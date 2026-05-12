"""
Workload generators for the CXL simulation.
Each workload is a callable: (engine, cycle) -> List[MemRequest]

Two categories:
  1. Synthetic: MBW, GUPS, random-read, stream - used in Cases 4/5
  2. BenchmarkProfile: parameter-driven profiles approximating SPEC/PARSEC/GAP
     access patterns (working set size, read/write ratio, locality) - Cases 1-3/6/7
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Callable, Optional
import random
import math

from simulation.request import MemRequest, ReqType


# Base workload interface 

class Workload:
    """Base class. Subclasses implement generate()."""
    name: str = "base"

    def __call__(self, engine, cycle: int) -> List[MemRequest]:
        return self.generate(engine, cycle)

    def generate(self, engine, cycle: int) -> List[MemRequest]:
        raise NotImplementedError


# Address generators

def random_addr(rng: random.Random, mem_bytes: int) -> int:
    return rng.randrange(0, mem_bytes, 64)

def sequential_addr(base: int, stride: int, step: int) -> int:
    return base + (step * stride) % (1 << 32)

def zipf_addr(rng: random.Random, hot_bytes: int, total_bytes: int,
              hot_prob: float = 0.9) -> int:
    if rng.random() < hot_prob:
        return rng.randrange(0, hot_bytes, 64)
    return rng.randrange(hot_bytes, total_bytes, 64)


# Synthetic workloads 

class MBWWorkload(Workload):
    """
    Memory bandwidth benchmark - sequential reads/writes.
    Mimics the 'MBW' tool used in Case 5.
    target_bw_gbps: desired injection rate.
    """
    def __init__(self, core_id: int, target_bw_gbps: float,
                 mem_size_mb: int = 512, rw_ratio: float = 1.0,
                 use_cxl: bool = True, seed: int = 0):
        self.name         = f"MBW_{target_bw_gbps:.0f}GB"
        self.core_id      = core_id
        self.target_bw    = target_bw_gbps
        self.mem_bytes    = mem_size_mb * 1024 * 1024
        self.rw_ratio     = rw_ratio      # fraction that are reads
        self.use_cxl      = use_cxl
        self.rng          = random.Random(seed)
        self._step        = 0

    def generate(self, engine, cycle: int) -> List[MemRequest]:
        # Throttle injection based on target bandwidth
        freq_ghz = engine.cfg["cpu"]["freq_ghz"]
        cls_per_cycle = max(1, int(self.target_bw * 1e9 / (64 * 8 * freq_ghz * 1e9)))

        reqs = []
        for _ in range(cls_per_cycle):
            addr = sequential_addr(0, 64, self._step)
            self._step += 1
            rtype = ReqType.DRD if self.rng.random() < self.rw_ratio else ReqType.DWR
            req = engine.make_request(rtype, addr, self.core_id)
            req.is_cxl = self.use_cxl
            reqs.append(req)
        return reqs


class GUPSWorkload(Workload):
    """
    GUPS (Giga Updates Per Second) - random read-modify-write.
    High spatial randomness, stresses CXL latency heavily.
    """
    def __init__(self, core_id: int, hot_set_mb: int, total_mb: int,
                 hot_prob: float = 0.9, rw_ratio: float = 0.5,
                 use_cxl: bool = True, seed: int = 0):
        self.name       = f"GUPS_hot{hot_set_mb}MB"
        self.core_id    = core_id
        self.hot_bytes  = hot_set_mb * 1024 * 1024
        self.total_bytes = total_mb * 1024 * 1024
        self.hot_prob   = hot_prob
        self.rw_ratio   = rw_ratio
        self.use_cxl    = use_cxl
        self.rng        = random.Random(seed)
        self._ops       = 0

    def generate(self, engine, cycle: int) -> List[MemRequest]:
        addr  = zipf_addr(self.rng, self.hot_bytes, self.total_bytes, self.hot_prob)
        rtype = ReqType.DRD if self.rng.random() < self.rw_ratio else ReqType.DWR
        req   = engine.make_request(rtype, addr, self.core_id)
        req.is_cxl = self.use_cxl
        self._ops += 1
        return [req]


class StreamWorkload(Workload):
    """STREAM-like: sequential array copy. Low locality, high BW pressure."""
    def __init__(self, core_id: int, array_mb: int = 256,
                 use_cxl: bool = True, seed: int = 0):
        self.name      = "STREAM"
        self.core_id   = core_id
        self.arr_bytes = array_mb * 1024 * 1024
        self.use_cxl   = use_cxl
        self.rng       = random.Random(seed)
        self._step     = 0

    def generate(self, engine, cycle: int) -> List[MemRequest]:
        reqs = []
        for rtype in (ReqType.DRD, ReqType.DWR):
            addr = sequential_addr(0, 64, self._step)
            req  = engine.make_request(rtype, addr, self.core_id)
            req.is_cxl = self.use_cxl
            reqs.append(req)
        self._step += 1
        return reqs


class RandomReadWorkload(Workload):
    """Pure random reads - worst case for CXL latency."""
    def __init__(self, core_id: int, mem_mb: int = 256,
                 use_cxl: bool = True, seed: int = 0):
        self.name     = "RandomRead"
        self.core_id  = core_id
        self.mem_bytes = mem_mb * 1024 * 1024
        self.use_cxl  = use_cxl
        self.rng      = random.Random(seed)

    def generate(self, engine, cycle: int) -> List[MemRequest]:
        addr = random_addr(self.rng, self.mem_bytes)
        req  = engine.make_request(ReqType.DRD, addr, self.core_id)
        req.is_cxl = self.use_cxl
        return [req]


# Benchmark profiles

@dataclass
class BenchmarkProfile:
    """
    Approximates a real benchmark's memory access behavior.
    Used to reproduce Case 1 (path classification) and Case 7 (optimization).

    Parameters tuned to match paper's SPEC/PARSEC/GAP observations.
    """
    name:            str
    working_set_mb:  int
    read_fraction:   float     # fraction DRD
    write_fraction:  float     # fraction DWR
    rfo_fraction:    float     # fraction RFO
    hwpf_fraction:   float     # fraction HWPF
    locality:        float     # 0=pure random, 1=perfect sequential
    hot_set_frac:    float     # fraction of WS that is "hot" (temporal locality)
    hot_prob:        float     # probability of accessing hot set
    cxl_miss_bias:   float     # extra CXL miss rate boost (0=same as local)
    description:     str = ""


# Paper's benchmark profiles - tuned to reproduce Table 7 and Figures 2/3/6
BENCHMARK_PROFILES: dict = {

    # SPEC CPU2017 - from paper's experimental data
    "GCC": BenchmarkProfile(
        name="602.gcc_s", working_set_mb=1367,
        read_fraction=0.55, write_fraction=0.25, rfo_fraction=0.12, hwpf_fraction=0.08,
        locality=0.45, hot_set_frac=0.15, hot_prob=0.70, cxl_miss_bias=0.15,
        description="Compiler - high RFO, moderate locality"),

    "MCF": BenchmarkProfile(
        name="505.mcf_s", working_set_mb=3961,
        read_fraction=0.70, write_fraction=0.10, rfo_fraction=0.08, hwpf_fraction=0.12,
        locality=0.10, hot_set_frac=0.05, hot_prob=0.45, cxl_miss_bias=0.40,
        description="Network simplex - highly irregular, pointer chasing"),

    "ROMS": BenchmarkProfile(
        name="654.roms_s", working_set_mb=10387,
        read_fraction=0.50, write_fraction=0.20, rfo_fraction=0.10, hwpf_fraction=0.20,
        locality=0.80, hot_set_frac=0.30, hot_prob=0.85, cxl_miss_bias=0.05,
        description="Ocean simulation - high HWPF, good locality"),

    "CAC": BenchmarkProfile(
        name="607.cactuBSSN_s", working_set_mb=6724,
        read_fraction=0.45, write_fraction=0.20, rfo_fraction=0.15, hwpf_fraction=0.20,
        locality=0.65, hot_set_frac=0.20, hot_prob=0.75, cxl_miss_bias=0.10,
        description="Numerical relativity - structured stencil"),

    "BWA": BenchmarkProfile(
        name="503.bwaves_s", working_set_mb=11467,
        read_fraction=0.50, write_fraction=0.15, rfo_fraction=0.10, hwpf_fraction=0.25,
        locality=0.75, hot_set_frac=0.25, hot_prob=0.80, cxl_miss_bias=0.08,
        description="Fluid dynamics - HWPF-heavy, moderate CXL miss"),

    "FOTS": BenchmarkProfile(
        name="649.fotonik3d_s", working_set_mb=9643,
        read_fraction=0.40, write_fraction=0.15, rfo_fraction=0.05, hwpf_fraction=0.40,
        locality=0.70, hot_set_frac=0.30, hot_prob=0.82, cxl_miss_bias=0.08,
        description="FDTD photonics - 59% HWPF to CXL (Table 7)"),

    "DEEP": BenchmarkProfile(
        name="631.deepsjeng_s", working_set_mb=6880,
        read_fraction=0.60, write_fraction=0.15, rfo_fraction=0.10, hwpf_fraction=0.15,
        locality=0.35, hot_set_frac=0.10, hot_prob=0.55, cxl_miss_bias=0.25,
        description="Chess engine - irregular, moderate locality"),

    "XZ": BenchmarkProfile(
        name="657.xz_s", working_set_mb=15344,
        read_fraction=0.55, write_fraction=0.20, rfo_fraction=0.10, hwpf_fraction=0.15,
        locality=0.50, hot_set_frac=0.15, hot_prob=0.65, cxl_miss_bias=0.20,
        description="Data compression - mixed locality"),

    "LBM": BenchmarkProfile(
        name="619.lbm_s", working_set_mb=3225,
        read_fraction=0.50, write_fraction=0.20, rfo_fraction=0.05, hwpf_fraction=0.25,
        locality=0.85, hot_set_frac=0.40, hot_prob=0.90, cxl_miss_bias=0.03,
        description="Lattice-Boltzmann - very high locality, LFB benefits from CXL"),

    "OMN": BenchmarkProfile(
        name="620.omnetpp_s", working_set_mb=242,
        read_fraction=0.55, write_fraction=0.15, rfo_fraction=0.15, hwpf_fraction=0.15,
        locality=0.30, hot_set_frac=0.10, hot_prob=0.50, cxl_miss_bias=0.30,
        description="Network simulation - irregular pointer access"),

    # PARSEC benchmarks
    "BODY": BenchmarkProfile(
        name="bodytrack", working_set_mb=33,
        read_fraction=0.55, write_fraction=0.20, rfo_fraction=0.15, hwpf_fraction=0.10,
        locality=0.60, hot_set_frac=0.25, hot_prob=0.75, cxl_miss_bias=0.12,
        description="Computer vision - structured, moderate CXL sensitivity"),

    "RAY": BenchmarkProfile(
        name="raytrace", working_set_mb=1283,
        read_fraction=0.65, write_fraction=0.10, rfo_fraction=0.10, hwpf_fraction=0.15,
        locality=0.25, hot_set_frac=0.08, hot_prob=0.45, cxl_miss_bias=0.35,
        description="Ray tracing - irregular, FlexBus-heavy stall"),

    "WATER": BenchmarkProfile(
        name="water_nsquared", working_set_mb=29,
        read_fraction=0.50, write_fraction=0.25, rfo_fraction=0.15, hwpf_fraction=0.10,
        locality=0.70, hot_set_frac=0.30, hot_prob=0.80, cxl_miss_bias=0.10,
        description="Molecular dynamics - small WS, good locality"),

    "VOL": BenchmarkProfile(
        name="volrend", working_set_mb=54,
        read_fraction=0.60, write_fraction=0.15, rfo_fraction=0.10, hwpf_fraction=0.15,
        locality=0.55, hot_set_frac=0.20, hot_prob=0.70, cxl_miss_bias=0.15,
        description="Volume rendering - moderate spatial locality"),

    "BARN": BenchmarkProfile(
        name="barnes", working_set_mb=1584,
        read_fraction=0.55, write_fraction=0.20, rfo_fraction=0.15, hwpf_fraction=0.10,
        locality=0.40, hot_set_frac=0.12, hot_prob=0.58, cxl_miss_bias=0.22,
        description="N-body simulation - tree traversal, irregular"),

    "FFT": BenchmarkProfile(
        name="fft", working_set_mb=12291,
        read_fraction=0.50, write_fraction=0.20, rfo_fraction=0.10, hwpf_fraction=0.20,
        locality=0.65, hot_set_frac=0.20, hot_prob=0.75, cxl_miss_bias=0.12,
        description="FFT - strided access, moderate HWPF"),

    "FREQ": BenchmarkProfile(
        name="freqmine", working_set_mb=632,
        read_fraction=0.55, write_fraction=0.15, rfo_fraction=0.10, hwpf_fraction=0.20,
        locality=0.60, hot_set_frac=0.20, hot_prob=0.72, cxl_miss_bias=0.10,
        description="Frequent item mining - moderate locality"),

    # GAP graph workloads
    "BFS": BenchmarkProfile(
        name="BFS", working_set_mb=15778,
        read_fraction=0.70, write_fraction=0.10, rfo_fraction=0.05, hwpf_fraction=0.15,
        locality=0.05, hot_set_frac=0.03, hot_prob=0.25, cxl_miss_bias=0.50,
        description="Breadth-first search - highly irregular, huge WS"),

    "CC": BenchmarkProfile(
        name="CC", working_set_mb=12381,
        read_fraction=0.65, write_fraction=0.15, rfo_fraction=0.10, hwpf_fraction=0.10,
        locality=0.08, hot_set_frac=0.04, hot_prob=0.28, cxl_miss_bias=0.45,
        description="Connected components - irregular graph traversal"),

    # Redis-like (key-value)
    "YCSB_A": BenchmarkProfile(
        name="YCSB-A", working_set_mb=512,
        read_fraction=0.50, write_fraction=0.30, rfo_fraction=0.15, hwpf_fraction=0.05,
        locality=0.40, hot_set_frac=0.15, hot_prob=0.65, cxl_miss_bias=0.25,
        description="YCSB workload A: 50/50 read-write"),

    "YCSB_B": BenchmarkProfile(
        name="YCSB-B", working_set_mb=512,
        read_fraction=0.95, write_fraction=0.02, rfo_fraction=0.02, hwpf_fraction=0.01,
        locality=0.40, hot_set_frac=0.15, hot_prob=0.65, cxl_miss_bias=0.20,
        description="YCSB workload B: 95% read"),

    "YCSB_C": BenchmarkProfile(
        name="YCSB-C", working_set_mb=512,
        read_fraction=1.00, write_fraction=0.00, rfo_fraction=0.00, hwpf_fraction=0.00,
        locality=0.45, hot_set_frac=0.15, hot_prob=0.70, cxl_miss_bias=0.18,
        description="YCSB workload C: read-only"),

    "YCSB_D": BenchmarkProfile(
        name="YCSB-D", working_set_mb=512,
        read_fraction=0.95, write_fraction=0.05, rfo_fraction=0.00, hwpf_fraction=0.00,
        locality=0.55, hot_set_frac=0.20, hot_prob=0.80, cxl_miss_bias=0.15,
        description="YCSB workload D: read latest"),

    "YCSB_F": BenchmarkProfile(
        name="YCSB-F", working_set_mb=512,
        read_fraction=0.50, write_fraction=0.00, rfo_fraction=0.50, hwpf_fraction=0.00,
        locality=0.40, hot_set_frac=0.15, hot_prob=0.65, cxl_miss_bias=0.22,
        description="YCSB workload F: read-modify-write"),
}


class BenchmarkWorkload(Workload):
    """
    Generates requests matching a BenchmarkProfile's statistical characteristics.
    """

    def __init__(self, profile: BenchmarkProfile, core_id: int,
                 use_cxl: bool = True, seed: int = 0,
                 issue_rate: int = 2):
        self.profile    = profile
        self.name       = profile.name
        self.core_id    = core_id
        self.use_cxl    = use_cxl
        self.rng        = random.Random(seed)
        self.issue_rate = issue_rate  # requests per cycle
        self.ws_bytes   = profile.working_set_mb * 1024 * 1024
        self.hot_bytes  = max(64, int(self.ws_bytes * profile.hot_set_frac))
        self._step      = 0
        self._hwpf_ahead = 8  # hardware prefetch distance (cachelines)

    def _pick_type(self) -> ReqType:
        r = self.rng.random()
        p = self.profile
        if r < p.read_fraction:                       return ReqType.DRD
        if r < p.read_fraction + p.rfo_fraction:      return ReqType.RFO
        if r < p.read_fraction + p.rfo_fraction + p.hwpf_fraction: return ReqType.HWPF
        return ReqType.DWR

    def _pick_addr(self) -> int:
        r = self.rng.random()
        p = self.profile
        if r < p.locality:
            # Sequential / strided
            addr = sequential_addr(0, 64, self._step)
        elif r < p.locality + p.hot_prob * (1 - p.locality):
            # Hot set random
            addr = random_addr(self.rng, self.hot_bytes)
        else:
            # Cold / random
            addr = random_addr(self.rng, self.ws_bytes)
        self._step += 1
        return addr

    def generate(self, engine, cycle: int) -> List[MemRequest]:
        reqs = []
        for _ in range(self.issue_rate):
            rtype = self._pick_type()
            addr  = self._pick_addr()
            req   = engine.make_request(rtype, addr, self.core_id)
            req.is_cxl = self.use_cxl
            reqs.append(req)
        return reqs


class MultiCoreWorkload(Workload):
    """
    Runs multiple workloads concurrently across different cores.
    Used in Cases 3/4 (interference analysis).
    """

    def __init__(self, workloads: list):
        self.workloads = workloads
        self.name = "+".join(w.name for w in workloads)

    def generate(self, engine, cycle: int) -> List[MemRequest]:
        reqs = []
        for wl in self.workloads:
            reqs.extend(wl.generate(engine, cycle))
        return reqs