"""
CHA (Caching and Home Agent) + LLC slice model.
Implements Table 2 PMU counters: TOR inserts/occupancy, LLC hit/miss by path,
serve-target distribution (local/SNC/remote/CXL), and cache coherence states.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import random

from simulation.request import MemRequest, ReqType, MemTarget


# Coherence states (MESIF)
class CohState:
    M = "M"  # Modified  (exclusive dirty)
    E = "E"  # Exclusive (clean)
    S = "S"  # Shared
    I = "I"  # Invalid
    F = "F"  # Forward


# PMU Counters

@dataclass
class CHACounters:
    """Mirrors Table 2: unc_cha_tor_inserts / occupancy, per DRd/RFO/HWPF/DWr."""

    # TOR inserts per path
    tor_insert_drd_total:   int = 0
    tor_insert_drd_hit:     int = 0
    tor_insert_drd_miss:    int = 0
    tor_insert_drd_cxl:     int = 0   # miss, target=CXL

    tor_insert_rfo_total:   int = 0
    tor_insert_rfo_hit:     int = 0
    tor_insert_rfo_miss:    int = 0
    tor_insert_rfo_cxl:     int = 0

    tor_insert_hwpf_total:  int = 0
    tor_insert_hwpf_hit:    int = 0
    tor_insert_hwpf_miss:   int = 0
    tor_insert_hwpf_cxl:    int = 0

    tor_insert_dwr_total:   int = 0
    tor_insert_dwr_hit:     int = 0
    tor_insert_dwr_miss:    int = 0

    # TOR occupancy (cumulative, divided by cycles = avg occupancy)
    tor_occ_drd_hit:        int = 0
    tor_occ_drd_miss:       int = 0
    tor_occ_rfo_hit:        int = 0
    tor_occ_rfo_miss:       int = 0
    tor_occ_hwpf_hit:       int = 0
    tor_occ_hwpf_miss:      int = 0

    # Serve targets for misses (unc_cha_tor_inserts.ia_drd, 9 scenarios)
    serve_local_mem:        int = 0
    serve_snc_llc:          int = 0
    serve_snc_mem:          int = 0
    serve_remote_llc:       int = 0
    serve_remote_mem:       int = 0
    serve_cxl_mem:          int = 0

    # LLC stall cycles (core-observed DRd stall + DRd response time)
    llc_stall_cycles:       int = 0
    llc_drd_resp_cycles:    int = 0

    def snapshot(self) -> dict:
        return {
            "cha.tor_drd_hit":    self.tor_insert_drd_hit,
            "cha.tor_drd_miss":   self.tor_insert_drd_miss,
            "cha.tor_drd_cxl":    self.tor_insert_drd_cxl,
            "cha.tor_rfo_hit":    self.tor_insert_rfo_hit,
            "cha.tor_rfo_miss":   self.tor_insert_rfo_miss,
            "cha.tor_rfo_cxl":    self.tor_insert_rfo_cxl,
            "cha.tor_hwpf_hit":   self.tor_insert_hwpf_hit,
            "cha.tor_hwpf_miss":  self.tor_insert_hwpf_miss,
            "cha.tor_hwpf_cxl":   self.tor_insert_hwpf_cxl,
            "cha.tor_dwr_hit":    self.tor_insert_dwr_hit,
            "cha.tor_dwr_miss":   self.tor_insert_dwr_miss,
            "cha.serve_local":    self.serve_local_mem,
            "cha.serve_snc_llc":  self.serve_snc_llc,
            "cha.serve_snc_mem":  self.serve_snc_mem,
            "cha.serve_remote_llc": self.serve_remote_llc,
            "cha.serve_remote_mem": self.serve_remote_mem,
            "cha.serve_cxl":      self.serve_cxl_mem,
            "cha.llc_stall":      self.llc_stall_cycles,
            "cha.drd_resp":       self.llc_drd_resp_cycles,
            "cha.occ_drd_hit":    self.tor_occ_drd_hit,
            "cha.occ_drd_miss":   self.tor_occ_drd_miss,
            "cha.occ_rfo_hit":    self.tor_occ_rfo_hit,
            "cha.occ_rfo_miss":   self.tor_occ_rfo_miss,
            "cha.occ_hwpf_hit":   self.tor_occ_hwpf_hit,
            "cha.occ_hwpf_miss":  self.tor_occ_hwpf_miss,
        }


# LLC Slice (one per CHA)

class LLCSlice:
    """Single LLC slice backed by a set-associative structure."""

    def __init__(self, size_kb: int, assoc: int):
        self.assoc    = assoc
        self.num_sets = (size_kb * 1024) // (assoc * 64)
        # sets: idx -> [(tag, state)]
        self.sets: Dict[int, List[Tuple[int, str]]] = {}

    def _decompose(self, addr: int):
        cl = addr >> 6
        return cl % self.num_sets, cl // self.num_sets

    def lookup(self, addr: int) -> Optional[str]:
        """Returns coherence state if hit, None on miss."""
        idx, tag = self._decompose(addr)
        for t, st in self.sets.get(idx, []):
            if t == tag:
                return st
        return None

    def fill(self, addr: int, state: str = CohState.E):
        idx, tag = self._decompose(addr)
        ways = self.sets.get(idx, [])
        if not any(t == tag for t, _ in ways):
            if len(ways) >= self.assoc:
                ways.pop()
            ways.insert(0, (tag, state))
            self.sets[idx] = ways

    def update_state(self, addr: int, state: str):
        idx, tag = self._decompose(addr)
        ways = self.sets.get(idx, [])
        self.sets[idx] = [(t, state if t == tag else s) for t, s in ways]

    def invalidate(self, addr: int):
        idx, tag = self._decompose(addr)
        self.sets[idx] = [(t, s) for t, s in self.sets.get(idx, []) if t != tag]


# TOR (Table Of Requests)

class TOR:
    """
    Hardware queue inside the CHA that tracks in-flight LLC misses.
    Allows PathFinder to compute occupancy (unc_cha_tor_occupancy).
    """

    def __init__(self, max_entries: int):
        self.max_entries = max_entries
        self.entries: Dict[int, MemRequest] = {}  # req_id -> req

    def insert(self, req: MemRequest) -> bool:
        if len(self.entries) >= self.max_entries:
            return False  # TOR full
        self.entries[req.req_id] = req
        return True

    def retire(self, req_id: int):
        self.entries.pop(req_id, None)

    @property
    def occupancy(self) -> int:
        return len(self.entries)

    @property
    def full(self) -> bool:
        return len(self.entries) >= self.max_entries


# CHA unit

class CHA:
    """
    CHA = LLC slice + CCD (cache coherent directory) + TOR.
    Handles LLC lookup, coherence, and decides serve target for misses.
    """

    def __init__(self, cha_id: int, cfg: dict, cxl_ratio: float = 0.3, rng: Optional[random.Random] = None):
        self.cha_id    = cha_id
        self.rng       = rng or random.Random()
        self.counters  = CHACounters()

        c = cfg["cha"]
        slice_kb = (c["llc_total_mb"] * 1024) // c["num_cha"]
        self.llc    = LLCSlice(slice_kb, c["llc_associativity"])
        self.tor    = TOR(c["tor_entries"])
        self.hit_lat  = c["llc_latency_cycles"]
        self.miss_lat = c["llc_miss_penalty_cycles"]

        # Fraction of LLC misses that go to CXL vs local DRAM
        # This is the key knob for reproducing CXL vs local experiments
        self.cxl_ratio = cxl_ratio

    def process(self, req: MemRequest, cycle: int) -> MemTarget:
        """
        Lookup LLC. Update TOR. Determine serve target.
        Returns MemTarget and updates req latency.
        """
        state = self.llc.lookup(req.address)
        hit   = state is not None

        self._tor_insert(req)
        self._count_tor_insert(req, hit)

        if hit:
            req.hit_llc = True
            req.latency_llc += self.hit_lat
            self.counters.llc_stall_cycles += self.hit_lat
            self._count_occ(req, hit=True)
            self.llc.update_state(req.address, CohState.E if req.is_store else state)
            self.tor.retire(req.req_id)
            return MemTarget.LOCAL_LLC

        # LLC miss — determine target
        req.latency_llc += self.miss_lat
        self.counters.llc_stall_cycles += self.miss_lat
        self.counters.llc_drd_resp_cycles += self.miss_lat
        self._count_occ(req, hit=False)

        target = self._route_miss(req)
        req.is_cxl = (target == MemTarget.CXL_DRAM)
        req.target  = target
        self._count_serve(target)
        self.llc.fill(req.address, CohState.E)
        self.tor.retire(req.req_id)
        return target

    def _route_miss(self, req: MemRequest) -> MemTarget:
        """Decide where a miss is served (local/SNC/remote/CXL)."""
        r = self.rng.random()
        if r < self.cxl_ratio:
            return MemTarget.CXL_DRAM
        elif r < self.cxl_ratio + 0.05:
            return MemTarget.SNC_LLC
        elif r < self.cxl_ratio + 0.10:
            return MemTarget.REMOTE_LLC
        else:
            return MemTarget.LOCAL_DRAM

    def _tor_insert(self, req: MemRequest):
        self.tor.insert(req)

    def _count_tor_insert(self, req: MemRequest, hit: bool):
        t = req.req_type
        if t == ReqType.DRD:
            self.counters.tor_insert_drd_total += 1
            if hit: self.counters.tor_insert_drd_hit += 1
            else:   self.counters.tor_insert_drd_miss += 1
        elif t == ReqType.RFO:
            self.counters.tor_insert_rfo_total += 1
            if hit: self.counters.tor_insert_rfo_hit += 1
            else:   self.counters.tor_insert_rfo_miss += 1
        elif t == ReqType.HWPF:
            self.counters.tor_insert_hwpf_total += 1
            if hit: self.counters.tor_insert_hwpf_hit += 1
            else:   self.counters.tor_insert_hwpf_miss += 1
        elif t in (ReqType.DWR, ReqType.WB):
            self.counters.tor_insert_dwr_total += 1
            if hit: self.counters.tor_insert_dwr_hit += 1
            else:   self.counters.tor_insert_dwr_miss += 1

    def _count_occ(self, req: MemRequest, hit: bool):
        t = req.req_type
        occ = self.tor.occupancy
        if t == ReqType.DRD:
            if hit: self.counters.tor_occ_drd_hit  += occ
            else:   self.counters.tor_occ_drd_miss += occ
        elif t == ReqType.RFO:
            if hit: self.counters.tor_occ_rfo_hit  += occ
            else:   self.counters.tor_occ_rfo_miss += occ
        elif t == ReqType.HWPF:
            if hit: self.counters.tor_occ_hwpf_hit  += occ
            else:   self.counters.tor_occ_hwpf_miss += occ

    def _count_serve(self, target: MemTarget):
        if   target == MemTarget.LOCAL_DRAM:  self.counters.serve_local_mem  += 1
        elif target == MemTarget.SNC_LLC:     self.counters.serve_snc_llc    += 1
        elif target == MemTarget.REMOTE_LLC:  self.counters.serve_remote_llc += 1
        elif target == MemTarget.CXL_DRAM:
            self.counters.serve_cxl_mem    += 1
            self.counters.tor_insert_drd_cxl  += 1

    def snapshot(self) -> dict:
        return {"cha_id": self.cha_id, **self.counters.snapshot()}

    def reset_counters(self):
        self.counters = CHACounters()


class CHAArray:
    """
    Manages all CHA slices for a socket.
    Routes LLC requests to the appropriate CHA using address hashing.
    """

    def __init__(self, cfg: dict, cxl_ratio: float = 0.3, rng: Optional[random.Random] = None):
        self.num_cha = cfg["cha"]["num_cha"]
        self.rng     = rng or random.Random()
        self.chas: List[CHA] = [
            CHA(i, cfg, cxl_ratio=cxl_ratio, rng=self.rng)
            for i in range(self.num_cha)
        ]

    def get_cha(self, addr: int) -> CHA:
        """Hash address to CHA slice."""
        return self.chas[(addr >> 6) % self.num_cha]

    def process(self, req: MemRequest, cycle: int) -> MemTarget:
        return self.get_cha(req.address).process(req, cycle)

    def snapshot(self) -> List[dict]:
        return [c.snapshot() for c in self.chas]

    def aggregate_snapshot(self) -> dict:
        """Aggregate all CHA counters into a single dict for analysis."""
        agg: dict = {}
        for cha in self.chas:
            for k, v in cha.counters.snapshot().items():
                agg[k] = agg.get(k, 0) + v
        return agg

    def reset_counters(self):
        for c in self.chas:
            c.reset_counters()