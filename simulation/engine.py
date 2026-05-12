"""
Main simulation engine.
Connects: Core[] -> CHAArray -> Mesh -> IMC/M2PCIe -> FlexBus -> DIMM/CXL_DIMM
Takes snapshots at configurable intervals for PathFinder to analyze.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Callable
import random
import yaml
from pathlib import Path

from simulation.request import MemRequest, ReqType, MemTarget, ReqState
from hardware.core   import Core
from hardware.cha    import CHAArray
from hardware.uncore import (MeshInterconnect, IMC, M2PCIe,
                              FlexBus, LocalDIMM, CXLDimmDevice)


class Snapshot:
    """One time-series snapshot of all PMU counters at a given cycle."""

    def __init__(self, cycle: int, snapshot_id: int):
        self.cycle       = cycle
        self.snapshot_id = snapshot_id
        self.cores:  List[dict] = []
        self.cha:    dict = {}
        self.mesh:   dict = {}
        self.imc:    dict = {}
        self.m2pcie: dict = {}
        self.flexbus: dict = {}
        self.dimm:   dict = {}
        self.cxl:    dict = {}
        self.workload_name: str = ""
        self.metadata: dict = {}

    def to_dict(self) -> dict:
        return {
            "cycle":       self.cycle,
            "snapshot_id": self.snapshot_id,
            "cores":       self.cores,
            "cha":         self.cha,
            "mesh":        self.mesh,
            "imc":         self.imc,
            "m2pcie":      self.m2pcie,
            "flexbus":     self.flexbus,
            "dimm":        self.dimm,
            "cxl":         self.cxl,
            "workload":    self.workload_name,
            "meta":        self.metadata,
        }


class SimulationEngine:
    """
    Cycle-driven simulation engine.

    Usage:
        engine = SimulationEngine.from_config("config/spr_agilex.yaml",
                                              cxl_ratio=0.3)
        engine.run(workload, cycles=1_000_000)
        snapshots = engine.snapshots
    """

    def __init__(self, cfg: dict, cxl_ratio: float = 0.3, seed: int = 42):
        self.cfg       = cfg
        self.cxl_ratio = cxl_ratio
        self.rng       = random.Random(seed)

        num_cores = cfg["cpu"]["num_cores"]
        self.cores: List[Core] = [
            Core(i, cfg, random.Random(seed + i)) for i in range(num_cores)
        ]
        self.cha_array  = CHAArray(cfg, cxl_ratio=cxl_ratio, rng=self.rng)
        self.mesh       = MeshInterconnect(cfg)
        self.imc        = IMC(cfg)
        self.m2pcie     = M2PCIe(cfg)
        self.flexbus    = FlexBus(cfg)
        self.local_dimm = LocalDIMM(cfg)
        self.cxl_dimm   = CXLDimmDevice(cfg)

        self.cycle      = 0
        self.snapshots: List[Snapshot] = []
        self.snap_interval = cfg["simulation"]["snapshot_interval_cycles"]
        self._snap_id   = 0

        # In-flight requests: req_id -> (req, expected_completion_cycle)
        self._in_flight: Dict[int, tuple] = {}
        self._completed: List[MemRequest] = []

        self._req_counter = 0

    @classmethod
    def from_config(cls, config_path: str, cxl_ratio: float = 0.3,
                    seed: int = 42) -> "SimulationEngine":
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cls(cfg, cxl_ratio=cxl_ratio, seed=seed)

    def make_request(self, req_type: ReqType, address: int,
                     core_id: int, mflow_id: int = 0) -> MemRequest:
        req = MemRequest(
            req_id=self._req_counter,
            req_type=req_type,
            address=address,
            core_id=core_id,
            mflow_id=mflow_id,
            issued_cycle=self.cycle,
        )
        self._req_counter += 1
        return req

    def submit(self, req: MemRequest) -> bool:
        """
        Submit a request into the pipeline.
        Returns True if accepted, False if stalled this cycle.
        """
        core  = self.cores[req.core_id % len(self.cores)]
        level = core.process(req)

        if level is not None:
            # Served by core-private hierarchy
            req.complete_cycle = self.cycle + req.total_latency
            req.state          = ReqState.COMPLETED
            self._completed.append(req)
            return True

        # Forward to CHA/LLC
        target = self.cha_array.process(req, self.cycle)

        if target == MemTarget.LOCAL_LLC:
            core.fill_complete(req)
            req.complete_cycle = self.cycle + req.latency_llc
            req.state          = ReqState.COMPLETED
            self._completed.append(req)
            return True

        if target == MemTarget.LOCAL_DRAM:
            done = self.imc.enqueue_read(req, self.cycle)
            self.local_dimm.serve(req, self.cycle)
            core.fill_complete(req)
            req.complete_cycle = self.cycle + req.latency_llc + req.latency_imc + req.latency_dimm
            req.state          = ReqState.COMPLETED
            self._completed.append(req)
            return True

        # CXL path: M2PCIe → FlexBus → CXL DIMM
        if target == MemTarget.CXL_DRAM:
            req.is_cxl = True
            ok = self.m2pcie.enqueue(req)
            if not ok:
                return False  # back-pressure stall

            sent = self.flexbus.send(req, self.cycle)
            if not sent:
                return False  # FlexBus credit stall

            enqueued = self.cxl_dimm.enqueue_m2s(req)
            if not enqueued:
                return False  # DIMM buffer full

            self._in_flight[req.req_id] = (req, self.cycle + req.latency_dimm)
            req.state = ReqState.IN_FLEXBUS
            return True

        # SNC/remote — treat as local DRAM + extra mesh hops for now
        done = self.local_dimm.serve(req, self.cycle)
        req.latency_mesh += self.mesh.hop_latency * 4
        core.fill_complete(req)
        req.complete_cycle = self.cycle + req.latency_llc + req.latency_dimm + req.latency_mesh
        req.state          = ReqState.COMPLETED
        self._completed.append(req)
        return True

    def _tick_cxl(self):
        """Process one cycle of CXL DIMM completions."""
        completed = self.cxl_dimm.tick(self.cycle)
        for req in completed:
            self.flexbus.return_credit(req.req_id)
            self.m2pcie.complete(req)
            core = self.cores[req.core_id % len(self.cores)]
            core.fill_complete(req)
            req.complete_cycle = self.cycle
            req.state          = ReqState.COMPLETED
            self._in_flight.pop(req.req_id, None)
            self._completed.append(req)

    def _take_snapshot(self, workload_name: str = "") -> Snapshot:
        snap = Snapshot(self.cycle, self._snap_id)
        snap.workload_name = workload_name
        snap.cores   = [c.snapshot() for c in self.cores]
        snap.cha     = self.cha_array.aggregate_snapshot()
        snap.mesh    = self.mesh.snapshot()
        snap.imc     = self.imc.snapshot()
        snap.m2pcie  = self.m2pcie.snapshot()
        snap.flexbus = self.flexbus.snapshot()
        snap.dimm    = self.local_dimm.snapshot()
        snap.cxl     = self.cxl_dimm.snapshot()
        snap.metadata = {
            "cxl_ratio":   self.cxl_ratio,
            "in_flight":   len(self._in_flight),
            "completed":   len(self._completed),
        }
        self._snap_id += 1
        return snap

    def run(self, request_generator, max_cycles: Optional[int] = None,
            workload_name: str = "", reset_between: bool = True) -> List[Snapshot]:
        """
        Main simulation loop.
        request_generator: callable(engine, cycle) -> List[MemRequest] | []
        """
        if max_cycles is None:
            max_cycles = self.cfg["simulation"]["clock_cycles"]

        if reset_between:
            self._reset_counters()

        self.snapshots = []
        self._completed = []

        for self.cycle in range(max_cycles):
            # IMC tick
            self.imc.tick()

            # CXL DIMM tick
            self._tick_cxl()

            # Generate and submit new requests
            new_reqs: List[MemRequest] = request_generator(self, self.cycle)
            for req in new_reqs:
                self.submit(req)

            # Periodic snapshot
            if self.cycle % self.snap_interval == 0:
                snap = self._take_snapshot(workload_name)
                self.snapshots.append(snap)

        # Final snapshot
        self.snapshots.append(self._take_snapshot(workload_name))
        return self.snapshots

    def _reset_counters(self):
        for c in self.cores:
            c.reset_counters()
        self.cha_array.reset_counters()
        self.cxl_dimm.reset_counters()
        self._completed = []
        self._in_flight = {}

    @property
    def completed_requests(self) -> List[MemRequest]:
        return self._completed