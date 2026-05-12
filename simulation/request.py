"""
MemRequest: the atomic unit that flows through the CXL simulation.
Models DRd, DWr, RFO, HWPF, SWPF as defined in PathFinder §2.2.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class ReqType(Enum):
    DRD  = auto()   # Demand read
    DWR  = auto()   # Demand write (triggers RFO internally)
    RFO  = auto()   # Read for ownership
    HWPF = auto()   # Hardware prefetch
    SWPF = auto()   # Software prefetch
    WB   = auto()   # Writeback (eviction)


class MemTarget(Enum):
    """Where a request is ultimately served from."""
    L1D        = auto()
    LFB        = auto()
    L2         = auto()
    LOCAL_LLC  = auto()   # LLC slice on same core's chiplet
    SNC_LLC    = auto()   # LLC slice in SNC cluster
    REMOTE_LLC = auto()   # LLC on another socket
    LOCAL_DRAM = auto()   # Local DIMM via IMC
    CXL_DRAM   = auto()   # CXL DIMM via M2PCIe + FlexBus


class ReqState(Enum):
    ISSUED    = auto()
    IN_SB     = auto()
    IN_LFB    = auto()
    IN_L1D    = auto()
    IN_L2     = auto()
    IN_LLC    = auto()
    IN_MESH   = auto()
    IN_IMC    = auto()
    IN_M2PCIE = auto()
    IN_FLEXBUS = auto()
    COMPLETED = auto()


@dataclass
class MemRequest:
    req_id:       int
    req_type:     ReqType
    address:      int           # memory address
    core_id:      int
    mflow_id:     int           # which mFlow this belongs to
    issued_cycle: int

    # Filled in as request progresses
    target:       Optional[MemTarget] = None
    state:        ReqState = ReqState.ISSUED
    complete_cycle: Optional[int] = None

    # Per-stage latency accounting (cycles spent at each stage)
    latency_sb:      int = 0
    latency_l1d:     int = 0
    latency_lfb:     int = 0
    latency_l2:      int = 0
    latency_llc:     int = 0
    latency_mesh:    int = 0
    latency_imc:     int = 0
    latency_m2pcie:  int = 0
    latency_flexbus: int = 0
    latency_dimm:    int = 0

    # Hit/miss flags at each level
    hit_l1d:  bool = False
    hit_lfb:  bool = False
    hit_l2:   bool = False
    hit_llc:  bool = False

    # Is this a CXL-destined request?
    is_cxl:   bool = False

    @property
    def total_latency(self) -> int:
        if self.complete_cycle is None:
            return 0
        return self.complete_cycle - self.issued_cycle

    @property
    def is_load(self) -> bool:
        return self.req_type in (ReqType.DRD, ReqType.RFO, ReqType.HWPF, ReqType.SWPF)

    @property
    def is_store(self) -> bool:
        return self.req_type in (ReqType.DWR, ReqType.WB)

    def stall_breakdown(self) -> dict:
        """Returns per-stage stall contribution as dict."""
        return {
            "SB":       self.latency_sb,
            "L1D":      self.latency_l1d,
            "LFB":      self.latency_lfb,
            "L2":       self.latency_l2,
            "LLC/CHA":  self.latency_llc,
            "Mesh":     self.latency_mesh,
            "IMC":      self.latency_imc,
            "FlexBus+MC": self.latency_m2pcie + self.latency_flexbus,
            "CXL_DIMM": self.latency_dimm,
        }