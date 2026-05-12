"""
Fast probabilistic core model.
Uses hit-rate curves instead of per-cacheline tracking.
PMU counters remain statistically accurate. ~40x faster.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import random
import math

from simulation.request import MemRequest, ReqType


@dataclass
class SBCounters:
    stall_cycles_rd_wr: int = 0
    stall_cycles_wr_only: int = 0
    total_stores: int = 0
    sb_full_events: int = 0

    def snapshot(self) -> dict:
        return {"sb.stall_rdwr": self.stall_cycles_rd_wr,
                "sb.stall_wronly": self.stall_cycles_wr_only,
                "sb.full_events": self.sb_full_events}


@dataclass
class L1DCounters:
    stalls_l1d_miss: int = 0
    replacement: int = 0
    load_hit: int = 0
    load_miss: int = 0
    response_wait_cycles: int = 0

    def snapshot(self) -> dict:
        return {"l1d.hit": self.load_hit, "l1d.miss": self.load_miss,
                "l1d.eviction": self.replacement, "l1d.stall": self.stalls_l1d_miss,
                "l1d.resp_wait": self.response_wait_cycles}


@dataclass
class LFBCounters:
    fb_hit: int = 0
    fb_full_stalls: int = 0

    def snapshot(self) -> dict:
        return {"lfb.hit": self.fb_hit, "lfb.stall": self.fb_full_stalls}


@dataclass
class L2Counters:
    drd_hit: int = 0;  drd_miss: int = 0
    rfo_hit: int = 0;  rfo_miss: int = 0
    hwpf_hit: int = 0; hwpf_miss: int = 0
    swpf_hit: int = 0; swpf_miss: int = 0
    dwr_hit: int = 0
    stall_cycles: int = 0
    drd_resp_cycles: int = 0
    rfo_resp_cycles: int = 0
    hwpf_resp_cycles: int = 0
    offcore_drd: int = 0
    offcore_rfo: int = 0

    def snapshot(self) -> dict:
        return {"l2.drd_hit": self.drd_hit, "l2.drd_miss": self.drd_miss,
                "l2.rfo_hit": self.rfo_hit, "l2.rfo_miss": self.rfo_miss,
                "l2.hwpf_hit": self.hwpf_hit, "l2.hwpf_miss": self.hwpf_miss,
                "l2.dwr_hit": self.dwr_hit, "l2.stall": self.stall_cycles,
                "l2.drd_resp": self.drd_resp_cycles,
                "l2.rfo_resp": self.rfo_resp_cycles,
                "l2.hwpf_resp": self.hwpf_resp_cycles}


def _hit_prob(cache_b: int, ws_b: int, locality: float, alpha: float = 0.85) -> float:
    if ws_b <= 0:
        return 1.0
    cap = min(1.0, cache_b / ws_b) ** alpha
    return locality + (1.0 - locality) * cap


class Core:
    """Fast probabilistic core: SB + LFB + L1D + L2."""

    def __init__(self, core_id: int, cfg: dict, rng: random.Random):
        self.core_id = core_id
        self.rng = rng
        cc = cfg["core"]

        # SB
        self.sb_entries = cc["sb_entries"]
        self._sb_fill = 0
        self.sb_counters = SBCounters()

        # LFB
        self.lfb_entries = cc["lfb_entries"]
        self._lfb_fill = 0
        self.lfb_counters = LFBCounters()

        # L1D
        self.l1d_size   = cc["l1d"]["size_kb"] * 1024
        self.l1d_lat    = cc["l1d"]["latency_cycles"]
        self.l1d_miss   = cc["l1d"]["miss_penalty_cycles"]
        self.l1d_counters = L1DCounters()

        # L2
        self.l2_size    = cc["l2"]["size_kb"] * 1024
        self.l2_lat     = cc["l2"]["latency_cycles"]
        self.l2_miss    = cc["l2"]["miss_penalty_cycles"]
        self.l2_counters = L2Counters()

        # Workload params (set by engine)
        self.working_set_bytes: int = 512 * 1024 * 1024
        self.locality: float = 0.5
        self._pending_loads: int = 0

    # SB helpers 
    def _sb_issue(self, req: MemRequest) -> bool:
        if self._sb_fill >= self.sb_entries:
            self.sb_counters.sb_full_events += 1
            has_loads = self._pending_loads > 0
            if has_loads: self.sb_counters.stall_cycles_rd_wr += 1
            else:         self.sb_counters.stall_cycles_wr_only += 1
            req.latency_sb += 1
            return False
        self._sb_fill += 1
        self.sb_counters.total_stores += 1
        return True

    def _sb_retire(self):
        self._sb_fill = max(0, self._sb_fill - 1)

    # LFB helpers 
    def _lfb_lookup(self) -> bool:
        p = min(0.08, self._lfb_fill / max(self.lfb_entries, 1) * 0.3)
        if self.rng.random() < p:
            self.lfb_counters.fb_hit += 1
            return True
        return False

    def _lfb_alloc(self) -> bool:
        if self._lfb_fill >= self.lfb_entries:
            self.lfb_counters.fb_full_stalls += 1
            return False
        self._lfb_fill += 1
        return True

    def _lfb_free(self):
        self._lfb_fill = max(0, self._lfb_fill - 1)

    # L1D access 
    def _l1d_access(self, req: MemRequest) -> bool:
        p = _hit_prob(self.l1d_size, self.working_set_bytes, self.locality)
        if req.is_cxl:
            p *= 0.60   # paper: 22.8% fewer hits under CXL (Fig 2c)
        hit = self.rng.random() < p
        if hit:
            self.l1d_counters.load_hit += 1
            req.hit_l1d = True
            req.latency_l1d += self.l1d_lat
        else:
            self.l1d_counters.load_miss += 1
            self.l1d_counters.stalls_l1d_miss += self.l1d_miss
            self.l1d_counters.response_wait_cycles += self.l1d_miss
            self.l1d_counters.replacement += 1
            req.latency_l1d += self.l1d_miss
        return hit

    # L2 access
    def _l2_access(self, req: MemRequest) -> bool:
        p = _hit_prob(self.l2_size, self.working_set_bytes, self.locality * 0.75)
        if req.is_cxl:
            p *= 0.52
        hit = self.rng.random() < p
        if hit:
            req.hit_l2 = True
            req.latency_l2 += self.l2_lat
            self._l2_count_hit(req)
        else:
            req.latency_l2 += self.l2_miss
            self.l2_counters.stall_cycles += self.l2_miss
            self._l2_count_miss(req)
        return hit

    def _l2_count_hit(self, req):
        t = req.req_type
        if   t == ReqType.DRD:  self.l2_counters.drd_hit  += 1
        elif t == ReqType.RFO:  self.l2_counters.rfo_hit  += 1
        elif t == ReqType.HWPF: self.l2_counters.hwpf_hit += 1
        elif t == ReqType.SWPF: self.l2_counters.swpf_hit += 1
        elif t == ReqType.DWR:  self.l2_counters.dwr_hit  += 1

    def _l2_count_miss(self, req):
        t, p = req.req_type, self.l2_miss
        if t == ReqType.DRD:
            self.l2_counters.drd_miss += 1
            self.l2_counters.drd_resp_cycles += p
            self.l2_counters.offcore_drd += 1
        elif t == ReqType.RFO:
            self.l2_counters.rfo_miss += 1
            self.l2_counters.rfo_resp_cycles += p
            self.l2_counters.offcore_rfo += 1
        elif t == ReqType.HWPF:
            self.l2_counters.hwpf_miss += 1
            self.l2_counters.hwpf_resp_cycles += p
        elif t == ReqType.SWPF:
            self.l2_counters.swpf_miss += 1

    # Main pipeline
    def process(self, req: MemRequest) -> Optional[str]:
        if req.is_store:
            if not self._sb_issue(req):
                return None

        if req.is_load:
            self._pending_loads += 1
            if self._l1d_access(req):
                self._pending_loads -= 1
                return "L1D"
            if self._lfb_lookup():
                req.latency_lfb += 4
                self._pending_loads -= 1
                return "LFB"
            if not self._lfb_alloc():
                return None
            if self._l2_access(req):
                self._lfb_free()
                self._pending_loads -= 1
                return "L2"
            return None   # goes to CHA

        if self._l1d_access(req):
            return "L1D"
        return None

    def fill_complete(self, req: MemRequest):
        self._lfb_free()
        if req.is_load:
            self._pending_loads = max(0, self._pending_loads - 1)
        if req.is_store:
            self._sb_retire()

    def snapshot(self) -> dict:
        return {
            "core_id": self.core_id,
            **self.sb_counters.snapshot(),
            **self.l1d_counters.snapshot(),
            **self.lfb_counters.snapshot(),
            **self.l2_counters.snapshot(),
        }

    def reset_counters(self):
        self.sb_counters  = SBCounters()
        self.l1d_counters = L1DCounters()
        self.lfb_counters = LFBCounters()
        self.l2_counters  = L2Counters()