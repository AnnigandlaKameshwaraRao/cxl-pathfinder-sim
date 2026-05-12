"""
PathFinder's four analysis components (S 4.3 – S 4.6):

  PFBuilder     - Path map construction from PMU hit/miss counters
  PFEstimator   - Bottom-up back-propagation of CXL-induced stall cycles
  PFAnalyzer    - Little's Law queue-length estimation, culprit detection
  PFMaterializer- Cross-snapshot time-series analysis, locality/anomaly
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math

from simulation.request import MemTarget, ReqType
from pathfinder.clos_network import (ClosNetwork, MFlow, Path, NodeType,
                                      CXL_PATH_STAGES, LOCAL_PATH_STAGES)


# PFBuilder  (S 4.3)

class PFBuilder:
    """
    Constructs the CXL data path map by synthesising PMU hit/miss counters.
    Mirrors the TOR-walk described in Table 5 of the paper.

    Input : list of Snapshots from SimulationEngine
    Output: path_map dict describing per-path traffic loads
    """

    def __init__(self, clos: ClosNetwork):
        self.clos = clos

    def build(self, snapshots: list) -> Dict[str, dict]:
        """
        Build a path map from all snapshots.
        Returns dict keyed by mflow_id → path summary.
        """
        path_map: Dict[str, dict] = {}

        for snap in snapshots:
            for core_snap in snap.cores:
                cid = core_snap["core_id"]
                path_map.setdefault(f"core_{cid}", {})
                self._extract_core_paths(core_snap, path_map[f"core_{cid}"])

            self._extract_uncore_paths(snap, path_map)

        return path_map

    def _extract_core_paths(self, core_snap: dict, entry: dict):
        """Per-core path extraction from SB/LFB/L1D/L2 counters."""
        for key in ("l1d.hit", "l1d.miss", "lfb.hit", "lfb.stall",
                    "l2.drd_hit", "l2.drd_miss", "l2.rfo_hit", "l2.rfo_miss",
                    "l2.hwpf_hit", "l2.hwpf_miss", "l2.dwr_hit"):
            entry[key] = entry.get(key, 0) + core_snap.get(key, 0)

    def _extract_uncore_paths(self, snap, path_map: dict):
        """CHA + M2PCIe TOR-based path extraction (mirrors Table 5)."""
        cha = snap.cha
        path_map.setdefault("uncore", {})
        unc = path_map["uncore"]

        # DRd path: TOR inserts hit/miss/CXL
        for key in ("cha.tor_drd_hit", "cha.tor_drd_miss", "cha.tor_drd_cxl",
                    "cha.tor_rfo_hit", "cha.tor_rfo_miss", "cha.tor_rfo_cxl",
                    "cha.tor_hwpf_hit","cha.tor_hwpf_miss","cha.tor_hwpf_cxl",
                    "cha.tor_dwr_hit", "cha.tor_dwr_miss",
                    "cha.serve_local",  "cha.serve_cxl",
                    "cha.serve_snc_llc","cha.serve_remote_llc"):
            unc[key] = unc.get(key, 0) + cha.get(key, 0)

        # M2PCIe FlexBus traffic (ground truth for CXL load/store)
        for key in ("m2pcie.rxc_ins", "m2pcie.txc_bl", "m2pcie.txc_ak"):
            unc[key] = unc.get(key, 0) + snap.m2pcie.get(key, 0)

    def path_report(self, path_map: dict) -> str:
        """Human-readable path classification report (Case 1)."""
        lines = ["PFBuilder Path Report"]
        unc = path_map.get("uncore", {})
        total_cxl = unc.get("cha.tor_drd_cxl", 0) + unc.get("cha.tor_rfo_cxl", 0) + \
                    unc.get("cha.tor_hwpf_cxl", 0)
        total_local = unc.get("cha.serve_local", 0)
        total = total_cxl + total_local + 1

        lines.append(f"  CXL memory hits:   {total_cxl:>10,}")
        lines.append(f"  Local DRAM hits:   {total_local:>10,}")
        lines.append(f"  CXL fraction:      {total_cxl/total*100:.1f}%")
        lines.append(f"  CXL load/store:    LD={unc.get('m2pcie.txc_bl',0):,}  "
                     f"ST={unc.get('m2pcie.txc_ak',0):,}")

        for core_key, cd in path_map.items():
            if not core_key.startswith("core_"):
                continue
            lines.append(f"\n  {core_key}:")
            drd_h = cd.get("l2.drd_hit", 0); drd_m = cd.get("l2.drd_miss", 0)
            rfo_h = cd.get("l2.rfo_hit", 0); rfo_m = cd.get("l2.rfo_miss", 0)
            hwpf_h= cd.get("l2.hwpf_hit",0); hwpf_m= cd.get("l2.hwpf_miss",0)
            lines.append(f"    DRd  hit={drd_h:,}  miss={drd_m:,}")
            lines.append(f"    RFO  hit={rfo_h:,}  miss={rfo_m:,}")
            lines.append(f"    HWPF hit={hwpf_h:,} miss={hwpf_m:,}")
        return "\n".join(lines)


# PFEstimator  (S 4.4  - Algorithm 2)

@dataclass
class StallBreakdown:
    """Per-stage stall cycle attribution for one mFlow."""
    sb:       float = 0.0
    l1d:      float = 0.0
    lfb:      float = 0.0
    l2:       float = 0.0
    llc:      float = 0.0
    flexbus_mc: float = 0.0
    cxl_dimm: float = 0.0

    def as_dict(self) -> dict:
        return {
            "SB":          self.sb,
            "L1D":         self.l1d,
            "LFB":         self.lfb,
            "L2":          self.l2,
            "LLC":         self.llc,
            "FlexBus+MC":  self.flexbus_mc,
            "CXL_DIMM":    self.cxl_dimm,
        }

    def total(self) -> float:
        return (self.sb + self.l1d + self.lfb + self.l2 +
                self.llc + self.flexbus_mc + self.cxl_dimm)


class PFEstimator:
    """
    Bottom-up back-propagation of CXL-induced stall cycles.
    Implements Algorithm 2 from the paper.

    Starting from CXL DIMM, iteratively attributes stall overhead
    backward through FlexBus → CHA → L2 → L1D → SB.
    """

    def __init__(self, clos: ClosNetwork):
        self.clos = clos

    def estimate(self, snapshots: list, req_type: str = "DRd") -> Dict[str, StallBreakdown]:
        """
        Returns per-mFlow stall breakdown across all snapshots.
        req_type: "DRd" | "RFO" | "HW PF" | "DWr"
        """
        results: Dict[str, StallBreakdown] = {}

        for snap in snapshots:
            bd = self._back_propagate(snap, req_type)
            key = snap.workload_name or f"snap_{snap.snapshot_id}"
            if key not in results:
                results[key] = StallBreakdown()
            r = results[key]
            r.sb         += bd.sb
            r.l1d        += bd.l1d
            r.lfb        += bd.lfb
            r.l2         += bd.l2
            r.llc        += bd.llc
            r.flexbus_mc += bd.flexbus_mc
            r.cxl_dimm   += bd.cxl_dimm

        return results

    def _back_propagate(self, snap, req_type: str) -> StallBreakdown:
        bd = StallBreakdown()
        cha = snap.cha
        fb  = snap.flexbus
        cx  = snap.cxl

        # Step 1: CXL DIMM stall (from packing buffer occupancy)
        cxl_stall = (cx.get("cxl.rxc_ne_req", 0) * 10 +
                     cx.get("cxl.rxc_full_req", 0) * 20)
        bd.cxl_dimm = float(cxl_stall)

        # Step 2: FlexBus + MC stall
        fb_stall = (fb.get("flexbus.starvation", 0) * 5 +
                    snap.m2pcie.get("m2pcie.rxc_ne", 0) * 2)
        bd.flexbus_mc = float(fb_stall) + bd.cxl_dimm * 0.15

        # Step 3: CHA/LLC stall - proportional to CXL TOR occupancy
        cxl_tor = cha.get("cha.tor_drd_cxl", 0) + cha.get("cha.tor_rfo_cxl", 0)
        total_tor = max(cha.get("cha.tor_drd_miss", 1), 1)
        cxl_weight = cxl_tor / total_tor
        llc_stall = cha.get("cha.llc_stall", 0) * cxl_weight
        bd.llc = float(llc_stall) + bd.flexbus_mc * 0.10

        # Step 4: Core stages - CXL stall propagates upward with attenuation
        # Attenuation factors derived from paper's Figure 6 observation: "CXL-induced stalls decrease by avg 74.5% from FlexBus+MC to L1D"
        locality_factor = 1.0 - 0.745  # 25.5% remains after cache hierarchy
        bd.l2  = bd.llc * locality_factor * 0.60
        bd.lfb = bd.l2  * 0.40
        bd.l1d = bd.lfb * 0.55
        bd.sb  = bd.l1d * 0.20   # SB gets smallest share (L1 data locality)

        # Adjust for RFO/HWPF/DWr paths (different attenuation)
        if req_type == "RFO":
            bd.sb  *= 0.95; bd.l1d *= 0.85
        elif req_type == "HW PF":
            bd.sb   = 0;    bd.l1d *= 0.70
        elif req_type == "DWr":
            bd.sb  *= 2.0;  bd.flexbus_mc *= 0.80

        return bd

    def stall_report(self, results: Dict[str, StallBreakdown]) -> str:
        lines = ["PFEstimator Stall Breakdown (ns) "]
        for wl, bd in results.items():
            lines.append(f"\n  {wl}:")
            for stage, val in bd.as_dict().items():
                lines.append(f"    {stage:<14}: {val:>10.1f} cycles")
            lines.append(f"    {'TOTAL':<14}: {bd.total():>10.1f} cycles")
        return "\n".join(lines)


# PFAnalyzer  (S 4.5  - Algorithm 1, Little's Law)

@dataclass
class QueueAnalysis:
    """Queue length estimate per hardware component via Little's Law (L = λW)."""
    component:   str
    queue_length: float    # L = λ * W
    lambda_hit:  float = 0.0
    lambda_miss: float = 0.0
    w_hit:       float = 0.0
    w_miss:      float = 0.0

    def is_bottleneck(self, threshold: float = 5.0) -> bool:
        return self.queue_length > threshold


class PFAnalyzer:
    """
    Delay-based queueing analysis (Algorithm 1).
    Models each hardware component as an FCFS queue (S3-FIFO variant).
    Applies Little's Law: L = λ * W.

    Identifies the culprit path (maximum queue length) at each snapshot.
    """

    # Hardware delays (cycles) - from paper's SPR measurements
    DELAYS = {
        NodeType.SB:       {"hit": 2,   "miss": 5},
        NodeType.L1D:      {"hit": 5,   "tag": 12},
        NodeType.LFB:      {"hit": 4,   "miss": 0},
        NodeType.L2:       {"hit": 14,  "tag": 50},
        NodeType.CHA:      {"hit": 50,  "miss": 120},
        NodeType.M2PCIE:   {"hit": 200, "miss": 400},
        NodeType.CXL_DIMM: {"hit": 710, "miss": 0},
        NodeType.IMC:      {"hit": 200, "miss": 0},
        NodeType.LOCAL_DIMM: {"hit": 200, "miss": 0},
    }

    def __init__(self, clos: ClosNetwork):
        self.clos = clos

    def analyze(self, snap) -> Dict[NodeType, QueueAnalysis]:
        """
        Apply Algorithm 1 to one snapshot.
        Returns queue analysis per hardware component.
        """
        clocks = max(snap.metadata.get("completed", 1), 1)
        results: Dict[NodeType, QueueAnalysis] = {}

        # Aggregate all core counters
        agg_cores = self._aggregate_cores(snap.cores)

        results[NodeType.L1D] = self._analyze_l1d(agg_cores, clocks)
        results[NodeType.LFB] = self._analyze_lfb(agg_cores, clocks)
        results[NodeType.L2]  = self._analyze_l2(agg_cores, clocks)
        results[NodeType.CHA] = self._analyze_cha(snap.cha, clocks)
        results[NodeType.M2PCIE] = self._analyze_flexbus(snap, clocks)
        results[NodeType.CXL_DIMM] = self._analyze_cxl(snap.cxl, clocks)

        return results

    def culprit_path(self, analysis: Dict[NodeType, QueueAnalysis]) -> NodeType:
        """Returns the component with maximum queue length (culprit)."""
        return max(analysis, key=lambda k: analysis[k].queue_length)

    def _aggregate_cores(self, cores: list) -> dict:
        agg: dict = {}
        for c in cores:
            for k, v in c.items():
                if k != "core_id":
                    agg[k] = agg.get(k, 0) + v
        return agg

    def _little(self, lam: float, w: float) -> float:
        """L = λ * W"""
        return lam * w

    def _analyze_l1d(self, agg: dict, clocks: int) -> QueueAnalysis:
        d = self.DELAYS[NodeType.L1D]
        lam_hit  = agg.get("l1d.hit",  0) / clocks
        lam_miss = agg.get("l1d.miss", 0) / clocks
        w_hit  = d["hit"]
        w_miss = d["tag"]
        L = self._little(lam_hit, w_hit) + self._little(lam_miss, w_miss)
        return QueueAnalysis("L1D", L, lam_hit, lam_miss, w_hit, w_miss)

    def _analyze_lfb(self, agg: dict, clocks: int) -> QueueAnalysis:
        d = self.DELAYS[NodeType.LFB]
        lam_hit = agg.get("lfb.hit", 0) / clocks
        L = self._little(lam_hit, d["hit"])
        return QueueAnalysis("LFB", L, lam_hit, 0, d["hit"], 0)

    def _analyze_l2(self, agg: dict, clocks: int) -> QueueAnalysis:
        d = self.DELAYS[NodeType.L2]
        drd_h = agg.get("l2.drd_hit",  0); drd_m = agg.get("l2.drd_miss", 0)
        rfo_h = agg.get("l2.rfo_hit",  0); rfo_m = agg.get("l2.rfo_miss", 0)
        lam_hit  = (drd_h + rfo_h) / clocks
        lam_miss = (drd_m + rfo_m) / clocks
        L = self._little(lam_hit, d["hit"]) + self._little(lam_miss, d["tag"])
        return QueueAnalysis("L2(DRd)", L, lam_hit, lam_miss, d["hit"], d["tag"])

    def _analyze_cha(self, cha: dict, clocks: int) -> QueueAnalysis:
        d = self.DELAYS[NodeType.CHA]
        lam_hit  = (cha.get("cha.tor_drd_hit",  0) +
                    cha.get("cha.tor_rfo_hit",  0)) / clocks
        lam_miss = (cha.get("cha.tor_drd_miss", 0) +
                    cha.get("cha.tor_rfo_miss", 0)) / clocks
        w_miss   = cha.get("cha.drd_resp", 0) / max(lam_miss * clocks, 1)
        L = (self._little(lam_hit, d["hit"]) +
             self._little(lam_miss, max(w_miss, d["miss"])))
        return QueueAnalysis("CHA(LLC)", L, lam_hit, lam_miss, d["hit"], w_miss)

    def _analyze_flexbus(self, snap, clocks: int) -> QueueAnalysis:
        d = self.DELAYS[NodeType.M2PCIE]
        rxc = snap.m2pcie.get("m2pcie.rxc_ins", 0)
        lam = rxc / clocks
        stall = snap.flexbus.get("flexbus.starvation", 0)
        w = d["hit"] + stall / max(lam * clocks, 1) * 10
        L = self._little(lam, w)
        return QueueAnalysis("FlexBus+MC", L, lam, 0, w, 0)

    def _analyze_cxl(self, cxl: dict, clocks: int) -> QueueAnalysis:
        d = self.DELAYS[NodeType.CXL_DIMM]
        lam = cxl.get("cxl.rxc_ins_req", 0) / clocks
        buf_util = cxl.get("cxl.buf_util_req", 0)
        w = d["hit"] * (1 + buf_util * 2)
        L = self._little(lam, w)
        return QueueAnalysis("CXL_DIMM", L, lam, 0, w, 0)

    def interference_report(self, snapshots: list) -> str:
        lines = ["PFAnalyzer Queue Length Report "]
        for snap in snapshots[::max(1, len(snapshots)//5)]:
            analysis = self.analyze(snap)
            culprit  = self.culprit_path(analysis)
            lines.append(f"\n  Snapshot {snap.snapshot_id} (cycle {snap.cycle}):")
            lines.append(f"  Culprit: {culprit.name}")
            for nt, qa in analysis.items():
                flag = " ← BOTTLENECK" if nt == culprit else ""
                lines.append(f"    {qa.component:<15}: L={qa.queue_length:6.2f}{flag}")
        return "\n".join(lines)


# PFMaterializer  (S 4.6)

@dataclass
class LocalityWindow:
    """A stable execution window with consistent memory access pattern."""
    start_snap: int
    end_snap:   int
    avg_l1d_hit: float
    avg_llc_hit: float
    avg_cxl_hit: float
    phase_label: str = ""


class PFMaterializer:
    """
    Cross-snapshot time-series analysis.
    Clusters snapshots into stable execution windows,
    detects locality changes, and identifies co-location interference.

    Implements S 4.6: time-series clustering, Holt-Winters-style trend,
    and cross-workload correlation.
    """

    def __init__(self, clos: ClosNetwork):
        self.clos = clos

    def locality_timeline(self, snapshots: list) -> List[dict]:
        """Returns per-snapshot locality metrics for plotting (Case 6)."""
        timeline = []
        for snap in snapshots:
            agg = {}
            for c in snap.cores:
                for k in ("l1d.hit", "l1d.miss", "lfb.hit",
                          "l2.drd_hit", "l2.drd_miss"):
                    agg[k] = agg.get(k, 0) + c.get(k, 0)
            cha = snap.cha
            timeline.append({
                "cycle":         snap.cycle,
                "l1d_hit":       agg.get("l1d.hit",  0),
                "l1d_miss":      agg.get("l1d.miss", 0),
                "lfb_hit":       agg.get("lfb.hit",  0),
                "l2_access":     agg.get("l2.drd_hit",0) + agg.get("l2.drd_miss",0),
                "llc_hit":       cha.get("cha.tor_drd_hit", 0),
                "llc_miss":      cha.get("cha.tor_drd_miss", 0),
                "cxl_hit":       cha.get("cha.tor_drd_cxl", 0),
                "cxl_bw":        snap.cxl.get("cxl.txc_ins_data", 0) * 64,
            })
        return timeline

    def detect_windows(self, timeline: List[dict],
                       metric: str = "llc_hit",
                       min_window: int = 3) -> List[LocalityWindow]:
        """
        Cluster timeline into stable phases (similar hit distributions).
        Simple threshold-based change detection.
        """
        if not timeline:
            return []

        windows: List[LocalityWindow] = []
        w_start = 0
        prev_val = timeline[0].get(metric, 0)

        for i, point in enumerate(timeline[1:], 1):
            val = point.get(metric, 0)
            change = abs(val - prev_val) / max(prev_val, 1)
            if change > 0.20 or i == len(timeline) - 1:  # 20% change = new phase
                if i - w_start >= min_window:
                    vals = [t.get(metric, 0) for t in timeline[w_start:i]]
                    avg = sum(vals) / len(vals)
                    windows.append(LocalityWindow(
                        start_snap=w_start, end_snap=i,
                        avg_l1d_hit=avg,
                        avg_llc_hit=sum(t.get("llc_hit", 0) for t in timeline[w_start:i]) / (i - w_start),
                        avg_cxl_hit=sum(t.get("cxl_hit", 0) for t in timeline[w_start:i]) / (i - w_start),
                        phase_label=f"phase_{len(windows)}",
                    ))
                w_start = i
                prev_val = val

        return windows

    def pearson_correlation(self, x: List[float], y: List[float]) -> float:
        """Pearson-r between two metric timelines."""
        n = min(len(x), len(y))
        if n < 2:
            return 0.0
        mx = sum(x[:n]) / n; my = sum(y[:n]) / n
        num = sum((xi - mx) * (yi - my) for xi, yi in zip(x[:n], y[:n]))
        dx  = math.sqrt(sum((xi - mx)**2 for xi in x[:n]))
        dy  = math.sqrt(sum((yi - my)**2 for yi in y[:n]))
        return num / max(dx * dy, 1e-12)

    def bandwidth_correlation(self, snapshots: list) -> float:
        """
        Case 5: Pearson-r between CXL request frequency and bandwidth.
        Paper reports r=0.998.
        """
        req_freq = [s.m2pcie.get("m2pcie.rxc_ins", 0) for s in snapshots]
        bw       = [s.cxl.get("cxl.txc_ins_data", 0) * 64 for s in snapshots]
        return self.pearson_correlation(req_freq, bw)

    def summary_report(self, snapshots: list) -> str:
        timeline = self.locality_timeline(snapshots)
        windows  = self.detect_windows(timeline)
        r        = self.bandwidth_correlation(snapshots)

        lines = ["PFMaterializer Cross-Snapshot Analysis ",
                 f"  Total snapshots  : {len(snapshots)}",
                 f"  Stable phases    : {len(windows)}",
                 f"  BW-Req Pearson-r : {r:.4f}",
                 "  Phases:"]
        for w in windows:
            lines.append(f"    {w.phase_label}: snaps {w.start_snap}–{w.end_snap} "
                         f"L1D_hit_avg={w.avg_l1d_hit:.0f} "
                         f"LLC_hit_avg={w.avg_llc_hit:.0f} "
                         f"CXL_hit_avg={w.avg_cxl_hit:.0f}")
        return "\n".join(lines)