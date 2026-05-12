"""
run_cases.py — PathFinder case studies: original 7 from the paper + 5 new ones.

Cases from paper:
    1  Path Classification          (PFBuilder)
    2  Pipeline Stall Breakdown     (PFEstimator)
    3  Local vs CXL Interference    (PFAnalyzer)
    4  Concurrent CXL Contention    (PFEstimator + PFAnalyzer)
    5  CXL Bandwidth Partition      (PFAnalyzer + PFMaterializer)
    6  Data Locality                (PFMaterializer)
    7  Performance Optimization     (TPP simulation)

New cases (beyond the paper):
    8  Hardware Prefetcher Efficacy -- measures how much HWPF actually helps under CXL (paper notes it can cause extra CXL loads); sweeps prefetch distance and shows net benefit vs overhead.
    9  SPR vs EMR Config Comparison -- runs identical workloads on both hardware configs, overlays stall ratios, and attributes improvement to EMR's larger LLC.

Usage:
    python run_cases.py                        # all cases, SPR config
    python run_cases.py --case 8               # single new case
    python run_cases.py --cycles 30000         # shorter run
    python run_cases.py --cxl-ratio 0.4
    python run_cases.py --config emr_cz120
"""
from __future__ import annotations
import argparse
import copy
import sys
import time
from pathlib import Path

import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simulation.engine import SimulationEngine
from simulation.request import ReqType
from workloads.benchmark_profiles import (
    BenchmarkWorkload, BenchmarkProfile, BENCHMARK_PROFILES,
    MBWWorkload, GUPSWorkload, MultiCoreWorkload,
)
from pathfinder.clos_network import ClosNetwork
from pathfinder.profiler import (PFBuilder, PFEstimator,
                                  PFAnalyzer, PFMaterializer,
                                  StallBreakdown)
from analysis.plotter import (
    plot_case1_path_classification,
    plot_case2_stall_breakdown,
    plot_case3_interference,
    plot_case4_contention,
    plot_case5_bandwidth,
    plot_case6_locality,
    plot_case7_optimization,
    plot_core_pmu_comparison,
    save,
)

OUTDIR = "results"
CYCLES_DEFAULT = 200_000


#  Shared helpers 

def make_engine(cfg: dict, cxl_ratio: float, seed: int = 42) -> SimulationEngine:
    return SimulationEngine(cfg, cxl_ratio=cxl_ratio, seed=seed)


def run_workload(engine: SimulationEngine, wl, cycles: int,
                 name: str = "") -> list:
    print(f"    Running {name or wl.name} ({cycles:,} cycles) ...", end="", flush=True)
    t0 = time.time()
    snaps = engine.run(wl, max_cycles=cycles, workload_name=name or wl.name)
    print(f" done ({time.time()-t0:.1f}s, {len(engine.completed_requests):,} reqs)")
    return snaps


def _profile(key: str, core: int = 0, cxl: bool = True,
             seed: int = 0, rate: int = 2) -> BenchmarkWorkload:
    return BenchmarkWorkload(BENCHMARK_PROFILES[key], core, use_cxl=cxl,
                             seed=seed, issue_rate=rate)


def _avg_stall(snaps: list, cfg: dict, req_type: str = "DRd") -> StallBreakdown:
    clos = ClosNetwork(cfg)
    est  = PFEstimator(clos)
    bds  = est.estimate(snaps, req_type=req_type)
    avg  = StallBreakdown()
    for bd in bds.values():
        avg.sb += bd.sb; avg.l1d += bd.l1d; avg.lfb += bd.lfb
        avg.l2 += bd.l2; avg.llc += bd.llc
        avg.flexbus_mc += bd.flexbus_mc; avg.cxl_dimm += bd.cxl_dimm
    n = max(len(bds), 1)
    avg.sb /= n; avg.l1d /= n; avg.lfb /= n; avg.l2 /= n
    avg.llc /= n; avg.flexbus_mc /= n; avg.cxl_dimm /= n
    return avg


# Cases 1-7: original paper cases (unchanged)

def case1(cfg: dict, cxl_ratio: float, cycles: int):
    print("\n[Case 1] Path Classification (PFBuilder)")
    workload_keys = ["FOTS", "GCC", "MCF", "BWA", "ROMS", "CAC"]
    path_maps = {}; local_snaps = []; cxl_snaps = []
    for key in workload_keys:
        eng = make_engine(cfg, cxl_ratio)
        wl  = _profile(key, core=0, cxl=True)
        snaps = run_workload(eng, wl, cycles, key)
        cxl_snaps.extend(snaps)
        clos = ClosNetwork(cfg)
        path_maps[key] = PFBuilder(clos).build(snaps)
        eng2 = make_engine(cfg, cxl_ratio=0.0)
        wl2  = _profile(key, core=0, cxl=False)
        local_snaps.extend(run_workload(eng2, wl2, cycles, f"{key}_local"))
    plot_case1_path_classification(path_maps, workload_keys, OUTDIR)
    plot_core_pmu_comparison(local_snaps, cxl_snaps, workload_keys, OUTDIR)
    return path_maps


def case2(cfg: dict, cxl_ratio: float, cycles: int):
    print("\n[Case 2] Pipeline Stall Breakdown (PFEstimator)")
    workload_keys = ["BFS", "CC", "FREQ", "RAY", "BARN", "FFT"]
    stall_results: dict = {}
    for rtype_name in ["DRd", "RFO", "HW PF", "DWr"]:
        stall_results[rtype_name] = {}
        for key in workload_keys:
            eng   = make_engine(cfg, cxl_ratio)
            wl    = _profile(key, core=0, cxl=True)
            snaps = run_workload(eng, wl, cycles, key)
            avg   = _avg_stall(snaps, cfg, rtype_name)
            stall_results[rtype_name][key] = avg
            print(f"      {key} [{rtype_name}]: total={avg.total():.0f}  "
                  f"FlexBus={avg.flexbus_mc:.0f}  CXL={avg.cxl_dimm:.0f}")
    plot_case2_stall_breakdown(stall_results, OUTDIR)
    return stall_results


def case3(cfg: dict, cxl_ratio: float, cycles: int):
    print("\n[Case 3] Local vs CXL Interference")
    workload_keys = ["WATER", "VOL", "RAY", "BODY"]
    cxl_loads     = [20, 40, 60, 80, 100]
    interference_data: dict = {k: {} for k in workload_keys}
    for wl_key in workload_keys:
        for load_pct in cxl_loads:
            eng      = make_engine(cfg, cxl_ratio=load_pct / 100.0)
            combined = MultiCoreWorkload([_profile(wl_key, 0, False),
                                          _profile(wl_key, 1, True)])
            snaps = run_workload(eng, combined, cycles // 4, f"{wl_key}_L{load_pct}")
            avg_bd = _avg_stall(snaps, cfg, "DRd")
            interference_data[wl_key][load_pct] = {
                "SB": avg_bd.sb, "L1D": avg_bd.l1d, "LFB": avg_bd.lfb,
                "L2": avg_bd.l2, "LLC": avg_bd.llc,
                "FlexBus+MC": avg_bd.flexbus_mc, "CXL_DIMM": avg_bd.cxl_dimm,
            }
    plot_case3_interference(interference_data, OUTDIR)
    return interference_data


def case4(cfg: dict, cxl_ratio: float, cycles: int):
    print("\n[Case 4] Concurrent CXL Access Contention")
    ycsb_keys = ["YCSB_A", "YCSB_B", "YCSB_D", "YCSB_F"]
    cxl_loads  = [20, 40, 60, 80, 100]
    contention_data = {"loads": cxl_loads}
    for yk in ycsb_keys:
        wl_name = BENCHMARK_PROFILES[yk].name
        contention_data[wl_name] = {}
        for load_pct in cxl_loads:
            num_flows = max(1, load_pct // 20)
            eng = make_engine(cfg, cxl_ratio=load_pct / 100.0)
            wls = [_profile(yk, core=i, cxl=True, seed=i) for i in range(num_flows)]
            snaps = run_workload(eng, MultiCoreWorkload(wls), cycles // 4,
                                 f"{wl_name}_L{load_pct}")
            throughput = len(eng.completed_requests) / max(cycles // 4, 1) * 1e4
            avg_bd = _avg_stall(snaps, cfg, "DRd")
            clos = ClosNetwork(cfg); an = PFAnalyzer(clos)
            from pathfinder.clos_network import NodeType
            qa = an.analyze(snaps[-1]) if snaps else {}
            fb_lat  = qa.get(NodeType.M2PCIE,  type("Q",(),{"queue_length":0})()).queue_length
            cha_lat = qa.get(NodeType.CHA,      type("Q",(),{"queue_length":0})()).queue_length
            contention_data[wl_name][load_pct] = {
                "throughput": throughput, "SB": avg_bd.sb, "L1D": avg_bd.l1d,
                "LFB": avg_bd.lfb, "L2": avg_bd.l2, "LLC": avg_bd.llc,
                "CHA_latency": cha_lat, "FlexBus+MC": avg_bd.flexbus_mc,
            }
    plot_case4_contention(contention_data, OUTDIR)
    return contention_data


def case5(cfg: dict, cxl_ratio: float, cycles: int):
    print("\n[Case 5] CXL Bandwidth Partition")
    target_bws = [500, 700, 1000, 3700]
    bw_data = {"MBW_no_full": {}, "MBW_full": {}, "GUPS_no_full": {}, "GUPS_full": {},
                "mbw_scatter": {"req": [], "bw": []}, "gups_scatter": {"req": [], "bw": []}}
    for i, bw in enumerate(target_bws):
        key = f"mFlow-{i+1}"
        for scenario, full in [("no_full", False), ("full", True)]:
            eng = make_engine(cfg, cxl_ratio=0.9 if full else 0.3)
            wl  = MBWWorkload(i, bw / 1000.0, use_cxl=True, seed=i)
            snaps = run_workload(eng, wl, cycles // 4, f"MBW_{bw}_{scenario}")
            actual_bw = (len(eng.completed_requests) * 64 /
                         max(cycles // 4, 1) * cfg["cpu"]["freq_ghz"] * 1e9 / 1e6)
            bw_data[f"MBW_{scenario}"][key] = actual_bw
            if not full:
                rxc = sum(s.m2pcie.get("m2pcie.rxc_ins", 0) for s in snaps)
                bw_data["mbw_scatter"]["req"].append(rxc)
                bw_data["mbw_scatter"]["bw"].append(actual_bw)
    for i, (hot, total) in enumerate([(24,72),(12,48),(6,24),(3,12)]):
        key = f"mFlow-{i+1}"
        for scenario, full in [("no_full", False), ("full", True)]:
            eng = make_engine(cfg, cxl_ratio=0.9 if full else 0.3)
            wl  = GUPSWorkload(i, hot, total, use_cxl=True, seed=i+10)
            snaps = run_workload(eng, wl, cycles // 4, f"GUPS_{hot}_{scenario}")
            actual_bw = (len(eng.completed_requests) * 64 /
                         max(cycles // 4, 1) * cfg["cpu"]["freq_ghz"] * 1e9 / 1e6)
            bw_data[f"GUPS_{scenario}"][key] = actual_bw
            if not full:
                rxc = sum(s.m2pcie.get("m2pcie.rxc_ins", 0) for s in snaps)
                bw_data["gups_scatter"]["req"].append(rxc)
                bw_data["gups_scatter"]["bw"].append(actual_bw)
    clos = ClosNetwork(cfg); mat = PFMaterializer(clos)
    bw_data["pearson_r"] = mat.pearson_correlation(
        bw_data["mbw_scatter"]["req"], bw_data["mbw_scatter"]["bw"])
    print(f"    Pearson-r: {bw_data['pearson_r']:.4f}")
    plot_case5_bandwidth(bw_data, OUTDIR)
    return bw_data


def case6(cfg: dict, cxl_ratio: float, cycles: int):
    print("\n[Case 6] Data Locality (PFMaterializer)")
    clos = ClosNetwork(cfg); mat = PFMaterializer(clos)
    locality_timelines = {}
    eng = make_engine(cfg, cxl_ratio)
    bwaves = _profile("BWA", 0, True, seed=1)
    snaps_a1 = run_workload(eng, bwaves, cycles // 2, "bwaves")
    lbm = _profile("LBM", 1, False, seed=2)
    snaps_a2 = run_workload(eng, MultiCoreWorkload([bwaves, lbm]), cycles // 2, "bwaves+lbm")
    locality_timelines["bwaves_lbm"] = mat.locality_timeline(snaps_a1 + snaps_a2)
    locality_timelines["bwaves_lbm_events"] = [(cycles // 2, "LBM")]
    eng2 = make_engine(cfg, cxl_ratio)
    snaps_b1 = run_workload(eng2, _profile("BWA", 0, True), cycles // 2, "bwaves")
    roms = _profile("ROMS", 1, True, seed=3)
    snaps_b2 = run_workload(eng2, MultiCoreWorkload([bwaves, roms]), cycles // 2, "bwaves+roms")
    locality_timelines["bwaves_roms"] = mat.locality_timeline(snaps_b1 + snaps_b2)
    locality_timelines["bwaves_roms_events"] = [(cycles // 2, "ROMS")]
    eng3 = make_engine(cfg, cxl_ratio)
    mcf  = _profile("MCF", 2, True, seed=4)
    combo = MultiCoreWorkload([bwaves, lbm, mcf, roms])
    snaps_c = run_workload(eng3, combo, cycles, "bwaves+multi")
    locality_timelines["bwaves_multi"] = mat.locality_timeline(snaps_c)
    locality_timelines["bwaves_multi_events"] = [
        (cycles//4,"LBM"),(cycles//2,"MCF"),(3*cycles//4,"ROM")]
    plot_case6_locality(locality_timelines, OUTDIR)
    return locality_timelines


def case7(cfg: dict, cxl_ratio: float, cycles: int):
    print("\n[Case 7] Performance Optimization (TPP)")
    opt_data = {}
    targets  = ["YCSB_C", "FOTS", "GUPS"]
    tpp_ratio = max(0.05, cxl_ratio * 0.15)
    for key in targets:
        wl_name = BENCHMARK_PROFILES[key].name if key in BENCHMARK_PROFILES else key
        eng_no = make_engine(cfg, cxl_ratio)
        wl_no  = (_profile(key, 0, True) if key in BENCHMARK_PROFILES
                  else GUPSWorkload(0, 24, 72, use_cxl=True))
        snaps_no = run_workload(eng_no, wl_no, cycles // 2, f"{wl_name}_no_tpp")
        eng_tp = make_engine(cfg, tpp_ratio)
        wl_tp  = (_profile(key, 0, True) if key in BENCHMARK_PROFILES
                  else GUPSWorkload(0, 24, 72, use_cxl=True))
        snaps_tp = run_workload(eng_tp, wl_tp, cycles // 2, f"{wl_name}_tpp")
        def _hit_dict(snaps):
            agg = {}
            for snap in snaps:
                for c in snap.cores:
                    for k in ("l2.drd_hit","l2.rfo_hit","l2.hwpf_hit"):
                        agg[k] = agg.get(k, 0) + c.get(k, 0)
                agg["cha.drd_cxl"] = agg.get("cha.drd_cxl",0) + snap.cha.get("cha.tor_drd_cxl",0)
                agg["m2p.ld"] = agg.get("m2p.ld",0) + snap.m2pcie.get("m2pcie.txc_bl",0)
                agg["m2p.st"] = agg.get("m2p.st",0) + snap.m2pcie.get("m2pcie.txc_ak",0)
            return agg
        x_keys = ["DRd-L","RFO-L","HWPF-L","DRd-C","RFO-C","HWPF-C","M2P-LD","M2P-ST"]
        km = {"DRd-L":"l2.drd_hit","RFO-L":"l2.rfo_hit","HWPF-L":"l2.hwpf_hit",
              "DRd-C":"cha.drd_cxl","RFO-C":"cha.drd_cxl","HWPF-C":"cha.drd_cxl",
              "M2P-LD":"m2p.ld","M2P-ST":"m2p.st"}
        agg_no = _hit_dict(snaps_no); agg_tp = _hit_dict(snaps_tp)
        opt_data[f"{wl_name}_no_tpp"] = {k: agg_no.get(km[k], 0) for k in x_keys}
        opt_data[f"{wl_name}_tpp"]    = {k: agg_tp.get(km[k], 0) for k in x_keys}
        bd_no = _avg_stall(snaps_no, cfg, "DRd"); bd_tp = _avg_stall(snaps_tp, cfg, "DRd")
        sl_keys = ["CHA-DRd","CHA-RFO","CHA-HWPF","CHA-DWr","FMC-DRd","FMC-RFO","FMC-HWPF"]
        opt_data[f"{wl_name}_stall_no_tpp"] = {k: bd_no.llc if "CHA" in k else bd_no.flexbus_mc for k in sl_keys}
        opt_data[f"{wl_name}_stall_tpp"]    = {k: bd_tp.llc if "CHA" in k else bd_tp.flexbus_mc for k in sl_keys}
        imp = (len(eng_tp.completed_requests) - len(eng_no.completed_requests)) / max(len(eng_no.completed_requests), 1) * 100
        print(f"    {wl_name}: TPP improvement = {imp:+.1f}%")
    plot_case7_optimization(opt_data, OUTDIR)
    return opt_data


# Case 8 — Hardware Prefetcher Efficacy under CXL  [NEW]

def case8(cfg: dict, cxl_ratio: float, cycles: int):
    """
    Sweep hardware prefetch fraction (0% to 40% of requests are HWPF).
    Measure net effect on total stall cycles: prefetch hides latency but
    also issues extra CXL loads that consume FlexBus credits.

    The paper (§2.2, §3.2) notes that HWPF-triggered CXL loads can
    crowd out demand reads.  This case quantifies that crossover point.
    """
    print("\n[Case 11] Hardware Prefetcher Efficacy under CXL  [NEW]")
    hwpf_fracs  = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
    workloads   = ["ROMS", "BWA", "GCC", "MCF"]
    results: dict = {wl: {} for wl in workloads}

    base_profiles = {k: copy.deepcopy(BENCHMARK_PROFILES[k]) for k in workloads}

    for hwpf_frac in hwpf_fracs:
        for wl_key in workloads:
            # Patch profile: redistribute hwpf fraction from read fraction
            p = copy.deepcopy(base_profiles[wl_key])
            transfer = hwpf_frac - p.hwpf_fraction
            p.hwpf_fraction = hwpf_frac
            p.read_fraction  = max(0.05, p.read_fraction - transfer)

            eng   = make_engine(cfg, cxl_ratio)
            wl    = BenchmarkWorkload(p, core_id=0, use_cxl=True, seed=7)
            snaps = run_workload(eng, wl, cycles // 4,
                                 f"{wl_key}_hwpf{int(hwpf_frac*100)}pct")
            bd    = _avg_stall(snaps, cfg, "HW PF")
            bd_drd = _avg_stall(snaps, cfg, "DRd")
            throughput = len(eng.completed_requests) / max(cycles // 4, 1)
            results[wl_key][hwpf_frac] = {
                "hwpf_stall":  bd.total(),
                "drd_stall":   bd_drd.total(),
                "total_stall": bd.total() + bd_drd.total(),
                "throughput":  throughput,
                "flexbus_mc":  bd.flexbus_mc + bd_drd.flexbus_mc,
            }
            print(f"      {wl_key} hwpf={hwpf_frac:.0%}: "
                  f"HWPF_stall={bd.total():.0f}  DRd_stall={bd_drd.total():.0f}  "
                  f"tput={throughput:.2f}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Case 11: Hardware Prefetcher Efficacy under CXL (New)\n"
                 "Net stall cycles and throughput vs HWPF fraction", fontsize=11)
    colors = ["#4878CF","#D65F5F","#6ACC65","#EE854A"]
    markers = ["o","s","^","D"]

    for ax_idx, (metric, ylabel) in enumerate([
        ("total_stall", "Total stall cycles"),
        ("throughput",  "Throughput (req/cycle)"),
        ("flexbus_mc",  "FlexBus+MC stall cycles"),
        ("hwpf_stall",  "HWPF-induced stall cycles"),
    ]):
        ax = axes.flat[ax_idx]
        for wl_key, col, mk in zip(workloads, colors, markers):
            xs = sorted(results[wl_key].keys())
            ys = [results[wl_key][x][metric] for x in xs]
            ax.plot([x*100 for x in xs], ys, f"{mk}-", color=col,
                    label=wl_key, linewidth=1.5, markersize=5)
        ax.set_xlabel("HWPF fraction (%)")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=8)
        # Mark the crossover (where total_stall is minimised)
        if metric == "total_stall":
            for wl_key, col in zip(workloads, colors):
                xs = sorted(results[wl_key].keys())
                ys = [results[wl_key][x][metric] for x in xs]
                best = xs[ys.index(min(ys))]
                ax.axvline(best*100, color=col, linestyle=":", linewidth=0.8, alpha=0.6)

    plt.tight_layout()
    save(fig, f"{OUTDIR}/case11_prefetcher_efficacy.png")
    return results


# Case 9 — SPR vs EMR Hardware Config Comparison  [NEW]

def case9(cfg_spr: dict, cfg_emr: dict, cxl_ratio: float, cycles: int):
    """
    Run identical workloads on SPR (60 MB LLC) and EMR (160 MB LLC) configs.
    Overlay key stall ratios to show how the larger EMR LLC reduces CXL impact.

    The paper validates on both platforms (§3.6) but does not produce a
    direct side-by-side comparison figure. This case fills that gap.
    """
    print("\n[Case 12] SPR vs EMR Hardware Config Comparison  [NEW]")
    workloads = ["GCC", "MCF", "BWA", "FFT", "BFS", "WATER"]
    configs   = {"SPR (60 MB LLC)": cfg_spr, "EMR (160 MB LLC)": cfg_emr}
    req_types = ["DRd", "RFO", "HW PF"]

    # results[config_label][wl_key][req_type] = StallBreakdown
    results: dict = {cl: {wl: {} for wl in workloads} for cl in configs}

    for cfg_label, cfg_hw in configs.items():
        for wl_key in workloads:
            for rt in req_types:
                eng   = make_engine(cfg_hw, cxl_ratio)
                wl    = _profile(wl_key, 0, True, seed=5)
                snaps = run_workload(eng, wl, cycles // 4,
                                     f"{cfg_label[:3]}_{wl_key}_{rt}")
                bd    = _avg_stall(snaps, cfg_hw, rt)
                results[cfg_label][wl_key][rt] = bd
                print(f"      [{cfg_label[:3]}] {wl_key} [{rt}]: "
                      f"total={bd.total():.0f}  CXL={bd.cxl_dimm:.0f}")

    # Plot: for each request type, bar chart of total stall, grouped SPR vs EMR
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Case 12: SPR vs EMR Config Comparison (New)\n"
                 "CXL-induced stall cycles: 60 MB vs 160 MB LLC", fontsize=11)
    cfg_colors = {"SPR (60 MB LLC)": "#4878CF", "EMR (160 MB LLC)": "#D65F5F"}
    x = np.arange(len(workloads)); w = 0.35

    for ax, rt in zip(axes, req_types):
        for i, (cfg_label, col) in enumerate(cfg_colors.items()):
            vals = [results[cfg_label][wl][rt].total() for wl in workloads]
            ax.bar(x + (i - 0.5) * w, vals, w, label=cfg_label,
                   color=col, edgecolor="white", linewidth=0.5)
        ax.set_title(f"{rt} path")
        ax.set_xticks(x); ax.set_xticklabels(workloads, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Total stall cycles" if rt == "DRd" else "")
        ax.legend(frameon=False, fontsize=8)

        # Annotate reduction percentage
        for j, wl in enumerate(workloads):
            spr_val = results["SPR (60 MB LLC)"][wl][rt].total()
            emr_val = results["EMR (160 MB LLC)"][wl][rt].total()
            if spr_val > 0:
                pct = (spr_val - emr_val) / spr_val * 100
                ax.text(j, max(spr_val, emr_val) * 1.02,
                        f"-{pct:.0f}%", ha="center", fontsize=6, color="#333333")

    plt.tight_layout()
    save(fig, f"{OUTDIR}/case12_spr_vs_emr.png")
    return results


#  Main 

def main():
    parser = argparse.ArgumentParser(description="CXL PathFinder Simulation")
    parser.add_argument("--config",    default="spr_agilex",
                        help="Primary config (spr_agilex | emr_cz120)")
    parser.add_argument("--case",      type=int, default=0,
                        help="Run single case 1-9 (0 = all)")
    parser.add_argument("--cycles",    type=int, default=CYCLES_DEFAULT)
    parser.add_argument("--cxl-ratio", type=float, default=0.3)
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    config_path = f"config/{args.config}.yaml"
    print(f"\nCXL PathFinder Simulation")
    print(f"  Config     : {config_path}")
    print(f"  CXL ratio  : {args.cxl_ratio:.0%}")
    print(f"  Cycles     : {args.cycles:,}")
    print(f"  Seed       : {args.seed}")
    print(f"  Output dir : {OUTDIR}/")
    Path(OUTDIR).mkdir(exist_ok=True)

    with open(config_path) as f:
        cfg_spr = yaml.safe_load(f)
    cfg_spr["simulation"]["seed"] = args.seed

    with open("config/emr_cz120.yaml") as f:
        cfg_emr = yaml.safe_load(f)
    cfg_emr["simulation"]["seed"] = args.seed

    # Case 9 always uses SPR as primary; pick the right one for cases 1-8
    cfg = cfg_spr

    cases = {
        1:  lambda: case1(cfg,  args.cxl_ratio, args.cycles),
        2:  lambda: case2(cfg,  args.cxl_ratio, args.cycles),
        3:  lambda: case3(cfg,  args.cxl_ratio, args.cycles),
        4:  lambda: case4(cfg,  args.cxl_ratio, args.cycles),
        5:  lambda: case5(cfg,  args.cxl_ratio, args.cycles),
        6:  lambda: case6(cfg,  args.cxl_ratio, args.cycles),
        7:  lambda: case7(cfg,  args.cxl_ratio, args.cycles),
        8:  lambda: case8(cfg,  args.cxl_ratio, args.cycles),
        9:  lambda: case9(cfg_spr, cfg_emr, args.cxl_ratio, args.cycles),
    }

    to_run = list(cases.keys()) if args.case == 0 else [args.case]
    summary_lines = [
        f"PathFinder Simulation Summary  --  {args.config}",
        f"CXL ratio: {args.cxl_ratio:.0%}   Cycles: {args.cycles:,}",
        "-" * 55,
    ]

    t_total = time.time()
    for n in to_run:
        t0 = time.time()
        try:
            cases[n]()
            elapsed = time.time() - t0
            tag = "[NEW]" if n >= 8 else "     "
            summary_lines.append(f"  Case {n:>2} {tag}: OK   ({elapsed:.1f}s)")
        except Exception as e:
            summary_lines.append(f"  Case {n:>2}      : FAILED -- {e}")
            import traceback; traceback.print_exc()

    summary_lines.append(f"\nTotal time : {time.time()-t_total:.1f}s")
    summary_lines.append(f"Figures    : {OUTDIR}/")
    summary = "\n".join(summary_lines)
    print("\n" + summary)
    with open(f"{OUTDIR}/summary.txt", "w") as f:
        f.write(summary + "\n")


if __name__ == "__main__":
    main()