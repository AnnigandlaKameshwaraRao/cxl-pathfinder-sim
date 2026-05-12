# CXL PathFinder Simulation

A cycle-driven simulation of the CXL.mem protocol stack, implementing the PathFinder profiler described in:

**"Understanding and Profiling CXL.mem Using PathFinder"**
Xiao Li, Zerui Guo, Yuebin Bai, Mahesh Ketkar, Hugh Wilkinson, Ming Liu - ACM SIGCOMM 2025, Coimbra, Portugal 

Paper Link: https://doi.org/10.1145/3718958.3750479

Original Arttfact Link: https://github.com/netlab-wisconsin/PathFinder

This project was built as part of a Bachelor's Thesis Project (BTP) to recreate and extend the paper's results through simulation, without access to physical CXL hardware.

---

## What this simulates

The full CXL.mem hardware stack from CPU cores down to the CXL DIMM:

```
        Core (SB -> LFB -> L1D -> L2)
                    |
    CHA / LLC  (TOR queue, snoop filter, MESIF coherence)
                    |
             Mesh Interconnect
          |                     |
     IMC (local DRAM)     M2PCIe (CXL path)
          |                     |
     Local DIMM            FlexBus I/O
                                |
                        CXL DIMM (Type-3)
```

Each hardware module carries PMU counters matching Tables 1-4 of the paper. The PathFinder profiler runs on top as four components: PFBuilder, PFEstimator, PFAnalyzer, PFMaterializer.

---

## Project structure

```
cxl_pathfinder_sim/
├── config/
│   ├── spr_agilex.yaml          # Intel SPR + Intel Agilex CXL DIMM
│   └── emr_cz120.yaml           # Intel EMR + Micron CZ120 CXL DIMMs
├── hardware/
│   ├── core.py                  # SB, LFB, L1D, L2, PMU counters
│   ├── cha.py                   # CHA, LLC slices, TOR, snoop filter
│   └── uncore.py                # Mesh, IMC, M2PCIe, FlexBus, DIMMs
├── simulation/
│   ├── request.py               # MemRequest, ReqType, MemTarget
│   └── engine.py                # Cycle-driven simulation loop
├── pathfinder/
│   ├── clos_network.py          # Clos graph G=(V,E), mFlow, Path
│   └── profiler.py              # PFBuilder, PFEstimator, PFAnalyzer, PFMaterializer
├── workloads/
│   └── benchmark_profiles.py   # 24 benchmark profiles + MBW/GUPS generators
├── analysis/
│   └── plotter.py               # Figure generation for all cases
├── results/                     # results produced by simulation
├── run_cases.py                 # Main runner (Cases 1-12)
└── requirements.txt
```

---

## Setup

Python 3.10 or higher is recommended.

```bash
git clone https://github.com/AnnigandlaKameshwaraRao/cxl-pathfinder-sim
cd cxl-pathfinder-sim
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Running the cases

### Run all cases (paper's 7 + 2 new)

```bash
python3 run_cases.py
```

### Run a single case

```bash
python3 run_cases.py --case 1        # Path Classification
python3 run_cases.py --case 2        # Pipeline Stall Breakdown
python3 run_cases.py --case 3        # Local vs CXL Interference
python3 run_cases.py --case 4        # Concurrent CXL Contention
python3 run_cases.py --case 5        # CXL Bandwidth Partition
python3 run_cases.py --case 6        # Data Locality
python3 run_cases.py --case 7        # Performance Optimization (TPP)
python3 run_cases.py --case 8        # Prefetcher Efficacy       [new]
python3 run_cases.py --case 9        # SPR vs EMR Comparison     [new]
```

### Change hardware config

```bash
python3 run_cases.py --config spr_agilex     # Intel SPR + Agilex (default)
python3 run_cases.py --config emr_cz120      # Intel EMR + Micron CZ120
```

### Tune key parameters

```bash
# Fraction of LLC misses served by CXL DIMM (0.0 = all local, 1.0 = all CXL)
python3 run_cases.py --cxl-ratio 0.4

# Simulation length in cycles (more = smoother results, slower)
python3 run_cases.py --cycles 50000

# Reproducibility seed
python3 run_cases.py --seed 123

# Combine options
python3 run_cases.py --case 3 --cycles 40000 --cxl-ratio 0.5 --config emr_cz120
```

### Check calibration against paper values

```bash
python3 calibrate.py                             # GCC workload, SPR config
python3 calibrate.py --workload MCF              # different workload
python3 calibrate.py --workload WATER --raw      # also print raw counters
python3 calibrate.py --all-workloads             # run all 6 and compare
python3 calibrate.py --cycles 60000              # longer run for accuracy
```

All output figures go to `results/`.

---

## Cases from the paper (1-7)

| Case | Title | PathFinder component |
|------|-------|---------------------|
| 1 | Path Classification | PFBuilder |
| 2 | Pipeline Stall Breakdown | PFEstimator |
| 3 | Local vs CXL Interference | PFAnalyzer |
| 4 | Concurrent CXL Contention | PFEstimator + PFAnalyzer |
| 5 | CXL Bandwidth Partition | PFAnalyzer + PFMaterializer |
| 6 | Data Locality | PFMaterializer |
| 7 | Performance Optimization (TPP) | All four |

## New cases (8-12)

| Case | Title | What it adds |
|------|-------|-------------|
| 8 | Prefetcher Efficacy | Sweeps HWPF fraction, finds crossover where prefetch hurts more than it helps |
| 9 | SPR vs EMR Comparison | Side-by-side stall comparison, attributes improvement to EMR's larger LLC |

---

## Hardware configs

Both configs mirror the paper's experimental testbeds (S 5.1).

**spr_agilex.yaml** — Intel Sapphire Rapids + Intel Agilex I-Series CXL DIMM
- 32 cores at 2.0 GHz, 60 MB LLC, 256 GB DDR5
- CXL DIMM: 16 GB DDR4, 355 ns random latency, 17.6 GB/s bandwidth

**emr_cz120.yaml** — Intel Emerald Rapids + Micron CZ120 CXL DIMMs
- 32 cores at 2.1 GHz, 160 MB LLC, 1536 GB DDR5
- CXL DIMM: 256 GB DDR5, 320 ns random latency, 38.4 GB/s bandwidth

---

## Benchmark profiles

24 profiles tuned to approximate real benchmark memory access patterns:

- SPEC CPU 2017: GCC, MCF, ROMS, CAC, BWA, FOTS, DEEP, XZ, LBM, OMN
- PARSEC: BODY, RAY, WATER, VOL, BARN, FFT, FREQ
- GAP graph: BFS, CC
- Redis / YCSB: YCSB-A, YCSB-B, YCSB-C, YCSB-D, YCSB-F

Synthetic workloads: MBWWorkload (sequential bandwidth), GUPSWorkload (random updates).

Each profile controls: working set size, read/write/RFO/HWPF fractions, spatial locality, hot-set probability, and CXL miss bias.

---

## Calibration targets

Key ratios from the paper that the simulation reproduces:

| Metric | Paper value | Source |
|--------|------------|--------|
| SB stall increase (CXL vs local) | 1.9x | S3.2, Fig 2a |
| L1D stall increase | 2.1x | S3.2, Fig 2b |
| L2 stall increase | 2.7x | S3.2, Fig 2e |
| LLC stall increase | 2.1x | S3.3, Fig 3a |
| LLC DRd miss increase | 4.2x | S3.3, Fig 3b |
| CXL-served fraction of DRd misses | 38.4% | S3.3, Fig 3c |
| Pearson-r (BW vs CXL req freq) | 0.998 | S5.6, Fig 11b |

---

## Tuning the model

All hardware parameters are in YAML configs. To change the model behaviour:

**Change CXL latency:**
```yaml
# config/spr_agilex.yaml
cxl_dimm:
  random_latency_ns: 355.3
  latency_cycles: 710
```

**Change LLC size:**
```yaml
cha:
  llc_total_mb: 60
```

**Change L1D CXL hit-rate penalty** (controls L1D stall ratio):
```python
# hardware/core.py, _l1d_access()
if req.is_cxl:
    p *= 0.48    # decrease to increase stall ratio
```

**Change CXL miss routing fractions** (controls Fig 3c):
```python
# hardware/cha.py, _route_miss()
cxl_fracs = {ReqType.DRD: 0.384, ReqType.RFO: 0.041, ...}
```

---

## Requirements

```
numpy>=1.24
matplotlib>=3.7
pyyaml>=6.0
pandas>=2.0
scipy>=1.11
```

---

## Reference

Xiao Li, Zerui Guo, Yuebin Bai, Mahesh Ketkar, Hugh Wilkinson, Ming Liu.
"Understanding and Profiling CXL.mem Using PathFinder."
In ACM SIGCOMM 2025 Conference, September 8-11, 2025, Coimbra, Portugal.
https://doi.org/10.1145/3718958.3750479

PathFinder open-source artifact: https://github.com/netlab-wisconsin/PathFinder
