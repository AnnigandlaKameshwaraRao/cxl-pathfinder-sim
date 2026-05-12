"""
PathFinder Clos network model (§4.2).
Models the server as G=(V,E) where V=hardware modules, E=interconnect links.
Defines mFlow and Path as the paper's abstractions.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple
import itertools

from simulation.request import MemRequest, ReqType, MemTarget


# Graph nodes (hardware modules = Clos stages)

class NodeType(Enum):
    CORE    = auto()
    SB      = auto()
    LFB     = auto()
    L1D     = auto()
    L2      = auto()
    CHA     = auto()
    LLC     = auto()
    MESH    = auto()
    IMC     = auto()
    M2PCIE  = auto()
    FLEXBUS = auto()
    LOCAL_DIMM = auto()
    CXL_DIMM   = auto()


# Ordered stage sequence for CXL path (matches Figure 5b of paper)
CXL_PATH_STAGES = [
    NodeType.CORE, NodeType.SB, NodeType.L1D, NodeType.LFB,
    NodeType.L2, NodeType.CHA, NodeType.MESH,
    NodeType.M2PCIE, NodeType.FLEXBUS, NodeType.CXL_DIMM
]

LOCAL_PATH_STAGES = [
    NodeType.CORE, NodeType.SB, NodeType.L1D, NodeType.LFB,
    NodeType.L2, NodeType.CHA, NodeType.MESH,
    NodeType.IMC, NodeType.LOCAL_DIMM
]


@dataclass
class Node:
    node_type: NodeType
    node_id:   int        # e.g. core_id, cha_id
    stage:     int        # Clos stage index

    def __hash__(self):
        return hash((self.node_type, self.node_id))

    def __eq__(self, other):
        return (self.node_type == other.node_type and
                self.node_id   == other.node_id)

    @property
    def label(self) -> str:
        return f"{self.node_type.name}[{self.node_id}]"


@dataclass
class Edge:
    src:       Node
    dst:       Node
    bandwidth: float = 0.0    # GB/s
    latency:   int   = 0      # cycles

    def __hash__(self):
        return hash((self.src, self.dst))


# mFlow

@dataclass
class Path:
    """
    One data path within an mFlow: e.g. Core_0 -> CXL_DIMM via DRd.
    Tracks quantitative traffic load (number of requests traversing).
    Corresponds to paper's W_{v_i}^{e_k} <- v_j notation.
    """
    path_id:    int
    req_type:   ReqType
    stages:     List[NodeType]
    target:     MemTarget

    # Traffic counters
    request_count: int = 0
    total_latency: int = 0
    stall_cycles_per_stage: Dict[NodeType, int] = field(default_factory=dict)
    hit_counts:             Dict[NodeType, int] = field(default_factory=dict)
    miss_counts:            Dict[NodeType, int] = field(default_factory=dict)

    @property
    def avg_latency(self) -> float:
        return self.total_latency / max(self.request_count, 1)

    @property
    def is_cxl(self) -> bool:
        return self.target == MemTarget.CXL_DRAM

    def record_request(self, req: MemRequest):
        self.request_count += 1
        self.total_latency += req.total_latency

        bd = req.stall_breakdown()
        for stage_name, cycles in bd.items():
            ntype = _stage_name_to_node(stage_name)
            if ntype:
                self.stall_cycles_per_stage[ntype] = (
                    self.stall_cycles_per_stage.get(ntype, 0) + cycles)

        # Hit/miss at each level
        for ntype, hit in [(NodeType.L1D, req.hit_l1d),
                           (NodeType.LFB, req.hit_lfb),
                           (NodeType.L2,  req.hit_l2),
                           (NodeType.CHA, req.hit_llc)]:
            if hit:
                self.hit_counts[ntype]  = self.hit_counts.get(ntype, 0) + 1
            else:
                self.miss_counts[ntype] = self.miss_counts.get(ntype, 0) + 1

    def traffic_load(self) -> int:
        return self.request_count


def _stage_name_to_node(name: str) -> Optional[NodeType]:
    mapping = {
        "SB": NodeType.SB, "L1D": NodeType.L1D, "LFB": NodeType.LFB,
        "L2": NodeType.L2, "LLC/CHA": NodeType.CHA, "Mesh": NodeType.MESH,
        "IMC": NodeType.IMC, "FlexBus+MC": NodeType.M2PCIE,
        "CXL_DIMM": NodeType.CXL_DIMM,
    }
    return mapping.get(name)


@dataclass
class MFlow:
    """
    Memory flow: Core_i <-> DIMM_j (local or CXL).
    Lifetime aligns with the workload; location-sensitive; bidirectional.
    mFlow number bounded by Core# × DIMM# (§4.2).
    """
    mflow_id:  int
    core_id:   int
    dimm_id:   int
    is_cxl:    bool
    pid:       int = 0         # process/workload ID

    paths: Dict[Tuple[ReqType, MemTarget], Path] = field(default_factory=dict)
    snapshots: List[dict] = field(default_factory=list)

    _path_counter: int = field(default=0, repr=False)

    def get_or_create_path(self, req_type: ReqType, target: MemTarget) -> Path:
        key = (req_type, target)
        if key not in self.paths:
            stages = CXL_PATH_STAGES if self.is_cxl else LOCAL_PATH_STAGES
            self.paths[key] = Path(
                path_id=self._path_counter,
                req_type=req_type,
                stages=stages,
                target=target,
            )
            self._path_counter += 1
        return self.paths[key]

    def record(self, req: MemRequest):
        target = req.target or (MemTarget.CXL_DRAM if req.is_cxl else MemTarget.LOCAL_DRAM)
        path   = self.get_or_create_path(req.req_type, target)
        path.record_request(req)

    def aggregate_traffic(self) -> Dict[ReqType, int]:
        traffic: Dict[ReqType, int] = {}
        for (rtype, _), path in self.paths.items():
            traffic[rtype] = traffic.get(rtype, 0) + path.request_count
        return traffic

    def snapshot(self) -> dict:
        return {
            "mflow_id": self.mflow_id,
            "core_id":  self.core_id,
            "dimm_id":  self.dimm_id,
            "is_cxl":   self.is_cxl,
            "paths": {
                f"{rt.name}->{tgt.name}": {
                    "count":   p.request_count,
                    "avg_lat": p.avg_latency,
                    "stalls":  {k.name: v for k, v in p.stall_cycles_per_stage.items()},
                    "hits":    {k.name: v for k, v in p.hit_counts.items()},
                    "misses":  {k.name: v for k, v in p.miss_counts.items()},
                }
                for (rt, tgt), p in self.paths.items()
            }
        }


# Clos Network Graph

class ClosNetwork:
    """
    Represents G=(V,E) as described in PathFinder §4.2.
    Nodes = hardware modules.  Edges = interconnect links.
    Used by PFBuilder to construct the data path map.
    """

    def __init__(self, cfg: dict):
        self.cfg   = cfg
        self.nodes: Dict[Tuple[NodeType, int], Node] = {}
        self.edges: Set[Edge] = set()
        self.mflows: Dict[int, MFlow] = {}
        self._mflow_counter = 0
        self._build_topology()

    def _build_topology(self):
        c    = self.cfg
        ncores = c["cpu"]["num_cores"]
        ncha   = c["cha"]["num_cha"]

        # Add nodes
        for i in range(ncores):
            for nt, stage in [(NodeType.CORE, 0), (NodeType.SB, 1),
                               (NodeType.L1D, 2), (NodeType.LFB, 3),
                               (NodeType.L2,  4)]:
                self.nodes[(nt, i)] = Node(nt, i, stage)

        for i in range(ncha):
            self.nodes[(NodeType.CHA, i)] = Node(NodeType.CHA, i, 5)
            self.nodes[(NodeType.LLC, i)] = Node(NodeType.LLC, i, 5)

        self.nodes[(NodeType.MESH,      0)] = Node(NodeType.MESH,      0, 6)
        self.nodes[(NodeType.IMC,       0)] = Node(NodeType.IMC,       0, 7)
        self.nodes[(NodeType.M2PCIE,    0)] = Node(NodeType.M2PCIE,    0, 7)
        self.nodes[(NodeType.FLEXBUS,   0)] = Node(NodeType.FLEXBUS,   0, 8)
        self.nodes[(NodeType.LOCAL_DIMM,0)] = Node(NodeType.LOCAL_DIMM,0, 9)
        self.nodes[(NodeType.CXL_DIMM,  0)] = Node(NodeType.CXL_DIMM,  0, 9)

        # Adding edges (simplified - full mesh would be O(n²))
        m   = self.nodes[(NodeType.MESH, 0)]
        imc = self.nodes[(NodeType.IMC,  0)]
        m2p = self.nodes[(NodeType.M2PCIE, 0)]
        fb  = self.nodes[(NodeType.FLEXBUS, 0)]
        ld  = self.nodes[(NodeType.LOCAL_DIMM, 0)]
        cx  = self.nodes[(NodeType.CXL_DIMM,   0)]

        for i in range(ncha):
            cha = self.nodes[(NodeType.CHA, i)]
            self.edges.add(Edge(cha, m))
            self.edges.add(Edge(m,  cha))

        self.edges.update([
            Edge(m, imc), Edge(imc, ld), Edge(ld, imc),
            Edge(m, m2p), Edge(m2p, fb), Edge(fb, cx),
            Edge(cx, fb), Edge(fb, m2p),
        ])

    def create_mflow(self, core_id: int, dimm_id: int, is_cxl: bool, pid: int = 0) -> MFlow:
        mflow = MFlow(self._mflow_counter, core_id, dimm_id, is_cxl, pid)
        self.mflows[self._mflow_counter] = mflow
        self._mflow_counter += 1
        return mflow

    def get_mflow(self, mflow_id: int) -> Optional[MFlow]:
        return self.mflows.get(mflow_id)

    def record_request(self, req: MemRequest):
        mflow = self.mflows.get(req.mflow_id)
        if mflow is None:
            is_cxl = getattr(req, "is_cxl", False)
            mflow  = self.create_mflow(req.core_id, 0, is_cxl)
        mflow.record(req)

    def all_cxl_paths(self) -> List[Path]:
        paths = []
        for mflow in self.mflows.values():
            if mflow.is_cxl:
                paths.extend(mflow.paths.values())
        return paths

    def all_paths(self) -> List[Path]:
        paths = []
        for mflow in self.mflows.values():
            paths.extend(mflow.paths.values())
        return paths