"""
Uncore hardware: Mesh interconnect, IMC (local DIMM), M2PCIe (CXL path),
FlexBus, Local DIMM, and CXL DIMM.
Counters mirror Tables 3 and 4 of the PathFinder paper.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, List, Optional
import random

from simulation.request import MemRequest, ReqType, MemTarget


# Mesh Interconnect 

@dataclass
class MeshCounters:
    cycles_active: int = 0
    requests_routed: int = 0
    stall_cycles: int = 0

    def snapshot(self) -> dict:
        return {
            "mesh.requests":  self.requests_routed,
            "mesh.stall":     self.stall_cycles,
        }


class MeshInterconnect:
    """
    Models the on-chip ring/mesh NoC between cores and CHA/MC.
    Adds hop latency per request based on source->destination distance.
    """

    def __init__(self, cfg: dict):
        m = cfg["mesh"]
        self.hop_latency = m["hop_latency_cycles"]
        self.num_stops   = m["num_stops"]
        self.bandwidth   = m["bandwidth_gbps"]
        self.counters    = MeshCounters()
        self._inflight:  Dict[int, int] = {}   # req_id -> completion_cycle

    def route(self, req: MemRequest, src: int, dst: int, cycle: int) -> int:
        """Returns cycle when request arrives at destination."""
        hops = abs(dst - src) % (self.num_stops // 2)
        lat  = hops * self.hop_latency + 1
        req.latency_mesh += lat
        self.counters.requests_routed += 1
        done = cycle + lat
        self._inflight[req.req_id] = done
        return done

    def snapshot(self) -> dict:
        return self.counters.snapshot()


# IMC (Integrated Memory Controller) - Local DIMM path 

@dataclass
class IMCCounters:
    """Table 3: unc_m_rpq/wpq_cycles_ne, unc_m_cas_count, unc_m_rpq/wpq_inserts."""
    rpq_cycles_ne_pch0:  int = 0   # RPQ not-empty cycles ch0
    rpq_cycles_ne_pch1:  int = 0
    wpq_cycles_ne_pch0:  int = 0
    wpq_cycles_ne_pch1:  int = 0
    rpq_inserts_pch0:    int = 0
    rpq_inserts_pch1:    int = 0
    wpq_inserts_pch0:    int = 0
    wpq_inserts_pch1:    int = 0
    cas_rd_pch0:         int = 0
    cas_rd_pch1:         int = 0
    cas_wr_pch0:         int = 0
    cas_wr_pch1:         int = 0
    rpq_occ_pch0:        int = 0
    rpq_occ_pch1:        int = 0
    wpq_occ_pch0:        int = 0
    wpq_occ_pch1:        int = 0

    def snapshot(self) -> dict:
        return {
            "imc.rpq_occ_pch0":    self.rpq_occ_pch0,
            "imc.rpq_occ_pch1":    self.rpq_occ_pch1,
            "imc.wpq_occ_pch0":    self.wpq_occ_pch0,
            "imc.wpq_occ_pch1":    self.wpq_occ_pch1,
            "imc.rpq_ne_pch0":     self.rpq_cycles_ne_pch0,
            "imc.cas_rd":          self.cas_rd_pch0 + self.cas_rd_pch1,
            "imc.cas_wr":          self.cas_wr_pch0 + self.cas_wr_pch1,
        }


class IMC:
    """Local memory controller with RPQ and WPQ queues."""

    def __init__(self, cfg: dict):
        c = cfg["imc"]
        self.rpq_depth   = c["rpq_depth"]
        self.wpq_depth   = c["wpq_depth"]
        self.cas_lat     = c["cas_latency_cycles"]
        self.num_channels = c["num_channels"]
        self.rpq: Deque[MemRequest] = deque()
        self.wpq: Deque[MemRequest] = deque()
        self.counters = IMCCounters()

    def enqueue_read(self, req: MemRequest, cycle: int) -> int:
        self.rpq.append(req)
        self.counters.rpq_inserts_pch0 += 1
        self.counters.rpq_occ_pch0     += len(self.rpq)
        req.latency_imc += max(0, len(self.rpq) - 1) * 2  # queueing delay
        return cycle + self.cas_lat + max(0, len(self.rpq) - 1) * 2

    def enqueue_write(self, req: MemRequest, cycle: int) -> int:
        self.wpq.append(req)
        self.counters.wpq_inserts_pch0 += 1
        self.counters.wpq_occ_pch0     += len(self.wpq)
        req.latency_imc += max(0, len(self.wpq) - 1) * 2
        return cycle + self.cas_lat + max(0, len(self.wpq) - 1) * 2

    def tick(self):
        if self.rpq:
            self.counters.rpq_cycles_ne_pch0 += 1
            self.counters.cas_rd_pch0        += 1
        if self.wpq:
            self.counters.wpq_cycles_ne_pch0 += 1
            self.counters.cas_wr_pch0        += 1

    def snapshot(self) -> dict:
        return self.counters.snapshot()


# M2PCIe (CXL MC / FlexBus interface)

@dataclass
class M2PCIeCounters:
    """Table 3: unc_m2p_rxc_cycles_ne, unc_m2p_rxc_inserts, txc_inserts_ak, txc_inserts_bl."""
    rxc_cycles_ne:  int = 0   # ingress not-empty cycles
    rxc_inserts:    int = 0   # inserts into ingress (CXL requests)
    txc_inserts_ak: int = 0   # egress acks (NDR)
    txc_inserts_bl: int = 0   # egress block data (DRS)

    def snapshot(self) -> dict:
        return {
            "m2pcie.rxc_ne":    self.rxc_cycles_ne,
            "m2pcie.rxc_ins":   self.rxc_inserts,
            "m2pcie.txc_ak":    self.txc_inserts_ak,
            "m2pcie.txc_bl":    self.txc_inserts_bl,
        }


class M2PCIe:
    """FlexBus-side memory controller — handles CXL.mem M2S / S2M transactions."""

    def __init__(self, cfg: dict):
        c = cfg["m2pcie"]
        self.ingress_depth = c["ingress_queue_depth"]
        self.egress_depth  = c["egress_queue_depth"]
        self.ingress: Deque[MemRequest] = deque()
        self.egress:  Deque[MemRequest] = deque()
        self.counters = M2PCIeCounters()

    def enqueue(self, req: MemRequest) -> bool:
        if len(self.ingress) >= self.ingress_depth:
            return False  # back-pressure
        self.ingress.append(req)
        self.counters.rxc_inserts += 1
        req.latency_m2pcie += max(0, len(self.ingress) - 1)
        return True

    def dequeue(self) -> Optional[MemRequest]:
        if self.ingress:
            self.counters.rxc_cycles_ne += 1
            return self.ingress.popleft()
        return None

    def complete(self, req: MemRequest):
        """Mark data returned from CXL DIMM."""
        if req.is_load:
            self.counters.txc_inserts_bl += 1   # DRS (data response)
        else:
            self.counters.txc_inserts_ak += 1   # NDR (ack)
        self.egress.append(req)

    def snapshot(self) -> dict:
        return {
            **self.counters.snapshot(),
            "m2pcie.load_count": self.counters.txc_inserts_bl,
            "m2pcie.store_count": self.counters.txc_inserts_ak,
        }


# FlexBus 

@dataclass
class FlexBusCounters:
    credit_starvation_cycles: int = 0
    flits_sent:               int = 0
    bytes_transferred:        int = 0

    def snapshot(self) -> dict:
        return {
            "flexbus.starvation": self.credit_starvation_cycles,
            "flexbus.flits":      self.flits_sent,
            "flexbus.bytes":      self.bytes_transferred,
        }


class FlexBus:
    """
    PCIe-based FlexBus I/O.
    Models credit-based flow control and link latency.
    One credit = one flit capacity.
    """

    def __init__(self, cfg: dict):
        f = cfg["flexbus"]
        self.link_latency     = int(f["link_latency_ns"] * cfg["cpu"]["freq_ghz"])
        self.credit_pool      = f["credit_pool"]
        self.available_credits = f["credit_pool"]
        self.flit_bytes       = 68 if f["flit_mode"] == "68B" else 256
        self.counters         = FlexBusCounters()
        self._in_flight:      Dict[int, int] = {}   # req_id -> return_cycle

    def send(self, req: MemRequest, cycle: int) -> bool:
        """Submit request onto FlexBus. Returns False if no credits (stall)."""
        if self.available_credits <= 0:
            self.counters.credit_starvation_cycles += 1
            req.latency_flexbus += 1
            return False
        self.available_credits -= 1
        req.latency_flexbus += self.link_latency
        self._in_flight[req.req_id] = cycle + self.link_latency
        self.counters.flits_sent       += 1
        self.counters.bytes_transferred += 64  # one cacheline
        return True

    def return_credit(self, req_id: int):
        """Credit returned when response comes back."""
        self._in_flight.pop(req_id, None)
        self.available_credits = min(self.available_credits + 1, self.credit_pool)

    @property
    def utilization(self) -> float:
        used = self.credit_pool - self.available_credits
        return used / max(self.credit_pool, 1)

    def snapshot(self) -> dict:
        return {
            **self.counters.snapshot(),
            "flexbus.util":    self.utilization,
            "flexbus.credits": self.available_credits,
        }


# Local DIMM

class LocalDIMM:
    """DDR5 local DIMM with configurable latency and bandwidth model."""

    def __init__(self, cfg: dict):
        d = cfg["local_dimm"]
        freq = cfg["cpu"]["freq_ghz"]
        self.latency_cycles  = d.get("latency_cycles",
                                     int(d["random_latency_ns"] * freq))
        self.bandwidth_gbps  = d["bandwidth_gbps"]
        self._requests_served: int = 0
        self._bytes_transferred: int = 0

    def serve(self, req: MemRequest, cycle: int) -> int:
        """Returns cycle when data is available."""
        req.latency_dimm += self.latency_cycles
        req.target        = MemTarget.LOCAL_DRAM
        self._requests_served   += 1
        self._bytes_transferred += 64
        return cycle + self.latency_cycles

    def snapshot(self) -> dict:
        return {
            "dram.requests": self._requests_served,
            "dram.bytes":    self._bytes_transferred,
        }


# ─ CXL DIMM ─

@dataclass
class CXLDIMMCounters:
    """Table 4: unc_cxlcm_rxc/txc_pack_buf_inserts/full/ne for mem_req and mem_data."""
    rxc_pack_buf_inserts:       int = 0   # total packing buf inserts (Req)
    rxc_pack_buf_inserts_data:  int = 0   # mem data packing buf inserts
    rxc_pack_buf_full_req:      int = 0   # cycles req buf full
    rxc_pack_buf_full_data:     int = 0   # cycles data buf full
    rxc_pack_buf_ne_req:        int = 0   # cycles req buf not-empty
    rxc_pack_buf_ne_data:       int = 0   # cycles data buf not-empty
    txc_pack_buf_inserts_req:   int = 0   # egress req inserts
    txc_pack_buf_inserts_data:  int = 0   # egress data inserts

    def snapshot(self) -> dict:
        return {
            "cxl.rxc_ins_req":  self.rxc_pack_buf_inserts,
            "cxl.rxc_ins_data": self.rxc_pack_buf_inserts_data,
            "cxl.rxc_full_req": self.rxc_pack_buf_full_req,
            "cxl.rxc_ne_req":   self.rxc_pack_buf_ne_req,
            "cxl.rxc_ne_data":  self.rxc_pack_buf_ne_data,
            "cxl.txc_ins_req":  self.txc_pack_buf_inserts_req,
            "cxl.txc_ins_data": self.txc_pack_buf_inserts_data,
        }

    def qos_level(self, req_buf_depth: int, data_buf_depth: int) -> str:
        """CXL spec 3.0 QoS telemetry."""
        req_util  = self.rxc_pack_buf_inserts / max(req_buf_depth * 100, 1)
        data_util = self.rxc_pack_buf_inserts_data / max(data_buf_depth * 100, 1)
        util = max(req_util, data_util)
        if util < 0.25: return "light_load"
        if util < 0.50: return "optimal_load"
        if util < 0.75: return "moderate_overload"
        return "severe_overload"


class CXLDimmDevice:
    """
    CXL Type-3 DIMM (host-managed device memory).
    Models M2S Req/RwD -> device processing -> S2M DRS/NDR responses.
    Packing buffers mimic ingress/egress queue behaviour tracked in Table 4.
    """

    def __init__(self, cfg: dict):
        d = cfg["cxl_dimm"]
        freq = cfg["cpu"]["freq_ghz"]
        self.latency_cycles     = d.get("latency_cycles",
                                        int(d["random_latency_ns"] * freq))
        self.bandwidth_gbps     = d["bandwidth_gbps"]
        self.req_buf_depth      = d["mem_req_packing_buf_depth"]
        self.data_buf_depth     = d["mem_data_packing_buf_depth"]
        self.m2s_lat            = d["m2s_latency_cycles"]
        self.s2m_lat            = d["s2m_latency_cycles"]
        self.qos_thresholds     = d["qos_thresholds"]

        self.mem_req_buf:  Deque[MemRequest] = deque()
        self.mem_data_buf: Deque[MemRequest] = deque()
        self.counters = CXLDIMMCounters()

    def enqueue_m2s(self, req: MemRequest) -> bool:
        """Host -> device (M2S: Req for read, RwD for write)."""
        if len(self.mem_req_buf) >= self.req_buf_depth:
            self.counters.rxc_pack_buf_full_req += 1
            return False
        self.mem_req_buf.append(req)
        self.counters.rxc_pack_buf_inserts += 1
        if req.is_load:
            self.counters.rxc_pack_buf_inserts_data += 1
        return True

    def tick(self, cycle: int) -> List[MemRequest]:
        """
        Process one cycle: move requests from req buf -> data buf -> complete.
        Returns list of completed requests (S2M responses ready).
        """
        completed = []

        # Update not-empty counters
        if self.mem_req_buf:
            self.counters.rxc_pack_buf_ne_req  += 1
        if self.mem_data_buf:
            self.counters.rxc_pack_buf_ne_data += 1
        if len(self.mem_req_buf) >= self.req_buf_depth:
            self.counters.rxc_pack_buf_full_req += 1
        if len(self.mem_data_buf) >= self.data_buf_depth:
            self.counters.rxc_pack_buf_full_data += 1

        # Process up to N requests per cycle (bandwidth-limited)
        bandwidth_budget = max(1, int(self.bandwidth_gbps * 1e9 / (64 * 8) / (1e9)))  # cachelines/cycle
        processed = 0
        while self.mem_req_buf and processed < bandwidth_budget:
            req = self.mem_req_buf.popleft()
            req.latency_dimm += self.latency_cycles
            req.target         = MemTarget.CXL_DRAM
            # S2M response
            if req.is_load:
                self.counters.txc_pack_buf_inserts_data += 1   # DRS
            else:
                self.counters.txc_pack_buf_inserts_req  += 1   # NDR
            completed.append(req)
            processed += 1

        return completed

    def snapshot(self) -> dict:
        return {
            **self.counters.snapshot(),
            "cxl.buf_util_req":  len(self.mem_req_buf)  / max(self.req_buf_depth, 1),
            "cxl.buf_util_data": len(self.mem_data_buf) / max(self.data_buf_depth, 1),
            "cxl.qos": self.counters.qos_level(self.req_buf_depth, self.data_buf_depth),
        }

    def reset_counters(self):
        self.counters = CXLDIMMCounters()