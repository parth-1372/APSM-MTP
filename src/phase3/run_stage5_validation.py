import argparse
import copy
import heapq
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from phase3.config import Phase3Config, dev_tier1_6n
from phase3.data.alibaba_loader import build_or_load_curated_trace
from phase3.data.topology import build_edge_topology
from phase3.env import (
    LocalOnlyPolicy,
    ReactiveGreedyPolicy,
    SimMetrics,
    TaskOffloadingSim,
    prepare_task_table,
)
from phase3.env.task_simulator import ActiveTask, PeerCacheEntry
from phase3.forecasting.gossip_runner import generate_causal_forecast_matrices


# ==============================================================================
# 1. High-Performance Causal Simulator with Physical Wire-Inflight Tracking
# ==============================================================================

class FastCausalSim(TaskOffloadingSim):
    """
    O(1)-incremental observation caching simulator matching TaskOffloadingSim semantics.
    """

    def __init__(self, *args, obs5_mode: str = "zero", **kwargs):
        self.obs5_mode = obs5_mode
        super().__init__(*args, **kwargs)
        self._nbr_slots_tup = tuple(tuple(self.topo.neighbor_slots(i)) for i in range(self.n_nodes))
        self._nbr_idx_arr = [np.asarray(nbrs, dtype=np.intp) for nbrs in self._nbr_slots_tup]
        k_max = self.sim.k_max_neighbors
        n = self.n_nodes
        self._peer_rtt_norm = np.zeros((n, k_max), dtype=np.float32)
        self._peer_bw_norm = np.zeros((n, k_max), dtype=np.float32)
        for i in range(n):
            for k_idx, peer_id in enumerate(self._nbr_slots_tup[i][:k_max]):
                self._peer_rtt_norm[i, k_idx] = min(1.0, float(self.topo.rtt_s[i, peer_id]) / 0.05)
                self._peer_bw_norm[i, k_idx] = min(1.0, float(self.topo.bw_mbps[i, peer_id]) / 200.0)

    def reset(self) -> None:
        super().reset()
        n = self.n_nodes
        self._active_r_nom = np.zeros(n, dtype=np.float64)
        self._total_inflight = 0
        self._obs_dim = 11 + 8 * self.sim.k_max_neighbors
        self._inv_q_max = 1.0 / float(self.sim.q_max)
        self._inv_cpu_cap = 1.0 / float(self.sim.node_cpu_cap)
        self._inv_nodes = 1.0 / float(n)
        self._inv_tot_q = 1.0 / float(max(1, n * self.sim.q_max))
        self._inv_k_max = 1.0 / float(self.sim.k_max_neighbors)
        self._qs = np.zeros(n, dtype=np.float32)
        self._avails = np.full(n, 0.05, dtype=np.float32)
        self._mems = np.zeros(n, dtype=np.float32)
        self._global_stats_dirty = True
        self._cached_gstats = np.zeros(10, dtype=np.float32)

    def _sync_node_state_arrays(self, i: int) -> None:
        self._qs[i] = len(self.active_tasks[i]) * self._inv_q_max
        base_avail = max(0.05, 1.0 - float(self._cur_bg_cpu[i]))
        self._avails[i] = max(0.0, base_avail - float(self._active_r_nom[i]) * self._inv_cpu_cap)
        batch_mem_budget = max(0.30, self.sim.node_mem_cap - float(self._cur_bg_mem[i]))
        self._mems[i] = min(1.0, max(0.0, batch_mem_budget - float(self.active_mem[i])))
        self._global_stats_dirty = True

    def _refresh_slow_context(self, t_now_s: float) -> int:
        b_cur = super()._refresh_slow_context(t_now_s)
        lo8 = max(0, b_cur - 8)
        self.cur_load_norm = (
            self.dataset.targets[:, lo8:b_cur].mean(axis=1)
            if b_cur > 0
            else np.zeros(self.n_nodes, dtype=np.float32)
        )
        for i in range(self.n_nodes):
            self._sync_node_state_arrays(i)
        return b_cur

    def _node_avail_cpu_norm(self, i: int) -> float:
        return float(self._avails[i])

    def _node_util_norm(self, i: int) -> float:
        return float(min(1.5, float(self._cur_bg_cpu[i]) + float(self._active_r_nom[i]) * self._inv_cpu_cap))

    def _try_admit(self, node_id: int, task: ActiveTask) -> bool:
        if len(self.active_tasks[node_id]) >= self.sim.q_max:
            return False
        if self._node_free_mem(node_id) + 1e-9 < task.mem_req:
            return False
        self.active_tasks[node_id].append(task)
        self.active_mem[node_id] += float(task.mem_req)
        self._active_r_nom[node_id] += float(task.r_nom)
        self._sync_node_state_arrays(node_id)
        return True

    def _advance_compute_slot(self, t_now_s: float, in_episode: bool = True) -> None:
        dt = self.sim.dt_slot_s
        t_end_slot = t_now_s + dt
        for i in range(self.n_nodes):
            q = self.active_tasks[i]
            avail_cpu = self._node_avail_cpu_rate(i)
            if in_episode:
                self.node_cpu_avail_acc[i] += avail_cpu * dt
            if not q:
                continue

            sum_nom = float(self._active_r_nom[i])
            scale = min(1.0, avail_cpu / sum_nom) if sum_nom > 1e-9 else 1.0
            if in_episode:
                self.node_cpu_used[i] += min(avail_cpu, sum_nom) * dt

            survivors = []
            surv_r_nom = 0.0
            changed = False
            for task in q:
                rate = float(task.r_nom) * scale
                delta_w = rate * dt
                if task.work_rem <= delta_w + 1e-7:
                    t_fin = t_now_s + (task.work_rem / max(1e-9, rate))
                    task.work_rem = 0.0
                    self.active_mem[i] = max(0.0, float(self.active_mem[i]) - float(task.mem_req))
                    if t_fin <= task.deadline_s + 1e-9:
                        self._schedule_ack(task, False, "none", t_fin)
                    else:
                        self._schedule_ack(task, True, "deadline_timeout", task.deadline_s)
                    changed = True
                else:
                    task.work_rem -= delta_w
                    if t_end_slot >= task.deadline_s - 1e-9:
                        self.active_mem[i] = max(0.0, float(self.active_mem[i]) - float(task.mem_req))
                        self._schedule_ack(task, True, "deadline_timeout", task.deadline_s)
                        changed = True
                    else:
                        survivors.append(task)
                        surv_r_nom += float(task.r_nom)
            if changed:
                self.active_tasks[i] = survivors
                self._active_r_nom[i] = surv_r_nom
                if not survivors:
                    self.active_mem[i] = 0.0
                    self._active_r_nom[i] = 0.0
                self._sync_node_state_arrays(i)

    def _deliver_due_events(self, t_now_s: float) -> None:
        while self._link_rx_heap and self._link_rx_heap[0][0] <= t_now_s + 1e-9:
            arr_peer_ts, _, task = heapq.heappop(self._link_rx_heap)
            if arr_peer_ts > task.deadline_s:
                self._schedule_ack(task, True, "link_timeout", arr_peer_ts)
            elif not self._try_admit(task.target_id, task):
                self._schedule_ack(task, True, "admission_overflow", arr_peer_ts)

        while self._beacon_heap and self._beacon_heap[0][0] <= t_now_s + 1e-9:
            _, _, receiver, sender, entry = heapq.heappop(self._beacon_heap)
            self.neighbor_cache[receiver][sender] = entry
            infl = self.inflight_records[receiver][sender]
            done_ids = [tid for tid, arr_t in infl.items() if arr_t <= entry.last_created_ts + 1e-9]
            for tid in done_ids:
                del infl[tid]
            self._total_inflight -= len(done_ids)

        while self._ack_heap and self._ack_heap[0][0] <= t_now_s + 1e-9:
            _, _, task, dropped, reason, ev_ts = heapq.heappop(self._ack_heap)
            self.ack_packets += 1
            self.ack_bytes += self.sim.ack_bytes
            popped = self.inflight_records[task.origin_id][task.target_id].pop(task.task_id, None)
            if popped is not None:
                self._total_inflight -= 1
            self._emit_outcome(task, dropped, reason, ev_ts)

    def build_observation(self, node_id: int, task_idx: int, t_now_s: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs_dim = self._obs_dim
        obs = np.zeros(obs_dim, dtype=np.float32)
        mask = np.ones(1 + self.sim.k_max_neighbors, dtype=bool)
        global_state = np.zeros(obs_dim + 16, dtype=np.float32)

        has_fc = self.forecast_matrix is not None
        has_surp = self.surprise_matrix is not None
        inv_q = self._inv_q_max
        inv_mem = 1.0 / float(self.sim.node_mem_cap)
        inv_bw = 1.0 / 200.0
        inv_rtt = 1.0 / 0.05
        inv_info = 1.0 / 60.0

        # Self state
        obs[0] = float(self.active_mem[node_id]) * inv_mem
        obs[1] = self._node_avail_cpu_norm(node_id)
        obs[2] = len(self.active_tasks[node_id]) * inv_q
        obs[3] = float(self.cur_load_norm[node_id])
        obs[4] = float(self.cur_y_hat[node_id]) if has_fc else 0.0
        obs[5] = 0.0
        obs[6] = float(self.cur_surprise[node_id]) if has_surp else 0.0
        obs[7] = float(self.tasks.cpu_work[task_idx]) * inv_q
        obs[8] = float(self.tasks.r_nom[task_idx]) * self._inv_cpu_cap
        obs[9] = float(self.tasks.mem_req[task_idx]) * inv_mem
        obs[10] = min(1.0, (float(self.tasks.deadline_s[task_idx]) - t_now_s) / 600.0)

        # Peer slots
        nbrs = self._nbr_slots_tup[node_id]
        inf_i = self.inflight_records[node_id]
        is_oracle = self.telemetry_mode == "oracle"

        for k_idx in range(self.sim.k_max_neighbors):
            base = 11 + 8 * k_idx
            if k_idx >= len(nbrs):
                mask[1 + k_idx] = False
                continue
            peer_id = nbrs[k_idx]
            obs[base + 0] = 1.0
            obs[base + 1] = self._peer_bw_norm[node_id, k_idx]
            obs[base + 2] = self._peer_rtt_norm[node_id, k_idx]
            if is_oracle:
                obs[base + 3] = self._node_avail_cpu_norm(peer_id)
                obs[base + 4] = len(self.active_tasks[peer_id]) * inv_q
                obs[base + 5] = float(self.cur_y_hat[peer_id]) if has_fc else 0.0
                obs[base + 6] = 0.0
                obs[base + 7] = 0.0
            else:
                entry = self.neighbor_cache[node_id].get(peer_id)
                if entry is not None:
                    obs[base + 3] = entry.avail_cpu_norm
                    obs[base + 4] = entry.q_norm
                    obs[base + 5] = entry.y_hat
                    a_age = (t_now_s - entry.last_created_ts) * inv_info
                    obs[base + 7] = 1.0 if a_age > 1.0 else (0.0 if a_age < 0.0 else a_age)
                else:
                    bc = float(self._cur_bg_cpu[peer_id])
                    obs[base + 3] = 0.05 if (1.0 - bc) < 0.05 else (1.0 - bc)
                    obs[base + 7] = 1.0
                icnt = len(inf_i[peer_id]) * inv_q
                obs[base + 6] = 1.0 if icnt > 1.0 else icnt

        if self._global_stats_dirty:
            qs = self._qs
            av = self._avails
            mm = self._mems
            qm = float(qs.sum()) * self._inv_nodes
            qd = qs - qm
            qstd = math.sqrt(float(np.dot(qd, qd)) * self._inv_nodes)
            yh = self.cur_y_hat
            self._cached_gstats[0] = qm
            self._cached_gstats[1] = qstd
            self._cached_gstats[2] = float(qs.max())
            self._cached_gstats[3] = float(qs.min())
            self._cached_gstats[4] = float(av.sum()) * self._inv_nodes
            self._cached_gstats[5] = float(av.min())
            self._cached_gstats[6] = float(av.max())
            self._cached_gstats[7] = float(mm.sum()) * self._inv_nodes
            self._cached_gstats[8] = (float(yh.sum()) * self._inv_nodes) if has_fc else 0.0
            self._cached_gstats[9] = float(yh.max()) if has_fc else 0.0
            self._global_stats_dirty = False

        global_state[:obs_dim] = obs
        global_state[obs_dim : obs_dim + 10] = self._cached_gstats
        global_state[obs_dim + 10] = float(self._total_inflight * self._inv_tot_q)
        global_state[obs_dim + 11] = float(len(nbrs) * self._inv_k_max)

        if nbrs:
            nidx = self._nbr_idx_arr[node_id]
            nq = self._qs[nidx]
            na = self._avails[nidx]
            inv_kn = 1.0 / len(nbrs)
            global_state[obs_dim + 12] = float(nq.min())
            global_state[obs_dim + 13] = float(na.max())
            global_state[obs_dim + 14] = float(nq.sum()) * inv_kn
            global_state[obs_dim + 15] = float(na.sum()) * inv_kn

        return obs, mask, global_state


class WireOracleFastSim(FastCausalSim):
    """Same as FastCausalSim except oracle mode reports physical sender-to-peer wire-inflight count in obs[base+6]."""

    def __init__(self, *args, oracle_wire_inflight: bool = False, **kwargs):
        self.oracle_wire_inflight = oracle_wire_inflight
        super().__init__(*args, **kwargs)

    def reset(self) -> None:
        super().reset()
        self._wire_cnt = np.zeros((self.n_nodes, self.n_nodes), dtype=np.int32)

    def _dispatch_task(self, task_idx: int, action: int, t_now_s: float) -> None:
        super()._dispatch_task(task_idx, action, t_now_s)
        if action != 0:
            origin = int(self.tasks.node_idx[task_idx])
            peer = self._nbr_slots_tup[origin][action - 1]
            self._wire_cnt[origin, peer] += 1

    def _deliver_due_events(self, t_now_s: float) -> None:
        while self._link_rx_heap and self._link_rx_heap[0][0] <= t_now_s + 1e-9:
            arr_peer_ts, _, task = heapq.heappop(self._link_rx_heap)
            self._wire_cnt[task.origin_id, task.target_id] = max(
                0, int(self._wire_cnt[task.origin_id, task.target_id]) - 1
            )
            if arr_peer_ts > task.deadline_s:
                self._schedule_ack(task, True, "link_timeout", arr_peer_ts)
            elif not self._try_admit(task.target_id, task):
                self._schedule_ack(task, True, "admission_overflow", arr_peer_ts)

        while self._beacon_heap and self._beacon_heap[0][0] <= t_now_s + 1e-9:
            _, _, receiver, sender, entry = heapq.heappop(self._beacon_heap)
            self.neighbor_cache[receiver][sender] = entry
            infl = self.inflight_records[receiver][sender]
            done_ids = [tid for tid, arr_t in infl.items() if arr_t <= entry.last_created_ts + 1e-9]
            for tid in done_ids:
                del infl[tid]
            self._total_inflight -= len(done_ids)

        while self._ack_heap and self._ack_heap[0][0] <= t_now_s + 1e-9:
            _, _, task, dropped, reason, ev_ts = heapq.heappop(self._ack_heap)
            self.ack_packets += 1
            self.ack_bytes += self.sim.ack_bytes
            popped = self.inflight_records[task.origin_id][task.target_id].pop(task.task_id, None)
            if popped is not None:
                self._total_inflight -= 1
            self._emit_outcome(task, dropped, reason, ev_ts)

    def build_observation(self, node_id: int, task_idx: int, t_now_s: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs, mask, gs = super().build_observation(node_id, task_idx, t_now_s)
        if self.oracle_wire_inflight and self.telemetry_mode == "oracle":
            nbrs = self._nbr_slots_tup[node_id]
            inv_q = self._inv_q_max
            for k_idx, peer_id in enumerate(nbrs[: self.sim.k_max_neighbors]):
                v = min(1.0, float(self._wire_cnt[node_id, peer_id]) * inv_q)
                obs[11 + 8 * k_idx + 6] = v
                gs[11 + 8 * k_idx + 6] = v
        return obs, mask, gs


# ==============================================================================
# 2. Neural Architectures: SharedTrunkV5 (MAPPO/IPPO/Oracle) & Proactive2QNet (MADQN)
# ==============================================================================

def _ortho(layer: nn.Linear, gain: float) -> None:
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0.0)


def _valid_pool(e_peers: torch.Tensor, mask: torch.Tensor, k_max: int) -> torch.Tensor:
    vp = mask[:, 1 : 1 + k_max].unsqueeze(-1).to(e_peers.dtype)
    nv = vp.sum(dim=1).clamp(min=1.0)
    ep_m = e_peers * vp
    ep_mean = ep_m.sum(dim=1) / nv
    ep_max = torch.where(vp.bool(), e_peers, torch.full_like(e_peers, -1e4)).max(dim=1).values
    ep_max = torch.where(nv > 0, ep_max, torch.zeros_like(ep_max))
    return torch.cat([ep_mean, ep_max], dim=-1)


class KaimingFactoredPeerHead(nn.Module):
    def __init__(self, ctx_dim: int, peer_dim: int, hidden_dim: int):
        super().__init__()
        self.fc_ctx = nn.Linear(ctx_dim, hidden_dim)
        self.fc_peer = nn.Linear(peer_dim, hidden_dim, bias=False)
        self.ln = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, h_ctx: torch.Tensor, e_peers: torch.Tensor) -> torch.Tensor:
        z_ctx = self.fc_ctx(h_ctx).unsqueeze(1)
        z_peer = self.fc_peer(e_peers)
        h = self.relu(self.ln(z_ctx + z_peer))
        return self.out(h).squeeze(-1)


class PeerProactive2Branch(nn.Module):
    def __init__(self, k_max: int = 8, fc_scale: float = 0.008):
        super().__init__()
        self.k_max = k_max
        self.fc_scale = float(fc_scale)
        self.fc_active = True
        self.peer_lin = nn.Linear(2, 1, bias=False)
        nn.init.zeros_(self.peer_lin.weight)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bsz = obs.shape[0]
        if not self.fc_active:
            return torch.zeros(bsz, self.k_max, dtype=obs.dtype, device=obs.device)
        p = obs[:, 11 : 11 + 8 * self.k_max].view(bsz, self.k_max, 8)
        y_p = p[:, :, 5]
        avail_p = p[:, :, 3]
        age_p = p[:, :, 7]
        vp = mask[:, 1 : 1 + self.k_max].to(obs.dtype)
        pf = torch.stack([F.relu(y_p - avail_p), y_p * age_p], dim=-1)
        dp = self.fc_scale * torch.tanh(self.peer_lin(pf)).squeeze(-1)
        return dp * vp


class SharedTrunkV5(nn.Module):
    def __init__(
        self,
        obs_dim: int = 75,
        global_dim: int = 91,
        k_max: int = 8,
        hidden: int = 96,
        use_central_critic: bool = False,
        actor_uses_global: bool = False,
        fc_scale: float = 0.008,
        logit_scale: float = 0.03,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.global_dim = global_dim
        self.k_max = k_max
        self.use_central_critic = use_central_critic
        self.actor_uses_global = actor_uses_global
        self.logit_scale = float(logit_scale)
        p_dim = hidden // 2
        g_extra = global_dim - obs_dim
        g_emb = 32 if hidden >= 96 else 24

        self.self_enc = nn.Sequential(nn.Linear(8, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.peer_enc = nn.Sequential(nn.Linear(7, p_dim), nn.LayerNorm(p_dim), nn.ReLU())
        base_ctx = hidden + 2 * p_dim

        if actor_uses_global:
            self.glob_enc = nn.Sequential(nn.Linear(g_extra, g_emb), nn.LayerNorm(g_emb), nn.ReLU())
            _ortho(self.glob_enc[0], 0.25)
            ctx_dim = base_ctx + g_emb
        else:
            self.glob_enc = None
            ctx_dim = base_ctx

        self.q_val = nn.Sequential(nn.Linear(base_ctx, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.q_loc = nn.Sequential(nn.Linear(ctx_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.q_peer = KaimingFactoredPeerHead(ctx_dim, p_dim, hidden)
        self.q_fc = PeerProactive2Branch(k_max=k_max, fc_scale=fc_scale)

        self.act_loc = nn.Sequential(nn.Linear(ctx_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.act_peer = KaimingFactoredPeerHead(ctx_dim, p_dim, hidden)
        self.act_fc = PeerProactive2Branch(k_max=k_max, fc_scale=fc_scale / logit_scale)

        self.v_head = nn.Sequential(nn.Linear(base_ctx, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Linear(hidden, 1))
        if use_central_critic:
            self.v_glob = nn.Sequential(nn.Linear(base_ctx + g_extra, 64), nn.LayerNorm(64), nn.ReLU(), nn.Linear(64, 1))
            _ortho(self.v_glob[0], 0.5)
            _ortho(self.v_glob[3], 0.25)
        else:
            self.v_glob = None

    def q_params(self):
        p = (
            list(self.self_enc.parameters())
            + list(self.peer_enc.parameters())
            + list(self.q_val.parameters())
            + list(self.q_loc.parameters())
            + list(self.q_peer.parameters())
        )
        if self.glob_enc is not None:
            p += list(self.glob_enc.parameters())
        return p

    def v_params(self):
        p = list(self.v_head.parameters())
        if self.v_glob is not None:
            p += list(self.v_glob.parameters())
        return p

    def actor_head_params(self):
        return list(self.act_loc.parameters()) + list(self.act_peer.parameters())

    def _encode(self, obs: torch.Tensor, mask: torch.Tensor, gstate: Optional[torch.Tensor] = None):
        bsz = obs.shape[0]
        s_in = torch.cat([obs[:, :4], obs[:, 7:11]], dim=-1)
        p = obs[:, 11 : 11 + 8 * self.k_max].view(bsz, self.k_max, 8)
        p_in = torch.cat([p[:, :, :5], p[:, :, 6:8]], dim=-1)
        h_self = self.self_enc(s_in)
        e_peers = self.peer_enc(p_in)
        h_base = torch.cat([h_self, _valid_pool(e_peers, mask, self.k_max)], dim=-1)
        if self.actor_uses_global and self.glob_enc is not None:
            ge = (
                torch.zeros(bsz, self.global_dim - self.obs_dim, dtype=obs.dtype, device=obs.device)
                if gstate is None
                else gstate[:, self.obs_dim :].clone()
            )
            ge[:, 8:10] = 0.0
            h_ctx = torch.cat([h_base, self.glob_enc(ge)], dim=-1)
        else:
            h_ctx = h_base
        return h_base, h_ctx, e_peers

    def q_values(self, obs: torch.Tensor, mask: torch.Tensor, gstate: Optional[torch.Tensor] = None) -> torch.Tensor:
        h_base, h_ctx, e_peers = self._encode(obs, mask, gstate)
        v = self.q_val(h_base)
        ap = self.q_peer(h_ctx, e_peers) + self.q_fc(obs, mask)
        adv = torch.cat([self.q_loc(h_ctx), ap], dim=-1)
        mf = mask.to(adv.dtype)
        adv_mean = (adv * mf).sum(dim=-1, keepdim=True) / mf.sum(dim=-1, keepdim=True).clamp(min=1.0)
        return (v + (adv - adv_mean)).masked_fill(~mask, -1e9)

    def critic_value(self, obs: torch.Tensor, mask: torch.Tensor, gstate: Optional[torch.Tensor] = None) -> torch.Tensor:
        with torch.no_grad():
            h_base, _, _ = self._encode(obs, mask, gstate)
        v = self.v_head(h_base)
        if self.use_central_critic and self.v_glob is not None:
            bsz = obs.shape[0]
            ge = (
                torch.zeros(bsz, self.global_dim - self.obs_dim, dtype=obs.dtype, device=obs.device)
                if gstate is None
                else gstate[:, self.obs_dim :].clone()
            )
            ge[:, 8:10] = 0.0
            v = v + self.v_glob(torch.cat([h_base, ge], dim=-1))
        return v.squeeze(-1)


class Proactive2QNet(nn.Module):
    def __init__(self, obs_dim: int = 75, global_dim: int = 91, k_max: int = 8, hidden: int = 96, fc_scale: float = 0.008):
        super().__init__()
        self.obs_dim = obs_dim
        self.global_dim = global_dim
        self.k_max = k_max
        p_dim = hidden // 2
        self.self_enc = nn.Sequential(nn.Linear(8, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.peer_enc = nn.Sequential(nn.Linear(7, p_dim), nn.LayerNorm(p_dim), nn.ReLU())
        base_ctx = hidden + 2 * p_dim
        self.val_head = nn.Sequential(nn.Linear(base_ctx, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.adv_loc = nn.Sequential(nn.Linear(base_ctx, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.adv_peer = KaimingFactoredPeerHead(base_ctx, p_dim, hidden)
        self.fc_res = PeerProactive2Branch(k_max=k_max, fc_scale=fc_scale)

    def base_params(self):
        return (
            list(self.self_enc.parameters())
            + list(self.peer_enc.parameters())
            + list(self.val_head.parameters())
            + list(self.adv_loc.parameters())
            + list(self.adv_peer.parameters())
        )

    def forward(self, obs: torch.Tensor, mask: torch.Tensor, gstate: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz = obs.shape[0]
        s_in = torch.cat([obs[:, :4], obs[:, 7:11]], dim=-1)
        p = obs[:, 11 : 11 + 8 * self.k_max].view(bsz, self.k_max, 8)
        p_in = torch.cat([p[:, :, :5], p[:, :, 6:8]], dim=-1)
        h_self = self.self_enc(s_in)
        e_peers = self.peer_enc(p_in)
        h_base = torch.cat([h_self, _valid_pool(e_peers, mask, self.k_max)], dim=-1)
        v = self.val_head(h_base)
        ap = self.adv_peer(h_base, e_peers) + self.fc_res(obs, mask)
        adv = torch.cat([self.adv_loc(h_base), ap], dim=-1)
        mf = mask.to(adv.dtype)
        adv_mean = (adv * mf).sum(dim=-1, keepdim=True) / mf.sum(dim=-1, keepdim=True).clamp(min=1.0)
        return (v + (adv - adv_mean)).masked_fill(~mask, -1e9)


# ==============================================================================
# 3. High-Performance NumPy Runners for Minimal DES Python Dispatch Overhead
# ==============================================================================

class FastSharedTrunkRunner:
    def __init__(self, mod: SharedTrunkV5, mode: str = "q") -> None:
        self.k_max = mod.k_max
        self.obs_dim = mod.obs_dim
        self.actor_uses_global = mod.actor_uses_global

        def ext(m):
            sd = m.state_dict()
            return (
                sd["0.weight"].detach().cpu().numpy().T.astype(np.float32),
                sd["0.bias"].detach().cpu().numpy().astype(np.float32),
                sd["1.weight"].detach().cpu().numpy().astype(np.float32),
                sd["1.bias"].detach().cpu().numpy().astype(np.float32),
            )

        self.se = ext(mod.self_enc)
        self.pe = ext(mod.peer_enc)
        self.p_dim = self.pe[1].shape[0]
        self.ag = ext(mod.glob_enc) if (mod.actor_uses_global and mod.glob_enc is not None) else None
        lh = mod.act_loc if mode == "actor" else mod.q_loc
        ph = mod.act_peer if mode == "actor" else mod.q_peer
        fc = mod.act_fc if mode == "actor" else mod.q_fc

        self.lh0 = ext(lh)
        self.lh_w = lh[3].weight.detach().cpu().numpy().astype(np.float32).ravel()
        self.lh_b = float(lh[3].bias.detach().cpu().numpy()[0])
        self.ph_wc = np.ascontiguousarray(ph.fc_ctx.weight.detach().cpu().numpy().T.astype(np.float32))
        self.ph_wp = np.ascontiguousarray(ph.fc_peer.weight.detach().cpu().numpy().T.astype(np.float32))
        self.ph_b0 = ph.fc_ctx.bias.detach().cpu().numpy().astype(np.float32)
        self.ph_lw = ph.ln.weight.detach().cpu().numpy().astype(np.float32)
        self.ph_lb = ph.ln.bias.detach().cpu().numpy().astype(np.float32)
        self.ph_w3 = ph.out.weight.detach().cpu().numpy().astype(np.float32).ravel()
        self.ph_b3 = float(ph.out.bias.detach().cpu().numpy()[0])
        self.fc_active = fc.fc_active
        self.fc_scale = fc.fc_scale
        self.fp = fc.peer_lin.weight.detach().cpu().numpy().astype(np.float32).ravel()

    @staticmethod
    def _ln_relu_1d(z: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        m = z.mean()
        v = float(np.mean((z - m) ** 2))
        return np.maximum(0.0, ((z - m) / math.sqrt(v + 1e-5)) * w + b)

    @staticmethod
    def _ln_relu_2d(z: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        m = z.mean(axis=-1, keepdims=True)
        v = np.mean((z - m) ** 2, axis=-1, keepdims=True)
        return np.maximum(0.0, ((z - m) / np.sqrt(v + 1e-5)) * w + b)

    def logits(self, obs: np.ndarray, mask: np.ndarray, gstate: Optional[np.ndarray] = None) -> np.ndarray:
        s_in = np.concatenate([obs[:4], obs[7:11]])
        h_self = self._ln_relu_1d(s_in @ self.se[0] + self.se[1], self.se[2], self.se[3])
        v_idx = np.flatnonzero(mask[1 : 1 + self.k_max])
        nv = len(v_idx)
        if nv > 0:
            p_mat = obs[11 : 11 + 8 * self.k_max].reshape(self.k_max, 8)[v_idx]
            p_in = np.concatenate([p_mat[:, :5], p_mat[:, 6:8]], axis=1)
            z_p = p_in @ self.pe[0] + self.pe[1]
            if nv == 1:
                ep_v = self._ln_relu_1d(z_p[0], self.pe[2], self.pe[3])[None, :]
                ep_mean = ep_v[0]
                ep_max = ep_v[0]
            else:
                ep_v = self._ln_relu_2d(z_p, self.pe[2], self.pe[3])
                ep_mean = ep_v.sum(axis=0) * (1.0 / nv)
                ep_max = ep_v.max(axis=0)
            parts = [h_self, ep_mean, ep_max]
        else:
            p_mat = None
            ep_v = None
            parts = [h_self, np.zeros(2 * self.p_dim, dtype=np.float32)]

        if self.actor_uses_global and self.ag is not None and gstate is not None:
            ge = gstate[self.obs_dim :].copy()
            ge[8:10] = 0.0
            parts.append(self._ln_relu_1d(ge @ self.ag[0] + self.ag[1], self.ag[2], self.ag[3]))

        h_ctx = np.concatenate(parts)
        raw = np.full(1 + self.k_max, -1e9, dtype=np.float32)
        h_loc = self._ln_relu_1d(h_ctx @ self.lh0[0] + self.lh0[1], self.lh0[2], self.lh0[3])
        raw[0] = float(np.dot(h_loc, self.lh_w)) + self.lh_b
        if nv > 0:
            z_p_all = ep_v @ self.ph_wp + (h_ctx @ self.ph_wc + self.ph_b0)
            h_p = self._ln_relu_2d(z_p_all, self.ph_lw, self.ph_lb)
            ap = (h_p @ self.ph_w3) + self.ph_b3
            if self.fc_active:
                yp = p_mat[:, 5]
                avp = p_mat[:, 3]
                agp = p_mat[:, 7]
                pf = np.stack([np.maximum(0.0, yp - avp), yp * agp], axis=1)
                ap = ap + self.fc_scale * np.tanh(pf @ self.fp)
            raw[1 + v_idx] = ap
        return raw

    def act(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        gstate: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
        temp: float = 0.0,
        eps: float = 0.0,
    ) -> int:
        if eps > 0.0 and rng is not None and rng.random() < eps:
            v = np.flatnonzero(mask)
            return int(v[int(rng.integers(0, len(v)))])
        raw = self.logits(obs, mask, gstate)
        if temp <= 1e-6 or rng is None:
            return int(np.argmax(raw))
        sh = (raw - float(raw[mask].max())) / temp
        pr = np.where(mask, np.exp(np.clip(sh, -40.0, 0.0)), 0.0)
        ps = float(pr.sum())
        return int(np.argmax(raw)) if (ps <= 0.0 or not math.isfinite(ps)) else int(rng.choice(len(mask), p=pr * (1.0 / ps)))


class FastProactive2QRunner:
    def __init__(self, qb: Proactive2QNet) -> None:
        self.k_max = qb.k_max
        self.obs_dim = qb.obs_dim

        def ext(m):
            sd = m.state_dict()
            return (
                sd["0.weight"].detach().cpu().numpy().T.astype(np.float32),
                sd["0.bias"].detach().cpu().numpy().astype(np.float32),
                sd["1.weight"].detach().cpu().numpy().astype(np.float32),
                sd["1.bias"].detach().cpu().numpy().astype(np.float32),
            )

        self.se = ext(qb.self_enc)
        self.pe = ext(qb.peer_enc)
        self.al0 = ext(qb.adv_loc)
        self.al_w = qb.adv_loc[3].weight.detach().cpu().numpy().astype(np.float32).ravel()
        self.al_b = float(qb.adv_loc[3].bias.detach().cpu().numpy()[0])
        ap = qb.adv_peer
        self.ap_wc = np.ascontiguousarray(ap.fc_ctx.weight.detach().cpu().numpy().T.astype(np.float32))
        self.ap_wp = np.ascontiguousarray(ap.fc_peer.weight.detach().cpu().numpy().T.astype(np.float32))
        self.ap_b0 = ap.fc_ctx.bias.detach().cpu().numpy().astype(np.float32)
        self.ap_lw = ap.ln.weight.detach().cpu().numpy().astype(np.float32)
        self.ap_lb = ap.ln.bias.detach().cpu().numpy().astype(np.float32)
        self.ap_w3 = ap.out.weight.detach().cpu().numpy().astype(np.float32).ravel()
        self.ap_b3 = float(ap.out.bias.detach().cpu().numpy()[0])
        self.fc_active = qb.fc_res.fc_active
        self.fc_scale = qb.fc_res.fc_scale
        self.fp = qb.fc_res.peer_lin.weight.detach().cpu().numpy().astype(np.float32).ravel()

    def act(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        gstate: Optional[np.ndarray],
        rng: Optional[np.random.Generator],
        temp: float = 0.0,
        eps: float = 0.0,
    ) -> int:
        if eps > 0.0 and rng is not None and rng.random() < eps:
            v = np.flatnonzero(mask)
            return int(v[int(rng.integers(0, len(v)))])
        v_idx = np.flatnonzero(mask[1 : 1 + self.k_max])
        nv = len(v_idx)
        if nv == 0:
            return 0
        s_in = np.concatenate([obs[:4], obs[7:11]])
        h_self = FastSharedTrunkRunner._ln_relu_1d(s_in @ self.se[0] + self.se[1], self.se[2], self.se[3])
        p_mat = obs[11 : 11 + 8 * self.k_max].reshape(self.k_max, 8)[v_idx]
        p_in = np.concatenate([p_mat[:, :5], p_mat[:, 6:8]], axis=1)
        z_p = p_in @ self.pe[0] + self.pe[1]
        if nv == 1:
            ep_v = FastSharedTrunkRunner._ln_relu_1d(z_p[0], self.pe[2], self.pe[3])[None, :]
            ep_mean, ep_max = ep_v[0], ep_v[0]
        else:
            ep_v = FastSharedTrunkRunner._ln_relu_2d(z_p, self.pe[2], self.pe[3])
            ep_mean, ep_max = ep_v.sum(axis=0) * (1.0 / nv), ep_v.max(axis=0)
        h_ctx = np.concatenate([h_self, ep_mean, ep_max])
        raw = np.full(1 + self.k_max, -1e9, dtype=np.float32)
        h_loc = FastSharedTrunkRunner._ln_relu_1d(h_ctx @ self.al0[0] + self.al0[1], self.al0[2], self.al0[3])
        raw[0] = float(np.dot(h_loc, self.al_w)) + self.al_b
        z_all = ep_v @ self.ap_wp + (h_ctx @ self.ap_wc + self.ap_b0)
        h_p = FastSharedTrunkRunner._ln_relu_2d(z_all, self.ap_lw, self.ap_lb)
        ap = (h_p @ self.ap_w3) + self.ap_b3
        if self.fc_active:
            yp = p_mat[:, 5]
            avp = p_mat[:, 3]
            agp = p_mat[:, 7]
            pf = np.stack([np.maximum(0.0, yp - avp), yp * agp], axis=1)
            ap = ap + self.fc_scale * np.tanh(pf @ self.fp)
        raw[1 + v_idx] = ap
        if temp <= 1e-6:
            return int(np.argmax(raw))
        sh = (raw - float(raw[mask].max())) / temp
        pr = np.where(mask, np.exp(np.clip(sh, -40.0, 0.0)), 0.0)
        ps = float(pr.sum())
        return int(np.argmax(raw)) if (ps <= 0.0 or not math.isfinite(ps)) else int(rng.choice(len(mask), p=pr * (1.0 / ps)))


# ==============================================================================
# 4. Worker Execution Routine
# ==============================================================================

def run_single_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    t0 = time.perf_counter()
    seed = spec["seed"]
    agent_name = spec["agent_name"]
    episodes = spec["episodes"]
    warmup = spec.get("warmup", 36)
    fc_scale = spec.get("fcs", 0.008)
    is_dqn = spec.get("is_dqn", False)
    tel_mode = spec.get("tel", "periodic")
    ow = spec.get("ow", False)

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed * 23 + 7)
    rng_a = np.random.default_rng(seed * 41 + 13)

    cfg = dev_tier1_6n(seed=seed)
    ds = build_or_load_curated_trace(cfg.trace, n_nodes=6)
    topo = build_edge_topology(cfg.trace.porto_cells_parquet, cfg.topology)

    fmat = spec.get("fmat", None)

    bv_sim = WireOracleFastSim(
        ds, topo, prepare_task_table(ds, cfg.sim, seed=seed, start_bin=380, end_bin=488),
        cfg, telemetry_mode=tel_mode, forecast_matrix=fmat, obs5_mode="zero", oracle_wire_inflight=ow
    )
    val_sim = WireOracleFastSim(
        ds, topo, prepare_task_table(ds, cfg.sim, split="val", seed=seed),
        cfg, telemetry_mode=tel_mode, forecast_matrix=fmat, obs5_mode="zero", oracle_wire_inflight=ow
    )
    test_sim = WireOracleFastSim(
        ds, topo, prepare_task_table(ds, cfg.sim, split="test", seed=seed),
        cfg, telemetry_mode=tel_mode, forecast_matrix=fmat, obs5_mode="zero", oracle_wire_inflight=ow
    )

    lo_bin = ds.lookback_L
    grps = [(lo_bin, 100), (210, 330), (260, 380), (260, 380), (488, 608), (688, 808)]
    tbls = [prepare_task_table(ds, cfg.sim, seed=seed, start_bin=s, end_bin=e) for s, e in grps]

    cap = 60000
    rb_o = np.zeros((cap, 75), np.float32)
    rb_m = np.zeros((cap, 9), bool)
    rb_g = np.zeros((cap, 91), np.float32)
    rb_a = np.zeros(cap, np.int64)
    rb_r = np.zeros(cap, np.float32)
    rb_no = np.zeros((cap, 75), np.float32)
    rb_nm = np.zeros((cap, 9), bool)
    rb_ng = np.zeros((cap, 91), np.float32)
    ptr = size = 0

    if is_dqn:
        qo = Proactive2QNet(fc_scale=fc_scale)
        qt = copy.deepcopy(qo)
        qt.eval()
        pg = [
            {"params": qo.base_params(), "weight_decay": 2e-4, "lr": 3e-4},
            {"params": qo.fc_res.parameters(), "weight_decay": 2e-2, "lr": 1.5e-4},
        ]
        opt = torch.optim.AdamW(pg, eps=1e-5)
    else:
        ac = SharedTrunkV5(
            use_central_critic=spec["central"],
            actor_uses_global=spec["act_g"],
            fc_scale=fc_scale,
            logit_scale=0.03,
        )
        ac_tgt = copy.deepcopy(ac)
        ac_tgt.eval()
        q_opt = torch.optim.AdamW(
            [
                {"params": ac.q_params(), "weight_decay": 2e-4, "lr": 3e-4},
                {"params": ac.q_fc.parameters(), "weight_decay": 2e-2, "lr": 1.5e-4},
            ],
            eps=1e-5,
        )
        v_opt = torch.optim.AdamW(ac.v_params(), lr=3e-4, weight_decay=2e-4, eps=1e-5)
        a_opt = torch.optim.AdamW(ac.actor_head_params(), lr=1.8e-3, weight_decay=5e-5, eps=1e-5)

    pg_w = 0.08
    T_pg = 0.25
    ema_a = 0.10
    kl_w = 0.06
    T_kl = 0.025

    best_vr = -1e9
    best_ep = 0
    best_tm = None
    history: List[Dict[str, Any]] = []

    eval_eps = {40, 44, 48} if episodes >= 40 else set(range(1, episodes + 1))
    order = []

    for ep in range(episodes):
        fc_on = (fmat is not None) and (ep >= warmup)
        frac = min(1.0, max(0, ep - 2) / 30.0)
        t_roll = 0.002 + 0.5 * (0.012 - 0.002) * (1.0 + math.cos(math.pi * frac))
        eps = 0.30 + min(1.0, ep / 28.0) * (0.01 - 0.30)

        if ep % len(tbls) == 0:
            order = rng.permutation(len(tbls)).tolist()
        sim = WireOracleFastSim(
            ds, topo, tbls[order[ep % len(tbls)]], cfg,
            telemetry_mode=tel_mode, forecast_matrix=fmat if fc_on else None,
            obs5_mode="zero", oracle_wire_inflight=ow
        )

        if is_dqn:
            opt.param_groups[0]["lr"] = 3e-4 if ep < warmup else 8e-5
            qo.fc_res.fc_active = fc_on
            qt.fc_res.fc_active = fc_on
            rn = FastProactive2QRunner(qo)
            ol, ml, al, nl, tl = [], [], [], [], []

            def _sq(o, m_, g, n, t, tid):
                a = rn.act(o, m_, None, rng, t_roll, eps)
                ol.append(o)
                ml.append(m_)
                al.append(a)
                nl.append(n)
                tl.append(tid)
                return a

            tr_m, out = sim.run_marl_episode(_sq)
            oa = np.stack(ol).astype(np.float32)
            ma = np.stack(ml).astype(bool)
            ga = np.zeros((len(oa), 91), np.float32)
            aa = np.array(al, np.int64)
            na = np.array(nl, np.int32)
            ra = np.array([out[t].reward for t in tl], np.float32)
        else:
            q_opt.param_groups[0]["lr"] = 3e-4 if ep < warmup else 8e-5
            ac.q_fc.fc_active = fc_on
            ac_tgt.q_fc.fc_active = fc_on
            ac.act_fc.fc_active = fc_on
            ac_tgt.act_fc.fc_active = fc_on
            alr = 1.8e-3 if ep < 36 else (4e-4 + 0.5 * (1.8e-3 - 4e-4) * (1.0 + math.cos(math.pi * (ep - 36) / 11.0)))
            a_opt.param_groups[0]["lr"] = alr
            rn = FastSharedTrunkRunner(ac, mode="q")
            ol, ml, gl, al, nl, tl = [], [], [], [], [], []

            def _st(o, m_, g, n, t, tid):
                a = rn.act(o, m_, g, rng, t_roll, eps)
                ol.append(o)
                ml.append(m_)
                gl.append(g)
                al.append(a)
                nl.append(n)
                tl.append(tid)
                return a

            tr_m, out = sim.run_marl_episode(_st)
            oa = np.stack(ol).astype(np.float32)
            ma = np.stack(ml).astype(bool)
            ga = np.stack(gl).astype(np.float32)
            aa = np.array(al, np.int64)
            na = np.array(nl, np.int32)
            ra = np.array([out[t].reward for t in tl], np.float32)

        noa, nma, nga = oa.copy(), ma.copy(), ga.copy()
        for nid in range(6):
            ix = np.flatnonzero(na == nid)
            if len(ix) > 1:
                noa[ix[:-1]] = oa[ix[1:]]
                nma[ix[:-1]] = ma[ix[1:]]
                nga[ix[:-1]] = ga[ix[1:]]

        for k in range(len(aa)):
            p = (ptr + k) % cap
            rb_o[p] = oa[k]
            rb_m[p] = ma[k]
            rb_g[p] = ga[k]
            rb_a[p] = aa[k]
            rb_r[p] = ra[k]
            rb_no[p] = noa[k]
            rb_nm[p] = nma[k]
            rb_ng[p] = nga[k]
        ptr = (ptr + len(aa)) % cap
        size = min(cap, size + len(aa))

        if size >= 256:
            if is_dqn:
                qo.train()
                for _ in range(48):
                    ix = rng.integers(0, size, size=256)
                    to, tm, ta, tr, tno, tnm = map(torch.from_numpy, (rb_o[ix], rb_m[ix], rb_a[ix], rb_r[ix], rb_no[ix], rb_nm[ix]))
                    with torch.no_grad():
                        ba = qo(tno, tnm).argmax(dim=-1, keepdim=True)
                        tgt = tr + 0.40 * qt(tno, tnm).gather(1, ba).squeeze(-1)
                    loss = F.smooth_l1_loss(qo(to, tm).gather(1, ta.unsqueeze(-1)).squeeze(-1), tgt, beta=1.0)
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(qo.parameters(), 2.0)
                    opt.step()
                    with torch.no_grad():
                        for p1, p2 in zip(qo.parameters(), qt.parameters()):
                            p2.data.mul_(0.995).add_(p1.data, alpha=0.005)
                qo.eval()
            else:
                ac.train()
                q_all_p = ac.q_params() + list(ac.q_fc.parameters())
                qt_all_p = ac_tgt.q_params() + list(ac_tgt.q_fc.parameters())
                v_p = ac.v_params()
                vt_p = ac_tgt.v_params()
                for _ in range(48):
                    ix = rng.integers(0, size, size=256)
                    to, tm, tg, ta, tr, tno, tnm, tng = map(torch.from_numpy, (rb_o[ix], rb_m[ix], rb_g[ix], rb_a[ix], rb_r[ix], rb_no[ix], rb_nm[ix], rb_ng[ix]))
                    with torch.no_grad():
                        ba = ac.q_values(tno, tnm, tng).argmax(dim=-1, keepdim=True)
                        tgt_q = tr + 0.40 * ac_tgt.q_values(tno, tnm, tng).gather(1, ba).squeeze(-1)
                    ql = F.smooth_l1_loss(ac.q_values(to, tm, tg).gather(1, ta.unsqueeze(-1)).squeeze(-1), tgt_q, beta=1.0)
                    q_opt.zero_grad()
                    ql.backward()
                    nn.utils.clip_grad_norm_(q_all_p, 2.0)
                    q_opt.step()
                    with torch.no_grad():
                        tgt_v = tr + 0.40 * ac_tgt.critic_value(tno, tnm, tng)
                    vl = F.smooth_l1_loss(ac.critic_value(to, tm, tg), tgt_v, beta=1.0)
                    v_opt.zero_grad()
                    vl.backward()
                    nn.utils.clip_grad_norm_(v_p, 2.0)
                    v_opt.step()
                    with torch.no_grad():
                        for p1, p2 in zip(q_all_p + v_p, qt_all_p + vt_p):
                            p2.data.mul_(0.995).add_(p1.data, alpha=0.005)
                        ac.act_fc.peer_lin.weight.data.copy_(ac.q_fc.peer_lin.weight.data)
                        ac_tgt.act_fc.peer_lin.weight.data.copy_(ac_tgt.q_fc.peer_lin.weight.data)

                if ep >= 2:
                    a_all_p = ac.actor_head_params()
                    at_all_p = ac_tgt.actor_head_params()
                    n_act = 128 if ep >= 32 else 80
                    for _ in range(n_act):
                        ix = rng_a.integers(0, size, size=512)
                        to, tm, tg = torch.from_numpy(rb_o[ix]), torch.from_numpy(rb_m[ix]), torch.from_numpy(rb_g[ix])
                        ta_t = torch.from_numpy(rb_a[ix])
                        tr_t = torch.from_numpy(rb_r[ix])
                        tno_t = torch.from_numpy(rb_no[ix])
                        tnm_t = torch.from_numpy(rb_nm[ix])
                        tng_t = torch.from_numpy(rb_ng[ix])
                        with torch.no_grad():
                            h_base_t, h_ctx_t, e_peers_t = ac_tgt._encode(to, tm, tg)
                            v_t = ac_tgt.q_val(h_base_t)
                            qfc_t = ac_tgt.q_fc(to, tm)
                            ap_q = ac_tgt.q_peer(h_ctx_t, e_peers_t) + qfc_t
                            adv_q_raw = torch.cat([ac_tgt.q_loc(h_ctx_t), ap_q], dim=-1)
                            mf = tm.to(adv_q_raw.dtype)
                            vcnt = mf.sum(dim=-1, keepdim=True).clamp(min=1.0)
                            qv_m = torch.where(tm, adv_q_raw, torch.zeros_like(adv_q_raw))
                            adv_q = torch.where(tm, qv_m - qv_m.sum(dim=-1, keepdim=True) / vcnt, torch.zeros_like(adv_q_raw))
                            qc_m = (v_t + adv_q).masked_fill(~tm, -1e9)
                            q_prob = F.softmax(qc_m / T_kl, dim=-1)
                            td_adv = (tr_t + 0.40 * ac_tgt.critic_value(tno_t, tnm_t, tng_t)) - ac_tgt.critic_value(to, tm, tg)
                            td_adv_n = torch.clamp((td_adv - td_adv.mean()) / td_adv.std().clamp(min=0.1), -2.0, 2.0)

                        l0 = ac.act_loc(h_ctx_t)
                        lp = ac.act_peer(h_ctx_t, e_peers_t) + qfc_t / 0.03
                        lg = torch.cat([l0, lp], dim=-1).masked_fill(~tm, -1e9)
                        lg_v = torch.where(tm, lg, torch.zeros_like(lg))
                        adv_pi = torch.where(tm, lg_v - lg_v.sum(dim=-1, keepdim=True) / vcnt, torch.zeros_like(lg))
                        l_reg = (F.smooth_l1_loss(0.03 * adv_pi, adv_q, beta=0.008, reduction="none") * mf).sum() / mf.sum().clamp(min=1.0)
                        logp_all = F.log_softmax((0.03 * adv_pi).masked_fill(~tm, -1e9) / T_kl, dim=-1)
                        l_ce = -(q_prob * torch.where(tm, logp_all, torch.zeros_like(logp_all))).sum(dim=-1).mean()
                        logp_pg = F.log_softmax((0.03 * adv_pi).masked_fill(~tm, -1e9) / T_pg, dim=-1)
                        l_pg = -(td_adv_n * logp_pg.gather(1, ta_t.unsqueeze(-1)).squeeze(-1)).mean()
                        p_loss = l_reg + kl_w * l_ce + pg_w * l_pg
                        a_opt.zero_grad()
                        p_loss.backward()
                        nn.utils.clip_grad_norm_(a_all_p, 1.0)
                        a_opt.step()
                        with torch.no_grad():
                            for p1, p2 in zip(a_all_p, at_all_p):
                                p2.data.mul_(1.0 - ema_a).add_(p1.data, alpha=ema_a)

                ac.eval()

        ep_stats = {
            "episode": ep + 1,
            "train_tsr": tr_m.tsr_pct,
            "train_reward": tr_m.mean_reward,
        }

        if (ep + 1) in eval_eps:
            er = np.random.default_rng(seed)
            if is_dqn:
                rn_eval = FastProactive2QRunner(qt)
                act_fn = lambda o, m_, g, n, t, tid: rn_eval.act(o, m_, None, er, 0.0, 0.0)
            else:
                rn_eval = FastSharedTrunkRunner(ac_tgt, mode="q")
                act_fn = lambda o, m_, g, n, t, tid: rn_eval.act(o, m_, g, er, 0.0, 0.0)

            vb, _ = bv_sim.run_marl_episode(act_fn)
            vc, _ = val_sim.run_marl_episode(act_fn)
            tm, _ = test_sim.run_marl_episode(act_fn)

            vr = 0.75 * vb.mean_reward + 0.25 * vc.mean_reward
            ep_stats["val_burst_tsr"] = vb.tsr_pct
            ep_stats["val_tsr"] = vc.tsr_pct
            ep_stats["val_reward"] = vr
            ep_stats["test_tsr"] = tm.tsr_pct

            if vr > best_vr:
                best_vr = vr
                best_ep = ep + 1
                best_tm = tm

        history.append(ep_stats)

    assert best_tm is not None, "Evaluation failed to generate test metrics"
    wall_s = time.perf_counter() - t0

    res_dict = {
        "agent": agent_name,
        "seed": seed,
        "best_val_ep": best_ep,
        "best_val_reward": float(best_vr),
        "wall_time_s": float(wall_s),
        "history": history,
        "test_metrics": asdict(best_tm),
    }

    return {"agent_name": agent_name, "seed": seed, "result": res_dict}


# ==============================================================================
# 5. Gate Summaries and Invariant Checks
# ==============================================================================

def _aggregate_seed_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [
        "tsr_pct",
        "dmr_pct",
        "drop_admission",
        "drop_deadline",
        "effective_latency_mean_s",
        "effective_latency_p95_s",
        "latency_completed_mean_s",
        "offload_rate_pct",
        "jains_fairness",
        "cb_total_bytes",
        "mean_reward",
    ]
    out: Dict[str, Any] = {"n_seeds": len(runs)}
    for k in keys:
        vals = [float(r["test_metrics"][k]) for r in runs]
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return out


def evaluate_gate_status(
    summary: Dict[str, Dict[str, Any]],
    per_seed: Dict[str, List[Dict[str, Any]]],
    b0_tsr: float,
    e1_tsr: float,
    seeds: List[int],
) -> Dict[str, Any]:
    """Verify strict inequalities on both 3-seed mean and individual seeds."""
    status: Dict[str, Any] = {}

    o1_mean = summary.get("o1_oracle", {}).get("tsr_pct_mean", 0.0)
    e3_mappo_mean = summary.get("e3_mappo", {}).get("tsr_pct_mean", 0.0)
    e2_mappo_mean = summary.get("e2_mappo", {}).get("tsr_pct_mean", 0.0)
    e2_ippo_mean = summary.get("e2_ippo", {}).get("tsr_pct_mean", 0.0)
    e3_madqn_mean = summary.get("e3_madqn", {}).get("tsr_pct_mean", 0.0)
    e2_madqn_mean = summary.get("e2_madqn", {}).get("tsr_pct_mean", 0.0)

    # 3-seed mean checks
    status["mean_o1_ge_e3_mappo"] = bool(o1_mean >= e3_mappo_mean - 1e-4)
    status["mean_e3_mappo_ge_e2_mappo"] = bool(e3_mappo_mean >= e2_mappo_mean - 1e-4)
    status["mean_e2_mappo_ge_e2_ippo"] = bool(e2_mappo_mean >= e2_ippo_mean - 1e-4)
    status["mean_e2_ippo_gt_e1"] = bool(e2_ippo_mean > e1_tsr)
    status["mean_e3_madqn_ge_e2_madqn"] = bool(e3_madqn_mean >= e2_madqn_mean - 1e-4)
    status["mean_e2_madqn_gt_e1"] = bool(e2_madqn_mean > e1_tsr)
    status["mean_e1_gt_b0"] = bool(e1_tsr > b0_tsr)

    # Per-seed checks
    per_seed_ok = True
    seed_details = {}
    for i, s in enumerate(seeds):
        o1_s = per_seed["o1_oracle"][i]["test_metrics"]["tsr_pct"] if "o1_oracle" in per_seed else 0.0
        e3m_s = per_seed["e3_mappo"][i]["test_metrics"]["tsr_pct"] if "e3_mappo" in per_seed else 0.0
        e2m_s = per_seed["e2_mappo"][i]["test_metrics"]["tsr_pct"] if "e2_mappo" in per_seed else 0.0
        ippo_s = per_seed["e2_ippo"][i]["test_metrics"]["tsr_pct"] if "e2_ippo" in per_seed else 0.0
        e3q_s = per_seed["e3_madqn"][i]["test_metrics"]["tsr_pct"] if "e3_madqn" in per_seed else 0.0
        e2q_s = per_seed["e2_madqn"][i]["test_metrics"]["tsr_pct"] if "e2_madqn" in per_seed else 0.0

        cond_mappo = (o1_s >= e3m_s - 1e-4) and (e3m_s >= e2m_s - 1e-4) and (e2m_s >= ippo_s - 1e-4) and (ippo_s > e1_tsr)
        cond_madqn = (e3q_s >= e2q_s - 1e-4) and (e2q_s > e1_tsr)

        seed_ok = cond_mappo and cond_madqn
        if not seed_ok:
            per_seed_ok = False
        seed_details[f"seed_{s}"] = {
            "o1": o1_s,
            "e3_mappo": e3m_s,
            "e2_mappo": e2m_s,
            "e2_ippo": ippo_s,
            "e3_madqn": e3q_s,
            "e2_madqn": e2q_s,
            "passes_monotonic": seed_ok,
        }

    status["per_seed_monotonic"] = per_seed_ok
    status["per_seed_details"] = seed_details

    # Overall gate passed
    all_mean_ok = all(
        status[k]
        for k in [
            "mean_o1_ge_e3_mappo",
            "mean_e3_mappo_ge_e2_mappo",
            "mean_e2_mappo_ge_e2_ippo",
            "mean_e2_ippo_gt_e1",
            "mean_e3_madqn_ge_e2_madqn",
            "mean_e2_madqn_gt_e1",
            "mean_e1_gt_b0",
        ]
    )
    status["gate_passed"] = bool(all_mean_ok and per_seed_ok)
    return status


# ==============================================================================
# 6. Main Orchestrator
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 5 MARL Gate Benchmark (E2, E3, IPPO, MADQN, O1)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--episodes", type=int, default=48)
    parser.add_argument("--window-bins", type=int, default=120)
    parser.add_argument(
        "--agents",
        type=str,
        nargs="+",
        default=["e2_mappo", "e2_ippo", "e2_madqn", "e3_mappo", "e3_madqn", "o1_oracle"],
    )
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    t_start_all = time.perf_counter()
    base = dev_tier1_6n(seed=args.seeds[0])
    base = replace(
        base,
        marl=replace(base.marl, n_train_episodes=args.episodes, train_window_bins=args.window_bins),
    )
    ds = build_or_load_curated_trace(base.trace, n_nodes=base.topology.n_nodes)
    topo = build_edge_topology(base.trace.porto_cells_parquet, base.topology)
    test_tasks = prepare_task_table(ds, base.sim, split="test", seed=base.seed)

    # Heuristic deterministic baselines (independent of RL seed)
    m_b0 = TaskOffloadingSim(ds, topo, test_tasks, base, telemetry_mode="none", verify_invariants=True).run_policy(
        LocalOnlyPolicy()
    )
    m_e1 = TaskOffloadingSim(ds, topo, test_tasks, base, telemetry_mode="periodic", verify_invariants=True).run_policy(
        ReactiveGreedyPolicy(base.sim)
    )

    print(
        f"=== Stage 5 Gate Suite (N={ds.n_nodes}, tasks={test_tasks.n_tasks}, seeds={args.seeds}, eps={args.episodes}) ==="
    )
    print(
        f"  B0 (LocalOnly)   | TSR={m_b0.tsr_pct:6.2f}% | EffLat={m_b0.effective_latency_mean_s:6.1f}s "
        f"| P95={m_b0.effective_latency_p95_s:6.1f}s | Jain={m_b0.jains_fairness:.4f} | Rew={m_b0.mean_reward:.3f}"
    )
    print(
        f"  E1 (Greedy-15s)  | TSR={m_e1.tsr_pct:6.2f}% | EffLat={m_e1.effective_latency_mean_s:6.1f}s "
        f"| P95={m_e1.effective_latency_p95_s:6.1f}s | Jain={m_e1.jains_fairness:.4f} | Rew={m_e1.mean_reward:.3f}"
    )

    selected = list(args.agents)
    agents_results: Dict[str, List[Dict[str, Any]]] = {k: [] for k in selected}

    # Agent config specifications
    arm_specs = {
        "e2_ippo":   {"is_dqn": False, "central": False, "act_g": False, "tel": "periodic", "ow": False, "fc_type": None,        "fcs": 0.008, "warmup": 36},
        "e2_mappo":  {"is_dqn": False, "central": True,  "act_g": False, "tel": "periodic", "ow": False, "fc_type": None,        "fcs": 0.008, "warmup": 36},
        "e3_mappo":  {"is_dqn": False, "central": True,  "act_g": False, "tel": "periodic", "ow": False, "fc_type": "fast_lstm", "fcs": 0.008, "warmup": 36},
        "o1_oracle": {"is_dqn": False, "central": True,  "act_g": True,  "tel": "oracle",   "ow": True,  "fc_type": "oracle",    "fcs": 0.008, "warmup": 36},
        "e2_madqn":  {"is_dqn": True,  "central": False, "act_g": False, "tel": "periodic", "ow": False, "fc_type": None,        "fcs": 0.008, "warmup": 36},
        "e3_madqn":  {"is_dqn": True,  "central": False, "act_g": False, "tel": "periodic", "ow": False, "fc_type": "fast_lstm", "fcs": 0.008, "warmup": 36},
    }

    # Precompute forecast matrices per seed
    oracle_fmat = np.clip(ds.targets.copy(), 0.0, 1.0).astype(np.float32)

    local_fmats: Dict[int, np.ndarray] = {}
    need_local = any(k in ("e3_mappo", "e3_madqn") for k in selected)
    if need_local:
        for s in args.seeds:
            cfg_s = dev_tier1_6n(seed=s)
            cache_path = Path(__file__).resolve().parent.parent.parent / "data" / "alibaba_v2017" / "processed" / f"cache_fast_lstm_s{s}.npz"
            if cache_path.exists():
                local_fmats[s] = np.load(cache_path)["forecast_matrix"]
            else:
                scratch_cache = Path(r"C:\Users\btpUser\.gemini\antigravity\brain\1cb424e4-013a-4681-923b-3c7a1355fcc4\scratch") / f"cache_fast_lstm_s{s}.npz"
                if scratch_cache.exists():
                    local_fmats[s] = np.load(scratch_cache)["forecast_matrix"]
                else:
                    print(f"--- Precomputing causal walk-forward forecast matrix for seed {s} ---", flush=True)
                    fc = generate_causal_forecast_matrices("local_lstm", ds, topo, cfg_s)
                    local_fmats[s] = fc["forecast_matrix"]

    # Build work specs
    tasks: List[Dict[str, Any]] = []
    for s in args.seeds:
        for k in selected:
            cfg_a = arm_specs[k]
            fc_type = cfg_a["fc_type"]
            if fc_type == "oracle":
                fmat = oracle_fmat
            elif fc_type == "fast_lstm":
                fmat = local_fmats[s]
            else:
                fmat = None

            tasks.append({
                "agent_name": k,
                "seed": s,
                "episodes": args.episodes,
                "window_bins": args.window_bins,
                "warmup": cfg_a["warmup"],
                "fcs": cfg_a["fcs"],
                "is_dqn": cfg_a["is_dqn"],
                "central": cfg_a["central"],
                "act_g": cfg_a["act_g"],
                "tel": cfg_a["tel"],
                "ow": cfg_a["ow"],
                "fmat": fmat,
            })

    print(f"\nExecuting {len(tasks)} benchmark tasks with {args.workers} workers in parallel...", flush=True)
    raw_results = []
    batch_size = max(1, args.workers)
    for b_start in range(0, len(tasks), batch_size):
        batch = tasks[b_start : b_start + batch_size]
        with ProcessPoolExecutor(max_workers=len(batch)) as ex:
            futs = [ex.submit(run_single_spec, sp) for sp in batch]
            for fut in as_completed(futs):
                r = fut.result()
                raw_results.append(r)
                k = r["agent_name"]
                seed = r["seed"]
                tm = r["result"]["test_metrics"]
                best_ep = r["result"]["best_val_ep"]
                wall_s = r["result"]["wall_time_s"]
                print(
                    f"  [s={seed:2d}] {k:10s} | TSR={tm['tsr_pct']:6.2f}% | EffLat={tm['effective_latency_mean_s']:5.1f}s "
                    f"| P95={tm['effective_latency_p95_s']:5.1f}s | Off={tm['offload_rate_pct']:5.1f}% "
                    f"| Jain={tm['jains_fairness']:.4f} | Rew={tm['mean_reward']:.3f} "
                    f"| bestEp={best_ep:2d} | t={wall_s:5.1f}s",
                    flush=True,
                )

    # Sort results into original agent/seed order
    for k in selected:
        for s in args.seeds:
            match = [r["result"] for r in raw_results if r["agent_name"] == k and r["seed"] == s]
            if match:
                agents_results[k].append(match[0])

    summary = {k: _aggregate_seed_runs(v) for k, v in agents_results.items()}
    print("\n=== Stage 5 Multi-Seed Summary ===", flush=True)
    for k, s in summary.items():
        print(
            f"  {k:10s} | TSR={s['tsr_pct_mean']:6.2f}%±{s['tsr_pct_std']:4.2f} "
            f"| EffLat={s['effective_latency_mean_s_mean']:5.1f}±{s['effective_latency_mean_s_std']:4.1f}s "
            f"| P95={s['effective_latency_p95_s_mean']:5.1f}s "
            f"| Off={s['offload_rate_pct_mean']:5.1f}% "
            f"| Jain={s['jains_fairness_mean']:.4f} "
            f"| Rew={s['mean_reward_mean']:.3f}±{s['mean_reward_std']:.3f}",
            flush=True,
        )

    gate_status = evaluate_gate_status(
        summary, agents_results, m_b0.tsr_pct, m_e1.tsr_pct, args.seeds
    )
    print("\n=== Gate Acceptance Status ===", flush=True)
    print(f"  All Mean Inequalities Passed: {all(gate_status[k] for k in ['mean_o1_ge_e3_mappo', 'mean_e3_mappo_ge_e2_mappo', 'mean_e2_mappo_ge_e2_ippo', 'mean_e2_ippo_gt_e1', 'mean_e3_madqn_ge_e2_madqn', 'mean_e2_madqn_gt_e1', 'mean_e1_gt_b0'])}")
    print(f"  Per-Seed Monotonicity Passed: {gate_status['per_seed_monotonic']}")
    print(f"  OVERALL STAGE 5 GATE PASSED : {gate_status['gate_passed']}")

    out_report = {
        "n_nodes": ds.n_nodes,
        "seeds": args.seeds,
        "episodes": args.episodes,
        "train_window_bins": args.window_bins,
        "wall_total_s": float(time.perf_counter() - t_start_all),
        "baselines": {"b0_local": asdict(m_b0), "e1_greedy": asdict(m_e1)},
        "gate_status": gate_status,
        "summary": summary,
        "per_seed": agents_results,
    }
    out_path = Path(__file__).resolve().parent.parent.parent / "results" / "phase3" / "stage5_marl_gate_6n.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_report, indent=2), encoding="utf-8")
    print(f"\nSaved Stage 5 Gate report -> {out_path} (total {out_report['wall_total_s']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
