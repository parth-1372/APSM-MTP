import heapq
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from phase3.config import Phase3Config, SimConfig
from phase3.data.alibaba_loader import CuratedTraceDataset
from phase3.data.topology import EdgeTopology


@dataclass(frozen=True)
class PreparedTaskTable:
    """
    Deterministic task table for a dataset split, completely decoupled from policy choices
    so all baselines & MARL agents face identical arrivals, work, sizes, and deadlines.
    """

    split_name: str
    t_ep_start_s: float
    t_ep_end_s: float
    n_slots: int
    n_tasks: int
    arr_ts_s: np.ndarray  # (M,) float64
    arr_slot: np.ndarray  # (M,) int32 relative to t_ep_start_s
    node_idx: np.ndarray  # (M,) int32
    cpu_work: np.ndarray  # (M,) float32
    avg_real_cpu: np.ndarray  # (M,) float32
    r_nom: np.ndarray  # (M,) float32 nominal cores/s execution rate
    mem_req: np.ndarray  # (M,) float32 scaled memory fraction
    payload_bytes: np.ndarray  # (M,) int32
    deadline_s: np.ndarray  # (M,) float64 absolute deadline timestamp


@dataclass
class ActiveTask:
    task_id: int
    origin_id: int
    target_id: int
    arr_ts_s: float
    deadline_s: float
    cpu_work: float
    work_rem: float
    r_nom: float
    mem_req: float
    payload_bytes: int
    hop_count: int
    already_offloaded: bool
    net_delay_s: float
    target_util_at_dec: float
    nbr_mean_util_at_dec: float
    decision_obs: Optional[np.ndarray] = None


@dataclass
class PeerCacheEntry:
    last_rx_ts: float
    last_created_ts: float
    q_norm: float
    avail_cpu_norm: float
    avail_mem_norm: float
    y_hat: float


@dataclass
class ResolvedOutcome:
    task_id: int
    origin_id: int
    target_id: int
    was_offloaded: bool
    dropped: bool
    drop_reason: str  # "none" | "admission_overflow" | "deadline_timeout" | "link_timeout"
    latency_s: float
    net_delay_s: float
    dl_budget_s: float
    reward: float


@dataclass
class SimMetrics:
    total_tasks: int
    completed_on_time: int
    dropped_tasks: int
    drop_admission: int
    drop_deadline: int
    drop_link_timeout: int
    tsr_pct: float
    dmr_pct: float
    latency_completed_mean_s: float
    latency_completed_p95_s: float
    latency_completed_p99_s: float
    effective_latency_mean_s: float
    effective_latency_p95_s: float
    offload_count: int
    offload_rate_pct: float
    offload_data_bytes: int
    telemetry_packets: int
    telemetry_bytes: int
    telemetry_suppressed: int
    ack_packets: int
    ack_bytes: int
    model_packets: int
    model_bytes: int
    cb_total_bytes: int
    mean_cpu_util: float
    jains_fairness: float
    util_cv: float
    mean_reward: float
    per_node_tsr_pct: List[float]
    per_node_cpu_util: List[float]


def prepare_task_table(
    dataset: CuratedTraceDataset,
    sim_cfg: SimConfig,
    split: Literal["train", "val", "test"] = "test",
    seed: int = 12345,
    start_bin: Optional[int] = None,
    end_bin: Optional[int] = None,
) -> PreparedTaskTable:
    """
    Build deterministic episode task arrays strictly inside the requested chronological split.
    Zero future split contamination: only rows in [t_ep_start, t_ep_end) are materialized.
    """
    t0 = float(dataset.t_start_s)
    dt = float(dataset.dt_bin_s)

    if start_bin is not None and end_bin is not None:
        b_start, b_end = int(start_bin), int(end_bin)
        split_label = f"custom_{b_start}_{b_end}"
    elif split == "train":
        b_start, b_end = dataset.lookback_L, dataset.train_end_bin
        split_label = "train"
    elif split == "val":
        b_start, b_end = dataset.train_end_bin, dataset.val_end_bin
        split_label = "val"
    else:
        b_start, b_end = dataset.val_end_bin, dataset.n_bins
        split_label = "test"

    t_ep_start = t0 + b_start * dt
    t_ep_end = t0 + b_end * dt
    n_slots = int(round((t_ep_end - t_ep_start) / sim_cfg.dt_slot_s))

    df = pd.read_parquet(dataset.task_events_path)
    sub = df[(df["start_ts"] >= t_ep_start) & (df["start_ts"] < t_ep_end)].copy()
    sub = sub.sort_values(["start_ts", "node_idx", "job_id", "task_id"]).reset_index(drop=True)

    m = len(sub)
    arr_ts = sub["start_ts"].to_numpy(dtype=np.float64)
    arr_slot = np.clip(
        np.floor((arr_ts - t_ep_start) / sim_cfg.dt_slot_s).astype(np.int32),
        0,
        max(0, n_slots - 1),
    )
    node_idx = sub["node_idx"].to_numpy(dtype=np.int32)
    cpu_work = sub["cpu_work"].to_numpy(dtype=np.float32)
    avg_cpu = sub["avg_real_cpu"].to_numpy(dtype=np.float32)
    mem_req = np.clip(
        sub["max_norm_mem"].to_numpy(dtype=np.float32) * sim_cfg.task_mem_scale,
        0.002,
        0.25,
    ).astype(np.float32)
    r_nom = np.maximum(1.0, avg_cpu * sim_cfg.task_nom_cpu_mult).astype(np.float32)
    t_exec_base = cpu_work / r_nom
    deadline_s = arr_ts + sim_cfg.deadline_base_slack_s + sim_cfg.deadline_slack_kappa * t_exec_base

    # Deterministic payload synthesis independent of routing policy ([Assumption])
    split_offset = {"train": 1, "val": 2, "test": 3}.get(split_label, 4)
    rng = np.random.default_rng(seed * 31 + split_offset)
    log_min = math.log(float(sim_cfg.payload_bytes_min))
    log_max = math.log(float(sim_cfg.payload_bytes_max))
    payload_bytes = np.exp(rng.uniform(log_min, log_max, size=m)).astype(np.int32)

    return PreparedTaskTable(
        split_name=split_label,
        t_ep_start_s=t_ep_start,
        t_ep_end_s=t_ep_end,
        n_slots=n_slots,
        n_tasks=m,
        arr_ts_s=arr_ts,
        arr_slot=arr_slot,
        node_idx=node_idx,
        cpu_work=cpu_work,
        avg_real_cpu=avg_cpu,
        r_nom=r_nom,
        mem_req=mem_req,
        payload_bytes=payload_bytes,
        deadline_s=deadline_s,
    )


class TaskOffloadingSim:
    """
    Deterministic Discrete-Slot / Sub-Second-Link-Queue Edge Task Offloading Simulator.
    Enforces strict Stage 4 invariants:
      - `max_hops = 1` anti-loop routing
      - Strict causal observations (`info_age > 0` for non-oracle modes)
      - Work & memory conservation across CFS-style proportional sharing
      - Exact separation of Control-Plane Bytes vs Offload Data-Plane Bytes (`H2`)
    """

    def __init__(
        self,
        dataset: CuratedTraceDataset,
        topo: EdgeTopology,
        tasks: PreparedTaskTable,
        cfg: Phase3Config,
        telemetry_mode: Literal["periodic", "apsm_t", "oracle", "none"] = "periodic",
        forecast_matrix: Optional[np.ndarray] = None,
        surprise_matrix: Optional[np.ndarray] = None,
        extra_model_packets: int = 0,
        extra_model_bytes: int = 0,
        apsm_t_threshold: float = 0.12,
        apsm_t_heartbeat: int = 4,
        verify_invariants: bool = False,
    ) -> None:
        self.dataset = dataset
        self.topo = topo
        self.tasks = tasks
        self.cfg = cfg
        self.sim = cfg.sim
        self.n_nodes = dataset.n_nodes
        self.telemetry_mode = telemetry_mode
        self.forecast_matrix = forecast_matrix
        self.surprise_matrix = surprise_matrix
        self.extra_model_packets = int(extra_model_packets)
        self.extra_model_bytes = int(extra_model_bytes)
        self.apsm_t_threshold = float(apsm_t_threshold)
        self.apsm_t_heartbeat = int(apsm_t_heartbeat)
        self.verify_invariants = bool(verify_invariants)

        # Group tasks cleanly by arrival slot
        self._slot_task_indices: List[List[int]] = [[] for _ in range(tasks.n_slots)]
        for idx, sl in enumerate(tasks.arr_slot):
            self._slot_task_indices[int(sl)].append(idx)

        self.reset()

    def reset(self) -> None:
        n = self.n_nodes
        self.active_tasks: List[List[ActiveTask]] = [[] for _ in range(n)]
        self.active_mem: np.ndarray = np.zeros(n, dtype=np.float64)
        self.link_free_at: np.ndarray = np.zeros((n, n), dtype=np.float64)
        self.link_free_at.fill(self.tasks.t_ep_start_s)

        # Event heaps: (deliver_ts, seq_id, payload)
        self._event_seq: int = 0
        self._link_rx_heap: List[Tuple[float, int, ActiveTask]] = []
        self._beacon_heap: List[Tuple[float, int, int, int, PeerCacheEntry]] = []
        self._ack_heap: List[Tuple[float, int, ActiveTask, bool, str, float]] = []
        self._wire_cnt: np.ndarray = np.zeros((n, n), dtype=np.int32)

        # Per-node neighbor cache indexed by receiver i -> sender j
        self.neighbor_cache: List[Dict[int, PeerCacheEntry]] = [{} for _ in range(n)]
        # Exact in-flight tracking: origin i -> peer j -> dict[task_id -> t_arr_peer]
        self.inflight_records: List[Dict[int, Dict[int, float]]] = [
            {j: {} for j in self.topo.neighbor_slots(i)} for i in range(n)
        ]

        # APSM-T state per node j
        self._last_tx_state: List[Optional[Tuple[float, float, float]]] = [None] * n
        self._apsm_t_supp_streak: List[int] = [0] * n

        # Slow-plane causal workload forecast cache (updated per bin)
        self.cur_y_hat: np.ndarray = np.zeros(n, dtype=np.float32)
        self.cur_surprise: np.ndarray = np.zeros(n, dtype=np.float32)
        self.cur_load_norm: np.ndarray = np.zeros(n, dtype=np.float32)

        # Current causal background utils
        self._cur_bg_cpu: np.ndarray = np.zeros(n, dtype=np.float32)
        self._cur_bg_mem: np.ndarray = np.zeros(n, dtype=np.float32)

        # Telemetry & H2 counters
        self.offload_count: int = 0
        self.offload_data_bytes: int = 0
        self.telemetry_packets: int = 0
        self.telemetry_bytes: int = 0
        self.telemetry_suppressed: int = 0
        self.ack_packets: int = 0
        self.ack_bytes: int = 0
        self.dynamic_model_packets: int = 0
        self.dynamic_model_bytes: int = 0

        self.node_cpu_used: np.ndarray = np.zeros(n, dtype=np.float64)
        self.node_cpu_avail_acc: np.ndarray = np.zeros(n, dtype=np.float64)
        self.node_total_tasks: np.ndarray = np.zeros(n, dtype=np.int64)
        self.node_ontime_tasks: np.ndarray = np.zeros(n, dtype=np.int64)

        self.outcomes: List[ResolvedOutcome] = []
        self._pending_task_store: Dict[int, ActiveTask] = {}

    def add_model_traffic(self, packets: int, n_bytes: int) -> None:
        self.dynamic_model_packets += int(packets)
        self.dynamic_model_bytes += int(n_bytes)

    def _causal_bg_bin(self, t_now_s: float) -> int:
        dt = float(self.dataset.dt_bin_s)
        # Strictly closed bin prior to or at t_now_s (zero future 300s snapshot leak)
        raw_b = int(math.floor((t_now_s - self.dataset.t_start_s - 1e-6) / dt))
        return max(0, min(self.dataset.n_bins - 1, raw_b))

    def _refresh_slow_context(self, t_now_s: float) -> int:
        b_bg = self._causal_bg_bin(t_now_s)
        self._cur_bg_cpu = np.minimum(self.sim.bg_cpu_cap, self.dataset.bg_cpu_util[:, b_bg])
        self._cur_bg_mem = np.minimum(self.sim.bg_mem_cap, self.dataset.bg_mem_util[:, b_bg])
        self.cur_load_norm = self.dataset.targets[:, b_bg]

        dt = float(self.dataset.dt_bin_s)
        b_cur = max(0, min(self.dataset.n_bins - 1, int(math.floor((t_now_s - self.dataset.t_start_s) / dt))))
        if self.forecast_matrix is not None:
            self.cur_y_hat = self.forecast_matrix[:, b_cur]
        if self.surprise_matrix is not None:
            self.cur_surprise = self.surprise_matrix[:, b_cur]
        return b_cur

    def _node_free_mem(self, i: int) -> float:
        # Enforce safe positive batch headroom against peak bg_mem cap
        batch_mem_budget = max(0.30, self.sim.node_mem_cap - float(self._cur_bg_mem[i]))
        return max(0.0, batch_mem_budget - float(self.active_mem[i]))

    def _node_avail_cpu_rate(self, i: int) -> float:
        return self.sim.node_cpu_cap * max(0.05, 1.0 - float(self._cur_bg_cpu[i]))

    def _node_avail_cpu_norm(self, i: int) -> float:
        # Available fraction subtracting active nominal queued CPU demand pressure
        base_avail = max(0.05, 1.0 - float(self._cur_bg_cpu[i]))
        demand_cores = sum(t.r_nom for t in self.active_tasks[i])
        headroom = max(0.0, base_avail - (demand_cores / self.sim.node_cpu_cap))
        return float(headroom)

    def _node_util_norm(self, i: int) -> float:
        base_used = float(self._cur_bg_cpu[i])
        demand_cores = sum(t.r_nom for t in self.active_tasks[i])
        return float(min(1.5, base_used + demand_cores / self.sim.node_cpu_cap))

    def _try_admit(self, node_id: int, task: ActiveTask) -> bool:
        if len(self.active_tasks[node_id]) >= self.sim.q_max:
            return False
        if self._node_free_mem(node_id) + 1e-9 < task.mem_req:
            return False
        self.active_tasks[node_id].append(task)
        self.active_mem[node_id] += float(task.mem_req)
        if self.verify_invariants:
            assert self.active_mem[node_id] <= max(0.30, self.sim.node_mem_cap - float(self._cur_bg_mem[node_id])) + 1e-6
        return True

    def _emit_outcome(
        self,
        task: ActiveTask,
        dropped: bool,
        drop_reason: str,
        resolved_ts: float,
    ) -> ResolvedOutcome:
        dl_budget = max(1.0, task.deadline_s - task.arr_ts_s)
        lat = max(0.0, min(dl_budget, resolved_ts - task.arr_ts_s)) if not dropped else dl_budget
        if not dropped:
            lat_norm = min(1.5, max(0.0, (resolved_ts - task.arr_ts_s) / dl_budget))
        else:
            lat_norm = 1.0

        net_norm = min(1.0, task.net_delay_s / dl_budget) if task.already_offloaded else 0.0
        imb_pen = abs(task.target_util_at_dec - task.nbr_mean_util_at_dec)
        reward = -(
            self.sim.reward_w_lat * lat_norm
            + self.sim.reward_w_drop * (1.0 if dropped else 0.0)
            + self.sim.reward_w_net * net_norm
            + self.sim.reward_w_imb * imb_pen
        )

        self.node_total_tasks[task.origin_id] += 1
        if not dropped:
            self.node_ontime_tasks[task.origin_id] += 1

        out = ResolvedOutcome(
            task_id=task.task_id,
            origin_id=task.origin_id,
            target_id=task.target_id,
            was_offloaded=task.already_offloaded,
            dropped=dropped,
            drop_reason=drop_reason,
            latency_s=float(max(0.0, resolved_ts - task.arr_ts_s)) if not dropped else float(dl_budget),
            net_delay_s=float(task.net_delay_s),
            dl_budget_s=float(dl_budget),
            reward=float(reward),
        )
        self.outcomes.append(out)
        return out

    def _schedule_ack(self, task: ActiveTask, dropped: bool, drop_reason: str, event_ts: float) -> None:
        if not task.already_offloaded:
            self._emit_outcome(task, dropped, drop_reason, event_ts)
            return
        i, j = task.origin_id, task.target_id
        bw_bps = float(self.topo.bw_mbps[j, i]) * 1e6
        ack_tx = float(self.topo.rtt_s[j, i]) * 0.5 + (8.0 * self.sim.ack_bytes) / max(1e6, bw_bps)
        deliver_ts = event_ts + ack_tx
        self._event_seq += 1
        heapq.heappush(self._ack_heap, (deliver_ts, self._event_seq, task, dropped, drop_reason, event_ts))

    def _emit_telemetry_check(self, t_now_s: float) -> None:
        if self.telemetry_mode in ("none", "oracle"):
            return

        beacon_bits = 8.0 * self.sim.telemetry_beacon_bytes
        for j in range(self.n_nodes):
            q_norm = len(self.active_tasks[j]) / float(self.sim.q_max)
            avail_cpu_n = self._node_avail_cpu_norm(j)
            avail_mem_n = min(1.0, self._node_free_mem(j))
            yhat_j = float(self.cur_y_hat[j])

            should_send = True
            if self.telemetry_mode == "apsm_t":
                prev = self._last_tx_state[j]
                if prev is not None and self._apsm_t_supp_streak[j] < self.apsm_t_heartbeat:
                    dq = abs(q_norm - prev[0])
                    dc = abs(avail_cpu_n - prev[1])
                    dy = 0.5 * abs(yhat_j - prev[2])
                    if max(dq, dc, dy) <= self.apsm_t_threshold:
                        should_send = False
                        self._apsm_t_supp_streak[j] += 1
                        self.telemetry_suppressed += 1

            if not should_send:
                continue

            self._apsm_t_supp_streak[j] = 0
            self._last_tx_state[j] = (q_norm, avail_cpu_n, yhat_j)
            for nbr in self.topo.neighbor_slots(j):
                bw_bps = float(self.topo.bw_mbps[j, nbr]) * 1e6
                tx_lat = float(self.topo.rtt_s[j, nbr]) * 0.5 + beacon_bits / max(1e6, bw_bps)
                entry = PeerCacheEntry(
                    last_rx_ts=t_now_s + tx_lat,
                    last_created_ts=t_now_s,
                    q_norm=float(q_norm),
                    avail_cpu_norm=float(avail_cpu_n),
                    avail_mem_norm=float(avail_mem_n),
                    y_hat=yhat_j,
                )
                self._event_seq += 1
                heapq.heappush(self._beacon_heap, (t_now_s + tx_lat, self._event_seq, nbr, j, entry))
                self.telemetry_packets += 1
                self.telemetry_bytes += self.sim.telemetry_beacon_bytes

    def build_observation(
        self, node_id: int, task_idx: int, t_now_s: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build strictly causal observation `o_i(t)`, valid discrete action mask `(1 + K_max,)`,
        and offline CTDE global summary `(obs_dim + 16,)` for centralized MAPPO critic.
        """
        k_max = self.sim.k_max_neighbors
        obs_dim = 11 + 8 * k_max
        obs = np.zeros(obs_dim, dtype=np.float32)
        mask = np.zeros(1 + k_max, dtype=bool)
        mask[0] = True  # Local execution always permitted in policy space

        # 1. Self operational state (4 dims)
        obs[0] = len(self.active_tasks[node_id]) / float(self.sim.q_max)
        obs[1] = self._node_avail_cpu_norm(node_id)
        obs[2] = min(1.0, self._node_free_mem(node_id))
        obs[3] = float(self._cur_bg_cpu[node_id])

        # 2. Self forecast & semantic surprise (3 dims; 0 if no forecaster attached)
        y_hat_i = float(self.cur_y_hat[node_id]) if self.forecast_matrix is not None else 0.0
        obs[4] = y_hat_i
        obs[5] = 0.0
        obs[6] = float(self.cur_surprise[node_id]) if self.surprise_matrix is not None else 0.0

        # 3. Current arriving task features (4 dims)
        w_m = float(self.tasks.cpu_work[task_idx])
        r_m = float(self.tasks.mem_req[task_idx])
        b_m = float(self.tasks.payload_bytes[task_idx])
        slack_s = float(self.tasks.deadline_s[task_idx] - t_now_s)
        obs[7] = min(2.0, w_m / self.sim.obs_w_scale)
        obs[8] = min(2.0, r_m / 0.25)
        obs[9] = min(2.0, b_m / self.sim.obs_bytes_scale)
        obs[10] = min(2.0, max(0.0, slack_s) / self.sim.obs_dl_scale)

        # 4. Fixed-slot neighbor summary (8 dims per slot)
        nbrs = self.topo.neighbor_slots(node_id)
        for k_idx, peer_id in enumerate(nbrs[:k_max]):
            mask[1 + k_idx] = True
            base = 11 + 8 * k_idx
            obs[base + 0] = 1.0  # valid_mask
            obs[base + 1] = min(1.0, float(self.topo.rtt_s[node_id, peer_id]) / 0.05)
            obs[base + 2] = min(1.0, float(self.topo.bw_mbps[node_id, peer_id]) / 200.0)

            if self.telemetry_mode == "oracle":
                obs[base + 3] = self._node_avail_cpu_norm(peer_id)
                obs[base + 4] = len(self.active_tasks[peer_id]) / float(self.sim.q_max)
                obs[base + 5] = float(self.cur_y_hat[peer_id]) if self.forecast_matrix is not None else 0.0
                obs[base + 6] = min(1.0, float(self._wire_cnt[node_id, peer_id]) / float(self.sim.q_max))
                obs[base + 7] = 0.0
            else:
                entry = self.neighbor_cache[node_id].get(peer_id)
                if entry is not None:
                    if self.verify_invariants:
                        assert entry.last_created_ts <= t_now_s + 1e-9
                        assert entry.last_rx_ts <= t_now_s + 1e-9
                    obs[base + 3] = entry.avail_cpu_norm
                    obs[base + 4] = entry.q_norm
                    obs[base + 5] = entry.y_hat
                    obs[base + 7] = min(1.0, max(0.0, t_now_s - entry.last_created_ts) / self.sim.info_age_max_s)
                else:
                    obs[base + 3] = max(0.05, 1.0 - float(self._cur_bg_cpu[peer_id]))
                    obs[base + 4] = 0.0
                    obs[base + 5] = 0.0
                    obs[base + 7] = 1.0
                inflight_cnt = len(self.inflight_records[node_id][peer_id])
                obs[base + 6] = min(1.0, inflight_cnt / float(self.sim.q_max))

        # 5. Offline CTDE global critic vector (fixed 16 extra dims -> works identically at N=6..100)
        qs = np.array([len(self.active_tasks[n_i]) / float(self.sim.q_max) for n_i in range(self.n_nodes)], dtype=np.float32)
        avails = np.array([self._node_avail_cpu_norm(n_i) for n_i in range(self.n_nodes)], dtype=np.float32)
        mems = np.array([min(1.0, self._node_free_mem(n_i)) for n_i in range(self.n_nodes)], dtype=np.float32)
        yhats = self.cur_y_hat if self.forecast_matrix is not None else np.zeros(self.n_nodes, dtype=np.float32)
        nbr_qs = np.array([qs[p] for p in nbrs], dtype=np.float32) if nbrs else np.zeros(1, dtype=np.float32)
        nbr_av = np.array([avails[p] for p in nbrs], dtype=np.float32) if nbrs else np.zeros(1, dtype=np.float32)
        total_inflight = (
            float(self._wire_cnt.sum())
            if self.telemetry_mode == "oracle"
            else sum(
                len(self.inflight_records[u][v]) for u in range(self.n_nodes) for v in self.inflight_records[u]
            )
        )
        summary = np.array(
            [
                float(qs.mean()),
                float(qs.std()),
                float(qs.max()),
                float(qs.min()),
                float(avails.mean()),
                float(avails.min()),
                float(avails.max()),
                float(mems.mean()),
                float(yhats.mean()),
                float(yhats.max()),
                float(total_inflight / max(1, self.n_nodes * self.sim.q_max)),
                float(len(self.topo.neighbor_slots(node_id)) / float(k_max)),
                float(nbr_qs.min()),
                float(nbr_av.max()),
                float(nbr_qs.mean()),
                float(nbr_av.mean()),
            ],
            dtype=np.float32,
        )
        global_state = np.concatenate([obs, summary], axis=0)
        return obs, mask, global_state

    def _deliver_due_events(self, t_now_s: float) -> None:
        # Step A: Process arriving offloaded tasks (strictly max_hops=1, NEVER routed again)
        while self._link_rx_heap and self._link_rx_heap[0][0] <= t_now_s + 1e-9:
            arr_peer_ts, _, task = heapq.heappop(self._link_rx_heap)
            self._wire_cnt[task.origin_id, task.target_id] = max(
                0, int(self._wire_cnt[task.origin_id, task.target_id]) - 1
            )
            if self.verify_invariants:
                assert task.already_offloaded is True and task.hop_count == 1
            if arr_peer_ts > task.deadline_s:
                self._schedule_ack(task, True, "link_timeout", arr_peer_ts)
            elif not self._try_admit(task.target_id, task):
                self._schedule_ack(task, True, "admission_overflow", arr_peer_ts)

        # Step B1: Deliver telemetry beacons into neighbor caches and prune seen inflight items
        while self._beacon_heap and self._beacon_heap[0][0] <= t_now_s + 1e-9:
            _, _, receiver, sender, entry = heapq.heappop(self._beacon_heap)
            self.neighbor_cache[receiver][sender] = entry
            infl = self.inflight_records[receiver][sender]
            done_ids = [tid for tid, arr_t in infl.items() if arr_t <= entry.last_created_ts + 1e-9]
            for tid in done_ids:
                del infl[tid]

        # Step B2: Deliver task outcome ACKs cleanly back to origin node
        while self._ack_heap and self._ack_heap[0][0] <= t_now_s + 1e-9:
            _, _, task, dropped, reason, ev_ts = heapq.heappop(self._ack_heap)
            self.ack_packets += 1
            self.ack_bytes += self.sim.ack_bytes
            self.inflight_records[task.origin_id][task.target_id].pop(task.task_id, None)
            self._emit_outcome(task, dropped, reason, ev_ts)

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

            sum_nom = sum(t.r_nom for t in q)
            scale = min(1.0, avail_cpu / sum_nom) if sum_nom > 1e-9 else 1.0
            if in_episode:
                self.node_cpu_used[i] += min(avail_cpu, sum_nom) * dt

            survivors: List[ActiveTask] = []
            for task in q:
                rate = float(task.r_nom) * scale
                delta_w = rate * dt
                if self.verify_invariants:
                    assert 0.0 < task.work_rem <= task.cpu_work + 1e-6

                if task.work_rem <= delta_w + 1e-7:
                    t_fin = t_now_s + (task.work_rem / max(1e-9, rate))
                    task.work_rem = 0.0
                    self.active_mem[i] = max(0.0, float(self.active_mem[i]) - float(task.mem_req))
                    if t_fin <= task.deadline_s + 1e-9:
                        self._schedule_ack(task, False, "none", t_fin)
                    else:
                        self._schedule_ack(task, True, "deadline_timeout", task.deadline_s)
                else:
                    task.work_rem -= delta_w
                    if t_end_slot >= task.deadline_s - 1e-9:
                        self.active_mem[i] = max(0.0, float(self.active_mem[i]) - float(task.mem_req))
                        self._schedule_ack(task, True, "deadline_timeout", task.deadline_s)
                    else:
                        survivors.append(task)
            self.active_tasks[i] = survivors
            if not survivors:
                self.active_mem[i] = 0.0

    def _dispatch_task(self, task_idx: int, action: int, t_now_s: float) -> None:
        i = int(self.tasks.node_idx[task_idx])
        nbrs = self.topo.neighbor_slots(i)
        nbr_mean_util = (
            float(np.mean([self._node_util_norm(p) for p in nbrs])) if nbrs else self._node_util_norm(i)
        )

        if action == 0:
            target_id = i
            already_offloaded = False
            hop_count = 0
            net_delay_s = 0.0
            target_util_dec = self._node_util_norm(i)
        else:
            peer_id = nbrs[action - 1]
            target_id = peer_id
            already_offloaded = True
            hop_count = 1
            bw_bps = float(self.topo.bw_mbps[i, peer_id]) * 1e6
            rtt_s = float(self.topo.rtt_s[i, peer_id])
            payload_b = int(self.tasks.payload_bytes[task_idx])
            tx_start = max(t_now_s, float(self.link_free_at[i, peer_id]))
            d_tx = (8.0 * payload_b) / max(1e6, bw_bps)
            tx_end = tx_start + d_tx
            self.link_free_at[i, peer_id] = tx_end
            arr_peer_ts = tx_end + 0.5 * rtt_s
            net_delay_s = arr_peer_ts - t_now_s
            target_util_dec = self._node_util_norm(peer_id)
            self.offload_count += 1
            self.offload_data_bytes += payload_b + self.sim.task_header_bytes

        task = ActiveTask(
            task_id=task_idx,
            origin_id=i,
            target_id=target_id,
            arr_ts_s=float(self.tasks.arr_ts_s[task_idx]),
            deadline_s=float(self.tasks.deadline_s[task_idx]),
            cpu_work=float(self.tasks.cpu_work[task_idx]),
            work_rem=float(self.tasks.cpu_work[task_idx]),
            r_nom=float(self.tasks.r_nom[task_idx]),
            mem_req=float(self.tasks.mem_req[task_idx]),
            payload_bytes=int(self.tasks.payload_bytes[task_idx]),
            hop_count=hop_count,
            already_offloaded=already_offloaded,
            net_delay_s=float(net_delay_s),
            target_util_at_dec=float(target_util_dec),
            nbr_mean_util_at_dec=float(nbr_mean_util),
        )

        if not already_offloaded:
            if not self._try_admit(i, task):
                self._emit_outcome(task, True, "admission_overflow", t_now_s)
        else:
            arr_peer_ts = t_now_s + net_delay_s
            self.inflight_records[i][target_id][task.task_id] = arr_peer_ts
            self._wire_cnt[i, target_id] += 1
            self._event_seq += 1
            heapq.heappush(self._link_rx_heap, (arr_peer_ts, self._event_seq, task))

    def _run_episode_loop(
        self,
        policy_fn: Callable[..., int],
        pass_global_state: bool,
        bin_callback: Optional[Callable[[int, float], None]] = None,
        drain_max_s: int = 20_000,
    ) -> Tuple[SimMetrics, Dict[int, ResolvedOutcome]]:
        self.reset()
        dt_bin = int(self.dataset.dt_bin_s)
        t0 = float(self.dataset.t_start_s)
        last_bin = -1
        tel_step = max(1, int(round(self.sim.telemetry_interval_s / self.sim.dt_slot_s)))

        for slot in range(self.tasks.n_slots):
            t_now_s = self.tasks.t_ep_start_s + slot * self.sim.dt_slot_s
            cur_bin = self._refresh_slow_context(t_now_s)
            if bin_callback is not None and cur_bin != last_bin:
                last_bin = cur_bin
                bin_callback(cur_bin, t_now_s)
                if self.forecast_matrix is not None:
                    self.cur_y_hat = self.forecast_matrix[:, cur_bin]
                if self.surprise_matrix is not None:
                    self.cur_surprise = self.surprise_matrix[:, cur_bin]

            self._deliver_due_events(t_now_s)
            if slot % tel_step == 0:
                self._emit_telemetry_check(t_now_s)

            for task_idx in self._slot_task_indices[slot]:
                node_id = int(self.tasks.node_idx[task_idx])
                obs, mask, gstate = self.build_observation(node_id, task_idx, t_now_s)
                if pass_global_state:
                    action = int(policy_fn(obs, mask, gstate, node_id, t_now_s, task_idx))
                else:
                    action = int(policy_fn(obs, mask, node_id, t_now_s, task_idx))
                assert 0 <= action < len(mask) and bool(mask[action]), f"Invalid masked action {action}"
                self._dispatch_task(task_idx, action, t_now_s)

            self._advance_compute_slot(t_now_s, in_episode=True)

        # Drain remaining queued/in-flight tasks post-horizon for 100% conservation
        extra = 0
        while extra < drain_max_s and (
            any(self.active_tasks) or self._link_rx_heap or self._ack_heap
        ):
            t_now_s = self.tasks.t_ep_start_s + (self.tasks.n_slots + extra) * self.sim.dt_slot_s
            self._refresh_slow_context(t_now_s)
            self._deliver_due_events(t_now_s)
            self._advance_compute_slot(t_now_s, in_episode=False)
            extra += 1

        # Drain any trailing beacons/acks at final timestamp
        final_ts = self.tasks.t_ep_start_s + (self.tasks.n_slots + extra + 10) * self.sim.dt_slot_s
        self._deliver_due_events(final_ts)

        outcomes_by_id: Dict[int, ResolvedOutcome] = {out.task_id: out for out in self.outcomes}
        if self.verify_invariants:
            assert len(self.outcomes) == self.tasks.n_tasks, (
                f"Conservation error: {len(self.outcomes)} != {self.tasks.n_tasks}"
            )
            assert len(outcomes_by_id) == self.tasks.n_tasks, (
                f"Outcome attribution error: {len(outcomes_by_id)} != {self.tasks.n_tasks}"
            )
            assert int(round(self.tasks.t_ep_start_s - t0)) % dt_bin >= 0

        return self.summarize_metrics(), outcomes_by_id

    def run_policy(
        self,
        policy_fn: Callable[[np.ndarray, np.ndarray, int, float, int], int],
        bin_callback: Optional[Callable[[int, float], None]] = None,
        drain_max_s: int = 20_000,
    ) -> SimMetrics:
        """
        Execute a complete episode using `policy_fn(obs, action_mask, node_id, t_now_s, task_idx) -> action`.
        """
        metrics, _ = self._run_episode_loop(
            policy_fn=policy_fn,
            pass_global_state=False,
            bin_callback=bin_callback,
            drain_max_s=drain_max_s,
        )
        return metrics

    def run_marl_episode(
        self,
        policy_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, int, float, int], int],
        bin_callback: Optional[Callable[[int, float], None]] = None,
        drain_max_s: int = 20_000,
    ) -> Tuple[SimMetrics, Dict[int, ResolvedOutcome]]:
        """
        Execute a complete MARL episode using
        `policy_fn(obs, action_mask, global_state, node_id, t_now_s, task_idx) -> action`
        and return `(SimMetrics, outcomes_by_task_id)` for delayed reward attribution.
        """
        return self._run_episode_loop(
            policy_fn=policy_fn,
            pass_global_state=True,
            bin_callback=bin_callback,
            drain_max_s=drain_max_s,
        )

    def summarize_metrics(self) -> SimMetrics:
        total = len(self.outcomes)
        on_time = sum(1 for o in self.outcomes if not o.dropped)
        dropped = total - on_time
        drop_adm = sum(1 for o in self.outcomes if o.drop_reason == "admission_overflow")
        drop_dl = sum(1 for o in self.outcomes if o.drop_reason == "deadline_timeout")
        drop_lnk = sum(1 for o in self.outcomes if o.drop_reason == "link_timeout")

        comp_lats = np.array([o.latency_s for o in self.outcomes if not o.dropped], dtype=np.float64)
        eff_lats = np.array(
            [
                o.latency_s if not o.dropped else o.dl_budget_s * self.sim.drop_penalty_lat_mult
                for o in self.outcomes
            ],
            dtype=np.float64,
        )
        rewards = np.array([o.reward for o in self.outcomes], dtype=np.float64)

        per_node_util = [
            float(self.node_cpu_used[i] / max(1.0, self.node_cpu_avail_acc[i]))
            for i in range(self.n_nodes)
        ]
        u_arr = np.asarray(per_node_util, dtype=np.float64)
        sum_u = float(u_arr.sum())
        sum_u2 = float((u_arr ** 2).sum())
        jain = float((sum_u ** 2) / (self.n_nodes * sum_u2)) if sum_u2 > 1e-12 else 1.0
        util_cv = float(u_arr.std() / max(1e-9, u_arr.mean()))

        per_node_tsr = [
            (float(self.node_ontime_tasks[i]) / float(self.node_total_tasks[i]) * 100.0)
            if self.node_total_tasks[i] > 0
            else 100.0
            for i in range(self.n_nodes)
        ]

        total_model_pkts = self.extra_model_packets + self.dynamic_model_packets
        total_model_bytes = self.extra_model_bytes + self.dynamic_model_bytes
        cb_total = total_model_bytes + self.telemetry_bytes + self.ack_bytes
        return SimMetrics(
            total_tasks=total,
            completed_on_time=on_time,
            dropped_tasks=dropped,
            drop_admission=drop_adm,
            drop_deadline=drop_dl,
            drop_link_timeout=drop_lnk,
            tsr_pct=(on_time / total * 100.0) if total > 0 else 100.0,
            dmr_pct=(dropped / total * 100.0) if total > 0 else 0.0,
            latency_completed_mean_s=float(comp_lats.mean()) if len(comp_lats) else 0.0,
            latency_completed_p95_s=float(np.percentile(comp_lats, 95)) if len(comp_lats) else 0.0,
            latency_completed_p99_s=float(np.percentile(comp_lats, 99)) if len(comp_lats) else 0.0,
            effective_latency_mean_s=float(eff_lats.mean()) if len(eff_lats) else 0.0,
            effective_latency_p95_s=float(np.percentile(eff_lats, 95)) if len(eff_lats) else 0.0,
            offload_count=self.offload_count,
            offload_rate_pct=(self.offload_count / total * 100.0) if total > 0 else 0.0,
            offload_data_bytes=self.offload_data_bytes,
            telemetry_packets=self.telemetry_packets,
            telemetry_bytes=self.telemetry_bytes,
            telemetry_suppressed=self.telemetry_suppressed,
            ack_packets=self.ack_packets,
            ack_bytes=self.ack_bytes,
            model_packets=total_model_pkts,
            model_bytes=total_model_bytes,
            cb_total_bytes=cb_total,
            mean_cpu_util=float(u_arr.mean()),
            jains_fairness=jain,
            util_cv=util_cv,
            mean_reward=float(rewards.mean()) if len(rewards) else 0.0,
            per_node_tsr_pct=per_node_tsr,
            per_node_cpu_util=per_node_util,
        )