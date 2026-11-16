import numpy as np

from phase3.config import SimConfig


class LocalOnlyPolicy:
    """Baseline B0: strictly local execution (zero offloading)."""

    def __call__(self, obs: np.ndarray, mask: np.ndarray, node_id: int, t_now_s: float, task_idx: int) -> int:
        return 0


class ReactiveGreedyPolicy:
    """
    Baseline E1: Reactive 1-hop greedy estimator using ONLY cached causal neighbor state.
    Protects uncongested local nodes from needless offloading churn, discounts peer headroom
    by self-dispatched inflight tasks, and offloads when a neighbor beats local cost by margin.
    """

    def __init__(
        self,
        sim_cfg: SimConfig,
        offload_hysteresis: float = 0.72,
        idle_q_guard: float = 0.15,
        idle_avail_guard: float = 0.25,
    ) -> None:
        self.sim = sim_cfg
        self.hysteresis = float(offload_hysteresis)
        self.idle_q_guard = float(idle_q_guard)
        self.idle_avail_guard = float(idle_avail_guard)

    def __call__(self, obs: np.ndarray, mask: np.ndarray, node_id: int, t_now_s: float, task_idx: int) -> int:
        q_local = float(obs[0])
        avail_local = max(0.01, float(obs[1]))
        avail_mem_local = float(obs[2])
        r_m = float(obs[8]) * 0.25
        payload_bytes = float(obs[9]) * self.sim.obs_bytes_scale

        # If local node has clear capacity/memory headroom, keep task locally to avoid herd churn
        if (q_local < self.idle_q_guard) and (avail_local > self.idle_avail_guard) and (avail_mem_local > r_m * 1.5):
            return 0

        local_cost = (1.0 + 3.0 * q_local) / avail_local
        best_action = 0
        best_cost = local_cost * self.hysteresis

        for k_idx in range(self.sim.k_max_neighbors):
            a = 1 + k_idx
            if not bool(mask[a]):
                continue
            base = 11 + 8 * k_idx
            rtt_s = float(obs[base + 1]) * 0.05
            bw_mbps = max(10.0, float(obs[base + 2]) * 200.0)
            avail_peer_raw = max(0.01, float(obs[base + 3]))
            q_peer = float(obs[base + 4])
            inflight_peer = float(obs[base + 6])
            info_age_norm = float(obs[base + 7])

            eff_q_peer = q_peer + inflight_peer
            if eff_q_peer >= 0.45:
                continue

            avail_peer_adj = max(0.01, avail_peer_raw * max(0.15, 1.0 - 3.0 * inflight_peer))
            tx_s = rtt_s * 0.5 + (8.0 * payload_bytes) / (bw_mbps * 1e6)
            tx_mult = 1.0 + tx_s / max(1.0, self.sim.obs_dl_scale * 0.25)
            stale_mult = 1.0 + 0.10 * info_age_norm
            peer_cost = tx_mult * stale_mult * (1.0 + 3.0 * eff_q_peer) / avail_peer_adj

            if peer_cost < best_cost:
                best_cost = peer_cost
                best_action = a

        return best_action

