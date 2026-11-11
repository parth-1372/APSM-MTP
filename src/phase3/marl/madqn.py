import copy
import math
import time
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from phase3.config import Phase3Config
from phase3.data.alibaba_loader import CuratedTraceDataset
from phase3.data.topology import EdgeTopology
from phase3.env.task_simulator import PreparedTaskTable, SimMetrics, TaskOffloadingSim, prepare_task_table


class MaskedDuelingQNet(nn.Module):
    """
    Parameter-Shared Masked Double Dueling Q-Network (`DECOFFEE` / `Adu` comparator family).
    Computes valid-action-only advantage mean so unconnected neighbor padding slots never bias Q-values.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        half = max(32, hidden_dim // 2)

        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.val_stream = nn.Sequential(
            nn.Linear(hidden_dim, half),
            nn.ReLU(),
            nn.Linear(half, 1),
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(hidden_dim, half),
            nn.ReLU(),
            nn.Linear(half, action_dim),
        )

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.feature(obs)
        val = self.val_stream(h)
        adv = self.adv_stream(h)
        mask_f = mask.to(dtype=adv.dtype)
        valid_cnt = mask_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
        adv_mean = (adv * mask_f).sum(dim=-1, keepdim=True) / valid_cnt
        q = val + (adv - adv_mean)
        return q.masked_fill(~mask, -1e9)


class _NumpyDuelingQRunner:
    """Exact float32 NumPy mirror of MaskedDuelingQNet for low-overhead DES evaluation & rollouts."""

    def __init__(self, qnet: MaskedDuelingQNet) -> None:
        sd = qnet.state_dict()
        self.fw0 = sd["feature.0.weight"].detach().cpu().numpy().astype(np.float32)
        self.fb0 = sd["feature.0.bias"].detach().cpu().numpy().astype(np.float32)
        self.fl1_w = sd["feature.1.weight"].detach().cpu().numpy().astype(np.float32)
        self.fl1_b = sd["feature.1.bias"].detach().cpu().numpy().astype(np.float32)
        self.fw3 = sd["feature.3.weight"].detach().cpu().numpy().astype(np.float32)
        self.fb3 = sd["feature.3.bias"].detach().cpu().numpy().astype(np.float32)
        self.fl4_w = sd["feature.4.weight"].detach().cpu().numpy().astype(np.float32)
        self.fl4_b = sd["feature.4.bias"].detach().cpu().numpy().astype(np.float32)

        self.vw0 = sd["val_stream.0.weight"].detach().cpu().numpy().astype(np.float32)
        self.vb0 = sd["val_stream.0.bias"].detach().cpu().numpy().astype(np.float32)
        self.vw2 = sd["val_stream.2.weight"].detach().cpu().numpy().astype(np.float32)
        self.vb2 = sd["val_stream.2.bias"].detach().cpu().numpy().astype(np.float32)

        self.aw0 = sd["adv_stream.0.weight"].detach().cpu().numpy().astype(np.float32)
        self.ab0 = sd["adv_stream.0.bias"].detach().cpu().numpy().astype(np.float32)
        self.aw2 = sd["adv_stream.2.weight"].detach().cpu().numpy().astype(np.float32)
        self.ab2 = sd["adv_stream.2.bias"].detach().cpu().numpy().astype(np.float32)

    @staticmethod
    def _layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        mean = x.mean()
        var = float(np.mean((x - mean) ** 2))
        return ((x - mean) / math.sqrt(var + eps)) * gamma + beta

    def q_values(self, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        h = np.maximum(0.0, self._layernorm(self.fw0 @ obs + self.fb0, self.fl1_w, self.fl1_b))
        h = np.maximum(0.0, self._layernorm(self.fw3 @ h + self.fb3, self.fl4_w, self.fl4_b))

        v = float((self.vw2 @ np.maximum(0.0, self.vw0 @ h + self.vb0) + self.vb2)[0])
        a = self.aw2 @ np.maximum(0.0, self.aw0 @ h + self.ab0) + self.ab2
        valid_adv = a[mask]
        a_mean = float(valid_adv.mean()) if len(valid_adv) > 0 else 0.0
        q = v + (a - a_mean)
        return np.where(mask, q, -1e9).astype(np.float32)

    def act(
        self, obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator, epsilon: float = 0.0
    ) -> int:
        if epsilon > 0.0 and rng.random() < epsilon:
            valid = np.flatnonzero(mask)
            return int(valid[int(rng.integers(0, len(valid)))])
        return int(np.argmax(self.q_values(obs, mask)))


class _ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int) -> None:
        self.cap = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.mask = np.zeros((capacity, action_dim), dtype=bool)
        self.act = np.zeros(capacity, dtype=np.int64)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_mask = np.zeros((capacity, action_dim), dtype=bool)
        self.ptr = 0
        self.size = 0

    def add_batch(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        act: np.ndarray,
        rew: np.ndarray,
        next_obs: np.ndarray,
        next_mask: np.ndarray,
    ) -> None:
        n = len(act)
        for k in range(n):
            idx = self.ptr
            self.obs[idx] = obs[k]
            self.mask[idx] = mask[k]
            self.act[idx] = act[k]
            self.rew[idx] = rew[k]
            self.next_obs[idx] = next_obs[k]
            self.next_mask[idx] = next_mask[k]
            self.ptr = (self.ptr + 1) % self.cap
            self.size = min(self.cap, self.size + 1)

    def sample(self, batch_size: int, rng: np.random.Generator) -> Tuple[np.ndarray, ...]:
        idx = rng.integers(0, self.size, size=batch_size)
        return (
            self.obs[idx],
            self.mask[idx],
            self.act[idx],
            self.rew[idx],
            self.next_obs[idx],
            self.next_mask[idx],
        )


def evaluate_dqn_policy(
    qnet: MaskedDuelingQNet,
    sim: TaskOffloadingSim,
    seed: int = 0,
    bin_callback: Optional[Callable[[int, float], None]] = None,
) -> SimMetrics:
    runner = _NumpyDuelingQRunner(qnet)
    rng = np.random.default_rng(seed)

    def _step(obs, mask, gstate, node_id, t_now_s, task_idx) -> int:
        return runner.act(obs, mask, rng, epsilon=0.0)

    metrics, _ = sim.run_marl_episode(_step, bin_callback=bin_callback)
    return metrics


class MADQNAgent:
    """
    Parameter-Shared Masked Double Dueling MADQN Trainer (`DECOFFEE` / `Adu` family comparator).
    Strictly independent/decentralized (`zero central critic`, `obs`-only during both train & eval).
    """

    def __init__(self, cfg: Phase3Config, name: str = "madqn") -> None:
        self.cfg = cfg
        self.marl = cfg.marl
        self.name = name
        self.device = torch.device(cfg.device)
        self.obs_dim = 11 + 8 * cfg.sim.k_max_neighbors
        self.action_dim = 1 + cfg.sim.k_max_neighbors

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.q_online = MaskedDuelingQNet(self.obs_dim, self.action_dim, self.marl.hidden_dim).to(self.device)
        self.q_target = copy.deepcopy(self.q_online).to(self.device)
        self.q_target.eval()
        self.opt = torch.optim.Adam(self.q_online.parameters(), lr=self.marl.dqn_lr)
        self.buf = _ReplayBuffer(self.marl.dqn_buffer_size, self.obs_dim, self.action_dim)
        self.rng = np.random.default_rng(cfg.seed * 23 + 7)

    def _collect_and_update(
        self,
        sim: TaskOffloadingSim,
        ep_idx: int,
        bin_callback: Optional[Callable[[int, float], None]] = None,
    ) -> Dict[str, float]:
        decay_frac = min(1.0, ep_idx / max(1, self.marl.dqn_eps_decay_episodes))
        eps = self.marl.dqn_eps_start + decay_frac * (self.marl.dqn_eps_end - self.marl.dqn_eps_start)

        runner = _NumpyDuelingQRunner(self.q_online)
        obs_list: List[np.ndarray] = []
        mask_list: List[np.ndarray] = []
        act_list: List[int] = []
        node_list: List[int] = []
        tid_list: List[int] = []

        def _step(obs, mask, gstate, node_id, t_now_s, task_idx) -> int:
            act = runner.act(obs, mask, self.rng, epsilon=eps)
            obs_list.append(obs)
            mask_list.append(mask)
            act_list.append(act)
            node_list.append(node_id)
            tid_list.append(task_idx)
            return act

        metrics, outcomes = sim.run_marl_episode(_step, bin_callback=bin_callback)
        m = len(tid_list)
        if m == 0:
            return {"train_tsr": 100.0, "train_reward": 0.0, "td_loss": 0.0, "eps": eps}

        obs_arr = np.stack(obs_list, axis=0).astype(np.float32)
        mask_arr = np.stack(mask_list, axis=0).astype(bool)
        act_arr = np.array(act_list, dtype=np.int64)
        node_arr = np.array(node_list, dtype=np.int32)
        rew_arr = np.array([outcomes[tid].reward for tid in tid_list], dtype=np.float32)

        next_obs_arr = obs_arr.copy()
        next_mask_arr = mask_arr.copy()
        for nid in range(sim.n_nodes):
            idxs = np.flatnonzero(node_arr == nid)
            if len(idxs) > 1:
                next_obs_arr[idxs[:-1]] = obs_arr[idxs[1:]]
                next_mask_arr[idxs[:-1]] = mask_arr[idxs[1:]]

        self.buf.add_batch(obs_arr, mask_arr, act_arr, rew_arr, next_obs_arr, next_mask_arr)

        losses = []
        if self.buf.size >= self.marl.dqn_batch_size:
            self.q_online.train()
            gamma = self.marl.gamma
            tau = self.marl.dqn_tau
            for _ in range(self.marl.dqn_updates_per_episode):
                bo, bm, ba, br, bno, bnm = self.buf.sample(self.marl.dqn_batch_size, self.rng)
                t_o = torch.from_numpy(bo).to(self.device)
                t_m = torch.from_numpy(bm).to(self.device)
                t_a = torch.from_numpy(ba).to(self.device)
                t_r = torch.from_numpy(br).to(self.device)
                t_no = torch.from_numpy(bno).to(self.device)
                t_nm = torch.from_numpy(bnm).to(self.device)

                with torch.no_grad():
                    next_q_on = self.q_online(t_no, t_nm)
                    best_a = next_q_on.argmax(dim=-1, keepdim=True)
                    next_q_tgt = self.q_target(t_no, t_nm).gather(1, best_a).squeeze(-1)
                    target = t_r + gamma * next_q_tgt

                q_taken = self.q_online(t_o, t_m).gather(1, t_a.unsqueeze(-1)).squeeze(-1)
                loss = F.smooth_l1_loss(q_taken, target, beta=1.0)
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.q_online.parameters(), 2.0)
                self.opt.step()

                with torch.no_grad():
                    for p_on, p_tg in zip(self.q_online.parameters(), self.q_target.parameters()):
                        p_tg.data.mul_(1.0 - tau).add_(p_on.data, alpha=tau)

                losses.append(float(loss.item()))
            self.q_online.eval()

        return {
            "train_tsr": metrics.tsr_pct,
            "train_eff_lat": metrics.effective_latency_mean_s,
            "train_offload_pct": metrics.offload_rate_pct,
            "train_reward": metrics.mean_reward,
            "td_loss": float(np.mean(losses)) if losses else 0.0,
            "eps": float(eps),
        }

    def fit_and_evaluate(
        self,
        dataset: CuratedTraceDataset,
        topo: EdgeTopology,
        telemetry_mode: str = "periodic",
        forecast_matrix: Optional[np.ndarray] = None,
        surprise_matrix: Optional[np.ndarray] = None,
        extra_test_model_packets: int = 0,
        extra_test_model_bytes: int = 0,
        apsm_t_threshold: float = 0.12,
        apsm_t_heartbeat: int = 4,
    ) -> Dict[str, Any]:
        t0_wall = time.perf_counter()
        lo_bin = dataset.lookback_L
        hi_bin = dataset.train_end_bin
        bv_s = max(lo_bin + 60, min(self.marl.burst_val_start_bin, hi_bin - 60))
        bv_e = min(hi_bin, max(bv_s + 40, self.marl.burst_val_end_bin))
        win_bins = self.marl.train_window_bins

        candidate_ranges: List[Tuple[int, int]] = []
        if bv_s - lo_bin >= 60:
            w_l = min(win_bins, bv_s - lo_bin)
            candidate_ranges.append((lo_bin, lo_bin + w_l))
            burst_lo = max(lo_bin, bv_s - w_l)
            candidate_ranges.append((burst_lo, bv_s))
            candidate_ranges.append((burst_lo, bv_s))
            mid_s = max(lo_bin, burst_lo - max(40, w_l // 3))
            candidate_ranges.append((mid_s, min(bv_s, mid_s + w_l)))
        if hi_bin - bv_e >= 60:
            w_r = min(win_bins, hi_bin - bv_e)
            candidate_ranges.append((bv_e, bv_e + w_r))
            candidate_ranges.append((max(bv_e, hi_bin - w_r), hi_bin))
        if not candidate_ranges:
            candidate_ranges.append((lo_bin, hi_bin))

        train_tables: List[PreparedTaskTable] = [
            prepare_task_table(dataset, self.cfg.sim, seed=self.cfg.seed, start_bin=s, end_bin=e)
            for (s, e) in candidate_ranges
        ]
        burst_val_table = prepare_task_table(
            dataset, self.cfg.sim, seed=self.cfg.seed, start_bin=bv_s, end_bin=bv_e
        )
        val_table = prepare_task_table(dataset, self.cfg.sim, split="val", seed=self.cfg.seed)
        test_table = prepare_task_table(dataset, self.cfg.sim, split="test", seed=self.cfg.seed)

        burst_val_sim = TaskOffloadingSim(
            dataset,
            topo,
            burst_val_table,
            self.cfg,
            telemetry_mode=telemetry_mode,
            forecast_matrix=forecast_matrix,
            surprise_matrix=surprise_matrix,
            apsm_t_threshold=apsm_t_threshold,
            apsm_t_heartbeat=apsm_t_heartbeat,
        )
        val_sim = TaskOffloadingSim(
            dataset,
            topo,
            val_table,
            self.cfg,
            telemetry_mode=telemetry_mode,
            forecast_matrix=forecast_matrix,
            surprise_matrix=surprise_matrix,
            apsm_t_threshold=apsm_t_threshold,
            apsm_t_heartbeat=apsm_t_heartbeat,
        )

        best_val_score = -1e9
        best_state = copy.deepcopy(self.q_online.state_dict())
        best_ep = 0
        history: List[Dict[str, float]] = []

        for ep in range(self.marl.n_train_episodes):
            if ep % len(train_tables) == 0:
                order = self.rng.permutation(len(train_tables)).tolist()
            tbl = train_tables[order[ep % len(train_tables)]]
            tr_sim = TaskOffloadingSim(
                dataset,
                topo,
                tbl,
                self.cfg,
                telemetry_mode=telemetry_mode,
                forecast_matrix=forecast_matrix,
                surprise_matrix=surprise_matrix,
                apsm_t_threshold=apsm_t_threshold,
                apsm_t_heartbeat=apsm_t_heartbeat,
            )
            stats = self._collect_and_update(tr_sim, ep)
            stats["episode"] = ep + 1

            if ((ep + 1) % self.marl.eval_every == 0) or (ep + 1 == self.marl.n_train_episodes):
                vm_b = evaluate_dqn_policy(self.q_online, burst_val_sim, seed=self.cfg.seed)
                vm_c = evaluate_dqn_policy(self.q_online, val_sim, seed=self.cfg.seed)
                val_score = 0.75 * vm_b.mean_reward + 0.25 * vm_c.mean_reward
                stats["val_burst_tsr"] = vm_b.tsr_pct
                stats["val_tsr"] = vm_c.tsr_pct
                stats["val_eff_lat"] = vm_b.effective_latency_mean_s
                stats["val_reward"] = val_score
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_state = copy.deepcopy(self.q_online.state_dict())
                    best_ep = ep + 1
            history.append(stats)

        self.q_online.load_state_dict(best_state)
        self.q_online.eval()
        test_sim = TaskOffloadingSim(
            dataset,
            topo,
            test_table,
            self.cfg,
            telemetry_mode=telemetry_mode,
            forecast_matrix=forecast_matrix,
            surprise_matrix=surprise_matrix,
            extra_model_packets=extra_test_model_packets,
            extra_model_bytes=extra_test_model_bytes,
            apsm_t_threshold=apsm_t_threshold,
            apsm_t_heartbeat=apsm_t_heartbeat,
            verify_invariants=True,
        )
        test_metrics = evaluate_dqn_policy(self.q_online, test_sim, seed=self.cfg.seed)
        wall_s = time.perf_counter() - t0_wall
        return {
            "agent": self.name,
            "seed": self.cfg.seed,
            "best_val_ep": best_ep,
            "best_val_reward": float(best_val_score),
            "wall_time_s": float(wall_s),
            "history": history,
            "test_metrics": asdict(test_metrics),
        }

