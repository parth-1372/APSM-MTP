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
        bsz, k_max, _ = e_peers.shape
        z_ctx = self.fc_ctx(h_ctx).unsqueeze(1)
        z_peer = self.fc_peer(e_peers)
        h = self.relu(self.ln(z_ctx + z_peer))
        return self.out(h).squeeze(-1)


class PeerProactive2Branch(nn.Module):
    """Peer proactive congestion residual using verified causal signals."""

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


class MaskedDuelingQNet(nn.Module):
    """
    Parameter-Shared Masked Double Dueling Q-Network (`DECOFFEE` / `Adu` comparator family)
    enhanced with permutation-invariant edge encoder and proactive peer branch.
    """

    def __init__(
        self,
        obs_dim: int = 75,
        action_dim: int = 9,
        hidden_dim: int = 96,
        fc_scale: float = 0.008,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.k_max = action_dim - 1
        self.hidden_dim = hidden_dim
        p_dim = hidden_dim // 2

        self.self_enc = nn.Sequential(nn.Linear(8, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())
        self.peer_enc = nn.Sequential(nn.Linear(7, p_dim), nn.LayerNorm(p_dim), nn.ReLU())
        base_ctx = hidden_dim + 2 * p_dim

        self.val_head = nn.Sequential(
            nn.Linear(base_ctx, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.adv_loc = nn.Sequential(
            nn.Linear(base_ctx, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.adv_peer = KaimingFactoredPeerHead(base_ctx, p_dim, hidden_dim)
        self.fc_res = PeerProactive2Branch(k_max=self.k_max, fc_scale=fc_scale)

        # Backwards compatible aliases
        self.feature = self.self_enc
        self.val_stream = self.val_head
        self.adv_stream = self.adv_loc

    def base_params(self):
        return (
            list(self.self_enc.parameters())
            + list(self.peer_enc.parameters())
            + list(self.val_head.parameters())
            + list(self.adv_loc.parameters())
            + list(self.adv_peer.parameters())
        )

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
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

    def q_values(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.forward(obs, mask)


class _NumpyDuelingQRunner:
    """Exact float32 NumPy mirror of MaskedDuelingQNet for low-overhead DES evaluation & rollouts."""

    def __init__(self, qnet: MaskedDuelingQNet) -> None:
        self.k_max = qnet.k_max
        self.obs_dim = qnet.obs_dim

        def ext(m):
            sd = m.state_dict()
            return (
                sd["0.weight"].detach().cpu().numpy().T.astype(np.float32),
                sd["0.bias"].detach().cpu().numpy().astype(np.float32),
                sd["1.weight"].detach().cpu().numpy().astype(np.float32),
                sd["1.bias"].detach().cpu().numpy().astype(np.float32),
            )

        self.se = ext(qnet.self_enc)
        self.pe = ext(qnet.peer_enc)
        self.p_dim = self.pe[1].shape[0]

        vh = qnet.val_head
        self.vh0 = ext(vh)
        self.vh_w = vh[3].weight.detach().cpu().numpy().astype(np.float32).ravel()
        self.vh_b = float(vh[3].bias.detach().cpu().numpy()[0])

        al = qnet.adv_loc
        self.al0 = ext(al)
        self.al_w = al[3].weight.detach().cpu().numpy().astype(np.float32).ravel()
        self.al_b = float(al[3].bias.detach().cpu().numpy()[0])

        ap = qnet.adv_peer
        self.ap_wc = np.ascontiguousarray(ap.fc_ctx.weight.detach().cpu().numpy().T.astype(np.float32))
        self.ap_wp = np.ascontiguousarray(ap.fc_peer.weight.detach().cpu().numpy().T.astype(np.float32))
        self.ap_b0 = ap.fc_ctx.bias.detach().cpu().numpy().astype(np.float32)
        self.ap_lw = ap.ln.weight.detach().cpu().numpy().astype(np.float32)
        self.ap_lb = ap.ln.bias.detach().cpu().numpy().astype(np.float32)
        self.ap_w3 = ap.out.weight.detach().cpu().numpy().astype(np.float32).ravel()
        self.ap_b3 = float(ap.out.bias.detach().cpu().numpy()[0])

        self.fc_active = qnet.fc_res.fc_active
        self.fc_scale = qnet.fc_res.fc_scale
        self.fp = qnet.fc_res.peer_lin.weight.detach().cpu().numpy().astype(np.float32).ravel()

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

    def q_values(self, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
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
            h_base = np.concatenate([h_self, ep_mean, ep_max])
        else:
            p_mat = None
            ep_v = None
            h_base = np.concatenate([h_self, np.zeros(2 * self.p_dim, dtype=np.float32)])

        h_val = self._ln_relu_1d(h_base @ self.vh0[0] + self.vh0[1], self.vh0[2], self.vh0[3])
        v = float(np.dot(h_val, self.vh_w)) + self.vh_b

        adv = np.zeros(1 + self.k_max, dtype=np.float32)
        h_loc = self._ln_relu_1d(h_base @ self.al0[0] + self.al0[1], self.al0[2], self.al0[3])
        adv[0] = float(np.dot(h_loc, self.al_w)) + self.al_b
        if nv > 0:
            z_all = ep_v @ self.ap_wp + (h_base @ self.ap_wc + self.ap_b0)
            h_p = self._ln_relu_2d(z_all, self.ap_lw, self.ap_lb)
            ap = (h_p @ self.ap_w3) + self.ap_b3
            if self.fc_active:
                yp = p_mat[:, 5]
                avp = p_mat[:, 3]
                agp = p_mat[:, 7]
                pf = np.stack([np.maximum(0.0, yp - avp), yp * agp], axis=1)
                ap = ap + self.fc_scale * np.tanh(pf @ self.fp)
            adv[1 + v_idx] = ap

        valid_cnt = int(mask.sum())
        adv_mean = float(adv[mask].mean()) if valid_cnt > 0 else 0.0
        q = np.where(mask, v + (adv - adv_mean), -1e9).astype(np.float32)
        return q

    def act(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        rng: Optional[np.random.Generator] = None,
        temp: float = 0.0,
        epsilon: float = 0.0,
    ) -> int:
        if epsilon > 0.0 and rng is not None and rng.random() < epsilon:
            valid = np.flatnonzero(mask)
            return int(valid[int(rng.integers(0, len(valid)))])
        raw = self.q_values(obs, mask)
        if temp <= 1e-6 or rng is None:
            return int(np.argmax(raw))
        sh = (raw - float(raw[mask].max())) / temp
        pr = np.where(mask, np.exp(np.clip(sh, -40.0, 0.0)), 0.0)
        ps = float(pr.sum())
        return (
            int(np.argmax(raw))
            if (ps <= 0.0 or not math.isfinite(ps))
            else int(rng.choice(len(mask), p=pr * (1.0 / ps)))
        )


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
    Parameter-Shared Masked Double Dueling MADQN Trainer (`DECOFFEE` / `Adu` family comparator)
    with proactive peer branch and fast NumPy DES execution.
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

        self.q_online = MaskedDuelingQNet(
            self.obs_dim, self.action_dim, self.marl.hidden_dim, fc_scale=0.008
        ).to(self.device)
        self.q_target = copy.deepcopy(self.q_online).to(self.device)
        self.q_target.eval()

        pg = [
            {"params": self.q_online.base_params(), "weight_decay": 2e-4, "lr": 3e-4},
            {"params": self.q_online.fc_res.parameters(), "weight_decay": 2e-2, "lr": 1.5e-4},
        ]
        self.opt = torch.optim.AdamW(pg, eps=1e-5)
        self.buf = _ReplayBuffer(60000, self.obs_dim, self.action_dim)
        self.rng = np.random.default_rng(cfg.seed * 23 + 7)

    def _collect_and_update(
        self,
        sim: TaskOffloadingSim,
        ep_idx: int,
        bin_callback: Optional[Callable[[int, float], None]] = None,
        forecast_active: bool = False,
    ) -> Dict[str, float]:
        warmup = 36
        self.opt.param_groups[0]["lr"] = 3e-4 if ep_idx < warmup else 8e-5

        fc_on = forecast_active and (ep_idx >= warmup)
        self.q_online.fc_res.fc_active = fc_on
        self.q_target.fc_res.fc_active = fc_on

        frac = min(1.0, max(0, ep_idx - 2) / 30.0)
        t_roll = 0.002 + 0.5 * (0.012 - 0.002) * (1.0 + math.cos(math.pi * frac))
        eps = 0.30 + min(1.0, ep_idx / 28.0) * (0.01 - 0.30)

        runner = _NumpyDuelingQRunner(self.q_online)
        obs_list: List[np.ndarray] = []
        mask_list: List[np.ndarray] = []
        act_list: List[int] = []
        node_list: List[int] = []
        tid_list: List[int] = []

        def _step(obs, mask, gstate, node_id, t_now_s, task_idx) -> int:
            act = runner.act(obs, mask, self.rng, temp=t_roll, epsilon=eps)
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
        min_buf = min(256, self.marl.dqn_batch_size)
        if self.buf.size >= min_buf:
            self.q_online.train()
            gamma = 0.40
            n_updates = min(48, getattr(self.marl, "dqn_updates_per_episode", 48))
            bs = min(self.buf.size, getattr(self.marl, "dqn_batch_size", 256))
            for _ in range(n_updates):
                bo, bm, ba, br, bno, bnm = self.buf.sample(bs, self.rng)
                t_o = torch.from_numpy(bo).to(self.device)
                t_m = torch.from_numpy(bm).to(self.device)
                t_a = torch.from_numpy(ba).to(self.device)
                t_r = torch.from_numpy(br).to(self.device)
                t_no = torch.from_numpy(bno).to(self.device)
                t_nm = torch.from_numpy(bnm).to(self.device)

                with torch.no_grad():
                    best_a = self.q_online(t_no, t_nm).argmax(dim=-1, keepdim=True)
                    tgt = t_r + gamma * self.q_target(t_no, t_nm).gather(1, best_a).squeeze(-1)

                q_taken = self.q_online(t_o, t_m).gather(1, t_a.unsqueeze(-1)).squeeze(-1)
                loss = F.smooth_l1_loss(q_taken, tgt, beta=1.0)
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.q_online.parameters(), 2.0)
                self.opt.step()

                with torch.no_grad():
                    for p1, p2 in zip(self.q_online.parameters(), self.q_target.parameters()):
                        p2.data.mul_(0.995).add_(p1.data, alpha=0.005)

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

        grps = [(lo_bin, 100), (210, 330), (260, 380), (260, 380), (488, 608), (688, 808)]
        train_tables = [
            prepare_task_table(dataset, self.cfg.sim, seed=self.cfg.seed, start_bin=s, end_bin=e)
            for s, e in grps
        ]
        burst_val_table = prepare_task_table(
            dataset, self.cfg.sim, seed=self.cfg.seed, start_bin=380, end_bin=488
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

        n_eps = self.marl.n_train_episodes
        eval_eps = {40, 44, 48} if n_eps >= 40 else set(range(1, n_eps + 1))
        order = []

        for ep in range(n_eps):
            if ep % len(train_tables) == 0:
                order = self.rng.permutation(len(train_tables)).tolist()
            tbl = train_tables[order[ep % len(train_tables)]]
            tr_sim = TaskOffloadingSim(
                dataset,
                topo,
                tbl,
                self.cfg,
                telemetry_mode=telemetry_mode,
                forecast_matrix=forecast_matrix if (forecast_matrix is not None and ep >= 36) else None,
                surprise_matrix=surprise_matrix if (surprise_matrix is not None and ep >= 36) else None,
                apsm_t_threshold=apsm_t_threshold,
                apsm_t_heartbeat=apsm_t_heartbeat,
            )
            stats = self._collect_and_update(
                tr_sim, ep, forecast_active=(forecast_matrix is not None)
            )
            stats["episode"] = ep + 1

            if ((ep + 1) in eval_eps) or (ep + 1 == n_eps):
                vm_b = evaluate_dqn_policy(self.q_target, burst_val_sim, seed=self.cfg.seed)
                vm_c = evaluate_dqn_policy(self.q_target, val_sim, seed=self.cfg.seed)
                val_score = 0.75 * vm_b.mean_reward + 0.25 * vm_c.mean_reward
                stats["val_burst_tsr"] = vm_b.tsr_pct
                stats["val_tsr"] = vm_c.tsr_pct
                stats["val_reward"] = val_score
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_state = copy.deepcopy(self.q_target.state_dict())
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
