import copy
import math
import time
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from phase3.config import Phase3Config
from phase3.data.alibaba_loader import CuratedTraceDataset
from phase3.data.topology import EdgeTopology
from phase3.env.task_simulator import PreparedTaskTable, SimMetrics, TaskOffloadingSim, prepare_task_table


def _ortho_init(layer: nn.Linear, gain: float) -> None:
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0.0)


class RunningMeanStd:
    """Running scalar mean/std normalizer for MAPPO critic value normalization."""

    def __init__(self, eps: float = 1e-4) -> None:
        self.mean: float = 0.0
        self.var: float = 1.0
        self.count: float = eps

    def update(self, arr: np.ndarray) -> None:
        if len(arr) == 0:
            return
        b_mean = float(np.mean(arr))
        b_var = float(np.var(arr))
        b_cnt = float(len(arr))
        delta = b_mean - self.mean
        tot_cnt = self.count + b_cnt
        new_mean = self.mean + delta * (b_cnt / tot_cnt)
        m_a = self.var * self.count
        m_b = b_var * b_cnt
        m2 = m_a + m_b + (delta ** 2) * (self.count * b_cnt / tot_cnt)
        self.mean = new_mean
        self.var = max(1e-4, m2 / tot_cnt)
        self.count = tot_cnt

    @property
    def std(self) -> float:
        return math.sqrt(max(1e-4, self.var))


class MaskedActorCritic(nn.Module):
    """
    Parameter-shared Masked Discrete Actor-Critic (`DR-7`, `DR-8`):
      - `use_central_critic=True`: MAPPO offline centralized critic over `global_state` (91 dims).
      - `use_central_critic=False`: IPPO strictly decentralized critic over local `obs` (75 dims).
      - `actor_uses_global=False`: 100% decentralized P2P runtime actor over `obs` (`E2-E5b`).
      - `actor_uses_global=True`: Oracle `O1` actor consuming `global_state`.
    """

    def __init__(
        self,
        obs_dim: int,
        global_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        use_central_critic: bool = True,
        actor_uses_global: bool = False,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.global_dim = global_dim
        self.action_dim = action_dim
        self.use_central_critic = use_central_critic
        self.actor_uses_global = actor_uses_global

        actor_in = global_dim if actor_uses_global else obs_dim
        critic_in = global_dim if use_central_critic else obs_dim

        self.actor = nn.Sequential(
            nn.Linear(actor_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(critic_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Clean orthogonal initialization (zero hardcoded action preference)
        _ortho_init(self.actor[0], math.sqrt(2.0))
        _ortho_init(self.actor[3], math.sqrt(2.0))
        _ortho_init(self.actor[6], 0.01)
        _ortho_init(self.critic[0], math.sqrt(2.0))
        _ortho_init(self.critic[3], math.sqrt(2.0))
        _ortho_init(self.critic[6], 1.0)

    def actor_logits(
        self, obs: torch.Tensor, mask: torch.Tensor, gstate: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = gstate if self.actor_uses_global else obs
        logits = self.actor(x)
        return logits.masked_fill(~mask, -1e9)

    def critic_value(self, obs: torch.Tensor, gstate: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = gstate if self.use_central_critic else obs
        return self.critic(x).squeeze(-1)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        gstate: torch.Tensor,
        mask: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor_logits(obs, mask, gstate)
        dist = Categorical(logits=logits)
        logp = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.critic_value(obs, gstate)
        return logp, entropy, values


class _NumpyActorRunner:
    """Zero-overhead float32 NumPy mirror of MaskedActorCritic.actor for fast DES step loops."""

    def __init__(self, ac: MaskedActorCritic) -> None:
        sd = ac.actor.state_dict()
        self.w0 = sd["0.weight"].detach().cpu().numpy().astype(np.float32)
        self.b0 = sd["0.bias"].detach().cpu().numpy().astype(np.float32)
        self.ln1_w = sd["1.weight"].detach().cpu().numpy().astype(np.float32)
        self.ln1_b = sd["1.bias"].detach().cpu().numpy().astype(np.float32)
        self.w3 = sd["3.weight"].detach().cpu().numpy().astype(np.float32)
        self.b3 = sd["3.bias"].detach().cpu().numpy().astype(np.float32)
        self.ln4_w = sd["4.weight"].detach().cpu().numpy().astype(np.float32)
        self.ln4_b = sd["4.bias"].detach().cpu().numpy().astype(np.float32)
        self.w6 = sd["6.weight"].detach().cpu().numpy().astype(np.float32)
        self.b6 = sd["6.bias"].detach().cpu().numpy().astype(np.float32)
        self.actor_uses_global = ac.actor_uses_global

    @staticmethod
    def _layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        mean = x.mean()
        var = float(np.mean((x - mean) ** 2))
        return ((x - mean) / math.sqrt(var + eps)) * gamma + beta

    def logits(self, obs: np.ndarray, mask: np.ndarray, gstate: Optional[np.ndarray] = None) -> np.ndarray:
        x = gstate if (self.actor_uses_global and gstate is not None) else obs
        h = np.maximum(0.0, self._layernorm(self.w0 @ x + self.b0, self.ln1_w, self.ln1_b))
        h = np.maximum(0.0, self._layernorm(self.w3 @ h + self.b3, self.ln4_w, self.ln4_b))
        raw = self.w6 @ h + self.b6
        return np.where(mask, raw, -1e9).astype(np.float32)

    def sample(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        gstate: Optional[np.ndarray],
        rng: np.random.Generator,
        deterministic: bool = False,
    ) -> int:
        lg = self.logits(obs, mask, gstate)
        if deterministic:
            return int(np.argmax(lg))
        shifted = lg - np.max(lg)
        probs = np.exp(shifted)
        probs = np.where(mask, probs, 0.0)
        p_sum = float(probs.sum())
        if p_sum <= 0.0 or not math.isfinite(p_sum):
            valid = np.flatnonzero(mask)
            return int(valid[0])
        probs = probs / p_sum
        return int(rng.choice(len(mask), p=probs))


def evaluate_marl_policy(
    ac: MaskedActorCritic,
    sim: TaskOffloadingSim,
    deterministic: bool = True,
    seed: int = 0,
    bin_callback: Optional[Callable[[int, float], None]] = None,
) -> SimMetrics:
    """Evaluate a trained MaskedActorCritic policy deterministically on any TaskOffloadingSim."""
    runner = _NumpyActorRunner(ac)
    rng = np.random.default_rng(seed)

    def _step(obs, mask, gstate, node_id, t_now_s, task_idx) -> int:
        return runner.sample(obs, mask, gstate, rng, deterministic=deterministic)

    metrics, _ = sim.run_marl_episode(_step, bin_callback=bin_callback)
    return metrics


class MAPPOAgent:
    """
    Parameter-Shared Masked Discrete MAPPO Trainer (with `use_central_critic=False` IPPO toggle
    and `actor_uses_global=True` Oracle `O1` toggle).
    """

    def __init__(
        self,
        cfg: Phase3Config,
        use_central_critic: bool = True,
        actor_uses_global: bool = False,
        name: str = "mappo",
    ) -> None:
        self.cfg = cfg
        self.marl = cfg.marl
        self.name = name
        self.device = torch.device(cfg.device)
        self.obs_dim = 11 + 8 * cfg.sim.k_max_neighbors
        self.global_dim = self.obs_dim + 16
        self.action_dim = 1 + cfg.sim.k_max_neighbors

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.ac = MaskedActorCritic(
            obs_dim=self.obs_dim,
            global_dim=self.global_dim,
            action_dim=self.action_dim,
            hidden_dim=self.marl.hidden_dim,
            use_central_critic=use_central_critic,
            actor_uses_global=actor_uses_global,
        ).to(self.device)

        self.actor_opt = torch.optim.Adam(self.ac.actor.parameters(), lr=self.marl.actor_lr, eps=1e-5)
        self.critic_opt = torch.optim.Adam(self.ac.critic.parameters(), lr=self.marl.critic_lr, eps=1e-5)
        self.val_rms = RunningMeanStd()
        self.rng = np.random.default_rng(cfg.seed * 17 + 5)

    def _collect_episode_rollout(
        self,
        sim: TaskOffloadingSim,
        runner: _NumpyActorRunner,
        bin_callback: Optional[Callable[[int, float], None]] = None,
    ) -> Tuple[SimMetrics, Optional[Dict[str, np.ndarray]]]:
        obs_list: List[np.ndarray] = []
        mask_list: List[np.ndarray] = []
        gstate_list: List[np.ndarray] = []
        act_list: List[int] = []
        node_list: List[int] = []
        tid_list: List[int] = []

        def _step(obs, mask, gstate, node_id, t_now_s, task_idx) -> int:
            act = runner.sample(obs, mask, gstate, self.rng, deterministic=False)
            obs_list.append(obs)
            mask_list.append(mask)
            gstate_list.append(gstate)
            act_list.append(act)
            node_list.append(node_id)
            tid_list.append(task_idx)
            return act

        metrics, outcomes = sim.run_marl_episode(_step, bin_callback=bin_callback)
        m = len(tid_list)
        if m == 0:
            return metrics, None

        obs_arr = np.stack(obs_list, axis=0).astype(np.float32)
        mask_arr = np.stack(mask_list, axis=0).astype(bool)
        gstate_arr = np.stack(gstate_list, axis=0).astype(np.float32)
        act_arr = np.array(act_list, dtype=np.int64)
        node_arr = np.array(node_list, dtype=np.int32)
        rew_arr = np.array([outcomes[tid].reward for tid in tid_list], dtype=np.float32)

        obs_t = torch.from_numpy(obs_arr).to(self.device)
        mask_t = torch.from_numpy(mask_arr).to(self.device)
        gstate_t = torch.from_numpy(gstate_arr).to(self.device)
        act_t = torch.from_numpy(act_arr).to(self.device)

        self.ac.eval()
        with torch.no_grad():
            old_logp_t, _, old_vnorm_t = self.ac.evaluate_actions(obs_t, gstate_t, mask_t, act_t)
            old_logp = old_logp_t.cpu().numpy()
            old_vnorm = old_vnorm_t.cpu().numpy()

        old_val_raw = old_vnorm * self.val_rms.std + self.val_rms.mean
        adv_arr = np.zeros(m, dtype=np.float32)
        ret_arr = np.zeros(m, dtype=np.float32)

        gamma = self.marl.gamma
        lam = self.marl.gae_lambda
        for nid in range(sim.n_nodes):
            idxs = np.flatnonzero(node_arr == nid)
            if len(idxs) == 0:
                continue
            last_gae = 0.0
            next_v = float(old_val_raw[idxs[-1]])
            for pos in range(len(idxs) - 1, -1, -1):
                g_i = idxs[pos]
                delta = float(rew_arr[g_i]) + gamma * next_v - float(old_val_raw[g_i])
                last_gae = delta + gamma * lam * last_gae
                adv_arr[g_i] = last_gae
                ret_arr[g_i] = last_gae + float(old_val_raw[g_i])
                next_v = float(old_val_raw[g_i])

        return metrics, {
            "obs": obs_arr,
            "mask": mask_arr,
            "gstate": gstate_arr,
            "act": act_arr,
            "old_logp": old_logp,
            "old_vnorm": old_vnorm,
            "adv": adv_arr,
            "ret": ret_arr,
        }

    def _update_on_rollouts(
        self,
        rollouts: List[Dict[str, np.ndarray]],
        progress_frac: float,
    ) -> Dict[str, float]:
        if not rollouts:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}

        obs_arr = np.concatenate([r["obs"] for r in rollouts], axis=0)
        mask_arr = np.concatenate([r["mask"] for r in rollouts], axis=0)
        gstate_arr = np.concatenate([r["gstate"] for r in rollouts], axis=0)
        act_arr = np.concatenate([r["act"] for r in rollouts], axis=0)
        old_logp_arr = np.concatenate([r["old_logp"] for r in rollouts], axis=0)
        old_vnorm_arr = np.concatenate([r["old_vnorm"] for r in rollouts], axis=0)
        adv_arr = np.concatenate([r["adv"] for r in rollouts], axis=0)
        ret_arr = np.concatenate([r["ret"] for r in rollouts], axis=0)

        m = len(act_arr)
        self.val_rms.update(ret_arr)
        ret_norm_arr = ((ret_arr - self.val_rms.mean) / self.val_rms.std).astype(np.float32)
        adv_std = max(0.25, float(adv_arr.std()))
        adv_norm_arr = ((adv_arr - float(adv_arr.mean())) / adv_std).astype(np.float32)

        obs_t = torch.from_numpy(obs_arr).to(self.device)
        mask_t = torch.from_numpy(mask_arr).to(self.device)
        gstate_t = torch.from_numpy(gstate_arr).to(self.device)
        act_t = torch.from_numpy(act_arr).to(self.device)
        old_logp_t = torch.from_numpy(old_logp_arr).to(self.device)
        adv_t = torch.from_numpy(adv_norm_arr).to(self.device)
        ret_norm_t = torch.from_numpy(ret_norm_arr).to(self.device)
        old_vnorm_t = torch.from_numpy(old_vnorm_arr).to(self.device)

        ent_coef = self.marl.entropy_coef + progress_frac * (self.marl.entropy_min - self.marl.entropy_coef)
        bs = min(max(self.marl.minibatch_size, 256), m)

        self.ac.train()
        p_losses, v_losses, ents, kls = [], [], [], []
        for _ in range(self.marl.ppo_epochs):
            perm = self.rng.permutation(m)
            epoch_kls = []
            for s0 in range(0, m, bs):
                mb = torch.from_numpy(perm[s0 : s0 + bs]).to(self.device)
                new_logp, entropy, new_vnorm = self.ac.evaluate_actions(
                    obs_t[mb], gstate_t[mb], mask_t[mb], act_t[mb]
                )
                log_ratio = new_logp - old_logp_t[mb]
                ratio = torch.exp(log_ratio)
                with torch.no_grad():
                    approx_kl = float(((ratio - 1.0) - log_ratio).mean().item())
                epoch_kls.append(approx_kl)

                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1.0 - self.marl.clip_eps, 1.0 + self.marl.clip_eps) * adv_t[mb]
                p_loss = -torch.min(surr1, surr2).mean() - ent_coef * entropy.mean()

                v_clipped = old_vnorm_t[mb] + torch.clamp(
                    new_vnorm - old_vnorm_t[mb], -self.marl.clip_eps, self.marl.clip_eps
                )
                v_loss_unclipped = (new_vnorm - ret_norm_t[mb]) ** 2
                v_loss_clipped = (v_clipped - ret_norm_t[mb]) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                self.actor_opt.zero_grad()
                p_loss.backward()
                nn.utils.clip_grad_norm_(self.ac.actor.parameters(), self.marl.max_grad_norm)
                self.actor_opt.step()

                self.critic_opt.zero_grad()
                (self.marl.value_coef * v_loss).backward()
                nn.utils.clip_grad_norm_(self.ac.critic.parameters(), self.marl.max_grad_norm)
                self.critic_opt.step()

                p_losses.append(float(p_loss.item()))
                v_losses.append(float(v_loss.item()))
                ents.append(float(entropy.mean().item()))
            mean_kl = float(np.mean(epoch_kls)) if epoch_kls else 0.0
            kls.append(mean_kl)
            if mean_kl > 0.06:
                break

        self.ac.eval()
        return {
            "policy_loss": float(np.mean(p_losses)),
            "value_loss": float(np.mean(v_losses)),
            "entropy": float(np.mean(ents)),
            "approx_kl": float(np.mean(kls)) if kls else 0.0,
        }

    def _collect_and_update(
        self,
        sim: TaskOffloadingSim,
        ep_idx: int,
        bin_callback: Optional[Callable[[int, float], None]] = None,
    ) -> Dict[str, float]:
        runner = _NumpyActorRunner(self.ac)
        metrics, ro = self._collect_episode_rollout(sim, runner, bin_callback=bin_callback)
        if ro is None:
            return {"train_tsr": 100.0, "train_reward": 0.0, "policy_loss": 0.0, "value_loss": 0.0}
        frac = min(1.0, ep_idx / max(1, self.marl.n_train_episodes - 1))
        upd = self._update_on_rollouts([ro], progress_frac=frac)
        return {
            "train_tsr": metrics.tsr_pct,
            "train_eff_lat": metrics.effective_latency_mean_s,
            "train_offload_pct": metrics.offload_rate_pct,
            "train_reward": metrics.mean_reward,
            **upd,
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
        """
        Train cleanly using mixed burst + calm multi-slice PPO batches on `train`,
        select best checkpoint on disjoint `burst_val` + `calm_val`, and evaluate once on `test`.
        """
        t0_wall = time.perf_counter()
        lo_bin = dataset.lookback_L
        hi_bin = dataset.train_end_bin
        bv_s = max(lo_bin + 60, min(self.marl.burst_val_start_bin, hi_bin - 60))
        bv_e = min(hi_bin, max(bv_s + 40, self.marl.burst_val_end_bin))
        win_bins = self.marl.train_window_bins

        # Construct 2-slice paired batches (each iter runs 1 burst + 1 asymmetric/calm slice -> 2x more outer PPO steps!)
        left_ranges: List[Tuple[int, int]] = []
        right_ranges: List[Tuple[int, int]] = []
        if bv_s - lo_bin >= 60:
            w_l = min(win_bins, bv_s - lo_bin)
            left_ranges.append((max(lo_bin, bv_s - w_l), bv_s))        # Peak Wave-1 burst
            mid_lo = max(lo_bin, bv_s - w_l - max(40, w_l // 2))
            left_ranges.append((mid_lo, min(bv_s, mid_lo + w_l)))      # Rising burst
        if hi_bin - bv_e >= 60:
            w_r = min(win_bins, hi_bin - bv_e)
            right_ranges.append((bv_e, min(hi_bin, bv_e + max(160, w_r))))  # Asymmetric spatial load [488..648)
            right_ranges.append((max(bv_e, hi_bin - w_r), hi_bin))          # Afternoon calm
        if not left_ranges and not right_ranges:
            left_ranges.append((lo_bin, hi_bin))
            right_ranges.append((lo_bin, hi_bin))
        elif not left_ranges:
            left_ranges = right_ranges
        elif not right_ranges:
            right_ranges = left_ranges

        burst_tables: List[PreparedTaskTable] = [
            prepare_task_table(dataset, self.cfg.sim, seed=self.cfg.seed, start_bin=s, end_bin=e)
            for (s, e) in left_ranges
        ]
        asym_tables: List[PreparedTaskTable] = [
            prepare_task_table(dataset, self.cfg.sim, seed=self.cfg.seed, start_bin=s, end_bin=e)
            for (s, e) in right_ranges
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

        batch_k = 2
        n_iters = max(1, (self.marl.n_train_episodes + batch_k - 1) // batch_k)
        eval_cadence = max(1, self.marl.eval_every // batch_k)
        best_val_score = -1e9
        best_state = copy.deepcopy(self.ac.state_dict())
        best_ep = 0
        history: List[Dict[str, float]] = []

        for it in range(n_iters):
            runner = _NumpyActorRunner(self.ac)
            rollouts: List[Dict[str, np.ndarray]] = []
            tsrs, rews, offs = [], [], []
            pair = [burst_tables[it % len(burst_tables)], asym_tables[it % len(asym_tables)]]
            for tbl in pair:
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
                m_ep, ro = self._collect_episode_rollout(tr_sim, runner)
                tsrs.append(m_ep.tsr_pct)
                rews.append(m_ep.mean_reward)
                offs.append(m_ep.offload_rate_pct)
                if ro is not None:
                    rollouts.append(ro)

            frac = min(1.0, it / max(1, n_iters - 1))
            upd = self._update_on_rollouts(rollouts, progress_frac=frac)
            ep_equiv = (it + 1) * batch_k
            stats: Dict[str, float] = {
                "episode": float(ep_equiv),
                "iter": float(it + 1),
                "train_tsr": float(np.mean(tsrs)),
                "train_offload_pct": float(np.mean(offs)),
                "train_reward": float(np.mean(rews)),
                **upd,
            }

            if ((it + 1) % eval_cadence == 0) or (it + 1 == n_iters):
                vm_b = evaluate_marl_policy(self.ac, burst_val_sim, deterministic=True, seed=self.cfg.seed)
                vm_c = evaluate_marl_policy(self.ac, val_sim, deterministic=True, seed=self.cfg.seed)
                val_score = 0.75 * vm_b.mean_reward + 0.25 * vm_c.mean_reward
                stats["val_burst_tsr"] = vm_b.tsr_pct
                stats["val_tsr"] = vm_c.tsr_pct
                stats["val_eff_lat"] = vm_b.effective_latency_mean_s
                stats["val_reward"] = val_score
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_state = copy.deepcopy(self.ac.state_dict())
                    best_ep = ep_equiv
            history.append(stats)

        self.ac.load_state_dict(best_state)
        self.ac.eval()
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
        test_metrics = evaluate_marl_policy(self.ac, test_sim, deterministic=True, seed=self.cfg.seed)
        wall_s = time.perf_counter() - t0_wall
        return {
            "agent": self.name,
            "seed": self.cfg.seed,
            "use_central_critic": self.ac.use_central_critic,
            "actor_uses_global": self.ac.actor_uses_global,
            "best_val_ep": best_ep,
            "best_val_reward": float(best_val_score),
            "wall_time_s": float(wall_s),
            "history": history,
            "test_metrics": asdict(test_metrics),
        }

