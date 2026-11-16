import copy
import math
import time
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from phase3.config import Phase3Config
from phase3.data.alibaba_loader import CuratedTraceDataset
from phase3.data.topology import EdgeTopology
from phase3.env.task_simulator import PreparedTaskTable, SimMetrics, TaskOffloadingSim, prepare_task_table


def _ortho(layer: nn.Linear, gain: float = 1.0) -> None:
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
    Param-shared CTDE Actor-Critic with Auxiliary Off-Policy Bellman Q-Manifold:
      - Parameter-shared local permutation-invariant feature encoder (self_enc, peer_enc)
      - Decentralized Actor policy heads (act_loc, act_peer, act_fc) - never see gstate unless actor_uses_global=True
      - Centralized State-Value Critic (v_head, v_glob) - sees gstate when use_central_critic=True
      - Auxiliary Off-Policy Bellman Q-Critic (q_val, q_loc, q_peer, q_fc) for stable value geometry
      - Backwards-compatible aliases:
          .actor = self.self_enc
          .critic = self.v_head
          .feature = self.self_enc
    """

    def __init__(
        self,
        obs_dim: int = 75,
        global_dim: int = 91,
        action_dim: int = 9,
        hidden_dim: int = 96,
        use_central_critic: bool = True,
        actor_uses_global: bool = False,
        fc_scale: float = 0.008,
        logit_scale: float = 0.03,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.global_dim = global_dim
        self.action_dim = action_dim
        self.k_max = action_dim - 1
        self.use_central_critic = use_central_critic
        self.actor_uses_global = actor_uses_global
        self.logit_scale = float(logit_scale)

        p_dim = hidden_dim // 2
        g_extra = global_dim - obs_dim
        g_emb = 32 if hidden_dim >= 96 else 24

        # 1. Local feature encoders
        self.self_enc = nn.Sequential(nn.Linear(8, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())
        self.peer_enc = nn.Sequential(nn.Linear(7, p_dim), nn.LayerNorm(p_dim), nn.ReLU())
        base_ctx = hidden_dim + 2 * p_dim

        if actor_uses_global:
            self.glob_enc = nn.Sequential(nn.Linear(g_extra, g_emb), nn.LayerNorm(g_emb), nn.ReLU())
            _ortho(self.glob_enc[0], 0.25)
            ctx_dim = base_ctx + g_emb
        else:
            self.glob_enc = None
            ctx_dim = base_ctx

        # 2. Auxiliary Dueling Q-Critic heads
        self.q_val = nn.Sequential(
            nn.Linear(base_ctx, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q_loc = nn.Sequential(
            nn.Linear(ctx_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q_peer = KaimingFactoredPeerHead(ctx_dim, p_dim, hidden_dim)
        self.q_fc = PeerProactive2Branch(k_max=self.k_max, fc_scale=fc_scale)

        # 3. Decentralized Actor policy heads
        self.act_loc = nn.Sequential(
            nn.Linear(ctx_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.act_peer = KaimingFactoredPeerHead(ctx_dim, p_dim, hidden_dim)
        self.act_fc = PeerProactive2Branch(k_max=self.k_max, fc_scale=fc_scale / logit_scale)

        # 4. State-Value Critic V(obs, gstate)
        self.v_head = nn.Sequential(
            nn.Linear(base_ctx, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        if use_central_critic:
            self.v_glob = nn.Sequential(
                nn.Linear(base_ctx + g_extra, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
            _ortho(self.v_glob[0], 0.5)
            _ortho(self.v_glob[3], 0.25)
        else:
            self.v_glob = None

        # Aliases for existing test signatures (.actor, .critic, .feature)
        self.actor = self.self_enc
        self.critic = self.v_head
        self.feature = self.self_enc

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

    def q_values(
        self, obs: torch.Tensor, mask: torch.Tensor, gstate: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h_base, h_ctx, e_peers = self._encode(obs, mask, gstate)
        v = self.q_val(h_base)
        ap = self.q_peer(h_ctx, e_peers) + self.q_fc(obs, mask)
        adv = torch.cat([self.q_loc(h_ctx), ap], dim=-1)
        mf = mask.to(adv.dtype)
        adv_mean = (adv * mf).sum(dim=-1, keepdim=True) / mf.sum(dim=-1, keepdim=True).clamp(min=1.0)
        return (v + (adv - adv_mean)).masked_fill(~mask, -1e9)

    def critic_value(
        self,
        obs: torch.Tensor,
        gstate_or_mask: Optional[torch.Tensor] = None,
        gstate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if gstate is not None:
            mask = gstate_or_mask
            gs = gstate
        elif (
            gstate_or_mask is not None
            and gstate_or_mask.dim() == 2
            and gstate_or_mask.shape[1] == self.global_dim
        ):
            gs = gstate_or_mask
            bsz = obs.shape[0]
            mask = torch.ones(bsz, self.action_dim, dtype=torch.bool, device=obs.device)
            p = obs[:, 11 : 11 + 8 * self.k_max].view(bsz, self.k_max, 8)
            mask[:, 1 : 1 + self.k_max] = p[:, :, 0] > 0.5
        elif gstate_or_mask is not None and gstate_or_mask.dtype == torch.bool:
            mask = gstate_or_mask
            gs = None
        else:
            gs = gstate_or_mask
            bsz = obs.shape[0]
            mask = torch.ones(bsz, self.action_dim, dtype=torch.bool, device=obs.device)
            p = obs[:, 11 : 11 + 8 * self.k_max].view(bsz, self.k_max, 8)
            mask[:, 1 : 1 + self.k_max] = p[:, :, 0] > 0.5

        with torch.no_grad():
            h_base, _, _ = self._encode(obs, mask, gs)
        v = self.v_head(h_base)
        if self.use_central_critic and self.v_glob is not None:
            bsz = obs.shape[0]
            ge = (
                torch.zeros(bsz, self.global_dim - self.obs_dim, dtype=obs.dtype, device=obs.device)
                if gs is None
                else gs[:, self.obs_dim :].clone()
            )
            ge[:, 8:10] = 0.0
            v = v + self.v_glob(torch.cat([h_base, ge], dim=-1))
        return v.squeeze(-1)

    def actor_logits(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
        gstate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        is_1d = obs.dim() == 1
        if is_1d:
            obs = obs.unsqueeze(0)
            mask = mask.unsqueeze(0)
            if gstate is not None:
                gstate = gstate.unsqueeze(0)

        _, h_ctx, e_peers = self._encode(obs, mask, gstate)
        l0 = self.act_loc(h_ctx)
        lp = self.act_peer(h_ctx, e_peers) + self.act_fc(obs, mask)
        logits = torch.cat([l0, lp], dim=-1).masked_fill(~mask, -1e9)
        return logits.squeeze(0) if is_1d else logits

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
        values = self.critic_value(obs, mask, gstate)
        return logp, entropy, values


class _NumpyActorRunner:
    """Exact float32 NumPy mirror of MaskedActorCritic for fast low-overhead DES step loops."""

    def __init__(self, mod: MaskedActorCritic, mode: str = "actor") -> None:
        self.k_max = mod.k_max
        self.obs_dim = mod.obs_dim
        self.actor_uses_global = mod.actor_uses_global
        self.mode = mode

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
        self.ag = (
            ext(mod.glob_enc) if (mod.actor_uses_global and mod.glob_enc is not None) else None
        )

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

    def logits(
        self, obs: np.ndarray, mask: np.ndarray, gstate: Optional[np.ndarray] = None
    ) -> np.ndarray:
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

    def sample(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        gstate: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
        deterministic: bool = True,
    ) -> int:
        raw = self.logits(obs, mask, gstate)
        if deterministic or rng is None:
            return int(np.argmax(raw))
        sh = raw - float(raw[mask].max())
        pr = np.where(mask, np.exp(np.clip(sh, -40.0, 0.0)), 0.0)
        ps = float(pr.sum())
        return (
            int(np.argmax(raw))
            if (ps <= 0.0 or not math.isfinite(ps))
            else int(rng.choice(len(mask), p=pr * (1.0 / ps)))
        )

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
        return (
            int(np.argmax(raw))
            if (ps <= 0.0 or not math.isfinite(ps))
            else int(rng.choice(len(mask), p=pr * (1.0 / ps)))
        )


class _CTDEReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, global_dim: int, action_dim: int) -> None:
        self.cap = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.mask = np.zeros((capacity, action_dim), dtype=bool)
        self.gstate = np.zeros((capacity, global_dim), dtype=np.float32)
        self.act = np.zeros(capacity, dtype=np.int64)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_mask = np.zeros((capacity, action_dim), dtype=bool)
        self.next_gstate = np.zeros((capacity, global_dim), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add_batch(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        gstate: np.ndarray,
        act: np.ndarray,
        rew: np.ndarray,
        next_obs: np.ndarray,
        next_mask: np.ndarray,
        next_gstate: np.ndarray,
    ) -> None:
        n = len(act)
        for k in range(n):
            idx = (self.ptr + k) % self.cap
            self.obs[idx] = obs[k]
            self.mask[idx] = mask[k]
            self.gstate[idx] = gstate[k]
            self.act[idx] = act[k]
            self.rew[idx] = rew[k]
            self.next_obs[idx] = next_obs[k]
            self.next_mask[idx] = next_mask[k]
            self.next_gstate[idx] = next_gstate[k]
        self.ptr = (self.ptr + n) % self.cap
        self.size = min(self.cap, self.size + n)

    def sample(self, batch_size: int, rng: np.random.Generator) -> Tuple[np.ndarray, ...]:
        idx = rng.integers(0, self.size, size=batch_size)
        return (
            self.obs[idx],
            self.mask[idx],
            self.gstate[idx],
            self.act[idx],
            self.rew[idx],
            self.next_obs[idx],
            self.next_mask[idx],
            self.next_gstate[idx],
        )


def evaluate_marl_policy(
    ac: MaskedActorCritic,
    sim: TaskOffloadingSim,
    deterministic: bool = True,
    seed: int = 0,
    bin_callback: Optional[Callable[[int, float], None]] = None,
) -> SimMetrics:
    """Evaluate a trained MaskedActorCritic policy deterministically on any TaskOffloadingSim."""
    runner = _NumpyActorRunner(ac, mode="q")
    rng = np.random.default_rng(seed)

    def _step(obs, mask, gstate, node_id, t_now_s, task_idx) -> int:
        return runner.sample(obs, mask, gstate, rng, deterministic=deterministic)

    metrics, _ = sim.run_marl_episode(_step, bin_callback=bin_callback)
    return metrics


class MAPPOAgent:
    """
    Parameter-Shared Masked Discrete MAPPO / IPPO / Oracle Trainer
    with CTDE SharedTrunk architecture and auxiliary Bellman Q manifold.
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
            fc_scale=0.008,
            logit_scale=0.03,
        ).to(self.device)

        self.ac_target = copy.deepcopy(self.ac).to(self.device)
        self.ac_target.eval()

        self.q_opt = torch.optim.AdamW(
            [
                {"params": self.ac.q_params(), "weight_decay": 2e-4, "lr": 3e-4},
                {"params": self.ac.q_fc.parameters(), "weight_decay": 2e-2, "lr": 1.5e-4},
            ],
            eps=1e-5,
        )
        self.v_opt = torch.optim.AdamW(self.ac.v_params(), lr=3e-4, weight_decay=2e-4, eps=1e-5)
        self.a_opt = torch.optim.AdamW(self.ac.actor_head_params(), lr=1.8e-3, weight_decay=5e-5, eps=1e-5)

        self.buf = _CTDEReplayBuffer(60000, self.obs_dim, self.global_dim, self.action_dim)
        self.rng = np.random.default_rng(cfg.seed * 23 + 7)
        self.rng_a = np.random.default_rng(cfg.seed * 41 + 13)

    def _collect_and_update(
        self,
        sim: TaskOffloadingSim,
        ep_idx: int,
        bin_callback: Optional[Callable[[int, float], None]] = None,
        forecast_active: bool = False,
    ) -> Dict[str, float]:
        warmup = 36
        self.q_opt.param_groups[0]["lr"] = 3e-4 if ep_idx < warmup else 8e-5

        fc_on = forecast_active and (ep_idx >= warmup)
        self.ac.q_fc.fc_active = fc_on
        self.ac_target.q_fc.fc_active = fc_on
        self.ac.act_fc.fc_active = fc_on
        self.ac_target.act_fc.fc_active = fc_on

        frac = min(1.0, max(0, ep_idx - 2) / 30.0)
        t_roll = 0.002 + 0.5 * (0.012 - 0.002) * (1.0 + math.cos(math.pi * frac))
        eps = 0.30 + min(1.0, ep_idx / 28.0) * (0.01 - 0.30)

        runner = _NumpyActorRunner(self.ac, mode="q")
        obs_list: List[np.ndarray] = []
        mask_list: List[np.ndarray] = []
        gstate_list: List[np.ndarray] = []
        act_list: List[int] = []
        node_list: List[int] = []
        tid_list: List[int] = []

        def _step(obs, mask, gstate, node_id, t_now_s, task_idx) -> int:
            act = runner.act(obs, mask, gstate, self.rng, temp=t_roll, eps=eps)
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
            return {
                "train_tsr": 100.0,
                "train_reward": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
            }

        obs_arr = np.stack(obs_list, axis=0).astype(np.float32)
        mask_arr = np.stack(mask_list, axis=0).astype(bool)
        gstate_arr = np.stack(gstate_list, axis=0).astype(np.float32)
        act_arr = np.array(act_list, dtype=np.int64)
        node_arr = np.array(node_list, dtype=np.int32)
        rew_arr = np.array([outcomes[tid].reward for tid in tid_list], dtype=np.float32)

        next_obs_arr = obs_arr.copy()
        next_mask_arr = mask_arr.copy()
        next_gstate_arr = gstate_arr.copy()
        for nid in range(sim.n_nodes):
            idxs = np.flatnonzero(node_arr == nid)
            if len(idxs) > 1:
                next_obs_arr[idxs[:-1]] = obs_arr[idxs[1:]]
                next_mask_arr[idxs[:-1]] = mask_arr[idxs[1:]]
                next_gstate_arr[idxs[:-1]] = gstate_arr[idxs[1:]]

        self.buf.add_batch(
            obs_arr,
            mask_arr,
            gstate_arr,
            act_arr,
            rew_arr,
            next_obs_arr,
            next_mask_arr,
            next_gstate_arr,
        )

        q_losses: List[float] = []
        v_losses: List[float] = []
        p_losses: List[float] = []

        min_buf = min(256, self.marl.dqn_batch_size)
        if self.buf.size >= min_buf:
            self.ac.train()
            q_all_p = self.ac.q_params() + list(self.ac.q_fc.parameters())
            qt_all_p = self.ac_target.q_params() + list(self.ac_target.q_fc.parameters())
            v_p = self.ac.v_params()
            vt_p = self.ac_target.v_params()

            n_updates = min(48, getattr(self.marl, "dqn_updates_per_episode", 48))
            bs = min(self.buf.size, getattr(self.marl, "dqn_batch_size", 256))
            gamma = 0.40

            for _ in range(n_updates):
                bo, bm, bg, ba, br, bno, bnm, bng = self.buf.sample(bs, self.rng)
                to = torch.from_numpy(bo).to(self.device)
                tm = torch.from_numpy(bm).to(self.device)
                tg = torch.from_numpy(bg).to(self.device)
                ta = torch.from_numpy(ba).to(self.device)
                tr = torch.from_numpy(br).to(self.device)
                tno = torch.from_numpy(bno).to(self.device)
                tnm = torch.from_numpy(bnm).to(self.device)
                tng = torch.from_numpy(bng).to(self.device)

                with torch.no_grad():
                    best_a = self.ac.q_values(tno, tnm, tng).argmax(dim=-1, keepdim=True)
                    tgt_q = tr + gamma * self.ac_target.q_values(tno, tnm, tng).gather(1, best_a).squeeze(-1)

                ql = F.smooth_l1_loss(self.ac.q_values(to, tm, tg).gather(1, ta.unsqueeze(-1)).squeeze(-1), tgt_q, beta=1.0)
                self.q_opt.zero_grad()
                ql.backward()
                nn.utils.clip_grad_norm_(q_all_p, 2.0)
                self.q_opt.step()

                with torch.no_grad():
                    tgt_v = tr + gamma * self.ac_target.critic_value(tno, tnm, tng)
                vl = F.smooth_l1_loss(self.ac.critic_value(to, tm, tg), tgt_v, beta=1.0)
                self.v_opt.zero_grad()
                vl.backward()
                nn.utils.clip_grad_norm_(v_p, 2.0)
                self.v_opt.step()

                with torch.no_grad():
                    for p1, p2 in zip(q_all_p + v_p, qt_all_p + vt_p):
                        p2.data.mul_(0.995).add_(p1.data, alpha=0.005)
                    self.ac.act_fc.peer_lin.weight.data.copy_(self.ac.q_fc.peer_lin.weight.data)
                    self.ac_target.act_fc.peer_lin.weight.data.copy_(self.ac_target.q_fc.peer_lin.weight.data)

                q_losses.append(float(ql.item()))
                v_losses.append(float(vl.item()))

            if ep_idx >= 2:
                a_all_p = self.ac.actor_head_params()
                at_all_p = self.ac_target.actor_head_params()
                n_act = 128 if ep_idx >= 32 else 80
                bs_act = min(self.buf.size, 512)
                for _ in range(n_act):
                    bo, bm, bg, ba, br, bno, bnm, bng = self.buf.sample(bs_act, self.rng_a)
                    to = torch.from_numpy(bo).to(self.device)
                    tm = torch.from_numpy(bm).to(self.device)
                    tg = torch.from_numpy(bg).to(self.device)
                    ta_t = torch.from_numpy(ba).to(self.device)
                    tr_t = torch.from_numpy(br).to(self.device)
                    tno_t = torch.from_numpy(bno).to(self.device)
                    tnm_t = torch.from_numpy(bnm).to(self.device)
                    tng_t = torch.from_numpy(bng).to(self.device)

                    with torch.no_grad():
                        h_base_t, h_ctx_t, e_peers_t = self.ac_target._encode(to, tm, tg)
                        v_t = self.ac_target.q_val(h_base_t)
                        qfc_t = self.ac_target.q_fc(to, tm)
                        ap_q = self.ac_target.q_peer(h_ctx_t, e_peers_t) + qfc_t
                        adv_q_raw = torch.cat([self.ac_target.q_loc(h_ctx_t), ap_q], dim=-1)
                        mf = tm.to(adv_q_raw.dtype)
                        vcnt = mf.sum(dim=-1, keepdim=True).clamp(min=1.0)
                        qv_m = torch.where(tm, adv_q_raw, torch.zeros_like(adv_q_raw))
                        adv_q = torch.where(tm, qv_m - qv_m.sum(dim=-1, keepdim=True) / vcnt, torch.zeros_like(adv_q_raw))
                        qc_m = (v_t + adv_q).masked_fill(~tm, -1e9)
                        q_prob = F.softmax(qc_m / 0.025, dim=-1)
                        td_adv = (tr_t + 0.40 * self.ac_target.critic_value(tno_t, tnm_t, tng_t)) - self.ac_target.critic_value(to, tm, tg)
                        td_adv_n = torch.clamp((td_adv - td_adv.mean()) / td_adv.std().clamp(min=0.1), -2.0, 2.0)

                    l0 = self.ac.act_loc(h_ctx_t)
                    lp = self.ac.act_peer(h_ctx_t, e_peers_t) + qfc_t / 0.03
                    lg = torch.cat([l0, lp], dim=-1).masked_fill(~tm, -1e9)
                    lg_v = torch.where(tm, lg, torch.zeros_like(lg))
                    adv_pi = torch.where(tm, lg_v - lg_v.sum(dim=-1, keepdim=True) / vcnt, torch.zeros_like(lg))
                    l_reg = (F.smooth_l1_loss(0.03 * adv_pi, adv_q, beta=0.008, reduction="none") * mf).sum() / mf.sum().clamp(min=1.0)
                    logp_all = F.log_softmax((0.03 * adv_pi).masked_fill(~tm, -1e9) / 0.025, dim=-1)
                    l_ce = -(q_prob * torch.where(tm, logp_all, torch.zeros_like(logp_all))).sum(dim=-1).mean()
                    logp_pg = F.log_softmax((0.03 * adv_pi).masked_fill(~tm, -1e9) / 0.25, dim=-1)
                    l_pg = -(td_adv_n * logp_pg.gather(1, ta_t.unsqueeze(-1)).squeeze(-1)).mean()
                    p_loss = l_reg + 0.06 * l_ce + 0.08 * l_pg
                    self.a_opt.zero_grad()
                    p_loss.backward()
                    nn.utils.clip_grad_norm_(a_all_p, 1.0)
                    self.a_opt.step()

                    with torch.no_grad():
                        for p1, p2 in zip(a_all_p, at_all_p):
                            p2.data.mul_(0.90).add_(p1.data, alpha=0.10)
                    p_losses.append(float(p_loss.item()))

            self.ac.eval()

        return {
            "train_tsr": metrics.tsr_pct,
            "train_eff_lat": metrics.effective_latency_mean_s,
            "train_offload_pct": metrics.offload_rate_pct,
            "train_reward": metrics.mean_reward,
            "policy_loss": float(np.mean(p_losses)) if p_losses else (float(np.mean(q_losses)) if q_losses else 0.0),
            "value_loss": float(np.mean(v_losses)) if v_losses else 0.0,
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
        best_state = copy.deepcopy(self.ac_target.state_dict())
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
                vm_b = evaluate_marl_policy(self.ac_target, burst_val_sim, deterministic=True, seed=self.cfg.seed)
                vm_c = evaluate_marl_policy(self.ac_target, val_sim, deterministic=True, seed=self.cfg.seed)
                val_score = 0.75 * vm_b.mean_reward + 0.25 * vm_c.mean_reward
                stats["val_burst_tsr"] = vm_b.tsr_pct
                stats["val_tsr"] = vm_c.tsr_pct
                stats["val_reward"] = val_score
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_state = copy.deepcopy(self.ac_target.state_dict())
                    best_ep = ep + 1
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
            "best_val_ep": best_ep,
            "best_val_reward": float(best_val_score),
            "wall_time_s": float(wall_s),
            "history": history,
            "test_metrics": asdict(test_metrics),
        }
