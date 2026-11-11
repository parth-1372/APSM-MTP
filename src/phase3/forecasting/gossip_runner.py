import copy
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from phase3.config import Phase3Config
from phase3.data.alibaba_loader import CuratedTraceDataset, make_supervised_windows
from phase3.data.topology import EdgeTopology
from phase3.forecasting.apsm import OnlineSemanticGate
from phase3.forecasting.models import EdgeLSTMForecaster, age_weighted_merge


@dataclass
class PendingModelMsg:
    deliver_time_s: float
    sender_id: int
    shared_weights: Dict[str, torch.Tensor]
    sample_age: float


class EdgeForecasterNode:
    """Single decentralized forecasting + APSM node in Stage 3 (and reusable inside Stage 6 MARL)."""

    def __init__(
        self,
        node_id: int,
        model: EdgeLSTMForecaster,
        optimizer: torch.optim.Optimizer,
        gate: OnlineSemanticGate,
        neighbors: List[int],
        initial_age: float,
        online_buffer: Deque[Tuple[np.ndarray, float]],
        cfg: Phase3Config,
    ) -> None:
        self.node_id = node_id
        self.model = model
        self.optimizer = optimizer
        self.gate = gate
        self.neighbors = neighbors
        self.model_age = float(initial_age)
        self.online_buffer = online_buffer
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.criterion = nn.MSELoss()

        self.inbox: List[PendingModelMsg] = []
        self.last_pred: Optional[float] = None
        self.last_val_loss: Optional[float] = None

        # Telemetry & H2 communication counters (counted strictly over test window)
        self.test_gossip_checks: int = 0
        self.test_suppressed_checks: int = 0
        self.test_broadcast_rounds: int = 0
        self.test_packets_sent: int = 0
        self.test_model_bytes_sent: int = 0
        self.total_grad_calls: int = 0
        self.test_grad_calls: int = 0

    def receive_and_merge_due(self, current_time_s: float, is_test: bool = False) -> int:
        """Apply all weights messages whose network arrival time <= current_time_s."""
        due = [m for m in self.inbox if m.deliver_time_s <= current_time_s + 1e-9]
        if not due:
            return 0
        self.inbox = [m for m in self.inbox if m.deliver_time_s > current_time_s + 1e-9]

        local_shared = self.model.export_shared_weights()
        pkgs = [(m.shared_weights, m.sample_age) for m in due]
        merged, new_age = age_weighted_merge(
            local_shared,
            self.model_age,
            pkgs,
            age_decay=self.cfg.gossip.age_decay,
            self_mix_alpha=self.cfg.gossip.self_mix_alpha,
        )
        self.model.load_shared_weights(merged)
        self.model_age = new_age

        # Optional post-merge local calibration step (or P3 fine-tune mode)
        if (
            self.cfg.forecaster.calibrate_after_merge
            or self.cfg.forecaster.personalization == "p3_finetune"
        ) and len(self.online_buffer) >= 4:
            self._gradient_step(steps=1, lr=self.cfg.forecaster.online_lr, is_test=is_test)

        return len(due)

    def predict_next(self, window_x: np.ndarray) -> float:
        x_t = torch.from_numpy(window_x).unsqueeze(0).to(self.device)
        pred = float(self.model.predict_nonneg(x_t).item())
        self.last_pred = pred
        return pred

    def _gradient_step(self, steps: int, lr: Optional[float] = None, is_test: bool = False) -> float:
        if len(self.online_buffer) < 4:
            return float("nan")
        if lr is not None:
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr
        xs = np.stack([item[0] for item in self.online_buffer], axis=0)
        ys = np.array([item[1] for item in self.online_buffer], dtype=np.float32)
        xt = torch.from_numpy(xs).to(self.device)
        yt = torch.from_numpy(ys).to(self.device)

        self.model.train()
        last_loss = 0.0
        for _ in range(steps):
            self.optimizer.zero_grad()
            out = self.model(xt)
            loss = self.criterion(out, yt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
            self.optimizer.step()
            last_loss = float(loss.item())
        self.model.eval()
        self.model_age += float(steps)
        self.total_grad_calls += 1
        if is_test:
            self.test_grad_calls += 1
        return last_loss

    def observe_outcome_and_adapt(
        self,
        window_x: np.ndarray,
        y_true: float,
        bin_idx: int,
        eval_val_x: Optional[np.ndarray] = None,
        eval_val_y: Optional[np.ndarray] = None,
        is_test: bool = False,
    ) -> Tuple[float, float, bool]:
        """Close forecast bin, update APSM surprise state, and trigger bounded local retrain."""
        pred = self.last_pred if self.last_pred is not None else y_true
        self.online_buffer.append((window_x.copy(), float(y_true)))

        triggered = False
        causal_exceeded = False
        if self.cfg.apsm.mode == "h1_residual":
            eps, tau, triggered = self.gate.record_causal_residual(y_true, pred)
            causal_exceeded = triggered
        elif self.cfg.apsm.mode == "val_delta":
            # Measure post-deployment loss on recent local slice (fixing B1/B3)
            with torch.no_grad():
                if eval_val_x is not None and len(eval_val_x) > 0:
                    vx = torch.from_numpy(eval_val_x).to(self.device)
                    vy = torch.from_numpy(eval_val_y).to(self.device)
                    vloss = float(self.criterion(self.model(vx), vy).item())
                else:
                    buf = list(self.online_buffer)[-min(16, len(self.online_buffer)) :]
                    vx = torch.from_numpy(np.stack([b[0] for b in buf], axis=0)).to(self.device)
                    vy = torch.tensor([b[1] for b in buf], dtype=torch.float32, device=self.device)
                    vloss = float(self.criterion(self.model(vx), vy).item())
            eps, tau, triggered = self.gate.record_val_loss_delta(vloss)
            self.last_val_loss = vloss
        else:
            # APSM disabled ("off"): still log causal residual passively for matched comparison without gating
            eps, tau, causal_exceeded = self.gate.record_causal_residual(y_true, pred)

        periodic_due = (
            self.cfg.apsm.periodic_online_retrain_interval > 0
            and (bin_idx % self.cfg.apsm.periodic_online_retrain_interval == 0)
        )
        local_surprise_adapt = self.cfg.forecaster.local_adapt_on_causal_surprise and causal_exceeded
        should_retrain = (self.cfg.apsm.retrain_on_trigger and triggered) or local_surprise_adapt or periodic_due
        if should_retrain:
            self._gradient_step(
                steps=self.cfg.forecaster.online_steps_per_trigger,
                lr=self.cfg.forecaster.online_lr,
                is_test=is_test,
            )

        return eps, tau, triggered


def _clone_pretrained_model(
    base_state: Dict[str, torch.Tensor], cfg: Phase3Config
) -> Tuple[EdgeLSTMForecaster, torch.optim.Optimizer]:
    model = EdgeLSTMForecaster(
        input_dim=cfg.forecaster.input_dim,
        hidden_dim=cfg.forecaster.hidden_dim,
        num_layers=cfg.forecaster.num_layers,
        dropout=cfg.forecaster.dropout,
        personalization=cfg.forecaster.personalization,
    ).to(torch.device(cfg.device))
    model.load_state_dict(copy.deepcopy(base_state))
    model.eval()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.forecaster.online_lr)
    return model, opt


def _pretrain_base_models(
    dataset: CuratedTraceDataset,
    cfg: Phase3Config,
    pretrain_history_frac: Union[float, Sequence[float]] = 1.0,
) -> Tuple[List[Dict[str, torch.Tensor]], List[Deque[Tuple[np.ndarray, float]]]]:
    """
    Deterministically pretrain one EdgeLSTMForecaster per node on ONLY
    that node's chronological training split `[0 .. train_end_bin)`.
    When `cfg.forecaster.common_init=True`, all nodes start from the exact
    same random seed initialization `theta_0` so gossip weight averaging does not
    collide across incompatible hidden-unit permutations.
    Supports scalar or per-node sequence `pretrain_history_frac` for scarce/mixed cold-start audits.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)
    criterion = nn.MSELoss()

    shared_init_state: Optional[Dict[str, torch.Tensor]] = None
    if cfg.forecaster.common_init:
        init_model = EdgeLSTMForecaster(
            input_dim=cfg.forecaster.input_dim,
            hidden_dim=cfg.forecaster.hidden_dim,
            num_layers=cfg.forecaster.num_layers,
            dropout=cfg.forecaster.dropout,
            personalization=cfg.forecaster.personalization,
        ).to(device)
        shared_init_state = copy.deepcopy(init_model.state_dict())

    base_states: List[Dict[str, torch.Tensor]] = []
    init_buffers: List[Deque[Tuple[np.ndarray, float]]] = []
    L = dataset.lookback_L

    for i in range(dataset.n_nodes):
        hf = (
            float(pretrain_history_frac[i])
            if isinstance(pretrain_history_frac, (list, tuple))
            else float(pretrain_history_frac)
        )
        # Strictly inside [0, train_end_bin)
        usable_end = dataset.train_end_bin
        if hf < 1.0:
            usable_start = max(L, int(usable_end * (1.0 - hf)))
        else:
            usable_start = L
        X_tr, y_tr = make_supervised_windows(
            dataset.features[i], dataset.targets[i], L, usable_start, usable_end
        )

        model = EdgeLSTMForecaster(
            input_dim=cfg.forecaster.input_dim,
            hidden_dim=cfg.forecaster.hidden_dim,
            num_layers=cfg.forecaster.num_layers,
            dropout=cfg.forecaster.dropout,
            personalization=cfg.forecaster.personalization,
        ).to(device)
        if shared_init_state is not None:
            model.load_state_dict(copy.deepcopy(shared_init_state))
        opt = torch.optim.Adam(model.parameters(), lr=cfg.forecaster.lr)

        if len(X_tr) > 0:
            xt = torch.from_numpy(X_tr).to(device)
            yt = torch.from_numpy(y_tr).to(device)
            bs = min(cfg.forecaster.pretrain_batch_size, len(X_tr))
            rng = np.random.default_rng(cfg.seed * 1000 + i)
            model.train()
            for _ in range(cfg.forecaster.pretrain_epochs):
                perm = rng.permutation(len(X_tr))
                for start in range(0, len(X_tr), bs):
                    idx = perm[start : start + bs]
                    opt.zero_grad()
                    loss = criterion(model(xt[idx]), yt[idx])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    opt.step()

        model.eval()
        base_states.append(copy.deepcopy(model.state_dict()))

        buf: Deque[Tuple[np.ndarray, float]] = deque(maxlen=cfg.forecaster.online_buffer_len)
        tail = min(len(X_tr), cfg.forecaster.online_buffer_len)
        for j in range(len(X_tr) - tail, len(X_tr)):
            buf.append((X_tr[j].copy(), float(y_tr[j])))
        init_buffers.append(buf)

    return base_states, init_buffers


def _run_neural_variant(
    variant_name: str,
    dataset: CuratedTraceDataset,
    topo: EdgeTopology,
    cfg: Phase3Config,
    base_states: List[Dict[str, torch.Tensor]],
    init_buffers: List[Deque[Tuple[np.ndarray, float]]],
    enable_gossip: bool,
    collect_trajectories: bool = False,
) -> Dict[str, Any]:
    L = dataset.lookback_L
    dt = dataset.dt_bin_s
    t_val_start = dataset.train_end_bin
    t_test_start = dataset.val_end_bin
    t_end = dataset.n_bins

    nodes: List[EdgeForecasterNode] = []
    # Local static validation slices for val_delta mode (strictly from train tail / early val, no test leakage)
    val_slices: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(dataset.n_nodes):
        model, opt = _clone_pretrained_model(base_states[i], cfg)
        gate = OnlineSemanticGate(cfg.apsm)
        buf = deque(init_buffers[i], maxlen=cfg.forecaster.online_buffer_len)
        nbrs = topo.neighbor_slots(i) if enable_gossip else []
        nodes.append(
            EdgeForecasterNode(
                node_id=i,
                model=model,
                optimizer=opt,
                gate=gate,
                neighbors=nbrs,
                initial_age=float(max(1, len(buf))),
                online_buffer=buf,
                cfg=cfg,
            )
        )
        vx, vy = make_supervised_windows(
            dataset.features[i], dataset.targets[i], L, max(L, t_val_start - 48), t_val_start
        )
        val_slices.append((vx, vy))

    payload_bytes = nodes[0].model.serialized_payload_bytes(cfg.forecaster.head_bytes_header)
    preds = np.zeros((dataset.n_nodes, t_end - t_test_start), dtype=np.float32)
    trues = dataset.targets[:, t_test_start:t_end].astype(np.float32)

    preds_all: Optional[np.ndarray] = None
    surp_all: Optional[np.ndarray] = None
    bin_pkts_all: Optional[np.ndarray] = None
    bin_bytes_all: Optional[np.ndarray] = None
    if collect_trajectories:
        preds_all = np.zeros((dataset.n_nodes, t_end), dtype=np.float32)
        surp_all = np.zeros((dataset.n_nodes, t_end), dtype=np.float32)
        bin_pkts_all = np.zeros(t_end, dtype=np.int32)
        bin_bytes_all = np.zeros(t_end, dtype=np.int64)

    t0_wall = time.perf_counter()
    for t in range(t_val_start, t_end):
        t_now_s = t * dt
        is_test = t >= t_test_start

        # 1. Deliver due model gossip messages BEFORE making predictions for bin t
        if enable_gossip:
            for node in nodes:
                node.receive_and_merge_due(t_now_s, is_test=is_test)

        # 2. Causal forecast at start of bin t using strictly <= t-1 features & <= t-1 surprise
        for i, node in enumerate(nodes):
            win = dataset.features[i, t - L : t]
            y_hat = node.predict_next(win)
            if is_test:
                preds[i, t - t_test_start] = y_hat
            if collect_trajectories:
                assert preds_all is not None and surp_all is not None
                preds_all[i, t] = y_hat
                surp_all[i, t] = float(node.gate.current_norm_surprise)

        # 3. Bin t closes -> observe true y_i(t), compute causal residual / APSM update / local adapt
        for i, node in enumerate(nodes):
            win = dataset.features[i, t - L : t]
            y_true = float(dataset.targets[i, t])
            vx, vy = val_slices[i]
            node.observe_outcome_and_adapt(win, y_true, t, vx, vy, is_test=is_test)

        # 4. Equal-cadence Gossip send gate evaluation (fixing Phase 2 B6)
        if enable_gossip and ((t - t_val_start) % cfg.gossip.cadence_bins == 0):
            for i, node in enumerate(nodes):
                dec = node.gate.evaluate_gossip_send()
                if is_test:
                    node.test_gossip_checks += 1
                    if not dec.should_send:
                        node.test_suppressed_checks += 1
                if dec.should_send and node.neighbors:
                    k_send = max(1, int(np.ceil(len(node.neighbors) * cfg.gossip.target_probability)))
                    targets = node.neighbors[:k_send]
                    shared_w = node.model.export_shared_weights()
                    if collect_trajectories:
                        assert bin_pkts_all is not None and bin_bytes_all is not None
                        bin_pkts_all[t] += len(targets)
                        bin_bytes_all[t] += len(targets) * payload_bytes
                    if is_test:
                        node.test_broadcast_rounds += 1
                        node.test_packets_sent += len(targets)
                        node.test_model_bytes_sent += len(targets) * payload_bytes
                    for peer_id in targets:
                        rtt = float(topo.rtt_s[i, peer_id])
                        bw_bps = float(topo.bw_mbps[i, peer_id]) * 1e6
                        tx_lat_s = (rtt / 2.0) + (payload_bytes * 8.0) / max(1e6, bw_bps)
                        nodes[peer_id].inbox.append(
                            PendingModelMsg(
                                deliver_time_s=t_now_s + tx_lat_s,
                                sender_id=i,
                                shared_weights={k: v.clone() for k, v in shared_w.items()},
                                sample_age=node.model_age,
                            )
                        )

    wall_s = time.perf_counter() - t0_wall
    sq_err = (preds - trues) ** 2
    abs_err = np.abs(preds - trues)
    per_node_mse = sq_err.mean(axis=1)

    total_checks = sum(n.test_gossip_checks for n in nodes)
    total_supp = sum(n.test_suppressed_checks for n in nodes)
    total_rounds = sum(n.test_broadcast_rounds for n in nodes)
    total_pkts = sum(n.test_packets_sent for n in nodes)
    total_bytes = sum(n.test_model_bytes_sent for n in nodes)
    total_grad_calls = sum(n.total_grad_calls for n in nodes)
    test_grad_calls = sum(n.test_grad_calls for n in nodes)

    out = {
        "variant": variant_name,
        "test_mse": float(sq_err.mean()),
        "test_rmse": float(np.sqrt(sq_err.mean())),
        "test_mae": float(abs_err.mean()),
        "per_node_mse_min_max": [float(per_node_mse.min()), float(per_node_mse.max())],
        "payload_bytes": int(payload_bytes),
        "shared_params": int(nodes[0].model.shared_num_params()),
        "total_params": int(nodes[0].model.total_num_params()),
        "test_gossip_checks": int(total_checks),
        "test_suppressed_checks": int(total_supp),
        "test_broadcast_rounds": int(total_rounds),
        "test_packets_sent": int(total_pkts),
        "test_model_bytes": int(total_bytes),
        "total_grad_calls": int(total_grad_calls),
        "test_grad_calls": int(test_grad_calls),
        "sr_dec_pct": (float(total_supp) / total_checks * 100.0) if total_checks > 0 else 0.0,
        "wall_time_s": float(wall_s),
    }
    if collect_trajectories:
        out["forecast_matrix"] = preds_all
        out["surprise_matrix"] = surp_all
        out["per_bin_packets"] = bin_pkts_all
        out["per_bin_bytes"] = bin_bytes_all
    return out


def generate_causal_forecast_matrices(
    variant_mode: str,
    dataset: CuratedTraceDataset,
    topo: EdgeTopology,
    base_cfg: Phase3Config,
) -> Dict[str, Any]:
    """
    Generate strictly causal `(N, n_bins)` forecast & normalized surprise matrices across
    ALL splits `[train .. val .. test]`:
      - `train` `[L .. train_end_bin)`: 4-block expanding walk-forward OOS forecasts
        so MARL training observes realistic OOS forecast errors & surprise spikes without train leakage.
      - `val` & `test` `[train_end_bin .. n_bins)`: exact online Stage 3 rollout pretrained strictly on `[0 .. train_end_bin)`.
    Supported `variant_mode`: `"local_lstm"`, `"gl_baseline"`, `"apsm_val_delta"`, `"apsm_h1"`, `"oracle"`.
    """
    n, n_bins = dataset.n_nodes, dataset.n_bins
    L = dataset.lookback_L
    forecast_mat = np.zeros((n, n_bins), dtype=np.float32)
    surprise_mat = np.zeros((n, n_bins), dtype=np.float32)

    if variant_mode == "oracle":
        # Ideal next-step causal-indexed benchmark upper bound: exact target for that bin, zero surprise
        forecast_mat[:, :] = dataset.targets
        return {
            "variant": "oracle",
            "forecast_matrix": forecast_mat,
            "surprise_matrix": surprise_mat,
            "per_bin_packets": np.zeros(n_bins, dtype=np.int32),
            "per_bin_bytes": np.zeros(n_bins, dtype=np.int64),
            "test_mse": 0.0,
            "test_packets_sent": 0,
            "test_model_bytes": 0,
        }

    enable_gl = variant_mode != "local_lstm"
    apsm_mode = "off" if variant_mode in ("local_lstm", "gl_baseline") else (
        "val_delta" if variant_mode == "apsm_val_delta" else "h1_residual"
    )
    v_cfg = replace(base_cfg, apsm=replace(base_cfg.apsm, mode=apsm_mode))

    # 1. Warmup early train bins [L .. block_bounds[1]) with causal EMA4 (strictly <= t-1 lags)
    tr_span = max(4, dataset.train_end_bin - L)
    block_bounds = [
        L,
        L + tr_span // 4,
        L + tr_span // 2,
        L + (3 * tr_span) // 4,
        dataset.train_end_bin,
    ]
    for t in range(L, block_bounds[1]):
        lo = max(0, t - 4)
        forecast_mat[:, t] = dataset.targets[:, lo:t].mean(axis=1)
        surprise_mat[:, t] = 0.0

    # Expanding OOS causal blocks over remaining training bins so RL never sees overfit train MSE
    device = torch.device(v_cfg.device)
    crit = nn.MSELoss()
    for blk_idx in range(1, len(block_bounds) - 1):
        tr_end = block_bounds[blk_idx]
        ev_end = block_bounds[blk_idx + 1]
        torch.manual_seed(v_cfg.seed + blk_idx)
        shared_init = None
        if v_cfg.forecaster.common_init:
            m0 = EdgeLSTMForecaster(
                input_dim=v_cfg.forecaster.input_dim,
                hidden_dim=v_cfg.forecaster.hidden_dim,
                num_layers=v_cfg.forecaster.num_layers,
                dropout=v_cfg.forecaster.dropout,
                personalization=v_cfg.forecaster.personalization,
            ).to(device)
            shared_init = copy.deepcopy(m0.state_dict())

        blk_nodes: List[EdgeForecasterNode] = []
        for i in range(n):
            m = EdgeLSTMForecaster(
                input_dim=v_cfg.forecaster.input_dim,
                hidden_dim=v_cfg.forecaster.hidden_dim,
                num_layers=v_cfg.forecaster.num_layers,
                dropout=v_cfg.forecaster.dropout,
                personalization=v_cfg.forecaster.personalization,
            ).to(device)
            if shared_init is not None:
                m.load_state_dict(copy.deepcopy(shared_init))
            opt = torch.optim.Adam(m.parameters(), lr=v_cfg.forecaster.lr)
            X_b, y_b = make_supervised_windows(dataset.features[i], dataset.targets[i], L, L, tr_end)
            if len(X_b) > 0:
                xt = torch.from_numpy(X_b).to(device)
                yt = torch.from_numpy(y_b).to(device)
                bs = min(v_cfg.forecaster.pretrain_batch_size, len(X_b))
                rng = np.random.default_rng(v_cfg.seed * 1000 + blk_idx * 100 + i)
                m.train()
                for _ in range(v_cfg.forecaster.pretrain_epochs):
                    perm = rng.permutation(len(X_b))
                    for s0 in range(0, len(X_b), bs):
                        idx = perm[s0 : s0 + bs]
                        opt.zero_grad()
                        loss = crit(m(xt[idx]), yt[idx])
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(m.parameters(), 2.0)
                        opt.step()
            m.eval()
            buf: Deque[Tuple[np.ndarray, float]] = deque(maxlen=v_cfg.forecaster.online_buffer_len)
            tail = min(len(X_b), v_cfg.forecaster.online_buffer_len)
            for j in range(len(X_b) - tail, len(X_b)):
                buf.append((X_b[j].copy(), float(y_b[j])))
            opt_onl = torch.optim.Adam(m.parameters(), lr=v_cfg.forecaster.online_lr)
            blk_nodes.append(
                EdgeForecasterNode(
                    node_id=i,
                    model=m,
                    optimizer=opt_onl,
                    gate=OnlineSemanticGate(v_cfg.apsm),
                    neighbors=topo.neighbor_slots(i) if enable_gl else [],
                    initial_age=float(max(1, len(buf))),
                    online_buffer=buf,
                    cfg=v_cfg,
                )
            )

        pb = blk_nodes[0].model.serialized_payload_bytes(v_cfg.forecaster.head_bytes_header)
        vx_blk = [make_supervised_windows(dataset.features[i], dataset.targets[i], L, max(L, tr_end - 48), tr_end) for i in range(n)]
        for t in range(tr_end, ev_end):
            t_now_s = t * dataset.dt_bin_s
            if enable_gl:
                for node in blk_nodes:
                    node.receive_and_merge_due(t_now_s, is_test=False)
            for i, node in enumerate(blk_nodes):
                win = dataset.features[i, t - L : t]
                forecast_mat[i, t] = node.predict_next(win)
                surprise_mat[i, t] = float(node.gate.current_norm_surprise)
            for i, node in enumerate(blk_nodes):
                win = dataset.features[i, t - L : t]
                node.observe_outcome_and_adapt(win, float(dataset.targets[i, t]), t, vx_blk[i][0], vx_blk[i][1], is_test=False)
            if enable_gl and ((t - tr_end) % v_cfg.gossip.cadence_bins == 0):
                for i, node in enumerate(blk_nodes):
                    dec = node.gate.evaluate_gossip_send()
                    if dec.should_send and node.neighbors:
                        shared_w = node.model.export_shared_weights()
                        for peer_id in node.neighbors:
                            tx_lat = (float(topo.rtt_s[i, peer_id]) / 2.0) + (pb * 8.0) / max(1e6, float(topo.bw_mbps[i, peer_id]) * 1e6)
                            blk_nodes[peer_id].inbox.append(
                                PendingModelMsg(t_now_s + tx_lat, i, {k: v.clone() for k, v in shared_w.items()}, node.model_age)
                            )

    # 2. Exact Stage 3 rollout on [train_end_bin .. n_bins) (full val & test splits)
    base_states, init_buffers = _pretrain_base_models(dataset, v_cfg, pretrain_history_frac=1.0)
    res = _run_neural_variant(
        variant_mode,
        dataset,
        topo,
        v_cfg,
        base_states,
        init_buffers,
        enable_gossip=enable_gl,
        collect_trajectories=True,
    )
    forecast_mat[:, dataset.train_end_bin :] = res["forecast_matrix"][:, dataset.train_end_bin :]
    surprise_mat[:, dataset.train_end_bin :] = res["surprise_matrix"][:, dataset.train_end_bin :]
    res["forecast_matrix"] = forecast_mat
    res["surprise_matrix"] = surprise_mat
    return res


def run_stage3_benchmark(
    dataset: CuratedTraceDataset,
    topo: EdgeTopology,
    base_cfg: Phase3Config,
    pretrain_history_frac: Union[float, Sequence[float]] = 1.0,
) -> Dict[str, Any]:
    """
    Run all Stage 3 forecasting comparators on matched data, topology, and initial weights,
    reporting strictly separated H2 communication metrics and Persistence skill scores.
    """
    t_test = dataset.val_end_bin
    t_end = dataset.n_bins
    y_test = dataset.targets[:, t_test:t_end]
    y_pers = dataset.targets[:, t_test - 1 : t_end - 1]
    y_ema4 = np.mean(
        [dataset.targets[:, t_test - k : t_end - k] for k in range(1, 5)],
        axis=0,
    )

    pers_mse = float(np.mean((y_test - y_pers) ** 2))
    ema4_mse = float(np.mean((y_test - y_ema4) ** 2))

    base_states, init_buffers = _pretrain_base_models(
        dataset, base_cfg, pretrain_history_frac=pretrain_history_frac
    )

    # Variants sharing identical pretrained weights & online buffer per seed
    configs = [
        ("local_lstm", False, replace(base_cfg, apsm=replace(base_cfg.apsm, mode="off"))),
        ("gl_baseline", True, replace(base_cfg, apsm=replace(base_cfg.apsm, mode="off"))),
        ("apsm_val_delta", True, replace(base_cfg, apsm=replace(base_cfg.apsm, mode="val_delta"))),
        ("apsm_h1", True, replace(base_cfg, apsm=replace(base_cfg.apsm, mode="h1_residual"))),
    ]

    results = {}
    for name, enable_gl, v_cfg in configs:
        results[name] = _run_neural_variant(
            name, dataset, topo, v_cfg, base_states, init_buffers, enable_gossip=enable_gl
        )
        mse_val = results[name]["test_mse"]
        results[name]["skill_vs_pers_pct"] = (pers_mse - mse_val) / pers_mse * 100.0

    # Add strict H2 communication comparison relative to gl_baseline (E4)
    gl_res = results["gl_baseline"]
    for name in ("apsm_val_delta", "apsm_h1"):
        r = results[name]
        r["br_avoid_count"] = gl_res["test_broadcast_rounds"] - r["test_broadcast_rounds"]
        r["br_avoid_pct"] = (
            r["br_avoid_count"] / max(1, gl_res["test_broadcast_rounds"]) * 100.0
        )
        r["pp_avoid_count"] = gl_res["test_packets_sent"] - r["test_packets_sent"]
        r["pp_avoid_pct"] = r["pp_avoid_count"] / max(1, gl_res["test_packets_sent"]) * 100.0
        r["mb_avoid_bytes"] = gl_res["test_model_bytes"] - r["test_model_bytes"]
        r["mse_delta_vs_gl_pct"] = (
            (r["test_mse"] - gl_res["test_mse"]) / gl_res["test_mse"] * 100.0
        )

    return {
        "n_nodes": dataset.n_nodes,
        "seed": base_cfg.seed,
        "personalization": base_cfg.forecaster.personalization,
        "pretrain_history_frac": pretrain_history_frac,
        "apsm_stat": base_cfg.apsm.stat,
        "apsm_k": base_cfg.apsm.k,
        "persistence_mse": pers_mse,
        "persistence_rmse": float(np.sqrt(pers_mse)),
        "ema4_mse": ema4_mse,
        "ema4_skill_vs_pers_pct": (pers_mse - ema4_mse) / pers_mse * 100.0,
        "variants": results,
    }

