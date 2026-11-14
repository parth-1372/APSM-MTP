"""
Stage 6 Live Decentralized Gossip Learning Engine & Multi-Channel Communication Ledger.

Implements LiveGossipManager:
  - Manages N EdgeForecasterNodes, each with an EdgeLSTMForecaster and OnlineSemanticGate (APSM-H1).
  - Synchronizes with TaskOffloadingSim via the 30-second bin callback.
  - Enforces physical link propagation & transmission delays for model parameter broadcasts.
  - Applies age_weighted_merge upon network message delivery.
  - Injects live workload predictions into sim.cur_y_hat and sim.cur_surprise for operational beacons.
  - Maintains full multi-channel communication ledger reconciling model, telemetry, ack, and header bytes.
"""

from collections import deque
import copy
from dataclasses import dataclass, field
import math
from typing import Deque, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from phase3.config import APSMConfig, ForecasterConfig, GossipConfig, Phase3Config
from phase3.data.alibaba_loader import CuratedTraceDataset, make_supervised_windows
from phase3.data.topology import EdgeTopology
from phase3.env.task_simulator import TaskOffloadingSim
from phase3.forecasting.apsm import GateDecision, OnlineSemanticGate
from phase3.forecasting.models import EdgeLSTMForecaster, age_weighted_merge


@dataclass
class PendingModelMsg:
    deliver_time_s: float
    sender_id: int
    shared_weights: Dict[str, torch.Tensor]
    sample_age: float


@dataclass
class CommunicationLedger:
    """Rigorous multi-channel communication accounting reconciling physical link layers."""
    # Model communication channel (Gossip Learning)
    model_packets: int = 0
    model_bytes: int = 0
    model_broadcast_rounds: int = 0
    suppressed_checks: int = 0
    surprise_sends: int = 0
    heartbeat_sends: int = 0
    warmup_sends: int = 0
    baseline_sends: int = 0
    total_gossip_checks: int = 0

    # Operational telemetry channel (15s beacons)
    telemetry_packets: int = 0
    telemetry_bytes: int = 0
    telemetry_suppressed: int = 0

    # Task offloading channel (Data plane + headers)
    offload_count: int = 0
    offload_data_bytes: int = 0
    header_packets: int = 0
    header_bytes: int = 0

    # Acknowledgment channel (ACKs)
    ack_packets: int = 0
    ack_bytes: int = 0

    # Aggregated metrics
    cb_control_bytes: int = 0
    cb_system_bytes: int = 0
    suppression_rate_pct: float = 0.0

    def reconcile(self, sim: TaskOffloadingSim) -> "CommunicationLedger":
        """Reconcile ledger with simulation metrics."""
        self.telemetry_packets = sim.telemetry_packets
        self.telemetry_bytes = sim.telemetry_bytes
        self.telemetry_suppressed = sim.telemetry_suppressed
        self.offload_count = sim.offload_count
        self.offload_data_bytes = sim.offload_data_bytes
        self.header_packets = sim.offload_count
        self.header_bytes = sim.header_bytes
        self.ack_packets = sim.ack_packets
        self.ack_bytes = sim.ack_bytes

        # Control Plane Bytes = Model + Telemetry + ACK + Header
        self.cb_control_bytes = (
            self.model_bytes
            + self.telemetry_bytes
            + self.ack_bytes
            + self.header_bytes
        )
        # System Total Bytes = Control Plane + Pure Offload Payload Bytes
        pure_payload = max(0, self.offload_data_bytes - self.header_bytes)
        self.cb_system_bytes = self.cb_control_bytes + pure_payload

        if self.total_gossip_checks > 0:
            self.suppression_rate_pct = (
                float(self.suppressed_checks) / float(self.total_gossip_checks) * 100.0
            )
        else:
            self.suppression_rate_pct = float("nan")
        return self


class LiveGossipManager:
    """
    Stage 6 Unified Decentralized Gossip Learning & Semantic Gating Manager.
    Operates at 10-node scale during DES rollouts.
    """

    def __init__(
        self,
        dataset: CuratedTraceDataset,
        topo: EdgeTopology,
        cfg: Phase3Config,
        mode: Literal["apsm", "local_only", "unfiltered_gl", "off"] = "apsm",
        base_states: Optional[List[Dict[str, torch.Tensor]]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.dataset = dataset
        self.topo = topo
        self.cfg = cfg
        self.mode = mode
        self.n_nodes = dataset.n_nodes
        self.lookback_L = dataset.lookback_L
        self.device = device or torch.device(cfg.device)

        # Build per-node models, gates, optimizers, and buffers
        self.models: List[EdgeLSTMForecaster] = []
        self.gates: List[OnlineSemanticGate] = []
        self.optimizers: List[torch.optim.Optimizer] = []
        self.online_buffers: List[Deque[Tuple[np.ndarray, float]]] = []
        self.model_ages: List[float] = [1.0] * self.n_nodes
        self.inboxes: List[List[PendingModelMsg]] = [[] for _ in range(self.n_nodes)]
        self.last_preds: List[float] = [0.0] * self.n_nodes

        # Mode configuration for the gate
        gate_mode = "h1_residual" if mode == "apsm" else ("off" if mode == "unfiltered_gl" else "h1_residual")
        gate_cfg = cfg.apsm
        if gate_cfg.mode != gate_mode:
            from dataclasses import replace
            gate_cfg = replace(gate_cfg, mode=gate_mode)

        for i in range(self.n_nodes):
            m = EdgeLSTMForecaster(
                input_dim=cfg.forecaster.input_dim,
                hidden_dim=cfg.forecaster.hidden_dim,
                num_layers=cfg.forecaster.num_layers,
                dropout=cfg.forecaster.dropout,
                personalization=cfg.forecaster.personalization,
            ).to(self.device)
            if base_states is not None and i < len(base_states):
                m.load_state_dict(copy.deepcopy(base_states[i]))
            m.eval()
            self.models.append(m)

            gate = OnlineSemanticGate(gate_cfg)
            self.gates.append(gate)

            opt = torch.optim.Adam(m.parameters(), lr=cfg.forecaster.online_lr)
            self.optimizers.append(opt)

            buf: Deque[Tuple[np.ndarray, float]] = deque(maxlen=cfg.forecaster.online_buffer_len)
            self.online_buffers.append(buf)

        self.payload_bytes = self.models[0].serialized_payload_bytes(
            header_bytes=cfg.forecaster.head_bytes_header
        )
        self.ledger = CommunicationLedger()
        self.cadence_bins = max(1, getattr(cfg.gossip, "cadence_bins", 2))
        self._last_step_bin = -1
        self._has_stepped = False
        self.all_y_true: List[float] = []
        self.all_y_pred: List[float] = []
        self.all_y_pers: List[float] = []

    def _deliver_due_messages(self, current_time_s: float) -> int:
        """Deliver arriving model messages and execute age-weighted merge."""
        total_delivered = 0
        for i in range(self.n_nodes):
            inbox = self.inboxes[i]
            due = [m for m in inbox if m.deliver_time_s <= current_time_s + 1e-9]
            if not due:
                continue
            self.inboxes[i] = [m for m in inbox if m.deliver_time_s > current_time_s + 1e-9]

            local_shared = self.models[i].export_shared_weights()
            pkgs = [(m.shared_weights, m.sample_age) for m in due]
            merged, new_age = age_weighted_merge(
                local_shared,
                self.model_ages[i],
                pkgs,
                age_decay=self.cfg.gossip.age_decay,
                self_mix_alpha=self.cfg.gossip.self_mix_alpha,
            )
            self.models[i].load_shared_weights(merged)
            self.model_ages[i] = new_age
            total_delivered += len(due)

            # Optional post-merge online gradient step
            if (
                self.cfg.forecaster.calibrate_after_merge
                or self.cfg.forecaster.personalization == "p3_finetune"
            ) and len(self.online_buffers[i]) >= 4:
                self._gradient_step(i, steps=1)

        return total_delivered

    def _gradient_step(self, node_id: int, steps: int = 1) -> float:
        """Execute local gradient update on recent online buffer."""
        buf = self.online_buffers[node_id]
        if len(buf) < 4:
            return float("nan")
        xs = np.stack([item[0] for item in buf], axis=0)
        ys = np.array([item[1] for item in buf], dtype=np.float32)
        xt = torch.from_numpy(xs).to(self.device)
        yt = torch.from_numpy(ys).to(self.device)

        model = self.models[node_id]
        opt = self.optimizers[node_id]
        criterion = nn.MSELoss()
        model.train()
        last_loss = 0.0
        for _ in range(steps):
            opt.zero_grad()
            out = model(xt)
            loss = criterion(out, yt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            last_loss = float(loss.item())
        model.eval()
        self.model_ages[node_id] += float(steps)
        return last_loss

    def on_bin_boundary(self, cur_bin: int, t_now_s: float, sim: TaskOffloadingSim) -> None:
        """
        Called exactly at the 30s bin boundary inside TaskOffloadingSim._run_episode_loop.
        Strict chronological causality:
          1. Deliver in-flight model messages whose delivery timestamp <= t_now_s.
          2. For the newly closed past bin (cur_bin - 1), observe realized load y_true,
             record residual eps = |y_true - y_hat|, and update gate state.
          3. Evaluate gossip broadcast on equal cadence (cadence_bins).
          4. Predict 1-step ahead load y_hat for upcoming bin and inject into sim.cur_y_hat.
        """
        if cur_bin == self._last_step_bin:
            return
        self._last_step_bin = cur_bin

        # 1. Deliver any arriving gossip model packets
        self._deliver_due_messages(t_now_s)

        # Causal startup initialization: populate last_preds before observing past_bin
        if not self._has_stepped:
            self._has_stepped = True
            if cur_bin > 0:
                past_b = cur_bin - 1
                L = self.lookback_L
                for i in range(self.n_nodes):
                    if past_b >= L:
                        x_win = self.dataset.features[i, past_b - L : past_b]
                    else:
                        x_win = self.dataset.features[i, :past_b]
                        pad = np.zeros((L - len(x_win), x_win.shape[-1]), dtype=np.float32)
                        x_win = np.concatenate([pad, x_win], axis=0) if len(x_win) > 0 else np.zeros((L, 7), dtype=np.float32)
                    xt = torch.from_numpy(x_win).unsqueeze(0).to(self.device)
                    self.last_preds[i] = float(self.models[i].predict_nonneg(xt).item())

        # 2. Observe closed past bin
        if cur_bin > 0:
            past_bin = cur_bin - 1
            for i in range(self.n_nodes):
                y_true = float(self.dataset.targets[i, past_bin])
                y_pred = self.last_preds[i]
                eps, tau, exceeded = self.gates[i].record_causal_residual(y_true, y_pred)
                self.all_y_true.append(y_true)
                self.all_y_pred.append(y_pred)
                if past_bin > 0:
                    self.all_y_pers.append(float(self.dataset.targets[i, past_bin - 1]))

                # Append to online buffer for adaptation
                if past_bin >= self.lookback_L:
                    w_x = self.dataset.features[i, past_bin - self.lookback_L : past_bin]
                    self.online_buffers[i].append((w_x.copy(), y_true))

                # Local burst adaptation on causal surprise
                if (
                    self.cfg.forecaster.local_adapt_on_causal_surprise
                    and exceeded
                    and len(self.online_buffers[i]) >= 4
                ):
                    self._gradient_step(i, steps=self.cfg.forecaster.online_steps_per_trigger)

        # 3. Evaluate Gossip Learning broadcast
        is_cadence_slot = (cur_bin % self.cadence_bins == 0)
        if is_cadence_slot and self.mode in ("apsm", "unfiltered_gl"):
            for i in range(self.n_nodes):
                self.ledger.total_gossip_checks += 1
                decision: GateDecision = self.gates[i].evaluate_gossip_send()

                if self.mode == "apsm":
                    should_send = decision.should_send
                    if decision.reason == "suppressed":
                        self.ledger.suppressed_checks += 1
                    elif decision.reason == "surprise_trigger":
                        self.ledger.surprise_sends += 1
                    elif decision.reason == "heartbeat":
                        self.ledger.heartbeat_sends += 1
                    elif decision.reason == "warmup_fail_open":
                        self.ledger.warmup_sends += 1
                else:  # unfiltered_gl
                    should_send = True
                    self.ledger.baseline_sends += 1

                if should_send:
                    shared_w = self.models[i].export_shared_weights()
                    nbrs = self.topo.neighbor_slots(i)
                    self.ledger.model_broadcast_rounds += 1

                    for j in nbrs:
                        # Schedule model transmission contending with physical link
                        deliver_ts = sim.schedule_model_packet(i, j, self.payload_bytes, t_now_s)

                        msg = PendingModelMsg(
                            deliver_time_s=deliver_ts,
                            sender_id=i,
                            shared_weights={k: v.clone() for k, v in shared_w.items()},
                            sample_age=self.model_ages[i],
                        )
                        self.inboxes[j].append(msg)

                        self.ledger.model_packets += 1
                        self.ledger.model_bytes += self.payload_bytes

        # 4. Predict 1-step ahead workload for next bin & inject into simulator
        L = self.lookback_L
        for i in range(self.n_nodes):
            if cur_bin >= L:
                x_win = self.dataset.features[i, cur_bin - L : cur_bin]
            else:
                x_win = self.dataset.features[i, :cur_bin]
                pad = np.zeros((L - len(x_win), x_win.shape[-1]), dtype=np.float32)
                x_win = np.concatenate([pad, x_win], axis=0) if len(x_win) > 0 else np.zeros((L, 7), dtype=np.float32)

            xt = torch.from_numpy(x_win).unsqueeze(0).to(self.device)
            pred = float(self.models[i].predict_nonneg(xt).item())
            self.last_preds[i] = pred
            sim.cur_y_hat[i] = pred
            sim.cur_surprise[i] = self.gates[i].current_norm_surprise

    def compute_forecasting_metrics(self) -> Dict[str, float]:
        """Compute test-window direct forecasting metrics: MSE, RMSE, MAE, and persistence baseline."""
        if not self.all_y_true:
            return {"fc_mse": 0.0, "fc_rmse": 0.0, "fc_mae": 0.0, "fc_pers_mse": 0.0}
        yt = np.array(self.all_y_true, dtype=np.float64)
        yp = np.array(self.all_y_pred, dtype=np.float64)
        mse = float(np.mean((yt - yp) ** 2))
        rmse = float(math.sqrt(mse))
        mae = float(np.mean(np.abs(yt - yp)))
        if self.all_y_pers and len(self.all_y_pers) == len(yt):
            ypers = np.array(self.all_y_pers, dtype=np.float64)
            pers_mse = float(np.mean((yt - ypers) ** 2))
        else:
            pers_mse = float("nan")
        return {
            "fc_mse": mse,
            "fc_rmse": rmse,
            "fc_mae": mae,
            "fc_pers_mse": pers_mse,
        }

