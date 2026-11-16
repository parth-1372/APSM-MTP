from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import numpy as np

from phase3.config import APSMConfig


@dataclass
class GateDecision:
    should_send: bool
    reason: str
    eps: float
    tau: float
    norm_surprise: float
    suppressed_streak: int


class OnlineSemanticGate:
    """
    Phase 3 Semantic Gating Controller (`APSM-W`).

    Supports:
      - `off`: Unfiltered Gossip Learning baseline (always sends on equal cadence)
      - `h1_residual`: Online causal prediction residual gate eps_pred(t) = |y(t) - y_hat(t|t-1)|
      - `val_delta`: Bug-fixed Phase 2 validation-loss delta baseline (same cadence & warmup guard)
    """

    def __init__(self, cfg: APSMConfig) -> None:
        self.cfg = cfg
        self._window: Deque[float] = deque(maxlen=max(2, cfg.window))
        self._suppressed_streak: int = 0
        self._best_val_loss: Optional[float] = None
        self._ewma_mean: Optional[float] = None
        self._ewma_var: float = 0.0

        # Track whether any causal surprise event occurred since last equal-cadence gossip check
        self._pending_surprise: bool = False
        self._last_eps: float = 0.0
        self._last_tau: float = float("inf")
        self._last_norm_surprise: float = 0.0

        # Audit counters (Strict H2 metric separation)
        self.total_gossip_checks: int = 0
        self.suppressed_checks: int = 0
        self.sent_checks: int = 0
        self.heartbeat_sends: int = 0
        self.surprise_sends: int = 0
        self.warmup_sends: int = 0
        self.baseline_sends: int = 0
        self.retrain_triggers: int = 0

        # Diagnostic trace per online step
        self.eps_history = []
        self.tau_history = []
        self.decision_history = []

    def _compute_tau(self) -> float:
        if len(self._window) < self.cfg.min_obs:
            return float("inf")
        arr = np.asarray(self._window, dtype=np.float64)
        if self.cfg.stat == "mad":
            med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - med)))
            robust_sigma = 1.4826 * mad
            return med + self.cfg.k * max(robust_sigma, 1e-6)
        if self.cfg.stat == "ewma":
            std = float(np.sqrt(max(self._ewma_var, 1e-12)))
            base = float(self._ewma_mean if self._ewma_mean is not None else np.mean(arr))
            return base + self.cfg.k * std
        # Default: rolling std multiplier matching Phase 2 math after clean warmup
        return self.cfg.k * float(np.std(arr))

    def record_causal_residual(self, y_true: float, y_pred: float) -> Tuple[float, float, bool]:
        """
        Called at the close of each forecast bin `t` (strictly after observing realized y_true).
        Returns `(eps, tau, exceeded_threshold)`.
        Prequential evaluation: threshold tau(t) is computed from prior history BEFORE
        the current observation eps(t) is appended to the rolling window.
        """
        eps = float(abs(float(y_true) - float(y_pred)))
        tau = self._compute_tau()
        exceeded = len(self._window) >= self.cfg.min_obs and eps > tau
        if exceeded:
            self._pending_surprise = True
            self.retrain_triggers += 1

        if np.isfinite(tau) and tau > 0:
            norm_s = min(eps / (tau + 1e-6), self.cfg.surprise_clip)
        else:
            norm_s = 0.0

        # Prequentially append to window for future evaluations
        self._window.append(eps)
        alpha = 0.1
        if self._ewma_mean is None:
            self._ewma_mean = eps
            self._ewma_var = 0.0
        else:
            diff = eps - self._ewma_mean
            self._ewma_mean = (1.0 - alpha) * self._ewma_mean + alpha * eps
            self._ewma_var = (1.0 - alpha) * self._ewma_var + alpha * (diff * diff)

        self._last_eps = eps
        self._last_tau = tau
        self._last_norm_surprise = norm_s
        self.eps_history.append(eps)
        self.tau_history.append(tau if np.isfinite(tau) else np.nan)
        return eps, tau, exceeded

    def record_val_loss_delta(self, val_loss: float) -> Tuple[float, float, bool]:
        """
        Bug-fixed Phase 2 validation-delta surprise (`B3/B5/B6` resolved):
        update lifetime best cleanly after minimum observation warmup without pushing raw val_loss into delta deque.
        Prequentially evaluates tau BEFORE appending delta.
        """
        if self._best_val_loss is None:
            self._best_val_loss = float(val_loss)
            self._last_eps = 0.0
            self._last_tau = float("inf")
            self._last_norm_surprise = 0.0
            self.eps_history.append(0.0)
            self.tau_history.append(np.nan)
            return 0.0, float("inf"), False

        eps = float(abs(float(val_loss) - self._best_val_loss))
        if val_loss < self._best_val_loss:
            self._best_val_loss = float(val_loss)

        tau = self._compute_tau()
        exceeded = len(self._window) >= self.cfg.min_obs and eps > tau
        if exceeded:
            self._pending_surprise = True
            self.retrain_triggers += 1

        self._window.append(eps)
        norm_s = min(eps / (tau + 1e-6), self.cfg.surprise_clip) if (np.isfinite(tau) and tau > 0) else 0.0
        self._last_eps = eps
        self._last_tau = tau
        self._last_norm_surprise = norm_s
        self.eps_history.append(eps)
        self.tau_history.append(tau if np.isfinite(tau) else np.nan)
        return eps, tau, exceeded

    def evaluate_gossip_send(self) -> GateDecision:
        """
        Evaluated on equal-cadence timer (`T_gl`) for ALL modes (eliminating B6 polling bias).
        """
        self.total_gossip_checks += 1
        tau = self._last_tau
        eps = self._last_eps

        if self.cfg.mode == "off":
            self.sent_checks += 1
            self.baseline_sends += 1
            self._pending_surprise = False
            self._suppressed_streak = 0
            dec = GateDecision(True, "baseline_always_send", eps, tau, 0.0, 0)
            self.decision_history.append(1)
            return dec

        if len(self._window) < self.cfg.min_obs:
            self.sent_checks += 1
            self.warmup_sends += 1
            self._pending_surprise = False
            self._suppressed_streak = 0
            dec = GateDecision(True, "warmup_fail_open", eps, tau, self._last_norm_surprise, 0)
            self.decision_history.append(1)
            return dec

        if self._suppressed_streak >= self.cfg.heartbeat:
            self.sent_checks += 1
            self.heartbeat_sends += 1
            self._pending_surprise = False
            self._suppressed_streak = 0
            dec = GateDecision(True, "heartbeat", eps, tau, self._last_norm_surprise, 0)
            self.decision_history.append(1)
            return dec

        if self._pending_surprise or (np.isfinite(tau) and eps > tau):
            self.sent_checks += 1
            self.surprise_sends += 1
            self._pending_surprise = False
            self._suppressed_streak = 0
            dec = GateDecision(True, "surprise_trigger", eps, tau, self._last_norm_surprise, 0)
            self.decision_history.append(1)
            return dec

        # Suppress redundant model broadcast
        self._suppressed_streak += 1
        self.suppressed_checks += 1
        dec = GateDecision(False, "suppressed", eps, tau, self._last_norm_surprise, self._suppressed_streak)
        self.decision_history.append(0)
        return dec

    @property
    def current_norm_surprise(self) -> float:
        return self._last_norm_surprise

