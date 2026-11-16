# Phase 3 Stage 6 Architecture Corrections & System Rectifications

**Thesis:** Decentralized Edge Workload Forecasting using Semantic Gossip Learning and Agentic AI  
**Scope:** $N = 10$ Nodes (Porto Edge Topology, $k=3$, 16 undirected links)  
**Branch:** `stage6-audit-fix-10n`  
**Audit Reference:** `docs/phase3_stage6_audit_v2/STAGE6_INDEPENDENT_AUDIT.md`  
**Date:** November 2026

---

## Executive Summary

Following a rigorous independent research audit of the initial Stage 6 implementation (`e7295860`), eight critical architectural, algorithmic, and measurement defects (designated C1 through C8) were identified. These defects spanned causal evaluation order, physical link contention, communication byte accounting, startup transient biases, and RL value function estimation.

This document records the exact theoretical defects, mathematical formulations, concrete code rectifications, and unit verification tests applied in branch `stage6-audit-fix-10n`.

---

## Detailed Architectural Rectifications (C1 – C8)

### C1. Enforcement of Strict Prequential Evaluation Order in APSM

- **Audit Finding:** In `src/phase3/forecasting/apsm.py`, when a new prediction residual $\epsilon_t = |y_t - \hat{y}_t|$ or validation loss delta was recorded, the residual was appended to the rolling window before computing the dynamic threshold $\tau_t$. Consequently, $\tau_t$ contained the instantaneous residual being tested, violating strict prequential evaluation ($p(y_t \mid y_{<t})$) and slightly softening threshold boundaries.
- **Correction:** The evaluation order was inverted:
  $$\hat{\tau}_t = \text{Quantile}_{q}(\mathcal{W}_{t-1}) \quad \text{evaluated against } \epsilon_t,$$
  $$\mathcal{W}_t = \mathcal{W}_{t-1} \cup \{\epsilon_t\}.$$
- **Files Modified:** `src/phase3/forecasting/apsm.py` (`record_causal_residual` and `record_val_loss_delta`).
- **Verification:** Unit test `tests/phase3/test_stage3_forecasting.py` passed (6/6 tests).

---

### C2. Shared Physical Link Contention across Model, Data, and Telemetry Traffic

- **Audit Finding:** Task offloading and Gossip model synchronization used separate tracking logic. Gossip model exchanges incurred no physical link queuing contention, allowing 56.9 KB model packets to travel without delaying concurrent task offloading or telemetry beacons.
- **Correction:** Unified link queue scheduling was implemented in `TaskOffloadingSim`:
  $$\text{Earliest Start} = \max(t_{\text{now}}, \text{link\_busy\_until}[u, v]),$$
  $$d_{\text{tx}} = \frac{\text{packet\_bytes} \times 8}{R_{uv}},$$
  $$\text{link\_busy\_until}[u, v] = \text{Earliest Start} + d_{\text{tx}}.$$
  All traffic classes—(1) task offload payloads, (2) 32-byte telemetry beacons, (3) 16-byte ACK packets, and (4) 56,920-byte Gossip model broadcasts—now serialize through the identical directed physical link queue.
- **Files Modified:** `src/phase3/env/task_simulator.py` (`_schedule_link_tx`, `schedule_model_packet`).
- **Verification:** Dedicated test `test_shared_physical_link_contention` in `tests/phase3/test_stage6_integration.py` confirmed that an offloaded task submitted during an active model packet transfer is queued and delayed by exactly the model's remaining transmission duration.

---

### C3. Discrete-Event Model Packet Serialization Delay

- **Audit Finding:** In `LiveGossipManager`, model packets were assigned an arbitrary idle-channel latency estimate rather than querying the discrete-event network simulator's channel state.
- **Correction:** The gossip manager now calls `sim.schedule_model_packet(sender_id, receiver_id, packet_bytes, t_now_s)`, reserving actual physical link bandwidth and computing the exact discrete-event delivery timestamp.
- **Files Modified:** `src/phase3/integration/live_gossip_engine.py` (`_broadcast_model_update`).
- **Verification:** Full discrete-event integration test `tests/phase3/test_stage6_integration.py` passing (7/7 tests).

---

### C4. Direct Workload Forecasting Evaluation Metrics

- **Audit Finding:** Forecasting performance in earlier stages was evaluated indirectly via downstream offloading success rate, without directly reporting the Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE) of the continuous workload predictions on the test split.
- **Correction:** Added explicit prequential evaluation tracking in `LiveGossipManager.compute_forecasting_metrics()`:
  $$\text{MSE} = \frac{1}{K \cdot N} \sum_{i=1}^N \sum_{k=1}^K (y_{i, k} - \hat{y}_{i, k})^2,$$
  $$\text{Persistence MSE} = \frac{1}{(K-1) \cdot N} \sum_{i=1}^N \sum_{k=2}^K (y_{i, k} - y_{i, k-1})^2.$$
  Forecasting metrics are logged in result records (`fc_mse`, `fc_rmse`, `fc_mae`, `fc_pers_mse`) and plotted in publication Figure 19.
- **Files Modified:** `src/phase3/integration/live_gossip_engine.py`, `src/phase3/run_stage6_integrated.py`, `src/phase3/plot_stage6_results.py`, `src/phase3/analyze_stage6_results.py`.
- **Verification:** Smoke test verified direct test-window MSE of 0.0368 vs 0.0517 for the persistence baseline (+28.8% error reduction).

---

### C5. Rigorous Byte Accounting and Disaggregation

- **Audit Finding:** Wire header bytes ($32\text{ bytes} \times N_{\text{offloads}}$) were inconsistently combined with application payload bytes, resulting in slight discrepancies between simulator byte counters and the formal `CommunicationLedger`.
- **Correction:** Enforced exact accounting equations:
  $$CB_{\text{control}} = M + T + A + H,$$
  $$P = \text{offload\_data\_bytes} - H,$$
  $$CB_{\text{system}} = CB_{\text{control}} + P = M + T + A + H + P,$$
  where:
  - $M$: Gossip model parameter bytes ($14,224 \times 4 = 56,920\text{ bytes}$ per accepted broadcast).
  - $T$: Periodic routing telemetry beacon bytes ($32\text{ bytes}$ per beacon).
  - $A$: Offload acknowledgment bytes ($16\text{ bytes}$ per offload).
  - $H$: Offload wire header encapsulation bytes ($32\text{ bytes}$ per offload).
  - $P$: Pure task application payload data.
  The discrete-event simulator asserts wire transmission serialization delay across $P + H$.
- **Files Modified:** `src/phase3/env/task_simulator.py`, `src/phase3/integration/live_gossip_engine.py`.
- **Verification:** Unit test `test_stage4_simulator.py` verified exact equality between ledger and simulator counters.

---

### C6. Elimination of Startup Causal Residual Spike

- **Audit Finding:** At the start of simulation (bin 0), the realized workload $y_0$ was compared against a dummy prediction $\hat{y}_0 = 0.0$, generating an artificial residual spike $|\epsilon_0| = y_0$ that biased initial adaptive threshold estimation.
- **Correction:** Prior to the first bin boundary, `last_preds` are initialized by causally running the base LSTM model over the preceding historical lookback window ($L=12$ bins), ensuring pre-warmed initial residual computation.
- **Files Modified:** `src/phase3/integration/live_gossip_engine.py` (`__init__`).
- **Verification:** Integration smoke test confirmed smooth startup thresholds with zero artificial bin-0 spikes.

---

### C7. Unfiltered Gossip Learning Baseline Arm (`e3_madqn_gl`)

- **Audit Finding:** APSM suppression claims previously compared APSM-filtered gossip against a purely theoretical communication volume calculation without running an active, unfiltered Gossip Learning arm that actually transmits every model update across the physical link.
- **Correction:** Introduced a 6th benchmark arm: `e3_madqn_gl`. This arm executes identical collaborative gossip learning with $\text{APSM}$ filtering deactivated (all checks broadcast). This provides empirical measurement of:
  1. The exact physical link congestion caused by unfiltered model exchanges.
  2. The exact empirical bandwidth reduction achieved by APSM ($>50\%$).
  3. Pre-specified non-inferiority testing between filtered and unfiltered gossip learning.
- **Files Modified:** `src/phase3/run_stage6_integrated.py`, `src/phase3/plot_stage6_results.py`, `src/phase3/analyze_stage6_results.py`.
- **Verification:** 6-arm smoke test verified $314.0\text{ MB}$ transmitted by `e3_madqn_gl` versus $149.7\text{ MB}$ by `e3_madqn_apsm` ($52.32\%$ empirical reduction).

---

### C8. Double-DQN Bellman Target Terminal Masking & State Augmentation

- **Audit Finding:**
  1. The Double-DQN target calculation omitted the terminal transition mask $(1 - d_t)$:
     $$y_t = r_t + \gamma Q(s_{t+1}, \arg\max_a Q(s_{t+1}, a)).$$
     For terminal states, future return was not masked.
  2. The self-forecast feature $\hat{y}_i$ (observation dimension 4) was omitted from the `MaskedDuelingQNet` state encoder.
- **Correction:**
  1. The replay buffer and target computation were corrected:
     $$y_t = r_t + (1 - d_t) \gamma Q_{\text{target}}(s_{t+1}, \arg\max_a Q(s_{t+1}, a)).$$
  2. The state encoder was expanded from 8 to 9 input dimensions to incorporate $\text{obs}[:, 4]$ ($\hat{y}_i$), and exact floating-point parity was verified between the PyTorch training network and the fast CPU NumPy rollout runner ($< 2.4 \times 10^{-7}$ absolute difference).
- **Files Modified:** `src/phase3/marl/madqn.py`.
- **Verification:** Unit tests `test_stage5_marl.py` (5/5) and `test_madqn_pytorch_numpy_parity` in `test_stage6_integration.py` passed with zero errors.

---

## Summary of Test Verification Suite

All rectifications were validated by the complete Phase 3 test suite:
- `tests/phase3/test_stage3_forecasting.py`: **6/6 PASS**
- `tests/phase3/test_stage4_simulator.py`: **3/3 PASS**
- `tests/phase3/test_stage5_marl.py`: **5/5 PASS**
- `tests/phase3/test_stage6_integration.py`: **7/7 PASS**
- **Total Phase 3 Suite: 21/21 Unit & Integration Tests Passing Cleanly.**
