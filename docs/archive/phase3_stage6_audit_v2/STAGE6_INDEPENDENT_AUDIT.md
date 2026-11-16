# Phase 3 Stage 6: Independent Research, Implementation & Architecture Audit

**Repository:** `APSM-MTP`  
**Branch:** `stage6-audit-fix-10n` (forked from `e7295860bdfde9516c6a3f8e80235cb01c9ccda6`)  
**Date:** November 14, 2026  
**Auditor:** Senior Distributed Systems Researcher, Reinforcement Learning Engineer, Independent Code Auditor  
**Scope:** Strictly $N = 10$ Nodes (Porto Partial Mesh, $k=3$, 16 undirected edges)  

---

## 1. Executive Audit Summary

A rigorous, line-by-line inspection of the Stage 6 implementation (`e729586`), historical Phase 2 codebase (`62124eb`), and experimental records was conducted across:
- `src/phase3/config.py`
- `src/phase3/data/alibaba_loader.py`
- `src/phase3/data/topology.py`
- `src/phase3/forecasting/models.py`
- `src/phase3/forecasting/apsm.py`
- `src/phase3/forecasting/gossip_runner.py`
- `src/phase3/integration/live_gossip_engine.py`
- `src/phase3/env/task_simulator.py`
- `src/phase3/env/baselines.py`
- `src/phase3/marl/madqn.py`
- `src/phase3/run_stage6_integrated.py`
- `src/phase3/analyze_stage6_results.py`
- `src/phase3/plot_stage6_results.py`
- `tests/phase3/test_stage6_integration.py`
- `results/phase3/stage6_integrated_10n.json`

The audit reveals that while the high-level architecture achieved substantial progress (successful convergence of MADQN, fast NumPy vectorization, GPU acceleration, and 100% task conservation), **several critical defects, experimental shortcuts, and documentation discrepancies undermine the scientific defensibility of the existing Stage 6 milestone**.

---

## 2. Independent Audit Matrix

| Audit Item | Classification | Source File & Function | Summary of Defect / Issue | Impact | Required Correction | Test to Prove Fix |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **C1. Shared Link Scheduling** | **CONFIRMED DEFECT** | `task_simulator.py` (`_dispatch_task`, `_emit_telemetry_check`, `_schedule_ack`), `live_gossip_engine.py` (`on_bin_boundary`) | Model gossip, telemetry beacons, and ACKs calculated serialization delays independently assuming an idle channel, never contending with task offloads for the physical link `link_free_at[i, j]`. | High: Model traffic did not compete for bandwidth with tasks despite claims of a shared channel. | Implement unified `PhysicalLinkScheduler` where all 4 traffic classes reserve transmission time on the directed link FIFO queue. | Unit test verifying that a task scheduled while a 56.9 KB model packet is transmitting experiences positive link queueing delay. |
| **C2. Communication Byte Double Counting** | **CONFIRMED DEFECT** | `task_simulator.py` (line 648, 655), `live_gossip_engine.py` (line 58, 67) | `offload_data_bytes` added `payload_b + task_header_bytes`. Meanwhile, `header_bytes` was counted separately in `cb_control_bytes`. In `cb_system_bytes`, header bytes were added twice. In addition, task serialization delay `d_tx` omitted header bytes from transmission time. | Medium: Distorts system byte totals and under-estimates task wire transmission delay. | Canonical accounting: `offload_data_bytes` = payloads only; `header_bytes` counted once in control plane; `d_tx` computes serialization over `payload + header`. | Hand-calculated packet size reconciliation test asserting `cb_system == cb_control + payload_bytes`. |
| **C3. APSM-H1 Threshold Formula** | **CONFIRMED DOCUMENTATION ERROR** | `apsm.py` (`_compute_tau`), `STAGE6_INTEGRATION_REPORT_10N.md` | Code implements $\tau = K \cdot \sigma$ (matching Phase 2 `threshold_sweep.py`), but documentation claimed $\tau = \mu + K\sigma$. | Medium: Inconsistent reporting between code and research report. | Align documentation and equations to confirm the intended Phase 2 standard deviation multiplier $\tau = K\sigma$. | Equation verification test against Phase 2 baseline. |
| **C3. APSM Prequential Ordering** | **CONFIRMED DEFECT** | `apsm.py` (`record_causal_residual`) | Current residual $\epsilon$ was appended to `_window` *before* calling `_compute_tau()`. An extreme residual thus inflated its own threshold. | High: Violates strict prequential statistical gating principles. | Compute threshold $\tau(t)$ on past window *before* appending current residual $\epsilon$. | Test confirming outlier does not inflate its own contemporaneous detection threshold. |
| **C3. Startup Prediction Residual** | **CONFIRMED DEFECT** | `live_gossip_engine.py` (`last_preds`) | `last_preds` initialized to zeros. At $t=30$s, residual was computed as $|y_0 - 0.0| = y_0$, treating raw load as error. | Medium: Artificially injects false surprise at bin 1. | Generate causal initial prediction using lookback window at $t=0$ so bin 1 evaluates genuine forecast error. | Test asserting `last_preds[i] > 0` before first bin closure. |
| **C3. Local-Only Suppression Misrepresentation** | **CONFIRMED DOCUMENTATION MISREPRESENTATION** | `live_gossip_engine.py` (lines 249-253) | LocalOnly cadence slots were counted as "suppressed checks" and reported as "100% APSM suppression" even though APSM was not active. | Medium: Misleading metric table. | Report suppression as `N/A` (or 0 checks) for arms lacking a Gossip channel. | Table generation test verifying `N/A` for B0, E1, E2, E3. |
| **C4. Forecast Checkpoint Warmup Leakage** | **CONFIRMED DEFECT** | `run_stage6_integrated.py` (lines 145-165, 260-310) | Validation checkpoints were eligible before warmup ($ep < 36$). On Seed 55, Episode 12 was chosen as "best", then evaluated on test with `fc_res` forcibly enabled even though it was never trained. | High: Invalidates Seed 55 forecast-augmented evaluation. | Restrict best-checkpoint selection to $ep \ge \text{warmup}$ for forecast arms; save branch activation in checkpoint metadata. | Test rejecting checkpoint evaluation if checkpoint was saved with branch inactive. |
| **C5. MADQN Terminal State Done Flag** | **CONFIRMED DEFECT** | `run_stage6_integrated.py`, `madqn.py` (`_ReplayBuffer`) | Replay buffer lacked a `done` flag. Terminal transitions looped to self with discount $\gamma = 0.40$. | Medium: DQN bootstraps across episode boundaries. | Add `done` array to replay buffer and set $d=1.0$ on final task transitions: $y = r + \gamma(1-d)Q'$. | Test asserting $Q$-target for terminal transition equals $r$ exactly. |
| **C6. CTDE / MAPPO Architecture Designation** | **CONFIRMED DOCUMENTATION ERROR** | `run_stage6_integrated.py`, `STAGE6_INTEGRATION_REPORT_10N.md` | Code passes `gstate` to callback but does not use it. There is no centralized critic or PPO actor. | Medium: Misleading algorithmic claims. | Accurately designate as Parameter-Shared Double Dueling MADQN with local observations and decentralized execution. | Architecture audit documentation. |
| **C7. Self vs Peer Forecast Feature Utilization** | **CONFIRMED DEFECT / INCONSISTENCY** | `madqn.py` (`MaskedDuelingQNet`, `_NumpyDuelingQRunner`), `TaskOffloadingSim` | Self forecast $\hat{y}_i$ (`obs[4]`) is computed by the simulator but completely omitted from `s_in` in the Q-network. The policy only uses peer forecasts. | High: Agent cannot proactively anticipate its own incoming burst. | Feed self forecast $\hat{y}_i$ into the self encoder `s_in`, and verify parity between PyTorch and NumPy runners. | PyTorch vs NumPy output parity test with numerical tolerance $< 10^{-5}$. |
| **C8. Age-Weighted Merge Semantics** | **DESIGN CHOICE THAT NEEDS JUSTIFICATION** | `models.py` (`age_weighted_merge`), `live_gossip_engine.py` | `model_age` increments with gradient steps. Higher age gives higher weight, reflecting accumulated sample experience, not staleness. | Low: Potential semantic confusion if age is mistaken for staleness. | Formally document `model_age` as accumulated training experience/maturity, matching FedAvg sample weighting. | Unit test verifying that higher experience weights peer models proportionally. |
| **D1. Weak Temporal Causality Test** | **CONFIRMED DEFECT** | `test_stage6_integration.py` (`test_strict_temporal_causality`) | Test used a dummy policy returning 0 and perturbed targets outside the simulation window. | Medium: Test gave false confidence without validating learned policy causality. | Implement strict in-window counterfactual test using real policy: assert bit-for-bit identical actions prior to cutoff. | Counterfactual test with in-window cutoff bin perturbation. |
| **E1. Calibration Data Leakage** | **CONFIRMED DEFECT** | `scratch/calibrate_10n.py`, `STAGE6_CALIBRATION_RESULTS_10N.md` | The 9-point grid sweep ($C_{\text{cpu}}$ vs $\kappa$) was evaluated directly on the held-out test split. | High: Violates foundational machine learning protocol. | Re-run calibration sweep strictly on training/validation split `[8, 828)` or `[828, 1035)`. | Calibration report verifying zero test split contamination. |
| **E2. Missing Unfiltered Gossip Arm** | **CONFIRMED DEFECT** | `run_stage6_integrated.py`, `plot_stage6_results.py` | Benchmark only evaluated 5 arms; unfiltered Gossip was estimated via theoretical multiplication rather than measured. | High: Inability to prove non-inferiority or quantify actual APSM savings vs a live baseline. | Implement and execute 6th arm: `e3_madqn_gl` (Unfiltered Gossip Learning baseline). | 6-arm benchmark verification test. |
| **E3. Direct Forecasting Evaluation Missing** | **CONFIRMED DEFECT** | `run_stage6_integrated.py` | Stage 6 only recorded downstream task TSR/DMR, omitting test-window MSE, RMSE, and MAE. | High: Cannot prove whether Gossip Learning or APSM improved forecasting accuracy. | Measure and record test-window MSE, RMSE, and MAE for E3, E3-GL, E3-APSM, and persistence baseline. | Forecasting metric verification test. |
| **E4. Deterministic Baseline Pseudo-Replication** | **CONFIRMED DEFECT** | `run_stage6_integrated.py`, `analyze_stage6_results.py` | B0 and E1 evaluated the exact same task table 5 times, producing identical numbers, which were analyzed as independent random samples ($t=\infty$). | Medium: Invalidates statistical hypothesis tests. | Explicitly distinguish deterministic baselines from stochastic MARL training runs; state $N_{\text{scenarios}} = 1$ for baselines. | Statistical reporting update. |

---

## 3. Detailed Audit Analyses

### 3.1 Network Contention & Channel Sharing (Issue C1)
In `src/phase3/env/task_simulator.py`:
- `_dispatch_task` computed transmission serialization and arrival time using `self.link_free_at[i, peer_id]`.
- However, `_emit_telemetry_check` (lines 431-446) computed:
  $$\Delta t_{\text{tel}} = 0.5 \cdot \text{RTT}_{ij} + \frac{8 \times 32}{\text{BW}_{ij}}$$
  and immediately enqueued the beacon into `_beacon_heap` without checking or updating `link_free_at[i, peer_id]`.
- Similarly, `_schedule_ack` (lines 396-401) and `LiveGossipManager.on_bin_boundary` (lines 233-247) computed link delays independently assuming an idle channel.
- **Consequence:** If node $i$ broadcasted a $56.9\text{ KB}$ model packet to peer $j$ taking $4.5\text{ ms}$, an offloaded task dispatched $1\text{ ms}$ later experienced $0\text{ ms}$ of queueing delay behind the model packet. The claim that model traffic competed with task traffic for wireless bandwidth was an unverified assumption not reflected in the code.
- **Resolution:** A unified `PhysicalLinkScheduler` must manage the directed FIFO queue for link $(i, j)$. When any packet (model, beacon, task, or ACK) is scheduled, it must start at $\max(t_{\text{now}}, \text{link\_free\_at}[i, j])$, serialized by $\frac{8 \times \text{bytes}}{\text{BW}_{ij}}$, and advance $\text{link\_free\_at}[i, j]$.

### 3.2 Double Counting in Communication Accounting (Issue C2)
In `src/phase3/env/task_simulator.py` line 655:
```python
self.offload_data_bytes += payload_b + self.sim.task_header_bytes
```
In `src/phase3/integration/live_gossip_engine.py` lines 58-69:
```python
self.header_bytes = sim.offload_count * sim.sim.task_header_bytes
self.cb_control_bytes = self.model_bytes + self.telemetry_bytes + self.ack_bytes + self.header_bytes
self.cb_system_bytes = self.cb_control_bytes + self.offload_data_bytes
```
`self.cb_system_bytes` sums `cb_control_bytes` (which includes `header_bytes`) and `offload_data_bytes` (which *already* includes `task_header_bytes`). Thus, task headers ($32\text{ bytes} \times N_{\text{offloads}}$) were double-counted in total system bandwidth. Furthermore, `d_tx` in line 648 only serialized `payload_b`, ignoring header transmission time.
- **Resolution:** Separate cleanly:
  - `offload_data_bytes` = sum of payload bytes only.
  - `header_bytes` = $N_{\text{offloads}} \times 32\text{ bytes}$.
  - $CB_{\text{system}} = CB_{\text{control}} + CB_{\text{data}}$.
  - Physical wire serialization time: $d_{\text{tx}} = \frac{8 \times (\text{payload} + 32)}{\text{BW}}$.

### 3.3 APSM-H1 Threshold Mathematics & Prequential Rigor (Issue C3)
In `src/phase3/forecasting/apsm.py` line 81:
```python
eps = float(abs(float(y_true) - float(y_pred)))
self._window.append(eps)  # <-- Appended BEFORE calculating threshold!
tau = self._compute_tau() # <-- arr includes current eps!
exceeded = len(self._window) >= self.cfg.min_obs and eps > tau
```
In statistical change detection (e.g. cumulative sum / Shewhart control charts), testing observation $x_t$ against a threshold derived from a window containing $x_t$ is a defect: an extreme spike inflates $\sigma$ simultaneously, suppressing true anomalies.
- **Resolution:** Prequential ordering:
  1. $\tau(t) = \text{compute\_tau}(\text{window}_{t-1})$
  2. $\text{exceeded} = (\epsilon(t) > \tau(t))$
  3. $\text{window}_t = \text{append}(\text{window}_{t-1}, \epsilon(t))$.

### 3.4 Checkpoint Selection Warmup Defect (Issue C4)
In `run_stage6_integrated.py`:
- Warmup was set to 36 episodes. Episodes 1..35 trained the base Q-network with `fc_res.fc_active = False`.
- Validation checks evaluated every 4 episodes ($4, 8, 12, \dots, 48$).
- On Seed 55, Episode 12 produced the highest validation reward ($-0.3549$) among all evaluated checkpoints.
- The runner restored weights from Episode 12 and then forced `fc_res.fc_active = True` during final evaluation on the test split.
- **Consequence:** The proactive forecast branch was evaluated using initialized, un-updated random parameters.
- **Resolution:** Checkpoint selection for forecast-augmented arms must be restricted to $ep \ge \text{warmup}$.

### 3.5 Missing Unfiltered Gossip Baseline (Issue E2)
In the previous report, the comparison between APSM-filtered Gossip and unfiltered Gossip was presented as:
"APSM suppressed 52.3% of model broadcasts compared to unfiltered Gossip Learning."
However, this baseline was never actually simulated. The baseline bytes were calculated by multiplying $1,720 \text{ checks} \times 3.2 \text{ degree} \times 56.9\text{ KB}$.
Without running `E3-GL` (Unfiltered Gossip Learning), one cannot demonstrate whether APSM filtering preserved model accuracy or if unfiltered sharing provided superior forecasting and offloading performance.
- **Resolution:** Add `e3_madqn_gl` as the 6th benchmark arm, executing identical 48-episode training and test rollouts where Gossip broadcasts occur unconditionally on every 60-second cadence.
