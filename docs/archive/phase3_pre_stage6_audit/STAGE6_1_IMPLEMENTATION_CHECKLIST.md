# Stage 6.1 Implementation Checklist: Integrated N=6 Benchmark

**Repository:** `APSM-MTP`  
**Branch:** `phase3-continuation`  
**Commit:** `02ace48`  
**Date:** November 13, 2026  
**Auditor:** Senior Distributed Systems Researcher, Multi-Agent RL Researcher, Independent Code Auditor  

---

## 1. Scope and Safeguards

### Strict Boundaries:
- **Phase 2 Baseline**: `62124eb` and historical directories (`results/Phase2_Full/*`, `src/gossiplearning/*`) remain **STRICTLY FROZEN**.
- **Stage 5 Records**: `results/phase3/stage5_marl_gate_6n.json` remains untouched as a historical record.
- **New Branch / Isolation**: All Stage 6.1 code must reside in new runners or explicit feature branches.

---

## 2. Component Implementation Checklist

### Phase 1: Environment & Engine Bridge
- [ ] **1.1 Live 30s Gossip Callback Hook**:
  - Connect `TaskOffloadingSim.run_marl_episode(step_fn, bin_callback=...)` to an online gossip callback.
  - At every $30\text{s}$ bin boundary ($t_{\text{now}} = 30, 60, 90, \dots$):
    - Ingest past 30s actual workload $y_i(t)$.
    - Compute prediction residual $H_{1, i}(t) = |y_i(t) - \hat{y}_i(t)|$.
    - Check APSM surprise threshold $\tau_i(t) = \mu_i(t) + k \cdot \sigma_i(t)$ and heartbeat ($10\text{ bins}$).
- [ ] **1.2 Dynamic Model Exchange**:
  - If $H_1 > \tau$, serialize LSTM weight tensor ($56{,}920\text{ bytes}$).
  - Incur transmission delay over simulated wireless link.
  - Increment `model_packets += 2`, `model_bytes += 2 * 56920`.
  - Perform `age_weighted_merge` at neighbor node.
  - If $H_1 \le \tau$, increment `telemetry_suppressed += 1`, transmit $0\text{ bytes}$.
- [ ] **1.3 Next-Step Forecast Generation**:
  - Local LSTM executes forward pass on causal history window ($L=12$ bins) to generate $\hat{y}_i(t+1)$.
  - Store $\hat{y}_i(t+1)$ in node's local telemetry cache for inclusion in next $15\text{s}$ operational beacon.

### Phase 2: Observation & MARL Policy Ingestion
- [ ] **2.1 Replace Precomputed Matrix Lookups**:
  - Remove all dependencies on `cache_fast_lstm_s*.npz` in `WireOracleFastSim.build_observation`.
  - Feed the live scalar $\hat{y}_j(t+1)$ stored in `neighbor_cache[node_id][peer_id].y_hat`.
- [ ] **2.2 Action Masking Verification**:
  - Verify action mask strictly enforces:
    - Action 0: Local CPU queue not full ($Q_i < Q_{\text{max}}$).
    - Actions 1..K: Physical link up, peer queue not full ($Q_j < Q_{\text{max}}$), peer memory available, and `already_offloaded == False`.

### Phase 3: Communication Accounting & Telemetry Separation
- [ ] **3.1 Separate Logical Counters**:
  - Verify `telemetry_bytes` only accumulates 32-byte UDP status beacons.
  - Verify `model_bytes` only accumulates 56,920-byte LSTM weight broadcasts.
  - Verify `ack_bytes` only accumulates 16-byte task completion ACKs.
  - Verify `offload_data_bytes` accumulates application payload bytes ($50\text{ KB} \text{--} 1\text{ MB}$).
- [ ] **3.2 Unified Total Byte Computation**:
  - Compute $CB_{\text{total}} = \text{telemetry\_bytes} + \text{ack\_bytes} + \text{model\_bytes} + \text{header\_bytes}$.

### Phase 4: Benchmark Runner Script
- [ ] **4.1 Create Integrated Runner**:
  - File: `src/phase3/run_stage6_integrated.py`.
  - Supports the 5 benchmark arms:
    1. `b0_local`
    2. `e1_greedy`
    3. `e2_madqn` (Reactive Q-learning, no forecast)
    4. `e3_madqn_local` (Forecast Q-learning, local LSTM only)
    5. `e3_madqn_apsm` (Integrated System, live Gossip + APSM-H1)
  - Seeds: `[11, 22, 33, 44, 55]` (5 seeds for formal statistical power).
  - Train: 48 episodes, evaluate on held-out test split $[1035, 1380)$.

### Phase 5: Verification & Gate Pass Criteria
- [ ] **5.1 Invariant Gate**:
  - 100% task conservation verified across all 5 arms:
    $$\text{total\_tasks} \equiv \text{completed} + \text{drop\_adm} + \text{drop\_dead} + \text{drop\_link}$$
- [ ] **5.2 Gossip Gate**:
  - APSM-H1 achieves $> 50\%$ reduction in model broadcast rounds in `e3_madqn_apsm`.
- [ ] **5.3 Performance Gate**:
  - Arm `e3_madqn_apsm` achieves higher Task Success Rate than `e2_madqn` with statistical significance ($p < 0.05$).
  - Effective latency of `e3_madqn_apsm` is lower than or equal to `e2_madqn`.
- [ ] **5.4 Output Persistence**:
  - Serialize full results to `results/phase3/stage6_1_integrated_6n.json`.

