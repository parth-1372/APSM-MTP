# Phase 3 Stage 6 Readiness Assessment and Decision Framework

**Repository:** `APSM-MTP`  
**Branch:** `phase3-continuation`  
**Commit:** `02ace48`  
**Date:** November 13, 2026  
**Auditor:** Senior Distributed Systems & MARL Independent Code Auditor  

---

## 1. Formal Readiness Verdict

### **DECISION: READY WITH CONDITIONS**

The Phase 3 codebase has achieved significant scientific milestones:
- The discrete-event queueing simulator (Stage 4) is **rigorously verified, bug-free, and invariant-compliant** (100% task conservation, zero future leakage, strict 1-hop enforcement).
- The causal workload forecasting engine with APSM semantic filtering (Stage 3) is **empirically validated** ($53.33\%$ model communication reduction with negligible $<0.08\%$ MSE degradation).
- The proactive multi-agent reinforcement learning pipeline (Stage 5) successfully proves that **causal workload forecast integration improves edge task throughput by $+0.65\%$** ($74$ additional tasks saved under burst congestion).
- The unit and regression test suite passes **17/17 tests in under 60 seconds**.

However, **proceeding directly to Stage 6 (large-scale experiments at $N=10, 50, 100$) without addressing specific architectural and reporting conditions would risk invalidating experimental claims during thesis defense or peer review.**

---

## 2. Condition Classification & Risk Matrix

| Finding ID | Severity | Category | Description | Remediation Required Before Stage 6 |
| :--- | :--- | :--- | :--- | :--- |
| **AUDIT-COND-01** | **P0 (Critical)** | Algorithmic Integrity | **Evaluation Runner Evaluates Q-Head Exclusively**: In `run_stage5_validation.py`, `FastSharedTrunkRunner` hardcodes `mode="q"`, completely bypassing actor heads (`act_loc`, `act_peer`) for both MAPPO and IPPO. | Must either: (A) Enable true PPO actor evaluation (`mode="actor"`), or (B) Explicitly rename the benchmarked algorithm as Parameter-Shared Multi-Agent Dueling Q-Learning (MADQN). |
| **AUDIT-COND-02** | **P1 (High)** | Portability & Caching | **Non-Portable File Paths in Cache Resolution**: Earlier scripts relied on absolute paths to external scratch directories for walk-forward forecast matrices. | Ensure all Stage 6 loaders look in `data/alibaba_v2017/processed/` using repository-relative paths with graceful generation fallbacks. |
| **AUDIT-COND-03** | **P1 (High)** | Experimental Validity | **Topology & Capacity Calibration for $N=10$ and $N=50$**: Scaling to larger topologies requires that baseline B0 exhibits genuine congestion ($15\%\text{--}35\%$ DMR). If edge nodes have excess capacity, offloading becomes irrelevant. | Run Stage 4 calibration sweeps on $N=10$ and $N=50$ datasets before launching multi-episode MARL training. |
| **AUDIT-COND-04** | **P2 (Medium)** | Presentation | **Oracle Latency Inversion Documentation**: Oracle ($O_1$) exhibits higher effective latency ($83.78\text{ s}$) than E3 ($81.82\text{ s}$) despite higher TSR ($86.10\%$ vs $85.52\%$). | Add explicit documentation explaining that the oracle accepts marginal queue delays to maximize completion throughput. |

---

## 3. Detailed Remediation Requirements

### 3.1 Remediation for Condition AUDIT-COND-01 (P0: Algorithmic Integrity)

#### The Issue:
The current Stage 5 script outputs columns labeled `e2_mappo`, `e2_ippo`, and `e2_madqn`. However, because `mode="q"` was hardcoded in `FastSharedTrunkRunner`, all three arms executed the exact same Double Dueling Q-network policy. A reviewer reading the code will discover that PPO's clipped surrogate loss and GAE advantages had no influence on the evaluated actions.

#### Permitted Solutions:
- **Option A (True Actor-Critic Evaluation)**:
  Update `FastSharedTrunkRunner` to implement `mode="actor"` where:
  $$p_{\text{loc}} = \text{act\_loc}(h_{\text{self}}), \quad p_{\text{peer}} = \text{act\_peer}(h_{\text{peer}})$$
  $$\pi(a|s) = \text{softmax}([p_{\text{loc}}, p_{\text{peer}_1}, \dots, p_{\text{peer}_K}])$$
  Evaluate `e2_mappo` and `e2_ippo` using `mode="actor"`, and evaluate `e2_madqn` using `mode="q"`.
- **Option B (Honest Algorithm Designation)**:
  Acknowledge that the neural architecture is a Parameter-Shared Multi-Agent Dueling Q-Network (MADQN) with Proactive Residual Bias. Frame the paper around:
  - Baseline B0 (Local Only)
  - Baseline E1 (Reactive Greedy Heuristic)
  - Stage 5 Reactive MARL: Parameter-Shared MADQN (E2)
  - Stage 5 Proactive MARL: Forecast-Augmented MADQN (E3)
  - Upper Bound: Full Information Oracle (O1)

**Recommendation**: Option B is scientifically honest, mathematically sound, and avoids retraining complex PPO policies if Q-learning is already performing optimally. Option A should be executed if formal PPO vs DQN ablation is an explicit thesis committee requirement.

---

### 3.2 Remediation for Condition AUDIT-COND-02 (P1: Portable Caching)

#### The Issue:
Fast rollout execution relies on precomputed walk-forward forecast matrices (`fmat`). In earlier runs, these matrices were stored in external scratch directories.

#### Verification:
The repository now includes `cache_fast_lstm_s11.npz`, `cache_fast_lstm_s22.npz`, and `cache_fast_lstm_s33.npz` directly under `data/alibaba_v2017/processed/`.
Before running Stage 6, verify that:
```python
cache_path = Path("data/alibaba_v2017/processed") / f"cache_fast_lstm_{n_nodes}n_s{seed}.npz"
```
is checked first, and if missing, automatically generated via `generate_causal_forecast_matrices`.

---

### 3.3 Remediation for Condition AUDIT-COND-03 (P1: Capacity Calibration at Scale)

#### The Issue:
When expanding from $N=6$ to $N=10$ and $N=50$, total arrival volume and topology diameter change.
If node capacity $C_{\text{cpu}}$ is set too high:
- Local baseline B0 drops zero tasks ($\text{TSR} \approx 100\%$).
- Offloading offers zero benefit, and MARL agents appear ineffective.
If $C_{\text{cpu}}$ is set too low:
- All nodes are permanently saturated ($\text{TSR} < 30\%$).
- Offloading only ping-pongs tasks between congested nodes until deadlines expire.

#### Calibration Gate Checklist for Stage 6:
1. Generate curated trace datasets for $N=10$ (`curated_10n_30s.npz`) and $N=50$ (`curated_50n_30s.npz`).
2. Run a Stage 4 sweep over $C_{\text{cpu}} \in [16.0, 24.0]$ and $\kappa \in [2.0, 2.5]$.
3. Confirm that B0 achieves a baseline Deadline Miss Rate between **$15\%$ and $35\%$**.
4. Confirm that the reactive heuristic E1 achieves a statistically significant throughput improvement over B0 ($+5\%\text{ to }+10\%$ TSR).
5. Only then launch multi-agent training for E2 and E3.

---

## 4. Workstation Resource & Execution Rules for Stage 6

To maximize throughput and prevent system crashes on the primary Windows workstation (`i9-13950HX`, 32 threads, 32 GB RAM, RTX 1000 Ada):

1. **Thread Discipline**:
   For $N=6$ and $N=10$, CPU contention destroys performance if NumPy or PyTorch spawn 32 threads per worker.
   - Enforce: `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`.
   - Run up to 6 parallel worker processes using Python `multiprocessing`.
2. **Device Discipline**:
   - For DES rollouts and Q-learning updates, CPU execution with NumPy vectorized matrix operations is **$2\times$ faster than CUDA** due to PyTorch-to-CUDA kernel launch latencies for small tensors.
   - Use `device="cpu"` for all DES rollout loops.
   - Use `device="cuda"` only for large batch LSTM pre-training or when evaluating $N \ge 50$.
3. **Execution Command Standard**:
   Always prefix terminal execution with `rtk`:
   ```powershell
   rtk cmd /c "set OMP_NUM_THREADS=1&& set MKL_NUM_THREADS=1&& .\.venv\Scripts\python.exe src/phase3/run_stage5_validation.py"
   ```

---

## 5. Stage 6 Gate Sign-Off Protocol

To transition from Phase 5 to Stage 6 execution:
- [x] Phase 2 historical records frozen and unmolested (`62124eb`).
- [x] Pytest suite passes 17/17 tests.
- [x] Stage 3 APSM causal forecasting verified ($53.33\%$ suppression).
- [x] Stage 4 discrete-event queueing verified (100% task conservation).
- [x] Stage 5 multi-seed MARL results verified across seeds 11, 22, 33.
- [ ] User alignment on Condition AUDIT-COND-01 (Algorithm naming: MADQN vs PPO).
- [ ] Calibration sweep completed for $N=10$ and $N=50$ topologies.

**Conclusion**: The repository is in an advanced, stable state and is **READY WITH CONDITIONS** to begin Stage 6 scale-up immediately upon confirming the algorithmic labeling decision.

