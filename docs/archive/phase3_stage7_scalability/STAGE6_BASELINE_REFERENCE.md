# STAGE 6 FROZEN BASELINE REFERENCE (N=10 NODES)

## 1. Milestone & Provenance Overview

This document records the immutable experimental baseline established in **Phase 3 Stage 6** on the 10-node edge topology. It serves as the formal scientific benchmark against which the **Stage 7 Scalability Experiments** ($N=50$ and $N=100$) are evaluated.

- **Milestone Designation:** `stage6_integrated_10n_final_reconciliation`
- **Base Commit Hash:** `3d7b8ec`
- **Canonical Results Artifact:** [`results/phase3/stage6_integrated_10n_reconciled.json`](file:///C:/Users/btpUser/Desktop/Parth_MTP/APSM-MTP/results/phase3/stage6_integrated_10n_reconciled.json)
- **Exact Checkpoint Evaluation Artifact:** [`results/phase3/stage6_10n_marl_eval_exact.json`](file:///C:/Users/btpUser/Desktop/Parth_MTP/APSM-MTP/results/phase3/stage6_10n_marl_eval_exact.json)
- **Controlled Forecasting Diagnostic Artifact:** [`results/phase3/stage6_10n_forecasting_aligned.json`](file:///C:/Users/btpUser/Desktop/Parth_MTP/APSM-MTP/results/phase3/stage6_10n_forecasting_aligned.json)
- **Calibration Artifact:** [`results/phase3/stage6_10n_calibration_train_burst.json`](file:///C:/Users/btpUser/Desktop/Parth_MTP/APSM-MTP/results/phase3/stage6_10n_calibration_train_burst.json)
- **SHA-256 Manifest:** [`results/phase3/stage6_10n_reconciled_manifest.json`](file:///C:/Users/btpUser/Desktop/Parth_MTP/APSM-MTP/results/phase3/stage6_10n_reconciled_manifest.json)
- **Complete Test Log (28/28 passing):** [`results/phase3/stage6_final_reconciliation_test_log.txt`](file:///C:/Users/btpUser/Desktop/Parth_MTP/APSM-MTP/results/phase3/stage6_final_reconciliation_test_log.txt)

---

## 2. Experimental Configuration & Environment

### 2.1 Hardware Environment
- **Host CPU:** 13th Gen Intel(R) Core(TM) i9-13950HX (24 cores, 32 threads)
- **Host RAM:** 31.69 GB DDR5
- **GPU Accelerator:** NVIDIA RTX 1000 Ada Generation Laptop GPU (6,141 MiB GDDR6 VRAM, compute capability 8.9)
- **CUDA Runtime:** 13.2 / PyTorch `2.11.0+cu130`

### 2.2 Workload & Temporal Topology
- **Trace Source:** Alibaba Cluster Trace 2017 (`batch_instance.csv`), top 10 machine IDs by instance density.
- **Bin Duration:** $\Delta t_{\text{bin}} = 30$ seconds.
- **Total Trace Duration:** 1,380 bins ($41,400\text{ s} = 11.5\text{ hours}$).
- **Splits:**
  - Training: $[0, 828)$ (828 bins = 6.9 hours)
  - Validation: $[828, 1035)$ (207 bins = 1.725 hours)
  - Testing: $[1035, 1380)$ (345 bins = 2.875 hours = 10,350 seconds)
- **Test Workload Volume:** Exactly 17,862 discrete tasks; exactly 3,450 time-aligned 1-step workload predictions ($10 \text{ nodes} \times 345 \text{ bins}$).
- **Task Payload Distribution:** Log-uniform $50\text{ KB} \le D_k \le 1000\text{ KB}$.
- **Base Deadline Slack:** $\sigma_0 = 3.0$ seconds.

### 2.3 Edge Topology ($N=10$)
- **Network Construction:** Porto base station coordinates (`assets/porto_cells.parquet`), $k$-edge augmentation ($k=3$), seed 42.
- **Graph Structure:** 10 nodes, 16 undirected edges (32 directed transmission channels). Node degrees range from 3 to 4.
- **Physical Link Parameters:**
  - Propagation RTT: $5\text{ ms} \le \text{RTT}_{ij} \le 40\text{ ms}$.
  - Link Bandwidth: $50\text{ Mbps} \le B_{ij} \le 200\text{ Mbps}$.
- **Link Contention:** Discrete-event packet queue simulator enforcing head-of-line serialization and serialization delay:
  $$d_{\text{tx}} = \frac{8 \cdot \text{bytes}}{B_{ij}}$$

### 2.4 Simulation Operating Regime
- **Simulated Node CPU Capacity:** $C_{\text{cpu}} = 20.0$ capacity units.
- **Deadline Slack Factor:** $\kappa = 2.2$.
- **Local Queue Capacity:** $q_{\max} = 48$ task slots.
- **Operating Regime Characterization:** Selected via a 9-point grid sweep over burst training slice $[380, 488)$ (10,678 tasks). Represents an **empirical design operating regime** balancing high task load with active offloading ($47.35\%$ offload rate, $18.68\%$ baseline drop rate), avoiding both saturation and ceiling effects.

### 2.5 Collaborative Gossip & APSM Semantic Gating
- **Base Forecaster:** EdgeLSTMForecaster (input dim 7, hidden dim 32, 2 layers, ~56,929 bytes serialized shared weights).
- **Gossip Cadence:** Evaluated every 2 bins ($60\text{ seconds}$).
- **Semantic Filter:** APSM $H_1$ causal residual filter tracking running prediction errors $\epsilon_t = |y_t - \hat{y}_t|$ against adaptive quantile threshold $\tau_t$.
- **Gating Invariants:** Warmup fail-open duration = 12 bins (360s); maximum heartbeat interval = 10 evaluation checks (600s).

---

## 3. Reference Benchmark Results across 5 Seeds

All metrics are reported as Mean $\pm$ Sample Standard Deviation ($ddof=1$) over seeds `[11, 22, 33, 44, 55]`:

| Benchmark Arm | Task Success Rate (%) | Effective Latency (s) | Deadline Miss Rate (%) | Offload Rate (%) | Forecast MSE | Model Transfer Volume (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0: LocalOnly** | $78.38 \pm 0.00$ | $85.37 \pm 0.00$ | $21.62 \pm 0.00$ | $0.00 \pm 0.00$ | N/A | $0.00 \pm 0.00$ |
| **E1: ReactiveGreedy** | $84.86 \pm 0.00$ | $76.31 \pm 0.00$ | $15.14 \pm 0.00$ | $40.62 \pm 0.00$ | N/A | $0.00 \pm 0.00$ |
| **E2: Reactive MADQN** | $83.83 \pm 0.53$ | $79.07 \pm 0.81$ | $16.17 \pm 0.53$ | $28.16 \pm 0.99$ | N/A | $0.00 \pm 0.00$ |
| **E3-Local: Local MADQN** | $83.57 \pm 1.05$ | $79.26 \pm 0.88$ | $16.43 \pm 1.05$ | $28.13 \pm 1.76$ | $0.037063 \pm 0.000142$ | $0.00 \pm 0.00$ |
| **E3-GL: Unfiltered GL** | $84.07 \pm 0.48$ | $78.81 \pm 0.48$ | $15.93 \pm 0.48$ | $30.35 \pm 0.82$ | $0.035884 \pm 0.000064$ | $313.29 \pm 0.00$ |
| **E3-APSM: Filtered GL** | $84.15 \pm 0.49$ | $78.67 \pm 0.62$ | $15.85 \pm 0.49$ | $30.36 \pm 0.97$ | $0.036000 \pm 0.000029$ | $149.86 \pm 0.86$ |

### Primary Findings & Baseline Invariants:
1. **Communication Reduction:** APSM achieves a **$52.16\%$ reduction in model transfer bytes** ($313.29\text{ MB} \to 149.86\text{ MB}$) and a **$52.23\%$ reduction in model packets** ($5,504 \to 2,629$ packets).
2. **Collaborative Gain Retention:** APSM retains **$90.16\%$ of the collaborative forecasting accuracy gain** over isolated local models:
   $$\mathcal{R} = \frac{0.037063 - 0.036000}{0.037063 - 0.035884} \times 100 = 90.16\%$$
3. **Exploratory Non-Inferiority Acceptance:** Under margin $\delta = 1.00\%$ pts, E3-APSM is strictly non-inferior to unfiltered Gossip Learning ($p_{\text{NI}} = 1.6 \times 10^{-5}$, observed difference $+0.08\% \pm 0.10\%$).

---

## 4. Known Scope Boundaries & Baseline Caveats

1. **Topological Scale:** Validated strictly at $N=10$ nodes.
2. **Task Success Effect Size:** The proactive task offloading advantage of E3-APSM over reactive E2 ($84.15\%$ vs $83.83\%$, $+0.32\%$) is modest. In small 10-node meshes with 30s binning, reactive feedback is frequently sufficient to divert traffic.
3. **Homogeneous Model Architectures:** All edge nodes utilize identical parameter-shared neural network topologies.
4. **Static Connectivity:** The 16 physical links are static and do not simulate wireless link drops or topology churn.
