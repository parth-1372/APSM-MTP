# Phase 3 Stage 6 Rigorous Statistical Analysis & Evidence Reconciliation

**Topology:** 10 Nodes (Porto partial mesh, $k=3$, 16 undirected edges)
**Seeds:** [11, 22, 33, 44, 55] ($M = 5$ policy training seeds, $\text{df} = 4$)
**Test Set:** Alibaba 2017 Trace (Test Split: 17,862 real tasks, 10,350 slots, Bins [1035, 1380), duration = 2.875 hours)
**Exact Student's $t$ Critical Values:** Two-sided $t_{4, 0.025} = 2.7764$, One-sided $t_{4, 0.05} = 2.1318$

> **Methodological Clarification on Sample Size & Workload Horizon:** The 5 evaluation seeds represent independent stochastic policy learning runs (varying initial weights, minibatch sampling, and exploration trajectories) evaluated on a single fixed macro trace trajectory ($N_{\text{scenarios}} = 1$). They do not represent five independent multi-day cluster traces. For deterministic baselines B0 (LocalOnly) and E1 (ReactiveGreedy), decisions are fixed functions of arrivals and deterministic channel state; sample variance across seeds is identically 0.00% by construction.

## 1. Primary Benchmark Summary Table (Mean ± Sample Std across 5 Seeds)

| Arm Designation | TSR (%) | DMR (%) | Effective Latency (s) | P95 Latency (s) | Offload Rate (%) | Jain Fairness | Control Overhead (Bytes) | Check Suppression | Byte Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0: LocalOnly** | 78.38 ± 0.00% | 21.62 ± 0.00% | 85.37s | 138.77s | 0.0% | 0.9713 | 0 B | N/A | N/A |
| **E1: ReactiveGreedy** | 84.86 ± 0.00% | 15.14 ± 0.00% | 76.31s | 145.97s | 40.6% | 0.9941 | 1,054,800 B | N/A | N/A |
| **E2: Reactive MADQN** | 83.83 ± 0.53% | 16.17 ± 0.53% | 79.07s | 157.18s | 28.2% | 0.9904 | 948,038 B | N/A | N/A |
| **E3: Local Forecast MADQN** | 83.59 ± 1.00% | 16.41 ± 1.00% | 79.22s | 154.15s | 28.1% | 0.9868 | 947,904 B | N/A | N/A |
| **E3-GL: Unfiltered Gossip** | 84.15 ± 0.45% | 15.85 ± 0.45% | 78.81s | 154.41s | 30.4% | 0.9883 | 314,255,235 B | 0.0% | 0.0% |
| **E3-APSM: Integrated Closed-Loop** | 84.04 ± 0.52% | 15.96 ± 0.52% | 78.78s | 154.71s | 30.3% | 0.9891 | 150,608,785 B | 52.5% | 52.2% |

## 2. Pairwise Hypothesis Testing on Task Success Rate (Holm-Bonferroni Corrected)

| Comparison Pair | Mean Δ TSR (%) | 95% Two-Sided CI ($t_{4}$) | Seed Differences (11, 22, 33, 44, 55) | Paired t | Unadjusted p | Holm Adjusted p | Cohen's $d_z$ | FWER Significant (p < 0.05)? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E1 vs B0 (Greedy Advantage)** | +6.48 ± 0.00% | ±0.00% | [+6.48%, +6.48%, +6.48%, +6.48%, +6.48%] | N/E | N/E (deterministic trace) | N/E | 0.00 | **Descriptive (+6.48%)** |
| **E2 vs E1 (RL vs Heuristic)** | -1.03 ± 0.53% | ±0.66% | [-0.36%, -1.56%, -1.58%, -0.81%, -0.84%] | -4.355 | 1.2112e-02 | 7.2672e-02 | -1.95 | **NO** |
| **E3 vs E2 (Local Forecast Utility)** | -0.24 ± 0.60% | ±0.74% | [-0.12%, -0.12%, -1.27%, +0.07%, +0.26%] | -0.891 | 4.2312e-01 | 1.0000e+00 | -0.40 | **NO** |
| **E3-APSM vs E3 (Collaborative Gain)** | +0.45 ± 1.35% | ±1.68% | [-0.21%, +0.27%, +2.83%, -0.18%, -0.45%] | 0.743 | 4.9854e-01 | 1.0000e+00 | 0.33 | **NO** |
| **E3-APSM vs E2 (End-to-End System Gain)** | +0.21 ± 0.77% | ±0.96% | [-0.34%, +0.15%, +1.56%, -0.12%, -0.19%] | 0.613 | 5.7293e-01 | 1.0000e+00 | 0.27 | **NO** |
| **E3-APSM vs E3-GL (APSM Filter vs Unfiltered GL)** | -0.11 ± 0.10% | ±0.12% | [-0.03%, -0.21%, -0.04%, -0.04%, -0.21%] | -2.497 | 6.7005e-02 | 3.3502e-01 | -1.12 | **NO** |
| **E3-GL vs E3-Local (Gossip Weight Exchange)** | +0.56 ± 1.32% | ±1.64% | [-0.18%, +0.48%, +2.87%, -0.14%, -0.24%] | 0.942 | 3.9937e-01 | 1.0000e+00 | 0.42 | **NO** |

## 3. Exploratory / Post Hoc Non-Inferiority Assessment: APSM vs Unfiltered Gossip Learning

> **Standard of Evidence Note:** The 1.0% TSR non-inferiority margin ($\delta = 1.00\%$) was formulated post hoc during Stage 6 synthesis rather than registered prior to experiment execution. It is formally reported as an exploratory assessment.

- **Hypothesis:** $H_0: \mu_{\text{APSM}} - \mu_{\text{GL}} \le -1.00\%$ vs $H_1: \mu_{\text{APSM}} - \mu_{\text{GL}} > -1.00\%$ (TSR)
- **Exploratory Margin ($\delta$):** 1.00% TSR
- **Observed Difference ($\bar{d}$):** -0.11% ± 0.10% (SE = 0.043%)
- **One-Sided 95% Lower Confidence Bound ($L_{95}$):** -0.20% (Threshold: > -1.00%)
- **Non-Inferiority t-Statistic ($t_{NI}$):** 20.730
- **One-Sided p-Value ($p_{NI}$):** 1.5996e-05
- **Assessment Verdict:** **CONFIRMED NON-INFERIOR (p < 0.0001)**
- **Important Boundary:** Non-inferiority in task scheduling success (TSR) does NOT imply identity in workload forecasting accuracy. Workload forecasting accuracy is evaluated independently as a distinct endpoint below.

## 4. Distinct Outcome Family: Direct Test-Window Workload Forecasting Error

Evaluated strictly over test bins $[1035, 1380)$ ($N=10$ nodes $\times$ $345$ test bins = $3,450$ aligned predictions). Validation observations strictly excluded.

| Arm Designation | Test MSE | Test RMSE | Test MAE | Persistence MSE | Error Reduction vs Persistence |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **E3: Local Forecast MADQN** | 0.037063 | 0.1925 | 0.1370 | 0.050809 | **+27.05%** |
| **E3-GL: Unfiltered Gossip** | 0.035884 | 0.1894 | 0.1348 | 0.050809 | **+29.38%** |
| **E3-APSM: Integrated Closed-Loop** | 0.036000 | 0.1897 | 0.1345 | 0.050809 | **+29.15%** |

### Paired Forecasting Error Contrast: E3-APSM vs E3-GL
- **E3-GL MSE (Unfiltered):** 0.035884
- **E3-APSM MSE (Filtered):** 0.036000
- **Paired Difference ($\Delta_{\text{MSE}}$):** +0.000117 ± 0.000042 (95% CI: $\pm$0.000052)
- **Cohen's d_z:** 2.78
- **Interpretation:** APSM incurs an empirical MSE difference of +0.000117 compared to full unfiltered Gossip Learning, while eliminating 52.23% of model exchange bytes.

## 5. Disaggregated Communication Traffic & Ledger Reconciliation

| Arm Designation | Scheduled Checks | Check Suppression (%) | Model Packets | Packet Reduction (%) | Model Bytes (MB) | Model Byte Reduction (%) | Control Traffic (MB) | Control Reduction (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0: LocalOnly** | 0 | N/A | 0 | N/A | 0.00 MB | N/A | 0.00 MB | N/A |
| **E1: ReactiveGreedy** | 0 | N/A | 0 | N/A | 0.00 MB | N/A | 1.05 MB | N/A |
| **E2: Reactive MADQN** | 0 | N/A | 0 | N/A | 0.00 MB | N/A | 0.95 MB | N/A |
| **E3: Local Forecast MADQN** | 0 | N/A | 0 | N/A | 0.00 MB | N/A | 0.95 MB | N/A |
| **E3-GL: Unfiltered Gossip** | 1,720 | 0.00% | 5,504 | 0.00% | 313.29 MB | 0.00% | 314.26 MB | 0.00% |
| **E3-APSM: Integrated Closed-Loop** | 1,720 | 52.55% | 2,629 | 52.23% | 149.64 MB | 52.23% | 150.61 MB | 52.07% |

## 6. Physical Network Contention Semantics

The discrete-event simulator implements a **directed-link, full-duplex FIFO contention model**:
- **Per-Directed-Link Serialization:** Each directed edge $(u \to v)$ maintains an independent transmission queue with serialization delay $d_{\text{tx}} = \frac{\text{bytes} \times 8}{R_{uv}}$.
- **Full-Duplex:** Uplink $(u \to v)$ and downlink $(v \to u)$ do not contend with each other.
- **Shared Capacity across Traffic Classes:** 56.9 KB model packets, task offload data packets ($P + 32\text{B}$ wire headers), 32B telemetry beacons, and 16B ACKs all serialize through the identical directed link queue.
- **No Inter-Link Radio Interference:** Links are modeled as isolated physical channels without shared wireless medium fading or channel-capture effects.
- **Buffer Capacity:** Link packet queues are unbounded (no wire queue drop); drop occurs via task deadline expiration or local queue overflow.
- **Batch Model Assimilation:** Model packets completing physical transmission during bin $t$ are buffered in `inboxes[v]` and assimilated by the recipient node at the next $30\text{s}$ bin callback via age-weighted merge.
