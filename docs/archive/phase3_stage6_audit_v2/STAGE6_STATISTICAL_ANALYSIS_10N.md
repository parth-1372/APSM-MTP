# Phase 3 Stage 6 Statistical Significance & Performance Analysis

**Topology:** 10 Nodes (Porto partial mesh, $k=3$, 16 undirected edges)
**Seeds:** [11, 22, 33, 44, 55] ($M = 5$ seeds, $\text{df} = 4$)
**Test Set:** Alibaba 2017 Trace, 17,862 tasks over 10,350 slots (bins [1035, 2070))
**Student's $t$ Critical Values:** Two-sided $t_{4, 0.025} = 2.7764$, One-sided $t_{4, 0.05} = 2.1318$

> **Methodological Note on Baseline Variance:** For deterministic baselines B0 (LocalOnly) and E1 (ReactiveGreedy), decisions are fixed functions of arrivals and deterministic channel state. On the fixed test scenario ($N_{\text{scenarios}} = 1$), variance across random seeds is identically 0.00% by construction. The paired hypothesis test against E1/B0 evaluates the variance introduced by stochastic exploration during MARL policy learning across seeds.

## 1. Benchmark Summary Table (Mean ± Sample Std across 5 Seeds)

| Arm Designation | TSR (%) | DMR (%) | Effective Latency (s) | P95 Latency (s) | Offload Rate (%) | Jain Fairness | Control Overhead (Bytes) | APSM Suppression |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **B0: LocalOnly** | 78.38 ± 0.00% | 21.62 ± 0.00% | 85.37s | 138.77s | 0.0% | 0.9713 | 0 B | 0.0% |
| **E1: ReactiveGreedy** | 84.86 ± 0.00% | 15.14 ± 0.00% | 76.31s | 145.97s | 40.6% | 0.9941 | 1,054,800 B | 0.0% |
| **E2: Reactive MADQN** | 83.83 ± 0.53% | 16.17 ± 0.53% | 79.07s | 157.18s | 28.2% | 0.9904 | 948,038 B | nan% |
| **E3: Local Forecast MADQN** | 83.59 ± 1.00% | 16.41 ± 1.00% | 79.22s | 154.15s | 28.1% | 0.9868 | 947,904 B | nan% |
| **E3-GL: Unfiltered Gossip** | 84.15 ± 0.45% | 15.85 ± 0.45% | 78.81s | 154.41s | 30.4% | 0.9883 | 314,255,235 B | 0.0% |
| **E3-APSM: Integrated Closed-Loop** | 84.04 ± 0.52% | 15.96 ± 0.52% | 78.78s | 154.71s | 30.3% | 0.9891 | 150,608,785 B | 52.5% |

## 2. Paired Hypothesis Testing Across 5 Seeds (Exact Small-Sample Student's t)

| Comparison Pair | Mean Δ TSR (%) | 95% Two-Sided CI ($t_{4}$) | Paired t-Statistic | p-Value | Cohen's d | Statistically Significant (p < 0.05)? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **E1 vs B0 (Greedy Advantage)** | +6.48 ± 0.00% | ±0.00% | inf | 0.0000e+00 | 0.00 | YES (p < 0.05) |
| **E2 vs E1 (RL vs Heuristic)** | -1.03 ± 0.53% | ±0.66% | -4.355 | 1.2112e-02 | -1.95 | YES (p < 0.05) |
| **E3 vs E2 (Local Forecast Utility)** | -0.24 ± 0.60% | ±0.74% | -0.891 | 4.2312e-01 | -0.40 | NO |
| **E3-APSM vs E3 (Gossip Collaborative Gain)** | +0.45 ± 1.35% | ±1.68% | 0.743 | 4.9854e-01 | 0.33 | NO |
| **E3-APSM vs E2 (End-to-End System Gain)** | +0.21 ± 0.77% | ±0.96% | 0.613 | 5.7293e-01 | 0.27 | NO |
| **E3-APSM vs E3-GL (APSM Filter vs Unfiltered GL)** | -0.11 ± 0.10% | ±0.12% | -2.497 | 6.7005e-02 | -1.12 | NO |
| **E3-GL vs E3-Local (Gossip Weight Exchange Gain)** | +0.56 ± 1.32% | ±1.64% | 0.942 | 3.9937e-01 | 0.42 | NO |

## 3. Pre-Specified Non-Inferiority Test: APSM vs Unfiltered Gossip Learning

- **Hypothesis:** $H_0: \mu_{\text{APSM}} - \mu_{\text{GL}} \le -1.0\%$ vs $H_1: \mu_{\text{APSM}} - \mu_{\text{GL}} > -1.0\%$ (TSR)
- **Pre-specified Non-Inferiority Margin ($\delta$):** 1.00% TSR
- **Observed Difference ($\bar{d}$):** -0.11% ± 0.10% (SE = 0.043%)
- **One-Sided 95% Lower Confidence Bound:** -0.20% (Threshold: > -1.00%)
- **Non-Inferiority t-Statistic ($t_{NI}$):** 20.730
- **One-Sided p-Value ($p_{NI}$):** 1.5996e-05
- **Formal Decision:** **CONFIRMED NON-INFERIOR (p < 0.05)**
- **Substantive Finding:** APSM achieves statistical non-inferiority in task scheduling performance while eliminating over 50% of the gossip model exchange bandwidth.

## 4. Direct Test-Window Workload Forecasting Error Evaluation

| Arm Designation | Test MSE | Test RMSE | Test MAE | Persistence MSE | Relative Gain vs Persistence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **E3: Local Forecast MADQN** | 0.0380 | 0.1948 | 0.1384 | 0.0517 | **+26.65%** |
| **E3-GL: Unfiltered Gossip** | 0.0367 | 0.1916 | 0.1361 | 0.0517 | **+29.02%** |
| **E3-APSM: Integrated Closed-Loop** | 0.0368 | 0.1919 | 0.1358 | 0.0517 | **+28.81%** |

## 5. Physical Link Contention & Communication Channel Reconciliation

- **Model Channel (Gossip Learning):** Broadcasts 14,224 parameters (56,920 bytes) across physical links with FIFO queuing contention.
- **APSM-H1 Bandwidth Suppression:** Successfully suppressed 52.5% of model broadcasts.
- **Operational Telemetry:** Fixed 32-byte UDP beacons sent every 15s to immediate 1-hop neighbors.
- **Channel Separation Verification:** Strict architectural separation maintained: zero model weights leaked into operational routing beacons, and zero task queue lengths transmitted over Gossip.
- **Byte Accounting Reconciliation:** Control bytes ($CB_{\text{control}} = M + T + A + H$) and system bytes ($CB_{\text{system}} = CB_{\text{control}} + P$) perfectly balance across all simulated nodes.
