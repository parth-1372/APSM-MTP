# Phase 3 Stage 6 Statistical Significance & Performance Analysis

**Topology:** 10 Nodes (Porto partial mesh, $k=3$, 16 undirected edges)
**Seeds:** [11, 22, 33, 44, 55] ($M = 5$ seeds)
**Test Set:** Alibaba 2017 Trace, 17,862 tasks over 10,350 slots

## 1. Benchmark Summary Table (Mean ± Sample Std)

| Arm Designation | TSR (%) | DMR (%) | Effective Latency (s) | P95 Latency (s) | Offload Rate (%) | Jain Fairness | Control Overhead (Bytes) | APSM Suppression |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **B0: LocalOnly** | 78.38 ± 0.00% | 21.62 ± 0.00% | 85.37s | 138.77s | 0.0% | 0.9713 | 0 B | 0.0% |
| **E1: ReactiveGreedy** | 84.86 ± 0.00% | 15.14 ± 0.00% | 76.31s | 145.97s | 40.5% | 0.9943 | 1,053,744 B | 0.0% |
| **E2: Reactive MADQN** | 84.13 ± 0.44% | 15.87 ± 0.44% | 78.70s | 152.13s | 27.2% | 0.9907 | 939,705 B | 0.0% |
| **E3: Local Forecast MADQN** | 84.18 ± 0.50% | 15.82 ± 0.50% | 78.64s | 151.53s | 27.1% | 0.9908 | 938,832 B | 100.0% |
| **E3-APSM: Integrated Closed-Loop** | 84.20 ± 0.55% | 15.80 ± 0.55% | 78.73s | 152.49s | 27.0% | 0.9906 | 151,354,788 B | 52.3% |

## 2. Paired Hypothesis Testing & Effect Sizes Across 5 Seeds

| Comparison Pair | Mean Δ TSR (%) | 95% CI | Paired t-Statistic | p-Value | Cohen's d | Statistically Significant (p < 0.05)? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **E1 vs B0 (Greedy Advantage)** | +6.48 ± 0.00% | ±0.00% | inf | 0.0000e+00 | 0.00 | YES (p < 0.05) |
| **E2 vs E1 (RL vs Heuristic)** | -0.73 ± 0.44% | ±0.39% | -3.704 | 2.0756e-02 | -1.66 | YES (p < 0.05) |
| **E3 vs E2 (Local Forecast Utility)** | +0.05 ± 0.12% | ±0.10% | 1.000 | 3.7390e-01 | 0.45 | NO |
| **E3-APSM vs E3 (Gossip Collaborative Gain)** | +0.01 ± 0.07% | ±0.06% | 0.452 | 6.7502e-01 | 0.20 | NO |
| **E3-APSM vs E2 (End-to-End System Gain)** | +0.07 ± 0.19% | ±0.16% | 0.805 | 4.6584e-01 | 0.36 | NO |

## 3. Communication Channel Reconciliation

- **Model Channel (Gossip Learning):** Broadcasts 14,224 parameters (56,920 bytes) across physical links.
- **APSM-H1 Suppression:** Successfully suppressed 52.3% of model broadcasts.
- **Operational Telemetry:** Fixed 32-byte UDP beacons sent every 15s to immediate 1-hop neighbors.
- **Separation Guarantee:** Zero model weights leaked into operational routing beacons, and zero task queue lengths transmitted over Gossip.
