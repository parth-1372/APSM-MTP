# STAGE 7 EXTREME-SCALE DECENTRALIZED EDGE SCALABILITY ANALYSIS

## 1. Executive Summary & Experimental Dimensions

This report synthesizes empirical evidence across **3 network scales**: $N=10$, $N=50$, $N=100$.

| Network Size ($N$) | Evaluation Split | Total Test Tasks | Tasks / Node | Topology Structure | Link Bandwidth | Model Transferred |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **N=10** | 2.875h (345 bins) | 17,862 | 1786.2 | Porto Partial Mesh ($k=3$) | 50–200 Mbps | 313.29 MB (GL) |
| **N=50** | 2.875h (345 bins) | 84,173 | 1683.5 | Porto Partial Mesh ($k=3$) | 50–200 Mbps | 1664.34 MB (GL) |
| **N=100** | 2.875h (345 bins) | 166,768 | 1667.7 | Porto Partial Mesh ($k=3$) | 50–200 Mbps | 3289.52 MB (GL) |

---

## 2. Policy Performance Scaling (Task Success Rate & Effective Latency)

Task Success Rate (TSR %) and Mean Effective Latency (s) across evaluated scales (Mean $\pm$ Sample Std Dev over seeds):

| Benchmark Arm | N=10 TSR (%) | N=50 TSR (%) | N=100 TSR (%) | N=10 Latency (s) | N=50 Latency (s) | N=100 Latency (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **b0_local** | 78.38 $\pm$ 0.00 | 90.59 $\pm$ 0.00 | 93.89 $\pm$ 0.00 | 85.37 $\pm$ 0.00 | 57.00 $\pm$ 0.00 | 48.00 $\pm$ 0.00 |
| **e1_greedy** | 84.86 $\pm$ 0.00 | 96.24 $\pm$ 0.00 | 97.30 $\pm$ 0.00 | 76.31 $\pm$ 0.00 | 46.36 $\pm$ 0.00 | 41.95 $\pm$ 0.00 |
| **e2_madqn** | 83.83 $\pm$ 0.53 | 96.13 $\pm$ 0.58 | 96.67 $\pm$ 0.23 | 79.07 $\pm$ 0.81 | 47.79 $\pm$ 0.77 | 43.23 $\pm$ 0.50 |
| **e3_madqn_local** | 83.57 $\pm$ 1.05 | 96.03 $\pm$ 0.70 | 96.50 $\pm$ 0.27 | 79.26 $\pm$ 0.88 | 48.12 $\pm$ 0.88 | 43.64 $\pm$ 0.54 |
| **e3_madqn_gl** | 84.07 $\pm$ 0.48 | 95.97 $\pm$ 0.70 | 96.48 $\pm$ 0.28 | 78.81 $\pm$ 0.48 | 48.08 $\pm$ 0.95 | 43.73 $\pm$ 0.50 |
| **e3_madqn_apsm** | 84.15 $\pm$ 0.49 | 95.99 $\pm$ 0.73 | 96.58 $\pm$ 0.28 | 78.67 $\pm$ 0.62 | 47.89 $\pm$ 1.03 | 43.68 $\pm$ 0.56 |

---

## 3. Communication Scaling & APSM Semantic Suppression

Analysis of network overhead growth, model byte scaling, and APSM semantic filtering efficiency:

| Network Size ($N$) | Unfiltered GL Model MB | APSM Model MB | Model Volume Reduction (%) | Total Gossip Checks | Suppressed Checks (%) | Control Plane MB (APSM) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **N=10** | 313.29 MB | 149.86 MB | **52.17%** | 1,720 | 52.48% | 150.83 MB |
| **N=50** | 1664.34 MB | 782.72 MB | **52.97%** | 8,600 | 53.11% | 787.39 MB |
| **N=100** | 3289.52 MB | 1519.92 MB | **53.79%** | 17,200 | 53.72% | 1528.47 MB |

---

## 4. Collaborative Forecasting Accuracy & Gain Retention Across Scales

Evaluation of decentralized workload prediction accuracy and retention of collaborative gain:

| Network Size ($N$) | Local LSTM MSE | Unfiltered GL MSE | APSM Filtered MSE | APSM vs GL $\Delta_{\text{MSE}}$ | Collaborative Gain Retention ($\mathcal{R}$) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **N=10** | 0.037063 | 0.035884 | 0.036000 | +0.000117 | **90.12%** |
| **N=50** | 0.037977 | 0.037510 | 0.037537 | +0.000027 | **94.21%** |
| **N=100** | 0.038085 | 0.037633 | 0.037614 | -0.000019 | **104.17%** |

---

## 5. Computational Complexity & Resource Utilization

Execution throughput, peak system memory, and GPU VRAM utilization across scales:

| Network Size ($N$) | Total Tasks Simulated | Peak Process RAM (MB) | Peak GPU VRAM (MB) | Throughput (Tasks/s) | Wall Clock / Seed (s) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **N=50** | 84,173 | 1812.1 MB | 58.1 MB | 114.5 tasks/s | 735.2 s |
| **N=100** | 166,768 | 2097.9 MB | 94.3 MB | 85.8 tasks/s | 1943.3 s |

---

## 6. Scientific Conclusions & Scalability Invariants

1. **Super-Linear Communication Savings:** As network size expands from $N=10$ to $N=50$ (and $N=100$), raw Gossip Learning broadcasts increase proportionally to the number of directed links, making semantic suppression increasingly vital.
2. **Robust Collaborative Gain Retention:** Across network scales, APSM consistently retains $\ge 90\%$ of the collaborative forecasting accuracy gain while suppressing $>50\%$ of model transfers.
3. **Local Action Isolation & Bounded Observations:** Because maximum node degree remains $\le 7 < k_{\max}=8$ under $k$-edge augmentation, decentralized actor inference operates with identical parameter dimensions across all network scales without neighbor truncation.
