# STAGE 7 EXTREME-SCALE DECENTRALIZED EDGE SCALABILITY ANALYSIS

## 1. Executive Summary & Experimental Dimensions

This report synthesizes empirical evidence across **3 network scales**: $N=10$, $N=50$, $N=100$, comprising **90 total experimental runs** (30 runs per scale: 6 arms $\times$ 5 seeds).

### 1.1 Provenance & Data Sources
- **N=10 Reference Data:** Loaded from `results/phase3/stage6_integrated_10n_reconciled.json` (Stage 6 exact MARL forecasting evaluation; 30 runs).
- **N=50 Scalability Data:** Loaded from `results/phase3/stage7_scalability/n50/raw_runs.json` (Stage 7 scalability benchmark; 30 runs).
- **N=100 Scalability Data:** Loaded from `results/phase3/stage7_scalability/n100/raw_runs.json` (Stage 7 scalability benchmark; 30 runs).
- **Total Reconciled Experiments:** 30 ($N=10$) + 30 ($N=50$) + 30 ($N=100$) = **90 evaluated runs**.
- **Workload Provenance:** The 5 seeds (11, 22, 33, 44, 55) represent policy initialisation, minibatch sampling, exploration jitter, and gossip partner selection randomness. The underlying edge workload trace across all seeds is the exact chronological 10,350s test split of the Alibaba Cluster Trace 2017 (bins 1035..1380).
- **Hardware Execution Attribution:** Discrete-event task simulation rollouts and single-sample NumPy policy inference execute on CPU (multi-core threaded). PyTorch neural network minibatch gradient updates and Edge LSTM pretraining/online training execute on GPU via CUDA (standard FP32).

| Network Size ($N$) | Evaluation Split | Total Test Tasks | Tasks / Node | Topology Structure | Link Bandwidth | Model Transferred (GL) |
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
| **e3_madqn_local** | 83.57 $\pm$ 1.05 | 95.75 $\pm$ 0.39 | 96.55 $\pm$ 0.27 | 79.26 $\pm$ 0.88 | 48.37 $\pm$ 0.57 | 43.61 $\pm$ 0.54 |
| **e3_madqn_gl** | 84.07 $\pm$ 0.48 | 95.75 $\pm$ 0.35 | 96.56 $\pm$ 0.29 | 78.81 $\pm$ 0.48 | 48.22 $\pm$ 0.76 | 43.50 $\pm$ 0.55 |
| **e3_madqn_apsm** | 84.15 $\pm$ 0.49 | 95.73 $\pm$ 0.32 | 96.63 $\pm$ 0.27 | 78.67 $\pm$ 0.62 | 48.17 $\pm$ 0.72 | 43.65 $\pm$ 0.57 |

---

## 3. Communication Scaling & APSM Semantic Suppression

Analysis of network overhead growth, model byte scaling, and APSM semantic filtering efficiency.
Four distinct communication reduction metrics are reported:
1. **Model Byte Reduction (%):** Reduction in total model weights transferred over the network.
2. **Model Packet Reduction (%):** Reduction in physical model transfer frames.
3. **Scheduled Check Suppression (%):** Fraction of periodic gossip intervals where local semantic surprise fell below the gating threshold $\delta_{\text{APSM}}$.
4. **Control Plane Byte Reduction (%):** Overall reduction in total system control traffic (telemetry beacons, acknowledgments, and model frames).

| Network Size ($N$) | Unfiltered GL Model MB | APSM Model MB | Model Byte Reduction (%) | Model Packet Reduction (%) | Scheduled Check Suppression (%) | Control Plane MB (GL) | Control Plane MB (APSM) | Control Plane Reduction (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **N=10** | 313.29 MB | 149.86 MB | **52.17%** | 52.17% | 52.48% | 314.25 MB | 150.83 MB | **52.01%** |
| **N=50** | 1664.34 MB | 782.72 MB | **52.97%** | 52.97% | 53.11% | 1668.97 MB | 787.32 MB | **52.83%** |
| **N=100** | 3289.52 MB | 1519.92 MB | **53.79%** | 53.79% | 53.72% | 3298.30 MB | 1528.50 MB | **53.66%** |

---

## 4. Collaborative Forecasting Accuracy & Gain Retention Across Scales

Evaluation of decentralized workload prediction accuracy and retention of collaborative gain.
Collaborative Gain Retention is defined as:
$$\mathcal{R} = \frac{\text{MSE}_{\text{local}} - \text{MSE}_{\text{APSM}}}{\text{MSE}_{\text{local}} - \text{MSE}_{\text{GL}}} \times 100\%$$

| Network Size ($N$) | Local LSTM MSE | Unfiltered GL MSE | APSM Filtered MSE | APSM vs GL $\Delta_{\text{MSE}}$ | Collaborative Gain Retention ($\mathcal{R}$) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **N=10** | 0.037063 | 0.035884 | 0.036000 | +0.000117 | **90.12%** |
| **N=50** | 0.037977 | 0.037510 | 0.037537 | +0.000027 | **94.21%** |
| **N=100** | 0.038085 | 0.037633 | 0.037614 | -0.000019 | **104.17%** |

---

## 5. Computational Complexity & Resource Utilization

Execution throughput, peak system memory, and GPU memory utilization across scales.
Note: Simulation event rollouts execute on CPU, while PyTorch neural network training updates execute on GPU (CUDA).

| Network Size ($N$) | Total Tasks Simulated | Peak Process RAM (MB) | Peak GPU Alloc (MB) | Peak GPU Reserved (MB) | Throughput (Tasks/s) | Mean Run Wall Clock (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **N=50** | 84,173 | 1816.9 MB | 58.1 MB | 64.0 MB | 120.3 tasks/s | 699.9 s |
| **N=100** | 166,768 | 2078.6 MB | 94.3 MB | 100.0 MB | 90.2 tasks/s | 1848.7 s |

---

## 6. Scientific Conclusions & Scalability Invariants

1. **Defensible Communication Volume Suppression:** Rather than super-linear scaling, APSM demonstrates a remarkably consistent semantic suppression rate of 52.17% ($N=10$), 52.97% ($N=50$), and 53.79% ($N=100$) in model exchange volume. Because raw Gossip Learning broadcast volume scales with network edges ($O(|E|)$), this ~53% suppression prevents an increasingly massive absolute volume of wireless model transfers (saving 1.77 GB at $N=100$) from congesting edge link bandwidth.
2. **Collaborative Forecasting Preservation & Denoising:** Across network scales, APSM consistently retains $\ge 90\%$ of the collaborative forecasting accuracy advantage over local isolated models ($90.12\%$ at $N=10$, $94.21\%$ at $N=50$, and $104.17\%$ at $N=100$). The fact that APSM achieves slightly lower MSE than unfiltered GL at $N=100$ suggests that semantic surprise filtering acts as an implicit regulariser by discarding noisy model perturbations.
3. **Local Action Isolation & Bounded Observations:** Because maximum node degree remains $\le 7 < k_{\max}=8$ under $k$-edge augmentation, decentralized actor inference operates with identical parameter dimensions across all network scales without neighbor truncation.
