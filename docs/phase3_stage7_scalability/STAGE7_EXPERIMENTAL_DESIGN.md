# STAGE 7 EXPERIMENTAL DESIGN & SCALABILITY METHODOLOGY

## 1. Research Objectives & Scalability Hypotheses

### 1.1 Objective
Stage 7 evaluates the scaling characteristics of the integrated **Decentralized Edge Workload Forecasting, Semantic Gossip Learning (APSM), and Parameter-Shared Double Dueling Multi-Agent Reinforcement Learning (MADQN)** system.

The primary objective is to investigate whether the communication efficiency, collaborative forecasting advantages, and task offloading stability established at $N=10$ nodes hold across larger decentralized edge topologies:
- **Intermediate Scalability:** $N = 50$ nodes
- **Extreme-Scale Decentralized Edge:** $N = 100$ nodes

### 1.2 Core Research Hypotheses
1. **Hypothesis 1 (Communication Scaling):** In unfiltered decentralized Gossip Learning, model exchange volume scales with total network edges $\mathcal{O}(|E|)$. The APSM semantic filter will maintain or increase its suppression efficiency ($\ge 50\%$) as $N$ scales from 10 to 50 and 100, preventing physical communication bottlenecks.
2. **Hypothesis 2 (Collaborative Forecasting Retention):** Even under larger topologies where model updates travel across wider network diameters, APSM will retain $\ge 90\%$ of the collaborative forecasting accuracy gain over isolated local forecasters.
3. **Hypothesis 3 (Bounded Actor Complexity):** Because edge node density grows with bounded spatial degrees ($k=3$ augmentation yields max degree $\le 7 < k_{\max}=8$), local observation and action-mask dimensions remain strictly invariant across network scales without neighbor truncation.
4. **Hypothesis 4 (Task Offloading Stability):** Proactive workload forecasting will prevent localized queue build-up and achieve stable task success rates across larger scales without central orchestration.

---

## 2. Experimental Topologies & Workload Generation

### 2.1 Network Scales & Graph Topologies
All edge networks are synthesized from real cellular tower coordinates in Porto (`assets/porto_cells.parquet`) using deterministic seed 42 and $k$-edge connectivity augmentation ($k=3$):

| Scale | Nodes ($N$) | Undirected Edges ($|E|$) | Directed Channels | Min Degree | Mean Degree | Max Degree | Degree Bound ($k_{\max}$) | Truncation Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage 6 Reference** | 10 | 16 | 32 | 3 | 3.20 | 4 | 8 | **Zero Truncation** |
| **Stage 7 Primary** | 50 | 85 | 170 | 3 | 3.40 | 7 | 8 | **Zero Truncation** |
| **Stage 7 Extreme** | 100 | 168 | 336 | 3 | 3.36 | 6 | 8 | **Zero Truncation** |

### 2.2 Trace-Derived Workloads (Alibaba Cluster Trace 2017)
- **Source:** Deterministic extraction from `batch_instance.csv` for the top $N$ machine IDs by instance density.
- **Bin Duration:** $\Delta t_{\text{bin}} = 30$ seconds.
- **Chronological Data Splits:**
  - Training: $[0, 828)$ (828 bins = 6.9 hours)
  - Validation: $[828, 1035)$ (207 bins = 1.725 hours)
  - Testing: $[1035, 1380)$ (345 bins = 2.875 hours = 10,350 seconds)
- **Workload Volumes on Test Horizon:**
  - $N=10$: 17,862 tasks (1,786.2 tasks/node)
  - $N=50$: 84,173 tasks (1,683.5 tasks/node)
  - $N=100$: 166,768 tasks (1,667.7 tasks/node)
- **Task Attributes:** Log-uniform payload $50\text{ KB} \le D_k \le 1000\text{ KB}$, base slack $\sigma_0 = 3.0$ seconds.

---

## 3. Benchmark Arms & Algorithmic Configurations

The evaluation compares six standardized benchmark arms across all scales:

| Arm Key | Full Designation | Model Sharing | Telemetry / Gating Mode | Action Policy |
| :--- | :--- | :---: | :---: | :---: |
| **B0** | `b0_local` | None | None | LocalOnly (Zero offloading) |
| **E1** | `e1_greedy` | None | 15s Periodic Beacons | Reactive Greedy (Hysteresis 0.72) |
| **E2** | `e2_madqn` | None | 15s Periodic Beacons | Parameter-Shared Double Dueling MADQN |
| **E3-Local** | `e3_madqn_local` | None | 15s Periodic Beacons | MADQN + Local Edge LSTM Forecaster |
| **E3-GL** | `e3_madqn_gl` | Full Gossip | Unfiltered Periodic Broadcast (60s) | MADQN + Collaborative Gossip Forecaster |
| **E3-APSM** | `e3_madqn_apsm` | Semantic Gossip | APSM $H_1$ Residual Filter + Gossip | MADQN + Semantic Gated Collaborative Forecaster |

---

## 4. Execution Protocol & Resource Gates

To safeguard hardware stability and detect defects early, execution follows a strict 5-stage gating protocol:

```
[Gate A: N=10 Smoke Regression] ---> [Gate B: N=50 Smoke Test] ---> [Gate C: N=50 Full 5-Seed Benchmark]
                                                                                |
                                                                                v
                                    [Gate E: N=100 Full Benchmark] <--- [Gate D: N=100 Smoke Test]
```

### Staged Gates:
- **Gate A (N=10 Smoke Regression):** Validates that `run_stage7_scalability.py` produces identical data schemas, ledger arithmetic, and forecasting metrics as Stage 6.
- **Gate B (N=50 Smoke Test):** 1 seed, 2 training episodes. Asserts zero dimension mismatches, valid task conservation, link contention, and atomic JSON persistence.
- **Gate C (N=50 Full Benchmark):** 5 seeds (`[11, 22, 33, 44, 55]`), 48 episodes (36 warmup), all 6 arms. Produces full statistical analysis.
- **Gate D (N=100 Smoke Test):** 1 seed, 2 training episodes. Monitors memory footprint (RAM and GPU VRAM) and DES queue event rates.
- **Gate E (N=100 Full Benchmark):** Evaluated conditional on Gate D resource checks.

---

## 5. Statistical Methodology

1. **Random Seeds:** 5 independent seeds `[11, 22, 33, 44, 55]` across all MARL arms.
2. **Paired Statistical Testing:** APSM and unfiltered GL are paired seed-by-seed under identical initializations and workload schedules.
3. **Multiplicity Correction:** Step-down Holm-Bonferroni adjustment applied to all primary stochastic comparisons.
4. **Disaggregated Communication Metrics:**
   $$\text{Suppression Rate} = \frac{\text{Suppressed Checks}}{\text{Total Checks}} \times 100$$
   $$\text{Model Volume Reduction} = \left(1 - \frac{\text{Model Bytes}_{\text{APSM}}}{\text{Model Bytes}_{\text{GL}}}\right) \times 100$$
5. **Collaborative Gain Retention:**
   $$\mathcal{R} = \frac{\text{MSE}_{\text{Local}} - \text{MSE}_{\text{APSM}}}{\text{MSE}_{\text{Local}} - \text{MSE}_{\text{GL}}} \times 100$$
