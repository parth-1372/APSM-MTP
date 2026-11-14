# Final Source Code vs. Documentation Audit (N=10 Reconciliation)

## 1. Executive Summary

During the Phase 3 Stage 6 research lifecycle, several discrepancies arose between the evolving documentation/narrative reports and the actual underlying Python implementation. This document provides an exhaustive, line-by-line reconciliation between the source code truth and past documentation claims.

All statements in this document have been verified directly against the production codebase at branch `stage6-final-reconcile-10n` (derived from audited milestone `5d0dec7`).

---

## 2. Parameter-by-Parameter Reconciliation Matrix

| Parameter / Dimension | Historical Misconception / Prior Documentation | Verified Source Code Truth | Primary Source Files & Line References | Impact & Scientific Significance |
| :--- | :--- | :--- | :--- | :--- |
| **Time Bin Duration ($\Delta t$)** | $\Delta t = 20\text{s}$ | $\Delta t = 30\text{s}$ | `src/phase3/config.py`: `dt_bin_s: float = 30.0`<br>`src/phase3/data/alibaba_loader.py` | 30-second granularity provides stable workload aggregation while capturing edge traffic dynamics without excessive discrete event overhead. |
| **Total Number of Bins ($N_{\text{bins}}$)** | 2,070 bins (claimed 11.5-hour total horizon) | 1,380 bins total (41,400s = 11.5 hours) | `src/phase3/config.py`: `n_bins: int = 1380` | The total duration is 11.5 hours: $1380 \times 30\text{s} = 41,400\text{s}$. The claim of 2,070 bins was an artifact of assuming $\Delta t = 20\text{s}$ ($41,400 / 20 = 2070$). |
| **Dataset Temporal Partitioning** | Train: $[0, 1242)$, Val: $[1242, 1552)$, Test: $[1552, 2070)$ | Train: $[0, 828)$ (60%)<br>Val: $[828, 1035)$ (15%)<br>Test: $[1035, 1380)$ (25%) | `src/phase3/config.py`: `train_end_bin: int = 828`, `val_end_bin: int = 1035`, `n_bins: int = 1380` | Train = 24,840s (6.90h), Val = 6,210s (1.725h), Test = 10,350s (2.875h). Total = 41,400s (11.5h). |
| **Test Horizon Duration** | 5.75 hours (20,700s) | 2.875 hours (10,350s, 345 bins) | `src/phase3/run_stage6_integrated.py`: `test_start_s = 1035 * 30 = 31050s`, `test_duration_s = 10350s` | Test window evaluates $345 \text{ bins} \times 10 \text{ nodes} = 3,450$ aligned forecast slots. |
| **Forecasting Lookback Window ($L$)** | $L = 12$ bins | $L = 8$ bins (240 seconds history) | `src/phase3/config.py`: `lookback_L: int = 8`<br>`src/phase3/models/forecaster_lstm.py` | Input feature vector uses 8 past bins of normalized task arrival rates. 8 bins $\times 30\text{s} = 240\text{s}$ past horizon. |
| **Task Arrival Process** | Synthetic Poisson process ($E[\lambda]=5\text{ tasks/s}$) | Trace-driven Alibaba 2018 cluster traces from `batch_instance.csv` | `src/phase3/data/alibaba_loader.py`<br>`src/phase3/env/task_simulator.py` | Real Alibaba job instance arrivals sampled and mapped to 10 edge nodes; preserves authentic burstiness, non-stationarity, and spatiotemporal skew. |
| **Task Data / Payload Sizes** | Directly extracted from Alibaba trace network fields | Synthetically generated via log-normal / uniform distributions | `src/phase3/env/task_simulator.py`: `data_size_mb ~ Uniform/LogNormal(0.5, 2.5)` | Alibaba `batch_instance.csv` lacks reliable per-task network payload bytes; realistic edge transmission payload distributions were calibrated to simulate link bandwidth contention. |
| **APSM Threshold Mathematics** | Rolling median ($q = 0.5$) with absolute deviation or quantile $q=0.5, W=20$ | Sample standard deviation over rolling window $W=50$: $\tau_t = 1.5 \cdot \sigma(\mathcal{W}_{t-1}^{(50)})$ | `src/phase3/apsm/apsm_filter.py`: `stat: str = "std"`, `k: float = 1.5`, `window_size: int = 50`, `min_obs: int = 10`, `heartbeat_interval: int = 5` | Code computes `np.std(window)` over prior residuals. If residual exceeds $\tau_t$, gossip sync is triggered; otherwise suppressed unless heartbeat (5 bins) expires. |
| **MADQN Offload Rate** | 3%–5% offload rate (erroneous early draft claim) | 28.3%–29.8% offload rate under burst conditions | `results/phase3/stage6_integrated_10n_reconciled.json`: `offload_action_ratio`: E2=0.298, E3-Local=0.287, E3-GL=0.283, E3-APSM=0.284 | Agents actively exploit neighbor compute capacity during bursts, yielding significant task success rate preservation over LocalOnly (B0). |
| **Physical Network Dynamics** | Symmetric instant delivery or unconstrained links | Directed-link full-duplex FIFO physical network contention | `src/phase3/env/task_simulator.py`: `EdgeNetworkSim`, per-link queues, transmission delay $\delta = \text{size} / R_{ij}$ | High-fidelity discrete-event queuing network where simultaneous task transmissions and model updates share physical channel capacity. |
| **Gossip Model Assimilation** | Continuous instantaneous weight averaging upon packet arrival | Timestamped message inbox delivery, discrete assimilation at 30s bin boundaries | `src/phase3/integration/live_gossip_engine.py`: `_deliver_due_messages()`, `on_bin_boundary()` | Models in transit arrive at physical arrival time $t_{\text{rx}}$, buffer in node inboxes, and are merged into local edge forecasters strictly at discrete bin ticks. |

---

## 3. Detailed Technical Analysis

### 3.1 Dataset Bins and Temporal Split
In `src/phase3/config.py`:
```python
dt_bin_s: float = 30.0
n_bins: int = 1380
train_end_bin: int = 828      # 60% of 1380 = 24,840s (6.9h)
val_end_bin: int = 1035       # 75% of 1380 = 31,050s (8.625h)
lookback_L: int = 8           # 8 * 30s = 240s
```
The total duration of the scenario is $1,380 \times 30 = 41,400\text{ seconds} = 11.5\text{ hours}$.
The test partition spans bins $[1035, 1380)$, which is $345\text{ bins} \times 30\text{s} = 10,350\text{s} = 2.875\text{ hours}$.
Earlier notes that asserted 2,070 bins arose from dividing 41,400s by an assumed 20s bin width ($41,400 / 20 = 2070$). When 30s bins were finalized to ensure sufficient trace events per bin, the total bin count became 1,380.

### 3.2 APSM Mathematical Formulation
In `src/phase3/apsm/apsm_filter.py`:
```python
@dataclass
class APSMConfig:
    k: float = 1.5
    stat: str = "std"          # "std" uses sample standard deviation
    window_size: int = 50      # W = 50
    min_obs: int = 10
    heartbeat_interval: int = 5
```
For incoming residual $\epsilon_t = |y_t - \hat{y}_t|$:
$$\tau_t = k \cdot \sigma(\mathcal{W}_{t-1}^{(W)}) = 1.5 \cdot \sqrt{\frac{1}{|\mathcal{W}|-1}\sum_{j \in \mathcal{W}} (\epsilon_j - \bar{\epsilon})^2}$$
A gossip update is transmitted if $\epsilon_t > \tau_t$ or if $(t - t_{\text{last\_tx}}) \ge 5$.
Prior claims mentioning median or interquartile filters were preliminary conceptual designs that were superseded by the moving standard-deviation implementation.

### 3.3 Network and Task Simulator
In `src/phase3/env/task_simulator.py`:
- Tasks are instantiated from genuine Alibaba trace events recorded in `data/alibaba/batch_instance.csv`.
- Each task arrival specifies an arrival time $t_{\text{arr}}$, CPU execution duration $d_{\text{cpu}}$, CPU capacity demand, and deadline slack $\kappa \cdot d_{\text{cpu}}$.
- Network payloads are drawn from a calibrated distribution to represent edge IoT/mobile inputs (0.5 to 2.5 MB).
- Links between the 10 nodes are modeled with topology-defined bandwidths ($10$ to $50$ Mbps) and propagation latencies ($5$ to $25$ ms).
- Queuing at network interfaces is modeled using discrete event updates: packets serialize through link transmission queues, correctly enforcing head-of-line blocking and physical contention.

### 3.4 Gossip Weight Exchange & Assimilation
- During live simulation, when a node triggers a gossip transmission, a `ModelUpdatePacket` containing serialized network weights is pushed to the network simulator.
- The packet incurs network serialization and propagation delay based on link bandwidth.
- When the packet reaches the destination node, it is placed in the recipient's incoming inbox.
- Nodes process their inbox at 30-second bin intervals (`on_bin_boundary`), taking all messages delivered up to $t_{\text{bin}}$ and performing Age-Weighted Semantic Averaging.
- This decoupling ensures physical network latency is faithfully simulated without violating discrete-time MARL state boundaries.
