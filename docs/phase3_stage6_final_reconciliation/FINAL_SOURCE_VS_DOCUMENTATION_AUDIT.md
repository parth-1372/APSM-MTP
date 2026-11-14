# Final Source Code vs. Documentation Audit (N=10 Reconciliation)

## 1. Executive Summary

During the Phase 3 Stage 6 research lifecycle, several discrepancies arose between the evolving documentation/narrative reports and the actual underlying Python implementation. This document provides an exhaustive, line-by-line reconciliation between the source code truth and past documentation claims.

All statements in this document have been verified directly against the production codebase on branch `stage6-final-reconcile-10n` (derived from audited milestone `5d0dec7` and updated at `e743f76`).

---

## 2. Parameter-by-Parameter Reconciliation Matrix

| Parameter / Dimension | Historical Misconception / Prior Documentation | Verified Source Code Truth | Primary Source Files & Line References | Impact & Scientific Significance |
| :--- | :--- | :--- | :--- | :--- |
| **Workload Trace Dataset** | Alibaba Cluster Trace 2018 | **Alibaba Cluster Trace 2017** (`batch_instance.csv`) | `src/phase3/config.py`: `data/alibaba_v2017/alibaba-trace-2017/batch_instance.csv` | Dataset contains authentic cluster job events from Alibaba 2017 release; trace task arrivals and CPU executions are preserved. |
| **Time Bin Duration ($\Delta t$)** | $\Delta t = 20\text{s}$ | **$\Delta t = 30\text{s}$** | `src/phase3/config.py`: `dt_bin_s: float = 30.0`<br>`src/phase3/data/alibaba_loader.py` | 30-second granularity provides stable workload aggregation while capturing edge traffic dynamics without excessive discrete event overhead. |
| **Total Number of Bins ($N_{\text{bins}}$)** | 2,070 bins (claimed 11.5-hour total horizon) | **1,380 bins total** (41,400s = 11.5 hours) | `src/phase3/config.py`: `n_bins: int = 1380` | The total duration is 11.5 hours: $1380 \times 30\text{s} = 41,400\text{s}$. The claim of 2,070 bins was an artifact of assuming $\Delta t = 20\text{s}$ ($41,400 / 20 = 2070$). |
| **Dataset Temporal Partitioning** | Train: $[0, 1242)$, Val: $[1242, 1552)$, Test: $[1552, 2070)$ | **Train: $[0, 828)$ (60%)<br>Val: $[828, 1035)$ (15%)<br>Test: $[1035, 1380)$ (25%)** | `src/phase3/config.py`: `train_end_bin: int = 828`, `val_end_bin: int = 1035`, `n_bins: int = 1380` | Train = 24,840s (6.90h), Val = 6,210s (1.725h), Test = 10,350s (2.875h). Total = 41,400s (11.5h). |
| **Test Horizon Duration** | 5.75 hours (20,700s) | **2.875 hours** (10,350s, 345 bins) | `src/phase3/run_stage6_integrated.py`: `test_start_s = 1035 * 30 = 31050s`, `test_duration_s = 10350s` | Test window evaluates $345 \text{ bins} \times 10 \text{ nodes} = 3,450$ aligned forecast slots. |
| **Forecasting Lookback Window ($L$)** | $L = 12$ bins | **$L = 8$ bins** (240 seconds history) | `src/phase3/config.py`: `lookback_L: int = 8`<br>`src/phase3/models/forecaster_lstm.py` | Input feature vector uses 8 past bins of normalized task arrival rates. 8 bins $\times 30\text{s} = 240\text{s}$ past horizon. |
| **Task Arrival Process** | Synthetic Poisson process ($E[\lambda]=5\text{ tasks/s}$) | **Trace-driven Alibaba 2017 cluster traces** from `batch_instance.csv` | `src/phase3/data/alibaba_loader.py`<br>`src/phase3/env/task_simulator.py` | Real Alibaba job instance arrivals sampled and mapped to 10 edge nodes; preserves authentic burstiness, non-stationarity, and spatiotemporal skew. |
| **Task Data / Payload Sizes** | Directly extracted from trace network fields or Log-Normal 0.5–2.5 MB | **Log-uniform sampling** between 50,000 bytes and 1,000,000 bytes (50 KB to 1 MB) | `src/phase3/env/task_simulator.py`: `log_min = math.log(sim_cfg.payload_bytes_min)`, `payload_bytes = np.exp(rng.uniform(log_min, log_max))` | Alibaba trace lacks per-instance network payload bytes; calibrated log-uniform distribution simulates edge task input data serialization. |
| **Task Deadline Formulation** | $D_i = t_{\text{arr}} + \kappa \cdot t_{\text{exec}}$ | **$D_i = t_{\text{arr}} + 3.0 + \kappa \cdot t_{\text{exec}}$** (base slack of 3.0 seconds) | `src/phase3/env/task_simulator.py`: `tasks.deadline_s = tasks.arr_time_s + 3.0 + kappa * tasks.cpu_work` | Base slack of 3.0s accounts for minimal link propagation and scheduling overhead in edge networks. |
| **APSM Threshold Mathematics** | Rolling median ($q = 0.5$) with absolute deviation or sample std ($N-1$) | **Population standard deviation** (`ddof=0`) over rolling window $W=50$: $\tau_t = 1.5 \cdot \sigma(\mathcal{W}_{t-1}^{(50)})$ | `src/phase3/forecasting/apsm.py`: `tau = k * np.std(previous_residuals)` (`ddof=0` default) | NumPy default population std dev is evaluated strictly over prior residuals before appending the current residual (prequential ordering). No median term. |
| **Gossip Cadence & Target** | Every bin to 1 randomly selected peer | **Every 2 bins (60 seconds)** broadcast to **all 1-hop neighbors** | `src/phase3/config.py`: `cadence_bins = 2`<br>`src/phase3/integration/live_gossip_engine.py`: loops over `topo.neighbor_slots(i)` | Cadence is 60s; sends to all direct neighbors. Heartbeat counter counts consecutive suppressed gossip checks (5 suppressed checks trigger heartbeat send). |
| **Network Topology & Links** | 10–50 Mbps bandwidth, 5–25 ms RTT | **50–200 Mbps bandwidth, 5–40 ms RTT, 16 undirected edges** | `src/phase3/data/topology.py`<br>`src/phase3/config.py` | Porto 10-node partial mesh topology with $k=3$ connectivity. |
| **Latency Metric Units** | Milliseconds (ms) | **Seconds (s)** | `src/phase3/env/task_simulator.py`: `effective_latency_mean_s`, `latency_completed_p95_s` | Cumulative task latencies are measured in seconds (mean effective latencies are ~76s to 85s across the 10,350s episode). |
| **Physical Network Dynamics** | Symmetric instant delivery or unconstrained links | **Directed-link full-duplex FIFO contention** | `src/phase3/env/task_simulator.py`: `EdgeNetworkSim`, per-link queues, transmission delay $\delta = \text{size} / R_{ij}$ | High-fidelity discrete-event queuing network where simultaneous task transmissions and model updates share physical channel capacity. |
| **Gossip Model Assimilation** | Instantaneous weight averaging upon packet arrival | **Timestamped message inbox buffering, discrete assimilation at 30s bin boundaries** | `src/phase3/integration/live_gossip_engine.py`: `_deliver_due_messages()`, `on_bin_boundary()` | Models in transit arrive at physical arrival time $t_{\text{rx}}$, buffer in node inboxes, and are merged into local edge forecasters strictly at discrete bin ticks. |

---

## 3. Detailed Technical Invariant Analysis

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

### 3.2 APSM Mathematical Formulation
In `src/phase3/forecasting/apsm.py`:
```python
@dataclass
class APSMConfig:
    k: float = 1.5
    stat: str = "std"          # population standard deviation (ddof=0)
    window_size: int = 50      # W = 50
    min_obs: int = 10
    heartbeat_interval: int = 5
```
For incoming residual $\epsilon_t = |y_t - \hat{y}_t|$:
$$\tau_t = k \cdot \sigma(\mathcal{W}_{t-1}^{(W)}) = 1.5 \cdot \sqrt{\frac{1}{|\mathcal{W}|}\sum_{j \in \mathcal{W}} (\epsilon_j - \bar{\epsilon})^2}$$
A gossip update is transmitted if $\epsilon_t > \tau_t$ or if the consecutive suppressed checks counter reaches 5.

### 3.3 Task Payloads and Deadlines
In `src/phase3/env/task_simulator.py`:
- Payloads are sampled log-uniformly:
  ```python
  log_min = math.log(sim_cfg.payload_bytes_min)  # 50,000 B
  log_max = math.log(sim_cfg.payload_bytes_max)  # 1,000,000 B
  payload_bytes = np.exp(rng.uniform(log_min, log_max, size=m))
  ```
- Task deadline includes a 3-second base slack:
  ```python
  deadline_s = arr_time_s + 3.0 + kappa * cpu_work
  ```
