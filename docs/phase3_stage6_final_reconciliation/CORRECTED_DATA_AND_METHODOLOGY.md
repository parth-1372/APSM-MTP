# Corrected Experimental Data & Methodology (N=10 Evaluation)

## 1. Trace Data Extraction and Preprocessing

### 1.1 Source Trace: Alibaba Cluster Trace 2017
The experimental evaluation uses authentic production workload data extracted from the **Alibaba Cluster Trace 2017** (`data/alibaba_v2017/alibaba-trace-2017/batch_instance.csv`).

To construct a realistic 10-node edge computing scenario:
1. **Host Selection:** 10 distinct, highly active machine IDs were identified and mapped to the 10 nodes of the topology.
2. **Task Filtering & Temporal Ordering:** Task instances from these machines were extracted, filtered for valid completion and resource metrics, and temporally ordered by instance start time over an 11.5-hour continuous period.
3. **Arrival Timestamps:** Real instance submission timestamps $t_{\text{submit}}$ were normalized to simulation time $t \in [0, 41400]\text{ seconds}$.
4. **Execution Workload:** Task CPU execution runtimes were derived from the trace CPU usage and execution duration metrics ($d_{\text{cpu}} = t_{\text{exec}}$).

### 1.2 Synthetic Network Payloads
While the Alibaba trace provides authentic task arrival burstiness, submission distributions, and CPU execution runtimes, it does not capture input network payload bytes suitable for edge-to-edge offloading. Therefore:
- Each task is assigned an input payload size sampled **log-uniformly** between 50,000 bytes and 1,000,000 bytes (50 KB to 1 MB):
  $$\log(S_{\text{net}}) \sim \text{Uniform}(\log(50000), \log(1000000))$$
- This payload represents input data (sensor feeds, images, telemetry) required to execute the task, which must be serialized across edge links if offloaded.

### 1.3 Task Deadlines
Task deadlines include a base slack of 3.0 seconds plus the scaled execution duration:
$$D_i = t_{\text{arr}} + 3.0 + \kappa \cdot t_{\text{exec}}$$
where $\kappa$ is the deadline slack factor.

### 1.4 Data Partitioning and Binning
Continuous task arrivals are aggregated into discrete temporal bins of width $\Delta t = 30\text{ seconds}$, yielding $N_{\text{bins}} = 1,380$ total bins across the 11.5-hour duration ($41,400\text{ seconds}$):
- **Training Partition:** Bins $[0, 828)$ (0 to 24,840 s; 6.90 hours; 60% of data).
- **Validation Partition:** Bins $[828, 1035)$ (24,840 to 31,050 s; 1.725 hours; 15% of data).
- **Test Partition:** Bins $[1035, 1380)$ (31,050 to 41,400 s; 2.875 hours; 25% of data).

Input features for edge forecasting models comprise a sliding lookback window of $L = 8$ bins (240 seconds history) of normalized task arrival counts.

---

## 2. Parameter Calibration & Operating Regime Selection

To evaluate offloading and forecasting algorithms under realistic conditions, the edge nodes must operate in an **empirically balanced burst-congested regime** where:
1. Under normal load, tasks succeed locally.
2. Under peak burst arrivals, local compute capacity is exceeded, requiring selective offloading.
3. Network links have non-zero transmission delays and finite bandwidth, penalizing indiscriminate offloading.

### 2.1 Calibration Methodology on Training Burst Slice
To prevent any data leakage or test-set snooping, calibration was conducted strictly on a training burst slice:
- **Slice Window:** Bins $[380, 488)$ (11,400 s to 14,640 s, duration = 3,240 s).
- **Task Count:** 10,678 tasks exhibiting acute temporal burstiness.
- **Evaluated Parameters:** Node CPU core capacity $C_{\text{cpu}} \in \{18.0, 20.0, 22.0\}$ and deadline slack factor $\kappa \in \{2.0, 2.2, 2.5\}$.

### 2.2 Complete 9-Point Calibration Grid Results
The table below is generated programmatically from the raw JSON artifact (`results/phase3/stage6_10n_calibration_train_burst.json`):

| $C_{\text{cpu}}$ (cores) | Slack factor $\kappa$ | B0 TSR (%) | B0 DMR (%) | E1 TSR (%) | E1 DMR (%) | $\Delta_{\text{TSR}}$ (E1 - B0) | E1 Offload Rate (%) | Operating Regime Assessment |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 18.0 | 2.0 | 68.58% | 31.42% | 67.46% | 32.54% | -1.12% | 36.37% | Severe congestion; heavy local queuing drops, offloading penalized by link queuing |
| 18.0 | 2.2 | 69.10% | 30.90% | 68.07% | 31.93% | -1.02% | 30.40% | Under-provisioned; persistent queue buildup across neighbor mesh |
| 18.0 | 2.5 | 69.74% | 30.26% | 71.88% | 28.12% | +2.14% | 29.66% | Lenient deadline partially mitigates queue wait, but capacity severely limited |
| 20.0 | 2.0 | 80.25% | 19.75% | 78.74% | 21.26% | -1.51% | 48.34% | Tight deadline budget limits multi-hop offloading window |
| **20.0** | **2.2** | **81.32%** | **18.68%** | **81.38%** | **18.62%** | **+0.07%** | **47.35%** | **Empirical/Design Operating Regime**: Balanced burst pressure (~81.3% local TSR), realistic offloading headroom (~18.7% drop risk) |
| 20.0 | 2.5 | 83.04% | 16.96% | 84.24% | 15.76% | +1.20% | 50.42% | Generous slack factor reduces burst pressure |
| 22.0 | 2.0 | 86.66% | 13.34% | 86.52% | 13.48% | -0.14% | 48.74% | Lightly loaded; local cores absorb majority of peak bursts |
| 22.0 | 2.2 | 87.93% | 12.07% | 88.85% | 11.15% | +0.92% | 49.44% | High local sufficiency; limited headroom for cooperative gain |
| 22.0 | 2.5 | 90.12% | 9.88% | 92.15% | 7.85% | +2.03% | 47.91% | Near-complete local sufficiency (>90% TSR); minimal offloading need |

### 2.3 Selection Rationale
The combination $C_{\text{cpu}} = 20.0$ and $\kappa = 2.2$ was selected as the **empirical design operating regime** because it establishes an authentic edge computing challenge:
- Local nodes successfully process approximately 81.3% of tasks during burst arrivals, leaving ~18.7% at risk of deadline expiration.
- This creates realistic headroom for multi-agent offloading and predictive scheduling to prevent queuing drops.
- It is NOT claimed to be a mathematical optimum across all dimensions, but rather the targeted operating point where offloading trade-offs are most acute.

---

## 3. Network Topology & Physical Constraints

- **Topology:** Porto 10-node partial mesh network ($N=10$, $k=3$ minimum connectivity, 16 undirected links / 32 directed channels).
- **Physical Link Bandwidths:** 50–200 Mbps per directed link.
- **Link Propagation Latencies (RTT):** 5–40 ms round-trip time.
- **Directed Contention Model:** Directed-link full-duplex FIFO contention in discrete-event simulation. Transmission serialization delay:
  $$d_{\text{tx}} = \frac{8 \cdot \text{bytes}}{R_{uv}}$$
- **Gossip Model Packet Size:** 56,920 bytes per serialized PyTorch model parameter broadcast.

---

## 4. Benchmark Arm Architecture

The evaluation compares six distinct system configurations across 5 policy training seeds ($11, 22, 33, 44, 55$):

### 4.1 B0: LocalOnly (Heuristic Baseline)
- All arriving tasks are queued and executed locally on the recipient node.
- No network offloading is performed ($0$ bytes transmitted).
- Tasks are dropped if the local queue capacity (48 tasks) overflows or if the task deadline is exceeded before completion.
- Deterministic policy on fixed macro trace ($s = 0.00\%$).

### 4.2 E1: ReactiveGreedy (Heuristic Baseline)
- Periodically exchanges 32-byte telemetry beacons every 15 seconds.
- When local queue occupancy exceeds 70%, greedily offloads arriving tasks to the least-loaded direct 1-hop neighbor.
- Deterministic policy on fixed macro trace ($s = 0.00\%$).

### 4.3 E2: Reactive MADQN (Multi-Agent RL without Forecasting)
- Parameter-shared Dueling Double-DQN across all 10 nodes.
- Local observation vector (queue depth, available CPU, background load, task features, neighbor telemetry).
- Action space: $\{0: \text{local execution}, 1\dots K: \text{offload to neighbor } k\}$.
- No workload forecasting input ($fc\_active = \text{False}$).

### 4.4 E3-Local: Forecast-Augmented MADQN with Isolated LSTMs
- Dueling Double-DQN augmented with 8-dimensional workload forecast features.
- Edge LSTM forecasters trained strictly locally on each node's historical arrivals.
- No gossip learning ($0$ model bytes exchanged).

### 4.5 E3-GL: Forecast-Augmented MADQN with Unfiltered Gossip Learning
- Same forecast-augmented MADQN policy as E3-Local.
- Forecasters exchange model weights every 2 bins (60 seconds) with all 1-hop neighbors.
- Received models are merged using Age-Weighted Semantic Averaging:
  $$\mathbf{w}_{\text{merged}} = \alpha \mathbf{w}_{\text{local}} + (1-\alpha) \mathbf{w}_{\text{remote}}, \quad \alpha = \frac{\text{age}_{\text{local}}}{\text{age}_{\text{local}} + \text{age}_{\text{remote}}}$$

### 4.6 E3-APSM: Forecast-Augmented MADQN with Semantic Model Suppression
- Identical architecture and learning algorithms as E3-GL.
- Incorporates the **Adaptive Periodic / Semantic Model (APSM)** suppression filter.
- Evaluates prediction residual $\epsilon_t = |y_t - \hat{y}_t|$ against population standard-deviation threshold:
  $$\tau_t = 1.5 \cdot \sigma(\mathcal{W}_{t-1}^{(50)}) \quad (\text{ddof}=0)$$
- Suppresses gossip transmission if $\epsilon_t \le \tau_t$, unless the consecutive suppressed checks counter reaches 5 (heartbeat synchronization).
