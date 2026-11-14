# Corrected Experimental Data & Methodology (N=10 Evaluation)

## 1. Trace Data Extraction and Preprocessing

### 1.1 Source Trace: Alibaba Cluster Trace 2018
The experimental evaluation uses authentic production workload data extracted from the Alibaba Cluster Trace 2018 (`batch_instance.csv`). 

To create a realistic 10-node edge computing scenario:
1. **Host Selection:** 10 distinct, highly active machine IDs were identified and mapped to the 10 nodes of the topology.
2. **Task Filtering & Temporal Ordering:** Task instances from these machines were extracted, filtered for valid completion and resource metrics, and temporally ordered by instance start time over an 11.5-hour continuous period.
3. **Arrival Timestamps:** Real instance submission timestamps $t_{\text{submit}}$ were normalized to simulation time $t \in [0, 41400]\text{ seconds}$.
4. **Execution Workload:** Task CPU execution requirements were derived from the trace CPU usage and execution duration metrics ($d_{\text{cpu}}$).

### 1.2 Synthetic Network Payloads
While the Alibaba trace provides accurate task arrival dynamics, submission burstiness, and CPU execution runtimes, it does not capture transmission payload sizes suitable for edge-to-edge offloading. Therefore:
- Each task is assigned a network transfer payload $S_{\text{net}} \sim \text{LogNormal}(\mu=1.0, \sigma=0.4)\text{ MB}$, bounded in $[0.5, 2.5]\text{ MB}$.
- This payload represents input data (sensor feeds, images, telemetry) required to process the task, which must be serialized across edge links if offloaded.

### 1.3 Data Partitioning and Binning
Continuous task arrivals are aggregated into discrete temporal bins of width $\Delta t = 30\text{ seconds}$, yielding $N_{\text{bins}} = 1,380$ total bins across the 11.5-hour duration:

- **Training Partition:** Bins $[0, 828)$ (0 to 24,840 s; 6.90 hours; 60% of data).
- **Validation Partition:** Bins $[828, 1035)$ (24,840 to 31,050 s; 1.725 hours; 15% of data).
- **Test Partition:** Bins $[1035, 1380)$ (31,050 to 41,400 s; 2.875 hours; 25% of data).

Input features for edge forecasting models comprise a sliding lookback window of $L = 8$ bins (240 seconds history) of normalized task arrival counts.

---

## 2. Parameter Calibration & Thesis Operating Regime

To evaluate offloading and forecasting algorithms under realistic conditions, the edge nodes must operate in a **burst-congested regime** where:
1. Under normal load, tasks succeed locally.
2. Under peak burst arrivals, local compute capacity is exceeded, requiring selective offloading.
3. Network links have non-zero transmission delays and finite bandwidth, penalizing indiscriminate offloading.

### 2.1 Calibration Methodology on Training Burst Slice
To prevent any data leakage or test-set snooping, calibration was conducted strictly on a training burst slice:
- **Slice Window:** Bins $[380, 488)$ (11,400 s to 14,640 s, duration = 3,240 s).
- **Task Count:** 10,678 tasks exhibiting acute temporal burstiness.
- **Evaluated Parameters:** Node CPU core capacity $C_{\text{cpu}} \in \{18.0, 20.0, 22.0\}$ and deadline slack factor $\kappa \in \{2.0, 2.2, 2.5\}$ (where deadline $D_i = t_{\text{arr}} + \kappa \cdot d_{\text{cpu}}$).

### 2.2 Calibration Results & Selection Rationale
The full 9-point grid sweep executed by `scripts/calibrate_10n_train.py` yielded the following results (persisted in `results/phase3/stage6_10n_calibration_train_burst.json`):

| $C_{\text{cpu}}$ (cores) | Slack $\kappa$ | LocalOnly (B0) TSR (%) | ReactiveGreedy (E1) TSR (%) | $\Delta_{\text{TSR}}$ (E1 - B0) | Operating Regime Assessment |
| :---: | :---: | :---: | :---: | :---: | :--- |
| 18.0 | 2.0 | 66.86% | 67.23% | +0.37% | Severe congestion; excessive queuing drops |
| 18.0 | 2.2 | 70.19% | 70.52% | +0.33% | Under-provisioned; queues rarely clear |
| 18.0 | 2.5 | 74.55% | 74.88% | +0.33% | Low baseline; tasks expire waiting |
| **20.0** | 2.0 | 78.42% | 78.50% | +0.08% | Tight deadlines constrain offload transit |
| **20.0** | **2.2** | **81.32%** | **81.38%** | **+0.06%** | **Optimal Thesis Regime: ~81% local TSR, realistic burst pressure** |
| 20.0 | 2.5 | 84.77% | 84.81% | +0.04% | Slack too lenient for edge SLA |
| 22.0 | 2.0 | 85.91% | 85.97% | +0.06% | Underloaded; local capacity absorbs bursts |
| 22.0 | 2.2 | 88.02% | 88.05% | +0.03% | Minimal offloading headroom |
| 22.0 | 2.5 | 90.64% | 90.67% | +0.03% | Near-complete local sufficiency |

**Conclusion:** $C_{\text{cpu}} = 20.0$ and $\kappa = 2.2$ represents the exact balanced operating regime: local nodes successfully process ~81% of tasks during bursts, leaving ~19% at risk of deadline violation. This creates realistic headroom for intelligent MARL offloading and workload forecasting without overwhelming the physical network links.

---

## 3. Experimental Arms & Architecture

The evaluation compares six distinct system configurations across 5 random seeds ($11, 22, 33, 44, 55$):

### 3.1 B0: LocalOnly (Heuristic Baseline)
- All arriving tasks are queued locally on the receiving node.
- No network offloading is performed ($0$ bytes transmitted).
- Tasks are dropped if the local queue capacity (50 tasks) is exceeded or if execution cannot complete before the deadline.

### 3.2 E1: ReactiveGreedy (Heuristic Offloading)
- When a node's local queue exceeds an occupancy threshold (70%), the node inspects the instantaneous queue depths of its direct one-hop neighbors.
- If a neighbor with lower queue occupancy exists, the task is offloaded across the physical link.
- No predictive lookahead or reinforcement learning is used.

### 3.3 E2: Reactive Parameter-Shared Dueling Double-DQN
- Multi-Agent Reinforcement Learning (MARL) where agents share network parameters across all 10 nodes.
- Dueling Q-network architecture (value stream $V(s)$ and advantage stream $A(s, a)$).
- State space includes local queue depth, current CPU utilization, and 1-hop neighbor queue depths.
- Action space: $\{0: \text{execute locally}, 1\dots K: \text{offload to neighbor } k\}$.
- Does not receive workload forecasting features.

### 3.4 E3-Local: Forecast-Augmented MADQN (Isolated Forecasters)
- Uses the same Dueling Double-DQN architecture as E2, augmented with an 8-dimensional workload forecast feature vector representing anticipated arrival rates over upcoming bins.
- Forecasters are trained locally on each node using an LSTM model.
- **No gossip learning:** Nodes do not communicate or share model updates. Forecasters rely strictly on isolated local observations.

### 3.5 E3-GL: Forecast-Augmented MADQN with Unfiltered Gossip Learning
- Same forecast-augmented MADQN policy as E3-Local.
- Forecasters exchange model parameter updates across physical topology links via Peer-to-Peer Gossip Learning.
- At every 30-second bin boundary, each node transmits its local model weights to a randomly selected peer.
- Received models are merged using Age-Weighted Semantic Averaging:
  $$\mathbf{w}_{\text{merged}} = \alpha \mathbf{w}_{\text{local}} + (1-\alpha) \mathbf{w}_{\text{remote}}$$
  where $\alpha = \frac{\text{age}_{\text{local}}}{\text{age}_{\text{local}} + \text{age}_{\text{remote}}}$.
- Generates high communication overhead on edge links.

### 3.6 E3-APSM: Forecast-Augmented MADQN with Semantic Model Suppression
- Identical architecture and learning algorithms as E3-GL.
- Incorporates the **Adaptive Periodic / Semantic Model (APSM)** suppression filter.
- At each bin boundary, the node evaluates its local prediction residual $\epsilon_t = |y_t - \hat{y}_t|$.
- Transmission is dynamically suppressed if the residual does not exceed an adaptive threshold:
  $$\tau_t = 1.5 \cdot \sigma(\mathcal{W}_{t-1}^{(50)})$$
- A heartbeat timer (5 bins / 150 seconds) guarantees periodic baseline synchronization even during prolonged quiescent periods.

---

## 4. Test Environment and Statistical Setup

- **Topology:** Porto 10-node partial mesh topology with realistic link bandwidths (10–50 Mbps) and propagation latencies (5–25 ms).
- **Test Set Tasks:** 17,862 real Alibaba task instances arriving during the 2.875-hour test partition.
- **Hardware Platform:** NVIDIA RTX 1000 Ada Generation Laptop GPU, Intel Core i7 processor (16 logical threads), Windows 11, PyTorch 2.11.0+cu130 with CUDA acceleration.
- **Execution Seeds:** 5 independent seeds (11, 22, 33, 44, 55) controlling MARL initialization, stochastic replay sampling, and randomized gossip peer selection.
- **Deterministic Baseline Property:** Because the Alibaba workload trace and task arrival sequence are fixed for the evaluation scenario ($N_{\text{scenarios}}=1$), heuristic policies B0 and E1 produce identical, deterministic results across all 5 seeds ($s = 0.00\%$).
