# Stage 6 Scale-Up Calibration Plan: 10-Node and 50-Node Edge Meshes

**Repository:** `APSM-MTP`  
**Branch:** `phase3-continuation`  
**Commit:** `02ace48`  
**Date:** November 13, 2026  
**Auditor:** Senior Distributed Systems Researcher, Multi-Agent RL Researcher, Independent Code Auditor  

---

## 1. Objective and Rationale

Before launching computationally expensive multi-agent reinforcement learning training on larger topologies ($N=10$ and $N=50$), the discrete-event simulator environment must be rigorously calibrated.

### The Over-Capacity / Under-Capacity Dilemma:
- **Trivial Over-Capacity ($C_{\text{cpu}} > 35.0$)**: If edge servers have excessive compute capacity, the local queue never congests. Baseline `LocalOnly` (B0) completes $\approx 100\%$ of tasks on time. Offloading offers zero performance advantage, and reinforcement learning agents appear ineffective.
- **Pathological Under-Capacity ($C_{\text{cpu}} < 12.0$)**: If edge servers are severely starved, all nodes in the mesh are permanently saturated. Every offloaded task arrives at an equally congested neighbor and drops due to deadline expiration, reducing the entire edge mesh to collapse.
- **The Realistic Edge Congestion Zone**: The goal is **not** to artificially force a specific metric, but to operate at a realistic edge tier capacity ($16\text{--}24$ cores per node) where burst arrivals during peak hours cause localized queue saturation ($15\%\text{--}30\%$ baseline drops in B0), creating a genuine cooperative load-balancing opportunity for offloading.

---

## 2. Workload and Topology Analysis Across Scales

### 2.1 Scale Comparison Matrix

| Topology Tier | Node Count ($N$) | Physical Graph | Connectivity ($K$) | Total Tasks ($M_{\text{test}}$) | Test Duration | Mean Arrival Rate ($\lambda_{\text{sys}}$) | Expected Bottleneck |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1 (Dev)** | $N=6$ | Porto Partial Mesh | $k=2$ | $11{,}366$ tasks | $10{,}350\text{ s}$ ($345$ bins) | $1.10\text{ tasks/s}$ | Highly localized bursts; 2 physical neighbors per node. |
| **Tier 2 (Primary)** | $N=10$ | Porto Partial Mesh | $k=3$ | $\approx 18{,}500$ tasks | $10{,}350\text{ s}$ ($345$ bins) | $1.79\text{ tasks/s}$ | Asymmetric degree distribution; partial mesh routing. |
| **Tier 3 (Large)** | $N=50$ | Porto Urban Mesh | $k=3$ | $\approx 92{,}000$ tasks | $10{,}350\text{ s}$ ($345$ bins) | $8.89\text{ tasks/s}$ | Large graph diameter; link queue contention. |

### 2.2 Workload Distribution
In the Alibaba 2017 trace, task arrivals are Pareto-distributed with heavy-tailed CPU requirements:
- **Mean Task CPU Work**: $W_{\text{cpu}} \approx 32.4\text{ core-seconds}$.
- **Mean Nominal Execution Rate**: $r_{\text{nom}} \approx 2.0\text{ cores/s}$.
- **Base Execution Time**: $\bar{T}_{\text{exec}} \approx 16.2\text{ seconds}$.
- **Synthetic Deadline Slack**: $t_{\text{deadline}} = t_{\text{arrive}} + 3.0\text{s} + \kappa \cdot \bar{T}_{\text{exec}}$, with $\kappa = 2.2$ yielding an average deadline window of $\approx 38.6\text{ seconds}$.

---

## 3. Physical Realism & Justification of Edge Capacity ($C_{\text{cpu}}$)

In commercial 5G Multi-access Edge Computing (MEC) deployments (e.g. AWS Wavelength, Azure Edge Zones), edge nodes are typically modular rack servers:
- **Processor**: Intel Xeon D-2700 (16–20 cores) or AMD EPYC Embedded 7002 (16–24 cores).
- **RAM**: 32 GB – 64 GB ECC DDR4.
- **Nominal Processing Capacity**: $16.0$ to $24.0$ normalized cores/s allocated to containerized application workloads.

### Justified Parameter Grid:
- **$C_{\text{cpu}} = 20.0\text{ cores/s}$**: Calibrated primary operating point.
- **$C_{\text{mem}} = 1.0$**: Normalized physical memory (tasks require $2\%\text{--}25\%$ of node memory).
- **$Q_{\text{max}} = 48\text{ tasks}$**: Maximum FIFO buffer per node before admission drop.
- **$\kappa = 2.2$**: Medium-slack deadline multiplier.

---

## 4. Step-by-Step Calibration Gate Protocol for $N=10$ and $N=50$

Before training MARL policies on $N=10$ or $N=50$, execute the following protocol:

### Step 1: Trace Ingestion and Verification
1. Run `build_or_load_curated_trace` with `n_nodes=10` and verify:
   - Output: `data/alibaba_v2017/processed/curated_10n_30s.npz` and `tasks_10n.parquet`.
   - Check that `metadata_10n.json` confirms zero future temporal leakage.
2. Construct topology via `build_edge_topology` ($k=3$):
   - Confirm symmetric adjacency matrix.
   - Verify all nodes have at least degree 2 (no disconnected islands).

### Step 2: Capacity Sweep via Deterministic Baselines
Run `src/phase3/run_stage4_validation.py` adapted for $N=10$ across the calibration grid:
$$C_{\text{cpu}} \in \{18.0, 20.0, 22.0\}, \quad \kappa \in \{2.0, 2.2, 2.5\}$$
For each grid point, execute:
- `LocalOnlyPolicy` (B0)
- `ReactiveGreedyPolicy` (E1)

### Step 3: Verification of Calibration Pass Criteria
The operating point $(C_{\text{cpu}}, \kappa)$ is deemed **scientifically justified** if and only if all of the following conditions are met:
1. **Task Conservation**: Total completed + admission drops + deadline drops + link drops $\equiv M_{\text{test}}$.
2. **Realistic Congestion**: B0 Task Success Rate falls between **$70.0\%$ and $80.0\%$** (Deadline Miss Rate between $20.0\%$ and $30.0\%$).
3. **Offloading Viability**: Reactive Greedy (E1) achieves a statistically significant throughput improvement:
   $$\text{TSR}_{\text{E1}} - \text{TSR}_{\text{B0}} \ge \mathbf{+5.0\%}$$
4. **Queue & Memory Conservation**: Zero memory leaks; active queue lengths never exceed $Q_{\text{max}} = 48$.
5. **Channel Contention**: Wireless link delays remain physically bounded; link drop rate $< 2.0\%$.

---

## 5. Specific Considerations for 50-Node Scale-Up

When scaling to $N=50$:
1. **Observation Vector Dimension**:
   In `Phase3Config`, `k_max_neighbors` is capped at $K_{\text{max}} = 8$. Even in a 50-node topology, each node only observes its top $K_{\text{max}}$ nearest physical neighbors. The observation dimension remains fixed at $75\text{-dim}$, preserving neural network weight compatibility!
2. **Link Delay Scaling**:
   With $50$ nodes, cross-link contention increases. Link transmission times must be verified using the Shannon channel capacity formula:
   $$R_{ij} = B \cdot \log_2(1 + \text{SINR}_{ij})$$
3. **Compute Efficiency**:
   At $N=50$, DES event scheduling volume increases $5\times$. The discrete-event queue min-heap must be profiled to ensure episode execution takes $< 15\text{ seconds}$ per episode.

