# Phase 3 Stage 6 Scientific Assessment and Remaining Limitations

**Thesis:** Decentralized Edge Workload Forecasting using Semantic Gossip Learning and Agentic AI  
**Scope:** $N = 10$ Nodes (Porto Edge Topology, $k=3$, 16 undirected links)  
**Branch:** `stage6-audit-fix-10n`  
**Date:** November 2026

---

## 1. Scope and Scale Boundaries

### 1.1 Network Scale ($N = 10$)
The validated experimental evaluation is conducted on an $N = 10$ node edge topology constructed from top urban cluster centroids in Porto, Portugal.
- **Limitation:** While $N = 10$ represents a realistic metropolitan micro-datacenter cluster with degree $k=3$ (16 undirected links), larger deployments ($N = 50$, $N = 100$) will encounter different gossip mixing times and graph diameter effects.
- **Future Direction:** Scaling laws for APSM suppression rate and gossip convergence as diameter increases ($O(\log N)$ or $O(\sqrt{N})$) will be investigated in subsequent large-scale experiments.

### 1.2 Evaluation Scenario Horizon ($N_{\text{scenarios}} = 1$)
- **Limitation:** Evaluation is conducted on a single 5.75-hour continuous test window from the Alibaba 2017 cluster trace ($17,862$ tasks across $10,350$ discrete-event slots). While this window provides high statistical power ($>17,000$ individual task arrival, queuing, and offloading events), it represents a single continuous macroeconomic workload trajectory.
- **Future Direction:** Cross-domain validation on heterogeneous traces such as the Azure Functions 2019/2021 dataset and Google Borg cluster traces across multi-day horizons.

---

## 2. Policy Dynamics: Heuristic vs Multi-Agent Reinforcement Learning

### 2.1 The Heuristic Offloading Advantage in High-Capacity Regimes
An important empirical finding of Stage 6 is that **ReactiveGreedy (E1)** achieves higher Task Success Rate ($84.86\%$) and lower deadline miss rate ($15.14\%$) than the learned MADQN policies ($78.38\% - 78.60\%$).
- **Mechanism:** E1 employs an aggressive, memoryless offloading policy: whenever local queue $Q_i > 4$, it immediately offloads to the neighbor with minimum queue length, offloading $40.6\%$ of all tasks. In a network where link bandwidths are relatively high ($50 - 200\text{ Mbps}$), this aggressive load shedding prevents local admission overflows very effectively.
- **RL Conservation & Conservative Offloading:** The parameter-shared Double Dueling MADQN policy learns to be far more conservative (offload rate $3\% - 5\%$). Because the reward function penalizes task drops heavily ($-1.0$) while penalizing offloading latency, the Q-network avoids network traversal risk unless local queue build-up is extreme.
- **Scientific Implication:** Deep RL policies in edge systems must be carefully rewarded to overcome risk-averse local execution equilibria. Future work will investigate curriculum learning and multi-objective Pareto reward formulation.

---

## 3. Physical Channel Modeling vs Full Kernel Network Stack

### 3.1 Discrete-Event FIFO Contention
- **Model Realism:** The Stage 6 simulator accurately implements shared physical link serialization ($d_{\text{tx}} = \text{bytes} \times 8 / R_{uv}$), propagation delay ($d_{\text{prop}} = \text{dist} / 2\times 10^8\text{ m/s}$), and link busy FIFO queuing. This ensures that offloaded tasks are physically delayed when sharing links with $56.9\text{ KB}$ Gossip model updates.
- **Limitation:** The simulator abstracts packet-level TCP congestion windows (e.g., CUBIC/BBR), bufferbloat, packet retransmission timeouts (RTO), and IP-level packet fragmentation.
- **Assessment:** For edge computing simulations where aggregate flow rate and queuing delay dominate, discrete-event link contention provides an accurate and computationally tractable representation, verified by exact byte accounting.

---

## 4. Semantic Filtering Parameter Sensitivity

### 4.1 Hyperparameter Selection ($q = 0.5, W = 20$)
- In APSM-H1, model broadcasts are triggered when prediction residual exceeds the rolling median ($q = 0.5$) over a window of $W = 20$ bins ($400\text{ seconds}$).
- **Finding:** This configuration consistently achieves $50\% - 53\%$ communication suppression while maintaining workload prediction MSE ($0.0368$ vs $0.0517$ persistence baseline).
- **Limitation:** $q$ and $W$ were selected based on validation split tuning. Dynamically adapting $q$ online as a function of real-time link utilization (e.g., increasing $q$ during link congestion to suppress more model packets) represents a promising adaptive extension.

---

## 5. Architectural Separation Verification

A key requirement of the thesis is the strict isolation between:
1. **Control-Plane Model Synchronization (Gossip Channel):** Carries $56.9\text{ KB}$ neural weight packets at low frequency ($60\text{s}$ cadence).
2. **Operational Routing Telemetry (Beacon Channel):** Carries lightweight $32\text{B}$ queue state beacons at high frequency ($15\text{s}$ cadence).

The code audit verified zero cross-contamination:
- Gossip messages carry strictly model state dictionaries.
- Routing beacons carry strictly scalar queue lengths and CPU capacities.
- Physical link bandwidth is shared under realistic FIFO queuing without protocol leakage.
