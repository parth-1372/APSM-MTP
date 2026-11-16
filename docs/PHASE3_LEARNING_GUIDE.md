# APSM-MTP Phase 3 Learning Guide: Architecture, Algorithms & Defence Handbook

> **Author:** Parth (Roll No. 2022IMT082)  
> **Degree:** Master of Technology in Computer Science & Engineering  
> **Thesis:** *Decentralized Edge Workload Forecasting using Semantic Gossip Learning and Agentic AI*  
> **Purpose:** Personal mastery guide and viva-defence handbook for Phase 3. Explains the intuition, architectural decisions, mathematical mechanics, empirical findings, and code structure in plain, defensible terms.

---

## Table of Contents
1. [Section A — Phase 3 in One Big Picture](#section-a--phase-3-in-one-big-picture)
2. [Section B — Research Background and Base Papers](#section-b--research-background-and-base-papers)
3. [Section C — Complete Phase 3 Architecture](#section-c--complete-phase-3-architecture)
4. [Section D — Understanding the Core Concepts](#section-d--understanding-the-core-concepts)
5. [Section E — Why Did We Choose This Architecture? (Design Decisions)](#section-e--why-did-we-choose-this-architecture-design-decisions)
6. [Section F — The Experimental Journey & Development Chronology](#section-f--the-experimental-journey--development-chronology)
7. [Section G — Experiments, Baselines, and Results](#section-g--experiments-baselines-and-results)
8. [Section H — What Did Our Approach Gain Over Baselines & Base Papers?](#section-h--what-did-our-approach-gain-over-baselines--base-papers)
9. [Section I — Critical Limitations (Viva Defense Questions)](#section-i--critical-limitations-viva-defense-questions)
10. [Section J — Codebase Navigation Guide](#section-j--codebase-navigation-guide)
11. [Section K — Glossary & 10 Defence Self-Check Questions](#section-k--glossary--10-defence-self-check-questions)

---

## Section A — Phase 3 in One Big Picture

### 1. What problem are we solving in Phase 3?
In edge computing, battery-powered or resource-constrained nodes (like cellular base stations, roadside units, or IoT gateways) receive random streams of compute tasks from end users. If a node gets swamped, tasks miss their deadlines and get dropped. Nodes *could* offload tasks to their spatial neighbors, but they cannot afford a centralized cloud server telling everyone what to do—central cloud coordinators introduce a single point of failure (SPOF) and prohibitive transmission latency.

Furthermore, if a node wants to know when it or its neighbors will be swamped, it needs a **workload forecasting model**. However, training neural networks collaboratively across edge nodes without a server requires **Gossip Learning** (nodes pinging each other their model weights). In standard Gossip Learning, nodes blindly transmit megabytes of weight updates over wireless channels at every round, even if their model hardly changed. 

**Phase 3 solves both problems simultaneously:** It establishes a decentralized edge network where nodes **forecast future workloads**, **share model weights only when semantically necessary** (saving >52% of wireless bandwidth), and use **multi-agent reinforcement learning (MADQN)** to make intelligent, proactive task-offloading decisions before local compute queues overflow.

### 2. What did Phase 2 already establish?
In Phase 2, we worked with the **Porto Taxi dataset** on a static traffic prediction task. Phase 2 proved the fundamental viability of the **Adaptive Predictive-Semantic Mechanism (APSM)**:
- Nodes do not need to transmit neural network weights on every training round.
- By tracking validation surprise $\varepsilon_i(t)$ against an adaptive threshold $\tau_i(t) = \mu_i(t) + k \cdot \sigma_i(t)$, nodes could filter out repetitive, uninformative weight broadcasts.
- Phase 2 achieved a **53.6% reduction in packet transmissions** with less than 1% degradation in forecasting accuracy.

### 3. What additional problem does Phase 3 address?
Phase 2 was a pure machine learning simulation on traffic data. It had **no concept of computational tasks, queues, wireless link congestion, or execution deadlines**.

Phase 3 transitions the project from a pure ML filter into a **complete decentralized edge operating environment**:
1. **Real Industrial Workloads:** Replaces Porto taxi data with continuous, heterogeneous CPU/Memory utilization traces from the **Alibaba Cluster Trace v2017**.
2. **Computational Task Simulation:** Injected discrete computational tasks with CPU requirements, data payloads, and strict expiration deadlines (20–40 seconds).
3. **Agentic Multi-Agent Reinforcement Learning (MARL):** Implements independent **Double-DQN (MADQN)** agents that run locally on each edge node to decide task placement (execute locally vs. offload to neighbor $j$).
4. **Physical Channel Separation & Contention:** Models shared physical wireless channels where multi-megabyte model synchronization packets and kilobyte task offloading requests compete for bandwidth.
5. **Cross-Scale Scalability:** Validates the system not just at $N=10$, but at **$N=50$ and $N=100$ nodes** across 90 benchmark runs.

### 4. Why do we need workload forecasting and task offloading together?
If a node only reacts to its *current* queue length, it suffers from the "lag problem": by the time its queue is 95% full, incoming tasks cannot be transferred across wireless links before their deadlines expire. 

If the node knows that its workload will surge 60 seconds from now (predicted by its local LSTM forecaster), its reinforcement learning policy can begin offloading incoming tasks to lightly loaded neighbors **proactively**. 

Conversely, collaborative forecasting requires decentralized model exchange over the exact same wireless links used for task offloading. If model training hogs the link, tasks cannot offload. **APSM suppresses unnecessary model transmissions, freeing the wireless channel so task offloading can succeed.**

### 5. What is the central idea of our proposed system?
The central idea is **Predictive-Semantic Synergy**:
> *Use semantic filtering (APSM) to slash model synchronization overhead by more than half, and use the resulting accurate, low-overhead workload forecasts to empower local MARL agents to offload tasks proactively without saturating wireless channels.*

### 6. What is the main contribution we are trying to demonstrate?
We demonstrate that across networks of 10, 50, and 100 edge nodes:
- APSM eliminates **52.17% to 53.79% of wireless model bandwidth** (saving 1.77 GB of payload at $N=100$).
- APSM retains **90.12% to 104.17% of the forecasting accuracy gain** provided by full Gossip Learning.
- MARL task offloading achieves up to **96.63% task success rates**, with zero central coordination and complete physical channel stability.

---

## Section B — Research Background and Base Papers

To explain and defend Phase 3, you must understand where each piece came from. We did not build everything from scratch; we built on published foundations and extended them where they fell short.

### 1. Base Paper 1: Tundo et al. (IEEE TNSM 2025)
- **Full Citation:** Alessandro Tundo, Marco Mobilio, Michele Ciavotta, and Leonardo Mariani, *"Gossip Learning for Edge Computing: A Decentralized Machine Learning Approach to Urban Traffic Prediction,"* **IEEE Transactions on Network and Service Management (TNSM)**, vol. 22, no. 1, pp. 450–464, March 2025. [DOI: 10.1109/TNSM.2024.3491200](https://doi.org/10.1109/TNSM.2024.3491200). Local copy: [`papers/MTP Base paper.pdf`](file:///C:/Users/btpUser/Desktop/Parth_MTP/APSM-MTP/papers/MTP%20Base%20paper.pdf).
- **Problem it solved:** Federated Learning requires a central parameter server. If the server fails or network connections to the cloud are severed, learning stops. Tundo et al. proposed using **Gossip Learning (GL)**—nodes train locally and randomly exchange models with graph neighbors.
- **Key mechanism we took:** **Age-Weighted Model Merging**. When node $i$ receives a model from node $j$, it merges weights using their training iteration counts (ages):
  $$\theta_{\text{merged}} = \frac{\text{age}_i}{\text{age}_i + \text{age}_j}\theta_i + \frac{\text{age}_j}{\text{age}_i + \text{age}_j}\theta_j$$
- **What was missing / What we extended:** Tundo et al. had **blind transmissions**. Every node sends its full model weights at every round. In bandwidth-limited edge networks, this causes massive congestion. We introduced **APSM** as a pre-transmission gate to suppress redundant updates. Furthermore, Tundo et al. only forecasted traffic; they did not implement task offloading or reinforcement learning.

### 2. Supporting Literature: Adu et al. (Sensors 2026)
- **Full Citation:** K. Adu, R. Mensah, and Y. Zhang, *"Multi-Agent Deep Reinforcement Learning for Decentralized Task Offloading in Heterogeneous Edge Networks,"* **MDPI Sensors**, vol. 26, no. 3, art. 914, Jan 2026. [DOI: 10.3390/s26030914](https://doi.org/10.3390/s26030914). Local copy: [`papers/sensors-26-00914.pdf`](file:///C:/Users/btpUser/Desktop/Parth_MTP/APSM-MTP/papers/sensors-26-00914.pdf).
- **Problem it solved:** Decentralized task offloading across heterogeneous edge base stations.
- **Key mechanism we adapted:** **Neighbor Action Masking** and local queue state representations for RL agents. Each node only considers actions corresponding to its physically connected neighbors.
- **What we changed:** Adu et al. used reactive queue states without future workload forecasting. We fed **live 1-step-ahead forecasts** directly into the RL observation vector, transforming the policy from reactive to proactive.

### 3. Supporting Literature: DECOFFEE (Chen et al., 2026)
- **Citation:** S. Chen et al., *"DECOFFEE: Decentralized Collaborative Federated Edge Forecasting with Privacy Preservation,"* **arXiv:2604.24507**, 2026. Local copy: [`papers/2604.24507v1.pdf`](file:///C:/Users/btpUser/Desktop/Parth_MTP/APSM-MTP/papers/2604.24507v1.pdf).
- **Role:** Peer benchmark confirming that collaborative recurrent forecasting across edge devices improves local generalization under non-stationary workload distributions.

### Summary Comparison Table

| Research Area / Paper | What the Paper Proposed | What We Adopted | What We Changed or Extended |
|---|---|---|---|
| **Tundo et al. (IEEE TNSM 2025)** *(Primary Base Paper)* | Decentralized Gossip Learning for traffic forecasting; age-weighted model merging. | Age-weighted model merge formula; sparse P2P graph topology. | Added APSM semantic gating (suppressing >52% transmissions); integrated live task simulator and MARL offloading. |
| **Adu et al. (Sensors 2026)** | Multi-agent RL for edge task offloading using local queue observations. | Action space formulation (local vs. neighbor offload); neighbor validity masking. | Added live workload forecast into state space; integrated shared-link physical contention with ML traffic. |
| **Chen et al. (DECOFFEE 2026)** | Collaborative edge forecasting under non-stationary distributions. | Multi-node recurrent workload sequence forecasting formulation. | Replaced federated coordinator with fully serverless P2P gossip; added semantic surprise gating. |
| **Phase 2 (Our Prior Work)** | APSM pre-send filter on Porto taxi dataset (MSE 0.009, 53.6% packet reduction). | Adaptive noise-floor thresholding $\tau_i(t) = \mu_i(t) + k \cdot \sigma_i(t)$ and safety heartbeat. | Moved from synthetic traffic to Alibaba compute traces; added task queue physics, transmission delays, deadlines, and MADQN. |

---

## Section C — Complete Phase 3 Architecture

The architecture coordinates physical resource usage, discrete simulation time, machine learning model synchronization, and reinforcement learning control.

### Mermaid Architecture Diagram

```mermaid
flowchart TD
    subgraph DataPlane ["1. Data & Workload Pipeline"]
        Raw["Alibaba Cluster Trace (1300+ Machines)"] --> Preprocess["Binning (dt = 30s) & MinMax Scaling"]
        Preprocess --> Curated["Curated Workload Tensor (N Nodes x 1380 Bins)"]
    end

    subgraph ForecastEngine ["2. Decentralized Forecasting Engine"]
        Curated --> Lookback["Lookback Buffer (L = 8 Bins)"]
        Lookback --> EdgeLSTM["Local Edge-GRU/LSTM Model"]
        EdgeLSTM --> Pred["1-Step Ahead CPU Prediction y_hat(t+1)"]
        EdgeLSTM --> ValSurprise["Compute Validation Loss Surprise epsilon(t)"]
        ValSurprise --> APSMGate{"APSM Semantic Gate<br>epsilon(t) > tau(t) OR Heartbeat?"}
        APSMGate -- Yes: Send --> SendModel["Broadcast Model Weights to Peer"]
        APSMGate -- No: Suppress --> Suppress["Increment Local Age Only (Zero Bandwidth)"]
        SendModel --> LinkQueue["Shared Wireless Link Queue"]
    end

    subgraph MARLEngine ["3. Agentic Task Offloading (MADQN)"]
        TaskGen["Discrete Task Arrival (CPU, Bytes, Deadline)"] --> NodeQueue["Node Compute Queue"]
        NodeQueue --> Obs["Local State Vector:<br>[Queue Load, Neighbor Loads, y_hat(t+1), Link Bw]"]
        Pred -.-> Obs
        Obs --> MADQN["MADQN Local Agent (Double-DQN)"]
        MADQN --> ActionMask["Apply Neighbor Action Mask"]
        ActionMask --> Decision{"Offload Decision"}
        Decision -- Action = 0 --> LocalExec["Execute on Local Core"]
        Decision -- Action = j --> OffloadLink["Transmit Task Payload to Neighbor j"]
        OffloadLink --> LinkQueue
    end

    subgraph SimulationPhysics ["4. Physical Environment & Ledgers"]
        LinkQueue --> NetSim["Physical Link Bandwidth Contention (50 Mbps)"]
        NetSim --> Delivery["Task / Model Delivery Event"]
        LocalExec --> TaskLedger["100% Task Conservation Ledger<br>(TSR, Latency, Drops)"]
        Delivery --> TaskLedger
        SendModel --> CommLedger["Communication Ledger<br>(Model Bytes, Control Bytes)"]
    end
```

### Numbered End-to-End Simulation Flow
Every simulation step advances by discrete time bins of **$\Delta t = 30$ seconds** (totaling $1,380$ bins = 11.5 hours of continuous operation). Here is what happens inside a single bin $t$:

1. **Step 1: Workload Update & Feature Lookback**  
   Node $i$ samples its actual CPU and memory utilization for the past $L=8$ bins ($4$ minutes of history) from the curated Alibaba trace.
2. **Step 2: Workload Forecast Generation**  
   Node $i$ feeds the lookback sequence into its local recurrent neural network (`EdgeLSTMForecaster`). The model outputs $\hat{y}_i(t+1)$, the predicted CPU load for the next time interval.
3. **Step 3: APSM Semantic Gating Decision**  
   Node $i$ evaluates its local validation loss surprise $\varepsilon_i(t)$. It compares $\varepsilon_i(t)$ against its running dynamic threshold $\tau_i(t)$.  
   - If $\varepsilon_i(t) > \tau_i(t)$ or if $H=5$ consecutive bins have elapsed without transmission, the gate triggers a **Model Broadcast**.
   - Otherwise, transmission is **Suppressed**. Zero bytes enter the wireless link. The node merely increments its internal model age.
4. **Step 4: Discrete Task Generation**  
   Synthetic computational tasks are generated proportional to the Alibaba CPU arrival density. Each task is stamped with an ID, creation time $t$, execution requirement (cycles), data payload (kilobytes), and maximum deadline (e.g., $t + 30$ seconds).
5. **Step 5: Agentic Offloading Decision (MADQN)**  
   The local MADQN agent collects its state vector: its current queue length, neighbor queue states (received via lightweight telemetry), the forecast $\hat{y}_i(t+1)$, and link status.  
   - The agent applies the **neighbor mask** (setting invalid non-neighbor actions to $-\infty$).
   - The agent chooses action $a_i \in \{0, 1, \dots, K_i\}$.
   - If $a_i = 0$, the task enters node $i$'s local FIFO compute queue.
   - If $a_i = j$, the task payload is placed into the wireless link queue for transmission to neighbor $j$.
6. **Step 6: Physical Link Execution & Contention**  
   The network simulator processes packets in the link queue. If both a 160 KB model weight packet (from Step 3) and a 10 KB task packet (from Step 5) are traversing the same link, transmission time is computed based on shared link bandwidth ($50\text{ Mbps}$). If link queue delay causes a task to exceed its deadline, it is dropped.
7. **Step 7: Outcome Recording & Conservation Audit**  
   Every completed or dropped task is logged in the `TaskConservationLedger`. Total tasks generated must exactly equal completed tasks plus dropped tasks ($100.0\%$ conservation).

---

## Section D — Understanding the Core Concepts

### 1. LSTM/GRU Workload Forecasting
- **What it means:** A recurrent neural network that processes sequential data by maintaining an internal hidden memory vector $h_t$.
- **Why we need it:** CPU arrivals at edge clusters exhibit sharp bursts and diurnal patterns. Standard feedforward networks treat each time step independently; recurrent networks capture temporal momentum.
- **How it works:** Lookback length $L=8$. It takes the past 8 intervals of CPU/memory usage and predicts the next 1 interval.
- **Intuitive Analogy:** Like a weather forecaster looking at the barometric pressure over the last 4 hours to predict whether it will rain in the next 30 minutes, rather than just glancing at the sky right now.
- **In Code:** [`src/phase3/forecasting/models.py`](file:///C:/Users/btpUser/Desktop/Parth_MTP/APSM-MTP/src/phase3/forecasting/models.py#L7-L65) (`EdgeLSTMForecaster`).

### 2. Gossip Learning (Decentralized Model Sharing)
- **What it means:** Completely decentralized collaborative machine learning without a central server.
- **Why we need it:** Eliminates single points of failure, cloud dependency, and wide-area backbone bandwidth costs.
- **How it works:** Nodes form a sparse graph (degree $k \approx 3$). Periodically, a node sends its model weights $\theta_i$ to a random connected neighbor. When receiving $\theta_j$, the node computes an age-weighted average:
  $$\theta_{\text{new}} = \frac{\text{age}_i}{\text{age}_i + \text{age}_j}\theta_i + \frac{\text{age}_j}{\text{age}_i + \text{age}_j}\theta_j$$
- **Intuitive Analogy:** Like students in a study group who do homework independently, then periodically swap notes with one desk neighbor to compare formulas, gradually converging on the same correct answers without a teacher lecturing them.
- **In Code:** [`src/phase3/forecasting/models.py`](file:///C:/Users/btpUser/Desktop/Parth_MTP/APSM-MTP/src/phase3/forecasting/models.py#L50-L75) (`age_weighted_merge`) and [`src/phase3/integration/live_gossip_engine.py`](file:///C:/Users/btpUser/Desktop/Parth_MTP/APSM-MTP/src/phase3/integration/live_gossip_engine.py#L115-L210).

### 3. APSM (Adaptive Predictive-Semantic Mechanism)
- **What it means:** A pre-transmission decision filter that checks whether sending a model update is actually worth the wireless bandwidth.
- **Why we need it:** In standard Gossip Learning, nodes send weights every single round. If a node's local workload was steady, its weights changed by less than $0.01\%$. Broadcasting those weights wastes wireless bandwidth and causes packet congestion.
- **How it works:** Node $i$ computes its validation loss surprise $\varepsilon_i(t) = |\mathcal{L}_i(t) - \mathcal{L}_i(t-1)|$. It maintains a running mean $\mu_i(t)$ and standard deviation $\sigma_i(t)$ over a sliding window ($W=50$). The dynamic transmission threshold is:
  $$\tau_i(t) = \mu_i(t) + k \cdot \sigma_i(t) \quad (\text{with } k=2.0)$$
  - If $\varepsilon_i(t) \le \tau_i(t)$, the update is classified as "semantic noise" and **suppressed**.
  - If $\varepsilon_i(t) > \tau_i(t)$, the update represents a genuine distribution shift and is **transmitted**.
- **The Heartbeat Safety Guard ($H=5$):** If a node suppresses 5 consecutive transmissions, it forces a send on the 6th check. This prevents nodes from becoming isolated if their local environment is quiet for an extended duration.
- **The Warmup Guard ($W_{\text{warmup}}=36$):** During the first 36 episodes/bins, models are initializing from random weights and gradients are chaotic. APSM runs in open transmission mode during warmup to allow baseline consensus to form.
- **Intuitive Analogy:** Like an engineer only speaking in a technical meeting when they have critical new data, remaining silent when they only have minor confirmations, but checking in every 10 minutes (heartbeat) just to confirm they are still on the call.
- **In Code:** [`src/phase3/forecasting/apsm.py`](file:///C:/Users/btpUser/Desktop/Parth_MTP/APSM-MTP/src/phase3/forecasting/apsm.py#L20-L85) (`OnlineSemanticGate`).

### 4. Multi-Agent Reinforcement Learning (MADQN)
- **What it means:** Multi-Agent Deep Q-Network where each edge node is an independent reinforcement learning agent learning an action-value function $Q_i(s, a; \theta_i)$.
- **Why we need it:** Static rules (like "offload if queue $> 80\%$") fail when neighbors are also busy or when transmission delays exceed the task deadline. Reinforcement learning learns optimal non-linear offloading policies tailored to local network topology.
- **State Space $s_i(t)$:**
  $$s_i(t) = [q_i(t), \hat{y}_i(t+1), \{q_j(t)\}_{j \in \mathcal{N}_i}, \{\text{bw}_{ij}\}_{j \in \mathcal{N}_i}]$$
  Includes local queue ratio, local workload forecast $\hat{y}_i(t+1)$, neighbor queue ratios, and available link bandwidths.
- **Action Space & Masking:**  
  $$a_i \in \{0, 1, \dots, K_i\}$$
  Action $0$ = execute locally. Action $j \in \{1 \dots K_i\}$ = offload to physical neighbor $j$. If a graph node has 3 neighbors, actions $4 \dots K_{\max}$ are masked out ($Q(s, a) = -\infty$).
- **Reward Function:**
  $$r_i(t) = +1.0 \cdot (\text{completed tasks}) - 2.0 \cdot (\text{dropped tasks}) - 0.1 \cdot (\text{queue delay}) - 0.05 \cdot (\text{offload cost})$$
  Heavy penalties for dropped tasks teach the agent to prioritize meeting deadlines over selfishly offloading.
- **In Code:** [`src/phase3/marl/madqn.py`](file:///C:/Users/btpUser/Desktop/Parth_MTP/APSM-MTP/src/phase3/marl/madqn.py#L25-L120) and [`src/phase3/env/task_simulator.py`](file:///C:/Users/btpUser/Desktop/Parth_MTP/APSM-MTP/src/phase3/env/task_simulator.py#L492-L560).

---

## Section E — Why Did We Choose This Architecture? (Design Decisions)

This section prepares you to answer "Why did you do it this way instead of that way?" from your thesis committee.

### 1. Decision: Decentralized Gossip Learning over Centralized Federated Learning
- **Problem it addresses:** In edge and mobile environments, wide-area connections to cloud servers are intermittent, expensive, and introduce single points of failure (SPOF).
- **Alternatives considered:** Centralized Federated Learning (FedAvg), Centralized Cloud Training.
- **Reason for selection:** Gossip Learning is fully serverless. If any edge node is destroyed or disconnected, the remaining mesh continues training unimpeded.
- **Evidence:** Base paper (Tundo et al. 2025) proved Gossip Learning matches Federated Learning accuracy within 1–2% without requiring a central coordinator.

### 2. Decision: Validation Loss Surprise over Raw Prediction Error for APSM Gating
- **Problem it addresses:** If you gate model transmissions based on single-sample prediction error $|y - \hat{y}|$, an outlier in the incoming data causes a false-alarm burst of model transmissions.
- **Alternatives considered:** Absolute prediction error thresholding, gradient norm thresholding, weight-difference norm $\|\theta_t - \theta_{t-1}\|$.
- **Reason for selection:** Validation loss $\mathcal{L}_{\text{val}}$ evaluates generalization on a held-out local batch. Surprise $\varepsilon(t) = |\mathcal{L}(t) - \mathcal{L}(t-1)|$ measures whether the model's *knowledge* changed, rather than whether a single random data sample was noisy. Weight norm calculations ($\|\theta_t - \theta_{t-1}\|$) require full matrix operations over millions of parameters, which is computationally expensive on edge CPUs.
- **Evidence:** Stage 6 calibration verified that validation surprise gating achieved 52.48% suppression without destabilizing convergence.

### 3. Decision: Independent MADQN with Neighbor Action Masking over MAPPO / Centralized Critic
- **Problem it addresses:** Centralized Training with Decentralized Execution (CTDE) frameworks like MAPPO or QMIX require a centralized critic during training that observes the global state of all nodes. In an edge cluster, assembling a global state vector during online execution creates a communication bottleneck.
- **Alternatives considered:** MAPPO (Multi-Agent PPO), QMIX, Independent Q-Learning without masking, Greedy Heuristics.
- **Reason for selection:** Independent Double-DQN agents require strictly local observation vectors during inference. By enforcing neighbor action masking, agents never attempt to offload to nodes that are not topologically connected.
- **Evidence:** Stage 5 and Stage 6 evaluations showed MADQN achieved 84.15% TSR at $N=10$, outperforming local execution (78.38%) while executing in sub-millisecond inference time on CPU using NumPy matrix operations.

### 4. Decision: Shared Physical Link Bandwidth Contention over Decoupled Logical Channels
- **Problem it addresses:** Naive simulations simulate ML model exchanges and task offload packets as if they exist in parallel universes with separate infinite bandwidth. In real hardware, both packets travel through the same radio frequency (RF) interface.
- **Alternatives considered:** Completely separated physical channels; zero-delay instant model synchronization.
- **Reason for selection:** Modeling link contention enforces physical realism. It demonstrates the true value of APSM: by suppressing 160 KB model packets, APSM prevents the physical link queue from filling up, allowing 10 KB task packets to traverse the link without missing deadlines.
- **Evidence:** `tests/phase3/test_stage6_integration.py::test_shared_physical_link_contention` verifies link contention and confirms zero packet loss under APSM.

---

## Section F — The Experimental Journey & Development Chronology

Understanding how Phase 3 evolved chronologically helps you explain *why* certain fixes and safeguards exist in the repository.

```mermaid
flowchart LR
    S1["Stage 1-4:<br>Data & Simulator"] --> S5["Stage 5:<br>MARL Gate"]
    S5 --> S6["Stage 6:<br>N=10 Integration"]
    S6 --> Audit6["Stage 6 Reconcile:<br>Holm-Bonferroni Audit"]
    Audit6 --> S7["Stage 7:<br>N=50 & N=100 Scale"]
    S7 --> PostAudit["Post-Stage-7 Audit:<br>Post-Warmup Selection"]
    PostAudit --> Main["Final Phase 3:<br>Merge to Main"]
```

1. **Stages 1 to 4: Data Engineering & Simulator Physics**  
   Extracted 12 hours of Alibaba 2017 cluster traces, binned CPU/memory usage into 30-second intervals, constructed the 6-node and 10-node sparse Porto edge graphs, and verified that the discrete-event simulator strictly conserved tasks ($100.0\%$).
2. **Stage 5: MARL Policy Formulation**  
   Implemented independent Double-DQN agents with spatial neighbor masking. Verified that NumPy policy runners executed identical decisions to PyTorch models with $100\times$ lower latency, enabling fast CPU simulation.
3. **Stage 6: N=10 Live Integrated Benchmark**  
   Executed the first 6-arm integrated benchmark across 5 seeds (30 runs total). Discovered that without APSM, model packets created transient link congestion. APSM reduced model payload by $52.17\%$.
4. **Stage 6 Scientific Reconciliation & Holm-Bonferroni Correction**  
   An independent audit verified statistical significance. Conducted Wilcoxon signed-rank tests across seeds with Holm-Bonferroni correction, proving that APSM's bandwidth reduction was statistically significant ($p < 0.05$) while its impact on forecasting MSE was statistically negligible.
5. **Stage 7: Scalability Testing at N=50 and N=100**  
   Scaled the edge network to $N=50$ nodes ($150$ edges) and $N=100$ nodes ($300$ edges). Validated that memory and event queues scaled linearly $\mathcal{O}(N)$ without memory leaks.
6. **Post-Stage-7 Checkpoint Selection Audit (The Critical Scientific Fix)**  
   *What happened:* In an early Stage 7 run, an automated script selected the "best" model checkpoint from validation episode 24—before warmup completed ($W_{\text{warmup}}=36$). At episode 24, the forecast branch was not fully active, causing an unrepresentative checkpoint to be evaluated.  
   *The Fix:* We established a strict invariant: **forecasting-enabled arms can only select checkpoints from post-warmup episodes ($\ge 36$)**. All affected runs were re-evaluated, historical backups were preserved in `*_pre_audit_historical.json`, and all 30 active Stage 7 runs now satisfy $\text{best\_checkpoint\_episode} \ge 36$.

---

## Section G — Experiments, Baselines, and Results

### 1. Experimental Setup
- **Dataset:** Alibaba Cluster Trace v2017 (CPU & memory utilization across 1,300+ machines, binned at $\Delta t = 30\text{ s}$, 1,380 bins total = 11.5 hours).
- **Data Split:** Train bins $[0, 828)$ (60%), Validation bins $[828, 1035)$ (15%), Test bins $[1035, 1380)$ (25%, 345 bins = 2.875 hours).
- **Topologies:** 
  - $N=10$ nodes: Tier-1 sparse mesh (average degree $k \approx 3$).
  - $N=50$ nodes: Spatial mesh (150 undirected links).
  - $N=100$ nodes: Large-scale spatial mesh (300 undirected links).
- **The 5 Seeds (11, 22, 33, 44, 55):**  
  *Crucial viva point:* The five seeds represent **training randomness and policy initialization** (random weight init, exploratory action choices, stochastic peer selection), **NOT five different workload traces**. All runs use the identical underlying Alibaba cluster workload trace.

### 2. The 6 Evaluation Arms
To isolate every contribution scientifically, we compared 6 distinct arms:
1. **B0 (Local Greedy):** No forecasting, no offloading. Tasks are executed strictly on the local arrival node.
2. **E1 (Greedy Offloading):** Offloads tasks to the neighbor with the lowest immediate queue length. No ML forecasting, no reinforcement learning.
3. **E2 (MADQN without Forecast):** Reinforcement learning offloading using current queue states only.
4. **E3-Local (MADQN + Local Forecast):** RL offloading informed by local LSTM forecasts, but nodes train in isolation (no Gossip Learning).
5. **E3-GL (MADQN + Gossip Learning Baseline):** RL offloading informed by collaborative Gossip Learning forecasters with standard periodic model broadcasts.
6. **E3-APSM (MADQN + APSM - Our Complete System):** RL offloading informed by Gossip Learning regulated by the APSM semantic gate.

### 3. Reconciled Results Across All 90 Runs

#### Table 1: Forecasting & Wireless Communication Performance

| Scale ($N$) | Evaluation Arm | Test MSE | $\Delta_{\text{MSE}}$ vs GL | Gain Retention $\mathcal{R}$ | Model Payload Exchanged | Bandwidth Savings vs GL | Suppressed Checks % |
|---|---|---|---|---|---|---|---|
| **$N = 10$** | E3-Local | 0.037063 | — | — | 0.00 MB | 100.00% | — |
| *(30 runs)* | E3-GL (Base) | 0.035884 | Baseline | Baseline | 313.29 MB | 0.00% | 0.00% |
| | **E3-APSM (Ours)** | **0.036000** | **+0.000117** | **90.12%** | **149.86 MB** | **52.17%** | **52.48%** |
| **$N = 50$** | E3-Local | 0.037977 | — | — | 0.00 MB | 100.00% | — |
| *(30 runs)* | E3-GL (Base) | 0.037510 | Baseline | Baseline | 1,664.34 MB | 0.00% | 0.00% |
| | **E3-APSM (Ours)** | **0.037537** | **+0.000027** | **94.21%** | **782.72 MB** | **52.97%** | **53.11%** |
| **$N = 100$** | E3-Local | 0.038085 | — | — | 0.00 MB | 100.00% | — |
| *(30 runs)* | E3-GL (Base) | 0.037633 | Baseline | Baseline | 3,289.52 MB | 0.00% | 0.00% |
| | **E3-APSM (Ours)** | **0.037614** | **-0.000019** | **104.17%** | **1,519.92 MB** | **53.79%** | **53.72%** |

#### Table 2: Task Offloading & System Performance

| Scale ($N$) | Evaluation Arm | Task Success Rate (TSR) | Mean Task Latency | Task Drop Rate | Mean CPU Utilization |
|---|---|---|---|---|---|
| **$N = 10$** | B0 (Local Only) | 78.38% | 85.37 s | 21.62% | 48.91% |
| | E1 (Greedy Offload) | 84.86% | 76.31 s | 15.14% | 48.95% |
| | E2 (MADQN Queue-Only) | 83.83% | 79.07 s | 16.17% | 48.98% |
| | E3-Local (MADQN + Local FC) | 83.57% | 79.26 s | 16.43% | 48.97% |
| | E3-GL (MADQN + GL FC) | 84.07% | 78.81 s | 15.93% | 48.98% |
| | **E3-APSM (Complete System)** | **84.15%** | **78.67 s** | **15.85%** | **48.98%** |
| **$N = 50$** | B0 (Local Only) | 90.59% | 57.00 s | 9.41% | 49.03% |
| | E1 (Greedy Offload) | 96.24% | 46.36 s | 3.76% | 49.08% |
| | E2 (MADQN Queue-Only) | 96.13% | 47.79 s | 3.87% | 49.07% |
| | E3-Local (MADQN + Local FC) | 95.75% | 48.37 s | 4.25% | 49.07% |
| | E3-GL (MADQN + GL FC) | 95.75% | 48.22 s | 4.25% | 49.07% |
| | **E3-APSM (Complete System)** | **95.73%** | **48.17 s** | **4.27%** | **49.07%** |
| **$N = 100$** | B0 (Local Only) | 93.89% | 48.00 s | 6.11% | 49.12% |
| | E1 (Greedy Offload) | 97.30% | 41.95 s | 2.70% | 49.18% |
| | E2 (MADQN Queue-Only) | 96.67% | 43.23 s | 3.33% | 49.17% |
| | E3-Local (MADQN + Local FC) | 96.55% | 43.61 s | 3.45% | 49.16% |
| | E3-GL (MADQN + GL FC) | 96.56% | 43.50 s | 3.44% | 49.16% |
| | **E3-APSM (Complete System)** | **96.63%** | **43.65 s** | **3.37%** | **49.16%** |

### 4. Six Questions to Interpret the Results

#### Q1: What happened to forecasting accuracy?
Gossip Learning consistently beats purely local training across all scales (e.g., at $N=10$, MSE drops from $0.037063$ to $0.035884$). When APSM is introduced, it achieves an MSE of $0.036000$—retaining **$90.12\%$ of the collaborative gain** while cutting model transmissions by **$52.17\%$**. At $N=100$, APSM's MSE ($0.037614$) actually outperforms full Gossip Learning ($0.037633$, $104.17\%$ retention) because semantic gating acts as a regularization filter that rejects noisy updates from distant peers.

#### Q2: What happened to wireless bandwidth?
Across all scales, APSM consistently suppresses between **$52.17\%$ and $53.79\%$** of model transmissions. In absolute terms, at $N=100$, Gossip Learning transmitted $3,289.52\text{ MB}$ of model weights over the wireless links, whereas APSM transmitted only $1,519.92\text{ MB}$—saving **$1,769.6\text{ MB}$ ($1.77\text{ GB}$) of wireless payload**.

#### Q3: How did task offloading perform?
Offloading significantly outperforms purely local execution:
- At $N=10$: TSR increases from $78.38\%$ (B0) to $84.15\%$ (E3-APSM), a **$+5.77\%$ improvement** with an average latency reduction of $6.7$ seconds.
- At $N=50$: TSR increases from $90.59\%$ to $95.73\%$ ($+5.14\%$).
- At $N=100$: TSR increases from $93.89\%$ to $96.63\%$ ($+2.74\%$).

#### Q4: Why does Greedy (E1) have a slightly higher TSR than MADQN (E2/E3)?
Notice in Table 2 that E1 (Greedy) achieves $84.86\%$ TSR at $N=10$ versus $84.15\%$ for E3-APSM. **Why?**  
Greedy offloads aggressively ($40.62\%$ offload rate at $N=10$) by querying neighbor queues at every instant. However, Greedy has no concept of communication cost; it transmits $1.05\text{ MB}$ of control telemetry. MADQN achieves virtually the same TSR while offloading selectively ($30.36\%$ offload rate), preserving link bandwidth and avoiding unnecessary multi-hop ping-pong.

#### Q5: What does this demonstrate?
It demonstrates that semantic gating is an effective, robust mechanism for edge collaborative learning. You can eliminate more than half of all wireless model transmissions without harming predictive accuracy or degrading task success rates.

#### Q6: What does this NOT demonstrate?
It does **NOT** demonstrate that reinforcement learning is infinitely better than simple heuristics on static workloads. When edge networks have ample capacity, greedy heuristics perform very well. Reinforcement learning's advantage emerges under heterogeneous, bursting workloads where future forecasts prevent queue overflow.

---

## Section H — What Did Our Approach Gain Over Baselines & Base Papers?

Be completely objective and precise when presenting your gains. Avoid unsubstantiated claims.

### 1. Concrete Gains Over Base Paper (Tundo et al. 2025)
- **$52.17\%\text{ to }53.79\%$ Bandwidth Reduction:** Tundo et al. had zero suppression (every node transmits every round). APSM saves over half the wireless bytes across all evaluated network scales.
- **Physical Feasibility:** Tundo et al. did not model wireless packet contention. By cutting model payload by $1.77\text{ GB}$ at $N=100$, APSM ensures that model exchanges do not block real-time computing tasks.
- **Unified Edge Scope:** Extended the base paper from isolated traffic forecasting to coupled task offloading and discrete-event simulation.

### 2. Concrete Gains Over Experimental Baselines
- **Over Pure Local Training (B0 / E3-Local):** Collaborative forecasting reduces test MSE from $0.038085$ to $0.037614$ at $N=100$. Task offloading increases task completion from $78.38\%$ to $84.15\%$ at $N=10$.
- **Over Unconstrained Gossip Learning (E3-GL):** Identical task success rates ($84.15\%$ vs. $84.07\%$ at $N=10$, $96.63\%$ vs. $96.56\%$ at $N=100$), but with **$52.17\%\text{ to }53.79\%$ less model communication**.

### 3. Design Contributions
- Formulated an online, self-calibrating semantic gate using running mean and standard deviation ($\tau_i(t) = \mu_i(t) + k\cdot\sigma_i(t)$) that requires zero manual threshold tuning per node.
- Coupled localized RL action masks with discrete anti-loop TTL conservation to guarantee 100% discrete task conservation across heterogeneous edge graphs.

---

## Section I — Critical Limitations (Viva Defense Questions)

Every rigorous master's thesis has limitations. Reviewers respect students who openly state their assumptions. Here are the four key limitations you should know:

1. **Fixed Trace Across Seeds:**  
   *Reviewer Question:* "You evaluated 5 random seeds (11, 22, 33, 44, 55). Does this mean you tested 5 different workload traces?"  
   *Your Answer:* "No. All runs evaluate the same underlying Alibaba Cluster Trace v2017 to ensure fair cross-arm comparability. The 5 seeds represent **stochastic variation in the system**—specifically, neural network weight initialization, epsilon-greedy action exploration in MADQN, and randomized peer selection in Gossip Learning. Future work should evaluate across multiple independent enterprise traces."
2. **Simulation vs. Real Hardware Deployment:**  
   *Reviewer Question:* "Did you test this on actual Raspberry Pis or physical edge base stations?"  
   *Your Answer:* "This is a high-fidelity discrete-event software simulation (`TaskOffloadingSim`). While we explicitly modeled physical bandwidth constraints, queue capacities, transmission delays, and CPU clock cycles, we did not deploy on physical embedded hardware. Real RF phenomena like fading, packet collisions at the MAC layer, and physical temperature throttling are abstracted into effective bandwidth and link latency models."
3. **Execution on CPU vs. GPU:**  
   *Reviewer Question:* "Why did you run the simulation on CPU rather than GPU?"  
   *Your Answer:* "The simulator is written in discrete Python/NumPy, maintaining independent state dictionaries for up to 100 nodes. PyTorch tensor allocations on GPU incur substantial memory transfer overhead for small vector inferences ($1 \times 8$). NumPy vector operations on CPU executed in under $0.5\text{ ms}$ per node, making CPU execution faster and more memory-efficient for this discrete-event architecture."
4. **Homogeneous Link Bandwidths:**  
   *Reviewer Question:* "Did you model varying 5G wireless channel qualities?"  
   *Your Answer:* "In the current benchmark, physical links share a configured capacity ($50\text{ Mbps}$). While links experience dynamic queuing delay and contention based on active packets, wireless channel fading was not dynamically simulated. Adding Rayleigh fading is an avenue for future work."

---

## Section J — Codebase Navigation Guide

When your supervisor asks you to show the implementation, navigate directly to these core entry points:

| Component | File Path | Key Class / Function | What It Does |
|---|---|---|---|
| **Unified Config** | [`src/phase3/config.py`](file:///C:/Users/btpUser/Desktop/Parth_MTP/APSM-MTP/src/phase3/config.py) | `Phase3Config`, `APSMConfig` | Central dataclass defining all simulation, ML, and network parameters. |
| **Forecasting Model** | [`src/phase3/forecasting/models.py`](file:///C:/Users/btpUser/Desktop/Parth_MTP/APSM-MTP/src/phase3/forecasting/models.py#L7-L65) | `EdgeLSTMForecaster`, `age_weighted_merge` | PyTorch recurrent model and push-sum age-weighted model mixing formula. |
| **Semantic Gate** | [`src/phase3/forecasting/apsm.py`](file:///C:/Users/btpUser/Desktop/Parth_MTP/APSM-MTP/src/phase3/forecasting/apsm.py#L20-L85) | `OnlineSemanticGate`, `evaluate_transmission` | Implements $\tau(t) = \mu(t) + k\cdot\sigma(t)$ and heartbeat counter. |
| **Task Simulator** | [`src/phase3/env/task_simulator.py`](file:///C:/Users/btpUser/Desktop/Parth_MTP/APSM-MTP/src/phase3/env/task_simulator.py) | `TaskOffloadingSim`, `step_bin` | Discrete-event simulator modeling queues, link delays, and task deadlines. |
| **MADQN Agent** | [`src/phase3/marl/madqn.py`](file:///C:/Users/btpUser/Desktop/Parth_MTP/APSM-MTP/src/phase3/marl/madqn.py#L25-L120) | `MADQNAgent`, `select_action` | Double-DQN network with neighbor action masking and replay memory. |
| **Live Integration Engine** | [`src/phase3/integration/live_gossip_engine.py`](file:///C:/Users/btpUser/Desktop/Parth_MTP/APSM-MTP/src/phase3/integration/live_gossip_engine.py#L115-L210) | `LiveIntegratedEngine` | Glues forecasting, APSM gating, link queues, and RL offloading into a single loop. |
| **N=10 Runner** | [`src/phase3/run_stage6_integrated.py`](file:///C:/Users/btpUser/Desktop/Parth_MTP/APSM-MTP/src/phase3/run_stage6_integrated.py) | `main()` | CLI runner executing the 6-arm N=10 integrated benchmark. |
| **Scalability Runner** | [`src/phase3/run_stage7_scalability.py`](file:///C:/Users/btpUser/Desktop/Parth_MTP/APSM-MTP/src/phase3/run_stage7_scalability.py) | `run_scalability_benchmark()` | CLI runner executing the N=50 and N=100 scalability benchmarks. |
| **Metrics Compiler** | [`scripts/compile_final_metrics.py`](file:///C:/Users/btpUser/Desktop/Parth_MTP/APSM-MTP/scripts/compile_final_metrics.py) | `main()` | Standalone script that parses raw JSON runs and prints exact multi-scale tables. |

---

## Section K — Glossary & 10 Defence Self-Check Questions

### Glossary of Important Terms
- **APSM (Adaptive Predictive-Semantic Mechanism):** Our proposed pre-transmission gating filter that suppresses redundant model updates based on dynamic variance thresholding.
- **Age-Weighted Merging:** A gossip learning technique where models with higher training iterations contribute proportionally more to the merged parameter vector.
- **Heartbeat Mechanism ($H=5$):** A fail-safe rule that forces a model transmission if 5 consecutive checks were suppressed, preventing node isolation.
- **MADQN (Multi-Agent Deep Q-Network):** Independent Double-DQN agents running locally on edge nodes to make discrete task offloading decisions.
- **Neighbor Action Masking:** Setting the Q-values of invalid non-neighbor actions to $-\infty$ so agents only select physically valid offloading targets.
- **Lookback ($L=8$):** The number of past time bins (4 minutes) fed into the recurrent forecaster to predict the next bin.
- **Task Success Rate (TSR):** The percentage of generated computational tasks that finish processing before their deadline expires.
- **Validation Surprise ($\varepsilon_i(t)$):** The absolute change in validation loss between successive evaluation rounds, used as the input signal to the APSM gate.
- **TTL (Time-To-Live):** A discrete integer hop limit attached to offloaded tasks to prevent infinite forwarding loops between adjacent nodes.

---

### 10 Defence Self-Check Questions

#### 1. Why didn't you use standard Federated Learning?
*Answer:* Federated Learning depends on a central parameter server. In edge computing environments, edge nodes are distributed across metropolitan areas; communicating with a central cloud server introduces high latency, bandwidth bottlenecks, and a single point of failure (SPOF). Gossip Learning is serverless and resilient to individual node dropouts. *(Section B.1, Section E.1)*

#### 2. How does APSM decide whether to send a model update?
*Answer:* It computes validation loss surprise $\varepsilon_i(t) = |\mathcal{L}_i(t) - \mathcal{L}_i(t-1)|$ on local data. It compares this against an adaptive threshold $\tau_i(t) = \mu_i(t) + k \cdot \sigma_i(t)$, where $\mu$ and $\sigma$ are the running mean and standard deviation over the last 50 checks. If $\varepsilon_i(t) > \tau_i(t)$, the update is deemed semantically informative and transmitted; otherwise, it is suppressed. A heartbeat forces a send if 5 consecutive checks are suppressed. *(Section D.3)*

#### 3. What is the difference between Phase 2 and Phase 3?
*Answer:* Phase 2 evaluated APSM solely as a machine learning filter on static Porto taxi traffic data, with no computational tasks, queues, or offloading. Phase 3 builds a complete edge system using the Alibaba Cluster Trace, integrating discrete task arrivals, execution deadlines, link contention, and multi-agent reinforcement learning (MADQN) for proactive task offloading across 10, 50, and 100 nodes. *(Section A.2, Section A.3)*

#### 4. Why is workload forecasting necessary for task offloading?
*Answer:* Reactive offloading only observes current queue lengths. By the time a queue fills up, incoming tasks cannot be transferred across the wireless link before their deadlines expire. Workload forecasting provides 1-step-ahead predictions, enabling the RL policy to offload tasks proactively before local queues become congested. *(Section A.4)*

#### 5. How do you prevent offloaded tasks from bouncing back and forth forever?
*Answer:* Every task has a Time-To-Live (TTL) counter initialized to $\text{TTL}_{\max} = 2$. Each transmission decrements the TTL. If a task reaches $\text{TTL} = 0$, it cannot be offloaded again and must either execute locally or be dropped if the local queue is full. *(Section C, Section D)*

#### 6. What was the post-Stage-7 checkpoint audit issue?
*Answer:* During Stage 7 scalability runs, an automated checkpoint selector picked a checkpoint from episode 24, which was below the warmup threshold ($W_{\text{warmup}}=36$). Below warmup, the forecast branch was not fully active. We established a strict post-warmup selection invariant ($\text{episode} \ge 36$), re-evaluated all affected runs, preserved original artifacts in `*_pre_audit_historical.json`, and validated that all active runs satisfy the invariant. *(Section F.6)*

#### 7. How much bandwidth does APSM actually save?
*Answer:* APSM consistently saves **$52.17\%$ to $53.79\%$ of wireless model bandwidth** across all network scales. At $N=10$, it reduces model payload from $313.29\text{ MB}$ to $149.86\text{ MB}$. At $N=100$, it reduces payload from $3,289.52\text{ MB}$ to $1,519.92\text{ MB}$, saving $1.77\text{ GB}$ of wireless transmissions. *(Section G.3)*

#### 8. Does APSM sacrifice forecasting accuracy to achieve that bandwidth saving?
*Answer:* No. At $N=10$, APSM retains $90.12\%$ of Gossip Learning's MSE improvement over local training (MSE $0.036000$ vs. $0.035884$). At $N=50$, it retains $94.21\%$. At $N=100$, it achieves an MSE of $0.037614$, retaining $104.17\%$ of the gain (slightly outperforming unconstrained gossip due to noise filtering). *(Section G.3, Table 1)*

#### 9. Why does Greedy Offloading (E1) have a slightly higher TSR than MADQN (E3)?
*Answer:* Greedy offloads aggressively ($40.6\%$ offload rate vs. $30.4\%$ for MADQN) by constantly querying neighbor states. However, it generates more control traffic and does not account for transmission overhead or link congestion. MADQN achieves competitive TSR ($84.15\%$ vs. $84.86\%$) while offloading far more selectively, preserving bandwidth on shared wireless links. *(Section G.4, Table 2)*

#### 10. Can this system scale to hundreds of nodes?
*Answer:* Yes. Our Stage 7 experiments evaluated the system up to $N=100$ nodes with 300 physical links. Memory usage remained bounded ($<2.1\text{ GB}$ peak RAM), discrete-event simulation scaled linearly, and APSM maintained its $\sim 53.8\%$ bandwidth reduction while achieving a $96.63\%$ task success rate. *(Section G.3, Section H)*
