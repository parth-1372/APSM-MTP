# Phase 3 Stage 6: Failure Modes, Edge Case Analysis & Research Limitations

**Repository:** `APSM-MTP`  
**Branch:** `stage6-integrated-10n`  
**Date:** November 13, 2026  
**Topology:** Porto 10-Node Undirected Mesh ($k=3$)  

---

## 1. Identified Failure Modes & Edge Boundary Conditions

In decentralized edge computing systems with dynamic workload offloading, several critical failure regimes emerge under extreme conditions:

### 1.1 The "Herd Offloading" Oscillation Failure
- **Mechanism:** When a burst of tasks arrives at node $i$, if $i$ determines that peer $j$ has low utilization based on a 15-second-old beacon, node $i$ may offload multiple tasks in rapid succession to $j$.
- **Failure Consequence:** Before node $j$ can process the tasks or emit an updated beacon reflecting its rising queue, peer $j$'s FIFO queue saturates ($Q_j \ge Q_{\text{max}} = 48$). Subsequent offloaded tasks from node $i$ are dropped due to admission overflow (`admission_overflow`), converting useful offloading into packet loss.
- **Mitigation in Stage 6:**
  1. **Self-Inflight Discounting:** Both Reactive Greedy (E1) and MADQN (E2/E3) track `inflight_records[origin_id][peer_id]`. As tasks are dispatched onto the wire, the local observation immediately marks those peer slots as partially occupied (`obs[base + 6]`), damping herd behavior.
  2. **Hysteresis Thresholds:** In E1, offloading requires peer estimated completion time to beat local execution by at least $1.2\times$, preventing borderline offloading churn.

### 1.2 Physical Link Congestion & Wire Timeout
- **Mechanism:** Wireless link bandwidth varies between $50\text{ Mbps}$ and $200\text{ Mbps}$. When multiple tasks with large payloads ($500\text{ KB} - 1\text{ MB}$) are serialized back-to-back, link queueing delay accumulates.
- **Failure Consequence:** If link delay exceeds the task's slack budget ($\Delta t_{\text{net}} > \kappa \cdot t_{\text{exec}} + \text{slack}$), the task expires while in transit over the physical wireless link (`link_timeout`).
- **Mitigation in Stage 6:** The simulator enforces strict wire serialization (`link_free_at[i, j]`). If the link is already busy, the policy observes non-zero in-flight queueing, discouraging further offloading across congested links.

### 1.3 Asymmetric Graph Degree Disparities
- **Topology Property:** In the Porto 10-node mesh ($k=3$), nodes 1 and 9 have degree 4, while nodes 0, 2, 3, 4, 5, 6, 7, 8 have degree 3.
- **Failure Consequence:** Degree-4 nodes act as natural topological hubs. If peripheral nodes blindly offload to degree-4 nodes, those hub nodes become computational bottlenecks.
- **Mitigation in Stage 6:** The MADQN agent employs permutation-invariant peer pooling with per-peer action masking, learning to distribute load based on actual peer availability rather than fixed node index preferences.

---

## 2. Scientific & System Limitations

While Stage 6 represents an end-to-end integrated research benchmark, the following fundamental limitations should be acknowledged:
1. **Single-Hop Offloading Constraint:** Tasks are constrained to `max_hops = 1`. Multi-hop routing ($h \ge 2$) is excluded to prevent cyclic routing loops and compounding network delays.
2. **Homogeneous Task Preemption Model:** Tasks are processed using non-preemptive CFS-style proportional sharing. Priority preemption of urgent tasks is not modeled.
3. **Synthetic Link Quality Fluctuation:** Link RTT and bandwidth are drawn from realistic empirical distributions ($5-40\text{ ms}$, $50-200\text{ Mbps}$), but dynamic wireless channel fading (e.g. Rayleigh/Rician fast fading) is not dynamically recomputed per second.
4. **Offline Trace Replay:** Workload demand is sourced from Alibaba's production cluster trace. While non-stationary and bursty, trace-driven arrival scheduling lacks interactive closed-loop user responsiveness.

