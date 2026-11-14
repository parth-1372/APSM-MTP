# Phase 3 Stage 6: Multi-Channel Communication Accounting & Reconciliation

**Repository:** `APSM-MTP`  
**Branch:** `stage6-integrated-10n`  
**Date:** November 13, 2026  
**Scope:** Strictly $N = 10$ Nodes Only  

---

## 1. Principles of Physical Channel Separation

In distributed edge computing, a common methodological pitfall is allowing background model synchronization to secretly piggyback operational status telemetry, or conversely, treating large neural network model exchanges as zero-cost virtual operations.

Stage 6 implements **strict logical channel separation** coupled with **unified physical link accounting**:

```
+-------------------------------------------------------------------------------+
|                      SHARED PHYSICAL WIRELESS CHANNEL                         |
|             Total Bandwidth Consumption = Control Plane + Data Plane          |
+-------------------------------------------------------------------------------+
       |                               |                       |
       v                               v                       v
[Channel 1: Model Gossip]   [Channel 2: Telemetry]     [Channel 3: Data Offloading]
- Cadence: 30s-60s          - Cadence: 15s Periodic    - Event: Real-time Arrival
- Payload: 56,920 Bytes     - Payload: 32 Bytes        - Payload: 50 KB - 1 MB
- Content: State Dict Only  - Content: Q, CPU, Mem, y   - Content: Task Bytes
- Gating: APSM-H1 Residual  - Scope: 1-Hop Neighbors   - Scope: 1-Hop Hop_Count=1
```

---

## 2. Channel Specifications & Byte Formats

### 2.1 Channel 1: Decentralized Model Gossip Channel
- **Message Type:** Serialized PyTorch parameter weights (`state_dict`).
- **Network Protocol:** P2P Broadcast to physical 1-hop neighbors ($k \le 4$).
- **Message Size:**
  - $14,224$ float32 weights $\times 4\text{ bytes} = 56,896\text{ bytes}$
  - Header overhead $= 24\text{ bytes}$
  - **Total Serialized Payload:** $56,920\text{ bytes}$ ($\approx 56.9\text{ KB}$).
- **Link Delay Modeling:**
  $$\Delta t_{\text{model}} = \frac{1}{2}\text{RTT}_{ij} + \frac{8 \times 56{,}920}{\text{BW}_{ij}}$$
  Under Porto link parameters ($\text{RTT} \in [5, 40]\text{ms}$, $\text{BW} \in [50, 200]\text{Mbps}$), transmission delay is $\approx 2.3\text{ms}$ to $9.1\text{ms}$, safely completing well before the next 30-second bin boundary.
- **APSM-H1 Gating Rule:**
  A broadcast is triggered if and only if:
  $$H_1(t) = |y_i(t) - \hat{y}_i(t|t-1)| > \tau_i(t) \quad \text{or} \quad \text{streak} \ge H_{\text{max}}$$
  Otherwise, transmission is completely suppressed ($0\text{ bytes}$ transmitted).

### 2.2 Channel 2: Operational Telemetry Channel
- **Message Type:** High-frequency UDP neighbor beacons.
- **Cadence:** Fixed 15-second interval ($T_{\text{telemetry}} = 15.0\text{s}$).
- **Payload Format (32 Bytes):**
  1. `sender_id` (int32, 4 bytes)
  2. `timestamp` (float64, 8 bytes)
  3. `q_len_norm` (float32, 4 bytes)
  4. `avail_cpu_norm` (float32, 4 bytes)
  5. `avail_mem_norm` (float32, 4 bytes)
  6. `forecast_y_hat` (float32, 4 bytes)
  7. `checksum` (uint32, 4 bytes)
- **Causal Guarantee:** Beacons incur simulated wireless transit time; receiving nodes only see status from `last_rx_ts <= t_now_s`. Zero instantaneous state leakage.

### 2.3 Channel 3: Application Task Offload Data Plane
- **Message Type:** Task computational execution binary and input dataset.
- **Payload Range:** Log-uniform synthetic distribution in $[50{,}000, 1{,}000{,}000]\text{ bytes}$ ($50\text{ KB}$ to $1\text{ MB}$).
- **FIFO Link Contention:** Modeled with physical link queueing: subsequent tasks wait until preceding transmissions finish (`link_free_at[i, j]`).

### 2.4 Channel 4: Acknowledgment (ACK) Channel
- **Message Type:** Task resolution confirmation returning to origin host.
- **Payload Size:** $16\text{ bytes}$ per completed or dropped offloaded task.

---

## 3. Communication Cost Ledger Equations

The multi-channel communication ledger computes:
$$\text{CB}_{\text{control}} = \text{CB}_{\text{model}} + \text{CB}_{\text{telemetry}} + \text{CB}_{\text{ack}} + \text{CB}_{\text{header}}$$
$$\text{CB}_{\text{system}} = \text{CB}_{\text{control}} + \text{CB}_{\text{data\_plane}}$$

Where:
- $\text{CB}_{\text{model}} = N_{\text{model\_pkts}} \times 56{,}920\text{ bytes}$.
- $\text{CB}_{\text{telemetry}} = N_{\text{tel\_pkts}} \times 32\text{ bytes}$.
- $\text{CB}_{\text{ack}} = N_{\text{ack\_pkts}} \times 16\text{ bytes}$.
- $\text{CB}_{\text{header}} = N_{\text{offloads}} \times 32\text{ bytes}$.
- $\text{CB}_{\text{data\_plane}} = \sum_{\text{offloads}} \text{payload\_bytes}$.

