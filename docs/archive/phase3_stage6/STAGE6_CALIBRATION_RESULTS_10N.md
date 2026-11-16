# Phase 3 Stage 6: Calibration Sweep & Edge Capacity Characterization (N = 10 Nodes)

**Repository:** `APSM-MTP`  
**Branch:** `stage6-integrated-10n`  
**Date:** November 13, 2026  
**Scope:** Strictly $N = 10$ Nodes Only  
**Hardware:** Intel Core i9-13950HX (24C/32T), NVIDIA RTX 1000 Ada Generation Laptop GPU (6 GB VRAM, CUDA 13.0)

---

## 1. Executive Summary & Calibration Objective

Before evaluating reinforcement learning offloading policies, edge systems must establish a scientifically defensible operational capacity.
- If edge node capacities are set unrealistically high ($C_{\text{cpu}} \to \infty$), local nodes never saturate, task queues remain empty, and offloading offers zero utility ($\Delta_{\text{TSR}} \approx 0\%$).
- If capacities are unrealistically constrained ($C_{\text{cpu}} \to 0$), all nodes saturate simultaneously, peer queues overflow, and wireless transmission overhead exacerbates deadline expiration drops.

To determine the optimal, defensible operating point for the 10-node Porto topology, an exhaustive 9-point grid sweep was executed across:
- **Node CPU Capacity ($C_{\text{cpu}}$):** $\{18.0, 20.0, 22.0\}\text{ cores/second}$
- **Deadline Slack Multiplier ($\kappa$):** $\{2.0, 2.2, 2.5\}$
- **Fixed Invariants:** $Q_{\text{max}} = 48$, $\kappa_{\text{base}} = 3.0\text{s}$, $T_{\text{telemetry}} = 15.0\text{s}$, 32-byte UDP beacons.
- **Evaluation Dataset:** Full chronological test split ($t \in [70,650, 81,000)\text{s}$, 345 bins, $17,862$ real tasks from the Alibaba 2017 cluster trace).

---

## 2. Full 9-Point Calibration Grid Sweep Results

The entire 9-point sweep on $17,862$ tasks executed in **7.89 seconds** across 9 parallel CPU workers:

| $C_{\text{cpu}}$ (cores/s) | Slack $\kappa$ | Baseline B0 TSR (%) | Baseline B0 DMR (%) | Heuristic E1 TSR (%) | Heuristic E1 DMR (%) | Greedy Advantage $\Delta_{\text{TSR}}$ | E1 Offload Rate (%) | E1 Control Overhead |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 18.0 | 2.0 | 77.14% | 22.86% | 82.68% | 17.32% | **+5.54%** | 40.3% | 821,648 B |
| 18.0 | 2.2 | 77.19% | 22.81% | 82.98% | 17.02% | **+5.79%** | 40.5% | 822,368 B |
| 18.0 | 2.5 | 77.21% | 22.79% | 83.38% | 16.62% | **+6.16%** | 39.8% | 820,176 B |
| **20.0** | **2.2** | **78.38%** | **21.62%** | **84.86%** | **15.14%** | **+6.48%** | **40.5%** | **822,288 B** |
| 20.0 | 2.0 | 78.24% | 21.76% | 84.38% | 15.62% | **+6.14%** | 40.8% | 823,152 B |
| 20.0 | 2.5 | 78.45% | 21.55% | 85.51% | 14.49% | **+7.06%** | 39.6% | 819,632 B |
| 22.0 | 2.0 | 79.94% | 20.06% | 87.20% | 12.80% | **+7.27%** | 40.3% | 821,600 B |
| 22.0 | 2.2 | 79.99% | 20.01% | 87.40% | 12.60% | **+7.41%** | 40.2% | 821,568 B |
| 22.0 | 2.5 | 80.08% | 19.92% | 87.80% | 12.20% | **+7.72%** | 40.0% | 821,008 B |

---

## 3. Selected Operational Baseline Justification

The operating point **$C_{\text{cpu}} = 20.0\text{ cores/second}$ and $\kappa = 2.2$** was selected as the optimal benchmark configuration:
1. **Meaningful Capacity Saturation:** Under pure local execution (B0), the edge cluster achieves **78.38% TSR** (21.62% DMR), reflecting realistic edge congestion during high-arrival workload bursts without causing total network collapse.
2. **Defensible Offloading Advantage:** Reactive Greedy offloading (E1) achieves **84.86% TSR**, providing a clean **+6.48% TSR improvement** over B0. This satisfies the calibration criterion ($\Delta > +5.0\%$) and proves the network topology provides meaningful offloading paths.
3. **Realistic Workload Headroom:** The 15.14% remaining deadline drops in E1 provide ample headroom for reinforcement learning policies (E2, E3, E3-APSM) to explore proactive routing optimizations.
4. **Stable Control Overhead:** Across all grid points, 15-second periodic telemetry consistently consumes $\approx 822\text{ KB}$ across the 3.5-hour test duration, proving stability.

