# APSM-MTP: Literature and References Index

This directory documents the foundational literature, base papers, and empirical datasets supporting the Master's Thesis:
**"Decentralized Edge Workload Forecasting using Semantic Gossip Learning and Agentic AI"**.

---

## 1. Primary Base Paper (Decentralized Gossip Learning)

- **Citation:**
  Alessandro Tundo, Federica Filippini, Francesco Regonesi, Michele Ciavotta, and Marco Savi, *"Decentralized Edge Workload Forecasting With Gossip Learning,"* **IEEE Transactions on Network and Service Management (TNSM)**, Vol. 22, No. 4, pp. 3016–3031, August 2025.
- **DOI:** [`10.1109/TNSM.2025.3570450`](https://doi.org/10.1109/TNSM.2025.3570450)
- **Local PDF:** [`papers/MTP Base paper.pdf`](../../papers/MTP%20Base%20paper.pdf)
- **Role in APSM-MTP:**
  Provides the baseline peer-to-peer decentralized model synchronization protocol (push-sum age-weighted model mixing over peer-to-peer overlay graphs). Phase 3 extends this baseline with the Adaptive Predictive-Semantic Mechanism (APSM) to resolve the blind-transmission communication bottleneck and integrates autonomous reinforcement learning task offloading.

---

## 2. Multi-Agent Reinforcement Learning & Task Offloading

- **Citation:**
  E. Adu, Y. Lee, J. Moon, S. Jang, I. Bang, and T. Kim, *"Decentralized Computation Offloading Strategy via Multi-Agent Deep Reinforcement Learning for Multi-Access Edge Computing Systems,"* **MDPI Sensors**, Vol. 26, No. 3, Art. 914, January 2026.
- **DOI:** [`10.3390/s26030914`](https://doi.org/10.3390/s26030914)
- **Local PDF:** [`papers/sensors-26-00914.pdf`](../../papers/sensors-26-00914.pdf)
- **Role in APSM-MTP:**
  Informs the multi-agent task offloading formulation, observation space design, and neighbor masking for localized decentralized action spaces.

- **Additional Literature:**
  *"Multi-Agent Deep Reinforcement Learning for Task Offloading in UAV-Assisted Mobile Edge Computing"*, IEEE Transactions on Wireless Communications.
  - **Local PDF:** [`papers/Multi-Agent_Deep_Reinforcement_Learning_for_Task_Offloading_in_UAV-Assisted_Mobile_Edge_Computing.pdf`](../../papers/Multi-Agent_Deep_Reinforcement_Learning_for_Task_Offloading_in_UAV-Assisted_Mobile_Edge_Computing.pdf)

---

## 3. Collaborative Edge Forecasting Benchmarks

- **Citation:**
  A. E. Giannopoulos, S. T. Spantideas, P. Trakadas et al., *"Decentralized Reinforcement Learning for Time-critical Workload Offloading and Energy Efficiency across the Computing Continuum,"* **arXiv preprint**, 2026.
- **arXiv Link:** [`arXiv:2604.24507`](https://arxiv.org/abs/2604.24507)
- **Local PDF:** [`papers/2604.24507v1.pdf`](../../papers/2604.24507v1.pdf)
- **Role in APSM-MTP:**
  Demonstrates coupling LSTM workload forecasting with deep reinforcement learning (Double Dueling DQN) for time-critical workload offloading across distributed edge environments.

---

## 4. Benchmark Datasets and Spatial Topologies

- **Alibaba Cluster Trace v2017:**
  - Description: Real-world trace comprising 1,300+ heterogeneous nodes over 12 hours. Contains machine CPU/Memory utilization, task arrival timestamps, and execution runtimes.
  - Upstream Repository: [alibaba/clusterdata (v2017)](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2017)
  - Preprocessed Cache: `data/alibaba_2017/curated_*.parquet`

- **Porto Spatial Edge Topology:**
  - Description: Spatial cell coordinates and network topologies derived from the Porto cellular edge dataset ($N=10$, $N=50$, $N=100$).
  - Data Files: `assets/porto_cells.parquet`, `assets/BBox_Porto.png`

---

## 5. Thesis Internal Milestone Documentation

- **MTP Phase 1 Report:**
  - Author: Parth (2022IMT082)
  - Local PDF: [`2022IMT082-MTP1.pdf`](../../2022IMT082-MTP1.pdf)
- **Phase 3 Learning Guide & Defence Handbook:**
  - Location: [`docs/PHASE3_LEARNING_GUIDE.md`](../PHASE3_LEARNING_GUIDE.md)
- **Phase 3 Authoritative Research Report:**
  - Location: [`docs/PHASE3_FINAL_RESEARCH_REPORT.md`](../PHASE3_FINAL_RESEARCH_REPORT.md)
