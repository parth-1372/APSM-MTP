# APSM-MTP: Literature and References Index

This directory documents the foundational literature, base papers, and empirical datasets supporting the Master's Thesis:
**"Decentralized Edge Workload Forecasting using Semantic Gossip Learning and Agentic AI"**.

---

## 1. Primary Base Paper (Decentralized Gossip Learning)

- **Citation:**
  Alessandro Tundo, Marco Mobilio, Michele Ciavotta, and Leonardo Mariani, *"Gossip Learning for Edge Computing: A Decentralized Machine Learning Approach to Urban Traffic Prediction,"* **IEEE Transactions on Network and Service Management (TNSM)**, vol. 22, no. 1, pp. 450–464, March 2025.
- **DOI:** [`10.1109/TNSM.2024.3491200`](https://doi.org/10.1109/TNSM.2024.3491200)
- **Local PDF:** [`papers/MTP Base paper.pdf`](file:///C:/Users/btpUser/Desktop/Parth_MTP/APSM-MTP/papers/MTP%20Base%20paper.pdf)
- **Role in APSM-MTP:**
  Provides the baseline peer-to-peer decentralized model synchronization protocol (push-sum age-weighted model mixing over peer-to-peer overlay graphs). Phase 3 extends this baseline with the Adaptive Predictive-Semantic Mechanism (APSM) to resolve the blind-transmission communication bottleneck.

---

## 2. Multi-Agent Reinforcement Learning & Task Offloading

- **Citation:**
  K. Adu, R. Mensah, and Y. Zhang, *"Multi-Agent Deep Reinforcement Learning for Decentralized Task Offloading in Heterogeneous Edge Networks,"* **MDPI Sensors**, vol. 26, no. 3, art. 914, January 2026.
- **DOI:** [`10.3390/s26030914`](https://doi.org/10.3390/s26030914)
- **Local PDF:** [`papers/sensors-26-00914.pdf`](file:///C:/Users/btpUser/Desktop/Parth_MTP/APSM-MTP/papers/sensors-26-00914.pdf)
- **Role in APSM-MTP:**
  Informs the multi-agent task offloading formulation, observation space design, and neighbor masking for localized decentralized action spaces.

- **Additional Literature:**
  *"Multi-Agent Deep Reinforcement Learning for Task Offloading in UAV-Assisted Mobile Edge Computing"*, IEEE Transactions on Wireless Communications.
  - **Local PDF:** [`papers/Multi-Agent_Deep_Reinforcement_Learning_for_Task_Offloading_in_UAV-Assisted_Mobile_Edge_Computing.pdf`](file:///C:/Users/btpUser/Desktop/Parth_MTP/APSM-MTP/papers/Multi-Agent_Deep_Reinforcement_Learning_for_Task_Offloading_in_UAV-Assisted_Mobile_Edge_Computing.pdf)

---

## 3. Collaborative Edge Forecasting Benchmarks

- **Citation:**
  S. Chen, H. Gao, J. Wang, and L. Zhao, *"DECOFFEE: Decentralized Collaborative Federated Edge Forecasting with Privacy Preservation,"* **arXiv preprint**, 2026.
- **arXiv Link:** [`arXiv:2604.24507`](https://arxiv.org/abs/2604.24507)
- **Local PDF:** [`papers/2604.24507v1.pdf`](file:///C:/Users/btpUser/Desktop/Parth_MTP/APSM-MTP/papers/2604.24507v1.pdf)
- **Role in APSM-MTP:**
  Serves as the contemporary peer benchmark for collaborative workload forecasting across edge nodes under dynamic traffic and temporal variation.

---

## 4. Benchmark Datasets and Spatial Topologies

- **Alibaba Cluster Trace v2017:**
  - Description: Real-world trace comprising 1,300+ heterogeneous nodes over 12 hours. Contains machine CPU/Memory utilization, task arrival timestamps, and execution runtimes.
  - Upstream Repository: [alibaba/clusterdata (v2017)](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2017)
  - Preprocessed Cache: `data/alibaba_2017/curated_*.parquet`

- **Porto Spatial Edge Topology:**
  - Description: Spatial cell coordinates and real network topologies derived from the Porto urban edge dataset.
  - Data Files: `assets/porto_cells.parquet`, `assets/BBox_Porto.png`

---

## 5. Thesis Internal Milestone Documentation

- **MTP Phase 1 Report:**
  - Author: Parth (2022IMT082)
  - Local PDF: [`2022IMT082-MTP1.pdf`](file:///C:/Users/btpUser/Desktop/Parth_MTP/APSM-MTP/2022IMT082-MTP1.pdf)
- **Phase 3 Authoritative Research Report:**
  - Location: [`docs/PHASE3_FINAL_RESEARCH_REPORT.md`](file:///C:/Users/btpUser/Desktop/Parth_MTP/APSM-MTP/docs/PHASE3_FINAL_RESEARCH_REPORT.md)
