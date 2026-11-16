# APSM-MTP: Decentralized Edge Workload Forecasting using Semantic Gossip Learning and Agentic AI

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Tests: 47 Passed](https://img.shields.io/badge/tests-47%20passed-brightgreen.svg)](tests/phase3)

> **Master's Thesis Research Project (APSM-MTP)**  
> Author: Parth (Roll No. 2022IMT082)  
> Department of Computer Science & Engineering  
> Base Paper: Alessandro Tundo, Federica Filippini, Francesco Regonesi, Michele Ciavotta, and Marco Savi, *"Decentralized Edge Workload Forecasting With Gossip Learning,"* **IEEE Transactions on Network and Service Management (TNSM)**, Vol. 22, No. 4, pp. 3016–3031, August 2025. [DOI: 10.1109/TNSM.2025.3570450](https://doi.org/10.1109/TNSM.2025.3570450).

---

## 1. Executive Overview

In decentralized edge computing environments, edge nodes must forecast incoming computational demand and dynamically offload tasks to resource-abundant neighbors without relying on a centralized coordinator or single point of failure (SPOF). However, conventional decentralized collaborative learning (such as Gossip Learning) suffers from **blind periodic transmissions**, exchanging complete neural network weights across wireless channels regardless of whether local parameter updates contain novel information.

**APSM-MTP** resolves this fundamental communication-computation tension through a tri-part integrated architecture:
1. **Adaptive Predictive-Semantic Mechanism (APSM):** An online pre-transmission gating filter that monitors local validation-loss surprise $\varepsilon_i(t)$ against a self-calibrating temporal noise floor $\tau_i(t) = \mu_i(t) + k \cdot \sigma_i(t)$. Redundant weight exchanges are suppressed while informative updates pass unimpeded, enforced by a fail-safe periodic heartbeat ($H=5$).
2. **Decentralized Sequence Forecasting Engine:** Edge nodes maintain local recurrent sequence forecasters (GRU/LSTM) trained on continuous workload traces, collaboratively regularized across sparse peer-to-peer overlay graphs via push-sum age-weighted model mixing.
3. **Agentic Multi-Agent Reinforcement Learning (MADQN):** Independent Double-DQN agents perform decentralized, proactive task offloading informed by local workload predictions and spatial neighbor queue perceptions, enforcing strict TTL conservation without central orchestration.

---

## 2. Validated Empirical Findings Across Scales ($N=10, 50, 100$)

All results are derived from complete, programmatically validated benchmark datasets comprising **90 experimental runs** (30 runs per scale across 5 independent seeds: 11, 22, 33, 44, 55) evaluated on real industrial workload traces from **Alibaba Cluster Data v2017** mapped to the **Porto Spatial Edge Topology**:

| Scale ($N$) | Forecasting Arm | Test MSE | $\Delta_{\text{MSE}}$ vs GL | Gain Retention $\mathcal{R}$ | Model Payload Exchanged | Bandwidth Reduction vs GL | Task Success Rate (TSR) |
|---|---|---|---|---|---|---|---|
| **$N = 10$** | **Local No-Gossip** | 0.037063 | — | — | 0.00 MB | 100.00% | 84.15% |
| | **Gossip Learning (Base)** | 0.035884 | Baseline | Baseline | 313.29 MB | 0.00% | 84.07% |
| | **APSM-MTP (Ours)** | **0.036000** | +0.000117 | **90.12%** | **149.86 MB** | **52.17%** | **84.15%** |
| **$N = 50$** | **Local No-Gossip** | 0.037977 | — | — | 0.00 MB | 100.00% | 95.75% |
| | **Gossip Learning (Base)** | 0.037510 | Baseline | Baseline | 1,664.34 MB | 0.00% | 95.75% |
| | **APSM-MTP (Ours)** | **0.037537** | +0.000027 | **94.21%** | **782.72 MB** | **52.97%** | **95.73%** |
| **$N = 100$** | **Local No-Gossip** | 0.038085 | — | — | 0.00 MB | 100.00% | 96.55% |
| | **Gossip Learning (Base)** | 0.037633 | Baseline | Baseline | 3,289.52 MB | 0.00% | 96.56% |
| | **APSM-MTP (Ours)** | **0.037614** | **-0.000019** | **104.17%** | **1,519.92 MB** | **53.79%** | **96.63%** |

### Key Experimental Insights:
- **Bandwidth Reduction:** APSM consistently cuts wireless model synchronization overhead by **52.17% to 53.79%** across all network scales, saving over **1.77 GB** of wireless transmission payload at $N=100$.
- **Forecasting Fidelity:** At $N=10$ and $N=50$, APSM retains **90.12%** and **94.21%** of the full collaborative gossip gain, with negligible absolute MSE differences ($\le 0.000117$). At $N=100$, APSM matches and marginally outperforms unconstrained Gossip Learning ($104.17\%$ gain retention) due to the noise-filtering effect of semantic gating over large peer graphs.
- **Task Offloading Performance:** MARL offloading achieves high task completion ($96.63\%$ at $N=100$), maintaining bounded service latency and zero packet loss.

---

## 3. Authoritative Research Documentation & Learning Guide

Comprehensive documentation, theoretical proofs, literature mapping, detailed statistical analyses, and a viva-defence learning guide are maintained in:

- **[Phase 3 Learning Guide & Defence Handbook (`docs/PHASE3_LEARNING_GUIDE.md`)](docs/PHASE3_LEARNING_GUIDE.md)**  
  *The core intuitive learning guide.* Written in straightforward English to teach the system architecture, mathematical mechanics, algorithm design decisions, experimental chronology, and viva-defence preparation (includes a glossary and 10 model viva self-check questions).
- **[Primary Research Report (`docs/PHASE3_FINAL_RESEARCH_REPORT.md`)](docs/PHASE3_FINAL_RESEARCH_REPORT.md)**  
  *The authoritative 17-section Phase 3 thesis research report.* Covers the complete problem formulation, literature survey and verified DOIs, research decision traceability, system architecture blueprints, mathematical derivations, experimental protocols, Holm-Bonferroni corrected hypothesis testing, and computational complexity analysis.
- **[Literature & References Index (`docs/references/REFERENCES_INDEX.md`)](docs/references/REFERENCES_INDEX.md)**  
  Complete bibliographic records, verified DOIs, arXiv preprints, and local PDF mappings for all foundational papers.
- **[Historical Archive Index (`docs/archive/ARCHIVE_INDEX.md`)](docs/archive/ARCHIVE_INDEX.md)**  
  Archival directory housing previous Phase 1, Phase 2, and intermediate Phase 3 audit reports and design reviews.

---

## 4. Repository Layout

```text
APSM-MTP/
├── README.md                                 # Primary repository overview
├── pyproject.toml                            # Build configuration and dependencies
├── 2022IMT082-MTP1.pdf                       # MTP Phase 1 reference thesis report
│
├── docs/                                     # Documentation root
│   ├── PHASE3_LEARNING_GUIDE.md              # ★ Phase 3 learning guide & viva handbook
│   ├── PHASE3_FINAL_RESEARCH_REPORT.md       # ★ Authoritative 17-section research report
│   ├── references/                           # Essential citations and paper mappings
│   │   └── REFERENCES_INDEX.md
│   └── archive/                              # Historically preserved audit & phase reports
│       └── ARCHIVE_INDEX.md
│
├── src/                                      # Source code
│   ├── phase3/                               # Phase 3 core implementation
│   │   ├── config.py                         # Unified dataclass configuration
│   │   ├── forecasting/                      # GRU models, age-weighted merge, APSM gate
│   │   ├── env/                              # Edge discrete-event network simulator
│   │   ├── marl/                             # MADQN agent, replay buffer, state encoders
│   │   ├── integration/                      # Live integrated simulation engine
│   │   ├── run_stage6_integrated.py          # N=10 integrated benchmark runner
│   │   └── run_stage7_scalability.py         # N=50 and N=100 scalability runner
│   └── gossiplearning/                       # Core gossip learning library routines
│
├── tests/                                    # Automated test suite (47 tests passing)
│   └── phase3/
│       ├── test_stage1_data.py               # Dataset causality & topology checks
│       ├── test_stage3_forecasting.py        # Forecaster, merge & APSM gate unit tests
│       ├── test_stage4_simulator.py          # Simulation physics & conservation tests
│       ├── test_stage5_marl.py               # MADQN training, execution & masking tests
│       ├── test_stage6_integration.py        # N=10 integration & causality tests
│       └── test_stage7_scalability.py        # N=50/100 scalability & checkpoint tests
│
├── results/                                  # Canonical benchmark results & checkpoints
│   └── phase3/                               # Verified JSON results & SHA-256 manifests
│       ├── stage6_integrated_10n_reconciled.json
│       ├── stage7_scalability/               # N=50 and N=100 verified results
│       └── ...
│
├── assets/                                   # Porto spatial topology & bounding boxes
├── data/                                     # Alibaba cluster data traces
├── papers/                                   # Foundational reference research papers (PDF)
└── scripts/                                  # Metric compilation and manifest generators
```

---

## 5. Quick Start & Reproducibility

### 5.1 Environment Setup

Ensure Python 3.10+ is installed:

```bash
# Clone the repository
git clone https://github.com/parth-1372/APSM-MTP.git
cd APSM-MTP

# Create virtual environment and activate
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1

# Install package in editable mode with development dependencies
pip install -e .
```

### 5.2 Running the Test Suite

Execute the complete regression and invariant test suite:

```bash
python -m pytest tests/phase3 -v
```

All 47 unit, integration, and scalability tests verify:
- Non-oracle temporal causality (no forward-looking data leakage).
- 100% discrete task conservation across nodes and networks.
- Physical link bandwidth contention and channel separation.
- Post-warmup checkpoint selection invariants.
- Strict RFC 8259 JSON compliance.

### 5.3 Executing Scalability Benchmarks

To execute or re-verify the integrated benchmark runs:

```bash
# N=10 Integrated Benchmark (6 arms, 5 seeds)
python -m src.phase3.run_stage6_integrated --seeds 11 22 33 44 55

# N=50 Scalability Benchmark
python -m src.phase3.run_stage7_scalability --nodes 50 --seeds 11 22 33 44 55

# N=100 Scalability Benchmark
python -m src.phase3.run_stage7_scalability --nodes 100 --seeds 11 22 33 44 55
```

---

## 6. Citation

If you build upon or reference this work in academic research, please cite:

```bibtex
@mastersthesis{parth2026apsmmtp,
  author       = {Parth},
  title        = {Decentralized Edge Workload Forecasting using Semantic Gossip Learning and Agentic AI},
  school       = {Indian Institute of Information Technology, Design and Manufacturing},
  year         = {2026},
  month        = {May},
  note         = {Extending IEEE TNSM 2025 Gossip Learning with Adaptive Predictive-Semantic Mechanism (APSM)}
}
```
