# APSM-MTP: Adaptive Predictive-Semantic Filter for Gossip Learning

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

> **M.Tech Research Project** — Extending Gossip Learning (Tundo et al. 2025, IEEE TNSM) with an Adaptive Predictive-Semantic Filter (APSM) to reduce bandwidth consumption in decentralised edge ML networks.

## Research Summary

The base paper (Tundo 2025) implements a fully decentralised, serverless Gossip Learning system for urban traffic prediction using the Porto Taxi dataset. Nodes broadcast their model weights after **every training round**, regardless of whether the update carries new information.

**Our contribution (Phase 2):** We introduce the **Adaptive Predictive-Semantic Filter (APSM)** — a pre-send gating mechanism that suppresses transmissions when the node's validation-loss surprise ε(t) falls within the adaptive noise band τ(t) = k·σ(ε). Result: **53.6% packet reduction with only 0.69% MSE degradation**.

## Quick Start

```bash
# Install dependencies
poetry install

# Run full GL-Baseline vs APSM comparison (requires Porto dataset in data/)
cd src/
python run_all_comparisons.py
```

## Results (Full Run — Kaggle GPU T4, June 2026)

| Metric | GL-Baseline | APSM Phase 2 |
|---|---|---|
| Avg MSE (scaled) | 0.009269 | 0.009332 (+0.69%) |
| Packets Sent | 1,447 | 1,314 |
| Packets Suppressed | 0 | **1,674** |
| **Packet Reduction** | 0% | **53.6% ✅** |

## Documentation

- [`docs/RESEARCH_REPORT.md`](docs/RESEARCH_REPORT.md) — Full research report (dataset, baseline, APSM design, results)
- [`docs/CODEBASE_GUIDE.md`](docs/CODEBASE_GUIDE.md) — File map + where to find what
- [`docs/phase2_explanation.md`](docs/phase2_explanation.md) — Conceptual explanation of the APSM mechanism

## Project Structure

```
MTP/
├── docs/                    ← Research report & codebase guide
├── src/
│   ├── gossiplearning/      ← Core simulation engine (DES + APSM)
│   ├── utils/               ← Dataset, metrics, model builders
│   ├── configs/             ← Secondary experiment configs
│   ├── scripts/             ← Data setup utilities
│   ├── review/              ← Files pending review (not canonical)
│   ├── experiments_full/    ← Canonical 7-hour run outputs
│   └── run_all_comparisons.py  ← Main experiment script
├── results/Phase2_Full/     ← Graphs, CSV, JSON, run log
├── data/                    ← Porto taxi dataset (npz + network graphs)
└── fl-baseline/             ← Reference Flower/FedAvg baseline
```

## Base Paper

> Tundo, A. et al. "Decentralized Edge Workload Forecasting with Gossip Learning."  
> *IEEE Transactions on Network and Service Management*, Special Issue on Next Generation Networks, 2025.  
> Data: [Zenodo](https://doi.org/10.5281/zenodo.15393791)

## License

Distributed under the GPL v3 license. See [LICENSE](LICENSE) for more information.
