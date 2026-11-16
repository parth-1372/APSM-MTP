# MTP Codebase Navigation Guide

Quick reference for understanding what every file does and how everything fits together.

---

## Project Layout at a Glance

```
MTP/
├── docs/                          ← All documentation
├── src/                           ← All Python source code
│   ├── gossiplearning/            ← Core simulation library (don't touch unless extending protocol)
│   ├── utils/                     ← Experiment utilities (data loading, metrics, model builders)
│   ├── configs/                   ← Secondary experiment config files
│   ├── scripts/                   ← One-off data setup scripts
│   ├── review/                    ← Files pending review / not canonical
│   ├── experiments_full/          ← ★ Full 7-hour run outputs (canonical results)
│   └── experiments/               ← Previous / test run outputs
├── results/Phase2_Full/           ← ★ Final graphs + CSV + log from full run
├── data/                          ← Porto taxi dataset (npz files + network graphs)
└── fl-baseline/                   ← Reference FedAvg baseline (separate package)
```

---

## To Run a New Experiment

```bash
cd src/
python run_all_comparisons.py
```

This runs GL-Baseline + APSM back to back, saves results to `results/phase2_full/` and plots 5 graphs.

---

## Core Library: `src/gossiplearning/`

| File | Role | Touch it when... |
|---|---|---|
| `models.py` | Type aliases (NodeId, Link, ModelWeights…) | Adding new message types |
| `config.py` | Pydantic config schema | Adding new APSM hyperparameters |
| `node.py` | **Node training + APSM filter logic** | Modifying the semantic filter algorithm |
| `event.py` | **Gossip protocol event handlers** | Changing how/when nodes send/receive |
| `simulator.py` | DES runner, event queue | Changing simulation-level orchestration |
| `aggregators.py` | Weight merging (simple_avg, age_weighted…) | Adding a new aggregation strategy |
| `history.py` | History dataclasses (messages, suppressed, ε/τ) | Adding new metrics to track |
| `weights_marshaling.py` | Flatten/unflatten model weights | Changing weight serialization/compression |
| `links_strategy.py` | Graph → node links conversion | Changing network topology logic |

---

## Experiment Scripts: `src/`

| Script | Purpose |
|---|---|
| `run_all_comparisons.py` | **MAIN**: Run GL-Baseline + APSM, generate 5 graphs, CSV, JSON |
| `threshold_sweep.py` | Sweep k ∈ {0.5..3.0} to find optimal APSM sensitivity |
| `scalability_simulation.py` | Test with 100-node network (porto_100n_3k) |
| `simulation.py` | Run a single simulation from a config file |
| `train_centralized.py` | Train a single centralized LSTM (comparison baseline) |
| `train_single.py` | Train a single node in isolation (comparison baseline) |
| `analysis.py` | Post-hoc analysis of experiment metrics CSVs |
| `evaluation.py` | Evaluate saved models against test sets |

---

## Notebooks: `src/` (numbered in pipeline order)

| Notebook | Pipeline Step |
|---|---|
| `0_cell_data_preparation.ipynb` | Encode raw GPS → spatial cells |
| `1_network_generation.ipynb` | Build K-NN graph (10-node, 3-neighbour) |
| `2_dataset_generation.ipynb` | Generate per-node `.npz` train/val/test splits |
| `3_simulation.ipynb` | Run a single gossip simulation interactively |
| `4_single_node_training.ipynb` | Single-node training baseline |
| `5_centralized_training.ipynb` | Centralized training baseline |
| `6_evaluation.ipynb` | Evaluate experiment results |
| `7_exchanged_data.ipynb` | Analyse what data gets exchanged in gossip |

**Review folder** (`src/review/`): hyperparameter_tuning, plots, failures_simulation, failures_analysis, history_test notebooks. Not part of the main pipeline — inspect before deciding to keep.

---

## APSM Phase 2 — Key Code Locations

| Concept | File | Function |
|---|---|---|
| Surprise score ε(t) | `gossiplearning/node.py` | `update_semantic_state()` L134 |
| Adaptive threshold τ(t) | `gossiplearning/node.py` | `update_semantic_state()` L150 |
| Gag rule + heartbeat | `gossiplearning/node.py` | `should_suppress_transmission()` L156 |
| Gate applied pre-send | `gossiplearning/event.py` | `process_send_model_event()` L162 |
| State updated after train | `gossiplearning/event.py` | `process_save_model_event()` L335 |
| Suppression counter | `gossiplearning/history.py` | `History.suppressed_packets` |
| ε/τ time series | `gossiplearning/history.py` | `History.semantic_surprise_scores` |
| APSM config params | `gossiplearning/config.py` | `TrainingConfig` L149–185 |

---

## Config Parameters Quick Reference

Edit `src/config.json` or override in `run_all_comparisons.py` constants:

```
APSM params:
  semantic_k         = 2.0   # Width of noise band (2σ rule)
  semantic_window    = 50    # Sliding window for σ computation
  semantic_heartbeat = 5     # Force send after N consecutive suppressions

Training params:
  fixed_updates      = 100   # Gossip rounds per node
  epochs_per_update  = 3     # Local epochs per round
  merge_strategy     = age_weighted
  is_baseline        = true/false  # true → disable APSM
```

---

## Results Interpretation

| File | Contains |
|---|---|
| `results/Phase2_Full/graph1_bandwidth.png` | Packets sent vs suppressed (bar chart) |
| `results/Phase2_Full/graph2_convergence.png` | MSE over time for all 10 nodes, both methods |
| `results/Phase2_Full/graph3_semantic_surprise.png` | ε(t) vs τ(t) per node — shows when suppression fires |
| `results/Phase2_Full/graph4_per_node_suppression.png` | How many packets each node suppressed |
| `results/Phase2_Full/graph5_mse_boxplot.png` | Final MSE distribution comparison |
| `results/Phase2_Full/phase2_comparison_full.csv` | Numeric summary (MSE, RMSE, MAE, packets) |
| `results/Phase2_Full/summary.json` | Machine-readable result summary |
| `results/Phase2_Full/phase2_run.log` | Full console log from the 7-hour Kaggle run |
