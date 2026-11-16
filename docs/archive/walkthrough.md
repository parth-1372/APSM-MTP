# MTP Cleanup & Documentation — Walkthrough

## What Was Done

### 1. Research Report Written (`docs/RESEARCH_REPORT.md`)

A complete, self-contained research document covering:
- **Research Gap**: Why the base paper (Tundo 2025) wastes bandwidth and what APSM fixes
- **Dataset**: Porto Taxi full pipeline — GPS → spatial cells → per-node `.npz` (21,946 train samples/node, 9 features, 4-timestep LSTM)
- **Baseline Deep Flow**: Every step traced — DES event queue, SendModels → Receive → Merge → Train → Save chain, age-weighted aggregation math, LSTM architecture
- **APSM Phase 2 Deep Flow**: ε(t) surprise score, τ(t) = k·σ(sliding window), heartbeat mechanism, exact integration points in `node.py` and `event.py`
- **Results**: 53.6% packet reduction, 0.69% MSE degradation — both success criteria met
- **Visual Evidence**: All 5 graphs explained in context

### 2. Codebase Guide Written (`docs/CODEBASE_GUIDE.md`)

Quick-reference table of every file, when to touch it, and how to run experiments.

### 3. README Rewritten (`README.md`)

Reflects the APSM contribution, links to docs, shows results summary, updated structure.

---

## Files Moved / Reorganized

### New directories created
| Directory | Purpose |
|---|---|
| `docs/` | All documentation |
| `src/scripts/` | Data setup utilities |
| `src/configs/` | Secondary experiment config JSONs |
| `src/review/` | Files pending review (not canonical) |

### Root → docs/
| From | To |
|---|---|
| `phase2_explanation.md` | `docs/phase2_explanation.md` |
| `thesis_analysis_report.md` | `docs/thesis_analysis_report.md` |
| `task.md` | `docs/task.md` |

### Root → src/scripts/
| From | To |
|---|---|
| `download_and_extract.py` | `src/scripts/download_and_extract.py` |
| `fix_typing.py` | `src/scripts/fix_typing.py` |
| `unzip.py` | `src/scripts/unzip.py` |

### src/ → src/configs/
| From | To |
|---|---|
| `failures_simulation_config.json` | `src/configs/failures_simulation_config.json` |
| `scalability_simulation_config.json` | `src/configs/scalability_simulation_config.json` |

### src/ → src/review/ (pending review)
| File | Reason |
|---|---|
| `history.json` (renamed `history_stray_run.json`) | Stray ~1.3MB test run artifact |
| `history_test.ipynb` | Test notebook, not canonical |
| `hyperparameter_tuning.ipynb` | Not part of main research pipeline |
| `plots.ipynb` | Large (489KB), ad-hoc, superseded by run_all_comparisons.py plots |
| `failures_analysis.ipynb` | Niche failure-mode notebook |
| `failures_simulation.ipynb` | Niche failure-mode notebook |

### src/ → root
| From | To |
|---|---|
| `requirements-data-analysis.txt` | `requirements-data-analysis.txt` (root) |

---

## Files Deleted

| File | Reason |
|---|---|
| `MTP.pem` | SSH key — user confirmed delete |
| `__MACOSX/` (root) | macOS zip artifact, empty |
| `src/__MACOSX/` | macOS zip artifact, empty |
| `results/Phase2_Full/mtp-notebook.log` | Exact duplicate of phase2 log |

---

## Files Renamed

| Old Name | New Name | Reason |
|---|---|---|
| `results/phase 1/` | `results/phase1/` | Removed space from dir name |
| `results/Phase2_Full/phase2._log.txt` | `results/Phase2_Full/phase2_run.log` | Proper extension, no dot before extension |
| `results/Phase2_Full/kaggle logs save and commit.txt` | `results/Phase2_Full/kaggle_run_log.txt` | Underscores, descriptive name |

---

## Final Clean Structure

```
MTP/
├── docs/
│   ├── RESEARCH_REPORT.md       ← Full research document
│   ├── CODEBASE_GUIDE.md        ← Navigation reference
│   ├── phase2_explanation.md    ← Conceptual explanation
│   ├── thesis_analysis_report.md
│   └── task.md
│
├── src/
│   ├── gossiplearning/          ← Core engine (untouched)
│   ├── utils/                   ← Utilities (untouched)
│   ├── configs/                 ← failures + scalability configs
│   ├── scripts/                 ← download, unzip, fix_typing
│   ├── review/                  ← 6 files for later inspection
│   ├── experiments_full/        ← Canonical 7-hour run outputs
│   ├── 0..7_*.ipynb             ← Pipeline notebooks (in order)
│   ├── run_all_comparisons.py   ← MAIN experiment script
│   ├── threshold_sweep.py
│   ├── scalability_simulation.py
│   ├── simulation.py
│   ├── train_centralized.py
│   ├── train_single.py
│   ├── analysis.py
│   ├── evaluation.py
│   ├── porto_taxi_dataset.ipynb ← Dataset exploration
│   └── config.json              ← Base config template
│
├── results/
│   ├── Phase2_Full/             ← 5 graphs + CSV + JSON + log
│   ├── phase1/                  ← Phase 1 results
│   └── phase2/                  ← Phase 2 preliminary results
│
├── data/                        ← Porto dataset
├── fl-baseline/                 ← FedAvg reference baseline
├── README.md                    ← Updated
├── pyproject.toml
└── poetry.lock
```
