# Phase 3 Stage 6 Reproducibility Specification

**Thesis:** Decentralized Edge Workload Forecasting using Semantic Gossip Learning and Agentic AI  
**Scope:** $N = 10$ Nodes (Porto Edge Topology, $k=3$, 16 undirected links)  
**Branch:** `stage6-audit-fix-10n`  
**Date:** November 2026

---

## 1. Hardware & System Environment

All experiments and benchmarks are reproduced in the following verified environment:

- **Host Operating System:** Windows 11 Enterprise (64-bit)
- **Central Processor:** Intel Core i9-13950HX (24 Physical Cores / 32 Threads, 2.20 GHz base, up to 5.50 GHz turbo)
- **System Memory:** 31.69 GiB DDR5 RAM
- **Dedicated Graphics Accelerator:** NVIDIA RTX 1000 Ada Generation Laptop GPU (6,141 MiB VRAM, Compute Capability 8.9)
- **CUDA Runtime:** CUDA 13.0 (Driver version compatible)
- **Python Environment:** Python 3.10.0 (`.venv` virtual environment)
- **Core Libraries:**
  - `torch == 2.11.0+cu130`
  - `numpy == 1.26.4`
  - `scipy == 1.13.1`
  - `matplotlib == 3.8.4`
  - `pandas == 2.2.2`
  - `pyarrow == 16.0.0`
  - `pytest == 8.1.1`

---

## 2. Repository and Branch Governance

- **Repository:** `https://github.com/parth-1372/APSM-MTP`
- **Active Working Branch:** `stage6-audit-fix-10n`
- **Frozen Stage 6 Milestone Baseline:** Commit `e7295860bdfde9516c6a3f8e80235cb01c9ccda6`
- **Phase 2 Baseline:** Commit `62124eb` (strictly preserved, zero modifications to `src/gossiplearning/`, `src/utils/`, or `results/phase2_*`)
- **Remote Push Policy:** Branch `stage6-audit-fix-10n` is maintained locally and is **not pushed to remote without explicit user approval**.

---

## 3. Step-by-Step Reproduction Guide

### Step 3.1: Verify Test Suite Integrity
Execute the complete Phase 3 unit and integration test suite (21 tests total):
```powershell
rtk cmd /c "cd /d C:\Users\btpUser\Desktop\Parth_MTP\APSM-MTP && .venv\Scripts\python.exe -m pytest tests/phase3 -v"
```
**Expected Outcome:** `21 passed` in $< 120\text{ seconds}$.

### Step 3.2: Execute the 6-Arm N=10 Integrated Benchmark
Run the full 6-arm benchmark across all 5 evaluation seeds ($M = 30$ runs total):
```powershell
rtk cmd /c "cd /d C:\Users\btpUser\Desktop\Parth_MTP\APSM-MTP && .venv\Scripts\python.exe src\phase3\run_stage6_integrated.py --seeds 11 22 33 44 55 --episodes 48 --warmup 36 --output results/phase3/stage6_integrated_10n_audited_v2.json --ckpt-dir results/phase3/checkpoints_stage6_10n_audited_v2 --manifest results/phase3/stage6_10n_audited_v2_manifest.json"
```

### Step 3.3: Generate Publication-Quality Figures (Figures 1 – 19)
Generate all 19 publication figures into the audited results directory:
```powershell
rtk cmd /c "cd /d C:\Users\btpUser\Desktop\Parth_MTP\APSM-MTP && .venv\Scripts\python.exe src\phase3\plot_stage6_results.py --results results/phase3/stage6_integrated_10n_audited_v2.json --plots-dir results/phase3/plots_stage6_10n_audited_v2"
```

### Step 3.4: Generate Statistical Significance and Non-Inferiority Report
Perform small-sample Student's $t$ paired hypothesis tests and pre-specified non-inferiority testing:
```powershell
rtk cmd /c "cd /d C:\Users\btpUser\Desktop\Parth_MTP\APSM-MTP && .venv\Scripts\python.exe src\phase3\analyze_stage6_results.py --results results/phase3/stage6_integrated_10n_audited_v2.json --output docs/phase3_stage6_audit_v2/STAGE6_STATISTICAL_ANALYSIS_10N.md"
```

---

## 4. Artifacts Directory Structure

```
APSM-MTP/
├── docs/
│   └── phase3_stage6_audit_v2/
│       ├── STAGE6_INDEPENDENT_AUDIT.md
│       ├── STAGE6_ARCHITECTURE_CORRECTIONS.md
│       ├── STAGE6_DATA_AND_EXPERIMENTAL_METHODOLOGY.md
│       ├── STAGE6_RESULTS_N10_AUDITED_V2.md
│       ├── STAGE6_STATISTICAL_ANALYSIS_10N.md
│       ├── STAGE6_REMAINING_LIMITATIONS.md
│       ├── STAGE6_REPRODUCIBILITY.md
│       └── audit_manifest_v2.json
├── results/
│   └── phase3/
│       ├── stage6_integrated_10n_audited_v2.json
│       ├── stage6_10n_audited_v2_manifest.json
│       ├── checkpoints_stage6_10n_audited_v2/
│       │   ├── e2_madqn_s11_best.pt ...
│       │   ├── e3_madqn_local_s11_best.pt ...
│       │   ├── e3_madqn_gl_s11_best.pt ...
│       │   └── e3_madqn_apsm_s11_best.pt ...
│       └── plots_stage6_10n_audited_v2/
│           ├── fig01_tsr_comparison_10n.png
│           ├── fig02_dmr_comparison_10n.png
│           ├── ...
│           ├── fig18_tradeoff_tsr_vs_bandwidth_10n.png
│           └── fig19_forecasting_error_comparison_10n.png
```

---

## 5. File Integrity and Cryptographic Manifest

Baseline file hashes prior to audit fixes were recorded in `audit_manifest_v2.json`. The final audited artifacts will be re-hashed upon benchmark completion and signed in `results/phase3/stage6_10n_audited_v2_manifest.json`.
