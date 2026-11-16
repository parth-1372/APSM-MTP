# STAGE 7 REPRODUCIBILITY GUIDE & EXECUTION PROTOCOL

This document provides exact reproduction instructions, execution commands, hardware requirements, and verification protocols for **Phase 3 Stage 7 Scalability Benchmarks**.

---

## 1. System Requirements & Software Environment

### 1.1 Hardware Specifications
- **CPU:** 13th Gen Intel(R) Core(TM) i9-13950HX (24 physical cores, 32 logical threads)
- **RAM:** Minimum 32 GB RAM DDR5
- **GPU Accelerator:** NVIDIA RTX 1000 Ada Generation Laptop GPU (6 GB VRAM) with CUDA Compute Capability 8.9
- **Storage:** Minimum 10 GB free disk space

### 1.2 Python Environment
- **Python Version:** 3.10.0
- **PyTorch:** `2.11.0+cu130`
- **Key Dependencies:** `numpy==2.2.6`, `scipy==1.15.3`, `pandas>=2.0`, `networkx>=3.0`, `pytest>=8.0`, `psutil>=5.9`

Activate the local environment:
```powershell
Set-Location C:\Users\btpUser\Desktop\Parth_MTP\APSM-MTP
.\.venv\Scripts\Activate.ps1
```

---

## 2. Test Suite Execution

Run the complete Phase 3 test suite, including all 17 Stage 7 scalability tests:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/phase3 -v
```
To run only the Stage 7 scalability tests:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/phase3/test_stage7_scalability.py -v
```

---

## 3. Scalability Benchmark Execution Commands

### 3.1 Gate A: N=10 Smoke Regression
```powershell
.\.venv\Scripts\python.exe src/phase3/run_stage7_scalability.py --nodes 10 --smoke --seeds 11
```

### 3.2 Gate B: N=50 Smoke Test
```powershell
.\.venv\Scripts\python.exe src/phase3/run_stage7_scalability.py --nodes 50 --smoke --seeds 11
```

### 3.3 Gate C: N=50 Full 5-Seed Scalability Benchmark
```powershell
.\.venv\Scripts\python.exe src/phase3/run_stage7_scalability.py --nodes 50 --seeds 11 22 33 44 55 --episodes 48 --warmup 36 --device cuda
```

### 3.4 Gate D: N=100 Smoke Test
```powershell
.\.venv\Scripts\python.exe src/phase3/run_stage7_scalability.py --nodes 100 --smoke --seeds 11
```

### 3.5 Gate E: N=100 Full 5-Seed Scalability Benchmark (Resource Permitting)
```powershell
.\.venv\Scripts\python.exe src/phase3/run_stage7_scalability.py --nodes 100 --seeds 11 22 33 44 55 --episodes 48 --warmup 36 --device cuda
```

---

## 4. Incremental Recovery & Resilience

The runner implements **atomic per-run persistence** to `results/phase3/stage7_scalability/n{N}/raw_runs.json`. If execution is interrupted by a power failure or user stop:
1. Re-run the exact same command.
2. The runner will inspect `raw_runs.json`, detect completed `(arm, seed)` pairs, and skip them automatically.
3. Execution resumes seamlessly from the first unfinished run.

---

## 5. Statistical Analysis Generation

To synthesize and generate the comparative markdown report across scales:
```powershell
.\.venv\Scripts\python.exe src/phase3/analyze_stage7_scalability.py --scales 10 50 100 --output docs/phase3_stage7_scalability/STAGE7_RESULTS_AND_ANALYSIS.md
```

---

## 6. Artifact Inventory & Integrity Verification

- **N=50 Results Directory:** `results/phase3/stage7_scalability/n50/`
  - `raw_runs.json` (30 full run records, strict RFC 8259 JSON)
  - `summary.json` (Mean $\pm$ Std per arm)
  - `calibration_and_topology.json` (Graph topology and simulation parameters)
  - `resource_usage.json` (Peak RAM, VRAM, and runtime)
- **N=100 Results Directory:** `results/phase3/stage7_scalability/n100/`
- **Checkpoints:** `results/phase3/stage7_scalability/checkpoints/`
