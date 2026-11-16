# Reproducibility Guide & Test Evidence (Stage 6 Reconciliation)

## 1. System Environment & Hardware Specifications

All experiments, simulations, calibration sweeps, and test suites were executed on the following verified hardware and software platform:

- **Operating System:** Microsoft Windows 11 (Build 10.0.26200, AMD64 architecture)
- **Host CPU:** 13th Gen Intel(R) Core(TM) i9-13950HX (24 Physical Cores, 32 Logical Processors @ 2.20 GHz base, up to 5.50 GHz turbo)
- **Host RAM:** 31.69 GB High-Speed Memory
- **Dedicated GPU:** NVIDIA RTX 1000 Ada Generation Laptop GPU (6,141 MiB GDDR6 VRAM, Compute Capability 8.9)
- **NVIDIA Display Driver:** Version 596.71 (CUDA Driver Version 13.2)
- **Python Environment:** Python 3.10.0 64-bit Virtual Environment (`.venv`)
- **Core Frameworks & Libraries:**
  - `torch`: 2.11.0+cu130 (CUDA GPU acceleration enabled)
  - `numpy`: 2.2.6
  - `scipy`: 1.15.3
  - `pandas`: 2.2.2
  - `pytest`: 8.4.2

---

## 2. Repository & Branch State

- **Repository:** `https://github.com/parth-1372/APSM-MTP.git`
- **Baseline Audited Milestone:** Commit `5d0dec71dd2061b10d1c3e64a67f81050133a9e2`
- **Working Reconciliation Branch:** `stage6-final-reconcile-10n`
- **Branch Tracking:** Strictly tracks `origin/stage6-final-reconcile-10n` without creating alternate branches or force-pushing.

---

## 3. Reproduction Commands

All commands are executed from the repository root directory `C:\Users\btpUser\Desktop\Parth_MTP\APSM-MTP`:

### 3.1 System Capacity Calibration Sweep (Training Burst Slice)
Executes the 9-point grid sweep across $C_{\text{cpu}} \in \{18.0, 20.0, 22.0\}$ and $\kappa \in \{2.0, 2.2, 2.5\}$ on bins $[380, 488)$ (10,678 tasks):
```powershell
.venv\Scripts\python.exe scripts/calibrate_10n_train.py
```
*Output:* `results/phase3/stage6_10n_calibration_train_burst.json`

### 3.2 Programmatic Calibration Table Generation
Verifies consistency and generates the exact Markdown table without manual transcription:
```powershell
.venv\Scripts\python.exe scripts/generate_calibration_table.py
```

### 3.3 Exact MARL Checkpoint Evaluation with Time-Aligned Forecasting
Evaluates the trained checkpoints from `results/phase3/checkpoints_stage6_10n_audited_v2/` on test split $[1035, 1380)$ with 3,450 aligned predictions:
```powershell
.venv\Scripts\python.exe scripts/eval_marl_forecasting_exact.py
```
*Output:* `results/phase3/stage6_10n_marl_eval_exact.json`

### 3.4 Controlled Forecasting Diagnostic (ReactiveGreedy Baseline Policy)
Evaluates edge LSTM forecasters under a fixed ReactiveGreedy baseline policy:
```powershell
.venv\Scripts\python.exe scripts/reevaluate_forecasting_metrics.py
```
*Output:* `results/phase3/stage6_10n_forecasting_aligned.json`

### 3.5 Strict JSON Reconciliation & Formatting
Reconciles audited results, preserves individual seed variance, converts bare NaNs/Infinities to standards-compliant JSON `null`, updates paired disaggregated suppression metrics, and validates with a strict parser:
```powershell
.venv\Scripts\python.exe scripts/reconcile_results_json.py
```
*Output:* `results/phase3/stage6_integrated_10n_reconciled.json`

### 3.6 Statistical Hypothesis Testing & Reporting
Executes inferential statistical tests, exact Student's $t$ with $df=4$, Cohen's $d_z$, Holm-Bonferroni correction, and exploratory non-inferiority analysis:
```powershell
.venv\Scripts\python.exe src/phase3/analyze_stage6_results.py --results results/phase3/stage6_integrated_10n_reconciled.json --output docs/phase3_stage6_final_reconciliation/CORRECTED_STATISTICAL_ANALYSIS.md
```
*Output:* `docs/phase3_stage6_final_reconciliation/CORRECTED_STATISTICAL_ANALYSIS.md`

### 3.7 Complete Automated Test Suite Execution
Runs all 28 unit and integration tests across Phase 3 and saves log:
```powershell
.venv\Scripts\python.exe -m pytest tests/phase3 -v > results/phase3/stage6_final_reconciliation_test_log.txt
```

---

## 4. Test Suite Structure & Verification Coverage

The automated test suite in `tests/phase3/` contains 28 tests providing end-to-end verification across every subsystem:

### Data Invariants (`test_stage1_data.py` - 3 tests)
1. `test_schema_verification`: Confirms schema fields, column data types, and non-empty data arrays.
2. `test_topology_tier1_sparse_partial_mesh`: Asserts node degrees, symmetric physical links, and partial mesh connectivity.
3. `test_curated_6n_dataset_causality_and_non_leakage`: Verifies non-leakage between train, validation, and test partitions.

### Forecasting Subsystem (`test_stage3_forecasting.py` - 6 tests)
4. `test_p2_local_head_isolation_and_bytes`: Asserts local prediction heads remain strictly private; only shared representation weights are exchanged.
5. `test_age_weighted_merge_fixes_zero_age_bug`: Validates weight interpolation stability even when sample age is zero.
6. `test_age_weighted_merge_self_mix_alpha`: Ensures that identical-age models merge with equal $0.5$ weights.
7. `test_common_init_zero_pretrain_identical_across_nodes`: Confirms all edge forecasters begin from identical pretrained parameter initialization.
8. `test_apsm_h1_warmup_and_heartbeat_invariants`: Asserts that heartbeat broadcasts trigger strictly when threshold is not exceeded for 5 consecutive bins.
9. `test_stage3_equal_cadence_and_finite_metrics`: Ensures finite, non-NaN MSE and MAE metrics across all nodes.

### Discrete-Event Simulator (`test_stage4_simulator.py` - 3 tests)
10. `test_stage4_anti_loop_conservation_and_byte_audit`: Confirms strict anti-loop TTL guards and exact task conservation ($N_{\text{completed}} + N_{\text{dropped}} = N_{\text{total}}$).
11. `test_stage4_strict_causality_and_non_oracle_staleness`: Verifies that telemetry updates reflect physical network latency and queue staleness.
12. `test_stage4_apsm_t_gating_and_e1_beats_b0`: Asserts that greedy offloading utilizes spare neighbor capacity.

### Multi-Agent Reinforcement Learning (`test_stage5_marl.py` - 5 tests)
13. `test_numpy_runners_match_pytorch_and_enforce_mask`: Tests numerical parity between PyTorch network and optimized NumPy inference runner.
14. `test_ctde_decentralized_actor_isolation_vs_ippo`: Verifies that actors observe only local and 1-hop neighbor states during inference.
15. `test_stage5_short_training_step_and_causal_local_lstm_matrix`: Asserts Q-loss minimization and backpropagation stability.
16. `test_run_marl_episode_conservation_delayed_outcomes_and_callbacks`: Asserts conservation across asynchronous task completion callbacks.
17. `test_run_marl_episode_gstate_strict_causality`: Verifies centralized critic observations do not violate evaluation isolation.

### Stage 6 Live Decentralized Integration (`test_stage6_integration.py` - 11 tests)
18. `test_channel_separation_and_ledger`: Asserts model bytes and telemetry bytes are strictly separated and unified in communication ledger.
19. `test_live_forecast_injection`: Confirms live one-step-ahead forecasts are correctly injected into MADQN observation vector without static matrices.
20. `test_apsm_suppression_at_10n`: Validates that APSM achieves $>45\%$ suppression on live 10-node mesh.
21. `test_100_percent_task_conservation`: Asserts exact task conservation ($N_{\text{success}} + N_{\text{drop}} = 17,862$) across all nodes.
22. `test_strict_temporal_causality`: Verifies that future perturbations after $T_{\text{cut}}$ do not affect prior decisions.
23. `test_shared_physical_link_contention`: Asserts that model transmissions serialize through link queues and delay concurrent task transmissions.
24. `test_madqn_pytorch_numpy_parity`: Asserts bit-for-bit equivalence between PyTorch model evaluation and NumPy runner.
25. `test_marl_deep_temporal_causality_with_instrumentation`: Deep causality test loading actual trained checkpoint and instrumenting Q-network actions, forecasts, residuals, thresholds, and decisions against future perturbations.
26. `test_model_packet_delivery_vs_bin_callback_assimilation`: Confirms models arriving at physical $t_{\text{rx}}$ wait in node inboxes and are only assimilated at discrete 30s bin boundary callbacks.
27. `test_holm_bonferroni_adjustment`: Validates step-down multiplier, monotonicity, and family-wise error rate control for Holm-Bonferroni correction.
28. `test_forecasting_metrics_time_alignment_and_non_leakage`: Asserts exactly 3,450 unique aligned test predictions, zero validation leakage, and persistence matching $y(t-1)$.

---

## 5. Artifact Verification & Cryptographic Manifest

Every generated artifact is cataloged with its relative path and SHA-256 digest in `results/phase3/stage6_10n_reconciled_manifest.json`.

Verification can be performed in PowerShell:
```powershell
Get-FileHash -Algorithm SHA256 results/phase3/stage6_integrated_10n_reconciled.json
```
