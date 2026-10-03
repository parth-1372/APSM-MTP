import sys

with open('src/run_all_comparisons.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Add concurrent.futures
code = code.replace("import time", "import time\nimport concurrent.futures")
code = code.replace("N_NODES        = 10", "N_NODES        = 50")
code = code.replace("N_SEEDS        = 1", "N_SEEDS        = 3")

# We will replace def main(): onwards.
main_idx = code.find("def main():")
top_part = code[:main_idx]

bottom_part = """
def run_seed(seed: int) -> tuple[dict, dict]:
    # Import locally to ensure independent contexts in subprocesses
    import tensorflow as tf
    import gc
    from pathlib import Path
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    seed_start = time.time()
    print(f"\\n{'━'*55}\\n  SEED {seed + 1}/{N_SEEDS}\\n{'━'*55}")

    # ── GL-Baseline ────────────────────────────────────────────
    b_workspace = str(SRC_DIR / f"experiments/gl_baseline_seed{seed}")
    b_hist_path = Path(b_workspace) / "0" / "history.json"

    if b_hist_path.exists():
        print(f"  ✅  GL-Baseline (seed {seed}) — cached, skipping.")
    else:
        print(f"  ▶️   GL-Baseline (seed {seed}) — running...")
        b_hist_path = run_one_simulation(b_workspace, seed, is_baseline=True)
        elapsed = time.time() - seed_start
        print(f"  ✅  GL-Baseline (seed {seed}) done  ({elapsed/60:.1f} min)")

    b_metrics = extract_metrics(b_hist_path)

    tf.keras.backend.clear_session()
    gc.collect()

    # ── APSM ────────────────────────────────────────────────────
    a_workspace = str(SRC_DIR / f"experiments/apsm_phase2_seed{seed}")
    a_hist_path = Path(a_workspace) / "0" / "history.json"

    a_start = time.time()
    if a_hist_path.exists():
        print(f"  ✅  APSM Phase 2 (seed {seed}) — cached, skipping.")
    else:
        print(f"  ▶️   APSM Phase 2 (seed {seed}) — running...")
        a_hist_path = run_one_simulation(a_workspace, seed, is_baseline=False)
        elapsed = time.time() - a_start
        print(f"  ✅  APSM (seed {seed}) done  ({elapsed/60:.1f} min)")

    a_metrics = extract_metrics(a_hist_path)

    tf.keras.backend.clear_session()
    gc.collect()

    return b_metrics, a_metrics

def main():
    if not DATASETS_FOLDER.exists() or not NETWORKS_FOLDER.exists():
        print(f"❌  Data not found at: {DATASETS_FOLDER}")
        print("    Run notebook 2_dataset_generation.ipynb first, then retry.")
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    total_start = time.time()

    print("\\n" + "=" * 60)
    print("  APSM Phase 2 — Full Research Run")
    print(f"  Seeds={N_SEEDS} | Updates={FIXED_UPDATES} | Epochs/round={EPOCHS_PER_UPDATE}")
    print(f"  MAX_SAMPLES={'FULL DATA' if MAX_SAMPLES is None else MAX_SAMPLES}")
    print("=" * 60 + "\\n")

    baseline_runs = [None] * N_SEEDS
    apsm_runs     = [None] * N_SEEDS

    with concurrent.futures.ProcessPoolExecutor(max_workers=min(N_SEEDS, 4)) as executor:
        futures = {executor.submit(run_seed, seed): seed for seed in range(N_SEEDS)}
        for future in concurrent.futures.as_completed(futures):
            seed = futures[future]
            try:
                b_metrics, a_metrics = future.result()
                baseline_runs[seed] = b_metrics
                apsm_runs[seed] = a_metrics
                
                # Progress snapshot
                total = b_metrics["total_sent"] + a_metrics["total_suppressed"]
                pr_now = (a_metrics["total_suppressed"] / total * 100) if total > 0 else 0
                print(f"\\n  📊  Seed {seed} snapshot:")
                print(f"      Baseline MSE = {b_metrics['avg_mse']:.5f}  |  APSM MSE = {a_metrics['avg_mse']:.5f}")
                print(f"      Packets sent: baseline={b_metrics['total_sent']}, apsm={a_metrics['total_sent']}")
                print(f"      Suppressed: {a_metrics['total_suppressed']}  →  {pr_now:.1f}% reduction")
            except Exception as e:
                import traceback
                print(f"❌  Seed {seed} failed with error: {e}")
                traceback.print_exc()

    # Remove Nones if failures occurred
    baseline_runs = [r for r in baseline_runs if r is not None]
    apsm_runs     = [r for r in apsm_runs if r is not None]

    if not baseline_runs:
        print("❌ All seeds failed.")
        sys.exit(1)

    # ── Final aggregated table ─────────────────────────────────────
    pkt_red, mse_delta = print_comparison_table(baseline_runs, apsm_runs)

    # ── Plots ──────────────────────────────────────────────────────
    print("📈  Generating plots...")
    plot_results(baseline_runs, apsm_runs, RESULTS_DIR)

    # ── Save CSVs / JSON ───────────────────────────────────────────
    save_results(baseline_runs, apsm_runs, RESULTS_DIR, pkt_red, mse_delta)

    total_elapsed = (time.time() - total_start) / 60
    print(f"\\n🎉  All done!  Total wall-clock time: {total_elapsed:.1f} min")
    print(f"📁  Results saved to: {RESULTS_DIR.resolve()}\\n")

if __name__ == '__main__':
    # Fix for multiprocessing in windows
    import multiprocessing
    multiprocessing.freeze_support()
    main()
"""

with open('src/run_all_comparisons.py', 'w', encoding='utf-8') as f:
    f.write(top_part + bottom_part)
