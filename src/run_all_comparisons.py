"""
run_all_comparisons.py
======================
Runs GL-Baseline and APSM (Phase 2: Semantic Filter) simulations back-to-back
and produces a visible comparison table + three plots saved to results/.

Usage (from src/ folder):
    poetry run python run_all_comparisons.py
"""
import json
import os
import dataclasses
import functools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # no GUI needed
import matplotlib.pyplot as plt
import numpy as np

from gossiplearning.config import Config
from gossiplearning.models import MergeStrategy, StopCriterion
from gossiplearning.utils import NpEncoder
from gossiplearning.weight import weight_by_dataset_size
from utils.data import get_common_test_set
from utils.gossip_training import get_node_dataset, round_trip_fn, run_simulation
from utils.model_creators import create_LSTM

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ─────────────────────────────────────────────
# Shared config
# ─────────────────────────────────────────────
N_NODES = 10
MAX_SAMPLES = 50  # Cap training data for fast CPU runs
K = 3
DATASETS_FOLDER = Path(f"data/datasets/porto_{N_NODES}n_{K}k")
NETWORKS_FOLDER = Path(f"data/networks/porto_{N_NODES}n_{K}k/seed1000")
RESULTS_DIR = Path("../results")


def load_base_config(workspace_dir: str) -> dict:
    with open("config.json", "r") as f:
        cfg = json.load(f)
    cfg["n_nodes"] = N_NODES
    cfg["workspace_dir"] = workspace_dir
    cfg["training"]["merge_strategy"] = MergeStrategy.AGE_WEIGHTED
    cfg["training"]["epochs_per_update"] = 1
    cfg["training"]["target_probability"] = 1
    cfg["training"]["stop_criterion"] = StopCriterion.FIXED_UPDATES.value
    cfg["training"]["fixed_updates"] = 10
    cfg["training"]["is_baseline"] = "baseline" in workspace_dir
    cfg["training"]["patience"] = 5
    cfg["training"]["min_delta"] = 0.1
    return cfg


def run_one_simulation(workspace_dir: str) -> Path:
    """Run a single simulation and return the path to its history.json."""
    cfg_json = load_base_config(workspace_dir)
    config = Config.model_validate(cfg_json)
    model_creator = functools.partial(create_LSTM, config=config)

    _raw_node_data_fn = functools.partial(
        get_node_dataset,
        base_folder=DATASETS_FOLDER,
        simulation_number=0,
        ds_name="",
    )
    def node_data_fn(node_index):
        ds = _raw_node_data_fn(node_index=node_index)
        for key in ("X_train", "Y_train", "X_val", "Y_val"):
            ds[key] = ds[key][:MAX_SAMPLES]
        return ds
    get_test_set = functools.partial(
        get_common_test_set,
        node_data_fn=node_data_fn,
        n_nodes=config.n_nodes,
        perc=0.1,
    )

    run_simulation(
        config=config,
        simulation_number=0,
        network_folder=NETWORKS_FOLDER / "0",
        round_trip_fn=round_trip_fn,
        model_transmission_fn=lambda i, j: 30,
        node_data_fn=node_data_fn,
        model_creator=model_creator,
        get_test_set=get_test_set,
        weight_fn=weight_by_dataset_size,
    )
    return Path(workspace_dir) / "0" / "history.json"


def extract_metrics(history_path: Path):
    """Extract MSE, total packets sent and suppressed packets from history.json."""
    with open(history_path) as f:
        h = json.load(f)

    mses = []
    for node_metrics in h.get("nodes_test_history", {}).values():
        vals = node_metrics.get("mse", [])
        if vals:
            mses.append(vals[-1])

    total_sent = len(h.get("messages", []))
    suppressed = sum(h.get("suppressed_packets", {}).values())

    return {
        "avg_mse": float(np.mean(mses)) if mses else float("nan"),
        "total_sent": total_sent,
        "total_suppressed": suppressed,
        "history": h,
    }


def print_comparison_table(baseline: dict, apsm: dict):
    """Print a nicely formatted side-by-side comparison table to the terminal."""
    packet_reduction = 0.0
    if baseline["total_sent"] > 0:
        packet_reduction = (apsm["total_suppressed"] / (baseline["total_sent"] + apsm["total_suppressed"])) * 100

    mse_delta_pct = 0.0
    if baseline["avg_mse"] > 0:
        mse_delta_pct = ((apsm["avg_mse"] - baseline["avg_mse"]) / baseline["avg_mse"]) * 100

    print("\n")
    print("=" * 65)
    print("   PHASE 2 COMPARISON: GL-Baseline vs APSM Semantic Filter")
    print("=" * 65)
    print(f"{'Metric':<35} {'GL-Baseline':>12} {'APSM':>12}")
    print("-" * 65)
    print(f"{'Avg MSE (scaled)':<35} {baseline['avg_mse']:>12.5f} {apsm['avg_mse']:>12.5f}")
    print(f"{'Avg MSE (unscaled ~raw)':<35} {baseline['avg_mse']*305**2:>12.1f} {apsm['avg_mse']*305**2:>12.1f}")
    print(f"{'MSE change (%)':<35} {'—':>12} {mse_delta_pct:>+11.2f}%")
    print(f"{'Total packets sent':<35} {baseline['total_sent']:>12d} {apsm['total_sent']:>12d}")
    print(f"{'Packets suppressed (APSM)':<35} {'—':>12} {apsm['total_suppressed']:>12d}")
    print(f"{'Packet Reduction Ratio':<35} {'—':>12} {packet_reduction:>11.1f}%")
    print("=" * 65)

    # Check success criteria
    print("\n  ✅ SUCCESS CRITERIA CHECK:")
    criterion1 = packet_reduction >= 40
    criterion2 = abs(mse_delta_pct) <= 5
    print(f"  {'✅' if criterion1 else '❌'} Packet reduction >= 40%  →  {packet_reduction:.1f}%")
    print(f"  {'✅' if criterion2 else '❌'} MSE delta   <= 5%      →  {abs(mse_delta_pct):.2f}%")
    print()


def plot_results(baseline: dict, apsm: dict, out_dir: Path):
    """Generate and save three publication-quality comparison plots."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Plot 1: Bandwidth Savings Bar Chart ──────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["GL-Baseline\n(Tundo 2025)", "APSM\n(Ours — Phase 2)"]
    sent = [baseline["total_sent"], apsm["total_sent"]]
    suppressed = [0, apsm["total_suppressed"]]
    x = np.arange(len(labels))
    ax.bar(x, sent, color=["#4C72B0", "#55A868"], label="Packets Sent", zorder=3)
    ax.bar(x[1:], suppressed, bottom=sent[1:], color="#C44E52", alpha=0.7, label="Packets Suppressed", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Total Gossip Packets", fontsize=12)
    ax.set_title("Graph 1: Network Bandwidth Usage", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.4, zorder=0)
    plt.tight_layout()
    plt.savefig(out_dir / "graph1_bandwidth.png", dpi=150)
    plt.close()
    print(f"  📊 Saved: {out_dir / 'graph1_bandwidth.png'}")

    # ── Plot 2: MSE Convergence per Node ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, h, title, color in [
        (axes[0], baseline["history"], "GL-Baseline (Tundo 2025)", "#4C72B0"),
        (axes[1], apsm["history"],     "APSM — Semantic Filter",   "#55A868"),
    ]:
        for node_id, metrics in h.get("nodes_test_history", {}).items():
            mse_vals = metrics.get("mse", [])
            if mse_vals:
                ax.plot(mse_vals, alpha=0.7, linewidth=1.5, label=f"Node {node_id}")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Evaluation Step", fontsize=10)
        ax.set_ylabel("MSE (scaled)", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("Graph 2: MSE Convergence — Baseline vs APSM", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "graph2_convergence.png", dpi=150)
    plt.close()
    print(f"  📊 Saved: {out_dir / 'graph2_convergence.png'}")

    # ── Plot 3: Semantic Surprise vs Threshold over Time ──────────────
    apsm_h = apsm["history"]
    node_keys = list(apsm_h.get("semantic_surprise_scores", {}).keys())
    if node_keys:
        node_key = node_keys[0]
        scores = apsm_h["semantic_surprise_scores"][node_key]
        if scores and len(scores[0]) == 3:
            times      = [s[0] for s in scores]
            surprises  = [s[1] for s in scores]
            thresholds = [s[2] for s in scores]

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(times, surprises,  label="Surprise ε(t)", color="#C44E52", linewidth=1.5)
            ax.plot(times, thresholds, label="Threshold τ(t) = 2·σ", color="#4C72B0",
                    linewidth=1.5, linestyle="--")
            ax.fill_between(times, 0, thresholds, alpha=0.1, color="#4C72B0", label="Silent Zone")
            ax.set_xlabel("Simulated Time (s)", fontsize=11)
            ax.set_ylabel("Error Magnitude (scaled)", fontsize=11)
            ax.set_title(f"Graph 3: Semantic Surprise vs Threshold — Node {node_key}",
                         fontsize=13, fontweight="bold")
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / "graph3_semantic_surprise.png", dpi=150)
            plt.close()
            print(f"  📊 Saved: {out_dir / 'graph3_semantic_surprise.png'}")
        else:
            print("  ⚠️  Surprise scores missing threshold column — skipping Graph 3")
    else:
        print("  ⚠️  No semantic surprise data found — skipping Graph 3 (window not filled yet?)")


def main():
    if not DATASETS_FOLDER.exists() or not NETWORKS_FOLDER.exists():
        print("❌  Data not found. Please run 2_dataset_generation.py first.")
        return

    results_phase2 = RESULTS_DIR / "phase 2"

    # ── 1. GL-Baseline ─────────────────────────────────────────────
    baseline_workspace = "experiments/gl_baseline"
    baseline_history_path = Path(baseline_workspace) / "0" / "history.json"

    if baseline_history_path.exists():
        print("✅  GL-Baseline history found — skipping re-run.")
    else:
        print("▶️   Running GL-Baseline (Tundo 2025)...")
        baseline_history_path = run_one_simulation(baseline_workspace)
        print("✅  GL-Baseline done.")

    baseline = extract_metrics(baseline_history_path)

    # ── 2. APSM — Semantic Filter ───────────────────────────────────
    apsm_workspace = "experiments/apsm_phase2"
    print("\n▶️   Running APSM — Phase 2: Semantic Filter...")
    apsm_history_path = run_one_simulation(apsm_workspace)
    print("✅  APSM simulation done.")

    apsm = extract_metrics(apsm_history_path)

    # ── 3. Print table & plots ──────────────────────────────────────
    print_comparison_table(baseline, apsm)

    print("📈  Generating plots...")
    plot_results(baseline, apsm, results_phase2)

    # ── 4. Save comparison summary CSV ──────────────────────────────
    results_phase2.mkdir(parents=True, exist_ok=True)
    csv_path = results_phase2 / "phase2_comparison.csv"
    with open(csv_path, "w") as f:
        f.write("Method,Avg_MSE_scaled,Avg_MSE_raw,Total_Packets_Sent,Suppressed_Packets,Packet_Reduction_pct\n")
        f.write(f"GL-Baseline,{baseline['avg_mse']:.6f},{baseline['avg_mse']*305**2:.1f},{baseline['total_sent']},0,0.0\n")
        pr = (apsm["total_suppressed"] / max(1, baseline["total_sent"] + apsm["total_suppressed"])) * 100
        f.write(f"APSM-Phase2,{apsm['avg_mse']:.6f},{apsm['avg_mse']*305**2:.1f},{apsm['total_sent']},{apsm['total_suppressed']},{pr:.1f}\n")

    print(f"\n  💾  Summary saved to: {csv_path}")
    print("\n✅  Phase 2 evaluation complete!\n")


if __name__ == "__main__":
    main()
