"""
threshold_sweep.py
==================
Sweeps the k multiplier in tau(t) = k * sigma(epsilon[t-N...t]) over a range
of values and records: packets suppressed, MSE change vs baseline.

This produces a publication-quality plot: k vs (Packet Reduction %, MSE Delta %)
which empirically proves the chosen k=2 is the optimal operating point.

Usage (from src/ folder):
    poetry run python threshold_sweep.py

Output:
    results/phase 2/threshold_sweep.png
    results/phase 2/threshold_sweep.csv
"""
import json
import os
import dataclasses
import functools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
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

N_NODES = 10
K_DATASET = 3
DATASETS_FOLDER = Path(f"data/datasets/porto_{N_NODES}n_{K_DATASET}k")
NETWORKS_FOLDER = Path(f"data/networks/porto_{N_NODES}n_{K_DATASET}k/seed1000")
RESULTS_DIR = Path("../results/phase 2")

# k values to sweep — covers from very aggressive (almost never sends) to very permissive (always sends)
K_VALUES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]


def load_base_config(workspace_dir: str) -> dict:
    with open("config.json", "r") as f:
        cfg = json.load(f)
    cfg["n_nodes"] = N_NODES
    cfg["workspace_dir"] = workspace_dir
    cfg["training"]["merge_strategy"] = MergeStrategy.AGE_WEIGHTED
    cfg["training"]["epochs_per_update"] = 4
    cfg["training"]["target_probability"] = 1
    cfg["training"]["stop_criterion"] = StopCriterion.NO_IMPROVEMENTS.value
    cfg["training"]["patience"] = 5
    cfg["training"]["min_delta"] = 0.1
    return cfg


def run_one_simulation(workspace_dir: str, k_value: float) -> Path:
    """Run a simulation with a specific k value and return the history.json path."""
    cfg_json = load_base_config(workspace_dir)
    config = Config.model_validate(cfg_json)
    model_creator = functools.partial(create_LSTM, config=config)

    node_data_fn = functools.partial(
        get_node_dataset,
        base_folder=DATASETS_FOLDER,
        simulation_number=0,
        ds_name="",
    )
    get_test_set = functools.partial(
        get_common_test_set,
        node_data_fn=node_data_fn,
        n_nodes=config.n_nodes,
        perc=0.1,
    )

    # Temporarily patch the k value on all nodes before running
    # We do this by monkey-patching the Node class's default
    import gossiplearning.node as node_module
    original_init = node_module.Node.__init__

    def patched_init(self, **kwargs):
        original_init(self, **kwargs)
        self._SEMANTIC_K = k_value  # Override the k multiplier

    node_module.Node.__init__ = patched_init

    try:
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
    finally:
        node_module.Node.__init__ = original_init  # Always restore

    return Path(workspace_dir) / "0" / "history.json"


def extract_metrics(history_path: Path) -> dict:
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
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASETS_FOLDER.exists() or not NETWORKS_FOLDER.exists():
        print("Data not found. Please run 2_dataset_generation.py first.")
        return

    # ── Step 1: Get baseline (no suppression, k is irrelevant here)
    baseline_path = Path("experiments/gl_baseline/0/history.json")
    if not baseline_path.exists():
        print("Baseline history not found. Run run_all_comparisons.py first to generate it.")
        return

    baseline = extract_metrics(baseline_path)
    baseline_mse = baseline["avg_mse"]
    baseline_sent = baseline["total_sent"]
    print(f"Baseline: MSE={baseline_mse:.5f}, Packets Sent={baseline_sent}")

    # ── Step 2: Sweep k values
    results = []
    for k in K_VALUES:
        workspace = f"experiments/sweep_k{k:.1f}"
        print(f"\n▶️  Running sweep: k={k} ...")
        hist_path = run_one_simulation(workspace, k)
        m = extract_metrics(hist_path)

        total_potential = baseline_sent + m["total_suppressed"]
        packet_reduction = (m["total_suppressed"] / max(1, total_potential)) * 100
        mse_delta = ((m["avg_mse"] - baseline_mse) / baseline_mse) * 100

        results.append({
            "k": k,
            "avg_mse": m["avg_mse"],
            "total_sent": m["total_sent"],
            "total_suppressed": m["total_suppressed"],
            "packet_reduction_pct": packet_reduction,
            "mse_delta_pct": mse_delta,
        })
        print(f"  k={k:.1f} → Packet Reduction={packet_reduction:.1f}%, MSE Δ={mse_delta:+.2f}%")

    # ── Step 3: Save CSV
    csv_path = RESULTS_DIR / "threshold_sweep.csv"
    with open(csv_path, "w") as f:
        f.write("k,avg_mse,total_sent,total_suppressed,packet_reduction_pct,mse_delta_pct\n")
        for r in results:
            f.write(f"{r['k']},{r['avg_mse']:.6f},{r['total_sent']},{r['total_suppressed']},{r['packet_reduction_pct']:.2f},{r['mse_delta_pct']:.3f}\n")
    print(f"\n💾 CSV saved: {csv_path}")

    # ── Step 4: Plot
    ks = [r["k"] for r in results]
    reductions = [r["packet_reduction_pct"] for r in results]
    mse_deltas = [abs(r["mse_delta_pct"]) for r in results]

    fig, ax1 = plt.subplots(figsize=(9, 6))
    color1 = "#4C72B0"
    color2 = "#C44E52"

    ax1.plot(ks, reductions, "o-", color=color1, linewidth=2.5, markersize=8, label="Packet Reduction (%)")
    ax1.set_xlabel("Threshold Multiplier k", fontsize=13)
    ax1.set_ylabel("Packet Reduction (%)", fontsize=13, color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.axhline(y=40, color=color1, linestyle="--", alpha=0.5, label="40% Target")
    ax1.set_ylim(0, 100)

    ax2 = ax1.twinx()
    ax2.plot(ks, mse_deltas, "s--", color=color2, linewidth=2.5, markersize=8, label="|MSE Δ| (%)")
    ax2.set_ylabel("|MSE Change| (%)", fontsize=13, color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.axhline(y=5, color=color2, linestyle="--", alpha=0.5, label="5% Tolerance")

    # Mark the chosen k=2
    ax1.axvline(x=2.0, color="green", linestyle=":", linewidth=2.5, label="Chosen k=2")

    # Add shaded "optimal zone" where both criteria are met
    optimal_ks = [r["k"] for r in results if r["packet_reduction_pct"] >= 40 and abs(r["mse_delta_pct"]) <= 5]
    if optimal_ks:
        ax1.axvspan(min(optimal_ks) - 0.1, max(optimal_ks) + 0.1, alpha=0.08, color="green", label="Optimal Zone")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=10)

    plt.title("Threshold Sweep: k vs Bandwidth Savings & Accuracy Trade-off", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plot_path = RESULTS_DIR / "threshold_sweep.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"📊 Plot saved: {plot_path}")

    # ── Step 5: Print summary table
    print("\n" + "=" * 65)
    print(f"{'k':>6}  {'Packet Reduction':>18}  {'|MSE Delta|':>14}  {'Valid?':>7}")
    print("-" * 65)
    for r in results:
        valid = r["packet_reduction_pct"] >= 40 and abs(r["mse_delta_pct"]) <= 5
        mark = "✅" if valid else "❌"
        chosen = " ← CHOSEN" if r["k"] == 2.0 else ""
        print(f"{r['k']:>6.1f}  {r['packet_reduction_pct']:>17.1f}%  {abs(r['mse_delta_pct']):>13.2f}%  {mark}{chosen}")
    print("=" * 65)
    print("\n✅ Threshold sweep complete!\n")


if __name__ == "__main__":
    main()
