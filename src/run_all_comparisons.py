"""
run_all_comparisons.py
======================
3-WAY FULL-RUN BENCHMARK:
Federated Learning (FedAvg) vs. Gossip Learning (Tundo 2025) vs. APSM (Phase 2)

Faithfully benchmarks all three decentralized/federated paradigms:
  1. Federated Learning (FL / FedAvg): Star topology, central aggregation parameter server.
  2. GL-Baseline: Serverless P2P mesh network, age-weighted merge, uninhibited gossip.
  3. APSM (Phase 2): Serverless P2P mesh network with adaptive predictive-semantic filtering.

Outputs produced:
  • Terminal 3-way comparison table (Accuracy, Convergence, Bandwidth, Topology)
  • 6 publication-quality graphs:
      - Graph 1: Total Network Bandwidth (FL vs GL vs APSM)
      - Graph 2: Convergence Trajectory (MSE vs Training Steps)
      - Graph 3: Accuracy vs Bandwidth Pareto Frontier (The core thesis trade-off)
      - Graph 4: APSM Semantic Surprise Scores vs Adaptive Threshold
      - Graph 5: APSM Per-Node Packet Suppression Breakdown
      - Graph 6: 3-Way Final MSE Distribution Boxplots across seeds
  • Detailed CSV with per-seed, per-paradigm metrics (three_way_comparison_full.csv)
  • Comprehensive JSON summary (three_way_summary.json)
"""

import json
import os
import sys
import time
import shutil
import argparse
import concurrent.futures
import functools
import datetime
import gc
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent

import matplotlib
matplotlib.use("Agg")          
import matplotlib.pyplot as plt
import numpy as np

from gossiplearning.config import Config
from gossiplearning.models import MergeStrategy, StopCriterion
from gossiplearning.weight import weight_by_dataset_size
from utils.data import get_common_test_set
from utils.gossip_training import get_node_dataset, round_trip_fn, run_simulation
from utils.model_creators import create_LSTM
from utils.federated_training import run_federated_simulation

# ──────────────────────────────────────────────────────────────────────
# TensorFlow / CPU+GPU setup — runs at import time in the MAIN process
# ──────────────────────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # silence TF info spam
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"  # Intel oneDNN optimized kernels (AVX2/AVX-512)
import multiprocessing as _mp
_N_CPUS = _mp.cpu_count()
os.environ["OMP_NUM_THREADS"]        = str(_N_CPUS)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(_N_CPUS)
os.environ["TF_NUM_INTEROP_THREADS"] = str(_N_CPUS)

import tensorflow as tf
import tensorflow.config
import tensorflow.config.experimental

tf.config.threading.set_intra_op_parallelism_threads(_N_CPUS)
tf.config.threading.set_inter_op_parallelism_threads(_N_CPUS)

# ── GPU discovery ─────────────────────────────────────────────────────────────
_GPUS = tf.config.list_physical_devices("GPU")
if _GPUS:
    for _gpu in _GPUS:
        try:
            tf.config.experimental.set_memory_growth(_gpu, True)
        except RuntimeError:
            pass
    print(f"🖥️  {len(_GPUS)} GPU(s) detected:")
    for _i, _g in enumerate(_GPUS):
        print(f"      GPU {_i}: {_g.name}")
else:
    print(f"⚡ Running on CPU with {_N_CPUS} threads (oneDNN AVX acceleration enabled)")


def _gpu_for_seed(seed: int, n_gpus: int) -> str:
    if n_gpus == 0:
        return ""
    return str(seed % n_gpus)


from experiment_config import (
    N_NODES,
    K_EDGE_CONNECTIVITY as K,
    MAX_SAMPLES,
    FIXED_UPDATES,
    EPOCHS_PER_UPDATE,
    N_SEEDS,
    PARALLEL_SEEDS,
    SEMANTIC_K,
    SEMANTIC_WINDOW,
    SEMANTIC_HEARTBEAT,
    RUN_FEDERATED,
    FL_ROUNDS,
    FL_EPOCHS,
    FL_FRACTION_FIT,
)

# ──────────────────────────────────────────────────────────────────────
DATASETS_FOLDER = ROOT_DIR / f"data/datasets/porto_{N_NODES}n_{K}k"
NETWORKS_FOLDER = ROOT_DIR / f"data/networks/porto_{N_NODES}n_{K}k"
RESULTS_DIR     = ROOT_DIR / f"results/phase2_full_{N_NODES}n"


# ──────────────────────────────────────────────────────────────────────
# Config helpers
# ──────────────────────────────────────────────────────────────────────

def load_base_config(workspace_dir: str, seed: int, is_baseline: bool) -> dict:
    """Build a simulation config dict from config.json with experiment overrides."""
    with open(SRC_DIR / "config.json", "r") as f:
        cfg = json.load(f)

    cfg["n_nodes"]          = N_NODES
    cfg["workspace_dir"]    = workspace_dir
    cfg["log_level"]        = 0          # ERROR only — keep logs clean

    t = cfg["training"]
    t["merge_strategy"]    = MergeStrategy.AGE_WEIGHTED.value
    t["epochs_per_update"] = EPOCHS_PER_UPDATE
    t["target_probability"] = 1.0        # broadcast to all neighbours
    t["stop_criterion"]    = StopCriterion.FIXED_UPDATES.value
    t["fixed_updates"]     = FIXED_UPDATES
    t["patience"]          = 10
    t["min_delta"]         = 0.001
    t["batch_size"]        = 128         # mini-batch size

    # Phase 2: Semantic Filter
    t["is_baseline"]       = is_baseline
    t["semantic_k"]        = SEMANTIC_K
    t["semantic_window"]   = SEMANTIC_WINDOW
    t["semantic_heartbeat"] = SEMANTIC_HEARTBEAT

    return cfg


# ──────────────────────────────────────────────────────────────────────
# Simulation runners
# ──────────────────────────────────────────────────────────────────────

_dataset_logged = set()
def _log_dataset_sizes(node_index, ds):
    if node_index not in _dataset_logged:
        _dataset_logged.add(node_index)
        print(f"    Node {node_index}: "
              f"train={len(ds['X_train'])}, val={len(ds['X_val'])}, test={len(ds['X_test'])}")


def run_one_simulation(workspace_dir: str, seed: int, is_baseline: bool) -> Path:
    """Run one full Gossip Learning simulation and return the path to its history.json."""
    np.random.seed(seed)
    tf.random.set_seed(seed)

    cfg_json = load_base_config(workspace_dir, seed, is_baseline)
    config   = Config.model_validate(cfg_json)
    model_creator = functools.partial(create_LSTM, config=config)

    _raw_node_data_fn = functools.partial(
        get_node_dataset,
        base_folder=DATASETS_FOLDER,
        simulation_number=0,
        ds_name="",
    )

    def node_data_fn(node_index):
        ds = _raw_node_data_fn(node_index=node_index)
        if MAX_SAMPLES is not None:
            for key in ("X_train", "Y_train", "X_val", "Y_val", "X_test", "Y_test"):
                if key in ds:
                    ds[key] = ds[key][:MAX_SAMPLES]
        _log_dataset_sizes(node_index, ds)
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


def run_one_fl_simulation(workspace_dir: str, seed: int) -> Path:
    """Run one full Federated Learning (FedAvg) simulation and return path to history.json."""
    np.random.seed(seed)
    tf.random.set_seed(seed)

    cfg_json = load_base_config(workspace_dir, seed, is_baseline=True)
    config   = Config.model_validate(cfg_json)
    model_creator = functools.partial(create_LSTM, config=config)

    _raw_node_data_fn = functools.partial(
        get_node_dataset,
        base_folder=DATASETS_FOLDER,
        simulation_number=0,
        ds_name="",
    )

    def node_data_fn(node_index):
        ds = _raw_node_data_fn(node_index=node_index)
        if MAX_SAMPLES is not None:
            for key in ("X_train", "Y_train", "X_val", "Y_val", "X_test", "Y_test"):
                if key in ds:
                    ds[key] = ds[key][:MAX_SAMPLES]
        _log_dataset_sizes(node_index, ds)
        return ds

    get_test_set = functools.partial(
        get_common_test_set,
        node_data_fn=node_data_fn,
        n_nodes=config.n_nodes,
        perc=0.1,
    )

    return run_federated_simulation(
        config=config,
        node_data_fn=node_data_fn,
        model_creator=model_creator,
        get_test_set=get_test_set,
        n_rounds=FL_ROUNDS,
        epochs_per_round=FL_EPOCHS,
        batch_size=config.training.batch_size,
        fraction_fit=FL_FRACTION_FIT,
        workspace_dir=workspace_dir,
        eval_freq=5,
        seed=seed,
    )


# ──────────────────────────────────────────────────────────────────────
# Metrics extraction
# ──────────────────────────────────────────────────────────────────────

def extract_metrics(history_path: Path) -> dict:
    """Extract all relevant metrics from a simulation history.json."""
    with open(history_path) as f:
        h = json.load(f)

    mses, rmses, maes = [], [], []
    for node_metrics in h.get("nodes_test_history", {}).values():
        if node_metrics.get("mse"):
            mses.append(node_metrics["mse"][-1])
        if node_metrics.get("rmse"):
            rmses.append(node_metrics["rmse"][-1])
        if node_metrics.get("mae"):
            maes.append(node_metrics["mae"][-1])

    # Also extract common/global test history if present (from FL)
    global_mse = None
    if h.get("global_test_history", {}).get("mse"):
        global_mse = h["global_test_history"]["mse"][-1]

    total_sent     = len(h.get("messages", []))
    suppressed     = sum(h.get("suppressed_packets", {}).values())
    per_node_supp  = h.get("suppressed_packets", {})

    return {
        "avg_mse":          float(np.mean(mses)) if mses else (global_mse if global_mse else float("nan")),
        "std_mse":          float(np.std(mses))  if mses else float("nan"),
        "avg_rmse":         float(np.mean(rmses)) if rmses else float("nan"),
        "avg_mae":          float(np.mean(maes))  if maes  else float("nan"),
        "total_sent":       total_sent,
        "total_suppressed": suppressed,
        "per_node_supp":    per_node_supp,
        "history":          h,
    }


# ──────────────────────────────────────────────────────────────────────
# 3-Way Pretty-print comparison table
# ──────────────────────────────────────────────────────────────────────

def print_comparison_table(fl_runs: list, b_runs: list, a_runs: list):
    """Aggregate across seeds and print 3-way benchmark comparison table."""

    def agg(runs, key):
        if not runs:
            return float("nan"), float("nan")
        vals = [r[key] for r in runs if not np.isnan(r[key])]
        return np.mean(vals), np.std(vals)

    has_fl = len(fl_runs) > 0
    if has_fl:
        fl_mse_mu,  fl_mse_sd  = agg(fl_runs, "avg_mse")
        fl_rmse_mu, _          = agg(fl_runs, "avg_rmse")
        fl_sent_mu, _          = agg(fl_runs, "total_sent")
    else:
        fl_mse_mu, fl_mse_sd, fl_rmse_mu, fl_sent_mu = float("nan"), float("nan"), float("nan"), float("nan")

    b_mse_mu,  b_mse_sd  = agg(b_runs, "avg_mse")
    a_mse_mu,  a_mse_sd  = agg(a_runs, "avg_mse")
    b_rmse_mu, _         = agg(b_runs, "avg_rmse")
    a_rmse_mu, _         = agg(a_runs, "avg_rmse")
    b_sent_mu, _         = agg(b_runs, "total_sent")
    a_sent_mu, _         = agg(a_runs, "total_sent")
    a_supp_mu, a_supp_sd = agg(a_runs, "total_suppressed")

    total_possible_apsm = b_sent_mu + a_supp_mu
    pkt_red_vs_gl = (a_supp_mu / total_possible_apsm * 100) if total_possible_apsm > 0 else 0
    pkt_red_vs_fl = ((fl_sent_mu - a_sent_mu) / fl_sent_mu * 100) if has_fl and fl_sent_mu > 0 else 0

    mse_delta_vs_gl = ((a_mse_mu - b_mse_mu) / b_mse_mu * 100) if b_mse_mu > 0 else 0
    mse_delta_vs_fl = ((a_mse_mu - fl_mse_mu) / fl_mse_mu * 100) if has_fl and fl_mse_mu > 0 else 0

    SCALE = 305 ** 2   # un-scale from normalised to raw m²

    W = 82
    print("\n" + "=" * W)
    print("   3-WAY BENCHMARK: Federated Learning vs GL-Baseline vs APSM (Phase 2)")
    print(f"   Seeds={N_SEEDS}  |  Rounds/Updates={FIXED_UPDATES}  |  Epochs/round={EPOCHS_PER_UPDATE}")
    print(f"   APSM params: k={SEMANTIC_K}, window={SEMANTIC_WINDOW}, heartbeat={SEMANTIC_HEARTBEAT}")
    print("=" * W)
    print(f"{'Metric':<36} {'FedAvg (FL)':>14} {'GL-Baseline':>14} {'APSM (Ours)':>14}")
    print("-" * W)
    fl_mse_str = f"{fl_mse_mu:.5f}±{fl_mse_sd:.4f}" if has_fl else "—"
    print(f"{'Avg MSE (scaled) mean±std':<36} {fl_mse_str:>14} {b_mse_mu:>10.5f}±{b_mse_sd:.4f} {a_mse_mu:>10.5f}±{a_mse_sd:.4f}")
    
    fl_raw_str = f"{fl_mse_mu*SCALE:.0f}" if has_fl else "—"
    print(f"{'Avg MSE (raw, m²)':<36} {fl_raw_str:>14} {b_mse_mu*SCALE:>14.0f} {a_mse_mu*SCALE:>14.0f}")

    fl_rmse_str = f"{fl_rmse_mu:.5f}" if has_fl else "—"
    print(f"{'Avg RMSE (scaled)':<36} {fl_rmse_str:>14} {b_rmse_mu:>14.5f} {a_rmse_mu:>14.5f}")

    print(f"{'MSE Delta vs GL (%)':<36} {'—':>14} {'0.00%':>14} {mse_delta_vs_gl:>+13.2f}%")
    if has_fl:
        print(f"{'MSE Delta vs FL (%)':<36} {'0.00%':>14} {((b_mse_mu - fl_mse_mu)/fl_mse_mu*100):>+13.2f}% {mse_delta_vs_fl:>+13.2f}%")

    print("-" * W)
    fl_sent_str = f"{int(fl_sent_mu)}" if has_fl else "—"
    print(f"{'Total Packets Transmitted':<36} {fl_sent_str:>14} {int(b_sent_mu):>14} {int(a_sent_mu):>14}")
    print(f"{'Packets Suppressed (Gated)':<36} {'0':>14} {'0':>14} {int(a_supp_mu):>14}")
    print(f"{'Packet Reduction vs GL':<36} {'—':>14} {'0.0%':>14} {pkt_red_vs_gl:>13.1f}%")
    if has_fl:
        print(f"{'Packet Reduction vs FL':<36} {'0.0%':>14} {((fl_sent_mu - b_sent_mu)/fl_sent_mu*100):>13.1f}% {pkt_red_vs_fl:>13.1f}%")

    print("-" * W)
    print(f"{'Network Topology':<36} {'Star (Central)':>14} {'P2P Mesh':>14} {'Sparse Mesh':>14}")
    print(f"{'Server Bottleneck / SPOF':<36} {'YES (High)':>14} {'NO (None)':>14} {'NO (None)':>14}")
    print("=" * W)

    print("\n  ✅ SUCCESS CRITERIA (APSM vs GL):")
    c1 = pkt_red_vs_gl >= 40
    c2 = abs(mse_delta_vs_gl) <= 5
    print(f"  {'✅' if c1 else '❌'}  Packet reduction vs GL ≥ 40%  →  {pkt_red_vs_gl:.1f}%")
    print(f"  {'✅' if c2 else '❌'}  MSE degradation vs GL ≤  5%  →  {abs(mse_delta_vs_gl):.2f}%")
    print()
    return pkt_red_vs_gl, mse_delta_vs_gl


# ──────────────────────────────────────────────────────────────────────
# Publication Plots (6 graphs)
# ──────────────────────────────────────────────────────────────────────

def plot_results(fl_runs: list, b_runs: list, a_runs: list, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Colour palette
    C_FL   = "#DD8452"   # Orange/Coral for FL
    C_BASE = "#4C72B0"   # Blue for GL-Baseline
    C_APSM = "#55A868"   # Green for APSM
    C_SUPP = "#C44E52"   # Red for Suppressed

    has_fl = len(fl_runs) > 0
    b_sent_avg = np.mean([r["total_sent"] for r in b_runs])
    a_sent_avg = np.mean([r["total_sent"] for r in a_runs])
    a_supp_avg = np.mean([r["total_suppressed"] for r in a_runs])
    fl_sent_avg = np.mean([r["total_sent"] for r in fl_runs]) if has_fl else 0

    # ── Graph 1: Bandwidth bar chart (3-way comparison) ──────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    if has_fl:
        labels = ["Federated\nLearning (FL)", "GL-Baseline\n(Tundo 2025)", "APSM\n(Ours — Phase 2)"]
        sent_vals = [fl_sent_avg, b_sent_avg, a_sent_avg]
        colors = [C_FL, C_BASE, C_APSM]
        x = np.arange(3)
        bars = ax.bar(x, sent_vals, color=colors, label="Packets Sent", zorder=3, width=0.5)
        ax.bar(x[2], a_supp_avg, bottom=a_sent_avg, color=C_SUPP, alpha=0.75,
               label="Packets Suppressed (APSM Gated)", zorder=3, width=0.5)
    else:
        labels = ["GL-Baseline\n(Tundo 2025)", "APSM\n(Ours — Phase 2)"]
        sent_vals = [b_sent_avg, a_sent_avg]
        x = np.arange(2)
        bars = ax.bar(x, sent_vals, color=[C_BASE, C_APSM], label="Packets Sent", zorder=3, width=0.5)
        ax.bar(x[1], a_supp_avg, bottom=a_sent_avg, color=C_SUPP, alpha=0.75,
               label="Packets Suppressed", zorder=3, width=0.5)

    for bar, v in zip(bars, sent_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 5, f"{int(v)}", ha="center", fontsize=10, fontweight="bold")
    if has_fl:
        ax.text(x[2], a_sent_avg + a_supp_avg + 5, f"Saved: {int(a_supp_avg)}", ha="center", fontsize=9, color=C_SUPP)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Total Network Messages / Packets", fontsize=12)
    ax.set_title("Graph 1: Network Bandwidth Usage Comparison\n"
                 f"(mean over {N_SEEDS} seeds, {FIXED_UPDATES} rounds)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.4, zorder=0)
    plt.tight_layout()
    plt.savefig(out_dir / "graph1_bandwidth_3way.png", dpi=180)
    plt.close()
    print("  📊  graph1_bandwidth_3way.png")

    # ── Graph 2: Convergence Curves ──────────────────────────────────
    n_cols = 3 if has_fl else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4.5), sharey=True)
    panels = []
    if has_fl:
        panels.append((axes[0], fl_runs, "Federated Learning (FedAvg)", C_FL))
        panels.append((axes[1], b_runs, "GL-Baseline (Tundo 2025)", C_BASE))
        panels.append((axes[2], a_runs, "APSM — Semantic Filter", C_APSM))
    else:
        panels.append((axes[0], b_runs, "GL-Baseline (Tundo 2025)", C_BASE))
        panels.append((axes[1], a_runs, "APSM — Semantic Filter", C_APSM))

    for ax, runs, title, color in panels:
        for s_idx, run in enumerate(runs):
            for node_id, metrics in run["history"].get("nodes_test_history", {}).items():
                mse_vals = metrics.get("mse", [])
                if mse_vals:
                    show_label = (s_idx == 0) and (int(node_id) < 6 or int(node_id) == N_NODES - 1) if N_NODES > 10 else (s_idx == 0)
                    ax.plot(mse_vals,
                            alpha=0.85 if s_idx == 0 else 0.25,
                            linewidth=1.4 if s_idx == 0 else 0.8,
                            color=color,
                            label=f"Node {node_id}" if show_label else "_nolegend_")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Evaluation Step", fontsize=10)
        ax.set_ylabel("MSE (scaled)", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper right", ncol=2)

    fig.suptitle("Graph 2: MSE Convergence Comparison across Paradigms (Seed 0)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "graph2_convergence_3way.png", dpi=180)
    plt.close()
    print("  📊  graph2_convergence_3way.png")

    # ── Graph 3: Pareto Frontier (Accuracy vs Bandwidth) ─────────────
    # The crucial thesis plot: x = Total Packets Sent, y = Test MSE
    fig, ax = plt.subplots(figsize=(8, 5.5))
    if has_fl:
        fl_mse_pts = [r["avg_mse"] for r in fl_runs]
        fl_pkt_pts = [r["total_sent"] for r in fl_runs]
        ax.scatter(fl_pkt_pts, fl_mse_pts, color=C_FL, s=120, marker="o", label="Federated Learning (FL)", zorder=5)

    b_mse_pts = [r["avg_mse"] for r in b_runs]
    b_pkt_pts = [r["total_sent"] for r in b_runs]
    ax.scatter(b_pkt_pts, b_mse_pts, color=C_BASE, s=120, marker="s", label="GL-Baseline (Tundo 2025)", zorder=5)

    a_mse_pts = [r["avg_mse"] for r in a_runs]
    a_pkt_pts = [r["total_sent"] for r in a_runs]
    ax.scatter(a_pkt_pts, a_mse_pts, color=C_APSM, s=160, marker="*", label="APSM (Ours — Phase 2)", zorder=5)

    # Annotate points
    if has_fl:
        ax.annotate("FL (Centralized Bottleneck)", (np.mean(fl_pkt_pts), np.mean(fl_mse_pts)),
                    textcoords="offset points", xytext=(10, 10), fontsize=9, fontweight="bold", color=C_FL)
    ax.annotate("GL-Baseline (High P2P Traffic)", (np.mean(b_pkt_pts), np.mean(b_mse_pts)),
                textcoords="offset points", xytext=(10, -15), fontsize=9, fontweight="bold", color=C_BASE)
    ax.annotate("APSM (Optimal Pareto Front: Low Bandwidth, High Accuracy)", (np.mean(a_pkt_pts), np.mean(a_mse_pts)),
                textcoords="offset points", xytext=(10, 10), fontsize=9, fontweight="bold", color=C_APSM)

    ax.set_xlabel("Total Packets Transmitted over Network (Lower is Better)", fontsize=11)
    ax.set_ylabel("Final Test MSE (Lower is Better)", fontsize=11)
    ax.set_title("Graph 3: Accuracy vs. Bandwidth Pareto Frontier", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / "graph3_pareto_accuracy_vs_bandwidth.png", dpi=180)
    plt.close()
    print("  📊  graph3_pareto_accuracy_vs_bandwidth.png")

    # ── Graph 4: Semantic Surprise vs Threshold (APSM) ───────────────
    apsm_h = a_runs[0]["history"]
    node_keys = list(apsm_h.get("semantic_surprise_scores", {}).keys())
    if node_keys:
        fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharey=False)
        axes = axes.flatten()
        for idx, nk in enumerate(node_keys[:10]):
            ax = axes[idx]
            scores = apsm_h["semantic_surprise_scores"].get(nk, [])
            if scores and len(scores[0]) == 3:
                times      = [s[0] for s in scores]
                surprises  = [s[1] for s in scores]
                thresholds = [s[2] for s in scores]
                ax.plot(times, surprises,  label="ε(t) surprise", color=C_SUPP,  lw=1.5)
                ax.plot(times, thresholds, label="τ(t) threshold", color=C_BASE, lw=1.5, ls="--")
                ax.fill_between(times, 0, thresholds, alpha=0.08, color=C_BASE)
                ax.set_title(f"Node {nk}", fontsize=9)
                ax.set_xlabel("Sim Time (s)", fontsize=8)
                ax.grid(alpha=0.3)
                if idx == 0:
                    ax.legend(fontsize=7)
        fig.suptitle("Graph 4: APSM Semantic Surprise ε(t) vs Adaptive Threshold τ(t) — Nodes 0..9",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(out_dir / "graph4_semantic_surprise.png", dpi=180)
        plt.close()
        print("  📊  graph4_semantic_surprise.png")

    # ── Graph 5: Per-node suppression count (APSM) ───────────────────
    per_node = a_runs[0]["per_node_supp"]
    if per_node:
        nodes  = sorted(per_node.keys(), key=lambda x: int(x))
        counts = [per_node[n] for n in nodes]
        fig_w = max(10, len(nodes) * 0.25)
        fig, ax = plt.subplots(figsize=(fig_w, 4.5))
        bars = ax.bar(nodes, counts, color=C_SUPP, alpha=0.8, zorder=3)
        if len(nodes) <= 25:
            for bar, c in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width() / 2, c + 0.3, str(c), ha="center", fontsize=8)
        ax.tick_params(axis="x", rotation=90 if len(nodes) > 15 else 0, labelsize=7 if len(nodes) > 20 else 9)
        ax.set_xlabel("Node ID", fontsize=11)
        ax.set_ylabel("Packets Suppressed", fontsize=11)
        ax.set_title("Graph 5: Per-Node Packet Suppression (APSM Phase 2 — Seed 0)",
                     fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.4, zorder=0)
        plt.tight_layout()
        plt.savefig(out_dir / "graph5_per_node_suppression.png", dpi=180)
        plt.close()
        print("  📊  graph5_per_node_suppression.png")

    # ── Graph 6: 3-Way Final MSE Distribution Boxplot ────────────────
    b_mses = [r["avg_mse"] for r in b_runs]
    a_mses = [r["avg_mse"] for r in a_runs]
    fig, ax = plt.subplots(figsize=(8, 5))
    if has_fl:
        fl_mses = [r["avg_mse"] for r in fl_runs]
        data_to_plot = [fl_mses, b_mses, a_mses]
        box_labels = ["Federated\nLearning", "GL-Baseline\n(Tundo 2025)", "APSM\n(Ours — Phase 2)"]
        colors = [C_FL, C_BASE, C_APSM]
    else:
        data_to_plot = [b_mses, a_mses]
        box_labels = ["GL-Baseline\n(Tundo 2025)", "APSM\n(Ours — Phase 2)"]
        colors = [C_BASE, C_APSM]

    bp = ax.boxplot(data_to_plot, patch_artist=True, notch=False,
                    boxprops=dict(facecolor="white"),
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color + "66")

    ax.set_xticklabels(box_labels, fontsize=11)
    ax.set_ylabel("Avg Final MSE (scaled)", fontsize=11)
    ax.set_title(f"Graph 6: Final MSE Distribution across {N_SEEDS} Seeds",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / "graph6_mse_boxplot_3way.png", dpi=180)
    plt.close()
    print("  📊  graph6_mse_boxplot_3way.png")


# ──────────────────────────────────────────────────────────────────────
# Save CSV + JSON
# ──────────────────────────────────────────────────────────────────────

def save_results(fl_runs: list, b_runs: list, a_runs: list, out_dir: Path, pkt_red: float, mse_delta: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    SCALE = 305 ** 2
    has_fl = len(fl_runs) > 0

    # Detailed per-seed CSV
    csv_path = out_dir / "three_way_comparison_full.csv"
    with open(csv_path, "w") as f:
        f.write("seed,method,avg_mse_scaled,avg_mse_raw,avg_rmse,avg_mae,total_sent,suppressed,packet_reduction_pct\n")
        for seed_idx in range(len(b_runs)):
            br = b_runs[seed_idx]
            ar = a_runs[seed_idx]
            total = br["total_sent"] + ar["total_suppressed"]
            pr = (ar["total_suppressed"] / total * 100) if total > 0 else 0

            if has_fl and seed_idx < len(fl_runs):
                fl = fl_runs[seed_idx]
                f.write(f"{seed_idx},FederatedLearning,{fl['avg_mse']:.6f},{fl['avg_mse']*SCALE:.1f},"
                        f"{fl['avg_rmse']:.6f},{fl['avg_mae']:.6f},{fl['total_sent']},0,0.0\n")

            f.write(f"{seed_idx},GL-Baseline,{br['avg_mse']:.6f},{br['avg_mse']*SCALE:.1f},"
                    f"{br['avg_rmse']:.6f},{br['avg_mae']:.6f},{br['total_sent']},0,0.0\n")
            f.write(f"{seed_idx},APSM-Phase2,{ar['avg_mse']:.6f},{ar['avg_mse']*SCALE:.1f},"
                    f"{ar['avg_rmse']:.6f},{ar['avg_mae']:.6f},{ar['total_sent']},"
                    f"{ar['total_suppressed']},{pr:.1f}\n")

    # Summary JSON
    summary = {
        "run_timestamp": datetime.datetime.now().isoformat(),
        "config": {
            "n_nodes": N_NODES,
            "fixed_updates": FIXED_UPDATES,
            "epochs_per_update": EPOCHS_PER_UPDATE,
            "n_seeds": N_SEEDS,
            "max_samples": MAX_SAMPLES,
            "semantic_k": SEMANTIC_K,
            "semantic_window": SEMANTIC_WINDOW,
            "semantic_heartbeat": SEMANTIC_HEARTBEAT,
            "run_federated": RUN_FEDERATED,
            "fl_rounds": FL_ROUNDS,
            "fl_epochs": FL_EPOCHS,
            "fl_fraction_fit": FL_FRACTION_FIT,
        },
        "results": {
            "packet_reduction_pct": round(pkt_red, 2),
            "mse_delta_pct": round(mse_delta, 2),
            "packet_reduction_pct_vs_gl": round(pkt_red, 2),
            "mse_delta_pct_vs_gl": round(mse_delta, 2),
            "baseline_mse_mean": round(float(np.mean([r["avg_mse"] for r in b_runs])), 6),
            "apsm_mse_mean":     round(float(np.mean([r["avg_mse"] for r in a_runs])), 6),
            "fl_mse_mean":       round(float(np.mean([r["avg_mse"] for r in fl_runs])), 6) if has_fl else None,
            "baseline_mse_std":  round(float(np.std([r["avg_mse"] for r in b_runs])), 6),
            "apsm_mse_std":      round(float(np.std([r["avg_mse"] for r in a_runs])), 6),
            "fl_mse_std":        round(float(np.std([r["avg_mse"] for r in fl_runs])), 6) if has_fl else None,
            "baseline_packets":  int(np.mean([r["total_sent"] for r in b_runs])),
            "apsm_packets_sent": int(np.mean([r["total_sent"] for r in a_runs])),
            "apsm_packets_suppressed": int(np.mean([r["total_suppressed"] for r in a_runs])),
            "fl_packets":        int(np.mean([r["total_sent"] for r in fl_runs])) if has_fl else None,
        }
    }
    json_path = out_dir / "three_way_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Backwards compatibility summary
    with open(out_dir / "phase2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  💾  CSV:  {csv_path}")
    print(f"  💾  JSON: {json_path}")


# ──────────────────────────────────────────────────────────────────────
# Seed Execution Runner
# ──────────────────────────────────────────────────────────────────────

def run_seed(seed: int, gpu_id: str = "", force_fl: bool = False, force_gl: bool = False, force_apsm: bool = False) -> tuple[dict, dict, dict]:
    import os as _os
    import sys as _sys
    import gc
    import shutil as _shutil
    import multiprocessing as _mp
    from pathlib import Path

    if gpu_id != "":
        _os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    _os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    _os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

    _src = str(Path(__file__).resolve().parent)
    if _src not in _sys.path:
        _sys.path.insert(0, _src)

    _n = _mp.cpu_count()
    _os.environ["OMP_NUM_THREADS"]        = str(_n)
    _os.environ["TF_NUM_INTRAOP_THREADS"] = str(_n)
    _os.environ["TF_NUM_INTEROP_THREADS"] = str(_n)

    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(_n)
    tf.config.threading.set_inter_op_parallelism_threads(_n)

    gpus = tf.config.list_physical_devices("GPU")
    gpu_label = f"GPU {gpu_id} ({gpus[0].name})" if gpus else "CPU"
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    seed_start = time.time()
    print(f"\n{'━'*60}\n  SEED {seed + 1}/{N_SEEDS}  →  {gpu_label}  |  PID {os.getpid()}\n{'━'*60}")

    # ── 1. Federated Learning (FedAvg) ─────────────────────────
    fl_metrics = None
    if RUN_FEDERATED:
        fl_workspace = str(SRC_DIR / f"experiments/fl_baseline_{N_NODES}n_seed{seed}")
        fl_hist_path = Path(fl_workspace) / "0" / "history.json"
        
        # Legacy fallback if running 10 nodes and un-suffixed directory exists
        if not fl_hist_path.exists() and N_NODES == 10:
            legacy_fl = Path(str(SRC_DIR / f"experiments/fl_baseline_seed{seed}")) / "0" / "history.json"
            if legacy_fl.exists():
                fl_hist_path = legacy_fl

        if force_fl and Path(fl_workspace).exists():
            _shutil.rmtree(fl_workspace, ignore_errors=True)

        fl_start = time.time()
        if fl_hist_path.exists():
            print(f"  ✅  Federated Learning (seed {seed}) — cached, skipping.")
        else:
            print(f"  ▶️   Federated Learning (seed {seed}) — running on {gpu_label}...")
            fl_hist_path = run_one_fl_simulation(fl_workspace, seed)
            elapsed = time.time() - fl_start
            print(f"  ✅  Federated Learning (seed {seed}) done  ({elapsed/60:.1f} min)")

        fl_metrics = extract_metrics(fl_hist_path)
        tf.keras.backend.clear_session()
        gc.collect()

    # ── 2. GL-Baseline ─────────────────────────────────────────
    b_workspace = str(SRC_DIR / f"experiments/gl_baseline_{N_NODES}n_seed{seed}")
    b_hist_path = Path(b_workspace) / "0" / "history.json"

    # Legacy fallback if running 10 nodes and un-suffixed directory exists
    if not b_hist_path.exists() and N_NODES == 10:
        legacy_b = Path(str(SRC_DIR / f"experiments/gl_baseline_seed{seed}")) / "0" / "history.json"
        if legacy_b.exists():
            b_hist_path = legacy_b

    if force_gl and Path(b_workspace).exists():
        _shutil.rmtree(b_workspace, ignore_errors=True)

    b_start = time.time()
    if b_hist_path.exists():
        print(f"  ✅  GL-Baseline (seed {seed}) — cached, skipping.")
    else:
        print(f"  ▶️   GL-Baseline (seed {seed}) — running on {gpu_label}...")
        b_hist_path = run_one_simulation(b_workspace, seed, is_baseline=True)
        elapsed = time.time() - b_start
        print(f"  ✅  GL-Baseline (seed {seed}) done  ({elapsed/60:.1f} min)")

    b_metrics = extract_metrics(b_hist_path)
    tf.keras.backend.clear_session()
    gc.collect()

    # ── 3. APSM (Phase 2) ──────────────────────────────────────
    a_workspace = str(SRC_DIR / f"experiments/apsm_phase2_{N_NODES}n_seed{seed}")
    a_hist_path = Path(a_workspace) / "0" / "history.json"

    # Legacy fallback if running 10 nodes and un-suffixed directory exists
    if not a_hist_path.exists() and N_NODES == 10:
        legacy_a = Path(str(SRC_DIR / f"experiments/apsm_phase2_seed{seed}")) / "0" / "history.json"
        if legacy_a.exists():
            a_hist_path = legacy_a

    if force_apsm and Path(a_workspace).exists():
        _shutil.rmtree(a_workspace, ignore_errors=True)

    a_start = time.time()
    if a_hist_path.exists():
        print(f"  ✅  APSM Phase 2 (seed {seed}) — cached, skipping.")
    else:
        print(f"  ▶️   APSM Phase 2 (seed {seed}) — running on {gpu_label}...")
        a_hist_path = run_one_simulation(a_workspace, seed, is_baseline=False)
        elapsed = time.time() - a_start
        print(f"  ✅  APSM (seed {seed}) done  ({elapsed/60:.1f} min)")

    a_metrics = extract_metrics(a_hist_path)
    tf.keras.backend.clear_session()
    gc.collect()

    return fl_metrics, b_metrics, a_metrics


def main():
    parser = argparse.ArgumentParser(description="3-Way Benchmark: FL vs GL vs APSM")
    parser.add_argument("--force", "--no-cache", dest="force", action="store_true",
                        help="Force rerun all paradigms (FL, GL, and APSM), ignoring cache.")
    parser.add_argument("--force-gossip", "--no-cache-gossip", dest="force_gossip", action="store_true",
                        help="Force rerun GL-Baseline and APSM Phase 2, keeping FL cache.")
    parser.add_argument("--force-fl", "--no-cache-fl", dest="force_fl", action="store_true",
                        help="Force rerun Federated Learning baseline only.")
    parser.add_argument("--force-gl", "--no-cache-gl", dest="force_gl", action="store_true",
                        help="Force rerun GL-Baseline only.")
    parser.add_argument("--force-apsm", "--no-cache-apsm", dest="force_apsm", action="store_true",
                        help="Force rerun APSM Phase 2 only.")
    args = parser.parse_args()

    force_fl   = args.force or args.force_fl
    force_gl   = args.force or args.force_gossip or args.force_gl
    force_apsm = args.force or args.force_gossip or args.force_apsm

    if not DATASETS_FOLDER.exists() or not NETWORKS_FOLDER.exists():
        print(f"❌  Data not found at: {DATASETS_FOLDER}")
        print("    Run 1_network_generation.py and 2_dataset_generation.py first, then retry.")
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    total_start = time.time()

    print("\n" + "=" * 65)
    print("  3-WAY BENCHMARK: FL (FedAvg) vs GL-Baseline vs APSM (Phase 2)")
    print(f"  Seeds={N_SEEDS} | Updates={FIXED_UPDATES} | Epochs/round={EPOCHS_PER_UPDATE}")
    print(f"  Include FL={RUN_FEDERATED} (Rounds={FL_ROUNDS}, Fraction={FL_FRACTION_FIT})")
    print(f"  MAX_SAMPLES={'FULL DATA' if MAX_SAMPLES is None else MAX_SAMPLES}")
    if force_fl or force_gl or force_apsm:
        print(f"  Force rerun flags: FL={force_fl}, GL={force_gl}, APSM={force_apsm}")
    print("=" * 65 + "\n")

    fl_runs       = [None] * N_SEEDS
    baseline_runs = [None] * N_SEEDS
    apsm_runs     = [None] * N_SEEDS

    max_workers = min(N_SEEDS, 4) if PARALLEL_SEEDS else 1

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        n_gpus = len(_GPUS)
        futures = {
            executor.submit(run_seed, seed, _gpu_for_seed(seed, n_gpus), force_fl, force_gl, force_apsm): seed
            for seed in range(N_SEEDS)
        }
        for future in concurrent.futures.as_completed(futures):
            seed = futures[future]
            try:
                fl_m, b_m, a_m = future.result()
                if fl_m is not None:
                    fl_runs[seed] = fl_m
                baseline_runs[seed] = b_m
                apsm_runs[seed] = a_m

                # Progress snapshot
                total = b_m["total_sent"] + a_m["total_suppressed"]
                pr_now = (a_m["total_suppressed"] / total * 100) if total > 0 else 0
                fl_str = f"FL MSE = {fl_m['avg_mse']:.5f} | " if fl_m else ""
                print(f"\n  📊  Seed {seed} snapshot:")
                print(f"      {fl_str}GL MSE = {b_m['avg_mse']:.5f} | APSM MSE = {a_m['avg_mse']:.5f}")
                print(f"      Packets sent: GL={b_m['total_sent']}, APSM={a_m['total_sent']}"
                      + (f", FL={fl_m['total_sent']}" if fl_m else ""))
                print(f"      Suppressed (APSM): {a_m['total_suppressed']} → {pr_now:.1f}% reduction vs GL")
            except Exception as e:
                import traceback
                print(f"❌  Seed {seed} failed with error: {e}")
                traceback.print_exc()

    baseline_runs = [r for r in baseline_runs if r is not None]
    apsm_runs     = [r for r in apsm_runs if r is not None]
    fl_runs       = [r for r in fl_runs if r is not None]

    if not baseline_runs or not apsm_runs:
        print("❌ Simulations failed.")
        sys.exit(1)

    # ── Final aggregated 3-way table ───────────────────────────────
    pkt_red, mse_delta = print_comparison_table(fl_runs, baseline_runs, apsm_runs)

    # ── 6 Master Plots ─────────────────────────────────────────────
    print("📈  Generating 6 master comparison plots...")
    plot_results(fl_runs, baseline_runs, apsm_runs, RESULTS_DIR)

    # ── Save CSVs / JSON ───────────────────────────────────────────
    save_results(fl_runs, baseline_runs, apsm_runs, RESULTS_DIR, pkt_red, mse_delta)

    total_elapsed = (time.time() - total_start) / 60
    print(f"\n🎉  All done!  Total wall-clock time: {total_elapsed:.1f} min")
    print(f"📁  Results saved to: {RESULTS_DIR.resolve()}\n")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
