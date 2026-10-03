"""
run_all_comparisons.py
======================
FULL-RUN VERSION  —  Designed for Kaggle GPU (T4 / P100).

Runs GL-Baseline and APSM (Phase 2: Semantic Filter) back-to-back across
multiple random seeds, producing:
  • A terminal comparison table
  • 5 publication-quality graphs  (bandwidth, convergence, surprise,
                                   per-node suppression, RMSE curve)
  • A detailed CSV with per-seed metrics
  • A JSON summary of all results

Usage (from src/ folder):
    python run_all_comparisons.py

Estimated time on Kaggle GPU T4:  ~45–90 min for N_SEEDS=3
"""

import json
import os
import sys
import time
import concurrent.futures
import functools
import datetime
import gc
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent

import matplotlib
matplotlib.use("Agg")          # headless — no display needed on Kaggle
import matplotlib.pyplot as plt
import numpy as np

from gossiplearning.config import Config
from gossiplearning.models import MergeStrategy, StopCriterion
from gossiplearning.weight import weight_by_dataset_size
from utils.data import get_common_test_set
from utils.gossip_training import get_node_dataset, round_trip_fn, run_simulation
from utils.model_creators import create_LSTM

# ──────────────────────────────────────────────────────────────────────
# TensorFlow / CPU+GPU setup — runs at import time in the MAIN process
# ──────────────────────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # silence TF info spam
# Use ALL available CPU cores for both inter-op and intra-op parallelism.
import multiprocessing as _mp
_N_CPUS = _mp.cpu_count()   # 24 on i9-13950HX
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
    if len(_GPUS) >= 2:
        print("      GPU 0 → 16 GB  (seeds 0, 2)")
        print("      GPU 1 →  6 GB  RTX 1000 Ada  (seed 1)")
else:
    print(f"⚠️  No GPU detected — running on CPU with {_N_CPUS} threads")


def _gpu_for_seed(seed: int, n_gpus: int) -> str:
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
    SEMANTIC_HEARTBEAT
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
    t["batch_size"]        = 128         # larger batch = faster GPU utilisation

    # Phase 2: Semantic Filter
    t["is_baseline"]       = is_baseline
    t["semantic_k"]        = SEMANTIC_K
    t["semantic_window"]   = SEMANTIC_WINDOW
    t["semantic_heartbeat"] = SEMANTIC_HEARTBEAT

    return cfg


# ──────────────────────────────────────────────────────────────────────
# Simulation runner
# ──────────────────────────────────────────────────────────────────────

def run_one_simulation(workspace_dir: str, seed: int, is_baseline: bool) -> Path:
    """Run one full simulation and return the path to its history.json."""
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
            for key in ("X_train", "Y_train", "X_val", "Y_val"):
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


_dataset_logged = set()
def _log_dataset_sizes(node_index, ds):
    if node_index not in _dataset_logged:
        _dataset_logged.add(node_index)
        print(f"    Node {node_index}: "
              f"train={len(ds['X_train'])}, val={len(ds['X_val'])}")


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

    total_sent     = len(h.get("messages", []))
    suppressed     = sum(h.get("suppressed_packets", {}).values())
    per_node_supp  = h.get("suppressed_packets", {})

    return {
        "avg_mse":          float(np.mean(mses))  if mses  else float("nan"),
        "std_mse":          float(np.std(mses))   if mses  else float("nan"),
        "avg_rmse":         float(np.mean(rmses)) if rmses else float("nan"),
        "avg_mae":          float(np.mean(maes))  if maes  else float("nan"),
        "total_sent":       total_sent,
        "total_suppressed": suppressed,
        "per_node_supp":    per_node_supp,
        "history":          h,
    }


# ──────────────────────────────────────────────────────────────────────
# Pretty-print comparison table
# ──────────────────────────────────────────────────────────────────────

def print_comparison_table(b_runs: list, a_runs: list):
    """Aggregate across seeds and print mean ± std."""

    def agg(runs, key):
        vals = [r[key] for r in runs if not np.isnan(r[key])]
        return np.mean(vals), np.std(vals)

    b_mse_mu,  b_mse_sd  = agg(b_runs, "avg_mse")
    a_mse_mu,  a_mse_sd  = agg(a_runs, "avg_mse")
    b_rmse_mu, _         = agg(b_runs, "avg_rmse")
    a_rmse_mu, _         = agg(a_runs, "avg_rmse")
    b_sent_mu, _         = agg(b_runs, "total_sent")
    a_sent_mu, _         = agg(a_runs, "total_sent")
    a_supp_mu, a_supp_sd = agg(a_runs, "total_suppressed")

    total_possible  = b_sent_mu + a_supp_mu
    packet_red_pct  = (a_supp_mu / total_possible * 100) if total_possible > 0 else 0
    mse_delta_pct   = ((a_mse_mu - b_mse_mu) / b_mse_mu * 100) if b_mse_mu > 0 else 0

    SCALE = 305 ** 2   # un-scale from normalised to raw m²

    W = 68
    print("\n" + "=" * W)
    print("   PHASE 2 FULL-RUN: GL-Baseline vs APSM Semantic Filter")
    print(f"   Seeds={N_SEEDS}  |  Updates={FIXED_UPDATES}  |  Epochs/round={EPOCHS_PER_UPDATE}")
    print(f"   APSM params: k={SEMANTIC_K}, window={SEMANTIC_WINDOW}, heartbeat={SEMANTIC_HEARTBEAT}")
    print("=" * W)
    print(f"{'Metric':<38} {'GL-Baseline':>14} {'APSM':>14}")
    print("-" * W)
    print(f"{'Avg MSE (scaled)  mean±std':<38} {b_mse_mu:>10.5f}±{b_mse_sd:.4f} {a_mse_mu:>10.5f}±{a_mse_sd:.4f}")
    print(f"{'Avg MSE (raw, m²)':<38} {b_mse_mu*SCALE:>14.0f} {a_mse_mu*SCALE:>14.0f}")
    print(f"{'Avg RMSE (scaled)':<38} {b_rmse_mu:>14.5f} {a_rmse_mu:>14.5f}")
    print(f"{'MSE delta (%)':<38} {'—':>14} {mse_delta_pct:>+13.2f}%")
    # Energy calculations based on Table X of the Base Paper
    energy_send_j = 0.07    # Joules to SendModel
    energy_recv_j = 20.74   # Joules to ReceiveModel
    energy_fit_j = 1.93     # Joules for Model fit
    
    # Each suppressed packet saves 1 Send, and K Recvs + K Fits on neighbor nodes
    energy_saved_per_packet = energy_send_j + (K * (energy_recv_j + energy_fit_j))
    total_energy_saved_j = a_supp_mu * energy_saved_per_packet
    total_energy_saved_kwh = total_energy_saved_j / 3.6e6  # Convert Joules to kWh

    print(f"{'Avg packets sent':<38} {b_sent_mu:>14.0f} {a_sent_mu:>14.0f}")
    print(f"{'Avg packets suppressed':<38} {'—':>14} {a_supp_mu:>14.0f}")
    print(f"{'Packet Reduction Ratio':<38} {'—':>14} {packet_red_pct:>13.1f}%")
    print("-" * W)
    print(f"{'Estimated Energy Saved (Joules)':<38} {'—':>14} {total_energy_saved_j:>14.1f}")
    print(f"{'Estimated Energy Saved (Wh)':<38} {'—':>14} {total_energy_saved_kwh * 1000:>14.4f}")
    print("=" * W)

    print("\n  ✅ SUCCESS CRITERIA:")
    c1 = packet_red_pct >= 40
    c2 = abs(mse_delta_pct) <= 5
    print(f"  {'✅' if c1 else '❌'}  Packet reduction ≥ 40%  →  {packet_red_pct:.1f}%")
    print(f"  {'✅' if c2 else '❌'}  MSE degradation ≤  5%  →  {abs(mse_delta_pct):.2f}%")
    print()
    return packet_red_pct, mse_delta_pct, total_energy_saved_j, total_energy_saved_kwh


# ──────────────────────────────────────────────────────────────────────
# Plots  (5 graphs)
# ──────────────────────────────────────────────────────────────────────

def plot_results(b_runs: list, a_runs: list, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Colour palette
    C_BASE = "#4C72B0"
    C_APSM = "#55A868"
    C_SUPP = "#C44E52"

    # ── Graph 1: Bandwidth bar chart (mean across seeds) ─────────────
    b_sent_avg = np.mean([r["total_sent"] for r in b_runs])
    a_sent_avg = np.mean([r["total_sent"] for r in a_runs])
    a_supp_avg = np.mean([r["total_suppressed"] for r in a_runs])

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["GL-Baseline\n(Tundo 2025)", "APSM\n(Ours — Phase 2)"]
    sent_vals = [b_sent_avg, a_sent_avg]
    supp_vals = [0, a_supp_avg]
    x = np.arange(len(labels))
    bars = ax.bar(x, sent_vals, color=[C_BASE, C_APSM], label="Packets Sent", zorder=3, width=0.5)
    ax.bar(x[1:], supp_vals, bottom=sent_vals[1:], color=C_SUPP, alpha=0.75,
           label="Packets Suppressed (saved)", zorder=3, width=0.5)
    for bar, v in zip(bars, sent_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1, f"{int(v)}", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Total Gossip Packets", fontsize=12)
    ax.set_title("Graph 1: Network Bandwidth Usage\n"
                 f"(mean over {N_SEEDS} seeds, {FIXED_UPDATES} rounds/node)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.4, zorder=0)
    plt.tight_layout()
    plt.savefig(out_dir / "graph1_bandwidth.png", dpi=180)
    plt.close()
    print(f"  📊  graph1_bandwidth.png")

    # ── Graph 2: MSE convergence — side-by-side (first seed) ─────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, runs, title, color in [
        (axes[0], b_runs, "GL-Baseline (Tundo 2025)", C_BASE),
        (axes[1], a_runs, "APSM — Semantic Filter",   C_APSM),
    ]:
        h = runs[0]["history"]   # plot first seed; others shown as faint lines
        for s_idx, run in enumerate(runs):
            for node_id, metrics in run["history"].get("nodes_test_history", {}).items():
                mse_vals = metrics.get("mse", [])
                if mse_vals:
                    ax.plot(mse_vals,
                            alpha=0.9 if s_idx == 0 else 0.25,
                            linewidth=1.6 if s_idx == 0 else 0.8,
                            color=color,
                            label=f"Node {node_id}" if s_idx == 0 else "_nolegend_")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Evaluation Step", fontsize=10)
        ax.set_ylabel("MSE (scaled)", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper right", ncol=2)
    fig.suptitle(f"Graph 2: MSE Convergence — Baseline vs APSM  (solid=seed 0, faint=other seeds)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "graph2_convergence.png", dpi=180)
    plt.close()
    print(f"  📊  graph2_convergence.png")

    # ── Graph 3: Semantic Surprise vs Threshold (APSM, first seed) ───
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
        fig.suptitle("Graph 3: APSM Semantic Surprise ε(t) vs Adaptive Threshold τ(t) — All Nodes",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(out_dir / "graph3_semantic_surprise.png", dpi=180)
        plt.close()
        print(f"  📊  graph3_semantic_surprise.png")
    else:
        print("  ⚠️   Graph 3 skipped — no surprise score data (window may not have filled)")

    # ── Graph 4: Per-node suppression count (APSM, first seed) ───────
    per_node = a_runs[0]["per_node_supp"]
    if per_node:
        nodes  = sorted(per_node.keys(), key=lambda x: int(x))
        counts = [per_node[n] for n in nodes]
        fig, ax = plt.subplots(figsize=(9, 4))
        bars = ax.bar(nodes, counts, color=C_SUPP, alpha=0.8, zorder=3)
        for bar, c in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, c + 0.3, str(c),
                    ha="center", fontsize=9)
        ax.set_xlabel("Node ID", fontsize=11)
        ax.set_ylabel("Packets Suppressed", fontsize=11)
        ax.set_title("Graph 4: Per-Node Packet Suppression (APSM Phase 2 — Seed 0)",
                     fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.4, zorder=0)
        plt.tight_layout()
        plt.savefig(out_dir / "graph4_per_node_suppression.png", dpi=180)
        plt.close()
        print(f"  📊  graph4_per_node_suppression.png")

    # ── Graph 5: MSE final distribution — box plot across seeds ──────
    b_mses = [r["avg_mse"] for r in b_runs]
    a_mses = [r["avg_mse"] for r in a_runs]
    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot([b_mses, a_mses],
                    patch_artist=True, notch=False,
                    boxprops=dict(facecolor="white"),
                    medianprops=dict(color="black", linewidth=2))
    bp["boxes"][0].set_facecolor(C_BASE + "55")
    bp["boxes"][1].set_facecolor(C_APSM + "55")
    ax.set_xticklabels(["GL-Baseline\n(Tundo 2025)", "APSM\n(Ours — Phase 2)"], fontsize=11)
    ax.set_ylabel("Avg Final MSE (scaled)", fontsize=11)
    ax.set_title(f"Graph 5: Final MSE Distribution across {N_SEEDS} Seeds",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / "graph5_mse_boxplot.png", dpi=180)
    plt.close()
    print(f"  📊  graph5_mse_boxplot.png")


# ──────────────────────────────────────────────────────────────────────
# Save CSV + JSON
# ──────────────────────────────────────────────────────────────────────

def save_results(b_runs: list, a_runs: list, out_dir: Path, pkt_red: float, mse_delta: float, energy_j: float, energy_kwh: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    SCALE = 305 ** 2

    # Detailed per-seed CSV
    csv_path = out_dir / "phase2_comparison_full.csv"
    with open(csv_path, "w") as f:
        f.write("seed,method,avg_mse_scaled,avg_mse_raw,avg_rmse,avg_mae,"
                "total_sent,suppressed,packet_reduction_pct\n")
        for seed_idx, (br, ar) in enumerate(zip(b_runs, a_runs)):
            total = br["total_sent"] + ar["total_suppressed"]
            pr = (ar["total_suppressed"] / total * 100) if total > 0 else 0
            f.write(f"{seed_idx},GL-Baseline,{br['avg_mse']:.6f},{br['avg_mse']*SCALE:.1f},"
                    f"{br['avg_rmse']:.6f},{br['avg_mae']:.6f},{br['total_sent']},0,0.0\n")
            f.write(f"{seed_idx},APSM-Phase2,{ar['avg_mse']:.6f},{ar['avg_mse']*SCALE:.1f},"
                    f"{ar['avg_rmse']:.6f},{ar['avg_mae']:.6f},{ar['total_sent']},"
                    f"{ar['total_suppressed']},{pr:.1f}\n")

    # Summary JSON
    summary = {
        "run_timestamp": datetime.datetime.now().isoformat(),
        "config": {
            "n_nodes": N_NODES, "fixed_updates": FIXED_UPDATES,
            "epochs_per_update": EPOCHS_PER_UPDATE, "n_seeds": N_SEEDS,
            "max_samples": MAX_SAMPLES,
            "semantic_k": SEMANTIC_K, "semantic_window": SEMANTIC_WINDOW,
            "semantic_heartbeat": SEMANTIC_HEARTBEAT,
        },
        "results": {
            "packet_reduction_pct": round(pkt_red, 2),
            "mse_delta_pct": round(mse_delta, 2),
            "baseline_mse_mean": round(float(np.mean([r["avg_mse"] for r in b_runs])), 6),
            "apsm_mse_mean":     round(float(np.mean([r["avg_mse"] for r in a_runs])), 6),
            "baseline_mse_std":  round(float(np.std([r["avg_mse"] for r in b_runs])), 6),
            "apsm_mse_std":      round(float(np.std([r["avg_mse"] for r in a_runs])), 6),
            "energy_saved_joules": round(energy_j, 2),
            "energy_saved_wh": round(energy_kwh * 1000, 4)
        }
    }
    json_path = out_dir / "phase2_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  💾  CSV:  {csv_path}")
    print(f"  💾  JSON: {json_path}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def run_seed(seed: int, gpu_id: str = "") -> tuple[dict, dict]:
    # ── Subprocess: pin GPU BEFORE importing TF, then configure threads ───
    import os as _os
    import sys as _sys
    import gc
    import multiprocessing as _mp
    from pathlib import Path

    # MUST be set before TF import so CUDA sees only the assigned device.
    # gpu_id="" means CPU-only (CUDA not installed yet).
    if gpu_id != "":
        _os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    _os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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
    print(f"\n{'━'*55}\n  SEED {seed + 1}/{N_SEEDS}  →  {gpu_label}  |  PID {os.getpid()}\n{'━'*55}")

    # ── GL-Baseline ────────────────────────────────────────────
    b_workspace = str(SRC_DIR / f"experiments/gl_baseline_seed{seed}")
    b_hist_path = Path(b_workspace) / "0" / "history.json"

    if b_hist_path.exists():
        print(f"  ✅  GL-Baseline (seed {seed}) — cached, skipping.")
    else:
        print(f"  ▶️   GL-Baseline (seed {seed}) — running on {gpu_label}...")
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
        print(f"  ▶️   APSM Phase 2 (seed {seed}) — running on {gpu_label}...")
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

    print("\n" + "=" * 60)
    print("  APSM Phase 2 — Full Research Run")
    print(f"  Seeds={N_SEEDS} | Updates={FIXED_UPDATES} | Epochs/round={EPOCHS_PER_UPDATE}")
    print(f"  MAX_SAMPLES={'FULL DATA' if MAX_SAMPLES is None else MAX_SAMPLES}")
    print("=" * 60 + "\n")

    baseline_runs = [None] * N_SEEDS
    apsm_runs     = [None] * N_SEEDS

    max_workers = min(N_SEEDS, 4) if PARALLEL_SEEDS else 1

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        n_gpus = len(_GPUS)
        futures = {
            executor.submit(run_seed, seed, _gpu_for_seed(seed, n_gpus)): seed
            for seed in range(N_SEEDS)
        }
        for future in concurrent.futures.as_completed(futures):
            seed = futures[future]
            try:
                b_metrics, a_metrics = future.result()
                baseline_runs[seed] = b_metrics
                apsm_runs[seed] = a_metrics
                
                # Progress snapshot
                total = b_metrics["total_sent"] + a_metrics["total_suppressed"]
                pr_now = (a_metrics["total_suppressed"] / total * 100) if total > 0 else 0
                print(f"\n  📊  Seed {seed} snapshot:")
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
    pkt_red, mse_delta, energy_j, energy_kwh = print_comparison_table(baseline_runs, apsm_runs)

    # ── Plots ──────────────────────────────────────────────────────
    print("📈  Generating plots...")
    plot_results(baseline_runs, apsm_runs, RESULTS_DIR)

    # ── Save CSVs / JSON ───────────────────────────────────────────
    save_results(baseline_runs, apsm_runs, RESULTS_DIR, pkt_red, mse_delta, energy_j, energy_kwh)

    total_elapsed = (time.time() - total_start) / 60
    print(f"\n🎉  All done!  Total wall-clock time: {total_elapsed:.1f} min")
    print(f"📁  Results saved to: {RESULTS_DIR.resolve()}\n")

if __name__ == '__main__':
    # Fix for multiprocessing in windows
    import multiprocessing
    multiprocessing.freeze_support()
    main()
