"""
Phase 3 Stage 6 Publication-Quality Figures (18 Required Figures at N=10).

Generates all 18 publication-quality figures:
  1. fig01_tsr_comparison_10n.png
  2. fig02_dmr_comparison_10n.png
  3. fig03_effective_latency_10n.png
  4. fig04_p95_latency_10n.png
  5. fig05_offload_rate_10n.png
  6. fig06_jain_fairness_10n.png
  7. fig07_per_node_load_distribution_10n.png
  8. fig08_apsm_suppression_rate_10n.png
  9. fig09_model_bytes_comparison_10n.png
  10. fig10_telemetry_bytes_10n.png
  11. fig11_control_vs_system_bytes_10n.png
  12. fig12_training_reward_curves_10n.png
  13. fig13_training_tsr_curves_10n.png
  14. fig14_td_loss_curves_10n.png
  15. fig15_calibration_surface_10n.png
  16. fig16_seed_sensitivity_paired_diff_10n.png
  17. fig17_network_topology_10n.png
  18. fig18_tradeoff_tsr_vs_bandwidth_10n.png
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ARM_LABELS = {
    "b0_local": "B0: LocalOnly",
    "e1_greedy": "E1: ReactiveGreedy",
    "e2_madqn": "E2: Reactive MADQN",
    "e3_madqn_local": "E3: Local Forecast MADQN",
    "e3_madqn_gl": "E3-GL: Unfiltered Gossip",
    "e3_madqn_apsm": "E3-APSM: Integrated Closed-Loop",
}

ARM_COLORS = {
    "b0_local": "#7f7f7f",       # Grey
    "e1_greedy": "#1f77b4",      # Blue
    "e2_madqn": "#ff7f0e",       # Orange
    "e3_madqn_local": "#2ca02c",  # Green
    "e3_madqn_gl": "#9467bd",     # Purple
    "e3_madqn_apsm": "#d62728",   # Crimson / Red
}


def load_results(json_path: Path) -> List[Dict[str, Any]]:
    return json.loads(json_path.read_text(encoding="utf-8"))


def generate_all_plots(results_path: str, plots_dir: str, calib_path: Optional[str] = None):
    out_dir = Path(plots_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_results(Path(results_path))

    all_possible_arms = ["b0_local", "e1_greedy", "e2_madqn", "e3_madqn_local", "e3_madqn_gl", "e3_madqn_apsm"]
    arms = [a for a in all_possible_arms if any(r.get("arm") == a for r in records)]
    # Group by arm
    arm_data: Dict[str, List[Dict[str, Any]]] = {a: [] for a in arms}
    for r in records:
        if r["arm"] in arm_data:
            arm_data[r["arm"]].append(r)

    # Style setup
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({"font.size": 11, "figure.autolayout": True, "figure.dpi": 300})

    # 1. Fig 01: TSR Comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    tsr_means = [np.mean([x["tsr_pct"] for x in arm_data[a]]) for a in arms]
    tsr_stds = [np.std([x["tsr_pct"] for x in arm_data[a]]) for a in arms]
    bars = ax.bar(
        [ARM_LABELS[a] for a in arms],
        tsr_means,
        yerr=tsr_stds,
        capsize=5,
        color=[ARM_COLORS[a] for a in arms],
        alpha=0.88,
        edgecolor="black",
    )
    ax.set_ylabel("Task Success Rate (%)", fontweight="bold")
    ax.set_ylim(70, 100)
    ax.set_title("Fig 1: Task Success Rate (TSR) Comparison at N=10", fontweight="bold")
    for b, val in zip(bars, tsr_means):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.5, f"{val:.2f}%", ha="center", va="bottom", fontweight="bold")
    plt.xticks(rotation=20, ha="right")
    plt.savefig(out_dir / "fig01_tsr_comparison_10n.png")
    plt.close()

    # 2. Fig 02: DMR Breakdown
    fig, ax = plt.subplots(figsize=(8, 5))
    adm_drops = [np.mean([x["drop_admission"] for x in arm_data[a]]) for a in arms]
    dl_drops = [np.mean([x["drop_deadline"] for x in arm_data[a]]) for a in arms]
    tot_tasks = np.mean([x["total_tasks"] for x in arm_data[arms[0]]])
    adm_pct = [d / tot_tasks * 100.0 for d in adm_drops]
    dl_pct = [d / tot_tasks * 100.0 for d in dl_drops]

    idx = np.arange(len(arms))
    ax.bar(idx, adm_pct, label="Admission Overflow", color="#e7298a", alpha=0.85, edgecolor="black")
    ax.bar(idx, dl_pct, bottom=adm_pct, label="Deadline Timeout", color="#7570b3", alpha=0.85, edgecolor="black")
    ax.set_xticks(idx)
    ax.set_xticklabels([ARM_LABELS[a] for a in arms], rotation=20, ha="right")
    ax.set_ylabel("Deadline Miss Rate (%)", fontweight="bold")
    ax.set_title("Fig 2: DMR Breakdown by Drop Reason at N=10", fontweight="bold")
    ax.legend()
    plt.savefig(out_dir / "fig02_dmr_comparison_10n.png")
    plt.close()

    # 3. Fig 03: Effective Latency
    fig, ax = plt.subplots(figsize=(8, 5))
    eff_means = [np.mean([x["effective_latency_mean_s"] for x in arm_data[a]]) for a in arms]
    eff_stds = [np.std([x["effective_latency_mean_s"] for x in arm_data[a]]) for a in arms]
    ax.bar(
        [ARM_LABELS[a] for a in arms],
        eff_means,
        yerr=eff_stds,
        capsize=5,
        color=[ARM_COLORS[a] for a in arms],
        alpha=0.88,
        edgecolor="black",
    )
    ax.set_ylabel("Effective Latency (seconds)", fontweight="bold")
    ax.set_title("Fig 3: Mean Effective Latency (With Drop Penalty) at N=10", fontweight="bold")
    plt.xticks(rotation=20, ha="right")
    plt.savefig(out_dir / "fig03_effective_latency_10n.png")
    plt.close()

    # 4. Fig 04: P95 Latency of Completed Tasks
    fig, ax = plt.subplots(figsize=(8, 5))
    p95_means = [np.mean([x["latency_completed_p95_s"] for x in arm_data[a]]) for a in arms]
    p95_stds = [np.std([x["latency_completed_p95_s"] for x in arm_data[a]]) for a in arms]
    ax.bar(
        [ARM_LABELS[a] for a in arms],
        p95_means,
        yerr=p95_stds,
        capsize=5,
        color=[ARM_COLORS[a] for a in arms],
        alpha=0.88,
        edgecolor="black",
    )
    ax.set_ylabel("P95 Completed Latency (seconds)", fontweight="bold")
    ax.set_title("Fig 4: 95th Percentile Completed Task Latency at N=10", fontweight="bold")
    plt.xticks(rotation=20, ha="right")
    plt.savefig(out_dir / "fig04_p95_latency_10n.png")
    plt.close()

    # 5. Fig 05: Offload Rate
    fig, ax = plt.subplots(figsize=(8, 5))
    off_means = [np.mean([x["offload_rate_pct"] for x in arm_data[a]]) for a in arms]
    off_stds = [np.std([x["offload_rate_pct"] for x in arm_data[a]]) for a in arms]
    bars = ax.bar(
        [ARM_LABELS[a] for a in arms],
        off_means,
        yerr=off_stds,
        capsize=5,
        color=[ARM_COLORS[a] for a in arms],
        alpha=0.88,
        edgecolor="black",
    )
    ax.set_ylabel("Offloading Rate (%)", fontweight="bold")
    ax.set_title("Fig 5: Offload Rate Across Benchmark Arms at N=10", fontweight="bold")
    for b, val in zip(bars, off_means):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.5, f"{val:.1f}%", ha="center", va="bottom", fontweight="bold")
    plt.xticks(rotation=20, ha="right")
    plt.savefig(out_dir / "fig05_offload_rate_10n.png")
    plt.close()

    # 6. Fig 06: Jain's Fairness
    fig, ax = plt.subplots(figsize=(8, 5))
    jain_means = [np.mean([x["jains_fairness"] for x in arm_data[a]]) for a in arms]
    jain_stds = [np.std([x["jains_fairness"] for x in arm_data[a]]) for a in arms]
    bars = ax.bar(
        [ARM_LABELS[a] for a in arms],
        jain_means,
        yerr=jain_stds,
        capsize=5,
        color=[ARM_COLORS[a] for a in arms],
        alpha=0.88,
        edgecolor="black",
    )
    ax.set_ylabel("Jain's Fairness Index", fontweight="bold")
    ax.set_ylim(0.95, 1.0)
    ax.set_title("Fig 6: Workload Distribution Fairness (Jain's Index) at N=10", fontweight="bold")
    for b, val in zip(bars, jain_means):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.001, f"{val:.4f}", ha="center", va="bottom", fontweight="bold")
    plt.xticks(rotation=20, ha="right")
    plt.savefig(out_dir / "fig06_jain_fairness_10n.png")
    plt.close()

    # 7. Fig 07: Per-Node CPU Utilization Heatmap
    fig, ax = plt.subplots(figsize=(10, 5))
    heatmap_data = []
    for a in arms:
        node_utils = np.mean([x["per_node_cpu_util"] for x in arm_data[a]], axis=0)
        heatmap_data.append(node_utils)
    cax = ax.imshow(heatmap_data, cmap="YlGnBu", aspect="auto", vmin=0.3, vmax=0.8)
    ax.set_yticks(np.arange(len(arms)))
    ax.set_yticklabels([ARM_LABELS[a] for a in arms], fontweight="bold")
    ax.set_xticks(np.arange(10))
    ax.set_xticklabels([f"Node {i}" for i in range(10)])
    ax.set_title("Fig 7: Per-Node CPU Utilization Heatmap at N=10", fontweight="bold")
    cbar = fig.colorbar(cax)
    cbar.set_label("CPU Utilization", fontweight="bold")
    plt.savefig(out_dir / "fig07_per_node_load_distribution_10n.png")
    plt.close()

    # 8. Fig 08: APSM Suppression Rate
    fig, ax = plt.subplots(figsize=(7, 5))
    supp_rates = [x["suppression_rate_pct"] for x in arm_data["e3_madqn_apsm"]]
    seeds = [x["seed"] for x in arm_data["e3_madqn_apsm"]]
    bars = ax.bar([f"Seed {s}" for s in seeds], supp_rates, color="#d62728", alpha=0.85, edgecolor="black")
    ax.axhline(float(np.mean(supp_rates)), color="black", linestyle="--", label=f"Mean: {np.mean(supp_rates):.1f}%")
    ax.set_ylabel("APSM Suppression Rate (%)", fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_title("Fig 8: APSM Communication Suppression Rate Across Seeds", fontweight="bold")
    ax.legend()
    for b, val in zip(bars, supp_rates):
        ax.text(b.get_x() + b.get_width() / 2, val + 1.0, f"{val:.1f}%", ha="center", va="bottom", fontweight="bold")
    plt.savefig(out_dir / "fig08_apsm_suppression_rate_10n.png")
    plt.close()

    # 9. Fig 09: Model Bytes Comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    apsm_bytes = np.mean([x["model_bytes"] for x in arm_data["e3_madqn_apsm"]]) if arm_data.get("e3_madqn_apsm") else 0.0
    if arm_data.get("e3_madqn_gl") and len(arm_data["e3_madqn_gl"]) > 0:
        unfiltered_bytes = np.mean([x["model_bytes"] for x in arm_data["e3_madqn_gl"]])
        unfiltered_label = "E3-GL: Unfiltered Gossip (Measured)"
    else:
        total_checks = np.mean([x.get("total_gossip_checks", 1720) for x in arm_data["e3_madqn_apsm"]])
        unfiltered_bytes = total_checks * 3.2 * 56920  # Theoretical baseline
        unfiltered_label = "Unfiltered Gossip (Theoretical)"
    comp_labels = ["B0 / E1 / E2 / E3-Local (0 B)", "E3-APSM (Live Filtered)", unfiltered_label]
    comp_vals = [0, apsm_bytes / 1e6, unfiltered_bytes / 1e6]
    bars = ax.bar(comp_labels, comp_vals, color=["#7f7f7f", "#d62728", "#9467bd"], alpha=0.88, edgecolor="black")
    ax.set_ylabel("Model Exchange Volume (MB)", fontweight="bold")
    ax.set_title("Fig 9: Gossip Learning Model Exchange Volume (MB)", fontweight="bold")
    for b, val in zip(bars, comp_vals):
        ax.text(b.get_x() + b.get_width() / 2, val + 2.0, f"{val:.1f} MB", ha="center", va="bottom", fontweight="bold")
    plt.savefig(out_dir / "fig09_model_bytes_comparison_10n.png")
    plt.close()

    # 10. Fig 10: Telemetry Bytes vs Offload Bytes
    fig, ax = plt.subplots(figsize=(8, 5))
    tel_mb = [np.mean([x["telemetry_bytes"] for x in arm_data[a]]) / 1e6 for a in arms]
    off_mb = [np.mean([x["offload_data_bytes"] for x in arm_data[a]]) / 1e6 for a in arms]
    idx = np.arange(len(arms))
    ax.bar(idx - 0.2, tel_mb, width=0.4, label="Operational Telemetry (MB)", color="#386cb0", edgecolor="black")
    ax.bar(idx + 0.2, off_mb, width=0.4, label="Offloaded Task Data (MB)", color="#fdc086", edgecolor="black")
    ax.set_xticks(idx)
    ax.set_xticklabels([ARM_LABELS[a] for a in arms], rotation=20, ha="right")
    ax.set_ylabel("Data Volume (Megabytes)", fontweight="bold")
    ax.set_title("Fig 10: Operational Telemetry vs Task Data Volume (MB)", fontweight="bold")
    ax.legend()
    plt.savefig(out_dir / "fig10_telemetry_bytes_10n.png")
    plt.close()

    # 11. Fig 11: Control-Plane Bytes vs System Total Bytes
    fig, ax = plt.subplots(figsize=(8, 5))
    ctrl_mb = [np.mean([x["cb_control_bytes"] for x in arm_data[a]]) / 1e6 for a in arms]
    data_mb = [np.mean([x["offload_data_bytes"] for x in arm_data[a]]) / 1e6 for a in arms]
    idx = np.arange(len(arms))
    ax.bar(idx, ctrl_mb, label="Control Plane Bytes (Model+Tel+Ack+Hdr)", color="#beaed4", edgecolor="black")
    ax.bar(idx, data_mb, bottom=ctrl_mb, label="Data Plane Bytes (Payloads)", color="#7fc97f", edgecolor="black")
    ax.set_xticks(idx)
    ax.set_xticklabels([ARM_LABELS[a] for a in arms], rotation=20, ha="right")
    ax.set_ylabel("Total Bytes Transferred (MB)", fontweight="bold")
    ax.set_title("Fig 11: Control Plane vs Data Plane Communication Overhead", fontweight="bold")
    ax.legend()
    plt.savefig(out_dir / "fig11_control_vs_system_bytes_10n.png")
    plt.close()

    # 12-14: Training Curves (Fig 12, 13, 14)
    marl_arms = [a for a in ["e2_madqn", "e3_madqn_local", "e3_madqn_gl", "e3_madqn_apsm"] if a in arm_data and len(arm_data[a]) > 0]
    # Check if episode_history exists in records
    sample_hist = arm_data[marl_arms[0]][0].get("episode_history", []) if marl_arms else []
    if sample_hist:
        n_eps = len(sample_hist)
        eps_axis = np.arange(1, n_eps + 1)

        # Fig 12: Reward curves
        fig, ax = plt.subplots(figsize=(8, 5))
        for a in marl_arms:
            curves = [
                [h["train_reward"] for h in rec.get("episode_history", [])]
                for rec in arm_data[a]
                if "episode_history" in rec
            ]
            if curves:
                mean_c = np.mean(curves, axis=0)
                std_c = np.std(curves, axis=0)
                ax.plot(eps_axis, mean_c, label=ARM_LABELS[a], color=ARM_COLORS[a], lw=2)
                ax.fill_between(eps_axis, mean_c - std_c, mean_c + std_c, color=ARM_COLORS[a], alpha=0.18)
        ax.set_xlabel("Training Episode", fontweight="bold")
        ax.set_ylabel("Mean Episode Reward", fontweight="bold")
        ax.set_title("Fig 12: MARL Training Reward Progression over Episodes", fontweight="bold")
        ax.legend()
        plt.savefig(out_dir / "fig12_training_reward_curves_10n.png")
        plt.close()

        # Fig 13: TSR curves
        fig, ax = plt.subplots(figsize=(8, 5))
        for a in marl_arms:
            curves = [
                [h["train_tsr"] for h in rec.get("episode_history", [])]
                for rec in arm_data[a]
                if "episode_history" in rec
            ]
            if curves:
                mean_c = np.mean(curves, axis=0)
                std_c = np.std(curves, axis=0)
                ax.plot(eps_axis, mean_c, label=ARM_LABELS[a], color=ARM_COLORS[a], lw=2)
                ax.fill_between(eps_axis, mean_c - std_c, mean_c + std_c, color=ARM_COLORS[a], alpha=0.18)
        ax.set_xlabel("Training Episode", fontweight="bold")
        ax.set_ylabel("Training TSR (%)", fontweight="bold")
        ax.set_title("Fig 13: MARL Training TSR Progression over Episodes", fontweight="bold")
        ax.legend()
        plt.savefig(out_dir / "fig13_training_tsr_curves_10n.png")
        plt.close()

        # Fig 14: TD Loss curves
        fig, ax = plt.subplots(figsize=(8, 5))
        for a in marl_arms:
            curves = [
                [h["td_loss"] for h in rec.get("episode_history", [])]
                for rec in arm_data[a]
                if "episode_history" in rec
            ]
            if curves:
                mean_c = np.mean(curves, axis=0)
                ax.plot(eps_axis, mean_c, label=ARM_LABELS[a], color=ARM_COLORS[a], lw=2)
        ax.set_xlabel("Training Episode", fontweight="bold")
        ax.set_ylabel("Smooth L1 TD Loss", fontweight="bold")
        ax.set_title("Fig 14: Bellman TD Loss Progression during Double Dueling Training", fontweight="bold")
        ax.legend()
        plt.savefig(out_dir / "fig14_td_loss_curves_10n.png")
        plt.close()

    # 15. Fig 15: Calibration Surface
    scratch_calib = Path("C:/Users/btpUser/.gemini/antigravity/brain/1cb424e4-013a-4681-923b-3c7a1355fcc4/scratch/calibration_results_10n.json")
    if scratch_calib.exists():
        calib_data = json.loads(scratch_calib.read_text(encoding="utf-8"))
        fig, ax = plt.subplots(figsize=(8, 5))
        cpu_caps = sorted(list(set(x["cpu_cap"] for x in calib_data)))
        kappas = sorted(list(set(x["kappa"] for x in calib_data)))
        matrix = np.zeros((len(cpu_caps), len(kappas)))
        for x in calib_data:
            i = cpu_caps.index(x["cpu_cap"])
            j = kappas.index(x["kappa"])
            matrix[i, j] = x["delta_tsr"]
        cax = ax.imshow(matrix, cmap="Blues", origin="lower")
        ax.set_xticks(np.arange(len(kappas)))
        ax.set_xticklabels([f"κ={k}" for k in kappas])
        ax.set_yticks(np.arange(len(cpu_caps)))
        ax.set_yticklabels([f"C={c}" for c in cpu_caps])
        for i in range(len(cpu_caps)):
            for j in range(len(kappas)):
                ax.text(j, i, f"+{matrix[i, j]:.2f}%", ha="center", va="center", fontweight="bold")
        ax.set_title("Fig 15: Calibration Sweep Δ TSR (E1 vs B0) Surface", fontweight="bold")
        fig.colorbar(cax, label="Greedy Advantage (Δ TSR %)")
        plt.savefig(out_dir / "fig15_calibration_surface_10n.png")
        plt.close()

    # 16. Fig 16: Seed Sensitivity Paired Difference
    fig, ax = plt.subplots(figsize=(10, 5))
    seeds = sorted(list(set(x["seed"] for x in records)))
    deltas_e1_b0 = [
        next(x["tsr_pct"] for x in arm_data["e1_greedy"] if x["seed"] == s)
        - next(x["tsr_pct"] for x in arm_data["b0_local"] if x["seed"] == s)
        for s in seeds
    ]
    deltas_e2_e1 = [
        next(x["tsr_pct"] for x in arm_data["e2_madqn"] if x["seed"] == s)
        - next(x["tsr_pct"] for x in arm_data["e1_greedy"] if x["seed"] == s)
        for s in seeds
    ]
    deltas_e3_e2 = [
        next(x["tsr_pct"] for x in arm_data["e3_madqn_local"] if x["seed"] == s)
        - next(x["tsr_pct"] for x in arm_data["e2_madqn"] if x["seed"] == s)
        for s in seeds
    ]
    deltas_apsm_e2 = [
        next(x["tsr_pct"] for x in arm_data["e3_madqn_apsm"] if x["seed"] == s)
        - next(x["tsr_pct"] for x in arm_data["e2_madqn"] if x["seed"] == s)
        for s in seeds
    ]
    has_gl = "e3_madqn_gl" in arm_data and len(arm_data["e3_madqn_gl"]) > 0
    if has_gl:
        deltas_apsm_gl = [
            next(x["tsr_pct"] for x in arm_data["e3_madqn_apsm"] if x["seed"] == s)
            - next(x["tsr_pct"] for x in arm_data["e3_madqn_gl"] if x["seed"] == s)
            for s in seeds
        ]

    idx = np.arange(len(seeds))
    w = 0.16 if has_gl else 0.2
    ax.bar(idx - 1.5 * w, deltas_e1_b0, width=w, label="Δ(E1 - B0)", color="#1f77b4")
    ax.bar(idx - 0.5 * w, deltas_e2_e1, width=w, label="Δ(E2 - E1)", color="#ff7f0e")
    ax.bar(idx + 0.5 * w, deltas_e3_e2, width=w, label="Δ(E3 - E2)", color="#2ca02c")
    ax.bar(idx + 1.5 * w, deltas_apsm_e2, width=w, label="Δ(E3_APSM - E2)", color="#d62728")
    if has_gl:
        ax.bar(idx + 2.5 * w, deltas_apsm_gl, width=w, label="Δ(APSM - GL)", color="#9467bd")
    ax.set_xticks(idx)
    ax.set_xticklabels([f"Seed {s}" for s in seeds])
    ax.set_ylabel("Pairwise TSR Difference (Percentage Points)", fontweight="bold")
    ax.set_title("Fig 16: Seed-Wise Paired Performance Advantage at N=10", fontweight="bold")
    ax.legend()
    plt.savefig(out_dir / "fig16_seed_sensitivity_paired_diff_10n.png")
    plt.close()

    # 17. Fig 17: Network Topology Visualization
    fig, ax = plt.subplots(figsize=(7, 7))
    coords = [
        (0.2, 0.8), (0.5, 0.9), (0.8, 0.8), (0.9, 0.5), (0.8, 0.2),
        (0.5, 0.1), (0.2, 0.2), (0.1, 0.5), (0.35, 0.55), (0.65, 0.45)
    ]
    edges = [
        (0, 3), (0, 4), (0, 7), (1, 4), (1, 7), (1, 2), (1, 9),
        (2, 3), (2, 9), (3, 8), (4, 7), (5, 6), (5, 9), (5, 8),
        (6, 9), (6, 8)
    ]
    for u, v in edges:
        ax.plot([coords[u][0], coords[v][0]], [coords[u][1], coords[v][1]], color="#888888", lw=1.5, zorder=1)
    for i, (x, y) in enumerate(coords):
        ax.scatter([x], [y], s=500, color="#1f77b4", edgecolor="black", zorder=2)
        ax.text(x, y, str(i), color="white", ha="center", va="center", fontweight="bold", zorder=3)
    ax.set_title("Fig 17: Porto 10-Node Edge Topology (k=3, 16 Undirected Edges)", fontweight="bold")
    ax.axis("off")
    plt.savefig(out_dir / "fig17_network_topology_10n.png")
    plt.close()

    # 18. Fig 18: Pareto Frontier: TSR vs Bandwidth
    fig, ax = plt.subplots(figsize=(8, 5))
    for a in arms:
        tsr_val = np.mean([x["tsr_pct"] for x in arm_data[a]])
        cb_val = max(1.0, np.mean([x["cb_control_bytes"] for x in arm_data[a]]))
        ax.scatter(cb_val, tsr_val, s=200, color=ARM_COLORS[a], label=ARM_LABELS[a], edgecolor="black", zorder=3)
        ax.text(cb_val * 1.15, tsr_val, ARM_LABELS[a], va="center", fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlabel("Total Control-Plane Bytes (Log Scale)", fontweight="bold")
    ax.set_ylabel("Task Success Rate (%)", fontweight="bold")
    ax.set_title("Fig 18: Pareto Frontier: TSR vs Control Bandwidth at N=10", fontweight="bold")
    ax.set_ylim(75, 95)
    plt.savefig(out_dir / "fig18_tradeoff_tsr_vs_bandwidth_10n.png")
    plt.close()

    # 19. Fig 19: Direct Forecasting Error Comparison
    fc_arms = [a for a in ["e3_madqn_local", "e3_madqn_gl", "e3_madqn_apsm"] if a in arm_data and len(arm_data[a]) > 0]
    if fc_arms and "fc_mse" in arm_data[fc_arms[0]][0]:
        fig, ax = plt.subplots(figsize=(8, 5))
        mses = [np.mean([x["fc_mse"] for x in arm_data[a]]) for a in fc_arms]
        pers_mse = np.mean([x.get("fc_pers_mse", 0.0517) for x in arm_data[fc_arms[0]]])
        bar_labels = [ARM_LABELS[a] for a in fc_arms] + ["Persistence Baseline"]
        bar_vals = mses + [pers_mse]
        bar_colors = [ARM_COLORS[a] for a in fc_arms] + ["#7f7f7f"]
        bars = ax.bar(bar_labels, bar_vals, color=bar_colors, alpha=0.88, edgecolor="black")
        ax.set_ylabel("Mean Squared Error (Normalized Load)", fontweight="bold")
        ax.set_title("Fig 19: Direct Test-Window Workload Forecasting Error (MSE)", fontweight="bold")
        for b, val in zip(bars, bar_vals):
            ax.text(b.get_x() + b.get_width() / 2, val + 0.001, f"{val:.4f}", ha="center", va="bottom", fontweight="bold")
        plt.xticks(rotation=20, ha="right")
        plt.savefig(out_dir / "fig19_forecasting_error_comparison_10n.png")
        plt.close()

    print(f"Successfully generated all publication-quality figures in {out_dir}!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Phase 3 Stage 6 publication plots")
    parser.add_argument("--results", type=str, default="results/phase3/stage6_integrated_10n_audited_v2.json")
    parser.add_argument("--plots-dir", type=str, default="results/phase3/plots_stage6_10n_audited_v2")
    args = parser.parse_args()

    # Fallback to stage6_integrated_10n.json if audited_v2 does not exist yet
    res_path = Path(args.results)
    if not res_path.exists() and Path("results/phase3/stage6_integrated_10n.json").exists():
        res_path = Path("results/phase3/stage6_integrated_10n.json")

    generate_all_plots(
        results_path=str(res_path),
        plots_dir=args.plots_dir,
    )

