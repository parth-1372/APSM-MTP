import json
import re
from pathlib import Path
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"

def load_scalability_data():
    data = []
    # Find all phase2_full_*n folders
    for path in RESULTS_DIR.glob("phase2_full_*n"):
        summary_file = path / "phase2_summary.json"
        if summary_file.exists():
            with open(summary_file, "r") as f:
                summary = json.load(f)
                n_nodes = summary["config"]["n_nodes"]
                packet_red = summary["results"]["packet_reduction_pct"]
                mse_delta = summary["results"]["mse_delta_pct"]
                base_mse = summary["results"]["baseline_mse_mean"]
                apsm_mse = summary["results"]["apsm_mse_mean"]
                data.append({
                    "n_nodes": n_nodes,
                    "packet_reduction_pct": packet_red,
                    "mse_delta_pct": mse_delta,
                    "base_mse": base_mse,
                    "apsm_mse": apsm_mse
                })
    # Sort by n_nodes
    data.sort(key=lambda x: x["n_nodes"])
    return data

def plot_scalability():
    data = load_scalability_data()
    if not data:
        print("No scalability data found! Run run_all_comparisons.py for different N_NODES first.")
        return

    n_nodes = [d["n_nodes"] for d in data]
    packet_red = [d["packet_reduction_pct"] for d in data]
    mse_delta = [d["mse_delta_pct"] for d in data]

    # Plot 1: Packet Reduction scaling
    plt.figure(figsize=(8, 5))
    plt.plot(n_nodes, packet_red, marker="o", color="#55A868", linewidth=2)
    plt.title("APSM Packet Reduction vs. Network Size (Scalability)")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Packets Suppressed (%)")
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    for x, y in zip(n_nodes, packet_red):
        plt.text(x, y + 2, f"{y:.1f}%", ha="center")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "scalability_bandwidth_plot.png", dpi=180)
    print(f"✅ Saved plot -> {RESULTS_DIR}/scalability_bandwidth_plot.png")

    # Plot 2: Accuracy degradation scaling
    plt.figure(figsize=(8, 5))
    plt.plot(n_nodes, mse_delta, marker="s", color="#C44E52", linewidth=2)
    plt.title("APSM MSE Degradation vs. Network Size")
    plt.xlabel("Number of Nodes")
    plt.ylabel("MSE Delta vs Baseline (%)")
    plt.axhline(0, color="black", linestyle="--", alpha=0.5)
    plt.grid(True, alpha=0.3)
    for x, y in zip(n_nodes, mse_delta):
        plt.text(x, y + 0.2, f"{y:+.1f}%", ha="center")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "scalability_accuracy_plot.png", dpi=180)
    print(f"✅ Saved plot -> {RESULTS_DIR}/scalability_accuracy_plot.png")

if __name__ == "__main__":
    plot_scalability()
