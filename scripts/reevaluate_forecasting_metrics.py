"""
Corrected Time-Aligned Forecasting Metrics Evaluation.

Evaluates:
  - Exact target indices: [1035, 1380) (345 bins per node, exactly 3,450 predictions across 10 nodes).
  - Validation observations (bins < 1035) strictly excluded from test MSE/RMSE/MAE.
  - Final test-bin prediction (bin 1379) evaluated against target 1379.
  - Persistence prediction: y_pers(t) = y(t-1) for identical target t.
  - Seeds: 11, 22, 33, 44, 55.
  - Arms: E3-local, E3-GL, E3-APSM.
"""

from collections import deque
import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn as nn

from phase3.config import Phase3Config, SimConfig
from phase3.data.alibaba_loader import build_or_load_curated_trace
from phase3.data.topology import build_edge_topology
from phase3.env.task_simulator import prepare_task_table, TaskOffloadingSim
from phase3.forecasting.gossip_runner import _pretrain_base_models
from phase3.integration.live_gossip_engine import LiveGossipManager


def reevaluate_forecasting(seeds=(11, 22, 33, 44, 55)):
    cfg = Phase3Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}")
    topo_cfg = replace(cfg.topology, n_nodes=10, k_edge_connectivity=3)
    sim_cfg = replace(cfg.sim, node_cpu_cap=20.0, deadline_slack_kappa=2.2, q_max=48)

    print("Loading N=10 Curated Alibaba Dataset and Building Topology...")
    ds = build_or_load_curated_trace(cfg.trace, n_nodes=10)
    topo = build_edge_topology(cfg.trace.porto_cells_parquet, topo_cfg)

    print("Pretraining Base Edge LSTM Forecasters on GPU...")
    cfg_dev = replace(cfg, device=device)
    base_lstm_states, _ = _pretrain_base_models(ds, cfg_dev)

    test_tbl = prepare_task_table(ds, sim_cfg, split="test", seed=42)
    print(f"Test tasks: {test_tbl.n_tasks}, test slots: {test_tbl.n_slots}")

    arms = ["e3_madqn_local", "e3_madqn_gl", "e3_madqn_apsm"]
    mode_map = {
        "e3_madqn_local": "local_only",
        "e3_madqn_gl": "unfiltered_gl",
        "e3_madqn_apsm": "apsm",
    }

    all_results = {}

    for arm in arms:
        mode = mode_map[arm]
        print(f"\n==================== Arm: {arm} (mode={mode}) ====================")
        all_results[arm] = []

        for s in seeds:
            torch.manual_seed(s)
            np.random.seed(s)

            sim = TaskOffloadingSim(ds, topo, test_tbl, cfg, telemetry_mode="periodic", has_live_forecaster=True)
            gossip_mgr = LiveGossipManager(
                dataset=ds,
                topo=topo,
                cfg=cfg,
                mode=mode,
                base_states=base_lstm_states,
                device=device,
                test_start_bin=1035,
                test_end_bin=1380,
            )

            # Run test episode with LiveGossipManager
            # Note: ReactiveGreedy or Local policy doesn't affect workload arrival/forecasting process
            # because arrivals and LSTM features are trace-driven
            from phase3.env.baselines import ReactiveGreedyPolicy
            sim.run_policy(
                ReactiveGreedyPolicy(sim_cfg),
                bin_callback=lambda cur_b, t_now: gossip_mgr.on_bin_boundary(cur_b, t_now, sim),
            )

            fc = gossip_mgr.compute_forecasting_metrics()
            all_results[arm].append(fc)
            print(
                f"  Seed {s}: MSE={fc['fc_mse']:.6f} | RMSE={fc['fc_rmse']:.4f} | MAE={fc['fc_mae']:.4f} | Pers_MSE={fc['fc_pers_mse']:.6f}"
            )

    print("\n" + "=" * 80)
    print("SUMMARY OF TIME-ALIGNED TEST FORECASTING METRICS (Mean across 5 seeds)")
    print("=" * 80)
    pers_mse = all_results["e3_madqn_local"][0]["fc_pers_mse"]
    print(f"Naive Persistence Baseline MSE: {pers_mse:.6f} (RMSE={math.sqrt(pers_mse):.4f})")

    summary_table = {}
    for arm in arms:
        mses = [x["fc_mse"] for x in all_results[arm]]
        rmses = [x["fc_rmse"] for x in all_results[arm]]
        maes = [x["fc_mae"] for x in all_results[arm]]
        mean_mse = float(np.mean(mses))
        std_mse = float(np.std(mses, ddof=1))
        rel_gain = (1.0 - mean_mse / pers_mse) * 100.0
        summary_table[arm] = {
            "mean_mse": mean_mse,
            "std_mse": std_mse,
            "mean_rmse": float(np.mean(rmses)),
            "mean_mae": float(np.mean(maes)),
            "pers_mse": pers_mse,
            "rel_gain_pct": rel_gain,
            "seed_mses": mses,
        }
        print(f"{arm:20}: MSE = {mean_mse:.6f} +/- {std_mse:.6f} | RMSE = {np.mean(rmses):.4f} | Rel Gain = +{rel_gain:.2f}%")

    out_file = ROOT / "results/phase3/stage6_10n_forecasting_aligned.json"
    out_file.write_text(json.dumps({"summary": summary_table, "raw": all_results}, indent=2), encoding="utf-8")
    print(f"\nSaved aligned forecasting results to {out_file.as_posix()}!")
    return summary_table


if __name__ == "__main__":
    reevaluate_forecasting()
