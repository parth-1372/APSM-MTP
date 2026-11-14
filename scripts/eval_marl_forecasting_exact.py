"""
Evaluate the exact trained MADQN checkpoints on the test split [1035, 1380)
with time-aligned LiveGossipManager (3,450 predictions).

Records:
  - Task Success Rate (TSR)
  - Dropped Tasks
  - Forecasting MSE, RMSE, MAE, Persistence MSE
  - Communication metrics (model packets, model bytes, control bytes)
  - Assert exactly 3,450 unique (node_id, target_bin) records.
"""

from collections import deque
import copy
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from phase3.config import Phase3Config
from phase3.data.alibaba_loader import build_or_load_curated_trace
from phase3.data.topology import build_edge_topology
from phase3.env.task_simulator import PreparedTaskTable, TaskOffloadingSim, prepare_task_table
from phase3.forecasting.gossip_runner import _pretrain_base_models
from phase3.integration.live_gossip_engine import CommunicationLedger, LiveGossipManager
from phase3.marl.madqn import MaskedDuelingQNet, _NumpyDuelingQRunner


def evaluate_marl_checkpoints():
    cfg = Phase3Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    print(f"Device: {device.upper()}")

    topo_cfg = replace(cfg.topology, n_nodes=10, k_edge_connectivity=3)
    sim_cfg = replace(cfg.sim, node_cpu_cap=20.0, deadline_slack_kappa=2.2, q_max=48)

    print("Loading dataset & topology...")
    ds = build_or_load_curated_trace(cfg.trace, n_nodes=10)
    topo = build_edge_topology(cfg.trace.porto_cells_parquet, topo_cfg)

    print("Pretraining base LSTM models...")
    cfg_dev = replace(cfg, device=device)
    base_lstm_states, _ = _pretrain_base_models(ds, cfg_dev)

    test_tbl = prepare_task_table(ds, sim_cfg, split="test", seed=42)
    print(f"Test tasks: {test_tbl.n_tasks}, test slots: {test_tbl.n_slots}")

    k_max = cfg.sim.k_max_neighbors
    obs_dim = 11 + 8 * k_max
    action_dim = 1 + k_max

    ckpt_dir = ROOT / "results/phase3/checkpoints_stage6_10n_audited_v2"
    seeds = [11, 22, 33, 44, 55]
    arms = ["e3_madqn_local", "e3_madqn_gl", "e3_madqn_apsm"]
    mode_map = {
        "e3_madqn_local": "local_only",
        "e3_madqn_gl": "unfiltered_gl",
        "e3_madqn_apsm": "apsm",
    }

    results = {}

    for arm in arms:
        gossip_mode = mode_map[arm]
        results[arm] = []
        print(f"\n==================== Evaluating Arm: {arm} ====================")

        for seed in seeds:
            ckpt_file = ckpt_dir / f"ckpt_best_{arm}_s{seed}.pt"
            assert ckpt_file.exists(), f"Missing checkpoint: {ckpt_file}"
            ckpt_data = torch.load(ckpt_file, map_location=dev)

            # Reconstruct network
            q_net = MaskedDuelingQNet(obs_dim, action_dim, hidden_dim=cfg.marl.hidden_dim, fc_scale=0.008).to(dev)
            q_net.load_state_dict(ckpt_data["state_dict"])
            q_net.eval()
            q_net.fc_res.fc_active = ckpt_data.get("fc_active", True)
            q_runner = _NumpyDuelingQRunner(q_net)

            # Test simulator & gossip manager
            test_sim = TaskOffloadingSim(ds, topo, test_tbl, cfg, telemetry_mode="periodic", has_live_forecaster=True)
            test_mgr = LiveGossipManager(
                ds, topo, cfg, mode=gossip_mode, base_states=base_lstm_states, device=dev,
                test_start_bin=1035, test_end_bin=1380
            )

            rng = np.random.default_rng(seed * 43 + 17)
            t0 = time.perf_counter()
            test_metrics, outcomes = test_sim.run_marl_episode(
                lambda o, m, g, nid, t, tid: q_runner.act(o, m, rng, epsilon=0.0),
                bin_callback=lambda b, t: test_mgr.on_bin_boundary(b, t, test_sim),
            )
            test_mgr.finalize_test_metrics()
            dur = time.perf_counter() - t0

            # Ledger & metrics
            ledger = test_mgr.ledger.reconcile(test_sim)
            fc_metrics = test_mgr.compute_forecasting_metrics()

            # Assertions
            records = test_mgr.test_records
            unique_pairs = {(r[0], r[1]) for r in records}
            assert len(records) == 3450, f"Expected 3450 records, got {len(records)}"
            assert len(unique_pairs) == 3450, f"Expected 3450 unique pairs, got {len(unique_pairs)}"
            assert all(1035 <= r[1] < 1380 for r in records), "Out of range target bin!"

            res = asdict(test_metrics)
            res.update(fc_metrics)
            res.update({
                "arm": arm,
                "seed": seed,
                "best_checkpoint_episode": ckpt_data.get("best_ep"),
                "best_val_reward": ckpt_data.get("best_val_reward"),
                "wall_clock_s": dur,
                "suppression_rate_pct": ledger.suppression_rate_pct,
                "cb_control_bytes": ledger.cb_control_bytes,
                "cb_system_bytes": ledger.cb_system_bytes,
                "model_broadcast_rounds": ledger.model_broadcast_rounds,
                "total_gossip_checks": ledger.total_gossip_checks,
                "suppressed_checks": ledger.suppressed_checks,
                "surprise_sends": ledger.surprise_sends,
                "heartbeat_sends": ledger.heartbeat_sends,
                "model_packets": ledger.model_packets,
                "model_bytes": ledger.model_bytes,
                "has_forecast": True,
            })
            results[arm].append(res)

            print(f"  Seed {seed}: TSR={res['tsr_pct']:.2f}% | Latency={res['effective_latency_mean_s']:.4f}s | "
                  f"MSE={fc_metrics['fc_mse']:.6f} | RMSE={fc_metrics['fc_rmse']:.4f} | "
                  f"ModelMB={ledger.model_bytes/1e6:.2f}MB | Supp={ledger.suppression_rate_pct:.1f}%")

    # Summary
    print("\n==================== EVALUATION SUMMARY (MEAN ACROSS 5 SEEDS) ====================")
    for arm in arms:
        tsrs = [r["tsr_pct"] for r in results[arm]]
        mses = [r["fc_mse"] for r in results[arm]]
        rmses = [r["fc_rmse"] for r in results[arm]]
        maes = [r["fc_mae"] for r in results[arm]]
        pers_mses = [r["fc_pers_mse"] for r in results[arm]]
        model_mbs = [r["model_bytes"] / 1e6 for r in results[arm]]
        print(f"{arm:18}: TSR={np.mean(tsrs):.2f}% +/- {np.std(tsrs, ddof=1):.2f}% | "
              f"MSE={np.mean(mses):.6f} +/- {np.std(mses, ddof=1):.6f} | "
              f"RMSE={np.mean(rmses):.4f} | MAE={np.mean(maes):.4f} | "
              f"Pers_MSE={np.mean(pers_mses):.6f} | ModelMB={np.mean(model_mbs):.2f}MB")

    out_file = ROOT / "results/phase3/stage6_10n_marl_eval_exact.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved exact MARL evaluation to {out_file}!")


if __name__ == "__main__":
    evaluate_marl_checkpoints()
