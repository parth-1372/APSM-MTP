"""
Phase 3 Stage 6 Complete Implementation & Execution:
Live APSM + Decentralized Gossip Learning + Parameter-Shared Double Dueling MADQN Integration
Strict Experimental Scope: N = 10 Nodes Only.

5 Benchmark Arms:
  - B0: LocalOnly
  - E1: ReactiveGreedy (15s periodic telemetry beacons)
  - E2: Reactive Parameter-Shared Double Dueling MADQN (No forecast)
  - E3: Forecast-Augmented Double Dueling MADQN (Local LSTM forecaster, no peer sharing)
  - E3-APSM: Integrated Closed-Loop System (Collaborative Gossip Learning + APSM-H1 filter + MADQN)

Hardware Acceleration:
  - GPU (RTX 1000 Ada via CUDA) for PyTorch neural network training minibatches (Double Dueling Q updates & LSTM pretraining)
  - CPU (i9-13950HX) float32 NumPy runner for fast microsecond DES simulation rollouts
"""

import argparse
import copy
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from phase3.config import Phase3Config, SimConfig
from phase3.data.alibaba_loader import CuratedTraceDataset, build_or_load_curated_trace
from phase3.data.topology import EdgeTopology, build_edge_topology
from phase3.env.baselines import LocalOnlyPolicy, ReactiveGreedyPolicy
from phase3.env.task_simulator import PreparedTaskTable, SimMetrics, TaskOffloadingSim, prepare_task_table
from phase3.forecasting.gossip_runner import _pretrain_base_models
from phase3.integration.live_gossip_engine import CommunicationLedger, LiveGossipManager
from phase3.marl.madqn import MaskedDuelingQNet, _NumpyDuelingQRunner, _ReplayBuffer


def run_baseline_b0(
    ds: CuratedTraceDataset,
    topo: EdgeTopology,
    test_tbl: PreparedTaskTable,
    cfg: Phase3Config,
    seed: int,
) -> Dict[str, Any]:
    """Execute B0: LocalOnly baseline."""
    t0 = time.perf_counter()
    sim = TaskOffloadingSim(ds, topo, test_tbl, cfg, telemetry_mode="none")
    policy = LocalOnlyPolicy()
    metrics = sim.run_policy(policy)
    dur = time.perf_counter() - t0

    ledger = CommunicationLedger().reconcile(sim)
    res = asdict(metrics)
    res.update(
        {
            "arm": "b0_local",
            "seed": seed,
            "wall_clock_s": dur,
            "suppression_rate_pct": 0.0,
            "cb_control_bytes": ledger.cb_control_bytes,
            "cb_system_bytes": ledger.cb_system_bytes,
        }
    )
    return res


def run_baseline_e1(
    ds: CuratedTraceDataset,
    topo: EdgeTopology,
    test_tbl: PreparedTaskTable,
    cfg: Phase3Config,
    sim_cfg: SimConfig,
    seed: int,
) -> Dict[str, Any]:
    """Execute E1: ReactiveGreedy heuristic baseline with 15s telemetry beacons."""
    t0 = time.perf_counter()
    sim = TaskOffloadingSim(ds, topo, test_tbl, cfg, telemetry_mode="periodic")
    policy = ReactiveGreedyPolicy(sim_cfg)
    metrics = sim.run_policy(policy)
    dur = time.perf_counter() - t0

    ledger = CommunicationLedger().reconcile(sim)
    res = asdict(metrics)
    res.update(
        {
            "arm": "e1_greedy",
            "seed": seed,
            "wall_clock_s": dur,
            "suppression_rate_pct": 0.0,
            "cb_control_bytes": ledger.cb_control_bytes,
            "cb_system_bytes": ledger.cb_system_bytes,
        }
    )
    return res


def run_marl_arm(
    arm: str,
    ds: CuratedTraceDataset,
    topo: EdgeTopology,
    train_tbls: List[PreparedTaskTable],
    val_tbl: PreparedTaskTable,
    test_tbl: PreparedTaskTable,
    base_lstm_states: List[Dict[str, torch.Tensor]],
    cfg: Phase3Config,
    seed: int,
    episodes: int = 48,
    warmup: int = 36,
    device: str = "cuda",
    ckpt_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Train and evaluate a Parameter-Shared Double Dueling MADQN Agent on 10-node topology.
    Arms:
      - 'e2_madqn': Reactive MADQN (No forecast feature, forecast_active=False)
      - 'e3_madqn_local': Forecast MADQN with local forecaster (mode='local_only')
      - 'e3_madqn_apsm': Closed-loop Live Gossip + APSM-H1 filter (mode='apsm')
    """
    t_start = time.perf_counter()
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed * 43 + 17)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    k_max = cfg.sim.k_max_neighbors
    obs_dim = 11 + 8 * k_max
    action_dim = 1 + k_max

    has_fc = arm in ("e3_madqn_local", "e3_madqn_gl", "e3_madqn_apsm")
    gossip_mode = (
        "apsm"
        if arm == "e3_madqn_apsm"
        else ("unfiltered_gl" if arm == "e3_madqn_gl" else ("local_only" if arm == "e3_madqn_local" else "off"))
    )

    # Initialize Q networks
    fc_scale = 0.008 if has_fc else 0.0
    q_online = MaskedDuelingQNet(obs_dim, action_dim, hidden_dim=cfg.marl.hidden_dim, fc_scale=fc_scale).to(dev)
    q_target = copy.deepcopy(q_online).to(dev)
    q_target.eval()

    pg = [
        {"params": q_online.base_params(), "weight_decay": 2e-4, "lr": 3e-4},
        {"params": q_online.fc_res.parameters(), "weight_decay": 2e-2, "lr": 1.5e-4},
    ]
    optimizer = torch.optim.AdamW(pg, eps=1e-5)
    replay_buf = _ReplayBuffer(capacity=60000, obs_dim=obs_dim, action_dim=action_dim)

    best_val_reward = -1e9
    best_ep = 0
    best_weights = copy.deepcopy(q_online.state_dict())
    gamma = 0.40
    episode_history: List[Dict[str, float]] = []

    eval_cadence = 4
    eval_eps = set(range(eval_cadence, episodes + 1, eval_cadence)) | {episodes}

    # Training Loop
    for ep in range(1, episodes + 1):
        fc_on = has_fc and (ep >= warmup)
        optimizer.param_groups[0]["lr"] = 3e-4 if ep < warmup else 8e-5
        q_online.fc_res.fc_active = fc_on
        q_target.fc_res.fc_active = fc_on

        # Exploration schedule: smooth cosine temperature & linear epsilon decay
        frac = min(1.0, max(0, ep - 2) / 30.0)
        t_roll = 0.002 + 0.5 * (0.012 - 0.002) * (1.0 + math.cos(math.pi * frac))
        eps = 0.30 + min(1.0, ep / 28.0) * (0.01 - 0.30)

        # Select a training table slice
        tr_tbl = train_tbls[(ep - 1) % len(train_tbls)]
        tr_sim = TaskOffloadingSim(ds, topo, tr_tbl, cfg, telemetry_mode="periodic", has_live_forecaster=fc_on)

        # Setup live gossip manager for this episode if forecast active
        mgr = None
        bin_cb = None
        if fc_on:
            mgr = LiveGossipManager(
                ds, topo, cfg, mode=gossip_mode, base_states=base_lstm_states, device=dev
            )
            bin_cb = lambda b, t: mgr.on_bin_boundary(b, t, tr_sim)

        # Fast NumPy runner for episode rollout
        runner = _NumpyDuelingQRunner(q_online)
        obs_l, mask_l, act_l, node_l, tid_l = [], [], [], [], []

        def _step_train(obs, mask, gstate, node_id, t_now_s, task_idx):
            act = runner.act(obs, mask, rng, temp=t_roll, epsilon=eps)
            obs_l.append(obs)
            mask_l.append(mask)
            act_l.append(act)
            node_l.append(node_id)
            tid_l.append(task_idx)
            return act

        tr_metrics, outcomes = tr_sim.run_marl_episode(_step_train, bin_callback=bin_cb)

        # Ingest transitions into replay buffer
        if len(tid_l) > 0:
            obs_arr = np.stack(obs_l, axis=0).astype(np.float32)
            mask_arr = np.stack(mask_l, axis=0).astype(bool)
            act_arr = np.array(act_l, dtype=np.int64)
            node_arr = np.array(node_l, dtype=np.int32)
            rew_arr = np.array([outcomes[tid].reward for tid in tid_l], dtype=np.float32)

            next_obs_arr = obs_arr.copy()
            next_mask_arr = mask_arr.copy()
            done_arr = np.zeros(len(tid_l), dtype=bool)
            for nid in range(ds.n_nodes):
                idxs = np.flatnonzero(node_arr == nid)
                if len(idxs) > 0:
                    if len(idxs) > 1:
                        next_obs_arr[idxs[:-1]] = obs_arr[idxs[1:]]
                        next_mask_arr[idxs[:-1]] = mask_arr[idxs[1:]]
                    done_arr[idxs[-1]] = True

            replay_buf.add_batch(obs_arr, mask_arr, act_arr, rew_arr, next_obs_arr, next_mask_arr, done_arr)

        # GPU-accelerated minibatch updates
        ep_loss = 0.0
        if replay_buf.size >= 256:
            q_online.train()
            n_updates = min(48, getattr(cfg.marl, "dqn_updates_per_episode", 48))
            bs = min(replay_buf.size, getattr(cfg.marl, "dqn_batch_size", 256))
            losses = []

            for _ in range(n_updates):
                bo, bm, ba, br, bno, bnm, bd = replay_buf.sample(bs, rng)
                t_o = torch.from_numpy(bo).to(dev)
                t_m = torch.from_numpy(bm).to(dev)
                t_a = torch.from_numpy(ba).to(dev)
                t_r = torch.from_numpy(br).to(dev)
                t_no = torch.from_numpy(bno).to(dev)
                t_nm = torch.from_numpy(bnm).to(dev)
                t_d = torch.from_numpy(bd).to(dev).float()

                with torch.no_grad():
                    # Double-DQN target calculation with terminal done discounting
                    best_a = q_online(t_no, t_nm).argmax(dim=-1, keepdim=True)
                    tgt = t_r + (1.0 - t_d) * gamma * q_target(t_no, t_nm).gather(1, best_a).squeeze(-1)

                q_taken = q_online(t_o, t_m).gather(1, t_a.unsqueeze(-1)).squeeze(-1)
                loss = F.smooth_l1_loss(q_taken, tgt, beta=1.0)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_online.parameters(), 2.0)
                optimizer.step()

                # Soft target network update
                with torch.no_grad():
                    for p1, p2 in zip(q_online.parameters(), q_target.parameters()):
                        p2.data.mul_(0.995).add_(p1.data, alpha=0.005)
                losses.append(float(loss.item()))

            q_online.eval()
            ep_loss = float(np.mean(losses)) if losses else 0.0

        episode_history.append({
            "episode": ep,
            "train_tsr": float(tr_metrics.tsr_pct),
            "train_reward": float(tr_metrics.mean_reward),
            "td_loss": float(ep_loss),
        })

        # Validation Checkpoint Evaluation
        if ep in eval_eps:
            val_sim = TaskOffloadingSim(ds, topo, val_tbl, cfg, telemetry_mode="periodic", has_live_forecaster=fc_on)
            val_mgr = None
            val_cb = None
            if fc_on:
                val_mgr = LiveGossipManager(
                    ds, topo, cfg, mode=gossip_mode, base_states=base_lstm_states, device=dev
                )
                val_cb = lambda b, t: val_mgr.on_bin_boundary(b, t, val_sim)

            val_runner = _NumpyDuelingQRunner(q_online)
            val_metrics, _ = val_sim.run_marl_episode(
                lambda o, m, g, nid, t, tid: val_runner.act(o, m, rng, epsilon=0.0),
                bin_callback=val_cb,
            )

            eligible = (not has_fc) or (ep >= warmup)
            if eligible and val_metrics.mean_reward > best_val_reward:
                best_val_reward = val_metrics.mean_reward
                best_ep = ep
                best_weights = copy.deepcopy(q_online.state_dict())

    # Fallback if no checkpoint was selected
    if best_ep == 0:
        best_ep = episodes
        best_weights = copy.deepcopy(q_online.state_dict())

    # Load best checkpoint for final evaluation on test split
    q_online.load_state_dict(best_weights)
    q_online.eval()
    q_online.fc_res.fc_active = has_fc and (best_ep >= warmup)

    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_file = ckpt_dir / f"ckpt_best_{arm}_s{seed}.pt"
        torch.save(
            {
                "arm": arm,
                "seed": seed,
                "best_ep": best_ep,
                "best_val_reward": best_val_reward,
                "fc_active": has_fc and (best_ep >= warmup),
                "warmup": warmup,
                "state_dict": best_weights,
            },
            ckpt_file,
        )

    # Final Evaluation on the complete Test split (17,862 tasks)
    test_sim = TaskOffloadingSim(ds, topo, test_tbl, cfg, telemetry_mode="periodic", has_live_forecaster=has_fc)
    test_mgr = None
    test_cb = None
    if has_fc:
        test_mgr = LiveGossipManager(
            ds, topo, cfg, mode=gossip_mode, base_states=base_lstm_states, device=dev
        )
        test_cb = lambda b, t: test_mgr.on_bin_boundary(b, t, test_sim)

    eval_runner = _NumpyDuelingQRunner(q_online)
    test_metrics, outcomes = test_sim.run_marl_episode(
        lambda o, m, g, nid, t, tid: eval_runner.act(o, m, rng, epsilon=0.0),
        bin_callback=test_cb,
    )

    ledger = (test_mgr.ledger.reconcile(test_sim)) if test_mgr is not None else CommunicationLedger().reconcile(test_sim)
    dur = time.perf_counter() - t_start

    if test_mgr is not None:
        fc_metrics = test_mgr.compute_forecasting_metrics()
    else:
        fc_metrics = {"fc_mse": None, "fc_rmse": None, "fc_mae": None, "fc_pers_mse": None}

    res = asdict(test_metrics)
    res.update(fc_metrics)
    res.update(
        {
            "arm": arm,
            "seed": seed,
            "best_checkpoint_episode": best_ep,
            "best_val_reward": best_val_reward,
            "wall_clock_s": dur,
            "suppression_rate_pct": ledger.suppression_rate_pct,
            "cb_control_bytes": ledger.cb_control_bytes,
            "cb_system_bytes": ledger.cb_system_bytes,
            "model_broadcast_rounds": ledger.model_broadcast_rounds,
            "total_gossip_checks": ledger.total_gossip_checks,
            "suppressed_checks": ledger.suppressed_checks,
            "surprise_sends": ledger.surprise_sends,
            "heartbeat_sends": ledger.heartbeat_sends,
            "has_forecast": has_fc,
            "warmup_episode": warmup,
            "episode_history": episode_history,
        }
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Stage 6 Live Integration Benchmark (N=10 Only)")
    parser.add_argument("--smoke", action="store_true", help="Run quick 2-episode smoke test")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55], help="Random seeds")
    parser.add_argument("--episodes", type=int, default=48, help="Number of training episodes")
    parser.add_argument("--warmup", type=int, default=36, help="Episode when proactive branch turns on")
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["b0_local", "e1_greedy", "e2_madqn", "e3_madqn_local", "e3_madqn_gl", "e3_madqn_apsm"],
        help="Benchmark arms to execute",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/phase3/stage6_integrated_10n_audited_v2.json",
        help="Path for final JSON results",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default="results/phase3/checkpoints_stage6_10n_audited_v2",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="results/phase3/stage6_10n_audited_v2_manifest.json",
        help="Path for progress manifest JSON",
    )
    args = parser.parse_args()

    cfg = Phase3Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"================================================================================")
    print(f"Phase 3 Stage 6 Benchmark Execution (N = 10 Nodes Only)")
    print(f"Device: {device.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Seeds: {args.seeds}")
    print(f"Arms: {args.arms}")
    print(f"Episodes: {args.episodes if not args.smoke else 2} (Warmup: {args.warmup if not args.smoke else 1})")
    print(f"================================================================================")

    # Operating parameters
    topo_cfg = replace(cfg.topology, n_nodes=10, k_edge_connectivity=3)
    sim_cfg = replace(cfg.sim, node_cpu_cap=20.0, deadline_slack_kappa=2.2, q_max=48)

    print("\n[1/5] Loading N=10 Curated Alibaba Dataset and Building Porto Topology...")
    ds = build_or_load_curated_trace(cfg.trace, n_nodes=10)
    topo = build_edge_topology(cfg.trace.porto_cells_parquet, topo_cfg)
    print(f"  Dataset: {ds.n_nodes} nodes, {ds.n_bins} bins, dt={ds.dt_bin_s}s")
    print(f"  Topology: {topo.n_nodes} nodes, {int(topo.adj_matrix.sum() // 2)} edges, degrees 3-4")

    print("\n[2/5] Pretraining Base Edge LSTM Forecasters on GPU...")
    t0_pre = time.perf_counter()
    cfg_dev = replace(cfg, device=device)
    base_lstm_states, _ = _pretrain_base_models(ds, cfg_dev)
    print(f"  Pretrained {len(base_lstm_states)} models in {time.perf_counter() - t0_pre:.2f}s!")

    print("\n[3/5] Materializing Task Tables (Test Split: 17,862 Tasks)...")
    test_tbl = prepare_task_table(ds, sim_cfg, split="test", seed=42)
    val_tbl = prepare_task_table(ds, sim_cfg, split="val", seed=42)

    # Materialize 6 diverse training window slices
    train_slices = [(ds.lookback_L, 128), (148, 268), (288, 408), (428, 548), (568, 688), (688, 808)]
    train_tbls = [
        prepare_task_table(ds, sim_cfg, seed=42, start_bin=s, end_bin=e) for s, e in train_slices
    ]
    print(f"  Test tasks: {test_tbl.n_tasks} (Slots: {test_tbl.n_slots})")
    print(f"  Val tasks: {val_tbl.n_tasks} (Slots: {val_tbl.n_slots})")
    print(f"  Train slices: {len(train_tbls)} slices (~{train_tbls[0].n_tasks} tasks each)")

    results: List[Dict[str, Any]] = []
    ckpt_path = Path(args.ckpt_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest)

    n_episodes = 2 if args.smoke else args.episodes
    warmup_ep = 1 if args.smoke else args.warmup

    print("\n[4/5] Executing Benchmark Arms across Seeds...")
    total_runs = len(args.arms) * len(args.seeds)
    run_idx = 0

    for arm in args.arms:
        print(f"\n--- Arm: {arm.upper()} ---")
        for seed in args.seeds:
            run_idx += 1
            print(f"  [{run_idx}/{total_runs}] Running {arm} (Seed {seed})...", end="", flush=True)

            if arm == "b0_local":
                res = run_baseline_b0(ds, topo, test_tbl, cfg, seed)
            elif arm == "e1_greedy":
                res = run_baseline_e1(ds, topo, test_tbl, cfg, sim_cfg, seed)
            elif arm in ("e2_madqn", "e3_madqn_local", "e3_madqn_gl", "e3_madqn_apsm"):
                res = run_marl_arm(
                    arm=arm,
                    ds=ds,
                    topo=topo,
                    train_tbls=train_tbls,
                    val_tbl=val_tbl,
                    test_tbl=test_tbl,
                    base_lstm_states=base_lstm_states,
                    cfg=cfg,
                    seed=seed,
                    episodes=n_episodes,
                    warmup=warmup_ep,
                    device=device,
                    ckpt_dir=ckpt_path,
                )
            else:
                raise ValueError(f"Unknown arm: {arm}")

            results.append(res)
            print(
                f" Done ({res['wall_clock_s']:.1f}s) | TSR={res['tsr_pct']:.2f}% | DMR={res['dmr_pct']:.2f}% | Offload={res['offload_rate_pct']:.1f}% | CB={res['cb_control_bytes']:,} B"
            )

            # Persist intermediate manifest
            manifest = {
                "completed_runs": len(results),
                "total_runs": total_runs,
                "latest_arm": arm,
                "latest_seed": seed,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n[5/5] Saving Final JSON Results...")
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"  Successfully wrote {len(results)} experiment records to {out_path}!")
    print(f"================================================================================")


if __name__ == "__main__":
    main()

