"""
Phase 3 Stage 7: Extreme-Scale Decentralized Edge Forecasting & Agentic Offloading Evaluation.
Generic, Configurable Runner for N = 10, N = 50, and N = 100 Edge Networks.

Arms:
  - B0: LocalOnly
  - E1: ReactiveGreedy (15s periodic telemetry beacons)
  - E2: Reactive Parameter-Shared Double Dueling MADQN (No forecast)
  - E3-Local: Forecast MADQN (Isolated local LSTM forecaster)
  - E3-GL: Forecast MADQN (Collaborative Gossip Learning, unfiltered)
  - E3-APSM: Integrated Closed-Loop System (APSM-H1 Semantic Gating + Gossip Learning + MADQN)

Hardware Acceleration:
  - GPU (RTX 1000 Ada via CUDA) for PyTorch neural network minibatches & LSTM pretraining.
  - CPU (i9-13950HX) float32 NumPy runner for fast microsecond discrete-event simulation rollouts.
  - Real-time memory & VRAM tracking with atomic per-run incremental persistence.
"""

import argparse
import copy
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F

from phase3.config import Phase3Config, SimConfig, TopologyConfig
from phase3.data.alibaba_loader import CuratedTraceDataset, build_or_load_curated_trace
from phase3.data.topology import EdgeTopology, build_edge_topology
from phase3.env.baselines import LocalOnlyPolicy, ReactiveGreedyPolicy
from phase3.env.task_simulator import PreparedTaskTable, SimMetrics, TaskOffloadingSim, prepare_task_table
from phase3.forecasting.gossip_runner import _pretrain_base_models
from phase3.integration.live_gossip_engine import CommunicationLedger, LiveGossipManager
from phase3.marl.madqn import MaskedDuelingQNet, _NumpyDuelingQRunner, _ReplayBuffer


def sanitize_json_obj(obj: Any) -> Any:
    """Recursively convert NaNs and Infs to None for strict RFC 8259 JSON compliance."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: sanitize_json_obj(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_json_obj(v) for v in obj]
    return obj


def get_resource_usage() -> Dict[str, float]:
    """Capture current process RAM and GPU VRAM usage."""
    proc = psutil.Process(os.getpid())
    ram_mb = proc.memory_info().rss / (1024.0 * 1024.0)
    vram_mb = 0.0
    if torch.cuda.is_available():
        vram_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    return {"ram_mb": ram_mb, "vram_mb": vram_mb}


def run_baseline_b0_stage7(
    ds: CuratedTraceDataset,
    topo: EdgeTopology,
    test_tbl: PreparedTaskTable,
    cfg: Phase3Config,
    seed: int,
    n_nodes: int,
) -> Dict[str, Any]:
    """Execute B0: LocalOnly baseline."""
    t0 = time.perf_counter()
    sim = TaskOffloadingSim(ds, topo, test_tbl, cfg, telemetry_mode="none")
    policy = LocalOnlyPolicy()
    metrics = sim.run_policy(policy)
    dur = time.perf_counter() - t0

    ledger = CommunicationLedger().reconcile(sim)
    res = asdict(metrics)
    res.update({
        "n_nodes": n_nodes,
        "arm": "b0_local",
        "seed": seed,
        "wall_clock_s": dur,
        "suppression_rate_pct": None,
        "cb_control_bytes": ledger.cb_control_bytes,
        "cb_system_bytes": ledger.cb_system_bytes,
        "model_broadcast_rounds": 0,
        "total_gossip_checks": 0,
        "suppressed_checks": 0,
        "surprise_sends": 0,
        "heartbeat_sends": 0,
        "model_packets": 0,
        "model_bytes": 0,
        "fc_mse": None,
        "fc_rmse": None,
        "fc_mae": None,
        "fc_pers_mse": None,
        "has_forecast": False,
        "resource_usage": get_resource_usage(),
    })
    return res


def run_baseline_e1_stage7(
    ds: CuratedTraceDataset,
    topo: EdgeTopology,
    test_tbl: PreparedTaskTable,
    cfg: Phase3Config,
    sim_cfg: SimConfig,
    seed: int,
    n_nodes: int,
) -> Dict[str, Any]:
    """Execute E1: ReactiveGreedy baseline."""
    t0 = time.perf_counter()
    sim = TaskOffloadingSim(ds, topo, test_tbl, cfg, telemetry_mode="periodic")
    policy = ReactiveGreedyPolicy(sim_cfg)
    metrics = sim.run_policy(policy)
    dur = time.perf_counter() - t0

    ledger = CommunicationLedger().reconcile(sim)
    res = asdict(metrics)
    res.update({
        "n_nodes": n_nodes,
        "arm": "e1_greedy",
        "seed": seed,
        "wall_clock_s": dur,
        "suppression_rate_pct": None,
        "cb_control_bytes": ledger.cb_control_bytes,
        "cb_system_bytes": ledger.cb_system_bytes,
        "model_broadcast_rounds": 0,
        "total_gossip_checks": 0,
        "suppressed_checks": 0,
        "surprise_sends": 0,
        "heartbeat_sends": 0,
        "model_packets": 0,
        "model_bytes": 0,
        "fc_mse": None,
        "fc_rmse": None,
        "fc_mae": None,
        "fc_pers_mse": None,
        "has_forecast": False,
        "resource_usage": get_resource_usage(),
    })
    return res


def train_stage7_marl(
    arm: str,
    ds: CuratedTraceDataset,
    topo: EdgeTopology,
    sim_cfg: SimConfig,
    train_tbls: List[PreparedTaskTable],
    val_tbl: PreparedTaskTable,
    test_tbl: PreparedTaskTable,
    base_lstm_states: List[Dict[str, torch.Tensor]],
    cfg: Phase3Config,
    seed: int,
    n_nodes: int,
    episodes: int = 48,
    warmup: int = 36,
    device: str = "cuda",
    ckpt_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train and evaluate Parameter-Shared Double Dueling MADQN Agent on N-node topology."""
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

        frac = min(1.0, max(0, ep - 2) / 30.0)
        t_roll = 0.002 + 0.5 * (0.012 - 0.002) * (1.0 + math.cos(math.pi * frac))
        eps = 0.30 + min(1.0, ep / 28.0) * (0.01 - 0.30)

        tr_tbl = train_tbls[(ep - 1) % len(train_tbls)]
        tr_sim = TaskOffloadingSim(ds, topo, tr_tbl, cfg, telemetry_mode="periodic", has_live_forecaster=fc_on)

        mgr = None
        bin_cb = None
        if fc_on:
            mgr = LiveGossipManager(ds, topo, cfg, mode=gossip_mode, base_states=base_lstm_states, device=dev)
            bin_cb = lambda b, t: mgr.on_bin_boundary(b, t, tr_sim)

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

        if len(tid_l) > 0:
            obs_arr = np.stack(obs_l, axis=0).astype(np.float32)
            mask_arr = np.stack(mask_l, axis=0).astype(bool)
            act_arr = np.array(act_l, dtype=np.int64)
            node_arr = np.array(node_l, dtype=np.int32)
            rew_arr = np.array([outcomes.get(tid, 0.0) for tid in tid_l], dtype=np.float32)

            n_samples = len(obs_arr)
            sub_idx = np.arange(n_samples)
            if n_samples > 12000:
                sub_idx = rng.choice(n_samples, size=12000, replace=False)
                sub_idx.sort()

            replay_buf.add_batch(
                obs_arr[sub_idx],
                mask_arr[sub_idx],
                act_arr[sub_idx],
                rew_arr[sub_idx],
                node_arr[sub_idx],
                gamma=gamma,
            )

        # Gradient updates
        if len(replay_buf) >= 128:
            n_updates = min(16, max(4, len(replay_buf) // 300))
            q_online.train()
            for _ in range(n_updates):
                s_obs, s_mask, s_act, s_ret, s_next_obs, s_next_mask = replay_buf.sample(128, rng, dev)
                q_vals = q_online(s_obs, s_mask)
                q_taken = q_vals.gather(1, s_act.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_q_online = q_online(s_next_obs, s_next_mask)
                    next_actions = next_q_online.argmax(dim=1, keepdim=True)
                    next_q_target = q_target(s_next_obs, s_next_mask)
                    next_target_val = next_q_target.gather(1, next_actions).squeeze(1)
                    target = s_ret + gamma * next_target_val

                loss = F.smooth_l1_loss(q_taken, target)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_online.parameters(), 1.0)
                optimizer.step()

                # Polyak target update
                tau = 0.005
                for p_t, p_o in zip(q_target.parameters(), q_online.parameters()):
                    p_t.data.lerp_(p_o.data, tau)

        # Validation Rollout
        if ep in eval_eps:
            q_online.eval()
            val_sim = TaskOffloadingSim(ds, topo, val_tbl, cfg, telemetry_mode="periodic", has_live_forecaster=fc_on)
            val_mgr = None
            val_cb = None
            if fc_on:
                val_mgr = LiveGossipManager(ds, topo, cfg, mode=gossip_mode, base_states=base_lstm_states, device=dev)
                val_cb = lambda b, t: val_mgr.on_bin_boundary(b, t, val_sim)

            v_runner = _NumpyDuelingQRunner(q_online)
            val_metrics, _ = val_sim.run_marl_episode(
                lambda o, m, g, nid, t, tid: v_runner.act(o, m, rng, epsilon=0.0),
                bin_callback=val_cb,
            )
            v_rew = val_metrics.mean_reward
            if v_rew > best_val_reward:
                best_val_reward = v_rew
                best_ep = ep
                best_weights = copy.deepcopy(q_online.state_dict())

            episode_history.append({
                "episode": ep,
                "val_reward": v_rew,
                "val_tsr": val_metrics.tsr_pct,
                "train_tsr": tr_metrics.tsr_pct,
            })

    # Final Evaluation on the complete Test split
    q_online.load_state_dict(best_weights)
    q_online.eval()
    q_online.fc_res.fc_active = has_fc and (best_ep >= warmup)

    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_file = ckpt_dir / f"ckpt_best_{arm}_s{seed}_n{n_nodes}.pt"
        torch.save(
            {
                "n_nodes": n_nodes,
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

    test_sim = TaskOffloadingSim(ds, topo, test_tbl, cfg, telemetry_mode="periodic", has_live_forecaster=has_fc)
    test_mgr = None
    test_cb = None
    if has_fc:
        test_mgr = LiveGossipManager(
            ds, topo, cfg, mode=gossip_mode, base_states=base_lstm_states, device=dev,
            test_start_bin=1035, test_end_bin=1380
        )
        test_cb = lambda b, t: test_mgr.on_bin_boundary(b, t, test_sim)

    eval_runner = _NumpyDuelingQRunner(q_online)
    test_metrics, outcomes = test_sim.run_marl_episode(
        lambda o, m, g, nid, t, tid: eval_runner.act(o, m, rng, epsilon=0.0),
        bin_callback=test_cb,
    )
    if test_mgr is not None:
        test_mgr.finalize_test_metrics()

    ledger = (test_mgr.ledger.reconcile(test_sim)) if test_mgr is not None else CommunicationLedger().reconcile(test_sim)
    dur = time.perf_counter() - t_start

    if test_mgr is not None:
        fc_metrics = test_mgr.compute_forecasting_metrics()
        records = test_mgr.test_records
        expected_fc = n_nodes * 345
        assert len(records) == expected_fc, f"Expected {expected_fc} predictions, got {len(records)}"
    else:
        fc_metrics = {"fc_mse": None, "fc_rmse": None, "fc_mae": None, "fc_pers_mse": None}

    res = asdict(test_metrics)
    res.update(fc_metrics)
    res.update({
        "n_nodes": n_nodes,
        "arm": arm,
        "seed": seed,
        "best_checkpoint_episode": best_ep,
        "best_val_reward": best_val_reward,
        "wall_clock_s": dur,
        "suppression_rate_pct": ledger.suppression_rate_pct if arm == "e3_madqn_apsm" else None,
        "cb_control_bytes": ledger.cb_control_bytes,
        "cb_system_bytes": ledger.cb_system_bytes,
        "model_broadcast_rounds": ledger.model_broadcast_rounds,
        "total_gossip_checks": ledger.total_gossip_checks,
        "suppressed_checks": ledger.suppressed_checks,
        "surprise_sends": ledger.surprise_sends,
        "heartbeat_sends": ledger.heartbeat_sends,
        "model_packets": ledger.model_packets,
        "model_bytes": ledger.model_bytes,
        "has_forecast": has_fc,
        "warmup_episode": warmup,
        "episode_history": episode_history,
        "resource_usage": get_resource_usage(),
    })
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


def run_scalability_benchmark(
    n_nodes: int,
    seeds: List[int],
    arms: List[str],
    episodes: int = 48,
    warmup: int = 36,
    smoke: bool = False,
    device: str = "cuda",
    output_base: Path = Path("results/phase3/stage7_scalability"),
    ckpt_base: Path = Path("results/phase3/stage7_scalability/checkpoints"),
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Execute complete scalability benchmark for a single network size N."""
    scale_dir = output_base / f"n{n_nodes}"
    scale_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = ckpt_base / f"n{n_nodes}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    raw_runs_file = scale_dir / "raw_runs.json"

    # Load existing runs for incremental recovery
    existing_runs: List[Dict[str, Any]] = []
    if raw_runs_file.exists():
        try:
            existing_runs = json.loads(raw_runs_file.read_text(encoding="utf-8"))
            print(f"Loaded {len(existing_runs)} existing runs from {raw_runs_file}")
        except Exception as e:
            print(f"Warning: could not load existing runs ({e}); starting fresh.")
            existing_runs = []

    completed_set = {(r["arm"], r["seed"]) for r in existing_runs}

    cfg = Phase3Config()
    dev_str = device if torch.cuda.is_available() else "cpu"
    print(f"\n================================================================================")
    print(f"Phase 3 Stage 7 Scalability Benchmark: N = {n_nodes} Nodes")
    print(f"Device: {dev_str.upper()} | Seeds: {seeds} | Arms: {arms}")
    print(f"Episodes: {episodes if not smoke else 2} (Warmup: {warmup if not smoke else 1})")
    print(f"================================================================================")

    topo_cfg = replace(cfg.topology, n_nodes=n_nodes, k_edge_connectivity=3)
    sim_cfg = replace(cfg.sim, node_cpu_cap=20.0, deadline_slack_kappa=2.2, q_max=48)

    print(f"Loading N={n_nodes} Curated Alibaba Dataset and Building Porto Topology...")
    ds = build_or_load_curated_trace(cfg.trace, n_nodes=n_nodes)
    topo = build_edge_topology(cfg.trace.porto_cells_parquet, topo_cfg)

    # Topology and degree audit
    degs = [len(x) for x in topo.neighbors]
    min_deg, mean_deg, max_deg = min(degs), float(np.mean(degs)), max(degs)
    n_edges = int(topo.adj_matrix.sum() // 2)
    print(f"  Dataset: {ds.n_nodes} nodes, {ds.n_bins} bins, dt={ds.dt_bin_s}s")
    print(f"  Topology: {topo.n_nodes} nodes, {n_edges} undirected edges, min_deg={min_deg}, mean_deg={mean_deg:.2f}, max_deg={max_deg}")
    assert max_deg <= cfg.sim.k_max_neighbors, f"Degree {max_deg} exceeds k_max_neighbors {cfg.sim.k_max_neighbors}!"

    # Save calibration and topology metadata
    topo_meta = {
        "n_nodes": n_nodes,
        "k_edge_connectivity": 3,
        "n_undirected_edges": n_edges,
        "min_degree": min_deg,
        "mean_degree": mean_deg,
        "max_degree": max_deg,
        "c_cpu_cap": sim_cfg.node_cpu_cap,
        "deadline_slack_kappa": sim_cfg.deadline_slack_kappa,
        "q_max": sim_cfg.q_max,
        "k_max_neighbors": sim_cfg.k_max_neighbors,
        "trace_time_horizon_s": ds.n_bins * ds.dt_bin_s,
        "test_horizon_s": (1380 - 1035) * ds.dt_bin_s,
    }
    (scale_dir / "calibration_and_topology.json").write_text(json.dumps(topo_meta, indent=2), encoding="utf-8")

    print("Pretraining Base Edge LSTM Forecasters...")
    t0_pre = time.perf_counter()
    cfg_dev = replace(cfg, device=dev_str)
    base_lstm_states, _ = _pretrain_base_models(ds, cfg_dev)
    print(f"  Pretrained {len(base_lstm_states)} base models in {time.perf_counter() - t0_pre:.2f}s!")

    print("Materializing Task Tables...")
    test_tbl = prepare_task_table(ds, sim_cfg, split="test", seed=42)
    val_tbl = prepare_task_table(ds, sim_cfg, split="val", seed=42)
    train_slices = [(ds.lookback_L, 128), (148, 268), (288, 408), (428, 548), (568, 688), (688, 808)]
    train_tbls = [prepare_task_table(ds, sim_cfg, seed=42, start_bin=s, end_bin=e) for s, e in train_slices]
    print(f"  Test tasks: {test_tbl.n_tasks} | Val tasks: {val_tbl.n_tasks} | Train slices: {len(train_tbls)}")

    total_runs = len(arms) * len(seeds)
    run_idx = 0
    all_runs: List[Dict[str, Any]] = list(existing_runs)

    for arm in arms:
        print(f"\n--- Arm: {arm.upper()} (N={n_nodes}) ---")
        for seed in seeds:
            run_idx += 1
            if (arm, seed) in completed_set:
                print(f"  [{run_idx}/{total_runs}] Skipping completed: {arm} (Seed {seed})")
                continue

            print(f"  [{run_idx}/{total_runs}] Running {arm} (Seed {seed})...", end="", flush=True)
            if arm == "b0_local":
                res = run_baseline_b0_stage7(ds, topo, test_tbl, cfg, seed, n_nodes)
            elif arm == "e1_greedy":
                res = run_baseline_e1_stage7(ds, topo, test_tbl, cfg, sim_cfg, seed, n_nodes)
            else:
                n_eps = 2 if smoke else episodes
                w_ep = 1 if smoke else warmup
                res = train_stage7_marl(
                    arm=arm,
                    ds=ds,
                    topo=topo,
                    sim_cfg=sim_cfg,
                    train_tbls=train_tbls,
                    val_tbl=val_tbl,
                    test_tbl=test_tbl,
                    base_lstm_states=base_lstm_states,
                    cfg=cfg,
                    seed=seed,
                    n_nodes=n_nodes,
                    episodes=n_eps,
                    warmup=w_ep,
                    device=dev_str,
                    ckpt_dir=ckpt_dir,
                )

            all_runs.append(res)
            completed_set.add((arm, seed))

            # Incremental atomic persistence
            clean_runs = sanitize_json_obj(all_runs)
            tmp_runs = raw_runs_file.with_suffix(".tmp")
            tmp_runs.write_text(json.dumps(clean_runs, indent=2, allow_nan=False), encoding="utf-8")
            tmp_runs.replace(raw_runs_file)

            fc_str = f" | MSE={res['fc_mse']:.6f}" if res.get("fc_mse") is not None else ""
            supp_str = f" | Supp={res['suppression_rate_pct']:.1f}%" if res.get("suppression_rate_pct") is not None else ""
            print(f" Done! TSR={res['tsr_pct']:.2f}% | Latency={res['effective_latency_mean_s']:.2f}s{fc_str}{supp_str} ({res['wall_clock_s']:.1f}s)")

    # Post-processing: Compute paired disaggregated communication reductions
    gl_by_seed = {r["seed"]: r for r in all_runs if r["arm"] == "e3_madqn_gl"}
    for r in all_runs:
        arm = r["arm"]
        seed = r["seed"]
        if arm == "e3_madqn_apsm" and seed in gl_by_seed:
            gl_ref = gl_by_seed[seed]
            gl_pkts = gl_ref.get("model_packets", 0)
            apsm_pkts = r.get("model_packets", 0)
            r["model_packet_reduction_pct"] = ((1.0 - apsm_pkts / gl_pkts) * 100.0) if gl_pkts > 0 else None

            gl_bytes = gl_ref.get("model_bytes", 0)
            apsm_bytes = r.get("model_bytes", 0)
            r["model_byte_reduction_pct"] = ((1.0 - apsm_bytes / gl_bytes) * 100.0) if gl_bytes > 0 else None

            gl_ctrl = gl_ref.get("cb_control_bytes", 0)
            apsm_ctrl = r.get("cb_control_bytes", 0)
            r["control_byte_reduction_pct"] = ((1.0 - apsm_ctrl / gl_ctrl) * 100.0) if gl_ctrl > 0 else None

            tot_chk = r.get("total_gossip_checks", 0)
            supp_chk = r.get("suppressed_checks", 0)
            r["scheduled_check_suppression_pct"] = ((supp_chk / tot_chk) * 100.0) if tot_chk > 0 else None
        elif arm == "e3_madqn_gl":
            r["model_packet_reduction_pct"] = 0.0
            r["model_byte_reduction_pct"] = 0.0
            r["control_byte_reduction_pct"] = 0.0
            r["scheduled_check_suppression_pct"] = 0.0
        else:
            r["model_packet_reduction_pct"] = None
            r["model_byte_reduction_pct"] = None
            r["control_byte_reduction_pct"] = None
            r["scheduled_check_suppression_pct"] = None

        # Normalized communication metrics
        r["bytes_per_node"] = float(r.get("cb_control_bytes", 0)) / float(n_nodes)
        r["bytes_per_edge"] = float(r.get("cb_control_bytes", 0)) / float(max(1, n_edges))
        r["bytes_per_simulated_hour"] = float(r.get("cb_control_bytes", 0)) / 2.875

    # Re-save raw runs with paired reductions
    clean_runs = sanitize_json_obj(all_runs)
    raw_runs_file.write_text(json.dumps(clean_runs, indent=2, allow_nan=False), encoding="utf-8")

    # Generate Summary Metrics
    summary: Dict[str, Any] = {"n_nodes": n_nodes, "arms": {}}
    for arm in arms:
        arm_runs = [r for r in all_runs if r["arm"] == arm]
        if not arm_runs:
            continue
        tsrs = [r["tsr_pct"] for r in arm_runs]
        lats = [r["effective_latency_mean_s"] for r in arm_runs]
        p95s = [r["effective_latency_p95_s"] for r in arm_runs]
        offs = [r["offload_rate_pct"] for r in arm_runs]
        drops = [r["dmr_pct"] for r in arm_runs]
        m_bytes = [r["model_bytes"] for r in arm_runs]
        m_pkts = [r["model_packets"] for r in arm_runs]
        ctrl_b = [r["cb_control_bytes"] for r in arm_runs]
        fcs = [r["fc_mse"] for r in arm_runs if r.get("fc_mse") is not None]

        summary["arms"][arm] = {
            "n_runs": len(arm_runs),
            "tsr_mean": float(np.mean(tsrs)),
            "tsr_std": float(np.std(tsrs, ddof=1)) if len(tsrs) > 1 else 0.0,
            "effective_latency_mean": float(np.mean(lats)),
            "effective_latency_p95": float(np.mean(p95s)),
            "offload_rate_mean": float(np.mean(offs)),
            "drop_rate_mean": float(np.mean(drops)),
            "model_bytes_mean": float(np.mean(m_bytes)),
            "model_packets_mean": float(np.mean(m_pkts)),
            "control_bytes_mean": float(np.mean(ctrl_b)),
            "fc_mse_mean": float(np.mean(fcs)) if fcs else None,
            "fc_mse_std": float(np.std(fcs, ddof=1)) if len(fcs) > 1 else (0.0 if fcs else None),
        }

    (scale_dir / "summary.json").write_text(json.dumps(sanitize_json_obj(summary), indent=2, allow_nan=False), encoding="utf-8")

    # Resource usage log
    res_log = {
        "n_nodes": n_nodes,
        "peak_ram_mb": max(r["resource_usage"]["ram_mb"] for r in all_runs),
        "peak_vram_mb": max(r["resource_usage"]["vram_mb"] for r in all_runs),
        "total_wall_clock_s": sum(r["wall_clock_s"] for r in all_runs),
    }
    (scale_dir / "resource_usage.json").write_text(json.dumps(res_log, indent=2), encoding="utf-8")
    print(f"\nCompleted Stage 7 Scalability Benchmark for N={n_nodes}! Summary written to {scale_dir / 'summary.json'}")
    return all_runs, summary


def main():
    parser = argparse.ArgumentParser(description="Stage 7 Extreme-Scale Decentralized Edge Scalability Runner")
    parser.add_argument("--nodes", nargs="+", type=int, default=[50], help="Network sizes N to evaluate (e.g. 50, 100)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55], help="Random seeds")
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["b0_local", "e1_greedy", "e2_madqn", "e3_madqn_local", "e3_madqn_gl", "e3_madqn_apsm"],
        help="Benchmark arms",
    )
    parser.add_argument("--episodes", type=int, default=48, help="Number of training episodes")
    parser.add_argument("--warmup", type=int, default=36, help="Warmup episode when forecast branch turns on")
    parser.add_argument("--smoke", action="store_true", help="Run 2-episode smoke test")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device (cuda or cpu)")
    parser.add_argument("--output-dir", type=str, default="results/phase3/stage7_scalability", help="Output directory")
    parser.add_argument("--ckpt-dir", type=str, default="results/phase3/stage7_scalability/checkpoints", help="Checkpoint directory")
    args = parser.parse_args()

    out_base = Path(args.output_dir)
    ckpt_base = Path(args.ckpt_dir)

    for n in args.nodes:
        run_scalability_benchmark(
            n_nodes=n,
            seeds=args.seeds,
            arms=args.arms,
            episodes=args.episodes,
            warmup=args.warmup,
            smoke=args.smoke,
            device=args.device,
            output_base=out_base,
            ckpt_base=ckpt_base,
        )


if __name__ == "__main__":
    main()
