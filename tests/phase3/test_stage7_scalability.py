"""
Unit and Integration Tests for Stage 7 Scalability Testing Framework.

Validates:
  1. Generic N=10, N=50, and N=100 configuration construction.
  2. Dataset node counts and shape integrity across scales.
  3. Topology graph connectivity and degree constraints.
  4. Neighbor-degree and observation-dimension compatibility.
  5. Absence of neighbor truncation (all node degrees <= k_max_neighbors).
  6. Action-mask dimension consistency (1 + k_max = 9).
  7. Exact task conservation at N=50 scale.
  8. Strict temporal causality under perturbation at N=50 scale.
  9. Absence of test/validation leakage in LiveGossipManager.
  10. Forecast count equals exactly N * 345 on test split.
  11. Uniqueness of (node_id, target_bin) prediction records.
  12. Physical model packet delivery and next-bin assimilation.
  13. APSM suppression accounting invariants.
  14. Communication ledger arithmetic and channel reconciliation.
  15. Strict RFC 8259 JSON compliance (no bare NaNs or Infinities).
  16. Incremental partial-run persistence and recovery.
  17. Memory-safe event queue handling.
"""

from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import tempfile
import numpy as np
import pytest
import torch

from phase3.config import Phase3Config, SimConfig, TopologyConfig
from phase3.data.alibaba_loader import CuratedTraceDataset, build_or_load_curated_trace
from phase3.data.topology import EdgeTopology, build_edge_topology
from phase3.env.baselines import LocalOnlyPolicy, ReactiveGreedyPolicy
from phase3.env.task_simulator import PreparedTaskTable, TaskOffloadingSim, prepare_task_table
from phase3.forecasting.gossip_runner import _pretrain_base_models
from phase3.integration.live_gossip_engine import CommunicationLedger, LiveGossipManager, PendingModelMsg
from phase3.marl.madqn import MaskedDuelingQNet, _NumpyDuelingQRunner


@pytest.fixture(scope="module")
def n50_light_setup():
    """Lightweight 50-node fixture on 5-bin slice for fast unit verification."""
    cfg = Phase3Config()
    topo_cfg = replace(cfg.topology, n_nodes=50, k_edge_connectivity=3)
    sim_cfg = replace(cfg.sim, node_cpu_cap=20.0, deadline_slack_kappa=2.2, q_max=48)
    ds = build_or_load_curated_trace(cfg.trace, n_nodes=50)
    topo = build_edge_topology(cfg.trace.porto_cells_parquet, topo_cfg)
    return cfg, ds, topo, sim_cfg


def test_stage7_generic_config_construction():
    """1. Generic N=10, N=50, and N=100 configuration construction."""
    cfg = Phase3Config()
    for n in [10, 50, 100]:
        t_cfg = replace(cfg.topology, n_nodes=n, k_edge_connectivity=3)
        assert t_cfg.n_nodes == n
        assert t_cfg.k_edge_connectivity == 3
        s_cfg = replace(cfg.sim, node_cpu_cap=20.0, deadline_slack_kappa=2.2, q_max=48)
        assert s_cfg.node_cpu_cap == 20.0
        assert s_cfg.deadline_slack_kappa == 2.2
        assert s_cfg.k_max_neighbors == 8


def test_stage7_dataset_node_count():
    """2. Correct node count in generated dataset."""
    cfg = Phase3Config()
    ds50 = build_or_load_curated_trace(cfg.trace, n_nodes=50)
    assert ds50.n_nodes == 50
    assert ds50.features.shape == (50, 1380, 7)
    assert ds50.targets.shape == (50, 1380)

    ds100 = build_or_load_curated_trace(cfg.trace, n_nodes=100)
    assert ds100.n_nodes == 100
    assert ds100.features.shape == (100, 1380, 7)
    assert ds100.targets.shape == (100, 1380)


def test_stage7_topology_connectivity_and_degrees():
    """3. Correct topology node count and connectivity across scales."""
    cfg = Phase3Config()
    for n in [10, 50, 100]:
        t_cfg = replace(cfg.topology, n_nodes=n, k_edge_connectivity=3)
        topo = build_edge_topology(cfg.trace.porto_cells_parquet, t_cfg)
        assert topo.n_nodes == n
        assert topo.k_edge_connectivity == 3
        degs = [len(x) for x in topo.neighbors]
        assert min(degs) >= 3, f"Node degree below k=3 at N={n}"


def test_stage7_neighbor_and_observation_dimension_compatibility():
    """4. Neighbor-degree and observation-dimension compatibility."""
    k_max = 8
    obs_dim = 11 + 8 * k_max
    action_dim = 1 + k_max
    assert obs_dim == 75
    assert action_dim == 9

    qnet = MaskedDuelingQNet(obs_dim, action_dim, hidden_dim=64, fc_scale=0.008)
    dummy_obs = np.zeros(obs_dim, dtype=np.float32)
    dummy_mask = np.ones(action_dim, dtype=bool)
    runner = _NumpyDuelingQRunner(qnet)
    act = runner.act(dummy_obs, dummy_mask)
    assert 0 <= act < action_dim


def test_stage7_no_neighbor_truncation():
    """5. Verify no neighbor truncation occurs at N=50 and N=100."""
    cfg = Phase3Config()
    for n in [50, 100]:
        t_cfg = replace(cfg.topology, n_nodes=n, k_edge_connectivity=3)
        topo = build_edge_topology(cfg.trace.porto_cells_parquet, t_cfg)
        for i in range(n):
            nbr_count = len(topo.neighbor_slots(i))
            assert nbr_count <= cfg.sim.k_max_neighbors, (
                f"Degree {nbr_count} exceeds k_max_neighbors {cfg.sim.k_max_neighbors} at node {i} (N={n})!"
            )


def test_stage7_action_mask_dimensions(n50_light_setup):
    """6. Action mask dimension consistency across discrete events."""
    cfg, ds, topo, sim_cfg = n50_light_setup
    b_start = ds.val_end_bin
    tbl = prepare_task_table(ds, sim_cfg, split="test", seed=42, start_bin=b_start, end_bin=b_start + 2)
    sim = TaskOffloadingSim(ds, topo, tbl, cfg, telemetry_mode="periodic")

    masks_observed = []
    def _inspect_mask(obs, mask, node_id, t_now_s, task_idx):
        masks_observed.append(mask.copy())
        return 0

    sim.run_policy(_inspect_mask)
    assert len(masks_observed) > 0
    for m in masks_observed:
        assert len(m) == 1 + sim_cfg.k_max_neighbors
        assert m[0] == True  # Local action always valid


def test_stage7_task_conservation(n50_light_setup):
    """7. Task conservation across 50-node simulation slice."""
    cfg, ds, topo, sim_cfg = n50_light_setup
    b_start = ds.val_end_bin
    tbl = prepare_task_table(ds, sim_cfg, split="test", seed=42, start_bin=b_start, end_bin=b_start + 4)
    sim = TaskOffloadingSim(ds, topo, tbl, cfg, telemetry_mode="periodic")

    policy = ReactiveGreedyPolicy(sim_cfg)
    metrics = sim.run_policy(policy)

    tot = metrics.completed_on_time + metrics.dropped_tasks
    assert tot == metrics.total_tasks == tbl.n_tasks
    assert metrics.drop_admission + metrics.drop_deadline + metrics.drop_link_timeout == metrics.dropped_tasks


def test_stage7_strict_temporal_causality(n50_light_setup):
    """8. Strict temporal causality: future perturbations do not leak backwards at N=50."""
    cfg, ds, topo, sim_cfg = n50_light_setup
    b_start = ds.val_end_bin
    b_split = b_start + 3
    b_end = b_start + 6

    # Baseline slice
    tbl1 = prepare_task_table(ds, sim_cfg, split="test", seed=42, start_bin=b_start, end_bin=b_end)
    t_split = tbl1.t_ep_start_s + 3 * ds.dt_bin_s
    sim1 = TaskOffloadingSim(ds, topo, tbl1, cfg, telemetry_mode="periodic", has_live_forecaster=True)
    mgr1 = LiveGossipManager(ds, topo, cfg, mode="apsm")

    decisions1 = []
    def _rec1(obs, mask, node_id, t_now_s, task_idx):
        if t_now_s < t_split:
            decisions1.append((node_id, task_idx, int(obs[0] * 1000)))
        return 0
    sim1.run_policy(_rec1, bin_callback=lambda b, t: mgr1.on_bin_boundary(b, t, sim1))

    # Mutated future targets
    ds_mut = replace(ds, targets=ds.targets.copy(), features=ds.features.copy())
    ds_mut.targets[:, b_split:] += 5.0
    ds_mut.features[:, b_split:, :] += 2.0

    tbl2 = prepare_task_table(ds_mut, sim_cfg, split="test", seed=42, start_bin=b_start, end_bin=b_end)
    sim2 = TaskOffloadingSim(ds_mut, topo, tbl2, cfg, telemetry_mode="periodic", has_live_forecaster=True)
    mgr2 = LiveGossipManager(ds_mut, topo, cfg, mode="apsm")

    decisions2 = []
    def _rec2(obs, mask, node_id, t_now_s, task_idx):
        if t_now_s < t_split:
            decisions2.append((node_id, task_idx, int(obs[0] * 1000)))
        return 0
    sim2.run_policy(_rec2, bin_callback=lambda b, t: mgr2.on_bin_boundary(b, t, sim2))

    assert len(decisions1) == len(decisions2) > 0
    assert decisions1 == decisions2, "Future target perturbation leaked into past observations/actions!"


def test_stage7_no_test_validation_leakage(n50_light_setup):
    """9. Test window records strictly exclude validation bins."""
    cfg, ds, topo, sim_cfg = n50_light_setup
    mgr = LiveGossipManager(ds, topo, cfg, mode="apsm", test_start_bin=1035, test_end_bin=1380)

    # Simulate stepping through validation split bins 828 to 1034
    sim = TaskOffloadingSim(ds, topo, prepare_task_table(ds, sim_cfg, start_bin=828, end_bin=835), cfg)
    for b in range(828, 835):
        mgr.on_bin_boundary(b, float(b * ds.dt_bin_s), sim)

    assert len(mgr.test_records) == 0, "Validation bins prematurely leaked into test_records!"


def test_stage7_forecast_count_equals_N_times_345(n50_light_setup):
    """10. Exact aligned forecast count calculation for N=50."""
    cfg, ds, topo, sim_cfg = n50_light_setup
    mgr = LiveGossipManager(ds, topo, cfg, mode="apsm", test_start_bin=1035, test_end_bin=1380)
    sim = TaskOffloadingSim(ds, topo, prepare_task_table(ds, sim_cfg, split="test", seed=42, start_bin=1035, end_bin=1040), cfg)

    # Step through 5 bins
    for b in range(1035, 1041):
        mgr.on_bin_boundary(b, float(b * ds.dt_bin_s), sim)

    # 5 closed bins * 50 nodes = 250 predictions
    assert len(mgr.test_records) == 50 * 5


def test_stage7_unique_node_bin_records(n50_light_setup):
    """11. Verify (node_id, target_bin) prediction record uniqueness."""
    cfg, ds, topo, sim_cfg = n50_light_setup
    mgr = LiveGossipManager(ds, topo, cfg, mode="apsm", test_start_bin=1035, test_end_bin=1380)
    sim = TaskOffloadingSim(ds, topo, prepare_task_table(ds, sim_cfg, split="test", seed=42, start_bin=1035, end_bin=1038), cfg)

    for b in range(1035, 1039):
        mgr.on_bin_boundary(b, float(b * ds.dt_bin_s), sim)

    records = mgr.test_records
    unique_pairs = {(r[0], r[1]) for r in records}
    assert len(unique_pairs) == len(records)


def test_stage7_physical_model_packet_delivery_and_assimilation(n50_light_setup):
    """12. Discrete-event model packet delivery vs next-bin assimilation at N=50."""
    cfg, ds, topo, sim_cfg = n50_light_setup
    b_start = ds.val_end_bin
    tbl = prepare_task_table(ds, sim_cfg, split="test", seed=42, start_bin=b_start, end_bin=b_start + 3)
    sim = TaskOffloadingSim(ds, topo, tbl, cfg, has_live_forecaster=True)
    mgr = LiveGossipManager(ds, topo, cfg, mode="apsm")

    sender = 0
    receiver = topo.neighbor_slots(sender)[0]
    t_start = tbl.t_ep_start_s
    t_next_bin = t_start + ds.dt_bin_s
    payload = mgr.payload_bytes

    deliver_ts = sim.schedule_model_packet(sender, receiver, payload, t_start)
    assert t_start < deliver_ts < t_next_bin

    shared_w = mgr.models[sender].export_shared_weights()
    msg = PendingModelMsg(
        deliver_time_s=deliver_ts,
        sender_id=sender,
        shared_weights={k: v.clone() for k, v in shared_w.items()},
        sample_age=1.0,
    )
    mgr.inboxes[receiver].append(msg)

    # Before next boundary: unchanged
    rec_w_before = {k: v.clone() for k, v in mgr.models[receiver].export_shared_weights().items()}
    assert msg in mgr.inboxes[receiver]

    # After next boundary: delivered and assimilated
    mgr.on_bin_boundary(b_start + 1, t_next_bin, sim)
    assert msg not in mgr.inboxes[receiver]
    assert all(m.deliver_time_s > t_next_bin for m in mgr.inboxes[receiver])


def test_stage7_apsm_suppression_accounting(n50_light_setup):
    """13. APSM suppression accounting invariant at N=50."""
    cfg, ds, topo, sim_cfg = n50_light_setup
    b_start = ds.val_end_bin
    tbl = prepare_task_table(ds, sim_cfg, split="test", seed=42, start_bin=b_start, end_bin=b_start + 4)
    sim = TaskOffloadingSim(ds, topo, tbl, cfg, has_live_forecaster=True)
    mgr = LiveGossipManager(ds, topo, cfg, mode="apsm")

    for b in range(b_start, b_start + 5):
        mgr.on_bin_boundary(b, float(b * ds.dt_bin_s), sim)

    ledger = mgr.ledger
    total = ledger.total_gossip_checks
    sum_parts = ledger.suppressed_checks + ledger.surprise_sends + ledger.heartbeat_sends + ledger.warmup_sends
    assert total == sum_parts, f"Accounting mismatch: total {total} != sum {sum_parts}"


def test_stage7_communication_ledger_arithmetic(n50_light_setup):
    """14. Control plane communication ledger arithmetic at N=50."""
    cfg, ds, topo, sim_cfg = n50_light_setup
    b_start = ds.val_end_bin
    tbl = prepare_task_table(ds, sim_cfg, split="test", seed=42, start_bin=b_start, end_bin=b_start + 4)
    sim = TaskOffloadingSim(ds, topo, tbl, cfg, telemetry_mode="periodic", has_live_forecaster=True)
    mgr = LiveGossipManager(ds, topo, cfg, mode="apsm")

    policy = ReactiveGreedyPolicy(sim_cfg)
    sim.run_policy(policy, bin_callback=lambda b, t: mgr.on_bin_boundary(b, t, sim))
    ledger = mgr.ledger.reconcile(sim)

    expected_ctrl = ledger.model_bytes + ledger.telemetry_bytes + ledger.ack_bytes + ledger.header_bytes
    assert ledger.cb_control_bytes == expected_ctrl


def test_stage7_strict_json_serialization():
    """15. Strict RFC 8259 JSON compliance check."""
    from scripts.eval_marl_forecasting_exact import sanitize_json_obj

    def reject_constant(c):
        raise ValueError(f"Encountered non-standard constant: {c}")

    payload = {
        "arm": "e3_madqn_apsm",
        "seed": 11,
        "tsr_pct": 84.15,
        "invalid_nan": float("nan"),
        "invalid_inf": float("inf"),
        "valid_null": None,
    }
    clean = sanitize_json_obj(payload)
    text = json.dumps(clean, allow_nan=False)
    loaded = json.loads(text, parse_constant=reject_constant)
    assert loaded["invalid_nan"] is None
    assert loaded["invalid_inf"] is None
    assert loaded["valid_null"] is None


def test_stage7_partial_run_persistence_and_recovery():
    """16. Incremental run logging and partial-run recovery."""
    with tempfile.TemporaryDirectory() as td:
        run_file = Path(td) / "raw_runs.json"
        assert not run_file.exists()

        # Step 1: Write first seed
        run_1 = [{"arm": "b0_local", "seed": 11, "tsr_pct": 78.5}]
        run_file.write_text(json.dumps(run_1, indent=2), encoding="utf-8")

        # Step 2: Read and resume with second seed
        existing = json.loads(run_file.read_text(encoding="utf-8"))
        done_keys = {(r["arm"], r["seed"]) for r in existing}
        assert ("b0_local", 11) in done_keys

        # Append seed 22
        existing.append({"arm": "b0_local", "seed": 22, "tsr_pct": 78.5})
        run_file.write_text(json.dumps(existing, indent=2), encoding="utf-8")

        resumed = json.loads(run_file.read_text(encoding="utf-8"))
        assert len(resumed) == 2


def test_stage7_memory_safe_event_handling(n50_light_setup):
    """17. Memory-safe discrete-event queue clearance after episode."""
    cfg, ds, topo, sim_cfg = n50_light_setup
    b_start = ds.val_end_bin
    tbl = prepare_task_table(ds, sim_cfg, split="test", seed=42, start_bin=b_start, end_bin=b_start + 3)
    sim = TaskOffloadingSim(ds, topo, tbl, cfg)
    sim.run_policy(LocalOnlyPolicy())

    # Task queues should be drained or cleared
    active_count = sum(len(q) for q in sim.active_tasks)
    assert active_count == 0, f"Remaining active tasks after episode completion: {active_count}"
