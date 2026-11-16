"""
Unit & Integration Tests for Stage 6 Live Decentralized Gossip Learning & MARL System.

Verifies:
  1. Live forecast injection with forecast_matrix=None.
  2. Strict temporal causality (future target perturbation does not leak into earlier actions).
  3. APSM-H1 surprise, heartbeat, and suppression mechanics at N=10.
  4. Physical link delay & age-weighted merge on model arrival.
  5. 100% task conservation across discrete-event simulation.
  6. Channel separation & communication ledger reconciliation.
"""

from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
import torch

from phase3.config import Phase3Config
from phase3.data.alibaba_loader import build_or_load_curated_trace
from phase3.data.topology import build_edge_topology
from phase3.env.baselines import LocalOnlyPolicy, ReactiveGreedyPolicy
from phase3.env.task_simulator import PreparedTaskTable, TaskOffloadingSim, prepare_task_table
from phase3.forecasting.gossip_runner import _pretrain_base_models
from phase3.forecasting.models import EdgeLSTMForecaster
from phase3.integration.live_gossip_engine import CommunicationLedger, LiveGossipManager, PendingModelMsg
from phase3.marl.madqn import MaskedDuelingQNet, _NumpyDuelingQRunner


@pytest.fixture(scope="module")
def stage6_setup():
    cfg = Phase3Config()
    topo_cfg = replace(cfg.topology, n_nodes=10, k_edge_connectivity=3)
    sim_cfg = replace(cfg.sim, node_cpu_cap=20.0, deadline_slack_kappa=2.2, q_max=48)

    ds = build_or_load_curated_trace(cfg.trace, n_nodes=10)
    topo = build_edge_topology(cfg.trace.porto_cells_parquet, topo_cfg)
    return cfg, ds, topo, sim_cfg


def test_channel_separation_and_ledger(stage6_setup):
    """Test that model bytes and telemetry bytes are strictly separated yet unified in ledger."""
    cfg, ds, topo, sim_cfg = stage6_setup

    manager = LiveGossipManager(ds, topo, cfg, mode="apsm")
    assert manager.payload_bytes == 56920
    assert manager.ledger.model_bytes == 0
    assert manager.ledger.model_packets == 0

    # Short test slice: 10 bins (300 seconds)
    tbl = prepare_task_table(ds, sim_cfg, split="test", seed=42, start_bin=ds.val_end_bin, end_bin=ds.val_end_bin + 10)
    sim = TaskOffloadingSim(ds, topo, tbl, cfg, telemetry_mode="periodic", has_live_forecaster=True)

    metrics = sim.run_policy(
        LocalOnlyPolicy(),
        bin_callback=lambda b, t: manager.on_bin_boundary(b, t, sim),
    )

    ledger = manager.ledger.reconcile(sim)
    assert ledger.model_bytes == sim.dynamic_model_bytes
    assert ledger.model_packets == sim.dynamic_model_packets
    assert ledger.telemetry_bytes == sim.telemetry_bytes
    assert ledger.telemetry_packets == sim.telemetry_packets
    assert ledger.cb_control_bytes == (
        ledger.model_bytes + ledger.telemetry_bytes + ledger.ack_bytes + ledger.header_bytes
    )
    pure_payload = max(0, ledger.offload_data_bytes - ledger.header_bytes)
    assert ledger.cb_system_bytes == ledger.cb_control_bytes + pure_payload


def test_live_forecast_injection(stage6_setup):
    """Verify that sim.cur_y_hat is populated live by LiveGossipManager without static matrix."""
    cfg, ds, topo, sim_cfg = stage6_setup

    manager = LiveGossipManager(ds, topo, cfg, mode="apsm")
    tbl = prepare_task_table(ds, sim_cfg, split="test", seed=42, start_bin=ds.val_end_bin, end_bin=ds.val_end_bin + 6)
    sim = TaskOffloadingSim(ds, topo, tbl, cfg, forecast_matrix=None, has_live_forecaster=True)

    y_hat_observed = []

    def tracking_policy(obs, mask, node_id, t_now_s, task_idx):
        y_hat_observed.append(float(obs[4]))
        return 0

    metrics = sim.run_policy(
        tracking_policy,
        bin_callback=lambda b, t: manager.on_bin_boundary(b, t, sim),
    )

    # Some observations should have positive predictions once stepped
    assert len(y_hat_observed) > 0
    assert any(y > 0.0 for y in y_hat_observed), "Live forecasts should be non-zero in obs[4]"


def test_apsm_suppression_at_10n(stage6_setup):
    """Verify that APSM suppresses model checks compared to unfiltered baseline."""
    cfg, ds, topo, sim_cfg = stage6_setup

    tbl = prepare_task_table(ds, sim_cfg, split="test", seed=42, start_bin=ds.val_end_bin, end_bin=ds.val_end_bin + 30)

    # Arm 1: APSM gated
    mgr_apsm = LiveGossipManager(ds, topo, cfg, mode="apsm")
    sim_apsm = TaskOffloadingSim(ds, topo, tbl, cfg, has_live_forecaster=True)
    sim_apsm.run_policy(LocalOnlyPolicy(), bin_callback=lambda b, t: mgr_apsm.on_bin_boundary(b, t, sim_apsm))
    ledger_apsm = mgr_apsm.ledger.reconcile(sim_apsm)

    # Arm 2: Unfiltered baseline
    mgr_gl = LiveGossipManager(ds, topo, cfg, mode="unfiltered_gl")
    sim_gl = TaskOffloadingSim(ds, topo, tbl, cfg, has_live_forecaster=True)
    sim_gl.run_policy(LocalOnlyPolicy(), bin_callback=lambda b, t: mgr_gl.on_bin_boundary(b, t, sim_gl))
    ledger_gl = mgr_gl.ledger.reconcile(sim_gl)

    assert ledger_apsm.total_gossip_checks == ledger_gl.total_gossip_checks
    assert ledger_apsm.model_bytes < ledger_gl.model_bytes, "APSM must transmit strictly fewer bytes than unfiltered GL"
    assert ledger_apsm.suppression_rate_pct > 30.0, f"APSM suppression rate too low: {ledger_apsm.suppression_rate_pct:.1f}%"


def test_100_percent_task_conservation(stage6_setup):
    """Verify 100% task conservation on test tasks under live integration."""
    cfg, ds, topo, sim_cfg = stage6_setup

    tbl = prepare_task_table(ds, sim_cfg, split="test", seed=42, start_bin=ds.val_end_bin, end_bin=ds.val_end_bin + 20)
    sim = TaskOffloadingSim(ds, topo, tbl, cfg, has_live_forecaster=True, verify_invariants=True)
    mgr = LiveGossipManager(ds, topo, cfg, mode="apsm")

    metrics, outcomes = sim.run_marl_episode(
        lambda o, m, g, nid, t, tid: 0,
        bin_callback=lambda b, t: mgr.on_bin_boundary(b, t, sim),
    )

    assert metrics.total_tasks == tbl.n_tasks
    assert metrics.completed_on_time + metrics.dropped_tasks == metrics.total_tasks
    assert len(outcomes) == tbl.n_tasks


def test_strict_temporal_causality(stage6_setup):
    """
    Verify Issue D1 fix: perturbing future target/feature data strictly after cutoff bin T_cut
    does NOT leak into or change any decisions taken prior to T_cut under a real policy.
    """
    cfg, ds, topo, sim_cfg = stage6_setup
    b_start = ds.val_end_bin
    b_end = ds.val_end_bin + 20
    b_cut = ds.val_end_bin + 10
    t_cut_s = ds.t_start_s + b_cut * ds.dt_bin_s

    tbl = prepare_task_table(ds, sim_cfg, split="test", seed=42, start_bin=b_start, end_bin=b_end)

    # Run 1: original data under ReactiveGreedyPolicy
    policy1 = ReactiveGreedyPolicy(sim_cfg)
    actions1 = []
    mgr1 = LiveGossipManager(ds, topo, cfg, mode="apsm")
    sim1 = TaskOffloadingSim(ds, topo, tbl, cfg, has_live_forecaster=True)
    sim1.run_policy(
        lambda o, m, nid, t, tid: (actions1.append((t, policy1(o, m, nid, t, tid))) or actions1[-1][1]),
        bin_callback=lambda b, t: mgr1.on_bin_boundary(b, t, sim1),
    )

    # Run 2: perturb targets strictly at and after b_cut
    ds_perturbed = replace(ds, targets=ds.targets.copy(), features=ds.features.copy())
    ds_perturbed.targets[:, b_cut:] += 5.0
    ds_perturbed.features[:, b_cut:, :] += 2.0

    policy2 = ReactiveGreedyPolicy(sim_cfg)
    actions2 = []
    mgr2 = LiveGossipManager(ds_perturbed, topo, cfg, mode="apsm")
    sim2 = TaskOffloadingSim(ds_perturbed, topo, tbl, cfg, has_live_forecaster=True)
    sim2.run_policy(
        lambda o, m, nid, t, tid: (actions2.append((t, policy2(o, m, nid, t, tid))) or actions2[-1][1]),
        bin_callback=lambda b, t: mgr2.on_bin_boundary(b, t, sim2),
    )

    prior1 = [act for (t, act) in actions1 if t < t_cut_s]
    prior2 = [act for (t, act) in actions2 if t < t_cut_s]
    assert len(prior1) > 0, "Should have prior actions before cutoff"
    assert prior1 == prior2, "Decisions prior to cutoff must be bit-for-bit identical (zero temporal leakage)"


def test_shared_physical_link_contention(stage6_setup):
    """
    Verify Issue C1 fix: when a model packet transmits on directed link (i, j),
    an offloaded task scheduled on link (i, j) queues behind the model packet,
    experiencing positive transmission queueing delay.
    """
    cfg, ds, topo, sim_cfg = stage6_setup
    tbl = prepare_task_table(ds, sim_cfg, split="test", seed=42, start_bin=ds.val_end_bin, end_bin=ds.val_end_bin + 5)
    sim = TaskOffloadingSim(ds, topo, tbl, cfg, has_live_forecaster=False)

    i = 0
    nbrs = topo.neighbor_slots(i)
    j = nbrs[0]
    t0 = tbl.t_ep_start_s
    model_bytes = 56920

    # Schedule model packet at t0 on directed link (i -> j)
    tx_end_model, arr_model = sim._schedule_link_tx(i, j, model_bytes, t0)
    bw_bps = float(topo.bw_mbps[i, j]) * 1e6
    expected_d_tx = (8.0 * model_bytes) / bw_bps
    assert abs((tx_end_model - t0) - expected_d_tx) < 1e-6
    assert sim.link_free_at[i, j] == tx_end_model

    # Now schedule a task on the same directed link (i -> j) at t0 + 1ms (while model is transmitting)
    t_task = t0 + 0.001
    task_payload = 100000
    task_total = task_payload + sim.sim.task_header_bytes
    tx_end_task, arr_task = sim._schedule_link_tx(i, j, task_total, t_task)

    # Task transmission start MUST be queued until tx_end_model
    task_d_tx = (8.0 * task_total) / bw_bps
    assert abs(tx_end_task - (tx_end_model + task_d_tx)) < 1e-6, "Task must queue behind model packet on shared link"
    assert tx_end_task > tx_end_model


def test_madqn_pytorch_numpy_parity():
    """Verify Issue C7 fix: float32 numerical parity between PyTorch MaskedDuelingQNet and NumPy runner."""
    torch.manual_seed(42)
    np.random.seed(42)
    obs_dim = 75
    action_dim = 9
    qnet = MaskedDuelingQNet(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=96, fc_scale=0.008)
    qnet.eval()
    runner = _NumpyDuelingQRunner(qnet)

    for i in range(10):
        obs = np.random.randn(obs_dim).astype(np.float32)
        obs[0:7] = np.abs(obs[0:7])
        mask = np.ones(action_dim, dtype=bool)
        if i % 2 == 0:
            mask[2] = False
            mask[5] = False

        q_np = runner.q_values(obs, mask)
        t_obs = torch.from_numpy(obs).unsqueeze(0)
        t_mask = torch.from_numpy(mask).unsqueeze(0)
        with torch.no_grad():
            q_pt = qnet(t_obs, t_mask).squeeze(0).cpu().numpy()

        max_diff = np.max(np.abs(q_np[mask] - q_pt[mask]))
        assert max_diff < 1e-4, f"Parity mismatch at iter {i}: max_diff={max_diff}"


def test_marl_deep_temporal_causality_with_instrumentation(stage6_setup):
    """
    Strengthened Temporal Causality Test:
    Loads a real trained MADQN checkpoint (or fixed deterministic initialization)
    and exercises the actual production inference path (_NumpyDuelingQRunner).
    Instruments:
      - Forecast outputs
      - APSM residuals
      - Gating thresholds
      - Actual gate send/suppress decision reasons
      - Model packet scheduling and transmissions
      - Learned policy actions
    Perturbs future target/features strictly after T_cut and asserts bit-for-bit identical outputs prior to T_cut.
    """
    cfg, ds, topo, sim_cfg = stage6_setup
    b_start = ds.val_end_bin
    b_end = ds.val_end_bin + 20
    b_cut = ds.val_end_bin + 10
    t_cut_s = ds.t_start_s + b_cut * ds.dt_bin_s

    tbl = prepare_task_table(ds, sim_cfg, split="test", seed=42, start_bin=b_start, end_bin=b_end)

    # Pretrain base LSTM forecasters with fixed seed for determinism across runs
    cfg_cpu = replace(cfg, device="cpu")
    base_lstm_states, _ = _pretrain_base_models(ds, cfg_cpu)

    # Load actual trained MADQN checkpoint if available
    ckpt_file = Path("results/phase3/checkpoints_stage6_10n_audited_v2/ckpt_best_e3_madqn_apsm_s11.pt")
    k_max = cfg.sim.k_max_neighbors
    obs_dim = 11 + 8 * k_max
    action_dim = 1 + k_max

    qnet = MaskedDuelingQNet(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=cfg.marl.hidden_dim, fc_scale=0.008)
    if ckpt_file.exists():
        ckpt_data = torch.load(ckpt_file, map_location="cpu")
        qnet.load_state_dict(ckpt_data["state_dict"])
        qnet.fc_res.fc_active = ckpt_data.get("fc_active", True)
    qnet.eval()
    q_runner = _NumpyDuelingQRunner(qnet)

    def run_instrumented(dataset_inst):
        mgr = LiveGossipManager(dataset_inst, topo, cfg, mode="apsm", base_states=base_lstm_states, device=torch.device("cpu"))
        sim = TaskOffloadingSim(dataset_inst, topo, tbl, cfg, has_live_forecaster=True)

        trace = {
            "actions": [],
            "forecasts": [],
            "residuals": [],
            "thresholds": [],
            "decisions": [],
            "tx_events": [],
        }

        def instrumented_policy(obs, mask, node_id, t_now_s, task_idx):
            act = int(q_runner.act(obs, mask, epsilon=0.0))
            trace["actions"].append((t_now_s, node_id, act))
            return act

        def instrumented_callback(cur_bin, t_now_s):
            mgr.on_bin_boundary(cur_bin, t_now_s, sim)
            for i in range(10):
                trace["forecasts"].append((t_now_s, i, float(mgr.last_preds[i])))
                trace["residuals"].append((t_now_s, i, float(mgr.gates[i]._last_eps)))
                trace["thresholds"].append((t_now_s, i, float(mgr.gates[i]._last_tau)))
                trace["decisions"].append((t_now_s, i, str(mgr.gates[i]._pending_surprise)))
            trace["tx_events"].append((t_now_s, mgr.ledger.model_packets, mgr.ledger.model_bytes))

        sim.run_policy(instrumented_policy, bin_callback=instrumented_callback)
        return trace

    # Run 1: Unperturbed
    trace1 = run_instrumented(ds)

    # Run 2: Strongly perturb future targets and features at and after b_cut
    ds_perturbed = replace(ds, targets=ds.targets.copy(), features=ds.features.copy())
    ds_perturbed.targets[:, b_cut:] += 10.0
    ds_perturbed.features[:, b_cut:, :] += 5.0
    trace2 = run_instrumented(ds_perturbed)

    # Filter all pre-cutoff events
    for key in ["actions", "forecasts", "residuals", "thresholds", "decisions", "tx_events"]:
        pre1 = [x for x in trace1[key] if x[0] < t_cut_s]
        pre2 = [x for x in trace2[key] if x[0] < t_cut_s]
        assert len(pre1) > 0, f"No pre-cutoff items for {key}"
        assert pre1 == pre2, f"Temporal causality violation in {key} prior to T_cut!"


def test_model_packet_delivery_vs_bin_callback_assimilation(stage6_setup):
    """
    Verify discrete-event model packet delivery vs next-bin assimilation:
    A model packet is scheduled on directed link (sender -> receiver) through the actual simulator.
    1. Verify physical transmission and arrival timestamp t_rx.
    2. Verify physical arrival occurs between bin boundary t_start and next bin boundary t_start + 30s.
    3. Verify that at t in (t_rx, t_start + 30.0s), arrival alone does NOT cause model assimilation (inbox has pending message).
    4. Verify that recipient weights prior to next bin boundary are unchanged.
    5. Invoke bin callback at next boundary (cur_bin = b_start + 1).
    6. Verify message is removed from inbox and model weights are assimilated via age_weighted_merge.
    """
    cfg, ds, topo, sim_cfg = stage6_setup
    b_start = ds.val_end_bin
    tbl = prepare_task_table(ds, sim_cfg, split="test", seed=42, start_bin=b_start, end_bin=b_start + 5)
    sim = TaskOffloadingSim(ds, topo, tbl, cfg, has_live_forecaster=True)
    mgr = LiveGossipManager(ds, topo, cfg, mode="apsm")

    sender = 0
    receiver = topo.neighbor_slots(sender)[0]
    t_start = tbl.t_ep_start_s
    t_next_bin = t_start + ds.dt_bin_s
    payload_bytes = mgr.payload_bytes

    # 1. Schedule model packet through the actual simulator method
    deliver_ts = sim.schedule_model_packet(sender, receiver, payload_bytes, t_start)

    bw_bps = float(topo.bw_mbps[sender, receiver]) * 1e6
    expected_d_tx = (8.0 * payload_bytes) / bw_bps
    prop_delay = float(topo.rtt_s[sender, receiver]) / 2.0
    expected_arr = t_start + expected_d_tx + prop_delay

    assert abs(deliver_ts - expected_arr) < 1e-5, f"Delivery TS mismatch: {deliver_ts} vs {expected_arr}"
    assert t_start < deliver_ts < t_next_bin, f"Delivery TS {deliver_ts} should arrive between {t_start} and {t_next_bin}"

    # 2. Put message into receiver inbox with arrival timestamp
    shared_w = mgr.models[sender].export_shared_weights()
    msg = PendingModelMsg(
        deliver_time_s=deliver_ts,
        sender_id=sender,
        shared_weights={k: v.clone() for k, v in shared_w.items()},
        sample_age=1.0,
    )
    mgr.inboxes[receiver].append(msg)

    # 3. Snapshot recipient model weights before assimilation
    rec_w_before = {k: v.clone() for k, v in mgr.models[receiver].export_shared_weights().items()}

    # 4. Check intermediate state: physical arrival has occurred, but bin callback has not fired
    assert len(mgr.inboxes[receiver]) == 1
    rec_w_intermediate = mgr.models[receiver].export_shared_weights()
    for k in rec_w_before:
        assert torch.equal(rec_w_before[k], rec_w_intermediate[k]), "Premature weight assimilation before bin boundary!"

    # 5. Invoke the next actual bin-boundary callback at t=t_next_bin (cur_bin=b_start + 1)
    mgr.on_bin_boundary(cur_bin=b_start + 1, t_now_s=t_next_bin, sim=sim)

    # 6. The pending message must now be delivered and removed from inbox
    # (Any new messages in inbox must be strictly in-flight for future bins, deliver_time_s > t_next_bin)
    assert msg not in mgr.inboxes[receiver]
    assert all(m.deliver_time_s > t_next_bin for m in mgr.inboxes[receiver])

    # 7. Recipient model weights must now reflect the merged weights
    rec_w_after = mgr.models[receiver].export_shared_weights()
    any_diff = any(not torch.equal(rec_w_before[k], rec_w_after[k]) for k in rec_w_before)
    assert any_diff, "Weights were not assimilated after bin boundary callback!"


def test_holm_bonferroni_adjustment():
    """Verify Holm-Bonferroni step-down multiplier, monotonicity, and family-wise control."""
    from phase3.analyze_stage6_results import holm_bonferroni

    raw_p = [0.012, 0.045, 0.002, 0.250]
    adj_p = holm_bonferroni(raw_p)
    assert len(adj_p) == len(raw_p)

    # Smallest raw p (0.002, index 2, rank 0): (4 - 0) * 0.002 = 0.008
    assert abs(adj_p[2] - 0.008) < 1e-6

    # Monotonicity check: higher raw p cannot have smaller adjusted p
    sorted_pairs = sorted(zip(raw_p, adj_p), key=lambda x: x[0])
    for i in range(len(sorted_pairs) - 1):
        assert sorted_pairs[i][1] <= sorted_pairs[i + 1][1]

    # Upper bound check
    assert all(0.0 <= p <= 1.0 for p in adj_p)


def test_forecasting_metrics_time_alignment_and_non_leakage(stage6_setup):
    """
    Verify test forecasting metric time alignment and non-leakage invariants:
      1. Exactly 3,450 aligned test predictions (10 nodes x 345 bins).
      2. No validation targets (bins < 1035) present in test records.
      3. Bin 1035 is included for all nodes.
      4. Final bin 1379 is included for all nodes.
      5. Persistence prediction uses exactly y(t-1) for matching target t.
    """
    cfg, ds, topo, sim_cfg = stage6_setup
    mgr = LiveGossipManager(ds, topo, cfg, mode="apsm", test_start_bin=1035, test_end_bin=1380)

    # Simulate bin callbacks across test split [1035, 1380)
    tbl = prepare_task_table(ds, sim_cfg, split="test", seed=42)
    sim = TaskOffloadingSim(ds, topo, tbl, cfg, has_live_forecaster=True)
    sim.run_policy(LocalOnlyPolicy(), bin_callback=lambda b, t: mgr.on_bin_boundary(b, t, sim))

    metrics = mgr.compute_forecasting_metrics()
    records = mgr.test_records

    # 1. Exactly 3,450 aligned predictions
    assert len(records) == 3450, f"Expected 3,450 records, got {len(records)}"

    # 2. No validation targets
    target_bins = [r[1] for r in records]
    assert all(b >= 1035 for b in target_bins), "Validation target leaked into test records!"
    assert all(b < 1380 for b in target_bins), "Out of range target in test records!"

    # 3. Bin 1035 is included for all 10 nodes
    nodes_1035 = [r[0] for r in records if r[1] == 1035]
    assert sorted(nodes_1035) == list(range(10)), "Bin 1035 missing nodes!"

    # 4. Bin 1379 is included for all 10 nodes
    nodes_1379 = [r[0] for r in records if r[1] == 1379]
    assert sorted(nodes_1379) == list(range(10)), "Bin 1379 missing nodes!"

    # 5. Persistence prediction matches y(t-1)
    for node_id, b, y_true, y_pred, y_pers in records:
        assert abs(y_true - float(ds.targets[node_id, b])) < 1e-6
        assert abs(y_pers - float(ds.targets[node_id, b - 1])) < 1e-6

    # 6. Finite positive metrics
    assert 0.0 < metrics["fc_mse"] < 0.10
    assert 0.0 < metrics["fc_rmse"] < 0.40
    assert 0.0 < metrics["fc_pers_mse"] < 0.10

