from dataclasses import asdict, replace

import numpy as np
import torch

from phase3.config import dev_tier1_6n
from phase3.data.alibaba_loader import build_or_load_curated_trace
from phase3.data.topology import build_edge_topology
from phase3.env import TaskOffloadingSim, prepare_task_table
from phase3.forecasting.gossip_runner import generate_causal_forecast_matrices
from phase3.marl.madqn import MADQNAgent, MaskedDuelingQNet, _NumpyDuelingQRunner, evaluate_dqn_policy
from phase3.marl.mappo import MAPPOAgent, MaskedActorCritic, _NumpyActorRunner, evaluate_marl_policy


def test_numpy_runners_match_pytorch_and_enforce_mask():
    obs_dim, global_dim, act_dim = 75, 91, 9
    rng = np.random.default_rng(123)
    ac = MaskedActorCritic(obs_dim, global_dim, act_dim, hidden_dim=64, use_central_critic=True, actor_uses_global=False)
    ac.eval()
    qnet = MaskedDuelingQNet(obs_dim, act_dim, hidden_dim=64)
    qnet.eval()

    np_actor = _NumpyActorRunner(ac)
    np_q = _NumpyDuelingQRunner(qnet)

    for _ in range(10):
        obs = rng.normal(size=obs_dim).astype(np.float32)
        gstate = rng.normal(size=global_dim).astype(np.float32)
        mask = np.zeros(act_dim, dtype=bool)
        mask[0] = True
        mask[1] = True
        mask[3] = True

        with torch.no_grad():
            t_lg = ac.actor_logits(torch.from_numpy(obs), torch.from_numpy(mask), torch.from_numpy(gstate)).numpy()
            t_q = qnet(torch.from_numpy(obs).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0)).squeeze(0).numpy()

        n_lg = np_actor.logits(obs, mask, gstate)
        n_q = np_q.q_values(obs, mask)

        assert np.allclose(t_lg[mask], n_lg[mask], atol=1e-4)
        assert np.allclose(t_q[mask], n_q[mask], atol=1e-4)
        assert np.all(n_lg[~mask] <= -1e8)
        assert np.all(n_q[~mask] <= -1e8)

        for _draw in range(50):
            a_p = np_actor.sample(obs, mask, gstate, rng, deterministic=False)
            a_q = np_q.act(obs, mask, rng, epsilon=0.5)
            assert bool(mask[a_p]) and bool(mask[a_q])


def test_ctde_decentralized_actor_isolation_vs_ippo():
    obs_dim, global_dim, act_dim = 75, 91, 9
    mappo = MaskedActorCritic(obs_dim, global_dim, act_dim, hidden_dim=64, use_central_critic=True, actor_uses_global=False)
    ippo = MaskedActorCritic(obs_dim, global_dim, act_dim, hidden_dim=64, use_central_critic=False, actor_uses_global=False)
    obs = torch.randn(4, obs_dim)
    mask = torch.ones(4, act_dim, dtype=torch.bool)
    mask[:, 4:] = False
    g_a = torch.cat([obs, torch.zeros(4, 16)], dim=-1)
    g_b = torch.cat([obs, torch.ones(4, 16) * 5.0], dim=-1)

    with torch.no_grad():
        # Actor logits must NEVER depend on central global summary unless actor_uses_global=True
        assert torch.allclose(mappo.actor_logits(obs, mask, g_a), mappo.actor_logits(obs, mask, g_b))
        # MAPPO critic depends on gstate; IPPO critic depends strictly on local obs
        assert not torch.allclose(mappo.critic_value(obs, g_a), mappo.critic_value(obs, g_b))
        assert torch.allclose(ippo.critic_value(obs, g_a), ippo.critic_value(obs, g_b))


def test_stage5_short_training_step_and_causal_local_lstm_matrix():
    cfg = dev_tier1_6n(seed=42)
    cfg = replace(
        cfg,
        forecaster=replace(cfg.forecaster, pretrain_epochs=4),
        marl=replace(cfg.marl, n_train_episodes=2, train_window_bins=30, eval_every=2, dqn_batch_size=32, dqn_updates_per_episode=2),
    )
    ds = build_or_load_curated_trace(cfg.trace, n_nodes=6)
    topo = build_edge_topology(cfg.trace.porto_cells_parquet, cfg.topology)

    fc = generate_causal_forecast_matrices("local_lstm", ds, topo, cfg)
    fmat, smat = fc["forecast_matrix"], fc["surprise_matrix"]
    assert fmat.shape == (6, ds.n_bins) and smat.shape == (6, ds.n_bins)
    assert np.isfinite(fmat).all() and np.isfinite(smat).all()
    assert np.all(smat >= 0.0) and np.all(smat <= cfg.apsm.surprise_clip + 1e-5)

    # Check causal update on MAPPO and MADQN over tiny episode slices
    tasks = prepare_task_table(ds, cfg.sim, start_bin=ds.lookback_L, end_bin=ds.lookback_L + 30)
    sim = TaskOffloadingSim(ds, topo, tasks, cfg, telemetry_mode="periodic", forecast_matrix=fmat, surprise_matrix=smat)

    mappo = MAPPOAgent(cfg, use_central_critic=True, name="test_mappo")
    w_before = mappo.ac.actor[0].weight.detach().clone()
    st_p = mappo._collect_and_update(sim, 0)
    assert np.isfinite(st_p["policy_loss"]) and np.isfinite(st_p["value_loss"])
    assert not torch.allclose(w_before, mappo.ac.actor[0].weight)

    madqn = MADQNAgent(cfg, name="test_madqn")
    q_before = madqn.q_online.feature[0].weight.detach().clone()
    st_q = madqn._collect_and_update(sim, 0)
    assert np.isfinite(st_q["td_loss"])
    assert not torch.allclose(q_before, madqn.q_online.feature[0].weight)

    # Also verify evaluation entrypoints calling run_marl_episode
    m_eval_p = evaluate_marl_policy(mappo.ac, sim, deterministic=True, seed=42)
    m_eval_q = evaluate_dqn_policy(madqn.q_online, sim, seed=42)
    assert m_eval_p.total_tasks == tasks.n_tasks
    assert m_eval_q.total_tasks == tasks.n_tasks


def test_run_marl_episode_conservation_delayed_outcomes_and_callbacks():
    cfg = dev_tier1_6n(seed=42)
    ds = build_or_load_curated_trace(cfg.trace, n_nodes=6)
    topo = build_edge_topology(cfg.trace.porto_cells_parquet, cfg.topology)
    b_start = ds.val_end_bin
    b_end = ds.val_end_bin + 20
    tasks = prepare_task_table(ds, cfg.sim, start_bin=b_start, end_bin=b_end)

    fmat = np.zeros((ds.n_nodes, ds.n_bins), dtype=np.float32)
    smat = np.zeros((ds.n_nodes, ds.n_bins), dtype=np.float32)

    seen_tids = []
    chosen_targets = {}
    seen_bins = []
    observed_yhat_after_cb = []

    def bin_cb(cur_bin: int, t_now_s: float) -> None:
        seen_bins.append((cur_bin, t_now_s))
        fmat[:, cur_bin] = 0.25 + 0.01 * (cur_bin - b_start)
        smat[:, cur_bin] = 0.5

    def marl_step(obs, mask, gstate, node_id, t_now_s, task_idx) -> int:
        assert obs.shape == (75,) and gstate.shape == (91,) and mask.shape == (9,)
        assert np.allclose(gstate[:75], obs)
        assert np.isfinite(obs).all() and np.isfinite(gstate).all()
        assert 0 <= node_id < ds.n_nodes
        assert t_now_s >= tasks.t_ep_start_s
        assert obs[4] >= 0.25 - 1e-6  # Set by bin_callback before task decisions in the bin
        observed_yhat_after_cb.append(float(obs[4]))

        act = 1 if (task_idx % 2 == 1 and bool(mask[1])) else 0
        nbrs = topo.neighbor_slots(node_id)
        seen_tids.append(task_idx)
        chosen_targets[task_idx] = node_id if act == 0 else nbrs[act - 1]
        return act

    sim_marl = TaskOffloadingSim(
        ds,
        topo,
        tasks,
        cfg,
        telemetry_mode="periodic",
        forecast_matrix=fmat,
        surprise_matrix=smat,
        verify_invariants=True,
    )
    m_marl, outcomes = sim_marl.run_marl_episode(marl_step, bin_callback=bin_cb)

    # 1. Callback invocation & exact bin coverage
    assert len(seen_bins) == (b_end - b_start)
    assert [b for b, _ in seen_bins] == list(range(b_start, b_end))
    assert seen_tids == list(range(tasks.n_tasks))
    assert len(observed_yhat_after_cb) == tasks.n_tasks

    # 2. Task conservation & exact 1-to-1 outcome attribution
    assert m_marl.total_tasks == tasks.n_tasks
    assert m_marl.completed_on_time + m_marl.dropped_tasks == tasks.n_tasks
    assert m_marl.offload_count > 0
    assert m_marl.ack_packets == m_marl.offload_count
    assert m_marl.ack_bytes == m_marl.ack_packets * cfg.sim.ack_bytes
    assert len(sim_marl.outcomes) == tasks.n_tasks
    assert set(outcomes.keys()) == set(range(tasks.n_tasks))

    # 3. Delayed ACKs & varying durations resolve out of dispatch order, yet keyed attribution stays exact
    resolved_order = [o.task_id for o in sim_marl.outcomes]
    assert resolved_order != list(range(tasks.n_tasks))
    for tid in range(tasks.n_tasks):
        out = outcomes[tid]
        origin = int(tasks.node_idx[tid])
        assert out.task_id == tid
        assert out.origin_id == origin
        assert out.target_id == chosen_targets[tid]
        assert out.was_offloaded == (chosen_targets[tid] != origin)
        assert np.isfinite(out.reward)

    # 4. Exact behavioral equivalence with run_policy under identical decisions
    fmat_ref = np.zeros((ds.n_nodes, ds.n_bins), dtype=np.float32)
    smat_ref = np.zeros((ds.n_nodes, ds.n_bins), dtype=np.float32)

    def bin_cb_ref(cur_bin: int, _t: float) -> None:
        fmat_ref[:, cur_bin] = 0.25 + 0.01 * (cur_bin - b_start)
        smat_ref[:, cur_bin] = 0.5

    def ref_policy(_obs, mask, _node_id, _t_now_s, task_idx) -> int:
        return 1 if (task_idx % 2 == 1 and bool(mask[1])) else 0

    sim_ref = TaskOffloadingSim(
        ds,
        topo,
        tasks,
        cfg,
        telemetry_mode="periodic",
        forecast_matrix=fmat_ref,
        surprise_matrix=smat_ref,
        verify_invariants=True,
    )
    m_ref = sim_ref.run_policy(ref_policy, bin_callback=bin_cb_ref)
    assert asdict(m_marl) == asdict(m_ref)


def test_run_marl_episode_gstate_strict_causality():
    cfg = dev_tier1_6n(seed=42)
    ds = build_or_load_curated_trace(cfg.trace, n_nodes=6)
    topo = build_edge_topology(cfg.trace.porto_cells_parquet, cfg.topology)
    tasks_a = prepare_task_table(ds, cfg.sim, start_bin=ds.val_end_bin, end_bin=ds.val_end_bin + 10)
    tasks_b = prepare_task_table(ds, cfg.sim, start_bin=ds.val_end_bin, end_bin=ds.val_end_bin + 10)

    cut_ts = tasks_a.t_ep_start_s + 5 * ds.dt_bin_s
    fut_mask = tasks_b.arr_ts_s >= cut_ts
    tasks_b.cpu_work[fut_mask] *= 10.0

    trace_a, trace_b = [], []

    def rec_step(log):
        def _fn(obs, mask, gstate, _node_id, t_now_s, task_idx) -> int:
            if t_now_s < cut_ts:
                log.append((obs.copy(), gstate.copy()))
            return 1 if (task_idx % 3 == 0 and bool(mask[1])) else 0
        return _fn

    _, out_a = TaskOffloadingSim(ds, topo, tasks_a, cfg, telemetry_mode="periodic", verify_invariants=True).run_marl_episode(
        rec_step(trace_a)
    )
    _, out_b = TaskOffloadingSim(ds, topo, tasks_b, cfg, telemetry_mode="periodic", verify_invariants=True).run_marl_episode(
        rec_step(trace_b)
    )

    assert len(trace_a) == len(trace_b) and len(trace_a) > 0
    for (oa, ga), (ob, gb) in zip(trace_a, trace_b):
        assert np.allclose(oa, ob)
        assert np.allclose(ga, gb)
    assert len(out_a) == tasks_a.n_tasks and len(out_b) == tasks_b.n_tasks

