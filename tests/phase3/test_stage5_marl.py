from dataclasses import replace

import numpy as np
import torch

from phase3.config import dev_tier1_6n
from phase3.data.alibaba_loader import build_or_load_curated_trace
from phase3.data.topology import build_edge_topology
from phase3.env import TaskOffloadingSim, prepare_task_table
from phase3.forecasting.gossip_runner import generate_causal_forecast_matrices
from phase3.marl.madqn import MADQNAgent, MaskedDuelingQNet, _NumpyDuelingQRunner
from phase3.marl.mappo import MAPPOAgent, MaskedActorCritic, _NumpyActorRunner


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

