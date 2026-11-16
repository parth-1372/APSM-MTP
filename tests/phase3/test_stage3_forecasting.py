from dataclasses import replace

import numpy as np
import torch

from phase3.config import APSMConfig, dev_tier1_6n
from phase3.data.alibaba_loader import build_or_load_curated_trace
from phase3.data.topology import build_edge_topology
from phase3.forecasting.apsm import OnlineSemanticGate
from phase3.forecasting.gossip_runner import _pretrain_base_models, run_stage3_benchmark
from phase3.forecasting.models import EdgeLSTMForecaster, age_weighted_merge


def test_p2_local_head_isolation_and_bytes():
    m_shared = EdgeLSTMForecaster(personalization="p1_shared")
    m_p2 = EdgeLSTMForecaster(personalization="p2_local_head")
    assert m_p2.shared_num_params() < m_p2.total_num_params()
    assert m_p2.serialized_payload_bytes() < m_shared.serialized_payload_bytes()

    # Modifying peer head weights must NEVER overwrite local P2 head weights
    peer_shared = m_p2.export_shared_weights()
    assert "head.weight" not in peer_shared and "head.bias" not in peer_shared
    orig_head_w = m_p2.head.weight.detach().clone()
    zero_shared = {k: torch.zeros_like(v) for k, v in peer_shared.items()}
    m_p2.load_shared_weights(zero_shared)
    assert torch.allclose(m_p2.head.weight, orig_head_w)


def test_age_weighted_merge_fixes_zero_age_bug():
    w_local = {"a": torch.tensor([1.0, 3.0])}
    w_peer = {"a": torch.tensor([5.0, 7.0])}
    merged, new_age = age_weighted_merge(w_local, 0.0, [(w_peer, 0.0)], age_decay=0.05)
    # With B8 fixed (floored weight >= 1.0), exact 50/50 average of [1,3] and [5,7] = [3,5]
    assert torch.allclose(merged["a"], torch.tensor([3.0, 5.0]))
    assert new_age >= 1.0


def test_age_weighted_merge_self_mix_alpha():
    w_local = {"a": torch.tensor([10.0])}
    w_p1 = {"a": torch.tensor([0.0])}
    w_p2 = {"a": torch.tensor([0.0])}
    # With self_mix_alpha=0.70, peers share 0.30 total weight regardless of node degree
    merged, new_age = age_weighted_merge(
        w_local, 10.0, [(w_p1, 10.0), (w_p2, 10.0)], age_decay=0.05, self_mix_alpha=0.70
    )
    assert torch.allclose(merged["a"], torch.tensor([7.0]))
    assert new_age == 9.5

    # Empty peer list preserves local tensor safely
    merged_empty, age_empty = age_weighted_merge(w_local, 0.0, [], age_decay=0.05, self_mix_alpha=0.70)
    assert torch.allclose(merged_empty["a"], torch.tensor([10.0]))
    assert age_empty == 1.0


def test_common_init_zero_pretrain_identical_across_nodes():
    cfg = dev_tier1_6n(seed=13)
    cfg = replace(cfg, forecaster=replace(cfg.forecaster, pretrain_epochs=0, common_init=True))
    ds = build_or_load_curated_trace(cfg.trace, n_nodes=6)
    base_states, _ = _pretrain_base_models(ds, cfg, pretrain_history_frac=1.0)
    for k in base_states[0].keys():
        for i in range(1, 6):
            assert torch.allclose(base_states[0][k], base_states[i][k])


def test_apsm_h1_warmup_and_heartbeat_invariants():
    cfg = APSMConfig(mode="h1_residual", k=1.5, window=20, min_obs=10, heartbeat=4)
    gate = OnlineSemanticGate(cfg)

    # First 9 steps must fail open with finite or infinite tau without raising spike artifacts
    for step in range(9):
        eps, tau, trig = gate.record_causal_residual(0.5 + 0.01 * step, 0.5)
        assert not trig
        assert np.isinf(tau)
        dec = gate.evaluate_gossip_send()
        assert dec.should_send and dec.reason == "warmup_fail_open"

    # Populate steady low variance -> calm steps should suppress until heartbeat hits 4
    for _ in range(10):
        gate.record_causal_residual(0.50, 0.50)

    reasons = []
    for _ in range(6):
        gate.record_causal_residual(0.50, 0.50)
        dec = gate.evaluate_gossip_send()
        reasons.append((dec.should_send, dec.reason))

    # Expect at least 4 suppressed followed by mandatory heartbeat
    assert (True, "heartbeat") in reasons

    # Sudden drift spike must immediately trigger broadcast
    _, _, trig = gate.record_causal_residual(1.40, 0.45)
    assert trig
    dec = gate.evaluate_gossip_send()
    assert dec.should_send and dec.reason == "surprise_trigger"


def test_stage3_equal_cadence_and_finite_metrics():
    cfg = dev_tier1_6n(seed=7)
    # Lightweight epochs just for fast invariant unit check
    cfg = replace(cfg, forecaster=replace(cfg.forecaster, pretrain_epochs=5))
    ds = build_or_load_curated_trace(cfg.trace, n_nodes=6)
    topo = build_edge_topology(cfg.trace.porto_cells_parquet, cfg.topology)
    out = run_stage3_benchmark(ds, topo, cfg, pretrain_history_frac=[1.0, 1.0, 0.1, 0.1, 0.1, 0.1])

    v = out["variants"]
    # Equal cadence check (fixing B6!): GL and APSM execute EXACT same total_gossip_checks
    assert v["gl_baseline"]["test_gossip_checks"] == v["apsm_h1"]["test_gossip_checks"]
    assert v["gl_baseline"]["test_gossip_checks"] == v["apsm_val_delta"]["test_gossip_checks"]
    assert v["gl_baseline"]["test_suppressed_checks"] == 0
    assert v["apsm_h1"]["test_packets_sent"] <= v["gl_baseline"]["test_packets_sent"]
    assert np.isfinite(v["apsm_h1"]["test_mse"])
    assert v["apsm_h1"]["total_grad_calls"] > 0
    assert v["local_lstm"]["total_grad_calls"] > 0

