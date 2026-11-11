from dataclasses import replace

import numpy as np

from phase3.config import dev_tier1_6n
from phase3.data.alibaba_loader import build_or_load_curated_trace
from phase3.data.topology import build_edge_topology
from phase3.env import LocalOnlyPolicy, ReactiveGreedyPolicy, TaskOffloadingSim, prepare_task_table


def test_stage4_anti_loop_conservation_and_byte_audit():
    cfg = dev_tier1_6n(seed=42)
    ds = build_or_load_curated_trace(cfg.trace, n_nodes=6)
    topo = build_edge_topology(cfg.trace.porto_cells_parquet, cfg.topology)
    # Use a dense 30-bin sub-slice for fast rigorous invariant verification
    tasks = prepare_task_table(ds, cfg.sim, start_bin=ds.val_end_bin, end_bin=ds.val_end_bin + 30)

    # Adversarial policy that ALWAYS tries to offload to neighbor slot 1 whenever connected!
    # Verifies max_hops=1 anti-loop invariant: offloaded tasks enter peer queue directly and NEVER bounce again.
    def always_offload_first_neighbor(obs, mask, node_id, t_now_s, task_idx):
        assert len(mask) == 1 + cfg.sim.k_max_neighbors
        # Unconnected neighbor slots MUST be masked False and observation zeroed
        for k in range(cfg.sim.k_max_neighbors):
            if not mask[1 + k]:
                base = 11 + 8 * k
                assert np.all(obs[base : base + 8] == 0.0)
        return 1 if bool(mask[1]) else 0

    sim = TaskOffloadingSim(
        ds,
        topo,
        tasks,
        cfg,
        telemetry_mode="periodic",
        extra_model_packets=10,
        extra_model_bytes=569200,
        verify_invariants=True,
    )
    m = sim.run_policy(always_offload_first_neighbor)

    assert m.total_tasks == tasks.n_tasks
    assert m.completed_on_time + m.dropped_tasks == m.total_tasks
    # Every task arrived at source and hopped at most ONCE
    assert m.offload_count == m.total_tasks
    assert m.telemetry_bytes == m.telemetry_packets * cfg.sim.telemetry_beacon_bytes
    assert m.ack_bytes == m.ack_packets * cfg.sim.ack_bytes
    assert m.cb_total_bytes == m.model_bytes + m.telemetry_bytes + m.ack_bytes
    assert m.offload_data_bytes == int(tasks.payload_bytes.sum()) + m.total_tasks * cfg.sim.task_header_bytes


def test_stage4_strict_causality_and_non_oracle_staleness():
    cfg = dev_tier1_6n(seed=42)
    ds = build_or_load_curated_trace(cfg.trace, n_nodes=6)
    topo = build_edge_topology(cfg.trace.porto_cells_parquet, cfg.topology)
    tasks = prepare_task_table(ds, cfg.sim, start_bin=ds.val_end_bin, end_bin=ds.val_end_bin + 10)

    # Modify FUTURE tasks strictly after bin val_end_bin+5; verify decisions & obs in first 5 bins are IDENTICAL!
    tasks_future_mut = prepare_task_table(ds, cfg.sim, start_bin=ds.val_end_bin, end_bin=ds.val_end_bin + 10)
    cut_ts = tasks.t_ep_start_s + 5 * ds.dt_bin_s
    fut_mask = tasks_future_mut.arr_ts_s >= cut_ts
    tasks_future_mut.cpu_work[fut_mask] *= 10.0

    obs_log_a = []
    obs_log_b = []

    def rec_policy(log):
        def _fn(obs, mask, node_id, t_now_s, task_idx):
            if t_now_s < cut_ts:
                # Non-oracle telemetry cache must carry positive info_age after t > t_ep_start + 0
                if t_now_s > tasks.t_ep_start_s + 0.5:
                    for k in range(cfg.sim.k_max_neighbors):
                        if mask[1 + k]:
                            assert obs[11 + 8 * k + 7] > 0.0
                log.append(obs.copy())
            return 0
        return _fn

    TaskOffloadingSim(ds, topo, tasks, cfg, telemetry_mode="periodic", verify_invariants=True).run_policy(rec_policy(obs_log_a))
    TaskOffloadingSim(ds, topo, tasks_future_mut, cfg, telemetry_mode="periodic", verify_invariants=True).run_policy(rec_policy(obs_log_b))

    assert len(obs_log_a) == len(obs_log_b) and len(obs_log_a) > 0
    for oa, ob in zip(obs_log_a, obs_log_b):
        assert np.allclose(oa, ob)


def test_stage4_apsm_t_gating_and_e1_beats_b0():
    cfg = dev_tier1_6n(seed=42)
    ds = build_or_load_curated_trace(cfg.trace, n_nodes=6)
    topo = build_edge_topology(cfg.trace.porto_cells_parquet, cfg.topology)
    tasks = prepare_task_table(ds, cfg.sim, start_bin=ds.val_end_bin, end_bin=ds.val_end_bin + 150)

    m_b0 = TaskOffloadingSim(ds, topo, tasks, cfg, telemetry_mode="none", verify_invariants=True).run_policy(LocalOnlyPolicy())
    m_e1 = TaskOffloadingSim(ds, topo, tasks, cfg, telemetry_mode="periodic", verify_invariants=True).run_policy(
        ReactiveGreedyPolicy(cfg.sim)
    )
    m_e1_t = TaskOffloadingSim(ds, topo, tasks, cfg, telemetry_mode="apsm_t", verify_invariants=True).run_policy(
        ReactiveGreedyPolicy(cfg.sim)
    )

    assert m_e1.tsr_pct > m_b0.tsr_pct
    assert m_e1.jains_fairness > m_b0.jains_fairness
    assert m_e1_t.telemetry_bytes < m_e1.telemetry_bytes
    assert m_e1_t.telemetry_suppressed > 0

