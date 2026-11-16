import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from phase3.config import dev_tier1_6n
from phase3.data.alibaba_loader import build_or_load_curated_trace
from phase3.data.topology import build_edge_topology
from phase3.env import LocalOnlyPolicy, ReactiveGreedyPolicy, TaskOffloadingSim, prepare_task_table


def main() -> None:
    base = dev_tier1_6n(seed=42)
    ds = build_or_load_curated_trace(base.trace, n_nodes=base.topology.n_nodes)
    topo = build_edge_topology(base.trace.porto_cells_parquet, base.topology)

    # Sweep clean calibration grid around node_cpu_cap & kappa to verify Gate 4:
    # B0 must exhibit genuine burst saturation (~15%-35% DMR) and E1 must strictly beat B0!
    calibration_grid = [
        (22.0, 2.2),
        (20.0, 2.2),
        (18.0, 2.2),
    ]

    report = {"n_nodes": ds.n_nodes, "runs": []}
    for cpu_cap, kappa in calibration_grid:
        cfg = replace(
            base,
            sim=replace(base.sim, node_cpu_cap=cpu_cap, deadline_slack_kappa=kappa),
        )
        tasks = prepare_task_table(ds, cfg.sim, split="test", seed=cfg.seed)

        t0 = time.perf_counter()
        sim_b0 = TaskOffloadingSim(ds, topo, tasks, cfg, telemetry_mode="none", verify_invariants=True)
        m_b0 = sim_b0.run_policy(LocalOnlyPolicy())
        dt_b0 = time.perf_counter() - t0

        t1 = time.perf_counter()
        sim_e1 = TaskOffloadingSim(ds, topo, tasks, cfg, telemetry_mode="periodic", verify_invariants=True)
        m_e1 = sim_e1.run_policy(ReactiveGreedyPolicy(cfg.sim))
        dt_e1 = time.perf_counter() - t1

        sim_e1_o = TaskOffloadingSim(ds, topo, tasks, cfg, telemetry_mode="oracle", verify_invariants=True)
        m_e1_o = sim_e1_o.run_policy(ReactiveGreedyPolicy(cfg.sim))

        print(f"\n=== Calibration (cpu_cap={cpu_cap:.1f}, kappa={kappa:.1f}, n_tasks={tasks.n_tasks}) ===")
        for label, m, wall in [("B0 (Local)", m_b0, dt_b0), ("E1 (Greedy-15s)", m_e1, dt_e1), ("E1-Oracle", m_e1_o, 0.0)]:
            print(
                f"  {label:16s} | TSR={m.tsr_pct:6.2f}% (DMR={m.dmr_pct:5.2f}% [adm={m.drop_admission}, dl={m.drop_deadline}, lnk={m.drop_link_timeout}])"
                f" | EffLat={m.effective_latency_mean_s:6.1f}s (P95={m.effective_latency_p95_s:6.1f}s)"
                f" | Offload={m.offload_rate_pct:5.1f}% | Jain={m.jains_fairness:.4f}"
                f" | CB={m.cb_total_bytes/1024.0:6.1f}KB | Reward={m.mean_reward:.3f} | t={wall:.2f}s"
            )
            print(f"    per_node_tsr : {[round(x, 1) for x in m.per_node_tsr_pct]}")
            print(f"    per_node_util: {[round(x, 3) for x in m.per_node_cpu_util]}")

        report["runs"].append(
            {
                "node_cpu_cap": cpu_cap,
                "deadline_slack_kappa": kappa,
                "b0": asdict(m_b0),
                "e1_periodic": asdict(m_e1),
                "e1_oracle": asdict(m_e1_o),
            }
        )

    out_path = Path(__file__).resolve().parent.parent.parent / "results" / "phase3" / "stage4_des_gate_6n.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved Stage 4 Gate report -> {out_path}")


if __name__ == "__main__":
    main()

