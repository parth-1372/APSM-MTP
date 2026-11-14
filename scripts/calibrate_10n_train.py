"""
Reproducible 9-Point Parameter Calibration Sweep on Training Burst Slice.

Strict Data Split Discipline:
  - Calibration Window: Bins [380, 488) (10,678 tasks) from training partition [0, 828).
  - ZERO test split leakage (test partition is strictly [1035, 1380)).
  - Grid: C_cpu in {18.0, 20.0, 22.0} MIPS x kappa in {2.0, 2.2, 2.5}.
"""

import json
from pathlib import Path
import sys
import time
from dataclasses import replace
from typing import Any, Dict, List

# Ensure src is in python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from phase3.config import Phase3Config
from phase3.data.alibaba_loader import build_or_load_curated_trace
from phase3.data.topology import build_edge_topology
from phase3.env.baselines import LocalOnlyPolicy, ReactiveGreedyPolicy
from phase3.env.task_simulator import prepare_task_table, TaskOffloadingSim


def run_calibration_sweep(output_json: str = "results/phase3/stage6_10n_calibration_train_burst.json") -> List[Dict[str, Any]]:
    cfg = Phase3Config()
    topo_cfg = replace(cfg.topology, n_nodes=10, k_edge_connectivity=3)
    ds = build_or_load_curated_trace(cfg.trace, n_nodes=10)
    topo = build_edge_topology(cfg.trace.porto_cells_parquet, topo_cfg)

    # Strictly inside training split: bins [380, 488) (MARLConfig.burst_val_start_bin)
    # Zero validation or test split contamination
    b_start, b_end = 380, 488
    tbl_base = prepare_task_table(ds, cfg.sim, start_bin=b_start, end_bin=b_end, seed=42)
    print(f"Calibration Slice: bins [{b_start}, {b_end}), tasks={tbl_base.n_tasks}", flush=True)

    cpu_caps = [18.0, 20.0, 22.0]
    kappas = [2.0, 2.2, 2.5]

    print("\n" + "=" * 88, flush=True)
    print(f"{'C_cpu':>6} | {'kappa':>6} | {'B0 TSR':>8} | {'B0 DMR':>8} | {'E1 TSR':>8} | {'E1 DMR':>8} | {'Delta':>8} | {'E1 Off%':>8}", flush=True)
    print("=" * 88, flush=True)

    results = []
    t_start = time.perf_counter()
    for c_cpu in cpu_caps:
        for kappa in kappas:
            sim_cfg = replace(cfg.sim, node_cpu_cap=c_cpu, deadline_slack_kappa=kappa, q_max=48)
            cfg_sweep = replace(cfg, sim=sim_cfg)
            tbl = prepare_task_table(ds, sim_cfg, start_bin=b_start, end_bin=b_end, seed=42)

            sim_b0 = TaskOffloadingSim(ds, topo, tbl, cfg_sweep, telemetry_mode="none")
            m_b0 = sim_b0.run_policy(LocalOnlyPolicy())

            sim_e1 = TaskOffloadingSim(ds, topo, tbl, cfg_sweep, telemetry_mode="periodic")
            m_e1 = sim_e1.run_policy(ReactiveGreedyPolicy(sim_cfg))

            delta = float(m_e1.tsr_pct - m_b0.tsr_pct)
            print(
                f"{c_cpu:6.1f} | {kappa:6.1f} | {m_b0.tsr_pct:7.2f}% | {m_b0.dmr_pct:7.2f}% | "
                f"{m_e1.tsr_pct:7.2f}% | {m_e1.dmr_pct:7.2f}% | {delta:+7.2f}% | {m_e1.offload_rate_pct:7.1f}%",
                flush=True,
            )
            results.append({
                "c_cpu": float(c_cpu),
                "kappa": float(kappa),
                "n_tasks": int(tbl.n_tasks),
                "start_bin": b_start,
                "end_bin": b_end,
                "b0_tsr_pct": float(m_b0.tsr_pct),
                "b0_dmr_pct": float(m_b0.dmr_pct),
                "b0_completed": int(m_b0.completed_on_time),
                "b0_dropped": int(m_b0.dropped_tasks),
                "e1_tsr_pct": float(m_e1.tsr_pct),
                "e1_dmr_pct": float(m_e1.dmr_pct),
                "e1_completed": int(m_e1.completed_on_time),
                "e1_dropped": int(m_e1.dropped_tasks),
                "delta_tsr_pct": delta,
                "e1_offload_rate_pct": float(m_e1.offload_rate_pct),
            })

    print("=" * 88, flush=True)
    dur = time.perf_counter() - t_start
    print(f"Sweep completed in {dur:.2f}s", flush=True)

    p_opt = [r for r in results if r["c_cpu"] == 20.0 and r["kappa"] == 2.2][0]
    print(f"\nOperating point (C_cpu=20.0, kappa=2.2) on Training Burst slice:")
    print(f"  B0 TSR: {p_opt['b0_tsr_pct']:.2f}%, E1 TSR: {p_opt['e1_tsr_pct']:.2f}%, Delta: {p_opt['delta_tsr_pct']:+.2f}%")
    print("  ZERO TEST SPLIT LEAKAGE: All 10,678 tasks drawn from bins [380, 488) < train_end_bin=828.")

    out_file = ROOT / output_json
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved complete 9-point calibration results to {out_file.as_posix()}!")
    return results


if __name__ == "__main__":
    run_calibration_sweep()
