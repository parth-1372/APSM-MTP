import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from phase3.config import Phase3Config, dev_tier1_6n
from phase3.data.alibaba_loader import build_or_load_curated_trace
from phase3.data.topology import build_edge_topology
from phase3.env import LocalOnlyPolicy, ReactiveGreedyPolicy, TaskOffloadingSim, prepare_task_table
from phase3.forecasting.gossip_runner import generate_causal_forecast_matrices
from phase3.marl import MADQNAgent, MAPPOAgent


def _aggregate_seed_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [
        "tsr_pct",
        "dmr_pct",
        "drop_admission",
        "drop_deadline",
        "effective_latency_mean_s",
        "effective_latency_p95_s",
        "latency_completed_mean_s",
        "offload_rate_pct",
        "jains_fairness",
        "cb_total_bytes",
        "mean_reward",
    ]
    out: Dict[str, Any] = {"n_seeds": len(runs)}
    for k in keys:
        vals = [float(r["test_metrics"][k]) for r in runs]
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 5 MARL Gate Benchmark (E2, E3, IPPO, MADQN, O1)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--episodes", type=int, default=48)
    parser.add_argument("--window-bins", type=int, default=120)
    parser.add_argument(
        "--agents",
        type=str,
        nargs="+",
        default=["e2_mappo", "e2_ippo", "e2_madqn", "e3_mappo", "e3_madqn", "o1_oracle"],
    )
    args = parser.parse_args()

    t_start_all = time.perf_counter()
    base = dev_tier1_6n(seed=args.seeds[0])
    base = replace(
        base,
        marl=replace(base.marl, n_train_episodes=args.episodes, train_window_bins=args.window_bins),
    )
    ds = build_or_load_curated_trace(base.trace, n_nodes=base.topology.n_nodes)
    topo = build_edge_topology(base.trace.porto_cells_parquet, base.topology)
    test_tasks = prepare_task_table(ds, base.sim, split="test", seed=base.seed)

    # Heuristic deterministic baselines (independent of RL seed)
    m_b0 = TaskOffloadingSim(ds, topo, test_tasks, base, telemetry_mode="none", verify_invariants=True).run_policy(
        LocalOnlyPolicy()
    )
    m_e1 = TaskOffloadingSim(ds, topo, test_tasks, base, telemetry_mode="periodic", verify_invariants=True).run_policy(
        ReactiveGreedyPolicy(base.sim)
    )

    print(
        f"=== Stage 5 Test Suite (N={ds.n_nodes}, tasks={test_tasks.n_tasks}, seeds={args.seeds}, eps={args.episodes}) ==="
    )
    print(
        f"  B0 (LocalOnly)   | TSR={m_b0.tsr_pct:6.2f}% | EffLat={m_b0.effective_latency_mean_s:6.1f}s "
        f"| P95={m_b0.effective_latency_p95_s:6.1f}s | Jain={m_b0.jains_fairness:.4f} | Rew={m_b0.mean_reward:.3f}"
    )
    print(
        f"  E1 (Greedy-15s)  | TSR={m_e1.tsr_pct:6.2f}% | EffLat={m_e1.effective_latency_mean_s:6.1f}s "
        f"| P95={m_e1.effective_latency_p95_s:6.1f}s | Jain={m_e1.jains_fairness:.4f} | Rew={m_e1.mean_reward:.3f}"
    )

    selected = list(args.agents)
    agents_results: Dict[str, List[Dict[str, Any]]] = {k: [] for k in selected}

    need_oracle = "o1_oracle" in selected
    oracle_fc = generate_causal_forecast_matrices("oracle", ds, topo, base) if need_oracle else None

    for seed in args.seeds:
        cfg_s = replace(base, seed=seed)
        need_local = any(k.startswith("e3_") for k in selected)
        if need_local:
            print(f"\n--- Seed {seed}: Generating causal local_lstm forecast matrix ---")
            local_fc = generate_causal_forecast_matrices("local_lstm", ds, topo, cfg_s)
            fmat_l, smat_l = local_fc["forecast_matrix"], local_fc["surprise_matrix"]
        else:
            fmat_l, smat_l = None, None

        if "e2_mappo" in selected:
            agents_results["e2_mappo"].append(
                MAPPOAgent(cfg_s, use_central_critic=True, name="e2_mappo").fit_and_evaluate(
                    ds, topo, telemetry_mode="periodic", forecast_matrix=None, surprise_matrix=None
                )
            )
        if "e2_ippo" in selected:
            agents_results["e2_ippo"].append(
                MAPPOAgent(cfg_s, use_central_critic=False, name="e2_ippo").fit_and_evaluate(
                    ds, topo, telemetry_mode="periodic", forecast_matrix=None, surprise_matrix=None
                )
            )
        if "e2_madqn" in selected:
            agents_results["e2_madqn"].append(
                MADQNAgent(cfg_s, name="e2_madqn").fit_and_evaluate(
                    ds, topo, telemetry_mode="periodic", forecast_matrix=None, surprise_matrix=None
                )
            )
        if "e3_mappo" in selected:
            agents_results["e3_mappo"].append(
                MAPPOAgent(cfg_s, use_central_critic=True, name="e3_mappo").fit_and_evaluate(
                    ds, topo, telemetry_mode="periodic", forecast_matrix=fmat_l, surprise_matrix=smat_l
                )
            )
        if "e3_madqn" in selected:
            agents_results["e3_madqn"].append(
                MADQNAgent(cfg_s, name="e3_madqn").fit_and_evaluate(
                    ds, topo, telemetry_mode="periodic", forecast_matrix=fmat_l, surprise_matrix=smat_l
                )
            )
        if "o1_oracle" in selected and oracle_fc is not None:
            agents_results["o1_oracle"].append(
                MAPPOAgent(
                    cfg_s, use_central_critic=True, actor_uses_global=True, name="o1_oracle"
                ).fit_and_evaluate(
                    ds,
                    topo,
                    telemetry_mode="oracle",
                    forecast_matrix=oracle_fc["forecast_matrix"],
                    surprise_matrix=oracle_fc["surprise_matrix"],
                )
            )

        for k in selected:
            tm = agents_results[k][-1]["test_metrics"]
            print(
                f"  [s={seed}] {k:10s} | TSR={tm['tsr_pct']:6.2f}% | EffLat={tm['effective_latency_mean_s']:5.1f}s "
                f"| P95={tm['effective_latency_p95_s']:5.1f}s | Off={tm['offload_rate_pct']:5.1f}% "
                f"| Jain={tm['jains_fairness']:.4f} | Rew={tm['mean_reward']:.3f} "
                f"| bestEp={agents_results[k][-1]['best_val_ep']} | t={agents_results[k][-1]['wall_time_s']:.1f}s"
            )

    summary = {k: _aggregate_seed_runs(v) for k, v in agents_results.items()}
    print("\n=== Stage 5 Multi-Seed Summary ===")
    for k, s in summary.items():
        print(
            f"  {k:10s} | TSR={s['tsr_pct_mean']:6.2f}%±{s['tsr_pct_std']:4.2f} "
            f"| EffLat={s['effective_latency_mean_s_mean']:5.1f}±{s['effective_latency_mean_s_std']:4.1f}s "
            f"| P95={s['effective_latency_p95_s_mean']:5.1f}s "
            f"| Off={s['offload_rate_pct_mean']:5.1f}% "
            f"| Jain={s['jains_fairness_mean']:.4f} "
            f"| Rew={s['mean_reward_mean']:.3f}±{s['mean_reward_std']:.3f}"
        )

    out_report = {
        "n_nodes": ds.n_nodes,
        "seeds": args.seeds,
        "episodes": args.episodes,
        "train_window_bins": args.window_bins,
        "wall_total_s": float(time.perf_counter() - t_start_all),
        "baselines": {"b0_local": asdict(m_b0), "e1_greedy": asdict(m_e1)},
        "summary": summary,
        "per_seed": agents_results,
    }
    out_path = Path(__file__).resolve().parent.parent.parent / "results" / "phase3" / "stage5_marl_gate_6n.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_report, indent=2), encoding="utf-8")
    print(f"\nSaved Stage 5 Gate report -> {out_path} (total {out_report['wall_total_s']:.1f}s)")


if __name__ == "__main__":
    main()

