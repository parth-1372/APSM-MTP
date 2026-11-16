import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from phase3.config import APSMConfig, ForecasterConfig, Phase3Config, dev_tier1_6n
from phase3.data.alibaba_loader import build_or_load_curated_trace
from phase3.data.topology import build_edge_topology
from phase3.forecasting.gossip_runner import run_stage3_benchmark


def _aggregate_seed_runs(runs: List[Dict]) -> Dict:
    pers_mses = [r["persistence_mse"] for r in runs]
    ema_mses = [r["ema4_mse"] for r in runs]
    summary = {
        "n_seeds": len(runs),
        "persistence_mse_mean": float(np.mean(pers_mses)),
        "ema4_mse_mean": float(np.mean(ema_mses)),
        "ema4_skill_pct": float(np.mean([r["ema4_skill_vs_pers_pct"] for r in runs])),
        "variants": {},
    }
    variant_names = list(runs[0]["variants"].keys())
    for vn in variant_names:
        mses = [r["variants"][vn]["test_mse"] for r in runs]
        skills = [r["variants"][vn]["skill_vs_pers_pct"] for r in runs]
        pkts = [r["variants"][vn]["test_packets_sent"] for r in runs]
        bytes_s = [r["variants"][vn]["test_model_bytes"] for r in runs]
        srs = [r["variants"][vn]["sr_dec_pct"] for r in runs]
        entry = {
            "test_mse_mean": float(np.mean(mses)),
            "test_mse_std": float(np.std(mses)),
            "test_rmse_mean": float(np.mean([r["variants"][vn]["test_rmse"] for r in runs])),
            "skill_vs_pers_pct_mean": float(np.mean(skills)),
            "packets_sent_mean": float(np.mean(pkts)),
            "model_bytes_mean": float(np.mean(bytes_s)),
            "sr_dec_pct_mean": float(np.mean(srs)),
            "total_grad_calls_mean": float(np.mean([r["variants"][vn]["total_grad_calls"] for r in runs])),
            "wall_time_s_mean": float(np.mean([r["variants"][vn]["wall_time_s"] for r in runs])),
        }
        if "pp_avoid_pct" in runs[0]["variants"][vn]:
            entry["pp_avoid_pct_mean"] = float(np.mean([r["variants"][vn]["pp_avoid_pct"] for r in runs]))
            entry["br_avoid_pct_mean"] = float(np.mean([r["variants"][vn]["br_avoid_pct"] for r in runs]))
            entry["mb_avoid_kb_mean"] = float(np.mean([r["variants"][vn]["mb_avoid_bytes"] / 1024.0 for r in runs]))
            entry["mse_delta_vs_gl_pct_mean"] = float(np.mean([r["variants"][vn]["mse_delta_vs_gl_pct"] for r in runs]))
        summary["variants"][vn] = entry
    return summary


def main() -> None:
    seeds = [11, 22, 33]
    base = dev_tier1_6n(seed=11)
    ds = build_or_load_curated_trace(base.trace, n_nodes=base.topology.n_nodes)
    topo = build_edge_topology(base.trace.porto_cells_parquet, base.topology)

    experiments = [
        ("full_hist_p2_std_k1.5", 1.0, "p2_local_head", "std", 1.5),
        ("full_hist_p1_std_k1.5", 1.0, "p1_shared", "std", 1.5),
        ("full_hist_p2_mad_k2.0", 1.0, "p2_local_head", "mad", 2.0),
        ("cold_start_25pct_p2_std_k1.5", 0.25, "p2_local_head", "std", 1.5),
        ("scarce_10pct_p2_std_k1.5", 0.10, "p2_local_head", "std", 1.5),
        ("mixed_2rich_4cold_p2_std_k1.5", [1.0, 1.0, 0.05, 0.05, 0.05, 0.05], "p2_local_head", "std", 1.5),
    ]

    full_report = {}
    for exp_name, hist_frac, pers_mode, stat, k_val in experiments:
        seed_runs = []
        for s in seeds:
            cfg = Phase3Config(
                tier_name=f"stage3_{exp_name}",
                seed=s,
                trace=base.trace,
                topology=base.topology,
                forecaster=ForecasterConfig(personalization=pers_mode),
                apsm=APSMConfig(mode="h1_residual", stat=stat, k=k_val),
                gossip=base.gossip,
            )
            seed_runs.append(run_stage3_benchmark(ds, topo, cfg, pretrain_history_frac=hist_frac))
        agg = _aggregate_seed_runs(seed_runs)
        full_report[exp_name] = {"summary": agg, "seeds": seed_runs}

        print(f"\n=== {exp_name} (3 seeds, N={ds.n_nodes}) ===")
        print(f"  Persistence MSE = {agg['persistence_mse_mean']:.5f} | EMA4 MSE = {agg['ema4_mse_mean']:.5f} ({agg['ema4_skill_pct']:+.2f}%)")
        for vn, vm in agg["variants"].items():
            extra = ""
            if "pp_avoid_pct_mean" in vm:
                extra = (
                    f" | PP_avoid={vm['pp_avoid_pct_mean']:5.1f}%"
                    f" | SR_dec={vm['sr_dec_pct_mean']:5.1f}%"
                    f" | dGL={vm['mse_delta_vs_gl_pct_mean']:+.2f}%"
                    f" | saved={vm['mb_avoid_kb_mean']:.1f}KB"
                )
            print(
                f"  {vn:15s} | MSE={vm['test_mse_mean']:.5f}+/-{vm['test_mse_std']:.5f}"
                f" | vsPers={vm['skill_vs_pers_pct_mean']:+6.2f}%"
                f" | grads={vm['total_grad_calls_mean']:5.0f}"
                f" | pkts={vm['packets_sent_mean']:6.1f}"
                f" | KB={vm['model_bytes_mean']/1024.0:7.1f}{extra}"
            )

    out_path = Path(__file__).resolve().parent.parent.parent / "results" / "phase3" / "stage3_forecast_gate_6n.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(full_report, indent=2), encoding="utf-8")
    print(f"\nSaved full Stage 3 report -> {out_path}")


if __name__ == "__main__":
    main()

