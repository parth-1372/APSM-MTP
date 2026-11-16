import json
from pathlib import Path
import numpy as np

base = Path("results/phase3")
n10_runs = json.loads((base / "stage6_integrated_10n_reconciled.json").read_text(encoding="utf-8"))
n50_summary = json.loads((base / "stage7_scalability/n50/summary.json").read_text(encoding="utf-8"))
n100_summary = json.loads((base / "stage7_scalability/n100/summary.json").read_text(encoding="utf-8"))
n50_res = json.loads((base / "stage7_scalability/n50/resource_usage.json").read_text(encoding="utf-8"))
n100_res = json.loads((base / "stage7_scalability/n100/resource_usage.json").read_text(encoding="utf-8"))

arms = ["b0_local", "e1_greedy", "e2_madqn", "e3_madqn_local", "e3_madqn_gl", "e3_madqn_apsm"]

print("="*80)
print("COMPREHENSIVE MULTI-SCALE EXACT METRICS SUMMARY")
print("="*80)

for scale, runs_data in [(10, n10_runs)]:
    print(f"\n--- N={scale} ---")
    for arm in arms:
        aruns = [r for r in runs_data if r["arm"] == arm]
        tsrs = [r["tsr_pct"] for r in aruns]
        lats = [r["effective_latency_mean_s"] for r in aruns]
        p95s = [r["effective_latency_p95_s"] for r in aruns]
        offs = [r["offload_rate_pct"] for r in aruns]
        drops = [r["dmr_pct"] for r in aruns]
        m_bytes = [r.get("model_bytes", 0) for r in aruns]
        ctrl_b = [r.get("cb_control_bytes", 0) for r in aruns]
        fcs = [r["fc_mse"] for r in aruns if r.get("fc_mse") is not None]
        print(f"  {arm:<16} | TSR: {np.mean(tsrs):.2f} +- {np.std(tsrs, ddof=1):.2f}% | Lat: {np.mean(lats):.2f} +- {np.std(lats, ddof=1):.2f}s (P95: {np.mean(p95s):.2f}s) | Off: {np.mean(offs):.2f}% | Drop: {np.mean(drops):.2f}% | ModelMB: {np.mean(m_bytes)/1e6:.2f} | CtrlMB: {np.mean(ctrl_b)/1e6:.2f} | MSE: {np.mean(fcs):.6f}" if fcs else f"  {arm:<16} | TSR: {np.mean(tsrs):.2f} +- {np.std(tsrs, ddof=1):.2f}% | Lat: {np.mean(lats):.2f} +- {np.std(lats, ddof=1):.2f}s (P95: {np.mean(p95s):.2f}s) | Off: {np.mean(offs):.2f}% | Drop: {np.mean(drops):.2f}% | ModelMB: {np.mean(m_bytes)/1e6:.2f} | CtrlMB: {np.mean(ctrl_b)/1e6:.2f}")

    gl_r = [r for r in runs_data if r["arm"] == "e3_madqn_gl"]
    apsm_r = [r for r in runs_data if r["arm"] == "e3_madqn_apsm"]
    m_red = np.mean([r["model_byte_reduction_pct"] for r in apsm_r if r.get("model_byte_reduction_pct") is not None])
    p_red = np.mean([r["model_packet_reduction_pct"] for r in apsm_r if r.get("model_packet_reduction_pct") is not None])
    c_red = np.mean([r["control_byte_reduction_pct"] for r in apsm_r if r.get("control_byte_reduction_pct") is not None])
    s_chk = np.mean([r["scheduled_check_suppression_pct"] for r in apsm_r if r.get("scheduled_check_suppression_pct") is not None])
    print(f"  APSM Reductions: ModelBytes: {m_red:.2f}%, ModelPkts: {p_red:.2f}%, TotalCtrlBytes: {c_red:.2f}%, SuppChecks: {s_chk:.2f}%")

for scale, summary, res in [(50, n50_summary, n50_res), (100, n100_summary, n100_res)]:
    print(f"\n--- N={scale} ---")
    for arm in arms:
        d = summary["arms"][arm]
        f_str = f" | MSE: {d['fc_mse_mean']:.6f} +- {d['fc_mse_std']:.6f}" if d.get("fc_mse_mean") is not None else ""
        print(f"  {arm:<16} | TSR: {d['tsr_mean']:.2f} +- {d['tsr_std']:.2f}% | Lat: {d['effective_latency_mean']:.2f}s (P95: {d['effective_latency_p95']:.2f}s) | Off: {d['offload_rate_mean']:.2f}% | Drop: {d['drop_rate_mean']:.2f}% | ModelMB: {d['model_bytes_mean']/1e6:.2f} | CtrlMB: {d['control_bytes_mean']/1e6:.2f}{f_str}")
    apsm_d = summary["arms"]["e3_madqn_apsm"]
    print(f"  APSM Reductions: ModelBytes: {apsm_d['model_byte_reduction_pct_mean']:.2f}%, ModelPkts: {apsm_d['model_packet_reduction_pct_mean']:.2f}%, TotalCtrlBytes: {apsm_d['control_byte_reduction_pct_mean']:.2f}%, SuppChecks: {apsm_d['scheduled_check_suppression_pct_mean']:.2f}%")
    print(f"  Resources: Peak RAM: {res['peak_ram_mb']:.1f} MB | Peak GPU Alloc: {res['peak_gpu_allocated_mb']:.1f} MB | Peak GPU Res: {res['peak_gpu_reserved_mb']:.1f} MB | Mean Wall Clock: {res['total_wall_clock_s']/30:.1f}s")
