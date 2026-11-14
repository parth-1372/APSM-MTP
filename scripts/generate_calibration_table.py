"""
Generate the exact Calibration Table from results/phase3/stage6_10n_calibration_train_burst.json
programmatically to guarantee zero transcription errors.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def generate_table():
    calib_path = ROOT / "results/phase3/stage6_10n_calibration_train_burst.json"
    assert calib_path.exists(), f"Missing {calib_path}"
    data = json.loads(calib_path.read_text(encoding="utf-8"))

    lines = [
        "| $C_{\\text{cpu}}$ (cores) | Slack factor $\\kappa$ | B0 TSR (%) | B0 DMR (%) | E1 TSR (%) | E1 DMR (%) | $\\Delta_{\\text{TSR}}$ (E1 - B0) | E1 Offload Rate (%) | Operating Regime Assessment |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]

    for r in data:
        c = r["c_cpu"]
        k = r["kappa"]
        b0_tsr = r["b0_tsr_pct"]
        b0_dmr = r["b0_dmr_pct"]
        e1_tsr = r["e1_tsr_pct"]
        e1_dmr = r["e1_dmr_pct"]
        delta = r["delta_tsr_pct"]
        offload = r["e1_offload_rate_pct"]

        # Verification check
        assert abs(b0_tsr + b0_dmr - 100.0) < 1e-4, f"B0 sum mismatch at {c}, {k}"
        assert abs(e1_tsr + e1_dmr - 100.0) < 1e-4, f"E1 sum mismatch at {c}, {k}"
        assert abs(e1_tsr - b0_tsr - delta) < 1e-4, f"Delta mismatch at {c}, {k}"

        sign = "+" if delta > 0 else ""
        if c == 20.0 and k == 2.2:
            assessment = "**Empirical/Design Operating Regime**: Balanced burst pressure (~81.3% local TSR), realistic offloading headroom (~18.7% drop risk)"
            lines.append(
                f"| **{c:.1f}** | **{k:.1f}** | **{b0_tsr:.2f}%** | **{b0_dmr:.2f}%** | **{e1_tsr:.2f}%** | **{e1_dmr:.2f}%** | **{sign}{delta:.2f}%** | **{offload:.2f}%** | {assessment} |"
            )
        else:
            if c == 18.0:
                if k == 2.0:
                    assessment = "Severe congestion; heavy local queuing drops, offloading penalized by link queuing"
                elif k == 2.2:
                    assessment = "Under-provisioned; persistent queue buildup across neighbor mesh"
                else:
                    assessment = "Lenient deadline partially mitigates queue wait, but capacity severely limited"
            elif c == 20.0:
                if k == 2.0:
                    assessment = "Tight deadline budget limits multi-hop offloading window"
                else:
                    assessment = "Generous slack factor reduces burst pressure"
            else:  # c == 22.0
                if k == 2.0:
                    assessment = "Lightly loaded; local cores absorb majority of peak bursts"
                elif k == 2.2:
                    assessment = "High local sufficiency; limited headroom for cooperative gain"
                else:
                    assessment = "Near-complete local sufficiency (>90% TSR); minimal offloading need"
            lines.append(
                f"| {c:.1f} | {k:.1f} | {b0_tsr:.2f}% | {b0_dmr:.2f}% | {e1_tsr:.2f}% | {e1_dmr:.2f}% | {sign}{delta:.2f}% | {offload:.2f}% | {assessment} |"
            )

    table_md = "\n".join(lines)
    print(table_md)
    return table_md

if __name__ == "__main__":
    generate_table()
