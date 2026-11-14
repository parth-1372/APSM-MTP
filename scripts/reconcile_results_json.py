"""
Strict Standards-Compliant Reconciliation for Stage 6 N=10 Results JSON.

Rules:
  - Exact seed association: preserves per-seed variance.
  - No copying of mean aggregates into individual seed records.
  - Strictly disaggregated suppression metrics computed per seed against matching GL seed.
  - Non-applicable metrics set to JSON null (never 0.0 or NaN).
  - Validated using strict custom parser disallowing NaN/Inf constants.
  - Preserves immutable historical audited-v2 artifact untouched.
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent


def sanitize_val(v: Any) -> Any:
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    elif isinstance(v, dict):
        return {k: sanitize_val(val) for k, val in v.items()}
    elif isinstance(v, list):
        return [sanitize_val(val) for val in v]
    return v


def strict_json_check(text: str) -> None:
    def reject_constant(c):
        raise ValueError(f"Strict JSON violation: non-standard constant '{c}' encountered.")
    json.loads(text, parse_constant=reject_constant)


def reconcile():
    v2_path = ROOT / "results/phase3/stage6_integrated_10n_audited_v2.json"
    exact_marl_path = ROOT / "results/phase3/stage6_10n_marl_eval_exact.json"
    out_path = ROOT / "results/phase3/stage6_integrated_10n_reconciled.json"

    assert v2_path.exists(), f"Missing {v2_path}"
    assert exact_marl_path.exists(), f"Missing {exact_marl_path}"

    v2_records: List[Dict[str, Any]] = json.loads(v2_path.read_text(encoding="utf-8"))
    exact_marl: Dict[str, List[Dict[str, Any]]] = json.loads(exact_marl_path.read_text(encoding="utf-8"))

    # Map exact marl by (arm, seed)
    exact_fc_by_arm_seed: Dict[tuple, Dict[str, Any]] = {}
    for arm, items in exact_marl.items():
        for r in items:
            exact_fc_by_arm_seed[(arm, r["seed"])] = r

    # Map GL reference metrics by seed for paired suppression calculation
    gl_by_seed: Dict[int, Dict[str, Any]] = {}
    for r in v2_records:
        if r["arm"] == "e3_madqn_gl":
            gl_by_seed[r["seed"]] = r

    reconciled_records = []

    for r in v2_records:
        rec = dict(r)
        arm = rec["arm"]
        seed = rec["seed"]

        # Update forecasting metrics with exact per-seed aligned values
        if arm in ("e3_madqn_local", "e3_madqn_gl", "e3_madqn_apsm"):
            aligned = exact_fc_by_arm_seed.get((arm, seed))
            if aligned is not None:
                rec["fc_mse"] = aligned["fc_mse"]
                rec["fc_rmse"] = aligned["fc_rmse"]
                rec["fc_mae"] = aligned["fc_mae"]
                rec["fc_pers_mse"] = aligned["fc_pers_mse"]
            else:
                raise ValueError(f"Missing aligned forecasting evaluation for {arm} seed {seed}")
        else:
            rec["fc_mse"] = None
            rec["fc_rmse"] = None
            rec["fc_mae"] = None
            rec["fc_pers_mse"] = None

        # Compute paired disaggregated suppression metrics
        if arm == "e3_madqn_apsm":
            gl_ref = gl_by_seed.get(seed)
            assert gl_ref is not None, f"Missing GL reference for seed {seed}"

            # 1. Scheduled check suppression
            tot_checks = rec.get("total_gossip_checks", 0)
            supp_checks = rec.get("suppressed_checks", 0)
            rec["scheduled_check_suppression_pct"] = (
                (supp_checks / tot_checks * 100.0) if tot_checks > 0 else None
            )

            # 2. Model packet reduction (paired vs matching GL seed)
            gl_pkts = gl_ref.get("model_packets", 0)
            apsm_pkts = rec.get("model_packets", 0)
            rec["model_packet_reduction_pct"] = (
                ((1.0 - apsm_pkts / gl_pkts) * 100.0) if gl_pkts > 0 else None
            )

            # 3. Model byte reduction (paired vs matching GL seed)
            gl_bytes = gl_ref.get("model_bytes", 0)
            apsm_bytes = rec.get("model_bytes", 0)
            rec["model_byte_reduction_pct"] = (
                ((1.0 - apsm_bytes / gl_bytes) * 100.0) if gl_bytes > 0 else None
            )

            # 4. Control byte reduction (paired vs matching GL seed)
            gl_ctrl = gl_ref.get("cb_control_bytes", 0)
            apsm_ctrl = rec.get("cb_control_bytes", 0)
            rec["control_byte_reduction_pct"] = (
                ((1.0 - apsm_ctrl / gl_ctrl) * 100.0) if gl_ctrl > 0 else None
            )

        elif arm == "e3_madqn_gl":
            rec["scheduled_check_suppression_pct"] = 0.0
            rec["model_packet_reduction_pct"] = 0.0
            rec["model_byte_reduction_pct"] = 0.0
            rec["control_byte_reduction_pct"] = 0.0
        else:
            rec["scheduled_check_suppression_pct"] = None
            rec["model_packet_reduction_pct"] = None
            rec["model_byte_reduction_pct"] = None
            rec["control_byte_reduction_pct"] = None

        # Sanitize all values
        rec_clean = sanitize_val(rec)
        reconciled_records.append(rec_clean)

    assert len(reconciled_records) == 30, f"Expected 30 records, got {len(reconciled_records)}"

    # Convert to JSON and strictly validate
    json_text = json.dumps(reconciled_records, indent=2)
    strict_json_check(json_text)

    # Atomic write
    tmp_path = out_path.with_suffix(".tmp")
    tmp_path.write_text(json_text, encoding="utf-8")
    tmp_path.replace(out_path)

    print(f"Successfully wrote {len(reconciled_records)} strictly validated records to {out_path}!")


if __name__ == "__main__":
    reconcile()
