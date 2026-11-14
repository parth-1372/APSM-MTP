import hashlib
import json
from pathlib import Path

def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    root = Path(__file__).resolve().parent.parent
    files_to_hash = [
        "results/phase3/stage6_integrated_10n_reconciled.json",
        "results/phase3/stage6_10n_calibration_train_burst.json",
        "results/phase3/stage6_10n_forecasting_aligned.json",
        "results/phase3/stage6_10n_marl_eval_exact.json",
        "results/phase3/stage6_final_reconciliation_test_log.txt",
        "scripts/calibrate_10n_train.py",
        "scripts/generate_calibration_table.py",
        "scripts/reevaluate_forecasting_metrics.py",
        "scripts/eval_marl_forecasting_exact.py",
        "scripts/reconcile_results_json.py",
        "scripts/generate_reconciled_manifest.py",
        "src/phase3/integration/live_gossip_engine.py",
        "src/phase3/analyze_stage6_results.py",
        "tests/phase3/test_stage6_integration.py",
        "docs/phase3_stage6_final_reconciliation/FINAL_SOURCE_VS_DOCUMENTATION_AUDIT.md",
        "docs/phase3_stage6_final_reconciliation/CORRECTED_DATA_AND_METHODOLOGY.md",
        "docs/phase3_stage6_final_reconciliation/CORRECTED_FORECAST_EVALUATION.md",
        "docs/phase3_stage6_final_reconciliation/CORRECTED_STATISTICAL_ANALYSIS.md",
        "docs/phase3_stage6_final_reconciliation/FINAL_RESEARCH_CONCLUSION.md",
        "docs/phase3_stage6_final_reconciliation/REPRODUCIBILITY_AND_TEST_EVIDENCE.md",
    ]

    manifest = {
        "benchmark_milestone": "stage6_integrated_10n_final_reconciliation",
        "base_commit": "5d0dec71dd2061b10d1c3e64a67f81050133a9e2",
        "branch": "stage6-final-reconcile-10n",
        "scope": "N=10 nodes only",
        "artifacts": {}
    }

    for rel_path in files_to_hash:
        p = root / rel_path
        if p.exists():
            manifest["artifacts"][rel_path] = {
                "sha256": compute_sha256(p),
                "bytes": p.stat().st_size,
            }
        else:
            print(f"Warning: {rel_path} not found!")

    out_file = root / "results/phase3/stage6_10n_reconciled_manifest.json"
    with open(out_file, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote manifest with {len(manifest['artifacts'])} artifacts to {out_file}!")

if __name__ == "__main__":
    main()
