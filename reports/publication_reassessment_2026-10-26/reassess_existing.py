"""Post hoc audit of existing transfer results. No model inference or primary-output edits."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/g0"))
import phase1_transfer_analysis as analysis

HERE = Path(__file__).resolve().parent
RAW = ROOT.parent / "transfer_runs/analysis_main"
COMMITTED = ROOT / "reports/intervention_transfer/results/main"


def load_rows(path):
    rows = {}
    integers = {"feature", "sentence", "cluster", "language_index",
                "n_rows", "n_positions", "n_active_q"}
    with path.open() as handle:
        for raw in csv.DictReader(handle):
            row = {}
            for key, value in raw.items():
                if key in ("kind", "split"):
                    row[key] = value
                elif key == "eligible":
                    row[key] = value == "True"
                elif not value:
                    row[key] = None
                elif key in integers:
                    row[key] = int(float(value))
                else:
                    row[key] = float(value)
            rows[row["feature"], row["sentence"], row["kind"]] = row
    return rows


def main():
    locked = {"status": "original FLORES calibration fits, exported after main outcomes; no external outcomes used by this script",
              "target": "error_norm / (fp_norm + FLORES floor for direction kind)",
              "eligibility": "fp_norm > 5 * FLORES real-direction floor",
              "models": {}, "code_sha256": hashlib.sha256((ROOT / "scripts/g0/phase1_transfer_analysis.py").read_bytes()).hexdigest()}
    result = {"status": "post hoc descriptive audit; no new uncertainty intervals",
              "states": {}, "inputs_sha256": {},
              "limits": [
                  "Unseen-document restriction was chosen after main outcomes; not confirmatory.",
                  "The strict subset has only 65 documents and 136 eligible items.",
                  "Joint eligibility does not equate FP effect norm or intervention strength.",
                  "No original fit, outcome, success rule or freeze is overwritten."]}
    reference = list(csv.DictReader((COMMITTED / "predictive_validation.csv").open()))
    for state in ("INT16", "INT6", "INT4"):
        path = RAW / f"per_context_{state}.csv"
        result["inputs_sha256"][str(path.relative_to(ROOT.parent))] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = load_rows(path)
        design = analysis.design(rows)
        calibration = design["split"] == "calibration"
        held = design["split"] == "heldout"
        fits = analysis._fit_models(design, analysis.MODEL_GROUPS,
                                    analysis.RIDGE_STRENGTHS, analysis.N_FOLDS)
        cal_documents = {r["cluster"] for r in rows.values()
                         if r["kind"] == "real" and r["split"] == "calibration"}
        new_document = np.array([c not in cal_documents
                                 for c in design["cluster"][held]])
        strongest = min(analysis.BASELINES, key=lambda name: fits[name]["cv_mae"])
        locked["models"][state] = {
            "calibration_selected_baseline": strongest,
            "null_calibration_median": float(np.median(design["y"][calibration])),
            "floor_by_kind": {r["kind"]: r["floor"] for r in rows.values()},
            "groups": {name: list(groups) for name, groups in analysis.MODEL_GROUPS.items()},
            "column_order": design["columns"],
            "fits": {name: {key: fit[key] for key in ("mean", "scale", "coef", "intercept", "lambda")}
                     for name, fit in fits.items()}}
        means = {}
        for name, fit in fits.items():
            expected = next(float(r["heldout_mae"]) for r in reference
                            if r["state"] == state and r["variant"] == "eligible"
                            and r["model"] == name)
            assert np.isclose(fit["heldout_mae"], expected, rtol=1e-10, atol=1e-12)
            means[name] = {
                "original_heldout_mae": float(fit["abs_error"].mean()),
                "unseen_document_subset_mae": float(fit["abs_error"][new_document].mean())}
        null_error = np.abs(design["y"][held] - np.median(design["y"][calibration]))
        means["calibration_median"] = {
            "original_heldout_mae": float(null_error.mean()),
            "unseen_document_subset_mae": float(null_error[new_document].mean())}
        paired, all_controls = [], 0
        for row in rows.values():
            if row["kind"] != "real" or not row["eligible"]:
                continue
            controls = [rows[row["feature"], row["sentence"], k]
                        for k in analysis.CONTROL_KINDS]
            all_controls += int(all(c["eligible"] for c in controls))
            swapped = rows[row["feature"], row["sentence"], "matched_swapped"]
            if swapped["eligible"]:
                paired.append([row["normalized_error"] - swapped["normalized_error"],
                               row["fp_norm"] / swapped["fp_norm"]])
        baseline_mae = means[strongest]["unseen_document_subset_mae"]
        diag_mae = means["A+B"]["unseen_document_subset_mae"]
        result["states"][state] = {
            "eligible_heldout_items": int(held.sum()),
            "unseen_document_items": int(new_document.sum()),
            "unseen_documents": int(len(set(design["cluster"][held][new_document]))),
            "calibration_selected_baseline": strongest,
            "unseen_document_diagnostic_relative_improvement": 1-diag_mae/baseline_mae,
            "mae": means,
            "joint_real_swapped_eligible_items": len(paired),
            "joint_real_all_controls_eligible_items": all_controls,
            "joint_swapped_median_normalized_error_gap": float(np.median(paired, axis=0)[0]),
            "joint_swapped_median_fp_effect_norm_ratio": float(np.median(paired, axis=0)[1])}
    group_path = COMMITTED / "group_per_language.csv"
    result["inputs_sha256"][str(group_path.relative_to(ROOT))] = hashlib.sha256(group_path.read_bytes()).hexdigest()
    group = list(csv.DictReader(group_path.open()))
    result["group_native_contrasts"] = {}
    for state in ("INT16", "INT6", "INT4"):
        rr = [r for r in group if r["state"] == state]
        residual = [float(r["net_Q_Q_native"])-float(r["net_FP_Q"]) for r in rr]
        # Exact path identity: native total = weights at FP context + remainder.
        for r, rem in zip(rr, residual):
            assert np.isclose(float(r["native_total"]), float(r["W"])+rem)
        result["group_native_contrasts"][state] = {
            "n_languages": len(rr),
            "median_native_net_harm": float(np.median([float(r["net_Q_Q_native"]) for r in rr])),
            "median_fixed_net_harm": float(np.median([float(r["net_Q_Q"]) for r in rr])),
            "median_weights_only_net_harm": float(np.median([float(r["net_FP_Q"]) for r in rr])),
            "median_paired_native_minus_weights_only": float(np.median(residual)),
            "median_abs_native_minus_weights_only": float(np.median(np.abs(residual)))}
    (HERE / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    locked["inputs_sha256"] = result["inputs_sha256"]
    locked_path = HERE / "locked_flores_models.json"
    serialized = json.dumps(locked, indent=2) + "\n"
    if locked_path.exists() and locked_path.read_text() != serialized:
        raise RuntimeError("Locked FLORES model export would change; preserve it and write a dated amendment instead")
    locked_path.write_text(serialized)
    print(json.dumps({"states": {s: {k:v for k,v in r.items() if k!="mae"}
                                for s,r in result["states"].items()},
                      "group_native_contrasts": result["group_native_contrasts"]}, indent=2))


if __name__ == "__main__":
    main()
