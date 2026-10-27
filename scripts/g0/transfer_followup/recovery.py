"""Locked FLORES-to-Wikipedia scoring of the intervention-transfer study (revision R1, task 1a of docs/superpowers/plans/2026-10-26-publication-recovery-after-results.md; POST HOC; protocol fixed in
reports/intervention_transfer/revision_R1.json before the run). The predictive models fitted on the main study's FLORES calibration items (exported, after the main outcomes, to
reports/publication_reassessment_2026-10-26/locked_flores_models.json) score the external panel's held-out-feature items with no refit, no external standardization, strength or comparator choice, and no filtering on quantized
quantities. PRIMARY target: the external error norm normalized with the FLORES per-kind floors (error_norm / (fp_norm + floor of the kind)) and eligibility from the FP effect only (fp_norm above five FLORES real-direction
floors); SECONDARY: the external floors of D-095 (the A11 E3 target). Two comparators for the reduction of A+B: the locked calibration-selected baseline among A and A+G (the old comparison, retained) and the comparator among
A, A+G and the calibration-median null chosen on the MAIN calibration items' out-of-fold errors. Before any external item is scored the run verifies that the exported fits equal a refit by the frozen protocol, that
they reproduce the committed held-out MAE of every model and state, and that the re-normalization reproduces the main normalized errors; it stops otherwise.

    python phase1_transfer_recovery.py --locked <locked_flores_models.json> --main <analysis_main> --external <analysis_external> --committed <results/main/predictive_validation.csv> --out <dir>
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .tables import sha256_file as _sha, write_csv

import phase1_transfer_analysis as ta
from . import cross_corpus as cc

COMPARATOR_ORDER = ("A", "A+G", "null")
PROTOCOLS = ("primary_flores_floors", "secondary_external_floors")
REFIT_TOLERANCE = 1e-9
MAE_TOLERANCE = 1e-10


def load_locked(path):
    """The locked models: {"sha256" of the file, "floors" (the per-kind FLORES floors, equal in every state), "states": {state: {"baseline", "null" (calibration median), "models": {name: mean, scale, coef, intercept, lambda}}}}."""
    raw = Path(path).read_bytes(); d = json.loads(raw); states, floors = {}, None
    for s, m in d["models"].items():
        if floors is None:
            floors = dict(m["floor_by_kind"])
        elif dict(m["floor_by_kind"]) != floors:
            raise ValueError(f"the floors of the state {s} differ from those of the other states")
        if {k: tuple(v) for k, v in m["groups"].items()} != {k: tuple(v) for k, v in ta.MODEL_GROUPS.items()}:
            raise ValueError(f"the model groups of the state {s} differ from the frozen groups")
        states[s] = {"baseline": m["calibration_selected_baseline"], "null": float(m["null_calibration_median"]),
                     "models": {n: {"mean": np.array(f["mean"], dtype=float), "scale": np.array(f["scale"], dtype=float), "coef": np.array(f["coef"], dtype=float), "intercept": float(f["intercept"]), "lambda": f["lambda"]} for n, f in m["fits"].items()}}
    return {"sha256": hashlib.sha256(raw).hexdigest(), "floors": floors, "states": states}


def renormalize(rows, floors, multiple=ta.ELIGIBILITY_MULTIPLE):
    """Copies of the rows with the normalized error of the floors given: error_norm / (fp_norm + floor of the row's kind) (None if either norm is missing), and eligibility from the FP effect alone: fp_norm above `multiple`
    times the floor of the kind. No quantized quantity enters the eligibility; the input rows are not changed."""
    out = {}
    for key, r in rows.items():
        q = dict(r); fl = floors[r["kind"]]; en, fp = r.get("error_norm"), r.get("fp_norm")
        q["normalized_error"] = None if en is None or fp is None else en / (fp + fl)
        q["eligible"] = bool(fp is not None and fp > multiple * fl); out[key] = q
    return out


def design_of(rows):
    """`ta.design` of the eligible real items; control rows without a normalized error are left out and counted (`n_controls_missing`), never filled."""
    keep = {k: r for k, r in rows.items() if not (r["kind"] in ta.CONTROL_KINDS and r.get("normalized_error") is None)}
    d = ta.design(keep); d["n_controls_missing"] = len(rows) - len(keep); return d


def _predict(model, X):
    return ta.ridge_predict({k: model[k] for k in ("mean", "scale", "coef", "intercept")}, X)


def calibration_cv(d, lam, n_folds=ta.N_FOLDS):
    """The out-of-fold mean absolute error on the CALIBRATION items of a design for the baselines A and A+G at their given ridge strengths (`lam` {model: strength}) and for the null (the median of the training folds),
    with the frozen feature-group folds; the held-out half is not used. -> {"A", "A+G", "null"}."""
    cal = d["split"] == "calibration"; y = d["y"][cal]; fold = np.array([ta.feature_fold(f, n_folds) for f in d["feature"][cal]]); errs = {n: [] for n in COMPARATOR_ORDER}
    for k in range(n_folds):
        te = fold == k
        if not te.any():
            continue
        for n in ("A", "A+G"):
            X = np.hstack([d[g][cal] for g in ta.MODEL_GROUPS[n]]); errs[n].append(np.abs(ta.ridge_predict(ta.ridge_fit(X[~te], y[~te], lam[n]), X[te]) - y[te]))
        errs["null"].append(np.abs(y[te] - np.median(y[~te])))
    return {n: float(np.concatenate(v).mean()) for n, v in errs.items()}


def choose_comparator(cv):
    """The comparator with the lowest out-of-fold calibration error among A, A+G and the null; ties go to the earlier one in that order. Never chosen on external outcomes."""
    return min(COMPARATOR_ORDER, key=lambda n: cv[n])


def score(state_locked, d, cv, n_boot=ta.BOOTSTRAP_REPLICATES, seed=ta.BOOTSTRAP_SEED, min_heldout=10):
    """The held-out items of the design `d` scored with the locked models of one state: absolute errors of A, A+G, A+B, A+B+G and of the null (the locked calibration median); the reduction of the mean absolute error of A+B
    against the old comparator (the locked calibration-selected baseline) and the new one (`choose_comparator(cv)`), and against A (the H3 quantity), each with the paired crossed bootstrap over features and clusters."""
    held = d["split"] == "heldout"; out = {"n_heldout": int(held.sum())}
    if held.sum() < min_heldout:
        return {**out, "status": "insufficient"}
    y = d["y"][held]; models = {}
    for name, groups in ta.MODEL_GROUPS.items():
        ae = np.abs(_predict(state_locked["models"][name], np.hstack([d[g][held] for g in groups])) - y); models[name] = {"heldout_mae": float(ae.mean()), "abs_error": ae}
    null_err = np.abs(y - state_locked["null"]); fh, ch = d["feature"][held], d["cluster"][held]; err_of = lambda n: null_err if n == "null" else models[n]["abs_error"]; comps = {}
    for which, name in (("old", state_locked["baseline"]), ("new", choose_comparator(cv))):
        comps[which] = {"name": name, "heldout_mae": float(err_of(name).mean()), "improvement": ta.paired_relative_improvement(err_of(name), models["A+B"]["abs_error"], fh, ch, n_boot, seed)}
    return {**out, "status": "fitted", "models": models, "null_mae": float(null_err.mean()), "null_abs_error": null_err, "comparators": comps, "n_features": int(len(np.unique(fh))), "n_clusters": int(len(np.unique(ch))),
            "improvement_over_A": ta.paired_relative_improvement(models["A"]["abs_error"], models["A+B"]["abs_error"], fh, ch, n_boot, seed)}


def verify_locked(locked, main_rows, committed):
    """The three checks before any external item is scored. refit_equals_locked: a refit on the main calibration items by the frozen protocol (standardization, strength by feature-group cross-validation, coefficients) equals the
    exported fit. committed_mae_reproduced: the locked models give the committed held-out MAE (`committed`: rows with state, variant, model, heldout_mae) of every model and state on the main held-out items.
    renormalization_reproduces_main: `renormalize` with the locked floors reproduces the normalized error and the eligibility of every main item. -> {"ok", "checks": {name: {"ok", "detail"}}}."""
    refit, mae, renorm = {"ok": True, "detail": {}}, {"ok": True, "detail": {}}, {"ok": True, "detail": {}}
    ref = {(r["state"], r["model"]): float(r["heldout_mae"]) for r in committed if r.get("variant") == "eligible"}
    for s, st in locked["states"].items():
        rows = main_rows[s]; d = design_of(rows); cal, held = d["split"] == "calibration", d["split"] == "heldout"
        for name, groups in ta.MODEL_GROUPS.items():
            Xc = np.hstack([d[g][cal] for g in groups]); ch = ta.choose_lambda(Xc, d["y"][cal], d["feature"][cal]); fit = ta.ridge_fit(Xc, d["y"][cal], ch["lambda"]); lk = st["models"][name]
            diff = max(float(np.max(np.abs(fit[k] - lk[k]) / (1.0 + np.abs(lk[k])))) for k in ("mean", "scale", "coef")); diff = max(diff, abs(fit["intercept"] - lk["intercept"]) / (1.0 + abs(lk["intercept"])))
            good = ch["lambda"] == lk["lambda"] and diff <= REFIT_TOLERANCE; refit["detail"][f"{s}/{name}"] = {"max_relative_difference": diff, "lambda_refit": ch["lambda"], "lambda_locked": lk["lambda"], "ok": good}; refit["ok"] &= good
            got = float(np.abs(_predict(lk, np.hstack([d[g][held] for g in groups])) - d["y"][held]).mean()); want = ref.get((s, name)); good = want is not None and abs(got - want) <= MAE_TOLERANCE
            mae["detail"][f"{s}/{name}"] = {"locked": got, "committed": want, "ok": good}; mae["ok"] &= good
        new = renormalize(rows, locked["floors"]); bad = 0; worst = 0.0
        for k, r in rows.items():
            a, b = r.get("normalized_error"), new[k]["normalized_error"]
            if (a is None) != (b is None) or new[k]["eligible"] != r["eligible"] or (a is not None and abs(a - b) > 1e-9 * (1.0 + abs(a))):
                bad += 1
            elif a is not None:
                worst = max(worst, abs(a - b))
        renorm["detail"][s] = {"items": len(rows), "mismatches": bad, "max_abs_difference": worst, "ok": bad == 0}; renorm["ok"] &= bad == 0
    checks = {"refit_equals_locked": refit, "committed_mae_reproduced": mae, "renormalization_reproduces_main": renorm}
    return {"ok": all(c["ok"] for c in checks.values()), "checks": checks}


def run(locked_path, main_dir, external_dir, committed_csv, states, out_dir, n_boot=ta.BOOTSTRAP_REPLICATES, seed=ta.BOOTSTRAP_SEED):
    """Verify, then score both protocols per state. Writes `cross_corpus_locked.csv` (one row per state and protocol) and `cross_corpus_locked.json` (all numbers, the verification, the hashes of the locked file and of
    every input). Stops with a RuntimeError, writing only `verification.json`, if a check fails. -> the table rows."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True); locked = load_locked(locked_path)
    main_rows = {s: cc.rows_from_csv(Path(main_dir) / f"per_context_{s}.csv") for s in states}; ext_rows = {s: cc.rows_from_csv(Path(external_dir) / f"per_context_{s}.csv") for s in states}
    with open(committed_csv, newline="") as fh:
        committed = list(csv.DictReader(fh))
    ver = verify_locked({**locked, "states": {s: locked["states"][s] for s in states}}, main_rows, committed); (out_dir / "verification.json").write_text(json.dumps(ver, indent=1, default=float))
    if not ver["ok"]:
        raise RuntimeError("verification failed: " + ", ".join(k for k, c in ver["checks"].items() if not c["ok"]))
    table, full = [], {}
    for s in states:
        st = locked["states"][s]; dmain = design_of(main_rows[s]); cv = calibration_cv(dmain, {n: st["models"][n]["lambda"] for n in ("A", "A+G")})
        for protocol, rows in (("primary_flores_floors", renormalize(ext_rows[s], locked["floors"])), ("secondary_external_floors", ext_rows[s])):
            d = design_of(rows); r = score(st, d, cv, n_boot, seed); real = [x for x in rows.values() if x["kind"] == "real"]
            row = {"state": s, "protocol": protocol, "status": r["status"], "locked_sha256": locked["sha256"], "n_external_items": len(real), "n_eligible_items": sum(1 for x in real if x["eligible"]), "n_design_items": int(len(d["y"])),
                   "n_dropped": d["n_dropped"], "n_controls_missing": d["n_controls_missing"], "n_heldout_eligible": r["n_heldout"]}
            if r["status"] == "fitted":
                o, n_ = r["comparators"]["old"], r["comparators"]["new"]
                row.update({"n_features": r["n_features"], "n_clusters": r["n_clusters"], "old_comparator": o["name"], "new_comparator": n_["name"], **{f"heldout_mae_{k}": m["heldout_mae"] for k, m in r["models"].items()}, "null_mae": r["null_mae"],
                            "old_comparator_mae": o["heldout_mae"], "new_comparator_mae": n_["heldout_mae"], "improvement_old_point": o["improvement"]["point"], "improvement_old_low": o["improvement"]["low"], "improvement_old_high": o["improvement"]["high"],
                            "improvement_new_point": n_["improvement"]["point"], "improvement_new_low": n_["improvement"]["low"], "improvement_new_high": n_["improvement"]["high"], "improvement_over_A_point": r["improvement_over_A"]["point"],
                            "improvement_over_A_low": r["improvement_over_A"]["low"], "improvement_over_A_high": r["improvement_over_A"]["high"], **{f"cv_mae_{k}": v for k, v in cv.items()}})
                full.setdefault(s, {})[protocol] = {**{k: v for k, v in r.items() if k not in ("models", "null_abs_error")}, "models": {k: {"heldout_mae": m["heldout_mae"]} for k, m in r["models"].items()}, "dropped_reasons": d["dropped_reasons"], "cv_mae_main_calibration": cv}
            else:
                full.setdefault(s, {})[protocol] = {**r, "dropped_reasons": d["dropped_reasons"]}
            table.append(row)
    write_csv(out_dir / "cross_corpus_locked.csv", table)
    inputs = {f"{side}/per_context_{s}.csv": _sha(Path(dir_) / f"per_context_{s}.csv") for side, dir_ in (("main", main_dir), ("external", external_dir)) for s in states}
    (out_dir / "cross_corpus_locked.json").write_text(json.dumps({"locked_sha256": locked["sha256"], "floors": locked["floors"], "n_boot": n_boot, "seed": seed, "verification": ver, "inputs_sha256": inputs, "states": full}, indent=1, default=float))
    return table


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--locked", required=True); ap.add_argument("--main", required=True); ap.add_argument("--external", required=True)
    ap.add_argument("--committed", required=True); ap.add_argument("--out", required=True); ap.add_argument("--states", nargs="*", default=["INT16", "INT6", "INT4"]); a = ap.parse_args()
    for row in run(a.locked, a.main, a.external, a.committed, a.states, a.out):
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if not k.startswith("cv_") and k != "locked_sha256"})


if __name__ == "__main__":
    main()
