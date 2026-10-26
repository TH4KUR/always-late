"""Cross-corpus diagnostic of the intervention-transfer study (amendment A11, E3; D-088, D-095): the predictive models fitted on the CALIBRATION items of the main study (FLORES, 27 languages) score the HELD-OUT items of the
external panel (Wikipedia, six languages) with no refit. A separate row next to the external claim rules, not part of them. Written after the external analysis had been run once; it reads the per-item tables
(per_context_<state>.csv) that the report driver wrote for both corpora and builds the designs with the frozen `phase1_transfer_analysis.design` (eligible real-direction items, each corpus with its own floor and
eligibility); it adds no change to the frozen analysis module.

    python phase1_transfer_cross_corpus.py --train <main analysis dir> --test <external analysis dir> --out <dir> [--states INT16 INT6 INT4]
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

import phase1_transfer_analysis as ta

_INTS = ("feature", "sentence", "cluster", "language_index")
_TEXT = ("kind", "split")


def rows_from_csv(path):
    """The per-item table of a report run as the row dictionaries the design reads: {(feature, sentence, kind): row}. Integers for the ids, a boolean for `eligible`, text for `kind` and `split`, floats for everything
    else, a blank cell is None (an item with a missing value is dropped by `design`, never filled)."""
    rows = {}
    with open(path, newline="") as fh:
        for rec in csv.DictReader(fh):
            r = {}
            for k, v in rec.items():
                if k in _TEXT:
                    r[k] = v
                elif v == "":
                    r[k] = None
                elif k in _INTS:
                    r[k] = int(v)
                elif k == "eligible":
                    r[k] = v == "True"
                else:
                    r[k] = float(v)
            key = (r["feature"], r["sentence"], r["kind"])
            if key in rows:
                raise ValueError(f"item {key} occurs twice in {path}")
            rows[key] = r
    return rows


def cross_corpus_diagnostic(d_train, d_test, n_boot=ta.BOOTSTRAP_REPLICATES, seed=ta.BOOTSTRAP_SEED, lams=ta.RIDGE_STRENGTHS, n_folds=ta.N_FOLDS, min_calibration=20, min_heldout=10):
    """The models A, A+G, A+B, A+B+G fitted on the CALIBRATION items of the training design (same protocol as `ta.diagnostic`: standardization, ridge strength by feature-group cross-validation inside them, the
    coefficients, the choice of the strongest baseline among A and A+G by its cross-validated calibration error) score the HELD-OUT items of the test design with no refit; nothing of the training held-out half
    or of the test calibration half is used. The relative reduction of the mean absolute error of A+B against the strongest baseline (and against A: the H3 quantity) has the paired crossed-bootstrap interval of
    the test items. Null model: the training calibration median. The two designs must have the same columns and the training calibration features must not occur among the test held-out features."""
    if d_train["columns"] != d_test["columns"]:
        raise ValueError(f"the designs have different columns: {d_train['columns']} against {d_test['columns']}")
    cal, held = d_train["split"] == "calibration", d_test["split"] == "heldout"
    out = {"n_train_calibration": int(cal.sum()), "n_test_heldout": int(held.sum()), "diagnostic": ta.DIAGNOSTIC, "baselines": list(ta.BASELINES)}
    shared = sorted(set(d_train["feature"][cal].tolist()) & set(d_test["feature"][held].tolist()))
    if shared:
        raise ValueError(f"the training calibration half and the test held-out half share {len(shared)} features (first: {shared[:5]}): the split must separate them")
    if cal.sum() < min_calibration or held.sum() < min_heldout:
        return {**out, "status": "insufficient"}
    models = {}
    for name, groups in ta.MODEL_GROUPS.items():
        Xtr = np.hstack([d_train[g][cal] for g in groups]); Xte = np.hstack([d_test[g][held] for g in groups]); ch = ta.choose_lambda(Xtr, d_train["y"][cal], d_train["feature"][cal], lams, n_folds)
        ae = np.abs(ta.ridge_predict(ta.ridge_fit(Xtr, d_train["y"][cal], ch["lambda"]), Xte) - d_test["y"][held])
        models[name] = {"lambda": ch["lambda"], "cv_mae": ch["cv_mae"][ch["lambda"]], "heldout_mae": float(ae.mean()), "abs_error": ae}
    strongest = min(ta.BASELINES, key=lambda n: models[n]["cv_mae"]); fh, ch_ = d_test["feature"][held], d_test["cluster"][held]
    imp = ta.paired_relative_improvement(models[strongest]["abs_error"], models[ta.DIAGNOSTIC]["abs_error"], fh, ch_, n_boot, seed)
    imp_a = ta.paired_relative_improvement(models["A"]["abs_error"], models[ta.DIAGNOSTIC]["abs_error"], fh, ch_, n_boot, seed)
    return {**out, "status": "fitted", "models": models, "null_mae": float(np.abs(d_test["y"][held] - np.median(d_train["y"][cal])).mean()), "strongest_baseline": strongest, "improvement": imp,
            "improvement_over_A": imp_a, "criterion_met": ta.utility_criterion(imp)}


def run(train_dir, test_dir, states, out_dir, n_boot=ta.BOOTSTRAP_REPLICATES, seed=ta.BOOTSTRAP_SEED):
    """Per state: the designs of the eligible real items of both corpora (`<dir>/per_context_<state>.csv`), the diagnostic, and the outputs `cross_corpus.csv` (one row per state) and `cross_corpus.json`
    (everything except the per-item errors). -> the list of the table rows."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True); table, full = [], {}
    for s in states:
        dtr = ta.design(rows_from_csv(Path(train_dir) / f"per_context_{s}.csv")); dte = ta.design(rows_from_csv(Path(test_dir) / f"per_context_{s}.csv"))
        r = cross_corpus_diagnostic(dtr, dte, n_boot, seed)
        row = {"state": s, "status": r["status"], "n_train_items": int(len(dtr["y"])), "n_train_dropped": dtr["n_dropped"], "n_test_items": int(len(dte["y"])), "n_test_dropped": dte["n_dropped"], "n_train_calibration": r["n_train_calibration"], "n_test_heldout": r["n_test_heldout"]}
        if r["status"] == "fitted":
            imp, impa = r["improvement"], r["improvement_over_A"]
            row.update({"strongest_baseline": r["strongest_baseline"], "null_mae": r["null_mae"], **{f"heldout_mae_{n}": m["heldout_mae"] for n, m in r["models"].items()}, "improvement_point": imp["point"], "improvement_low": imp["low"], "improvement_high": imp["high"],
                        "improvement_over_A_point": impa["point"], "improvement_over_A_low": impa["low"], "improvement_over_A_high": impa["high"], "criterion_met": r["criterion_met"]})
            full[s] = {**{k: v for k, v in r.items() if k != "models"}, "models": {n: {k: v for k, v in m.items() if k != "abs_error"} for n, m in r["models"].items()}, "train_dropped_reasons": dtr["dropped_reasons"], "test_dropped_reasons": dte["dropped_reasons"]}
        else:
            full[s] = {**r, "train_dropped_reasons": dtr["dropped_reasons"], "test_dropped_reasons": dte["dropped_reasons"]}
        table.append(row)
    keys = []
    for row in table:
        for k in row:
            if k not in keys:
                keys.append(k)
    with open(out_dir / "cross_corpus.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(table)
    (out_dir / "cross_corpus.json").write_text(json.dumps({"train": str(train_dir), "test": str(test_dir), "n_boot": n_boot, "seed": seed, "states": full}, indent=1, default=float))
    return table


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--train", required=True); ap.add_argument("--test", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--states", nargs="*", default=["INT16", "INT6", "INT4"]); a = ap.parse_args()
    for row in run(a.train, a.test, a.states, a.out):
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})


if __name__ == "__main__":
    main()
