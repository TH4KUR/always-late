"""Native versus fixed edit at the group level: paired per-language contrasts with sentence resampling (revision R4, task 1 of docs/superpowers/plans/2026-10-26-publication-recovery-after-results.md; POST HOC; protocol
fixed in reports/intervention_transfer/revision_R4_native_fixed.json before the code and the run).

From the net harm of a language (group minus the mean of the control draws, pooled by target-row count) in the five cells FF, QF, FQ, QQ, QN: context C = QF - FF, weights W = FQ - FF, interaction I, fixed total
F = QQ - FF, native total N = QN - FF, the NATIVE REMAINDER R = QN - FQ (N = W + R) and the amplitude difference D = QN - QQ (N = F + D). The summaries over languages are medians of PAIRED per-language values (never
ratios of medians); the intervals resample the source sentences with the scheme and seed of `ta.bootstrap_a3` (one multiplicity vector per replicate, shared by every language and cell, so that both cells of a paired
difference see the same sentences). It does not say that amplitude 'causes' a share of the harm.

    python phase1_transfer_native_fixed.py --main-dirs INT16=<dir> INT6=<dir> INT4=<dir> --external-dirs INT16=<dir> ... --out <dir>
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import phase1_transfer_analysis as ta

CELL_NAMES = ("FF", "QF", "FQ", "QQ", "QN")
SUMMARY_METRICS = ("median_R", "median_D", "median_N", "median_F", "median_W", "median_abs_R_minus_abs_W", "n_abs_R_gt_abs_W", "n_D_negative", "n_R_positive")
INTERVAL_QUANTITIES = ("W", "F", "N", "R", "D")


def contrasts(net):
    """C, W, I, F, N, R, D of the net harm [L, 5] in the order FF, QF, FQ, QQ, QN."""
    net = np.asarray(net, dtype=float); ff, qf, fq, qq, qn = (net[:, i] for i in range(5))
    return {"C": qf - ff, "W": fq - ff, "I": qq - qf - fq + ff, "F": qq - ff, "N": qn - ff, "R": qn - fq, "D": qn - qq}


def identity_error(net):
    """The largest absolute violation over languages of N = W + R (both sides computed from the cells) and N = F + D."""
    c = contrasts(net); return float(max(np.nanmax(np.abs(c["N"] - (c["W"] + c["R"]))), np.nanmax(np.abs(c["N"] - (c["F"] + c["D"])))))


def summarize(c):
    """The summaries over languages of the paired per-language values (languages with a missing value in any of R, D, N, F, W are left out and counted): the medians (lower weighted median, as in A3) of R, D, N, F, W
    and of |R| - |W|, the number of languages with |R| > |W|, with D < 0 and with R > 0."""
    ok = np.ones(len(c["R"]), dtype=bool)
    for k in ("R", "D", "N", "F", "W"):
        ok &= np.isfinite(c[k])
    n = int(ok.sum()); out = {"n_languages": n, "n_missing_languages": int((~ok).sum())}
    if n == 0:
        return {**out, **{m: float("nan") for m in SUMMARY_METRICS}}
    one = np.ones(n); R, D, W = c["R"][ok], c["D"][ok], c["W"][ok]
    out.update({"median_R": ta.weighted_median(R, one), "median_D": ta.weighted_median(D, one), "median_N": ta.weighted_median(c["N"][ok], one), "median_F": ta.weighted_median(c["F"][ok], one), "median_W": ta.weighted_median(W, one),
                "median_abs_R_minus_abs_W": ta.weighted_median(np.abs(R) - np.abs(W), one), "n_abs_R_gt_abs_W": int((np.abs(R) > np.abs(W)).sum()), "n_D_negative": int((D < 0).sum()), "n_R_positive": int((R > 0).sum())})
    return out


def bootstrap(A, n_boot=ta.BOOTSTRAP_REPLICATES, seed=ta.BOOTSTRAP_SEED, level=ta.BOOTSTRAP_LEVEL):
    """The contrasts of the data (`point`, per language) and their sentence-bootstrap intervals: per language for W, F, N, R, D (`per_language`: (low [L], high [L])) and for every summary metric (`summary_ci`). The
    sentences are drawn exactly as in `ta.bootstrap_a3` (np.random.default_rng(seed); multiplicities = bincount of S draws) and the same multiplicities serve every language and cell. Replicates without any language are
    left out and counted (`n_finite`)."""
    rng = np.random.default_rng(seed); S = A["dloss"].shape[1]; per = {k: [] for k in INTERVAL_QUANTITIES}; summ = {m: [] for m in SUMMARY_METRICS}
    for _ in range(n_boot):
        mult = np.bincount(rng.integers(0, S, S), minlength=S); c = contrasts(ta.pooled_harms(A, mult=mult)["net"]); s = summarize(c)
        if s["n_languages"] == 0:
            continue
        for k in INTERVAL_QUANTITIES:
            per[k].append(c[k])
        for m in SUMMARY_METRICS:
            summ[m].append(s[m])
    lo, hi = (100 - level) / 2, 100 - (100 - level) / 2; pool = ta.pooled_harms(A); point = contrasts(pool["net"]); L = len(pool["n_sentences"])
    per_ci = {k: (np.nanpercentile(np.array(v), lo, axis=0), np.nanpercentile(np.array(v), hi, axis=0)) if v else (np.full(L, np.nan), np.full(L, np.nan)) for k, v in per.items()}
    summ_ci = {m: (float(np.percentile(v, lo)), float(np.percentile(v, hi))) if v else (float("nan"), float("nan")) for m, v in summ.items()}
    return {"point": point, "cells": pool["net"], "per_language": per_ci, "summary_point": summarize(point), "summary_ci": summ_ci, "n_finite": len(summ["median_R"]), "n_sentences": pool["n_sentences"], "n_boot": int(n_boot), "seed": int(seed), "level": level}


def _check_planned(where, got, planned):
    if got != planned:
        raise ValueError(f"{where}: {got} items plus skipped, planned {planned}: the run is incomplete or belongs to another design (chunks lost, wrong core or wrong grid)")


def run(dirs, languages, n_sentences, n_controls, states, out_dir, n_boot=ta.BOOTSTRAP_REPLICATES, seed=ta.BOOTSTRAP_SEED):
    """Per corpus and state: load the group chunks (`dirs[corpus][state]`), check the planned item count, build the group arrays, run the contrasts and the bootstrap. Writes `native_fixed_per_language.csv` (the five
    cells, C, W, I, F, N, R, D and the intervals of W, F, N, R, D per language), `native_fixed_summary.csv` (the summary metrics with intervals) and `native_fixed.json` (also the largest identity error and the hashes of the
    meta files of the inputs). -> (per-language rows, summary rows)."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True); per_rows, sum_rows, worst, inputs = [], [], 0.0, {}
    for corpus, by_state in dirs.items():
        langs = languages[corpus]
        for s in states:
            d = by_state[s]; items, _, skipped = ta.load_chunks(d, "group", s, pools=False); planned = len(langs) * n_sentences * (1 + n_controls); _check_planned(f"group level {corpus}, state {s}", len(items) + len(skipped), planned)
            meta = Path(d) / "meta_transfer.json"
            if meta.exists():
                inputs[f"{corpus}/{s}/meta_transfer.json"] = hashlib.sha256(meta.read_bytes()).hexdigest()
            A = ta.group_arrays(items, langs, n_sentences, n_controls); r = bootstrap(A, n_boot, seed); worst = max(worst, identity_error(r["cells"])); c = r["point"]
            for li, lang in enumerate(langs):
                row = {"corpus": corpus, "state": s, "language": lang, "n_sentences": int(r["n_sentences"][li]), **{n: float(r["cells"][li, i]) for i, n in enumerate(CELL_NAMES)}, **{k: float(c[k][li]) for k in ("C", "W", "I", "F", "N", "R", "D")}}
                for k in INTERVAL_QUANTITIES:
                    row[f"{k}_low"], row[f"{k}_high"] = float(r["per_language"][k][0][li]), float(r["per_language"][k][1][li])
                per_rows.append(row)
            for m in SUMMARY_METRICS:
                sum_rows.append({"corpus": corpus, "state": s, "metric": m, "point": r["summary_point"][m], "low": r["summary_ci"][m][0], "high": r["summary_ci"][m][1], "n_languages": r["summary_point"]["n_languages"], "n_boot": r["n_boot"]})
    for name, rows in (("native_fixed_per_language.csv", per_rows), ("native_fixed_summary.csv", sum_rows)):
        with open(out_dir / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out_dir / "native_fixed.json").write_text(json.dumps({"max_identity_error": worst, "inputs_meta_sha256": inputs, "n_boot": n_boot, "seed": seed, "summary": sum_rows, "per_language": per_rows}, indent=1, default=float))
    return per_rows, sum_rows


def _dirs(pairs):
    return {p.split("=")[0]: p.split("=", 1)[1] for p in pairs}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--main-dirs", nargs="+", required=True); ap.add_argument("--external-dirs", nargs="+", required=True); ap.add_argument("--core", required=True)
    ap.add_argument("--external-languages", nargs="+", required=True); ap.add_argument("--out", required=True); ap.add_argument("--sentences", type=int, default=300); ap.add_argument("--controls", type=int, default=3); a = ap.parse_args()
    main_langs = json.loads(Path(a.core).read_text())["languages"]; md, ed = _dirs(a.main_dirs), _dirs(a.external_dirs); states = list(md)
    per, summ = run({"flores_main": md, "wikipedia_external": ed}, {"flores_main": main_langs, "wikipedia_external": a.external_languages}, a.sentences, a.controls, states, a.out)
    for r in summ:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "n_boot"})


if __name__ == "__main__":
    main()
