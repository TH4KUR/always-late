"""A3 (plan Task 8, specified 2026-10-23 before any number was computed): does the FUNCTION of a feature (F5 write-side similarity W1: the cosine of the effect fingerprint in the quantized state with the one in full
precision) depend on the full-precision margin that predicts the loss of its D-035 language association (A2, `phase1_assoc_margin.py`)? If the association status measures the margin and not the function, the margin should
predict the association loss but not W1.

Joins the F5 main run (per-state files of the jobs, merged as in the analysis; the tested features) with the D-035 per-feature statuses (`feature_status.csv`, read by `phase1_assoc_margin.read_features`: established
features, m0 = margin, a0 = scale, s0 = share, the lost flag per precision) by (language, feature id) through the F5 sample (feature id to language). Per precision (INT8, INT6, INT5, INT4): the within-language Spearman
correlation of m0 with W1 (mean over the languages with enough features, bootstrap interval over languages); the within-language AUROC of -m0 for the D-035 loss and for the function event W1 < floor on the SAME features
(the floor: the 5th percentile of W1 at INT16 over all tested features, as in D-059); an OLS of W1 on z(log m0), z(log a0), z(s0) with language fixed effects and cluster-robust errors; the cohort without the excluded
languages; the verdicts of the hypotheses written in the plan. Exploratory; the F5 sample is stratified by role, so class shares are not the population's.

    ~/.venvs/anlp-torch/bin/python scripts/g0/phase1_assoc_function_link.py --status <feature_status.csv> --run <F5 job A> --run <F5 job B> --out reports/association_retention/results/margin-function"""
import argparse
import csv
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np

import phase1_assoc_margin as am
import phase1_func_f5_analysis as an
import phase1_func_f5_metrics as met

PRECISIONS = ("INT8", "INT6", "INT5", "INT4")
OLS_NAMES = ["m0", "a0", "s0"]


def join_features(feats, write_features, sample):
    """Match the tested features (their ids, in the order of the write rows of the run) to the established ones by (language, id), the language coming from the F5 sample. -> {"positions": indices into the tested
    features, "rows": indices into the arrays of `feats`, "n_tested", "n_joined", "n_dropped"}; a tested feature that is not in the sample, or not established in the status file for its language, is dropped and counted."""
    index = {(str(l), int(i)): k for k, (l, i) in enumerate(zip(feats["language"], feats["feature_id"]))}; positions, rows = [], []
    for pos, fid in enumerate(write_features):
        rec = sample["features"].get(str(int(fid)))
        row = index.get((str(rec["language"]), int(fid))) if rec is not None else None
        if row is not None:
            positions.append(pos); rows.append(row)
    return {"positions": np.array(positions, dtype=int), "rows": np.array(rows, dtype=int), "n_tested": len(write_features), "n_joined": len(rows), "n_dropped": len(write_features) - len(rows)}


def _bootstrap_mean(vals, n_boot, seed):
    draws = np.random.default_rng(seed).integers(0, len(vals), (n_boot, len(vals))); means = vals[draws].mean(axis=1)
    return float(vals.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def within_language_spearman(x, y, langs, min_n=20, n_boot=2000, seed=0):
    """The Spearman correlation of x with y within each language that has at least `min_n` pairs with both values defined (and variation); the mean over those languages and a percentile interval from `n_boot` bootstrap draws over languages."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float); langs = np.asarray(langs); per = {}
    for lg in np.unique(langs):
        sel = langs == lg; r = met.spearman(x[sel], y[sel])
        if r["n"] >= min_n and r["rho"] == r["rho"]:
            per[str(lg)] = r["rho"]
    vals = np.array(list(per.values()), dtype=float)
    if not len(vals):
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_languages": 0, "per_language": {}}
    mean, lo, hi = _bootstrap_mean(vals, n_boot, seed)
    return {"mean": mean, "lo": lo, "hi": hi, "n_languages": len(vals), "per_language": per}


def fixed_effects_ols(y, X, langs, names):
    """OLS of y on the columns of X with one intercept per language and cluster-robust standard errors by language; rows with a missing value are left out and counted. -> {"n", "n_missing", "n_languages",
    "coefficients": {name: {"coef", "lo", "hi", "p"}}}."""
    import statsmodels.api as sm
    y = np.asarray(y, dtype=float); X = np.asarray(X, dtype=float); langs = np.asarray(langs); ok = ~(np.isnan(X).any(axis=1) | np.isnan(y)); n_missing = int((~ok).sum()); y, X, langs = y[ok], X[ok], langs[ok]
    used = [lg for lg in np.unique(langs) if (langs == lg).sum() >= 2]; keep = np.isin(langs, used); y, X, langs = y[keep], X[keep], langs[keep]; res = {"n": int(len(y)), "n_missing": n_missing, "n_languages": len(used), "coefficients": {}}
    if len(used) < 2 or not len(y):
        res["error"] = "too few languages"; return res
    dummies = (langs[:, None] == np.array(used)[None, :]).astype(float); codes = np.searchsorted(np.array(used), langs)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore"); fit = sm.OLS(y, np.column_stack([X, dummies])).fit(cov_type="cluster", cov_kwds={"groups": codes})
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"; return res
    for j, name in enumerate(names):
        b, se = float(fit.params[j]), float(fit.bse[j]); res["coefficients"][name] = {"coef": b, "lo": b - 1.96 * se, "hi": b + 1.96 * se, "p": float(fit.pvalues[j])}
    return res


def analyse(feats, write_features, states, sample, precisions=PRECISIONS, floor_state="INT16", min_n=20, min_class=3, n_boot=2000, seed=0, exclude=()):
    """The analyses of the module docstring. `write_features` and the arrays of `states[s]["write"]["cos"]` are the tested features of the run in the same order."""
    if floor_state not in states:
        raise ValueError(f"the floor state {floor_state} is missing")
    tau = met.floor_threshold(np.asarray(states[floor_state]["write"]["cos"], dtype=float)); j = join_features(feats, write_features, sample); pos, rows = j["positions"], j["rows"]
    keep = ~np.isin(feats["language"][rows], list(exclude)); pos, rows = pos[keep], rows[keep]; lang = feats["language"][rows]; v = {n: feats[n][rows] for n in ("m0", "a0", "s0")}
    X = np.column_stack([am._z(am._log(v["m0"])), am._z(am._log(v["a0"])), am._z(v["s0"])])
    out = {"n_tested": j["n_tested"], "n_joined": int(len(rows)), "n_dropped": j["n_dropped"], "floor": tau, "floor_state": floor_state, "excluded": [str(x) for x in exclude], "languages": sorted(str(x) for x in np.unique(lang)),
           "min_n": min_n, "min_class": min_class, "n_boot": n_boot, "seed": seed, "by_precision": {}}
    for p in precisions:
        if p not in states:
            continue
        w1 = np.asarray(states[p]["write"]["cos"], dtype=float)[pos]; defined = ~np.isnan(w1); event = w1 < tau; lost = feats["lost"][p][rows]; score = np.where(defined, -v["m0"], np.nan)
        out["by_precision"][p] = {"n": int(defined.sum()), "n_events": int(event[defined].sum()), "event_share": float(event[defined].mean()) if defined.any() else float("nan"),
                                  "spearman": within_language_spearman(v["m0"], w1, lang, min_n, n_boot, seed), "auroc_association": am.within_language_auroc(-v["m0"], lost, lang, min_n, min_class, n_boot, seed),
                                  "auroc_function": am.within_language_auroc(score, event, lang, min_n, min_class, n_boot, seed), "ols": fixed_effects_ols(w1, X, lang, OLS_NAMES)}
    return out


def _val(res, p, *keys):
    d = res["by_precision"].get(p)
    for k in keys:
        d = d.get(k) if isinstance(d, dict) else None
    return float("nan") if d is None else float(d)


def hypotheses(res):
    """The verdicts of the hypotheses written in the plan (A3). H-A3a: the within-language Spearman correlation of m0 with W1 is below 0.15 in absolute value at INT6, INT5 and INT4. H-A3b: at INT6 the AUROC of -m0 for the D-035 loss is at least 0.70
    and the AUROC of -m0 for the function event is at most 0.60 at INT6 and INT5. H-A3c: at INT6 and INT5 the first AUROC exceeds the second by at least 0.15. 'not evaluable' when a needed value is missing."""
    h = {}
    sp = {p: _val(res, p, "spearman", "mean") for p in ("INT6", "INT5", "INT4")}
    h["H-A3a"] = {"verdict": "not evaluable" if any(x != x for x in sp.values()) else ("met" if all(abs(x) < 0.15 for x in sp.values()) else "not met"), "spearman_mean": sp}
    aa = {p: _val(res, p, "auroc_association", "mean") for p in ("INT6", "INT5")}; af = {p: _val(res, p, "auroc_function", "mean") for p in ("INT6", "INT5")}
    bad = any(x != x for x in list(aa.values()) + list(af.values()))
    h["H-A3b"] = {"verdict": "not evaluable" if bad else ("met" if aa["INT6"] >= 0.70 and af["INT6"] <= 0.60 and af["INT5"] <= 0.60 else "not met"), "auroc_association": aa, "auroc_function": af}
    diff = {p: aa[p] - af[p] for p in aa}
    h["H-A3c"] = {"verdict": "not evaluable" if bad else ("met" if all(d >= 0.15 - 1e-12 for d in diff.values()) else "not met"), "difference": diff}
    return h


def _fmt(x, nd=3):
    return "n/a" if x is None or x != x else f"{x:.{nd}f}"


def _table(res, title):
    L = [f"### {title}", "", f"Tested features {res['n_tested']}, joined with an established D-035 feature {res['n_joined']}" + (f" (without {', '.join(res['excluded'])})" if res["excluded"] else "") + f", dropped {res['n_dropped']}; numerical floor tau = {res['floor']:.4f} (5th percentile of W1 at {res['floor_state']} over all tested features); languages: {len(res['languages'])}.", "",
         f"Within-language correlation and AUROCs (mean over the languages with at least {res['min_n']} features, and {res['min_class']} of each class for an AUROC; percentile interval over languages, {res['n_boot']} draws). The AUROCs score -m0, so 0.5 = no information:", "",
         "| precision | features | function events (W1 < tau) | Spearman(m0, W1) | AUROC of -m0, association lost | AUROC of -m0, function event |", "|---|---|---|---|---|---|"]
    for p, b in res["by_precision"].items():
        f = lambda d: f"{_fmt(d['mean'])} [{_fmt(d['lo'])}, {_fmt(d['hi'])}] ({d['n_languages']})"
        L.append(f"| {p} | {b['n']} | {b['n_events']} ({_fmt(b['event_share'])}) | {f(b['spearman'])} | {f(b['auroc_association'])} | {f(b['auroc_function'])} |")
    L += ["", "OLS of W1 on z(log m0), z(log a0), z(s0) with a fixed effect per language (change of W1 per standard deviation; cluster-robust 95 % interval by language):", "", "| precision | m0 | scale a0 | share s0 | features |", "|---|---|---|---|---|"]
    for p, b in res["by_precision"].items():
        c = b["ols"]["coefficients"]; L.append(f"| {p} | " + " | ".join((f"{_fmt(c[n]['coef'], 4)} [{_fmt(c[n]['lo'], 4)}, {_fmt(c[n]['hi'], 4)}]" if n in c else "n/a") for n in OLS_NAMES) + f" | {b['ols']['n']} |")
    return L


def report_text(res_all, res_ex, hyp, meta):
    L = ["# A3: does the function of a feature depend on the margin that predicts the loss of its association?", "",
         "**Exploratory and descriptive; nothing here is causal.** Function = the F5 write-side similarity W1 (cosine of the effect fingerprint in the quantized state with the one in full precision); the function event is W1 below the numerical floor. Association = the D-035 status. "
         f"Margin m0, scale a0 and share s0 are read from the full-precision row of the D-035 per-feature file `{meta['status']}` (sha256 `{meta['sha256']}`). **The F5 sample is stratified by role (retained and lost at each precision, at most 12 features per language cell): class shares are not the population's; AUROCs and correlations are not affected by the shares, a pooled loss fraction would be.** "
         "m0 was measured on devtest like the statuses.", "", "## Verdicts of the hypotheses written before any number (plan A3)", "",
         f"- **H-A3a** (|Spearman(m0, W1)| below 0.15 at INT6, INT5, INT4): **{hyp['H-A3a']['verdict']}** (INT6 {_fmt(hyp['H-A3a']['spearman_mean']['INT6'])}, INT5 {_fmt(hyp['H-A3a']['spearman_mean']['INT5'])}, INT4 {_fmt(hyp['H-A3a']['spearman_mean']['INT4'])}).",
         f"- **H-A3b** (AUROC for the association loss at INT6 at least 0.70; AUROC for the function event at most 0.60 at INT6 and INT5): **{hyp['H-A3b']['verdict']}** (association INT6 {_fmt(hyp['H-A3b']['auroc_association']['INT6'])}; function INT6 {_fmt(hyp['H-A3b']['auroc_function']['INT6'])}, INT5 {_fmt(hyp['H-A3b']['auroc_function']['INT5'])}).",
         f"- **H-A3c** (the two AUROCs differ by at least 0.15 at INT6 and INT5): **{hyp['H-A3c']['verdict']}** (differences INT6 {_fmt(hyp['H-A3c']['difference']['INT6'])}, INT5 {_fmt(hyp['H-A3c']['difference']['INT5'])}).", ""]
    L += _table(res_all, "All languages") + [""]
    if res_ex["excluded"]:
        L += _table(res_ex, f"Cohort without {', '.join(res_ex['excluded'])}") + [""]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="A3: the F5 write-side similarity against the full-precision margin that predicts the D-035 association loss.")
    ap.add_argument("--status", required=True); ap.add_argument("--run", action="append", required=True); ap.add_argument("--sample", default=str(Path(__file__).resolve().parents[2] / "reports/functional_validation/f5/f5_sample.json")); ap.add_argument("--out", required=True)
    ap.add_argument("--min-n", type=int, default=20); ap.add_argument("--min-class", type=int, default=3); ap.add_argument("--n-boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--exclude", nargs="*", default=["eng_Latn"]); a = ap.parse_args(argv)
    path = Path(a.status); feats = am.read_features(path); sa, states, run_meta = an.load_runs(a.run); sample = json.loads(Path(a.sample).read_text()); wf = [int(x) for x in sa["write_features"]]
    kw = dict(min_n=a.min_n, min_class=a.min_class, n_boot=a.n_boot, seed=a.seed); res_all = analyse(feats, wf, states, sample, **kw); res_ex = analyse(feats, wf, states, sample, exclude=tuple(a.exclude), **kw)
    hyp, hyp_ex = hypotheses(res_all), hypotheses(res_ex); meta = {"status": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "runs": [Path(r).name for r in a.run], "sample": Path(a.sample).name}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    doc = {"input": meta, "all": res_all, "without_excluded": res_ex, "hypotheses": hyp, "hypotheses_without_excluded": hyp_ex}
    (out / "analysis.json").write_text(json.dumps(am._clean(doc), indent=1, allow_nan=False) + "\n"); (out / "report.md").write_text(report_text(res_all, res_ex, hyp, meta))
    cols = ["precision", "n", "n_events", "event_share", "spearman_mean", "spearman_lo", "spearman_hi", "auroc_association", "auroc_function", "ols_m0", "ols_a0", "ols_s0"]
    with open(out / "margin_vs_function.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(cols)
        for p, b in res_all["by_precision"].items():
            c = b["ols"]["coefficients"]; w.writerow([p, b["n"], b["n_events"], b["event_share"], b["spearman"]["mean"], b["spearman"]["lo"], b["spearman"]["hi"], b["auroc_association"]["mean"], b["auroc_function"]["mean"]] + [c[n]["coef"] if n in c else "" for n in OLS_NAMES])
    print("wrote", out, "|", {k: v["verdict"] for k, v in hyp.items()})


if __name__ == "__main__":
    main()
