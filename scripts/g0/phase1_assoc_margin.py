"""A2 (plan Task 8, specified 2026-10-23 before any number was computed): is the loss of language association (D-035) predicted by how borderline a feature was in full precision? Reads the per-feature D-035 status
file (`feature_status.csv` of the `eq` stream), keeps the selective features that were established in full precision (`fp_status` retained) and, for each precision, whether the association was lost (`status` not
retained). Predictors come from the full-precision row of the same feature: m0 = `margin` (the peak language's rate minus the runner-up's), r0 = m0 / `target_rate`, s0 = `target_share`, a0 = `target_mean`.

Per precision: the loss fraction by within-language quintile of m0 and of r0 (pooled counts, Wilson intervals); the within-language AUROC of -m0, -r0, -s0 and -a0 for loss (mean over the languages with enough
features and both outcomes, percentile interval from a bootstrap over languages); a logistic regression of loss on z(log m0), z(log a0), z(s0) with language fixed effects and cluster-robust standard errors by language
(odds ratios per standard deviation). The same for the cohort without the excluded languages (default English, which holds a fifth of the features) and the verdicts of the three hypotheses written in the plan.
Exploratory and descriptive: the statuses were measured on devtest and so was m0, nothing is causal.

    ~/.venvs/anlp-torch/bin/python scripts/g0/phase1_assoc_margin.py --status <feature_status.csv> --out reports/association_retention/results/within-language"""
import argparse
import csv
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from phase1_func_mc_calibration import wilson

PRECISIONS = ["INT16", "INT12", "INT8", "INT7", "INT6", "INT5", "INT4"]
PREDICTORS = ("m0", "r0", "s0", "a0")
LOGIT_NAMES = ["m0", "a0", "s0"]


def _f(x):
    return float("nan") if x in ("", None) else float(x)


def read_features(path):
    """The selective features established in full precision, sorted by (language, feature id): {"language", "feature_id", "m0", "r0", "s0", "a0" (arrays), "lost": {precision: bool array}, "n_selective",
    "n_established", "n_missing_predictor"}. A status other than `retained` counts as lost. An established feature without a row for every precision is an error."""
    fp, cond, selective = {}, {}, set()
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["group"] != "selective":
                continue
            key = (r["original_language"], int(r["feature_id"])); selective.add(key)
            if r["condition"] == "FP":
                fp[key] = r
            elif r["condition"] in PRECISIONS:
                cond.setdefault(key, {})[r["condition"]] = r["status"]
    keys = sorted(k for k, r in fp.items() if r["fp_status"] == "retained"); out = {"language": [], "feature_id": [], "m0": [], "r0": [], "s0": [], "a0": [], "lost": {p: [] for p in PRECISIONS}}; n_missing = 0
    for k in keys:
        miss = [p for p in PRECISIONS if p not in cond.get(k, {})]
        if miss:
            raise ValueError(f"feature {k} has no row for {miss}")
        r = fp[k]; m, rate = _f(r["margin"]), _f(r["target_rate"]); r0 = m / rate if rate == rate and rate != 0 and m == m else float("nan")
        vals = (m, r0, _f(r["target_share"]), _f(r["target_mean"])); n_missing += int(any(v != v for v in vals))
        out["language"].append(k[0]); out["feature_id"].append(k[1])
        for name, v in zip(PREDICTORS, vals):
            out[name].append(v)
        for p in PRECISIONS:
            out["lost"][p].append(cond[k][p] != "retained")
    res = {"language": np.array(out["language"]), "feature_id": np.array(out["feature_id"], dtype=int), "lost": {p: np.array(v, dtype=bool) for p, v in out["lost"].items()}}
    for name in PREDICTORS:
        res[name] = np.array(out[name], dtype=float)
    res.update({"n_selective": len(selective), "n_established": len(keys), "n_missing_predictor": n_missing})
    return res


def quantile_bins(values, langs, k=5):
    """Within-language equal-count bins by the rank of `values` (0 = the smallest values of the language; ties in the order of the rows); NaN gets -1 and no rank."""
    values = np.asarray(values, dtype=float); langs = np.asarray(langs); out = np.full(len(values), -1, dtype=int)
    for lg in np.unique(langs):
        idx = np.flatnonzero((langs == lg) & ~np.isnan(values)); n = len(idx)
        if n == 0:
            continue
        order = idx[np.argsort(values[idx], kind="stable")]; out[order] = (np.arange(n) * k) // n
    return out


def loss_by_bin(bins, lost, k=5):
    """Per bin 0..k-1: the number of features, the number lost, the fraction and its Wilson 95 % interval (rows with bin -1 are not counted)."""
    bins = np.asarray(bins); lost = np.asarray(lost, dtype=bool); out = []
    for i in range(k):
        sel = bins == i; n = int(sel.sum()); c = int(lost[sel].sum()); lo, hi = wilson(c, n) if n else (float("nan"), float("nan"))
        out.append({"bin": i, "n": n, "lost": c, "fraction": c / n if n else float("nan"), "lo": lo, "hi": hi})
    return out


def auroc(score, y):
    """Probability that a randomly chosen lost feature has a larger score than a kept one (ties count one half); NaN scores are left out; NaN when one class is empty."""
    score = np.asarray(score, dtype=float); y = np.asarray(y, dtype=bool); d = ~np.isnan(score); s, y = score[d], y[d]; npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    ranks = rankdata(s)
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def within_language_auroc(score, y, langs, min_n=20, min_class=3, n_boot=2000, seed=0):
    """The AUROC within each language that has at least `min_n` defined features and `min_class` lost and kept ones; the mean over those languages and a percentile interval from `n_boot` bootstrap draws over languages."""
    score = np.asarray(score, dtype=float); y = np.asarray(y, dtype=bool); langs = np.asarray(langs); per = {}
    for lg in np.unique(langs):
        sel = (langs == lg) & ~np.isnan(score); npos = int(y[sel].sum()); nneg = int(sel.sum()) - npos
        if sel.sum() >= min_n and npos >= min_class and nneg >= min_class:
            per[str(lg)] = auroc(score[sel], y[sel])
    vals = np.array(list(per.values()), dtype=float)
    if not len(vals):
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_languages": 0, "per_language": {}}
    draws = np.random.default_rng(seed).integers(0, len(vals), (n_boot, len(vals))); means = vals[draws].mean(axis=1)
    return {"mean": float(vals.mean()), "lo": float(np.percentile(means, 2.5)), "hi": float(np.percentile(means, 97.5)), "n_languages": len(vals), "per_language": per}


def fixed_effects_logit(y, X, langs, names):
    """Logistic regression of y on the columns of X with one intercept per language, cluster-robust standard errors by language; rows with a missing predictor and languages without variation in y are left out and
    counted. -> {"n", "n_missing", "n_languages_used", "n_languages_dropped", "odds_ratios": {name: {"or", "lo", "hi", "p"}}}."""
    import statsmodels.api as sm
    y = np.asarray(y, dtype=float); X = np.asarray(X, dtype=float); langs = np.asarray(langs); ok = ~np.isnan(X).any(axis=1); n_missing = int((~ok).sum()); y, X, langs = y[ok], X[ok], langs[ok]
    used = [lg for lg in np.unique(langs) if 0 < y[langs == lg].mean() < 1]; dropped = len(np.unique(langs)) - len(used); keep = np.isin(langs, used); y, X, langs = y[keep], X[keep], langs[keep]
    res = {"n": int(len(y)), "n_missing": n_missing, "n_languages_used": len(used), "n_languages_dropped": dropped, "odds_ratios": {}}
    if len(used) < 2 or not len(y):
        res["error"] = "too few languages with variation"; return res
    dummies = (langs[:, None] == np.array(used)[None, :]).astype(float); design = np.column_stack([X, dummies]); codes = np.searchsorted(np.array(used), langs)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore"); fit = sm.Logit(y, design).fit(disp=0, maxiter=200, cov_type="cluster", cov_kwds={"groups": codes})
    except Exception as e:                                                                                                                                      # separation or a singular design: reported, not hidden
        res["error"] = f"{type(e).__name__}: {e}"; return res
    for j, name in enumerate(names):
        b, se = float(fit.params[j]), float(fit.bse[j]); res["odds_ratios"][name] = {"or": float(np.exp(b)), "lo": float(np.exp(b - 1.96 * se)), "hi": float(np.exp(b + 1.96 * se)), "p": float(fit.pvalues[j])}
    return res


def _z(x):
    x = np.asarray(x, dtype=float); sd = np.nanstd(x)
    return (x - np.nanmean(x)) / sd if sd > 0 else x * 0.0


def _log(x):
    x = np.asarray(x, dtype=float); return np.where(x > 0, np.log(np.where(x > 0, x, 1.0)), np.nan)


def analyse(feats, precisions=PRECISIONS, min_n=20, min_class=3, n_boot=2000, seed=0, exclude=(), k=5):
    """The analyses of the module docstring for the features of `read_features`, without the languages in `exclude`."""
    keep = ~np.isin(feats["language"], list(exclude)); lang = feats["language"][keep]; v = {n: feats[n][keep] for n in PREDICTORS}
    bins = {"m0": quantile_bins(v["m0"], lang, k), "r0": quantile_bins(v["r0"], lang, k)}; X = np.column_stack([_z(_log(v["m0"])), _z(_log(v["a0"])), _z(v["s0"])])
    out = {"n_established": int(keep.sum()), "languages": sorted(str(x) for x in np.unique(lang)), "excluded": [str(x) for x in exclude], "min_n": min_n, "min_class": min_class, "n_boot": n_boot, "seed": seed, "k": k, "by_precision": {}}
    for p in precisions:
        lost = feats["lost"][p][keep]
        out["by_precision"][p] = {"loss_overall": float(lost.mean()) if len(lost) else float("nan"), "bins_m0": loss_by_bin(bins["m0"], lost, k), "bins_r0": loss_by_bin(bins["r0"], lost, k),
                                  "auroc": {n: within_language_auroc(-v[n], lost, lang, min_n, min_class, n_boot, seed) for n in PREDICTORS}, "logit": fixed_effects_logit(lost, X, lang, LOGIT_NAMES)}
    return out


def hypotheses(res):
    """The verdicts of the three hypotheses written in the plan (A2). H-A2a: at INT6 the within-language AUROC of -m0 is at least 0.70 and at INT6, INT5 and INT4 the lowest margin quintile loses more than the highest. H-A2b: at INT4 the loss
    exceeds the floor (INT16) in every m0 quintile (Wilson lower bound of INT4 above the upper bound of INT16); otherwise 'at the floor in the top quintile' when the top one does not, else 'mixed'. H-A2c: at INT6, INT5 and INT4 the odds ratio
    of m0 is further from 1 on the log scale than those of scale and share."""
    bp = res["by_precision"]; h = {}
    a6 = bp["INT6"]["auroc"]["m0"]["mean"]; trend = {p: bp[p]["bins_m0"][0]["fraction"] > bp[p]["bins_m0"][-1]["fraction"] for p in ("INT6", "INT5", "INT4")}
    h["H-A2a"] = {"verdict": "met" if (a6 >= 0.70 and all(trend.values())) else "not met", "auroc_m0_int6": a6, "lowest_exceeds_highest": trend}
    q4, q16 = bp["INT4"]["bins_m0"], bp["INT16"]["bins_m0"]; above = [bool(a["lo"] > b["hi"]) for a, b in zip(q4, q16)]
    h["H-A2b"] = {"verdict": "exceeds the floor in every quintile" if all(above) else ("at the floor in the top quintile" if not above[-1] else "mixed"), "above_floor_by_quintile": above, "int4_top": q4[-1]["fraction"], "floor_top": q16[-1]["fraction"]}
    ok, det = [], {}
    for p in ("INT6", "INT5", "INT4"):
        o = bp[p]["logit"]["odds_ratios"]
        if not o:
            ok.append(False); continue
        lm = {n: abs(np.log(o[n]["or"])) for n in LOGIT_NAMES}; det[p] = lm; ok.append(lm["m0"] > lm["a0"] and lm["m0"] > lm["s0"])
    h["H-A2c"] = {"verdict": "met" if all(ok) else "not met", "abs_log_or": det}
    return h


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if x != x else float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _fmt(x, nd=3):
    return "n/a" if x is None or x != x else f"{x:.{nd}f}"


def _cohort_tables(res, title):
    bp = res["by_precision"]; L = [f"### {title}", "", f"Established features: {res['n_established']} in {len(res['languages'])} languages" + (f" (without {', '.join(res['excluded'])})" if res["excluded"] else "") + ".", "",
                                   "Lost fraction by within-language quintile of the full-precision margin m0 (quintile 1 = the most borderline fifth of each language; pooled counts, Wilson 95 % intervals):", "",
                                   "| precision | all lost | q1 (smallest m0) | q2 | q3 | q4 | q5 (largest m0) |", "|---|---|---|---|---|---|---|"]
    for p, b in bp.items():
        cells = " | ".join(f"{_fmt(x['fraction'])} [{_fmt(x['lo'])}, {_fmt(x['hi'])}] ({x['lost']}/{x['n']})" for x in b["bins_m0"]); L.append(f"| {p} | {_fmt(b['loss_overall'])} | {cells} |")
    L += ["", f"Within-language AUROC for loss (the score is minus the predictor, so 0.5 = no information and values above 0.5 mean that a smaller value goes with loss; mean over the languages with at least {res['min_n']} features and {res['min_class']} lost and kept ones, percentile interval over languages, {res['n_boot']} draws):", "",
          "| precision | -m0 | -r0 | -s0 (share) | -a0 (scale) | languages |", "|---|---|---|---|---|---|"]
    for p, b in bp.items():
        a = b["auroc"]; L.append(f"| {p} | " + " | ".join(f"{_fmt(a[n]['mean'])} [{_fmt(a[n]['lo'])}, {_fmt(a[n]['hi'])}]" for n in PREDICTORS) + f" | {a['m0']['n_languages']} |")
    L += ["", "Logistic regression of loss on z(log m0), z(log a0) and z(s0) with a fixed effect per language (odds ratio per standard deviation, cluster-robust 95 % interval by language; languages without variation in the outcome are dropped):", "",
          "| precision | m0 | scale a0 | share s0 | features | languages used |", "|---|---|---|---|---|---|"]
    for p, b in bp.items():
        g = b["logit"]; o = g["odds_ratios"]; cells = " | ".join((f"{_fmt(o[n]['or'])} [{_fmt(o[n]['lo'])}, {_fmt(o[n]['hi'])}]" if n in o else "n/a") for n in LOGIT_NAMES); L.append(f"| {p} | {cells} | {g['n']} | {g['n_languages_used']} |" + (f" {g['error']}" if "error" in g else ""))
    return L


def report_text(res_all, res_ex, hyp, meta):
    L = ["# A2: is the loss of language association predicted by how borderline the feature was? (D-035 per-feature statuses, within language)", "",
         "**Exploratory and descriptive; nothing here is causal.** The statuses are the D-035 statuses of the `eq` stream (measured on devtest; features selected and ranked on dev). Established = `retained` in full precision; lost at a precision = not `retained` there. "
         f"The predictors are read from the full-precision row of the same feature: m0 = the peak language's rate minus the runner-up's (`margin`), r0 = m0 over the target rate, s0 = the target share, a0 = the activity scale (`target_mean`). Input: `{meta['input']}`, sha256 `{meta['sha256']}`.", "",
         "**Limit that travels with the numbers:** m0 was measured on the same text as the statuses, so a borderline feature's m0 is itself noisy and regression to the mean inflates the association of m0 with loss; the rates at INT16 and INT12 (rounding noise only) are the floor to compare with.", "",
         "## Verdicts of the hypotheses written before any number (plan A2)", "",
         f"- **H-A2a** (loss decreases with the margin; AUROC of -m0 at INT6 at least 0.70; lowest quintile loses more than the highest at INT6, INT5, INT4): **{hyp['H-A2a']['verdict']}** (AUROC at INT6 {_fmt(hyp['H-A2a']['auroc_m0_int6'])}).",
         f"- **H-A2b** (at INT4 the loss exceeds the floor in every m0 quintile): **{hyp['H-A2b']['verdict']}** (top quintile: {_fmt(hyp['H-A2b']['int4_top'])} at INT4 against {_fmt(hyp['H-A2b']['floor_top'])} at INT16).",
         f"- **H-A2c** (given m0, scale and share have odds ratios closer to 1): **{hyp['H-A2c']['verdict']}**.", ""]
    L += _cohort_tables(res_all, "All languages") + [""]
    if res_ex["excluded"]:
        L += _cohort_tables(res_ex, f"Cohort without {', '.join(res_ex['excluded'])}") + [""]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="A2: within-language models of D-035 association loss against the full-precision margin, scale and share.")
    ap.add_argument("--status", required=True, help="feature_status.csv of the eq stream"); ap.add_argument("--out", required=True); ap.add_argument("--min-n", type=int, default=20); ap.add_argument("--min-class", type=int, default=3)
    ap.add_argument("--n-boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--exclude", nargs="*", default=["eng_Latn"]); a = ap.parse_args(argv)
    path = Path(a.status); feats = read_features(path); sha = hashlib.sha256(path.read_bytes()).hexdigest(); meta = {"input": path.name, "sha256": sha, "bytes": path.stat().st_size}
    res_all = analyse(feats, min_n=a.min_n, min_class=a.min_class, n_boot=a.n_boot, seed=a.seed); res_ex = analyse(feats, min_n=a.min_n, min_class=a.min_class, n_boot=a.n_boot, seed=a.seed, exclude=tuple(a.exclude))
    hyp, hyp_ex = hypotheses(res_all), hypotheses(res_ex); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    doc = {"input": {**meta, "n_selective": feats["n_selective"], "n_established": feats["n_established"], "n_missing_predictor": feats["n_missing_predictor"]}, "all": res_all, "without_excluded": res_ex, "hypotheses": hyp, "hypotheses_without_excluded": hyp_ex}
    (out / "analysis.json").write_text(json.dumps(_clean(doc), indent=1, allow_nan=False) + "\n"); (out / "report.md").write_text(report_text(res_all, res_ex, hyp, meta))
    with open(out / "loss_by_margin_quintile.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["precision", "quintile", "n", "lost", "fraction", "lo", "hi"])
        for p, b in res_all["by_precision"].items():
            for x in b["bins_m0"]:
                w.writerow([p, x["bin"] + 1, x["n"], x["lost"], x["fraction"], x["lo"], x["hi"]])
    print("wrote", out, "|", {k: v["verdict"] for k, v in hyp.items()})


if __name__ == "__main__":
    main()
