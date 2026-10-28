"""Stage 1 of the survival measurement-validity study (D-117; plan docs/superpowers/plans/2026-10-29-survival-measurement-validity-study.md; rules frozen BEFORE any value in
reports/survival_measurement/stage1_freeze.json). CPU only, from the Phase 1 survival.pt files of the three arms (legacy RTN, matched RTN, GPTQ).

lambda = log(1/r^2 - 1) is the log noise-to-signal ratio of a feature: if b = g*a + c + e then r^2 = g^2 var(a) / (g^2 var(a) + var(e)), so lambda = log(var(e) / (g^2 var(a))).
Duan's survival (r > 0.9) is lambda < log(1/0.81 - 1). Post hoc and exploratory: the correlations were used before (N6, N9, D-104).

  OMP_NUM_THREADS=4 ~/.venvs/anlp-torch/bin/python -B scripts/g0/survival_lambda.py --out reports/survival_measurement/stage1"""
import argparse, csv, json, math, sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parents[1]

R_LO, R_HI = 0.001, 1 - 1e-6
LAMBDA_CUT = math.log(1 / 0.81 - 1)
FIRING_THRESHOLD = 0.001
ARMS = {"legacy": "~/.cache/anlp-always-late/phase1_runs/p1-mask/{stream}/survival.pt",
        "matched_rtn": "/home/karma/Documents/ANLP/project/transfer_runs/rtnm-p1-mask/{stream}/survival.pt",
        "gptq": "/home/karma/Documents/ANLP/project/transfer_runs/gptq-p1-mask/{stream}/survival.pt"}
ARM_LABEL = {"matched_rtn": "matched RTN", "gptq": "GPTQ"}
WIDTHS = (7, 6, 5, 4, 16)
PRIMARY = (6, 4)


def lam(r):
    """-> (lambda, clipped): r clipped to [R_LO, R_HI]; NaN, zero and negative correlations count as clipped at R_LO."""
    r = np.asarray(r, dtype=float)
    clipped = ~np.isfinite(r) | (r < R_LO) | (r > R_HI)
    rc = np.clip(np.nan_to_num(r, nan=R_LO), R_LO, R_HI)
    return np.log(1 / rc ** 2 - 1), clipped


def survival_share(corr, firing_rate, threshold=FIRING_THRESHOLD):
    """Duan's summary (phase1_run.survival_summary): active = firing rate strictly above threshold (compared in the file's dtype); share r > 0.9 in percent; lower-middle median."""
    fr = np.asarray(firing_rate)
    c = np.asarray(corr, dtype=float)[fr > fr.dtype.type(threshold)]
    s = np.sort(c)
    return float((c > 0.9).mean() * 100), float(s[(len(s) - 1) // 2]), int(len(c))


def h1_stats(lam_leg, lam_arm, clip_leg, clip_arm):
    """-> (Spearman over all features, geometric-mean slope sd(arm)/sd(legacy) over features with neither value clipped)."""
    keep = ~(clip_leg | clip_arm)
    return float(spearmanr(lam_leg, lam_arm).statistic), float(np.std(lam_arm[keep]) / np.std(lam_leg[keep]))


def h1_met(rhos, slopes):
    return bool(np.median(rhos) >= 0.8 and 0.8 <= np.median(slopes) <= 1.25)


def shift_model(lam_leg_by_lang, lam_arm_by_lang):
    """One shift per (arm, width): delta = median over all (language, feature) of lambda_arm - lambda_legacy; predicted share per language from the shifted legacy values."""
    delta = float(np.median(np.concatenate([b - a for a, b in zip(lam_leg_by_lang, lam_arm_by_lang)])))
    pred = [float(((a + delta) < LAMBDA_CUT).mean() * 100) for a in lam_leg_by_lang]
    obs = [float((b < LAMBDA_CUT).mean() * 100) for b in lam_arm_by_lang]
    none = [float((a < LAMBDA_CUT).mean() * 100) for a in lam_leg_by_lang]
    return {"delta": delta, "predicted": pred, "observed": obs,
            "mae": float(np.mean(np.abs(np.subtract(pred, obs)))), "mae_no_shift": float(np.mean(np.abs(np.subtract(none, obs))))}


def h2_met(mae):
    return bool(mae <= 2.0)


def h3_lambda_met(median_rho):
    return bool(abs(median_rho) > 0.5)


def ols_r2(y, X):
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    res = y - X1 @ beta
    return float(1 - res @ res / ((y - y.mean()) @ (y - y.mean())))


def load(arm, stream):
    import torch
    d = torch.load(Path(ARMS[arm].format(stream=stream)).expanduser(), map_location="cpu", weights_only=False)
    return {"languages": list(d["languages"]), "corr": {int(b): d["corr"][b].numpy() for b in d["corr"]},
            "firing_rate": d["firing_rate"].numpy(), "mean_activation": d["mean_activation"].numpy(), "max_activation": d["max_activation"].numpy()}


def reproduction_control(runs):
    """Committed tables: legacy -> reports/phase1/mask/eq/per_language.csv; arms -> stage1_paired_per_language.csv (a = arm, b = legacy). Returns the largest difference."""
    worst = 0.0; checked = 0
    with open(REPO / "reports/phase1/mask/eq/per_language.csv") as f:
        for row in csv.DictReader(f):
            b = int(row["bits"]); leg = runs["legacy"]; i = leg["languages"].index(row["language"])
            if b not in leg["corr"]:
                continue
            pct, med, n = survival_share(leg["corr"][b][i], leg["firing_rate"][i])
            worst = max(worst, abs(pct - float(row["survived_>0.9_pct"])), abs(med - float(row["median_corr"])), abs(n - int(row["n_active_features"]))); checked += 1
    names = {"matched RTN minus legacy RTN": "matched_rtn", "GPTQ minus legacy RTN": "gptq"}
    with open(REPO / "reports/gptq_extension/results/stage1/stage1_paired_per_language.csv") as f:
        for row in csv.DictReader(f):
            if row["contrast"] not in names:
                continue
            run = runs[names[row["contrast"]]]; b = int(row["bits"]); i = run["languages"].index(row["language"])
            pct, med, _ = survival_share(run["corr"][b][i], run["firing_rate"][i])
            worst = max(worst, abs(pct - float(row["survived_pct_a"])), abs(med - float(row["median_corr_a"]))); checked += 1
    return worst, checked


def analyse(runs, n_perm=10000, seed=0):
    import phase1_analysis as an
    import phase1_language_level as ll
    leg = runs["legacy"]; langs = leg["languages"]
    for arm in runs.values():
        assert arm["languages"] == langs and np.array_equal(arm["firing_rate"], leg["firing_rate"]), "FP statistics differ between arms"
    roster = {l["code"]: l for l in json.loads((REPO / "reports/g0/phase1_languages.json").read_text())["languages"]}
    tier = np.array([an.tier_score(roster[c]["tier"]) for c in langs])
    fams = sorted({roster[c]["script_family"] for c in langs}); script = np.array([fams.index(roster[c]["script_family"]) for c in langs])
    scale = ll.language_activation_scale({"languages": langs, "firing_rate": leg["firing_rate"], "max_activation": leg["max_activation"]})
    active = [leg["firing_rate"][i] > np.float32(FIRING_THRESHOLD) for i in range(len(langs))]
    L = {}   # (arm, width) -> list per language of (lambda, clipped)
    for name, run in runs.items():
        for w in WIDTHS:
            L[name, w] = [lam(run["corr"][w][i][active[i]]) for i in range(len(langs))]
    out = {"languages": langs, "n_active": [int(a.sum()) for a in active], "cells": {}}
    for name, run in runs.items():
        for w in WIDTHS:
            cell = {"clipped_high": int(sum(((c) & (run["corr"][w][i][active[i]] > R_HI)).sum() for i, (_, c) in enumerate(L[name, w]))),
                    "clipped_low": int(sum(((c) & ~(run["corr"][w][i][active[i]] > R_HI)).sum() for i, (_, c) in enumerate(L[name, w]))),
                    "median_lambda_by_language": [float(np.median(l)) for l, _ in L[name, w]]}
            rhos, r2s = [], []
            for i in range(len(langs)):
                l, c = L[name, w][i]; mx = run["max_activation"][i][active[i]].astype(float)
                rhos.append(float(spearmanr(l, np.log10(mx)).statistic))
                X = np.column_stack([np.log10(mx), np.log10(run["mean_activation"][i][active[i]].astype(float)), np.log10(run["firing_rate"][i][active[i]].astype(float))])
                r2s.append(ols_r2(l[~c], X[~c]))
            cell["H3_spearman_lambda_logmax_median"] = float(np.median(rhos)); cell["H3_spearman_by_language"] = rhos
            cell["H3_r2_median"] = float(np.median(r2s)); cell["H3_lambda_half_met"] = h3_lambda_met(np.median(rhos))
            y = -np.array(cell["median_lambda_by_language"])
            t0 = an.freedman_lane_tier_test(y, tier, script, n_perm, seed); t1 = an.freedman_lane_tier_test(y, tier, script, n_perm, seed, covariates=scale)
            cell["tier"] = {"slope": float(t0["slope"]), "p": float(t0["p"]), "slope_with_scale": float(t1["slope"]), "p_with_scale": float(t1["p"])}
            if name != "legacy":
                st = [h1_stats(L["legacy", w][i][0], L[name, w][i][0], L["legacy", w][i][1], L[name, w][i][1]) for i in range(len(langs))]
                cell["H1_spearman_by_language"] = [s[0] for s in st]; cell["H1_slope_by_language"] = [s[1] for s in st]
                cell["H1_spearman_median"] = float(np.median([s[0] for s in st])); cell["H1_slope_median"] = float(np.median([s[1] for s in st]))
                cell["H1_met"] = h1_met([s[0] for s in st], [s[1] for s in st])
                sm = shift_model([L["legacy", w][i][0] for i in range(len(langs))], [L[name, w][i][0] for i in range(len(langs))])
                cell["H2"] = sm; cell["H2_met"] = h2_met(sm["mae"])
            out["cells"][f"{name}/{w}"] = cell
    prim = [out["cells"][f"{a}/{w}"] for a in ("matched_rtn", "gptq") for w in PRIMARY]
    out["verdict"] = {"H1": all(c["H1_met"] for c in prim), "H2": all(c["H2_met"] for c in prim),
                      "H3_lambda_half": all(out["cells"][f"{a}/{w}"]["H3_lambda_half_met"] for a in runs for w in PRIMARY)}
    return out


def render_md(res, stream):
    f = lambda x: f"{x:.3f}"
    L = [f"# Stage 1: what the survival correlation measures (stream `{stream}`; D-117)", "",
         "Generated by `scripts/g0/survival_lambda.py` from the Phase 1 `survival.pt` files of the three arms; rules frozen before any value in "
         "[`../stage1_freeze.json`](../stage1_freeze.json). **Post hoc and exploratory** (the correlations were used before: N6, N9, D-104).", "",
         f"Reproduction control: {res['control']['checked']} committed per-language values recomputed, largest difference {res['control']['worst']:.2e} (must be <= 1e-6).", "",
         "lambda = log(1/r^2 - 1) = log(noise variance / (gain^2 x signal variance)); Duan's survival r > 0.9 is lambda < -1.4500.", "",
         f"**Verdicts (primary cells: matched RTN and GPTQ at INT6 and INT4):** H1 {'met' if res['verdict']['H1'] else 'not met'}, H2 {'met' if res['verdict']['H2'] else 'not met'}, "
         f"lambda half of H3 {'met' if res['verdict']['H3_lambda_half'] else 'not met'}.", "",
         "## H1 and H2: are the quantizers one shift of the same feature-wise noise-to-signal?", "",
         "| arm | bits | median Spearman with legacy | median slope | H1 | shift delta | survival MAE with shift (points) | MAE without shift | H2 |", "|---|---|---|---|---|---|---|---|---|"]
    for a in ("matched_rtn", "gptq"):
        for w in WIDTHS:
            c = res["cells"][f"{a}/{w}"]
            L.append(f"| {ARM_LABEL[a]} | INT{w} | {f(c['H1_spearman_median'])} | {f(c['H1_slope_median'])} | {'met' if c['H1_met'] else 'not met'} | {f(c['H2']['delta'])} | "
                     f"{c['H2']['mae']:.2f} | {c['H2']['mae_no_shift']:.2f} | {'met' if c['H2_met'] else 'not met'} |")
    L += ["", "Rules are applied to INT6 and INT4 only; INT7, INT5 and INT16 are reported.", "",
          "## H3 (lambda half) and tier", "",
          "| arm | bits | median Spearman(lambda, log max act.) | median R^2 (log max, log mean, log firing) | clipped low / high | tier slope of -median lambda (p) | with scale covariate (p) |",
          "|---|---|---|---|---|---|---|"]
    for a in ("legacy", "matched_rtn", "gptq"):
        for w in WIDTHS:
            c = res["cells"][f"{a}/{w}"]; t = c["tier"]
            L.append(f"| {ARM_LABEL.get(a, 'legacy RTN')} | INT{w} | {f(c['H3_spearman_lambda_logmax_median'])} | {f(c['H3_r2_median'])} | {c['clipped_low']} / {c['clipped_high']} | "
                     f"{f(t['slope'])} ({t['p']:.4f}) | {f(t['slope_with_scale'])} ({t['p_with_scale']:.4f}) |")
    L += ["", "Tier: language outcome = minus the median lambda over its active features (higher = better), slope on the ordinal tier (low 0, mid 1, high 2), one-sided Freedman-Lane with script-family dummies; "
          "the covariate is N9's log10 median max activation. Descriptive only: no equivalence claim from a large p.", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--n-perm", type=int, default=10000)
    args = ap.parse_args(argv); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for stream in ("eq", "all"):
        runs = {a: load(a, stream) for a in ARMS}
        res = {"stream": stream}
        if stream == "eq":
            worst, checked = reproduction_control(runs)
            res["control"] = {"worst": worst, "checked": checked}
            if worst > 1e-6:
                raise SystemExit(f"reproduction control failed: largest difference {worst}")
        else:
            res["control"] = {"worst": float("nan"), "checked": 0}
        res.update(analyse(runs, n_perm=args.n_perm))
        (out / f"stage1_{stream}.json").write_text(json.dumps(res, indent=1) + "\n")
        (out / f"stage1_{stream}.md").write_text(render_md(res, stream) + "\n")
        print(stream, json.dumps(res["verdict"]))


if __name__ == "__main__":
    main()
