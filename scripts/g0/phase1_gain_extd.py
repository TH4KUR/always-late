"""Study G, extension D (design: docs/superpowers/specs/2026-10-24-graded-gain-study-design.md section 12, D-083): does the optimal gain of a language at INT4 follow the shrinkage of its group's activation?
Post hoc in origin (suggested by D-082), exploratory, CPU only. Input: the item file of `gainc-4b` (INT4, 562 reserve sentences) and the baseline G1 (`g_ratio`, `controls_ratio` of the INT4 rows).
Per language the optimal gain is the ratio among {1} + the extension C ratios with the lowest group harm (grid, primary; equal values go to the ratio nearest 1) and the vertex of a quadratic in ln(ratio) through
the origin, clipped to [0, ln of the largest ratio] (smooth, secondary). Rules (fixed before any number):
  R-D1  Spearman rho(alpha, 1 / g_ratio) over the languages, one-sided permutation p-value (p = (1 + number of permutations with rho at least the observed) / (permutations + 1); 10,000, seed 0);
        supported: rho >= 0.30 and p < 0.05 and the split-half reliability >= 0.30; uninformative: reliability < 0.30; otherwise not supported;
  R-D2  the same with the smooth estimator (descriptive);
  R-D3  rho(alpha, 1 / controls_ratio) and the difference of the two correlations with a 95 % bootstrap interval over languages (5,000 draws, seed 0), descriptive;
  R-D4  split-half: the sentences split by the parity of their position, alpha on each half, Spearman correlation between halves, Spearman-Brown reliability, the corrected R-D1 correlation (descriptive).

    ~/.venvs/anlp-torch/bin/python scripts/g0/phase1_gain_extd.py --runs <gainc-4b dir> --baseline-csv <baseline.csv> --out <dir>"""
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

import phase1_func_analysis as fa
import phase1_gain_extc as ec
import phase1_gain_ratios as rt

RHO_MIN, P_MAX, RELIABILITY_MIN = 0.30, 0.05, 0.30


def alpha_grid(harm):
    """The ratio with the lowest group harm among 1 (harm 0) and the ratios of `harm` {ratio: harm}; equal values go to the ratio nearest 1 on the log scale."""
    grid = [1.0] + sorted(harm)
    return min(grid, key=lambda a: (0.0 if a == 1.0 else harm[a], abs(math.log(a))))


def alpha_smooth(harm):
    """Least-squares fit harm = c1 x + c2 x^2 (x = ln ratio, through the origin) over the given ratios; the vertex clipped to [0, ln of the largest ratio] when the fit is convex, otherwise the end of that range
    with the lower fitted harm (1 on a tie)."""
    ratios = sorted(harm)
    x = np.log(ratios); y = np.array([harm[a] for a in ratios]); top = float(x[-1])
    c1, c2 = np.linalg.lstsq(np.column_stack([x, x ** 2]), y, rcond=None)[0]
    if c2 > 1e-12:
        star = min(max(-c1 / (2 * c2), 0.0), top)
    else:
        star = top if c1 * top + c2 * top ** 2 < 0 else 0.0
    return float(math.exp(star))


def _ranks_centred(v):
    r = stats.rankdata(np.asarray(v, dtype=float)); r = r - r.mean()
    return r, float(np.sqrt((r ** 2).sum()))


def _rho(x, y):
    """Spearman correlation; 0.0 when either vector is constant (no information)."""
    rx, nx = _ranks_centred(x); ry, ny = _ranks_centred(y)
    return 0.0 if nx == 0 or ny == 0 else float((rx @ ry) / (nx * ny))


def spearman_perm(x, y, n_perm=10000, seed=0):
    """{"rho", "p" (one-sided, rho > 0, (1 + #{permuted rho >= observed}) / (n_perm + 1)), "constant"}."""
    rx, nx = _ranks_centred(x); ry, ny = _ranks_centred(y)
    if nx == 0 or ny == 0:
        return {"rho": 0.0, "p": 1.0, "constant": True}
    rho = float((rx @ ry) / (nx * ny))
    perms = np.random.default_rng(seed).permuted(np.tile(ry, (n_perm, 1)), axis=1)
    rhos = (perms @ rx) / (nx * ny)
    return {"rho": rho, "p": float((1 + np.sum(rhos >= rho - 1e-12)) / (n_perm + 1)), "constant": False}


def split_halves(n):
    """Positions of the two halves: even and odd positions in the id list."""
    return np.arange(0, n, 2), np.arange(1, n, 2)


def reliability(alpha_a, alpha_b):
    """Correlation between the two half-sample estimates over the languages and its Spearman-Brown step-up (0 when the correlation is not positive)."""
    rho = _rho(alpha_a, alpha_b)
    return {"rho_half": rho, "sb": 0.0 if rho <= 0 else float(2 * rho / (1 + rho))}


def corrected(rho, rel):
    """The correlation corrected for the unreliability of the estimator, rho / sqrt(reliability), within [-1, 1]; None when the reliability is 0."""
    if rel <= 0:
        return None
    return float(min(max(rho / math.sqrt(rel), -1.0), 1.0))


def boot_diff(alpha, tau, tau_c, n_boot=5000, seed=0):
    """rho(alpha, tau) - rho(alpha, tau_c) with a 95 % percentile interval from a bootstrap over languages; resamples in which a vector is constant are counted and left out."""
    alpha, tau, tau_c = (np.asarray(v, dtype=float) for v in (alpha, tau, tau_c))
    idx = np.random.default_rng(seed).integers(0, len(alpha), size=(n_boot, len(alpha)))
    vals = []
    for i in idx:
        a = alpha[i]
        if np.ptp(a) == 0 or np.ptp(tau[i]) == 0 or np.ptp(tau_c[i]) == 0:
            continue
        vals.append(_rho(a, tau[i]) - _rho(a, tau_c[i]))
    lo, hi = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"), float("nan"))
    return {"diff": _rho(alpha, tau) - _rho(alpha, tau_c), "lo": lo, "hi": hi, "n_nan": n_boot - len(vals)}


def verdict(rho, p, rel):
    if rel < RELIABILITY_MIN:
        return "uninformative"
    return "supported" if (rho >= RHO_MIN and p < P_MAX) else "not supported"


def per_language_harm(e, state="INT4"):
    """{language: {"all"|"even"|"odd": {ratio: pooled group harm in nats per token}}} for the sentences of the whole sample and of the even and odd positions."""
    run = e.runs[state]; out = {}
    for code in e.languages:
        cell = {"all": {}, "even": {}, "odd": {}}
        for a in e.ratios:
            dg, nt = run.delta(code, rt.group_name(a, e.label, code))
            even, odd = split_halves(len(dg))
            cell["all"][a] = fa.pooled(dg, nt); cell["even"][a] = fa.pooled(dg, nt, even); cell["odd"][a] = fa.pooled(dg, nt, odd)
        out[code] = cell
    return out


def load_baseline(path, state="INT4"):
    """{language: {"g_ratio", "controls_ratio"}} of the rows of `state` of baseline.csv."""
    with open(path) as fh:
        rows = [r for r in csv.DictReader(fh) if r["state"] == state]
    if not rows:
        raise ValueError(f"{path} has no row of state {state}")
    return {r["language"]: {"g_ratio": float(r["g_ratio"]), "controls_ratio": float(r["controls_ratio"])} for r in rows}


def evaluate(harm, baseline, languages, n_perm=10000, n_boot=5000, seed=0):
    langs = list(languages)
    tau = np.array([1.0 / baseline[c]["g_ratio"] for c in langs]); tau_c = np.array([1.0 / baseline[c]["controls_ratio"] for c in langs])
    alpha = np.array([alpha_grid(harm[c]["all"]) for c in langs]); smooth = np.array([alpha_smooth(harm[c]["all"]) for c in langs])
    a_even = np.array([alpha_grid(harm[c]["even"]) for c in langs]); a_odd = np.array([alpha_grid(harm[c]["odd"]) for c in langs])
    rd1 = spearman_perm(alpha, tau, n_perm, seed); rd2 = spearman_perm(smooth, tau, n_perm, seed)
    d = boot_diff(alpha, tau, tau_c, n_boot, seed)
    rd3 = {"rho_placebo": _rho(alpha, tau_c), **d}
    rel = reliability(a_even, a_odd)
    rd4 = {**rel, "corrected": corrected(rd1["rho"], rel["sb"])}
    return {"verdict": verdict(rd1["rho"], rd1["p"], rel["sb"]), "rd1": rd1, "rd2": rd2, "rd3": rd3, "rd4": rd4, "n_perm": n_perm, "n_boot": n_boot, "seed": seed,
            "languages": {c: {"alpha": float(alpha[i]), "alpha_smooth": float(smooth[i]), "alpha_even": float(a_even[i]), "alpha_odd": float(a_odd[i]), "tau": float(tau[i]), "tau_controls": float(tau_c[i]),
                              "harm": {a: harm[c]["all"][a] for a in sorted(harm[c]["all"])}} for i, c in enumerate(langs)}}


def _v(x):
    return {"supported": "**supported**", "uninformative": "**uninformative**", "not supported": "**not supported**"}[x]


def report_text(res, e, meta):
    r1, r2, r3, r4 = res["rd1"], res["rd2"], res["rd3"], res["rd4"]; n = len(res["languages"])
    corr = "undefined" if r4["corrected"] is None else f"{r4['corrected']:.3f}"
    interval = "interval undefined (no usable resample)" if r3["n_nan"] >= res["n_boot"] else f"[{r3['lo']:+.3f}, {r3['hi']:+.3f}] (95 %)"
    tau = [v["tau"] for v in res["languages"].values()]
    out = ["# Study G, extension D: does the optimal gain follow the quantization shrinkage?", "",
           f"Post hoc in origin (suggested by the result of extension C, D-082), exploratory, no new job; rules fixed before the numbers (spec section 12, D-083). Data: INT4, {e.n_sentences} reserve sentences per language, {n} languages; "
           f"alpha = the ratio among 1, {', '.join(f'{a:g}' for a in e.ratios)} with the lowest group harm; target 1 / g_ratio (baseline G1, INT4; range {min(tau):.2f} to {max(tau):.2f}).", "",
           "## Verdict", "", "| rule | statistic | result | verdict |", "|---|---|---|---|",
           f"| R-D1 rank correlation of alpha with 1 / g_ratio | rho >= 0.30, one-sided permutation p < 0.05 ({res['n_perm']} permutations, seed {res['seed']}), reliability >= 0.30 | rho = {r1['rho']:.3f}, p = {r1['p']:.4f}; split-half reliability {r4['sb']:.3f} | {_v(res['verdict'])} |",
           f"| R-D2 the same with the smooth estimator | descriptive | rho = {r2['rho']:.3f}, p = {r2['p']:.4f} | |",
           f"| R-D3 placebo target 1 / controls_ratio | descriptive | rho = {r3['rho_placebo']:.3f}; difference (group target minus control target) {r3['diff']:+.3f} {interval} ({res['n_boot']} bootstrap draws over languages, {r3['n_nan']} resamples left out) | |",
           f"| R-D4 split-half (even and odd sentence positions) | descriptive | correlation between halves {r4['rho_half']:.3f}; Spearman-Brown reliability {r4['sb']:.3f}; corrected R-D1 correlation {corr} | |", "",
           "## Per language", "", "| language | 1 / g | 1 / controls | alpha (grid) | alpha (smooth) | alpha, even half | alpha, odd half |", "|---|---|---|---|---|---|---|"]
    for c, v in res["languages"].items():
        out.append(f"| {c} | {v['tau']:.3f} | {v['tau_controls']:.3f} | {v['alpha']:g} | {v['alpha_smooth']:.3f} | {v['alpha_even']:g} | {v['alpha_odd']:g} |")
    out += ["", "## Limits", "",
            "- Post hoc: the question came from the extension C result; the rules were written before this analysis was run, but the data are the same data. A language-level correlation over 27 points; alpha lies on a grid of four ratios.",
            "- 'uninformative' (reliability below 0.30) is not a refutation; 'not supported' with a reliable estimator is a failure to find the relation at this resolution, not proof of its absence (the spread of 1 / g is about 0.06 in standard deviation).",
            "- g_ratio is measured on the selection-independent text of the baseline, not on the evaluation sentences; one model, layer, SAE and quantizer.", ""]
    return "\n".join(out)


def write_outputs(out_dir, res, e, meta):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    ratios = e.ratios
    with open(out / "alpha.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["language", "tau", "tau_controls", "alpha", "alpha_smooth", "alpha_even", "alpha_odd"] + [f"harm_{a:g}" for a in ratios])
        for c, v in res["languages"].items():
            w.writerow([c, v["tau"], v["tau_controls"], v["alpha"], v["alpha_smooth"], v["alpha_even"], v["alpha_odd"]] + [v["harm"][a] for a in ratios])
    body = {k: v for k, v in res.items() if k != "languages"}
    (out / "analysis.json").write_text(json.dumps({**body, "n_languages": len(res["languages"]), "ratios": ratios, "n_sentences": e.n_sentences, "meta": meta}, indent=1, default=float) + "\n")
    (out / "report.md").write_text(report_text(res, e, meta))


def main(argv=None):
    import argparse
    import phase1_func_tasks as ft
    ap = argparse.ArgumentParser(description="Study G extension D: optimal gain against the quantization shrinkage (rules R-D1 to R-D4).")
    ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--baseline-csv", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n-perm", type=int, default=10000); ap.add_argument("--n-boot", type=int, default=5000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--expect-ids", help="comma-separated sentence ids the runs must have (default: the reserve ids of lm_ids(1012, 450, 562, 0))")
    ap.add_argument("--forbid-ids", help="comma-separated ids that must not occur (default: the pilot and evaluation ids of the main study)")
    a = ap.parse_args(argv)
    main_ids = ft.lm_ids(1012, 150, 300, 0)
    expect = [int(x) for x in a.expect_ids.split(",")] if a.expect_ids else ft.lm_ids(1012, 450, 562, 0)["eval"]
    forbid = [int(x) for x in a.forbid_ids.split(",")] if a.forbid_ids else main_ids["eval"] + main_ids["pilot"]
    e, info = ec.load_runs(a.runs, ec.EXTC_RATIOS, expect, forbid)
    if "INT4" not in e.states:
        raise ValueError(f"extension D needs the INT4 run, got {e.states}")
    res = evaluate(per_language_harm(e, "INT4"), load_baseline(a.baseline_csv, "INT4"), e.languages, a.n_perm, a.n_boot, a.seed)
    write_outputs(a.out, res, e, {"runs": info, "baseline_sha256": ec._sha256(a.baseline_csv)})
    print(f"wrote {a.out}: R-D1 rho {res['rd1']['rho']:.3f} p {res['rd1']['p']:.4f} reliability {res['rd4']['sb']:.3f} -> {res['verdict']}")


if __name__ == "__main__":
    main()
