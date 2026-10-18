"""N6: is the dose-response of N2 (more language-selective features survive less than matched controls) a selectivity effect or an activation-magnitude effect? (decision D-020;
pre-specified in handoff 002 before any magnitude number was looked at; one amendment, also made before any real number: several magnitude-bin resolutions.)
Source of the question: issues.md C6 / PLAN P1-5. Duan et al.'s own released per-feature data show survival follows maximum activation (Spearman 0.78) and not firing rate
(-0.085), while the proposal matches controls on firing-rate quintile only. Everything is computed from the files phase1_run.save_outputs wrote; no new estimator is introduced:
(a) threshold-free dose-response of survival on S inside firing-rate strata, (b) a descriptive magnitude comparison, (c) the unchanged D-014/D-016 analysis with controls matched on
firing-rate stratum x maximum-activation bin (phase1_threshold.run_thresholds with its `transform` hook). Magnitude m(f) = maximum activation of the feature at its peak language,
full-precision stream; survival y_b(f) = r > 0.9 at the peak language (their strict rule); evaluable = full-precision firing rate at the peak language above their 0.001 threshold."""
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

import phase1_analysis as an
import phase1_threshold as th

BAND = 0.01          # +-1 pp: the pre-specified 'shrinks to within' band
ALPHA = 0.05
RETAIN = 0.5         # operationalisation (fixed before the numbers) of 'in between': retaining less than half of the rate-only gap = partly explained


def _np(x):
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def evaluable_table(survival, selp, bits, threshold=0.001):
    """Supported features that are evaluable at their peak language: S, firing-rate stratum, magnitude m, correlation r and survival y at width `bits`."""
    peak = _np(selp["peak"]); F = len(peak); idx = np.arange(F); supported = _np(selp["supported"]).astype(bool)
    ev = supported & (_np(survival["firing_rate"])[peak, idx] > threshold)
    r = _np(survival["corr"][bits])[peak, idx][ev].astype(float)
    if np.isnan(r).any():
        raise ValueError("NaN correlation among the evaluable features")
    return {"feature": idx[ev], "peak": peak[ev], "S": _np(selp["S"]).astype(float)[ev], "stratum": _np(selp["strata"])[ev], "m": _np(survival["max_activation"]).astype(float)[peak, idx][ev],
            "r": r, "y": (r > an.CUT).astype(float)}


def within_stratum_rho(x, y, strata, n_perm=10000, seed=0):
    """n-weighted mean over firing-rate strata of the Spearman correlation of x with y (average ranks), two-sided permutation p with x permuted WITHIN strata
    (+1 correction). A stratum with fewer than three features or a constant x or y is left out and listed."""
    x, y, strata = np.asarray(x, float), np.asarray(y, float), np.asarray(strata)
    per, undefined, cells = {}, [], []
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        if len(idx) < 3:
            undefined.append(int(s)); continue
        rx, ry = rankdata(x[idx]), rankdata(y[idx]); rxc, ryc = rx - rx.mean(), ry - ry.mean(); den = float(np.sqrt((rxc ** 2).sum() * (ryc ** 2).sum()))
        if den == 0:
            undefined.append(int(s)); continue
        per[int(s)] = {"n": int(len(idx)), "rho": float((rxc * ryc).sum() / den)}; cells.append((rxc, ryc, den, len(idx)))
    if not cells:
        raise ValueError("no stratum with a defined correlation")
    N = sum(c[3] for c in cells); obs = sum(v["n"] * v["rho"] for v in per.values()) / N
    rng = np.random.default_rng(seed); cnt = 0
    for _ in range(n_perm):
        tot = sum(n * float(rxc[rng.permutation(n)] @ ryc) / den for rxc, ryc, den, n in cells)
        cnt += abs(tot / N) >= abs(obs) - 1e-12
    return {"rho": float(obs), "p": (1 + cnt) / (n_perm + 1), "per_stratum": per, "undefined_strata": undefined, "n": int(N), "n_perm": n_perm}


def decile_table(S, y, strata, n_bins=10):
    """Survival by equal-count bin of S (ordinal ranks, ties broken by order): the raw rate and the rate standardised to the overall firing-rate-stratum shares
    (strata missing from a bin are dropped and counted)."""
    S, y, strata = np.asarray(S, float), np.asarray(y, float), np.asarray(strata); n = len(S)
    if n < n_bins:
        raise ValueError("fewer features than bins")
    rank0 = np.empty(n, int); rank0[np.argsort(S, kind="stable")] = np.arange(n); dec = rank0 * n_bins // n
    vals = np.unique(strata); share = {s: float((strata == s).mean()) for s in vals}; rows = []
    for d in range(n_bins):
        m = dec == d; num = den = 0.0; missing = 0
        for s in vals:
            c = m & (strata == s)
            if not c.any():
                missing += 1; continue
            num += share[s] * y[c].mean(); den += share[s]
        rows.append({"decile": d, "n": int(m.sum()), "mean_S": float(S[m].mean()), "raw_rate": float(y[m].mean()), "standardised_rate": num / den, "n_strata_missing": missing})
    return rows


def magnitude_strata(strata, supported, evaluable, m, n=3):
    """Combined stratum id = firing-rate stratum x n + magnitude bin; bins are equal-count within each firing-rate stratum among the EVALUABLE supported features
    (a supported feature that is not evaluable gets the bin its magnitude falls in); unsupported features keep -1."""
    strata = torch.as_tensor(strata); supported = torch.as_tensor(supported).bool(); evaluable = torch.as_tensor(evaluable).bool(); m = torch.as_tensor(m).double()
    out = torch.full_like(strata, -1)
    for s in strata.unique().tolist():
        if s < 0:
            continue
        inside = (strata == s) & supported; ev = np.sort(m[inside & evaluable].numpy())
        if inside.any():
            b = np.zeros(int(inside.sum()), dtype=int) if len(ev) == 0 else np.minimum(n - 1, np.searchsorted(ev, m[inside].numpy(), side="left") * n // len(ev))
            out[inside] = s * n + torch.as_tensor(b, dtype=out.dtype)
    return out


def reading(rate_only_delta, matched_delta, matched_holm, band=BAND, alpha=ALPHA, retain=RETAIN):
    """Pre-specified reading of the INT6 contrast with magnitude-matched controls: not significant or within +-1 pp -> 'explained by magnitude'; significant and positive ->
    'reversed'; significant, more negative than -1 pp and retaining at least half of the rate-only gap -> 'supported'; retaining less than half -> 'partly explained'."""
    if matched_holm >= alpha:
        return "explained by magnitude"
    if matched_delta > 0:
        return "reversed"
    if abs(matched_delta) <= band:
        return "explained by magnitude"
    if rate_only_delta < 0 and matched_delta / rate_only_delta < retain:
        return "partly explained"
    return "supported"


def final_reading(by_bins, min_retained=0.5):
    """Headline = the reading at the FINEST bin resolution that keeps at least `min_retained` of the selective features (common support); all readings are reported."""
    ok = [n for n in sorted(by_bins, reverse=True) if by_bins[n]["retained"] >= min_retained]; note = ""
    n = ok[0] if ok else min(by_bins)
    if not ok:
        note = "no resolution kept enough common support: the coarsest is reported"
    return {"bins": n, "reading": by_bins[n]["reading"], "retained": by_bins[n]["retained"], "all_agree": len({v["reading"] for v in by_bins.values()}) == 1, "note": note}


def descriptives(tab, selp, labels_taus, seed=0, n_perm=10000):
    """Magnitude of selective features against their controls at each threshold (controls as in N2, firing-rate strata): medians of m, and the stratum-standardised
    difference of log10 m with its within-stratum permutation p."""
    rows = []
    for label, tau in labels_taus:
        alt = th.alternative_selection(selp, tau, seed); sel_ids = set(alt["selective"].nonzero().squeeze(1).tolist()); ctl_ids = set(alt["controls"].tolist())
        feats = tab["feature"]; g = np.array([1 if f in sel_ids else 0 if f in ctl_ids else -1 for f in feats]); keep = g >= 0
        row = {"label": label, "tau": float(tau), "n_selective": int((g == 1).sum()), "n_control": int((g == 0).sum())}
        if (g == 1).any() and (g == 0).any():
            lm = np.log10(tab["m"][keep]); c = an.stratified_permutation_contrast(g[keep], lm, tab["stratum"][keep], n_perm, seed)
            row.update(median_m_selective=float(np.median(tab["m"][g == 1])), median_m_control=float(np.median(tab["m"][g == 0])), log10_m_delta=float(c["delta"]), log10_m_p=float(c["p"]))
        else:
            row["undefined"] = "one of the groups is empty among the evaluable features"
        rows.append(row)
    return rows


def dose_response(survival, selp, widths, threshold=0.001, n_perm=10000, seed=0):
    out = {}
    for b in widths:
        t = evaluable_table(survival, selp, b, threshold)
        out[str(b)] = {"n_features": int(len(t["S"])), "rho_S_survival": within_stratum_rho(t["S"], t["y"], t["stratum"], n_perm, seed), "rho_S_r": within_stratum_rho(t["S"], t["r"], t["stratum"], n_perm, seed),
                       "rho_m_survival": within_stratum_rho(t["m"], t["y"], t["stratum"], n_perm, seed), "rho_S_m": within_stratum_rho(t["S"], t["m"], t["stratum"], n_perm, seed),
                       "deciles": decile_table(t["S"], t["y"], t["stratum"], 10)}
    return out


def _t2(row):
    return row["tests"]["T2_int6"], row["holm"]["T2_int6"]


def run_magnitude(run_dir, roster_path, out_dir, taus, widths=(7, 6, 5), bins=(3, 5, 10), n_perm=10000, n_boot=2000, seed=0, rate_only_json=None, threshold=0.001):
    """(a) dose-response, (b) descriptives, (c) the rate-only rows (N2, read from `rate_only_json` if given, else recomputed) and the magnitude-matched rows at every bin
    resolution, with the pre-specified reading per threshold and resolution; writes magnitude.json / magnitude.md."""
    run_dir, out_dir = Path(run_dir), Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    surv = torch.load(run_dir / "survival.pt"); selp = torch.load(run_dir / "selectivity.pt"); taus = [float(t) for t in taus]
    plan = [("null", float(selp["null_cutoff"]))] + [(str(t), t) for t in taus]
    peak = _np(selp["peak"]); idx = np.arange(len(peak)); supported = _np(selp["supported"]).astype(bool)
    ev_full = supported & (_np(surv["firing_rate"])[peak, idx] > threshold); m_full = _np(surv["max_activation"]).astype(float)[peak, idx]
    res = {"decision": "D-020", "run": str(run_dir), "seed": seed, "n_perm": n_perm, "n_boot": n_boot, "taus": taus, "widths": list(widths), "bins": list(bins),
           "dose_response": dose_response(surv, selp, widths, threshold, n_perm, seed), "descriptives": descriptives(evaluable_table(surv, selp, 6, threshold), selp, plan, seed, n_perm)}
    res["rate_only"] = json.loads(Path(rate_only_json).read_text()) if rate_only_json else th.run_thresholds(run_dir, roster_path, out_dir / "rate_only", taus, n_perm=n_perm, n_boot=n_boot, seed=seed)
    res["matched"] = {}
    for n in bins:
        tf = lambda s, n=n: {**s, "strata": magnitude_strata(s["strata"], s["supported"], ev_full, m_full, n)}
        res["matched"][str(n)] = th.run_thresholds(run_dir, roster_path, out_dir / f"matched_{n}", taus, n_perm=n_perm, n_boot=n_boot, seed=seed, transform=tf)
    res["readings"] = {}
    rate_by_label = {r["label"]: r for r in res["rate_only"]["rows"]}                     # matched by label: a reference file may hold other thresholds (N2 has 2.0 as well)
    for i, (label, _) in enumerate(plan):
        rate_row = rate_by_label[label]; by = {}
        for n in bins:
            row = res["matched"][str(n)]["rows"][i]
            if "tests" not in row or "tests" not in rate_row:
                continue
            (t_m, h_m), (t_r, _) = _t2(row), _t2(rate_row)
            by[n] = {"reading": reading(t_r["delta"], t_m["delta"], h_m), "retained": 1.0 - t_m["n_selective_dropped"] / t_m["n_selective"], "rate_only_delta": t_r["delta"], "matched_delta": t_m["delta"], "matched_holm": h_m}
        res["readings"][label] = {"by_bins": {str(k): v for k, v in by.items()}, "final": final_reading(by) if by else None}
    (out_dir / "magnitude.json").write_text(json.dumps(res, indent=1, default=float) + "\n"); (out_dir / "magnitude.md").write_text(render_md(res)); return res


def render_md(res):
    L = ["# Activation-magnitude check of the selectivity dose-response (decision D-020)", "",
         "Question: is the N2 gap (selective features survive less than firing-rate-matched controls, growing with the S threshold) a selectivity effect or an activation-magnitude effect? "
         "Source: `issues.md` C6 / `PLAN.md` P1-5 (Duan et al.'s released per-feature data: survival follows maximum activation, Spearman 0.78, not firing rate, -0.085); the proposal matches controls on firing-rate quintile only. "
         "Magnitude m = maximum activation at the feature's peak language (full-precision stream); survival = r > 0.9 there. No new estimator: (a) threshold-free within-stratum dose-response, (b) descriptive magnitude comparison, "
         "(c) the unchanged D-014 / D-016 analysis with controls matched on firing-rate stratum x magnitude bin (3 bins = terciles, pre-specified; 5 and 10 added before any real number because coarse bins leave residual confounding when "
         "survival follows magnitude steeply). Thresholds are the fixed S thresholds of N2 (labelled deviation, D-019).", "",
         "**Pre-specified reading rule** (INT6 contrast at the threshold; written before the numbers): not significant (Holm-adjusted p >= 0.05) or within +-1 pp -> *explained by magnitude*; significant (Holm < 0.05) and positive -> *reversed*; "
         "significant (Holm < 0.05), more negative than -1 pp and retaining at least half of the rate-only gap -> *supported* (selectivity reading); significant but retaining less than half of the rate-only gap -> *partly explained*. "
         "Headline reading = the finest bin resolution that keeps at least half of the selective features (common support); all resolutions are shown.", "", "## (a) Threshold-free dose-response (S against survival inside firing-rate strata)", "",
         "| width | features | rho(S, survival) | p | rho(S, r) | rho(m, survival) | rho(S, m) |", "|---|---|---|---|---|---|---|"]
    for w, d in res["dose_response"].items():
        L.append(f"| INT{w} | {d['n_features']} | {d['rho_S_survival']['rho']:+.3f} | {d['rho_S_survival']['p']:.3g} | {d['rho_S_r']['rho']:+.3f} | {d['rho_m_survival']['rho']:+.3f} | {d['rho_S_m']['rho']:+.3f} |")
    L += ["", "Stratum-standardised survival by decile of S (INT6):", "", "| decile of S | features | mean S | raw survival | standardised survival |", "|---|---|---|---|---|"]
    for r in res["dose_response"].get("6", next(iter(res["dose_response"].values())))["deciles"]:
        L.append(f"| {r['decile'] + 1} | {r['n']} | {r['mean_S']:.2f} | {r['raw_rate']:.3f} | {r['standardised_rate']:.3f} |")
    L += ["", "## (b) Magnitude of selective features against their controls", "", "| threshold | selective | controls | median max activation (selective / control) | standardised difference of log10 m (p) |", "|---|---|---|---|---|"]
    for r in res["descriptives"]:
        L.append(f"| {r['label']} | {r['n_selective']} | {r['n_control']} | " + ("undefined |  |" if "undefined" in r else f"{r['median_m_selective']:.3g} / {r['median_m_control']:.3g} | {r['log10_m_delta']:+.3f} ({r['log10_m_p']:.3g}) |"))
    L += ["", "## (c) INT6 contrast, firing-rate matching only against firing-rate x magnitude matching", "", "| threshold | matching | INT6 selective - control (p, Holm) | selective kept | reading |", "|---|---|---|---|---|"]
    rate_by_label = {r["label"]: r for r in res["rate_only"]["rows"]}
    for label, rd in res["readings"].items():
        rate = rate_by_label[label]
        if "tests" in rate:
            t, h = _t2(rate); L.append(f"| {label} | firing rate only (N2) | {t['delta'] * 100:+.2f} pp (p {t['p']:.3g}, Holm {h:.3g}) | | |")
        for n, v in rd["by_bins"].items():
            L.append(f"| {label} | rate x {n} magnitude bins | {v['matched_delta'] * 100:+.2f} pp (Holm {v['matched_holm']:.3g}) | {v['retained'] * 100:.0f} % | {v['reading']} |")
        if rd["final"]:
            f = rd["final"]; L.append(f"| **{label}** | **headline ({f['bins']} bins)** | | {f['retained'] * 100:.0f} % | **{f['reading']}**{' (all resolutions agree)' if f['all_agree'] else ' (resolutions disagree)'} {f['note']} |")
    return "\n".join(L) + "\n"


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=run_magnitude.__doc__)
    ap.add_argument("--run", required=True); ap.add_argument("--out", required=True); ap.add_argument("--rate-only-json", default=None)
    ap.add_argument("--roster", default=str(Path(__file__).resolve().parents[2] / "reports/g0/phase1_languages.json"))
    ap.add_argument("--taus", type=float, nargs="*", default=[3.0, 5.0]); ap.add_argument("--bins", type=int, nargs="*", default=[3, 5, 10])
    ap.add_argument("--n-perm", type=int, default=10000); ap.add_argument("--n-boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv); r = run_magnitude(a.run, a.roster, a.out, a.taus, bins=tuple(a.bins), n_perm=a.n_perm, n_boot=a.n_boot, seed=a.seed, rate_only_json=a.rate_only_json)
    for label, rd in r["readings"].items():
        print(f"tau {label}: " + "; ".join(f"{n} bins {v['matched_delta'] * 100:+.2f} pp -> {v['reading']} (kept {v['retained'] * 100:.0f}%)" for n, v in rd["by_bins"].items()) + (f"  HEADLINE {rd['final']['reading']}" if rd["final"] else ""))
    print("wrote", Path(a.out) / "magnitude.md")


if __name__ == "__main__":
    main()
