"""Phase 1 inference (decision D-014; proposal sections 6-7), numpy only: feature table, within-stratum permutation contrast, cluster bootstrap, per-language
gaps, Freedman-Lane tier test with script covariate, Holm, TOST. Tested on synthetic data with planted effects (tests_torch/test_phase1_analysis.py)."""
import numpy as np

CUT = 0.9


def survived(r, cut=CUT):
    return np.asarray(r) > cut


def feature_table(corr, fire, sel, meta, bits_list, threshold=0.001, features=None):
    """Selective features and their matched controls, each read at its peak language (D-013): r at every width, whether the feature is evaluable there
    (full-precision devtest firing rate above their threshold), stratum, rank score, tier and script of the peak language."""
    feats = np.concatenate([np.flatnonzero(np.asarray(sel["selective"])), np.asarray(sel["controls"], dtype=int)]) if features is None else np.asarray(features, dtype=int)
    lang = np.asarray(sel["peak"])[feats]
    pick = lambda key: np.array([meta[l][key] for l in lang])
    return {"feature": feats, "group": np.asarray(sel["selective"])[feats].astype(int), "lang": lang, "stratum": np.asarray(sel["strata"])[feats],
            "S_rank": np.asarray(sel["S_rank"])[feats], "r": {b: np.asarray(corr[b])[lang, feats] for b in bits_list}, "evaluable": np.asarray(fire)[lang, feats] > threshold,
            "tier": pick("tier"), "script_family": pick("script_family"), "joshi_class": pick("joshi_class")}


def _delta(group, y):
    return float(y[group == 1].mean() - y[group == 0].mean())


def stratified_permutation_contrast(group, y, strata, n_perm=10000, seed=0):
    """Two-sided permutation test of the difference of group means; labels are permuted WITHIN strata (the quintile-matching of the proposal)."""
    group, y, strata = np.asarray(group), np.asarray(y, dtype=float), np.asarray(strata)
    if not ((group == 1).any() and (group == 0).any()):
        raise ValueError("both groups are needed")
    rng = np.random.default_rng(seed); cells = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    obs, cnt = _delta(group, y), 0
    for _ in range(n_perm):
        g = group.copy()
        for idx in cells:
            g[idx] = group[idx][rng.permutation(len(idx))]
        cnt += abs(_delta(g, y)) >= abs(obs) - 1e-12
    return {"delta": obs, "p": (1 + cnt) / (n_perm + 1), "n_selective": int((group == 1).sum()), "n_control": int((group == 0).sum()), "n_perm": n_perm}


def _bootstrap_indices(group, lang, rng):
    """Resample features with replacement within every (language, group) cell: cell sizes stay fixed (cluster bootstrap over features within language)."""
    out = []
    for lg in np.unique(lang):
        for g in (0, 1):
            idx = np.flatnonzero((lang == lg) & (group == g))
            if len(idx):
                out.append(idx[rng.integers(0, len(idx), len(idx))])
    return np.concatenate(out)


def cluster_bootstrap_ci(group, y, lang, n_boot=2000, seed=0, alpha=0.05):
    group, y, lang = np.asarray(group), np.asarray(y, dtype=float), np.asarray(lang)
    rng = np.random.default_rng(seed)
    d = [_delta(group[i], y[i]) for i in (_bootstrap_indices(group, lang, rng) for _ in range(n_boot))]
    return float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))


def language_gaps(lang, group, r, evaluable, n_lang, min_features=10):
    """Per language: median r of the evaluable selective features minus median r of the evaluable matched controls (NaN if either set is empty);
    `populated` = at least `min_features` evaluable selective features."""
    lang, group, r, evaluable = map(np.asarray, (lang, group, r, evaluable))
    gap, ns, nc = np.full(n_lang, np.nan), np.zeros(n_lang, int), np.zeros(n_lang, int)
    for l in range(n_lang):
        s, c = r[(lang == l) & (group == 1) & evaluable], r[(lang == l) & (group == 0) & evaluable]
        ns[l], nc[l] = len(s), len(c)
        if len(s) and len(c):
            gap[l] = np.median(s) - np.median(c)
    return {"gap": gap, "n_selective": ns, "n_control": nc, "populated": (ns >= min_features) & (nc >= 1)}


def _design(script):
    cats = np.unique(script)
    return np.column_stack([np.ones(len(script))] + [(script == c).astype(float) for c in cats[1:]])


def freedman_lane_tier_test(gap, tier_score, script, n_perm=10000, seed=0):
    """One-sided (positive slope = lower gap for lower tiers) permutation test of the slope of the per-language gap on the ordinal tier, with script-family
    dummies as nuisance covariates (Freedman and Lane 1983: permute the residuals of the reduced model, refit the full model, compare t statistics)."""
    gap, tier, script = np.asarray(gap, dtype=float), np.asarray(tier_score, dtype=float), np.asarray(script)
    n = len(gap); Z = _design(script); X = np.column_stack([Z, tier])
    Hz = Z @ np.linalg.pinv(Z); tier_r = tier - Hz @ tier; ss = float(tier_r @ tier_r)
    fit_z = Hz @ gap; res_z = gap - fit_z; Hx = X @ np.linalg.pinv(X); dof = n - np.linalg.matrix_rank(X)

    def slope_t(y):
        b = float(tier_r @ y) / ss; rss = float(((y - Hx @ y) ** 2).sum())
        return b, b / np.sqrt(rss / dof / ss) if rss > 0 else (np.inf if b > 0 else -np.inf if b < 0 else 0.0)
    b_obs, t_obs = slope_t(gap)
    rng = np.random.default_rng(seed); cnt = 0
    for _ in range(n_perm):
        cnt += slope_t(fit_z + res_z[rng.permutation(n)])[1] >= t_obs - 1e-12
    hi, lo = gap[tier == 2], gap[tier == 0]
    return {"slope": b_obs, "t": float(t_obs), "p": (1 + cnt) / (n_perm + 1), "n": n, "n_perm": n_perm,
            "high_minus_low": float(hi.mean() - lo.mean()) if len(hi) and len(lo) else float("nan")}


def partial_slope(gap, tier, script):
    """OLS coefficient of the tier score in gap ~ script dummies + tier (Frisch-Waugh: regress on the tier residualised on the script dummies)."""
    gap, tier = np.asarray(gap, dtype=float), np.asarray(tier, dtype=float); Z = _design(np.asarray(script))
    tr = tier - Z @ np.linalg.pinv(Z) @ tier
    return float(tr @ gap / (tr @ tr))


def holm(pvals):
    p = np.asarray(pvals, dtype=float); m = len(p); order = np.argsort(p); adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * p[i])); adj[i] = run
    return adj


def tost(boot, bound, alpha=0.05):
    """Equivalence by the confidence-interval approach to TOST: the (1 - 2 alpha) bootstrap interval must lie inside (-bound, bound)."""
    lo, hi = float(np.quantile(boot, alpha)), float(np.quantile(boot, 1 - alpha))
    return {"interval": (lo, hi), "equivalent": bool(lo > -bound and hi < bound), "bound": bound}


TIER_SCORE = {"low": 0, "mid": 1, "high": 2}


def tier_score(name):
    return TIER_SCORE[str(name).lower()]
MIXED_FORMULA = "bits_c + C(stratum) + S_rank + tier + S_rank:tier"


def _fit_random_intercept_logit(X, Y, n_nodes=25):
    """Maximum likelihood of logit Pr(Y_kf = 1) = X_kf beta + u_f, u_f ~ N(0, sigma^2), by Gauss-Hermite quadrature over u_f (what lme4's glmer does with
    nAGQ = n_nodes for a scalar random intercept). X [K, F, p], Y [K, F] in {0, 1}. Gradient and Hessian by autograd; standard errors from the inverse Hessian
    (pseudo-inverse if sigma collapses to 0). Returns names-free arrays: beta, se, sigma, converged, nll."""
    import torch
    from scipy.optimize import minimize
    nodes, weights = np.polynomial.hermite_e.hermegauss(n_nodes)
    z = torch.tensor(nodes, dtype=torch.float64); logw = torch.log(torch.tensor(weights / np.sqrt(2 * np.pi), dtype=torch.float64))
    Xt, sign = torch.tensor(X, dtype=torch.float64), 2 * torch.tensor(Y, dtype=torch.float64) - 1
    p = X.shape[2]

    def nll(theta):
        lin = (Xt @ theta[:p])[..., None] + torch.exp(theta[p]) * z                      # [K, F, nodes]
        return -torch.logsumexp(torch.nn.functional.logsigmoid(sign[..., None] * lin).sum(0) + logw, dim=1).sum()

    def fun(x):
        th = torch.tensor(x, dtype=torch.float64, requires_grad=True); v = nll(th); v.backward()
        return v.item(), th.grad.numpy().copy()

    x0 = np.zeros(p + 1); x0[p] = np.log(0.5)
    res = minimize(fun, x0, jac=True, method="L-BFGS-B", options={"maxiter": 500, "maxfun": 2000})
    th = torch.tensor(res.x, dtype=torch.float64)
    cov = np.linalg.pinv(torch.autograd.functional.hessian(nll, th).numpy())
    return {"beta": res.x[:p], "se": np.sqrt(np.clip(np.diag(cov)[:p], 0, None)), "sigma": float(np.exp(res.x[p])),
            "converged": bool(res.success or np.abs(res.jac).max() < 1e-3), "nll": float(res.fun)}


def mixed_logit(table, bits_list, max_features=None, seed=0, gee=True):
    """Section 7 model: logit Pr(survived) = b0 + b_bits * bits + b_q * quintile + b_S * S~ + b_t * tier + b_St * S~ x tier + u_feature (random intercept per
    feature), fitted by quadrature maximum likelihood (above); as a robustness check also as a population-averaged logistic GEE with an exchangeable correlation
    within feature (statsmodels). Non-evaluable features are dropped; with `max_features` a seeded random subset is used. The quintile enters as a categorical,
    the tier as the ordinal 0 / 1 / 2, bits centred at the mean width."""
    import math
    import pandas as pd
    import patsy
    keep = np.flatnonzero(np.asarray(table["evaluable"]))
    if max_features is not None and len(keep) > max_features:
        keep = np.sort(np.random.default_rng(seed).choice(keep, max_features, replace=False))
    centre = float(np.mean(bits_list))
    df = pd.concat([pd.DataFrame({"feature": np.asarray(table["feature"])[keep], "y": survived(np.asarray(table["r"][b])[keep]).astype(int), "bits_c": b - centre,
                                  "S_rank": np.asarray(table["S_rank"])[keep], "tier": [tier_score(t) for t in np.asarray(table["tier"])[keep]],
                                  "stratum": np.asarray(table["stratum"])[keep]}) for b in bits_list], ignore_index=True)
    X = patsy.dmatrix(MIXED_FORMULA, df, return_type="dataframe"); names = list(X.columns); K, F = len(bits_list), len(keep)
    fit = _fit_random_intercept_logit(X.to_numpy().reshape(K, F, -1), df["y"].to_numpy().reshape(K, F))
    z = fit["beta"] / np.where(fit["se"] > 0, fit["se"], np.nan)
    out = {"mixed": {"names": names, "coef": [float(x) for x in fit["beta"]], "se": [float(x) for x in fit["se"]], "z": [float(x) for x in z],
                     "p": [float(math.erfc(abs(v) / math.sqrt(2))) if v == v else float("nan") for v in z], "sigma": fit["sigma"], "converged": fit["converged"], "nll": fit["nll"]},
           "gee": None, "n_features": int(F), "n_obs": int(len(df))}
    if gee:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        g = smf.gee("y ~ " + MIXED_FORMULA, groups="feature", data=df, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
        out["gee"] = {"names": list(g.params.index), "coef": [float(x) for x in g.params], "se": [float(x) for x in g.bse], "p": [float(x) for x in g.pvalues]}
    return out


def _py(x):
    """JSON-friendly copy (numpy scalars / arrays, tuples, torch-free)."""
    if isinstance(x, dict):
        return {str(k): _py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_py(v) for v in x]
    if isinstance(x, np.ndarray):
        return _py(x.tolist())
    if isinstance(x, (np.floating, float)):
        return None if x != x else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _avg_gaps(table, bits, mask, n_lang, min_features, idx=None):
    """Per-language median-r gap averaged over `bits` (and the matching survival-rate gap); `idx` resamples the evaluable rows (bootstrap)."""
    sel = np.flatnonzero(mask) if idx is None else np.flatnonzero(mask)[idx]
    g, lang = table["group"][sel], table["lang"][sel]
    gaps = [language_gaps(lang, g, table["r"][b][sel], np.ones(len(sel), bool), n_lang, min_features) for b in bits]
    rate = [language_gaps(lang, g, survived(table["r"][b][sel]).astype(float), np.ones(len(sel), bool), n_lang, min_features) for b in bits]
    return {"gap": np.nanmean([x["gap"] for x in gaps], axis=0) if len(gaps) else None, "rate_gap": np.nanmean([x["gap"] for x in rate], axis=0),
            "n_selective": gaps[0]["n_selective"], "populated": gaps[0]["populated"]}


def run_analysis(run_dir, roster_path, out_dir, n_perm=10000, n_boot=2000, seed=0, min_features=10, min_languages=20, tost_median_r=0.02, tost_rate=0.05, mixed_cap=None):
    """Confirmatory family (INT7 and INT6 selectivity contrasts, language-level tier test, Holm; decision D-014) plus labelled exploratory sections, from the files
    phase1_run.save_outputs wrote. Writes analysis.json and analysis.md into out_dir and returns the result dict."""
    import json
    from pathlib import Path
    import torch
    from scipy.stats import mannwhitneyu
    run_dir, out_dir = Path(run_dir), Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    surv = torch.load(run_dir / "survival.pt"); selp = torch.load(run_dir / "selectivity.pt"); meta = json.loads((run_dir / "meta.json").read_text())
    roster = {l["code"]: l for l in json.loads(Path(roster_path).read_text())["languages"]}
    langs = list(surv["languages"]); lm = [roster[c] for c in langs]; n_lang = len(langs); bits = list(surv["bits"])
    corr = {b: surv["corr"][b].numpy() for b in bits}; fire = surv["firing_rate"].numpy()
    sel = {k: selp[k].numpy() for k in ("selective", "supported", "peak", "S_rank", "strata", "controls")}
    table = feature_table(corr, fire, sel, lm, bits); ev = table["evaluable"]
    tier = np.array([tier_score(l["tier"]) for l in lm]); joshi = np.array([l["joshi_class"] for l in lm], float)
    fam = sorted({l["script_family"] for l in lm}); script = np.array([fam.index(l["script_family"]) for l in lm])

    def contrast(b):
        m = ev; y = survived(table["r"][b][m]).astype(float); g = table["group"][m]
        r = stratified_permutation_contrast(g, y, table["stratum"][m], n_perm, seed)
        r["ci"] = cluster_bootstrap_ci(g, y, table["lang"][m], n_boot, seed); r["survival_selective"] = float(y[g == 1].mean()); r["survival_control"] = float(y[g == 0].mean())
        r["n_dropped_not_evaluable"] = int((~ev).sum()); return r
    t1, t2 = contrast(7), contrast(6)

    ag = _avg_gaps(table, (7, 6), ev, n_lang, min_features); used = ag["populated"] & ~np.isnan(ag["gap"]); enough = bool(used.sum() >= min_languages)
    t3 = {"n_languages": int(used.sum()), "enough_languages": enough, "min_features": min_features, "min_languages": min_languages}
    if used.sum() >= 4 and len(set(tier[used])) > 1:
        fl = freedman_lane_tier_test(ag["gap"][used], tier[used], script[used], n_perm, seed); t3.update(fl)
        rng = np.random.default_rng(seed); mask_idx = np.flatnonzero(ev); slopes, hl = [], []
        for _ in range(n_boot):
            idx = _bootstrap_indices(table["group"][mask_idx], table["lang"][mask_idx], rng)
            bg = _avg_gaps(table, (7, 6), ev, n_lang, min_features, idx=idx)["gap"][used]
            if np.isnan(bg).any():
                continue
            slopes.append(partial_slope(bg, tier[used], script[used])); hl.append(bg[tier[used] == 2].mean() - bg[tier[used] == 0].mean() if (tier[used] == 2).any() and (tier[used] == 0).any() else np.nan)
        t3["tost_slope"] = tost(np.array(slopes), tost_median_r); t3["tost_high_minus_low"] = tost(np.array(hl)[~np.isnan(hl)], tost_median_r)
        rate_used = ag["populated"] & ~np.isnan(ag["rate_gap"]); t3["survival_rate_gap_sensitivity"] = freedman_lane_tier_test(ag["rate_gap"][rate_used], tier[rate_used], script[rate_used], n_perm, seed)
        t3["joshi_class_sensitivity"] = freedman_lane_tier_test(ag["gap"][used], joshi[used], script[used], n_perm, seed)
    else:
        t3.update({"slope": None, "p": 1.0, "note": "too few populated languages or a single tier"})
    t3["languages"] = [{"code": langs[i], "tier": lm[i]["tier"], "joshi_class": lm[i]["joshi_class"], "gap_median_r": ag["gap"][i], "gap_survival_rate": ag["rate_gap"][i],
                        "n_selective": int(ag["n_selective"][i]), "used": bool(used[i])} for i in range(n_lang)]
    ps = [t1["p"], t2["p"], t3["p"]]; adj = holm(ps)
    result = {"regime": meta.get("regime"), "bits": bits, "n_perm": n_perm, "n_boot": n_boot, "seed": seed, "decision": "D-014",
              "tests": {"T1_int7": t1, "T2_int6": t2, "T3_tier": t3}, "holm": dict(zip(("T1_int7", "T2_int6", "T3_tier"), (float(x) for x in adj)))}

    exploratory = {"label": "not part of the confirmatory family; p-values are uncorrected", "contrasts": {}}
    for b in (5, 4):
        if b in table["r"]:
            exploratory["contrasts"][f"int{b}"] = contrast(b)
    sup = np.flatnonzero(sel["supported"]); st = feature_table(corr, fire, sel, lm, [b for b in (7, 6) if b in corr], features=sup)
    exploratory["mixed_logit"] = mixed_logit(st, [b for b in (7, 6) if b in corr], max_features=mixed_cap, seed=seed)
    rate = selp["rates"].numpy().mean(0); a, c = rate[sel["selective"]], rate[sel["supported"] & ~sel["selective"]]
    exploratory["P1_firing_rate"] = {"mean_rate_selective": float(a.mean()) if len(a) else None, "mean_rate_other_supported": float(c.mean()) if len(c) else None,
                                     "mannwhitney_p_selective_lower": float(mannwhitneyu(a, c, alternative="less").pvalue) if len(a) and len(c) else None}
    result["exploratory"] = exploratory
    result["controls"] = {"label": "survival of the same features at the noise-floor widths INT16, INT12 and restore-only INT0 (all should be at the ceiling)",
                          "survival_at": {f"int{b}": {"selective": float(survived(table["r"][b][ev & (table["group"] == 1)]).mean()), "control": float(survived(table["r"][b][ev & (table["group"] == 0)]).mean())}
                                          for b in (16, 12, 0) if b in table["r"]}}
    result = _py(result)
    (out_dir / "analysis.json").write_text(json.dumps(result, indent=1) + "\n"); (out_dir / "analysis.md").write_text(render_markdown(result))
    return result


def render_markdown(r):
    t = r["tests"]; L = [f"# Phase 1 analysis, regime `{r['regime']}` (decision D-014)", "", f"{r['n_perm']} permutations, {r['n_boot']} bootstrap draws, seed {r['seed']}.", "",
                         "## Confirmatory family (Holm over three tests)", "", "| test | estimate | 95% CI / detail | p | Holm p |", "|---|---|---|---|---|"]
    for k, nm in (("T1_int7", "INT7 selective minus matched-universal survival"), ("T2_int6", "INT6 selective minus matched-universal survival")):
        x = t[k]; L.append(f"| {nm} | {x['delta']:+.4f} | [{x['ci'][0]:+.4f}, {x['ci'][1]:+.4f}] (selective {x['survival_selective']:.4f}, control {x['survival_control']:.4f}, n {x['n_selective']}+{x['n_control']}) | {x['p']:.4g} | {r['holm'][k]:.4g} |")
    x = t["T3_tier"]
    L.append(f"| language-level tier slope (median-r gap, INT7+INT6, one-sided) | {x['slope'] if x.get('slope') is None else format(x['slope'], '+.4f')} | languages used {x['n_languages']} (need {x['min_languages']}: {'ok' if x['enough_languages'] else 'P2 DEMOTED to exploratory'}); High minus Low {x.get('high_minus_low')} | {x['p']:.4g} | {r['holm']['T3_tier']:.4g} |")
    if "tost_slope" in x:
        L += ["", "TOST (90% bootstrap interval vs bound): slope " + f"{x['tost_slope']['interval'][0]:+.4f}..{x['tost_slope']['interval'][1]:+.4f}, equivalent within +-{x['tost_slope']['bound']}: {x['tost_slope']['equivalent']}; "
              f"High minus Low {x['tost_high_minus_low']['interval'][0]:+.4f}..{x['tost_high_minus_low']['interval'][1]:+.4f}, equivalent: {x['tost_high_minus_low']['equivalent']}."]
    L += ["", "### Per-language gaps (median r of selective minus matched controls, mean of INT7 and INT6)", "", "| language | tier | Joshi | selective features | gap (median r) | gap (survival rate) | used |", "|---|---|---|---|---|---|---|"]
    for l in x["languages"]:
        f = lambda v: "" if v is None else f"{v:+.4f}"
        L.append(f"| {l['code']} | {l['tier']} | {l['joshi_class']} | {l['n_selective']} | {f(l['gap_median_r'])} | {f(l['gap_survival_rate'])} | {'yes' if l['used'] else 'no'} |")
    e = r["exploratory"]; L += ["", f"## Exploratory analyses ({e['label']})", "", "| width | selective minus control | 95% CI | p (uncorrected) |", "|---|---|---|---|"]
    for k, v in e["contrasts"].items():
        L.append(f"| {k.upper()} | {v['delta']:+.4f} | [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}] | {v['p']:.4g} |")
    m = e["mixed_logit"]["mixed"]; L += ["", f"Mixed-effects logistic model (random intercept per feature, quadrature ML; {e['mixed_logit']['n_features']} features, sigma {m['sigma']:.3f}, converged {m['converged']}):", "",
                                         "| term | estimate | SE | p |", "|---|---|---|---|"]
    L += [f"| {n} | {c:+.4f} | {s_:.4f} | {p:.4g} |" if p is not None else f"| {n} | {c:+.4f} | {s_:.4f} | |" for n, c, s_, p in zip(m["names"], m["coef"], m["se"], m["p"])]
    L += ["", "P1 (selective features fire less): mean panel firing rate selective %.5f, other supported %.5f, Mann-Whitney p %.4g." % (e["P1_firing_rate"]["mean_rate_selective"], e["P1_firing_rate"]["mean_rate_other_supported"], e["P1_firing_rate"]["mannwhitney_p_selective_lower"]),
          "", f"## Instrument check: {r['controls']['label']}", "", "| width | selective | control |", "|---|---|---|"]
    L += [f"| {k.upper()} | {v['selective']:.4f} | {v['control']:.4f} |" for k, v in r["controls"]["survival_at"].items()]
    return "\n".join(L) + "\n"
