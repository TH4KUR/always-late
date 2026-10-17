"""Phase 1 inference (decision D-014; proposal sections 6-7), numpy only: feature table, within-stratum permutation contrast, cluster bootstrap, per-language
gaps, Freedman-Lane tier test with script covariate, Holm, TOST. Tested on synthetic data with planted effects (tests_torch/test_phase1_analysis.py)."""
import numpy as np

CUT = 0.9


def survived(r, cut=CUT):
    return np.asarray(r) > cut


def feature_table(corr, fire, sel, meta, bits_list, threshold=0.001):
    """Selective features and their matched controls, each read at its peak language (D-013): r at every width, whether the feature is evaluable there
    (full-precision devtest firing rate above their threshold), stratum, rank score, tier and script of the peak language."""
    feats = np.concatenate([np.flatnonzero(np.asarray(sel["selective"])), np.asarray(sel["controls"], dtype=int)])
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


TIER_SCORE = {"Low": 0, "Mid": 1, "High": 2}
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
                                  "S_rank": np.asarray(table["S_rank"])[keep], "tier": [TIER_SCORE[t] for t in np.asarray(table["tier"])[keep]],
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
