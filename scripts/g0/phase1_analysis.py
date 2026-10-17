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
