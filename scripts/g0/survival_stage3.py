"""Stage 3 of the survival measurement-validity study (D-117): the frozen analysis of the Stage 2 sidecar sums. Rules: reports/survival_measurement/analysis_freeze.json (D-120, sha256 in
analysis_freeze.sha256, committed before any Stage 2 output existed); this module is the generator and refuses to run on a changed freeze. The functions here are pure numpy on the saved sums
(tested in tests_torch/test_survival_stage3.py on sums written by the real SurvivalSink); `main` (added with the Stage 2 data) reads the validated job directories."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_analysis as an  # noqa: E402
import survival_effect as se  # noqa: E402
import survival_lambda as sl  # noqa: E402
import survival_sidecar as sc  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MIN_RATE, MIN_TOKENS = 0.001, 20                    # FP-active cell: firing rate above 0.001 and at least 20 FP-active tokens (freeze, population)
IDENTITY_TOL = 1e-4                                 # identity gate (freeze, gates)
H3_NU_MAX, H3_LAMBDA_MIN = 0.2, 0.5                 # H3 (the plan's proposed thresholds)
H4_BOUND = 0.20                                     # equivalence bound in standardized slope units (D-120)
N_BOOT, SEED = 2000, 20261029                       # language cluster bootstrap
MAX_DROPPED_SHARE = 0.05                            # more degenerate replicates than this: the interval is not estimable


class FreezeError(RuntimeError):
    pass


def check_freeze(directory=ROOT / "reports/survival_measurement"):
    """The generator runs only on the committed freeze: sha256 of analysis_freeze.json must equal the one in analysis_freeze.sha256."""
    d = Path(directory); want = (d / "analysis_freeze.sha256").read_text().split()[0]; got = hashlib.sha256((d / "analysis_freeze.json").read_bytes()).hexdigest()
    if got != want:
        raise FreezeError(f"analysis_freeze.json (sha256 {got[:16]}...) is not the frozen file ({want[:16]}...): a deviation is written into the report, the freeze is never edited")
    return {"ok": True, "sha256": got}


# ------------------------------------------------------------------ per-cell quantities from the saved sums
def derived(sums, theta):
    """Every per-(language, feature) quantity of the freeze from one condition's sums {"stats": {name: [L, d]}, "n": [L]} and theta [d]; numpy float64, NaN where undefined."""
    st = {k: v.numpy() for k, v in sums["stats"].items()}; n = sums["n"].numpy().astype(np.float64)[:, None]; th = theta.double().numpy()[None, :]
    S = {k: st[k].astype(np.float64) for k in sc.STAT_NAMES}
    with np.errstate(all="ignore"):
        mean_a, mean_b = S["Sa"] / n, S["Sb"] / n
        var_a, var_b = np.maximum(S["Saa"] / n - mean_a ** 2, 0.0), np.maximum(S["Sbb"] / n - mean_b ** 2, 0.0); cov = S["Sab"] / n - mean_a * mean_b
        r = sc.corr_from_sums(sums).numpy(); lam, clipped = sl.lam(r)
        g = np.where(var_a > 0, cov / var_a, np.nan); c = mean_b - g * mean_a; sigma_e2 = var_b - g ** 2 * var_a
        n_a = S["n_a"]; nu = np.where(n_a > 0, np.sqrt(S["S_dpre2"] / n_a) / th, np.nan)
        den2 = (S["Saa"] - S["Sa"] ** 2 / n) * (S["Sbb"] - S["Sb"] ** 2 / n)
        d = {"n": n[:, 0], "sigma_a": np.sqrt(var_a), "sigma_b": np.sqrt(var_b), "var_a": var_a, "var_b": var_b, "r": r, "lam": lam, "clipped": clipped, "clipped_low": clipped & ~(r > sl.R_HI),
             "g": g, "c": c, "sigma_e2": sigma_e2, "nu": nu, "log_nu": np.log(np.maximum(nu, 1e-12)), "on_off": np.where(n_a > 0, S["n_on_off"] / n_a, np.nan),
             "off_on": np.where(n - n_a > 0, S["n_off_on"] / (n - n_a), np.nan), "exact_flip": np.where(n_a > 0, S["n_exceed1"] / n_a, np.nan), "pred_flip": np.where(n_a > 0, S["n_exceed2"] / n_a, np.nan),
             "mean_margin": np.where(n_a > 0, S["S_d"] / n_a / th, np.nan), "silent": S["n_b"] == 0, "n_a": n_a, "den_clamped": den2 < 1e-12}
    d["identity_ok"] = (var_a > 0) & (var_b > 0) & (sigma_e2 > 0) & (np.abs(r) < 1.0) & ~d["den_clamped"] & (g != 0) & np.isfinite(g)
    return d


def fp_active(n_a, n, min_rate=MIN_RATE, min_tokens=MIN_TOKENS):
    """FP-active cells: rate n_a / n above min_rate AND n_a at least min_tokens. Depends on the full-precision model only: the same set in every condition and arm."""
    n_a = np.asarray(n_a, dtype=np.float64); n = np.asarray(n, dtype=np.float64)[:, None]
    return (n_a / n > min_rate) & (n_a >= min_tokens)


def populations(d, active):
    """Per quantity (freeze): nu, flips, margin, lambda (clip convention) use EVERY FP-active cell with no condition on the quantized side; the decomposition (g, sigma_e) needs g != 0 and sigma_e^2 > 0."""
    decomp = active & np.isfinite(d["g"]) & (d["g"] != 0) & (d["sigma_e2"] > 0)
    return {"nu": active.copy(), "lambda": active.copy(), "decomposition": decomp, "silent_count": int((d["silent"] & active).sum()),
            "clipped_low": int((d["clipped_low"] & active).sum()), "clipped_high": int((d["clipped"] & ~d["clipped_low"] & active).sum()), "active": int(active.sum())}


def identity_gate(d, active, tol=IDENTITY_TOL):
    """r recomputed from g, sigma_a, sigma_e against r from the sums on every FP-active cell that is not degenerate; degenerate cells (a variance of zero, sigma_e^2 <= 0 from cancellation,
    a clamped r) are excluded and counted. -> {"passed", "checked", "excluded": {reason: count}, "max_abs_diff"}."""
    excluded = {"sigma_b_zero": 0, "residual_variance_not_positive": 0, "r_clamped": 0}; check = np.zeros(active.shape, dtype=bool)
    for l, j in zip(*np.nonzero(active)):
        if not (d["var_a"][l, j] > 0 and d["var_b"][l, j] > 0 and np.isfinite(d["g"][l, j])):
            excluded["sigma_b_zero"] += 1
        elif not d["sigma_e2"][l, j] > 0:
            excluded["residual_variance_not_positive"] += 1
        elif abs(d["r"][l, j]) >= 1.0 or d["den_clamped"][l, j]:
            excluded["r_clamped"] += 1
        else:
            check[l, j] = True
    if check.any():
        g, sa, se2 = d["g"][check], d["sigma_a"][check], d["sigma_e2"][check]
        diff = np.abs(g * sa / np.sqrt(g ** 2 * sa ** 2 + se2) - d["r"][check]); mx = float(diff.max())
    else:
        mx = float("nan")
    return {"passed": bool(check.any() and mx <= tol), "checked": int(check.sum()), "excluded": excluded, "max_abs_diff": mx, "tolerance": tol}


# ------------------------------------------------------------------ H3
def h3_cell(d, active, min_cells=3, scale=None):
    """Per language over its FP-active cells: Spearman(lambda, log sigma_a) and Spearman(nu, log sigma_a); the median over languages (languages with fewer than min_cells cells or a constant column are skipped)."""
    lam_rho, nu_rho = [], []
    for l in range(active.shape[0]):
        m = active[l]
        if m.sum() < min_cells:
            lam_rho.append(float("nan")); nu_rho.append(float("nan")); continue
        ls = np.log((d["sigma_a"] if scale is None else scale)[l][m])
        with np.errstate(all="ignore"):
            lam_rho.append(float(spearmanr(d["lam"][l][m], ls)[0])); nu_rho.append(float(spearmanr(d["nu"][l][m], ls)[0]))
    return {"median_spearman_lambda": float(np.nanmedian(lam_rho)) if np.isfinite(lam_rho).any() else float("nan"),
            "median_spearman_nu": float(np.nanmedian(nu_rho)) if np.isfinite(nu_rho).any() else float("nan"), "per_language_lambda": lam_rho, "per_language_nu": nu_rho,
            "languages_used": int(np.isfinite(nu_rho).sum())}


def h3_rule(median_nu, median_lambda):
    """The plan's thresholds: |median Spearman(nu, log sigma_a)| < 0.2 AND |median Spearman(lambda, log sigma_a)| > 0.5 (NaN fails)."""
    return bool(np.isfinite(median_nu) and np.isfinite(median_lambda) and abs(median_nu) < H3_NU_MAX and abs(median_lambda) > H3_LAMBDA_MIN)


def h3_overall(cells, arms=("legacy", "matched", "gptq"), widths=("INT6", "INT4")):
    """H3 is met only if it is met in all six (arm, width) cells; a missing cell is not a pass."""
    expected = [(a, w) for a in arms for w in widths]
    missing = [k for k in expected if k not in cells]; failing = [k for k in expected if k in cells and not cells[k]]
    return {"met": not missing and not failing, "failing": failing, "missing": missing}


# ------------------------------------------------------------------ H4
def tier_slope(y, tier, script, covariates=None):
    """OLS coefficient of the tier score in y ~ script dummies (+ covariates) + tier (as phase1_analysis.partial_slope and freedman_lane_tier_test); NaN when the residualised tier has no variation (a degenerate resample)."""
    y, tier, script = np.asarray(y, dtype=float), np.asarray(tier, dtype=float), np.asarray(script)
    Z = an._design(script)
    if covariates is not None:
        C = np.asarray(covariates, dtype=float); Z = np.column_stack([Z, C[:, None] if C.ndim == 1 else C])
    tr = tier - Z @ np.linalg.pinv(Z) @ tier; ss = float(tr @ tr)
    return float(tr @ y / ss) if ss > 1e-12 else float("nan")


def z_slope(y, tier, script, covariates=None):
    """Standardized slope: slope * SD(tier) / SD(y) over the languages of the sample (each width with ITS OWN SD(y)); NaN if y is constant."""
    y = np.asarray(y, dtype=float); sd_y = float(y.std()); sd_t = float(np.asarray(tier, dtype=float).std())
    return tier_slope(y, tier, script, covariates) * sd_t / sd_y if sd_y > 0 else float("nan")


def h4_rule(estimate, ci95, ci90, bound=H4_BOUND):
    """'present' if the 95 % interval excludes 0 and |estimate| >= bound; 'absent' ONLY if the 90 % interval lies strictly inside (-bound, +bound) (equivalence); otherwise 'inconclusive'."""
    vals = [estimate, *ci95, *ci90]
    if not np.all(np.isfinite(vals)):
        return "inconclusive"
    if (ci95[0] > 0 or ci95[1] < 0) and abs(estimate) >= bound:
        return "present"
    if ci90[0] > -bound and ci90[1] < bound:
        return "absent"
    return "inconclusive"


def _intervals(reps, ok):
    if not ok:
        return (float("nan"), float("nan")), (float("nan"), float("nan"))
    return tuple(float(x) for x in np.percentile(reps, [2.5, 97.5])), tuple(float(x) for x in np.percentile(reps, [5, 95]))


def h4_cell(y_w, y16, tier, script, n_boot=N_BOOT, seed=SEED, bound=H4_BOUND, covariates=None):
    """One (arm, width): the PRIMARY quantization-specific standardized difference s_w = z_w - z_INT16, the SECONDARY z_w against 0 and the SENSITIVITY (slope_w - slope_INT16) * SD(tier) / SD(y_INT16),
    with language-cluster bootstrap intervals (the same resample for every width) and the rule of the freeze. Degenerate replicates are dropped and counted; more than MAX_DROPPED_SHARE -> not estimable."""
    y_w, y16, tier, script = np.asarray(y_w, float), np.asarray(y16, float), np.asarray(tier, float), np.asarray(script)
    cov = None if covariates is None else np.asarray(covariates, float)
    def stats(idx):
        yw, y0, t, sc_ = y_w[idx], y16[idx], tier[idx], script[idx]; cv = None if cov is None else cov[idx]
        zw, z0 = z_slope(yw, t, sc_, cv), z_slope(y0, t, sc_, cv); sw, s0 = tier_slope(yw, t, sc_, cv), tier_slope(y0, t, sc_, cv)
        sens = (sw - s0) * float(t.std()) / float(y0.std()) if y0.std() > 0 else float("nan")
        return zw - z0, zw, sens
    n = len(y_w); point = stats(np.arange(n)); rng = np.random.default_rng(seed); reps = []
    for _ in range(n_boot):
        v = stats(rng.integers(0, n, n))
        if np.all(np.isfinite(v)):
            reps.append(v)
    reps = np.array(reps).reshape(-1, 3); dropped = n_boot - len(reps); ok = dropped / n_boot <= MAX_DROPPED_SHARE and len(reps) > 1
    out = {"n_boot": n_boot, "valid_replicates": int(len(reps)), "dropped_replicates": int(dropped), "slope_w": tier_slope(y_w, tier, script, cov), "slope_16": tier_slope(y16, tier, script, cov),
           "z_w": z_slope(y_w, tier, script, cov), "z_16": z_slope(y16, tier, script, cov), "bound": bound, "scale_covariate": cov is not None}
    for name, k in (("primary", 0), ("secondary", 1), ("sensitivity_common_denominator", 2)):
        ci95, ci90 = _intervals(reps[:, k], ok)
        out[name] = {"estimate": float(point[k]), "ci95": ci95, "ci90": ci90, "not_estimable": not ok, "outcome": h4_rule(point[k], ci95, ci90, bound)}
    return out


# ------------------------------------------------------------------ H5: g and nu added to model B (D-113 code and rule unchanged)
def h5_cell(base, r1, gain, log_nu, w1, lang, mask, n_a, n_boot=se.N_BOOT, seed=se.SEED):
    """One (arm, state) cell of H5 on the F5 features: models A (base), B (base + R1), B+ (base + R1 + g + log nu) and the g-only and nu-only variants, all on the SAME rows (the reliability mask `mask`,
    n_a >= MIN_TOKENS in the sidecar, g and nu defined), leave-language-out ridge as in D-113 (survival_effect, unchanged); the improvement of B+ over B in mean absolute error with the language-cluster
    bootstrap of survival_effect and its rule (>= 10 % and the interval above 0)."""
    base, r1, gain, log_nu, w1, lang, mask, n_a = (np.asarray(x) for x in (base, r1, gain, log_nu, w1, lang, mask, n_a))
    mask = mask.astype(bool); enough = n_a >= MIN_TOKENS; defined = np.isfinite(gain) & np.isfinite(log_nu)
    pop = mask & enough & defined
    dropped = {"outside_mask": int((~mask).sum()), "n_a_below_floor": int((mask & ~enough).sum()), "undefined_g_or_nu": int((mask & enough & ~defined).sum())}
    y = w1[pop].astype(np.float64); lg = lang[pop]; b, r, g, nu = base[pop], r1[pop], gain[pop], log_nu[pop]
    X = {"A": b, "B": np.column_stack([b, r]), "B+": np.column_stack([b, r, g, nu]), "B+g": np.column_stack([b, r, g]), "B+nu": np.column_stack([b, r, nu])}
    err = {k: np.abs(y - se.loo_language(v, y, lg)[0]) for k, v in X.items()}
    imp = lambda a, c: se.improvement(err[a], err[c], lg, n_boot, seed)
    out = {"n": int(pop.sum()), "n_languages": int(len(np.unique(lg))), "dropped": dropped, "mae": {k: float(v.mean()) for k, v in err.items()},
           "improvement": {"B_over_A": imp("A", "B"), "Bplus_over_B": imp("B", "B+"), "Bplus_over_A": imp("A", "B+"), "Bg_over_B": imp("B", "B+g"), "Bnu_over_B": imp("B", "B+nu")}}
    b_ = out["improvement"]["Bplus_over_B"]; out["cell_met"] = se.cell_met(b_["improvement"], b_["lo"])
    return out


def h5_overall(cells):
    """The D-113 overall statement over the cells of (arm, INT6, INT4)."""
    return se.overall_statement(cells)


# ------------------------------------------------------------------ descriptive tables of the freeze
def _pct(v, qs, names):
    v = np.asarray(v, dtype=float); v = v[np.isfinite(v)]
    return {n: (float(np.percentile(v, q)) if len(v) else float("nan")) for q, n in zip(qs, names)}


def lambda_decomposition(d_a, d_b, active):
    """Cross-arm decomposition of lambda (b minus a) on the cells with a decomposition in both arms and an unclipped lambda in both: lambda = log sigma_e^2 - 2 log|g| - 2 log sigma_a and sigma_a is the same
    in every arm, so delta lambda = delta log sigma_e^2 - 2 delta log|g| exactly (the identity error is reported); medians of the three differences."""
    m = populations(d_a, active)["decomposition"] & populations(d_b, active)["decomposition"] & ~d_a["clipped"] & ~d_b["clipped"]
    dl = (d_b["lam"] - d_a["lam"])[m]; ds = np.log(d_b["sigma_e2"][m]) - np.log(d_a["sigma_e2"][m]); dg = -2 * (np.log(np.abs(d_b["g"][m])) - np.log(np.abs(d_a["g"][m])))
    med = lambda v: float(np.median(v)) if len(v) else float("nan")
    return {"cells": int(m.sum()), "median_delta_lambda": med(dl), "median_delta_log_sigma_e2": med(ds), "median_minus2_delta_log_abs_g": med(dg),
            "identity_max_abs_error": float(np.abs(dl - (ds + dg)).max()) if len(dl) else float("nan")}


def arm_summary(d, active, tier):
    """One (arm, width): decomposition medians, flip-rate percentiles, nu quartiles overall and per tier, and the silent / clipped counts overall and per tier (tier = tier score of each language, 0 low, 1 mid, 2 high)."""
    pop = populations(d, active); dec = pop["decomposition"]; tier = np.asarray(tier)
    by_tier = {int(t): active & (tier == t)[:, None] for t in np.unique(tier)}
    count = lambda m: {"cells": int(m.sum()), "silent": int((d["silent"] & m).sum()), "share": float((d["silent"] & m).sum() / m.sum()) if m.sum() else float("nan")}
    clip = lambda m: {"low": int((d["clipped_low"] & m).sum()), "high": int((d["clipped"] & ~d["clipped_low"] & m).sum())}
    med = lambda v: float(np.median(v)) if len(v) else float("nan")
    return {"median_g": med(d["g"][dec]), "median_log_sigma_e2": med(np.log(d["sigma_e2"][dec])), "median_lambda": med(d["lam"][active]), "cells": int(active.sum()),
            "flip": {k: _pct(d[k][active], (10, 50, 90), ("p10", "p50", "p90")) for k in ("on_off", "off_on")},
            "nu": {"overall": _pct(d["nu"][active], (25, 50, 75), ("p25", "p50", "p75")), "by_tier": {t: _pct(d["nu"][m], (25, 50, 75), ("p25", "p50", "p75")) for t, m in by_tier.items()}},
            "silent": {"overall": count(active), "by_tier": {t: count(m) for t, m in by_tier.items()}}, "clipped": {"overall": clip(active), "by_tier": {t: clip(m) for t, m in by_tier.items()}}}


def flip_prediction(d, active):
    """Predicted flip share (|delta pre| > margin, both signs) against the exact one (n_exceed1 / n_a): Spearman over cells and the median difference."""
    m = active & np.isfinite(d["pred_flip"]) & np.isfinite(d["exact_flip"])
    with np.errstate(all="ignore"):
        rho = float(spearmanr(d["pred_flip"][m], d["exact_flip"][m])[0]) if m.sum() > 2 else float("nan")
    return {"cells": int(m.sum()), "spearman": rho, "median_pred_minus_exact": float(np.median((d["pred_flip"] - d["exact_flip"])[m])) if m.any() else float("nan")}


def nu_between_arms(d_a, d_b, active):
    """The metric-2 analogue of Stage 1's H1: log nu of arm b against arm a over the FP-active cells (no condition on the quantized side): Spearman, the SD-ratio slope, the OLS slope, the median difference."""
    m = active & np.isfinite(d_a["log_nu"]) & np.isfinite(d_b["log_nu"]); x, y = d_a["log_nu"][m], d_b["log_nu"][m]
    rho, sd_slope = sl.h1_stats(x, y, np.zeros(len(x), dtype=bool), np.zeros(len(y), dtype=bool))
    return {"cells": int(m.sum()), "spearman": rho, "sd_ratio_slope": sd_slope, "ols_slope": float(np.polyfit(x, y, 1)[0]), "median_difference": float(np.median(y - x))}


# ------------------------------------------------------------------ wording (D-121), selectivity (secondary), assembly, report
class AnalysisStopped(RuntimeError):
    pass


def improvement_wording(improvement, lo, hi):
    """D-121: every improvement is reported with its interval; its size does not decide how it is worded (the 10 % target is a convention, not a claim)."""
    if not np.all(np.isfinite([improvement, lo, hi])):
        return "not estimable"
    if lo > 0:
        return "improvement (interval above zero)"
    if hi < 0:
        return "worse (interval below zero)"
    return "not distinguishable from zero (interval includes zero)"


def selectivity_difference(d, active, selective, min_cells=20, n_boot=N_BOOT, seed=SEED):
    """Secondary, descriptive: per language the difference of mean log nu between selective and non-selective FP-active cells (languages with at least min_cells selective cells), the mean over languages and a 95 % language-cluster interval."""
    rows = []
    for l in range(active.shape[0]):
        a, b = active[l] & selective[l], active[l] & ~selective[l]
        rows.append(float(d["log_nu"][l][a].mean() - d["log_nu"][l][b].mean()) if a.sum() >= min_cells and b.sum() > 0 else float("nan"))
    v = np.array(rows); ok = np.isfinite(v)
    if not ok.any():
        return {"languages_used": 0, "mean_difference": float("nan"), "ci95": (float("nan"), float("nan")), "per_language": rows}
    rng = np.random.default_rng(seed); u = v[ok]; reps = [float(u[rng.integers(0, len(u), len(u))].mean()) for _ in range(n_boot)]
    return {"languages_used": int(ok.sum()), "mean_difference": float(u.mean()), "ci95": tuple(float(x) for x in np.percentile(reps, [2.5, 97.5])), "per_language": rows}


def jsonable(o):
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


ARMS3 = ("legacy", "matched", "gptq")
F5_LABEL = {"legacy": "legacy RTN", "matched": "matched RTN", "gptq": "GPTQ"}


def analyse(sums_by_arm, theta, tier, script, f5=None, selective=None, committed=None, max_activation=None, n_boot=N_BOOT, seed=SEED, h5_n_boot=None):
    """Everything of the freeze from the validated sums {arm: {width: sums}} (legacy: INT16, INT6, INT4; the others INT6, INT4). Stops (AnalysisStopped) if the arms disagree on the FP-active counts or an identity
    gate fails; labels every number 'sidecar-internal' if a committed comparison failed. `f5` = {arm: {"T": feature table, "mask": reliability mask}} adds H5; `selective` [n_lang, d] adds the selectivity secondary."""
    ref = sums_by_arm["legacy"]["INT6"]; n_lang = int(ref["n"].numel()); tier = np.asarray(tier); script = np.asarray(script)
    for arm, widths in sums_by_arm.items():
        for w, sm in widths.items():
            if not (torch_equal(sm["stats"]["n_a"], ref["stats"]["n_a"]) and torch_equal(sm["n"], ref["n"])):
                raise AnalysisStopped(f"{arm} {w}: the FP-active counts (n_a, n) differ from the reference: the FP side must be identical in every condition")
    active = fp_active(ref["stats"]["n_a"].numpy(), ref["n"].numpy())
    D = {(a, w): derived(sm, theta) for a, ws in sums_by_arm.items() for w, sm in ws.items()}
    gates = {f"{a}|{w}": identity_gate(d, active) for (a, w), d in D.items()}
    bad = [k for k, g in gates.items() if not g["passed"]]
    if bad:
        raise AnalysisStopped(f"identity gate failed in {bad}: the analysis stops (freeze)")
    pops = {f"{a}|{w}": populations(d, active) for (a, w), d in D.items()}
    primary = [(a, w) for a in ARMS3 for w in ("INT6", "INT4") if (a, w) in D]
    h3c = {}
    for a, w in primary:
        r = h3_cell(D[a, w], active); r["met"] = h3_rule(r["median_spearman_nu"], r["median_spearman_lambda"]); h3c[f"{a}|{w}"] = r
    H3 = {"cells": h3c, "overall": h3_overall({tuple(k.split("|")): v["met"] for k, v in h3c.items()}), "reported": {"legacy|INT16": h3_cell(D["legacy", "INT16"], active)}}
    if max_activation is not None:
        H3["sensitivity_log_max_activation"] = {f"{a}|{w}": h3_cell(D[a, w], active, scale=np.asarray(max_activation, dtype=float)) for a, w in primary}
    def outcome(d):
        return np.array([-d["log_nu"][l][active[l]].mean() if active[l].any() else np.nan for l in range(n_lang)])
    y16 = outcome(D["legacy", "INT16"]); cov = np.array([np.log10(np.median(D["legacy", "INT6"]["sigma_a"][l][active[l]])) if active[l].any() else np.nan for l in range(n_lang)])
    H4 = {"cells": {f"{a}|{w}": {"standard": h4_cell(outcome(D[a, w]), y16, tier, script, n_boot=n_boot, seed=seed),
                                 "with_scale_covariate": h4_cell(outcome(D[a, w]), y16, tier, script, n_boot=n_boot, seed=seed, covariates=cov)} for a, w in primary}}
    pair = {f"{a}_vs_legacy|{w}": (D["legacy", w], D[a, w]) for a in ("matched", "gptq") for w in ("INT6", "INT4") if (a, w) in D}
    desc = {"arm_summary": {f"{a}|{w}": arm_summary(d, active, tier) for (a, w), d in D.items()}, "lambda_decomposition": {k: lambda_decomposition(x, y, active) for k, (x, y) in pair.items()},
            "nu_between_arms": {k: nu_between_arms(x, y, active) for k, (x, y) in pair.items()}, "flip_prediction": {f"{a}|{w}": flip_prediction(d, active) for (a, w), d in D.items()},
            "selectivity": None if selective is None else {f"{a}|{w}": selectivity_difference(D[a, w], active, np.asarray(selective, dtype=bool), n_boot=n_boot, seed=seed) for a, w in primary}}
    H5 = None
    if f5 is not None:
        cells = {}
        for a, w in primary:
            T, mask = f5[a]["T"], f5[a]["mask"]; d = D[a, w]; li, ii = np.asarray(T["lang"]).astype(int), np.asarray(T["ids"]).astype(int)
            res = h5_cell(T["base"], T[w]["r1"], d["g"][li, ii], d["log_nu"][li, ii], T[w]["w1"], T["lang"], mask, d["n_a"][li, ii], n_boot=h5_n_boot or n_boot)
            for k, v in res["improvement"].items():
                v["wording"] = improvement_wording(v["improvement"], v["lo"], v["hi"])
            cells[f"{a}|{w}"] = {"result": res, "wording": res["improvement"]["Bplus_over_B"]["wording"]}
        H5 = {"cells": cells, "statement": h5_overall([c["result"]["cell_met"] for c in cells.values()]),
              "note": "the 10 % relative improvement is the D-113 convention, not a claim: every improvement is reported with its interval and worded from the interval (D-121)"}
    failed = sorted(a for a, c in (committed or {}).items() if not c["G2_committed"]["passed"])
    return {"n_languages": n_lang, "gates": {"identity": gates}, "populations": pops, "H3": H3, "H4": H4, "descriptive": desc, "H5": H5,
            "labels": {"sidecar_internal": bool(failed), "failed_arms": failed, "committed_comparison": "run" if committed else "not run"}}


def torch_equal(a, b):
    import torch
    return bool(torch.equal(a, b))


def _f(x, k=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{k}f}"


def _ci(c, k=3):
    return f"[{_f(c[0], k)}, {_f(c[1], k)}]"


def render_md(res):
    """The generated report: every number comes from `res`; thresholds are conventions and are shown next to the numbers they label (D-120, D-121)."""
    L = ["# Stage 3 results (generated; D-117 survival measurement-validity study)", "",
         "Rules: `reports/survival_measurement/analysis_freeze.json` (D-120, committed before the Stage 2 outputs existed). **Every threshold below (H3: 0.2 and 0.5; H4: the 0.20 bound; H5: the 10 % target) is a pre-set convention, not a claim: the numbers and their intervals are always shown next to the label (D-121).**", ""]
    lab = res["labels"]
    if lab["sidecar_internal"]:
        L += [f"**LABEL: sidecar-internal.** The comparison with the committed stage-1 files (G2(ii)) failed for {', '.join(lab['failed_arms'])}: every number is a statement about the sidecar's own sums, and the claim that the sidecar sees the same weights and tokens as stage 1 is not made.", ""]
    else:
        L += [f"Committed comparison (G2(ii)): {lab['committed_comparison']}.", ""]
    L += ["## Gates: identity (r from g, sigma_a, sigma_e against r from the sums)", "", "| cell | checked | excluded (degenerate) | largest difference |", "|---|---|---|---|"]
    for k, g in res["gates"]["identity"].items():
        L.append(f"| {k} | {g['checked']} | {sum(g['excluded'].values())} | {g['max_abs_diff']:.2e} |")
    L += ["", "## Populations (FP-active cells; silent = no quantized activation; clipped lambda counted, never dropped)", "", "| cell | active | silent | clipped low | clipped high |", "|---|---|---|---|---|"]
    for k, p in res["populations"].items():
        L.append(f"| {k} | {p['active']} | {p['silent_count']} | {p['clipped_low']} | {p['clipped_high']} |")
    h3 = res["H3"]
    L += ["", "## H3: does lambda follow the feature's scale, and nu not?", "", "| cell | median Spearman(lambda, log sigma_a) | median Spearman(nu, log sigma_a) | rule (convention) |", "|---|---|---|---|"]
    for k, c in h3["cells"].items():
        L.append(f"| {k} | {_f(c['median_spearman_lambda'])} | {_f(c['median_spearman_nu'])} | {'met' if c['met'] else 'not met'} |")
    for k, c in h3["reported"].items():
        L.append(f"| {k} (reported) | {_f(c['median_spearman_lambda'])} | {_f(c['median_spearman_nu'])} | - |")
    o = h3["overall"]; L += ["", f"Overall: {'met in all six cells' if o['met'] else 'not met'}; failing cells {o['failing'] or 'none'}; missing {o['missing'] or 'none'}.", "",
                             "## H4: tier pattern in the encoder perturbation (outcome = minus the mean of log nu per language)", "",
                             "Primary s = z(width) - z(INT16) (quantization-specific part, standardized slopes); secondary z(width) against 0. Outcome words: present (95 % interval excludes 0 and |estimate| >= 0.20), absent (90 % interval inside +-0.20), otherwise inconclusive; inconclusive is an accepted result.", "",
                             "| cell | covariate | slope INT16 | slope width | primary s [95 %] | primary outcome | secondary z [95 %] | secondary outcome | dropped replicates |", "|---|---|---|---|---|---|---|---|---|"]
    for k, v in res["H4"]["cells"].items():
        for name in ("standard", "with_scale_covariate"):
            c = v[name]; L.append(f"| {k} | {'sigma_a' if name != 'standard' else 'none'} | {_f(c['slope_16'])} | {_f(c['slope_w'])} | {_f(c['primary']['estimate'])} {_ci(c['primary']['ci95'])} | {c['primary']['outcome']} | {_f(c['secondary']['estimate'])} {_ci(c['secondary']['ci95'])} | {c['secondary']['outcome']} | {c['dropped_replicates']} |")
    if res["H5"] is not None:
        L += ["", "## H5: do g and log nu add to model B for W1? (every improvement with its interval)", "", res["H5"]["note"] + ".", "",
              "| cell | n | B over A | B+ over B | B+ over A | g only over B | nu only over B | wording of B+ over B | meets the 10 % convention |", "|---|---|---|---|---|---|---|---|---|"]
        for k, c in res["H5"]["cells"].items():
            r = c["result"]; im = r["improvement"]; cell = lambda n: f"{_f(im[n]['improvement'])} {_ci((im[n]['lo'], im[n]['hi']))}"
            L.append(f"| {k} | {r['n']} | {cell('B_over_A')} | {cell('Bplus_over_B')} | {cell('Bplus_over_A')} | {cell('Bg_over_B')} | {cell('Bnu_over_B')} | {c['wording']} | {'yes' if r['cell_met'] else 'no'} |")
        L += ["", f"D-113 overall statement over the six cells: {res['H5']['statement']}."]
    d = res["descriptive"]
    L += ["", "## Descriptive tables", "", "### Lambda decomposition between arms (arm minus legacy; lambda = log sigma_e^2 - 2 log|g| - 2 log sigma_a, exact)", "", "| pair | cells | median delta lambda | median delta log sigma_e^2 | median -2 delta log abs g | identity error |", "|---|---|---|---|---|---|"]
    for k, c in d["lambda_decomposition"].items():
        L.append(f"| {k} | {c['cells']} | {_f(c['median_delta_lambda'])} | {_f(c['median_delta_log_sigma_e2'])} | {_f(c['median_minus2_delta_log_abs_g'])} | {c['identity_max_abs_error']:.1e} |")
    L += ["", "### log nu of an arm against legacy", "", "| pair | cells | Spearman | SD-ratio slope | OLS slope | median difference |", "|---|---|---|---|---|---|"]
    for k, c in d["nu_between_arms"].items():
        L.append(f"| {k} | {c['cells']} | {_f(c['spearman'])} | {_f(c['sd_ratio_slope'])} | {_f(c['ols_slope'])} | {_f(c['median_difference'])} |")
    L += ["", "### Per arm and width", "", "| cell | median g | median log sigma_e^2 | median lambda | on-off median | off-on median | nu median [p25, p75] | silent share | predicted vs exact flips (Spearman) |", "|---|---|---|---|---|---|---|---|---|"]
    for k, c in d["arm_summary"].items():
        L.append(f"| {k} | {_f(c['median_g'])} | {_f(c['median_log_sigma_e2'])} | {_f(c['median_lambda'])} | {_f(c['flip']['on_off']['p50'], 4)} | {_f(c['flip']['off_on']['p50'], 4)} | {_f(c['nu']['overall']['p50'])} [{_f(c['nu']['overall']['p25'])}, {_f(c['nu']['overall']['p75'])}] | {_f(c['silent']['overall']['share'], 4)} | {_f(d['flip_prediction'][k]['spearman'])} |")
    if d["selectivity"]:
        L += ["", "### Selectivity (secondary, descriptive): mean log nu of selective minus non-selective cells", "", "| cell | languages | mean difference [95 %] |", "|---|---|---|"]
        for k, c in d["selectivity"].items():
            L.append(f"| {k} | {c['languages_used']} | {_f(c['mean_difference'])} {_ci(c['ci95'])} |")
    L += ["", "## Limits", "", "One model, one layer, one SAE, simulated weight-only quantizers, 27 languages (clusters). Single-language script families are absorbed by the script dummies, so the tier slope rests on about 23 languages. Post hoc relative to Stage 1 (same tokens and features), with the Stage 3 rules fixed before the new statistics existed. Nulls, negatives and inconclusive outcomes are results."]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ inputs and entry point
JOBS = {"legacy": ("sc-legacy", ("INT16", "INT6", "INT4"), (16, 6, 4, 0)), "matched": ("sc-matched", ("INT6", "INT4"), (6, 4, 0)), "gptq": ("sc-gptq", ("INT6", "INT4"), (6, 4, 0))}
ARM_WEIGHTS = {"legacy": "legacy", "matched": "rtn-matched", "gptq": "gptq"}


def load_job(directory, arm):
    """The saved sums and meta of one validated job directory; refuses a job whose meta did not pass every gate (survival_validate.check_meta) or whose conditions are not those of the arm."""
    import survival_validate as sv
    name, widths, bits = JOBS[arm]; d = Path(directory) / name; meta = json.loads((d / "meta.json").read_text())
    chk = sv.check_meta(meta, list(bits), full=True, n_lang=len(meta["languages"]), n_feat=meta["n_feat"], arm=ARM_WEIGHTS[arm])
    if not chk["ok"]:
        raise AnalysisStopped(f"{name}: meta.json is not that of a passed full job: {chk['why']}")
    sc.validate_dir(d, list(widths), n_lang=len(meta["languages"]), n_feat=meta["n_feat"])
    return {w: sc.load_sums(d / f"sums_{w}.pt") for w in widths}, meta


def selective_cells(selective, peak, n_lang):
    """The committed selectivity file holds `selective` per FEATURE and `peak` (the language where the feature peaks): the cell (language l, feature j) is selective iff the feature is selective and its peak language is l
    (this is how the freeze's 'the committed `selective` mask' is read for the per-cell secondary; stated in the report)."""
    selective = np.asarray(selective, dtype=bool); peak = np.asarray(peak).astype(int); cells = np.zeros((n_lang, len(selective)), dtype=bool)
    j = np.nonzero(selective)[0]; cells[peak[j], j] = True
    return cells


def load_roster(path, languages):
    import json as _j
    roster = {l["code"]: l for l in _j.loads(Path(path).read_text())["languages"]}
    fams = sorted({roster[c]["script_family"] for c in languages})
    return np.array([an.tier_score(roster[c]["tier"]) for c in languages]), np.array([fams.index(roster[c]["script_family"]) for c in languages])


def load_f5(func_inputs, phase1_survival, languages):
    """The F5 feature tables of the three arms and the D-113 primary (reliable) mask, exactly as survival_effect.main builds them."""
    import torch
    sv = torch.load(phase1_survival, map_location="cpu", weights_only=False)
    meta_langs = json.loads((Path(func_inputs) / "jobs/gptq-f5-int64/meta_f5.json").read_text())["languages"]
    if list(meta_langs) != list(languages):
        raise AnalysisStopped("the F5 language list differs from the sidecar's language list")
    tabs = {}
    for arm in ARMS3:
        sa, states = se.load_arm(func_inputs, F5_LABEL[arm]); tabs[arm] = se.feature_table(sa, states, sv, meta_langs)
    rel = se.reliable_mask(tabs["legacy"]["cos16"])
    return {arm: {"T": T, "mask": rel} for arm, T in tabs.items()}


def main(argv=None):
    import argparse
    import survival_validate as sv
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", default="/home/karma/Documents/ANLP/project/transfer_runs"); ap.add_argument("--out", default=str(ROOT / "reports/survival_measurement/stage3"))
    ap.add_argument("--roster", default=str(ROOT / "reports/g0/phase1_languages.json")); ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--committed", nargs=3, metavar=("LEGACY", "MATCHED", "GPTQ"), default=[str(Path.home() / ".cache/anlp-always-late/phase1_runs/p1-mask/eq"), "/home/karma/Documents/ANLP/project/transfer_runs/rtnm-p1-mask/eq", "/home/karma/Documents/ANLP/project/transfer_runs/gptq-p1-mask/eq"])
    ap.add_argument("--no-committed", action="store_true"); ap.add_argument("--no-f5", action="store_true"); ap.add_argument("--no-selectivity", action="store_true"); ap.add_argument("--func-inputs", default="/home/karma/Documents/ANLP/project/transfer_runs/func_inputs")
    ap.add_argument("--phase1", default=str(Path.home() / ".cache/anlp-always-late/phase1_runs/p1-mask/eq/survival.pt")); ap.add_argument("--selectivity", default=str(Path.home() / ".cache/anlp-always-late/phase1_runs/p1-mask/eq/selectivity.pt"))
    a = ap.parse_args(argv)
    import torch
    check_freeze()
    jobs, metas = {}, {}
    for arm in ARMS3:
        jobs[arm], metas[arm] = load_job(a.runs, arm)
    langs = metas["legacy"]["languages"]
    for arm in ARMS3:
        if metas[arm]["languages"] != langs or metas[arm]["theta_sha256"] != metas["legacy"]["theta_sha256"]:
            raise AnalysisStopped(f"{arm}: languages or thresholds differ from the legacy job")
    theta = torch.tensor(metas["legacy"]["theta"], dtype=torch.float32); tier, script = load_roster(a.roster, langs)
    committed = None if a.no_committed else {arm: sc.check_committed(Path(a.runs) / JOBS[arm][0], path) for arm, path in zip(ARMS3, a.committed)}
    max_act = selective = None
    if not a.no_committed and Path(a.phase1).exists():
        max_act = torch.load(a.phase1, map_location="cpu", weights_only=False)["max_activation"].double().numpy()
    if not a.no_selectivity and Path(a.selectivity).exists():
        sel = torch.load(a.selectivity, map_location="cpu", weights_only=False)
        if list(sel["languages"]) != list(langs):
            raise AnalysisStopped("the selectivity file's language list differs from the sidecar's")
        selective = selective_cells(sel["selective"].numpy(), sel["peak"].numpy(), len(langs))
    f5 = None if a.no_f5 else load_f5(a.func_inputs, a.phase1, langs)
    res = analyse(jobs, theta, tier, script, f5=f5, selective=selective, committed=committed, max_activation=max_act, n_boot=a.n_boot)
    res["freeze_sha256"] = check_freeze()["sha256"]; res["committed_checks"] = committed
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "stage3_results.json").write_text(json.dumps(res, indent=1, default=jsonable) + "\n"); (out / "stage3_results.md").write_text(render_md(res))
    print("wrote", out / "stage3_results.json", "and", out / "stage3_results.md", "| labels:", res["labels"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
