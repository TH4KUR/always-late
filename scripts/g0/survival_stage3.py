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
def h3_cell(d, active, min_cells=3):
    """Per language over its FP-active cells: Spearman(lambda, log sigma_a) and Spearman(nu, log sigma_a); the median over languages (languages with fewer than min_cells cells or a constant column are skipped)."""
    lam_rho, nu_rho = [], []
    for l in range(active.shape[0]):
        m = active[l]
        if m.sum() < min_cells:
            lam_rho.append(float("nan")); nu_rho.append(float("nan")); continue
        ls = np.log(d["sigma_a"][l][m])
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
def tier_slope(y, tier, script):
    """OLS coefficient of the tier score in y ~ script dummies + tier (as phase1_analysis.partial_slope); NaN when the script-residualised tier has no variation (a degenerate resample)."""
    y, tier, script = np.asarray(y, dtype=float), np.asarray(tier, dtype=float), np.asarray(script)
    Z = an._design(script); tr = tier - Z @ np.linalg.pinv(Z) @ tier; ss = float(tr @ tr)
    return float(tr @ y / ss) if ss > 1e-12 else float("nan")


def z_slope(y, tier, script):
    """Standardized slope: slope * SD(tier) / SD(y) over the languages of the sample (each width with ITS OWN SD(y)); NaN if y is constant."""
    y = np.asarray(y, dtype=float); sd_y = float(y.std()); sd_t = float(np.asarray(tier, dtype=float).std())
    return tier_slope(y, tier, script) * sd_t / sd_y if sd_y > 0 else float("nan")


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


def h4_cell(y_w, y16, tier, script, n_boot=N_BOOT, seed=SEED, bound=H4_BOUND):
    """One (arm, width): the PRIMARY quantization-specific standardized difference s_w = z_w - z_INT16, the SECONDARY z_w against 0 and the SENSITIVITY (slope_w - slope_INT16) * SD(tier) / SD(y_INT16),
    with language-cluster bootstrap intervals (the same resample for every width) and the rule of the freeze. Degenerate replicates are dropped and counted; more than MAX_DROPPED_SHARE -> not estimable."""
    y_w, y16, tier, script = np.asarray(y_w, float), np.asarray(y16, float), np.asarray(tier, float), np.asarray(script)
    def stats(idx):
        yw, y0, t, sc_ = y_w[idx], y16[idx], tier[idx], script[idx]
        zw, z0 = z_slope(yw, t, sc_), z_slope(y0, t, sc_); sw, s0 = tier_slope(yw, t, sc_), tier_slope(y0, t, sc_)
        sens = (sw - s0) * float(t.std()) / float(y0.std()) if y0.std() > 0 else float("nan")
        return zw - z0, zw, sens
    n = len(y_w); point = stats(np.arange(n)); rng = np.random.default_rng(seed); reps = []
    for _ in range(n_boot):
        v = stats(rng.integers(0, n, n))
        if np.all(np.isfinite(v)):
            reps.append(v)
    reps = np.array(reps).reshape(-1, 3); dropped = n_boot - len(reps); ok = dropped / n_boot <= MAX_DROPPED_SHARE and len(reps) > 1
    out = {"n_boot": n_boot, "valid_replicates": int(len(reps)), "dropped_replicates": int(dropped), "slope_w": tier_slope(y_w, tier, script), "slope_16": tier_slope(y16, tier, script),
           "z_w": z_slope(y_w, tier, script), "z_16": z_slope(y16, tier, script), "bound": bound}
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
