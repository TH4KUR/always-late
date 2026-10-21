"""D-035 Task 3 (design: reports/survival_method_reassessment.md section 7; plan: docs/superpowers/plans/2026-10-21-language-association-retention.md): metrics of language
association from the per-language response profiles of one condition. Pure NumPy on [language, feature] arrays; nothing here reads or writes files.

Definitions (r = firing rate z > 0 per content token, mu = mean activation with zeros included, l0[f] = the frozen dev peak language of feature f):
  target response r[l0], strongest off-target response max over the other languages, margin = target - off-target maximum;
  share p[l] = x[l] / sum_l x[l] (undefined for an all-zero profile), target-anchored selectivity L * p[l0], global S = max r / mean r (0 for a never-firing feature, as
  phase1_lib.selectivity_S), Deng et al.'s nu at l0 = mu[l0] - mean of mu over the other languages (as phase1_lib.deng_nu);
  peak sets keep ties (a language is in the set when its response is within TIE_TOL of the maximum; an all-zero profile has no peak);
  total variation distance between two normalised profiles 0.5 * sum |p_quant - p_fp| (undefined, never 0, when either profile is all zero).
A common positive rescaling leaves shares and the distance unchanged, so amplitudes are always reported next to them. TIE_TOL is for tie handling only: it never rounds a
reported quantity. Thresholds of the status function reuse the tagging defaults (50 positives, panel-mean rate 1e-5) as operational safeguards, not as validity claims."""
import numpy as np

TIE_TOL = 1e-12
MIN_POSITIVES = 50
MIN_PANEL_RATE = 1e-5


class MetricsError(ValueError):
    pass


def _profile(x, name):
    a = np.asarray(x)
    if a.ndim != 2:
        raise MetricsError(f"{name} must be a [language, feature] array, got shape {a.shape}")
    a = a.astype(np.float64)
    if not np.isfinite(a).all():
        raise MetricsError(f"{name} contains non-finite values")
    if (a < 0).any():
        raise MetricsError(f"{name} contains negative values")
    return a


def _labels(original_peak, n_lang, n_feat):
    p = np.asarray(original_peak)
    if p.ndim != 1 or p.shape[0] != n_feat:
        raise MetricsError(f"original_peak must hold one language index per feature ({n_feat}), got shape {p.shape}")
    if p.dtype.kind not in "iu" or (p < 0).any() or (p >= n_lang).any():
        raise MetricsError(f"original_peak must be integer language indices in [0, {n_lang})")
    return p.astype(np.int64)


def _peak_set(x):
    """[L, F] boolean (languages within TIE_TOL of the maximum and above zero) and [F] counts."""
    mx = x.max(0)
    s = (x >= mx[None, :] - TIE_TOL) & (x > 0)
    return s, s.sum(0)


def _one(x, l0, n_lang, suffix):
    f = np.arange(x.shape[1]); target = x[l0, f]
    others = x.copy(); others[l0, f] = -np.inf; off = others.max(0)
    total = x.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(total > 0, target / total, np.nan)
    sets, count = _peak_set(x)
    return {f"target_{suffix}": target, f"offtarget_max_{suffix}": off, f"margin_{suffix}": target - off, f"share_{suffix}": share, f"peak_set_{suffix}": sets,
            f"peak_count_{suffix}": count, f"has_peak_{suffix}": count > 0, f"unique_peak_{suffix}": count == 1, f"peak_is_original_{suffix}": sets[l0, f]}


def association_metrics(rates, means, original_peak):
    """rates, means: [language, feature] firing rates and mean activations of ONE condition; original_peak: [feature] frozen dev peak language indices. -> dict of arrays."""
    r, m = _profile(rates, "rates"), _profile(means, "means")
    if r.shape != m.shape:
        raise MetricsError(f"rates {r.shape} and means {m.shape} differ in shape")
    n_lang, n_feat = r.shape
    if n_lang < 2:
        raise MetricsError("at least two languages are needed")
    l0 = _labels(original_peak, n_lang, n_feat); f = np.arange(n_feat)
    out = {"original_peak": l0, **_one(r, l0, n_lang, "rate"), **_one(m, l0, n_lang, "mean")}
    mean_r = r.mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["global_S"] = np.where(mean_r > 0, r.max(0) / np.maximum(mean_r, 1e-300), 0.0)
    out["target_anchored_S"] = n_lang * out["share_rate"]
    out["nu_target"] = m[l0, f] - (m.sum(0) - m[l0, f]) / (n_lang - 1)
    return out


def compare_profiles(fp, quant):
    """Absolute change and total variation distance between the full-precision and the quantized profile of every feature. `defined` is False where either profile is all
    zero (the distance is then NaN, never 0: an inactive feature is not a perfectly preserved one)."""
    a, b = _profile(fp, "fp"), _profile(quant, "quant")
    if a.shape != b.shape:
        raise MetricsError(f"fp {a.shape} and quant {b.shape} differ in shape")
    sa, sb = a.sum(0), b.sum(0); defined = (sa > 0) & (sb > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        tv = np.where(defined, 0.5 * np.abs(b / sb - a / sa).sum(0), np.nan)
    return {"delta": b - a, "tv_distance": tv, "defined": defined}


def association_status(counts, n_tokens, original_peak, min_positives=MIN_POSITIVES, min_panel_rate=MIN_PANEL_RATE):
    """Support, activity and peak status of every feature from positive-token counts [language, feature] and tokens per language [language].
    status: inactive (no positive token anywhere) | low_support (panel-mean rate below min_panel_rate or fewer than min_positives positives at the current peak)
    | ambiguous_peak (supported, several languages tied at the peak) | different_peak (supported, unique peak elsewhere) | retained (supported, unique peak at l0).
    Positives at a tied peak are the largest count among the tied languages. The target-specific flags are kept even when the peak moved."""
    c = np.asarray(counts)
    if c.ndim != 2 or c.dtype.kind not in "iu":
        raise MetricsError("counts must be an integer [language, feature] array")
    if (c < 0).any():
        raise MetricsError("counts contain negative values")
    n = np.asarray(n_tokens)
    if n.ndim != 1 or n.shape[0] != c.shape[0] or n.dtype.kind not in "iu" or (n <= 0).any():
        raise MetricsError("n_tokens must hold a positive integer token count for every language")
    n_lang, n_feat = c.shape
    l0 = _labels(original_peak, n_lang, n_feat); f = np.arange(n_feat)
    rates = c.astype(np.float64) / n.astype(np.float64)[:, None]
    sets, count = _peak_set(rates)
    pos_peak = np.where(sets, c, 0).max(0)
    inactive = c.sum(0) == 0; panel = rates.mean(0)
    low = ~inactive & ((panel < min_panel_rate) | (pos_peak < min_positives))
    supported = ~inactive & ~low
    ambiguous = supported & (count > 1); unique = supported & (count == 1)
    retained = unique & sets[l0, f]; different = unique & ~sets[l0, f]
    status = np.select([inactive, low, ambiguous, different, retained], ["inactive", "low_support", "ambiguous_peak", "different_peak", "retained"], default="unknown")
    assert not (status == "unknown").any()
    return {"status": status, "retained": retained, "inactive": inactive, "low_support": low, "ambiguous": ambiguous, "different_peak": different, "peak_count": count,
            "positives_at_peak": pos_peak, "positives_at_target": c[l0, f], "target_supported": c[l0, f] >= min_positives, "target_peak_in_set": sets[l0, f], "panel_rate": panel}
