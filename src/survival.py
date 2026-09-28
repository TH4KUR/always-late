"""Feature activity / survival statistics, matching Duan (2026)'s protocol.

Activity rule: a feature is "active" over a corpus if its firing rate
(fraction of tokens with nonzero post-gate activation) exceeds 0.001.
Survival metric: per-feature Pearson correlation of activations between
the FP16 baseline and a compressed model, on identical tokens.
Thresholds: survive if r >= 0.9, damaged if r <= 0.5 (Section 4.2).
"""

from __future__ import annotations

import numpy as np


def firing_rate(acts: np.ndarray) -> np.ndarray:
    """acts: [n_tokens, n_features] -> [n_features] fraction of tokens firing."""
    return (acts > 0).mean(axis=0)


def active_feature_mask(acts: np.ndarray, threshold: float = 0.001) -> np.ndarray:
    return firing_rate(acts) > threshold


def pearson_corr_per_feature(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorized per-feature Pearson correlation across the token axis.

    a, b: [n_tokens, n_features], aligned on identical tokens.
    Returns NaN for features with zero variance in either array (e.g.
    never fired in this sample) rather than raising.
    """
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    num = (a * b).sum(axis=0)
    den = np.sqrt((a ** 2).sum(axis=0) * (b ** 2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    return r


def classify_survival(
    r: np.ndarray, ts: float = 0.9, td: float = 0.5
) -> np.ndarray:
    """Returns an array of labels: 'survived' (r>=ts), 'damaged' (r<=td),
    'partial' (td < r < ts), 'undefined' (NaN, e.g. dead feature)."""
    labels = np.full(r.shape, "partial", dtype=object)
    labels[np.isnan(r)] = "undefined"
    labels[(~np.isnan(r)) & (r >= ts)] = "survived"
    labels[(~np.isnan(r)) & (r <= td)] = "damaged"
    return labels


def selectivity_score(rate_by_lang: np.ndarray) -> np.ndarray:
    """Deng et al. (2025)-style selectivity score S(f), Section 4.1.

    rate_by_lang: [n_languages, n_features] mean per-language firing rate.
    S(f) = max_lang rate / L1-norm over languages of rate (per feature).
    """
    l1 = rate_by_lang.sum(axis=0)
    peak = rate_by_lang.max(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = peak / l1
    return s
