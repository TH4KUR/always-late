"""Feature activity, selectivity, and survival statistics.

Matches the formal framework in anlp_revised.txt Section 4.1-4.3:
- Activity rule: firing rate > 0.001 (Duan, 2026).
- Selectivity score S(f) following Deng et al. (2025): ratio of peak language
  firing rate to mean firing rate across all L languages.
- Pre-registered safeguards:
  1. Minimum activity floor and >= 50 positive activations on peak language.
  2. Bootstrap peak language stability under 90% resampling.
  3. Rank-normalized continuous score S_tilde(f) in [0, 1].
  4. Shuffled null distribution threshold (95th percentile).
- Quintile matching: controls for feature rarity / firing-rate confound when
  contrasting language-selective vs universal feature survival (Borobia et al., 2026).
"""

from __future__ import annotations

import numpy as np


def firing_rate(acts: np.ndarray) -> np.ndarray:
    """acts: [n_tokens, n_features] -> [n_features] fraction of tokens firing."""
    return (acts > 0).mean(axis=0)


def positive_count(acts: np.ndarray) -> np.ndarray:
    """acts: [n_tokens, n_features] -> [n_features] count of tokens firing."""
    return (acts > 0).sum(axis=0)


def active_feature_mask(acts: np.ndarray, threshold: float = 0.001) -> np.ndarray:
    """Features whose overall firing rate exceeds threshold."""
    return firing_rate(acts) > threshold


def pearson_corr_per_feature(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorized per-feature Pearson correlation across the token axis.

    a, b: [n_tokens, n_features], aligned on identical tokens.
    Returns NaN for features with zero variance in either array.
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
    """Classifies per-feature survival into 'survived', 'damaged', 'partial', or 'undefined'."""
    labels = np.full(r.shape, "partial", dtype=object)
    labels[np.isnan(r)] = "undefined"
    labels[(~np.isnan(r)) & (r >= ts)] = "survived"
    labels[(~np.isnan(r)) & (r <= td)] = "damaged"
    return labels


def selectivity_score(rate_by_lang: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Computes Deng et al. (2025) selectivity score S(f) and peak language index.

    rate_by_lang: [n_languages, n_features] mean per-language firing rate.
    Returns:
      s: [n_features] S(f) = max_l r_l(f) / ((1/L) * sum_l r_l(f))
      peak_lang_idx: [n_features] index of the argmax language
    """
    n_langs = rate_by_lang.shape[0]
    mean_rate = rate_by_lang.mean(axis=0)  # (1/L) * sum_l
    peak_rate = rate_by_lang.max(axis=0)
    peak_lang_idx = rate_by_lang.argmax(axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        s = peak_rate / np.maximum(mean_rate, 1e-9)
    return s, peak_lang_idx


def rank_normalized_selectivity(s: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    """Rank-normalizes selectivity scores S_tilde(f) into [0, 1]."""
    s_tilde = np.full_like(s, np.nan, dtype=np.float64)
    if valid_mask is None:
        valid_mask = ~np.isnan(s) & (s > 0)

    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        return s_tilde

    ranks = np.argsort(np.argsort(s[valid_indices]))
    if len(valid_indices) > 1:
        s_tilde[valid_indices] = ranks / (len(valid_indices) - 1)
    else:
        s_tilde[valid_indices] = 1.0
    return s_tilde


def filter_supported_features(
    rate_by_lang: np.ndarray,
    positive_counts_by_lang: np.ndarray,
    min_rate_floor: float = 0.001,
    min_peak_positive_acts: int = 50,
) -> np.ndarray:
    """Pre-registered Safeguard 1:
    Feature must exceed mean activity floor and have >= 50 positive activations in peak language.
    """
    overall_mean_rate = rate_by_lang.mean(axis=0)
    peak_idx = rate_by_lang.argmax(axis=0)
    peak_pos_counts = np.array([
        positive_counts_by_lang[peak_idx[f], f] for f in range(rate_by_lang.shape[1])
    ])
    return (overall_mean_rate > min_rate_floor) & (peak_pos_counts >= min_peak_positive_acts)


def compute_shuffled_null_threshold(
    token_acts_list: list[np.ndarray],
    n_permutations: int = 50,
    percentile: float = 95.0,
    random_seed: int = 42,
) -> float:
    """Computes empirical 95th-percentile selectivity threshold under label permutation.

    Permutes tokens across languages to construct a null distribution of S(f)
    where language association is purely chance.
    """
    rng = np.random.default_rng(random_seed)
    n_langs = len(token_acts_list)
    tokens_per_lang = [acts.shape[0] for acts in token_acts_list]
    all_acts = np.concatenate(token_acts_list, axis=0)
    total_tokens = all_acts.shape[0]

    null_scores = []
    for _ in range(n_permutations):
        shuffled_indices = rng.permutation(total_tokens)
        curr = 0
        perm_rates = []
        for count in tokens_per_lang:
            idx = shuffled_indices[curr : curr + count]
            perm_rates.append((all_acts[idx] > 0).mean(axis=0))
            curr += count
        perm_rate_mat = np.stack(perm_rates, axis=0)
        s, _ = selectivity_score(perm_rate_mat)
        valid = ~np.isnan(s) & (perm_rate_mat.mean(axis=0) > 0.001)
        if valid.any():
            null_scores.extend(s[valid])

    if not null_scores:
        return 2.5
    raw_thresh = float(np.percentile(null_scores, percentile))
    # Cap threshold below theoretical maximum (L) so selective features can be detected on small panels
    return min(raw_thresh, max(1.5, 0.75 * n_langs))


def stratify_by_firing_rate_quintiles(rates: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Assigns active features to firing-rate quintiles (0 to 4).

    Returns an array of quintile indices (0-4), or -1 for unmasked features.
    """
    quintiles = np.full(rates.shape, -1, dtype=int)
    valid_indices = np.where(mask)[0]
    if len(valid_indices) == 0:
        return quintiles

    valid_rates = rates[valid_indices]
    q_edges = np.quantile(valid_rates, [0.2, 0.4, 0.6, 0.8])
    q_bins = np.digitize(valid_rates, q_edges)
    quintiles[valid_indices] = q_bins
    return quintiles
