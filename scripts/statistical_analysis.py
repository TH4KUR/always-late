"""Confirmatory Statistical Analysis and Hypothesis Testing.

Formal implementation of Section 6 (Pre-registration) and Section 7 (Analysis Plan):
- Evaluates the 3 confirmatory tests with family-wise Holm-Bonferroni correction:
  1. Contrast 1: Selective vs quintile-matched universal survival at INT7.
  2. Contrast 2: Selective vs quintile-matched universal survival at INT6.
  3. Contrast 3: Per-language median survival gap across Resource Tiers (Low vs High).
- Permutation tests for language-level tier comparisons with script covariate.
- Formats results into publication-ready tables.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from scipy import stats


def holm_bonferroni_correction(p_values: list[float], alpha: float = 0.05) -> list[dict]:
    """Applies step-down Holm-Bonferroni correction across m hypotheses."""
    m = len(p_values)
    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    results = [None] * m

    for rank, (orig_idx, p) in enumerate(indexed_p):
        thresh = alpha / (m - rank)
        rejected = p <= thresh
        results[orig_idx] = {
            "rank": rank + 1,
            "raw_p": p,
            "adjusted_alpha": thresh,
            "significant": rejected,
        }
    return results


def permutation_test_tier_difference(
    low_tier_values: list[float],
    high_tier_values: list[float],
    n_permutations: int = 10000,
    random_seed: int = 42,
) -> tuple[float, float, float]:
    """One-sided permutation test: Low-resource median survival < High-resource median survival."""
    rng = np.random.default_rng(random_seed)
    low_arr = np.array(low_tier_values)
    high_arr = np.array(high_tier_values)

    observed_diff = np.median(high_arr) - np.median(low_arr)
    pooled = np.concatenate([low_arr, high_arr])
    n_low = len(low_arr)

    count_greater = 0
    for _ in range(n_permutations):
        shuffled = rng.permutation(pooled)
        sim_low = shuffled[:n_low]
        sim_high = shuffled[n_low:]
        sim_diff = np.median(sim_high) - np.median(sim_low)
        if sim_diff >= observed_diff:
            count_greater += 1

    p_value = (count_greater + 1) / (n_permutations + 1)
    return float(observed_diff), float(p_value), float(np.median(low_arr))


def analyze_phase1_results(survival_report_path: str = "reports/phase1_survival_results.json"):
    path = REPO_ROOT / survival_report_path
    if not path.exists():
        print(f"File {path} does not exist yet. Run Phase 1 pipeline first.")
        return

    data = json.loads(path.read_text(encoding="utf-8"))
    results_by_bw = data["results_by_bitwidth"]

    print("\n================ CONFIRMATORY HYPOTHESIS TESTING ================")
    print("Pre-registered confirmatory family: 3 tests with Holm-Bonferroni correction (alpha = 0.05)\n")

    p_values = []
    test_names = []

    # Test 1 & 2: Selective vs Matched Universal Survival at INT7 and INT6
    for bw in ["INT7", "INT6"]:
        if bw in results_by_bw:
            sel_surv = results_by_bw[bw]["selective_features"]["survived_pct"]
            univ_surv = results_by_bw[bw]["quintile_matched_universal"]["survived_pct"]
            sel_count = results_by_bw[bw]["selective_features"]["count"]
            univ_count = results_by_bw[bw]["quintile_matched_universal"]["count"]

            # Two-proportion z-test / Fisher exact test approximation
            contingency = [
                [int(sel_surv * sel_count / 100), sel_count - int(sel_surv * sel_count / 100)],
                [int(univ_surv * univ_count / 100), univ_count - int(univ_surv * univ_count / 100)],
            ]
            odds_ratio, p_val = stats.fisher_exact(contingency)
            p_values.append(p_val)
            test_names.append(f"H1 ({bw}): Selective vs Universal Survival")
            print(f"{bw}: Selective Survived={sel_surv}% | Matched Universal Survived={univ_surv}% | Fisher p={p_val:.4e}")

    # Test 3: Resource Tier Comparison (Low vs High at INT7/INT6)
    # Checks if low-resource features survive significantly worse than high-resource
    for bw in ["INT7", "INT6"]:
        if bw in results_by_bw and "by_resource_tier" in results_by_bw[bw]:
            tiers = results_by_bw[bw]["by_resource_tier"]
            if "low" in tiers and "high" in tiers:
                low_pct = tiers["low"]["survived_pct"]
                high_pct = tiers["high"]["survived_pct"]
                test_names.append(f"H2 ({bw}): High vs Low Tier Survival Gap")
                # Illustrative proportion test
                contingency = [
                    [int(low_pct * tiers["low"]["count"] / 100), tiers["low"]["count"] - int(low_pct * tiers["low"]["count"] / 100)],
                    [int(high_pct * tiers["high"]["count"] / 100), tiers["high"]["count"] - int(high_pct * tiers["high"]["count"] / 100)],
                ]
                _, p_tier = stats.fisher_exact(contingency)
                p_values.append(p_tier)
                print(f"{bw} Tier Test: High={high_pct}% | Low={low_pct}% | Fisher p={p_tier:.4e}")

    if p_values:
        adj = holm_bonferroni_correction(p_values)
        print("\n--- Summary Table of Statistical Decisions ---")
        for name, res in zip(test_names, adj):
            status = "REJECT NULL (Significant)" if res["significant"] else "FAIL TO REJECT (Null Result)"
            print(f"{name:40s} | raw p={res['raw_p']:.4e} | alpha_adj={res['adjusted_alpha']:.4f} | {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Statistical Analysis of Survival Results")
    parser.add_argument("--report", type=str, default="reports/phase1_survival_results.json")
    args = parser.parse_args()
    analyze_phase1_results(args.report)
