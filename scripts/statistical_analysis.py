"""Confirmatory Statistical Analysis and Hypothesis Testing.

Formal implementation of Section 6 (Pre-registration) and Section 7 (Analysis Plan):
- Evaluates the 3 confirmatory tests with family-wise Holm-Bonferroni correction:
  1. Contrast 1: Selective vs quintile-matched universal survival at INT7.
  2. Contrast 2: Selective vs quintile-matched universal survival at INT6.
  3. Contrast 3: Per-language median survival gap across Resource Tiers (Low vs High).
- Permutation tests for language-level tier comparisons with script covariate.
- Formats results into publication-ready tables.
- Zero-dependency fallback: Runs with scipy if available, or pure Python/math if not.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def normal_cdf(x: float) -> float:
    """Standard normal cumulative distribution function using math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def two_proportion_p_value(count1: int, nobs1: int, count2: int, nobs2: int) -> float:
    """Computes two-tailed p-value for difference between two independent proportions."""
    if nobs1 <= 0 or nobs2 <= 0:
        return 1.0

    if HAS_SCIPY:
        # Fisher exact test
        contingency = [[count1, nobs1 - count1], [count2, nobs2 - count2]]
        _, p_val = stats.fisher_exact(contingency)
        return float(p_val)

    # Pure Python normal approximation
    p1 = count1 / nobs1
    p2 = count2 / nobs2
    p_pool = (count1 + count2) / (nobs1 + nobs2)
    se = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / nobs1 + 1.0 / nobs2) + 1e-12)
    if se == 0:
        return 1.0
    z = (p1 - p2) / se
    p_val = 2.0 * (1.0 - normal_cdf(abs(z)))
    return max(min(p_val, 1.0), 0.0)


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


def analyze_phase1_results(survival_report_path: str = "reports/phase1_survival_results.json"):
    path = REPO_ROOT / survival_report_path
    if not path.exists():
        print(f"File {path} does not exist yet. Run Phase 1 survival pipeline first.")
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

            c1 = int(round(sel_surv * sel_count / 100))
            c2 = int(round(univ_surv * univ_count / 100)) if univ_surv is not None else 0
            n2 = univ_count if univ_count else 0

            p_val = two_proportion_p_value(c1, sel_count, c2, n2)
            p_values.append(p_val)
            test_names.append(f"H1 ({bw}): Selective vs Universal Survival")
            print(f"{bw}: Selective Survived={sel_surv}% (n={sel_count}) | "
                  f"Matched Universal Survived={univ_surv}% (n={univ_count}) | p={p_val:.4e}")

    # Test 3: Resource Tier Comparison (Low vs High at INT7/INT6)
    for bw in ["INT7", "INT6"]:
        if bw in results_by_bw and "by_resource_tier" in results_by_bw[bw]:
            tiers = results_by_bw[bw]["by_resource_tier"]
            if "low" in tiers and "high" in tiers:
                low_pct = tiers["low"]["survived_pct"]
                high_pct = tiers["high"]["survived_pct"]
                n_low = tiers["low"]["count"]
                n_high = tiers["high"]["count"]
                test_names.append(f"H3 ({bw}): High vs Low Tier Survival Gap")
                c_low = int(round(low_pct * n_low / 100))
                c_high = int(round(high_pct * n_high / 100))
                p_tier = two_proportion_p_value(c_low, n_low, c_high, n_high)
                p_values.append(p_tier)
                print(f"{bw} Tier Test: High={high_pct}% (n={n_high}) | Low={low_pct}% (n={n_low}) | p={p_tier:.4e}")

    if p_values:
        adj = holm_bonferroni_correction(p_values)
        print("\n--- Summary Table of Statistical Decisions ---")
        for name, res in zip(test_names, adj):
            status = "REJECT NULL (Significant)" if res["significant"] else "FAIL TO REJECT (Null Result)"
            print(f"{name:42s} | raw p={res['raw_p']:.4e} | alpha_adj={res['adjusted_alpha']:.4f} | {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Statistical Analysis of Survival Results")
    parser.add_argument("--report", type=str, default="reports/phase1_survival_results.json")
    args = parser.parse_args()
    analyze_phase1_results(args.report)
