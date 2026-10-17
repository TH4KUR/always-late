"""Tests for scripts/g0/phase1_anchor.py: the S5 anchor compares our Phase 1 runner (wikitext, no BOS, their 3a_test budget) with the vendored smoke run of
their own script, with tolerances fixed BEFORE the run. stdlib only."""
import csv, sys, tempfile, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phase1_anchor as an  # noqa: E402

COLS = ["layer", "condition", "method", "bits", "perplexity", "ppl_delta_pct", "n_active_features", "mean_corr", "median_corr", "survived_>0.9_pct", "degraded_0.5_0.9_pct", "damaged_<0.5_pct", "n_tokens", "n_total_features", "loss"]
REF = [dict(layer=12, condition="FP16 baseline", method="FP16", bits=16, perplexity=414.07, ppl_delta_pct=0.0, n_active_features=6702, mean_corr=1.0, median_corr=1.0, **{"survived_>0.9_pct": 100.0, "degraded_0.5_0.9_pct": 0.0, "damaged_<0.5_pct": 0.0}, n_tokens=19968, n_total_features=16384, loss=6.026),
       dict(layer=12, condition="RTN INT8", method="RTN", bits=8, perplexity=422.62, ppl_delta_pct=2.06, n_active_features=6702, mean_corr=0.964, median_corr=0.971, **{"survived_>0.9_pct": 96.15, "degraded_0.5_0.9_pct": 3.85, "damaged_<0.5_pct": 0.0}, n_tokens=19968, n_total_features=16384, loss=6.046)]


def write(rows):
    p = Path(tempfile.mkdtemp()) / "s.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)
    return p


def ours(changes=None):
    rows = [dict(r) for r in REF]; rows[0]["bits"] = 32                      # our FP baseline row carries bits 32 (wandb schema v1)
    for (i, k), v in (changes or {}).items(): rows[i][k] = v
    return rows


class Compare(unittest.TestCase):
    def test_identical_runs_have_no_failures_and_fp_rows_match_despite_the_bits_label(self):
        r = an.compare_summary(write(ours()), write(REF)); self.assertEqual(r["failures"], []); self.assertEqual(r["compared_conditions"], ["FP16 baseline", "RTN INT8"])

    def test_values_inside_the_tolerance_pass_and_outside_fail_naming_the_field(self):
        ok = an.compare_summary(write(ours({(1, "perplexity"): 422.62 * 1.004, (1, "survived_>0.9_pct"): 96.15 + 0.4, (0, "n_active_features"): 6702 + 8})), write(REF))
        self.assertEqual(ok["failures"], [])
        bad = an.compare_summary(write(ours({(1, "perplexity"): 422.62 * 1.01})), write(REF)); self.assertEqual([f["field"] for f in bad["failures"]], ["perplexity"])
        bad = an.compare_summary(write(ours({(1, "survived_>0.9_pct"): 96.15 - 0.6})), write(REF)); self.assertEqual([f["field"] for f in bad["failures"]], ["survived_>0.9_pct"])
        bad = an.compare_summary(write(ours({(0, "n_active_features"): 6702 - 11})), write(REF)); self.assertEqual([f["field"] for f in bad["failures"]], ["n_active_features"])
        bad = an.compare_summary(write(ours({(1, "median_corr"): 0.971 + 0.006})), write(REF)); self.assertEqual([f["field"] for f in bad["failures"]], ["median_corr"])

    def test_no_common_condition_is_a_failure_not_a_pass(self):
        other = [dict(r, condition="RTN INT5") for r in REF[1:]]
        r = an.compare_summary(write(other), write(REF)); self.assertEqual(r["compared_conditions"], [])
        self.assertTrue(any(f["field"] == "conditions" for f in r["failures"]))                    # nothing in common: the anchor did not test anything

    def test_extra_conditions_in_ours_are_ignored_but_missing_ones_are_reported(self):
        extra = ours() + [dict(REF[1], condition="RTN INT4", bits=4)]
        self.assertEqual(an.compare_summary(write(extra), write(REF))["failures"], [])
        miss = an.compare_summary(write(ours()[:1]), write(REF)); self.assertTrue(any(f["field"] == "conditions" and "RTN INT8" in f["detail"] for f in miss["failures"]))


if __name__ == "__main__":
    unittest.main()
