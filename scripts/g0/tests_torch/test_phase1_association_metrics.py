"""D-035 Task 3 tests for scripts/g0/phase1_association_metrics.py (design: reports/survival_method_reassessment.md section 7; plan Task 3). Pure NumPy functions on [language, feature]
arrays; hand-computed fixtures. The cases are the plan's: identity, uniform positive scaling (normalised profiles identical, absolute means change), off-target broadening with an
unchanged target response (target share falls), a full language switch, equal-height peaks kept as sets, inactivity (no peak, undefined distance), weak support, and a dev-selected
peak that the full-precision evaluation does not reproduce. S and Deng's nu are compared with the existing phase1_lib definitions on shared fixtures."""
import sys, unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import phase1_association_metrics as am   # noqa: E402
import phase1_lib as p1   # noqa: E402

L = 4


def col(*values):
    """one feature's profile over L languages as an [L, 1] array"""
    return np.array(values, dtype=np.float64).reshape(-1, 1)


class MetricsTests(unittest.TestCase):
    def test_target_offtarget_margin_and_shares_of_a_hand_computed_profile(self):
        r = col(0.00, 0.06, 0.02, 0.02); m = col(0.0, 0.3, 0.1, 0.1)
        out = am.association_metrics(r, m, np.array([1]))
        self.assertAlmostEqual(out["target_rate"][0], 0.06); self.assertAlmostEqual(out["offtarget_max_rate"][0], 0.02); self.assertAlmostEqual(out["margin_rate"][0], 0.04)
        self.assertAlmostEqual(out["share_rate"][0], 0.6)                                         # 0.06 / 0.10
        self.assertAlmostEqual(out["target_anchored_S"][0], L * 0.6)
        self.assertAlmostEqual(out["global_S"][0], 0.06 / (0.10 / L))
        self.assertAlmostEqual(out["target_mean"][0], 0.3); self.assertAlmostEqual(out["share_mean"][0], 0.6)
        self.assertAlmostEqual(out["nu_target"][0], 0.3 - (0.0 + 0.1 + 0.1) / 3)

    def test_global_S_and_nu_agree_with_the_existing_definitions(self):
        g = np.random.default_rng(0); r = g.random((L, 9)) * (g.random((L, 9)) > 0.4); m = g.random((L, 9)) * (r > 0)
        out = am.association_metrics(r, m, g.integers(0, L, 9))
        np.testing.assert_allclose(out["global_S"], p1.selectivity_S(torch.tensor(r)).numpy(), rtol=1e-12)
        nu = p1.deng_nu(torch.tensor(m)).numpy(); peak = out["original_peak"]
        np.testing.assert_allclose(out["nu_target"], nu[peak, np.arange(9)], rtol=1e-12, atol=1e-15)

    def test_target_anchored_S_equals_global_S_when_the_target_is_the_peak(self):
        r = col(0.01, 0.05, 0.0, 0.01); out = am.association_metrics(r, r, np.array([1]))
        self.assertAlmostEqual(out["target_anchored_S"][0], out["global_S"][0])
        out2 = am.association_metrics(r, r, np.array([0]))                                          # a non-peak target: anchored S is smaller than global S
        self.assertLess(out2["target_anchored_S"][0], out2["global_S"][0])

    def test_equal_height_peaks_are_kept_as_a_set_not_resolved_by_language_order(self):
        r = col(0.05, 0.0, 0.05, 0.01)
        out = am.association_metrics(r, r, np.array([2]))
        self.assertEqual(out["peak_set_rate"][:, 0].tolist(), [True, False, True, False]); self.assertEqual(int(out["peak_count_rate"][0]), 2)
        self.assertTrue(out["peak_is_original_rate"][0]); self.assertFalse(out["unique_peak_rate"][0])

    def test_peak_is_original_is_false_when_the_peak_is_elsewhere(self):
        out = am.association_metrics(col(0.05, 0.01, 0.0, 0.0), col(0.05, 0.01, 0.0, 0.0), np.array([1]))
        self.assertTrue(out["has_peak_rate"][0]); self.assertFalse(out["peak_is_original_rate"][0]); self.assertFalse(out["peak_is_original_mean"][0])
        self.assertTrue(out["unique_peak_rate"][0]); self.assertEqual(out["peak_set_rate"][:, 0].tolist(), [True, False, False, False])

    def test_tie_tolerance_is_1e_12_and_applies_to_ties_only(self):
        close = am.association_metrics(col(0.05, 0.05 - 1e-13, 0, 0), col(0, 0, 0, 0), np.array([0]))
        far = am.association_metrics(col(0.05, 0.05 - 1e-9, 0, 0), col(0, 0, 0, 0), np.array([0]))
        self.assertEqual(int(close["peak_count_rate"][0]), 2); self.assertEqual(int(far["peak_count_rate"][0]), 1)
        self.assertAlmostEqual(far["margin_rate"][0], 1e-9, places=12)                           # the tolerance never rounds a difference in a reported quantity

    def test_all_zero_profile_has_no_peak_and_undefined_shares(self):
        out = am.association_metrics(col(0, 0, 0, 0), col(0, 0, 0, 0), np.array([1]))
        self.assertEqual(int(out["peak_count_rate"][0]), 0); self.assertFalse(out["has_peak_rate"][0]); self.assertFalse(out["peak_is_original_rate"][0])
        self.assertTrue(np.isnan(out["share_rate"][0])); self.assertTrue(np.isnan(out["target_anchored_S"][0])); self.assertEqual(out["global_S"][0], 0.0)

    def test_rate_and_mean_peak_sets_are_separate(self):
        out = am.association_metrics(col(0.05, 0.02, 0, 0), col(0.1, 0.9, 0, 0), np.array([0]))
        self.assertEqual(out["peak_set_rate"][:, 0].tolist(), [True, False, False, False]); self.assertEqual(out["peak_set_mean"][:, 0].tolist(), [False, True, False, False])

    def test_features_are_independent_columns(self):
        g = np.random.default_rng(1); r = g.random((L, 6)); m = g.random((L, 6)); pk = g.integers(0, L, 6); perm = g.permutation(6)
        a = am.association_metrics(r, m, pk); b = am.association_metrics(r[:, perm], m[:, perm], pk[perm])
        for k in ("target_rate", "margin_rate", "share_rate", "target_anchored_S", "nu_target"):
            np.testing.assert_allclose(a[k][perm], b[k], rtol=1e-12)

    def test_float32_input_is_accepted_and_reported_in_float64(self):
        out = am.association_metrics(col(0.1, 0.2, 0, 0).astype(np.float32), col(0.1, 0.2, 0, 0).astype(np.float32), np.array([1]))
        self.assertEqual(out["target_rate"].dtype, np.float64)

    def test_invalid_inputs_raise(self):
        ok = col(0.1, 0.2, 0.0, 0.0)
        for bad in (col(-0.1, 0.2, 0, 0), col(np.nan, 0.2, 0, 0), col(np.inf, 0.2, 0, 0)):
            with self.assertRaises(am.MetricsError):
                am.association_metrics(bad, ok, np.array([0]))
        with self.assertRaises(am.MetricsError):
            am.association_metrics(ok, np.zeros((L + 1, 1)), np.array([0]))                       # shape mismatch
        with self.assertRaises(am.MetricsError):
            am.association_metrics(ok, ok, np.array([L]))                                         # peak language out of range
        with self.assertRaises(am.MetricsError):
            am.association_metrics(ok, ok, np.array([0, 1]))                                      # one peak label per feature
        with self.assertRaises(am.MetricsError):
            am.association_metrics(ok.ravel(), ok.ravel(), np.array([0]))                         # not [language, feature]


class CompareTests(unittest.TestCase):
    def test_identity_has_zero_change_and_zero_distance(self):
        x = col(0.1, 0.2, 0.0, 0.3); c = am.compare_profiles(x, x)
        self.assertEqual(float(np.abs(c["delta"]).max()), 0.0); self.assertEqual(c["tv_distance"][0], 0.0); self.assertTrue(c["defined"][0])

    def test_uniform_scaling_leaves_normalised_profiles_unchanged_but_not_amplitudes(self):
        x = col(0.1, 0.2, 0.0, 0.3); c = am.compare_profiles(x, 2.5 * x)
        self.assertAlmostEqual(c["tv_distance"][0], 0.0, places=15); self.assertGreater(float(np.abs(c["delta"]).max()), 0.1)

    def test_broadening_with_an_unchanged_target_changes_shares_and_distance(self):
        fp = col(0.0, 0.01, 0.0, 0.0); q = col(0.0, 0.01, 0.01, 0.0)
        c = am.compare_profiles(fp, q); self.assertAlmostEqual(c["tv_distance"][0], 0.5)
        a, b = am.association_metrics(fp, fp, np.array([1])), am.association_metrics(q, q, np.array([1]))
        self.assertEqual(a["target_rate"][0], b["target_rate"][0]); self.assertAlmostEqual(a["share_rate"][0], 1.0); self.assertAlmostEqual(b["share_rate"][0], 0.5)
        self.assertAlmostEqual(b["margin_rate"][0], 0.0)                                          # the margin vanishes although the target response is unchanged

    def test_hand_computed_distance_of_a_language_switch(self):
        c = am.compare_profiles(col(0.5, 0.5, 0.0, 0.0), col(0.5, 0.0, 0.5, 0.0)); self.assertAlmostEqual(c["tv_distance"][0], 0.5)
        c2 = am.compare_profiles(col(1, 0, 0, 0), col(0, 1, 0, 0)); self.assertAlmostEqual(c2["tv_distance"][0], 1.0)

    def test_all_zero_vectors_are_undefined_not_perfectly_preserved(self):
        c = am.compare_profiles(col(0, 0, 0, 0), col(0, 0, 0, 0))
        self.assertFalse(c["defined"][0]); self.assertTrue(np.isnan(c["tv_distance"][0]))
        c2 = am.compare_profiles(col(0.1, 0.2, 0, 0), col(0, 0, 0, 0)); self.assertFalse(c2["defined"][0]); self.assertTrue(np.isnan(c2["tv_distance"][0]))
        self.assertAlmostEqual(float(c2["delta"][1, 0]), -0.2)                                    # the absolute change is still reported

    def test_shape_mismatch_and_negative_values_raise(self):
        with self.assertRaises(am.MetricsError):
            am.compare_profiles(np.zeros((L, 2)), np.zeros((L, 3)))
        with self.assertRaises(am.MetricsError):
            am.compare_profiles(col(-1, 0, 0, 0), col(0, 0, 0, 0))


def counts_col(*c):
    return np.array(c, dtype=np.int64).reshape(-1, 1)


NT = np.full(L, 20000, dtype=np.int64)


class StatusTests(unittest.TestCase):
    def test_retained_unique_original_peak_with_support(self):
        s = am.association_status(counts_col(0, 80, 20, 10), NT, np.array([1]))
        self.assertEqual(s["status"][0], "retained"); self.assertTrue(s["retained"][0]); self.assertTrue(s["target_supported"][0]); self.assertEqual(int(s["positives_at_target"][0]), 80)

    def test_a_different_unique_peak_is_reported_with_target_support_kept_as_a_flag(self):
        s = am.association_status(counts_col(0, 60, 90, 10), NT, np.array([1]))
        self.assertEqual(s["status"][0], "different_peak"); self.assertFalse(s["retained"][0]); self.assertTrue(s["target_supported"][0])      # 60 positives remain at the target

    def test_an_ambiguous_peak_set_is_neither_retained_nor_a_switch(self):
        s = am.association_status(counts_col(0, 80, 80, 10), NT, np.array([1]))
        self.assertEqual(s["status"][0], "ambiguous_peak"); self.assertFalse(s["retained"][0]); self.assertTrue(s["target_peak_in_set"][0])

    def test_inactive_everywhere(self):
        s = am.association_status(counts_col(0, 0, 0, 0), NT, np.array([1])); self.assertEqual(s["status"][0], "inactive"); self.assertTrue(s["inactive"][0]); self.assertFalse(s["retained"][0])

    def test_fewer_than_50_positives_at_the_peak_is_low_support_even_with_the_original_peak(self):
        s = am.association_status(counts_col(0, 49, 10, 5), NT, np.array([1])); self.assertEqual(s["status"][0], "low_support"); self.assertFalse(s["retained"][0])
        s50 = am.association_status(counts_col(0, 50, 10, 5), NT, np.array([1])); self.assertEqual(s50["status"][0], "retained")          # the rule is "at least 50"

    def test_panel_rate_below_the_floor_is_low_support(self):
        s = am.association_status(counts_col(0, 80, 20, 10), np.full(L, 10 ** 9, dtype=np.int64), np.array([1]))
        self.assertEqual(s["status"][0], "low_support"); self.assertLess(s["panel_rate"][0], 1e-5)

    def test_full_precision_evaluation_that_does_not_reproduce_the_dev_peak_is_not_established(self):
        s = am.association_status(counts_col(0, 5, 70, 0), NT, np.array([1]))
        self.assertEqual(s["status"][0], "different_peak"); self.assertFalse(s["retained"][0]); self.assertFalse(s["target_supported"][0])

    def test_unequal_token_counts_use_rates_for_the_peak_and_counts_for_support(self):
        s = am.association_status(counts_col(60, 80, 0, 0), np.array([1000, 100000, 20000, 20000]), np.array([0]))                 # language 0 has the higher RATE (0.06 vs 0.0008)
        self.assertEqual(s["status"][0], "retained"); self.assertEqual(int(s["positives_at_peak"][0]), 60)

    def test_vectorised_over_features(self):
        c = np.array([[0, 5, 0], [80, 0, 0], [20, 90, 0], [10, 0, 0]], dtype=np.int64)
        s = am.association_status(c, NT, np.array([1, 0, 0])); self.assertEqual(s["status"].tolist(), ["retained", "different_peak", "inactive"])

    def test_invalid_inputs_raise(self):
        with self.assertRaises(am.MetricsError):
            am.association_status(counts_col(0, -1, 0, 0), NT, np.array([1]))
        with self.assertRaises(am.MetricsError):
            am.association_status(np.array([[0.0], [1.5], [0.0], [0.0]]), NT, np.array([1]))      # counts are integers
        with self.assertRaises(am.MetricsError):
            am.association_status(counts_col(0, 5, 0, 0), np.array([20000, 0, 20000, 20000]), np.array([1]))   # a language without tokens
        with self.assertRaises(am.MetricsError):
            am.association_status(counts_col(0, 5, 0, 0), NT[:3], np.array([1]))
        with self.assertRaises(am.MetricsError):
            am.association_status(counts_col(0, 5, 0, 0), NT, np.array([7]))


if __name__ == "__main__":
    unittest.main()
