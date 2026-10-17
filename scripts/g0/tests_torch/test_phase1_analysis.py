"""S7 tests for scripts/g0/phase1_analysis.py on SYNTHETIC data with planted effects (decision D-014): feature table, stratified permutation contrast, cluster bootstrap,
per-language gaps, Freedman-Lane tier test with a script covariate, Holm, TOST. numpy only."""
import sys, unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import phase1_analysis as an   # noqa: E402


def make_inputs(n_lang=4, F=60, seed=0):
    rng = np.random.default_rng(seed)
    corr = {b: rng.uniform(0.2, 1.0, (n_lang, F)) for b in (7, 6)}
    fire = rng.uniform(0.0, 0.01, (n_lang, F)); fire[0, 5] = 0.0005                       # feature 5 is inactive at language 0 (below their 0.001 threshold)
    selective = np.zeros(F, bool); selective[:6] = True
    controls = np.arange(10, 16)
    peak = rng.integers(0, n_lang, F); peak[5] = 0
    sel = {"selective": selective, "controls": controls, "peak": peak, "S_rank": rng.uniform(0, 1, F), "strata": rng.integers(0, 5, F)}
    meta = [{"code": f"l{i}", "tier": ["Low", "Mid", "High", "High"][i], "script_family": ["A", "A", "B", "B"][i], "joshi_class": [1, 3, 5, 4][i]} for i in range(n_lang)]
    return corr, fire, sel, meta


class FeatureTable(unittest.TestCase):
    def test_rows_are_selective_plus_controls_read_at_the_peak_language(self):
        corr, fire, sel, meta = make_inputs(); t = an.feature_table(corr, fire, sel, meta, [7, 6], threshold=0.001)
        self.assertEqual(sorted(t["feature"].tolist()), list(range(6)) + list(range(10, 16)))
        for k, f in enumerate(t["feature"]):
            self.assertEqual(t["group"][k], int(f < 6)); self.assertEqual(t["lang"][k], sel["peak"][f])
            for b in (7, 6): self.assertEqual(t["r"][b][k], corr[b][sel["peak"][f], f])
            self.assertEqual(bool(t["evaluable"][k]), bool(fire[sel["peak"][f], f] > 0.001))
            self.assertEqual(t["tier"][k], meta[sel["peak"][f]]["tier"]); self.assertEqual(t["script_family"][k], meta[sel["peak"][f]]["script_family"])
        self.assertFalse(bool(t["evaluable"][list(t["feature"]).index(5)]))                        # the inactive feature is flagged, not silently kept

    def test_survival_is_strictly_above_the_cut(self):
        self.assertEqual(an.survived(np.array([0.9, 0.9000001, 1.0, 0.5])).tolist(), [False, True, True, False])


class StratifiedPermutation(unittest.TestCase):
    def data(self, effect, n=400, seed=1):
        rng = np.random.default_rng(seed); strata = rng.integers(0, 5, n); group = rng.integers(0, 2, n)
        p = 0.5 + 0.08 * (strata - 2) + effect * group
        return group, (rng.random(n) < p).astype(float), strata

    def test_delta_is_the_difference_of_group_means(self):
        g, y, s = self.data(0.3); r = an.stratified_permutation_contrast(g, y, s, n_perm=50, seed=0)
        self.assertAlmostEqual(r["delta"], y[g == 1].mean() - y[g == 0].mean())

    def test_a_planted_effect_is_detected_and_a_null_is_not(self):
        g, y, s = self.data(0.25); self.assertLess(an.stratified_permutation_contrast(g, y, s, 999, 0)["p"], 0.01)
        ps = [an.stratified_permutation_contrast(*self.data(0.0, seed=k), 199, k)["p"] for k in range(40)]
        self.assertLess(np.mean(np.array(ps) < 0.05), 0.15); self.assertGreater(np.mean(ps), 0.3)

    def test_labels_are_permuted_only_within_strata_so_a_stratum_effect_cannot_leak(self):
        strata = np.repeat([0, 1], 20); group = np.array([1] * 2 + [0] * 18 + [1] * 18 + [0] * 2); y = strata.astype(float)      # selective features sit mostly in the high stratum, outcome = stratum
        r = an.stratified_permutation_contrast(group, y, strata, 500, 0)
        self.assertAlmostEqual(r["delta"], 0.8); self.assertEqual(r["p"], 1.0)             # within-stratum permutation keeps the composition: delta is 0.8 in every permutation

    def test_the_test_is_two_sided_and_a_negative_effect_is_detected(self):
        g, y, s = self.data(-0.25); r = an.stratified_permutation_contrast(g, y, s, 999, 0); self.assertLess(r["delta"], -0.1); self.assertLess(r["p"], 0.01)

    def test_p_value_has_the_plus_one_correction_so_it_is_never_zero(self):
        g = np.array([0, 1] * 20); y = g.astype(float); r = an.stratified_permutation_contrast(g, y, np.zeros(40, int), 99, 0)
        self.assertEqual(r["delta"], 1.0); self.assertEqual(r["p"], 0.01)

    def test_reproducible_for_a_seed(self):
        g, y, s = self.data(0.1); a = an.stratified_permutation_contrast(g, y, s, 300, 5); b = an.stratified_permutation_contrast(g, y, s, 300, 5)
        self.assertEqual(a, b)


class ClusterBootstrap(unittest.TestCase):
    def test_interval_brackets_the_estimate_is_reproducible_and_degenerate_for_constant_outcomes(self):
        rng = np.random.default_rng(0); n = 300; lang = rng.integers(0, 5, n); g = rng.integers(0, 2, n); y = (rng.random(n) < 0.5 + 0.2 * g).astype(float)
        lo, hi = an.cluster_bootstrap_ci(g, y, lang, n_boot=400, seed=1); d = y[g == 1].mean() - y[g == 0].mean()
        self.assertLess(lo, d); self.assertGreater(hi, d); self.assertEqual((lo, hi), an.cluster_bootstrap_ci(g, y, lang, n_boot=400, seed=1))
        self.assertEqual(an.cluster_bootstrap_ci(g, np.ones(n), lang, n_boot=50, seed=0), (0.0, 0.0))

    def test_resampling_stays_within_language_and_group_so_group_sizes_are_fixed(self):
        g = np.array([0, 0, 1, 1, 0, 1]); lang = np.array([0, 0, 0, 0, 1, 1]); y = np.arange(6, dtype=float)
        idx = an._bootstrap_indices(g, lang, np.random.default_rng(0))
        self.assertEqual(sorted(zip(lang[idx], g[idx])), sorted(zip(lang, g)))


class Gaps(unittest.TestCase):
    def test_gap_is_median_r_of_selective_minus_median_r_of_controls_per_language_with_a_population_floor(self):
        lang = np.array([0] * 6 + [1] * 3); g = np.array([1, 1, 1, 0, 0, 0, 1, 0, 0]); r = np.array([.9, .8, .7, .95, .9, .6, .5, .4, .3])
        out = an.language_gaps(lang, g, r, np.ones(9, bool), n_lang=3, min_features=3)
        self.assertAlmostEqual(out["gap"][0], 0.8 - 0.9); self.assertEqual(out["n_selective"].tolist(), [3, 1, 0]); self.assertEqual(out["populated"].tolist(), [True, False, False])
        self.assertTrue(np.isnan(out["gap"][2]))

    def test_non_evaluable_features_are_left_out(self):
        lang = np.zeros(6, int); g = np.array([1, 1, 1, 0, 0, 0]); r = np.array([.9, .8, .1, .5, .5, .5]); ev = np.array([1, 1, 0, 1, 1, 1], bool)
        out = an.language_gaps(lang, g, r, ev, n_lang=1, min_features=2); self.assertAlmostEqual(out["gap"][0], 0.85 - 0.5); self.assertEqual(out["n_selective"].tolist(), [2])


class TierTest(unittest.TestCase):
    def panel(self, slope, script_shift, seed=0, n=27):
        rng = np.random.default_rng(seed); tier = np.repeat([0, 1, 2], n // 3); script = rng.integers(0, 3, n)
        gap = slope * tier + script_shift * script + rng.normal(0, 0.05, n)
        return gap, tier, script

    def test_a_planted_slope_is_detected_one_sided_and_the_opposite_sign_is_not(self):
        gap, tier, script = self.panel(0.05, 0.0); r = an.freedman_lane_tier_test(gap, tier, script, n_perm=2000, seed=0)
        self.assertGreater(r["slope"], 0.03); self.assertLess(r["p"], 0.01)
        r = an.freedman_lane_tier_test(-gap, tier, script, n_perm=2000, seed=0); self.assertGreater(r["p"], 0.9)

    @staticmethod
    def confounded(k):
        rng = np.random.default_rng(100 + k); tier = np.repeat([0, 1, 2], 9)
        script = np.where(tier == 0, rng.integers(0, 2, 27), rng.integers(1, 3, 27))                   # scripts correlate with tier
        return 0.2 * script + rng.normal(0, 0.05, 27), tier, script                                     # the gap depends on script only

    def test_script_differences_alone_do_not_produce_a_tier_effect(self):
        with_cov = [an.freedman_lane_tier_test(*self.confounded(k), n_perm=499, seed=k)["p"] < 0.05 for k in range(30)]
        self.assertLessEqual(sum(with_cov), 5)                                                          # nominal 5% of 30 = 1.5
        without = [an.freedman_lane_tier_test(g, t, np.zeros(27, int), n_perm=499, seed=k)["p"] < 0.05 for k, (g, t, _) in ((k, self.confounded(k)) for k in range(30))]
        self.assertGreater(sum(without), 15)                                                            # the same data without the covariate are rejected: the covariate matters

    def test_p_value_agrees_with_the_exact_freedman_lane_enumeration_on_a_tiny_design(self):
        gap = np.array([-0.6, -0.67, -0.69, 1.55, 3.75, 2.6]); tier = np.array([0, 1, 2, 0, 1, 2]); script = np.array([0, 0, 0, 1, 1, 1])
        r = an.freedman_lane_tier_test(gap, tier, script, n_perm=20000, seed=0)
        self.assertAlmostEqual(r["p"], 0.4, delta=0.02)               # all 720 permutations of the reduced-model residuals give 0.4; permuting the raw outcome would give 0.283

    def test_high_minus_low_contrast_is_reported(self):
        gap, tier, script = self.panel(0.05, 0.0); r = an.freedman_lane_tier_test(gap, tier, script, n_perm=50, seed=0)
        self.assertAlmostEqual(r["high_minus_low"], gap[tier == 2].mean() - gap[tier == 0].mean())

    def test_reproducible_for_a_seed(self):
        gap, tier, script = self.panel(0.02, 0.1); self.assertEqual(an.freedman_lane_tier_test(gap, tier, script, 300, 3), an.freedman_lane_tier_test(gap, tier, script, 300, 3))


class HolmAndTost(unittest.TestCase):
    def test_holm_adjusts_in_order_enforces_monotonicity_and_caps_at_one(self):
        self.assertTrue(np.allclose(an.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])); self.assertTrue(np.allclose(an.holm([0.5, 0.9]), [1.0, 1.0]))
        self.assertTrue(np.allclose(an.holm([0.001]), [0.001])); self.assertTrue(np.allclose(an.holm([0.6, 0.7, 0.9]), [1.0, 1.0, 1.0]))      # 3 x 0.6 would exceed 1

    def test_tost_declares_equivalence_only_when_the_90_percent_interval_is_inside_the_bounds(self):
        rng = np.random.default_rng(0); inside = rng.normal(0.0, 0.004, 4000); wide = rng.normal(0.0, 0.03, 4000); shifted = rng.normal(0.03, 0.004, 4000)
        self.assertTrue(an.tost(inside, 0.02)["equivalent"]); self.assertFalse(an.tost(wide, 0.02)["equivalent"]); self.assertFalse(an.tost(shifted, 0.02)["equivalent"])
        lo, hi = an.tost(inside, 0.02)["interval"]; self.assertAlmostEqual(lo, np.quantile(inside, 0.05)); self.assertAlmostEqual(hi, np.quantile(inside, 0.95))


if __name__ == "__main__":
    unittest.main()
