"""S7 tests for scripts/g0/phase1_analysis.py on SYNTHETIC data with planted effects (decision D-014): feature table, stratified permutation contrast, cluster bootstrap,
per-language gaps, Freedman-Lane tier test with a script covariate, Holm, TOST. numpy only."""
import shutil, sys, unittest
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

    def test_adjusted_values_subtract_the_stratum_median_r_and_the_stratum_survival_rate(self):
        corr, fire, sel, meta = make_inputs(); t = an.feature_table(corr, fire, sel, meta, [7, 6], threshold=0.001)
        for b in (7, 6):
            r = t["r"][b]; y = an.survived(r).astype(float)
            for st in np.unique(t["stratum"][t["evaluable"]]):
                m = t["evaluable"] & (t["stratum"] == st)
                self.assertTrue(np.allclose(t["r_adj"][b][m], r[m] - np.median(r[m]))); self.assertTrue(np.allclose(t["y_adj"][b][m], y[m] - y[m].mean()))
                self.assertAlmostEqual(float(np.median(t["r_adj"][b][m])), 0.0); self.assertAlmostEqual(float(t["y_adj"][b][m].mean()), 0.0)

    def test_survival_is_strictly_above_the_cut(self):
        self.assertEqual(an.survived(np.array([0.9, 0.9000001, 1.0, 0.5])).tolist(), [False, True, True, False])


class StratifiedDelta(unittest.TestCase):
    """The contrast is standardised to the selective group's stratum distribution: sum_s n1s (ybar1s - ybar0s) / sum n1s over strata that hold both groups (D-016)."""
    def test_hand_computed_value_and_dropped_strata(self):
        g = np.array([1, 1, 1, 0, 0, 1, 0, 1, 1]); s_ = np.array([0, 0, 0, 0, 0, 1, 1, 2, 2]); y = np.array([1., 1., 0., 1., 0., 1., 0., 1., 1.])
        # stratum 0: sel 2/3, ctrl 1/2 -> +1/6 (n1=3); stratum 1: sel 1, ctrl 0 -> +1 (n1=1); stratum 2: only selective -> dropped
        r = an.stratified_delta(g, y, s_); self.assertAlmostEqual(r["delta"], (3 * (2 / 3 - 1 / 2) + 1 * 1.0) / 4); self.assertEqual(r["n_strata_used"], 2); self.assertEqual(r["n_selective_dropped"], 2)

    def test_equals_the_plain_difference_when_the_two_groups_have_the_same_composition(self):
        rng = np.random.default_rng(0); s_ = np.repeat(np.arange(5), 40); g = np.tile([0, 1], 100); y = (rng.random(200) < 0.4 + 0.1 * (s_ // 2)).astype(float)
        self.assertAlmostEqual(an.stratified_delta(g, y, s_)["delta"], y[g == 1].mean() - y[g == 0].mean())

    def test_a_pure_composition_difference_gives_zero(self):
        s_ = np.repeat([0, 1], 40); g = np.array([1] * 30 + [0] * 10 + [1] * 10 + [0] * 30); y = s_.astype(float)                # selective features sit mostly in the low stratum, outcome = stratum
        self.assertAlmostEqual(an.stratified_delta(g, y, s_)["delta"], 0.0); self.assertLess(y[g == 1].mean() - y[g == 0].mean(), -0.4)         # while the plain difference is -0.5

    def test_no_stratum_with_both_groups_is_an_error(self):
        with self.assertRaises(ValueError): an.stratified_delta(np.array([1, 0]), np.array([1., 0.]), np.array([0, 1]))


class StratifiedPermutation(unittest.TestCase):
    def data(self, effect, n=400, seed=1):
        rng = np.random.default_rng(seed); strata = rng.integers(0, 5, n); group = rng.integers(0, 2, n)
        p = 0.5 + 0.08 * (strata - 2) + effect * group
        return group, (rng.random(n) < p).astype(float), strata

    def test_delta_is_the_stratum_standardised_difference(self):
        g, y, s = self.data(0.3); r = an.stratified_permutation_contrast(g, y, s, n_perm=50, seed=0)
        self.assertAlmostEqual(r["delta"], an.stratified_delta(g, y, s)["delta"]); self.assertEqual(r["n_selective_dropped"], 0)

    def test_a_planted_effect_is_detected_and_a_null_is_not(self):
        g, y, s = self.data(0.25); self.assertLess(an.stratified_permutation_contrast(g, y, s, 999, 0)["p"], 0.01)
        ps = [an.stratified_permutation_contrast(*self.data(0.0, seed=k), 199, k)["p"] for k in range(40)]
        self.assertLess(np.mean(np.array(ps) < 0.05), 0.15); self.assertGreater(np.mean(ps), 0.3)

    def test_labels_are_permuted_only_within_strata_so_a_stratum_effect_cannot_leak(self):
        rng = np.random.default_rng(3); strata = np.repeat([0, 1], 1000); group = np.concatenate([np.array([1] * 800 + [0] * 200), np.array([1] * 200 + [0] * 800)])
        rate = np.where(strata == 0, 0.8, 0.3) + 0.0 * group                                                                      # outcome depends on the stratum only
        y = (rng.random(2000) < rate).astype(float); r = an.stratified_permutation_contrast(group, y, strata, 999, 0)
        self.assertLess(abs(r["delta"]), 0.08); self.assertGreater(r["p"], 0.05)           # selective features sit in the high stratum: the plain difference is about 0.3, the stratified one is 0
        self.assertGreater(y[group == 1].mean() - y[group == 0].mean(), 0.25)
        r2 = an.stratified_permutation_contrast(group, (rng.random(2000) < rate + 0.1 * group).astype(float), strata, 999, 0); self.assertGreater(r2["delta"], 0.04); self.assertLess(r2["p"], 0.05)   # a true within-stratum effect is still found

    def test_p_value_equals_the_exact_within_stratum_enumeration_on_a_tiny_design(self):
        strata = np.repeat([0, 1], 6); group = np.array([1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0])
        y = np.array([-0.54, -0.04, -0.08, -0.04, -0.03, -0.65, 0.95, 1.34, 3.07, 2.37, 2.59, 3.38])
        r = an.stratified_permutation_contrast(group, y, strata, n_perm=20000, seed=0)
        self.assertAlmostEqual(r["delta"], -0.45916666666666667, places=10)
        self.assertAlmostEqual(r["p"], 0.15556, delta=0.015)             # all 15 x 15 = 225 within-stratum label assignments; permuting labels globally would give 0.330

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

    def test_with_strata_the_interval_is_for_the_stratum_standardised_difference(self):
        rng = np.random.default_rng(0); strata = np.repeat([0, 1], 400); group = np.concatenate([np.array([1] * 300 + [0] * 100), np.array([1] * 100 + [0] * 300)])
        lang = rng.integers(0, 4, 800); y = strata.astype(float)                                   # outcome = stratum: no group effect inside any stratum
        lo, hi = an.cluster_bootstrap_ci(group, y, lang, 300, 0, strata=strata); self.assertTrue(lo <= 0.0 <= hi)
        lo2, hi2 = an.cluster_bootstrap_ci(group, y, lang, 300, 0); self.assertLess(hi2, -0.3)       # the plain difference is about -0.5 (selective features sit in the low stratum)

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

    def test_tost_without_any_valid_bootstrap_draw_is_undefined_not_a_crash(self):
        r = an.tost(np.array([]), 0.02)                          # found with fixed S thresholds (N2): few selective features, every tier-slope draw skipped
        self.assertIsNone(r["interval"]); self.assertIsNone(r["equivalent"]); self.assertEqual(r["n_draws"], 0); self.assertEqual(r["bound"], 0.02)

    def test_tost_of_ordinary_input_keeps_exactly_its_original_keys_so_earlier_outputs_stay_byte_identical(self):
        self.assertEqual(set(an.tost(np.linspace(-0.01, 0.01, 50), 0.02)), {"interval", "equivalent", "bound"})

    def test_the_tost_sentence_keeps_its_original_wording_and_says_undefined_when_there_is_no_interval(self):
        ok = {"tost_slope": {"interval": (-0.0075, -0.0016), "equivalent": True, "bound": 0.02}, "tost_high_minus_low": {"interval": (-0.0129, -0.0031), "equivalent": True, "bound": 0.02}}
        self.assertEqual(an._tost_text(ok), "TOST (90% bootstrap interval vs bound): slope -0.0075..-0.0016, equivalent within +-0.02: True; High minus Low -0.0129..-0.0031, equivalent: True.")
        bad = {"tost_slope": {"interval": None, "equivalent": None, "bound": 0.02, "n_draws": 0}, "tost_high_minus_low": ok["tost_high_minus_low"]}
        text = an._tost_text(bad); self.assertIn("slope undefined (no valid bootstrap draw)", text); self.assertIn("equivalent within +-0.02: None", text); self.assertIn("High minus Low -0.0129..-0.0031", text)


class MixedLogit(unittest.TestCase):
    """The section-7 random-intercept logistic model, fitted by maximum likelihood with Gauss-Hermite quadrature over the feature effect (statsmodels' variational and
    MAP GLMM were measured to be intractable beyond ~1,000 features: 14.8 s at 1,000, minutes at 2,000, MAP not converging)."""
    def table(self, beta_s=0.8, beta_t=0.4, sigma=0.8, n=3000, seed=0):
        rng = np.random.default_rng(seed); u = rng.normal(0, sigma, n); s_rank = rng.uniform(0, 1, n); tier = rng.integers(0, 3, n); stratum = rng.integers(0, 5, n)
        t = {"feature": np.arange(n), "S_rank": s_rank, "tier": np.array(["Low", "Mid", "High"])[tier], "stratum": stratum, "evaluable": np.ones(n, bool), "r": {}}
        for b, shift in ((7, 1.0), (6, 0.0)):
            logit = -0.3 + shift + beta_s * s_rank + beta_t * tier + 0.1 * (stratum - 2) + u
            t["r"][b] = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), 0.91, 0.89)         # r just above / just below the 0.9 cut: only the 0.9 cut separates them
        return t

    def test_planted_coefficients_are_recovered_within_three_standard_errors_and_sigma_is_close(self):
        fit = an.mixed_logit(self.table(), [7, 6], gee=False); m = fit["mixed"]; est = dict(zip(m["names"], m["coef"])); se = dict(zip(m["names"], m["se"]))
        truth = {"bits_c": 1.0, "S_rank": 0.8, "tier": 0.4, "S_rank:tier": 0.0, "C(stratum)[T.1]": 0.1, "C(stratum)[T.2]": 0.2, "C(stratum)[T.3]": 0.3, "C(stratum)[T.4]": 0.4}
        for k, v in truth.items():
            self.assertLess(abs(est[k] - v), 3 * se[k], (k, est[k], v, se[k]))
        self.assertGreater(m["sigma"], 0.5); self.assertLess(m["sigma"], 1.1); self.assertTrue(m["converged"])
        self.assertEqual(fit["n_features"], 3000); self.assertEqual(fit["n_obs"], 6000)
        import math
        for zv, pv in zip(m["z"], m["p"]):
            self.assertAlmostEqual(pv, 2 * (1 - 0.5 * (1 + math.erf(abs(zv) / math.sqrt(2)))), places=10)          # two-sided normal p-values

    def test_without_a_feature_effect_it_agrees_with_plain_logistic_regression(self):
        import pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
        t = self.table(sigma=0.0, n=2500, seed=3); fit = an.mixed_logit(t, [7, 6], gee=False)["mixed"]
        df = pd.concat([pd.DataFrame({"y": an.survived(t["r"][b]).astype(int), "bits_c": b - 6.5, "S_rank": t["S_rank"], "tier": [an.tier_score(x) for x in t["tier"]], "stratum": t["stratum"]}) for b in (7, 6)], ignore_index=True)
        ref = smf.logit("y ~ bits_c + C(stratum) + S_rank + tier + S_rank:tier", df).fit(disp=0)
        for name, c in zip(fit["names"], fit["coef"]):
            self.assertAlmostEqual(c, ref.params[name], delta=0.08, msg=name)
        self.assertLess(fit["sigma"], 0.15)

    def test_gee_cross_check_is_returned_with_the_same_coefficient_names(self):
        fit = an.mixed_logit(self.table(n=800), [7, 6]); self.assertEqual(fit["gee"]["names"], fit["mixed"]["names"])
        g = dict(zip(fit["gee"]["names"], fit["gee"]["coef"])); self.assertGreater(g["bits_c"], 0.3)

    def test_non_evaluable_features_are_dropped_and_the_cap_subsamples_features_reproducibly(self):
        t = self.table(n=300); t["evaluable"][:50] = False
        fit = an.mixed_logit(t, [7, 6], max_features=100, seed=1, gee=False); self.assertEqual(fit["n_features"], 100); self.assertEqual(fit["n_obs"], 200)
        self.assertEqual(an.mixed_logit(t, [7, 6], max_features=100, seed=1, gee=False)["mixed"]["coef"], fit["mixed"]["coef"])
        self.assertEqual(an.mixed_logit(t, [7, 6], gee=False)["n_features"], 250)


class Cli(unittest.TestCase):
    def test_command_line_runs_the_analysis_on_a_run_directory_and_writes_the_report(self):
        import subprocess
        out = RunAnalysis().synth(low_penalty=0.05, seed=11); dest = Path(out) / "cli_out"; self.addCleanup(shutil.rmtree, out, ignore_errors=True)
        r = subprocess.run([sys.executable, "-B", str(Path(an.__file__)), "--run", str(out), "--out", str(dest), "--n-perm", "100", "--n-boot", "50"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-800:]); self.assertTrue((dest / "analysis.json").exists() and (dest / "analysis.md").exists())
        import json
        j = json.loads((dest / "analysis.json").read_text()); self.assertEqual((j["n_perm"], j["n_boot"]), (100, 50))               # the command-line counts reach the analysis
        for k in ("T1_int7", "T2_int6", "T3_tier"): self.assertIn(f"{k}: p {j['tests'][k]['p']:.4g}  Holm {j['holm'][k]:.4g}", r.stdout)


class PartialSlope(unittest.TestCase):
    def test_equals_the_ols_coefficient_of_the_full_model_with_script_dummies(self):
        rng = np.random.default_rng(0); tier = rng.integers(0, 3, 27).astype(float); script = rng.integers(0, 3, 27); gap = 0.3 * tier + 0.5 * script + rng.normal(0, 0.1, 27)
        X = np.column_stack([np.ones(27), (script == 1), (script == 2), tier]).astype(float); ref = np.linalg.lstsq(X, gap, rcond=None)[0][3]
        self.assertAlmostEqual(an.partial_slope(gap, tier, script), ref)


class RunAnalysis(unittest.TestCase):
    """End to end on a synthetic run directory with the files phase1_run.save_outputs writes (survival.pt, selectivity.pt, meta.json) and the real roster."""
    ROOT = Path(__file__).resolve().parents[3]

    def synth(self, low_penalty=0.0, selective_penalty=0.0, seed=0, per_lang=15, F=4000, silent=0, penalty_bits=(7, 6, 5, 4), script_penalty=0.0, stratum_effect=0.0, low_skew=False, controls_low_strata=False):
        import json, tempfile, torch
        roster = json.loads((self.ROOT / "reports/g0/phase1_languages.json").read_text()); langs = roster["languages"] if isinstance(roster, dict) else roster
        L = len(langs); rng = np.random.default_rng(seed); n_sel = L * per_lang
        selective = np.zeros(F, bool); selective[:n_sel] = True; controls = np.arange(n_sel, 2 * n_sel)
        peak = np.concatenate([np.repeat(np.arange(L), per_lang), rng.integers(0, L, n_sel), rng.integers(0, L, F - 2 * n_sel)])
        sel_strata = rng.integers(0, 5, n_sel)
        if low_skew:                                                       # selective features of Low-tier languages sit in the high strata (3, 4: lower r with stratum_effect), the others in 0-2
            lowlang = np.repeat([l["tier"].lower() == "low" for l in langs], per_lang); sel_strata = np.where(lowlang, rng.integers(3, 5, n_sel), rng.integers(0, 3, n_sel))
        ctrl_strata = rng.integers(0, 3, n_sel) if controls_low_strata else sel_strata               # a short control pool: no controls in the rarest strata
        strata = np.concatenate([sel_strata, ctrl_strata] + [np.full(F - 2 * n_sel, -1)])
        supported = np.zeros(F, bool); supported[:2 * n_sel] = True; S_rank = np.where(selective, rng.uniform(0.9, 1, F), np.where(supported, rng.uniform(0.1, 0.6, F), 0.0))
        low = np.array([l["tier"].lower() == "low" for l in langs])
        brahmic = np.array([l["script_family"] in ("Indo-Aryan Brahmic", "Dravidian Brahmic", "SE-Asian Brahmic", "Ethiopic") for l in langs])      # non-Latin scripts of the Mid / Low tiers
        corr = {}
        for b, base in ((7, 0.985), (6, 0.95), (5, 0.9), (4, 0.8), (16, 1.0), (12, 1.0), (0, 1.0)):
            r = np.clip(base + rng.normal(0, 0.03, (L, F)) - stratum_effect * (np.maximum(strata, 0)[None, :] - 2) / 2.0, -1, 1) if b in penalty_bits else np.clip(base + rng.normal(0, 0.03, (L, F)), -1, 1)
            at_peak = selective[None, :] & (np.arange(L)[:, None] == peak[None, :])
            hit = at_peak & low[peak][None, :]
            r = np.where(at_peak & brahmic[peak][None, :] & (b in penalty_bits), r - script_penalty, r)
            corr[b] = torch.tensor(np.where(hit, r - low_penalty - selective_penalty, np.where(selective[None, :], r - selective_penalty, r)) if b in penalty_bits else r.clip(max=1.0), dtype=torch.float32)
        out = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, out, ignore_errors=True)         # synthetic runs are several MB each; /tmp is RAM-backed
        fire = np.full((L, F), 0.01); fire[peak[:silent], np.arange(silent)] = 0.0005                      # the first `silent` selective features are inactive at their peak language
        torch.save({"languages": [l["code"] for l in langs], "bits": [7, 6, 5, 4, 16, 12, 0], "n": [1000] * L, "corr": corr, "firing_rate": torch.tensor(fire, dtype=torch.float32), "mean_activation": torch.zeros(L, F),
                    "max_activation": torch.zeros(L, F), "loss_sum": {}, "loss_n": {}, "pooled_corr": {}, "pooled_firing_rate": torch.zeros(F)}, out / "survival.pt")
        rates = torch.zeros(L, F); rates[:, :2 * n_sel] = torch.tensor(np.where(selective[None, :2 * n_sel], 0.002, 0.02), dtype=torch.float32)
        torch.save({"selective": torch.tensor(selective), "supported": torch.tensor(supported), "peak": torch.tensor(peak), "S_rank": torch.tensor(S_rank, dtype=torch.float32),
                    "strata": torch.tensor(strata), "controls": torch.tensor(controls), "rates": rates, "S": torch.ones(F), "null_cutoff": 1.2, "stable_count": torch.full((F,), 100),
                    "languages": [l["code"] for l in langs]}, out / "selectivity.pt")
        (out / "meta.json").write_text(json.dumps({"regime": "mask", "languages": [l["code"] for l in langs]}))
        return out

    def analyse(self, out, **kw):
        return an.run_analysis(out, self.ROOT / "reports/g0/phase1_languages.json", Path(out) / "analysis", n_perm=300, n_boot=100, seed=0, **kw)

    def test_a_planted_tier_effect_gives_a_small_p_and_a_null_does_not(self):
        r = self.analyse(self.synth(low_penalty=0.08)); t3 = r["tests"]["T3_tier"]
        self.assertGreater(t3["slope"], 0); self.assertLess(t3["p"], 0.05); self.assertEqual(t3["n_languages"], 27); self.assertTrue(t3["enough_languages"])
        r0 = self.analyse(self.synth(low_penalty=0.0, seed=1)); self.assertGreater(r0["tests"]["T3_tier"]["p"], 0.05)

    def test_a_planted_selectivity_effect_is_detected_at_the_two_confirmatory_widths(self):
        r = self.analyse(self.synth(selective_penalty=0.08, seed=2))
        for k in ("T1_int7", "T2_int6"):
            self.assertLess(r["tests"][k]["delta"], 0); self.assertLess(r["tests"][k]["p"], 0.01); self.assertIn("ci", r["tests"][k])
        self.assertEqual(sorted(r["holm"]), ["T1_int7", "T2_int6", "T3_tier"]); self.assertTrue(all(0 <= v <= 1 for v in r["holm"].values()))

    def test_p2_is_demoted_when_too_few_languages_have_a_populated_selective_set(self):
        r = self.analyse(self.synth(seed=4), min_features=16)                       # 15 selective features per language: no language qualifies
        t3 = r["tests"]["T3_tier"]; self.assertEqual(t3["n_languages"], 0); self.assertFalse(t3["enough_languages"]); self.assertEqual(t3["p"], 1.0)
        out = self.synth(seed=4); r = self.analyse(out, min_languages=28); self.assertFalse(r["tests"]["T3_tier"]["enough_languages"])
        self.assertIn("P2 DEMOTED", (Path(out) / "analysis" / "analysis.md").read_text())
        self.assertTrue(self.analyse(self.synth(seed=4), min_languages=27)["tests"]["T3_tier"]["enough_languages"])          # exactly 27 populated languages: the bound is inclusive
        self.assertTrue(self.analyse(self.synth(seed=4))["tests"]["T3_tier"]["enough_languages"])

    def test_a_script_effect_that_correlates_with_tier_is_not_mistaken_for_a_tier_effect(self):
        out = self.synth(script_penalty=0.12, seed=8); r = self.analyse(out); self.assertGreater(r["tests"]["T3_tier"]["p"], 0.05)         # with the script-family covariate
        self.assertGreater(r["tests"]["T3_tier"]["high_minus_low"], 0.03)                                                                  # while the raw High-Low difference is large

    def test_t1_is_the_int7_contrast_and_t2_the_int6_contrast(self):
        r = self.analyse(self.synth(selective_penalty=0.08, seed=9, penalty_bits=(6,)))
        self.assertGreater(r["tests"]["T1_int7"]["p"], 0.05); self.assertLess(abs(r["tests"]["T1_int7"]["delta"]), 0.05)
        self.assertLess(r["tests"]["T2_int6"]["p"], 0.01); self.assertLess(r["tests"]["T2_int6"]["delta"], -0.1)

    def test_a_rarity_effect_that_correlates_with_tier_is_not_mistaken_for_a_tier_effect(self):
        out = self.synth(stratum_effect=0.12, low_skew=True, seed=12); r = self.analyse(out)
        self.assertGreater(r["tests"]["T3_tier"]["p"], 0.05)                                           # r depends on the firing-rate stratum only; Low languages' features are simply rarer
        self.assertLess(abs(r["tests"]["T1_int7"]["delta"]), 0.1)                                      # and the stratified contrast sees no selectivity effect either
        self.assertGreater(r["tests"]["T3_tier"]["survival_rate_gap_sensitivity"]["p"], 0.05)          # the survival-rate version of the tier test is adjusted too

    def test_a_short_control_pool_does_not_turn_rarity_into_a_selectivity_effect(self):
        r = self.analyse(self.synth(stratum_effect=0.12, controls_low_strata=True, seed=13)); t1 = r["tests"]["T1_int7"]
        self.assertGreater(t1["n_selective_dropped"], 0); self.assertLess(abs(t1["delta"]), 0.05); self.assertTrue(t1["ci"][0] <= 0.0 <= t1["ci"][1])      # strata without controls are dropped and counted; no false effect
        self.assertGreater(t1["survival_control"] - t1["survival_selective"], 0.1)                     # while the raw rates differ a lot (selective features sit in the strata where survival is lower)

    def test_the_tier_test_uses_both_confirmatory_widths(self):
        for pb in ((6,), (7,)):
            r = self.analyse(self.synth(low_penalty=0.2, seed=7, penalty_bits=pb)); self.assertGreater(r["tests"]["T3_tier"]["slope"], 0.02); self.assertLess(r["tests"]["T3_tier"]["p"], 0.05)

    def test_features_inactive_at_their_peak_language_are_reported_and_left_out(self):
        r = self.analyse(self.synth(seed=5, silent=30)); self.assertEqual(r["tests"]["T1_int7"]["n_dropped_not_evaluable"], 30); self.assertEqual(r["tests"]["T1_int7"]["n_selective"], 405 - 30)

    def test_holm_values_are_the_holm_adjustment_of_the_three_reported_p_values(self):
        r = self.analyse(self.synth(selective_penalty=0.05, low_penalty=0.0, seed=6)); ps = [r["tests"][k]["p"] for k in ("T1_int7", "T2_int6", "T3_tier")]
        for k, v in zip(("T1_int7", "T2_int6", "T3_tier"), an.holm(ps)): self.assertAlmostEqual(r["holm"][k], v)

    def test_files_and_exploratory_sections_are_written_and_labelled(self):
        import json
        out = self.synth(seed=3); r = self.analyse(out); d = Path(out) / "analysis"
        self.assertTrue((d / "analysis.json").exists() and (d / "analysis.md").exists()); md = (d / "analysis.md").read_text()
        self.assertIn("exploratory", md.lower()); self.assertIn("Holm", md); self.assertIn("INT5", md); self.assertEqual(json.loads((d / "analysis.json").read_text())["regime"], "mask")
        self.assertEqual(sorted(r["exploratory"]["contrasts"]), ["int4", "int5"]); self.assertIn("mixed_logit", r["exploratory"]); self.assertIn("controls", r)
        self.assertEqual(sorted(r["controls"]["survival_at"]), ["int0", "int12", "int16"])                  # noise-floor widths: survival of the same features


if __name__ == "__main__":
    unittest.main()
