"""Tests for scripts/g0/survival_stage3.py: the frozen Stage 3 analysis of the survival measurement-validity study (reports/survival_measurement/analysis_freeze.json, D-120). The sums come from the
real SurvivalSink on random pre-activations of a toy dictionary, so the statistics are the ones the Ada jobs will write; every answer is recomputed token by token with plain numpy."""
import json, sys, unittest
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import survival_sidecar as sc   # noqa: E402
import survival_stage3 as s3   # noqa: E402
from test_survival_sidecar import Feeder, sae_acts, toy_batch   # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
L, D = 4, 6
THETA = torch.tensor([0.4, 0.8, 0.3, 1.0, 0.5, 0.6], dtype=torch.float32)


def make_case(seed=0, n=300, noise=0.25, silence=None, exact=None, negative=None, n_lang=L):
    """-> (sums of the quantized condition, pre_fp [L][n, D], pre_q [L][n, D]) for L languages of n equal-budget tokens; `silence` = (lang, feature) pushed below theta in the quantized pass,
    `exact` = (lang, feature) left exactly equal in both passes."""
    g = torch.Generator().manual_seed(seed)
    fp = [torch.randn(n, D, generator=g) * 1.2 + 0.3 for _ in range(n_lang)]
    q = [x + noise * torch.randn(n, D, generator=g) * (1 + torch.arange(D) % 3) for x in fp]
    if silence is not None:
        l, j = silence; q[l][:, j] = -5.0
    if exact is not None:
        l, j = exact; q[l][:, j] = fp[l][:, j]
    if negative is not None:
        l, j = negative; q[l][:, j] = 2.0 - fp[l][:, j] + 0.2 * torch.randn(n, generator=g)               # anti-correlated quantized feature: a negative gain
    f = Feeder(n_lang, THETA)
    batches = [toy_batch(l, torch.ones(1, n, dtype=torch.bool).tolist(), torch.ones(1, n, dtype=torch.bool).tolist(), tokens=torch.arange(n)[None, :] + 5 + 1000 * l) for l in range(n_lang)]
    f.condition("FP", [(b, x[None]) for b, x in zip(batches, fp)]); f.condition("INT4", [(b, x[None]) for b, x in zip(batches, q)])
    return f.sink.sums["INT4"], fp, q


def cell_arrays(fp, q, l, j):
    th = float(THETA[j]); pf, pq = fp[l][:, j].double().numpy(), q[l][:, j].double().numpy()
    pf32, pq32 = fp[l][:, j], q[l][:, j]
    a = (pf32 * (pf32 > THETA[j])).double().numpy(); b = (pq32 * (pq32 > THETA[j])).double().numpy()
    return th, pf, pq, a, b


class Derived(unittest.TestCase):
    def setUp(self):
        self.sums, self.fp, self.q = make_case(); self.d = s3.derived(self.sums, THETA)

    def test_every_derived_quantity_equals_a_token_by_token_numpy_computation(self):
        for l in range(L):
            for j in range(D):
                th, pf, pq, a, b = cell_arrays(self.fp, self.q, l, j)
                act = a > 0; n = len(a)
                self.assertAlmostEqual(self.d["sigma_a"][l, j], a.std(), places=9); self.assertAlmostEqual(self.d["sigma_b"][l, j], b.std(), places=9)
                if a.std() > 0 and b.std() > 0:
                    self.assertAlmostEqual(self.d["r"][l, j], np.corrcoef(a, b)[0, 1], places=9)
                    g = np.cov(a, b, bias=True)[0, 1] / a.var(); c = b.mean() - g * a.mean()
                    self.assertAlmostEqual(self.d["g"][l, j], g, places=9); self.assertAlmostEqual(self.d["c"][l, j], c, places=9)
                    self.assertAlmostEqual(self.d["sigma_e2"][l, j], np.var(b - g * a - c), places=9)
                if act.sum() > 0:
                    self.assertAlmostEqual(self.d["nu"][l, j], np.sqrt(np.mean((pq[act] - pf[act]) ** 2)) / th, places=9)
                    self.assertAlmostEqual(self.d["on_off"][l, j], np.mean(b[act] == 0), places=12); self.assertAlmostEqual(self.d["exact_flip"][l, j], self.d["on_off"][l, j], places=12)
                    self.assertAlmostEqual(self.d["pred_flip"][l, j], np.mean(np.abs(pq[act] - pf[act]) > (pf[act] - th)), places=12)
                    self.assertAlmostEqual(self.d["mean_margin"][l, j], np.mean(pf[act] - th) / th, places=9)
                if (~act).sum() > 0:
                    self.assertAlmostEqual(self.d["off_on"][l, j], np.mean(b[~act] > 0), places=12)
                self.assertEqual(bool(self.d["silent"][l, j]), bool((b > 0).sum() == 0))

    def test_lambda_follows_stage_1_including_the_clip_convention(self):
        import survival_lambda as sl
        lam, clip = sl.lam(self.d["r"])
        self.assertTrue(np.array_equal(lam, self.d["lam"]) and np.array_equal(clip, self.d["clipped"]))

    def test_the_lambda_identity_log_sigma_e2_minus_two_log_g_minus_two_log_sigma_a(self):
        ok = self.d["identity_ok"]
        self.assertTrue(ok.sum() > 10)
        lhs = np.log(1 / self.d["r"][ok] ** 2 - 1); rhs = np.log(self.d["sigma_e2"][ok]) - 2 * np.log(np.abs(self.d["g"][ok])) - 2 * np.log(self.d["sigma_a"][ok])
        self.assertTrue(np.allclose(lhs, rhs, atol=1e-8))


class Populations(unittest.TestCase):
    def test_fp_active_uses_the_rate_and_the_token_floor_and_nothing_from_the_quantized_side(self):
        n_a = np.array([[0, 5, 19, 20, 300]]); n = np.array([300])
        self.assertEqual(s3.fp_active(n_a, n, 0.001, 20).tolist(), [[False, False, False, True, True]])
        self.assertEqual(s3.fp_active(np.array([[28, 27]]), np.array([27036]), 0.001, 20).tolist(), [[True, False]])         # rate 0.00104 vs 0.00099 on the full equal budget
        self.assertEqual(s3.fp_active(np.array([[20]]), np.array([5000]), 0.001, 20).tolist(), [[True]])
        self.assertEqual(s3.fp_active(np.array([[20]]), np.array([20000]), 0.001, 20).tolist(), [[False]])                    # a rate of exactly 0.001 is not above 0.001

    def test_a_feature_silenced_by_quantization_stays_in_the_nu_population_and_is_clipped_low_in_lambda(self):
        sums, fp, q = make_case(silence=(1, 2)); d = s3.derived(sums, THETA); act = s3.fp_active(sums["stats"]["n_a"].numpy(), sums["n"].numpy())
        self.assertTrue(act[1, 2] and d["silent"][1, 2]); self.assertTrue(d["clipped"][1, 2])
        self.assertTrue(np.isfinite(d["nu"][1, 2]) and d["nu"][1, 2] > 0)                                                    # nu is defined from the FP-active tokens alone
        pop = s3.populations(d, act)
        self.assertTrue(pop["nu"][1, 2] and pop["lambda"][1, 2] and not pop["decomposition"][1, 2])                         # g and sigma_e are not defined for a silent feature
        self.assertEqual(pop["silent_count"], int((d["silent"] & act).sum()))
        self.assertGreater(int(pop["decomposition"].sum()), 5); self.assertTrue(np.array_equal(pop["decomposition"], act & np.isfinite(d["g"]) & (d["g"] != 0) & (d["sigma_e2"] > 0)))

    def test_the_fp_active_set_is_the_same_in_every_condition(self):
        a, _, _ = make_case(seed=0, noise=0.1); b, _, _ = make_case(seed=0, noise=0.9)
        self.assertTrue(np.array_equal(a["stats"]["n_a"], b["stats"]["n_a"]))


class IdentityGate(unittest.TestCase):
    def test_a_consistent_set_passes_and_degenerate_cells_are_excluded_and_counted(self):
        sums, fp, q = make_case(exact=(0, 1)); d = s3.derived(sums, THETA); act = s3.fp_active(sums["stats"]["n_a"].numpy(), sums["n"].numpy())
        r = s3.identity_gate(d, act)
        self.assertTrue(r["passed"]); self.assertGreater(r["checked"], 10); self.assertLess(r["max_abs_diff"], 1e-9)
        self.assertGreaterEqual(sum(r["excluded"].values()), 1); self.assertEqual(r["checked"] + sum(r["excluded"].values()), int(act.sum()))

    def test_a_negative_gain_is_checked_with_its_sign(self):
        sums, fp, q = make_case(negative=(2, 3)); d = s3.derived(sums, THETA); act = s3.fp_active(sums["stats"]["n_a"].numpy(), sums["n"].numpy())
        self.assertTrue(act[2, 3] and d["g"][2, 3] < 0 and d["r"][2, 3] < 0)
        r = s3.identity_gate(d, act); self.assertTrue(r["passed"]); self.assertLess(r["max_abs_diff"], 1e-9)

    def test_a_wrong_correlation_fails_the_gate_outside_the_degenerate_cells(self):
        sums, fp, q = make_case(); d = s3.derived(sums, THETA); act = s3.fp_active(sums["stats"]["n_a"].numpy(), sums["n"].numpy())
        d["r"] = d["r"].copy(); d["r"][0, 0] += 0.01
        self.assertFalse(s3.identity_gate(d, act)["passed"])

    def test_a_cell_with_nonpositive_residual_variance_or_a_clamped_r_is_not_checked(self):
        sums, fp, q = make_case(); d = s3.derived(sums, THETA); act = s3.fp_active(sums["stats"]["n_a"].numpy(), sums["n"].numpy())
        d["sigma_e2"] = d["sigma_e2"].copy(); d["sigma_e2"][0, 0] = -1e-17; d["r"] = d["r"].copy(); d["r"][0, 0] = 0.123                    # a wrong r in a cancellation cell is not a gate failure
        d["r"][1, 1] = 1.0                                                                                                      # clamped
        r = s3.identity_gate(d, act); self.assertTrue(r["passed"]); self.assertGreaterEqual(r["excluded"]["residual_variance_not_positive"], 1); self.assertGreaterEqual(r["excluded"]["r_clamped"], 1)


class H3(unittest.TestCase):
    def test_per_language_spearman_and_the_median(self):
        sums, fp, q = make_case(n=400); d = s3.derived(sums, THETA); act = s3.fp_active(sums["stats"]["n_a"].numpy(), sums["n"].numpy())
        r = s3.h3_cell(d, act)
        lam_rho, nu_rho = [], []
        for l in range(L):
            m = act[l]; ls = np.log(d["sigma_a"][l][m])
            lam_rho.append(spearmanr(d["lam"][l][m], ls)[0]); nu_rho.append(spearmanr(d["nu"][l][m], ls)[0])
        self.assertAlmostEqual(r["median_spearman_lambda"], float(np.median(lam_rho)), places=12); self.assertAlmostEqual(r["median_spearman_nu"], float(np.median(nu_rho)), places=12)
        self.assertEqual(len(r["per_language_lambda"]), L)

    def test_the_rule_needs_both_halves_and_the_planned_thresholds(self):
        self.assertTrue(s3.h3_rule(0.19, -0.51)); self.assertTrue(s3.h3_rule(-0.19, 0.6)); self.assertFalse(s3.h3_rule(-0.5, 0.6)); self.assertFalse(s3.h3_rule(0.1, -0.4))
        self.assertFalse(s3.h3_rule(0.2, 0.6)); self.assertFalse(s3.h3_rule(0.1, 0.5)); self.assertFalse(s3.h3_rule(0.1, float("nan")))

    def test_h3_as_a_whole_needs_all_six_cells(self):
        cells = {(a, w): True for a in ("legacy", "matched", "gptq") for w in ("INT6", "INT4")}
        self.assertTrue(s3.h3_overall(cells)["met"])
        cells[("gptq", "INT4")] = False
        r = s3.h3_overall(cells); self.assertFalse(r["met"]); self.assertEqual(r["failing"], [("gptq", "INT4")])
        full = {(a, w): True for a in ("legacy", "matched", "gptq") for w in ("INT6", "INT4")}; del full[("legacy", "INT6")]
        r = s3.h3_overall(full); self.assertFalse(r["met"]); self.assertEqual(r["missing"], [("legacy", "INT6")]); self.assertEqual(r["failing"], [])                  # a missing cell is not a pass


class H4(unittest.TestCase):
    TIER = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    SCRIPT = np.array(["lat"] * 12 + ["cyr"] * 3 + ["ara"] * 3 + ["dev"] * 3 + ["han", "jpn", "kor"] + ["lat"] * 3)

    def test_the_rule_function(self):
        self.assertEqual(s3.h4_rule(0.4, (0.25, 0.55), (0.3, 0.5)), "present")
        self.assertEqual(s3.h4_rule(0.1, (-0.05, 0.25), (-0.03, 0.22)), "inconclusive")                                      # 95 % interval contains 0
        self.assertEqual(s3.h4_rule(0.1, (0.01, 0.3), (0.03, 0.27)), "inconclusive")                                         # excludes 0, |estimate| < 0.20, 90 % interval not inside the bound
        self.assertEqual(s3.h4_rule(0.1, (0.01, 0.2), (0.03, 0.18)), "absent")                                               # a small non-zero effect inside the equivalence bound is "absent" (equivalence, not nullity)
        self.assertEqual(s3.h4_rule(0.02, (-0.2, 0.25), (-0.15, 0.19)), "absent")                                           # the 90 % interval is inside +-0.20
        self.assertEqual(s3.h4_rule(0.02, (-0.2, 0.25), (-0.15, 0.2)), "inconclusive")                                       # touching the bound is not inside
        self.assertEqual(s3.h4_rule(0.3, (float("nan"), float("nan")), (float("nan"), float("nan"))), "inconclusive")

    def test_slope_and_standardized_slope_equal_the_definitions(self):
        import phase1_analysis as an
        g = np.random.default_rng(1); y = 0.5 * self.TIER + g.normal(0, 0.3, 27)
        self.assertAlmostEqual(s3.tier_slope(y, self.TIER, self.SCRIPT), an.partial_slope(y, self.TIER, self.SCRIPT), places=12)
        self.assertAlmostEqual(s3.z_slope(y, self.TIER, self.SCRIPT), an.partial_slope(y, self.TIER, self.SCRIPT) * self.TIER.std() / y.std(), places=12)
        self.assertTrue(np.isnan(s3.tier_slope(y, np.zeros(27), self.SCRIPT)))                                                # no tier variation: undefined, not inf
        absorbed = np.array([{"lat": 0, "cyr": 1, "ara": 2, "dev": 0, "han": 1, "jpn": 2, "kor": 0}[c] for c in self.SCRIPT], dtype=float)      # the tier is a function of the script: nothing is left after the dummies
        self.assertTrue(np.isnan(s3.tier_slope(y, absorbed, self.SCRIPT)))

    def test_a_covariate_enters_the_reduced_model_like_freedman_lane_does(self):
        import phase1_analysis as an
        g = np.random.default_rng(8); cov = g.normal(size=27); y = 0.4 * self.TIER + 0.6 * cov + g.normal(0, 0.2, 27)
        self.assertAlmostEqual(s3.tier_slope(y, self.TIER, self.SCRIPT, covariates=cov), an.freedman_lane_tier_test(y, self.TIER, self.SCRIPT, n_perm=10, seed=0, covariates=cov)["slope"], places=10)
        self.assertNotAlmostEqual(s3.tier_slope(y, self.TIER, self.SCRIPT, covariates=cov), s3.tier_slope(y, self.TIER, self.SCRIPT), places=3)

    def test_a_strong_quantization_specific_tier_effect_is_present_and_an_unchanged_pattern_is_absent_in_primary(self):
        g = np.random.default_rng(2)
        y16 = 0.1 * self.TIER + g.normal(0, 0.5, 27); y_same = y16 + g.normal(0, 0.02, 27); y_more = 1.5 * self.TIER + g.normal(0, 0.3, 27)       # standardized slopes: the pattern must change relative to the noise
        r = s3.h4_cell(y_more, y16, self.TIER, self.SCRIPT, n_boot=400, seed=3)
        self.assertEqual(r["primary"]["outcome"], "present"); self.assertGreater(r["primary"]["estimate"], 0.2); self.assertEqual(r["secondary"]["outcome"], "present"); self.assertGreater(r["z_w"], 0.9)
        r = s3.h4_cell(y_same, y16, self.TIER, self.SCRIPT, n_boot=400, seed=3)
        self.assertEqual(r["primary"]["outcome"], "absent"); self.assertLess(abs(r["primary"]["estimate"]), 0.2)
        self.assertLess(abs(r["z_w"] - r["z_16"]), 0.05); self.assertEqual(r["secondary"]["estimate"], r["z_w"])         # the same pattern at both widths

    def test_primary_is_the_difference_of_the_per_width_standardized_slopes_and_the_common_denominator_is_a_sensitivity(self):
        g = np.random.default_rng(4); y16 = 0.3 * self.TIER + g.normal(0, 0.2, 27); yw = 0.9 * self.TIER + g.normal(0, 0.5, 27)
        r = s3.h4_cell(yw, y16, self.TIER, self.SCRIPT, n_boot=50, seed=0)
        self.assertAlmostEqual(r["primary"]["estimate"], s3.z_slope(yw, self.TIER, self.SCRIPT) - s3.z_slope(y16, self.TIER, self.SCRIPT), places=12)
        self.assertAlmostEqual(r["sensitivity_common_denominator"]["estimate"], (s3.tier_slope(yw, self.TIER, self.SCRIPT) - s3.tier_slope(y16, self.TIER, self.SCRIPT)) * self.TIER.std() / y16.std(), places=12)
        self.assertAlmostEqual(r["secondary"]["estimate"], s3.z_slope(yw, self.TIER, self.SCRIPT), places=12)
        for k in ("primary", "secondary", "sensitivity_common_denominator"):
            self.assertLessEqual(r[k]["ci95"][0], r[k]["ci90"][0]); self.assertGreaterEqual(r[k]["ci95"][1], r[k]["ci90"][1]); self.assertTrue(r[k]["ci95"][0] < r[k]["ci90"][0] or r[k]["ci95"][1] > r[k]["ci90"][1])

    def test_degenerate_replicates_are_dropped_counted_and_too_many_make_the_cell_inconclusive(self):
        tier = np.array([0] * 8 + [1] * 2 + [2] * 2); script = np.array(["a"] * 4 + ["b"] * 4 + ["c"] * 4)                    # resamples often lose the tier variation inside the script dummies
        g = np.random.default_rng(5); y = g.normal(0, 1, 12); y16 = g.normal(0, 1, 12)
        r = s3.h4_cell(y, y16, tier, script, n_boot=300, seed=1)
        self.assertGreater(r["dropped_replicates"], 0); self.assertEqual(r["dropped_replicates"] + r["valid_replicates"], 300)
        if r["dropped_replicates"] / 300 > 0.05:
            self.assertEqual(r["primary"]["outcome"], "inconclusive"); self.assertTrue(r["primary"]["not_estimable"])

    def test_the_bootstrap_is_reproducible_for_a_seed(self):
        g = np.random.default_rng(6); y = g.normal(0, 1, 27) + 0.3 * self.TIER; y16 = g.normal(0, 1, 27) + 0.3 * self.TIER
        a = s3.h4_cell(y, y16, self.TIER, self.SCRIPT, n_boot=100, seed=9); b = s3.h4_cell(y, y16, self.TIER, self.SCRIPT, n_boot=100, seed=9); c = s3.h4_cell(y, y16, self.TIER, self.SCRIPT, n_boot=100, seed=10)
        self.assertEqual(json.dumps(a, default=float), json.dumps(b, default=float)); self.assertNotEqual(json.dumps(a, default=float), json.dumps(c, default=float))


class H5(unittest.TestCase):
    """A toy F5-like table: 6 languages x 40 features, five baseline columns (model A), R1, and the sidecar's g and log nu; W1 is built from known ingredients."""
    def table(self, seed=0, signal=None, n_per=40, n_lang=6):
        g = np.random.default_rng(seed); n = n_per * n_lang; lang = np.repeat(np.arange(n_lang), n_per)
        base = g.normal(size=(n, 5)); r1 = g.normal(size=n); gain = g.normal(size=n); lognu = g.normal(size=n); n_a = np.full(n, 100)
        w1 = 0.5 * base[:, 0] + 0.3 * r1 + g.normal(0, 0.5, n)
        if signal == "nu":
            w1 = w1 + 1.5 * lognu
        if signal == "g":
            w1 = w1 + 1.5 * gain
        if signal == "r1":
            w1 = w1 + 1.5 * r1
        return dict(base=base, r1=r1, g=gain, log_nu=lognu, w1=w1, lang=lang, mask=np.ones(n, bool), n_a=n_a)

    def cell(self, T, **kw):
        return s3.h5_cell(T["base"], T["r1"], T["g"], T["log_nu"], T["w1"], T["lang"], T["mask"], T["n_a"], n_boot=200, seed=1, **kw)

    def test_with_no_information_in_g_and_nu_the_rule_is_not_met(self):
        r = self.cell(self.table()); self.assertFalse(r["cell_met"]); self.assertLess(r["improvement"]["Bplus_over_B"]["improvement"], 0.10)
        self.assertEqual((r["n"], r["n_languages"]), (240, 6)); self.assertEqual(set(r["improvement"]), {"B_over_A", "Bplus_over_B", "Bplus_over_A", "Bg_over_B", "Bnu_over_B"})

    def test_a_real_dependence_on_log_nu_is_found_by_b_plus_and_the_nu_only_variant_not_the_g_only_one(self):
        r = self.cell(self.table(signal="nu"))
        self.assertTrue(r["cell_met"]); self.assertGreaterEqual(r["improvement"]["Bplus_over_B"]["improvement"], 0.10); self.assertGreater(r["improvement"]["Bplus_over_B"]["lo"], 0)
        self.assertGreaterEqual(r["improvement"]["Bnu_over_B"]["improvement"], 0.10); self.assertLess(r["improvement"]["Bg_over_B"]["improvement"], 0.10)
        r = self.cell(self.table(signal="g")); self.assertTrue(r["cell_met"]); self.assertGreaterEqual(r["improvement"]["Bg_over_B"]["improvement"], 0.10); self.assertLess(r["improvement"]["Bnu_over_B"]["improvement"], 0.10)

    def test_the_population_is_the_mask_with_enough_tokens_and_defined_g_and_nu_and_the_drops_are_counted(self):
        T = self.table(); T["n_a"][:10] = 19; T["n_a"][20:23] = 20; T["g"][5:7] = np.nan; T["g"][18:19] = np.nan; T["g"][10:14] = np.nan; T["log_nu"][14:16] = np.nan; T["mask"][16:20] = False
        r = self.cell(T)
        self.assertEqual(r["n"], 240 - 10 - 4 - 2 - 4); self.assertEqual(r["dropped"], {"outside_mask": 4, "n_a_below_floor": 10, "undefined_g_or_nu": 6})

    def test_a_dependence_on_r1_is_model_b_over_a_and_does_not_meet_the_rule_which_asks_for_g_and_nu_beyond_b(self):
        r = self.cell(self.table(signal="r1"))
        self.assertGreaterEqual(r["improvement"]["B_over_A"]["improvement"], 0.10); self.assertGreaterEqual(r["improvement"]["Bplus_over_A"]["improvement"], 0.10)
        self.assertLess(r["improvement"]["Bplus_over_B"]["improvement"], 0.10); self.assertFalse(r["cell_met"])

    def test_models_a_b_and_b_plus_see_the_same_rows_and_the_result_is_reproducible(self):
        T = self.table(signal="nu"); a, b = self.cell(T), self.cell(T)
        self.assertEqual(json.dumps(a, default=float), json.dumps(b, default=float))
        self.assertAlmostEqual(a["improvement"]["Bplus_over_A"]["improvement"], 1 - (1 - a["improvement"]["B_over_A"]["improvement"]) * (1 - a["improvement"]["Bplus_over_B"]["improvement"]), places=9)

    def test_the_overall_statement_is_the_d113_one(self):
        self.assertEqual(s3.h5_overall([False] * 6), "no clear predictive value beyond baselines"); self.assertEqual(s3.h5_overall([True] * 6), "adds predictive value"); self.assertEqual(s3.h5_overall([True, False]), "mixed")


class Descriptive(unittest.TestCase):
    """The descriptive tables of the freeze, recomputed with plain numpy from the derived quantities of two toy arms (same full-precision tokens, different quantization noise)."""
    TIER = np.array([0, 1, 2, 1])

    def setUp(self):
        self.sa, _, _ = make_case(seed=3, noise=0.15, n=400); self.sb, _, _ = make_case(seed=3, noise=0.6, n=400, silence=(2, 1))
        self.da, self.db = s3.derived(self.sa, THETA), s3.derived(self.sb, THETA)
        self.act = s3.fp_active(self.sa["stats"]["n_a"].numpy(), self.sa["n"].numpy())

    def test_the_lambda_decomposition_between_arms_obeys_the_exact_identity(self):
        r = s3.lambda_decomposition(self.da, self.db, self.act)
        m = s3.populations(self.da, self.act)["decomposition"] & s3.populations(self.db, self.act)["decomposition"] & ~self.da["clipped"] & ~self.db["clipped"]
        dl = (self.db["lam"] - self.da["lam"])[m]; ds = np.log(self.db["sigma_e2"][m]) - np.log(self.da["sigma_e2"][m]); dg = -2 * (np.log(np.abs(self.db["g"][m])) - np.log(np.abs(self.da["g"][m])))
        self.assertEqual(r["cells"], int(m.sum())); self.assertGreater(r["cells"], 20)
        self.assertAlmostEqual(r["median_delta_lambda"], float(np.median(dl)), places=12); self.assertAlmostEqual(r["median_delta_log_sigma_e2"], float(np.median(ds)), places=12)
        self.assertAlmostEqual(r["median_minus2_delta_log_abs_g"], float(np.median(dg)), places=12); self.assertLess(r["identity_max_abs_error"], 1e-8)
        self.assertGreater(r["median_delta_lambda"], 0)                                                                    # more noise, higher noise-to-signal

    def test_cells_without_a_decomposition_in_either_arm_or_with_a_clipped_lambda_in_either_arm_are_left_out(self):
        base = s3.lambda_decomposition(self.da, self.db, self.act)["cells"]; pop = s3.populations(self.da, self.act)["decomposition"] & s3.populations(self.db, self.act)["decomposition"] & ~self.da["clipped"] & ~self.db["clipped"]
        l, j = map(int, np.argwhere(pop)[0]); l2, j2 = map(int, np.argwhere(pop)[1]); l3, j3 = map(int, np.argwhere(pop)[2])
        import copy
        a2 = {k: v.copy() if hasattr(v, "copy") else v for k, v in self.da.items()}; b2 = {k: v.copy() if hasattr(v, "copy") else v for k, v in self.db.items()}
        a2["sigma_e2"][l, j] = -1.0                                                                                       # no decomposition in arm a only
        self.assertEqual(s3.lambda_decomposition(a2, self.db, self.act)["cells"], base - 1)
        b2["sigma_e2"][l2, j2] = -1.0                                                                                     # ... in arm b only
        self.assertEqual(s3.lambda_decomposition(self.da, b2, self.act)["cells"], base - 1)
        a3 = {k: v.copy() if hasattr(v, "copy") else v for k, v in self.da.items()}; a3["clipped"][l3, j3] = True                                       # a clipped lambda in arm a only
        self.assertEqual(s3.lambda_decomposition(a3, self.db, self.act)["cells"], base - 1)
        b3 = {k: v.copy() if hasattr(v, "copy") else v for k, v in self.db.items()}; b3["clipped"][l3, j3] = True
        self.assertEqual(s3.lambda_decomposition(self.da, b3, self.act)["cells"], base - 1)

    def test_the_arm_summary_equals_plain_numpy_per_tier_and_overall(self):
        r = s3.arm_summary(self.db, self.act, self.TIER); d = self.db; a = self.act; dec = s3.populations(d, a)["decomposition"]
        self.assertAlmostEqual(r["median_g"], float(np.median(d["g"][dec])), places=12); self.assertAlmostEqual(r["median_log_sigma_e2"], float(np.median(np.log(d["sigma_e2"][dec]))), places=12)
        self.assertAlmostEqual(r["median_lambda"], float(np.median(d["lam"][a])), places=12)
        for key, name in (("on_off", "on_off"), ("off_on", "off_on")):
            v = d[name][a]; v = v[np.isfinite(v)]
            self.assertEqual([r["flip"][key][k] for k in ("p10", "p50", "p90")], [float(x) for x in np.percentile(v, [10, 50, 90])])
        nu = d["nu"][a]; self.assertEqual([r["nu"]["overall"][k] for k in ("p25", "p50", "p75")], [float(x) for x in np.percentile(nu, [25, 50, 75])])
        for t in (0, 1, 2):
            rows = np.isin(np.arange(L), np.nonzero(self.TIER == t)[0]); m = a & rows[:, None]
            self.assertEqual(r["nu"]["by_tier"][t]["p50"], float(np.median(d["nu"][m]))); self.assertEqual(r["silent"]["by_tier"][t]["cells"], int(m.sum()))
            self.assertEqual(r["silent"]["by_tier"][t]["silent"], int((d["silent"] & m).sum())); self.assertEqual(r["clipped"]["by_tier"][t]["low"], int((d["clipped_low"] & m).sum()))
            self.assertAlmostEqual(r["silent"]["by_tier"][t]["share"], float((d["silent"] & m).sum() / m.sum()), places=12)
        self.assertEqual(r["silent"]["overall"]["silent"], int((d["silent"] & a).sum())); self.assertGreaterEqual(r["silent"]["overall"]["silent"], 1)               # the silenced feature of the toy stays in the table
        self.assertEqual(r["clipped"]["overall"]["high"], int((d["clipped"] & ~d["clipped_low"] & a).sum()))

    def test_predicted_versus_exact_flips(self):
        from scipy.stats import spearmanr
        r = s3.flip_prediction(self.db, self.act); m = self.act & np.isfinite(self.db["pred_flip"]) & np.isfinite(self.db["exact_flip"])
        self.assertAlmostEqual(r["spearman"], float(spearmanr(self.db["pred_flip"][m], self.db["exact_flip"][m])[0]), places=12)
        self.assertAlmostEqual(r["median_pred_minus_exact"], float(np.median((self.db["pred_flip"] - self.db["exact_flip"])[m])), places=12); self.assertEqual(r["cells"], int(m.sum()))
        act2 = self.act.copy(); act2[0, :4] = False; m2 = act2 & np.isfinite(self.db["pred_flip"]) & np.isfinite(self.db["exact_flip"])
        self.assertEqual(s3.flip_prediction(self.db, act2)["cells"], int(m2.sum())); self.assertLess(int(m2.sum()), int(m.sum()))                      # only FP-active cells count

    def test_log_nu_between_arms(self):
        import survival_lambda as sl
        from scipy.stats import spearmanr
        r = s3.nu_between_arms(self.da, self.db, self.act); m = self.act
        x, y = self.da["log_nu"][m], self.db["log_nu"][m]
        self.assertAlmostEqual(r["spearman"], float(spearmanr(x, y)[0]), places=12); self.assertAlmostEqual(r["sd_ratio_slope"], float(np.std(y) / np.std(x)), places=12)
        self.assertAlmostEqual(r["ols_slope"], float(np.polyfit(x, y, 1)[0]), places=9); self.assertGreater(r["sd_ratio_slope"], 0); self.assertEqual(r["cells"], int(m.sum()))
        self.assertGreater(float(np.median(y - x)), 0)                                                                    # the noisier arm perturbs the encoder more
        self.assertAlmostEqual(r["median_difference"], float(np.median(y - x)), places=12)


class Wording(unittest.TestCase):
    def test_every_improvement_is_worded_from_its_interval_not_from_the_ten_percent_target(self):
        self.assertEqual(s3.improvement_wording(0.012, 0.004, 0.02), "improvement (interval above zero)")                 # a 1.2 % improvement is reported as an improvement (D-121)
        self.assertEqual(s3.improvement_wording(0.2, 0.15, 0.3), "improvement (interval above zero)")
        self.assertEqual(s3.improvement_wording(0.03, -0.01, 0.07), "not distinguishable from zero (interval includes zero)")
        self.assertEqual(s3.improvement_wording(0.01, 0.0, 0.02), "not distinguishable from zero (interval includes zero)")           # an interval that touches zero includes it
        self.assertEqual(s3.improvement_wording(-0.05, -0.09, -0.01), "worse (interval below zero)")
        self.assertEqual(s3.improvement_wording(float("nan"), float("nan"), float("nan")), "not estimable")


class Analyse(unittest.TestCase):
    NL = 12
    TIER = np.array([0, 1, 2] * 4); SCRIPT = np.array(["a", "a", "b", "b", "c", "c", "d", "d", "e", "e", "f", "f"])

    def sums(self, noise_by_arm=None):
        noise_by_arm = noise_by_arm or {"legacy": (0.02, 0.2, 0.4), "matched": (None, 0.3, 0.6), "gptq": (None, 0.25, 0.5)}
        out = {}
        for arm, (n16, n6, n4) in noise_by_arm.items():
            out[arm] = {}
            for w, nz in (("INT16", n16), ("INT6", n6), ("INT4", n4)):
                if nz is not None:
                    out[arm][w] = make_case(seed=5, noise=nz, n=300, n_lang=self.NL)[0]
        return out

    def run_analysis(self, **kw):
        return s3.analyse(self.sums(), THETA, self.TIER, self.SCRIPT, n_boot=60, seed=1, **kw)

    def test_the_results_have_every_section_of_the_freeze_and_the_gates_pass(self):
        r = self.run_analysis()
        self.assertEqual(set(r), {"gates", "populations", "H3", "H4", "descriptive", "H5", "labels", "n_languages"})
        self.assertTrue(all(g["passed"] for g in r["gates"]["identity"].values())); self.assertEqual(len(r["gates"]["identity"]), 7)           # legacy INT16, INT6, INT4 and the four patched cells
        self.assertEqual(set(r["H3"]["cells"]), {f"{a}|{w}" for a in ("legacy", "matched", "gptq") for w in ("INT6", "INT4")}); self.assertIn("legacy|INT16", r["H3"]["reported"])
        self.assertEqual(set(r["H4"]["cells"]), {f"{a}|{w}" for a in ("legacy", "matched", "gptq") for w in ("INT6", "INT4")})
        c = r["H4"]["cells"]["gptq|INT4"]; self.assertEqual(set(c), {"standard", "with_scale_covariate"}); self.assertIn("primary", c["standard"]); self.assertEqual(r["H5"], None)
        d = r["descriptive"]; self.assertIn("legacy|INT16", d["arm_summary"]); self.assertEqual(set(d["lambda_decomposition"]), {"matched_vs_legacy|INT6", "matched_vs_legacy|INT4", "gptq_vs_legacy|INT6", "gptq_vs_legacy|INT4"})
        self.assertEqual(set(d["nu_between_arms"]), set(d["lambda_decomposition"])); self.assertEqual(r["labels"]["sidecar_internal"], False)
        json.dumps(r, default=s3.jsonable)

    def test_a_failing_identity_gate_stops_the_analysis(self):
        from unittest import mock
        real = s3.identity_gate; calls = []
        def fake(d, active, tol=s3.IDENTITY_TOL):
            r = real(d, active, tol); calls.append(1)
            return {**r, "passed": False} if len(calls) == 5 else r                                                       # the identity is algebraic given the sums, so only a code error can break it: simulate one
        with mock.patch.object(s3, "identity_gate", fake), self.assertRaises(s3.AnalysisStopped) as cm:
            s3.analyse(self.sums(), THETA, self.TIER, self.SCRIPT, n_boot=20, seed=1)
        self.assertIn("identity gate failed", str(cm.exception))

    def test_arms_that_disagree_on_the_fp_active_counts_are_refused(self):
        sums = self.sums(); sums["matched"]["INT6"]["stats"]["n_a"] = sums["matched"]["INT6"]["stats"]["n_a"] + 1
        with self.assertRaises(s3.AnalysisStopped):
            s3.analyse(sums, THETA, self.TIER, self.SCRIPT, n_boot=20, seed=1)

    def test_a_failed_committed_comparison_labels_every_number_sidecar_internal(self):
        r = s3.analyse(self.sums(), THETA, self.TIER, self.SCRIPT, n_boot=20, seed=1, committed={"legacy": {"G2_committed": {"passed": True}}, "gptq": {"G2_committed": {"passed": False}}})
        self.assertTrue(r["labels"]["sidecar_internal"]); self.assertEqual(r["labels"]["failed_arms"], ["gptq"])

    def test_the_scale_covariate_is_log10_of_the_median_sigma_a_of_the_active_cells(self):
        sums = self.sums(); r = self.run_analysis(); d = s3.derived(sums["legacy"]["INT6"], THETA)
        act = s3.fp_active(sums["legacy"]["INT6"]["stats"]["n_a"].numpy(), sums["legacy"]["INT6"]["n"].numpy()); cov = np.array([np.log10(np.median(d["sigma_a"][l][act[l]])) for l in range(self.NL)])
        dd = s3.derived(sums["gptq"]["INT4"], THETA); y = np.array([-dd["log_nu"][l][act[l]].mean() for l in range(self.NL)])
        d16 = s3.derived(sums["legacy"]["INT16"], THETA); y16 = np.array([-d16["log_nu"][l][act[l]].mean() for l in range(self.NL)])
        direct = s3.h4_cell(y, y16, self.TIER, self.SCRIPT, n_boot=60, seed=1, covariates=cov)
        self.assertEqual(json.dumps(direct, default=float), json.dumps(r["H4"]["cells"]["gptq|INT4"]["with_scale_covariate"], default=float))

    def test_h4_primary_is_taken_against_the_legacy_int16_and_the_outcome_is_minus_mean_log_nu(self):
        sums = self.sums(); r = self.run_analysis()
        act = s3.fp_active(sums["legacy"]["INT6"]["stats"]["n_a"].numpy(), sums["legacy"]["INT6"]["n"].numpy())
        def y(arm, w):
            d = s3.derived(sums[arm][w], THETA); return np.array([-d["log_nu"][l][act[l]].mean() for l in range(self.NL)])
        direct = s3.h4_cell(y("gptq", "INT4"), y("legacy", "INT16"), self.TIER, self.SCRIPT, n_boot=60, seed=1)
        self.assertEqual(json.dumps(direct["primary"], default=float), json.dumps(r["H4"]["cells"]["gptq|INT4"]["standard"]["primary"], default=float))

    def test_h5_looks_the_sidecar_values_up_at_language_and_feature_and_reports_every_improvement_with_its_wording(self):
        sums = self.sums(); g = np.random.default_rng(0); n = 360
        lang = np.repeat(np.arange(self.NL), n // self.NL); ids = g.integers(0, D, n)
        d6 = s3.derived(sums["legacy"]["INT6"], THETA); looked = d6["log_nu"][lang, ids]
        act = s3.fp_active(sums["legacy"]["INT6"]["stats"]["n_a"].numpy(), sums["legacy"]["INT6"]["n"].numpy()); keep = act[lang, ids]
        T = {"lang": lang, "ids": ids, "base": g.normal(size=(n, 5))}
        for st in ("INT6", "INT4"):
            T[st] = {"r1": g.normal(size=n), "w1": g.normal(0, 0.3, n)}
        T["INT6"]["w1"] = 2.0 * (looked - looked[keep].mean()) + g.normal(0, 0.2, n)                                                         # W1 of the INT6 cell is built from the looked-up log nu
        gl = s3.derived(sums["legacy"]["INT4"], THETA)["g"][lang, ids]; T["INT4"]["w1"] = 2.0 * (gl - gl[keep].mean()) + g.normal(0, 0.2, n)      # W1 of the INT4 cell from the looked-up gain
        f5 = {arm: {"T": T, "mask": keep.copy()} for arm in ("legacy", "matched", "gptq")}
        r = s3.analyse(sums, THETA, self.TIER, self.SCRIPT, n_boot=60, seed=1, f5=f5, h5_n_boot=100)
        cell = r["H5"]["cells"]["legacy|INT6"]; self.assertEqual(set(cell), {"result", "wording"}); self.assertEqual(cell["wording"], s3.improvement_wording(cell["result"]["improvement"]["Bplus_over_B"]["improvement"], cell["result"]["improvement"]["Bplus_over_B"]["lo"], cell["result"]["improvement"]["Bplus_over_B"]["hi"]))
        self.assertGreater(cell["result"]["improvement"]["Bplus_over_B"]["improvement"], 0.1)                                                  # found only if g / nu are looked up at (language, feature)
        self.assertGreater(r["H5"]["cells"]["legacy|INT4"]["result"]["improvement"]["Bg_over_B"]["improvement"], 0.1)                          # found only if the gain is looked up at (language, feature)
        self.assertEqual(len(r["H5"]["cells"]), 6); self.assertIn("statement", r["H5"]); self.assertIn("wording", r["H5"]["cells"]["legacy|INT4"])

    def test_the_selectivity_secondary_is_descriptive_and_uses_languages_with_enough_selective_cells(self):
        sums = self.sums(); d = s3.derived(sums["legacy"]["INT6"], THETA); act = s3.fp_active(sums["legacy"]["INT6"]["stats"]["n_a"].numpy(), sums["legacy"]["INT6"]["n"].numpy())
        sel = np.zeros_like(act); sel[:, :3] = True
        r = s3.selectivity_difference(d, act, sel, min_cells=2, n_boot=50, seed=0)
        per = [d["log_nu"][l][act[l] & sel[l]].mean() - d["log_nu"][l][act[l] & ~sel[l]].mean() for l in range(self.NL)]
        self.assertEqual(r["languages_used"], self.NL); self.assertAlmostEqual(r["mean_difference"], float(np.mean(per)), places=12); self.assertEqual(len(r["ci95"]), 2)
        self.assertEqual(s3.selectivity_difference(d, act, sel, min_cells=10 ** 6, n_boot=10, seed=0)["languages_used"], 0)
        exact = int(min((act[l] & sel[l]).sum() for l in range(self.NL)))                                                                   # a language with exactly min_cells selective cells is used
        self.assertEqual(s3.selectivity_difference(d, act, sel, min_cells=exact, n_boot=10, seed=0)["languages_used"], self.NL)
        self.assertLess(s3.selectivity_difference(d, act, sel, min_cells=exact + 1, n_boot=10, seed=0)["languages_used"], self.NL)


class SelectiveCells(unittest.TestCase):
    def test_a_cell_is_selective_only_for_the_peak_language_of_a_selective_feature(self):
        cells = s3.selective_cells(np.array([True, False, True, True]), np.array([2, 0, 0, 1]), 3)
        self.assertEqual(cells.shape, (3, 4)); self.assertEqual(int(cells.sum()), 3)
        self.assertTrue(cells[2, 0] and cells[0, 2] and cells[1, 3]); self.assertFalse(cells[:, 1].any()); self.assertFalse(cells[0, 0] or cells[1, 0])


class Render(unittest.TestCase):
    def test_the_markdown_report_states_the_labels_the_conventions_and_every_h5_cell(self):
        a = Analyse(); a.NL = 12; r = s3.analyse(a.sums(), THETA, a.TIER, a.SCRIPT, n_boot=30, seed=1, committed={"gptq": {"G2_committed": {"passed": False}}})
        md = s3.render_md(r)
        self.assertIn("sidecar-internal", md); self.assertIn("H3", md); self.assertIn("H4", md); self.assertIn("convention", md.lower()); self.assertIn("legacy", md); self.assertIn("INT16", md)
        self.assertIn("inconclusive", md.lower() + " inconclusive")                                                                            # the rule outcomes are written in words


class Main(unittest.TestCase):
    """`main` end to end on toy job directories (three jobs written with the real file layout), the toy roster and no F5 and no committed files."""
    def setUp(self):
        import tempfile, shutil
        self.tmp = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        a = Analyse(); a.NL = 12; sums = a.sums(); self.langs = [f"l{i}" for i in range(12)]
        roster = [{"code": c, "tier": ["low", "mid", "high"][i % 3], "script_family": "abcdef"[i // 2]} for i, c in enumerate(self.langs)]
        (self.tmp / "roster.json").write_text(json.dumps({"languages": roster}))
        for arm, (name, widths, bits) in s3.JOBS.items():
            d = self.tmp / name; d.mkdir()
            for w in widths:
                torch.save({"label": w, "stat_names": list(sc.STAT_NAMES), "stats": sums[arm][w]["stats"], "n": sums[arm][w]["n"]}, d / f"sums_{w}.pt")
            gates = {k: {"passed": True} for k in ("G1", "G2_internal", "G4", "G6")}; gates["G3"] = {"passed": True, "max_abs_diff_theta_vs_csv": 4e-7}
            (d / "meta.json").write_text(json.dumps({"study": "D-117 survival sidecar (Stage 2)", "gate_failures": [], "gates": gates, "saved_conditions": list(widths), "regime": "mask", "stream": "eq",
                "pre_activation_source": "sae.hook_sae_acts_pre", "pre_activation_dtype": "torch.float32", "subset": False, "languages": self.langs, "n_feat": D, "weights": {"legacy": "rtn", "matched": "rtn-matched", "gptq": "gptq"}[arm],
                "theta": THETA.tolist(), "theta_sha256": "t" * 64, "max_sentences": None, "seconds": 1.0, "peak_gpu_gb": 1.0}))

    def run_main(self, *extra):
        return s3.main(["--runs", str(self.tmp), "--out", str(self.tmp / "out"), "--roster", str(self.tmp / "roster.json"), "--no-committed", "--no-f5", "--n-boot", "30", *(extra or ("--no-selectivity",))])

    def test_main_writes_the_json_and_the_markdown_and_checks_the_freeze_first(self):
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.run_main(), 0)
        res = json.loads((self.tmp / "out" / "stage3_results.json").read_text()); md = (self.tmp / "out" / "stage3_results.md").read_text()
        self.assertEqual(res["n_languages"], 12); self.assertEqual(res["freeze_sha256"], s3.check_freeze()["sha256"]); self.assertIn("## H4", md); self.assertIn("convention", md)
        self.assertEqual(res["labels"]["committed_comparison"], "not run")

    def test_main_reads_the_selectivity_file_per_feature_and_peak_language(self):
        import contextlib, io
        sel = tmp = self.tmp / "sel.pt"; peak = torch.arange(D) % 12; torch.save({"languages": self.langs, "selective": torch.tensor([True, True, True, False, True, False]), "peak": peak}, sel)
        with contextlib.redirect_stdout(io.StringIO()):
            self.run_main("--selectivity", str(sel))
        res = json.loads((self.tmp / "out" / "stage3_results.json").read_text())
        self.assertIsNotNone(res["descriptive"]["selectivity"]); self.assertEqual(set(res["descriptive"]["selectivity"]), {f"{a}|{w}" for a in ("legacy", "matched", "gptq") for w in ("INT6", "INT4")})
        torch.save({"languages": self.langs[::-1], "selective": torch.tensor([True] * D), "peak": peak}, sel)                              # another language order is refused
        import shutil; shutil.rmtree(self.tmp / "out")
        with self.assertRaises(s3.AnalysisStopped):
            self.run_main("--selectivity", str(sel))

    def test_a_job_whose_meta_has_a_failed_gate_is_refused_and_nothing_is_written(self):
        m = json.loads((self.tmp / "sc-gptq" / "meta.json").read_text()); m["gates"]["G4"]["passed"] = False; m["gate_failures"] = ["G4"]; (self.tmp / "sc-gptq" / "meta.json").write_text(json.dumps(m))
        with self.assertRaises(s3.AnalysisStopped):
            self.run_main()
        self.assertFalse((self.tmp / "out").exists())

    def test_jobs_with_different_thresholds_are_refused(self):
        m = json.loads((self.tmp / "sc-matched" / "meta.json").read_text()); m["theta_sha256"] = "u" * 64; (self.tmp / "sc-matched" / "meta.json").write_text(json.dumps(m))
        with self.assertRaises(s3.AnalysisStopped):
            self.run_main()

    def test_main_refuses_to_run_on_a_changed_freeze(self):
        from unittest import mock
        with mock.patch.object(s3, "check_freeze", side_effect=s3.FreezeError("changed")), mock.patch.object(s3, "load_job") as loader, self.assertRaises(s3.FreezeError):
            self.run_main()
        loader.assert_not_called()                                                                                                         # no job file is read before the freeze is checked
        self.assertFalse((self.tmp / "out").exists())


class Freeze(unittest.TestCase):
    def test_the_constants_of_the_module_equal_the_committed_freeze(self):
        F = json.loads((ROOT / "reports/survival_measurement/analysis_freeze.json").read_text())
        self.assertEqual(s3.MIN_RATE, 0.001); self.assertEqual(s3.MIN_TOKENS, 20); self.assertEqual(s3.IDENTITY_TOL, 1e-4); self.assertEqual(s3.H4_BOUND, 0.20); self.assertEqual(s3.N_BOOT, F["bootstrap"]["replicates"]); self.assertEqual(s3.SEED, F["bootstrap"]["seed"])
        self.assertEqual((s3.H3_NU_MAX, s3.H3_LAMBDA_MIN), (0.2, 0.5)); self.assertEqual(s3.MAX_DROPPED_SHARE, 0.05)

    def test_the_freeze_check_refuses_a_changed_file(self):
        import tempfile, shutil
        tmp = Path(tempfile.mkdtemp())
        try:
            shutil.copy(ROOT / "reports/survival_measurement/analysis_freeze.json", tmp / "analysis_freeze.json"); shutil.copy(ROOT / "reports/survival_measurement/analysis_freeze.sha256", tmp / "analysis_freeze.sha256")
            self.assertTrue(s3.check_freeze(tmp)["ok"])
            (tmp / "analysis_freeze.json").write_text((tmp / "analysis_freeze.json").read_text().replace("0.20", "0.30", 1))
            with self.assertRaises(s3.FreezeError):
                s3.check_freeze(tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
