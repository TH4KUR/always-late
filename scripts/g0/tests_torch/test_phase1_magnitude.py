"""N6 tests for scripts/g0/phase1_magnitude.py (decision D-020; handoff 002 section 4, pre-specified before any magnitude number was looked at). Source of the question:
issues.md C6 / PLAN P1-5 (Duan's released per-feature data: survival follows maximum activation, Spearman 0.78, not firing rate, -0.085). Three parts:
(a) threshold-free dose-response of survival on S inside firing-rate strata, (b) descriptive magnitude comparison, (c) the unchanged contrast with controls matched on
firing-rate stratum x maximum-activation tercile. Hand-computed values and planted effects on synthetic data."""
import json, shutil, sys, tempfile, unittest
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_magnitude as mg   # noqa: E402
import phase1_threshold as th   # noqa: E402
import test_phase1_analysis as tpa   # noqa: E402   (imported as a module so its test class is not collected a second time)

ROOT = Path(__file__).resolve().parents[3]


class WithinStratumRho(unittest.TestCase):
    def test_weighted_mean_of_per_stratum_spearman_by_hand(self):
        x = np.array([1, 2, 3, 1, 2, 3, 4.]); y = np.array([1, 2, 3, 4, 3, 2, 1.]); s = np.array([0, 0, 0, 1, 1, 1, 1])
        r = mg.within_stratum_rho(x, y, s, n_perm=20, seed=0)
        self.assertAlmostEqual(r["per_stratum"][0]["rho"], 1.0); self.assertAlmostEqual(r["per_stratum"][1]["rho"], -1.0)
        self.assertAlmostEqual(r["rho"], (3 * 1.0 + 4 * -1.0) / 7); self.assertEqual(r["n"], 7)

    def test_ties_use_average_ranks_like_scipy(self):
        rng = np.random.default_rng(0); x = rng.normal(size=60); y = (x + rng.normal(size=60) > 0).astype(float); s = np.repeat([0, 1, 2], 20)
        r = mg.within_stratum_rho(x, y, s, n_perm=10, seed=0)
        for k in range(3):
            m = s == k; self.assertAlmostEqual(r["per_stratum"][k]["rho"], spearmanr(x[m], y[m]).statistic, places=12)

    def test_a_stratum_with_constant_outcome_or_fewer_than_three_features_is_left_out_and_reported(self):
        x = np.array([1, 2, 3, 4, 1, 2, 1, 2.]); y = np.array([0, 1, 1, 0, 1, 1, 0, 1.]); s = np.array([0, 0, 0, 0, 1, 1, 2, 2])
        r = mg.within_stratum_rho(x, y, s, n_perm=10, seed=0)
        self.assertEqual(sorted(r["per_stratum"]), [0]); self.assertEqual(sorted(r["undefined_strata"]), [1, 2]); self.assertAlmostEqual(r["rho"], r["per_stratum"][0]["rho"])

    def test_a_large_stratum_with_a_constant_outcome_is_left_out_and_does_not_make_the_result_undefined(self):
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8.]); y = np.array([0, 1, 0, 1, 1, 1, 1, 1.]); s = np.array([0, 0, 0, 0, 1, 1, 1, 1])             # stratum 1: four features, y constant
        r = mg.within_stratum_rho(x, y, s, n_perm=10, seed=0); self.assertEqual(r["undefined_strata"], [1]); self.assertEqual(sorted(r["per_stratum"]), [0]); self.assertFalse(np.isnan(r["rho"]))

    def test_a_planted_association_is_found_two_sided_and_a_null_is_not(self):
        rng = np.random.default_rng(1); n = 600; s = rng.integers(0, 5, n); x = rng.normal(size=n)
        pos = mg.within_stratum_rho(x, x + rng.normal(size=n), s, n_perm=300, seed=0); neg = mg.within_stratum_rho(x, -x + rng.normal(size=n), s, n_perm=300, seed=0)
        null = mg.within_stratum_rho(x, rng.normal(size=n), s, n_perm=300, seed=0)
        self.assertLess(pos["p"], 0.01); self.assertLess(neg["p"], 0.01); self.assertGreater(pos["rho"], 0.3); self.assertLess(neg["rho"], -0.3); self.assertGreater(null["p"], 0.05)

    def test_permutation_stays_inside_strata_so_a_between_stratum_confound_is_not_an_association(self):
        rng = np.random.default_rng(2); s = np.repeat(np.arange(5), 100); x = s * 10.0 + rng.normal(size=500); y = s * 10.0 + rng.normal(size=500)       # x and y both follow the stratum, nothing within it
        r = mg.within_stratum_rho(x, y, s, n_perm=300, seed=0); self.assertGreater(r["p"], 0.05); self.assertLess(abs(r["rho"]), 0.2)

    def test_p_has_the_plus_one_correction_and_is_reproducible(self):
        x = np.arange(30.0); s = np.zeros(30, int); r1 = mg.within_stratum_rho(x, x, s, n_perm=50, seed=4); r2 = mg.within_stratum_rho(x, x, s, n_perm=50, seed=4)
        self.assertAlmostEqual(r1["p"], 1 / 51); self.assertEqual(r1["p"], r2["p"])


class Deciles(unittest.TestCase):
    def test_deciles_are_equal_count_ranks_and_raw_rates_are_by_hand(self):
        S = np.arange(1, 21, dtype=float); y = np.array([1] * 10 + [0] * 10, float); st = np.tile([0, 1], 10)
        t = mg.decile_table(S, y, st, n_bins=10)
        self.assertEqual([r["n"] for r in t], [2] * 10); self.assertEqual([r["raw_rate"] for r in t], [1.0] * 5 + [0.0] * 5); self.assertAlmostEqual(t[0]["mean_S"], 1.5)

    def test_standardised_rate_uses_the_overall_stratum_shares_and_drops_missing_cells(self):
        # stratum 0 survives 100 %, stratum 1 survives 0 %; shares 3/8 and 5/8. Decile A holds only stratum 1 features, decile B both strata
        S = np.arange(8.0); st = np.array([1, 1, 1, 1, 0, 0, 1, 0]); y = (st == 0).astype(float)
        t = mg.decile_table(S, y, st, n_bins=2)                       # decile 0 = S 0..3 (all stratum 1), decile 1 = S 4..7 (strata 0, 0, 1, 0)
        self.assertEqual(t[0]["n_strata_missing"], 1); self.assertAlmostEqual(t[0]["standardised_rate"], 0.0)
        self.assertEqual(t[1]["n_strata_missing"], 0); w0 = 3 / 8; self.assertAlmostEqual(t[1]["standardised_rate"], w0 * 1.0 + (1 - w0) * 0.0)       # stratum 0 holds 3 of the 8 features (indices 4, 5, 7), stratum 1 holds 5
        self.assertAlmostEqual(t[1]["raw_rate"], 3 / 4)                # raw rate is confounded by composition, the standardised one is not


class MagnitudeStrata(unittest.TestCase):
    def test_terciles_are_equal_count_within_each_stratum_among_evaluable_features(self):
        strata = torch.tensor([0] * 9 + [1] * 6 + [-1]); supported = strata >= 0; ev = supported.clone(); m = torch.tensor([float(i) for i in range(9)] + [10.0 * i for i in range(6)] + [5.0])
        s2 = mg.magnitude_strata(strata, supported, ev, m)
        self.assertEqual(s2[:9].tolist(), [0, 0, 0, 1, 1, 1, 2, 2, 2]); self.assertEqual(s2[9:15].tolist(), [3, 3, 4, 4, 5, 5]); self.assertEqual(int(s2[15]), -1)

    def test_a_supported_but_not_evaluable_feature_gets_the_tercile_its_magnitude_falls_in(self):
        strata = torch.zeros(10, dtype=torch.long); supported = torch.ones(10, dtype=torch.bool); ev = torch.tensor([1] * 9 + [0], dtype=torch.bool)
        m = torch.tensor([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 7.5])           # the last feature is not evaluable; 7.5 lies above the 6th evaluable value
        s2 = mg.magnitude_strata(strata, supported, ev, m); self.assertEqual(int(s2[9]), 2); self.assertEqual(s2[:9].tolist(), [0, 0, 0, 1, 1, 1, 2, 2, 2])

    def test_a_not_evaluable_feature_above_every_evaluable_magnitude_stays_in_the_top_bin_of_its_own_stratum(self):
        strata = torch.tensor([0] * 4 + [1] * 3); supported = torch.ones(7, dtype=torch.bool); ev = torch.tensor([1, 1, 1, 0, 1, 1, 1], dtype=torch.bool)
        m = torch.tensor([1.0, 2.0, 3.0, 100.0, 1.0, 2.0, 3.0]); s2 = mg.magnitude_strata(strata, supported, ev, m)
        self.assertEqual(int(s2[3]), 2); self.assertEqual(s2[4:].tolist(), [3, 4, 5])                  # without the cap the 100.0 would get bin 3 = the next stratum's id 3

    def test_unsupported_features_keep_minus_one_and_the_combined_id_is_stratum_times_three_plus_tercile(self):
        strata = torch.tensor([2, 2, 2, -1]); supported = strata >= 0; s2 = mg.magnitude_strata(strata, supported, supported.clone(), torch.tensor([1.0, 2.0, 3.0, 9.0]))
        self.assertEqual(s2.tolist(), [6, 7, 8, -1])


class EvaluableTable(unittest.TestCase):
    def data(self):
        # 2 languages x 5 features; every value differs by language so reading the wrong language is visible
        peak = torch.tensor([0, 1, 0, 1, 1]); supported = torch.tensor([1, 1, 1, 1, 0], dtype=torch.bool)
        fire = torch.tensor([[0.5, 0.5, 0.001, 0.5, 0.5], [0.0005, 0.9, 0.5, 0.002, 0.5]])
        maxa = torch.tensor([[11.0, 12, 13, 14, 15], [21.0, 22, 23, 24, 25]])
        corr = {6: torch.tensor([[0.95, 0.10, 0.91, 0.20, 0.30], [0.50, 0.60, 0.70, 0.85, 0.99]])}
        surv = {"firing_rate": fire, "max_activation": maxa, "corr": corr}
        selp = {"peak": peak, "supported": supported, "S": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]), "strata": torch.tensor([0, 1, 2, 3, -1])}
        return surv, selp

    def test_values_are_read_at_each_features_peak_language_and_only_for_supported_evaluable_features(self):
        surv, selp = self.data(); t = mg.evaluable_table(surv, selp, 6)
        # feature 0: peak 0, rate .5 ok; feature 1: peak 1 ok; feature 2: peak 0, rate exactly .001 -> not evaluable; feature 3: peak 1, rate .002 ok; feature 4 unsupported
        self.assertEqual(t["feature"].tolist(), [0, 1, 3]); self.assertEqual(t["peak"].tolist(), [0, 1, 1])
        self.assertEqual(t["m"].tolist(), [11.0, 22.0, 24.0]); self.assertTrue(np.allclose(t["r"], [0.95, 0.60, 0.85])); self.assertEqual(t["y"].tolist(), [1.0, 0.0, 0.0])
        self.assertEqual(t["S"].tolist(), [1.0, 2.0, 4.0]); self.assertEqual(t["stratum"].tolist(), [0, 1, 3])

    def test_survival_is_strictly_above_0_9_and_a_nan_correlation_is_an_error(self):
        surv, selp = self.data(); surv["corr"][6] = surv["corr"][6].double(); surv["corr"][6][0, 0] = 0.9          # float64 so that the value is exactly their cut (float32 0.9 is not 0.9)
        self.assertEqual(mg.evaluable_table(surv, selp, 6)["y"][0], 0.0)
        surv["corr"][6][0, 0] = float("nan")
        with self.assertRaises(ValueError): mg.evaluable_table(surv, selp, 6)


class Descriptives(unittest.TestCase):
    def test_medians_and_the_log10_difference_by_hand(self):
        tab = {"feature": np.arange(4), "m": np.array([10.0, 10.0, 100.0, 100.0]), "stratum": np.zeros(4, int)}
        selp = {"S": torch.tensor([5.0, 5.0, 1.0, 1.0]), "supported": torch.ones(4, dtype=torch.bool), "stable_count": torch.full((4,), 100), "min_stable_needed": 80,
                "strata": torch.zeros(4, dtype=torch.long), "null_cutoff": 1.8}
        r = mg.descriptives(tab, selp, [("2.0", 2.0)], seed=0, n_perm=50)[0]
        self.assertEqual((r["n_selective"], r["n_control"]), (2, 2)); self.assertEqual((r["median_m_selective"], r["median_m_control"]), (10.0, 100.0)); self.assertAlmostEqual(r["log10_m_delta"], -1.0)
        self.assertLess(r["log10_m_p"], 1.0)

    def test_an_empty_group_is_reported_undefined(self):
        tab = {"feature": np.arange(2), "m": np.array([1.0, 2.0]), "stratum": np.zeros(2, int)}
        selp = {"S": torch.tensor([1.0, 1.0]), "supported": torch.ones(2, dtype=torch.bool), "stable_count": torch.full((2,), 100), "min_stable_needed": 80, "strata": torch.zeros(2, dtype=torch.long), "null_cutoff": 1.8}
        self.assertIn("undefined", mg.descriptives(tab, selp, [("5.0", 5.0)], seed=0, n_perm=10)[0])


class DoseResponseWiring(unittest.TestCase):
    def test_each_statistic_is_computed_from_the_right_columns(self):
        rng = np.random.default_rng(0); L, F = 2, 80
        peak = torch.tensor(rng.integers(0, L, F)); fire = torch.tensor(rng.uniform(0.002, 0.01, (L, F)), dtype=torch.float32); maxa = torch.tensor(rng.uniform(1, 9, (L, F)), dtype=torch.float32)
        corr = {6: torch.tensor(np.clip(rng.uniform(0.7, 1.0, (L, F)), -1, 1), dtype=torch.float32)}
        surv = {"firing_rate": fire, "max_activation": maxa, "corr": corr}
        selp = {"peak": peak, "supported": torch.ones(F, dtype=torch.bool), "S": torch.tensor(rng.uniform(1, 8, F), dtype=torch.float32), "strata": torch.tensor(rng.integers(0, 3, F))}
        d = mg.dose_response(surv, selp, [6], n_perm=20, seed=0)["6"]; t = mg.evaluable_table(surv, selp, 6)
        for key, x, y in (("rho_S_survival", t["S"], t["y"]), ("rho_S_r", t["S"], t["r"]), ("rho_m_survival", t["m"], t["y"]), ("rho_S_m", t["S"], t["m"])):
            self.assertAlmostEqual(d[key]["rho"], mg.within_stratum_rho(x, y, t["stratum"], n_perm=20, seed=0)["rho"], places=12)
        self.assertEqual(d["n_features"], len(t["S"])); self.assertEqual(len(d["deciles"]), 10)
        self.assertGreater(len({round(d[k]["rho"], 6) for k in ("rho_S_survival", "rho_S_r", "rho_m_survival", "rho_S_m")}), 1)       # the four statistics are not all the same number


class FinalReading(unittest.TestCase):
    """Pre-specified (handoff 002, amended before any real magnitude number was seen): coarse magnitude bins leave residual confounding when survival follows magnitude steeply
    (synthetic calibration, D-020), which would favour the selectivity reading; so the matching is run at 3, 5 and 10 bins per firing-rate stratum, and the headline reading is the one
    at the FINEST resolution that still keeps at least half of the selective features (common support); the readings at all resolutions are reported."""
    def test_finest_resolution_with_enough_common_support_gives_the_headline(self):
        f = mg.final_reading({3: {"reading": "supported", "retained": 0.95}, 5: {"reading": "supported", "retained": 0.9}, 10: {"reading": "explained by magnitude", "retained": 0.7}})
        self.assertEqual(f["bins"], 10); self.assertEqual(f["reading"], "explained by magnitude"); self.assertFalse(f["all_agree"])
        g = mg.final_reading({3: {"reading": "supported", "retained": 0.95}, 5: {"reading": "supported", "retained": 0.8}, 10: {"reading": "explained by magnitude", "retained": 0.3}})
        self.assertEqual(g["bins"], 5); self.assertEqual(g["reading"], "supported"); self.assertFalse(g["all_agree"])

    def test_all_resolutions_agreeing_is_reported_and_no_resolution_with_support_falls_back_to_the_coarsest(self):
        self.assertTrue(mg.final_reading({3: {"reading": "supported", "retained": 0.9}, 10: {"reading": "supported", "retained": 0.6}})["all_agree"])
        h = mg.final_reading({3: {"reading": "partly explained", "retained": 0.4}, 10: {"reading": "explained by magnitude", "retained": 0.1}})
        self.assertEqual(h["bins"], 3); self.assertIn("support", h["note"])
        self.assertEqual(mg.final_reading({3: {"reading": "supported", "retained": 0.9}, 10: {"reading": "explained by magnitude", "retained": 0.5}})["bins"], 10)      # exactly half is enough (>=)


class Reading(unittest.TestCase):
    """The reading rule written BEFORE the numbers (handoff 002): at tau = 5 the INT6 contrast with magnitude-matched controls (a) stays negative with Holm < 0.05 -> selectivity
    reading supported; (b) is not significant or shrinks to within +-1 pp -> magnitude explains it; in between -> partly explained. Operationalisation of 'in between' (fixed now,
    recorded in D-020): significant, more negative than -1 pp, but retaining less than half of the rate-only gap."""
    def test_cases(self):
        r = lambda rate, matched, holm: mg.reading(rate, matched, holm)
        self.assertEqual(r(-0.0716, -0.065, 0.001), "supported")                    # keeps almost all of the gap
        self.assertEqual(r(-0.07, -0.04, 0.001), "supported")                       # keeps more than half
        self.assertEqual(r(-0.07, -0.02, 0.01), "partly explained")                 # significant, beyond -1 pp, less than half left
        self.assertEqual(r(-0.07, -0.004, 0.3), "explained by magnitude")           # not significant
        self.assertEqual(r(-0.07, -0.008, 0.001), "explained by magnitude")         # significant but within +-1 pp: the explicit 'shrinks' case takes precedence
        self.assertEqual(r(-0.07, -0.01, 0.001), "explained by magnitude")          # exactly -1 pp is within the band
        self.assertEqual(r(-0.07, +0.03, 0.001), "reversed")                        # significant and positive
        self.assertEqual(r(-0.07, -0.0101, 0.0499), "partly explained")             # just outside the band, just significant, little retained
        self.assertEqual(r(-0.07, -0.04, 0.05), "explained by magnitude")           # Holm exactly 0.05 is NOT significant (strict, as in the analysis)
        self.assertEqual(r(-0.08, -0.04, 0.001), "supported")                       # exactly half of the gap retained still counts as retaining at least half
        self.assertEqual(r(0.0, -0.03, 0.001), "supported")                         # no rate-only gap to explain: a significant negative matched gap stands


def build_run(case, F=4000, seed=0):
    """case 'magnitude': survival depends on max activation only and selective features have low activations; case 'selectivity': selective features lose survival whatever their
    activation (matched magnitude distributions). The synthetic run directory is the analysis tests' own, with S, stability, max activations and correlations rewritten."""
    helper = tpa.RunAnalysis("test_a_planted_tier_effect_gives_a_small_p_and_a_null_does_not")
    out = helper.synth(F=F, selective_penalty=0.0); sel = torch.load(out / "selectivity.pt"); surv = torch.load(out / "survival.pt"); rng = np.random.default_rng(seed)
    selective = sel["selective"].numpy(); peak = sel["peak"].numpy(); L = surv["corr"][7].shape[0]
    S = np.ones(F); S[selective] = np.linspace(2.1, 9.0, int(selective.sum())); sel.update(S=torch.tensor(S, dtype=torch.float32), null_cutoff=2.0, min_stable_needed=80, stable_count=torch.full((F,), 100))
    m = np.exp(rng.normal(0, 0.5, F)); m[selective] *= (0.3 if case == "magnitude" else 1.0)
    surv["max_activation"] = torch.tensor(np.tile(m, (L, 1)), dtype=torch.float32)
    for b, base in ((7, 0.985), (6, 0.95), (5, 0.9), (4, 0.8)):
        r = base + rng.normal(0, 0.02, (L, F)) + (0.05 * np.log(m)[None, :] if case == "magnitude" else 0.0)
        if case == "selectivity":
            r = r - 0.08 * selective[None, :]
        surv["corr"][b] = torch.tensor(np.clip(r, -1, 1), dtype=torch.float32)
    torch.save(sel, out / "selectivity.pt"); torch.save(surv, out / "survival.pt"); return helper, out


class EndToEnd(unittest.TestCase):
    def go(self, case, n_perm=60, bins=(3, 10)):
        helper, out = build_run(case); self.addCleanup(helper.doCleanups)
        return out, mg.run_magnitude(out, ROOT / "reports/g0/phase1_languages.json", Path(out) / "mag", [3.0], bins=bins, n_perm=n_perm, n_boot=20, seed=0)

    def test_a_magnitude_only_effect_looks_like_selectivity_under_rate_matching_and_finer_magnitude_bins_remove_it(self):
        # calibrated on synthetic data only (D-020): rate-only gap about -0.52; terciles leave about -0.08 (coarse bins under-correct a steep link), 5 bins about -0.03, 10 bins about -0.01
        out, res = self.go("magnitude")
        rate = res["rate_only"]["rows"][1]["tests"]["T2_int6"]["delta"]; m3 = res["matched"]["3"]["rows"][1]["tests"]["T2_int6"]; m10 = res["matched"]["10"]["rows"][1]["tests"]["T2_int6"]
        self.assertLess(rate, -0.40)                                              # the spurious selectivity gap that rate matching cannot remove
        self.assertLess(abs(m10["delta"]), 0.03)                                  # finest magnitude matching removes it
        self.assertLess(abs(m10["delta"]), abs(m3["delta"]))                      # coarse bins leave more residual confounding than fine ones
        self.assertLess(abs(m3["delta"]), 0.5 * abs(rate))                        # but even terciles remove most of it
        rd = res["readings"]["3.0"]; self.assertEqual(rd["final"]["reading"], "explained by magnitude"); self.assertEqual(rd["final"]["bins"], 10)
        for n, row in ((3, res["matched"]["3"]["rows"][1]), (10, res["matched"]["10"]["rows"][1])):
            t = row["tests"]["T2_int6"]; self.assertAlmostEqual(rd["by_bins"][str(n) if str(n) in rd["by_bins"] else n]["retained"], 1 - t["n_selective_dropped"] / t["n_selective"])

    def test_a_real_selectivity_effect_survives_every_magnitude_matching_and_is_read_as_supported(self):
        out, res = self.go("selectivity", n_perm=200)                             # 200 permutations so that the smallest possible p (1/201) can clear Holm over three tests
        rate = res["rate_only"]["rows"][1]["tests"]["T2_int6"]["delta"]; self.assertLess(rate, -0.4)
        for n in ("3", "10"):
            row = res["matched"][n]["rows"][1]; self.assertLess(row["tests"]["T2_int6"]["delta"], -0.4); self.assertLess(row["holm"]["T2_int6"], 0.05)
        self.assertEqual(res["readings"]["3.0"]["final"]["reading"], "supported"); self.assertTrue(res["readings"]["3.0"]["final"]["all_agree"])

    def test_dose_response_and_descriptives_are_reported_for_every_width_and_threshold(self):
        out, res = self.go("magnitude")
        for w in ("7", "6", "5"):
            d = res["dose_response"][w]; self.assertIn("rho_S_survival", d); self.assertIn("rho_m_survival", d); self.assertEqual(len(d["deciles"]), 10)
            self.assertLess(d["rho_m_survival"]["rho"], 1.0); self.assertGreater(d["rho_m_survival"]["rho"], 0.0)          # planted: survival rises with magnitude
        desc = res["descriptives"]; self.assertEqual([r["label"] for r in desc], ["null", "3.0"]); self.assertLess(desc[1]["median_m_selective"], desc[1]["median_m_control"])     # planted: selective features have lower activations
        self.assertLess(desc[1]["log10_m_delta"], 0)

    def test_the_report_carries_the_pre_specified_reading_rule_and_the_decision_id(self):
        out, res = self.go("selectivity"); md = (Path(out) / "mag" / "magnitude.md").read_text()
        for needle in ("D-020", "issues.md", "C6", "Holm < 0.05", "within +-1 pp", "retaining less than half", "terciles", "reading", "5 and 10"): self.assertIn(needle, md)
        self.assertTrue((Path(out) / "mag" / "magnitude.json").exists())

    def test_a_rate_only_reference_with_other_thresholds_is_matched_by_label_not_by_position(self):
        helper, out = build_run("selectivity"); self.addCleanup(helper.doCleanups); roster = ROOT / "reports/g0/phase1_languages.json"
        n2 = th.run_thresholds(out, roster, Path(out) / "n2", [2.0, 3.0], n_perm=30, n_boot=10, seed=0)                        # rows: null, 2.0, 3.0
        res = mg.run_magnitude(out, roster, Path(out) / "mag", [3.0], bins=(3,), n_perm=30, n_boot=10, seed=0, rate_only_json=Path(out) / "n2" / "thresholds.json")     # plan: null, 3.0
        want = {r["label"]: r["tests"]["T2_int6"]["delta"] for r in n2["rows"]}
        self.assertEqual(res["readings"]["null"]["by_bins"]["3"]["rate_only_delta"], want["null"]); self.assertEqual(res["readings"]["3.0"]["by_bins"]["3"]["rate_only_delta"], want["3.0"])
        self.assertNotEqual(want["2.0"], want["3.0"]); md = (Path(out) / "mag" / "magnitude.md").read_text(); self.assertIn(f"{want['3.0'] * 100:+.2f} pp", md)

    def test_the_original_files_are_untouched(self):
        helper, out = build_run("selectivity"); self.addCleanup(helper.doCleanups); before = {f: (Path(out) / f).read_bytes() for f in ("survival.pt", "selectivity.pt", "meta.json")}
        mg.run_magnitude(out, ROOT / "reports/g0/phase1_languages.json", Path(out) / "mag", [3.0], bins=(3,), n_perm=20, n_boot=10, seed=0)
        for f, b in before.items(): self.assertEqual((Path(out) / f).read_bytes(), b)


class TransformHook(unittest.TestCase):
    """run_thresholds(transform=...) is how the magnitude strata enter the unchanged analysis; the identity transform must change nothing (the N2 plumbing is untouched)."""
    def test_identity_transform_gives_exactly_the_rows_without_a_transform(self):
        helper, out = build_run("selectivity"); self.addCleanup(helper.doCleanups); roster = ROOT / "reports/g0/phase1_languages.json"
        a = th.run_thresholds(out, roster, Path(out) / "a", [3.0], n_perm=30, n_boot=10, seed=0); b = th.run_thresholds(out, roster, Path(out) / "b", [3.0], n_perm=30, n_boot=10, seed=0, transform=lambda s: dict(s))
        for ra, rb in zip(a["rows"], b["rows"]):
            self.assertEqual(ra["n_selective"], rb["n_selective"]); self.assertEqual(ra["n_controls"], rb["n_controls"]); self.assertEqual(ra["tests"]["T2_int6"]["delta"], rb["tests"]["T2_int6"]["delta"])

    def test_a_transform_that_replaces_the_strata_changes_the_controls(self):
        helper, out = build_run("selectivity"); self.addCleanup(helper.doCleanups); roster = ROOT / "reports/g0/phase1_languages.json"
        def coarse(s): s = dict(s); st = s["strata"].clone(); st[st >= 0] = 0; s["strata"] = st; return s
        a = th.run_thresholds(out, roster, Path(out) / "a", [3.0], n_perm=30, n_boot=10, seed=0); c = th.run_thresholds(out, roster, Path(out) / "c", [3.0], n_perm=30, n_boot=10, seed=0, transform=coarse)
        self.assertNotEqual(a["rows"][1]["n_controls"], c["rows"][1]["n_controls"]) if a["rows"][1]["control_strata_short"] != c["rows"][1]["control_strata_short"] else None
        self.assertEqual(c["rows"][1]["n_selective"], a["rows"][1]["n_selective"])           # the selective set does not depend on the strata


if __name__ == "__main__":
    unittest.main()
