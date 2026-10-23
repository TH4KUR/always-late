import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phase1_assoc_margin as am
from phase1_func_mc_calibration import wilson

COLS = ["feature_id", "group", "original_language", "condition", "fp_status", "status", "target_rate", "target_share", "margin", "target_mean"]


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLS})


def feature_rows(fid, lang, fp_status, statuses, margin=0.002, rate=0.01, share=0.1, mean=0.05, group="selective"):
    """The FP row and one row per precision of a feature; statuses: {precision: status}."""
    base = {"feature_id": fid, "group": group, "original_language": lang, "fp_status": fp_status}
    rows = [{**base, "condition": "FP", "status": fp_status, "target_rate": rate, "target_share": share, "margin": margin, "target_mean": mean}]
    rows += [{**base, "condition": p, "status": s, "target_rate": rate, "target_share": share, "margin": margin, "target_mean": mean} for p, s in statuses.items()]
    return rows


def all_p(status):
    return {p: status for p in am.PRECISIONS}


def planted(seed=0, n_per_lang=120, langs=("aaa", "bbb", "ccc", "ddd", "eee", "fff"), beta=None, base=None):
    """Established features with a loss that depends on the baseline margin: logit P(lost_p) = base_p - beta_p * z(log m0); share and scale carry no signal."""
    rng = np.random.default_rng(seed); beta = beta or {p: 0.0 for p in am.PRECISIONS}; base = base or {p: -2.0 for p in am.PRECISIONS}
    lang = np.repeat(np.array(langs), n_per_lang); n = len(lang); logm = rng.normal(-6.0, 1.0, n); z = (logm - logm.mean()) / logm.std()
    rate = np.exp(rng.normal(-4.5, 0.5, n)); m0 = np.exp(logm); feats = {"language": lang, "feature_id": np.arange(n), "m0": m0, "r0": m0 / rate, "s0": rng.uniform(0.05, 0.9, n), "a0": np.exp(rng.normal(-3.0, 0.7, n)), "lost": {}}
    for p in am.PRECISIONS:
        pr = 1 / (1 + np.exp(-(base[p] - beta[p] * z))); feats["lost"][p] = rng.random(n) < pr
    feats["n_selective"] = n + 10; feats["n_established"] = n; feats["n_missing_predictor"] = 0
    return feats


class Reading(unittest.TestCase):
    def test_only_established_selective_features_are_read_with_their_fp_predictors_and_their_flags(self):
        rows = feature_rows(1, "aaa", "retained", {**all_p("retained"), "INT4": "different_peak"}, margin=0.004, rate=0.02, share=0.2, mean=0.3)
        rows += feature_rows(2, "aaa", "different_peak", all_p("different_peak"))                               # not established in full precision
        rows += feature_rows(3, "bbb", "retained", all_p("retained"), group="controls")                         # a control, not selective
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.csv"; write_csv(p, rows); f = am.read_features(p)
        self.assertEqual(f["n_selective"], 2); self.assertEqual(f["n_established"], 1); self.assertEqual(list(f["language"]), ["aaa"]); self.assertEqual(list(f["feature_id"]), [1])
        self.assertAlmostEqual(f["m0"][0], 0.004); self.assertAlmostEqual(f["r0"][0], 0.004 / 0.02); self.assertAlmostEqual(f["s0"][0], 0.2); self.assertAlmostEqual(f["a0"][0], 0.3)
        self.assertTrue(f["lost"]["INT4"][0]); self.assertFalse(f["lost"]["INT6"][0]); self.assertEqual(sorted(f["lost"]), sorted(am.PRECISIONS))

    def test_every_non_retained_status_counts_as_lost(self):
        rows = []
        for k, s in enumerate(["different_peak", "low_support", "ambiguous_peak", "retained"]):
            rows += feature_rows(k, "aaa", "retained", {**all_p("retained"), "INT5": s})
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.csv"; write_csv(p, rows); f = am.read_features(p)
        self.assertEqual(list(f["lost"]["INT5"]), [True, True, True, False])

    def test_a_missing_precision_row_is_an_error(self):
        rows = feature_rows(1, "aaa", "retained", {p: "retained" for p in am.PRECISIONS if p != "INT6"})
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.csv"; write_csv(p, rows)
            with self.assertRaises(ValueError):
                am.read_features(p)

    def test_a_missing_predictor_value_is_nan_and_counted(self):
        rows = feature_rows(1, "aaa", "retained", all_p("retained")); rows[0]["target_mean"] = ""; rows += feature_rows(2, "aaa", "retained", all_p("retained"))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.csv"; write_csv(p, rows); f = am.read_features(p)
        self.assertTrue(np.isnan(f["a0"][0])); self.assertFalse(np.isnan(f["a0"][1])); self.assertEqual(f["n_missing_predictor"], 1)


class Bins(unittest.TestCase):
    def test_bins_are_equal_count_within_each_language_and_do_not_mix_languages(self):
        langs = np.array(["a"] * 10 + ["b"] * 10); v = np.concatenate([np.arange(10) + 100.0, np.arange(10) * 1e-3])
        b = am.quantile_bins(v, langs, k=5)
        for lg in ("a", "b"):
            self.assertEqual(sorted(np.bincount(b[langs == lg]).tolist()), [2, 2, 2, 2, 2]); self.assertEqual(list(b[langs == lg]), [0, 0, 1, 1, 2, 2, 3, 3, 4, 4])         # 0 = the smallest values of the language

    def test_the_bin_follows_the_value_not_the_row_order(self):
        langs = np.array(["a"] * 5); b = am.quantile_bins(np.array([5.0, 1.0, 4.0, 2.0, 3.0]), langs, k=5); self.assertEqual(list(b), [4, 0, 3, 1, 2])

    def test_a_nan_value_gets_bin_minus_one_and_does_not_take_a_rank(self):
        langs = np.array(["a"] * 5); b = am.quantile_bins(np.array([np.nan, 1.0, 2.0, 3.0, 4.0]), langs, k=4); self.assertEqual(list(b), [-1, 0, 1, 2, 3])

    def test_the_loss_fraction_by_bin_has_counts_and_wilson_intervals(self):
        bins = np.array([0, 0, 0, 0, 1, 1, 1, 1, -1]); lost = np.array([True, True, True, False, False, False, False, True, True]); r = am.loss_by_bin(bins, lost, k=2)
        self.assertEqual([x["n"] for x in r], [4, 4]); self.assertEqual([x["lost"] for x in r], [3, 1]); self.assertAlmostEqual(r[0]["fraction"], 0.75)
        lo, hi = wilson(3, 4); self.assertAlmostEqual(r[0]["lo"], lo); self.assertAlmostEqual(r[0]["hi"], hi)                                                              # the row with bin -1 is not counted


class Auroc(unittest.TestCase):
    def test_known_values_ties_and_the_undefined_cases(self):
        self.assertAlmostEqual(am.auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([False, False, True, True])), 1.0)
        self.assertAlmostEqual(am.auroc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([False, False, True, True])), 0.0)
        self.assertAlmostEqual(am.auroc(np.array([0.5, 0.5, 0.5, 0.5]), np.array([False, True, False, True])), 0.5)                             # all tied
        self.assertAlmostEqual(am.auroc(np.array([1.0, 2.0, 2.0, 3.0]), np.array([False, False, True, True])), 0.875)                                                                                                              # positives 2, 3 against negatives 1, 2: pairs 1, .5, 1, 1 -> 3.5 / 4
        self.assertTrue(np.isnan(am.auroc(np.array([1.0, 2.0]), np.array([True, True])))); self.assertTrue(np.isnan(am.auroc(np.array([1.0, 2.0]), np.array([False, False]))))

    def test_a_nan_score_is_left_out(self):
        self.assertAlmostEqual(am.auroc(np.array([np.nan, 0.1, 0.9]), np.array([True, False, True])), 1.0)

    def test_the_within_language_auroc_sees_a_planted_dependence_and_a_null(self):
        dep = planted(0, beta={p: 1.5 for p in am.PRECISIONS}); nul = planted(1, beta={p: 0.0 for p in am.PRECISIONS})
        a = am.within_language_auroc(-dep["m0"], dep["lost"]["INT4"], dep["language"], n_boot=300, seed=0); b = am.within_language_auroc(-nul["m0"], nul["lost"]["INT4"], nul["language"], n_boot=300, seed=0)
        self.assertGreater(a["mean"], 0.75); self.assertLess(a["lo"], a["mean"]); self.assertGreater(a["hi"], a["mean"]); self.assertEqual(a["n_languages"], 6); self.assertLess(abs(b["mean"] - 0.5), 0.06)

    def test_a_language_with_too_few_features_or_a_single_class_is_excluded_and_the_bootstrap_is_reproducible(self):
        f = planted(2, n_per_lang=60, beta={p: 1.0 for p in am.PRECISIONS}); lang = f["language"].copy(); lost = f["lost"]["INT4"].copy()
        lang = np.concatenate([lang, np.array(["tiny"] * 5)]); m0 = np.concatenate([f["m0"], np.full(5, 1e-3)]); lost = np.concatenate([lost, np.array([True, False, True, False, True])])
        lang = np.concatenate([lang, np.array(["same"] * 40)]); m0 = np.concatenate([m0, np.linspace(1e-3, 2e-3, 40)]); lost = np.concatenate([lost, np.zeros(40, dtype=bool)])
        r = am.within_language_auroc(-m0, lost, lang, min_n=20, min_class=3, n_boot=200, seed=0); self.assertEqual(r["n_languages"], 6); self.assertNotIn("tiny", r["per_language"]); self.assertNotIn("same", r["per_language"])
        r2 = am.within_language_auroc(-m0, lost, lang, min_n=20, min_class=3, n_boot=200, seed=0); self.assertEqual((r["lo"], r["hi"]), (r2["lo"], r2["hi"]))
        r3 = am.within_language_auroc(-m0, lost, lang, min_n=20, min_class=3, n_boot=200, seed=1); self.assertEqual(r["mean"], r3["mean"])                                  # the point estimate does not depend on the seed


class AurocDetails(unittest.TestCase):
    def test_the_bootstrap_interval_is_the_percentile_interval_of_language_resamples_with_the_given_seed(self):
        rng = np.random.default_rng(0); langs = np.repeat(np.array(list("abcdef")), 40); score = rng.normal(size=240); y = rng.random(240) < 0.4
        r = am.within_language_auroc(score, y, langs, n_boot=500, seed=7); vals = np.array([r["per_language"][k] for k in "abcdef"]); draws = np.random.default_rng(7).integers(0, 6, (500, 6)); means = vals[draws].mean(axis=1)
        self.assertAlmostEqual(r["lo"], float(np.percentile(means, 2.5))); self.assertAlmostEqual(r["hi"], float(np.percentile(means, 97.5))); self.assertAlmostEqual(r["mean"], float(vals.mean()))

    def test_a_language_with_exactly_the_minimum_number_of_features_and_of_each_class_is_included_and_one_less_is_not(self):
        lost = np.array([True] * 3 + [False] * 17); score = np.arange(20, dtype=float); langs = np.array(["a"] * 20)
        self.assertEqual(am.within_language_auroc(score, lost, langs, min_n=20, min_class=3, n_boot=10)["n_languages"], 1)
        self.assertEqual(am.within_language_auroc(score, lost, langs, min_n=21, min_class=3, n_boot=10)["n_languages"], 0)
        self.assertEqual(am.within_language_auroc(score, lost, langs, min_n=20, min_class=4, n_boot=10)["n_languages"], 0)
        lost2 = np.array([True] * 17 + [False] * 3); self.assertEqual(am.within_language_auroc(score, lost2, langs, min_n=20, min_class=3, n_boot=10)["n_languages"], 1); self.assertEqual(am.within_language_auroc(score, lost2, langs, min_n=20, min_class=4, n_boot=10)["n_languages"], 0)


def fake_res(auroc6, q4, q16, ors, trend_ok=True):
    """A hand-made analysis result for the verdict rules: q4 and q16 are lists of (lost, n) per quintile at INT4 and INT16."""
    from phase1_func_mc_calibration import wilson as w
    def bins(pairs):
        out = []
        for i, (c, n) in enumerate(pairs):
            lo, hi = w(c, n); out.append({"bin": i, "n": n, "lost": c, "fraction": c / n, "lo": lo, "hi": hi})
        return out
    flat = bins([(10, 100)] * 5); steep = bins([(40, 100), (30, 100), (20, 100), (10, 100), (5, 100)]) if trend_ok else flat
    mk = lambda **k: {"auroc": {"m0": {"mean": k.get("a", 0.5)}}, "bins_m0": k["bins"], "logit": {"odds_ratios": k.get("ors", {})}}
    return {"by_precision": {"INT6": mk(a=auroc6, bins=steep, ors=ors), "INT5": mk(bins=steep, ors=ors), "INT4": mk(bins=bins(q4), ors=ors), "INT16": mk(bins=bins(q16))}}


class Verdicts(unittest.TestCase):
    ORS = {"m0": {"or": 0.3}, "a0": {"or": 0.9}, "s0": {"or": 1.1}}

    def test_h_a2a_needs_the_auroc_at_int6_of_at_least_070_and_the_trend_at_three_precisions(self):
        q = [(40, 100), (30, 100), (20, 100), (10, 100), (5, 100)]
        self.assertEqual(am.hypotheses(fake_res(0.70, q, [(1, 100)] * 5, self.ORS))["H-A2a"]["verdict"], "met"); self.assertEqual(am.hypotheses(fake_res(0.699, q, [(1, 100)] * 5, self.ORS))["H-A2a"]["verdict"], "not met")
        self.assertEqual(am.hypotheses(fake_res(0.9, [(5, 100), (10, 100), (20, 100), (30, 100), (40, 100)], [(1, 100)] * 5, self.ORS))["H-A2a"]["verdict"], "not met")           # INT4 reversed: the lowest quintile loses less

    def test_h_a2b_distinguishes_all_quintiles_above_the_floor_from_a_top_quintile_at_the_floor(self):
        floor = [(1, 100)] * 5
        self.assertEqual(am.hypotheses(fake_res(0.8, [(40, 100)] * 5, floor, self.ORS))["H-A2b"]["verdict"], "exceeds the floor in every quintile")
        self.assertEqual(am.hypotheses(fake_res(0.8, [(40, 100)] * 4 + [(1, 100)], floor, self.ORS))["H-A2b"]["verdict"], "at the floor in the top quintile")
        self.assertEqual(am.hypotheses(fake_res(0.8, [(1, 100)] + [(40, 100)] * 4, floor, self.ORS))["H-A2b"]["verdict"], "mixed"); h = am.hypotheses(fake_res(0.8, [(40, 100)] * 4 + [(7, 100)], floor, self.ORS))["H-A2b"]
        self.assertAlmostEqual(h["int4_top"], 0.07); self.assertAlmostEqual(h["floor_top"], 0.01)

    def test_h_a2b_compares_the_lower_bound_of_int4_with_the_upper_bound_of_the_floor(self):
        self.assertEqual(am.hypotheses(fake_res(0.8, [(10, 100)] * 5, [(8, 100)] * 5, self.ORS))["H-A2b"]["verdict"], "at the floor in the top quintile")           # overlapping intervals: not above the floor

    def test_h_a2c_needs_the_margin_to_beat_both_the_scale_and_the_share(self):
        a = am.hypotheses(fake_res(0.8, [(40, 100)] * 5, [(1, 100)] * 5, self.ORS))["H-A2c"]; self.assertEqual(a["verdict"], "met")
        b = am.hypotheses(fake_res(0.8, [(40, 100)] * 5, [(1, 100)] * 5, {"m0": {"or": 0.3}, "a0": {"or": 0.9}, "s0": {"or": 4.0}}))["H-A2c"]; self.assertEqual(b["verdict"], "not met")           # the share beats the margin
        c = am.hypotheses(fake_res(0.8, [(40, 100)] * 5, [(1, 100)] * 5, {"m0": {"or": 0.3}, "a0": {"or": 0.1}, "s0": {"or": 1.0}}))["H-A2c"]; self.assertEqual(c["verdict"], "not met")           # the scale beats the margin
        d = am.hypotheses(fake_res(0.8, [(40, 100)] * 5, [(1, 100)] * 5, {"m0": {"or": 3.0}, "a0": {"or": 0.9}, "s0": {"or": 1.1}}))["H-A2c"]; self.assertEqual(d["verdict"], "met")            # the direction of the effect does not matter, only its size


class FixedEffectsLogit(unittest.TestCase):
    def test_a_planted_odds_ratio_is_recovered_and_a_null_predictor_is_not_significant(self):
        rng = np.random.default_rng(2); n_l, n = 8, 150; lang = np.repeat(np.array([f"l{i}" for i in range(n_l)]), n); x1 = rng.normal(size=n_l * n); x2 = rng.normal(size=n_l * n); off = np.repeat(rng.normal(0, 1.0, n_l), n)
        y = rng.random(n_l * n) < 1 / (1 + np.exp(-(-1.0 + off + 0.8 * x1)))
        r = am.fixed_effects_logit(y, np.column_stack([x1, x2]), lang, ["x1", "x2"])
        self.assertAlmostEqual(np.log(r["odds_ratios"]["x1"]["or"]), 0.8, delta=0.15); self.assertLess(r["odds_ratios"]["x1"]["p"], 1e-6)
        self.assertAlmostEqual(np.log(r["odds_ratios"]["x2"]["or"]), 0.0, delta=0.15); self.assertGreater(r["odds_ratios"]["x2"]["p"], 0.01); self.assertLess(r["odds_ratios"]["x1"]["lo"], r["odds_ratios"]["x1"]["or"]); self.assertGreater(r["odds_ratios"]["x1"]["hi"], r["odds_ratios"]["x1"]["or"])

    def test_languages_without_variation_in_the_outcome_are_dropped_and_counted(self):
        rng = np.random.default_rng(1); n = 100; lang = np.repeat(np.array(["a", "b", "c", "d"]), n); x = rng.normal(size=4 * n); y = rng.random(4 * n) < 1 / (1 + np.exp(-0.7 * x)); y[lang == "c"] = False; y[lang == "d"] = True
        r = am.fixed_effects_logit(y, x[:, None], lang, ["x"]); self.assertEqual(r["n_languages_used"], 2); self.assertEqual(r["n_languages_dropped"], 2); self.assertEqual(r["n"], 2 * n)
        self.assertGreater(r["odds_ratios"]["x"]["or"], 1.5)

    def test_a_row_with_any_missing_predictor_is_left_out_not_only_a_row_with_all_missing(self):
        rng = np.random.default_rng(2); n = 120; lang = np.repeat(np.array(["a", "b"]), n); X = rng.normal(size=(2 * n, 2)); y = rng.random(2 * n) < 0.4; X[:5, 0] = np.nan; X[5:9, 1] = np.nan; X[9, :] = np.nan
        r = am.fixed_effects_logit(y, X, lang, ["x1", "x2"]); self.assertEqual(r["n"], 2 * n - 10); self.assertEqual(r["n_missing"], 10)

    def test_the_standard_errors_are_cluster_robust_by_language_a_slope_that_differs_between_languages_widens_the_interval(self):
        rng = np.random.default_rng(4); n_l, n = 8, 200; lang = np.repeat(np.array([f"l{i}" for i in range(n_l)]), n); x = rng.normal(size=n_l * n); slopes = np.repeat(rng.normal(0.8, 0.6, n_l), n)
        y = rng.random(n_l * n) < 1 / (1 + np.exp(-(-0.5 + slopes * x))); o = am.fixed_effects_logit(y, x[:, None], lang, ["x"])["odds_ratios"]["x"]
        self.assertGreater((np.log(o["hi"]) - np.log(o["lo"])) / (2 * 1.96), 0.12)                                                                           # the naive standard error would be about 0.05

    def test_rows_with_a_missing_predictor_are_left_out_and_counted(self):
        rng = np.random.default_rng(2); n = 120; lang = np.repeat(np.array(["a", "b"]), n); x = rng.normal(size=2 * n); y = rng.random(2 * n) < 0.4; x[:7] = np.nan
        r = am.fixed_effects_logit(y, x[:, None], lang, ["x"]); self.assertEqual(r["n"], 2 * n - 7); self.assertEqual(r["n_missing"], 7)


class Analysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        beta = {"INT16": 0.0, "INT12": 0.0, "INT8": 0.3, "INT7": 0.5, "INT6": 1.0, "INT5": 1.3, "INT4": 1.6}; base = {"INT16": -4.0, "INT12": -4.0, "INT8": -3.0, "INT7": -2.8, "INT6": -2.0, "INT5": -1.2, "INT4": -0.2}
        cls.f = planted(3, n_per_lang=150, beta=beta, base=base); cls.res = am.analyse(cls.f, n_boot=100, seed=0)

    def test_there_is_a_block_per_precision_with_the_overall_loss_the_bins_the_auroc_and_the_logit(self):
        self.assertEqual(sorted(self.res["by_precision"]), sorted(am.PRECISIONS)); b = self.res["by_precision"]["INT4"]
        self.assertAlmostEqual(b["loss_overall"], float(self.f["lost"]["INT4"].mean())); self.assertEqual(len(b["bins_m0"]), 5); self.assertEqual(sum(x["n"] for x in b["bins_m0"]), len(self.f["language"]))
        self.assertEqual(sorted(b["auroc"]), ["a0", "m0", "r0", "s0"]); self.assertEqual(sorted(b["logit"]["odds_ratios"]), ["a0", "m0", "s0"])

    def test_the_planted_dependence_shows_in_the_bins_the_auroc_and_the_odds_ratio_of_the_margin(self):
        b = self.res["by_precision"]["INT4"]; fr = [x["fraction"] for x in b["bins_m0"]]; self.assertGreater(fr[0], fr[-1] + 0.2); self.assertGreater(b["auroc"]["m0"]["mean"], 0.7)
        self.assertAlmostEqual(np.log(b["logit"]["odds_ratios"]["m0"]["or"]), -1.6, delta=0.3); self.assertLess(b["logit"]["odds_ratios"]["m0"]["or"], 0.5); self.assertLess(abs(np.log(b["logit"]["odds_ratios"]["s0"]["or"])), 0.3)                       # a higher margin lowers the odds of loss; the share has no planted effect
        fl = self.res["by_precision"]["INT16"]; self.assertLess(fl["loss_overall"], b["loss_overall"])

    def test_the_english_excluded_cohort_has_fewer_features_and_the_excluded_languages_are_named(self):
        r = am.analyse(self.f, n_boot=50, seed=0, exclude=("aaa",)); self.assertEqual(r["excluded"], ["aaa"]); self.assertEqual(r["n_established"], len(self.f["language"]) - 150); self.assertEqual(len(r["languages"]), 5)

    def test_the_hypothesis_verdicts_follow_the_declared_rules(self):
        h = am.hypotheses(self.res); self.assertEqual(h["H-A2a"]["verdict"], "met"); self.assertTrue(h["H-A2b"]["verdict"] in ("exceeds the floor in every quintile", "at the floor in the top quintile")); self.assertEqual(h["H-A2c"]["verdict"], "met")
        null = am.analyse(planted(4, n_per_lang=150), n_boot=50, seed=0); self.assertEqual(am.hypotheses(null)["H-A2a"]["verdict"], "not met")

    def test_the_floor_comparison_uses_the_same_quintile_at_int16_and_int4(self):
        h = am.hypotheses(self.res)["H-A2b"]; top = self.res["by_precision"]["INT4"]["bins_m0"][-1]["fraction"]; floor_top = self.res["by_precision"]["INT16"]["bins_m0"][-1]["fraction"]
        self.assertAlmostEqual(h["int4_top"], top); self.assertAlmostEqual(h["floor_top"], floor_top)


class Output(unittest.TestCase):
    def test_the_command_line_writes_json_csv_and_a_report_with_the_status_words_and_the_floor_and_the_excluded_cohort(self):
        f = planted(5, n_per_lang=60, beta={p: 1.0 for p in am.PRECISIONS}); rows = []
        for k in range(len(f["language"])):
            st = {p: ("different_peak" if f["lost"][p][k] else "retained") for p in am.PRECISIONS}; rows += feature_rows(k, str(f["language"][k]), "retained", st, margin=float(f["m0"][k]), rate=float(f["m0"][k] / f["r0"][k]), share=float(f["s0"][k]), mean=float(f["a0"][k]))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "feature_status.csv"; write_csv(p, rows); out = Path(d) / "o"
            am.main(["--status", str(p), "--out", str(out), "--n-boot", "40", "--exclude", "aaa"])
            j = json.loads((out / "analysis.json").read_text()); self.assertEqual(j["all"]["n_established"], 360); self.assertEqual(j["without_excluded"]["excluded"], ["aaa"]); self.assertIn("INT4", j["all"]["by_precision"]); self.assertIn("hypotheses", j)
            md = (out / "report.md").read_text(); self.assertIn("Exploratory", md); self.assertIn("H-A2a", md); self.assertIn("INT16", md); self.assertIn("without aaa", md); self.assertIn("regression to the mean", md)
            with open(out / "loss_by_margin_quintile.csv", newline="") as fh:
                rows_csv = list(csv.DictReader(fh))
            self.assertEqual(len(rows_csv), 7 * 5); self.assertEqual(set(rows_csv[0]), {"precision", "quintile", "n", "lost", "fraction", "lo", "hi"}); self.assertEqual(sorted({int(r["quintile"]) for r in rows_csv}), [1, 2, 3, 4, 5]); self.assertEqual(sum(int(r["n"]) for r in rows_csv if r["precision"] == "INT4"), 360)


if __name__ == "__main__":
    unittest.main()
