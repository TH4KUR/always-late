import copy
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_assoc_function_link as fl
import phase1_assoc_margin as am
from test_phase1_assoc_margin import feature_rows, planted, write_csv

PRECS = ("INT8", "INT6", "INT5", "INT4")


def world(seed=0, n_per_lang=120, assoc_beta=1.8, w1_beta=0.0):
    """Established features with a loss that depends on the margin (assoc_beta) and a write-side similarity that depends on it (w1_beta, in W1 units per standard deviation of log m0); the sample maps ids to languages."""
    beta = {p: 0.0 for p in am.PRECISIONS}; beta.update({"INT8": assoc_beta / 2, "INT6": assoc_beta, "INT5": assoc_beta, "INT4": assoc_beta}); base = {p: -2.0 for p in am.PRECISIONS}
    feats = planted(seed, n_per_lang=n_per_lang, beta=beta, base=base); rng = np.random.default_rng(seed + 100); n = len(feats["language"]); z = (np.log(feats["m0"]) - np.log(feats["m0"]).mean()) / np.log(feats["m0"]).std()
    mid = {"INT16": 0.997, "INT8": 0.99, "INT6": 0.97, "INT5": 0.94, "INT4": 0.84}; states = {}
    for s, m in mid.items():
        slope = 0.0 if s == "INT16" else w1_beta
        states[s] = {"write": {"cos": np.minimum(m + slope * z + rng.normal(0, 0.02, n), 1.0)}}
    sample = {"features": {str(int(i)): {"language": str(l)} for i, l in zip(feats["feature_id"], feats["language"])}}
    return feats, [int(i) for i in feats["feature_id"]], states, sample


class Join(unittest.TestCase):
    def test_tested_features_are_matched_to_established_ones_by_language_and_id_and_the_rest_is_counted(self):
        feats, wf, states, sample = world(0, n_per_lang=20)
        wf2 = wf + [9001, 9002, 9003]; sample["features"]["9001"] = {"language": "aaa"}; sample["features"]["9002"] = {"language": "zzz"}                       # 9001: not established; 9002: a language without features; 9003: not in the sample
        j = fl.join_features(feats, wf2, sample)
        self.assertEqual(j["n_tested"], len(wf2)); self.assertEqual(j["n_joined"], len(wf)); self.assertEqual(j["n_dropped"], 3); np.testing.assert_array_equal(j["positions"], np.arange(len(wf))); np.testing.assert_array_equal(j["rows"], np.arange(len(wf)))

    def test_the_positions_follow_the_order_of_the_tested_features_and_the_rows_those_of_the_status_arrays(self):
        feats, wf, states, sample = world(1, n_per_lang=20); order = np.random.default_rng(0).permutation(len(wf)); wf_shuffled = [wf[k] for k in order]
        j = fl.join_features(feats, wf_shuffled, sample); self.assertEqual(sorted(j["positions"].tolist()), list(range(len(wf))))
        for pos, row in zip(j["positions"], j["rows"]):
            self.assertEqual(wf_shuffled[pos], int(feats["feature_id"][row]))

    def test_the_same_id_in_another_language_is_not_a_match(self):
        feats, wf, states, sample = world(2, n_per_lang=10); other = [l for l in set(feats["language"]) if l != sample["features"][str(wf[0])]["language"]][0]; sample["features"][str(wf[0])]["language"] = other
        j = fl.join_features(feats, wf, sample); self.assertEqual(j["n_dropped"], 1)


class SpearmanAndOls(unittest.TestCase):
    def test_the_within_language_spearman_sees_a_planted_dependence_and_a_null(self):
        rng = np.random.default_rng(0); langs = np.repeat(np.array(list("abcdef")), 100); x = rng.normal(size=600); dep = -0.6 * x + rng.normal(size=600); nul = rng.normal(size=600)
        a = fl.within_language_spearman(x, dep, langs, n_boot=200, seed=0); b = fl.within_language_spearman(x, nul, langs, n_boot=200, seed=0)
        self.assertLess(a["mean"], -0.4); self.assertEqual(a["n_languages"], 6); self.assertLess(a["lo"], a["mean"]); self.assertGreater(a["hi"], a["mean"]); self.assertLess(abs(b["mean"]), 0.1)

    def test_languages_below_the_minimum_are_excluded_and_the_interval_is_the_percentile_interval_of_language_resamples(self):
        rng = np.random.default_rng(1); langs = np.concatenate([np.repeat(np.array(list("abcd")), 40), np.array(["tiny"] * 10)]); x = rng.normal(size=170); y = rng.normal(size=170)
        r = fl.within_language_spearman(x, y, langs, min_n=20, n_boot=300, seed=5); self.assertEqual(r["n_languages"], 4); self.assertNotIn("tiny", r["per_language"])
        vals = np.array([r["per_language"][k] for k in "abcd"]); means = vals[np.random.default_rng(5).integers(0, 4, (300, 4))].mean(axis=1)
        self.assertAlmostEqual(r["lo"], float(np.percentile(means, 2.5))); self.assertAlmostEqual(r["hi"], float(np.percentile(means, 97.5))); self.assertAlmostEqual(r["mean"], float(vals.mean()))

    def test_a_language_without_variation_has_no_correlation_and_is_not_counted(self):
        rng = np.random.default_rng(6); langs = np.repeat(np.array(["a", "b"]), 30); x = rng.normal(size=60); y = rng.normal(size=60); y[30:] = 0.5
        r = fl.within_language_spearman(x, y, langs, min_n=20, n_boot=20); self.assertEqual(r["n_languages"], 1); self.assertEqual(list(r["per_language"]), ["a"])

    def test_the_minimum_is_inclusive(self):
        rng = np.random.default_rng(2); langs = np.array(["a"] * 20); self.assertEqual(fl.within_language_spearman(rng.normal(size=20), rng.normal(size=20), langs, min_n=20, n_boot=10)["n_languages"], 1)
        self.assertEqual(fl.within_language_spearman(rng.normal(size=20), rng.normal(size=20), langs, min_n=21, n_boot=10)["n_languages"], 0)

    def test_a_planted_slope_is_recovered_by_the_fixed_effects_regression_with_cluster_robust_intervals(self):
        rng = np.random.default_rng(3); n_l, n = 8, 150; lang = np.repeat(np.array([f"l{i}" for i in range(n_l)]), n); x1 = rng.normal(size=n_l * n); x2 = rng.normal(size=n_l * n); off = np.repeat(rng.normal(0, 0.3, n_l), n)
        y = 0.9 + off - 0.05 * x1 + rng.normal(0, 0.05, n_l * n); r = fl.fixed_effects_ols(y, np.column_stack([x1, x2]), lang, ["x1", "x2"]); c = r["coefficients"]
        self.assertAlmostEqual(c["x1"]["coef"], -0.05, delta=0.01); self.assertLess(c["x1"]["p"], 1e-4); self.assertLess(abs(c["x2"]["coef"]), 0.01); self.assertLess(c["x1"]["lo"], c["x1"]["coef"]); self.assertGreater(c["x1"]["hi"], c["x1"]["coef"]); self.assertEqual(r["n_languages"], n_l)

    def test_a_language_with_a_single_row_is_not_given_a_fixed_effect(self):
        rng = np.random.default_rng(7); lang = np.concatenate([np.repeat(np.array(["a", "b"]), 50), np.array(["lonely"])]); x = rng.normal(size=101); y = rng.normal(size=101)
        r = fl.fixed_effects_ols(y, x[:, None], lang, ["x"]); self.assertEqual(r["n_languages"], 2); self.assertEqual(r["n"], 100)

    def test_the_standard_errors_are_cluster_robust_a_slope_that_differs_between_languages_widens_the_interval(self):
        rng = np.random.default_rng(8); n_l, n = 8, 200; lang = np.repeat(np.array([f"l{i}" for i in range(n_l)]), n); x = rng.normal(size=n_l * n); slopes = np.repeat(rng.normal(0.05, 0.04, n_l), n)
        y = slopes * x + rng.normal(0, 0.05, n_l * n); c = fl.fixed_effects_ols(y, x[:, None], lang, ["x"])["coefficients"]["x"]
        self.assertGreater((c["hi"] - c["lo"]) / (2 * 1.96), 0.008)                                                                                              # the naive standard error would be about 0.0016

    def test_rows_with_a_missing_value_are_left_out_and_counted(self):
        rng = np.random.default_rng(4); n = 100; lang = np.repeat(np.array(["a", "b"]), n); x = rng.normal(size=2 * n); y = rng.normal(size=2 * n); x[:4] = np.nan; y[5] = np.nan
        r = fl.fixed_effects_ols(y, x[:, None], lang, ["x"]); self.assertEqual(r["n"], 2 * n - 5); self.assertEqual(r["n_missing"], 5)


class Analysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.feats, cls.wf, cls.states, cls.sample = world(3, n_per_lang=150)
        cls.res = fl.analyse(cls.feats, cls.wf, cls.states, cls.sample, n_boot=60, seed=0)

    def test_there_is_a_block_per_precision_present_in_the_states_with_the_counts(self):
        self.assertEqual(sorted(self.res["by_precision"]), sorted(PRECS)); b = self.res["by_precision"]["INT6"]; self.assertEqual(self.res["n_joined"], 900); self.assertEqual(b["n"], 900)
        self.assertEqual(sorted(b), sorted(["n", "n_events", "event_share", "spearman", "auroc_association", "auroc_function", "ols"])); self.assertEqual(sorted(b["ols"]["coefficients"]), ["a0", "m0", "s0"])

    def test_the_floor_is_the_fifth_percentile_of_w1_at_int16_over_all_tested_features(self):
        self.assertAlmostEqual(self.res["floor"], float(np.percentile(self.states["INT16"]["write"]["cos"], 5)))
        s2 = copy.deepcopy(self.states); k = 50; wf2 = self.wf[:-k]                                                                                                 # the last features are untested: the floor must still be over what is passed as tested
        s2["INT16"]["write"]["cos"] = s2["INT16"]["write"]["cos"][:-k]
        for p in PRECS:
            s2[p]["write"]["cos"] = s2[p]["write"]["cos"][:-k]
        r2 = fl.analyse(self.feats, wf2, s2, self.sample, n_boot=20, seed=0); self.assertAlmostEqual(r2["floor"], float(np.percentile(s2["INT16"]["write"]["cos"], 5)))

    def test_the_events_are_w1_below_the_floor(self):
        b = self.res["by_precision"]["INT5"]; w1 = self.states["INT5"]["write"]["cos"]; self.assertEqual(b["n_events"], int((w1 < self.res["floor"]).sum())); self.assertAlmostEqual(b["event_share"], b["n_events"] / 900)

    def test_a_margin_that_drives_the_association_loss_but_not_the_function_gives_all_three_verdicts(self):
        b = self.res["by_precision"]["INT6"]; self.assertGreater(b["auroc_association"]["mean"], 0.7); self.assertLess(abs(b["auroc_function"]["mean"] - 0.5), 0.1); self.assertLess(abs(b["spearman"]["mean"]), 0.1)
        h = fl.hypotheses(self.res); self.assertEqual({k: v["verdict"] for k, v in h.items()}, {"H-A3a": "met", "H-A3b": "met", "H-A3c": "met"})

    def test_a_margin_that_also_drives_the_function_fails_the_first_hypothesis(self):
        f, wf, st, sm = world(4, n_per_lang=150, w1_beta=0.03); r = fl.analyse(f, wf, st, sm, n_boot=60, seed=0); h = fl.hypotheses(r)
        self.assertGreater(r["by_precision"]["INT5"]["spearman"]["mean"], 0.15); self.assertEqual(h["H-A3a"]["verdict"], "not met"); self.assertGreater(r["by_precision"]["INT5"]["auroc_function"]["mean"], 0.55)       # the lower the margin the more events

    def test_the_regression_slope_of_w1_on_the_margin_is_planted_and_recovered(self):
        f, wf, st, sm = world(5, n_per_lang=150, w1_beta=0.03); r = fl.analyse(f, wf, st, sm, n_boot=20, seed=0); c = r["by_precision"]["INT6"]["ols"]["coefficients"]["m0"]; self.assertAlmostEqual(c["coef"], 0.03, delta=0.008)

    def test_a_w1_exactly_at_the_floor_is_not_an_event(self):
        feats = planted(9, n_per_lang=21, langs=("aaa",)); wf = [int(i) for i in feats["feature_id"]]; sample = {"features": {str(i): {"language": "aaa"} for i in wf}}
        c16 = np.sort(np.random.default_rng(0).uniform(0.9, 1.0, 21)); tau = c16[1]                                                                              # the 5th percentile of 21 sorted values is the second one
        order = np.random.default_rng(1).permutation(21); c16 = c16[order]; c5 = c16.copy()
        r = fl.analyse(feats, wf, {"INT16": {"write": {"cos": c16}}, "INT5": {"write": {"cos": c5}}}, sample, precisions=("INT5",), min_n=5, n_boot=10)
        self.assertAlmostEqual(r["floor"], tau); self.assertEqual(r["by_precision"]["INT5"]["n_events"], 1)

    def test_the_floor_is_over_all_tested_features_not_only_the_joined_ones(self):
        feats, wf, states, sample = world(10, n_per_lang=40); extra = 60; wf2 = wf + [100000 + k for k in range(extra)]                                            # tested but not established: dropped from the join
        s2 = copy.deepcopy(self.states) if False else {k: {"write": {"cos": np.concatenate([v["write"]["cos"], np.full(extra, 0.2 if k == "INT16" else 0.9)])}} for k, v in states.items()}
        r = fl.analyse(feats, wf2, s2, sample, n_boot=10, seed=0); self.assertAlmostEqual(r["floor"], float(np.percentile(s2["INT16"]["write"]["cos"], 5))); self.assertEqual(r["n_dropped"], extra); self.assertEqual(r["n_joined"], len(wf))

    def test_a_precision_missing_from_the_states_is_skipped(self):
        s2 = {k: v for k, v in self.states.items() if k != "INT5"}; r = fl.analyse(self.feats, self.wf, s2, self.sample, n_boot=20, seed=0); self.assertNotIn("INT5", r["by_precision"])

    def test_the_floor_state_is_required(self):
        s2 = {k: v for k, v in self.states.items() if k != "INT16"}
        with self.assertRaises(ValueError):
            fl.analyse(self.feats, self.wf, s2, self.sample, n_boot=20, seed=0)

    def test_the_excluded_languages_are_left_out_and_named(self):
        r = fl.analyse(self.feats, self.wf, self.states, self.sample, n_boot=20, seed=0, exclude=("aaa",)); self.assertEqual(r["excluded"], ["aaa"]); self.assertEqual(r["n_joined"], 900 - 150); self.assertEqual(r["by_precision"]["INT6"]["n"], 750)


def fake(sp, aa, af):
    """A hand-made result for the verdict rules: sp, aa, af = {precision: value} (the mean Spearman, the AUROC for the association loss and for the function event)."""
    mk = lambda p: {"spearman": {"mean": sp[p]}, "auroc_association": {"mean": aa.get(p, 0.5)}, "auroc_function": {"mean": af.get(p, 0.5)}}
    return {"by_precision": {p: mk(p) for p in ("INT8", "INT6", "INT5", "INT4")}}


class Verdicts(unittest.TestCase):
    SP = {"INT8": 0.0, "INT6": 0.1, "INT5": -0.1, "INT4": 0.14}

    def test_h_a3a_needs_the_correlation_to_stay_below_015_in_absolute_value_at_three_precisions(self):
        self.assertEqual(fl.hypotheses(fake(self.SP, {"INT6": 0.8}, {}))["H-A3a"]["verdict"], "met")
        for p, v in (("INT6", 0.15), ("INT5", -0.15), ("INT4", 0.2)):
            self.assertEqual(fl.hypotheses(fake({**self.SP, p: v}, {"INT6": 0.8}, {}))["H-A3a"]["verdict"], "not met", p)

    def test_h_a3b_needs_the_association_auroc_at_int6_at_least_070_and_the_function_auroc_at_most_060_at_int6_and_int5(self):
        ok = fl.hypotheses(fake(self.SP, {"INT6": 0.70, "INT5": 0.8}, {"INT6": 0.60, "INT5": 0.60}))["H-A3b"]["verdict"]; self.assertEqual(ok, "met")
        self.assertEqual(fl.hypotheses(fake(self.SP, {"INT6": 0.699}, {"INT6": 0.5, "INT5": 0.5}))["H-A3b"]["verdict"], "not met")
        self.assertEqual(fl.hypotheses(fake(self.SP, {"INT6": 0.8}, {"INT6": 0.601, "INT5": 0.5}))["H-A3b"]["verdict"], "not met"); self.assertEqual(fl.hypotheses(fake(self.SP, {"INT6": 0.8}, {"INT6": 0.5, "INT5": 0.601}))["H-A3b"]["verdict"], "not met")

    def test_h_a3c_needs_a_difference_of_at_least_015_at_int6_and_int5(self):
        self.assertEqual(fl.hypotheses(fake(self.SP, {"INT6": 0.80, "INT5": 0.75}, {"INT6": 0.55, "INT5": 0.58}))["H-A3c"]["verdict"], "met")                       # 0.25 and 0.17
        self.assertEqual(fl.hypotheses(fake(self.SP, {"INT6": 0.80, "INT5": 0.70}, {"INT6": 0.55, "INT5": 0.56}))["H-A3c"]["verdict"], "not met")                  # INT5: 0.14
        self.assertEqual(fl.hypotheses(fake(self.SP, {"INT6": 0.70, "INT5": 0.80}, {"INT6": 0.56, "INT5": 0.55}))["H-A3c"]["verdict"], "not met")                  # INT6: 0.14
        self.assertEqual(fl.hypotheses(fake(self.SP, {"INT6": 0.80, "INT5": 0.80}, {"INT6": 0.65, "INT5": 0.5}))["H-A3c"]["verdict"], "met")                       # INT6: exactly 0.15 passes (the rule is at least 0.15)
        self.assertEqual(fl.hypotheses(fake(self.SP, {"INT6": 0.80, "INT5": 0.80}, {"INT6": 0.651, "INT5": 0.5}))["H-A3c"]["verdict"], "not met")                  # 0.149 does not
        self.assertEqual(fl.hypotheses(fake(self.SP, {"INT6": 0.70, "INT5": 0.80}, {"INT6": 0.55, "INT5": 0.5}))["H-A3c"]["verdict"], "met")                       # 0.70 - 0.55 is 0.15 on paper and 0.14999999999999991 in floating point

    def test_a_missing_value_gives_not_evaluable(self):
        sp = dict(self.SP); sp["INT4"] = float("nan"); self.assertEqual(fl.hypotheses(fake(sp, {"INT6": 0.8}, {}))["H-A3a"]["verdict"], "not evaluable")


class Output(unittest.TestCase):
    def test_the_command_line_writes_json_csv_and_a_report_from_run_directories_the_sample_and_the_status_file(self):
        from test_phase1_func_f5_analysis import make_world, save_run
        rng = np.random.default_rng(0)
        cos = {s: (lambda r, n, m=m: np.minimum(m + r.normal(0, 0.02, n), 1.0)) for s, m in {"INT16": 0.997, "INT8": 0.99, "INT6": 0.97, "INT5": 0.94, "INT4": 0.84}.items()}
        w = make_world(per_lang=40, n_excluded_per_lang=0, cos_by_state=cos); sa, st, sm = w[:3]
        with tempfile.TemporaryDirectory() as d:
            a, b, o = Path(d) / "a", Path(d) / "b", Path(d) / "o"; save_run(w, a, ["FP", "INT16", "INT8"]); save_run(w, b, ["FP", "INT6", "INT5", "INT4"])
            rows = []
            for fid, f in sm["features"].items():
                m0 = float(np.exp(rng.normal(-6, 1))); lost = {p: bool(rng.random() < 1 / (1 + np.exp(-(-2 - 1.5 * (np.log(m0) + 6))))) for p in am.PRECISIONS}
                rows += feature_rows(int(fid), f["language"], "retained", {p: ("different_peak" if lost[p] else "retained") for p in am.PRECISIONS}, margin=m0, rate=m0 * 3, share=0.2, mean=0.05)
            status = Path(d) / "feature_status.csv"; write_csv(status, rows); sample = Path(d) / "sample.json"; sample.write_text(json.dumps(sm))
            fl.main(["--status", str(status), "--run", str(a), "--run", str(b), "--sample", str(sample), "--out", str(o), "--n-boot", "30", "--exclude", sm["languages"][0], "--min-n", "20"])
            j = json.loads((o / "analysis.json").read_text()); self.assertEqual(j["all"]["n_joined"], len(sm["features"])); self.assertEqual(j["without_excluded"]["excluded"], [sm["languages"][0]]); self.assertIn("hypotheses", j); self.assertIn("INT4", j["all"]["by_precision"])
            md = (o / "report.md").read_text(); self.assertIn("Exploratory", md); self.assertIn("H-A3a", md); self.assertIn("H-A3c", md); self.assertIn("stratified", md); self.assertIn("without " + sm["languages"][0], md)
            with open(o / "margin_vs_function.csv", newline="") as fh:
                rows_csv = list(csv.DictReader(fh))
            self.assertEqual(len(rows_csv), 4); self.assertEqual([r["precision"] for r in rows_csv], ["INT8", "INT6", "INT5", "INT4"]); self.assertIn("auroc_function", rows_csv[0])


if __name__ == "__main__":
    unittest.main()
