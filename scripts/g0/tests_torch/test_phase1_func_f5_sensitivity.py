import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_func_f5_sensitivity as sn


def run(ids, cos_by_state, n_extra=0):
    """A minimal run: stage A with the tested feature ids and the W1 of each state."""
    ids = list(ids)
    return {"stage": {"write_features": ids, "write_rows": list(range(len(ids))), "feat_ids": np.array(ids)},
            "states": {s: {"write": {"cos": np.asarray(c, dtype=float), "retrieval": {"top1": 1.0}}} for s, c in cos_by_state.items()}}


class Alignment(unittest.TestCase):
    def test_features_are_aligned_by_id_and_in_ascending_id_order(self):
        a = run([5, 1, 9, 3], {"INT16": [0.5, 0.1, 0.9, 0.3]}); b = run([9, 5, 7], {"INT16": [0.91, 0.52, 0.7]})
        ia, ib = sn.align(a["stage"], b["stage"])
        self.assertEqual([a["stage"]["write_features"][i] for i in ia], [5, 9]); self.assertEqual([b["stage"]["write_features"][i] for i in ib], [5, 9])

    def test_no_common_feature_is_an_error(self):
        with self.assertRaises(ValueError):
            sn.align(run([1, 2], {"INT16": [0.1, 0.2]})["stage"], run([3], {"INT16": [0.3]})["stage"])


class CompareState(unittest.TestCase):
    def test_the_summary_of_two_measurements_of_the_same_features(self):
        a = np.array([0.99, 0.95, 0.90, 0.80, 0.70]); b = np.array([0.99, 0.96, 0.85, 0.88, 0.60])
        c = sn.compare_state(a, b, tau_a=0.92, tau_b=0.92)
        self.assertEqual(c["n"], 5); self.assertAlmostEqual(c["median_a"], 0.90); self.assertAlmostEqual(c["median_b"], 0.88); self.assertAlmostEqual(c["median_diff"], float(np.median(b - a)))
        self.assertAlmostEqual(c["mean_abs_diff"], float(np.mean(np.abs(b - a)))); self.assertAlmostEqual(c["spearman"], 1.0 - 6 * ((0 ** 2 + 0 ** 2 + 0 ** 2 + 1 ** 2 + 1 ** 2)) / (5 * 24))
        self.assertAlmostEqual(c["above_a"], 0.4); self.assertAlmostEqual(c["above_b"], 0.4); self.assertAlmostEqual(c["flip_share"], 0.0)

    def test_the_classification_can_flip_and_the_two_floors_may_differ(self):
        a = np.array([0.95, 0.90, 0.85, 0.99]); b = np.array([0.93, 0.91, 0.80, 0.99])
        c = sn.compare_state(a, b, tau_a=0.92, tau_b=0.90)
        self.assertAlmostEqual(c["above_a"], 0.5); self.assertAlmostEqual(c["above_b"], 0.75); self.assertAlmostEqual(c["flip_share"], 0.25)            # feature 2: below 0.92 in a, at or above 0.90 in b

    def test_a_value_exactly_at_the_floor_counts_as_at_or_above(self):
        c = sn.compare_state(np.array([0.92, 0.5]), np.array([0.90, 0.5]), tau_a=0.92, tau_b=0.90); self.assertAlmostEqual(c["above_a"], 0.5); self.assertAlmostEqual(c["above_b"], 0.5); self.assertEqual(c["flip_share"], 0.0)

    def test_undefined_values_are_left_out_and_counted(self):
        a = np.array([0.9, np.nan, 0.8, 0.7]); b = np.array([0.9, 0.5, np.nan, 0.6]); c = sn.compare_state(a, b, 0.5, 0.5)
        self.assertEqual(c["n"], 2); self.assertEqual(c["n_undefined"], 2)

    def test_identical_measurements_have_no_difference_and_correlation_one(self):
        a = np.array([0.5, 0.6, 0.7, 0.9]); c = sn.compare_state(a, a.copy(), 0.6, 0.6)
        self.assertEqual(c["mean_abs_diff"], 0.0); self.assertAlmostEqual(c["spearman"], 1.0); self.assertEqual(c["flip_share"], 0.0)


class CompareRuns(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0); ids = list(range(100, 300)); self.main = run(ids, {"INT16": 1 - np.abs(rng.normal(0, 0.03, 200)), "INT4": 0.8 + rng.normal(0, 0.05, 200)})
        sub = ids[::4]; k = [ids.index(i) for i in sub]
        self.sens = run(sub, {"INT16": self.main["states"]["INT16"]["write"]["cos"][k] - 0.002, "INT4": self.main["states"]["INT4"]["write"]["cos"][k] + 0.01}); self.k = k

    def test_the_comparison_uses_the_common_features_and_the_floor_of_each_run(self):
        c = sn.compare_runs(self.main, self.sens, ["INT16", "INT4"], q=0.05)
        self.assertEqual(c["n_common"], 50); m16 = self.main["states"]["INT16"]["write"]["cos"][self.k]; s16 = self.sens["states"]["INT16"]["write"]["cos"]
        self.assertAlmostEqual(c["floor"]["tau_a"], float(np.percentile(m16, 5))); self.assertAlmostEqual(c["floor"]["tau_b"], float(np.percentile(s16, 5)))
        self.assertAlmostEqual(c["states"]["INT4"]["median_diff"], 0.01, places=9); self.assertAlmostEqual(c["states"]["INT16"]["median_diff"], -0.002, places=9)
        self.assertAlmostEqual(c["states"]["INT4"]["spearman"], 1.0)

    def test_the_shares_at_or_above_the_floor_use_each_runs_own_floor(self):
        rng = np.random.default_rng(5); ids = list(range(50)); a16 = rng.uniform(0.90, 1.0, 50); a4 = rng.uniform(0.6, 1.0, 50)
        a = run(ids, {"INT16": a16, "INT4": a4}); b = run(ids, {"INT16": a16 - 0.05, "INT4": a4 - 0.03}); c = sn.compare_runs(a, b, ["INT4"]); ta, tb = c["floor"]["tau_a"], c["floor"]["tau_b"]
        self.assertNotAlmostEqual(ta, tb, places=3); self.assertAlmostEqual(c["states"]["INT4"]["above_a"], float((a4 >= ta).mean())); self.assertAlmostEqual(c["states"]["INT4"]["above_b"], float(((a4 - 0.03) >= tb).mean()))

    def test_a_state_missing_in_one_run_is_skipped_and_listed(self):
        del self.sens["states"]["INT4"]; c = sn.compare_runs(self.main, self.sens, ["INT16", "INT4"]); self.assertEqual(sorted(c["states"]), ["INT16"]); self.assertEqual(c["missing"], ["INT4"])

    def test_the_floor_state_is_required(self):
        del self.sens["states"]["INT16"]
        with self.assertRaises(ValueError):
            sn.compare_runs(self.main, self.sens, ["INT4"], floor_state="INT16")

    def test_the_report_names_the_run_and_every_number(self):
        c = sn.compare_runs(self.main, self.sens, ["INT16", "INT4"]); md = sn.report_text(c, "M = 8 against M = 16")
        self.assertIn("M = 8 against M = 16", md); self.assertIn("| INT4 |", md); self.assertIn(f"{c['states']['INT4']['spearman']:.3f}", md); self.assertIn(f"{c['floor']['tau_a']:.4f}", md); self.assertIn("exploratory", md)


class PairArrays(unittest.TestCase):
    def test_the_aligned_arrays_the_floors_and_the_count_for_the_figure(self):
        rng = np.random.default_rng(0); ids = list(range(100, 160)); main = run(ids, {"INT16": 1 - np.abs(rng.normal(0, 0.03, 60)), "INT4": 0.8 + rng.normal(0, 0.05, 60)}); sub = ids[::3]; k = [ids.index(i) for i in sub]
        sens = run(sub[::-1], {"INT16": main["states"]["INT16"]["write"]["cos"][k][::-1] - 0.001, "INT4": main["states"]["INT4"]["write"]["cos"][k][::-1] + 0.02})                         # another order of the rows
        pa = sn.pair_arrays(main, sens, ["INT16", "INT4"]); self.assertEqual(pa["n_common"], 20); np.testing.assert_allclose(pa["states"]["INT4"]["b"] - pa["states"]["INT4"]["a"], 0.02, atol=1e-12)
        np.testing.assert_allclose(pa["states"]["INT4"]["a"], main["states"]["INT4"]["write"]["cos"][k]); self.assertEqual(sorted(pa["states"]), ["INT16", "INT4"])
        c = sn.compare_runs(main, sens, ["INT16", "INT4"]); self.assertAlmostEqual(pa["floor"]["tau_a"], c["floor"]["tau_a"]); self.assertAlmostEqual(pa["floor"]["tau_b"], c["floor"]["tau_b"])

    def test_a_state_missing_in_one_run_is_not_in_the_arrays(self):
        a = run([1, 2, 3], {"INT16": [0.9, 0.95, 0.99], "INT4": [0.5, 0.6, 0.7]}); b = run([1, 2, 3], {"INT16": [0.9, 0.95, 0.99]}); self.assertEqual(sorted(sn.pair_arrays(a, b, ["INT16", "INT4"])["states"]), ["INT16"])


class Main(unittest.TestCase):
    def test_the_command_line_writes_json_and_markdown(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import test_phase1_func_f5_analysis as ta
        w = ta.build_world("sensible", per_lang=20)
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b, tempfile.TemporaryDirectory() as o:
            ta.save_run(w, a, ["FP", "INT16", "INT4"]); ta.save_run(w, b, ["FP", "INT16", "INT4"])
            sn.main(["--main-run", a, "--sens-run", b, "--label", "same run twice", "--states", "INT16", "INT4", "--out", str(Path(o) / "r")])
            j = json.loads((Path(o) / "r" / "sensitivity.json").read_text()); self.assertEqual(j["n_common"], len(w[0]["write_features"])); self.assertEqual(j["states"]["INT4"]["flip_share"], 0.0)
            self.assertTrue((Path(o) / "r" / "sensitivity.md").read_text().startswith("# F5 sensitivity"))
            sn.main(["--main-run", a, "--sens-run", b, "--label", "same run twice", "--states", "INT16", "INT4", "--out", str(Path(o) / "r2"), "--figures", "--label-a", "run one", "--label-b", "run two", "--figure-note", "exploratory, test"])
            self.assertEqual(sorted(p.name for p in (Path(o) / "r2").iterdir() if p.name.startswith("pair_")), ["pair_dark.png", "pair_dark.svg", "pair_light.png", "pair_light.svg"])
            self.assertFalse(any(p.name.startswith("pair_") for p in (Path(o) / "r").iterdir()))                                                      # no figure unless asked


class LadderArrays(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0); self.ids = list(range(100, 160)); n = 60; self.sub = self.ids[::3]; self.k = [self.ids.index(i) for i in self.sub]
        self.main = run(self.ids, {"INT16": 1 - np.abs(rng.normal(0, 0.02, n)), "INT8": 0.99 - np.abs(rng.normal(0, 0.02, n)), "INT5": 0.95 + rng.normal(0, 0.02, n), "INT4": 0.8 + rng.normal(0, 0.05, n)})
        m = lambda s: self.main["states"][s]["write"]["cos"][self.k]
        self.c1 = run(self.sub, {"INT16": m("INT16") - 0.004, "INT8": m("INT8") - 0.004, "INT5": m("INT5") - 0.01})                                  # the control run with the high precisions
        self.c2 = run(self.sub[::-1], {"INT16": (m("INT16") - 0.0041)[::-1], "INT4": (m("INT4") - 0.012)[::-1]})                                         # the control run with INT4, another row order

    def test_every_state_comes_from_the_first_control_run_that_has_it_on_the_features_common_to_all_runs(self):
        la = sn.ladder_arrays(self.main, [self.c1, self.c2], ["INT16", "INT8", "INT5", "INT4"])
        self.assertEqual(la["n_common"], len(self.sub)); self.assertEqual(list(la["states"]), ["INT16", "INT8", "INT5", "INT4"]); self.assertEqual(la["missing"], [])
        m = lambda s: self.main["states"][s]["write"]["cos"][self.k]
        np.testing.assert_allclose(la["states"]["INT8"]["a"], m("INT8")); np.testing.assert_allclose(la["states"]["INT8"]["b"], m("INT8") - 0.004)
        np.testing.assert_allclose(la["states"]["INT4"]["a"], m("INT4")); np.testing.assert_allclose(la["states"]["INT4"]["b"], m("INT4") - 0.012)              # aligned by id although c2 has the rows reversed
        np.testing.assert_allclose(la["states"]["INT16"]["b"], m("INT16") - 0.004)                                                                              # INT16 from the FIRST control run

    def test_the_floors_are_those_of_the_features_and_of_the_first_control_run_at_int16(self):
        la = sn.ladder_arrays(self.main, [self.c1, self.c2], ["INT16", "INT4"]); m16 = self.main["states"]["INT16"]["write"]["cos"][self.k]
        self.assertAlmostEqual(la["floor"]["tau_a"], float(np.percentile(m16, 5))); self.assertAlmostEqual(la["floor"]["tau_b"], float(np.percentile(m16 - 0.004, 5)))

    def test_a_state_that_no_control_run_has_is_listed_as_missing(self):
        la = sn.ladder_arrays(self.main, [self.c1, self.c2], ["INT16", "INT6", "INT4"]); self.assertEqual(list(la["states"]), ["INT16", "INT4"]); self.assertEqual(la["missing"], ["INT6"])

    def test_control_runs_that_do_not_test_the_same_features_are_an_error(self):
        other = run(self.sub[:-2], {"INT16": self.c2["states"]["INT16"]["write"]["cos"][2:], "INT4": self.c2["states"]["INT4"]["write"]["cos"][2:]})
        with self.assertRaises(ValueError):
            sn.ladder_arrays(self.main, [self.c1, other], ["INT16", "INT4"])

    def test_control_runs_with_the_same_number_of_features_but_other_ones_are_an_error(self):
        ids = self.ids[1::3]; self.assertEqual(len(ids), len(self.sub)); other = run(ids, {"INT16": np.linspace(0.9, 1.0, len(ids)), "INT4": np.linspace(0.7, 0.9, len(ids))})
        with self.assertRaises(ValueError):
            sn.ladder_arrays(self.main, [self.c1, other], ["INT16", "INT4"])

    def test_no_control_run_is_an_error(self):
        with self.assertRaises(ValueError):
            sn.ladder_arrays(self.main, [], ["INT16"])


class LadderStats(unittest.TestCase):
    def setUp(self):
        a = {"INT16": np.array([0.99, 0.995, 0.998, 1.0, 0.97]), "INT4": np.array([0.7, 0.8, 0.9, 0.85, 0.6])}; b = {"INT16": a["INT16"] - 0.004, "INT4": a["INT4"] - 0.02}
        self.la = {"n_common": 5, "q": 0.05, "floor_state": "INT16", "floor": {"tau_a": 0.98, "tau_b": 0.97}, "states": {s: {"a": a[s], "b": b[s]} for s in a}, "missing": []}; self.a, self.b = a, b

    def test_medians_quartiles_and_differences_of_each_state(self):
        s = sn.ladder_stats(self.la)["INT4"]
        self.assertEqual(s["n"], 5); self.assertAlmostEqual(s["median_a"], 0.8); self.assertAlmostEqual(s["median_b"], 0.78); self.assertAlmostEqual(s["q25_a"], float(np.percentile(self.a["INT4"], 25))); self.assertAlmostEqual(s["q75_b"], float(np.percentile(self.b["INT4"], 75)))
        self.assertAlmostEqual(s["q75_a"], float(np.percentile(self.a["INT4"], 75))); self.assertAlmostEqual(s["q25_b"], float(np.percentile(self.b["INT4"], 25)))
        self.assertAlmostEqual(s["median_diff"], -0.02); self.assertAlmostEqual(s["diff_of_medians"], -0.02)

    def test_the_shortfall_ratio_is_the_controls_shortfall_below_one_over_the_features(self):
        s = sn.ladder_stats(self.la)["INT4"]; self.assertAlmostEqual(s["shortfall_ratio"], (1 - 0.78) / (1 - 0.8))
        one = {**self.la, "states": {"INT16": {"a": np.ones(4), "b": np.array([1.0, 0.99, 1.0, 1.0])}}}; self.assertTrue(np.isnan(sn.ladder_stats(one)["INT16"]["shortfall_ratio"]))      # features exactly at 1: no ratio

    def test_the_shares_at_or_above_the_floor_use_each_runs_own_floor(self):
        s = sn.ladder_stats(self.la)["INT16"]; self.assertAlmostEqual(s["above_a"], float((self.a["INT16"] >= 0.98).mean())); self.assertAlmostEqual(s["above_b"], float((self.b["INT16"] >= 0.97).mean()))

    def test_a_value_exactly_at_a_floor_counts_as_at_or_above_it_in_each_run(self):
        la = {**self.la, "floor": {"tau_a": 0.98, "tau_b": 0.90}, "states": {"INT4": {"a": np.array([0.98, 0.5, 0.99, 0.97]), "b": np.array([0.90, 0.95, 0.5, 0.89])}}}; s = sn.ladder_stats(la)["INT4"]
        self.assertAlmostEqual(s["above_a"], 0.5); self.assertAlmostEqual(s["above_b"], 0.5)

    def test_undefined_pairs_are_left_out_and_counted_and_an_empty_state_is_all_nan(self):
        la = {**self.la, "states": {"INT4": {"a": np.array([0.8, np.nan, 0.7]), "b": np.array([0.7, 0.6, np.nan])}, "INT5": {"a": np.array([np.nan]), "b": np.array([0.5])}}}; st = sn.ladder_stats(la)
        self.assertEqual(st["INT4"]["n"], 1); self.assertEqual(st["INT4"]["n_undefined"], 2); self.assertEqual(st["INT5"]["n"], 0); self.assertTrue(np.isnan(st["INT5"]["median_a"]))


class LadderReport(unittest.TestCase):
    def test_the_table_has_a_row_per_state_the_floors_and_says_exploratory(self):
        a = {"INT16": np.array([0.99, 0.995, 0.998]), "INT4": np.array([0.7, 0.8, 0.9])}; la = {"n_common": 3, "q": 0.05, "floor_state": "INT16", "floor": {"tau_a": 0.98, "tau_b": 0.97}, "states": {s: {"a": v, "b": v - 0.01} for s, v in a.items()}, "missing": ["INT6"]}
        md = sn.ladder_report_text(la, "swapped directions against features"); self.assertIn("swapped directions against features", md); self.assertIn("exploratory", md); self.assertIn("| INT4 | 3 |", md); self.assertIn("| INT16 | 3 |", md)
        self.assertIn("0.9800 (features) and 0.9700 (control)", md); self.assertIn("INT6", md.split("Not compared")[1]); self.assertIn("0.8000", md)
        rows = sn.ladder_csv_rows(la); self.assertEqual([r["state"] for r in rows], ["INT16", "INT4"]); self.assertAlmostEqual(float(rows[1]["median_features"]), 0.8); self.assertAlmostEqual(float(rows[1]["median_control"]), 0.79)


if __name__ == "__main__":
    unittest.main()
