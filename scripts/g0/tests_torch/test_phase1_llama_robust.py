import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_llama_robust as lrb
from test_phase1_gain_analysis import planted_run, save_state

LANGS = ["a", "b", "c", "d", "e"]


def run_with(group, draws, state="FP", langs=LANGS, n_controls=3):
    """A Run in which language l has the zero-ablation group harm group[l] and the control draws draws[l] (zeroed controls; the scaled configurations are planted too, as the runner writes them)."""
    def harm(code, a, kind):
        if a == 0.0:
            return group[code] if kind == "g" else draws[code][int(kind[1:])]
        return 0.0
    return planted_run(state, harm, langs=langs, n_controls=n_controls)


G = {"a": 0.5, "b": 0.5, "c": 0.5, "d": 0.5, "e": 0.5}
D = {"a": [0.1, 0.2, 0.3], "b": [0.1, 0.2, 2.0], "c": [0.5, 0.1, 0.1], "d": [0.9, 1.0, 1.1], "e": [0.0, 0.0, 0.005]}


class Draws(unittest.TestCase):
    def test_the_harms_are_the_pooled_group_harm_and_the_pooled_harm_of_every_control_draw(self):
        g, d = lrb.draw_harms(run_with(G, D), "b")
        self.assertAlmostEqual(g, 0.5); np.testing.assert_allclose(d, [0.1, 0.2, 2.0])

    def test_five_draws_are_found_and_ordered_by_their_index(self):
        draws = {l: [0.1 * (k + 1) for k in range(5)] for l in LANGS}
        g, d = lrb.draw_harms(run_with(G, draws, n_controls=5), "a")
        np.testing.assert_allclose(d, [0.1, 0.2, 0.3, 0.4, 0.5])

    def test_eleven_draws_are_ordered_by_the_number_and_not_by_the_text_of_the_index(self):
        draws = {l: [0.01 * (k + 1) for k in range(11)] for l in LANGS}
        g, d = lrb.draw_harms(run_with(G, draws, n_controls=11), "a")
        np.testing.assert_allclose(d, [0.01 * (k + 1) for k in range(11)])

    def test_a_language_without_the_group_or_the_controls_is_an_error(self):
        r = run_with(G, D, langs=LANGS)
        with self.assertRaises(KeyError):
            lrb.draw_harms(r, "zzz")


class Outliers(unittest.TestCase):
    def test_a_draw_above_five_times_the_median_of_the_other_draws_is_flagged(self):
        self.assertEqual(lrb.outlier_flags([0.05, 0.06, 0.9]), [False, False, True])
        self.assertEqual(lrb.outlier_flags([0.05, 0.06, 0.20]), [False, False, False])                  # 5 x median(0.05, 0.06) = 0.275
        self.assertEqual(lrb.outlier_flags([3.72, 0.07, 0.198]), [True, False, False])

    def test_the_boundary_is_strict_and_a_floor_keeps_noise_from_being_flagged(self):
        self.assertEqual(lrb.outlier_flags([1.0, 0.2, 0.2]), [False, False, False])                       # exactly 5 x 0.2
        self.assertEqual(lrb.outlier_flags([1.01, 0.2, 0.2]), [True, False, False])
        self.assertEqual(lrb.outlier_flags([0.0, 0.0, 0.005]), [False, False, False])                     # the median of the others is 0: the floor of 0.01 gives 0.05
        self.assertEqual(lrb.outlier_flags([0.0, 0.0, 0.051]), [False, False, True])
        self.assertEqual(lrb.outlier_flags([-0.2, -0.1, 0.02]), [False, False, False])

    def test_the_median_of_the_others_not_their_mean_and_not_the_median_including_the_draw_decides(self):
        self.assertEqual(lrb.outlier_flags([0.0, 0.0, 0.0, 0.9, 0.5]), [False, False, False, True, True])      # the mean of the four others (0.225) would not flag 0.5; the median (0) does
        self.assertEqual(lrb.outlier_flags([0.01, 0.2, 0.9]), [False, False, True])                            # the median of the others is 0.105 (threshold 0.525); the median with the draw is 0.2 (threshold 1.0)

    def test_with_five_draws_the_median_of_the_four_others_is_used(self):
        self.assertEqual(lrb.outlier_flags([0.1, 0.1, 0.1, 0.1, 0.6]), [False, False, False, False, True])
        self.assertEqual(lrb.outlier_flags([0.1, 0.1, 0.1, 0.1, 0.5]), [False, False, False, False, False])


class Rows(unittest.TestCase):
    def setUp(self):
        self.rows = {r["language"]: r for r in lrb.language_rows(run_with(G, D), LANGS)}

    def test_e_with_the_mean_and_with_the_median_of_the_draws(self):
        self.assertAlmostEqual(self.rows["b"]["E_mean"], 0.5 - (0.1 + 0.2 + 2.0) / 3); self.assertAlmostEqual(self.rows["b"]["E_median"], 0.5 - 0.2)
        self.assertAlmostEqual(self.rows["a"]["E_mean"], 0.3); self.assertAlmostEqual(self.rows["a"]["E_median"], 0.3)

    def test_the_group_must_exceed_every_single_draw_strictly(self):
        self.assertTrue(self.rows["a"]["exceeds_all"]); self.assertFalse(self.rows["b"]["exceeds_all"]); self.assertFalse(self.rows["c"]["exceeds_all"])                 # c: a draw equal to the group harm
        self.assertFalse(self.rows["d"]["exceeds_all"])

    def test_the_outlier_draw_is_listed_and_e_without_it_uses_the_other_draws(self):
        b = self.rows["b"]
        self.assertEqual(b["outlier_draws"], [2]); self.assertAlmostEqual(b["E_without_outliers"], 0.5 - 0.15)
        a = self.rows["a"]
        self.assertEqual(a["outlier_draws"], []); self.assertAlmostEqual(a["E_without_outliers"], a["E_mean"])

    def test_without_an_outlier_the_value_is_the_plain_mean_and_the_median_ignores_the_outlier(self):
        c = self.rows["c"]
        self.assertAlmostEqual(c["E_mean"], 0.5 - 0.7 / 3); self.assertEqual(c["outlier_draws"], [])                  # 0.5 is exactly five times 0.1: not above it
        self.assertAlmostEqual(c["E_median"], 0.5 - 0.1)

    def test_all_draws_flagged_is_impossible_but_two_flagged_draws_leave_the_remaining_one(self):
        d = lrb.language_rows(run_with({**G, "a": 0.5}, {**D, "a": [0.01, 0.9, 0.95]}), ["a"])[0]
        self.assertEqual(sorted(d["outlier_draws"]), [])                                                      # the median of the other two is 0.475 or 0.92: nothing is above five times that
        d2 = lrb.language_rows(run_with(G, {**D, "a": [0.01, 0.01, 0.9]}), ["a"])[0]
        self.assertEqual(d2["outlier_draws"], [2])


class Summary(unittest.TestCase):
    def test_counts_over_the_tested_languages_only(self):
        runs = {"FP": run_with(G, D)}
        res = lrb.evaluate(runs, ["a", "b", "c", "d"])
        s = res["FP"]["summary"]
        self.assertEqual(s["n"], 4)
        self.assertEqual(s["E_mean_positive"], 2)                                                                   # a and c have a positive E with the mean of the draws; b (one draw of 2.0) and d do not
        self.assertEqual([r["language"] for r in res["FP"]["rows"]], ["a", "b", "c", "d"])

    def test_the_counts_and_medians_of_every_variant(self):
        res = lrb.evaluate({"FP": run_with(G, D)}, LANGS); s = res["FP"]["summary"]
        rows = res["FP"]["rows"]
        self.assertEqual(s["E_mean_positive"], sum(r["E_mean"] > 0 for r in rows)); self.assertEqual(s["E_median_positive"], sum(r["E_median"] > 0 for r in rows))
        self.assertEqual(s["exceeds_all_count"], sum(r["exceeds_all"] for r in rows)); self.assertEqual(s["E_without_outliers_positive"], sum(r["E_without_outliers"] > 0 for r in rows))
        self.assertEqual(s["languages_with_an_outlier_draw"], sum(bool(r["outlier_draws"]) for r in rows))
        self.assertAlmostEqual(s["E_mean_median"], float(np.median([r["E_mean"] for r in rows]))); self.assertAlmostEqual(s["E_median_median"], float(np.median([r["E_median"] for r in rows])))

    def test_a_negative_e_that_comes_from_one_draw_turns_positive_with_the_median(self):
        g = {"a": 0.1}; d = {"a": [0.07, 0.2, 3.7]}
        row = lrb.language_rows(run_with(g, d, langs=["a"]), ["a"])[0]
        self.assertLess(row["E_mean"], 0); self.assertLess(row["E_median"], 0)                               # the median draw 0.2 still exceeds the group: robustly negative
        g2 = {"a": 0.3}; row2 = lrb.language_rows(run_with(g2, d, langs=["a"]), ["a"])[0]
        self.assertLess(row2["E_mean"], 0); self.assertGreater(row2["E_median"], 0); self.assertGreater(row2["E_without_outliers"], 0)

    def test_several_states_and_an_empty_language_list(self):
        res = lrb.evaluate({"FP": run_with(G, D), "INT4": run_with({l: 2 * v for l, v in G.items()}, D, state="INT4")}, LANGS)
        self.assertEqual(list(res), ["FP", "INT4"]); self.assertGreater(res["INT4"]["summary"]["E_mean_positive"], res["FP"]["summary"]["E_mean_positive"] - 1)
        with self.assertRaises(ValueError):
            lrb.evaluate({"FP": run_with(G, D)}, [])


class Report(unittest.TestCase):
    def test_the_report_is_labelled_post_hoc_and_says_that_it_cannot_change_the_verdict(self):
        res = lrb.evaluate({"FP": run_with(G, D)}, LANGS)
        text = lrb.report_text(res, {})
        for k in ("post hoc", "cannot change the verdict", "median of the control draws", "exceeds every", "outlier", "FP"):
            self.assertIn(k, text)


class Files(unittest.TestCase):
    def test_main_reads_job_directories_and_the_manifest_and_writes_the_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            save_state(d / "j1", run_with(G, D), "preset-x"); save_state(d / "j2", run_with({l: 2 * v for l, v in G.items()}, D, state="INT4"), "preset-x")
            (d / "man.json").write_text(json.dumps({"languages": LANGS, "groups": {l: {"all": list(range(64 if l != "e" else 5))} for l in LANGS}}))
            with contextlib.redirect_stdout(io.StringIO()) as out:
                lrb.main(["--runs", str(d / "j2"), str(d / "j1"), "--manifest", str(d / "man.json"), "--out", str(d / "out")])
            for f in ("report.md", "robust.json", "languages.csv"):
                self.assertTrue((d / "out" / f).exists(), f)
            js = json.loads((d / "out" / "robust.json").read_text())
            self.assertEqual(js["states"], ["FP", "INT4"]); self.assertEqual(js["tested"], ["a", "b", "c", "d"]); self.assertEqual(js["untested"], ["e"]); self.assertTrue(js["post_hoc"])
            with open(d / "out" / "languages.csv") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 8); self.assertIn("E_median", rows[0]); self.assertIn("outlier_draws", rows[0])


if __name__ == "__main__":
    unittest.main()
