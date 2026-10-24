import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phase1_func_posctrl_summary as ps

LANGS = ["eng_Latn", "aaa", "bbb", "ccc"]


def table(none, zero, ctrl, sham=None, langs=LANGS):
    """Rows of an accuracy.csv of one dose: lists in the order of `langs`."""
    sham = none if sham is None else sham
    return [{"precision": "FP", "language": l, "n_items": 385, "acc_none": a, "acc_zero": b, "acc_control_mean": c, "acc_sham": s} for l, a, b, c, s in zip(langs, none, zero, ctrl, sham)]


def write(path, rows, extra=()):
    cols = ["precision", "language", "n_items", "acc_none", "acc_zero", "acc_control_mean", "acc_sham"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); [w.writerow(r) for r in list(rows) + list(extra)]


NONE = [0.50, 0.40, 0.40, 0.40]


class Summarise(unittest.TestCase):
    def setUp(self):
        self.big = table(NONE, [0.44, 0.36, 0.36, 0.36], [0.38, 0.30, 0.30, 0.30]); self.small = table(NONE, [0.48, 0.38, 0.38, 0.38], [0.45, 0.35, 0.35, 0.35])

    def test_macro_means_and_differences_of_a_dose(self):
        s = ps.summarise_dose(self.big, n_boot=200, seed=0)
        self.assertAlmostEqual(s["macro"]["none"], 0.425); self.assertAlmostEqual(s["macro"]["zero"], 0.38); self.assertAlmostEqual(s["macro"]["controls"], 0.32); self.assertAlmostEqual(s["macro"]["sham"], 0.425)
        d = s["diff"]; self.assertAlmostEqual(d["zero_minus_none"]["mean"], -0.045); self.assertAlmostEqual(d["controls_minus_none"]["mean"], -0.105); self.assertAlmostEqual(d["zero_minus_controls"]["mean"], 0.06); self.assertAlmostEqual(d["sham_minus_none"]["mean"], 0.0)
        self.assertEqual(s["n_languages"], 4); self.assertEqual(d["zero_minus_none"]["n_negative"], 4); self.assertEqual(d["zero_minus_controls"]["n_negative"], 0); self.assertEqual(d["sham_minus_none"]["n_negative"], 0)             # a difference of exactly 0 is not a fall

    def test_the_interval_is_the_percentile_interval_of_language_resamples_with_the_given_seed(self):
        import numpy as np
        s = ps.summarise_dose(self.big, n_boot=300, seed=7); v = np.array([-0.06, -0.04, -0.04, -0.04]); means = v[np.random.default_rng(7).integers(0, 4, (300, 4))].mean(axis=1)
        self.assertAlmostEqual(s["diff"]["zero_minus_none"]["lo"], float(np.percentile(means, 2.5))); self.assertAlmostEqual(s["diff"]["zero_minus_none"]["hi"], float(np.percentile(means, 97.5)))
        self.assertEqual(ps.summarise_dose(self.big, n_boot=300, seed=7)["diff"]["zero_minus_none"], s["diff"]["zero_minus_none"])

    def test_the_interval_limits_are_the_2_5th_and_97_5th_percentiles_with_enough_languages_to_tell_them_from_the_5th_and_95th(self):
        import numpy as np
        langs = [f"l{i}" for i in range(12)]; none = [0.40] * 12; zero = [0.40 - 0.01 * (i + 1) ** 1.3 / 3 for i in range(12)]; rows = table(none, zero, none, langs=langs)
        s = ps.summarise_dose(rows, n_boot=400, seed=3, english="none"); v = np.array(zero) - 0.40; means = v[np.random.default_rng(3).integers(0, 12, (400, 12))].mean(axis=1)
        self.assertAlmostEqual(s["diff"]["zero_minus_none"]["lo"], float(np.percentile(means, 2.5))); self.assertAlmostEqual(s["diff"]["zero_minus_none"]["hi"], float(np.percentile(means, 97.5)))
        self.assertNotAlmostEqual(float(np.percentile(means, 2.5)), float(np.percentile(means, 5)), places=6)

    def test_the_control_effect_is_given_for_english_as_target_and_for_the_other_languages_separately(self):
        s = ps.summarise_dose(self.big, n_boot=200, seed=0, english="eng_Latn"); e = s["controls_by_target"]
        self.assertAlmostEqual(e["english"], -0.12); self.assertAlmostEqual(e["others_mean"], -0.10); self.assertEqual(e["n_others"], 3)
        self.assertIsNone(ps.summarise_dose(table(NONE[1:], NONE[1:], NONE[1:], langs=LANGS[1:]), n_boot=50, seed=0)["controls_by_target"]["english"])

    def test_only_full_precision_rows_count_and_the_languages_are_those_of_the_table(self):
        rows = self.big + [{**self.big[0], "precision": "INT6", "acc_zero": 0.0}]; self.assertAlmostEqual(ps.summarise_dose(rows, n_boot=50, seed=0)["macro"]["zero"], 0.38)

    def test_missing_columns_or_no_rows_are_errors(self):
        with self.assertRaises(ValueError):
            ps.summarise_dose([], n_boot=10, seed=0)
        with self.assertRaises(ValueError):
            ps.summarise_dose([{"precision": "FP", "language": "a", "acc_none": 0.4}], n_boot=10, seed=0)


class Verdicts(unittest.TestCase):
    def run_doses(self, doses):
        return ps.verdicts({k: ps.summarise_dose(v, n_boot=50, seed=0) for k, v in doses.items()})

    def test_p1_needs_the_whole_pool_fall_of_at_least_five_points(self):
        ok = table(NONE, [0.45, 0.35, 0.35, 0.35], NONE); self.assertEqual(self.run_doses({"all": ok})["P1"]["verdict"], "met")                                  # -0.05 exactly
        no = table(NONE, [0.451, 0.351, 0.351, 0.351], NONE); self.assertEqual(self.run_doses({"all": no})["P1"]["verdict"], "not met")
        self.assertEqual(self.run_doses({"1024": ok})["P1"]["verdict"], "not evaluable")                                                                        # no whole-pool dose

    def test_the_thresholds_hold_for_decimal_values_whatever_the_floating_point_rounding(self):
        for a, b in ((0.50, 0.45), (0.55, 0.50), (0.45, 0.40), (0.60, 0.55), (0.65, 0.60), (0.70, 0.65), (0.75, 0.70)):                                           # a fall of exactly 0.05 on paper
            self.assertEqual(self.run_doses({"all": table([a], [b], [a], langs=["x"])})["P1"]["verdict"], "met", (a, b))
        for a, b in ((0.50, 0.49), (0.55, 0.54), (0.45, 0.44), (0.35, 0.34), (0.41, 0.40), (0.57, 0.56), (0.69, 0.68), (0.82, 0.81)):                                           # a fall of exactly 0.01 is still 'intermediate'
            self.assertEqual(self.run_doses({"all": table([a], [b], [a], langs=["x"])})["P1"]["reading"], "intermediate", (a, b))

    def test_p1_reports_an_intermediate_fall_as_such(self):
        mid = table(NONE, [0.48, 0.38, 0.38, 0.38], NONE); v = self.run_doses({"all": mid})["P1"]; self.assertEqual(v["verdict"], "not met"); self.assertEqual(v["reading"], "intermediate")
        small = table(NONE, [0.499, 0.399, 0.399, 0.399], NONE); self.assertEqual(self.run_doses({"all": small})["P1"]["reading"], "no fall")

    def test_p2_needs_a_monotone_fall_over_the_doses_present(self):
        big = table(NONE, [0.44, 0.36, 0.36, 0.36], NONE); mid = table(NONE, [0.47, 0.37, 0.37, 0.37], NONE); small = table(NONE, [0.49, 0.39, 0.39, 0.39], NONE)
        self.assertEqual(self.run_doses({"256": small, "1024": mid, "all": big})["P2"]["verdict"], "met"); self.assertEqual(self.run_doses({"256": mid, "1024": small, "all": big})["P2"]["verdict"], "not met")
        self.assertEqual(self.run_doses({"1024": mid, "all": big})["P2"]["verdict"], "met"); self.assertEqual(self.run_doses({"all": big})["P2"]["verdict"], "not evaluable")

    def test_p2_accepts_equal_falls(self):
        same = table(NONE, [0.44, 0.36, 0.36, 0.36], NONE); self.assertEqual(self.run_doses({"1024": same, "all": same})["P2"]["verdict"], "met")

    def test_p3_needs_the_sham_to_equal_none_in_every_language(self):
        a = table(NONE, NONE, NONE); self.assertEqual(self.run_doses({"all": a})["P3"]["verdict"], "met")
        b = table(NONE, NONE, NONE, sham=[0.50, 0.40, 0.40, 0.4000001]); self.assertEqual(self.run_doses({"all": b})["P3"]["verdict"], "not met")

    def test_p4_asks_whether_the_zero_edit_lowers_the_accuracy_more_than_the_controls_at_the_whole_pool_dose(self):
        more = table(NONE, [0.40, 0.30, 0.30, 0.30], [0.44, 0.36, 0.36, 0.36]); self.assertEqual(self.run_doses({"all": more})["P4"]["verdict"], "met")
        less = table(NONE, [0.44, 0.36, 0.36, 0.36], [0.40, 0.30, 0.30, 0.30]); self.assertEqual(self.run_doses({"all": less})["P4"]["verdict"], "not met")
        same = table(NONE, [0.44, 0.36, 0.36, 0.36], [0.44, 0.36, 0.36, 0.36]); self.assertEqual(self.run_doses({"all": same})["P4"]["verdict"], "not met")

    def test_the_largest_fall_of_any_edit_set_is_reported_for_the_can_the_edit_path_register_an_effect_statement(self):
        v = self.run_doses({"1024": table(NONE, [0.49, 0.39, 0.39, 0.39], [0.46, 0.36, 0.36, 0.36]), "all": table(NONE, [0.47, 0.37, 0.37, 0.37], [0.40, 0.30, 0.30, 0.30])})["largest_fall"]
        self.assertEqual(v["set"], "controls"); self.assertEqual(v["dose"], "all"); self.assertAlmostEqual(v["fall"], -0.10)


class Command(unittest.TestCase):
    def test_the_command_line_writes_json_and_a_report_that_says_what_the_labels_stand_for(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); write(d / "a.csv", table(NONE, [0.44, 0.36, 0.36, 0.36], [0.38, 0.30, 0.30, 0.30])); write(d / "b.csv", table(NONE, [0.48, 0.38, 0.38, 0.38], [0.45, 0.35, 0.35, 0.35]))
            ps.main(["--dose", "all", str(d / "a.csv"), "--dose", "1024", str(d / "b.csv"), "--out", str(d / "o"), "--n-boot", "100"])
            j = json.loads((d / "o" / "summary.json").read_text()); self.assertEqual(sorted(j["doses"]), ["1024", "all"]); self.assertIn("verdicts", j); self.assertEqual(j["verdicts"]["P2"]["verdict"], "met")
            md = (d / "o" / "summary.md").read_text(); self.assertIn("Exploratory", md); self.assertIn("positive control", md); self.assertIn("P1", md); self.assertIn("English", md); self.assertIn("random sets of the same size from the selective features of the other languages", md)

    def test_an_unknown_dose_label_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); write(d / "a.csv", table(NONE, NONE, NONE))
            with self.assertRaises(ValueError):
                ps.main(["--dose", "512", str(d / "a.csv"), "--out", str(d / "o")])

    def test_doses_must_cover_the_same_languages(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); write(d / "a.csv", table(NONE, NONE, NONE)); write(d / "b.csv", table(NONE[1:], NONE[1:], NONE[1:], langs=LANGS[1:]))
            with self.assertRaises(ValueError):
                ps.main(["--dose", "all", str(d / "a.csv"), "--dose", "1024", str(d / "b.csv"), "--out", str(d / "o")])


if __name__ == "__main__":
    unittest.main()
