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

    def test_a_state_missing_in_one_run_is_skipped_and_listed(self):
        del self.sens["states"]["INT4"]; c = sn.compare_runs(self.main, self.sens, ["INT16", "INT4"]); self.assertEqual(sorted(c["states"]), ["INT16"]); self.assertEqual(c["missing"], ["INT4"])

    def test_the_floor_state_is_required(self):
        del self.sens["states"]["INT16"]
        with self.assertRaises(ValueError):
            sn.compare_runs(self.main, self.sens, ["INT4"], floor_state="INT16")

    def test_the_report_names_the_run_and_every_number(self):
        c = sn.compare_runs(self.main, self.sens, ["INT16", "INT4"]); md = sn.report_text(c, "M = 8 against M = 16")
        self.assertIn("M = 8 against M = 16", md); self.assertIn("| INT4 |", md); self.assertIn(f"{c['states']['INT4']['spearman']:.3f}", md); self.assertIn(f"{c['floor']['tau_a']:.4f}", md); self.assertIn("exploratory", md)


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


if __name__ == "__main__":
    unittest.main()
