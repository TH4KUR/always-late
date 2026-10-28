"""Tests for scripts/g0/survival_effect_llama.py (Llama-3.2-1B F5 check; addendum reports/survival_effect_rescue/llama/llama_f5_addendum_2026-10-28.md): the table of a Llama F5 run, the instrument gate with Llama's own
floor, and the labels of the side-by-side with Gemma. Synthetic arrays only; numpy."""
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import survival_effect_llama as sl  # noqa: E402


def run(n=6):
    rng = np.random.default_rng(0)
    sa = {"feat_ids": np.array([100, 7, 55, 9, 31, 64]), "feat_lang": np.array([0, 1, 1, 2, 2, 2]), "write_rows": np.arange(n), "n_active": np.array([10, 20, 30, 40, 50, 60])}
    sample = {"features": {str(i): {"language": "x", "roles": ["k64"], "native": {"FP": [0.01 * (i + 1), 0.5 + i]}} for i in (100, 7, 55, 9, 31, 64)}}
    mk = lambda: {"read": {"corr": rng.uniform(0.5, 1, n), "jaccard": rng.uniform(0.5, 1, n), "scale": rng.uniform(0.9, 1.1, n)},
                  "write": {"cos": rng.uniform(0.7, 1, n), "dnll": rng.normal(size=n) * 0.01, "n_pos": rng.integers(10, 100, n)}}
    states = {s: mk() for s in ("FP", "INT16", "INT6", "INT4")}
    return sa, states, sample


class Table(unittest.TestCase):
    def test_baselines_come_from_stage_a_the_fp_state_and_the_sample_native_activity(self):
        sa, states, sample = run(); T = sl.feature_table(sa, states, sample)
        np.testing.assert_array_equal(T["ids"], [100, 7, 55, 9, 31, 64]); np.testing.assert_array_equal(T["lang"], [0, 1, 1, 2, 2, 2])
        np.testing.assert_allclose(T["base"][:, 0], [10, 20, 30, 40, 50, 60])                 # n_active
        np.testing.assert_allclose(T["base"][:, 1], [1.01, 0.08, 0.56, 0.10, 0.32, 0.65])      # native firing rate of the sample
        np.testing.assert_allclose(T["base"][:, 2], [100.5, 7.5, 55.5, 9.5, 31.5, 64.5])       # native mean activation of the sample
        np.testing.assert_allclose(T["base"][:, 3], np.abs(states["FP"]["write"]["dnll"])); np.testing.assert_allclose(T["base"][:, 4], states["FP"]["write"]["n_pos"])

    def test_the_state_columns_are_the_read_and_write_values_of_that_state_only(self):
        sa, states, sample = run(); T = sl.feature_table(sa, states, sample)
        for s in ("INT6", "INT4"):
            np.testing.assert_allclose(T[s]["r1"], states[s]["read"]["corr"]); np.testing.assert_allclose(T[s]["w1"], states[s]["write"]["cos"])
            np.testing.assert_allclose(T[s]["scale_dev"], np.abs(np.asarray(states[s]["read"]["scale"]) - 1.0))
        np.testing.assert_allclose(T["cos16"], states["INT16"]["write"]["cos"])

    def test_write_rows_select_the_read_side_rows(self):
        sa, states, sample = run(); sa["write_rows"] = np.array([5, 4, 3, 2, 1, 0]); sa["feat_ids"] = sa["feat_ids"][::-1].copy(); sa["feat_lang"] = sa["feat_lang"][::-1].copy()
        T = sl.feature_table(sa, states, sample)
        np.testing.assert_allclose(T["INT6"]["r1"], np.asarray(states["INT6"]["read"]["corr"])[[5, 4, 3, 2, 1, 0]])

    def test_a_feature_missing_from_the_sample_is_refused(self):
        sa, states, sample = run(); del sample["features"]["55"]
        with self.assertRaises(ValueError):
            sl.feature_table(sa, states, sample)


class Gate(unittest.TestCase):
    def test_the_floor_is_the_fifth_percentile_of_the_int16_cosine_of_this_system(self):
        c = np.linspace(0.5, 1.0, 101); self.assertAlmostEqual(sl.own_floor(c), np.percentile(c, 5))

    def test_the_instrument_is_inadequate_below_a_median_of_point_seven_or_with_too_few_reliable_features(self):
        good = np.full(400, 0.98); g = sl.instrument_gate(good, min_reliable=300)
        self.assertTrue(g["ok"]); self.assertEqual(g["n_reliable"], 400 - int((good < g["floor"]).sum()))
        self.assertFalse(sl.instrument_gate(np.full(400, 0.65), min_reliable=300)["ok"])
        mix = np.concatenate([np.full(100, 0.99), np.full(300, 0.30), np.full(100, 0.995)]); self.assertFalse(sl.instrument_gate(mix, min_reliable=300)["ok"])   # median 0.30 is fine to fail on median
        self.assertFalse(sl.instrument_gate(np.full(200, 0.98), min_reliable=300)["ok"])      # too few reliable features
        self.assertIn("median", sl.instrument_gate(np.full(400, 0.65), min_reliable=300)["why"][0])

    def test_features_exactly_at_the_floor_count_as_reliable(self):
        c = np.array([0.5, 0.9, 0.95, 0.99]); fl = sl.own_floor(c); g = sl.instrument_gate(np.append(c, np.full(400, 0.99)), min_reliable=1)
        self.assertGreaterEqual(g["n_reliable"], 400 + 1)


class Labels(unittest.TestCase):
    def test_the_association_label_follows_the_partial_spearman_interval(self):
        self.assertEqual(sl.association_label(0.01, 0.4), "repeats"); self.assertEqual(sl.association_label(-0.2, 0.3), "cannot be resolved")
        self.assertEqual(sl.association_label(-0.5, -0.01), "reverses"); self.assertEqual(sl.association_label(0.0, 0.3), "cannot be resolved"); self.assertEqual(sl.association_label(-0.4, 0.0), "cannot be resolved")
        self.assertEqual(sl.association_label(float("nan"), 0.3), "cannot be resolved")

    def test_predictive_value_repeats_only_when_the_cell_meets_the_rule(self):
        self.assertEqual(sl.value_label(True), "repeats"); self.assertEqual(sl.value_label(False), "does not repeat")


if __name__ == "__main__":
    unittest.main()
