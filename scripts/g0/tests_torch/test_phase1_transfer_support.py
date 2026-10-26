import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phase1_transfer_analysis as ta
import phase1_transfer_cross_corpus as cc
import phase1_transfer_report as rep
import phase1_transfer_support as sup

CTRL = ("matched_swapped", "random1", "random2", "random3")
STATES = ("INT16", "INT6", "INT4")


def item(f, sc, kind, fp, ne=0.5, cos=0.5, edit=5.0, resid=100.0, eligible=None, floor=30.0):
    return {"feature": f, "sentence": sc, "kind": kind, "cluster": sc, "split": "calibration", "fp_norm": fp, "edit_norm_fp": edit, "resid_norm_fp": resid, "normalized_error": ne, "cosine": cos, "error_norm": ne * (fp + floor),
            "eligible": (fp > 5 * floor) if eligible is None else eligible}


def mk(seed, n_features=30, per=6, gap=0.3, cos_gap=0.1, state_shift=0.0):
    """One state table: for every (feature, context) a real item and four control items; the real FP effect is 0.5 to 2 times that of a control; the real normalized error exceeds the control's by `gap`."""
    rng = np.random.default_rng(seed); rows = {}; fp_rng = np.random.default_rng(99)
    for f in range(n_features):
        for j in range(per):
            sc = 1000 * f + j; fpr = float(fp_rng.uniform(100, 400)); edit = float(fp_rng.uniform(2, 9)); resid = float(fp_rng.uniform(80, 150)); rows[(f, sc, "real")] = None; base = float(rng.uniform(0.2, 0.8)); cosb = float(rng.uniform(0.2, 0.7))
            for k in CTRL:
                fpc = fpr / float(fp_rng.uniform(0.5, 2.0)); rows[(f, sc, k)] = item(f, sc, k, fpc, ne=base + float(rng.normal(0, 0.01)), cos=cosb, edit=edit, resid=resid, eligible=True)
            rows[(f, sc, "real")] = item(f, sc, "real", fpr, ne=base + gap + state_shift, cos=cosb + cos_gap, edit=edit, resid=resid, eligible=True)
    return rows


class Pairs(unittest.TestCase):
    def test_a_pair_is_the_real_item_and_the_control_item_of_the_same_feature_and_context_sorted_by_feature_and_context(self):
        rows = {(2, 20, "real"): item(2, 20, "real", 100.0, ne=1.0), (2, 20, "matched_swapped"): item(2, 20, "matched_swapped", 50.0, ne=0.4), (1, 10, "real"): item(1, 10, "real", 200.0, ne=2.0), (1, 10, "matched_swapped"): item(1, 10, "matched_swapped", 80.0, ne=0.8),
                (3, 30, "real"): item(3, 30, "real", 300.0)}
        p = sup.pairs(rows, "matched_swapped"); self.assertEqual(p["feature"].tolist(), [1, 2]); self.assertEqual(p["fp_real"].tolist(), [200.0, 100.0]); self.assertEqual(p["fp_ctrl"].tolist(), [80.0, 50.0]); self.assertEqual(p["ne_real"].tolist(), [2.0, 1.0]); self.assertEqual(p["ne_ctrl"].tolist(), [0.8, 0.4])
        self.assertEqual(p["n_dropped"], 0)

    def test_pairs_without_a_finite_positive_fp_effect_in_either_item_are_dropped_and_counted_never_filled(self):
        rows = {(1, 1, "real"): item(1, 1, "real", 100.0), (1, 1, "random1"): item(1, 1, "random1", 0.0), (2, 2, "real"): item(2, 2, "real", float("nan")), (2, 2, "random1"): item(2, 2, "random1", 50.0), (3, 3, "real"): item(3, 3, "real", 100.0), (3, 3, "random1"): item(3, 3, "random1", 50.0),
                (4, 4, "real"): item(4, 4, "real", float("inf")), (4, 4, "random1"): item(4, 4, "random1", 50.0)}
        p = sup.pairs(rows, "random1"); self.assertEqual(p["feature"].tolist(), [3]); self.assertEqual(p["n_dropped"], 3)

    def test_a_missing_cosine_is_nan_and_the_other_fields_are_carried(self):
        rows = {(1, 1, "real"): item(1, 1, "real", 100.0, cos=None, edit=4.0, resid=90.0), (1, 1, "random2"): item(1, 1, "random2", 80.0, cos=0.3, edit=5.0, resid=95.0)}; p = sup.pairs(rows, "random2")
        self.assertTrue(np.isnan(p["cos_real"][0])); self.assertEqual(p["cos_ctrl"][0], 0.3); self.assertEqual((p["edit_real"][0], p["edit_ctrl"][0], p["resid_real"][0], p["resid_ctrl"][0]), (4.0, 5.0, 90.0, 95.0)); self.assertEqual(p["cluster"].tolist(), [1])


class Sets(unittest.TestCase):
    def test_s1_needs_both_items_eligible_and_s2_the_inclusive_band_of_the_fp_effect_ratio(self):
        rows = {}
        for n_, (fr, fc, er, ec) in enumerate([(80.0, 100.0, True, True), (125.0, 100.0, True, True), (79.0, 100.0, True, True), (126.0, 100.0, True, True), (100.0, 100.0, True, False), (100.0, 100.0, False, True), (100.0, 100.0, True, True)]):
            rows[(n_, n_, "real")] = item(n_, n_, "real", fr, eligible=er); rows[(n_, n_, "matched_swapped")] = item(n_, n_, "matched_swapped", fc, eligible=ec)
        p = sup.pairs(rows, "matched_swapped"); s = sup.sets(p)
        self.assertEqual(s["S0"].tolist(), [True] * 7); self.assertEqual(s["S1"].tolist(), [True, True, True, True, False, False, True]); self.assertEqual(s["S2"].tolist(), [True, True, False, False, False, False, True])
        self.assertTrue(np.allclose(s["rho"], [0.8, 1.25, 0.79, 1.26, 1.0, 1.0, 1.0]))

    def test_the_band_can_be_changed_and_is_nested_in_s1(self):
        rows = {(0, 0, "real"): item(0, 0, "real", 150.0, eligible=True), (0, 0, "random1"): item(0, 0, "random1", 100.0, eligible=True)}; p = sup.pairs(rows, "random1")
        self.assertFalse(sup.sets(p)["S2"][0]); self.assertTrue(sup.sets(p, lo=0.5, hi=2.0)["S2"][0]); self.assertTrue(np.array_equal(sup.sets(p, lo=0.0, hi=1e9)["S2"], sup.sets(p)["S1"]))


class Coverage(unittest.TestCase):
    def test_counts_shares_and_quartiles_of_the_three_ratios_in_each_set(self):
        rows = mk(1); p = sup.pairs(rows, "matched_swapped"); s = sup.sets(p); c = sup.coverage(p, s)
        for name in ("S0", "S1", "S2"):
            m = s[name]; self.assertEqual(c[name]["n_pairs"], int(m.sum())); self.assertEqual(c[name]["n_features"], len(np.unique(p["feature"][m]))); self.assertEqual(c[name]["n_clusters"], len(np.unique(p["cluster"][m])))
            for key, ratio in (("rho", p["fp_real"] / p["fp_ctrl"]), ("edit_ratio", p["edit_real"] / p["edit_ctrl"]), ("resid_ratio", p["resid_real"] / p["resid_ctrl"])):
                self.assertTrue(np.allclose([c[name][key + "_p25"], c[name][key + "_p50"], c[name][key + "_p75"]], np.percentile(ratio[m], [25, 50, 75])), (name, key))
        self.assertAlmostEqual(c["S1"]["share_in_band"], float(s["S2"].sum() / s["S1"].sum())); self.assertIsNone(c["S0"].get("share_in_band")); self.assertEqual(c["n_dropped"], p["n_dropped"])

    def test_the_share_in_band_is_relative_to_s1_and_the_edit_and_residual_ratios_are_real_over_control(self):
        rows = mk(1)
        for i, k in enumerate(sorted(k for k in rows if k[2] == "matched_swapped")):
            rows[k]["eligible"] = i % 3 != 0; rows[k]["edit_norm_fp"] *= 2.0; rows[k]["resid_norm_fp"] *= 4.0
        p = sup.pairs(rows, "matched_swapped"); s = sup.sets(p); c = sup.coverage(p, s); self.assertLess(s["S1"].sum(), s["S0"].sum()); self.assertAlmostEqual(c["S1"]["share_in_band"], float(s["S2"].sum() / s["S1"].sum()))
        self.assertAlmostEqual(c["S0"]["edit_ratio_p50"], 0.5); self.assertAlmostEqual(c["S0"]["resid_ratio_p50"], 0.25); self.assertEqual(c["S0"]["n_pairs"], len(p["feature"]))

    def test_an_empty_set_has_zero_counts_and_no_quartiles(self):
        rows = {(0, 0, "real"): item(0, 0, "real", 100.0, eligible=False), (0, 0, "random3"): item(0, 0, "random3", 100.0, eligible=False)}; p = sup.pairs(rows, "random3"); c = sup.coverage(p, sup.sets(p))
        self.assertEqual((c["S1"]["n_pairs"], c["S2"]["n_pairs"]), (0, 0)); self.assertIsNone(c["S2"]["rho_p50"]); self.assertIsNone(c["S1"]["share_in_band"])

    def test_the_real_items_with_all_four_controls_in_s1_are_counted(self):
        rows = {}
        for f, flags in ((1, (True, True, True, True)), (2, (True, True, True, False)), (3, (False, True, True, True))):
            rows[(f, f, "real")] = item(f, f, "real", 200.0, eligible=flags[0] if f == 3 else True)
            for k, e in zip(CTRL, flags if f != 3 else (True, True, True, True)):
                rows[(f, f, k)] = item(f, f, k, 150.0, eligible=e)
        self.assertEqual(sup.all_four(rows), 1)


class Contrast(unittest.TestCase):
    def setUp(self):
        self.rows = mk(2, gap=0.3, cos_gap=0.1); self.p = sup.pairs(self.rows, "matched_swapped"); self.s = sup.sets(self.p)

    def test_a_planted_gap_is_recovered_in_the_median_the_interval_and_the_share_with_the_counts(self):
        r = sup.contrast(self.p, self.s["S1"], n_boot=200, seed=1); self.assertEqual(r["status"], "fitted"); m = self.s["S1"]; self.assertEqual(r["n"], int(m.sum()))
        self.assertAlmostEqual(r["ne_gap"]["point"], ta.weighted_median(self.p["ne_real"][m] - self.p["ne_ctrl"][m], np.ones(int(m.sum())))); self.assertAlmostEqual(r["ne_gap"]["point"], 0.3, delta=0.02); self.assertGreater(r["ne_gap"]["low"], 0.25)
        self.assertAlmostEqual(r["cos_gap"]["point"], 0.1, delta=1e-9); self.assertEqual(r["share_real_larger"]["point"], 1.0); lr = np.log2(self.p["err_real"][m] / self.p["err_ctrl"][m]); self.assertAlmostEqual(r["log2_error_ratio"]["point"], ta.weighted_median(lr, np.ones(len(lr))))

    def test_the_interval_is_the_crossed_bootstrap_of_the_pair_values_with_the_seed(self):
        r = sup.contrast(self.p, self.s["S1"], n_boot=60, seed=4); m = self.s["S1"]; ref = ta.crossed_bootstrap(self.p["ne_real"][m] - self.p["ne_ctrl"][m], self.p["feature"][m], self.p["cluster"][m], "median", 60, 4)
        self.assertEqual((r["ne_gap"]["low"], r["ne_gap"]["high"]), (ref["low"], ref["high"])); ref2 = ta.crossed_bootstrap((self.p["ne_real"][m] > self.p["ne_ctrl"][m]).astype(float), self.p["feature"][m], self.p["cluster"][m], "mean", 60, 4); self.assertEqual(r["share_real_larger"]["point"], ref2["point"])

    def test_missing_cosines_and_zero_control_errors_are_left_out_of_their_statistics_and_counted(self):
        rows = mk(3); keys = [k for k in rows if k[2] == "real"][:7]
        for k in keys[:4]:
            rows[k]["cosine"] = None
        for k in keys[4:7]:
            rows[(k[0], k[1], "matched_swapped")]["error_norm"] = 0.0
        p = sup.pairs(rows, "matched_swapped"); r = sup.contrast(p, sup.sets(p)["S0"], n_boot=20, seed=1); self.assertEqual(r["cos_gap"]["n_missing"], 4); self.assertEqual(r["log2_error_ratio"]["n_missing"], 3); self.assertEqual(r["ne_gap"]["n_missing"], 0)

    def test_a_set_below_the_minimum_coverage_is_reported_as_insufficient_support_with_its_counts(self):
        m = np.zeros(len(self.p["feature"]), dtype=bool); m[np.where(self.s["S1"])[0][:99]] = True; r = sup.contrast(self.p, m, n_boot=10, seed=1); self.assertEqual(r["status"], "insufficient support"); self.assertEqual(r["n"], 99); self.assertNotIn("ne_gap", r)
        few = np.zeros(len(self.p["feature"]), dtype=bool); idx = np.where(self.p["feature"] < 19)[0]; few[idx] = True; r2 = sup.contrast(self.p, few, n_boot=10, seed=1); self.assertEqual((r2["status"], r2["n_features"]), ("insufficient support", 19))

    def test_ninety_nine_pairs_are_not_enough_even_with_many_features(self):
        rows = mk(6, n_features=33, per=3); p = sup.pairs(rows, "random1"); m = np.ones(len(p["feature"]), dtype=bool); self.assertEqual((int(m.sum()), len(np.unique(p["feature"]))), (99, 33)); r = sup.contrast(p, m, n_boot=10, seed=1)
        self.assertEqual((r["status"], r["n"], r["n_features"]), ("insufficient support", 99, 33))

    def test_a_real_item_with_a_zero_error_norm_is_left_out_of_the_log_ratio_and_ties_do_not_count_as_larger(self):
        rows = mk(7, n_features=20, per=6, gap=0.3)
        for k in [k for k in rows if k[2] == "real"][:5]:
            rows[k]["error_norm"] = 0.0
        for k in [k for k in rows if k[2] == "real"][5:15]:
            rows[k]["normalized_error"] = rows[(k[0], k[1], "matched_swapped")]["normalized_error"]
        p = sup.pairs(rows, "matched_swapped"); r = sup.contrast(p, np.ones(len(p["feature"]), dtype=bool), n_boot=10, seed=1)
        self.assertEqual(r["log2_error_ratio"]["n_missing"], 5); self.assertAlmostEqual(r["share_real_larger"]["point"], (len(p["feature"]) - 10) / len(p["feature"]))

    def test_exactly_the_minimum_pairs_and_features_are_enough(self):
        rows = mk(5, n_features=20, per=5); p = sup.pairs(rows, "random1"); m = np.ones(len(p["feature"]), dtype=bool); self.assertEqual((int(m.sum()), len(np.unique(p["feature"]))), (100, 20)); self.assertEqual(sup.contrast(p, m, n_boot=10, seed=1)["status"], "fitted")


class Run(unittest.TestCase):
    def build(self, root, gap_by_state, shift_fp_state=None):
        (root / "main").mkdir(); (root / "ext").mkdir()
        for side, off in (("main", 0), ("ext", 50)):
            for i, s in enumerate(STATES):
                rows = mk(off + i, n_features=60, per=8, gap=gap_by_state[s])
                if shift_fp_state == (side, s):
                    k0 = next(iter(rows)); rows[k0]["fp_norm"] *= 1.5; rows[(k0[0], k0[1], "matched_swapped")]["eligible"] = False
                rep.write_csv(root / side / f"per_context_{s}.csv", [r for _, r in sorted(rows.items())])

    def test_the_driver_writes_coverage_once_per_corpus_and_kind_and_a_contrast_per_state_kind_and_set_and_hashes_the_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); self.build(root, {"INT16": 0.0, "INT6": 0.1, "INT4": 0.5}); cov, con = sup.run(root / "main", root / "ext", STATES, root / "out", n_boot=20, seed=1)
            self.assertEqual(len(cov), 2 * 4 * 3 + 2); self.assertEqual(len(con), 2 * 3 * 4 * 2); self.assertEqual({r["control_kind"] for r in con}, set(CTRL))
            js = json.loads((root / "out" / "support.json").read_text()); self.assertTrue(js["sets_identical_across_states"]["flores_main"]); self.assertEqual(len(js["inputs_sha256"]), 6)
            for name in ("support_coverage.csv", "support_contrast.csv"):
                with (root / "out" / name).open() as fh:
                    self.assertGreater(len(list(csv.DictReader(fh))), 0)
            r = [x for x in con if x["corpus"] == "flores_main" and x["state"] == "INT4" and x["control_kind"] == "matched_swapped" and x["set"] == "S1"][0]; self.assertAlmostEqual(float(r["ne_gap"]), 0.5, delta=0.03)
            r16 = [x for x in con if x["corpus"] == "flores_main" and x["state"] == "INT16" and x["control_kind"] == "matched_swapped" and x["set"] == "S1"][0]; self.assertAlmostEqual(float(r16["ne_gap"]), 0.0, delta=0.03)

    def test_a_contrast_row_equals_the_direct_computation_on_the_table_read_back(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); self.build(root, {"INT16": 0.0, "INT6": 0.1, "INT4": 0.5}); cov, con = sup.run(root / "main", root / "ext", STATES, root / "out", n_boot=25, seed=3)
            rows = cc.rows_from_csv(root / "ext" / "per_context_INT6.csv"); p = sup.pairs(rows, "random2"); m = sup.sets(p)["S2"]; ref = ta.crossed_bootstrap(p["ne_real"][m] - p["ne_ctrl"][m], p["feature"][m], p["cluster"][m], "median", 25, 3)
            r = [x for x in con if x["corpus"] == "wikipedia_external" and x["state"] == "INT6" and x["control_kind"] == "random2" and x["set"] == "S2"][0]
            self.assertEqual(int(r["n"]), int(m.sum())); self.assertAlmostEqual(float(r["ne_gap"]), ref["point"], places=12); self.assertAlmostEqual(float(r["ne_gap_low"]), ref["low"], places=12); self.assertAlmostEqual(float(r["ne_gap_high"]), ref["high"], places=12)

    def test_support_sets_that_differ_between_the_state_tables_are_reported(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); self.build(root, {"INT16": 0.0, "INT6": 0.1, "INT4": 0.5}, shift_fp_state=("ext", "INT4")); sup.run(root / "main", root / "ext", STATES, root / "out", n_boot=10, seed=1)
            js = json.loads((root / "out" / "support.json").read_text()); self.assertTrue(js["sets_identical_across_states"]["flores_main"]); self.assertFalse(js["sets_identical_across_states"]["wikipedia_external"])
            rows = cc.rows_from_csv(root / "ext" / "per_context_INT16.csv"); first = int(sup.sets(sup.pairs(rows, "matched_swapped"))["S1"].sum()); got = [r for r in js["coverage"] if r["corpus"] == "wikipedia_external" and r["control_kind"] == "matched_swapped" and r["set"] == "S1"][0]
            self.assertEqual(got["n_pairs"], first)                      # the coverage comes from the first state table (INT16)


if __name__ == "__main__":
    unittest.main()
