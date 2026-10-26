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
import phase1_transfer_gradient_check as gc
import phase1_transfer_report as rep

STATES = ("INT16", "INT6", "INT4")


def row(feature, sentence, kind="real", gp=0.0, gqf=0.0, gqn=0.0, dff=0.0, dqq=0.0, dqn=0.0, edit=1.0, eligible=True):
    return {"feature": feature, "sentence": sentence, "kind": kind, "cluster": sentence, "split": "calibration", "eligible": eligible, "edit_norm_fp": edit, "g_pred_fp": gp, "g_pred_q_fixed": gqf, "g_pred_q_native": gqn,
            "dloss_FP_FP": dff, "dloss_Q_Q": dqq, "dloss_Q_Q_native": dqn}


def table(seed, n_features=12, per=6, noise=0.2, drift=0.0):
    """Rows of one state: the actual loss change follows the first-order prediction with slope 2 plus noise; the quantized state adds `drift`."""
    rng = np.random.default_rng(seed); rows = {}
    for f in range(n_features):
        for j in range(per):
            sc = 1000 * f + j
            for k in ("real", "matched_swapped", "random1", "random2", "random3"):
                gp = float(rng.normal(0, 1)); gqf = gp + float(rng.normal(0, 0.1)) + drift; gqn = gqf + float(rng.normal(0, 0.1))
                rows[(f, sc, k)] = row(f, sc, k, gp=gp, gqf=gqf, gqn=gqn, dff=2 * gp + float(rng.normal(0, noise)), dqq=2 * gqf + float(rng.normal(0, noise)), dqn=2 * gqn + float(rng.normal(0, noise)), edit=float(rng.uniform(1, 10)))
    return rows


class WeightedMetrics(unittest.TestCase):
    def test_a_proportional_relation_has_unit_correlation_the_slope_the_full_origin_r2_and_full_sign_agreement(self):
        p = np.array([-2.0, -1.0, 0.5, 1.0, 3.0]); m = gc.weighted_metrics(p, 2 * p, np.ones((1, 5)))
        self.assertAlmostEqual(m["r"][0], 1.0); self.assertAlmostEqual(m["slope"][0], 2.0); self.assertAlmostEqual(m["r2_origin"][0], 1.0); self.assertAlmostEqual(m["sign_agree"][0], 1.0)

    def test_an_inverted_relation_has_negative_correlation_and_slope_and_no_sign_agreement(self):
        p = np.array([-2.0, -1.0, 0.5, 1.0, 3.0]); m = gc.weighted_metrics(p, -p, np.ones((1, 5))); self.assertAlmostEqual(m["r"][0], -1.0); self.assertAlmostEqual(m["slope"][0], -1.0); self.assertAlmostEqual(m["sign_agree"][0], 0.0)

    def test_the_formulas_on_a_small_example_and_integer_weights_equal_repeated_rows(self):
        p = np.array([1.0, 2.0, -1.0, 0.5]); a = np.array([0.5, 3.0, -2.0, -0.2]); m = gc.weighted_metrics(p, a, np.ones((1, 4)))
        self.assertAlmostEqual(m["r"][0], float(np.corrcoef(p, a)[0, 1])); beta = (p * a).sum() / (p * p).sum(); self.assertAlmostEqual(m["slope"][0], beta); self.assertAlmostEqual(m["r2_origin"][0], 1 - ((a - beta * p) ** 2).sum() / (a * a).sum())
        self.assertAlmostEqual(m["sign_agree"][0], 0.75); w = np.array([[2.0, 1.0, 3.0, 1.0]]); rw = gc.weighted_metrics(p, a, w); rep_ = np.repeat(np.arange(4), [2, 1, 3, 1]); rr = gc.weighted_metrics(p[rep_], a[rep_], np.ones((1, len(rep_))))
        for k in m:
            self.assertAlmostEqual(rw[k][0], rr[k][0], places=12)

    def test_the_noise_mask_restricts_the_sign_agreement_to_changes_above_the_threshold_and_zero_changes_are_not_counted(self):
        p = np.array([1.0, 1.0, -1.0, 1.0, 1.0, 1.0]); a = np.array([0.05, -0.05, 2.0, 2.0, 0.0, 0.1]); m = gc.weighted_metrics(p, a, np.ones((1, 6)), noise=0.1)
        self.assertAlmostEqual(m["sign_agree"][0], 3 / 5); self.assertAlmostEqual(m["sign_agree_above_noise"][0], 1 / 2); self.assertEqual(m["n_above_noise"], 2)          # a change equal to the threshold is not above it


class Arrays(unittest.TestCase):
    def test_the_five_comparisons_use_the_stored_scalars_and_the_shifts_are_differences_of_both_sides(self):
        rows = {(1, 1, "real"): row(1, 1, gp=0.1, gqf=0.3, gqn=0.5, dff=1.0, dqq=2.0, dqn=4.0), (1, 2, "random1"): row(1, 2, "random1", gp=9.0, dff=9.0)}
        a = gc.comparison_arrays(rows, ("real",))
        self.assertEqual({k: (float(v["p"][0]), float(v["a"][0])) for k, v in a.items()}, {"C0": (0.1, 1.0), "C1": (0.3, 2.0), "C2": (0.5, 4.0), "S1": (0.3 - 0.1, 2.0 - 1.0), "S2": (0.5 - 0.1, 4.0 - 1.0)})
        self.assertEqual(len(a["C0"]["p"]), 1); self.assertEqual(len(gc.comparison_arrays(rows, ("random1",))["C0"]["p"]), 1)

    def test_an_item_with_a_missing_value_is_left_out_of_the_comparisons_that_need_it_and_counted(self):
        rows = {(1, 1, "real"): row(1, 1, gp=0.1, gqf=0.3, gqn=0.5, dff=1.0, dqq=None, dqn=4.0), (1, 2, "real"): row(1, 2, gp=0.2, gqf=0.1, gqn=0.0, dff=1.0, dqq=2.0, dqn=3.0)}
        a = gc.comparison_arrays(rows, ("real",)); self.assertEqual(len(a["C0"]["p"]), 2); self.assertEqual(len(a["C1"]["p"]), 1); self.assertEqual(len(a["S1"]["p"]), 1); self.assertEqual(len(a["C2"]["p"]), 2); self.assertEqual(a["C1"]["n_missing"], 1); self.assertEqual(a["C0"]["n_missing"], 0)

    def test_a_non_finite_value_is_left_out_like_a_missing_one(self):
        rows = {(1, 1, "real"): row(1, 1, gp=float("nan"), gqf=0.3, gqn=0.5, dff=1.0, dqq=2.0, dqn=4.0), (1, 2, "real"): row(1, 2, gp=0.2, gqf=0.1, gqn=0.0, dff=1.0, dqq=float("inf"), dqn=3.0)}
        a = gc.comparison_arrays(rows, ("real",)); self.assertEqual((len(a["C0"]["p"]), a["C0"]["n_missing"]), (1, 1)); self.assertEqual((len(a["C1"]["p"]), a["C1"]["n_missing"]), (1, 1)); self.assertEqual(len(a["C2"]["p"]), 2)

    def test_the_arrays_carry_the_feature_cluster_edit_size_and_eligibility_of_each_item_in_a_fixed_order(self):
        rows = {(2, 20, "real"): row(2, 20, edit=5.0, eligible=False), (1, 10, "real"): row(1, 10, edit=3.0, eligible=True)}; a = gc.comparison_arrays(rows, ("real",))["C0"]
        self.assertEqual(a["feature"].tolist(), [1, 2]); self.assertEqual(a["cluster"].tolist(), [10, 20]); self.assertEqual(a["edit"].tolist(), [3.0, 5.0]); self.assertEqual(a["eligible"].tolist(), [True, False])


class Noise(unittest.TestCase):
    def test_the_threshold_is_the_95th_percentile_of_the_int16_shift_of_the_real_items_only(self):
        rows = table(1); t = gc.noise_threshold(rows); real = [abs(r["dloss_Q_Q"] - r["dloss_FP_FP"]) for r in rows.values() if r["kind"] == "real"]; self.assertAlmostEqual(t, float(np.percentile(real, 95)))
        every = [abs(r["dloss_Q_Q"] - r["dloss_FP_FP"]) for r in rows.values()]; self.assertNotAlmostEqual(t, float(np.percentile(every, 95)), places=6)


class MetricsCI(unittest.TestCase):
    def setUp(self):
        self.rows = table(2); self.a = gc.comparison_arrays(self.rows, ("real",))["C1"]

    def test_the_point_is_the_unweighted_statistic_and_the_interval_brackets_it_with_the_counts(self):
        m = gc.metrics_ci(self.a["p"], self.a["a"], self.a["feature"], self.a["cluster"], 0.3, n_boot=200, seed=1); ref = gc.weighted_metrics(self.a["p"], self.a["a"], np.ones((1, len(self.a["p"]))), noise=0.3)
        for k in ("r", "slope", "r2_origin", "sign_agree", "sign_agree_above_noise"):
            self.assertAlmostEqual(m[k]["point"], ref[k][0], places=12); self.assertLessEqual(m[k]["low"], m[k]["point"] + 1e-9); self.assertGreaterEqual(m[k]["high"], m[k]["point"] - 1e-9); self.assertLess(m[k]["low"], m[k]["high"])
        self.assertEqual((m["n"], m["n_features"], m["n_clusters"], m["n_above_noise"]), (len(self.a["p"]), 12, 72, int((np.abs(self.a["a"]) > 0.3).sum())))
        self.assertGreater(m["slope"]["low"], 1.5); self.assertLess(m["slope"]["high"], 2.5); self.assertGreater(m["r"]["low"], 0.8)

    def test_the_interval_is_the_2_5_and_97_5_percentile_of_the_replicates_of_the_seeded_multiplicities(self):
        p, a_, fi, ci = self.a["p"], self.a["a"], self.a["feature"], self.a["cluster"]; _, f = np.unique(fi, return_inverse=True); _, c = np.unique(ci, return_inverse=True); mf, mc = ta.bootstrap_multiplicities(f.max() + 1, c.max() + 1, 80, 5)
        reps = gc.replicates(p, a_, f, c, mf, mc, 0.3); m = gc.metrics_ci(p, a_, fi, ci, 0.3, n_boot=80, seed=5)
        for k in ("r", "slope", "r2_origin", "sign_agree", "sign_agree_above_noise"):
            self.assertAlmostEqual(m[k]["low"], float(np.nanpercentile(reps[k], 2.5)), places=12); self.assertAlmostEqual(m[k]["high"], float(np.nanpercentile(reps[k], 97.5)), places=12)

    def test_the_same_seed_gives_the_same_interval_and_another_seed_a_different_one(self):
        a = gc.metrics_ci(self.a["p"], self.a["a"], self.a["feature"], self.a["cluster"], 0.3, 100, 1); b = gc.metrics_ci(self.a["p"], self.a["a"], self.a["feature"], self.a["cluster"], 0.3, 100, 1); c = gc.metrics_ci(self.a["p"], self.a["a"], self.a["feature"], self.a["cluster"], 0.3, 100, 2)
        self.assertEqual(a, b); self.assertNotEqual(a["slope"]["low"], c["slope"]["low"])

    def test_the_chunked_replicates_equal_one_pass_over_the_same_multiplicities(self):
        p, a_, fi, ci = self.a["p"], self.a["a"], self.a["feature"], self.a["cluster"]; _, f = np.unique(fi, return_inverse=True); _, c = np.unique(ci, return_inverse=True); mf, mc = ta.bootstrap_multiplicities(f.max() + 1, c.max() + 1, 60, 3)
        one = gc.weighted_metrics(p, a_, mf[:, f] * mc[:, c], noise=0.3); self.assertEqual(one["slope"].shape, (60,)); chunk = gc.replicates(p, a_, f, c, mf, mc, 0.3, chunk=7)
        for k in one:
            if k != "n_above_noise":
                self.assertTrue(np.allclose(one[k], chunk[k], equal_nan=True), k)


class Run(unittest.TestCase):
    def build(self, root, drift_by_state, seed0=0, ordered_edits=False):
        (root / "main").mkdir(); (root / "ext").mkdir()
        ev = [float(i) for i in range(72)]
        for i in list(range(20, 29)):
            ev[i] = 5.0                                                              # the first tercile boundary falls on a value shared by several items
        for i in list(range(44, 53)):
            ev[i] = 8.0
        for i, s in enumerate(STATES):
            for side, off in (("main", 0), ("ext", 100)):
                rows = table(seed0 + off + i, drift=drift_by_state[s]); fp = table(seed0 + off)                                         # the FP columns are the same in every state
                for k, r in rows.items():
                    r["g_pred_fp"], r["dloss_FP_FP"] = fp[k]["g_pred_fp"], fp[k]["dloss_FP_FP"]; r["edit_norm_fp"] = fp[k]["edit_norm_fp"]
                if ordered_edits:
                    for n_, k in enumerate(sorted(k for k in rows if k[2] == "real")):
                        rows[k]["edit_norm_fp"] = ev[n_]; rows[k]["eligible"] = n_ % 3 != 0
                rep.write_csv(root / side / f"per_context_{s}.csv", [r for _, r in sorted(rows.items())])

    def test_the_driver_writes_one_row_per_corpus_state_comparison_and_group_with_the_threshold_the_counts_and_the_inputs_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); self.build(root, {"INT16": 0.0, "INT6": 0.0, "INT4": 1.0}); out = gc.run(root / "main", root / "ext", STATES, root / "out", n_boot=30, seed=1)
            groups = {"real_all", "real_eligible", "controls_all", "real_edit_tercile_1", "real_edit_tercile_2", "real_edit_tercile_3"}; self.assertEqual({r["group"] for r in out}, groups); self.assertEqual({r["corpus"] for r in out}, {"flores_main", "wikipedia_external"})
            self.assertEqual(len(out), 2 * 3 * 5 * 6)
            with (root / "out" / "gradient_vs_nll.csv").open() as fh:
                table_ = list(csv.DictReader(fh))
            self.assertEqual(len(table_), len(out)); js = json.loads((root / "out" / "gradient_vs_nll.json").read_text()); self.assertTrue(js["c0_identical_across_states"]["flores_main"]); self.assertEqual(len(js["inputs_sha256"]), 6)
            t16 = js["noise_threshold"]["flores_main"]; self.assertTrue(all(abs(float(r["noise_threshold"]) - t16) < 1e-12 for r in table_ if r["corpus"] == "flores_main"))
            # the three terciles partition the real items of a comparison
            for corp in ("flores_main", "wikipedia_external"):
                n = {g: [int(r["n"]) for r in table_ if r["corpus"] == corp and r["state"] == "INT6" and r["comparison"] == "C1" and r["group"] == g][0] for g in groups}; self.assertEqual(n["real_edit_tercile_1"] + n["real_edit_tercile_2"] + n["real_edit_tercile_3"], n["real_all"])

    def test_every_table_row_equals_the_direct_computation_on_its_group_with_independent_masks_and_the_int16_threshold(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); self.build(root, {"INT16": 0.0, "INT6": 0.3, "INT4": 1.0}, ordered_edits=True); out = gc.run(root / "main", root / "ext", STATES, root / "out", n_boot=40, seed=7)
            for corpus, side in (("flores_main", "main"), ("wikipedia_external", "ext")):
                tabs = {s: cc.rows_from_csv(root / side / f"per_context_{s}.csv") for s in STATES}; real16 = [abs(r["dloss_Q_Q"] - r["dloss_FP_FP"]) for r in tabs["INT16"].values() if r["kind"] == "real"]; t = float(np.percentile(real16, 95))
                for s in STATES:
                    rows = tabs[s]; edit = np.array([r["edit_norm_fp"] for r in rows.values() if r["kind"] == "real"]); q1, q2 = np.quantile(edit, [1 / 3, 2 / 3])
                    for cname in gc.COMPARISONS:
                        real = gc.comparison_arrays(rows, ("real",))[cname]; ctrl = gc.comparison_arrays(rows, ("matched_swapped", "random1", "random2", "random3"))[cname]; e = real["edit"]
                        masks = {"real_all": np.ones(len(e), bool), "real_eligible": real["eligible"], "real_edit_tercile_1": e <= q1, "real_edit_tercile_2": (e > q1) & (e <= q2), "real_edit_tercile_3": e > q2}
                        for g, m in masks.items():
                            want = gc.metrics_ci(real["p"][m], real["a"][m], real["feature"][m], real["cluster"][m], t, 40, 7); got = [r for r in out if (r["corpus"], r["state"], r["comparison"], r["group"]) == (corpus, s, cname, g)][0]
                            self.assertEqual((got["n"], got["n_above_noise"], got["n_features"], got["n_clusters"]), (want["n"], want["n_above_noise"], want["n_features"], want["n_clusters"])); self.assertAlmostEqual(got["noise_threshold"], t, places=12)
                            for k in gc.METRICS:
                                for suffix, key in (("", "point"), ("_low", "low"), ("_high", "high")):
                                    a_, b_ = got[k + suffix], want[k][key]; self.assertTrue((np.isnan(a_) and np.isnan(b_)) or abs(a_ - b_) < 1e-12, (corpus, s, cname, g, k, suffix, a_, b_))
                        want = gc.metrics_ci(ctrl["p"], ctrl["a"], ctrl["feature"], ctrl["cluster"], t, 40, 7); got = [r for r in out if (r["corpus"], r["state"], r["comparison"], r["group"]) == (corpus, s, cname, "controls_all")][0]
                        self.assertEqual(got["n"], want["n"]); self.assertAlmostEqual(got["slope_low"], want["slope"]["low"], places=12); self.assertAlmostEqual(got["slope_high"], want["slope"]["high"], places=12)

    def test_a_state_whose_quantized_model_drifts_loses_the_correlation_of_the_shift_while_the_rounding_reference_keeps_the_prediction(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); self.build(root, {"INT16": 0.0, "INT6": 0.0, "INT4": 0.0}); out = gc.run(root / "main", root / "ext", STATES, root / "out", n_boot=30, seed=1)
            c1 = [r for r in out if r["corpus"] == "flores_main" and r["comparison"] == "C1" and r["group"] == "real_all"]; self.assertTrue(all(float(r["slope"]) > 1.5 and float(r["slope"]) < 2.5 for r in c1)); self.assertTrue(all(float(r["r"]) > 0.9 for r in c1))

    def test_a_c0_that_differs_between_the_state_tables_is_reported_not_hidden(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); self.build(root, {"INT16": 0.0, "INT6": 0.0, "INT4": 0.0}); path = root / "main" / "per_context_INT4.csv"; recs = list(csv.DictReader(path.open())); recs[0]["dloss_FP_FP"] = str(float(recs[0]["dloss_FP_FP"]) + 1.0)
            with path.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(recs[0])); w.writeheader(); w.writerows(recs)
            gc.run(root / "main", root / "ext", STATES, root / "out", n_boot=10, seed=1); js = json.loads((root / "out" / "gradient_vs_nll.json").read_text()); self.assertFalse(js["c0_identical_across_states"]["flores_main"]); self.assertTrue(js["c0_identical_across_states"]["wikipedia_external"])


if __name__ == "__main__":
    unittest.main()
