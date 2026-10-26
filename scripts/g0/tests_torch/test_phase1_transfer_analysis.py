import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phase1_transfer_analysis as ta
import phase1_transfer_core as tc
import phase1_transfer_run as tr


def gram_of(vectors):
    """A stored item: the Gram of the five cell vectors (FP/FP, Q/FP, FP/Q, Q/Q, Q/Q-native) given as rows."""
    V = torch.tensor(vectors, dtype=torch.float64); return (V @ V.T).numpy()


def item(fp, qq, kind_gap=0.0, n_rows=2, **extra):
    """An item whose FP effect is the first unit vector times fp and whose Q/Q effect is (qq, kind_gap)."""
    vecs = [[fp, 0.0], [fp, 0.0], [fp, 0.0], [qq, kind_gap], [qq, kind_gap]]
    return {"gram": gram_of(vecs), "cells": list(tc.CELLS), "dloss": [0.1] * 5, "loss_base": [2.0] * 5, "n_rows": n_rows, "n_target_rows": 5, "n_active_q": 1, "n_positions": n_rows, **extra}


CORE = {"feature_ids": [10, 11, 12, 13], "feature_language_index": [0, 0, 1, 1], "direction_ids": [11, 10, 13, 12], "contexts": [[1000, 2000], [1001, 2001], [3000], [3001]]}
SPLIT = {"calibration": [10, 12], "heldout": [11, 13]}


class Chunks(unittest.TestCase):
    def test_chunk_files_are_merged_and_a_repeated_item_key_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            tr.atomic_save({"items": {(10, 1000, "real"): item(3, 3)}, "pooled": {(10, "real"): {"gram": np.eye(5), "n_rows": 2, "n_contexts": 1}}, "skipped": []}, d / "transfer_INT4_c000.pt", "k0")
            tr.atomic_save({"items": {(11, 1001, "real"): item(4, 2)}, "pooled": {}, "skipped": [((12, 3000, "real"), "no_fp_active_position")]}, d / "transfer_INT4_c001.pt", "k1")
            items, pooled, skipped = ta.load_chunks(d, "transfer", "INT4")
            self.assertEqual(set(items), {(10, 1000, "real"), (11, 1001, "real")}); self.assertEqual(len(pooled), 1); self.assertEqual(skipped, [((12, 3000, "real"), "no_fp_active_position")])
            tr.atomic_save({"items": {(10, 1000, "real"): item(3, 3)}, "pooled": {}, "skipped": []}, d / "transfer_INT4_c002.pt", "k2")
            with self.assertRaisesRegex(ValueError, "twice"):
                ta.load_chunks(d, "transfer", "INT4")
            with self.assertRaisesRegex(FileNotFoundError, "INT6"):
                ta.load_chunks(d, "transfer", "INT6")
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); pool = {(10, "real"): {"gram": np.eye(5), "n_rows": 2, "n_contexts": 1}}
            tr.atomic_save({"items": {}, "pooled": pool, "skipped": []}, d / "transfer_INT4_c000.pt", "k0"); tr.atomic_save({"items": {}, "pooled": pool, "skipped": []}, d / "transfer_INT4_c001.pt", "k1")
            with self.assertRaisesRegex(ValueError, "pool"):
                ta.load_chunks(d, "transfer", "INT4")


class Table(unittest.TestCase):
    def test_a_row_per_item_with_the_norms_the_split_and_the_source_cluster(self):
        items = {(10, 1000, "real"): item(3.0, 5.0, 4.0), (11, 2001, "matched_swapped"): item(4.0, 4.0)}
        rows = ta.item_table(items, CORE, SPLIT, cluster_mod=997)
        a, b = rows[(10, 1000, "real")], rows[(11, 2001, "matched_swapped")]
        self.assertAlmostEqual(a["fp_norm"], 3.0); self.assertAlmostEqual(a["error_norm"], float(np.hypot(5.0 - 3.0, 4.0))); self.assertEqual((a["split"], b["split"]), ("calibration", "heldout"))
        self.assertEqual((a["cluster"], b["cluster"]), (1000 % 997, 2001 % 997)); self.assertEqual((a["language_index"], b["language_index"]), (0, 0)); self.assertEqual(b["kind"], "matched_swapped")
        c = ta.item_table({(12, 3000, "real"): item(1, 1)}, CORE, SPLIT, 997)[(12, 3000, "real")]; self.assertEqual((c["language_index"], c["split"]), (1, "calibration"))
        self.assertAlmostEqual(b["error_norm"], 0.0); self.assertAlmostEqual(b["cosine"], 1.0); self.assertAlmostEqual(a["norm_ratio"], float(np.hypot(5.0, 4.0) / 3.0))

    def test_a_feature_outside_the_split_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "split"):
            ta.item_table({(99, 1000, "real"): item(1, 1)}, CORE, SPLIT, 997)

    def test_the_components_norms_come_from_the_gram(self):
        rows = ta.item_table({(10, 1000, "real"): item(3.0, 5.0, 4.0)}, CORE, SPLIT, 997); r = rows[(10, 1000, "real")]
        n = tc.decomposition_norms(torch.tensor(item(3.0, 5.0, 4.0)["gram"]))
        for k in ("C", "W", "I", "total"):
            self.assertAlmostEqual(r[f"{k}_norm"], n[k])


class TableExtras(unittest.TestCase):
    """Quantities of the hypotheses and of the baselines that the stored Gram and the scalars give exactly (H1: how much of the error a rescaling or the native amplitude removes; baseline A: activation cosine, firing; gradient scalars)."""
    @staticmethod
    def row(vecs, **extra):
        it = {"gram": gram_of(vecs), "cells": list(tc.CELLS), "dloss": [0.1] * 5, "loss_base": [2.0] * 5, "n_rows": 2, "n_target_rows": 5, "n_active_q": 1, "n_positions": 2, **extra}
        return ta.item_table({(10, 1000, "real"): it}, CORE, SPLIT, 997)[(10, 1000, "real")]

    def test_the_native_error_and_the_error_left_after_the_best_rescaling_of_the_fp_effect(self):
        r = self.row([[3.0, 0.0], [3.0, 0.0], [3.0, 0.0], [5.0, 4.0], [4.0, 2.0]])
        self.assertAlmostEqual(r["error_norm"], float(np.hypot(2.0, 4.0))); self.assertAlmostEqual(r["native_error_norm"], float(np.hypot(1.0, 2.0)))
        self.assertAlmostEqual(r["scalar_residual_norm"], 4.0); self.assertAlmostEqual(r["native_scalar_residual_norm"], 2.0)                                      # the component of the effect orthogonal to the FP effect

    def test_a_zero_fp_effect_has_no_rescaling_residual(self):
        r = self.row([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        self.assertIsNone(r["scalar_residual_norm"]); self.assertIsNone(r["native_scalar_residual_norm"]); self.assertAlmostEqual(r["error_norm"], float(np.sqrt(2.0)))

    def test_the_activation_cosine_is_the_cosine_of_the_amplitude_vectors_zero_when_nothing_fires_and_the_firing_share_comes_from_the_counts(self):
        base = [[1.0, 0.0]] * 5
        self.assertAlmostEqual(self.row(base, amp_fp=[1.0, 2.0], amp_q=[2.0, 4.0], n_active_q=1, n_positions=2)["act_cos"], 1.0)
        self.assertAlmostEqual(self.row(base, amp_fp=[1.0, 0.0], amp_q=[0.0, 1.0])["act_cos"], 0.0); self.assertAlmostEqual(self.row(base, amp_fp=[3.0, 4.0], amp_q=[4.0, 3.0])["act_cos"], 24.0 / 25.0)
        self.assertEqual(self.row(base, amp_fp=[1.0], amp_q=[0.0])["act_cos"], 0.0); self.assertAlmostEqual(self.row(base, n_active_q=1, n_positions=4)["firing"], 0.25); self.assertNotIn("act_cos", self.row(base))

    def test_the_gradient_scalars_pass_through_and_the_shift_is_the_change_of_the_first_order_loss_effect(self):
        r = self.row([[1.0, 0.0]] * 5, g_pred_fp=0.5, g_pred_q_fixed=0.75, g_pred_q_native=1.25, g_norm_fp=2.0, g_norm_q=3.0)
        self.assertEqual((r["g_pred_fp"], r["g_pred_q_fixed"], r["g_pred_q_native"], r["g_norm_fp"], r["g_norm_q"]), (0.5, 0.75, 1.25, 2.0, 3.0)); self.assertAlmostEqual(r["g_shift_native"], 0.75); self.assertAlmostEqual(r["g_shift_fixed"], 0.25)
        self.assertNotIn("g_shift_native", self.row([[1.0, 0.0]] * 5))


class Floor(unittest.TestCase):
    def rows(self, errs, kinds=("real",), split="calibration"):
        out = {}
        for i, e in enumerate(errs):
            for k in kinds:
                out[(i, 0, k)] = {"kind": k, "split": split, "error_norm": e, "fp_norm": 10.0}
        return out

    def test_the_floor_is_the_95th_percentile_of_the_int16_error_over_the_calibration_items_of_each_kind(self):
        errs = list(range(1, 101)); f = ta.floors_by_kind(self.rows(errs))
        self.assertAlmostEqual(f["real"], float(np.percentile(errs, 95)))

    def test_held_out_items_never_enter_the_floor(self):
        rows = self.rows(list(range(1, 101))); held = self.rows([1000.0] * 50, split="heldout"); held = {(k[0] + 1000, k[1], k[2]): v for k, v in held.items()}
        f = ta.floors_by_kind({**rows, **held}); self.assertAlmostEqual(f["real"], float(np.percentile(list(range(1, 101)), 95)))
        with self.assertRaisesRegex(ValueError, "calibration"):
            ta.floors_by_kind(held)

    def test_each_direction_kind_has_its_own_floor(self):
        rows = {**self.rows([1.0] * 20, kinds=("real",)), **{(k[0] + 100, k[1], k[2]): v for k, v in self.rows([9.0] * 20, kinds=("random1",)).items()}}
        f = ta.floors_by_kind(rows); self.assertAlmostEqual(f["real"], 1.0); self.assertAlmostEqual(f["random1"], 9.0)

    def test_eligibility_is_fp_norm_above_five_times_the_floor_and_the_normalized_error_adds_the_floor(self):
        rows = {("a", 0, "real"): {"kind": "real", "split": "heldout", "fp_norm": 51.0, "error_norm": 10.0, "language_index": 0}, ("b", 0, "real"): {"kind": "real", "split": "heldout", "fp_norm": 50.0, "error_norm": 10.0, "language_index": 0}}
        out = ta.apply_floor(rows, {"real": 10.0}); self.assertTrue(out[("a", 0, "real")]["eligible"]); self.assertFalse(out[("b", 0, "real")]["eligible"])
        self.assertAlmostEqual(out[("a", 0, "real")]["normalized_error"], 10.0 / (51.0 + 10.0)); self.assertAlmostEqual(out[("a", 0, "real")]["error_over_floor"], 1.0)

    def test_the_coverage_table_counts_every_item_and_every_eligible_item_by_split_and_kind(self):
        rows = {(i, 0, "real"): {"kind": "real", "split": "heldout" if i % 2 else "calibration", "fp_norm": 60.0 if i < 3 else 10.0, "error_norm": 1.0, "language_index": 0} for i in range(10)}
        cov = ta.coverage(ta.apply_floor(rows, {"real": 10.0}))
        self.assertEqual(cov[("real", "calibration")], {"items": 5, "eligible": 2}); self.assertEqual(cov[("real", "heldout")], {"items": 5, "eligible": 1})


def gitem(d, ntok=10):
    """A group-level item with the loss changes d of the five cells."""
    return {"gram": np.eye(5), "cells": list(tc.CELLS), "dloss": list(d), "loss_base": [2.0] * 5, "n_rows": 3, "n_target_rows": ntok, "n_active_q": 1, "n_positions": 3}


def group_items_for(langs, n_sent, group, controls, ntok=lambda l, j: 10):
    """group: function (lang, j) -> five loss changes of the group; controls: list of such functions (one per draw)."""
    out = {}
    for l in langs:
        for j in range(n_sent):
            out[(l, j, "group")] = gitem(group(l, j), ntok(l, j))
            for r, c in enumerate(controls):
                out[(l, j, f"control{r}")] = gitem(c(l, j), ntok(l, j))
    return out


LANGS2 = ["a", "b"]


class GroupLevel(unittest.TestCase):
    def test_arrays_have_one_slot_per_language_sentence_set_and_cell_and_mark_missing_sets(self):
        items = group_items_for(LANGS2, 3, lambda l, j: [1, 2, 3, 4, 5], [lambda l, j: [0.1] * 5, lambda l, j: [0.2] * 5])
        del items[("b", 1, "control1")]
        A = ta.group_arrays(items, LANGS2, 3, 2)
        self.assertEqual(A["dloss"].shape, (2, 3, 3, 5)); self.assertEqual(A["ntok"].shape, (2, 3)); self.assertEqual(A["valid"].tolist(), [[True, True, True], [True, False, True]])
        self.assertTrue(np.isnan(A["dloss"][1, 1, 2]).all()); self.assertEqual(float(A["dloss"][0, 2, 1, 4]), 0.1)

    def test_the_net_harm_is_the_group_minus_the_mean_of_the_controls_pooled_by_token_count(self):
        group = lambda l, j: [1.0 + j] * 5; ctrl = [lambda l, j: [0.2] * 5, lambda l, j: [0.4] * 5]
        A = ta.group_arrays(group_items_for(["a"], 2, group, ctrl, ntok=lambda l, j: 10 * (j + 1)), ["a"], 2, 2)
        pool = ta.pooled_harms(A)                                                                                           # token weights 10 and 20
        self.assertAlmostEqual(pool["group"][0, 0], (10 * 1.0 + 20 * 2.0) / 30); self.assertAlmostEqual(pool["control"][0, 0], 0.3); self.assertAlmostEqual(pool["net"][0, 0], (10 * 1.0 + 20 * 2.0) / 30 - 0.3)

    def test_a_sentence_with_a_missing_set_is_left_out_of_the_net_and_counted(self):
        items = group_items_for(["a"], 2, lambda l, j: [1.0 + 9 * j] * 5, [lambda l, j: [0.0] * 5]); del items[("a", 1, "control0")]
        A = ta.group_arrays(items, ["a"], 2, 1); pool = ta.pooled_harms(A)
        self.assertAlmostEqual(pool["net"][0, 0], 1.0); self.assertEqual(pool["n_sentences"].tolist(), [1])

    def test_the_decomposition_of_the_net_harm_has_the_identity_and_the_native_difference(self):
        net = np.array([[0.9, 1.9, 0.7, 1.6, 0.6, ]]); dec = ta.decompose_net(net)
        self.assertAlmostEqual(dec["C"][0], 1.0); self.assertAlmostEqual(dec["W"][0], -0.2); self.assertAlmostEqual(dec["I"][0], 1.6 - 1.9 - 0.7 + 0.9); self.assertAlmostEqual(dec["total"][0], 0.7)
        self.assertAlmostEqual(dec["C"][0] + dec["W"][0] + dec["I"][0], dec["total"][0]); self.assertAlmostEqual(dec["native_minus_fixed"][0], 0.6 - 1.6); self.assertAlmostEqual(dec["native_total"][0], 0.6 - 0.9)

    def test_the_a3_summary_counts_languages_and_takes_the_median_of_the_paired_difference(self):
        dec = {"C": np.array([1.0, -2.0, 0.1]), "W": np.array([0.5, 0.5, -0.4])}                                         # |C| - |W| = 0.5, 1.5, -0.3
        s = ta.a3_summary(dec)
        self.assertEqual(s["n_languages"], 3); self.assertEqual(s["c_gt_w"], 2); self.assertAlmostEqual(s["median_abs_c"], 1.0); self.assertAlmostEqual(s["median_abs_w"], 0.5); self.assertAlmostEqual(s["median_difference"], 0.5); self.assertTrue(s["holds"])
        dec["C"] = np.array([0.1, 0.2, 0.1]); self.assertFalse(ta.a3_summary(dec)["holds"])

    def test_the_bootstrap_resamples_source_sentences_jointly_across_languages_and_is_seeded(self):
        rng = np.random.default_rng(0); eff = rng.normal(size=(2, 30))
        group = lambda l, j: [1.0, 1.0 + eff[LANGS2.index(l), j], 1.0, 1.0, 1.0]
        A = ta.group_arrays(group_items_for(LANGS2, 30, group, [lambda l, j: [0.0] * 5]), LANGS2, 30, 1)
        r1 = ta.bootstrap_a3(A, n_boot=200, seed=3); r2 = ta.bootstrap_a3(A, n_boot=200, seed=3); r3 = ta.bootstrap_a3(A, n_boot=200, seed=4)
        self.assertEqual(r1, r2); self.assertNotEqual(r1["median_difference"], r3["median_difference"])
        lo, hi = r1["median_difference"]; self.assertLessEqual(lo, hi)
        const = ta.group_arrays(group_items_for(LANGS2, 30, lambda l, j: [1.0, 2.0, 1.5, 1.7, 1.0], [lambda l, j: [0.0] * 5]), LANGS2, 30, 1)
        c = ta.bootstrap_a3(const, n_boot=50, seed=0); self.assertEqual(c["median_difference"][0], c["median_difference"][1])           # nothing varies over sentences: a degenerate interval

    def test_languages_are_resampled_with_the_same_sentence_multiplicities(self):
        A = ta.group_arrays(group_items_for(LANGS2, 4, lambda l, j: [1.0, 1.0 + j, 1.0, 1.0, 1.0], [lambda l, j: [0.0] * 5]), LANGS2, 4, 1)
        m = np.array([0, 2, 1, 1]); pool = ta.pooled_harms(A, mult=m)
        want = (0 * 1.0 + 2 * 1.0 + 1 * 1.0 + 1 * 1.0) / 4                                                                 # cell 0 is 1.0 everywhere, cell 1 is 1 + j: pooled over the multiplicities
        self.assertAlmostEqual(pool["net"][0, 0], want); self.assertAlmostEqual(pool["net"][1, 1], (2 * 2.0 + 1 * 3.0 + 1 * 4.0) / 4)

    def test_items_with_other_cells_or_another_token_count_than_the_group_are_errors(self):
        items = group_items_for(["a"], 2, lambda l, j: [1.0] * 5, [lambda l, j: [0.0] * 5]); bad = dict(items); bad[("a", 0, "group")] = {**items[("a", 0, "group")], "cells": ["FP/FP", "Q/FP", "FP/Q", "Q/Q-native", "Q/Q"]}
        with self.assertRaises(ValueError):
            ta.group_arrays(bad, ["a"], 2, 1)
        other = dict(items); other[("a", 1, "control0")] = {**items[("a", 1, "control0")], "n_target_rows": 11}
        with self.assertRaises(ValueError):
            ta.group_arrays(other, ["a"], 2, 1)

    def test_the_net_harm_needs_a_control_draw(self):
        A = ta.group_arrays(group_items_for(["a"], 2, lambda l, j: [1.0] * 5, []), ["a"], 2, 0)
        with self.assertRaises(ValueError):
            ta.pooled_harms(A)

    def test_a3_ties_do_not_count_for_the_context_term_and_a_zero_median_difference_does_not_hold(self):
        s = ta.a3_summary({"C": np.array([1.0, -1.0]), "W": np.array([1.0, 1.0])})
        self.assertEqual(s["c_gt_w"], 0); self.assertEqual(s["median_difference"], 0.0); self.assertFalse(s["holds"])

    def test_the_interval_is_the_percentile_interval_of_the_seeded_replicates_replayed_independently(self):
        rng = np.random.default_rng(5); c, w = rng.normal(size=12), rng.normal(size=12)                                  # one language: C and W come from the sentence-level loss changes of cells 1 and 2
        group = lambda l, j: [0.0, c[j], w[j], 0.0, 0.0]
        A = ta.group_arrays(group_items_for(["a"], 12, group, [lambda l, j: [0.0] * 5]), ["a"], 12, 1)
        res = ta.bootstrap_a3(A, n_boot=300, seed=9, level=90); r = np.random.default_rng(9); diffs, holds = [], []
        for _ in range(300):
            m = np.bincount(r.integers(0, 12, 12), minlength=12); d = abs((m * c).sum() / m.sum()) - abs((m * w).sum() / m.sum()); diffs.append(d); holds.append(d > 0)
        lo, hi = np.percentile(diffs, [5, 95]); self.assertAlmostEqual(res["median_difference"][0], lo); self.assertAlmostEqual(res["median_difference"][1], hi)
        self.assertAlmostEqual(res["share_holds"], float(np.mean(holds))); self.assertTrue(0 < res["share_holds"] < 1); self.assertEqual((res["n_boot"], res["seed"], res["level"]), (300, 9, 90))


class CrossedBootstrap(unittest.TestCase):
    def test_the_weighted_median_is_the_smallest_value_with_half_of_the_weight_below_it(self):
        self.assertEqual(ta.weighted_median(np.array([4.0, 1.0, 3.0, 2.0]), np.ones(4)), 2.0)                              # lower median of 1, 2, 3, 4, whatever the order
        self.assertEqual(ta.weighted_median(np.array([1.0, 2.0, 3.0, 4.0]), np.array([0.0, 0.0, 5.0, 1.0])), 3.0); self.assertEqual(ta.weighted_median(np.array([1.0, 2.0, 3.0, 4.0]), np.array([3.0, 0.0, 0.0, 3.0])), 1.0)
        self.assertTrue(np.isnan(ta.weighted_median(np.array([1.0, 2.0]), np.zeros(2)))); self.assertEqual(ta.weighted_median(np.array([7.0]), np.array([2.0])), 7.0)

    def test_the_multiplicities_are_drawn_per_feature_and_per_cluster_with_the_total_preserved_and_are_seeded(self):
        a = ta.bootstrap_multiplicities(5, 7, 50, seed=1); b = ta.bootstrap_multiplicities(5, 7, 50, seed=1); c = ta.bootstrap_multiplicities(5, 7, 50, seed=2)
        self.assertEqual(a[0].shape, (50, 5)); self.assertEqual(a[1].shape, (50, 7)); self.assertTrue((a[0].sum(1) == 5).all()); self.assertTrue((a[1].sum(1) == 7).all())
        self.assertTrue(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])); self.assertFalse(np.array_equal(a[0], c[0]))

    def test_a_row_weighs_the_product_of_the_multiplicity_of_its_feature_and_of_its_cluster(self):
        values = np.array([1.0, 2.0, 3.0, 10.0]); fidx = np.array([0, 0, 1, 1]); cidx = np.array([0, 1, 0, 1])
        mf = np.array([[3, 0], [1, 1]]); mc = np.array([[1, 1], [0, 2]])                                                    # replicate 0: weights 3, 3, 0, 0; replicate 1: weights 0, 1*2, 0, 1*2
        self.assertTrue(np.allclose(ta.replicate_stats(values, fidx, cidx, mf, mc, "mean"), [(3 * 1 + 3 * 2) / 6, (2 * 2 + 10 * 2) / 4]))
        self.assertTrue(np.allclose(ta.replicate_stats(values, fidx, cidx, mf, mc, "median"), [1.0, 2.0]))
        with self.assertRaises(ValueError):
            ta.replicate_stats(values, fidx, cidx, mf, mc, "mode")

    def test_the_interval_follows_the_clusters_not_the_number_of_rows(self):
        rng = np.random.default_rng(0); n_f, n_c = 40, 20; eff = rng.normal(size=n_c)                                       # the value of a row is the effect of its source cluster
        fidx = np.repeat(np.arange(n_f), n_c); cidx = np.tile(np.arange(n_c), n_f); values = eff[cidx]
        r = ta.crossed_bootstrap(values, fidx, cidx, "mean", n_boot=400, seed=3)
        self.assertAlmostEqual(r["point"], float(values.mean())); self.assertGreater(r["high"] - r["low"], 0.5); self.assertEqual(r["n"], n_f * n_c)       # iid rows would give a width of about 0.15

    def test_constant_values_give_a_degenerate_interval_and_nan_values_are_dropped_and_counted(self):
        r = ta.crossed_bootstrap(np.full(6, 2.5), np.array([0, 0, 1, 1, 2, 2]), np.array([0, 1, 0, 1, 0, 1]), "median", n_boot=50, seed=0); self.assertEqual((r["low"], r["point"], r["high"]), (2.5, 2.5, 2.5))
        v = np.array([1.0, np.nan, 3.0, 5.0]); r = ta.crossed_bootstrap(v, np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]), "mean", n_boot=20, seed=0); self.assertEqual((r["n"], r["n_missing"]), (3, 1)); self.assertAlmostEqual(r["point"], 3.0)
        e = ta.crossed_bootstrap(np.array([]), np.array([], dtype=int), np.array([], dtype=int), "mean", n_boot=20, seed=0); self.assertEqual(e["n"], 0); self.assertTrue(np.isnan(e["point"]))

    def test_the_interval_is_the_percentile_interval_of_the_replicates_replayed_independently_and_the_point_follows_the_statistic(self):
        rng = np.random.default_rng(2); v = rng.exponential(size=60); f = np.repeat(np.arange(6), 10); c = np.tile(np.arange(10), 6)
        for stat in ("mean", "median"):
            r = ta.crossed_bootstrap(v, f, c, stat, n_boot=300, seed=11, level=90); mf, mc = ta.bootstrap_multiplicities(6, 10, 300, 11); reps = ta.replicate_stats(v, f, c, mf, mc, stat)
            self.assertAlmostEqual(r["low"], float(np.percentile(reps, 5))); self.assertAlmostEqual(r["high"], float(np.percentile(reps, 95))); self.assertEqual((r["n_features"], r["n_clusters"], r["level"], r["stat"]), (6, 10, 90, stat))
            self.assertAlmostEqual(r["point"], float(v.mean()) if stat == "mean" else ta.weighted_median(v, np.ones(60)))
        self.assertAlmostEqual(ta.crossed_bootstrap(np.array([1.0, 2.0, 10.0]), np.array([0, 1, 2]), np.array([0, 0, 0]), "median", n_boot=10, seed=0)["point"], 2.0)

    def test_the_default_replicates_and_seed_are_the_frozen_ones_and_the_same_seed_gives_the_same_interval(self):
        self.assertEqual((ta.BOOTSTRAP_REPLICATES, ta.BOOTSTRAP_SEED), (2000, 20261026))
        v = np.arange(12.0); f = np.repeat(np.arange(3), 4); c = np.tile(np.arange(4), 3)
        self.assertEqual(ta.crossed_bootstrap(v, f, c, "mean", n_boot=100, seed=5), ta.crossed_bootstrap(v, f, c, "mean", n_boot=100, seed=5))
        self.assertNotEqual(ta.crossed_bootstrap(v, f, c, "mean", n_boot=100, seed=5), ta.crossed_bootstrap(v, f, c, "mean", n_boot=100, seed=6))


def floored(specs, floors):
    """apply_floor(item_table(...)) of items {(feature, sentence, kind): item}."""
    return ta.apply_floor(ta.item_table(specs, CORE, SPLIT, 997), floors)


def find(records, **kw):
    hit = [r for r in records if all(r[k] == v for k, v in kw.items())]
    assert len(hit) == 1, (kw, len(hit)); return hit[0]


class ComponentSummary(unittest.TestCase):
    def setUp(self):
        # floor 1: eligible means fp > 5. rows: (10, 1000) fp 10, error 0; (10, 2000) fp 10, error 3; (11, 1001) fp 10, error 4 (heldout); (12, 3000) fp 4 (not eligible)
        self.rows = floored({(10, 1000, "real"): item(10, 10), (10, 2000, "real"): item(10, 13), (11, 1001, "real"): item(10, 10, 4.0), (12, 3000, "real"): item(4, 4, 2.0), (10, 1000, "random1"): item(10, 10, 1.0)}, {"real": 1.0, "random1": 1.0})

    def test_the_summary_covers_every_kind_and_split_with_the_eligible_items_only(self):
        rec = ta.component_summary(self.rows, n_boot=40, seed=1)
        a = find(rec, kind="real", split="all", metric="normalized_error"); self.assertEqual(a["n"], 3); self.assertEqual(a["n_items"], 4); self.assertAlmostEqual(a["point"], 3.0 / 11.0)           # lower median of 0, 3/11, 4/11
        self.assertEqual(find(rec, kind="real", split="calibration", metric="normalized_error")["n"], 2); self.assertEqual(find(rec, kind="real", split="heldout", metric="normalized_error")["n"], 1)
        self.assertEqual(find(rec, kind="random1", split="all", metric="normalized_error")["n"], 1); self.assertEqual(find(rec, kind="real", split="all", metric="normalized_error")["stat"], "median")

    def test_ineligible_items_enter_when_the_eligibility_rule_is_switched_off_and_n_items_always_counts_them(self):
        rec = ta.component_summary(self.rows, n_boot=40, seed=1, eligible_only=False); a = find(rec, kind="real", split="all", metric="normalized_error")
        self.assertEqual((a["n"], a["n_items"]), (4, 4))

    def test_the_components_are_relative_to_the_fp_norm_plus_the_floor_and_c_versus_w_is_a_share(self):
        rec = ta.component_summary(self.rows, n_boot=40, seed=1); r = self.rows[(11, 1001, "real")]
        h = find(rec, kind="real", split="heldout", metric="C_rel"); self.assertAlmostEqual(h["point"], r["C_norm"] / (r["fp_norm"] + 1.0))
        self.assertAlmostEqual(find(rec, kind="real", split="heldout", metric="native_normalized_error")["point"], r["native_error_norm"] / (r["fp_norm"] + 1.0))
        share = find(rec, kind="real", split="all", metric="c_gt_w"); self.assertEqual(share["stat"], "mean"); self.assertAlmostEqual(share["point"], float(np.mean([self.rows[k]["C_norm"] > self.rows[k]["W_norm"] for k in self.rows if k[2] == "real" and self.rows[k]["eligible"]])))

    def test_the_native_error_the_interaction_share_and_the_unit_counts_come_from_the_right_fields(self):
        mine = {"gram": gram_of([[10.0, 0.0], [10.0, 1.0], [10.0, 2.0], [10.0, 4.0], [10.0, 1.0]]), "cells": list(tc.CELLS), "dloss": [0.1] * 5, "loss_base": [2.0] * 5, "n_rows": 2, "n_target_rows": 5, "n_active_q": 1, "n_positions": 2}
        rows = floored({(10, 1000, "real"): mine, (10, 1997, "real"): mine, (11, 1001, "real"): mine}, {"real": 1.0}); rec = ta.component_summary(rows, n_boot=20, seed=1); r = rows[(10, 1000, "real")]
        self.assertAlmostEqual(find(rec, kind="real", split="all", metric="native_normalized_error")["point"], 1.0 / 11.0); self.assertAlmostEqual(find(rec, kind="real", split="all", metric="normalized_error")["point"], 4.0 / 11.0)
        self.assertAlmostEqual(find(rec, kind="real", split="all", metric="interaction_share")["point"], r["I_norm"] / (r["C_norm"] + r["W_norm"]))
        a = find(rec, kind="real", split="all", metric="normalized_error"); self.assertEqual((a["n_features"], a["n_clusters"]), (2, 2))                          # sentences 1000 and 1997 share the source cluster 3; 1001 is cluster 4

    def test_h1_shares_are_the_error_left_after_a_rescaling_and_after_the_native_amplitude_relative_to_the_error(self):
        mine = {"gram": gram_of([[10.0, 0.0], [10.0, 1.0], [10.0, 2.0], [10.0, 4.0], [10.0, 1.0]]), "cells": list(tc.CELLS), "dloss": [0.1] * 5, "loss_base": [2.0] * 5, "n_rows": 2, "n_target_rows": 5, "n_active_q": 1, "n_positions": 2}
        same = {**mine, "gram": gram_of([[10.0, 0.0], [10.0, 0.0], [10.0, 0.0], [10.0, 0.0], [10.0, 0.0]])}                                      # no error at all: the shares are undefined
        rows = floored({(10, 1000, "real"): mine, (10, 2000, "real"): same}, {"real": 1.0}); rec = ta.component_summary(rows, n_boot=20, seed=1, eligible_only=False)
        s = find(rec, kind="real", split="all", metric="scalar_residual_share"); n = find(rec, kind="real", split="all", metric="native_error_share")
        self.assertAlmostEqual(s["point"], 4.0 / 4.0); self.assertAlmostEqual(n["point"], 1.0 / 4.0); self.assertEqual((s["n"], s["n_missing"], n["n"], n["n_missing"]), (1, 1, 1, 1))

    def test_a_metric_that_is_undefined_for_an_item_is_counted_as_missing(self):
        rows = floored({(10, 1000, "real"): item(10, 10), (10, 2000, "real"): {**item(10, 10), "gram": gram_of([[0, 0], [0, 0], [0, 0], [1, 1], [0, 1]])}}, {"real": 0.0001})
        a = find(ta.component_summary(rows, n_boot=20, seed=1, eligible_only=False), kind="real", split="all", metric="scalar_normalized_error"); self.assertEqual((a["n"], a["n_missing"]), (1, 1))


def pool_gram(ff, qq, native=None):
    """The Gram of an averaged fingerprint: FP effect `ff`, Q/Q effect `qq` (the Q/FP and FP/Q cells copy the nearer end, the native cell the Q/Q cell)."""
    return {"gram": gram_of([ff, ff, ff, qq, native if native is not None else qq]), "n_rows": 2, "n_contexts": 2}


class AveragedFingerprints(unittest.TestCase):
    """H4: the transfer of the averaged fingerprint of a feature against the transfer of its single contexts, on the SAME feature-context pairs."""
    def setUp(self):
        two = lambda a, b, c, d: {(10, 1000, "real"): {**item(1, 1), "gram": gram_of([a, a, a, b, b])}, (10, 2000, "real"): {**item(1, 1), "gram": gram_of([c, c, c, d, d])}}
        # feature 10: contexts FP (1,0) and (0,1), quantized (0,1) and (1,0): each context has cosine 0, the average (.5,.5) is identical in both states
        self.items = two([1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0])
        self.pooled = {(10, "real"): pool_gram([0.5, 0.5], [0.5, 0.5])}
        self.rows = floored(self.items, {"real": 1.0})

    def test_the_averaged_fingerprint_can_look_identical_while_every_context_is_orthogonal(self):
        h = ta.averaged_fingerprints(self.rows, self.pooled)[(10, "real")]
        self.assertAlmostEqual(h["pooled_cos"], 1.0); self.assertAlmostEqual(h["item_cos"], 0.0); self.assertAlmostEqual(h["cos_gap"], 1.0); self.assertAlmostEqual(h["pooled_error"], 0.0)
        self.assertAlmostEqual(h["pooled_normalized_error"], 0.0); self.assertAlmostEqual(h["item_normalized_error"], float(np.sqrt(2.0)) / (1.0 + 1.0)); self.assertEqual(h["n_contexts"], 2)

    def test_opposing_contexts_give_a_zero_mean_cloud_with_large_context_effects(self):
        items = {(10, 1000, "real"): {**item(1, 1), "gram": gram_of([[3.0, 0.0]] * 5)}, (10, 2000, "real"): {**item(1, 1), "gram": gram_of([[-3.0, 0.0]] * 5)}}
        h = ta.averaged_fingerprints(floored(items, {"real": 1.0}), {(10, "real"): pool_gram([0.0, 0.0], [0.0, 0.0])})[(10, "real")]
        self.assertIsNone(h["pooled_cos"]); self.assertIsNone(h["cos_gap"]); self.assertAlmostEqual(h["coherence_fp"], 0.0); self.assertAlmostEqual(h["item_cos"], 1.0)

    def test_coherence_is_the_norm_of_the_average_over_the_average_norm_weighted_by_rows(self):
        h = ta.averaged_fingerprints(self.rows, self.pooled)[(10, "real")]
        self.assertAlmostEqual(h["coherence_fp"], float(np.hypot(0.5, 0.5)) / 1.0); self.assertAlmostEqual(h["coherence_qq"], float(np.hypot(0.5, 0.5)))
        items = {(10, 1000, "real"): {**item(1, 1), "gram": gram_of([[2.0, 0.0]] * 5), "n_rows": 3}, (10, 2000, "real"): {**item(1, 1), "gram": gram_of([[1.0, 0.0]] * 5), "n_rows": 1}}      # 3 rows of 2 and 1 row of 1: average (7/4), norms 7/4: coherent
        g = ta.averaged_fingerprints(floored(items, {"real": 1.0}), {(10, "real"): {"gram": gram_of([[1.75, 0.0]] * 5), "n_rows": 4, "n_contexts": 2}})[(10, "real")]
        self.assertAlmostEqual(g["coherence_fp"], 1.0); self.assertAlmostEqual(g["item_cos"], 1.0)

    def test_a_tie_between_the_pooled_and_the_context_error_does_not_count_for_the_average_and_the_interval_resamples_features_only(self):
        mk = lambda f, pe, ie, gap: {"feature": f, "kind": "real", "cos_gap": gap, "coherence_fp": 0.5, "coherence_qq": 0.5, "pooled_normalized_error": pe, "item_normalized_error": ie, "pooled_cos": 0.1, "item_cos": 0.1}
        self.assertEqual(find(ta.averaged_summary({(1, "real"): mk(1, 0.2, 0.2, 0.0)}, n_boot=10, seed=0), kind="real", metric="averaged_looks_better")["point"], 0.0)
        rng = np.random.default_rng(4); gaps = rng.normal(size=14); hs = {(f, "real"): mk(f, 0.1, 0.2, float(gaps[f])) for f in range(14)}
        r = find(ta.averaged_summary(hs, n_boot=60, seed=7), kind="real", metric="cos_gap"); ref = ta.crossed_bootstrap(gaps, np.arange(14), np.zeros(14, dtype=int), "median", 60, 7)
        self.assertEqual((r["low"], r["high"], r["point"]), (ref["low"], ref["high"], ref["point"]))

    def test_the_pool_is_read_at_the_fixed_q_q_cell_and_the_pooled_error_is_normalized_by_the_pooled_fp_norm_plus_the_floor(self):
        items = {(10, 1000, "real"): {**item(1, 1), "gram": gram_of([[2.0, 0.0], [2.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 0.0]])}}
        h = ta.averaged_fingerprints(floored(items, {"real": 1.0}), {(10, "real"): {"gram": gram_of([[2.0, 0.0], [2.0, 0.0], [2.0, 0.0], [2.0, 2.0], [5.0, 5.0]]), "n_rows": 2, "n_contexts": 1}})[(10, "real")]
        self.assertAlmostEqual(h["pooled_error"], 2.0); self.assertAlmostEqual(h["pooled_normalized_error"], 2.0 / 3.0); self.assertAlmostEqual(h["pooled_cos"], float(4.0 / (2.0 * np.hypot(2.0, 2.0)))); self.assertAlmostEqual(h["item_normalized_error"], 2.0 / 3.0)

    def test_coherence_in_each_state_uses_the_norms_of_that_state(self):
        items = {(10, 1000, "real"): {**item(1, 1), "gram": gram_of([[2.0, 0.0]] * 3 + [[0.0, 4.0]] * 2)}, (10, 2000, "real"): {**item(1, 1), "gram": gram_of([[2.0, 0.0]] * 3 + [[0.0, -4.0]] * 2)}}
        h = ta.averaged_fingerprints(floored(items, {"real": 1.0}), {(10, "real"): {"gram": gram_of([[2.0, 0.0]] * 3 + [[0.0, 0.0]] * 2), "n_rows": 4, "n_contexts": 2}})[(10, "real")]
        self.assertAlmostEqual(h["coherence_fp"], 1.0); self.assertAlmostEqual(h["coherence_qq"], 0.0)

    def test_the_context_mean_is_weighted_by_rows_and_a_missing_pool_or_a_count_mismatch_is_an_error(self):
        items = {(10, 1000, "real"): {**item(1, 1), "gram": gram_of([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]), "n_rows": 3}, (10, 2000, "real"): {**item(1, 1), "gram": gram_of([[1.0, 0.0]] * 5), "n_rows": 1}}
        pool = {"gram": gram_of([[1.0, 0.0]] * 5), "n_rows": 4, "n_contexts": 2}
        self.assertAlmostEqual(ta.averaged_fingerprints(floored(items, {"real": 1.0}), {(10, "real"): pool})[(10, "real")]["item_cos"], (3 * 0.0 + 1 * 1.0) / 4)
        with self.assertRaisesRegex(ValueError, "pool"):
            ta.averaged_fingerprints(floored(items, {"real": 1.0}), {})
        with self.assertRaisesRegex(ValueError, "contexts"):
            ta.averaged_fingerprints(floored(items, {"real": 1.0}), {(10, "real"): {**pool, "n_contexts": 3}})

    def test_the_summary_is_over_features_with_the_gap_and_the_share_of_features_where_the_average_overstates_the_transfer(self):
        hs = {(10, "real"): {"feature": 10, "kind": "real", "cos_gap": 1.0, "coherence_fp": 0.5, "coherence_qq": 0.5, "pooled_normalized_error": 0.1, "item_normalized_error": 0.4, "pooled_cos": 1.0, "item_cos": 0.0},
              (11, "real"): {"feature": 11, "kind": "real", "cos_gap": -0.2, "coherence_fp": 0.9, "coherence_qq": 0.8, "pooled_normalized_error": 0.5, "item_normalized_error": 0.3, "pooled_cos": 0.5, "item_cos": 0.7},
              (12, "real"): {"feature": 12, "kind": "real", "cos_gap": None, "coherence_fp": 0.0, "coherence_qq": 0.0, "pooled_normalized_error": 0.3, "item_normalized_error": 0.2, "pooled_cos": None, "item_cos": 0.5}}
        rec = ta.averaged_summary(hs, n_boot=30, seed=1)
        g = find(rec, kind="real", metric="cos_gap"); self.assertEqual((g["n"], g["n_missing"]), (2, 1)); self.assertAlmostEqual(g["point"], -0.2)                                  # lower median of -0.2 and 1.0
        e = find(rec, kind="real", metric="error_gap"); self.assertAlmostEqual(e["point"], 0.1)                                                                                              # pooled minus context-mean normalized error: -0.3, +0.2, +0.1
        self.assertAlmostEqual(find(rec, kind="real", metric="averaged_looks_better")["point"], 1.0 / 3.0)                                                               # one feature of three: pooled error below the context mean by more than nothing
        ref = ta.crossed_bootstrap(np.array([1.0, -0.2]), np.array([10, 11]), np.zeros(2, dtype=int), "median", 30, 1); self.assertEqual((g["low"], g["high"], g["n_features"]), (ref["low"], ref["high"], 3))   # features only, no cluster level


def prow(feature, sentence, kind="real", err=1.0, split=None, eligible=True, **kw):
    """A row with the fields of the predictive design (the output of item_table and apply_floor, by hand)."""
    r = {"feature": feature, "sentence": sentence, "kind": kind, "cluster": sentence % 997, "split": split or ("calibration" if feature in SPLIT["calibration"] else "heldout"), "eligible": eligible, "normalized_error": err,
         "act_cos": 0.5, "amp_fp_mean": 2.0, "firing": 0.5, "fp_norm": 10.0, "resid_norm_fp": 100.0, "g_shift_native": 0.5, "g_pred_fp": 0.25, "g_pred_q_native": 0.75, "g_norm_fp": 2.0, "g_norm_q": 4.0}
    r.update(kw); return r


class Design(unittest.TestCase):
    def setUp(self):
        self.rows = {}
        for f in (10, 11):
            for s in (1000, 2000):
                self.rows[(f, s, "real")] = prow(f, s, "real", err=1.0 + f / 100)
                for k, e in (("matched_swapped", 0.5), ("random1", 0.2), ("random2", 0.3), ("random3", 0.4)):
                    self.rows[(f, s, k)] = prow(f, s, k, err=e)

    def test_one_row_per_eligible_real_item_with_the_baseline_columns_and_the_generic_sensitivity_from_the_controls_of_the_same_context(self):
        d = ta.design(self.rows)
        self.assertEqual(len(d["y"]), 4); self.assertEqual(d["keys"][0], (10, 1000)); self.assertAlmostEqual(d["y"][0], 1.10); self.assertEqual(list(d["split"]), ["calibration", "calibration", "heldout", "heldout"])
        self.assertEqual(d["columns"]["A"], ["act_cos", "log_amp_fp", "firing", "log_fp_norm", "log_resid_norm"]); self.assertEqual(d["columns"]["B"], ["generic_sensitivity"]); self.assertEqual(len(d["columns"]["G"]), 4)
        self.assertTrue(np.allclose(d["A"][0], [0.5, np.log(2.0), 0.5, np.log(10.0), np.log(100.0)])); self.assertAlmostEqual(d["B"][0, 0], float(np.log(np.mean([0.5, 0.2, 0.3, 0.4]) + 0.01)))
        self.assertTrue(np.allclose(d["G"][0], [np.log(0.5 + 1e-6), np.log(0.25 + 1e-6), np.log(0.75 + 1e-6), np.log(4.0) - np.log(2.0)])); self.assertEqual(d["n_dropped"], 0)

    def test_the_real_item_itself_never_enters_its_own_generic_sensitivity_and_the_average_uses_the_controls_that_exist(self):
        del self.rows[(10, 1000, "random1")]; del self.rows[(10, 1000, "random2")]; d = ta.design(self.rows)
        self.assertAlmostEqual(d["B"][0, 0], float(np.log(np.mean([0.5, 0.4]) + 0.01)))
        rows = {k: v for k, v in self.rows.items() if not (k[:2] == (10, 1000) and k[2] != "real")}; d = ta.design(rows)
        self.assertEqual(len(d["y"]), 3); self.assertEqual(d["n_dropped"], 1); self.assertEqual(d["dropped_reasons"], {"no_control": 1})

    def test_ineligible_items_are_left_out_unless_asked_for_and_items_with_a_missing_baseline_value_are_dropped_with_the_reason(self):
        self.rows[(10, 1000, "real")]["eligible"] = False
        self.assertEqual(len(ta.design(self.rows)["y"]), 3); self.assertEqual(len(ta.design(self.rows, eligible_only=False)["y"]), 4)
        del self.rows[(11, 2000, "real")]["act_cos"]; d = ta.design(self.rows); self.assertEqual(len(d["y"]), 2); self.assertEqual(d["dropped_reasons"], {"missing:act_cos": 1})
        self.rows[(11, 1000, "real")]["g_pred_fp"] = None; d = ta.design(self.rows); self.assertEqual(d["dropped_reasons"], {"missing:act_cos": 1, "missing:g_pred_fp": 1})

    def test_the_gradient_columns_use_absolute_values(self):
        self.rows[(10, 1000, "real")].update(g_shift_native=-0.5, g_pred_fp=-0.25, g_pred_q_native=-0.75); d = ta.design(self.rows)
        self.assertTrue(np.allclose(d["G"][0], [np.log(0.5 + 1e-6), np.log(0.25 + 1e-6), np.log(0.75 + 1e-6), np.log(2.0)]))

    def test_the_cluster_and_the_feature_of_every_row_come_with_the_design(self):
        d = ta.design(self.rows); self.assertEqual(list(d["feature"]), [10, 10, 11, 11]); self.assertEqual(list(d["cluster"]), [1000 % 997, 2000 % 997] * 2)


class Ridge(unittest.TestCase):
    def test_the_fit_is_the_closed_form_ridge_on_columns_standardized_with_the_training_mean_and_deviation(self):
        rng = np.random.default_rng(0); X = rng.normal(size=(30, 3)) * [1.0, 5.0, 0.2] + [3.0, -2.0, 0.0]; y = X @ [1.0, 0.1, 4.0] + rng.normal(size=30) * 0.1; lam = 2.0
        m = ta.ridge_fit(X, y, lam); mu, sd = X.mean(0), X.std(0); Z = (X - mu) / sd; w = np.linalg.solve(Z.T @ Z + lam * np.eye(3), Z.T @ (y - y.mean()))
        Xt = rng.normal(size=(5, 3)); self.assertTrue(np.allclose(ta.ridge_predict(m, Xt), ((Xt - mu) / sd) @ w + y.mean())); self.assertTrue(np.allclose(m["mean"], mu) and np.allclose(m["scale"], sd))

    def test_a_larger_strength_shrinks_the_coefficients_and_a_constant_column_is_ignored(self):
        rng = np.random.default_rng(1); X = rng.normal(size=(40, 2)); y = 3 * X[:, 0] + rng.normal(size=40)
        small, large = ta.ridge_fit(X, y, 0.1), ta.ridge_fit(X, y, 100.0); self.assertLess(np.abs(large["coef"]).sum(), np.abs(small["coef"]).sum())
        Xc = np.hstack([X, np.full((40, 1), 7.0)]); m = ta.ridge_fit(Xc, y, 1.0); self.assertEqual(float(m["coef"][2]), 0.0); self.assertTrue(np.isfinite(ta.ridge_predict(m, Xc)).all())

    def test_the_intercept_is_not_penalized(self):
        rng = np.random.default_rng(2); X = rng.normal(size=(50, 2)); y = X[:, 0] + 100.0
        self.assertTrue(np.allclose(ta.ridge_predict(ta.ridge_fit(X, y, 1e6), X), y.mean(), atol=1e-3))                                          # huge penalty: the prediction is the training mean of y, far from zero


class Folds(unittest.TestCase):
    def test_a_feature_belongs_to_one_deterministic_fold_and_every_fold_is_used(self):
        f = [ta.feature_fold(i, 5) for i in range(200)]; self.assertEqual(f, [ta.feature_fold(i, 5) for i in range(200)]); self.assertEqual(set(f), set(range(5))); self.assertNotEqual(f, [ta.feature_fold(i, 5, salt="other") for i in range(200)])

    def test_the_strength_with_the_lowest_cross_validated_mae_over_feature_groups_is_chosen_and_ties_go_to_the_smaller_strength(self):
        rng = np.random.default_rng(3); feats = np.repeat(np.arange(40), 3); X = rng.normal(size=(120, 4)); y = X[:, 0] * 2 + rng.normal(size=120) * 0.5
        res = ta.choose_lambda(X, y, feats, lams=(0.1, 1, 10), n_folds=5)
        manual = {}
        for lam in (0.1, 1, 10):
            err = []
            for k in range(5):
                te = np.array([ta.feature_fold(f, 5) == k for f in feats]); m = ta.ridge_fit(X[~te], y[~te], lam); err += list(np.abs(ta.ridge_predict(m, X[te]) - y[te]))
            manual[lam] = float(np.mean(err))
        self.assertTrue(all(abs(res["cv_mae"][l] - manual[l]) < 1e-12 for l in manual)); self.assertEqual(res["lambda"], min(manual, key=lambda l: (manual[l], l)))
        const = ta.choose_lambda(np.zeros((30, 1)), np.arange(30.0), np.arange(30), lams=(0.1, 1, 10), n_folds=3); self.assertEqual(const["lambda"], 0.1)                  # a constant column: all strengths are the same


class FoldsDegenerate(unittest.TestCase):
    def test_fewer_than_two_nonempty_folds_cannot_be_cross_validated(self):
        with self.assertRaisesRegex(ValueError, "folds"):
            ta.choose_lambda(np.random.default_rng(0).normal(size=(6, 2)), np.arange(6.0), np.full(6, 7), lams=(0.1, 1), n_folds=5)


class PairedImprovement(unittest.TestCase):
    def test_the_relative_reduction_of_the_mean_absolute_error_with_an_interval_from_the_replicates_replayed_independently(self):
        rng = np.random.default_rng(5); base = rng.exponential(size=40) + 0.5; diag = base * rng.uniform(0.6, 1.0, size=40); f = np.repeat(np.arange(8), 5); c = np.tile(np.arange(5), 8)
        r = ta.paired_relative_improvement(base, diag, f, c, n_boot=200, seed=3, level=90); mf, mc = ta.bootstrap_multiplicities(8, 5, 200, 3)
        reps = 1.0 - ta.replicate_stats(diag, f, c, mf, mc, "mean") / ta.replicate_stats(base, f, c, mf, mc, "mean")
        self.assertAlmostEqual(r["point"], 1.0 - diag.mean() / base.mean()); self.assertAlmostEqual(r["low"], float(np.percentile(reps, 5))); self.assertAlmostEqual(r["high"], float(np.percentile(reps, 95)))
        self.assertEqual((r["n"], r["n_features"], r["n_clusters"]), (40, 8, 5))

    def test_equal_errors_give_no_improvement(self):
        e = np.arange(1.0, 9.0); r = ta.paired_relative_improvement(e, e, np.arange(8), np.arange(8) % 3, n_boot=30, seed=0); self.assertEqual((r["point"], r["low"], r["high"]), (0.0, 0.0, 0.0))


def synthetic_design(seed=0, n_cal_features=24, n_held_features=18, per=4, signal="B", noise=0.1):
    """A design of the predictive diagnostic with known structure: y follows the column of the group `signal` (None: pure noise)."""
    rng = np.random.default_rng(seed); n_c, n_h = n_cal_features * per, n_held_features * per; n = n_c + n_h
    feature = np.concatenate([np.repeat(np.arange(n_cal_features), per), 1000 + np.repeat(np.arange(n_held_features), per)]); split = np.array(["calibration"] * n_c + ["heldout"] * n_h)
    d = {"A": rng.normal(size=(n, 5)), "B": rng.normal(size=(n, 1)), "G": rng.normal(size=(n, 4)), "split": split, "feature": feature, "cluster": rng.integers(0, 12, size=n)}
    d["y"] = 3.0 + (d[signal][:, 0] * 2.0 if signal else 0.0) + rng.normal(size=n) * noise if signal else 3.0 + rng.normal(size=n)
    return d


class Diagnostic(unittest.TestCase):
    def test_a_generic_sensitivity_that_carries_the_signal_beats_both_baselines_and_meets_the_criterion(self):
        r = ta.diagnostic(synthetic_design(signal="B"), n_boot=200, seed=1)
        self.assertEqual((r["n_calibration"], r["n_heldout"]), (96, 72)); self.assertEqual(set(r["models"]), {"A", "A+G", "A+B", "A+B+G"}); self.assertEqual(r["diagnostic"], "A+B")
        self.assertGreater(r["improvement"]["point"], 0.5); self.assertGreater(r["improvement"]["low"], 0.0); self.assertGreater(r["improvement_over_A"]["low"], 0.0); self.assertTrue(r["criterion_met"]); self.assertLess(r["models"]["A+B"]["heldout_mae"], r["models"]["A"]["heldout_mae"])

    def test_the_improvement_over_the_activation_baseline_alone_is_the_h3_quantity_and_uses_a_not_the_strongest_baseline(self):
        d = synthetic_design(signal="G", seed=5); r = ta.diagnostic(d, n_boot=50, seed=1); self.assertEqual(r["strongest_baseline"], "A+G")
        held = d["split"] == "heldout"; ref = ta.paired_relative_improvement(r["models"]["A"]["abs_error"], r["models"]["A+B"]["abs_error"], d["feature"][held], d["cluster"][held], 50, 1)
        self.assertEqual(r["improvement_over_A"], ref); self.assertNotEqual(r["improvement_over_A"], r["improvement"])

    def test_pure_noise_gives_no_utility_and_the_null_model_is_the_calibration_median(self):
        r = ta.diagnostic(synthetic_design(signal=None, seed=3), n_boot=200, seed=1); self.assertFalse(r["criterion_met"]); self.assertLess(r["improvement"]["point"], 0.1)
        d = synthetic_design(signal=None, seed=3); cal = d["split"] == "calibration"; held = ~cal; self.assertAlmostEqual(r["null_mae"], float(np.abs(d["y"][held] - np.median(d["y"][cal])).mean()))

    def test_the_strongest_baseline_is_chosen_on_the_calibration_half_by_cross_validated_error(self):
        d = synthetic_design(signal="G", seed=5); r = ta.diagnostic(d, n_boot=50, seed=1); self.assertEqual(r["strongest_baseline"], "A+G")
        d2 = synthetic_design(signal="B", seed=5); self.assertEqual(ta.diagnostic(d2, n_boot=50, seed=1)["strongest_baseline"], "A")                        # G is noise: the extra columns do not pay

    def test_nothing_from_the_held_out_half_enters_the_fits_the_strengths_or_the_choice_of_the_baseline(self):
        d = synthetic_design(signal="B", seed=7); a = ta.diagnostic(d, n_boot=50, seed=1); e = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in d.items()}; held = e["split"] == "heldout"
        e["y"][held] = e["y"][held] * 50 + 1000; e["A"][held] = e["A"][held] * 1e6; e["B"][held] = -e["B"][held] * 1e6; b = ta.diagnostic(e, n_boot=50, seed=1)
        self.assertEqual(a["strongest_baseline"], b["strongest_baseline"])
        for name in a["models"]:
            for k in ("lambda", "cv_mae", "coef", "mean", "scale"):
                self.assertEqual(np.asarray(a["models"][name][k]).tolist(), np.asarray(b["models"][name][k]).tolist(), (name, k))
        self.assertNotEqual(a["models"]["A"]["heldout_mae"], b["models"]["A"]["heldout_mae"])

    def test_too_few_items_are_reported_not_fitted(self):
        r = ta.diagnostic(synthetic_design(n_cal_features=3, n_held_features=2, per=2), n_boot=10, seed=0); self.assertEqual(r["status"], "insufficient"); self.assertEqual((r["n_calibration"], r["n_heldout"]), (6, 4)); self.assertNotIn("improvement", r)

    def test_a_small_held_out_half_alone_is_enough_to_refuse_and_the_coefficients_are_those_of_the_chosen_strength(self):
        r = ta.diagnostic(synthetic_design(n_cal_features=24, n_held_features=2, per=2), n_boot=10, seed=0); self.assertEqual((r["status"], r["n_heldout"]), ("insufficient", 4))
        d = synthetic_design(signal="B", seed=9); r = ta.diagnostic(d, n_boot=10, seed=0, lams=(0.1, 10, 1000)); cal = d["split"] == "calibration"
        for name, groups in ta.MODEL_GROUPS.items():
            fit = ta.ridge_fit(np.hstack([d[g][cal] for g in groups]), d["y"][cal], r["models"][name]["lambda"]); self.assertTrue(np.allclose(fit["coef"], r["models"][name]["coef"]))
        self.assertTrue(any(r["models"][n]["lambda"] != 1 for n in r["models"]))

    def test_the_criterion_needs_ten_percent_and_an_interval_above_zero(self):
        self.assertTrue(ta.utility_criterion({"point": 0.10, "low": 0.01, "high": 0.2})); self.assertFalse(ta.utility_criterion({"point": 0.0999, "low": 0.01, "high": 0.2}))
        self.assertFalse(ta.utility_criterion({"point": 0.3, "low": 0.0, "high": 0.4})); self.assertFalse(ta.utility_criterion({"point": 0.3, "low": -0.1, "high": 0.4}))


if __name__ == "__main__":
    unittest.main()
