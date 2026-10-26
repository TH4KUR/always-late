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


if __name__ == "__main__":
    unittest.main()
