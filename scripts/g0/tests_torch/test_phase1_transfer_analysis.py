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


if __name__ == "__main__":
    unittest.main()
