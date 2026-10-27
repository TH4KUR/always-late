"""The FP bridge of the transfer stage (freeze gate G_fp_bridge): the quantities of a new job's chunk files that depend only on the full-precision model (the full-precision edit and residual norms and amplitudes, the number of
rows and positions, the FP-prefix FP-suffix cell and its loss quantities, the full-precision gradient scalars) must equal the old job's EXACTLY, item by item; the chunk keys (which bind the freeze, core and state) must be equal."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
G0 = HERE.parent
ROOT = G0.parents[1]
sys.path.insert(0, str(G0))
from gptq_ext import bridge_transfer as bt  # noqa: E402

RUNS = Path("/home/karma/Documents/ANLP/project/transfer_runs")


def item(seed=0, shift=0.0):
    r = np.random.default_rng(seed)
    return {"gram": np.eye(5), "cells": ["FP/FP", "Q/FP", "FP/Q", "Q/Q", "Q/Q-native"], "dloss": [0.1 * seed + float(r.normal()) * 1e-3, 0.2 + shift, 0.3, 0.4, 0.5], "loss_base": [3.0 + seed, 3.5 + shift, 3.5, 3.5, 3.5],
            "n_rows": 2 + seed, "n_target_rows": 22, "n_active_q": 1 + int(shift * 10), "n_positions": 2, "edit_norm_fp": 7.5 + seed, "edit_norm_native": 2.8 + shift, "resid_norm_fp": 152.5, "resid_norm_q": 138.0 + shift,
            "amp_fp_mean": 7.5, "amp_q_mean": 2.8 + shift, "direction_norm": 1.0, "amp_fp": [7.5, 7.6], "amp_q": [2.8 + shift, 2.9], "g_pred_fp": -0.0059, "g_pred_q_fixed": -0.002 + shift, "g_pred_q_native": -0.0017, "g_norm_fp": 0.0077, "g_norm_q": 0.01 + shift}


def write_dir(d, n_chunks=2, n_items=3, shift=0.0, key="k" * 64, mutate=None):
    d = Path(d); d.mkdir(parents=True, exist_ok=True)
    for c in range(n_chunks):
        items = {(c, i, "real"): item(seed=c * 10 + i, shift=shift) for i in range(n_items)}
        if mutate:
            mutate(c, items)
        torch.save({"key": key, "payload": {"items": items, "pooled": {}, "skipped": [], "features": []}}, d / f"transfer_INT4_c{c:03d}.pt")
    return d


class Bridge(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_quantized_differences_are_expected_and_do_not_break_the_bridge(self):
        old = write_dir(self.d / "old"); new = write_dir(self.d / "new", shift=0.05)
        r = bt.bridge_dirs(old, new, "transfer_INT4")
        self.assertTrue(r["ok"], r)
        self.assertEqual((r["chunks"], r["items"]), (2, 6))
        self.assertTrue(all(v == 0 for v in r["fields_differing"].values()))

    def test_a_changed_fp_only_field_is_found_per_field(self):
        old = write_dir(self.d / "old")
        def m(c, items):
            if c == 1:
                items[(1, 0, "real")]["edit_norm_fp"] += 1e-9
                items[(1, 2, "real")]["dloss"][0] += 1e-12
        new = write_dir(self.d / "new", mutate=m)
        r = bt.bridge_dirs(old, new, "transfer_INT4")
        self.assertFalse(r["ok"])
        self.assertEqual(r["fields_differing"]["edit_norm_fp"], 1); self.assertEqual(r["fields_differing"]["dloss[0]"], 1)
        self.assertEqual(sum(v for k, v in r["fields_differing"].items() if k not in ("edit_norm_fp", "dloss[0]")), 0)

    def test_every_documented_fp_only_field_is_checked(self):
        for f in ("edit_norm_fp", "resid_norm_fp", "amp_fp", "amp_fp_mean", "g_pred_fp", "g_norm_fp", "n_rows", "n_target_rows", "n_positions", "direction_norm", "dloss[0]", "loss_base[0]"):
            old = write_dir(self.d / f"old_{f}")
            def m(c, items, f=f):
                if c != 0:
                    return
                it = items[(0, 0, "real")]
                if f.endswith("[0]"):
                    name = f[:-3]; it[name] = [it[name][0] + 1e-6] + it[name][1:]
                elif isinstance(it[f], list):
                    it[f] = [it[f][0] + 1e-6] + it[f][1:]
                else:
                    it[f] = it[f] + 1e-6
            new = write_dir(self.d / f"new_{f}", mutate=m)
            r = bt.bridge_dirs(old, new, "transfer_INT4")
            self.assertFalse(r["ok"], f); self.assertEqual(r["fields_differing"][f], 1, f)

    def test_a_changed_cell_label_order_breaks_the_bridge(self):
        old = write_dir(self.d / "old")
        def m(c, items):
            if c == 0:
                items[(0, 0, "real")]["cells"] = ["Q/FP", "FP/FP", "FP/Q", "Q/Q", "Q/Q-native"]
        r = bt.bridge_dirs(old, write_dir(self.d / "new", mutate=m), "transfer_INT4")
        self.assertFalse(r["ok"]); self.assertEqual(r["fields_differing"]["cells"], 1)

    def test_different_chunk_keys_or_item_sets_or_files_break_the_bridge(self):
        old = write_dir(self.d / "old")
        self.assertFalse(bt.bridge_dirs(old, write_dir(self.d / "k", key="z" * 64), "transfer_INT4")["ok"])
        self.assertFalse(bt.bridge_dirs(old, write_dir(self.d / "n", n_items=2), "transfer_INT4")["ok"])
        self.assertFalse(bt.bridge_dirs(old, write_dir(self.d / "c", n_chunks=1), "transfer_INT4")["ok"])

    def test_no_chunk_files_is_not_a_pass(self):
        (self.d / "e1").mkdir(); (self.d / "e2").mkdir()
        self.assertFalse(bt.bridge_dirs(self.d / "e1", self.d / "e2", "transfer_INT4")["ok"])

    @unittest.skipUnless((RUNS / "transfer-feature-int4/transfer_INT4_c000.pt").exists(), "old transfer runs not on this machine")
    def test_the_old_run_bridges_to_itself(self):
        r = bt.bridge_dirs(RUNS / "transfer-feature-int4", RUNS / "transfer-feature-int4", "transfer_INT4")
        self.assertTrue(r["ok"]); self.assertEqual(r["chunks"], 17)


if __name__ == "__main__":
    unittest.main()
