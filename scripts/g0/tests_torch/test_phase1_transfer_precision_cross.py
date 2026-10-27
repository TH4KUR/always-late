import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phase1_transfer_core as tc
import phase1_transfer_precision_cross as px

V = 7


def toy_suffix(seed):
    """A deterministic nonlinear map [B, L, d] -> [B, L, V]: the same weights for the same seed, so two 'modes' can differ by a small perturbation."""
    g = torch.Generator().manual_seed(seed); W = torch.randn(5, V, generator=g)
    return lambda x: torch.tanh(x) @ W


class Observer(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(1); self.h = torch.randn(6, 5, generator=g); self.edits = [(torch.tensor([1, 3]), torch.randn(2, 5, generator=g)), (torch.tensor([2]), torch.randn(1, 5, generator=g))]
        self.targets = torch.tensor([1, 2, 3, -1, 0, 4]); self.suffix = toy_suffix(2)

    def test_the_vectors_equal_the_effect_vectors_that_effect_many_returns_for_each_item(self):
        ob = px.VectorObserver(); out = tc.effect_many(self.h, self.edits, lambda x: self.suffix(x), self.targets, True, ob, ("FP/FP", [("f", 1, "real"), ("f", 2, "real")]))
        for k, key in enumerate([("f", 1, "real"), ("f", 2, "real")]):
            self.assertTrue(torch.equal(ob.vectors[(key, "FP/FP")], out[k]["vector"].float().cpu()))
        self.assertEqual(len(ob.vectors), 2)

    def test_the_combined_quantized_call_splits_into_the_fixed_edit_cell_and_the_native_cell(self):
        keys = [("f", 1, "real"), ("f", 2, "real")]; native = [(r, e * 0.5) for r, e in self.edits]; ob = px.VectorObserver()
        out = tc.effect_many(self.h, self.edits + native, self.suffix, self.targets, True, ob, ("Q/Q+Q/Q-native", keys))
        for k, key in enumerate(keys):
            self.assertTrue(torch.equal(ob.vectors[(key, "Q/Q")], out[k]["vector"])); self.assertTrue(torch.equal(ob.vectors[(key, "Q/Q-native")], out[len(keys) + k]["vector"]))

    def test_without_centering_the_vectors_are_the_raw_mean_logit_changes(self):
        ob = px.VectorObserver(center=False); out = tc.effect_many(self.h, self.edits, self.suffix, self.targets, False, ob, ("FP/Q", [("f", 1, "real"), ("f", 2, "real")]))
        self.assertTrue(torch.equal(ob.vectors[(("f", 1, "real"), "FP/Q")], out[0]["vector"])); ob2 = px.VectorObserver(center=True); tc.effect_many(self.h, self.edits, self.suffix, self.targets, True, ob2, ("FP/Q", [("f", 1, "real"), ("f", 2, "real")]))
        self.assertFalse(torch.equal(ob.vectors[(("f", 1, "real"), "FP/Q")], ob2.vectors[(("f", 1, "real"), "FP/Q")]))

    def test_a_repeated_item_and_cell_an_unknown_tag_and_a_wrong_number_of_edits_are_errors(self):
        keys = [("f", 1, "real"), ("f", 2, "real")]; ob = px.VectorObserver(); tc.effect_many(self.h, self.edits, self.suffix, self.targets, True, ob, ("FP/FP", keys))
        with self.assertRaisesRegex(ValueError, "twice"):
            tc.effect_many(self.h, self.edits, self.suffix, self.targets, True, ob, ("FP/FP", keys))
        with self.assertRaisesRegex(ValueError, "unknown"):
            tc.effect_many(self.h, self.edits, self.suffix, self.targets, True, px.VectorObserver(), ("Q/Q-grad@0.05", keys))
        with self.assertRaisesRegex(ValueError, "edits"):
            tc.effect_many(self.h, self.edits, self.suffix, self.targets, True, px.VectorObserver(), ("FP/FP", keys[:1]))


def vecs(**kw):
    """Cell vectors {mode: {(key, cell): tensor}} for one item from arrays given as bf_FF=..., f32_QQ=... (the cell names with / replaced by _)."""
    out = {"bf16": {}, "fp32": {}}; names = {"FF": "FP/FP", "QF": "Q/FP", "FQ": "FP/Q", "QQ": "Q/Q", "QN": "Q/Q-native"}
    for k, v in kw.items():
        mode, cell = k.split("_"); out["bf16" if mode == "bf" else "fp32"][(("f", 1, "real"), names[cell])] = torch.tensor(v, dtype=torch.float32)
    return out


def full(a, b, da, db, extra=None):
    """bf16: FF = a, QQ = b; fp32: FF = a + da, QQ = b + db; the other cells small distinct vectors."""
    a, b, da, db = (np.array(x, dtype=float) for x in (a, b, da, db)); z = np.zeros_like(a)
    kw = {"bf_FF": a, "bf_QF": z + 0.1, "bf_FQ": z + 0.2, "bf_QQ": b, "bf_QN": z + 0.3, "f32_FF": a + da, "f32_QF": z + 0.4, "f32_FQ": z + 0.5, "f32_QQ": b + db, "f32_QN": z + 0.6}
    return vecs(**kw)


class Gram(unittest.TestCase):
    def test_the_cross_mode_gram_is_the_matrix_of_inner_products_of_the_ten_cell_vectors_in_a_fixed_order(self):
        v = full([1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]); G = px.cross_gram(v, ("f", 1, "real")); self.assertEqual(G.shape, (10, 10)); self.assertEqual(px.NAMES[0], "bf16:FP/FP"); self.assertEqual(px.NAMES[9], "fp32:Q/Q-native")
        self.assertAlmostEqual(G[0, 0], 1.0); self.assertAlmostEqual(G[3, 3], 4.0); self.assertAlmostEqual(G[5, 5], 1.0 + 9.0); self.assertAlmostEqual(G[8, 8], 4.0 + 16.0); self.assertTrue(np.allclose(G, G.T)); self.assertAlmostEqual(G[0, 5], 1.0)

    def test_a_missing_cell_of_either_mode_is_an_error_naming_it(self):
        v = full([1, 0], [0, 1], [0, 0], [0, 0]); del v["fp32"][(("f", 1, "real"), "Q/Q")]
        with self.assertRaisesRegex(KeyError, "cell Q/Q of item .* in the mode fp32"):
            px.cross_gram(v, ("f", 1, "real"))

    def test_norms_dots_and_cosines_of_linear_combinations_equal_the_direct_vector_computation(self):
        g = torch.Generator().manual_seed(4); V10 = torch.randn(10, 6, generator=g, dtype=torch.float64); G = (V10 @ V10.T).numpy(); c1 = np.zeros(10); c1[[0, 3, 8]] = [1.0, -2.0, 0.5]; c2 = np.zeros(10); c2[[1, 5]] = [3.0, 1.0]
        d1, d2 = (torch.tensor(c1) @ V10), (torch.tensor(c2) @ V10); self.assertAlmostEqual(px.combo_norm(G, c1), float(d1.norm())); self.assertAlmostEqual(px.combo_dot(G, c1, c2), float(d1 @ d2)); self.assertAlmostEqual(px.combo_cosine(G, c1, c2), float(d1 @ d2 / (d1.norm() * d2.norm())))
        self.assertIsNone(px.combo_cosine(G, np.zeros(10), c2))

    def test_the_combination_of_named_cells_places_each_weight_at_the_position_of_its_mode_and_cell(self):
        c = px.comb([(("fp32", "Q/Q"), 1.0), (("bf16", "FP/FP"), -1.0)]); self.assertEqual(c.tolist(), [-1, 0, 0, 0, 0, 0, 0, 0, 1, 0]); self.assertEqual(px.comb([(("bf16", "Q/Q"), 2.0), (("bf16", "Q/Q"), 3.0)])[3], 5.0)


class ItemMetrics(unittest.TestCase):
    def test_the_within_mode_error_and_cosine_and_the_cross_mode_arithmetic_changes_follow_from_the_gram(self):
        a, b, da, db = [3.0, 0, 0, 0], [0, 4.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 2.0]; G = px.cross_gram(full(a, b, da, db), ("f", 1, "real")); m = px.item_metrics(G)
        self.assertAlmostEqual(m["fp_norm_bf16"], 3.0); self.assertAlmostEqual(m["fp_norm_fp32"], np.sqrt(9 + 1)); self.assertAlmostEqual(m["qq_norm_bf16"], 4.0); self.assertAlmostEqual(m["err_bf16"], 5.0)
        self.assertAlmostEqual(m["err_fp32"], float(np.linalg.norm(np.array(b) + db - np.array(a) - da))); self.assertAlmostEqual(m["cos_bf16"], 0.0); self.assertAlmostEqual(m["arith_ff"], 1.0); self.assertAlmostEqual(m["arith_qq"], 2.0)
        self.assertAlmostEqual(m["arith_qq_over_err_bf16"], 2.0 / 5.0); self.assertAlmostEqual(m["arith_qq_over_err_fp32"], 2.0 / m["err_fp32"]); self.assertAlmostEqual(m["arith_ff_over_fp_norm_bf16"], 1.0 / 3.0)
        e_bf = np.array(b) - np.array(a); self.assertAlmostEqual(m["cos_arith_vs_err"], float(np.array(db) @ e_bf / (np.linalg.norm(db) * np.linalg.norm(e_bf))))

    def test_cosines_and_ratios_are_none_not_nan_or_inf_when_a_norm_is_zero(self):
        z = [0.0, 0, 0, 0]; G = px.cross_gram(full([1.0, 0, 0, 0], [1.0, 0, 0, 0], z, z), ("f", 1, "real")); m = px.item_metrics(G)
        self.assertEqual(m["err_bf16"], 0.0); self.assertIsNone(m["arith_qq_over_err_bf16"]); self.assertIsNone(m["arith_qq_over_err_fp32"]); self.assertEqual(m["arith_qq"], 0.0); self.assertIsNone(m["cos_arith_vs_err"])

    def test_a_pure_scalar_rescaling_by_the_arithmetic_has_cosine_one_between_the_modes_and_a_relative_change_equal_to_the_scale(self):
        a = np.array([1.0, 2.0, 0, 0]); b = np.array([0, 1.0, 2.0, 0]); s = 1.5; G = px.cross_gram(full(a, b, (s - 1) * a, (s - 1) * b), ("f", 1, "real")); m = px.item_metrics(G)
        self.assertAlmostEqual(m["err_fp32"], s * m["err_bf16"]); self.assertAlmostEqual(m["cos_fp32"], m["cos_bf16"]); self.assertAlmostEqual(m["arith_ff_over_fp_norm_bf16"], s - 1)


class Floors(unittest.TestCase):
    def test_the_mode_specific_floor_is_the_95th_percentile_of_the_int16_error_per_kind_and_mode(self):
        rows = [{"kind": "real", "err_bf16": float(i), "err_fp32": float(2 * i)} for i in range(1, 101)] + [{"kind": "matched_swapped", "err_bf16": 5.0, "err_fp32": 7.0}]
        f = px.mode_floors(rows); self.assertAlmostEqual(f[("real", "bf16")], float(np.percentile(np.arange(1, 101), 95))); self.assertAlmostEqual(f[("real", "fp32")], float(np.percentile(2 * np.arange(1, 101), 95))); self.assertEqual(f[("matched_swapped", "bf16")], 5.0)

    def test_the_normalized_error_and_the_error_in_floors_use_the_given_floor_of_the_kind(self):
        r = {"kind": "real", "fp_norm_bf16": 110.0, "err_bf16": 30.0, "fp_norm_fp32": 120.0, "err_fp32": 36.0}; out = px.with_floors(r, {"real": 20.0, "matched_swapped": 10.0}, {("real", "bf16"): 15.0, ("real", "fp32"): 18.0})
        self.assertAlmostEqual(out["nerr_old_bf16"], 30.0 / 130.0); self.assertAlmostEqual(out["eof_old_bf16"], 1.5); self.assertAlmostEqual(out["nerr_mode_fp32"], 36.0 / 138.0); self.assertAlmostEqual(out["eof_mode_fp32"], 2.0)
        self.assertTrue(out["eligible_old_bf16"]); self.assertFalse(px.with_floors({**r, "fp_norm_bf16": 100.0}, {"real": 20.0}, {("real", "bf16"): 15.0, ("real", "fp32"): 18.0})["eligible_old_bf16"])      # 100 is not ABOVE five floors of 20
        self.assertTrue(out["eligible_mode_bf16"]); self.assertFalse(px.with_floors({**r, "fp_norm_bf16": 75.0}, {"real": 20.0}, {("real", "bf16"): 15.0, ("real", "fp32"): 18.0})["eligible_mode_bf16"])


class Batching(unittest.TestCase):
    def test_the_difference_of_two_executions_is_the_norm_of_the_difference_of_the_stored_vectors_of_the_cell(self):
        key = ("f", 1, "real"); A = {(key, "FP/FP"): torch.tensor([1.0, 2.0, 3.0]), (key, "Q/Q"): torch.tensor([0.0, 0.0, 1.0])}; B = {(key, "FP/FP"): torch.tensor([1.0, 2.0, 5.0]), (key, "Q/Q"): torch.tensor([0.0, 3.0, 1.0])}
        d = px.batching_difference(A, B, [key]); self.assertAlmostEqual(d[key]["FP/FP"], 2.0); self.assertAlmostEqual(d[key]["Q/Q"], 3.0)
        with self.assertRaisesRegex(KeyError, "cell Q/Q of item .* missing from the second execution"):
            px.batching_difference(A, {(key, "FP/FP"): B[(key, "FP/FP")]}, [key])


if __name__ == "__main__":
    unittest.main()
