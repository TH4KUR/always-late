import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phase1_transfer_core as tc


def scalar_suffix(w):
    """The hand-computable suffix F_s(h) = w h^2 on a one-dimensional 'vocabulary' (no centering: V = 1 and centering is switched off in these tests)."""
    return lambda x: w * x ** 2


def scalar_cells(h_fp=1.0, h_q=2.0, v=0.1, w_fp=1.0, w_q=2.0):
    res = {"FP": torch.tensor([[h_fp]]), "Q": torch.tensor([[h_q]])}; suf = {"FP": scalar_suffix(w_fp), "Q": scalar_suffix(w_q)}
    return tc.factorial_effects(res, suf, rows=torch.tensor([0]), fp_edit=torch.tensor([[v]]), center=False)


class Hand(unittest.TestCase):
    """The values of the plan (Task 2): F_s(h) = w_s h^2, h_FP = 1, h_Q = 2, v = 0.1, w_FP = 1, w_Q = 2 give the four effects -0.19, -0.39, -0.38, -0.78 and C = -0.20, W = -0.19, I = -0.20, total = -0.59."""
    def test_the_four_cell_effects(self):
        c = scalar_cells()
        for name, want in (("FP/FP", -0.19), ("Q/FP", -0.39), ("FP/Q", -0.38), ("Q/Q", -0.78)):
            self.assertAlmostEqual(float(c[name]["vector"][0]), want, places=5, msg=name)

    def test_the_decomposition_with_the_interaction_and_the_identity(self):
        d = tc.decompose(scalar_cells())
        v = {k: float(d[k]["vector"][0]) for k in ("C", "W", "I", "total")}
        for k, want in (("C", -0.20), ("W", -0.19), ("I", -0.20), ("total", -0.59)):
            self.assertAlmostEqual(v[k], want, places=5, msg=k)
        self.assertAlmostEqual(v["C"] + v["W"] + v["I"], v["total"], places=6)

    def test_a_linear_suffix_has_no_interaction_and_no_context_term(self):
        res = {"FP": torch.tensor([[1.0]]), "Q": torch.tensor([[2.0]])}; suf = {"FP": lambda x: 1.0 * x, "Q": lambda x: 3.0 * x}
        c = tc.factorial_effects(res, suf, torch.tensor([0]), torch.tensor([[0.1]]), center=False); d = tc.decompose(c)
        self.assertAlmostEqual(float(d["C"]["vector"][0]), 0.0, places=6); self.assertAlmostEqual(float(d["I"]["vector"][0]), 0.0, places=6)
        self.assertAlmostEqual(float(d["W"]["vector"][0]), -0.2, places=6)                                        # (3 - 1) * (-0.1)

    def test_unequal_native_amplitudes_are_a_separate_cell_and_not_a_substitution(self):
        res = {"FP": torch.tensor([[1.0]]), "Q": torch.tensor([[2.0]])}; suf = {"FP": scalar_suffix(1.0), "Q": scalar_suffix(2.0)}
        c = tc.factorial_effects(res, suf, torch.tensor([0]), torch.tensor([[0.1]]), native_edit=torch.tensor([[0.08]]), center=False)
        self.assertAlmostEqual(float(c["Q/Q"]["vector"][0]), -0.78, places=5)                                      # the fixed-amplitude cell is unchanged
        self.assertAlmostEqual(float(c["Q/Q-native"]["vector"][0]), 2.0 * ((2 - 0.08) ** 2 - 4), places=5)
        d = tc.decompose(c); self.assertAlmostEqual(float(d["native_minus_fixed"]["vector"][0]), float(c["Q/Q-native"]["vector"][0]) - float(c["Q/Q"]["vector"][0]), places=6)

    def test_the_signed_loss_change_is_decomposed_in_the_same_way(self):
        d = tc.decompose(scalar_cells())                                                                              # no targets given: the loss field is absent for every cell
        self.assertIsNone(d["C"]["dloss"]); self.assertIsNone(d["total"]["dloss"])


class Random(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(0); self.d, self.V, self.L = 6, 9, 5
        self.res = {"FP": torch.randn(self.L, self.d, generator=g), "Q": torch.randn(self.L, self.d, generator=g)}
        Wf, Wq = torch.randn(self.d, self.V, generator=g), torch.randn(self.d, self.V, generator=g)
        self.suf = {"FP": lambda x: torch.tanh(x @ Wf) * 2.0, "Q": lambda x: torch.tanh(x @ Wq) * 2.0 + 0.3}
        self.rows = torch.tensor([1, 3]); self.edit = torch.randn(2, self.d, generator=g) * 0.4; self.native = torch.randn(2, self.d, generator=g) * 0.3
        self.targets = torch.tensor([0, 4, 2, 7, -1])

    def cells(self, **kw):
        return tc.factorial_effects(self.res, self.suf, self.rows, self.edit, native_edit=self.native, targets=self.targets, **kw)

    def test_total_is_exactly_c_plus_w_plus_i_for_vectors_and_for_the_loss(self):
        d = tc.decompose(self.cells())
        self.assertTrue(torch.allclose(d["C"]["vector"] + d["W"]["vector"] + d["I"]["vector"], d["total"]["vector"], atol=1e-5))
        self.assertAlmostEqual(d["C"]["dloss"] + d["W"]["dloss"] + d["I"]["dloss"], d["total"]["dloss"], places=5)

    def test_every_cell_uses_its_own_unedited_baseline(self):
        shifted = {"FP": self.suf["FP"], "Q": lambda x: self.suf["Q"](x) + 5.0}                                        # a constant offset of the suffix changes no effect
        a = tc.factorial_effects(self.res, self.suf, self.rows, self.edit); b = tc.factorial_effects(self.res, shifted, self.rows, self.edit)
        for k in ("FP/Q", "Q/Q"):
            self.assertTrue(torch.allclose(a[k]["vector"], b[k]["vector"], atol=1e-5))

    def test_the_vector_is_the_mean_over_the_edited_rows_of_the_centered_logit_change(self):
        e = tc.effect(self.res["FP"], self.rows, self.edit, lambda x: self.suf["FP"](x))
        manual = []
        for j, r in enumerate(self.rows.tolist()):
            h = self.res["FP"].clone(); h2 = h.clone(); h2[r] = h[r] - self.edit[j]
            dz = self.suf["FP"](h2)[r] - self.suf["FP"](h)[r]; manual.append(dz - dz.mean())
        self.assertTrue(torch.allclose(e["vector"], torch.stack(manual).mean(0), atol=1e-6)); self.assertEqual(e["n_rows"], 2)
        self.assertAlmostEqual(float(e["vector"].sum()), 0.0, places=5)

    def test_rows_that_are_not_edited_do_not_enter_the_vector(self):
        a = tc.effect(self.res["FP"], torch.tensor([1]), self.edit[:1], self.suf["FP"])
        h = self.res["FP"].clone(); h[1] = h[1] - self.edit[0]; dz = self.suf["FP"](h)[1] - self.suf["FP"](self.res["FP"])[1]
        self.assertTrue(torch.allclose(a["vector"], dz - dz.mean(), atol=1e-6))

    def test_the_loss_change_is_the_mean_over_all_rows_with_a_target(self):
        e = tc.effect(self.res["FP"], self.rows, self.edit, self.suf["FP"], targets=self.targets)
        h2 = self.res["FP"].clone()
        for j, r in enumerate(self.rows.tolist()):
            h2[r] = h2[r] - self.edit[j]
        zb, ze = self.suf["FP"](self.res["FP"]), self.suf["FP"](h2); ok = self.targets >= 0; ar = torch.arange(5)
        nb = -torch.log_softmax(zb, -1)[ar, self.targets.clamp(min=0)]; ne = -torch.log_softmax(ze, -1)[ar, self.targets.clamp(min=0)]
        self.assertAlmostEqual(e["dloss"], float((ne - nb)[ok].mean()), places=5); self.assertAlmostEqual(e["loss_base"], float(nb[ok].mean()), places=5)

    def test_gram_matrix_order_symmetry_and_entries(self):
        c = self.cells(); G = tc.gram(c)
        self.assertEqual(G.shape, (5, 5)); self.assertEqual(G.dtype, torch.float64); self.assertTrue(torch.allclose(G, G.T))
        for i, a in enumerate(tc.CELLS):
            for j, b in enumerate(tc.CELLS):
                self.assertAlmostEqual(float(G[i, j]), float((c[a]["vector"].double() * c[b]["vector"].double()).sum()), places=5)

    def test_norms_from_the_gram_equal_the_norms_of_the_vectors(self):
        c = self.cells(); n = tc.decomposition_norms(tc.gram(c)); d = tc.decompose(c)
        for k in ("C", "W", "I", "total", "native_minus_fixed"):
            self.assertAlmostEqual(n[k], float(d[k]["vector"].double().norm()), places=5, msg=k)

    def test_the_transfer_error_is_normalized_by_the_fp_effect_norm_plus_the_floor(self):
        c = self.cells(); G = tc.gram(c); err = tc.transfer_error(G, floor=0.25)
        want = float((c["Q/Q"]["vector"] - c["FP/FP"]["vector"]).norm()) / (float(c["FP/FP"]["vector"].norm()) + 0.25)
        self.assertAlmostEqual(err["normalized_error"], want, places=5)
        self.assertAlmostEqual(err["error_norm"], float((c["Q/Q"]["vector"] - c["FP/FP"]["vector"]).norm()), places=5)
        self.assertAlmostEqual(err["norm_ratio"], float(c["Q/Q"]["vector"].norm() / c["FP/FP"]["vector"].norm()), places=5)
        self.assertAlmostEqual(err["cosine"], float(torch.nn.functional.cosine_similarity(c["Q/Q"]["vector"], c["FP/FP"]["vector"], dim=0)), places=5)

    def test_a_zero_fp_effect_gives_an_undefined_cosine_and_ratio_not_a_division_by_zero(self):
        G = torch.zeros(5, 5, dtype=torch.float64); G[3, 3] = 1.0
        err = tc.transfer_error(G, floor=0.5); self.assertIsNone(err["cosine"]); self.assertIsNone(err["norm_ratio"]); self.assertAlmostEqual(err["normalized_error"], 2.0)

    def test_pooling_over_contexts_weights_each_context_by_its_edited_rows(self):
        a = {"vector": torch.tensor([1.0, 0.0]), "n_rows": 1}; b = {"vector": torch.tensor([0.0, 4.0]), "n_rows": 3}
        self.assertTrue(torch.allclose(tc.pool_contexts([a, b]), torch.tensor([0.25, 3.0])))
        with self.assertRaises(ValueError):
            tc.pool_contexts([])


class Gates(unittest.TestCase):
    def test_a_bfloat16_residual_is_refused_because_the_edit_would_be_rounded(self):
        with self.assertRaises(ValueError):
            tc.effect(torch.zeros(2, 3, dtype=torch.bfloat16), torch.tensor([0]), torch.zeros(1, 3), lambda x: x)

    def test_nonfinite_output_shape_mismatch_and_bad_rows_are_errors(self):
        h = torch.zeros(3, 2)
        with self.assertRaises(ValueError):
            tc.effect(h, torch.tensor([0]), torch.zeros(1, 2), lambda x: x / 0.0)
        with self.assertRaises(ValueError):
            tc.effect(h, torch.tensor([0, 1]), torch.zeros(1, 2), lambda x: x)
        with self.assertRaises(ValueError):
            tc.effect(h, torch.tensor([3]), torch.zeros(1, 2), lambda x: x)
        with self.assertRaises(ValueError):
            tc.effect(h, torch.tensor([-1]), torch.zeros(1, 2), lambda x: x)
        with self.assertRaises(ValueError):
            tc.effect(h, torch.tensor([], dtype=torch.long), torch.zeros(0, 2), lambda x: x)
        with self.assertRaises(ValueError):
            tc.effect(h, torch.tensor([1, 1]), torch.zeros(2, 2), lambda x: x)                                       # a repeated row would subtract twice

    def test_a_zero_edit_is_an_exact_zero_effect_in_every_cell(self):                                                    # the sham gate
        g = torch.Generator().manual_seed(1); res = {"FP": torch.randn(4, 3, generator=g), "Q": torch.randn(4, 3, generator=g)}; W = torch.randn(3, 5, generator=g)
        suf = {"FP": lambda x: torch.tanh(x @ W), "Q": lambda x: torch.tanh(x @ W) * 1.1}
        c = tc.factorial_effects(res, suf, torch.tensor([0, 2]), torch.zeros(2, 3), native_edit=torch.zeros(2, 3), targets=torch.tensor([0, 1, 2, 3]))
        for k, e in c.items():
            self.assertEqual(float(e["vector"].abs().max()), 0.0, msg=k); self.assertEqual(e["dloss"], 0.0, msg=k)

    def test_missing_cells_missing_states_and_mismatched_inputs_are_errors(self):
        c = scalar_cells()
        for k in ("FP/FP", "Q/FP", "FP/Q", "Q/Q"):
            part = {x: y for x, y in c.items() if x != k}
            with self.assertRaises(ValueError):
                tc.decompose(part)
        with self.assertRaises(ValueError):
            tc.gram(c)                                                                                                    # the native-amplitude cell is missing
        res = {"FP": torch.tensor([[1.0]])}
        with self.assertRaises(ValueError):
            tc.factorial_effects(res, {"FP": scalar_suffix(1.0), "Q": scalar_suffix(2.0)}, torch.tensor([0]), torch.tensor([[0.1]]))
        with self.assertRaises(ValueError):
            tc.factorial_effects({"FP": torch.zeros(2, 1), "Q": torch.zeros(3, 1)}, {"FP": scalar_suffix(1.0), "Q": scalar_suffix(2.0)}, torch.tensor([0]), torch.tensor([[0.1]]))

    def test_the_same_state_hybrid_must_equal_the_native_forward(self):
        a = torch.randn(2, 4); tc.check_same_state(a, a.clone(), atol=0.0)
        with self.assertRaises(ValueError):
            tc.check_same_state(a, a + 1e-3, atol=1e-6)
        tc.check_same_state(a, a + 1e-7, atol=1e-6)
        with self.assertRaises(ValueError):
            tc.check_same_state(a, torch.zeros(3, 4), atol=1.0)                                                           # different shapes are never 'close'


class Contexts(unittest.TestCase):
    def test_decomposition_over_contexts_preserves_the_item_keys(self):
        per = {("f1", 3): scalar_cells(), ("f1", 7): scalar_cells(h_q=3.0), ("f2", 3): scalar_cells(v=0.2)}
        out = tc.decompose_contexts(per)
        self.assertEqual(list(out), list(per)); self.assertAlmostEqual(float(out[("f1", 3)]["total"]["vector"][0]), -0.59, places=5)
        self.assertNotAlmostEqual(float(out[("f1", 7)]["total"]["vector"][0]), -0.59, places=3)

    def test_a_context_with_a_missing_cell_is_reported_not_dropped(self):
        bad = {k: v for k, v in scalar_cells().items() if k != "Q/Q"}
        with self.assertRaisesRegex(ValueError, "f9"):
            tc.decompose_contexts({("f1", 3): scalar_cells(), ("f9", 1): bad})


class Loss(unittest.TestCase):
    def test_the_loss_change_of_a_two_class_example_is_computed_by_hand(self):
        h = torch.tensor([[2.0, 0.0]]); e = tc.effect(h, torch.tensor([0]), torch.tensor([[0.5, 0.0]]), lambda x: x, targets=torch.tensor([0]))
        base = math.log(1 + math.exp(-2.0)); edit = math.log(1 + math.exp(-1.5))
        self.assertAlmostEqual(e["loss_base"], base, places=5); self.assertAlmostEqual(e["dloss"], edit - base, places=5)
        self.assertTrue(torch.allclose(e["vector"], torch.tensor([-0.25, 0.25]), atol=1e-6))                            # the logit change (-0.5, 0), centered


class Many(unittest.TestCase):
    """effect_many: several edits of ONE context in ONE call, sharing the unedited row (fewer sequences, the same exactness: base and edits take the same computation)."""
    def setUp(self):
        g = torch.Generator().manual_seed(4); self.d, self.V, self.L = 6, 9, 5
        self.h = torch.randn(self.L, self.d, generator=g); W = torch.randn(self.d, self.V, generator=g)
        self.calls = []
        def suffix(x):
            self.calls.append(tuple(x.shape)); return torch.tanh(x @ W) * 2.0
        self.suf = suffix; self.targets = torch.tensor([0, 4, 2, 7, -1])
        self.edits = [(torch.tensor([1, 3]), torch.randn(2, self.d, generator=g) * 0.4), (torch.tensor([0]), torch.randn(1, self.d, generator=g) * 0.3), (torch.tensor([2, 3, 4]), torch.randn(3, self.d, generator=g) * 0.2)]

    def test_each_result_equals_the_single_effect_of_that_edit(self):
        many = tc.effect_many(self.h, self.edits, self.suf, self.targets)
        for (rows, e), m in zip(self.edits, many):
            one = tc.effect(self.h, rows, e, self.suf, self.targets)
            self.assertTrue(torch.allclose(m["vector"], one["vector"], atol=1e-6)); self.assertAlmostEqual(m["dloss"], one["dloss"], places=5); self.assertAlmostEqual(m["loss_base"], one["loss_base"], places=5); self.assertEqual(m["n_rows"], one["n_rows"])

    def test_the_unedited_row_is_shared_in_one_suffix_call(self):
        tc.effect_many(self.h, self.edits, self.suf, self.targets)
        self.assertEqual(self.calls, [(1 + 3, self.L, self.d)])                                                           # one call with the base and three edited residuals

    def test_the_single_effect_is_a_batch_of_two_as_before(self):
        tc.effect(self.h, self.edits[0][0], self.edits[0][1], self.suf); self.assertEqual(self.calls, [(2, self.L, self.d)])

    def test_a_zero_edit_among_other_edits_is_exactly_zero(self):
        edits = [self.edits[0], (torch.tensor([1, 2]), torch.zeros(2, self.d)), self.edits[2]]
        res = tc.effect_many(self.h, edits, self.suf, self.targets)
        self.assertEqual(float(res[1]["vector"].abs().max()), 0.0); self.assertEqual(res[1]["dloss"], 0.0); self.assertGreater(float(res[0]["vector"].abs().max()), 0.0)

    def test_empty_lists_and_bad_edits_are_errors_naming_the_edit(self):
        with self.assertRaises(ValueError):
            tc.effect_many(self.h, [], self.suf)
        bad = [self.edits[0], (torch.tensor([1, 1]), torch.zeros(2, self.d))]
        with self.assertRaisesRegex(ValueError, "edit 1"):
            tc.effect_many(self.h, bad, self.suf)
        bad = [self.edits[0], (torch.tensor([9]), torch.zeros(1, self.d))]
        with self.assertRaisesRegex(ValueError, "edit 1"):
            tc.effect_many(self.h, bad, self.suf)
        bad = [(torch.tensor([1]), torch.zeros(2, self.d))]
        with self.assertRaisesRegex(ValueError, "edit 0"):
            tc.effect_many(self.h, bad, self.suf)

    def test_a_float32_residual_is_required_and_a_nonfinite_output_is_an_error(self):
        with self.assertRaises(ValueError):
            tc.effect_many(self.h.to(torch.bfloat16), self.edits, self.suf)
        with self.assertRaises(ValueError):
            tc.effect_many(self.h, self.edits, lambda x: x[..., :1] / 0.0 + torch.zeros(*x.shape[:-1], self.V))


class Observer(Many):
    """An optional observer sees the raw logits of a call (for sidecar measurements such as decision margins) and changes nothing."""
    def test_the_observer_is_called_once_per_call_with_the_raw_logits_the_rows_and_the_tag(self):
        seen = []
        res = tc.effect_many(self.h, self.edits, self.suf, self.targets, observer=lambda tag, z, rows, targets: seen.append((tag, z.shape, [r.tolist() for r in rows], targets)), tag=("cell", ["a", "b", "c"]))
        self.assertEqual(len(seen), 1); tag, shape, rows, targets = seen[0]
        self.assertEqual(tag, ("cell", ["a", "b", "c"])); self.assertEqual(tuple(shape), (4, self.L, self.V)); self.assertEqual(rows, [[1, 3], [0], [2, 3, 4]]); self.assertTrue(torch.equal(targets, self.targets))
        plain = tc.effect_many(self.h, self.edits, self.suf, self.targets)
        for a, b in zip(res, plain):
            self.assertTrue(torch.equal(a["vector"], b["vector"])); self.assertEqual(a["dloss"], b["dloss"])

    def test_the_observer_gets_the_uncentered_logits_of_the_base_in_row_zero(self):
        got = []
        tc.effect_many(self.h, self.edits, self.suf, observer=lambda tag, z, rows, targets: got.append(z.clone()))
        W_free = self.suf(torch.stack([self.h]))[0]
        self.assertTrue(torch.allclose(got[0][0], W_free, atol=1e-6))


class Constants(unittest.TestCase):
    def test_the_cells_are_those_of_the_frozen_spec(self):
        import phase1_transfer_freeze as tf
        self.assertEqual(list(tc.CELLS), tf.SPEC["cells"])


if __name__ == "__main__":
    unittest.main()
