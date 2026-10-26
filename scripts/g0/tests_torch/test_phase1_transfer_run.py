import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phase1_transfer_core as tc
import phase1_transfer_run as tr

D, V = 5, 7


class Directions(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(0); self.W = torch.randn(10, D, generator=g)

    def test_a_random_direction_has_the_norm_of_the_real_one_and_is_orthogonal_to_it(self):
        d = self.W[3]; r = tr.random_direction(d, 1, 3)
        self.assertAlmostEqual(float(r.norm()), float(d.norm()), places=5); self.assertLess(abs(float(r @ d)) / float(d @ d), 1e-5)

    def test_it_is_reproducible_and_depends_on_the_seed_and_the_feature(self):
        d = self.W[3]
        self.assertTrue(torch.equal(tr.random_direction(d, 1, 3), tr.random_direction(d, 1, 3)))
        self.assertFalse(torch.allclose(tr.random_direction(d, 1, 3), tr.random_direction(d, 2, 3))); self.assertFalse(torch.allclose(tr.random_direction(d, 1, 3), tr.random_direction(d, 1, 4)))

    def test_the_direction_set_has_the_five_frozen_kinds_in_order(self):
        s = tr.build_directions(self.W, 3, 7, [1, 2, 3])
        self.assertEqual(list(s), ["real", "matched_swapped", "random1", "random2", "random3"]); self.assertEqual(tuple(s), tr.DIRECTION_KINDS)
        self.assertTrue(torch.equal(s["real"], self.W[3])); self.assertTrue(torch.equal(s["matched_swapped"], self.W[7]))
        for k in ("random1", "random2", "random3"):
            self.assertAlmostEqual(float(s[k].norm()), float(self.W[3].norm()), places=5)
        self.assertFalse(torch.allclose(s["random1"], s["random2"]))

    def test_the_matched_direction_must_differ_from_the_feature_and_the_seeds_must_be_three(self):
        with self.assertRaises(ValueError):
            tr.build_directions(self.W, 3, 3, [1, 2, 3])
        with self.assertRaises(ValueError):
            tr.build_directions(self.W, 3, 7, [1, 2])


class Targets(unittest.TestCase):
    def test_row_r_predicts_content_token_r_and_the_last_row_has_no_target(self):
        self.assertEqual(tr.context_targets(torch.tensor([5, 6, 7])).tolist(), [5, 6, 7, -1])
        self.assertEqual(tr.context_targets([9]).tolist(), [9, -1])


class Chunks(unittest.TestCase):
    def test_features_are_split_in_order_into_chunks_that_cover_all_of_them(self):
        c = tr.plan_chunks(list(range(10)), 4)
        self.assertEqual(c, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]); self.assertEqual(tr.plan_chunks([], 4), [])
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                tr.plan_chunks([1, 2], bad)

    def test_the_key_of_a_chunk_depends_on_the_freeze_the_state_and_the_features(self):
        k = tr.chunk_key("abc", "INT4", [1, 2]); self.assertEqual(k, tr.chunk_key("abc", "INT4", [1, 2]))
        for other in (tr.chunk_key("abd", "INT4", [1, 2]), tr.chunk_key("abc", "INT6", [1, 2]), tr.chunk_key("abc", "INT4", [1, 3]), tr.chunk_key("abc", "INT4", [2, 1])):
            self.assertNotEqual(k, other)

    def test_a_saved_chunk_is_loaded_only_with_its_key_and_a_temporary_file_is_never_a_result(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "chunk.pt"
            self.assertIsNone(tr.load_chunk(p, "k"))
            tr.atomic_save({"x": torch.arange(3)}, p, "k")
            self.assertEqual(tr.load_chunk(p, "k")["x"].tolist(), [0, 1, 2]); self.assertEqual([x.name for x in Path(d).iterdir()], ["chunk.pt"])
            with self.assertRaisesRegex(ValueError, "stale"):
                tr.load_chunk(p, "other")
            (Path(d) / "partial.pt.tmp").write_bytes(b"junk"); self.assertIsNone(tr.load_chunk(Path(d) / "partial.pt", "k"))


class Fixture:
    """Two states of a tiny model: the suffix is tanh(x W_s) with another W per state; the prefix residuals of the two states differ; one feature with two contexts."""
    def setUp(self):
        g = torch.Generator().manual_seed(3)
        self.Wfp, self.Wq = torch.randn(D, V, generator=g), torch.randn(D, V, generator=g) * 1.2
        self.suf = {"FP": lambda x: torch.tanh(x @ self.Wfp), "Q": lambda x: torch.tanh(x @ self.Wq) + 0.1}
        self.W_dec = torch.randn(6, D, generator=g)
        self.contexts = {2: [10, 11], 4: [11]}
        self.tokens = {10: torch.tensor([1, 2, 3]), 11: torch.tensor([4, 5])}
        self.caches = {"FP": {10: torch.randn(4, D, generator=g).to(torch.bfloat16), 11: torch.randn(3, D, generator=g).to(torch.bfloat16)},
                       "Q": {10: torch.randn(4, D, generator=g).to(torch.bfloat16), 11: torch.randn(3, D, generator=g).to(torch.bfloat16)}}
        self.positions = {(2, 10): torch.tensor([0, 2]), (2, 11): torch.tensor([1]), (4, 11): torch.tensor([0, 1])}
        self.amps = {"FP": {(2, 10): torch.tensor([0.5, 1.5]), (2, 11): torch.tensor([0.7]), (4, 11): torch.tensor([0.2, 0.9])},
                     "Q": {(2, 10): torch.tensor([0.4, 1.2]), (2, 11): torch.tensor([0.0]), (4, 11): torch.tensor([0.3, 0.8])}}
        self.dirs = {f: tr.build_directions(self.W_dec, f, (f + 1) % 6, [1, 2, 3]) for f in (2, 4)}
        self.log = []
        self.switches = {"FP": lambda: self.log.append("FP"), "Q": lambda: self.log.append("Q")}

    def run_chunk(self, features=(2, 4), **kw):
        return tr.process_chunk(list(features), {f: self.contexts[f] for f in features}, self.caches, self.positions, self.amps, self.dirs, self.tokens, self.suf, self.switches, **kw)


class Process(Fixture, unittest.TestCase):
    def reference(self, f, s, kind):
        rows = self.positions[(f, s)] + 1; d = self.dirs[f][kind]
        res = {"FP": self.caches["FP"][s].float(), "Q": self.caches["Q"][s].float()}
        edit = self.amps["FP"][(f, s)][:, None] * d[None, :]; native = self.amps["Q"][(f, s)][:, None] * d[None, :]
        cells = tc.factorial_effects(res, self.suf, rows, edit, native_edit=native, targets=tr.context_targets(self.tokens[s]))
        return cells

    def test_every_item_equals_the_cells_of_the_core_computed_directly(self):
        out = self.run_chunk()
        self.assertEqual(len(out["items"]), 5 * (2 + 1))
        for (f, s, kind), item in out["items"].items():
            ref = self.reference(f, s, kind)
            self.assertTrue(np.allclose(item["gram"], tc.gram(ref).numpy(), atol=1e-6), msg=(f, s, kind))
            self.assertEqual(item["gram"].dtype, np.float64); self.assertEqual(item["n_rows"], len(self.positions[(f, s)]))
            for i, c in enumerate(tc.CELLS):
                self.assertAlmostEqual(item["dloss"][i], ref[c]["dloss"], places=5); self.assertAlmostEqual(item["loss_base"][i], ref[c]["loss_base"], places=5)

    def test_the_weights_are_switched_once_per_state_and_the_fp_suffix_comes_first(self):
        self.run_chunk(); self.assertEqual(self.log, ["FP", "Q"])
        self.log.clear(); self.run_chunk(features=(4,)); self.assertEqual(self.log, ["FP", "Q"])

    def test_the_native_cell_uses_the_amplitudes_of_the_quantized_state_and_the_other_cells_the_fp_ones(self):
        item = self.run_chunk()["items"][(2, 10, "real")]
        ref = self.reference(2, 10, "real")
        self.assertNotAlmostEqual(item["gram"][3, 3], item["gram"][4, 4], places=4)
        self.assertAlmostEqual(item["gram"][4, 4], float(ref["Q/Q-native"]["vector"].double() @ ref["Q/Q-native"]["vector"].double()), places=6)

    def test_the_matched_swapped_direction_uses_the_amplitudes_of_the_feature_itself(self):
        item = self.run_chunk()["items"][(2, 10, "matched_swapped")]; ref = self.reference(2, 10, "matched_swapped")
        self.assertTrue(np.allclose(item["gram"], tc.gram(ref).numpy(), atol=1e-6))
        real = self.run_chunk()["items"][(2, 10, "real")]; self.assertFalse(np.allclose(item["gram"], real["gram"]))

    def test_the_edited_rows_are_the_positions_shifted_by_the_bos_row(self):
        ident = {"FP": lambda x: x[..., :V] if D >= V else torch.nn.functional.pad(x, (0, V - D)), "Q": lambda x: torch.nn.functional.pad(x, (0, V - D))}
        out = tr.process_chunk([2], {2: [10]}, self.caches, self.positions, self.amps, self.dirs, self.tokens, ident, self.switches)
        item = out["items"][(2, 10, "real")]
        h = self.caches["FP"][10].float(); rows = self.positions[(2, 10)] + 1; edit = self.amps["FP"][(2, 10)][:, None] * self.dirs[2]["real"][None, :]
        dz = -torch.nn.functional.pad(edit, (0, V - D)); dz = dz - dz.mean(-1, keepdim=True); v = dz.mean(0)
        self.assertAlmostEqual(item["gram"][0, 0], float(v.double() @ v.double()), places=5)                                # row r+1, not r: a shifted row would still give this for a linear map, so check the loss too
        base = torch.nn.functional.pad(h, (0, V - D)); tg = tr.context_targets(self.tokens[10]); ok = tg >= 0
        edited = h.clone(); edited[rows] = h[rows] - edit; ne = -torch.log_softmax(torch.nn.functional.pad(edited, (0, V - D)), -1)[torch.arange(4), tg.clamp(min=0)]
        nb = -torch.log_softmax(base, -1)[torch.arange(4), tg.clamp(min=0)]
        self.assertAlmostEqual(item["dloss"][0], float((ne - nb)[ok].mean()), places=5)

    def test_the_pooled_gram_is_the_gram_of_the_row_weighted_mean_vectors_of_the_contexts(self):
        out = self.run_chunk(); f, kind = 2, "real"
        cells = {c: [] for c in tc.CELLS}
        for s in self.contexts[f]:
            ref = self.reference(f, s, kind)
            for c in tc.CELLS:
                cells[c].append({"vector": ref[c]["vector"], "n_rows": ref[c]["n_rows"]})
        pooled = {c: {"vector": tc.pool_contexts(cells[c])} for c in tc.CELLS}
        self.assertTrue(np.allclose(out["pooled"][(f, kind)]["gram"], tc.gram(pooled).numpy(), atol=1e-6)); self.assertEqual(out["pooled"][(f, kind)]["n_rows"], 3)

    def test_the_activity_counts_of_the_quantized_state_are_recorded(self):
        out = self.run_chunk()
        self.assertEqual(out["items"][(2, 11, "real")]["n_active_q"], 0); self.assertEqual(out["items"][(2, 10, "real")]["n_active_q"], 2)

    def test_a_context_without_an_fp_active_position_is_reported_not_dropped_silently(self):
        self.positions[(2, 11)] = torch.zeros(0, dtype=torch.long); self.amps["FP"][(2, 11)] = torch.zeros(0); self.amps["Q"][(2, 11)] = torch.zeros(0)
        out = self.run_chunk(features=(2,))
        self.assertEqual(out["skipped"], [(2, 11, "no_fp_active_position")]); self.assertNotIn((2, 11, "real"), out["items"]); self.assertEqual(len(out["items"]), 5)

    def test_a_missing_cache_and_a_nonfinite_suffix_name_the_item(self):
        del self.caches["Q"][11]
        with self.assertRaisesRegex(KeyError, "cached residual of state Q for context sentence 11"):
            self.run_chunk()
        self.setUp(); self.amps["Q"][(2, 10)] = torch.tensor([0.4])
        with self.assertRaisesRegex(ValueError, "amplitudes"):
            self.run_chunk()
        self.setUp(); bad = dict(self.suf); bad["Q"] = lambda x: x[..., :1] / 0.0 + torch.zeros(*x.shape[:-1], V)
        with self.assertRaisesRegex(ValueError, "context"):
            tr.process_chunk([2], {2: [10]}, self.caches, self.positions, self.amps, self.dirs, self.tokens, bad, self.switches)


class Constants(unittest.TestCase):
    def test_the_direction_kinds_and_seeds_are_those_of_the_frozen_spec(self):
        import phase1_transfer_freeze as tf
        self.assertEqual(list(tr.DIRECTION_KINDS), tf.SPEC["direction_kinds"]); self.assertEqual(tr.RANDOM_SEEDS, tf.SPEC["random_direction_seeds"])


if __name__ == "__main__":
    unittest.main()
