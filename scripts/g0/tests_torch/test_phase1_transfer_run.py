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


class ProcessItems(Fixture, unittest.TestCase):
    """The generic item processor behind the feature-level and the group-level runs: an item brings its own edits."""
    def items(self):
        out = []
        for f in (2,):
            for s in self.contexts[f]:
                rows = self.positions[(f, s)] + 1; d = self.dirs[f]["real"]
                out.append({"key": (f, s, "real"), "rows": rows, "h_fp": self.caches["FP"][s].float(), "h_q": self.caches["Q"][s].float(), "edit_fp": self.amps["FP"][(f, s)][:, None] * d[None, :],
                            "edit_native": self.amps["Q"][(f, s)][:, None] * d[None, :], "targets": tr.context_targets(self.tokens[s]), "pool": (f, "real"), "n_active_q": int((self.amps["Q"][(f, s)] > 0).sum())})
        return out

    def test_an_item_equals_the_cells_of_the_core_and_the_weights_are_switched_once_per_state(self):
        out = tr.process_items(self.items(), self.suf, self.switches)
        self.assertEqual(self.log, ["FP", "Q"]); self.assertEqual(set(out["items"]), {(2, 10, "real"), (2, 11, "real")})
        for (f, s, kind), it in out["items"].items():
            res = {"FP": self.caches["FP"][s].float(), "Q": self.caches["Q"][s].float()}; rows = self.positions[(f, s)] + 1; d = self.dirs[f][kind]
            ref = tc.factorial_effects(res, self.suf, rows, self.amps["FP"][(f, s)][:, None] * d[None, :], native_edit=self.amps["Q"][(f, s)][:, None] * d[None, :], targets=tr.context_targets(self.tokens[s]))
            self.assertTrue(np.allclose(it["gram"], tc.gram(ref).numpy(), atol=1e-6)); self.assertEqual(it["n_target_rows"], len(self.tokens[s]))

    def test_items_with_a_pool_key_are_pooled_and_items_without_one_are_not(self):
        its = self.items(); out = tr.process_items(its, self.suf, self.switches)
        self.assertEqual(list(out["pooled"]), [(2, "real")]); self.assertEqual(out["pooled"][(2, "real")]["n_contexts"], 2)
        for it in its:
            it["pool"] = None
        self.assertEqual(tr.process_items(its, self.suf, self.switches)["pooled"], {})

    def test_the_item_keys_must_be_unique_and_an_item_without_rows_is_skipped_with_its_key(self):
        its = self.items()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            tr.process_items(its + [dict(its[0])], self.suf, self.switches)
        out0 = tr.process_items(its, self.suf, self.switches); self.assertEqual(out0["items"][(2, 10, "real")]["n_target_rows"], 3)
        its[0]["targets"] = torch.tensor([0, 1, 2, -1]); self.assertEqual(tr.process_items(its, self.suf, self.switches)["items"][(2, 10, "real")]["n_target_rows"], 3)       # token id 0 is a target
        its = self.items()
        its[1]["rows"] = torch.zeros(0, dtype=torch.long); its[1]["edit_fp"] = torch.zeros(0, D); its[1]["edit_native"] = torch.zeros(0, D)
        out = tr.process_items(its, self.suf, self.switches); self.assertEqual(out["skipped"], [((2, 11, "real"), "no_edited_row")]); self.assertEqual(len(out["items"]), 1)


class GroupEdits(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(8); self.W = torch.randn(6, D, generator=g)
        self.act_fp = torch.tensor([[1.0, 0.0, 2.0, 0.0, 0.0, 0.5], [0.0, 0.0, 0.0, 3.0, 0.0, 0.0], [0.0, 1.5, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        self.act_q = torch.tensor([[0.8, 0.0, 1.8, 0.0, 0.0, 0.6], [0.0, 0.0, 0.0, 2.5, 0.0, 0.1], [0.0, 1.4, 0.0, 0.0, 0.0, 0.0], [0.2, 0.0, 0.0, 0.0, 0.0, 0.0]])

    def test_the_edit_is_the_sum_of_activation_times_decoder_row_over_the_group_at_the_rows_where_it_is_active_in_fp(self):
        rows, e_fp, e_q = tr.group_edit(self.act_fp, self.act_q, self.W, [0, 2, 3])
        self.assertEqual(rows.tolist(), [1, 2]); self.assertEqual(tuple(e_fp.shape), (2, D))                                      # content rows 0 and 1 (the third has no active group feature), shifted by the BOS row
        self.assertTrue(torch.allclose(e_fp[0], 1.0 * self.W[0] + 2.0 * self.W[2], atol=1e-6)); self.assertTrue(torch.allclose(e_fp[1], 3.0 * self.W[3], atol=1e-6))
        self.assertTrue(torch.allclose(e_q[0], 0.8 * self.W[0] + 1.8 * self.W[2], atol=1e-6)); self.assertTrue(torch.allclose(e_q[1], 2.5 * self.W[3], atol=1e-6))

    def test_a_row_that_is_active_only_in_the_quantized_state_is_not_edited_and_an_inactive_group_gives_no_rows(self):
        rows, e_fp, e_q = tr.group_edit(self.act_fp, self.act_q, self.W, [0, 2, 3]); self.assertNotIn(4, rows.tolist())            # row 3 of the content has only a Q activation of feature 0
        rows, e_fp, e_q = tr.group_edit(self.act_fp, self.act_q, self.W, [4]); self.assertEqual(len(rows), 0); self.assertEqual(tuple(e_fp.shape), (0, D))

    def test_group_items_cover_the_group_and_the_control_draws_with_shared_residuals(self):
        sets = {"group": [0, 2, 3], "control0": [1], "control1": [5], "control2": [4]}
        h_fp, h_q = torch.randn(5, D), torch.randn(5, D); tg = torch.tensor([1, 2, 3, 4, -1])
        its, skipped = tr.group_items("eng", 7, h_fp, h_q, self.act_fp, self.act_q, self.W, sets, tg)
        self.assertEqual([i["key"] for i in its], [("eng", 7, "group"), ("eng", 7, "control0"), ("eng", 7, "control1")]); self.assertEqual(skipped, [(("eng", 7, "control2"), "no_active_group_feature")])
        self.assertTrue(all(i["h_fp"] is h_fp and i["h_q"] is h_q for i in its)); self.assertTrue(all(i["pool"] == ("eng", i["key"][2]) for i in its)); self.assertEqual(its[0]["rows"].tolist(), [1, 2])
        self.assertEqual(its[0]["n_active_q"], 2)                                                                     # counted over the edited rows only (the Q-only activation at content row 3 does not count)


class FakeModel:
    """A tiny stand-in for the real model: the prefix (embedding times a state-dependent scale), a ReLU 'SAE' and a tanh suffix with a state-dependent matrix. prefix_differs / suffix_differs decide which part
    of the model changes between FP and the quantized states (both off: the states are identical)."""
    d, V, F = 6, 10, 8

    def __init__(self, prefix_differs=True, suffix_differs=True):
        g = torch.Generator().manual_seed(11)
        self.E = torch.randn(10, self.d, generator=g); self.We = torch.randn(self.d, self.F, generator=g); self.Ws = torch.randn(self.d, self.V, generator=g); self.W_dec = torch.randn(self.F, self.d, generator=g)
        self.prefix_differs, self.suffix_differs, self.state, self.calls = prefix_differs, suffix_differs, "FP", []

    def setup(self, label):
        self.state = label; self.calls.append(label)

    def lower(self, tokens):
        s = 1.15 if (self.prefix_differs and self.state != "FP") else 1.0
        return self.E[tokens] * s

    def encode(self, x):
        return torch.relu(x @ self.We + 0.4)

    def suffix(self, x):
        W = self.Ws * (1.3 if (self.suffix_differs and self.state != "FP") else 1.0)
        return torch.tanh(x @ W)


def fake_core():
    return {"feature_ids": [0, 1, 2, 3], "direction_ids": [1, 0, 3, 2], "feature_language_index": [0, 0, 1, 1], "contexts": [[0, 1, 2], [1, 2, 3], [6, 7, 8], [7, 8, 9]]}


def fake_dev():
    g = torch.Generator().manual_seed(5)
    return [torch.randint(2, 10, (int(torch.randint(3, 7, (1,), generator=g)),), generator=g).tolist() for _ in range(12)]


def run_fake(model, out, states=("INT6",), per_chunk=2, freeze_sha="f", **kw):
    import phase1_lib as p1
    batcher = lambda sents: p1.make_batches({0: sents}, 1, "mask", 0, 256, 4, None)
    return tr.run_transfer(list(states), model.setup, model.lower, model.suffix, model.encode, model.W_dec, kw.pop("core", fake_core()), fake_dev(), batcher, out, freeze_sha, per_chunk, **kw)


def read_chunks(out, label):
    items = {}
    for p in sorted(Path(out).glob(f"transfer_{label}_c*.pt")):
        items.update(torch.load(p, weights_only=False)["payload"]["items"])
    return items


class Subset(unittest.TestCase):
    def test_the_smoke_subset_is_evenly_spaced_in_features_and_takes_the_first_contexts(self):
        core = {"feature_ids": list(range(10)), "direction_ids": list(range(1, 10)) + [0], "feature_language_index": [0] * 10, "contexts": [[i, i + 1, i + 2] for i in range(10)]}
        s = tr.subset_core(core, 4, 2)
        self.assertEqual(s["feature_ids"], [0, 3, 6, 9]); self.assertEqual(s["direction_ids"], [1, 4, 7, 0]); self.assertEqual(s["contexts"], [[0, 1], [3, 4], [6, 7], [9, 10]])
        self.assertEqual(tr.subset_core(core, None, None), core); self.assertEqual(tr.subset_core(core, 100, None)["feature_ids"], list(range(10)))


class RunTransfer(unittest.TestCase):
    def test_all_items_of_all_features_contexts_and_directions_are_written_for_every_state(self):
        with tempfile.TemporaryDirectory() as out:
            m = FakeModel(); meta = run_fake(m, out, states=("INT16", "INT6"))
            for label in ("INT16", "INT6"):
                items = read_chunks(out, label); core = fake_core()
                want = {(f, s, k) for f, c in zip(core["feature_ids"], core["contexts"]) for s in c for k in tr.DIRECTION_KINDS}
                skipped = {(f, s) for f, s, k in want} - {(f, s) for f, s, k in items}
                self.assertEqual(set(items) | {(f, s, k) for f, s in skipped for k in tr.DIRECTION_KINDS}, want)
                for it in items.values():
                    self.assertEqual(it["gram"].shape, (5, 5)); self.assertTrue(np.allclose(it["gram"], it["gram"].T))
            self.assertEqual(meta["states"]["INT6"]["chunks"], 2); self.assertEqual(meta["n_features"], 4)

    def test_identical_states_give_zero_context_weight_and_interaction_terms_and_a_zero_error(self):
        with tempfile.TemporaryDirectory() as out:
            run_fake(FakeModel(prefix_differs=False, suffix_differs=False), out)
            for it in read_chunks(out, "INT6").values():
                n = tc.decomposition_norms(torch.tensor(it["gram"])); self.assertAlmostEqual(n["total"], 0.0, places=5); self.assertAlmostEqual(n["C"] + n["W"] + n["I"], 0.0, places=5)

    def test_a_change_of_the_suffix_only_gives_no_context_term_and_a_change_of_the_prefix_only_no_weight_term(self):
        with tempfile.TemporaryDirectory() as out:
            run_fake(FakeModel(prefix_differs=False, suffix_differs=True), out); n = [tc.decomposition_norms(torch.tensor(i["gram"])) for i in read_chunks(out, "INT6").values()]
            self.assertTrue(all(x["C"] < 1e-5 for x in n)); self.assertTrue(any(x["W"] > 1e-3 for x in n))
        with tempfile.TemporaryDirectory() as out:
            run_fake(FakeModel(prefix_differs=True, suffix_differs=False), out); n = [tc.decomposition_norms(torch.tensor(i["gram"])) for i in read_chunks(out, "INT6").values()]
            self.assertTrue(all(x["W"] < 1e-5 for x in n)); self.assertTrue(any(x["C"] > 1e-3 for x in n))

    def test_the_call_order_of_the_states_is_fp_first_then_one_cache_per_state_and_two_switches_per_chunk(self):
        with tempfile.TemporaryDirectory() as out:
            m = FakeModel(); run_fake(m, out, states=("INT6",), per_chunk=2)
            self.assertEqual(m.calls[:2], ["FP", "INT6"]); self.assertEqual(m.calls[2:], ["FP", "INT6", "FP", "INT6", "FP"]); self.assertEqual(m.state, "FP")     # two chunks, then the weights are restored

    def test_a_second_run_resumes_from_the_chunk_files_without_recomputing_them(self):
        with tempfile.TemporaryDirectory() as out:
            run_fake(FakeModel(), out); first = read_chunks(out, "INT6"); m = FakeModel(); meta = run_fake(m, out)
            self.assertEqual(meta["states"]["INT6"]["resumed"], 2); self.assertEqual(m.calls, ["FP", "INT6", "FP"])                                              # only the caches were rebuilt, then the restore
            second = read_chunks(out, "INT6"); self.assertEqual(set(first), set(second))
            for k in first:
                self.assertTrue(np.array_equal(first[k]["gram"], second[k]["gram"]))

    def test_a_chunk_file_of_another_freeze_is_a_stale_error(self):
        with tempfile.TemporaryDirectory() as out:
            run_fake(FakeModel(), out)
            with self.assertRaisesRegex(ValueError, "stale"):
                run_fake(FakeModel(), out, freeze_sha="other")

    def test_the_subset_arguments_limit_features_and_contexts(self):
        with tempfile.TemporaryDirectory() as out:
            meta = run_fake(FakeModel(), out, max_features=2, max_contexts=1)
            self.assertEqual(meta["n_features"], 2); self.assertEqual({(f, s) for f, s, k in read_chunks(out, "INT6")}, {(0, 0), (3, 7)})

    def test_the_matched_direction_of_every_feature_is_taken_from_the_core_and_the_sham_gate_is_exact(self):
        with tempfile.TemporaryDirectory() as out:
            meta = run_fake(FakeModel(), out)
            self.assertTrue(meta["gates"]["sham_exact"]); self.assertEqual(meta["gates"]["sham_max_abs"], 0.0)


class NoisyQ(FakeModel):
    """The suffix of the quantized state is not deterministic: the unedited and the edited row would take different computations, so the sham gate must report it."""
    def suffix(self, x):
        z = super().suffix(x)
        return z + (0.01 * torch.randn_like(z) if self.state != "FP" else 0.0)


class Reference(unittest.TestCase):
    def test_an_item_of_the_run_equals_the_cells_computed_by_hand_from_the_fake_model(self):
        m = FakeModel(); core = fake_core(); f, s, kind = 1, 2, "matched_swapped"; dev = fake_dev()
        with tempfile.TemporaryDirectory() as out:
            run_fake(m, out); got = read_chunks(out, "INT6")[(f, s, kind)]
        toks = torch.tensor([[1] + dev[s]])
        def prefix(state):
            m.state = state; return m.lower(toks)[0]
        h_fp, h_q = prefix("FP"), prefix("INT6"); m.state = "FP"
        act_fp = m.encode(h_fp[1:])[:, f]; pos = torch.nonzero(act_fp > 0, as_tuple=True)[0]; a_fp = act_fp[pos]
        m.state = "INT6"; a_q = m.encode(h_q[1:])[:, f][pos]
        d = m.W_dec[core["direction_ids"][core["feature_ids"].index(f)]]                                              # the matched swapped direction
        tg = tr.context_targets(dev[s]); rows = pos + 1
        def cell(h, state, amp):
            m.state = state; return tc.effect(h, rows, amp[:, None] * d[None, :], m.suffix, tg)
        cells = {"FP/FP": cell(h_fp, "FP", a_fp), "Q/FP": cell(h_q, "FP", a_fp), "FP/Q": cell(h_fp, "INT6", a_fp), "Q/Q": cell(h_q, "INT6", a_fp), "Q/Q-native": cell(h_q, "INT6", a_q)}
        self.assertTrue(np.allclose(got["gram"], tc.gram(cells).numpy(), atol=1e-6)); self.assertEqual(got["n_rows"], len(pos))
        for i, c in enumerate(tc.CELLS):
            self.assertAlmostEqual(got["dloss"][i], cells[c]["dloss"], places=5)


class NoisyFP(FakeModel):
    def suffix(self, x):
        z = super().suffix(x)
        return z + (0.01 * torch.randn_like(z) if self.state == "FP" else 0.0)


class NoisyBOS(FakeModel):
    """Noise on row 0 only (never an edited row): the edited-row vector stays zero, the loss change over all rows does not."""
    def suffix(self, x):
        z = super().suffix(x).clone(); z[..., 0, :] = z[..., 0, :] + 0.05 * torch.randn_like(z[..., 0, :]); return z


class ShamGate(unittest.TestCase):
    def test_a_nondeterministic_quantized_suffix_fails_the_sham_gate(self):
        with tempfile.TemporaryDirectory() as out:
            meta = run_fake(NoisyQ(), out, states=("INT6",), per_chunk=4)
        self.assertFalse(meta["gates"]["sham_exact"]); self.assertGreater(meta["gates"]["sham_max_abs"], 0.0)

    def test_noise_in_the_fp_suffix_and_noise_on_unedited_rows_fail_the_gate_too(self):
        with tempfile.TemporaryDirectory() as out:
            self.assertFalse(run_fake(NoisyFP(), out, states=("INT6",), per_chunk=4)["gates"]["sham_exact"])
        with tempfile.TemporaryDirectory() as out:
            meta = run_fake(NoisyBOS(), out, states=("INT6",), per_chunk=4); self.assertFalse(meta["gates"]["sham_exact"])

    def test_the_chunk_size_decides_the_number_of_chunks_and_of_weight_switches(self):
        with tempfile.TemporaryDirectory() as out:
            m = FakeModel(); meta = run_fake(m, out, states=("INT6",), per_chunk=1)
        self.assertEqual(meta["states"]["INT6"]["chunks"], 4); self.assertEqual(m.calls.count("INT6"), 1 + 4)


class FreezeCheck(unittest.TestCase):
    def setUp(self):
        import json as _json
        import phase1_transfer_freeze as tf
        self.tf = tf; self.tmp = tempfile.TemporaryDirectory(); d = Path(self.tmp.name); (d / "f").write_text("x")
        self.core = {"feature_ids": [3, 5], "direction_ids": [5, 3], "feature_language_index": [0, 0], "contexts": [[1, 2], [2, 3]], "excluded_features": [], "parameters": {"M": 2}}
        self.freeze = _json.loads(_json.dumps(tf.build_freeze({"f": d / "f"}, self.core, "abc", "now", root=d)))

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_committed_core_and_the_spec_of_the_code_must_equal_the_freeze(self):
        tr.check_core_against_freeze(self.freeze, self.core)
        for key, mutate in (("feature_ids", lambda c: c["feature_ids"].__setitem__(0, 4)), ("direction_ids", lambda c: c["direction_ids"].__setitem__(0, 6)), ("contexts", lambda c: c["contexts"][0].append(9))):
            import json as _json
            c = _json.loads(_json.dumps(self.core)); mutate(c)
            with self.assertRaisesRegex(ValueError, key):
                tr.check_core_against_freeze(self.freeze, c)

    def test_a_changed_spec_in_the_code_is_reported(self):
        self.freeze["spec"]["bootstrap"]["replicates"] = 5
        with self.assertRaisesRegex(ValueError, "spec"):
            tr.check_core_against_freeze(self.freeze, self.core)

    def test_the_sbatch_wrapper_and_the_deploy_script_name_the_frozen_files(self):
        root = Path(__file__).resolve().parents[3]
        wrapper = (root / "scripts/g0/run_transfer.sbatch").read_text(); deploy = (root / "scripts/g0/deploy_phase1.sh").read_text()
        for needle in ("phase1_transfer_run.py", "reports/intervention_transfer/freeze.json", "reports/intervention_transfer/core_sample.json", "HF_HUB_OFFLINE=1"):
            self.assertIn(needle, wrapper)
        for needle in ("reports/intervention_transfer/freeze.json", "reports/intervention_transfer/core_sample.json"):
            self.assertIn(needle, deploy)


class MakeSuffix(unittest.TestCase):
    def test_the_readout_of_f5_takes_rows_so_the_suffix_flattens_batch_and_length_and_restores_them(self):
        g = torch.Generator().manual_seed(2); W = torch.randn(D, V, generator=g)

        def logits_fn(h):                                                       # like the F5 readout: rows [n, d] only
            if h.dim() != 2:
                raise ValueError("rows expected")
            return h @ W
        suffix = tr.make_suffix(lambda x: x * 2.0, logits_fn, "cpu"); x = torch.randn(2, 4, D, generator=g)
        z = suffix(x)
        self.assertEqual(tuple(z.shape), (2, 4, V)); self.assertTrue(torch.allclose(z, (x * 2.0) @ W, atol=1e-6))

    def test_the_input_is_moved_to_the_device_of_the_model(self):
        seen = []
        suffix = tr.make_suffix(lambda x: seen.append(x.device.type) or x, lambda h: h, "meta"); suffix(torch.zeros(1, 2, 3)); self.assertEqual(seen, ["meta"])             # a device that is not the input's own

    def test_main_builds_the_suffix_with_the_helper(self):
        src = (Path(__file__).resolve().parents[1] / "phase1_transfer_run.py").read_text()
        self.assertIn("suffix = make_suffix(upper_final, logits_fn, DEVICE)", src); self.assertNotIn("suffix = lambda x: logits_fn(upper_final", src)


class GateConstants(unittest.TestCase):
    def test_the_instrument_gate_thresholds_are_the_amended_ones_and_not_looser_than_documented(self):
        self.assertEqual((tr.GATE_MIN_TOP1, tr.GATE_MAX_ABS, tr.GATE_PROBE_SENTENCES, tr.GATE_ROWS), (0.90, 2.0, 8, 512))
        src = (Path(__file__).resolve().parents[1] / "phase1_transfer_run.py").read_text()
        self.assertIn("min_top1=GATE_MIN_TOP1", src); self.assertIn("max_abs=GATE_MAX_ABS", src); self.assertNotIn("min_top1=0.95", src)


class Constants(unittest.TestCase):
    def test_the_direction_kinds_and_seeds_are_those_of_the_frozen_spec(self):
        import phase1_transfer_freeze as tf
        self.assertEqual(list(tr.DIRECTION_KINDS), tf.SPEC["direction_kinds"]); self.assertEqual(tr.RANDOM_SEEDS, tf.SPEC["random_direction_seeds"])


if __name__ == "__main__":
    unittest.main()
