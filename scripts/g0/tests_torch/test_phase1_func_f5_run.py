import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_func_f5_metrics as met
import phase1_func_f5_run as r5
import phase1_func_intervene as fi
import phase1_lib as p1
from test_phase1_func_intervene import ToySAE
from test_phase1_func_run import ToyModel, corpus, V, D, F, BOS


class ToyStack:
    """The toy model split into lower(tokens) -> residual, upper_final(residual) -> final hidden and logits_fn(hidden) -> float32 logits (the three callables of the F5 runner). linear=True makes the upper half a purely
    position-wise linear map (hidden = resid @ A), for which the effect of a single-feature ablation has a closed form."""
    def __init__(self, seed=0, linear=False, perturb=None):
        self.m = ToyModel(seed); self.linear = linear
        g = torch.Generator().manual_seed(seed + 100); self.A = torch.randn(D, D, generator=g) * 0.5
        self.perturb = perturb or (lambda A: A)

    def lower(self, tokens):
        return self.m.lower(tokens)

    def upper_final(self, resid):
        if self.linear:
            return resid.float() @ self.perturb(self.A)
        return torch.tanh(self.m._cmean(resid) @ self.m.M2 + resid)

    def logits_fn(self, h):
        return h.float() @ self.m.U


def sentences(n=40, seed=0):
    return corpus(seed=seed, sizes=(n,), lo=3, hi=9)[0]


def batches_of(sents, batch_size=7):
    return p1.make_batches({0: sents}, BOS, "mask", 0, 256, batch_size)


class SelectContexts(unittest.TestCase):
    def test_the_contexts_are_the_most_active_sentences_with_ties_to_the_smaller_index(self):
        ma = torch.tensor([[1.0, 0.0], [3.0, 2.0], [3.0, 0.0], [2.0, 5.0], [0.0, 0.0]])                       # 5 sentences, 2 features
        r = r5.select_contexts(ma, M=3, min_ctx=1)
        self.assertEqual(r["contexts"][0].tolist(), [1, 2, 3])                                               # 3.0 (index 1) before 3.0 (index 2), then 2.0
        self.assertEqual(r["contexts"][1].tolist(), [3, 1, -1]); self.assertEqual(r["n_ctx"].tolist(), [3, 2])

    def test_only_active_sentences_count_and_rare_features_are_excluded(self):
        ma = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 1.0, 0.0]])
        r = r5.select_contexts(ma, M=8, min_ctx=3)
        self.assertEqual(r["n_active"].tolist(), [4, 1, 0]); self.assertEqual(r["excluded"].tolist(), [False, True, True]); self.assertEqual(r["n_ctx"].tolist(), [4, 1, 0])
        self.assertEqual(r["contexts"][2].tolist(), [-1] * 8)

    def test_a_feature_with_exactly_min_ctx_active_sentences_is_kept(self):
        r = r5.select_contexts(torch.tensor([[1.0, 1.0], [2.0, 1.0], [3.0, 0.0]]), M=4, min_ctx=2)
        self.assertEqual(r["n_active"].tolist(), [3, 2]); self.assertEqual(r["excluded"].tolist(), [False, False])

    def test_a_zero_activation_is_not_a_context(self):
        r = r5.select_contexts(torch.tensor([[0.0], [0.0], [1.0]]), M=2, min_ctx=1)
        self.assertEqual(r["contexts"][0].tolist(), [2, -1])


class Plan(unittest.TestCase):
    def test_unique_sentences_in_order_of_first_use_and_per_feature_local_indices(self):
        contexts = torch.tensor([[5, 2, -1], [2, 7, 5], [9, -1, -1]]); excluded = torch.tensor([False, False, True])
        dev = [[i] for i in range(10)]
        pl = r5.plan_contexts(contexts, excluded, dev)
        self.assertEqual(pl["dev_index"], [5, 2, 7]); self.assertEqual(pl["sentences"], [[5], [2], [7]])
        self.assertEqual(pl["ctx"], {0: [0, 1], 1: [1, 2, 0]})                                               # the excluded feature (row 2) has no contexts and adds no sentence


class StateAndPositions(unittest.TestCase):
    def setUp(self):
        self.stack = ToyStack(); self.sae = ToySAE(); self.sents = sentences()
        self.feat = torch.tensor([3, 11, 20, 7])
        ma = r5.max_activation_per_sentence(self.stack.lower, self.sae.encode, batches_of(self.sents), torch.arange(F), len(self.sents))
        self.ma = ma
        sel = r5.select_contexts(ma[:, self.feat], M=4, min_ctx=2)
        self.plan = r5.plan_contexts(sel["contexts"], sel["excluded"], self.sents); self.sel = sel

    def test_the_maximum_activation_per_sentence_equals_the_brute_force_value(self):
        for s in (0, 5, 17):
            tokens = torch.tensor([[BOS] + self.sents[s]]); valid = torch.zeros_like(tokens, dtype=torch.bool); valid[:, 1:] = True
            a = fi.encode_valid(self.stack.lower(tokens), valid, self.sae.encode)
            torch.testing.assert_close(self.ma[s], a.max(0).values)

    def test_the_full_precision_active_positions_equal_the_brute_force_positions(self):
        bt = batches_of(self.plan["sentences"], 3)
        pos = r5.fp_active_positions(self.stack.lower, self.sae.encode, bt, self.plan, self.feat)
        self.assertTrue(len(pos) > 0)
        for (k, s), p in pos.items():
            sent = self.plan["sentences"][s]
            tokens = torch.tensor([[BOS] + sent]); valid = torch.zeros_like(tokens, dtype=torch.bool); valid[:, 1:] = True
            a = fi.encode_valid(self.stack.lower(tokens), valid, self.sae.encode)[:, self.feat[k]]
            self.assertEqual(p.tolist(), torch.nonzero(a > 0, as_tuple=True)[0].tolist())
        self.assertEqual(len(pos), sum(len(v) for v in self.plan["ctx"].values()))                          # one entry per (feature, context sentence)

    def test_the_state_cache_holds_the_residual_rows_and_the_activations_at_those_positions(self):
        bt = batches_of(self.plan["sentences"], 3)
        pos = r5.fp_active_positions(self.stack.lower, self.sae.encode, bt, self.plan, self.feat)
        cache = r5.cache_state(self.stack.lower, self.sae.encode, bt, self.plan, self.feat, pos)
        for s, sent in enumerate(self.plan["sentences"]):
            tokens = torch.tensor([[BOS] + sent]); want = self.stack.lower(tokens)[0]
            self.assertEqual(tuple(cache["resid"][s].shape), (len(sent) + 1, D)); torch.testing.assert_close(cache["resid"][s], want)    # BOS row included, padding trimmed
        for (k, s), a in cache["act"].items():
            self.assertEqual(len(a), len(pos[(k, s)])); self.assertTrue(bool((a > 0).all()))                    # in the full-precision state the feature is active exactly at those positions


class Fingerprint(unittest.TestCase):
    def setUp(self):
        self.stack = ToyStack(linear=True); self.sae = ToySAE(); self.sents = sentences(60, seed=3)
        self.feat = torch.arange(F)
        ma = r5.max_activation_per_sentence(self.stack.lower, self.sae.encode, batches_of(self.sents), self.feat, len(self.sents))
        self.sel = r5.select_contexts(ma, M=5, min_ctx=3)
        self.plan = r5.plan_contexts(self.sel["contexts"], self.sel["excluded"], self.sents)
        self.bt = batches_of(self.plan["sentences"], 4)
        self.pos = r5.fp_active_positions(self.stack.lower, self.sae.encode, self.bt, self.plan, self.feat)
        self.cache = r5.cache_state(self.stack.lower, self.sae.encode, self.bt, self.plan, self.feat, self.pos)

    def fp(self, k, **kw):
        return r5.fingerprint_feature(k, self.plan, self.cache, self.pos, self.sae.W_dec[self.feat[k]], self.stack.upper_final, self.stack.logits_fn, V, "cpu", **kw)

    def test_the_fingerprint_of_a_linear_model_is_the_centred_direct_effect_averaged_over_the_ablated_positions(self):
        ks = [k for k in self.plan["ctx"]][:6]
        self.assertTrue(len(ks) > 0)
        for k in ks:
            out = self.fp(k)
            acts = torch.cat([self.cache["act"][(k, s)] for s in self.plan["ctx"][k]])
            want = -(acts.mean()) * (self.sae.W_dec[self.feat[k]] @ self.stack.A @ self.stack.m.U)             # -a d A U averaged: the mean activation times the direct effect
            want = want - want.mean()
            torch.testing.assert_close(out["D"], want, rtol=1e-4, atol=1e-5)
            self.assertEqual(out["n_pos"], len(acts)); self.assertEqual(out["n_active_q"], len(acts))

    def test_a_sham_ablation_of_amplitude_zero_changes_nothing_exactly(self):
        k = next(iter(self.plan["ctx"]))
        out = self.fp(k, sham=True)
        self.assertEqual(float(out["D"].abs().max()), 0.0); self.assertEqual(out["dnll"], 0.0)

    def test_a_feature_ablated_at_the_last_content_position_is_included_in_the_fingerprint_but_not_in_the_loss(self):
        # found by the closed-form test: the last content position has a logit vector (it predicts the end) but no target token
        sent = self.sents[0]; n = len(sent)
        plan = {"sentences": [sent], "ctx": {0: [0]}}; tokens = torch.tensor([[BOS] + sent])
        cache = {"resid": {0: self.stack.lower(tokens)[0]}, "act": {(0, 0): torch.tensor([2.0])}}; positions = {(0, 0): torch.tensor([n - 1])}
        d = self.sae.W_dec[3]
        out = r5.fingerprint_feature(0, plan, cache, positions, d, self.stack.upper_final, self.stack.logits_fn, V, "cpu")
        want = -2.0 * (d @ self.stack.A @ self.stack.m.U); want = want - want.mean()
        torch.testing.assert_close(out["D"], want, rtol=1e-4, atol=1e-5); self.assertEqual(out["n_pos"], 1)
        self.assertEqual(out["dnll"], 0.0)                                                                  # nothing earlier in the sentence is affected (position-wise map) and the ablated row has no target

    def test_the_loss_change_is_the_mean_per_predicted_token_over_the_context_sentences(self):
        k = next(iter(self.plan["ctx"])); out = self.fp(k)
        tot, n = 0.0, 0
        for s in self.plan["ctx"][k]:
            sent = self.plan["sentences"][s]; tokens = torch.tensor([[BOS] + sent]); resid = self.stack.lower(tokens)[0].clone(); base = resid.clone()
            p = self.pos[(k, s)]; a = self.cache["act"][(k, s)]
            resid[p + 1] = resid[p + 1] - a[:, None] * self.sae.W_dec[self.feat[k]][None, :]
            for x, sign in ((resid, 1.0), (base, -1.0)):
                z = self.stack.logits_fn(self.stack.upper_final(x))[:len(sent)]
                tot += sign * float(-torch.log_softmax(z, -1)[torch.arange(len(sent)), torch.tensor(sent)].sum())
            n += len(sent)
        self.assertAlmostEqual(out["dnll"], tot / n, places=4)

    def test_reordering_the_contexts_gives_the_same_fingerprint(self):
        k = next(k for k in self.plan["ctx"] if len(self.plan["ctx"][k]) >= 3)
        a = self.fp(k)["D"]
        orig = list(self.plan["ctx"][k]); self.plan["ctx"][k] = orig[::-1]
        try:
            b = self.fp(k)["D"]
        finally:
            self.plan["ctx"][k] = orig
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)                                              # the numerical floor of the toy is rounding only

    def test_reversing_the_contexts_and_padding_more_leave_the_fingerprint_unchanged_in_the_toy_and_sham_stays_exact(self):
        k = next(k for k in self.plan["ctx"] if len(self.plan["ctx"][k]) >= 3)
        torch.testing.assert_close(self.fp(k)["D"], self.fp(k, reverse=True, extra_pad=5)["D"], rtol=1e-5, atol=1e-6)
        self.assertEqual(float(self.fp(k, sham=True, reverse=True, extra_pad=5)["D"].abs().max()), 0.0)

    def test_a_feature_that_is_silent_in_the_state_has_a_zero_fingerprint_and_the_count_says_so(self):
        k = next(iter(self.plan["ctx"])); silent = {key: torch.zeros_like(v) for key, v in self.cache["act"].items()}
        cache = dict(self.cache, act=silent)
        out = r5.fingerprint_feature(k, self.plan, cache, self.pos, self.sae.W_dec[self.feat[k]], self.stack.upper_final, self.stack.logits_fn, V, "cpu")
        self.assertEqual(float(out["D"].abs().max()), 0.0); self.assertEqual(out["n_active_q"], 0); self.assertGreater(out["n_pos"], 0)

    def test_a_planted_change_of_the_downstream_map_along_the_decoder_direction_changes_that_features_function_only(self):
        k = next(k for k in self.plan["ctx"] if len(self.pos[(k, self.plan["ctx"][k][0])]) > 0)
        dj = self.sae.W_dec[self.feat[k]]; r = torch.randn(D, generator=torch.Generator().manual_seed(9)) * 2
        stack2 = ToyStack(linear=True, perturb=lambda A: A + torch.outer(dj / (dj @ dj), r) * 3.0)
        base = self.fp(k)["D"]
        q = r5.fingerprint_feature(k, self.plan, self.cache, self.pos, dj, stack2.upper_final, stack2.logits_fn, V, "cpu")["D"]
        cos = float(met.cosine_rows(base[None], q[None])[0][0])
        self.assertLess(cos, 0.9)                                                                           # its own function changed
        k2 = next(x for x in self.plan["ctx"] if x != k)
        d2 = self.sae.W_dec[self.feat[k2]]
        same = float(met.cosine_rows(self.fp(k2)["D"][None], r5.fingerprint_feature(k2, self.plan, self.cache, self.pos, d2, ToyStack(linear=True).upper_final, self.stack.logits_fn, V, "cpu")["D"][None])[0][0])
        self.assertAlmostEqual(same, 1.0, places=5)                                                         # an unchanged stack leaves every other feature at cosine 1


class Similarity(unittest.TestCase):
    def test_the_chunked_similarity_matrix_equals_the_dense_cosine_and_handles_zero_rows(self):
        g = torch.Generator().manual_seed(0); Dfp = torch.randn(7, 50, generator=g); Dq = Dfp + 0.3 * torch.randn(7, 50, generator=g)
        Dfp[2] = 0; Dq[4] = 0                                                                                # no effect in FP for 2, in the quantized model for 4
        S = r5.similarity_matrix(Dfp.half(), Dq.half(), device="cpu", chunk=3)
        want = torch.nn.functional.normalize(Dfp.half().float(), dim=1) @ torch.nn.functional.normalize(Dq.half().float(), dim=1).T
        keep = [i for i in range(7) if i != 2]
        torch.testing.assert_close(S[keep], want[keep], rtol=1e-3, atol=1e-3)
        self.assertTrue(bool(torch.isnan(S[2]).all())); self.assertEqual(float(torch.nan_to_num(S[:, 4]).abs().max()), 0.0)   # a zero candidate matches nothing; a zero query is undefined

    def test_topk_indices_overlap_agrees_with_the_metrics_module(self):
        g = torch.Generator().manual_seed(1); U = torch.randn(6, 40, generator=g); W = U + 0.8 * torch.randn(6, 40, generator=g)
        got = r5.topk_overlap_from_indices(r5.topk_indices(U, 5), r5.topk_indices(W, 5), 5)
        np.testing.assert_allclose(got.numpy(), met.topk_overlap(U.numpy(), W.numpy(), 5), atol=1e-12)


class StatesEndToEnd(unittest.TestCase):
    """The whole orchestration on the toy stack: full precision, the numerical-floor pseudo-state (the same weights, another batching), a 'quantized' state Q with a planted change of the downstream map along one
    feature's decoder direction, and a state S in which one feature is silent. The sample is a SHUFFLED subset of the features, so that a row index is never a feature index."""
    @classmethod
    def setUpClass(cls):
        cls.stack = ToyStack(linear=True); cls.sae = ToySAE(); cls.dev = sentences(80, seed=5); cls.ev = sentences(30, seed=6)
        cls.feat = torch.tensor([29, 3, 17, 8, 22, 1, 12, 25, 5, 30, 14, 9, 20, 7, 26, 11, 2, 19, 23, 15])
        cls.batcher = staticmethod(lambda sents, bs=6: p1.make_batches({0: sents}, BOS, "mask", 0, 256, bs))
        ma = r5.max_activation_per_sentence(cls.stack.lower, cls.sae.encode, cls.batcher(cls.dev), cls.feat, len(cls.dev))
        sel = r5.select_contexts(ma, 5, 3); testable = [int(cls.feat[k]) for k in range(len(cls.feat)) if not bool(sel["excluded"][k])]
        assert len(testable) >= 8
        cls.drift_feature, cls.silent_feature = testable[1], testable[3]
        dj = cls.sae.W_dec[cls.drift_feature]; r = torch.randn(D, generator=torch.Generator().manual_seed(11)) * 3
        cls.state = {"label": "FP"}

        def setup(label):
            cls.state["label"] = label
            cls.stack.perturb = (lambda A: A + torch.outer(dj / (dj @ dj), r)) if label == "Q" else (lambda A: A)
        cls.setup = staticmethod(setup)

        def encode(x):
            a = cls.sae.encode(x)
            if cls.state["label"] == "S":
                a[:, cls.silent_feature] = 0.0
            if cls.state["label"] == "Z":
                a = a * 0.5                                                                                   # every activation halved: the same features, half the scale
            return a
        cls.encode = staticmethod(encode)
        cls.groups = (cls.feat % 3).numpy()
        cls.out = r5.run_f5(["FP", "FP_rebatched", "Q", "Z", "S"], setup, cls.stack.lower, cls.stack.upper_final, cls.stack.logits_fn, encode, cls.sae.W_dec, cls.feat, cls.batcher(cls.ev), cls.dev, cls.batcher,
                            M=5, min_ctx=3, V=V, device="cpu", topk=3, groups=cls.groups)

    def w(self):
        return self.out["stage_a"]["write_features"]

    def test_the_stage_a_result_lists_the_tested_and_the_excluded_features_by_feature_id(self):
        a = self.out["stage_a"]; self.assertEqual(len(a["write_features"]) + int(a["excluded"].sum()), len(self.feat)); self.assertGreaterEqual(len(a["write_features"]), 8)
        self.assertEqual(a["write_features"], [int(self.feat[k]) for k in a["write_rows"]]); self.assertEqual(a["write_rows"], sorted(a["write_rows"]))

    def test_full_precision_against_itself_is_exactly_the_identity_on_the_read_side(self):
        r = self.out["states"]["FP"]["read"]; ok = ~np.isnan(r["corr"])
        self.assertTrue(ok.any()); np.testing.assert_allclose(r["corr"][ok], 1.0, atol=1e-9)
        np.testing.assert_array_equal(r["jaccard"][~np.isnan(r["jaccard"])], 1.0); np.testing.assert_allclose(r["scale"][~np.isnan(r["scale"])], 1.0)

    def test_the_numerical_floor_state_has_similarity_one_and_the_same_retrieval_as_the_identity(self):
        c = self.out["states"]["FP_rebatched"]["write"]
        np.testing.assert_allclose(c["cos"][~np.isnan(c["cos"])], 1.0, atol=1e-5)
        self.assertEqual(c["retrieval"]["top1"], self.out["states"]["FP"]["write"]["retrieval"]["top1"])

    def test_the_identity_state_retrieves_every_feature_whose_fingerprint_is_unique(self):
        c = self.out["states"]["FP"]["write"]
        self.assertGreaterEqual(c["retrieval"]["top1"], 0.9); self.assertEqual(c["retrieval"]["n_undefined"], 0)

    def test_the_within_group_retrieval_uses_the_groups_of_the_tested_features_by_row(self):
        wi = self.out["states"]["FP"]["write"]["retrieval"]["within"]
        self.assertIn("chance", wi); self.assertTrue(0 < wi["chance"] < 1); self.assertEqual(wi["n_queries"] + wi["n_singletons"], len(self.w()))

    def test_the_planted_drift_is_found_for_that_feature_and_not_for_the_others(self):
        c = self.out["states"]["Q"]["write"]; self.assertIn(self.drift_feature, self.w()); i = self.w().index(self.drift_feature)
        self.assertLess(c["cos"][i], 0.9)
        others = np.delete(c["cos"], i); self.assertGreater(np.nanmedian(others), c["cos"][i])

    def test_the_fingerprint_of_a_feature_uses_the_decoder_of_that_feature_not_of_its_row(self):
        c = self.out["states"]["FP_rebatched"]["write"]; fp = self.out["states"]["FP"]["write"]
        self.assertTrue(np.isfinite(c["cos"]).all()); self.assertTrue(np.isfinite(fp["cos"]).all())
        # the closed form for the first tested feature, with its own decoder direction
        k = self.out["stage_a"]["write_rows"][0]; fid = int(self.feat[k])
        self.assertEqual(fid, self.out["stage_a"]["write_features"][0]); self.assertNotEqual(k, fid)

    def test_the_read_side_of_the_quantized_state_is_untouched_because_only_the_upper_map_changed(self):
        r = self.out["states"]["Q"]["read"]; ok = ~np.isnan(r["corr"])
        np.testing.assert_allclose(r["corr"][ok], 1.0, atol=1e-9)

    def test_a_silent_feature_has_similarity_zero_flagged_no_overlap_and_no_active_positions_while_the_others_are_unchanged(self):
        s = self.out["states"]["S"]; w = self.w(); i = w.index(self.silent_feature); c = s["write"]
        self.assertTrue(c["zero_q"][i]); self.assertEqual(c["cos"][i], 0.0); self.assertTrue(np.isnan(c["topk_overlap"][i])); self.assertEqual(int(c["n_active_q"][i]), 0); self.assertGreater(int(c["n_pos"][i]), 0)
        others = np.delete(c["cos"], i); np.testing.assert_allclose(others[~np.isnan(others)], 1.0, atol=1e-5)
        j = int(np.flatnonzero(self.feat.numpy() == self.silent_feature)[0]); self.assertEqual(float(s["read"]["jaccard"][j]), 0.0)          # active in FP, never active in S

    def test_halving_every_activation_halves_the_scale_and_the_fingerprint_but_keeps_correlation_jaccard_and_direction(self):
        z = self.out["states"]["Z"]; r = z["read"]; c = z["write"]
        ok = ~np.isnan(r["scale"]); np.testing.assert_allclose(r["scale"][ok], 0.5, atol=1e-6); np.testing.assert_allclose(r["corr"][~np.isnan(r["corr"])], 1.0, atol=1e-6)
        self.assertTrue(((r["jaccard"][~np.isnan(r["jaccard"])]) == 1.0).all())                              # the ablation removes the state's own, halved, contribution
        np.testing.assert_allclose(c["magnitude_ratio"], 0.5, atol=1e-3); np.testing.assert_allclose(c["cos"], 1.0, atol=1e-3)

    def test_the_magnitude_ratio_is_one_for_the_identity_and_zero_for_a_silent_feature(self):
        np.testing.assert_allclose(self.out["states"]["FP"]["write"]["magnitude_ratio"], 1.0, atol=1e-3)
        s = self.out["states"]["S"]["write"]; self.assertEqual(float(s["magnitude_ratio"][self.w().index(self.silent_feature)]), 0.0)

    def test_the_floor_state_is_presented_reversed_and_padded_and_the_others_are_not(self):
        calls = []; orig = r5.fingerprint_feature
        def spy(*a, **kw):
            calls.append((kw.get("reverse"), kw.get("extra_pad")))
            return orig(*a, **kw)
        r5.fingerprint_feature = spy
        try:
            r5.run_f5(["FP", "FP_rebatched"], self.setup, self.stack.lower, self.stack.upper_final, self.stack.logits_fn, self.encode, self.sae.W_dec, self.feat, self.batcher(self.ev), self.dev, self.batcher,
                      M=5, min_ctx=3, V=V, device="cpu", topk=3)
        finally:
            r5.fingerprint_feature = orig
        n = len(self.w()); self.assertEqual(calls[:n], [(False, 0)] * n); self.assertEqual(calls[n:], [(True, 3)] * n)

    def test_stage_a_always_runs_in_full_precision_whatever_state_the_model_was_left_in(self):
        self.setup("S")                                                                                      # the model is left in the state with a silent feature
        out = r5.run_f5(["FP"], self.setup, self.stack.lower, self.stack.upper_final, self.stack.logits_fn, self.encode, self.sae.W_dec, self.feat, self.batcher(self.ev), self.dev, self.batcher,
                        M=5, min_ctx=3, V=V, device="cpu", topk=3)
        self.assertEqual(out["stage_a"]["write_features"], self.w()); self.assertIn(self.silent_feature, out["stage_a"]["write_features"])

    def test_the_compact_statistics_have_one_row_per_tested_feature(self):
        c = self.out["states"]["Q"]["write"]; n = len(self.w())
        for key in ("cos", "topk_overlap", "magnitude_ratio", "n_pos", "n_active_q", "dnll", "zero_q", "zero_fp"):
            self.assertEqual(len(c[key]), n, key)
        self.assertEqual(tuple(c["S"].shape), (n, n))

    def test_every_requested_state_is_reported_in_order_and_the_full_precision_state_must_come_first(self):
        self.assertEqual(list(self.out["states"]), ["FP", "FP_rebatched", "Q", "Z", "S"])
        with self.assertRaises(ValueError):
            r5.run_f5(["Q", "FP"], self.setup, self.stack.lower, self.stack.upper_final, self.stack.logits_fn, self.encode, self.sae.W_dec, self.feat, self.batcher(self.ev), self.dev, self.batcher,
                      M=5, min_ctx=3, V=V, device="cpu", topk=3)

class ExcludedFeaturesShiftTheRows(unittest.TestCase):
    """A feature that never fires in the dev sentences is excluded, so the tested rows are a subset of the sample rows: everything per row (the groups, the decoders, the feature ids) must follow the row."""
    @classmethod
    def setUpClass(cls):
        cls.stack = ToyStack(linear=True); cls.sae = ToySAE(); cls.dev = sentences(80, seed=5); cls.ev = sentences(30, seed=6)
        cls.feat = torch.tensor([29, 3, 17, 8, 22, 1, 12, 25, 5, 30, 14, 9])
        cls.batcher = staticmethod(lambda sents, bs=6: p1.make_batches({0: sents}, BOS, "mask", 0, 256, bs))
        cls.dead = int(cls.feat[2])

        def encode(x):
            a = cls.sae.encode(x); a[:, cls.dead] = 0.0
            return a
        cls.encode = staticmethod(encode)
        cls.groups = np.array([0, 0, 1, 1, 0, 2, 2, 1, 0, 2, 1, 0])
        cls.out = r5.run_f5(["FP"], lambda label: None, cls.stack.lower, cls.stack.upper_final, cls.stack.logits_fn, encode, cls.sae.W_dec, cls.feat, cls.batcher(cls.ev), cls.dev, cls.batcher,
                            M=5, min_ctx=3, V=V, device="cpu", topk=3, groups=cls.groups)

    def test_the_dead_feature_is_excluded_and_the_tested_rows_are_a_proper_subset(self):
        a = self.out["stage_a"]; self.assertTrue(bool(a["excluded"][2])); self.assertNotIn(2, a["write_rows"]); self.assertLess(len(a["write_rows"]), len(self.feat))
        self.assertEqual(a["write_features"], [int(self.feat[k]) for k in a["write_rows"]])

    def test_the_within_group_chance_level_is_computed_from_the_groups_of_the_tested_rows(self):
        rows = self.out["stage_a"]["write_rows"]; g = self.groups[rows]
        sizes = np.array([int((g == g[i]).sum()) for i in range(len(g)) if int((g == g[i]).sum()) > 1], dtype=float)
        wi = self.out["states"]["FP"]["write"]["retrieval"]["within"]
        self.assertAlmostEqual(wi["chance"], float((1.0 / sizes).mean()), places=12); self.assertEqual(wi["n_singletons"], int(len(g) - len(sizes)))
        self.assertEqual(wi["n_queries"] + wi["n_singletons"], len(rows))


if __name__ == "__main__":
    unittest.main()
