import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

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

    def test_the_ablation_is_applied_in_the_work_dtype_and_not_rounded_to_the_dtype_of_the_cache(self):
        k = next(iter(self.plan["ctx"])); d = self.sae.W_dec[self.feat[k]]; sents = self.plan["ctx"][k]; M = len(sents)
        bf = {"resid": {s: r.to(torch.bfloat16) for s, r in self.cache["resid"].items()}, "act": self.cache["act"]}; seen = {}

        def spy(x):
            seen["x"] = x.clone(); return self.stack.upper_final(x.float())

        def realized_error(work_dtype):
            r5.fingerprint_feature(k, self.plan, bf, self.pos, d, spy, self.stack.logits_fn, V, "cpu", work_dtype=work_dtype); x = seen["x"]; err = 0.0
            for j, s in enumerate(sents):
                p = self.pos[(k, s)]; a = self.cache["act"][(k, s)]
                err = max(err, float(((x[M + j, p + 1].double() - x[j, p + 1].double()) + a[:, None].double() * d[None, :].double()).abs().max()))
            return err, x.dtype
        e_native, dt_native = realized_error(None); e_wide, dt_wide = realized_error(torch.float32)
        self.assertEqual(dt_native, torch.bfloat16); self.assertGreater(e_native, 1e-3)                      # the edit lost to bf16 rounding: this is what the work dtype avoids
        self.assertEqual(dt_wide, torch.float32); self.assertLess(e_wide, 1e-5)

    def test_a_sham_in_the_work_dtype_is_still_exactly_zero(self):
        k = next(iter(self.plan["ctx"])); out = self.fp(k, sham=True, work_dtype=torch.float64)
        self.assertEqual(float(out["D"].abs().max()), 0.0); self.assertEqual(out["dnll"], 0.0)

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
        sel = r5.select_contexts(ma, 5, 3); cls.sel = sel; testable = [int(cls.feat[k]) for k in range(len(cls.feat)) if not bool(sel["excluded"][k])]
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
        cls.groups = (cls.feat % 3).numpy(); cls.calls = []
        cls.out = r5.run_f5(["FP", "FP_rebatched", "Q", "Z", "S"], setup, cls.stack.lower, cls.stack.upper_final, cls.stack.logits_fn, encode, cls.sae.W_dec, cls.feat, cls.batcher(cls.ev), cls.dev, cls.batcher,
                            M=5, min_ctx=3, V=V, device="cpu", topk=3, groups=cls.groups, on_state=lambda label, res: cls.calls.append((label, res)))

    def w(self):
        return self.out["stage_a"]["write_features"]

    def test_stage_a_keeps_the_context_sentences_of_every_feature_as_indices_into_the_candidate_list(self):
        c = self.out["stage_a"]["contexts"]; self.assertEqual(c.shape, (len(self.feat), 5))
        np.testing.assert_array_equal(c, self.sel["contexts"].numpy()); self.assertTrue(((c >= -1) & (c < len(self.dev))).all())

    def test_the_state_callback_receives_every_state_once_in_order_with_its_own_result(self):
        self.assertEqual([c[0] for c in self.calls], ["FP", "FP_rebatched", "Q", "Z", "S"])
        for label, res in self.calls:
            self.assertIs(res, self.out["states"][label])

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
        calls, sham = [], []; orig = r5.fingerprint_feature
        def spy(*a, **kw):
            (sham if kw.get("sham") else calls).append((kw.get("reverse"), kw.get("extra_pad")))
            return orig(*a, **kw)
        r5.fingerprint_feature = spy
        try:
            r5.run_f5(["FP", "FP_rebatched"], self.setup, self.stack.lower, self.stack.upper_final, self.stack.logits_fn, self.encode, self.sae.W_dec, self.feat, self.batcher(self.ev), self.dev, self.batcher,
                      M=5, min_ctx=3, V=V, device="cpu", topk=3)
        finally:
            r5.fingerprint_feature = orig
        n = len(self.w()); self.assertEqual(calls[:n], [(False, 0)] * n); self.assertEqual(calls[n:], [(True, 3)] * n)
        self.assertEqual(len(sham), 2)                                                                        # the sham gate (two features, in full precision) is not a state measurement

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

class FakeFinal:
    """The attributes of a HookedTransformer that the float32 logits use: ln_final.w, unembed.W_U and b_U, cfg.eps and cfg.output_logits_soft_cap."""
    def __init__(self, d=8, V=20, cap=30.0, dtype=torch.float32, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.ln_final = SimpleNamespace(w=(torch.randn(d, generator=g) * 0.3 + 1.0).to(dtype))
        self.unembed = SimpleNamespace(W_U=(torch.randn(d, V, generator=g) * 3).to(dtype), b_U=(torch.randn(V, generator=g) * 0.7).to(dtype))
        self.cfg = SimpleNamespace(eps=1e-6, output_logits_soft_cap=cap, n_layers=26)


def reference_logits(m, h):
    h = h.double(); n = h / torch.sqrt((h ** 2).mean(-1, keepdim=True) + m.cfg.eps) * m.ln_final.w.double()
    z = n @ m.unembed.W_U.double() + m.unembed.b_U.double(); cap = m.cfg.output_logits_soft_cap
    return cap * torch.tanh(z / cap) if cap else z


class FinalLogitsTests(unittest.TestCase):
    def test_float32_logits_equal_the_float64_closed_form_of_final_norm_unembedding_and_soft_cap(self):
        m = FakeFinal(); h = torch.randn(11, 8) * 40
        np.testing.assert_allclose(r5.make_logits_fn(m)(h).double().numpy(), reference_logits(m, h).numpy(), atol=2e-5)

    def test_the_norm_epsilon_matters_for_a_hidden_state_of_tiny_scale_and_is_the_models(self):
        m = FakeFinal(); h = torch.randn(7, 8) * 1e-3
        np.testing.assert_allclose(r5.make_logits_fn(m)(h).double().numpy(), reference_logits(m, h).numpy(), atol=2e-5)
        m.cfg.eps = 1e-2; np.testing.assert_allclose(r5.make_logits_fn(m)(h).double().numpy(), reference_logits(m, h).numpy(), atol=2e-5)

    def test_the_vocabulary_is_processed_in_slices_with_the_same_result(self):
        m = FakeFinal(V=20); h = torch.randn(5, 8); want = reference_logits(m, h).numpy()
        for chunk in (1, 3, 7, 19, 20, 64):
            np.testing.assert_allclose(r5.make_logits_fn(m, vocab_chunk=chunk)(h).double().numpy(), want, atol=2e-5, err_msg=f"chunk {chunk}")
        self.assertEqual(tuple(r5.make_logits_fn(m, vocab_chunk=7)(h).shape), (5, 20))

    def test_the_output_is_float32_even_for_bfloat16_parameters_and_input_and_uses_their_values_exactly(self):
        m = FakeFinal(dtype=torch.bfloat16); h = (torch.randn(5, 8) * 40).to(torch.bfloat16); z = r5.make_logits_fn(m)(h)
        self.assertEqual(z.dtype, torch.float32)
        np.testing.assert_allclose(z.double().numpy(), reference_logits(m, h).numpy(), atol=5e-5)

    def test_the_soft_cap_bounds_the_logits_and_is_absent_when_the_cap_is_none_or_zero(self):
        h = torch.randn(40, 8)
        big = FakeFinal(cap=30.0); big.unembed.W_U = big.unembed.W_U * 6                                           # the norm removes the input scale: the unembedding makes the logits large
        self.assertLess(float(r5.make_logits_fn(big)(h).abs().max()), 30.0 + 1e-4)
        for cap in (None, 0.0):
            m = FakeFinal(cap=cap); m.unembed.W_U = m.unembed.W_U * 6; z = r5.make_logits_fn(m)(h)
            self.assertGreater(float(z.abs().max()), 30.0); np.testing.assert_allclose(z.double().numpy(), reference_logits(m, h).numpy(), rtol=1e-5, atol=1e-3)

    def test_the_parameters_are_read_when_the_function_is_called_not_when_it_is_made(self):
        m = FakeFinal(); h = torch.randn(4, 8); fn = r5.make_logits_fn(m); before = fn(h).clone()
        m.unembed.W_U = m.unembed.W_U * 0.5                                         # a quantizer or a state restore replaces the parameter data
        np.testing.assert_allclose(fn(h).double().numpy(), reference_logits(m, h).numpy(), atol=2e-5); self.assertFalse(torch.allclose(fn(h), before))

    def test_the_input_is_not_modified_and_the_row_order_is_kept(self):
        m = FakeFinal(); h = torch.randn(6, 8); h0 = h.clone(); fn = r5.make_logits_fn(m); z = fn(h)
        self.assertTrue(torch.equal(h, h0)); self.assertTrue(torch.allclose(fn(h[[3, 1]]), z[[3, 1]], atol=1e-6))


class UpperFinalSpy(unittest.TestCase):
    def test_the_upper_half_starts_after_the_hooked_layer_and_stops_before_the_final_norm(self):
        calls = []

        class Model:
            cfg = SimpleNamespace(n_layers=26)

            def __call__(self, x, **kw):
                calls.append(kw); return x + 1
        out = r5.make_upper_final(Model(), 12)(torch.zeros(2, 3, 4))
        self.assertEqual(calls, [{"start_at_layer": 13, "stop_at_layer": 26}]); self.assertTrue(torch.equal(out, torch.ones(2, 3, 4)))


class RealRunHelpers(unittest.TestCase):
    SAMPLE = {"languages": ["aaa", "bbb", "ccc"], "features": {"30": {"language": "ccc"}, "4": {"language": "aaa"}, "17": {"language": "bbb"}, "9": {"language": "aaa"}, "250": {"language": "ccc"},
                                                                    "11": {"language": "bbb"}, "3": {"language": "aaa"}, "88": {"language": "ccc"}, "61": {"language": "bbb"}, "7": {"language": "aaa"}}}

    def test_the_sample_arrays_are_the_sorted_feature_ids_with_the_language_index_of_each(self):
        ids, groups = r5.sample_arrays(self.SAMPLE)
        self.assertEqual(ids.tolist(), [3, 4, 7, 9, 11, 17, 30, 61, 88, 250]); self.assertEqual(groups.tolist(), [0, 0, 0, 0, 1, 1, 2, 1, 2, 2])

    def test_a_feature_limit_takes_evenly_spaced_features_including_the_first_and_the_last(self):
        ids, groups = r5.sample_arrays(self.SAMPLE, max_features=4); full, fg = r5.sample_arrays(self.SAMPLE)
        self.assertEqual(len(ids), 4); self.assertEqual(int(ids[0]), 3); self.assertEqual(int(ids[-1]), 250); self.assertEqual(ids.tolist(), sorted(ids.tolist()))
        for i, g in zip(ids.tolist(), groups.tolist()):
            self.assertEqual(g, int(fg[full.tolist().index(i)]))
        for n in (None, 10, 99):
            self.assertEqual(len(r5.sample_arrays(self.SAMPLE, max_features=n)[0]), 10)

    SAMPLE2 = {"languages": ["aa", "bb"], "features": {
        **{str(i): {"language": "aa", "roles": ["k64"]} for i in range(100, 110)}, **{str(i): {"language": "aa", "roles": ["lost@INT4"]} for i in range(200, 206)},
        **{str(i): {"language": "aa", "roles": ["k64", "retained@INT4"]} for i in range(300, 304)}, **{str(i): {"language": "bb", "roles": ["k64"]} for i in range(400, 403)},
        **{str(i): {"language": "bb", "roles": ["lost@INT4", "retained@INT5"]} for i in range(500, 512)}}}

    @staticmethod
    def cells(sample):
        out = {}
        for k, f in sample["features"].items():
            for r in f["roles"]:
                out.setdefault((f["language"], r), []).append(int(k))
        return out

    def test_a_per_cell_cap_keeps_at_most_that_many_features_per_language_and_role_by_a_hash_order(self):
        import hashlib
        ids, groups = r5.sample_arrays(self.SAMPLE2, per_cell=3, seed=7); kept = set(ids.tolist()); key = lambda i: hashlib.sha256(f"7:{i}".encode()).hexdigest()
        want = set()
        for cell, members in self.cells(self.SAMPLE2).items():
            want |= set(sorted(members, key=key)[:3])
        self.assertEqual(kept, want); self.assertEqual(ids.tolist(), sorted(ids.tolist()))

    def test_cells_smaller_than_the_cap_are_kept_whole_and_the_languages_follow_the_features(self):
        ids, groups = r5.sample_arrays(self.SAMPLE2, per_cell=50)
        self.assertEqual(len(ids), len(self.SAMPLE2["features"])); langs = {int(i): g for i, g in zip(ids.tolist(), groups.tolist())}
        self.assertEqual(langs[100], 0); self.assertEqual(langs[400], 1); self.assertEqual(langs[505], 1)

    def test_the_subset_is_deterministic_and_changes_with_the_seed(self):
        a = r5.sample_arrays(self.SAMPLE2, per_cell=2, seed=0)[0].tolist(); b = r5.sample_arrays(self.SAMPLE2, per_cell=2, seed=0)[0].tolist(); c = r5.sample_arrays(self.SAMPLE2, per_cell=2, seed=1)[0].tolist()
        self.assertEqual(a, b); self.assertNotEqual(a, c)

    def test_a_cap_and_a_feature_limit_combine_cap_first(self):
        capped = r5.sample_arrays(self.SAMPLE2, per_cell=2)[0]; both = r5.sample_arrays(self.SAMPLE2, per_cell=2, max_features=4)[0]
        self.assertEqual(len(both), 4); self.assertTrue(set(both.tolist()) <= set(capped.tolist())); self.assertEqual(int(both[0]), int(capped[0])); self.assertEqual(int(both[-1]), int(capped[-1]))

    def test_every_role_that_has_members_stays_represented_after_the_cap(self):
        ids, groups = r5.sample_arrays(self.SAMPLE2, per_cell=1); kept = set(ids.tolist())
        for cell, members in self.cells(self.SAMPLE2).items():
            self.assertTrue(kept & set(members), cell)

    def test_an_unknown_language_in_the_sample_is_an_error(self):
        bad = {"languages": ["aaa"], "features": {"1": {"language": "zzz"}}}
        with self.assertRaises(ValueError):
            r5.sample_arrays(bad)

    def test_the_frozen_evaluation_ids_are_read_from_the_freeze_and_checked_against_the_recomputation(self):
        import phase1_func_tasks as ft
        good = {"ids": {"n_total": 1012, "n_pilot": 150, "n_eval": 300, "seed": 0, "eval_ids": ft.lm_ids(1012, 150, 300, 0)["eval"]}}
        self.assertEqual(r5.frozen_eval_ids(good), good["ids"]["eval_ids"])
        bad = {"ids": dict(good["ids"], eval_ids=good["ids"]["eval_ids"][:-1] + [5000])}
        with self.assertRaises(ValueError):
            r5.frozen_eval_ids(bad)

    def test_the_logit_agreement_gate_passes_for_noise_and_fails_for_another_function(self):
        z = torch.randn(200, 50) * 5; noisy = z + torch.randn_like(z) * 0.05
        g = r5.logit_agreement(z, noisy, min_top1=0.97, max_abs=1.0); self.assertTrue(g["ok"]); self.assertGreater(g["top1"], 0.97)
        other = r5.logit_agreement(z, torch.randn(200, 50) * 5, min_top1=0.97, max_abs=1.0); self.assertFalse(other["ok"])
        self.assertFalse(r5.logit_agreement(z, z + 3.0 * torch.sign(torch.randn_like(z)), min_top1=0.0, max_abs=1.0)["ok"])
        flat = torch.randn(200, 50) * 0.1                                              # nearly flat logits: a small absolute difference still changes the top-1 token
        self.assertFalse(r5.logit_agreement(flat, flat + torch.randn_like(flat) * 0.5, min_top1=0.97, max_abs=10.0)["ok"])

    def test_the_sham_gate_requires_exactly_zero_everywhere(self):
        self.assertTrue(r5.sham_is_exact({"D": torch.zeros(5), "dnll": 0.0})); self.assertFalse(r5.sham_is_exact({"D": torch.tensor([0, 0, 1e-12, 0, 0.0]), "dnll": 0.0}))
        self.assertFalse(r5.sham_is_exact({"D": torch.zeros(5), "dnll": 1e-15}))

class GateAndCallbacks(unittest.TestCase):
    """The sham gate of run_f5 (a sham ablation of the first tested features must give exactly zero in full precision), the stage-A callback and the work dtype."""
    @classmethod
    def setUpClass(cls):
        cls.stack = ToyStack(linear=True); cls.sae = ToySAE(); cls.dev = sentences(80, seed=5); cls.ev = sentences(30, seed=6)
        cls.feat = torch.tensor([29, 3, 17, 8, 22, 1, 12, 25, 5, 30, 14, 9]); cls.batcher = staticmethod(lambda sents, bs=6: p1.make_batches({0: sents}, BOS, "mask", 0, 256, bs))

    def run_(self, logits_fn=None, upper_final=None, **kw):
        return r5.run_f5(["FP"], lambda label: None, self.stack.lower, upper_final or self.stack.upper_final, logits_fn or self.stack.logits_fn, self.sae.encode, self.sae.W_dec, self.feat,
                         self.batcher(self.ev), self.dev, self.batcher, M=5, min_ctx=3, V=V, device="cpu", topk=3, **kw)

    def test_the_sham_gate_runs_on_the_first_tested_features_and_is_recorded(self):
        out = self.run_(); g = out["stage_a"]["sham_gate"]; w = out["stage_a"]["write_features"]
        self.assertEqual([x["feature"] for x in g], w[:2]); self.assertTrue(all(x["exact"] for x in g))
        self.assertEqual(len(self.run_(sham_gate=3)["stage_a"]["sham_gate"]), 3); self.assertEqual(self.run_(sham_gate=0)["stage_a"]["sham_gate"], [])

    def test_the_gate_stops_the_run_when_baseline_and_ablated_rows_do_not_agree_exactly(self):
        calls = {"n": 0}

        def noisy(h):
            calls["n"] += 1; return self.stack.logits_fn(h) + 1e-4 * calls["n"] * torch.linspace(-1, 1, V)                # a different vocabulary-dependent offset at every call: baseline and ablated logits cannot agree
        with self.assertRaises(r5.GateError):
            self.run_(logits_fn=noisy)
        calls["n"] = 0; self.run_(logits_fn=noisy, sham_gate=0)                                                       # with the gate off the same stack runs

    def test_the_gate_stops_the_run_when_only_the_second_gate_feature_is_inexact(self):
        calls = {"n": 0}

        def noisy_once(h):
            calls["n"] += 1                                                                                             # calls 1 and 2: baseline and ablated rows of the first gate feature; 3 and 4: of the second
            return self.stack.logits_fn(h) + (1e-4 * torch.linspace(-1, 1, V) if calls["n"] == 4 else 0.0)
        with self.assertRaises(r5.GateError) as cm:
            self.run_(logits_fn=noisy_once)
        self.assertIn(f"features [{self.run_(sham_gate=0)['stage_a']['write_features'][1]}]:", str(cm.exception))             # only the inexact one is named

    def test_the_stage_a_callback_is_called_once_before_any_state(self):
        events = []
        r5.run_f5(["FP", "FP_rebatched"], lambda label: None, self.stack.lower, self.stack.upper_final, self.stack.logits_fn, self.sae.encode, self.sae.W_dec, self.feat, self.batcher(self.ev), self.dev,
                  self.batcher, M=5, min_ctx=3, V=V, device="cpu", topk=3, on_stage_a=lambda sa: events.append(("A", sa)), on_state=lambda label, res: events.append((label, res)))
        self.assertEqual([e[0] for e in events], ["A", "FP", "FP_rebatched"]); self.assertIn("write_features", events[0][1]); self.assertIn("sham_gate", events[0][1])

    def test_the_work_dtype_reaches_the_upper_half(self):
        seen = []

        def spy(x):
            seen.append(x.dtype); return self.stack.upper_final(x)
        self.run_(upper_final=spy, work_dtype=torch.float64); self.assertTrue(seen); self.assertTrue(all(d == torch.float64 for d in seen))
        seen.clear(); self.run_(upper_final=spy); self.assertTrue(all(d == torch.float32 for d in seen))

class StateLabels(unittest.TestCase):
    def test_labels_are_normalised_and_the_quantizer_bits_are_extracted(self):
        self.assertEqual([r5.state_label(x) for x in ("FP", "FP_rebatched", "8", "INT6", "16")], ["FP", "FP_rebatched", "INT8", "INT6", "INT16"])
        self.assertEqual([r5.state_bits(x) for x in ("FP", "FP_rebatched", "INT8", "INT4")], [None, None, 8, 4])

    def test_nonsense_labels_are_rejected(self):
        for bad in ("fp", "INT", "INT0", "INT-4", "FP8", "", "4.5", "int6", "INT06", "INT1", "1", "0"):
            with self.assertRaises(ValueError, msg=bad):
                r5.state_label(bad)

class DefaultStates(unittest.TestCase):
    def test_the_default_states_include_the_weight_rounding_floor_int16_and_the_batching_floor(self):
        self.assertEqual([r5.state_label(x) for x in r5.DEFAULT_STATES], ["FP", "FP_rebatched", "INT16", "INT8", "INT6", "INT5", "INT4"])


if __name__ == "__main__":
    unittest.main()
