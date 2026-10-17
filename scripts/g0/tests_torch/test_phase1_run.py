"""S4 tests for scripts/g0/phase1_run.py orchestration with a FAKE causal model, SAE and quantizer (CPU): per-language results must equal a brute-force
computation done sentence by sentence without padding, whatever the batching; position 0 is excluded under `mask`; INT0 equals FP; restore leaves the model untouched."""
import sys, unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import phase1_lib as p1   # noqa: E402
import phase1_run as run   # noqa: E402

V, D, F, BOS, PAD = 40, 8, 24, 2, 0


class Fake:
    """Causal toy 'transformer': resid_t = cumsum(E[tokens])_t @ W, logits = resid @ U; the SAE is a JumpReLU-like encoder; quantization perturbs W."""
    def __init__(self, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.E = torch.randn(V, D, generator=g)
        self.W0 = torch.randn(D, D, generator=g) * 0.4; self.W = self.W0.clone(); self.U = torch.randn(D, V, generator=g)
        self.We = torch.randn(D, F, generator=g); self.be = torch.randn(F, generator=g) * 0.5; self.thr = torch.rand(F, generator=g) * 0.3
        self.sink = torch.randn(D, generator=g) * 25.0                                     # the BOS position alone carries a huge residual, like the real attention-sink token
    def forward(self, tokens):
        resid = torch.cumsum(self.E[tokens], dim=1) @ self.W + (tokens == BOS)[..., None] * self.sink
        return resid, resid @ self.U
    def encode(self, acts):
        pre = acts @ self.We + self.be
        return (pre > self.thr) * torch.relu(pre)
    def quantize(self, bits):
        step = 2.0 ** -(bits - 2); self.W = torch.round(self.W / step) * step * (1 - 2.0 ** -bits)       # acts on the CURRENT weights: compounds unless restored
    def restore(self):
        self.W = self.W0.clone()


def corpus(seed=1, lens=((5, 9, 4, 12, 7, 3), (6, 6, 10, 2), (8, 5, 11))):
    g = torch.Generator().manual_seed(seed)
    return {l: [torch.randint(3, V, (n,), generator=g).tolist() for n in ls] for l, ls in enumerate(lens)}


def call(fake, data, regime="mask", bits=(8, 4), **kw):
    return run.run_survival(fake.forward, fake.encode, fake.quantize, fake.restore, data, regime, BOS, PAD, F, list(bits), **kw)


def brute(fake, data, regime, bits=None):
    """Per language, sentence by sentence, no padding: (x, y, nll_sum, nll_n)."""
    out = {}
    for l, sents in data.items():
        xs, ys, nll, n = [], [], 0.0, 0
        for s in sents:
            toks = ([BOS] if regime == "mask" else []) + s; off = 1 if regime == "mask" else 0
            fake.restore(); r, lg = fake.forward(torch.tensor([toks])); x = fake.encode(r[0])[off:]
            if bits is not None: fake.quantize(bits)
            r2, lg2 = fake.forward(torch.tensor([toks])); y = fake.encode(r2[0])[off:]
            xs.append(x); ys.append(y)
            lp = torch.log_softmax(lg2[0].float(), -1)
            for t in range(max(off, 1), len(toks)):
                nll -= float(lp[t - 1, toks[t]]); n += 1
        fake.restore(); out[l] = (torch.cat(xs).double(), torch.cat(ys).double(), nll, n)
    return out


class Survival(unittest.TestCase):
    def test_per_language_correlation_equals_brute_force_for_a_quantized_rung(self):
        fake, data = Fake(), corpus()
        res = call(fake, data, "mask", bits=(8,), batch_size=4)
        ref = brute(Fake(), data, "mask", bits=8)
        for l in data:
            x, y, _, _ = ref[l]; act = (x > 0).any(0)
            expect = torch.tensor([np.corrcoef(x[:, j].numpy(), y[:, j].numpy())[0, 1] for j in range(F) if act[j] and x[:, j].std() > 0 and y[:, j].std() > 0])
            got = res[8]["corr"][l][act & (x.std(0) > 0) & (y.std(0) > 0)].double()
            self.assertTrue(torch.allclose(got, expect, atol=1e-6)); self.assertEqual(res[8]["n"][l], len(x))

    def test_results_do_not_depend_on_how_sentences_are_batched_or_padded(self):
        fake, data = Fake(), corpus()
        a = call(fake, data, "mask", batch_size=1); b = call(fake, data, "mask", batch_size=64); c = call(fake, data, "mask", batch_size=64, max_tokens=24)
        for key in (4, 8, "FP"):
            for other in (b, c):
                self.assertTrue(torch.allclose(a[key]["corr"], other[key]["corr"], atol=1e-6)); self.assertEqual(a[key]["n"], other[key]["n"])
                self.assertTrue(np.allclose(a[key]["loss_sum"], other[key]["loss_sum"], rtol=1e-5))

    def test_mask_regime_excludes_the_bos_position_so_its_huge_activation_never_enters_a_statistic(self):
        fake, data = Fake(), corpus()
        res = call(fake, data, "mask", bits=(8,)); content = {l: sum(len(s) for s in ss) for l, ss in data.items()}
        self.assertEqual(res["FP"]["n"], [content[l] for l in sorted(data)])         # content tokens only, never one extra BOS row per sentence
        ref = brute(Fake(), data, "mask")                                               # content positions only, sentence by sentence
        bos_vec = fake.encode(fake.forward(torch.tensor([[BOS]]))[0][0])
        for l in data:
            self.assertTrue(torch.allclose(res["FP"]["max_activation"][l].double(), ref[l][0].max(0).values, atol=1e-4))    # maxima are content-only ...
            self.assertTrue(bool((bos_vec > ref[l][0].max(0).values).any()))                                                 # ... and counting BOS would have changed them

    def test_none_regime_counts_every_content_token_and_has_no_bos(self):
        fake, data = Fake(), corpus()
        res = call(fake, data, "none", bits=(8,)); ref = brute(Fake(), data, "none", bits=8)
        for l in data:
            self.assertEqual(res["FP"]["n"][l], len(ref[l][0]))

    def test_int0_restore_only_control_equals_fp_exactly(self):
        fake, data = Fake(), corpus(); fake.quantize(3)                                   # start from a perturbed model: restore must undo it
        res = call(fake, data, "mask", bits=(0, 6))
        act = res["FP"]["firing_rate"] > 0
        self.assertTrue(act.sum() > 20)                                                  # the check below must not be vacuous
        self.assertTrue(bool((res[0]["corr"][act] >= 1 - 1e-6).all()))                    # float32 results: identical features give r = 1
        self.assertTrue(bool((res[6]["corr"][act].mean() < 1.0)))                         # a real rung does change the features

    def test_a_rung_does_not_depend_on_which_rungs_ran_before_it(self):
        data = corpus(); alone = call(Fake(), data, "mask", bits=(4,)); after = call(Fake(), data, "mask", bits=(8, 6, 4))
        self.assertTrue(torch.allclose(alone[4]["corr"], after[4]["corr"], atol=1e-6)); self.assertTrue(np.allclose(alone[4]["loss_sum"], after[4]["loss_sum"], rtol=1e-6))

    def test_firing_rate_mean_and_max_of_every_rung_are_those_of_the_full_precision_features(self):
        res = call(Fake(), corpus(), "mask", bits=(8, 4))                                  # their definition: activity statistics come from model A (the full-precision side)
        for key in (8, 4):
            for stat in ("firing_rate", "mean_activation", "max_activation"):
                self.assertTrue(torch.equal(res[key][stat], res["FP"][stat])); self.assertTrue(torch.equal(res[key]["pooled"][stat], res["FP"]["pooled"][stat]))

    def test_the_model_is_restored_after_every_rung(self):
        fake, data = Fake(), corpus(); call(fake, data, "mask", bits=(8, 4))
        self.assertTrue(torch.equal(fake.W, fake.W0))

    def test_per_language_loss_equals_brute_force(self):
        fake, data = Fake(), corpus(); res = call(fake, data, "mask", bits=(8,), batch_size=3); ref = brute(Fake(), data, "mask", bits=8)
        for l in data:
            self.assertAlmostEqual(res[8]["loss_sum"][l] / res[8]["loss_n"][l], ref[l][2] / ref[l][3], places=4)
            self.assertEqual(res[8]["loss_n"][l], ref[l][3])


class Pooled(unittest.TestCase):
    def test_pooled_result_equals_brute_force_over_all_languages(self):
        fake, data = Fake(), corpus(); res = call(fake, data, "mask", bits=(8,), batch_size=4); ref = brute(Fake(), data, "mask", bits=8)
        x = torch.cat([ref[l][0] for l in data]); y = torch.cat([ref[l][1] for l in data]); P = res[8]["pooled"]
        self.assertEqual(P["n"], len(x)); ok = (x.std(0) > 0) & (y.std(0) > 0)
        expect = torch.tensor([np.corrcoef(x[:, j].numpy(), y[:, j].numpy())[0, 1] for j in range(F) if ok[j]])
        self.assertTrue(torch.allclose(P["corr"][ok].double(), expect, atol=1e-6))
        self.assertTrue(torch.allclose(res["FP"]["pooled"]["firing_rate"].double(), (x > 0).double().mean(0), atol=1e-6))


class SurvivalSummary(unittest.TestCase):
    def test_follows_their_definitions_including_the_strict_boundaries(self):
        corr = torch.tensor([1.0, 0.95, 0.9, 0.7, 0.5, 0.49, 0.2, 0.0, 0.99], dtype=torch.float64)
        rate = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.001, 0.0], dtype=torch.float64)   # firing rate exactly at the threshold is NOT active
        r = run.survival_summary(corr, rate, 0.001)
        self.assertEqual(r["n_active_features"], 7)
        self.assertAlmostEqual(r["survived_>0.9_pct"], 2 / 7 * 100)       # r > 0.9 strictly: 0.9 itself is "degraded"
        self.assertAlmostEqual(r["degraded_0.5_0.9_pct"], 2 / 7 * 100)    # 0.5 < r <= 0.9
        self.assertAlmostEqual(r["damaged_<0.5_pct"], 2 / 7 * 100)        # r < 0.5 strictly: r == 0.5 is in none of the three bins (their definition)
        self.assertAlmostEqual(r["mean_corr"], float(corr[:7].mean())); self.assertAlmostEqual(r["median_corr"], 0.7)

    def test_median_is_torchs_lower_middle_value_as_in_their_code(self):
        r = run.survival_summary(torch.tensor([0.2, 0.4, 0.8, 0.95], dtype=torch.float64), torch.full((4,), 0.5, dtype=torch.float64), 0.001)
        self.assertAlmostEqual(r["median_corr"], 0.4)

    def test_no_active_feature_raises_like_theirs(self):
        with self.assertRaises(RuntimeError): run.survival_summary(torch.ones(3), torch.zeros(3), 0.001)


class Rows(unittest.TestCase):
    HEADER = open(Path(__file__).resolve().parents[3] / "reports/g0/baseline_smoke_vendored/phase3_summary_final.csv").readline().strip().split(",")

    def test_condition_rows_use_their_summary_columns_and_perplexity_from_token_weighted_loss(self):
        fake, data = Fake(), corpus(); res = call(fake, data, "mask", bits=(8, 0))
        rows = run.condition_rows(res, [8, 0], F, layer=12, threshold=0.001)
        self.assertEqual(list(rows[0].keys())[:len(self.HEADER)], self.HEADER)
        self.assertEqual([r["bits"] for r in rows], [32, 8, 0]); self.assertEqual(rows[0]["condition"], "FP16 baseline"); self.assertEqual(rows[1]["condition"], "RTN INT8")
        fp = np.exp(sum(res["FP"]["loss_sum"]) / sum(res["FP"]["loss_n"])); q = np.exp(sum(res[8]["loss_sum"]) / sum(res[8]["loss_n"]))
        self.assertAlmostEqual(rows[0]["perplexity"], fp); self.assertAlmostEqual(rows[1]["perplexity"], q); self.assertAlmostEqual(rows[1]["ppl_delta_pct"], (q / fp - 1) * 100)
        self.assertEqual(rows[0]["n_tokens"], sum(res["FP"]["n"])); self.assertEqual(rows[0]["n_total_features"], F)
        self.assertAlmostEqual(rows[2]["survived_>0.9_pct"], 100.0); self.assertAlmostEqual(rows[2]["ppl_delta_pct"], 0.0, places=6)    # INT0 control

    def test_language_rows_give_one_row_per_language_and_rung(self):
        fake, data = Fake(), corpus(); res = call(fake, data, "mask", bits=(8, 6))
        rows = run.language_rows(res, [8, 6], ["aaa", "bbb", "ccc"], threshold=0.001)
        self.assertEqual(len(rows), 3 * 2); self.assertEqual({r["language"] for r in rows}, {"aaa", "bbb", "ccc"})
        fp = res["FP"]; pooled_active = fp["pooled"]["firing_rate"] > 0.001
        self.assertTrue(any(bool(((fp["firing_rate"][li] > 0.001) != pooled_active).any()) for li in range(3)))      # premise: per-language active sets differ from the pooled one
        for r in rows:
            li = ["aaa", "bbb", "ccc"].index(r["language"]); ref = run.survival_summary(res[r["bits"]]["corr"][li].double(), fp["firing_rate"][li].double(), 0.001)
            self.assertEqual(r["n_active_features"], ref["n_active_features"]); self.assertEqual(int(fp["firing_rate"][li].gt(0.001).sum()), r["n_active_features"])
            self.assertAlmostEqual(r["survived_>0.9_pct"], ref["survived_>0.9_pct"]); self.assertAlmostEqual(r["median_corr"], ref["median_corr"]); self.assertEqual(r["n_tokens"], fp["n"][li])


class Tagging(unittest.TestCase):
    def test_sentence_statistics_equal_brute_force_and_languages_have_offsets(self):
        fake, data = Fake(), corpus()
        tag, offsets, lang_of = run.run_tagging(fake.forward, fake.encode, data, "mask", BOS, PAD, F, batch_size=4)
        self.assertEqual(offsets, [0, 6, 10]); self.assertEqual(lang_of.tolist(), [0] * 6 + [1] * 4 + [2] * 3)
        for l, sents in data.items():
            for i, s in enumerate(sents):
                r, _ = fake.forward(torch.tensor([[BOS] + s])); x = fake.encode(r[0])[1:]
                g = offsets[l] + i
                self.assertTrue(torch.equal(tag.counts[g].long(), (x > 0).sum(0))); self.assertEqual(int(tag.ntok[g]), len(s))
                self.assertTrue(torch.allclose(tag.sums[g], x.sum(0), atol=1e-4))


class ChunkedNll(unittest.TestCase):
    def test_equals_a_direct_computation(self):
        g = torch.Generator().manual_seed(0); logits = torch.randn(3, 6, V, generator=g); toks = torch.randint(0, V, (3, 6), generator=g)
        valid = torch.tensor([[1, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0]], dtype=torch.bool)
        s, n = p1.chunked_nll(logits, toks, valid, chunk=2)
        lp = torch.log_softmax(logits[:, :-1].float(), -1).gather(-1, toks[:, 1:, None])[..., 0]; m = valid[:, 1:]
        self.assertTrue(torch.allclose(s, -(lp * m).sum(1).double(), atol=1e-5)); self.assertEqual(n.tolist(), m.sum(1).tolist())


class TheirLossNumerics(unittest.TestCase):
    """Their perplexity comes from TransformerLens' lm_cross_entropy_loss, which takes log_softmax of the bf16 logits WITHOUT a float32 cast. The anchor run showed the
    float32 version is 0.72% lower in perplexity (411.09 vs 414.07 at 20k tokens), so the Phase 1 runner reproduces their numerics when asked (cast_float=False)."""
    def data(self):
        g = torch.Generator().manual_seed(1); logits = (torch.randn(4, 9, 4000, generator=g) * 6).bfloat16(); toks = torch.randint(0, 4000, (4, 9), generator=g)
        return logits, toks, torch.ones(4, 9, dtype=torch.bool)

    def test_native_dtype_path_equals_log_softmax_on_the_bf16_logits_and_differs_from_the_float32_path(self):
        logits, toks, valid = self.data()
        s_native, n = p1.chunked_nll(logits, toks, valid, chunk=2, cast_float=False); s_f32, _ = p1.chunked_nll(logits, toks, valid, chunk=2)
        lp = torch.log_softmax(logits, dim=-1)[:, :-1].gather(-1, toks[:, 1:, None])[..., 0].double()
        self.assertTrue(torch.equal(s_native, -lp.sum(1))); self.assertEqual(n.tolist(), [8] * 4)
        self.assertFalse(torch.equal(s_native, s_f32)); self.assertEqual(s_native.dtype, torch.float64)           # premise: the two numerics really differ; sums are float64

    def test_run_survival_uses_the_injected_loss_function(self):
        calls = []
        def nll(logits, tokens, valid):
            calls.append(1); return torch.full((logits.shape[0],), 2.0, dtype=torch.float64), valid[:, 1:].sum(1)
        res = call(Fake(), corpus(), "mask", bits=(8,), nll_fn=nll)
        self.assertTrue(len(calls) > 0); self.assertEqual(res["FP"]["loss_sum"], [12.0, 8.0, 6.0]); self.assertEqual(res[8]["loss_sum"], [12.0, 8.0, 6.0])    # 2.0 per sentence row: 6, 4, 3 sentences


class TheirWindowLoss(unittest.TestCase):
    """Their reported loss is TransformerLens' lm_cross_entropy_loss per window: `-predicted_log_probs.sum() / n` computed in bf16 (a sum near 1536 is rounded to a multiple of 8,
    so every window loss is a multiple of 1/32 and the mean carries a systematic offset: diagnostics diag_loss.py / diag_loss2.py, anomaly A-024). The anchor reproduces it by calling
    that function window by window; the production loss is the exact float64 token sum."""
    def test_row_wise_loss_is_scaled_to_a_token_sum_and_matches_the_function_per_row(self):
        g = torch.Generator().manual_seed(0); logits = torch.randn(3, 7, V, generator=g); toks = torch.randint(0, V, (3, 7), generator=g); valid = torch.ones(3, 7, dtype=torch.bool)
        calls = []
        def loss_fn(lg, tk):
            calls.append(tuple(lg.shape)); lp = torch.log_softmax(lg, -1)[:, :-1].gather(-1, tk[:, 1:, None])[..., 0]; return -lp.sum() / lp.numel()
        s_, n = run.tl_window_nll(loss_fn)(logits, toks, valid)
        self.assertEqual(calls, [(1, 7, V)] * 3); self.assertEqual(n.tolist(), [6, 6, 6])
        lp = torch.log_softmax(logits, -1)[:, :-1].gather(-1, toks[:, 1:, None])[..., 0]
        self.assertTrue(torch.allclose(s_, (-lp.sum(1)).double(), atol=1e-5))                 # sums = per-window loss x positions

    def test_padding_or_masked_positions_are_refused_because_the_function_averages_every_position(self):
        logits = torch.randn(2, 5, V); toks = torch.randint(0, V, (2, 5)); valid = torch.ones(2, 5, dtype=torch.bool); valid[1, 3:] = False
        with self.assertRaises(AssertionError): run.tl_window_nll(lambda lg, tk: torch.tensor(1.0))(logits, toks, valid)
        valid = torch.ones(2, 5, dtype=torch.bool); valid[0, 0] = False                        # a masked BOS position 0 does not matter for position >= 1 targets
        run.tl_window_nll(lambda lg, tk: torch.tensor(1.0))(logits, toks, valid)


class MaxTokens(unittest.TestCase):
    def test_batches_respect_a_token_budget_without_losing_sentences(self):
        data = {0: [[5] * n for n in (3, 4, 9, 2, 8)]}
        bs = p1.make_batches(data, BOS, "mask", PAD, batch_size=64, max_tokens=20)
        self.assertTrue(all(b["tokens"].numel() <= 20 or len(b["sent"]) == 1 for b in bs)); self.assertEqual(sorted(i for b in bs for i in b["sent"].tolist()), [0, 1, 2, 3, 4])


class SelectivityReport(unittest.TestCase):
    """Synthetic tagging statistics: 3 languages x 60 sentences x 20 tokens; features 0-3 fire only in one language (3 positives per sentence of that language),
    4-73 are universal (the same count pattern in every language, so S = 1; a third of them have rate 0.05 like the selective ones' panel mean),
    74-77 are rare noise (40 positives at the peak language: below the 50 floor), 78-79 never fire."""
    NL, SP, NT, F_, NU = 3, 60, 20, 80, 74

    def tag(self):
        S = self.NL * self.SP; counts = torch.zeros(S, self.F_, dtype=torch.int16); lang = torch.arange(S) // self.SP; odd = torch.arange(S) % 2
        for j in range(4):
            counts[lang == j % self.NL, j] = 3
        for j in range(4, self.NU):
            counts[:, j] = 1 + (j % 3) * odd
        for j in range(74, 78):
            counts[(lang == 1) & (torch.arange(S) % self.SP < 10), j] = 4
        t = p1.TagStats(S, self.F_); t.counts, t.sums, t.ntok = counts, counts.float() * 0.5, torch.full((S,), self.NT, dtype=torch.long)
        return t, lang

    def report(self, **kw):
        t, lang = self.tag(); args = dict(n_boot=20, n_perm=10, seed=0); args.update(kw)
        return run.selectivity_report(t, lang, self.NL, **args)

    def test_supported_features_are_the_ones_with_enough_positives_at_the_peak_language(self):
        self.assertEqual(self.report()["supported"].nonzero().squeeze(1).tolist(), list(range(self.NU)))

    def test_one_language_features_are_selective_and_universal_ones_are_not(self):
        r = self.report()
        self.assertEqual(r["selective"].nonzero().squeeze(1).tolist(), [0, 1, 2, 3])
        self.assertEqual(r["peak"][:4].tolist(), [0, 1, 2, 0]); self.assertAlmostEqual(float(r["S"][0]), 3.0)
        self.assertTrue(torch.allclose(r["S"][4:self.NU], torch.ones(self.NU - 4, dtype=r["S"].dtype)))
        self.assertEqual(r["stable_count"][:4].tolist(), [20] * 4)

    def test_rank_normalised_score_is_defined_only_within_supported_features(self):
        sr = self.report()["S_rank"]
        self.assertTrue(bool((sr[self.NU:] == 0).all())); self.assertTrue(bool(((sr[:self.NU] > 0) & (sr[:self.NU] <= 1)).all())); self.assertAlmostEqual(float(sr[:self.NU].max()), 1.0)
        self.assertTrue(float(sr[:4].min()) > float(sr[4:self.NU].max()))              # the four selective features rank above every universal one

    def test_nu_is_the_deng_statistic_of_the_per_language_mean_activations(self):
        t, lang = self.tag(); r = self.report(); means = p1.mean_activations(t.sums, t.ntok, lang, self.NL)
        self.assertTrue(torch.allclose(r["nu"], p1.deng_nu(means))); self.assertTrue(torch.allclose(r["mean_activation"], means))
        self.assertGreater(float(r["nu"][0, 0]), 0)                                      # feature 0 fires in language 0 only: positive nu there

    def test_matched_universal_controls_come_from_the_non_selective_supported_pool_and_match_the_strata(self):
        r = self.report(); c = r["controls"].tolist()
        self.assertEqual(r["control_strata_short"], []); self.assertEqual(len(c), 4); self.assertEqual(len(c), len(set(c))); self.assertTrue(set(c) <= set(range(4, self.NU)))
        st = r["strata"]; self.assertEqual(sorted(st[c].tolist()), sorted(st[:4].tolist()))      # the controls have the selective features' firing-rate strata

    def test_a_pool_too_small_for_a_stratum_is_reported_not_hidden(self):
        t, lang = self.tag(); t.counts[:, 4:self.NU] = 0; t.sums[:, 4:self.NU] = 0                # every universal feature silenced: nothing to match with
        r = run.selectivity_report(t, lang, self.NL, n_boot=20, n_perm=10, seed=0)
        self.assertEqual(len(r["controls"]), 0); self.assertTrue(len(r["control_strata_short"]) > 0)

    def test_a_feature_whose_score_equals_the_null_cutoff_is_not_selective(self):
        S = self.NL * self.SP; t = p1.TagStats(S, 10); t.counts = torch.ones(S, 10, dtype=torch.int16); t.sums = t.counts.float(); t.ntok = torch.full((S,), self.NT, dtype=torch.long)
        r = run.selectivity_report(t, torch.arange(S) // self.SP, self.NL, n_boot=20, n_perm=5, seed=0)    # constant counts: S == 1 in every shuffle
        self.assertEqual(float(r["S"][0]), r["null_cutoff"]); self.assertEqual(int(r["selective"].sum()), 0)

    def test_the_stability_threshold_is_inclusive(self):
        self.assertEqual(self.report(min_stable_frac=1.0)["selective"].nonzero().squeeze(1).tolist(), [0, 1, 2, 3])        # stable_count == n_boot, needed == n_boot: still passes

    def test_a_supported_high_score_feature_with_an_unstable_peak_is_not_selective(self):
        t, lang = self.tag(); S = len(lang)
        i = torch.arange(S) % self.SP                                                                                           # languages 0 and 1 tied (150 positives each; first index wins) with
        t.counts[:, 79] = torch.where(lang == 0, 2 + i % 2, torch.where(lang == 1, 2 + (i // 2) % 2, 0)).to(torch.int16)   # different sentences carrying them: the peak flips under resampling
        t.sums = t.counts.float() * 0.5
        r = run.selectivity_report(t, lang, self.NL, n_boot=20, n_perm=10, seed=0)
        self.assertTrue(bool(r["supported"][79])); self.assertGreater(float(r["S"][79]), r["null_cutoff"]); self.assertLess(int(r["stable_count"][79]), 16)
        self.assertFalse(bool(r["selective"][79])); self.assertEqual(r["selective"].nonzero().squeeze(1).tolist(), [0, 1, 2, 3])

    def test_strata_are_quantiles_of_the_panel_mean_firing_rate_within_supported_features(self):
        r = self.report(); sup = r["supported"]
        self.assertTrue(torch.equal(r["strata"][sup], p1.strata_ids(r["rates"].mean(0)[sup], 5))); self.assertTrue(bool((r["strata"][~sup] == -1).all()))

    def test_the_stability_requirement_scales_with_the_number_of_draws_80_percent_as_in_the_proposal(self):
        for n_boot, need in ((7, 6), (10, 8), (20, 16), (100, 80)):                 # 0.8 * 7 = 5.6 rounds UP
            r = self.report(n_boot=n_boot); self.assertEqual(r["min_stable_needed"], need)
            self.assertEqual(r["selective"].nonzero().squeeze(1).tolist(), [0, 1, 2, 3])               # a smoke run with fewer draws must not silently select nothing

    def test_the_selected_sets_are_reproducible_for_a_given_seed(self):
        a, b = self.report(seed=3), self.report(seed=3)
        self.assertTrue(torch.equal(a["controls"], b["controls"])); self.assertEqual(a["null_cutoff"], b["null_cutoff"])


class SaveOutputs(unittest.TestCase):
    def test_files_roundtrip_and_carry_the_provenance(self):
        import csv, json, tempfile
        fake, data = Fake(), corpus(); res = call(fake, data, "mask", bits=(8, 0)); names = ["aaa", "bbb", "ccc"]
        out = Path(tempfile.mkdtemp()); meta = {"regime": "mask", "corpus": "flores", "seed": 0}
        run.save_outputs(out, meta, res, [8, 0], names, F, threshold=0.001, report=None)
        rows = list(csv.DictReader(open(out / "summary.csv")))
        self.assertEqual([r["condition"] for r in rows], ["FP16 baseline", "RTN INT8", "restore-only control (INT0)"]); self.assertEqual(list(rows[0])[:3], ["layer", "condition", "method"])
        lang = list(csv.DictReader(open(out / "per_language.csv"))); self.assertEqual(len(lang), 6)
        z = torch.load(out / "survival.pt"); self.assertEqual(z["languages"], names); self.assertEqual(z["bits"], [8, 0])
        self.assertTrue(torch.equal(z["corr"][8], res[8]["corr"])); self.assertTrue(torch.equal(z["firing_rate"], res["FP"]["firing_rate"])); self.assertEqual(z["n"], res["FP"]["n"])
        self.assertEqual(z["loss_sum"]["FP"], res["FP"]["loss_sum"]); self.assertEqual(z["loss_n"][8], res[8]["loss_n"])
        m = json.loads((out / "meta.json").read_text()); self.assertEqual(m["regime"], "mask"); self.assertEqual(m["languages"], names); self.assertIn("their_functions_sha256", m)
        self.assertFalse((out / "selectivity.pt").exists())

    def test_selectivity_is_saved_when_a_report_is_given(self):
        import tempfile
        t, lang = SelectivityReport().tag(); r = run.selectivity_report(t, lang, 3, n_boot=20, n_perm=5, seed=0)
        out = Path(tempfile.mkdtemp()); fake, data = Fake(), corpus(); res = call(fake, data, "mask", bits=(8,))
        run.save_outputs(out, {}, res, [8], ["a", "b", "c"], F, threshold=0.001, report=r)
        z = torch.load(out / "selectivity.pt"); self.assertTrue(torch.equal(z["selective"], r["selective"])); self.assertEqual(z["null_cutoff"], r["null_cutoff"]); self.assertTrue(torch.equal(z["controls"], r["controls"]))


class PackageVersions(unittest.TestCase):
    def test_versions_come_from_package_metadata_and_a_missing_package_is_reported_not_raised(self):
        v = run.package_versions(["torch", "no-such-package-xyz"])
        self.assertEqual(list(v), ["torch", "no-such-package-xyz"]); self.assertTrue(v["torch"].startswith(torch.__version__.split("+")[0])); self.assertEqual(v["no-such-package-xyz"], "not installed")


class ForwardPathCheck(unittest.TestCase):
    def test_identical_paths_pass_and_a_differing_path_raises(self):
        toks = torch.randint(3, V, (2, 6)); f = Fake(); full = lambda t: f.forward(t); short = lambda t: (f.forward(t)[0], None)
        run.check_forward_paths(full, short, toks)
        bad = lambda t: (f.forward(t)[0] * 1.001, None)
        with self.assertRaises(AssertionError): run.check_forward_paths(full, bad, toks)


if __name__ == "__main__":
    unittest.main()
