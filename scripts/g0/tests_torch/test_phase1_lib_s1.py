"""S1 tests for scripts/g0/phase1_lib.py (CPU, torch): batching with BOS regimes and valid masks, per-language streaming Pearson
(their float64 sufficient-statistics math), sentence-level tagging statistics. Run:  <torch python> -m unittest discover -s scripts/g0/tests_torch -v"""
import sys, unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import phase1_lib as p1  # noqa: E402

BOS, PAD = 2, 0


def sents(lengths, start=10):
    out, v = [], start
    for n in lengths:
        out.append(list(range(v, v + n))); v += n
    return out


class Batching(unittest.TestCase):
    def test_mask_regime_prepends_bos_and_excludes_exactly_position_zero_and_padding(self):
        bs = p1.make_batches({0: sents([3, 5, 2])}, BOS, "mask", PAD, max_len=256, batch_size=8)
        self.assertEqual(len(bs), 1); b = bs[0]
        self.assertTrue((b["tokens"][:, 0] == BOS).all()); self.assertFalse(b["valid"][:, 0].any())
        lens = (b["valid"].sum(1)).tolist(); self.assertEqual(sorted(lens), [2, 3, 5])           # only real content tokens are valid
        self.assertTrue((b["tokens"][~b["valid"] & (b["tokens"] != BOS)] == PAD).all())          # padding is the pad id

    def test_none_regime_has_no_bos_and_every_real_token_is_valid(self):
        b = p1.make_batches({0: sents([3, 5])}, BOS, "none", PAD, max_len=256, batch_size=8)[0]
        self.assertFalse((b["tokens"] == BOS).any()); self.assertEqual(sorted(b["valid"].sum(1).tolist()), [3, 5])
        self.assertTrue(b["valid"][:, 0].all())

    def test_every_sentence_appears_once_batches_are_single_language_and_length_sorted(self):
        data = {0: sents([9, 4, 7, 2, 5]), 1: sents([8, 3, 3], start=500)}
        bs = p1.make_batches(data, BOS, "mask", PAD, max_len=256, batch_size=2)
        seen = {(b["lang"], int(i)) for b in bs for i in b["sent"]}
        self.assertEqual(seen, {(0, i) for i in range(5)} | {(1, i) for i in range(3)})
        self.assertEqual(sum(len(b["sent"]) for b in bs), 8); self.assertTrue(all(len(b["sent"]) <= 2 for b in bs))
        for b in bs:
            lens = b["valid"].sum(1).tolist(); self.assertEqual(lens, sorted(lens))

    def test_content_longer_than_max_len_is_truncated_and_counted(self):
        data = {0: sents([10, 3])}
        bs = p1.make_batches(data, BOS, "mask", PAD, max_len=6, batch_size=8)
        self.assertEqual(max(b["valid"].sum(1).max().item() for b in bs), 6)
        self.assertEqual(sum(b["n_truncated"] for b in bs), 1)

    def test_unknown_regime_is_rejected(self):
        with self.assertRaises(ValueError):
            p1.make_batches({0: sents([3])}, BOS, "keep-ish", PAD)

    def test_right_padding_cannot_change_activations_of_a_causal_model_at_real_positions(self):
        """The design assumption behind padded batches: for a causal model the outputs at real positions equal the per-sentence outputs."""
        g = torch.Generator().manual_seed(0); emb = torch.randn(600, 4, generator=g)
        causal = lambda toks: torch.cumsum(emb[toks], dim=1)            # position t depends only on tokens <= t
        data = {0: sents([4, 7, 2])}
        b = p1.make_batches(data, BOS, "mask", PAD, max_len=256, batch_size=8)[0]
        out = causal(b["tokens"])
        for row, sid in enumerate(b["sent"].tolist()):
            alone = p1.make_batches({0: [data[0][sid]]}, BOS, "mask", PAD, max_len=256, batch_size=1)[0]
            ref = causal(alone["tokens"])[0][alone["valid"][0]]
            self.assertTrue(torch.allclose(out[row][b["valid"][row]], ref))


class LangPearson(unittest.TestCase):
    def data(self, seed=0, n=400, d=12):
        g = torch.Generator().manual_seed(seed)
        x = torch.relu(torch.randn(n, d, generator=g) - 0.3); y = x + 0.2 * torch.randn(n, d, generator=g)
        return x, y

    def test_per_language_corr_equals_numpy_in_float64_and_languages_do_not_mix(self):
        x, y = self.data(); lang = (torch.arange(400) % 3)
        s = p1.LangPearson(3, 12, "cpu")
        for i in range(0, 400, 50):
            for l in range(3):
                m = lang[i:i + 50] == l
                if m.any(): s.update(l, x[i:i + 50][m], y[i:i + 50][m])
        for l in range(3):
            r = s.result(l); xs, ys = x[lang == l].double().numpy(), y[lang == l].double().numpy()
            ref = np.array([np.corrcoef(xs[:, j], ys[:, j])[0, 1] for j in range(12)])
            self.assertLess(np.abs(r["corr"].numpy() - ref).max(), 1e-9); self.assertEqual(r["n"], int((lang == l).sum()))
            self.assertTrue(torch.allclose(r["firing_rate"], (x[lang == l] > 0).double().mean(0)))
            self.assertTrue(torch.equal(r["max_activation"], x[lang == l].double().max(0).values))

    def test_pooled_result_is_the_sum_over_languages(self):
        x, y = self.data(1); s = p1.LangPearson(2, 12, "cpu")
        s.update(0, x[:150], y[:150]); s.update(1, x[150:], y[150:])
        ref = np.array([np.corrcoef(x[:, j].double().numpy(), y[:, j].double().numpy())[0, 1] for j in range(12)])
        self.assertLess(np.abs(s.pooled()["corr"].numpy() - ref).max(), 1e-9)
        pooled = s.pooled(); self.assertEqual(pooled["n"], 400)
        self.assertTrue(torch.equal(pooled["max_activation"], x.double().max(0).values))
        self.assertTrue(torch.allclose(pooled["firing_rate"], (x > 0).double().mean(0))); self.assertTrue(torch.allclose(pooled["mean_activation"], x.double().mean(0)))

    def test_zero_variance_feature_gives_corr_zero_like_their_clamped_denominator_and_empty_language_is_safe(self):
        x, y = self.data(2); x[:, 0] = 0.0; s = p1.LangPearson(2, 12, "cpu"); s.update(0, x, y)
        self.assertEqual(float(s.result(0)["corr"][0]), 0.0)
        self.assertEqual(s.result(1)["n"], 0)                           # a language with no data yields n = 0, not an exception


class TagStats(unittest.TestCase):
    def test_per_sentence_counts_and_sums_equal_brute_force(self):
        g = torch.Generator().manual_seed(3); d = 9
        feats = [torch.relu(torch.randn(n, d, generator=g) - 0.5) for n in (4, 6, 3)]       # three sentences
        t = p1.TagStats(n_sentences=3, d=d)
        row_sent = torch.cat([torch.full((len(f),), i) for i, f in enumerate(feats)])
        t.update(torch.cat(feats), row_sent)
        for i, f in enumerate(feats):
            self.assertTrue(torch.equal(t.counts[i].long(), (f > 0).sum(0)))
            self.assertTrue(torch.allclose(t.sums[i], f.sum(0), atol=1e-5)); self.assertEqual(int(t.ntok[i]), len(f))

    def test_updates_accumulate_across_batches(self):
        d = 5; a, b = torch.ones(2, d), 2 * torch.ones(3, d); t = p1.TagStats(1, d)
        t.update(a, torch.zeros(2, dtype=torch.long)); t.update(b, torch.zeros(3, dtype=torch.long))
        self.assertEqual(int(t.ntok[0]), 5); self.assertTrue(torch.allclose(t.sums[0], torch.full((d,), 8.0)))


if __name__ == "__main__":
    unittest.main()
