"""Tests for the equal-token-budget position sampling (decision D-015): exactly N content positions per language, uniform without replacement, seeded and independent of
batching and of the other languages; BOS and padding never count."""
import sys, unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import phase1_lib as p1   # noqa: E402

BOS, PAD = 2, 0


def corpus():
    g = torch.Generator().manual_seed(3)
    lens = {0: [5, 9, 4, 12, 7, 3], 1: [6, 6, 10, 2, 40, 8, 9], 2: [8, 5, 11, 300, 4]}        # language 2 holds a sentence longer than max_len=256
    return {l: [torch.randint(3, 40, (n,), generator=g).tolist() for n in ns] for l, ns in lens.items()}


def content_tokens(data, max_len=256):
    return {l: sum(min(len(s), max_len) for s in ss) for l, ss in data.items()}


class KeepMasks(unittest.TestCase):
    def test_budget_is_the_smallest_per_language_token_count_with_truncation(self):
        d = corpus(); self.assertEqual(p1.budget_for(d), min(content_tokens(d).values())); self.assertEqual(p1.budget_for(d, max_len=8), min(content_tokens(d, 8).values()))

    def test_every_language_keeps_exactly_the_budget_and_masks_match_the_truncated_sentences(self):
        d = corpus(); n = p1.budget_for(d); keep = p1.keep_masks(d, n, seed=0)
        for l, ss in d.items():
            self.assertEqual([len(m) for m in keep[l]], [min(len(s), 256) for s in ss]); self.assertEqual(int(sum(m.sum() for m in keep[l])), n)
            self.assertTrue(all(m.dtype == torch.bool for m in keep[l]))

    def test_a_language_with_fewer_tokens_than_the_budget_keeps_everything(self):
        d = corpus(); keep = p1.keep_masks(d, 10 ** 6, seed=0)
        for l in d: self.assertEqual(int(sum(m.sum() for m in keep[l])), content_tokens(d)[l])

    def test_reproducible_for_a_seed_different_for_another_and_independent_of_the_other_languages(self):
        d = corpus(); n = p1.budget_for(d); a, b, c = p1.keep_masks(d, n, 0), p1.keep_masks(d, n, 0), p1.keep_masks(d, n, 1)
        flat = lambda k, l: torch.cat(k[l])
        for l in d: self.assertTrue(torch.equal(flat(a, l), flat(b, l)))
        self.assertTrue(any(not torch.equal(flat(a, l), flat(c, l)) for l in d))
        only1 = p1.keep_masks({1: d[1]}, n, 0); self.assertTrue(torch.equal(flat(only1, 1), flat(a, 1)))        # language 1's draw does not depend on languages 0 and 2

    def test_two_languages_of_identical_size_get_different_draws(self):
        same = [list(range(3, 23)) for _ in range(5)]; keep = p1.keep_masks({0: same, 1: same}, 30, seed=0)
        self.assertFalse(torch.equal(torch.cat(keep[0]), torch.cat(keep[1])))

    def test_split_offset_gives_an_independent_draw(self):
        d = corpus(); n = p1.budget_for(d); a, b = p1.keep_masks(d, n, 0, split_offset=0), p1.keep_masks(d, n, 0, split_offset=1)
        self.assertTrue(any(not torch.equal(torch.cat(a[l]), torch.cat(b[l])) for l in d))

    def test_selection_is_uniform_every_position_is_kept_with_probability_budget_over_count(self):
        d = {0: [list(range(3, 23)) for _ in range(5)]}; n = 30; total = 100; freq = torch.zeros(total)
        for seed in range(400): freq += torch.cat(p1.keep_masks(d, n, seed)[0]).float()
        freq /= 400; self.assertLess(float((freq - n / total).abs().max()), 0.12); self.assertAlmostEqual(float(freq.mean()), n / total, places=6)


class BatchesCarryTheMask(unittest.TestCase):
    def test_eq_is_a_subset_of_valid_never_contains_bos_or_padding_and_counts_the_budget_per_language(self):
        d = corpus(); n = p1.budget_for(d); keep = p1.keep_masks(d, n, 0)
        for regime in ("mask", "none"):
            totals = {l: 0 for l in d}
            for b in p1.make_batches(d, BOS, regime, PAD, batch_size=3, keep=keep):
                self.assertTrue(bool((b["eq"] & ~b["valid"]).sum() == 0))
                if regime == "mask": self.assertFalse(bool(b["eq"][:, 0].any()))
                totals[b["lang"]] += int(b["eq"].sum())
            self.assertEqual(totals, {l: n for l in d})

    def test_the_selected_positions_are_the_same_whatever_the_batching(self):
        d = corpus(); keep = p1.keep_masks(d, p1.budget_for(d), 5)
        def selected(**kw):
            out = {}
            for b in p1.make_batches(d, BOS, "mask", PAD, keep=keep, **kw):
                for r, si in enumerate(b["sent"].tolist()):
                    out[(b["lang"], si)] = (b["tokens"][r][b["eq"][r]]).tolist(), b["eq"][r].nonzero().squeeze(-1).tolist()
            return out
        a, b = selected(batch_size=1), selected(batch_size=64, max_tokens=50)
        self.assertEqual(a, b)
        for (l, si), (toks, pos) in a.items():                                                   # the kept tokens are the kept positions of the original sentence (offset by BOS)
            m = keep[l][si]; self.assertEqual(pos, [i + 1 for i in m.nonzero().squeeze(-1).tolist()])

    def test_without_a_keep_argument_eq_equals_valid(self):
        for b in p1.make_batches(corpus(), BOS, "mask", PAD, batch_size=4): self.assertTrue(torch.equal(b["eq"], b["valid"]))


if __name__ == "__main__":
    unittest.main()
