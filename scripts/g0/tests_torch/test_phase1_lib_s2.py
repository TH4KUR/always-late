"""S2 tests for scripts/g0/phase1_lib.py: selectivity S(f) (proposal section 4.1), support rule, peak language, bootstrap stability,
shuffled-label null cutoff, Deng et al.'s nu, quantile-matched controls. Synthetic data with known structure; CPU."""
import sys, unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import phase1_lib as p1  # noqa: E402

N_LANG, SENT, NTOK = 3, 200, 30
RATES = np.array([   # [feature][language] true firing rates
    [0.05, 0.0, 0.0],        # f0 selective for language 0
    [0.02, 0.02, 0.02],      # f1 universal
    [0.0, 0.0005, 0.0],      # f2 almost never fires: peak evidence below 50 positives
    [0.0, 0.0, 0.0],         # f3 dead
    [0.01, 0.01, 0.03],      # f4 selective for language 2 (3x)
    [0.040, 0.038, 0.0],     # f5 peak between languages 0 and 1 is a near tie: unstable peak
])


def synth(seed=0):
    g = np.random.default_rng(seed); d = RATES.shape[0]
    lang = np.repeat(np.arange(N_LANG), SENT); S = len(lang)
    ntok = np.full(S, NTOK)
    counts = np.stack([g.binomial(NTOK, RATES[:, l]) for l in lang]).astype(np.int16)           # [S, d] positives per sentence
    l0, l1 = np.where(lang == 0)[0], np.where(lang == 1)[0]          # f5: languages 0 and 1 have the SAME multiset of per-sentence counts, language 0 wins by one count
    counts[l1, 5] = counts[l0, 5][g.permutation(SENT)]; counts[l0[0], 5] += 1
    vals = g.uniform(0.5, 2.0, size=(S, d)); sums = (counts * vals).astype(np.float32)
    return torch.tensor(counts), torch.tensor(sums), torch.tensor(ntok), torch.tensor(lang)


class Rates(unittest.TestCase):
    def test_rates_means_and_totals_equal_brute_force(self):
        counts, sums, ntok, lang = synth()
        r = p1.firing_rates(counts, ntok, lang, N_LANG); m = p1.mean_activations(sums, ntok, lang, N_LANG)
        for l in range(N_LANG):
            sel = (lang == l).numpy()
            self.assertTrue(np.allclose(r[l].numpy(), counts.numpy()[sel].sum(0) / ntok.numpy()[sel].sum()))
            self.assertTrue(np.allclose(m[l].numpy(), sums.numpy()[sel].sum(0) / ntok.numpy()[sel].sum(), rtol=1e-5))
        self.assertTrue(torch.equal(p1.language_totals(counts, lang, N_LANG)[1].long(), counts[lang == 1].long().sum(0)))


class Selectivity(unittest.TestCase):
    def test_S_is_max_over_mean_of_the_language_rates(self):
        r = torch.tensor([[0.06, 0.02, 0.0], [0.0, 0.02, 0.0], [0.0, 0.02, 0.0]])
        S = p1.selectivity_S(r)
        self.assertAlmostEqual(float(S[0]), 3.0, places=6); self.assertAlmostEqual(float(S[1]), 1.0, places=6); self.assertEqual(float(S[2]), 0.0)   # dead feature: 0, never NaN

    def test_structure_of_the_synthetic_features(self):
        counts, sums, ntok, lang = synth()
        r = p1.firing_rates(counts, ntok, lang, N_LANG); S = p1.selectivity_S(r)
        self.assertGreater(float(S[0]), 2.9); self.assertAlmostEqual(float(S[1]), 1.0, delta=0.15); self.assertGreater(float(S[4]), 1.6)
        self.assertEqual(p1.peak_language(r).tolist()[0], 0); self.assertEqual(p1.peak_language(r).tolist()[4], 2)

    def test_support_needs_the_rate_floor_and_fifty_positives_at_the_peak_language(self):
        counts, sums, ntok, lang = synth()
        r = p1.firing_rates(counts, ntok, lang, N_LANG); tot = p1.language_totals(counts, lang, N_LANG)
        sup = p1.supported_mask(r, tot, min_panel_rate=1e-5, min_pos=50)
        self.assertEqual(sup.tolist(), [True, True, False, False, True, True])      # f2 has < 50 positives, f3 never fires
        self.assertFalse(bool(p1.supported_mask(r, tot, min_panel_rate=0.5, min_pos=50)[1]))   # the floor binds when raised


class Stability(unittest.TestCase):
    def test_clear_peaks_are_stable_and_a_near_tie_is_not(self):
        counts, sums, ntok, lang = synth()
        r = p1.firing_rates(counts, ntok, lang, N_LANG); peak = p1.peak_language(r)
        st = p1.bootstrap_peak_stability(counts, ntok, lang, N_LANG, peak, n_boot=100, frac=0.9, seed=0)
        self.assertEqual(int(st[0]), 100); self.assertGreaterEqual(int(st[4]), 90); self.assertLess(int(st[5]), 80)

    def test_deterministic_given_the_seed(self):
        counts, sums, ntok, lang = synth(); peak = p1.peak_language(p1.firing_rates(counts, ntok, lang, N_LANG))
        a = p1.bootstrap_peak_stability(counts, ntok, lang, N_LANG, peak, n_boot=30, seed=5); b = p1.bootstrap_peak_stability(counts, ntok, lang, N_LANG, peak, n_boot=30, seed=5)
        self.assertTrue(torch.equal(a, b))


class NullCutoff(unittest.TestCase):
    def test_cutoff_sits_between_the_null_and_the_real_selective_features(self):
        counts, sums, ntok, lang = synth()
        r = p1.firing_rates(counts, ntok, lang, N_LANG); S = p1.selectivity_S(r)
        sup = p1.supported_mask(r, p1.language_totals(counts, lang, N_LANG))
        cut = p1.shuffled_null_cutoff(counts, ntok, lang, N_LANG, sup, n_perm=20, q=0.95, seed=0)
        self.assertGreater(cut, 1.0); self.assertLess(cut, float(S[0])); self.assertLess(cut, float(S[4]))
        self.assertGreater(cut, float(S[1]))                                          # the universal feature is below the null cutoff

    def test_deterministic_given_the_seed(self):
        counts, sums, ntok, lang = synth(); sup = torch.ones(6, dtype=torch.bool)
        self.assertEqual(p1.shuffled_null_cutoff(counts, ntok, lang, N_LANG, sup, n_perm=5, seed=2), p1.shuffled_null_cutoff(counts, ntok, lang, N_LANG, sup, n_perm=5, seed=2))


class DengNu(unittest.TestCase):
    def test_nu_is_mean_on_the_language_minus_the_mean_of_the_other_languages_means(self):
        m = torch.tensor([[3.0, 1.0], [1.0, 1.0], [2.0, 4.0]])
        nu = p1.deng_nu(m)
        self.assertAlmostEqual(float(nu[0, 0]), 3.0 - (1.0 + 2.0) / 2, places=6); self.assertAlmostEqual(float(nu[2, 1]), 4.0 - (1.0 + 1.0) / 2, places=6)
        self.assertAlmostEqual(float(nu[1, 0]), 1.0 - (3.0 + 2.0) / 2, places=6)          # signed: negative when a language is below the others


class Controls(unittest.TestCase):
    def test_strata_are_equal_count_quantiles(self):
        v = torch.arange(100.0); st = p1.strata_ids(v, 5)
        self.assertEqual(torch.bincount(st).tolist(), [20] * 5); self.assertEqual(int(st[0]), 0); self.assertEqual(int(st[99]), 4)

    def test_matched_sample_has_the_selective_stratum_counts_no_overlap_and_flags_shortfalls(self):
        g = torch.Generator().manual_seed(0); v = torch.rand(300, generator=g); st = p1.strata_ids(v, 5)
        sel = torch.zeros(300, dtype=torch.bool); sel[:40] = True; pool = ~sel
        idx, short = p1.matched_sample(st, sel, pool, seed=1)
        self.assertEqual(torch.bincount(st[idx], minlength=5).tolist(), torch.bincount(st[sel], minlength=5).tolist())
        self.assertFalse(bool(sel[idx].any())); self.assertEqual(len(set(idx.tolist())), len(idx)); self.assertEqual(short, [])
        idx2, _ = p1.matched_sample(st, sel, pool, seed=1); self.assertTrue(torch.equal(idx, idx2))
        tiny = torch.zeros(300, dtype=torch.bool); tiny[100:103] = True                    # a pool far too small for the selective set
        _, short2 = p1.matched_sample(st, sel, tiny, seed=1); self.assertTrue(len(short2) > 0)


if __name__ == "__main__":
    unittest.main()
