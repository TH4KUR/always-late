import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_func_intervene as fi
import phase1_func_run as fr
import phase1_lib as p1
from test_phase1_func_intervene import ToySAE

V, D, F, BOS = 11, 8, 32, 1


class ToyModel:
    """A tiny causal model split at 'layer 12': lower(tokens) -> residual, upper(residual) -> logits. Causal mixing by cumulative means, so padding on the right cannot influence earlier positions."""
    def __init__(self, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.E = torch.randn(V, D, generator=g); self.M1 = torch.randn(D, D, generator=g) * 0.3
        self.M2 = torch.randn(D, D, generator=g) * 0.3; self.U = torch.randn(D, V, generator=g)

    @staticmethod
    def _cmean(x):
        return x.cumsum(1) / torch.arange(1, x.shape[1] + 1, dtype=x.dtype)[None, :, None]

    def lower(self, tokens):
        x = self.E[tokens]
        return torch.tanh(self._cmean(x) @ self.M1 + x) * 3

    def upper(self, resid):
        return torch.tanh(self._cmean(resid) @ self.M2 + resid) @ self.U

    def full(self, tokens):
        return self.upper(self.lower(tokens))


def corpus(seed=0, sizes=(5, 4, 6), lo=2, hi=9):
    g = torch.Generator().manual_seed(seed)
    data = {}
    for l, n in enumerate(sizes):
        data[l] = [torch.randint(2, V, (int(torch.randint(lo, hi, (1,), generator=g)),), generator=g).tolist() for _ in range(n)]
    return data


def brute_nll(model, sae, sent, idx=None, kind=None, **kw):
    """NLL sum and token count of ONE sentence run alone (BOS prepended), with an optional group edit at the valid positions."""
    tokens = torch.tensor([[BOS] + sent]); valid = torch.zeros_like(tokens, dtype=torch.bool); valid[:, 1:] = True
    resid = model.lower(tokens)
    if kind is not None:
        a = fi.encode_valid(resid, valid, sae.encode)
        resid, _ = fi.apply_edit(resid, valid, a, sae.W_dec, idx, kind, **kw)
    s, n = p1.chunked_nll(model.upper(resid), tokens, valid)
    return float(s[0]), int(n[0])


class Engine(unittest.TestCase):
    def setUp(self):
        self.model, self.sae = ToyModel(), ToySAE(d=D, F=F)
        self.data = corpus()
        self.batches = p1.make_batches(self.data, BOS, "mask", 0, 256, batch_size=3)
        self.n_sent = {l: len(v) for l, v in self.data.items()}
        self.idx = torch.tensor([2, 5, 9, 17])

    def run_engine(self, configs, **kw):
        return fr.evaluate_precision(self.model.lower, self.model.upper, self.sae.encode, self.sae.W_dec, self.batches, self.n_sent, configs, decode=self.sae.decode, **kw)

    def test_baseline_equals_one_sentence_at_a_time(self):
        out = self.run_engine([])
        for l, sents in self.data.items():
            for i, s in enumerate(sents):
                nll, n = brute_nll(self.model, self.sae, s)
                self.assertAlmostEqual(float(out["items"][(l, "none")]["nll"][i]), nll, places=4)
                self.assertEqual(int(out["items"][(l, "none")]["ntok"][i]), n)

    def test_ablation_equals_brute_force_per_sentence_in_batches_with_padding(self):
        cfg = {"name": "zero", "kind": "ablate", "idx": self.idx, "langs": {0, 1, 2}}
        out = self.run_engine([cfg])
        changed = 0
        for l, sents in self.data.items():
            for i, s in enumerate(sents):
                nll, _ = brute_nll(self.model, self.sae, s, self.idx, "ablate")
                self.assertAlmostEqual(float(out["items"][(l, "zero")]["nll"][i]), nll, places=4)
                changed += abs(nll - brute_nll(self.model, self.sae, s)[0]) > 1e-6
        self.assertGreater(changed, 0)                                       # the edit does something on the toy

    def test_identity_edits_equal_the_baseline(self):
        cfgs = [{"name": "scale1", "kind": "scale", "idx": self.idx, "langs": {0, 1, 2}, "kw": {"alpha": 1.0}},
                {"name": "sham", "kind": "sham", "idx": self.idx, "langs": {0, 1, 2}}]
        out = self.run_engine(cfgs)
        for l in range(3):
            base = out["items"][(l, "none")]
            self.assertTrue(torch.equal(out["items"][(l, "scale1")]["nll"], base["nll"]))          # exact: no arithmetic in the identity edit
            self.assertTrue(torch.allclose(out["items"][(l, "sham")]["nll"], base["nll"], rtol=1e-5))   # float32 round-off of the sham round trip only
            for name in ("scale1", "sham"):
                self.assertTrue(torch.equal(out["items"][(l, name)]["ntok"], base["ntok"]))

    def test_configs_apply_only_to_their_languages(self):
        cfgs = [{"name": "only1", "kind": "ablate", "idx": self.idx, "langs": {1}}, {"name": "panel", "kind": "ablate", "idx": self.idx, "langs": {0, 2}}]
        out = self.run_engine(cfgs)
        self.assertEqual(sorted(k for k in out["items"] if k[1] == "only1"), [(1, "only1")])
        self.assertEqual(sorted(k for k in out["items"] if k[1] == "panel"), [(0, "panel"), (2, "panel")])
        self.assertEqual(sorted(k for k in out["items"] if k[1] == "none"), [(0, "none"), (1, "none"), (2, "none")])

    def test_items_are_paired_across_configs_and_complete(self):
        out = self.run_engine([{"name": "zero", "kind": "ablate", "idx": self.idx, "langs": {0, 1, 2}}])
        for l in range(3):
            self.assertEqual(out["items"][(l, "none")]["ntok"].shape[0], self.n_sent[l])
            self.assertTrue(torch.equal(out["items"][(l, "zero")]["ntok"], out["items"][(l, "none")]["ntok"]))
            self.assertTrue(bool((out["items"][(l, "none")]["ntok"] > 0).all()))

    def test_perturbation_statistics(self):
        out = self.run_engine([{"name": "zero", "kind": "ablate", "idx": self.idx, "langs": {0}}, {"name": "scale1", "kind": "scale", "idx": self.idx, "langs": {0}, "kw": {"alpha": 1.0}}])
        n_valid = sum(len(s) for s in self.data[0])
        st = out["stats"][(0, "zero")]
        self.assertEqual(st["rows"], n_valid)
        self.assertGreater(st["sum_norm_ratio"], 0)
        self.assertEqual(out["stats"][(0, "scale1")]["sum_norm_ratio"], 0.0)

    def test_fidelity_sums_match_brute_force(self):
        out = self.run_engine([])
        for l, sents in self.data.items():
            e2 = x2 = l0 = rows = 0.0
            for s in sents:
                tokens = torch.tensor([[BOS] + s]); x = self.model.lower(tokens)[0, 1:]
                a = self.sae.encode(x); err = x - self.sae.decode(a)
                e2 += float((err.double() ** 2).sum()); x2 += float((x.double() ** 2).sum()); l0 += float((a > 0).sum()); rows += x.shape[0]
            f = out["fidelity"][l]
            self.assertAlmostEqual(f["sum_err2"], e2, delta=1e-5 * abs(e2)); self.assertAlmostEqual(f["sum_x2"], x2, delta=1e-5 * abs(x2))
            self.assertEqual(f["sum_l0"], l0); self.assertEqual(f["rows"], rows)

    def test_resample_with_a_fixed_pool_matches_brute_force(self):
        pool = torch.tensor([[0.5, 0.0, 1.0, 2.0]])                           # one pool row: deterministic whatever the draw
        out = self.run_engine([{"name": "res", "kind": "resample", "idx": self.idx, "langs": {2}, "kw": {"pool": pool, "seed": 0}}])
        for i, s in enumerate(self.data[2]):
            nll, _ = brute_nll(self.model, self.sae, s, self.idx, "resample", pool=pool, generator=torch.Generator().manual_seed(0))
            self.assertAlmostEqual(float(out["items"][(2, "res")]["nll"][i]), nll, places=4)

    def test_donor_patch_with_the_recipients_own_features_is_identity(self):
        donors = {}
        with torch.no_grad():
            for bi, b in enumerate(self.batches):
                resid = self.model.lower(b["tokens"]); donors[bi] = fi.encode_valid(resid, b["valid"], self.sae.encode)
        out = self.run_engine([{"name": "don", "kind": "donor", "idx": self.idx, "langs": {0, 1, 2}}], donors=donors)
        for l in range(3):
            self.assertTrue(torch.equal(out["items"][(l, "don")]["nll"], out["items"][(l, "none")]["nll"]))

    def test_donor_without_donors_is_an_error(self):
        with self.assertRaises(ValueError):
            self.run_engine([{"name": "don", "kind": "donor", "idx": self.idx, "langs": {0}}])

    def test_unknown_kind_is_rejected_before_any_batch_runs(self):
        with self.assertRaises(ValueError):
            self.run_engine([{"name": "t", "kind": "teleport", "idx": self.idx, "langs": set()}])

    def test_duplicate_config_names_are_an_error(self):
        c = {"name": "x", "kind": "ablate", "idx": self.idx, "langs": {0}}
        with self.assertRaises(ValueError):
            self.run_engine([c, dict(c)])
        with self.assertRaises(ValueError):
            self.run_engine([{"name": "none", "kind": "ablate", "idx": self.idx, "langs": {0}}])


def toy_manifest():
    return {"languages": ["aa", "bb", "cc"], "Ks": [2, None], "Rs": [2, 1],
            "groups": {"aa": {"all": [9, 8, 7, 6]}, "bb": {"all": [5, 4]}, "cc": {"all": [3, 2, 1]}},
            "sizes": {"aa": {"K2": 2, "all": 4}, "bb": {"K2": 2, "all": 2}, "cc": {"K2": 2, "all": 3}},
            "controls": {"aa": {"K2": [[20, 21], [22, 23]], "all": [[24, 25, 26, 27]]}, "bb": {"K2": [[30, 31], [32, 33]], "all": [[34, 35]]},
                         "cc": {"K2": [[40, 41], [42, 43]], "all": [[44, 45, 46]]}}}


class BuildConfigs(unittest.TestCase):
    def build(self, targets=("aa", "bb"), families=None, run_langs=("aa", "bb", "cc"), panel=("aa", "cc"), off_target_label="K2", **kw):
        fam = families or {"zero": ["K2", "all"], "controls": ["K2"], "sham": ["K2"], "scale": {"K2": [0.5, 2.0]}}
        return fr.build_configs(toy_manifest(), list(run_langs), list(targets), list(panel), fam, off_target_label=off_target_label, **kw)

    def names(self, cfgs):
        return [c["name"] for c in cfgs]

    def test_names_groups_and_controls_come_from_the_manifest(self):
        cfgs = {c["name"]: c for c in self.build()}
        self.assertEqual(cfgs["zero_K2@aa"]["idx"].tolist(), [9, 8])
        self.assertEqual(cfgs["zero_all@aa"]["idx"].tolist(), [9, 8, 7, 6])
        self.assertEqual(cfgs["ctrl0_K2@aa"]["idx"].tolist(), [20, 21])
        self.assertEqual(cfgs["ctrl1_K2@aa"]["idx"].tolist(), [22, 23])
        self.assertEqual(cfgs["sham_K2@bb"]["idx"].tolist(), [5, 4])
        self.assertEqual(cfgs["scale0.5_K2@aa"]["kw"], {"alpha": 0.5})
        self.assertEqual(cfgs["scale2_K2@aa"]["kw"], {"alpha": 2.0})
        self.assertEqual(len(self.names(self.build())), len(set(self.names(self.build()))))

    def test_text_languages_target_for_all_and_panel_for_the_primary_zero_only(self):
        cfgs = {c["name"]: c for c in self.build(families={"zero": ["K2", "all"], "controls": ["K2"]})}
        self.assertEqual(cfgs["zero_K2@aa"]["langs"], {0, 2})                  # target aa (0) and panel {aa, cc} minus nothing off-target except cc: aa itself is the target
        self.assertEqual(cfgs["zero_K2@bb"]["langs"], {1, 0, 2})              # target bb plus panel aa and cc
        self.assertEqual(cfgs["zero_all@bb"]["langs"], {1})
        self.assertEqual(cfgs["ctrl0_K2@bb"]["langs"], {1})

    def test_panel_is_limited_to_the_languages_of_the_run(self):
        cfgs = {c["name"]: c for c in self.build(run_langs=("aa", "bb"), targets=("bb",), panel=("aa", "cc"))}
        self.assertEqual(cfgs["zero_K2@bb"]["langs"], {1, 0})

    def test_mean_and_resample_get_their_arguments_from_the_callables(self):
        calls = []
        def mean_fn(code, idx):
            calls.append(("mean", code)); return torch.full((idx.numel(),), 0.25)
        def pool_fn(code, idx):
            calls.append(("pool", code)); return torch.ones(3, idx.numel())
        cfgs = {c["name"]: c for c in self.build(targets=("cc",), families={"mean": ["K2"], "resample": ["K2"]}, mean_vals_fn=mean_fn, pool_fn=pool_fn)}
        self.assertTrue(torch.equal(cfgs["mean_K2@cc"]["kw"]["mean_vals"], torch.full((2,), 0.25)))
        self.assertEqual(cfgs["res_K2@cc"]["kw"]["pool"].shape, (3, 2))
        self.assertEqual(calls, [("mean", "cc"), ("pool", "cc")])
        with self.assertRaises(ValueError):
            self.build(targets=("cc",), families={"mean": ["K2"]})

    def test_the_number_of_control_draws_can_be_capped_per_size(self):
        cfgs = {c["name"]: c for c in self.build(families={"zero": ["K2"], "controls": ["K2", "all"], "n_controls": {"K2": 1}, "rescue": ["K2"]})}
        self.assertIn("ctrl0_K2@aa", cfgs); self.assertNotIn("ctrl1_K2@aa", cfgs)              # K2 has two draws in the manifest, one is kept
        self.assertIn("ctrl0_all@aa", cfgs)                                                    # a size without a cap keeps all its draws
        self.assertIn("rescuectrl0_K2@aa", cfgs); self.assertNotIn("rescuectrl1_K2@aa", cfgs)  # the cap also applies to the rescue controls
        with self.assertRaises(ValueError):
            self.build(families={"controls": ["K2"], "n_controls": {"K2": 0}})

    def test_unknown_label_target_or_family_is_an_error(self):
        for fam in ({"zero": ["K9"]}, {"bogus": ["K2"]}):
            with self.assertRaises(ValueError):
                self.build(families=fam)
        with self.assertRaises(ValueError):
            self.build(targets=("zz",))


class OtherLanguageMeans(unittest.TestCase):
    def test_mean_over_the_other_languages(self):
        mu = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [5.0, 0.0, 2.0]])      # [language, feature]
        got = fr.other_language_means({"mean_activation": mu}, 0, torch.tensor([0, 2]))
        self.assertTrue(torch.allclose(got, torch.tensor([(3.0 + 5.0) / 2, (1.0 + 2.0) / 2])))


class CollectPool(unittest.TestCase):
    def test_pool_rows_come_from_valid_positions_with_language_labels(self):
        model, sae = ToyModel(), ToySAE(d=D, F=F)
        data = corpus(sizes=(6, 5, 4))
        batches = p1.make_batches(data, BOS, "mask", 0, 256, batch_size=3)
        pool, lang = fr.collect_pool(model.lower, sae.encode, batches, rows_per_lang=7, seed=0)
        self.assertEqual(pool.shape, (21, F))
        self.assertEqual(lang.tolist(), [0] * 7 + [1] * 7 + [2] * 7)
        allrows = {}
        for l, sents in data.items():
            rows = []
            for s in sents:
                x = model.lower(torch.tensor([[BOS] + s]))[0, 1:]
                rows.append(sae.encode(x))
            allrows[l] = torch.cat(rows)
        for l in range(3):
            for r in pool[lang == l]:
                self.assertTrue(bool((torch.isclose(allrows[l], r[None, :].float(), rtol=1e-3, atol=1e-3).all(1)).any()))
        pool2, _ = fr.collect_pool(model.lower, sae.encode, batches, rows_per_lang=7, seed=0)
        self.assertTrue(torch.equal(pool, pool2))
        pool3, _ = fr.collect_pool(model.lower, sae.encode, batches, rows_per_lang=7, seed=1)
        self.assertFalse(torch.equal(pool, pool3))

    def test_a_language_with_too_few_positions_gives_all_of_them(self):
        model, sae = ToyModel(), ToySAE(d=D, F=F)
        data = {0: [[3, 4]], 1: corpus(sizes=(0, 5))[1]}
        batches = p1.make_batches(data, BOS, "mask", 0, 256, batch_size=3)
        pool, lang = fr.collect_pool(model.lower, sae.encode, batches, rows_per_lang=50, seed=0)
        self.assertEqual(int((lang == 0).sum()), 2)


def enc_letter(tokens, cand, answer=0):
    return {"tokens": list(tokens), "cand_ids": list(cand), "answer": answer, "format": "letter"}


def enc_cloze(tokens, options, answer=0):
    return {"tokens": list(tokens), "option_ids": [list(o) for o in options], "answer": answer, "format": "cloze"}


class QuantizedToy(ToyModel):
    """The 'quantized recipient': the same toy architecture with perturbed weights in both halves."""
    def __init__(self, seed=0, eps=0.15):
        super().__init__(seed)
        g = torch.Generator().manual_seed(99)
        self.E = self.E + eps * torch.randn_like(self.E, generator=g); self.M1 = self.M1 + eps * torch.randn_like(self.M1, generator=g)
        self.M2 = self.M2 + eps * torch.randn_like(self.M2, generator=g); self.U = self.U + eps * torch.randn_like(self.U, generator=g)


class Rescue(unittest.TestCase):
    def setUp(self):
        self.fp, self.q, self.sae = ToyModel(), QuantizedToy(), ToySAE(d=D, F=F)
        self.data = corpus()
        self.batches = p1.make_batches(self.data, BOS, "mask", 0, 256, batch_size=3)
        self.n_sent = {l: len(v) for l, v in self.data.items()}
        self.idx = torch.tensor([2, 5, 9, 17])
        self.cache = fr.build_donor_cache(self.fp.lower, self.sae.encode, self.batches)

    def run_q(self, configs):
        return fr.evaluate_precision(self.q.lower, self.q.upper, self.sae.encode, self.sae.W_dec, self.batches, self.n_sent, configs, donors=self.cache)

    def brute(self, sent, kind, idx=None):
        tokens = torch.tensor([[BOS] + sent]); valid = torch.zeros_like(tokens, dtype=torch.bool); valid[:, 1:] = True
        xq, xf = self.q.lower(tokens), self.fp.lower(tokens)
        aq, af = fi.encode_valid(xq, valid, self.sae.encode), fi.encode_valid(xf, valid, self.sae.encode)
        W = self.sae.W_dec
        if kind == "group":
            x2 = xq.clone(); x2[valid] = xq[valid] + (af[:, idx] - aq[:, idx]) @ W[idx]
        elif kind == "all":
            x2 = xq.clone(); x2[valid] = xq[valid] + (af - aq) @ W
        elif kind == "error":
            x2 = xq.clone(); x2[valid] = xf[valid] - (af - aq) @ W
        else:
            x2 = xq.clone(); x2[valid] = xf[valid]                      # BOS (position 0) is never patched
        s, n = p1.chunked_nll(self.q.upper(x2), tokens, valid)
        return float(s[0])

    def test_group_patch_equals_brute_force_and_differs_from_the_recipient(self):
        out = self.run_q([{"name": "rg", "kind": "donor", "idx": self.idx, "langs": {0, 1, 2}}])
        moved = 0
        for l, sents in self.data.items():
            for i, s in enumerate(sents):
                self.assertAlmostEqual(float(out["items"][(l, "rg")]["nll"][i]), self.brute(s, "group", self.idx), places=3)
                moved += abs(float(out["items"][(l, "rg")]["nll"][i]) - float(out["items"][(l, "none")]["nll"][i])) > 1e-6
        self.assertGreater(moved, 0)

    def test_bounds_equal_brute_force(self):
        allidx = torch.arange(F)
        cfgs = [{"name": "r_all", "kind": "donor", "idx": allidx, "langs": {0, 1, 2}}, {"name": "r_err", "kind": "donor_error", "idx": allidx, "langs": {0, 1, 2}},
                {"name": "r_full", "kind": "donor_resid", "idx": allidx, "langs": {0, 1, 2}}]
        out = self.run_q(cfgs)
        for l, sents in self.data.items():
            for i, s in enumerate(sents):
                for name, kind in (("r_all", "all"), ("r_err", "error"), ("r_full", "full")):
                    self.assertAlmostEqual(float(out["items"][(l, name)]["nll"][i]), self.brute(s, kind), places=3)

    def test_the_full_residual_patch_restores_the_donor_loss_when_the_upper_halves_agree(self):
        same_upper = QuantizedToy(eps=0.0)                              # recipient identical to the donor: every rescue is a no-op
        out = fr.evaluate_precision(same_upper.lower, same_upper.upper, self.sae.encode, self.sae.W_dec, self.batches, self.n_sent,
                                    [{"name": "r_full", "kind": "donor_resid", "idx": torch.arange(F), "langs": {0}}, {"name": "r_err", "kind": "donor_error", "idx": torch.arange(F), "langs": {0}},
                                     {"name": "r_g", "kind": "donor", "idx": self.idx, "langs": {0}}], donors=self.cache)
        for name in ("r_full", "r_err", "r_g"):
            self.assertTrue(torch.allclose(out["items"][(0, name)]["nll"], out["items"][(0, "none")]["nll"], atol=1e-4))

    def run_with(self, cache):
        return fr.evaluate_precision(self.q.lower, self.q.upper, self.sae.encode, self.sae.W_dec, self.batches, self.n_sent,
                                     [{"name": "rg", "kind": "donor", "idx": self.idx, "langs": {0, 1, 2}}], donors=cache)

    def test_a_cache_with_another_number_of_batches_is_rejected(self):
        other = p1.make_batches(corpus(seed=5, sizes=(2, 2, 2)), BOS, "mask", 0, 256, batch_size=2)
        with self.assertRaisesRegex(ValueError, "donor cache holds"):
            self.run_with(fr.build_donor_cache(self.fp.lower, self.sae.encode, other))
        extra = fr.build_donor_cache(self.fp.lower, self.sae.encode, self.batches + [self.batches[0]])           # the first batches agree: only the count differs
        with self.assertRaisesRegex(ValueError, "donor cache holds"):
            self.run_with(extra)

    def test_a_cache_with_another_number_of_rows_in_a_batch_is_rejected(self):
        altered = [dict(b) for b in self.batches]
        altered[0]["valid"] = self.batches[0]["valid"].clone(); altered[0]["valid"][0, 1] = False
        with self.assertRaisesRegex(ValueError, "does not match batch 0"):
            self.run_with(fr.build_donor_cache(self.fp.lower, self.sae.encode, altered))

    def test_the_donor_cache_returns_rows_and_sparse_activations_per_batch(self):
        b0 = self.batches[0]
        resid = self.fp.lower(b0["tokens"])
        self.assertTrue(torch.equal(self.cache.get_x(0, "cpu"), resid[b0["valid"]]))
        self.assertTrue(torch.allclose(self.cache.get_a(0, "cpu"), self.sae.encode(resid[b0["valid"]])))


class RescueConfigs(unittest.TestCase):
    def test_group_controls_and_bounds(self):
        man = toy_manifest()
        cfgs = {c["name"]: c for c in fr.build_configs(man, ["aa", "bb", "cc"], ["aa", "bb"], [], {"rescue": ["K2"], "rescue_bounds": [True]}, n_features=F)}
        self.assertEqual(cfgs["rescue_K2@aa"]["kind"], "donor"); self.assertEqual(cfgs["rescue_K2@aa"]["idx"].tolist(), [9, 8]); self.assertEqual(cfgs["rescue_K2@aa"]["langs"], {0})
        self.assertEqual(cfgs["rescuectrl0_K2@bb"]["idx"].tolist(), [30, 31]); self.assertEqual(cfgs["rescuectrl1_K2@bb"]["idx"].tolist(), [32, 33])
        self.assertEqual(cfgs["rescue_all"]["idx"].numel(), F); self.assertEqual(cfgs["rescue_all"]["kind"], "donor")
        self.assertEqual(cfgs["rescue_error"]["kind"], "donor_error"); self.assertEqual(cfgs["rescue_full"]["kind"], "donor_resid")
        for n in ("rescue_all", "rescue_error", "rescue_full"):
            self.assertEqual(cfgs[n]["langs"], {0, 1, 2})
        with self.assertRaises(ValueError):
            fr.build_configs(man, ["aa", "bb", "cc"], ["aa"], [], {"rescue_bounds": [True]})                          # needs n_features


class McBatches(unittest.TestCase):
    def test_letter_batches_are_sorted_padded_and_carry_last_positions_and_candidates(self):
        data = {0: [enc_letter([4, 5, 6], [3, 5, 7, 9]), enc_letter([4], [3, 5, 7, 9]), enc_letter([4, 5, 6, 7, 8], [2, 4, 6, 8])]}
        bs = fr.make_mc_batches(data, "letter", BOS, 0, max_tokens=64)
        self.assertEqual(len(bs), 1)
        b = bs[0]
        self.assertEqual(b["sent"].tolist(), [1, 0, 2])                                  # by length: 1, 3, 5 tokens
        self.assertEqual(b["tokens"].shape, (3, 6))
        self.assertEqual(b["last"].tolist(), [1, 3, 5])                                  # index of the last prompt token (BOS at 0)
        self.assertEqual(b["cand"].tolist(), [[3, 5, 7, 9], [3, 5, 7, 9], [2, 4, 6, 8]])
        self.assertTrue(bool((b["tokens"][:, 0] == BOS).all()))
        self.assertEqual(b["valid"].sum(1).tolist(), [1, 3, 5])                          # content positions only: no BOS, no padding
        self.assertFalse(bool(b["valid"][:, 0].any()))

    def test_max_tokens_splits_batches_and_an_oversized_item_gets_its_own(self):
        data = {0: [enc_letter([4] * 5, [3, 5, 7, 9]) for _ in range(4)] + [enc_letter([4] * 20, [3, 5, 7, 9])]}
        bs = fr.make_mc_batches(data, "letter", BOS, 0, max_tokens=14)                   # each padded row is 6 or 21 long
        self.assertEqual(sorted(len(b["sent"]) for b in bs), [1, 2, 2])
        self.assertEqual(sorted(i for b in bs for i in b["sent"].tolist()), [0, 1, 2, 3, 4])

    def test_cloze_rows_are_item_times_four_with_a_score_mask_on_the_option(self):
        data = {0: [enc_cloze([4, 5], [[6], [6, 7], [8], [9, 9, 9]]), enc_cloze([5], [[2], [3], [4], [5, 6]])]}
        bs = fr.make_mc_batches(data, "cloze", BOS, 0, max_tokens=64)
        self.assertEqual(fr.mc_n_rows(data, "cloze"), {0: 8})
        rows = {}
        for b in bs:
            for r, s in enumerate(b["sent"].tolist()):
                L = int(b["valid"][r].sum())
                rows[s] = (b["tokens"][r, 1:1 + L].tolist(), [int(i) for i in b["score"][r].nonzero().squeeze(1)])
        self.assertEqual(rows[0], ([4, 5, 6], [3]))                                      # item 0, option 0: score the single option token at position 3
        self.assertEqual(rows[1], ([4, 5, 6, 7], [3, 4]))
        self.assertEqual(rows[7], ([5, 5, 6], [2, 3]))
        self.assertTrue(all(len(v[1]) > 0 for v in rows.values()))


class McEngine(unittest.TestCase):
    def setUp(self):
        self.model, self.sae = ToyModel(), ToySAE(d=D, F=F)
        g = torch.Generator().manual_seed(3)
        self.letter = {0: [enc_letter(torch.randint(2, V, (int(torch.randint(3, 9, (1,), generator=g)),), generator=g).tolist(), [3, 5, 7, 9], answer=i % 4) for i in range(5)],
                       1: [enc_letter(torch.randint(2, V, (int(torch.randint(3, 9, (1,), generator=g)),), generator=g).tolist(), [2, 4, 6, 8], answer=1) for i in range(4)]}
        self.idx = torch.tensor([2, 5, 9, 17])

    def run_letter(self, configs):
        bs = fr.make_mc_batches(self.letter, "letter", BOS, 0, max_tokens=40)
        n = {l: len(v) for l, v in self.letter.items()}
        return fr.evaluate_precision(self.model.lower, self.model.upper, self.sae.encode, self.sae.W_dec, bs, n, configs, readout=fr.letter_readout)

    def brute_letter(self, enc, idx=None, kind=None):
        tokens = torch.tensor([[BOS] + enc["tokens"]]); valid = torch.zeros_like(tokens, dtype=torch.bool); valid[:, 1:] = True
        resid = self.model.lower(tokens)
        if kind:
            a = fi.encode_valid(resid, valid, self.sae.encode)
            resid, _ = fi.apply_edit(resid, valid, a, self.sae.W_dec, idx, kind)
        lp = torch.log_softmax(self.model.upper(resid)[0, -1].float(), -1)
        return lp[enc["cand_ids"]]

    def test_letter_baseline_and_edit_match_brute_force_per_item(self):
        out = self.run_letter([{"name": "zero", "kind": "ablate", "idx": self.idx, "langs": {0, 1}}])
        for l, items in self.letter.items():
            for i, e in enumerate(items):
                self.assertTrue(torch.allclose(out["items"][(l, "none")]["cand_logp"][i].float(), self.brute_letter(e), atol=1e-4))
                self.assertTrue(torch.allclose(out["items"][(l, "zero")]["cand_logp"][i].float(), self.brute_letter(e, self.idx, "ablate"), atol=1e-4))
        self.assertEqual(out["items"][(0, "none")]["cand_logp"].shape, (5, 4))

    def test_cloze_rows_score_the_option_tokens_only(self):
        g = torch.Generator().manual_seed(4)
        data = {0: [enc_cloze(torch.randint(2, V, (4,), generator=g).tolist(), [torch.randint(2, V, (int(torch.randint(1, 4, (1,), generator=g)),), generator=g).tolist() for _ in range(4)], answer=2) for _ in range(3)]}
        bs = fr.make_mc_batches(data, "cloze", BOS, 0, max_tokens=64)
        out = fr.evaluate_precision(self.model.lower, self.model.upper, self.sae.encode, self.sae.W_dec, bs, fr.mc_n_rows(data, "cloze"), [])
        for i, e in enumerate(data[0]):
            for j, o in enumerate(e["option_ids"]):
                tokens = torch.tensor([[BOS] + e["tokens"] + o])
                lp = torch.log_softmax(self.model.full(tokens)[0].float(), -1)
                want = -sum(float(lp[len(e["tokens"]) + k, o[k]]) for k in range(len(o)))
                self.assertAlmostEqual(float(out["items"][(0, "none")]["nll"][4 * i + j]), want, places=4)
                self.assertEqual(int(out["items"][(0, "none")]["ntok"][4 * i + j]), len(o))


class McItems(unittest.TestCase):
    ORIGIN = {"p0": "dev", "p1": "dev", "p2": "devtest", "p3": "devtest", "p4": "mixed"}

    def items(self, order=None):
        its = [{"passage_key": f"p{i % 5}", "question_number": i // 5 + 1, "options": ["a", "b", "c", "d"], "answer": 0, "passage": "x", "question": "q"} for i in range(20)]
        return its if order is None else [its[i] for i in order]

    def keys(self, items):
        return [(i["passage_key"], i["question_number"]) for i in items]

    def test_split_membership_and_cap(self):
        cal = fr.mc_items_for(self.items(), self.ORIGIN, "cal", 0, seed=0)
        ev = fr.mc_items_for(self.items(), self.ORIGIN, "eval", 0, seed=0)
        self.assertEqual({k[0] for k in self.keys(cal)}, {"p0", "p1"}); self.assertEqual({k[0] for k in self.keys(ev)}, {"p2", "p3"})
        self.assertEqual(len(cal), 8); self.assertEqual(len(fr.mc_items_for(self.items(), self.ORIGIN, "cal", 3, seed=0)), 3)

    def test_the_subset_does_not_depend_on_the_file_order_and_changes_with_the_seed(self):
        a = fr.mc_items_for(self.items(), self.ORIGIN, "eval", 3, seed=0)
        b = fr.mc_items_for(self.items(order=list(range(19, -1, -1))), self.ORIGIN, "eval", 3, seed=0)
        self.assertEqual(self.keys(a), self.keys(b))
        self.assertNotEqual(self.keys(a), self.keys(fr.mc_items_for(self.items(), self.ORIGIN, "eval", 3, seed=1)))


class TimeBudget(unittest.TestCase):
    def test_slurm_time_formats(self):
        self.assertEqual(fr.parse_slurm_time("3:12"), 192)
        self.assertEqual(fr.parse_slurm_time("1:02:03"), 3723)
        self.assertEqual(fr.parse_slurm_time("1-02:03:04"), 86400 + 7384)
        self.assertEqual(fr.parse_slurm_time("00:45"), 45)
        for bad in ("UNLIMITED", "INVALID", "NOT_SET", "", "abc", None, "45", "1:2:3:4"):
            self.assertIsNone(fr.parse_slurm_time(bad))

    def test_remaining_seconds_from_squeue_with_an_injected_runner(self):
        calls = []
        def runner(args):
            calls.append(args); return "2:30:00\n"
        self.assertEqual(fr.slurm_remaining_seconds({"SLURM_JOB_ID": "4242"}, runner), 9000)
        self.assertEqual(calls[0][:3], ["squeue", "-h", "-j"]); self.assertIn("4242", calls[0])
        self.assertIsNone(fr.slurm_remaining_seconds({}, runner))                                      # not a Slurm job: no limit known
        def boom(args):
            raise OSError("no squeue")
        self.assertIsNone(fr.slurm_remaining_seconds({"SLURM_JOB_ID": "4242"}, boom))                  # never fails the run

    def test_the_next_precision_starts_only_if_it_fits(self):
        self.assertTrue(fr.next_precision_fits(None, [3500]))                                          # no limit known
        self.assertTrue(fr.next_precision_fits(100, []))                                               # nothing measured yet: the first precision always starts
        self.assertTrue(fr.next_precision_fits(4000, [3500], margin=1.1))                              # 3850 needed
        self.assertFalse(fr.next_precision_fits(3800, [3500], margin=1.1))
        self.assertFalse(fr.next_precision_fits(3600, [3000, 3500], margin=1.1))                       # the slowest precision so far counts
        self.assertTrue(fr.next_precision_fits(3500, [3500], margin=1.0))                              # exactly enough
        self.assertFalse(fr.next_precision_fits(3499, [3500], margin=1.0))


class Gates(unittest.TestCase):
    def setUp(self):
        self.model, self.sae = ToyModel(), ToySAE(d=D, F=F)
        data = corpus(sizes=(4,))
        b = p1.make_batches(data, BOS, "mask", 0, 256, batch_size=4)[0]
        self.tokens, self.valid = b["tokens"], b["valid"]
        self.idx = torch.tensor([2, 5, 9, 17])

    def run_gates(self, full=None, upper=None):
        return fr.run_gates(full or self.model.full, self.model.lower, upper or self.model.upper, self.sae.encode, self.sae.W_dec, self.tokens, self.valid, self.idx)

    def test_all_gates_pass_on_a_consistent_model(self):
        g = self.run_gates()
        self.assertTrue(all(g.values()))
        self.assertEqual(set(g), {"split_equals_full", "scale_one_identity", "donor_self_identity", "sham_identity"})

    def test_a_split_that_differs_from_the_full_forward_fails_loudly(self):
        with self.assertRaisesRegex(fr.GateError, "split_equals_full"):
            self.run_gates(upper=lambda r: self.model.upper(r) * 1.0001)

    def test_a_noisy_edit_path_fails_exactly_the_matching_identity_gate(self):
        orig = fi.apply_edit
        for broken_kind, gate in (("scale", "scale_one_identity"), ("donor", "donor_self_identity"), ("sham", "sham_identity")):
            def broken(resid, valid, a, W, idx, kind, _k=broken_kind, **kw):
                out, st = orig(resid, valid, a, W, idx, kind, **kw)
                return (out + 1e-3 if kind == _k else out), st
            fi.apply_edit = broken
            try:
                with self.assertRaises(fr.GateError) as cm:
                    self.run_gates()
                self.assertIn(gate, str(cm.exception))
                for other in ("scale_one_identity", "donor_self_identity", "sham_identity"):
                    if other != gate:
                        self.assertNotIn(other, str(cm.exception))
            finally:
                fi.apply_edit = orig


if __name__ == "__main__":
    unittest.main()
