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
