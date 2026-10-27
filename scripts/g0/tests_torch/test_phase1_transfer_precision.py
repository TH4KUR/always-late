import hashlib
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phase1_transfer_precision as pp

LAYER = 1                                                                      # the SAE site: blocks 0 and 1 are the prefix, blocks 2 and 3 the suffix of the fake model


class Cfg:
    def __init__(self):
        self.dtype = torch.bfloat16


class Hook(nn.Module):
    def forward(self, x):
        return x


class Norm(nn.Module):
    """Like TransformerLens' RMSNorm: computed in float32 when cfg.dtype is lower precision, returned in cfg.dtype."""
    def __init__(self, cfg, d):
        super().__init__(); self.cfg = cfg; self.w = nn.Parameter(torch.ones(d, dtype=cfg.dtype)); self.hook_normalized = Hook()

    def forward(self, x):
        x = x.to(torch.float32) if self.cfg.dtype not in (torch.float32, torch.float64) else x
        return self.hook_normalized((x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)).to(self.cfg.dtype)) * self.w


class Attn(nn.Module):
    def __init__(self, cfg, d):
        super().__init__(); self.cfg = cfg; self.W_Q = nn.Parameter(torch.zeros(d, d, dtype=cfg.dtype)); self.hook_q = Hook(); self.register_buffer("rotary", torch.arange(d, dtype=torch.float32))

    def forward(self, h):
        return self.hook_q(h @ self.W_Q)


class Block(nn.Module):
    def __init__(self, cfg, d):
        super().__init__(); self.ln1 = Norm(cfg, d); self.attn = Attn(cfg, d)

    def forward(self, x):
        return x + self.attn(self.ln1(x))                                   # a float32 residual plus a lower-precision output promotes to float32, as in the real blocks


class FakeModel(nn.Module):
    def __init__(self, d=8, n_layers=4):
        super().__init__(); self.cfg = Cfg(); self.blocks = nn.ModuleList([Block(self.cfg, d) for _ in range(n_layers)]); self.ln_final = Norm(self.cfg, d)
        g = torch.Generator().manual_seed(0)
        for b in self.blocks:
            b.attn.W_Q.data = (torch.randn(d, d, generator=g) * 0.3).to(torch.bfloat16)

    def forward(self, resid, start_at_layer=0, stop_at_layer=None):
        for b in self.blocks[start_at_layer:stop_at_layer]:
            resid = b(resid)
        return resid


class PromoteDemote(unittest.TestCase):
    def setUp(self):
        self.m = FakeModel(); self.snap = pp.snapshot(self.m, LAYER)

    def test_the_snapshot_covers_the_parameters_and_floating_buffers_of_the_suffix_blocks_and_the_final_norm_only(self):
        names = set(self.snap); self.assertIn("blocks.2.attn.W_Q", names); self.assertIn("blocks.3.ln1.w", names); self.assertIn("ln_final.w", names); self.assertIn("blocks.2.attn.rotary", names)
        self.assertFalse(any(n.startswith(("blocks.0.", "blocks.1.")) for n in names)); self.assertEqual(self.snap["blocks.2.attn.W_Q"], torch.bfloat16); self.assertEqual(self.snap["blocks.2.attn.rotary"], torch.float32)

    def test_promotion_makes_every_suffix_tensor_float32_with_the_same_values_and_leaves_the_prefix_alone(self):
        before = {n: t.detach().clone() for n, t in pp._tensors(self.m, LAYER)}; prefix = self.m.blocks[0].attn.W_Q.detach().clone(); pp.promote_suffix(self.m, LAYER)
        for n, t in pp._tensors(self.m, LAYER):
            self.assertEqual(t.dtype, torch.float32, n); self.assertTrue(torch.equal(t, before[n].to(torch.float32)), n)
        self.assertEqual(self.m.blocks[0].attn.W_Q.dtype, torch.bfloat16); self.assertTrue(torch.equal(self.m.blocks[0].attn.W_Q, prefix))

    def test_demotion_restores_the_original_dtype_of_every_tensor_exactly_including_a_float32_buffer(self):
        before = {n: t.detach().clone() for n, t in pp._tensors(self.m, LAYER)}; pp.promote_suffix(self.m, LAYER); pp.demote_suffix(self.m, LAYER, self.snap)
        for n, t in pp._tensors(self.m, LAYER):
            self.assertEqual(t.dtype, self.snap[n], n); self.assertTrue(torch.equal(t, before[n]), n)
        self.assertEqual(self.m.blocks[2].attn.rotary.dtype, torch.float32)

    def test_a_tensor_missing_from_the_snapshot_is_an_error_not_a_guess(self):
        pp.promote_suffix(self.m, LAYER); snap = dict(self.snap); del snap["ln_final.w"]
        with self.assertRaisesRegex(KeyError, "ln_final.w"):
            pp.demote_suffix(self.m, LAYER, snap)
        self.assertTrue(all(t.dtype == torch.float32 for _, t in pp._tensors(self.m, LAYER)))                              # nothing was demoted before the error


class Hashes(unittest.TestCase):
    def setUp(self):
        self.m = FakeModel(); self.snap = pp.snapshot(self.m, LAYER)

    def test_the_hash_is_the_sha256_of_the_named_bytes_and_is_the_same_before_promotion_and_after_the_round_trip(self):
        h = pp.suffix_hashes(self.m, LAYER); self.assertEqual(set(h), set(self.snap)); t = self.m.blocks[3].ln1.w.detach()
        self.assertEqual(h["blocks.3.ln1.w"], hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
        pp.promote_suffix(self.m, LAYER); self.assertEqual(pp.suffix_hashes(self.m, LAYER, self.snap), h); pp.demote_suffix(self.m, LAYER, self.snap); self.assertEqual(pp.suffix_hashes(self.m, LAYER), h)

    def test_one_changed_value_changes_the_hash_of_that_tensor_only_and_the_prefix_is_not_hashed(self):
        h = pp.suffix_hashes(self.m, LAYER); self.m.blocks[2].attn.W_Q.data[0, 0] += 1.0; h2 = pp.suffix_hashes(self.m, LAYER)
        self.assertEqual([n for n in h if h[n] != h2[n]], ["blocks.2.attn.W_Q"]); self.m.blocks[0].attn.W_Q.data[0, 0] += 1.0; self.assertEqual(pp.suffix_hashes(self.m, LAYER), h2)

    def test_the_round_trip_check_passes_for_exact_promotion_and_fails_when_a_float32_value_is_not_representable_in_bf16(self):
        h = pp.suffix_hashes(self.m, LAYER); pp.promote_suffix(self.m, LAYER); self.assertTrue(pp.roundtrip_equal(self.m, LAYER, self.snap, h))
        self.m.blocks[2].attn.W_Q.data[0, 0] = 1.0 + 2 ** -12; self.assertFalse(pp.roundtrip_equal(self.m, LAYER, self.snap, h))                       # not a bf16 value: demotion would round it


class Switch(unittest.TestCase):
    def test_the_state_change_runs_in_the_original_precision_between_a_demotion_and_a_promotion_and_the_values_survive_exactly(self):
        m = FakeModel(); snap = pp.snapshot(m, LAYER); seen = []
        def setup(label):
            seen.append((label, m.blocks[2].attn.W_Q.dtype, m.ln_final.w.dtype))                                    # the dtypes while the weights are set
            m.blocks[2].attn.W_Q.data = (torch.round(m.blocks[2].attn.W_Q.data.float() * 4) / 4).to(m.blocks[2].attn.W_Q.dtype)               # a coarse-grid 'quantization' in the stored precision
        pp.promote_suffix(m, LAYER); wrapped = pp.fp32_setup(setup, m, LAYER, snap); wrapped("INT4")
        self.assertEqual(seen, [("INT4", torch.bfloat16, torch.bfloat16)]); self.assertEqual(m.blocks[2].attn.W_Q.dtype, torch.float32); w = m.blocks[2].attn.W_Q.detach()
        self.assertTrue(torch.equal(w, w.to(torch.bfloat16).to(torch.float32))); self.assertTrue(torch.equal(w, torch.round(w * 4) / 4))

    def test_an_exception_in_the_state_change_still_leaves_the_suffix_promoted(self):
        m = FakeModel(); snap = pp.snapshot(m, LAYER); pp.promote_suffix(m, LAYER)
        def bad(label):
            raise RuntimeError("boom")
        with self.assertRaisesRegex(RuntimeError, "boom"):
            pp.fp32_setup(bad, m, LAYER, snap)("INT4")
        self.assertEqual(m.blocks[2].attn.W_Q.dtype, torch.float32)


class SwitchRecord(unittest.TestCase):
    def test_every_state_change_appends_the_hash_check_with_the_label_the_tensor_count_and_a_digest_of_the_stored_values(self):
        m = FakeModel(); snap = pp.snapshot(m, LAYER); orig = m.blocks[2].attn.W_Q.data.clone(); pp.promote_suffix(m, LAYER); rec = []
        def setup(label):
            m.blocks[2].attn.W_Q.data = (orig.float() * (2.0 if label == "INT4" else 1.0)).to(torch.bfloat16)                  # restore, then 'quantize'
        
        w = pp.fp32_setup(setup, m, LAYER, snap, record=rec); w("INT16"); w("INT4"); self.assertEqual([r["label"] for r in rec], ["INT16", "INT4"]); self.assertTrue(all(r["ok"] for r in rec)); self.assertEqual(rec[0]["n_tensors"], len(snap)); self.assertNotEqual(rec[0]["digest"], rec[1]["digest"])
        w("INT16"); self.assertNotEqual(rec[2]["digest"], rec[1]["digest"])

    def test_a_promotion_that_does_not_convert_back_to_the_stored_values_raises_the_g1_error_and_is_recorded_as_failed(self):
        m = FakeModel(); snap = pp.snapshot(m, LAYER); pp.promote_suffix(m, LAYER); rec = []; real = pp.promote_suffix
        def corrupt(model, layer):
            real(model, layer); model.blocks[2].attn.W_Q.data[0, 0] += 1e-3                                        # a value that is not a bf16 value
        pp.promote_suffix = corrupt
        try:
            with self.assertRaisesRegex(RuntimeError, "G1"):
                pp.fp32_setup(lambda label: None, m, LAYER, snap, record=rec)("INT4")
        finally:
            pp.promote_suffix = real
        self.assertEqual(rec[0]["ok"], False)

    def test_without_a_record_nothing_is_hashed_and_the_return_value_of_the_setup_is_passed_on(self):
        m = FakeModel(); snap = pp.snapshot(m, LAYER); pp.promote_suffix(m, LAYER); self.assertEqual(pp.fp32_setup(lambda label: ("done", label), m, LAYER, snap)("INT4"), ("done", "INT4"))


class ModeSuffix(unittest.TestCase):
    def setUp(self):
        self.m = FakeModel(); self.snap = pp.snapshot(self.m, LAYER); g = torch.Generator().manual_seed(5); self.x = (torch.randn(2, 3, 8, generator=g) * 4.0 + 1.0 + 2 ** -10).float()
        self.upper = lambda r: self.m(r, start_at_layer=LAYER + 1, stop_at_layer=len(self.m.blocks))

    def reference(self):
        """The suffix computed in float64 with the (bf16-valued) weights."""
        r = self.x.double()
        for b in self.m.blocks[LAYER + 1:]:
            w1, wq = b.ln1.w.detach().double(), b.attn.W_Q.detach().double(); n = r / torch.sqrt(r.pow(2).mean(-1, keepdim=True) + 1e-6) * w1; r = r + n @ wq
        return r

    def test_the_fp32_mode_agrees_with_a_float64_reference_to_float32_rounding_and_the_bf16_mode_does_not(self):
        sfx = pp.mode_suffix(self.m, LAYER, self.upper, "bf16"); bf = sfx(self.x)
        pp.promote_suffix(self.m, LAYER); sf32 = pp.mode_suffix(self.m, LAYER, self.upper, "fp32"); f32 = sf32(self.x); ref = self.reference()
        self.assertLess(float((f32.double() - ref).abs().max()), 1e-4); self.assertGreater(float((bf.double() - ref).abs().max()), 1e-3)

    def test_the_dtype_of_the_configuration_is_switched_only_for_the_call_and_restored_also_after_an_exception(self):
        pp.promote_suffix(self.m, LAYER); seen = []
        def upper(r):
            seen.append(self.m.cfg.dtype); raise ValueError("fail")
        sfx = pp.mode_suffix(self.m, LAYER, upper, "fp32")
        with self.assertRaises(ValueError):
            sfx(self.x)
        self.assertEqual(seen, [torch.float32]); self.assertEqual(self.m.cfg.dtype, torch.bfloat16)

    def test_every_distinct_configuration_object_of_the_suffix_modules_is_switched(self):
        for b in self.m.blocks[LAYER + 1:]:
            b.cfg = Cfg(); b.ln1.cfg = Cfg(); b.attn.cfg = Cfg()                                                        # blocks, norms and attentions with their own copies of the configuration
        seen = []
        def upper(r):
            seen.append([c.dtype for b in self.m.blocks[LAYER + 1:] for c in (b.cfg, b.ln1.cfg, b.attn.cfg)] + [self.m.ln_final.cfg.dtype]); return r
        pp.mode_suffix(self.m, LAYER, upper, "fp32")(self.x); self.assertEqual(seen, [[torch.float32] * 7]); self.assertTrue(all(c.dtype == torch.bfloat16 for b in self.m.blocks[LAYER + 1:] for c in (b.cfg, b.ln1.cfg, b.attn.cfg)))

    def test_an_unknown_mode_is_refused(self):
        with self.assertRaisesRegex(ValueError, "mode"):
            pp.mode_suffix(self.m, LAYER, self.upper, "fp16")


class Probe(unittest.TestCase):
    def setUp(self):
        self.m = FakeModel(); self.x = torch.ones(1, 2, 8)
        self.paths = {"normalized": "blocks.2.ln1.hook_normalized", "q": "blocks.2.attn.hook_q"}
        self.run = lambda: self.m(self.x, start_at_layer=LAYER + 1, stop_at_layer=len(self.m.blocks))

    def test_the_probe_records_the_dtype_of_the_tensors_at_the_hooked_modules_in_each_mode(self):
        bf = pp.dtype_probe(self.m, self.paths, self.run); self.assertEqual(bf, {"normalized": "torch.bfloat16", "q": "torch.bfloat16"})
        pp.promote_suffix(self.m, LAYER); self.m.cfg.dtype = torch.float32
        try:
            f32 = pp.dtype_probe(self.m, self.paths, self.run)
        finally:
            self.m.cfg.dtype = torch.bfloat16
        self.assertEqual(f32, {"normalized": "torch.float32", "q": "torch.float32"})

    def test_the_probe_removes_its_hooks_and_the_assertion_names_the_tensor_that_is_not_float32(self):
        pp.dtype_probe(self.m, self.paths, self.run); self.assertEqual(len(self.m.blocks[2].attn.hook_q._forward_hooks), 0)
        pp.assert_dtypes({"normalized": "torch.float32", "q": "torch.float32"}, "fp32")
        with self.assertRaisesRegex(AssertionError, "q"):
            pp.assert_dtypes({"normalized": "torch.float32", "q": "torch.bfloat16"}, "fp32")
        pp.assert_dtypes({"normalized": "torch.bfloat16", "q": "torch.float32"}, "bf16")                                    # the bf16 mode only records what it observes


if __name__ == "__main__":
    unittest.main()
