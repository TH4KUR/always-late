"""Task 1 of the Gemma-2-9B three-arm plan (docs/superpowers/plans/2026-10-28-gemma9b-three-arm-confirmation.md): the canonical adapter and the sequential calibrator of the GPTQ extension were written and tested on a toy
whose attention inner width equals d_model. Gemma-2-9B (16 heads x d_head 256 = 4096 against d_model 3584, 8 key/value heads) and Gemma-2-2B (8 x 256 = 2048 against 2304) do not have that property. These tests run the
whole calibrator test suite of `test_gptq_ext_calibrate.WithModel` on a toy with the same property (4 heads x d_head 16 = 64 against d_model 32, 2 key/value heads), plus the round trip of every target and a check that GPTQ
beats the matched RTN on the objective it optimises. Model tests need TransformerLens (~/.venvs/anlp-tl) and are skipped elsewhere."""
import sys
import unittest
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
G0 = HERE.parent
sys.path.insert(0, str(G0)); sys.path.insert(0, str(HERE))
import test_gptq_ext_calibrate as base  # noqa: E402  (the module, not its classes: discovery must not collect them twice)
from gptq_ext import calibrate as cb, calibration as cal, canonical  # noqa: E402
from gptq_ext_tiny import DIMS, HAVE_TL, tiny_gemma2  # noqa: E402

DIMS_9B = {**DIMS, "d_head": 16}                      # 4 heads x 16 = 64 != d_model 32: the layout property of Gemma-2-2B and -9B


@unittest.skipUnless(HAVE_TL, "needs TransformerLens (~/.venvs/anlp-tl)")
class Layout(unittest.TestCase):
    def test_the_toy_has_the_9b_layout_property(self):
        m = tiny_gemma2(dims=DIMS_9B)
        self.assertNotEqual(m.cfg.n_heads * m.cfg.d_head, m.cfg.d_model)
        self.assertEqual((m.cfg.n_heads, m.cfg.n_key_value_heads, m.cfg.d_head), (4, 2, 16))

    def test_the_default_toy_is_unchanged(self):
        m = tiny_gemma2()
        self.assertEqual((m.cfg.d_head, m.cfg.n_heads, m.cfg.d_model), (DIMS["d_head"], DIMS["n_heads"], DIMS["d_model"]))

    def test_canonical_round_trip_for_every_target_when_heads_times_dhead_differs_from_dmodel(self):
        m = tiny_gemma2(dims=DIMS_9B)
        targets = canonical.targets(m)
        self.assertEqual(len(targets), 7 * 2)                                       # W_Q, _W_K, _W_V, W_O, W_in, W_gate, W_out in two blocks
        for name, p in targets:
            lin = canonical.to_linear(name, p.data)
            self.assertTrue(torch.equal(canonical.from_linear(name, lin, p.data), p.data), name)
        shapes = {n.split(".")[-1]: tuple(canonical.to_linear(n, p.data).shape) for n, p in targets}
        self.assertEqual(shapes, {"W_Q": (64, 32), "_W_K": (32, 32), "_W_V": (32, 32), "W_O": (32, 64), "W_in": (64, 32), "W_gate": (64, 32), "W_out": (32, 64)})


@unittest.skipUnless(HAVE_TL, "needs TransformerLens (~/.venvs/anlp-tl)")
class CalibratorOnThe9bLayout(base.WithModel):
    """Every model test of the calibrator suite (identity control, matched RTN equal to the direct rounding, stage inputs after the quantized earlier stages, determinism, padding, model left as found) on the 9B layout."""

    @classmethod
    def setUpClass(cls):
        cls._original_factory, cls._original_dims = base.tiny_gemma2, base.DIMS
        original = cls._original_factory
        base.tiny_gemma2 = lambda *a, **k: original(*a, dims=DIMS_9B, **k)
        base.DIMS = DIMS_9B                                                           # the base tests read the expected widths from DIMS (the stage input of W_O is heads x d_head wide)
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        base.tiny_gemma2, base.DIMS = cls._original_factory, cls._original_dims

    def test_gptq_beats_the_matched_rtn_on_the_objective_it_optimises_at_4_bits(self):
        rows = {}
        observe = lambda block, stage, name, r: rows.setdefault(name, r.clone()) if block == 0 and stage == "qkv" else None
        _, gptq, _ = self.run_cal("gptq", observer=observe)
        _, rtn, _ = self.run_cal("rtn_matched")
        ref = base.tiny_gemma2()
        err = {"gptq": 0.0, "rtn_matched": 0.0}
        for name in ("blocks.0.attn.W_Q", "blocks.0.attn._W_K", "blocks.0.attn._W_V"):
            w = canonical.to_linear(name, dict(ref.named_parameters())[name].data)
            for method, art in (("gptq", gptq), ("rtn_matched", rtn)):
                d = canonical.to_linear(name, art[name]) - w
                err[method] += float(((rows[name] @ d.T) ** 2).sum())              # block 0 attention input is the same for both methods: nothing is quantized before it
        self.assertLess(err["gptq"], err["rtn_matched"])


if __name__ == "__main__":
    unittest.main()
