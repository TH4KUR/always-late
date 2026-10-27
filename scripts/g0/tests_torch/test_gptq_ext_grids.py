"""Task 3 of the GPTQ extension plan: the authors' symmetric per-output-channel grid and the matched RTN (same grid routine, FP32 arithmetic, no Hessian compensation) that GPTQ is compared with. The legacy
quantizer of the project (third_party/duan2026 `quantize_rtn_per_output_channel_tl`: scale = max|w| / (2^(b-1)-1), reduced over all dimensions except the last) is reproduced per tensor only to show how the grids differ
(reports/quantization_gptq_audit_2026-10-27.md section 3)."""
import sys
import unittest
from pathlib import Path

import torch

G0 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(G0))
from gptq_ext import grids, upstream  # noqa: E402


def randw(rows=6, cols=32, seed=0, dtype=torch.float32):
    return torch.randn(rows, cols, generator=torch.Generator().manual_seed(seed)).to(dtype)


class Scales(unittest.TestCase):
    def test_four_bit_scale_is_two_fifteenths_for_authors_and_one_seventh_for_legacy(self):
        w = torch.tensor([[-1.0, -0.5, 0.0, 0.5, 1.0]])
        scale, zero = grids.authors_grid(w, 4)
        self.assertAlmostEqual(float(scale), 2 / 15, places=6)
        self.assertEqual(float(zero), 8.0)
        self.assertAlmostEqual(float(grids.legacy_scale(w.t(), 4)), 1 / 7, places=6)      # legacy layout [in, out]: the same row as ONE output channel

    def test_authors_grid_has_one_scale_per_output_row(self):
        scale, zero = grids.authors_grid(randw(rows=5), 6)
        self.assertEqual(tuple(scale.shape), (5, 1))
        self.assertEqual(tuple(zero.shape), (5, 1))
        self.assertEqual(scale.dtype, torch.float32)

    def test_an_all_zero_row_gets_a_usable_scale(self):
        w = randw(rows=3); w[1] = 0
        out = grids.rtn_matched(w, 4)
        self.assertTrue(torch.equal(out[1], torch.zeros_like(out[1])))
        self.assertTrue(torch.isfinite(out).all())


class MatchedRtn(unittest.TestCase):
    def test_matched_rtn_equals_gptq_with_identity_hessian(self):
        core = upstream.load()
        for bits in (4, 6, 8):
            w = randw(rows=8, cols=48, seed=bits)
            layer = torch.nn.Linear(48, 8, bias=False); layer.weight.data = w.clone()
            gp = core.GPTQ(layer); gp.quantizer = core.Quantizer(); gp.quantizer.configure(bits, perchannel=True, sym=True, mse=False); gp.H = torch.eye(48)
            gp.fasterquant(blocksize=128, percdamp=0.01, groupsize=-1, actorder=False)
            self.assertTrue(torch.equal(layer.weight.data, grids.rtn_matched(w, bits)), bits)

    def test_matched_rtn_levels_are_within_the_grid(self):
        for bits in (4, 6):
            w = randw(rows=7, cols=64, seed=3)
            out = grids.rtn_matched(w, bits)
            scale, zero = grids.authors_grid(w, bits)
            k = out / scale + zero
            self.assertTrue(torch.allclose(k, k.round(), atol=1e-4))
            self.assertTrue(((k.round() >= 0) & (k.round() <= 2 ** bits - 1)).all())
            self.assertTrue(all(len(torch.unique(r)) <= 2 ** bits for r in out))

    def test_matched_rtn_is_deterministic_and_dtype_preserving(self):
        w = randw(dtype=torch.bfloat16)
        a, b = grids.rtn_matched(w, 4), grids.rtn_matched(w, 4)
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(a.dtype, torch.bfloat16)
        self.assertEqual(a.shape, w.shape)

    def test_matched_rtn_is_computed_in_float32_even_for_bf16_weights(self):
        w = randw(dtype=torch.bfloat16, seed=5)
        expected = grids.rtn_matched(w.float(), 6).to(torch.bfloat16)
        self.assertTrue(torch.equal(grids.rtn_matched(w, 6), expected))

    def test_rows_are_independent(self):
        w = randw(rows=4, seed=7); w2 = w.clone(); w2[2] *= 3.0
        a, b = grids.rtn_matched(w, 4), grids.rtn_matched(w2, 4)
        self.assertTrue(torch.equal(a[[0, 1, 3]], b[[0, 1, 3]]))
        self.assertFalse(torch.equal(a[2], b[2]))

    def test_dead_columns_are_zeroed_like_upstream_and_the_scale_ignores_the_zeroing(self):
        w = randw(rows=4, cols=16, seed=9); dead = torch.zeros(16, dtype=torch.bool); dead[[3, 11]] = True
        out = grids.rtn_matched(w, 4, dead=dead)
        self.assertTrue((out[:, dead] == 0).all())
        keep = ~dead
        self.assertTrue(torch.equal(out[:, keep], grids.rtn_matched(w, 4)[:, keep]))

    def test_matched_rtn_error_is_at_most_half_a_step(self):
        w = randw(rows=5, cols=40, seed=11)
        scale, _ = grids.authors_grid(w, 6)
        err = (grids.rtn_matched(w, 6) - w).abs()
        inside = w.abs() <= (scale * 31)             # values beyond the clamp range (the asymmetric top level) can exceed half a step
        self.assertTrue((err[inside] <= scale.expand_as(w)[inside] * 0.5 + 1e-6).all())


class LegacyContrast(unittest.TestCase):
    def test_legacy_and_matched_grids_differ_for_the_same_symmetric_label(self):
        w = randw(rows=4, cols=64, seed=13)
        legacy = grids.legacy_rtn(w, 4)
        matched = grids.rtn_matched(w, 4)
        self.assertFalse(torch.equal(legacy, matched))

    def test_legacy_rtn_reduces_over_all_dimensions_except_the_last(self):
        w = randw(rows=4, cols=6, seed=15)            # legacy layout [in, out]: one scale per LAST index
        out = grids.legacy_rtn(w, 4)
        scale = w.abs().amax(dim=0, keepdim=True) / 7
        self.assertTrue(torch.equal(out, (torch.round(w / scale).clamp(-8, 7) * scale).to(w.dtype)))


if __name__ == "__main__":
    unittest.main()
