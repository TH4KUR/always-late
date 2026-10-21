import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phase1_func_intervene as fi


class ToySAE:
    """JumpReLU SAE with a decoder and a decoder bias, as in the real one (shapes small enough to check by hand)."""
    def __init__(self, d=8, F=32, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.W_enc = torch.randn(d, F, generator=g) * 0.5
        self.b_enc = torch.randn(F, generator=g) * 0.1
        self.theta = torch.rand(F, generator=g) * 0.3
        self.W_dec = torch.randn(F, d, generator=g)
        self.b_dec = torch.randn(d, generator=g) * 0.1

    def encode(self, x):
        pre = x.float() @ self.W_enc + self.b_enc
        return pre * (pre > self.theta)

    def decode(self, a):
        return a @ self.W_dec + self.b_dec


def make_batch(B=3, L=6, d=8, seed=1, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    resid = (torch.randn(B, L, d, generator=g) * 2).to(dtype)
    valid = torch.ones(B, L, dtype=torch.bool)
    valid[:, 0] = False                         # BOS
    valid[1, -2:] = False                       # padding of the second row
    return resid, valid


class Edits(unittest.TestCase):
    def setUp(self):
        self.sae = ToySAE()
        self.resid, self.valid = make_batch()
        self.a = fi.encode_valid(self.resid, self.valid, self.sae.encode)
        self.idx = torch.tensor([3, 7, 11])

    def edit(self, kind, **kw):
        return fi.apply_edit(self.resid, self.valid, self.a, self.sae.W_dec, self.idx, kind, **kw)

    def test_encode_valid_shape_and_rows(self):
        self.assertEqual(self.a.shape, (int(self.valid.sum()), 32))
        self.assertTrue(torch.equal(self.a, self.sae.encode(self.resid[self.valid])))

    def test_identity_scale_one_is_bitwise(self):
        for dtype in (torch.float32, torch.bfloat16):
            resid, valid = make_batch(dtype=dtype)
            a = fi.encode_valid(resid, valid, self.sae.encode)
            out, _ = fi.apply_edit(resid, valid, a, self.sae.W_dec, self.idx, "scale", alpha=1.0)
            self.assertTrue(torch.equal(out, resid))
            self.assertEqual(out.dtype, dtype)

    def test_error_term_is_preserved_when_every_feature_is_ablated(self):
        allidx = torch.arange(32)
        out, _ = fi.apply_edit(self.resid, self.valid, self.a, self.sae.W_dec, allidx, "ablate")
        x = self.resid[self.valid]
        xhat = self.sae.decode(self.a)
        err = x - xhat
        # all features removed: what is left is the decoder bias plus the SAE's own error term
        self.assertTrue(torch.allclose(out[self.valid], self.sae.b_dec + err, atol=1e-5))

    def test_change_is_confined_to_the_decoder_span_of_the_group(self):
        out, _ = self.edit("ablate")
        delta = (out - self.resid)[self.valid]                      # [N, d]
        B = self.sae.W_dec[self.idx]                                # [K, d], K < d
        coef = torch.linalg.lstsq(B.T, delta.T).solution            # delta^T ~ B^T c
        resid_after_projection = delta.T - B.T @ coef
        self.assertLess(float(resid_after_projection.abs().max()), 1e-5)
        self.assertGreater(float(delta.abs().max()), 1e-3)          # and the edit is not a no-op

    def test_ablation_removes_exactly_the_group_contribution(self):
        out, _ = self.edit("ablate")
        expect = self.resid[self.valid] - self.a[:, self.idx] @ self.sae.W_dec[self.idx]
        self.assertTrue(torch.allclose(out[self.valid], expect, atol=1e-5))

    def test_invalid_positions_are_bit_identical(self):
        out, _ = self.edit("ablate")
        self.assertTrue(torch.equal(out[~self.valid], self.resid[~self.valid]))        # BOS and padding
        self.assertTrue(torch.equal(out[:, 0], self.resid[:, 0]))

    def test_input_is_not_modified_in_place(self):
        before = self.resid.clone()
        self.edit("ablate")
        self.assertTrue(torch.equal(self.resid, before))

    def test_scale_zero_equals_ablate_and_scale_matches_formula(self):
        o0, _ = self.edit("scale", alpha=0.0)
        oa, _ = self.edit("ablate")
        self.assertTrue(torch.equal(o0, oa))
        o2, _ = self.edit("scale", alpha=2.0)
        expect = self.resid[self.valid] + self.a[:, self.idx] @ self.sae.W_dec[self.idx]
        self.assertTrue(torch.allclose(o2[self.valid], expect, atol=1e-5))

    def test_resample_uses_pool_rows_only_and_is_seeded(self):
        pool = torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 0.5]])
        v1 = fi.resample_values(pool, 50, torch.Generator().manual_seed(5))
        v2 = fi.resample_values(pool, 50, torch.Generator().manual_seed(5))
        self.assertTrue(torch.equal(v1, v2))
        for row in v1:
            self.assertTrue(any(torch.equal(row, p) for p in pool))
        self.assertTrue(any(torch.equal(r, pool[0]) for r in v1) and any(torch.equal(r, pool[1]) for r in v1))

    def test_resample_edit_with_a_single_pool_row(self):
        pool = torch.tensor([[1.0, 2.0, 3.0]])
        out, _ = self.edit("resample", pool=pool, generator=torch.Generator().manual_seed(0))
        expect = self.resid[self.valid] + (pool - self.a[:, self.idx]) @ self.sae.W_dec[self.idx]
        self.assertTrue(torch.allclose(out[self.valid], expect, atol=1e-5))

    def test_mean_ablation_sets_the_group_to_the_given_values(self):
        mean_vals = torch.tensor([0.5, 0.0, 2.0])
        out, _ = self.edit("mean", mean_vals=mean_vals)
        expect = self.resid[self.valid] + (mean_vals[None, :] - self.a[:, self.idx]) @ self.sae.W_dec[self.idx]
        self.assertTrue(torch.allclose(out[self.valid], expect, atol=1e-5))
        with self.assertRaises(ValueError):
            self.edit("mean")
        with self.assertRaises(ValueError):
            self.edit("mean", mean_vals=torch.zeros(2))

    def test_donor_equal_to_recipient_is_identity_and_other_donor_moves_by_the_difference(self):
        out, _ = fi.apply_edit(self.resid, self.valid, self.a, self.sae.W_dec, self.idx, "donor", a_donor=self.a)
        self.assertTrue(torch.equal(out, self.resid))
        donor = self.a.clone(); donor[:, self.idx] += 1.5
        out2, _ = fi.apply_edit(self.resid, self.valid, self.a, self.sae.W_dec, self.idx, "donor", a_donor=donor)
        expect = self.resid[self.valid] + 1.5 * self.sae.W_dec[self.idx].sum(0)
        self.assertTrue(torch.allclose(out2[self.valid], expect, atol=1e-4))

    def test_donor_only_touches_the_group_columns(self):
        donor = self.a + 10.0                                       # every column differs, only the group's may be used
        out, _ = fi.apply_edit(self.resid, self.valid, self.a, self.sae.W_dec, self.idx, "donor", a_donor=donor)
        expect = self.resid[self.valid] + 10.0 * self.sae.W_dec[self.idx].sum(0)
        self.assertTrue(torch.allclose(out[self.valid], expect, atol=1e-4))

    def donor_setup(self):
        g = torch.Generator().manual_seed(11)
        x_d = self.resid[self.valid] + torch.randn(int(self.valid.sum()), 8, generator=g) * 0.7          # the donor model's residual at the same positions
        a_d = self.sae.encode(x_d)
        return x_d, a_d

    def test_full_residual_patch_replaces_the_valid_rows_and_leaves_the_rest(self):
        x_d, a_d = self.donor_setup()
        out, st = fi.apply_edit(self.resid, self.valid, self.a, self.sae.W_dec, torch.arange(32), "donor_resid", x_donor=x_d)
        self.assertTrue(torch.allclose(out[self.valid], x_d))
        self.assertTrue(torch.equal(out[~self.valid], self.resid[~self.valid]))
        self.assertGreater(st["sum_norm_ratio"], 0)

    def test_error_term_patch_keeps_the_recipients_reconstruction(self):
        x_d, a_d = self.donor_setup()
        allidx = torch.arange(32)
        out, _ = fi.apply_edit(self.resid, self.valid, self.a, self.sae.W_dec, allidx, "donor_error", a_donor=a_d, x_donor=x_d)
        x = self.resid[self.valid]
        xhat_b, xhat_d = self.sae.decode(self.a), self.sae.decode(a_d)
        err_d = x_d - xhat_d
        self.assertTrue(torch.allclose(out[self.valid], xhat_b + err_d, atol=1e-4))      # recipient reconstruction + donor error term

    def test_all_features_patch_plus_error_patch_adds_up_to_the_donor_residual(self):
        x_d, a_d = self.donor_setup()
        allidx = torch.arange(32)
        o_all, _ = fi.apply_edit(self.resid, self.valid, self.a, self.sae.W_dec, allidx, "donor", a_donor=a_d)
        o_err, _ = fi.apply_edit(self.resid, self.valid, self.a, self.sae.W_dec, allidx, "donor_error", a_donor=a_d, x_donor=x_d)
        x = self.resid[self.valid]
        # x_all = x_b + (xhat_d - xhat_b); x_err = xhat_b + err_d; their combination x_all + x_err - x_b = x_d
        self.assertTrue(torch.allclose(o_all[self.valid] + o_err[self.valid] - x, x_d, atol=1e-4))

    def test_donor_kinds_need_their_arguments_and_matching_shapes(self):
        x_d, a_d = self.donor_setup()
        with self.assertRaises(ValueError):
            fi.apply_edit(self.resid, self.valid, self.a, self.sae.W_dec, self.idx, "donor_resid")
        with self.assertRaises(ValueError):
            fi.apply_edit(self.resid, self.valid, self.a, self.sae.W_dec, self.idx, "donor_resid", x_donor=x_d[:-1])
        with self.assertRaises(ValueError):
            fi.apply_edit(self.resid, self.valid, self.a, self.sae.W_dec, self.idx, "donor_error", a_donor=a_d)
        with self.assertRaises(ValueError):
            fi.apply_edit(self.resid, self.valid, self.a, self.sae.W_dec, self.idx, "donor_error", x_donor=x_d)

    def test_sham_is_bitwise_in_bf16_and_tiny_in_float32(self):
        resid, valid = make_batch(dtype=torch.bfloat16)
        a = fi.encode_valid(resid, valid, self.sae.encode)
        out, st = fi.apply_edit(resid, valid, a, self.sae.W_dec, self.idx, "sham")
        self.assertTrue(torch.equal(out, resid))
        self.assertLess(st["sum_norm_ratio"], 1e-5)               # float32 round-off of (x + d) - d only
        out32, _ = self.edit("sham")
        self.assertLess(float((out32 - self.resid).abs().max()), 1e-4)

    def test_norm_and_activity_statistics_on_a_hand_case(self):
        W = torch.zeros(4, 3); W[0] = torch.tensor([3.0, 4.0, 0.0]); W[1] = torch.tensor([0.0, 0.0, 1.0])
        resid = torch.tensor([[[0.0, 0.0, 0.0], [6.0, 8.0, 0.0]]])      # position 0 is BOS (invalid)
        valid = torch.tensor([[False, True]])
        a = torch.tensor([[2.0, 0.0, 0.0, 0.0]])
        out, st = fi.apply_edit(resid, valid, a, W, torch.tensor([0, 1]), "ablate")
        self.assertTrue(torch.allclose(out[0, 1], torch.tensor([0.0, 0.0, 0.0])))
        self.assertAlmostEqual(st["sum_norm_ratio"], 1.0, places=6)        # ||(-6,-8,0)|| / ||(6,8,0)||
        self.assertEqual(st["sum_active"], 1.0)                            # one group feature is active at the one edited row
        self.assertEqual(st["rows"], 1)

    def test_invalid_arguments_are_errors(self):
        with self.assertRaises(ValueError):
            fi.apply_edit(self.resid, self.valid, self.a, self.sae.W_dec, torch.tensor([1, 1]), "ablate")      # duplicate
        with self.assertRaises(ValueError):
            fi.apply_edit(self.resid, self.valid, self.a, self.sae.W_dec, torch.tensor([99]), "ablate")        # out of range
        with self.assertRaises(ValueError):
            fi.apply_edit(self.resid, self.valid, self.a[:-1], self.sae.W_dec, self.idx, "ablate")             # rows do not match valid
        with self.assertRaises(ValueError):
            fi.apply_edit(self.resid, self.valid, self.a, self.sae.W_dec, self.idx, "scale")                   # alpha missing
        with self.assertRaises(ValueError):
            fi.apply_edit(self.resid, self.valid, self.a, self.sae.W_dec, self.idx, "resample")                # pool missing
        with self.assertRaises(ValueError):
            fi.apply_edit(self.resid, self.valid, self.a, self.sae.W_dec, self.idx, "donor")                   # donor missing
        with self.assertRaises(ValueError):
            fi.apply_edit(self.resid, self.valid, self.a, self.sae.W_dec, self.idx, "teleport")                # unknown kind

    def test_edits_are_independent_of_batch_composition(self):
        out, _ = self.edit("ablate")
        # the same row edited alone gives the same result as inside the batch
        r, v = self.resid[:1], self.valid[:1]
        a1 = fi.encode_valid(r, v, self.sae.encode)
        o1, _ = fi.apply_edit(r, v, a1, self.sae.W_dec, self.idx, "ablate")
        self.assertTrue(torch.allclose(o1[0], out[0], atol=1e-6))


if __name__ == "__main__":
    unittest.main()
