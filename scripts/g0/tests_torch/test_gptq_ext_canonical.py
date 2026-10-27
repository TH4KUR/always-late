"""Task 2 of the GPTQ extension plan: the mapping between TransformerLens parameters and the canonical [out, in] linear matrices that GPTQ and the matched RTN work on (audit section 3: Q/K/V [h,m,d] -> [h*d, m],
O [h,d,m] -> [m, h*d], MLP [in,out] -> [out,in]; grouped-query attention uses the STORED `_W_K` and `_W_V`). Tests that need TransformerLens run in ~/.venvs/anlp-tl and are skipped elsewhere."""
import sys
import unittest
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
G0 = HERE.parent
ROOT = G0.parents[1]
sys.path.insert(0, str(G0)); sys.path.insert(0, str(HERE))
import their_code  # noqa: E402
from gptq_ext import canonical  # noqa: E402
from gptq_ext_tiny import DIMS, HAVE_TL, tiny_gemma2, tokens  # noqa: E402

H, M, D, KV, MLP = DIMS["n_heads"], DIMS["d_model"], DIMS["d_head"], DIMS["n_kv"], DIMS["d_mlp"]
SHAPES = {"blocks.3.attn.W_Q": (H, M, D), "blocks.3.attn._W_K": (KV, M, D), "blocks.3.attn._W_V": (KV, M, D), "blocks.3.attn.W_O": (H, D, M),
          "blocks.3.mlp.W_in": (M, MLP), "blocks.3.mlp.W_gate": (M, MLP), "blocks.3.mlp.W_out": (MLP, M)}


def legacy_filter():
    ns = their_code.build_namespace(their_code.extract_functions(their_code.PRISTINE, ["target_tl_weight"]), {})
    return ns["target_tl_weight"]


class Mapping(unittest.TestCase):
    def test_round_trip_is_exact_for_every_target_kind(self):
        for dtype in (torch.float32, torch.bfloat16):
            for name, shape in SHAPES.items():
                w = torch.randn(*shape, generator=torch.Generator().manual_seed(len(name))).to(dtype)
                lin = canonical.to_linear(name, w)
                self.assertEqual(lin.dtype, dtype)
                self.assertTrue(lin.is_contiguous())
                back = canonical.from_linear(name, lin, w)
                self.assertTrue(torch.equal(back, w), (name, dtype))
                self.assertEqual(back.shape, w.shape)

    def test_canonical_shapes_are_out_by_in(self):
        want = {"W_Q": (H * D, M), "_W_K": (KV * D, M), "_W_V": (KV * D, M), "W_O": (M, H * D), "W_in": (MLP, M), "W_gate": (MLP, M), "W_out": (M, MLP)}
        for name, shape in SHAPES.items():
            self.assertEqual(tuple(canonical.to_linear(name, torch.zeros(*shape)).shape), want[name.split(".")[-1]], name)

    def test_row_order_is_head_major_then_head_coordinate(self):
        w = torch.arange(H * M * D, dtype=torch.float32).reshape(H, M, D)
        lin = canonical.to_linear("blocks.0.attn.W_Q", w)
        self.assertTrue(torch.equal(lin[2 * D + 5], w[2, :, 5]))             # row (head 2, coordinate 5) = column 5 of head 2's [m, d] matrix
        wo = torch.arange(H * D * M, dtype=torch.float32).reshape(H, D, M)
        lo = canonical.to_linear("blocks.0.attn.W_O", wo)
        self.assertTrue(torch.equal(lo[7], wo[:, :, 7].reshape(H * D)))      # output channel 7 reads (head, coordinate) in head-major order

    def test_one_dimensional_and_unknown_tensors_are_refused(self):
        with self.assertRaises(ValueError):
            canonical.to_linear("blocks.0.attn.W_Q", torch.zeros(4))
        with self.assertRaises(ValueError):
            canonical.to_linear("blocks.0.ln1.w", torch.zeros(4, 4))
        with self.assertRaises(ValueError):
            canonical.from_linear("blocks.0.attn.W_Q", torch.zeros(3, 3), torch.zeros(H, M, D))

    def test_from_linear_takes_dtype_and_shape_of_the_reference_tensor(self):
        like = torch.zeros(H, M, D, dtype=torch.bfloat16)
        out = canonical.from_linear("blocks.0.attn.W_Q", torch.randn(H * D, M), like)
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(out.shape, like.shape)

    def test_a_parameter_with_a_target_substring_but_unknown_kind_is_refused(self):
        m = torch.nn.Module(); m.W_Q_extra = torch.nn.Parameter(torch.zeros(2, 2))
        with self.assertRaises(ValueError):
            canonical.targets(m)

    def test_the_filter_equals_the_legacy_filter_on_all_names_of_a_layer(self):
        legacy = legacy_filter()
        names = list(SHAPES) + ["blocks.3.ln1.w", "blocks.3.attn.b_Q", "blocks.3.attn._b_K", "blocks.3.mlp.b_in", "blocks.3.mlp.b_out", "embed.W_E", "unembed.W_U", "unembed.b_U", "ln_final.w"]
        for n in names:
            self.assertEqual(canonical.is_target(n), bool(legacy(n)), n)

    def test_the_legacy_quantizer_on_one_tensor_equals_grids_legacy_rtn(self):
        from gptq_ext import grids
        ns = their_code.build_namespace(their_code.extract_functions(their_code.PRISTINE, ["target_tl_weight", "quantize_rtn_per_output_channel_tl"]), {"torch": torch})
        for dtype in (torch.float32, torch.bfloat16):
            model = torch.nn.Module(); model.W_Q = torch.nn.Parameter(torch.randn(H, M, D, generator=torch.Generator().manual_seed(1)).to(dtype), requires_grad=False)
            w0 = model.W_Q.data.clone()
            ns["quantize_rtn_per_output_channel_tl"](model, bits=6)
            self.assertTrue(torch.equal(model.W_Q.data, grids.legacy_rtn(w0, 6)), dtype)


@unittest.skipUnless(HAVE_TL, "needs TransformerLens (~/.venvs/anlp-tl)")
class WithTransformerLens(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = tiny_gemma2()

    def test_target_filter_equals_the_legacy_filter_on_a_tiny_gemma2_model(self):
        legacy = legacy_filter()
        want = [n for n, _ in self.model.named_parameters() if legacy(n)]
        got = [n for n, _ in canonical.targets(self.model)]
        self.assertEqual(got, want)
        self.assertEqual(len(got), 7 * self.model.cfg.n_layers)
        self.assertTrue(all(p.ndim == 3 or p.ndim == 2 for _, p in canonical.targets(self.model)))

    def test_gqa_uses_stored_kv_parameters_not_expanded_views(self):
        shapes = {n: tuple(p.shape) for n, p in canonical.targets(self.model)}
        self.assertEqual(shapes["blocks.0.attn._W_K"], (KV, M, D))
        self.assertEqual(shapes["blocks.0.attn._W_V"], (KV, M, D))
        self.assertEqual(shapes["blocks.0.attn.W_Q"], (H, M, D))
        self.assertFalse(any(n.endswith(".W_K") or n.endswith(".W_V") for n in shapes))

    def test_canonical_scale_count_is_out_channels(self):
        from gptq_ext import grids
        p = dict(canonical.targets(self.model))["blocks.0.attn.W_Q"]
        scale, _ = grids.authors_grid(canonical.to_linear("blocks.0.attn.W_Q", p.data), 4)
        self.assertEqual(scale.numel(), H * D)                                  # 32 canonical scales
        self.assertEqual(grids.legacy_scale(p.data, 4).numel(), D)              # 8 legacy scales shared across heads

    def test_projection_agrees_with_the_model_forward(self):
        model, toks = self.model, tokens()
        cap = {}
        names = ["blocks.1.attn.hook_q", "blocks.1.attn.hook_k", "blocks.1.attn.hook_v", "blocks.1.attn.hook_z", "blocks.1.mlp.hook_pre", "blocks.1.mlp.hook_pre_linear", "blocks.1.mlp.hook_post"]
        # the projection inputs are the OUTPUTS of the norm modules (`hook_normalized` is taken before the learned scale `w`)
        handles = [model.blocks[1].ln1.register_forward_hook(lambda m, i, o: cap.__setitem__("ln1_out", o.detach().clone())),
                   model.blocks[1].ln2.register_forward_hook(lambda m, i, o: cap.__setitem__("ln2_out", o.detach().clone()))]
        try:
            with torch.no_grad():
                model.run_with_hooks(toks, fwd_hooks=[(n, lambda t, hook: cap.__setitem__(hook.name, t.detach().clone())) for n in names])
        finally:
            for h in handles:
                h.remove()
        lin = {n: canonical.to_linear(n, p.data.double()) for n, p in canonical.targets(model)}
        b, s = toks.shape
        x1 = cap["ln1_out"].double()
        for proj, hook, heads in (("blocks.1.attn.W_Q", "blocks.1.attn.hook_q", H), ("blocks.1.attn._W_K", "blocks.1.attn.hook_k", KV), ("blocks.1.attn._W_V", "blocks.1.attn.hook_v", KV)):
            got = (x1 @ lin[proj].T).reshape(b, s, heads, D)
            self.assertTrue(torch.allclose(got, cap[hook].double(), atol=1e-5), proj)
        x2 = cap["ln2_out"].double()
        self.assertTrue(torch.allclose(x2 @ lin["blocks.1.mlp.W_gate"].T, cap["blocks.1.mlp.hook_pre"].double(), atol=1e-5))
        self.assertTrue(torch.allclose(x2 @ lin["blocks.1.mlp.W_in"].T, cap["blocks.1.mlp.hook_pre_linear"].double(), atol=1e-5))
        post = cap["blocks.1.mlp.hook_post"].double()
        out_tl = torch.einsum("bsf,fm->bsm", post, dict(canonical.targets(model))["blocks.1.mlp.W_out"].data.double())
        self.assertTrue(torch.allclose(post @ lin["blocks.1.mlp.W_out"].T, out_tl, atol=1e-9))
        z = cap["blocks.1.attn.hook_z"].double()
        out_o = torch.einsum("bshd,hdm->bsm", z, dict(canonical.targets(model))["blocks.1.attn.W_O"].data.double())
        self.assertTrue(torch.allclose(z.reshape(b, s, H * D) @ lin["blocks.1.attn.W_O"].T, out_o, atol=1e-9))


if __name__ == "__main__":
    unittest.main()
