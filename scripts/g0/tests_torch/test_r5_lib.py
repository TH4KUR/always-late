"""Torch-dependent tests for scripts/g0/r5_lib.py (R5: OUR pipeline on THEIR tokens, with factor toggles).
Run with a torch-enabled python:  python -m unittest discover -s scripts/g0/tests_torch -v     (CPU is enough)
The decisive test executes THEIR quantizer (extracted verbatim from the vendored pristine script) on a TransformerLens-layout
copy of a tiny Gemma-like layer and requires our HF-layout implementation to give bit-identical weights."""
import ast, sys, unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(ROOT))
import r5_lib as r5  # noqa: E402
from src.rtn_quant import rtn_fake_quantize  # noqa: E402  (OUR quantizer)

PRISTINE03 = ROOT / "third_party" / "duan2026" / "pristine" / "scripts" / "03_gemma_phase3_streaming_sweep.py"


def their_functions(*names):
    src = PRISTINE03.read_text().split("# %% Cell 22")[0]
    tree = ast.parse(src)
    code = "\n\n".join(ast.get_source_segment(src, n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names)
    ns = {"torch": torch, "np": np}
    exec(code, ns)
    return ns


class Windows(unittest.TestCase):
    def test_build_windows_follows_their_rule(self):
        w = r5.build_windows(list(range(1000)), token_budget=600, seq_len=256)
        self.assertEqual(tuple(w.shape), (2, 256)); self.assertEqual(w[1, -1].item(), 511)      # usable = 512 of min(600, 1000)
        self.assertEqual(tuple(r5.build_windows(list(range(300)), 500_000, 256).shape), (1, 256))  # budget larger than the stream
        self.assertEqual(tuple(r5.build_windows(list(range(500_000 + 7)), 500_000, 256).shape), (1953, 256))  # their 499,968 tokens

    def test_bos_regimes(self):
        w = torch.arange(512).reshape(2, 256)
        self.assertEqual(tuple(r5.apply_regime(w, "none", 2).shape), (2, 256))
        for reg in ("mask", "keep"):
            x = r5.apply_regime(w, reg, 2)
            self.assertEqual(tuple(x.shape), (2, 257)); self.assertTrue((x[:, 0] == 2).all()); self.assertTrue(torch.equal(x[:, 1:], w))   # identical text windows
        self.assertEqual(r5.stats_start("none"), 0); self.assertEqual(r5.stats_start("keep"), 0); self.assertEqual(r5.stats_start("mask"), 1)
        with self.assertRaises(ValueError):
            r5.apply_regime(w, "bogus", 2)


class PearsonAndSummary(unittest.TestCase):
    def test_streaming_pearson_matches_numpy_in_float64(self):
        g = torch.Generator().manual_seed(0)
        x = torch.relu(torch.randn(500, 40, generator=g) - 0.5); y = x + 0.1 * torch.randn(500, 40, generator=g)
        s = r5.PearsonStream(40, "cpu")
        for i in range(0, 500, 77):
            s.update(x[i:i + 77], y[i:i + 77])
        res = s.result()
        ref = np.array([np.corrcoef(x[:, j].double().numpy(), y[:, j].double().numpy())[0, 1] for j in range(40)])
        self.assertLess(np.abs(res["corr"].numpy() - ref).max(), 1e-9)
        self.assertEqual(res["n"], 500)
        self.assertTrue(torch.allclose(res["firing_rate"], (x > 0).double().mean(0)))

    def test_zero_variance_in_the_quantized_arm_gives_corr_zero_like_theirs(self):
        x = torch.rand(100, 3); y = x.clone(); y[:, 1] = 0.0
        s = r5.PearsonStream(3, "cpu"); s.update(x, y)
        c = s.result()["corr"]
        self.assertEqual(c[1].item(), 0.0); self.assertAlmostEqual(c[0].item(), 1.0, places=9)

    def test_summary_uses_their_thresholds_and_torch_lower_median(self):
        corr = torch.tensor([1.0, 0.95, 0.9, 0.7, 0.5, 0.3, 0.99, 0.2], dtype=torch.float64)
        rate = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0005, 0.5], dtype=torch.float64)       # feature 6 is inactive (rate <= 0.001)
        s = r5.summarize(corr, rate)
        self.assertEqual(s["n_active"], 7)
        self.assertAlmostEqual(s["survived_pct"], 100 * 2 / 7)        # > 0.9 only: 1.0, 0.95 (0.9 itself is NOT survived)
        self.assertAlmostEqual(s["damaged_pct"], 100 * 2 / 7)         # < 0.5: 0.3, 0.2 (0.5 itself is not damaged)
        self.assertAlmostEqual(s["degraded_pct"], 100 * 2 / 7)        # 0.5 < r <= 0.9: 0.9, 0.7. Their three bins do not partition: r == 0.5 exactly is in none
        self.assertEqual(s["median_corr"], torch.tensor([1.0, 0.95, 0.9, 0.7, 0.5, 0.3, 0.2], dtype=torch.float64).median().item())


class SparseStore(unittest.TestCase):
    def test_roundtrip_is_exact_for_bf16_and_fp32(self):
        for dt in (torch.bfloat16, torch.float32):
            g = torch.Generator().manual_seed(1)
            dense = (torch.rand(64, 300, generator=g) * (torch.rand(64, 300, generator=g) < 0.02)).to(dt)
            st = r5.SparseFeatureStore()
            st.add(dense[:40]); st.add(dense[40:])
            self.assertTrue(torch.equal(st.get(0, "cpu"), dense[:40])); self.assertTrue(torch.equal(st.get(1, "cpu"), dense[40:]))
            self.assertEqual(st.n_batches, 2)


class QuantizerEquivalence(unittest.TestCase):
    """THEIR quantizer (TransformerLens layout, extracted verbatim) vs OUR HF-layout reimplementation, all seven module kinds."""
    H, KV, D, M, MLP = 2, 1, 8, 16, 24

    def hf_weights(self, seed=0):
        g = torch.Generator().manual_seed(seed); mk = lambda *s: (torch.randn(*s, generator=g) * 0.05).to(torch.bfloat16)
        H, KV, D, M, MLP = self.H, self.KV, self.D, self.M, self.MLP
        return {"q_proj": mk(H * D, M), "k_proj": mk(KV * D, M), "v_proj": mk(KV * D, M), "o_proj": mk(M, H * D),
                "gate_proj": mk(MLP, M), "up_proj": mk(MLP, M), "down_proj": mk(M, MLP)}

    def to_tl(self, hf):
        H, KV, D, M = self.H, self.KV, self.D, self.M
        return {"attn.W_Q": hf["q_proj"].view(H, D, M).permute(0, 2, 1).contiguous(), "attn._W_K": hf["k_proj"].view(KV, D, M).permute(0, 2, 1).contiguous(),
                "attn._W_V": hf["v_proj"].view(KV, D, M).permute(0, 2, 1).contiguous(), "attn.W_O": hf["o_proj"].view(M, H, D).permute(1, 2, 0).contiguous(),
                "mlp.W_in": hf["up_proj"].T.contiguous(), "mlp.W_gate": hf["gate_proj"].T.contiguous(), "mlp.W_out": hf["down_proj"].T.contiguous()}

    def from_tl(self, tl):
        H, KV, D, M = self.H, self.KV, self.D, self.M
        return {"q_proj": tl["attn.W_Q"].permute(0, 2, 1).reshape(H * D, M), "k_proj": tl["attn._W_K"].permute(0, 2, 1).reshape(KV * D, M),
                "v_proj": tl["attn._W_V"].permute(0, 2, 1).reshape(KV * D, M), "o_proj": tl["attn.W_O"].permute(2, 0, 1).reshape(M, H * D),
                "up_proj": tl["mlp.W_in"].T, "gate_proj": tl["mlp.W_gate"].T, "down_proj": tl["mlp.W_out"].T}

    def test_layout_conversions_are_inverse(self):
        hf = self.hf_weights()
        back = self.from_tl(self.to_tl(hf))
        for k in hf:
            self.assertTrue(torch.equal(hf[k], back[k]), k)

    def test_our_hf_layout_quantizer_is_bit_identical_to_their_tl_quantizer(self):
        theirs = their_functions("target_tl_weight", "quantize_rtn_per_output_channel_tl")["quantize_rtn_per_output_channel_tl"]
        for bits in (8, 7, 6, 4):
            hf = self.hf_weights(seed=bits)
            tl = self.to_tl(hf)
            class FakeTL:        # minimal stand-in: only named_parameters() is used by their function; names keep TransformerLens spelling
                def __init__(s):
                    s.params = {k: torch.nn.Parameter(v.clone(), requires_grad=False) for k, v in tl.items()}
                def named_parameters(s, *a, **k):
                    for k, p in s.params.items():
                        yield f"blocks.0.{k}", p
            m = FakeTL(); theirs(m, bits=bits)
            theirs_hf = self.from_tl({k: p.data for k, p in m.params.items()})
            for kind, w in hf.items():
                ours = r5.duan_fake_quantize(w, bits, r5.group_for(kind), head_dim=self.D)
                self.assertTrue(torch.equal(ours, theirs_hf[kind]), f"INT{bits} {kind}: our HF-layout result differs from their TL-layout result")

    def test_qkv_use_shared_scales_but_every_other_module_equals_plain_per_row(self):
        hf = self.hf_weights(seed=3)
        for kind in ("o_proj", "gate_proj", "up_proj", "down_proj"):
            self.assertEqual(r5.group_for(kind), "row")
            self.assertTrue(torch.equal(r5.duan_fake_quantize(hf[kind], 6, "row", 8), rtn_fake_quantize(hf[kind], 6)), f"{kind}: ours != theirs")  # OUR src quantizer
        for kind in ("q_proj", "k_proj", "v_proj"):
            self.assertEqual(r5.group_for(kind), "head_shared")
        # the single real difference: shared-across-heads scales are coarser than per-row for q/k/v
        q = hf["q_proj"]
        self.assertFalse(torch.equal(r5.duan_fake_quantize(q, 6, "head_shared", 8), rtn_fake_quantize(q, 6)))


if __name__ == "__main__":
    unittest.main()
