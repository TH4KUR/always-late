"""Tests for scripts/g0/r5_run.py (R5 cell runner) on a tiny random Gemma-2 (CPU, no downloads).
Run:  /tmp/claude-1000/torchvenv/bin/python -m unittest discover -s scripts/g0/tests_torch -v     (needs torch, transformers, pandas)"""
import copy, sys, tempfile, unittest
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(ROOT))
import r5_run  # noqa: E402
from src.sae_models import JumpReLUSAE  # noqa: E402
from transformers import Gemma2Config, Gemma2ForCausalLM  # noqa: E402

LAYER, L, N_WIN, D_SAE = 2, 16, 6, 64
SUMMARY_COLS = ["condition", "n_tokens", "n_total_features", "n_active_features", "mean_corr", "median_corr", "survived_>0.9_pct",
                "degraded_0.5_0.9_pct", "damaged_<0.5_pct", "layer", "bits", "perplexity", "loss", "ppl_delta_pct", "method"]


def tiny_model(attn="eager", seed=0):
    torch.manual_seed(seed)
    cfg = Gemma2Config(vocab_size=300, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
                       head_dim=8, query_pre_attn_scalar=8, sliding_window=16, max_position_embeddings=64,
                       attn_logit_softcapping=50.0, final_logit_softcapping=30.0)
    return Gemma2ForCausalLM._from_config(cfg, attn_implementation=attn, dtype=torch.bfloat16).eval()


def tiny_sae(seed=1):
    torch.manual_seed(seed)
    s = JumpReLUSAE(32, D_SAE)
    with torch.no_grad():
        s.W_enc.copy_(torch.randn(32, D_SAE) * 4.0); s.b_enc.copy_(torch.randn(D_SAE) * 0.1); s.threshold.copy_(torch.rand(D_SAE) * 0.05)
    return s.eval()


def tiny_windows(seed=2):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(3, 300, (N_WIN, L), generator=g)


def run(model, regime="none", quant="ours", sae_dtype="fp32", attn="eager", bits=(8, 4), out=None, logger=None):
    spec = r5_run.CellSpec(attn=attn, sae_dtype=sae_dtype, quant=quant, regime=regime)
    out = Path(out or tempfile.mkdtemp())
    rows = r5_run.run_cell("X", spec, model, tiny_sae(), tiny_windows(), bos_id=2, bits=list(bits), out_dir=out, layer=LAYER, device="cpu", logger=logger)
    return rows, out


class Losses(unittest.TestCase):
    def test_tl_style_loss_is_log_softmax_in_the_logits_dtype_then_gathered_mean(self):
        g = torch.Generator().manual_seed(3)
        logits = (torch.randn(1, 9, 50, generator=g) * 10).to(torch.bfloat16); ids = torch.randint(0, 50, (1, 9), generator=g)
        ref = -torch.log_softmax(logits, dim=-1)[..., :-1, :].gather(-1, ids[..., 1:, None])[..., 0].mean()   # what TransformerLens does
        self.assertEqual(r5_run.tl_style_loss(logits, ids), float(ref))
        self.assertAlmostEqual(r5_run.fp32_loss(logits, ids), float(torch.nn.functional.cross_entropy(logits[0, :-1].float(), ids[0, 1:])), places=6)

    def test_losses_cover_only_next_token_predictions(self):
        logits = torch.zeros(1, 5, 7, dtype=torch.bfloat16)
        self.assertAlmostEqual(r5_run.fp32_loss(logits, torch.zeros(1, 5, dtype=torch.long)), float(torch.log(torch.tensor(7.0))), places=5)


class AttentionSwitch(unittest.TestCase):
    def test_switch_changes_the_computation_and_switching_back_is_bit_exact(self):
        m, ids = tiny_model("eager"), tiny_windows()[:2]
        with torch.no_grad():
            before = m(input_ids=ids).logits.clone()
            r5_run.set_attention(m, "sdpa"); self.assertEqual(m.config._attn_implementation, "sdpa")
            mid = m(input_ids=ids).logits.clone()
            r5_run.set_attention(m, "eager"); self.assertEqual(m.config._attn_implementation, "eager")
            after = m(input_ids=ids).logits.clone()
        self.assertFalse(torch.equal(before, mid))     # soft-capping of attention logits is applied by eager only
        self.assertTrue(torch.equal(before, after))

    def test_rejects_unknown_implementation(self):
        with self.assertRaises(ValueError):
            r5_run.set_attention(tiny_model(), "flash")


class RunCell(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = tiny_model("eager")
        cls.orig = {k: v.clone() for k, v in cls.model.state_dict().items()}
        cls.rows, cls.out = run(cls.model)

    def test_schema_matches_their_summary_table(self):
        self.assertEqual([r["condition"] for r in self.rows], ["FP16 baseline", "RTN INT8", "RTN INT4"])
        for r in self.rows:
            for c in SUMMARY_COLS:
                self.assertIn(c, r)
        df = pd.read_csv(self.out / "phase3_summary_final.csv")
        self.assertEqual(list(df.columns[: len(SUMMARY_COLS)]), SUMMARY_COLS)
        self.assertEqual(len(df), 3)

    def test_fp_row_is_a_perfect_self_comparison_and_signal_exists(self):
        fp = self.rows[0]
        self.assertGreater(fp["n_active_features"], 5)
        self.assertGreater(fp["mean_corr"], 1 - 1e-9); self.assertEqual(fp["survived_>0.9_pct"], 100.0); self.assertEqual(fp["ppl_delta_pct"], 0.0)
        self.assertEqual(fp["n_tokens"], N_WIN * L); self.assertEqual(fp["bits"], 16)

    def test_fewer_bits_damage_more(self):
        i8, i4 = self.rows[1], self.rows[2]
        self.assertGreater(i8["mean_corr"], i4["mean_corr"]); self.assertLess(i4["mean_corr"], 1.0)
        self.assertAlmostEqual(i4["ppl_delta_pct"], (i4["perplexity"] / self.rows[0]["perplexity"] - 1) * 100, places=9)

    def test_weights_and_hooks_are_left_untouched(self):
        for k, v in self.model.state_dict().items():
            self.assertTrue(torch.equal(v, self.orig[k]), k)
        self.assertEqual(len(self.model.model.layers[LAYER]._forward_hooks), 0)

    def test_per_feature_files_follow_their_schema(self):
        pf = pd.read_csv(self.out / "RTN_INT4_per_feature.csv")
        self.assertEqual(list(pf.columns), ["feature_id", "condition", "layer", "corr", "firing_rate", "mean_activation", "max_activation", "active",
                                            "survived_>0.9", "damaged_<0.5"])
        self.assertEqual(len(pf), D_SAE); self.assertTrue((self.out / "FP16_baseline_per_feature.csv").exists())

    def test_extra_columns_record_the_fp32_loss_and_the_cell(self):
        for r in self.rows:
            self.assertIn("perplexity_fp32", r); self.assertEqual(r["cell"], "X")
            self.assertAlmostEqual(r["perplexity_fp32"] / r["perplexity"], 1.0, delta=0.05)   # same quantity up to bf16 log-softmax rounding

    def test_bos_regimes_change_what_enters_the_statistics(self):
        m = tiny_model("eager")
        keep, _ = run(m, regime="keep", bits=(8,)); mask, _ = run(m, regime="mask", bits=(8,))
        self.assertEqual(keep[0]["n_tokens"], N_WIN * (L + 1)); self.assertEqual(mask[0]["n_tokens"], N_WIN * L)

    def test_flipping_a_factor_changes_the_result(self):
        m = tiny_model("eager")
        base, _ = run(m, bits=(4,))
        for kw in (dict(quant="duan"), dict(sae_dtype="bf16"), dict(attn="sdpa"), dict(regime="keep")):
            other, _ = run(m, bits=(4,), **kw)
            self.assertNotEqual((base[1]["mean_corr"], base[0]["n_active_features"]), (other[1]["mean_corr"], other[0]["n_active_features"]), kw)

    def test_the_attention_setting_is_restored_after_a_cell(self):
        m = tiny_model("eager"); run(m, attn="sdpa", bits=(8,))
        self.assertEqual(m.config._attn_implementation, "eager")

    def test_each_toggle_points_the_right_way(self):
        """Direction, not just 'something changes': firing rates are recomputed by hand for every (SAE dtype, BOS regime) and attention is read mid-pass."""
        import numpy as np
        from src.pipeline import ResidualCapture
        for sae_dtype in ("fp32", "bf16"):
            for regime in ("none", "mask", "keep"):
                m = tiny_model("eager"); seen = []

                class Spy:
                    def log_condition(self, s): pass
                    def heartbeat(self, *a, **k): seen.append(m.config._attn_implementation)
                rows, out = run(m, regime=regime, sae_dtype=sae_dtype, attn="sdpa", bits=(8,), logger=Spy())
                self.assertEqual(set(seen), {"sdpa"})                      # the cell's attention setting is active while it runs
                sae = tiny_sae(); sae = sae.to(torch.bfloat16) if sae_dtype == "bf16" else sae
                r5_run.set_attention(m, "sdpa"); cap = ResidualCapture(m, LAYER); fired, total, n = torch.zeros(D_SAE, dtype=torch.float64), torch.zeros(D_SAE, dtype=torch.float64), 0
                for ids in r5_run.lib.apply_regime(tiny_windows(), regime, 2):
                    with torch.no_grad():
                        m(input_ids=ids[None])
                        f = sae.encode(cap.activation[0]).float()[r5_run.lib.stats_start(regime):]
                    fired += (f > 0).sum(0); total += f.double().sum(0); n += f.shape[0]
                cap.remove(); r5_run.set_attention(m, "eager")
                pf = pd.read_csv(out / "FP16_baseline_per_feature.csv")
                self.assertTrue(np.allclose(pf["firing_rate"].values, (fired / n).numpy(), atol=1e-12, rtol=0), (sae_dtype, regime))
                self.assertTrue(np.allclose(pf["mean_activation"].values, (total / n).numpy(), atol=1e-12, rtol=1e-12), (sae_dtype, regime))

    def test_a_no_op_quantizer_must_give_perfect_survival_so_cached_features_stay_aligned_with_windows(self):
        from contextlib import nullcontext
        from unittest import mock
        with mock.patch.object(r5_run, "_quant_ctx", lambda q, m: (lambda bits: nullcontext())):
            rows, _ = run(tiny_model("eager"), bits=(8,))
        self.assertGreater(rows[1]["mean_corr"], 1 - 1e-9); self.assertEqual(rows[1]["survived_>0.9_pct"], 100.0); self.assertEqual(rows[1]["ppl_delta_pct"], 0.0)

    def test_quantizer_choice_maps_to_their_grouping_or_ours(self):
        from src.rtn_quant import rtn_fake_quantize
        m = tiny_model("eager"); w0 = m.model.layers[0].self_attn.q_proj.weight.data.clone()
        with r5_run._quant_ctx("duan", m)(4):
            self.assertTrue(torch.equal(m.model.layers[0].self_attn.q_proj.weight.data, r5_run.lib.duan_fake_quantize(w0, 4, "head_shared", head_dim=8)))
        with r5_run._quant_ctx("ours", m)(4):
            self.assertTrue(torch.equal(m.model.layers[0].self_attn.q_proj.weight.data, rtn_fake_quantize(w0, 4)))
        self.assertFalse(torch.equal(rtn_fake_quantize(w0, 4), r5_run.lib.duan_fake_quantize(w0, 4, "head_shared", head_dim=8)))   # the two really differ for q
        self.assertTrue(torch.equal(m.model.layers[0].self_attn.q_proj.weight.data, w0))

    def test_logger_sees_every_condition_and_progress(self):
        class Spy:
            def __init__(self): self.conds, self.beats = [], 0
            def log_condition(self, s): self.conds.append(s["condition"])
            def heartbeat(self, *a, **k): self.beats += 1
        spy = Spy(); run(tiny_model("eager"), bits=(8,), logger=spy)
        self.assertEqual(spy.conds, ["FP16 baseline", "RTN INT8"]); self.assertGreater(spy.beats, 0)


class Diagnostics(unittest.TestCase):
    def test_int0_identity_passes_for_a_correct_restore_and_fails_for_a_broken_one(self):
        m, w = tiny_model("eager"), tiny_windows()[:3]
        self.assertTrue(r5_run.int0_identity(m, w, "cpu")["identical"])

        @contextmanager
        def leaky(model, bits):          # quantizes and never restores
            from src.rtn_quant import rtn_fake_quantize, _is_quant_target
            for n, mod in model.named_modules():
                if _is_quant_target(n, mod):
                    mod.weight.data.copy_(rtn_fake_quantize(mod.weight.data, bits))
            yield []
        self.assertFalse(r5_run.int0_identity(tiny_model("eager"), w, "cpu", ctx=leaky)["identical"])

    def test_sdpa_vs_eager_reports_the_size_of_the_softcap_effect_and_restores_the_setting(self):
        m = tiny_model("eager")
        d = r5_run.sdpa_vs_eager(m, tiny_windows()[:3], LAYER, "cpu")
        for k in ("resid_rel_l2", "logit_max_abs", "loss_delta"):
            self.assertIn(k, d)
        self.assertGreater(d["resid_rel_l2"], 0); self.assertEqual(m.config._attn_implementation, "eager")

    def test_feature_agreement_counts_mask_and_value_differences(self):
        a = torch.tensor([[0., 1., 2.], [0., 0., 3.]]); b = torch.tensor([[0., 1., 0.], [0.5, 0., 3.]])
        d = r5_run.feature_agreement(a, b)
        self.assertEqual(d["n_active_a"], 3); self.assertEqual(d["n_active_b"], 3); self.assertEqual(d["n_both"], 2)
        self.assertAlmostEqual(d["active_jaccard"], 2 / 4); self.assertAlmostEqual(d["max_abs_diff_common"], 0.0)

    def test_cell_done_marker(self):
        with tempfile.TemporaryDirectory() as t:
            self.assertFalse(r5_run.cell_is_done(Path(t)))
            (Path(t) / "cell_done.json").write_text("{}")
            self.assertTrue(r5_run.cell_is_done(Path(t)))


class Specs(unittest.TestCase):
    def test_cells_flip_exactly_one_factor_from_the_anchor(self):
        T = r5_run.CELLS["T"]
        self.assertEqual((T.attn, T.sae_dtype, T.quant, T.regime), ("eager", "fp32", "duan", "none"))
        one = {"T_sdpa": "attn", "T_bf16sae": "sae_dtype", "T_rowquant": "quant", "T_mask": "regime", "T_keep": "regime"}
        for name, field in one.items():
            c = r5_run.CELLS[name]
            diff = [f for f in ("attn", "sae_dtype", "quant", "regime") if getattr(c, f) != getattr(T, f)]
            self.assertEqual(diff, [field], name)
        O = r5_run.CELLS["O"]
        self.assertEqual((O.attn, O.sae_dtype, O.quant, O.regime), ("sdpa", "bf16", "ours", "none"))
        OK = r5_run.CELLS["O_keep"]     # our full pipeline with BOS on their 500k tokens: the cell that links R5 back to our original 98% / 94%
        self.assertEqual((OK.attn, OK.sae_dtype, OK.quant, OK.regime), ("sdpa", "bf16", "ours", "keep"))


if __name__ == "__main__":
    unittest.main()
