import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phase1_llama_sae as ls

D_IN, D_SAE, K = 6, 20, 4
CFG = {"architecture": "topk", "d_in": D_IN, "d_sae": D_SAE, "dtype": "torch.bfloat16", "hook_name": "blocks.12.hook_resid_pre", "hook_layer": 12, "activation_fn_str": "topk",
       "activation_fn_kwargs": {"k": K}, "apply_b_dec_to_input": True, "finetuning_scaling_factor": False, "prepend_bos": True, "normalize_activations": "none", "context_size": 512}


def weights(seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    return {"W_enc": torch.randn(D_IN, D_SAE, generator=g).to(dtype), "W_dec": torch.randn(D_SAE, D_IN, generator=g).to(dtype), "b_enc": torch.randn(D_SAE, generator=g).to(dtype) * 0.1,
            "b_dec": torch.randn(D_IN, generator=g).to(dtype) * 0.1}


def write(d, cfg=None, w=None):
    d = Path(d); d.mkdir(parents=True, exist_ok=True)
    (d / "cfg.json").write_text(json.dumps(cfg or CFG)); save_file(w or weights(), str(d / "sae_weights.safetensors"))
    return d


def reference_encode(x, w, k):
    """Straight-line reference: ReLU of the decoder-bias-centred projection, then keep the k largest of every token."""
    pre = torch.relu((x - w["b_dec"]) @ w["W_enc"] + w["b_enc"])
    out = torch.zeros_like(pre)
    for i in range(pre.shape[0]):
        idx = torch.argsort(pre[i], descending=True)[:k]
        out[i, idx] = pre[i, idx]
    return out


class Encode(unittest.TestCase):
    def setUp(self):
        self.w = weights(1); self.tmp = tempfile.TemporaryDirectory(); self.sae = ls.load_topk_sae(write(self.tmp.name, w=self.w))

    def tearDown(self):
        self.tmp.cleanup()

    def test_encode_matches_the_reference_formula(self):
        x = torch.randn(30, D_IN)
        torch.testing.assert_close(self.sae.encode(x), reference_encode(x, self.w, K))

    def test_at_most_k_features_are_active_per_token_and_exactly_k_when_enough_are_positive(self):
        x = torch.randn(200, D_IN) * 3
        f = self.sae.encode(x); active = (f > 0).sum(1)
        self.assertTrue(bool((active <= K).all())); self.assertTrue(bool((active == K).any()))
        pre = (x - self.w["b_dec"]) @ self.w["W_enc"] + self.w["b_enc"]; positive = (pre > 0).sum(1)
        self.assertTrue(torch.equal(active, torch.minimum(positive, torch.tensor(K))))

    def test_a_token_with_fewer_than_k_positive_pre_activations_keeps_only_those(self):
        w = {k: v.clone() for k, v in self.w.items()}; w["b_enc"] = torch.full((D_SAE,), -1000.0); w["b_enc"][:2] = 1000.0
        with tempfile.TemporaryDirectory() as d:
            sae = ls.load_topk_sae(write(d, w=w))
            f = sae.encode(torch.randn(5, D_IN))
            self.assertTrue(bool(((f > 0).sum(1) == 2).all())); self.assertTrue(bool((f[:, 2:] == 0).all()))

    def test_active_values_are_the_relu_pre_activations_not_rescaled(self):
        x = torch.randn(10, D_IN); f = self.sae.encode(x); pre = torch.relu((x - self.w["b_dec"]) @ self.w["W_enc"] + self.w["b_enc"])
        mask = f > 0
        torch.testing.assert_close(f[mask], pre[mask])

    def test_leading_dimensions_are_kept(self):
        x = torch.randn(3, 7, D_IN); f = self.sae.encode(x)
        self.assertEqual(tuple(f.shape), (3, 7, D_SAE)); torch.testing.assert_close(f.reshape(21, D_SAE), self.sae.encode(x.reshape(21, D_IN)))

    def test_a_bf16_input_is_computed_in_float32_and_returns_float32(self):
        x = torch.randn(4, D_IN).to(torch.bfloat16); f = self.sae.encode(x)
        self.assertEqual(f.dtype, torch.float32); torch.testing.assert_close(f, self.sae.encode(x.float()))

    def test_decode_is_the_linear_map_with_the_decoder_bias(self):
        f = torch.rand(5, D_SAE); torch.testing.assert_close(self.sae.decode(f), f @ self.w["W_dec"] + self.w["b_dec"])

    def test_the_error_term_is_what_the_sae_does_not_explain(self):
        x = torch.randn(8, D_IN); f = self.sae.encode(x); err = x - self.sae.decode(f)
        torch.testing.assert_close(self.sae.decode(f) + err, x)

    def test_the_decoder_matrix_is_exposed_as_w_dec(self):
        self.assertEqual(tuple(self.sae.W_dec.shape), (D_SAE, D_IN)); torch.testing.assert_close(self.sae.W_dec.detach(), self.w["W_dec"])


class Loading(unittest.TestCase):
    def test_bf16_weights_are_loaded_as_float32(self):
        with tempfile.TemporaryDirectory() as d:
            w = weights(2, torch.bfloat16); sae = ls.load_topk_sae(write(d, w=w))
            for name in ("W_enc", "W_dec", "b_enc", "b_dec"):
                p = getattr(sae, name); self.assertEqual(p.dtype, torch.float32); torch.testing.assert_close(p.detach(), w[name].float())

    def test_the_config_is_exposed_with_the_sizes_the_hook_and_k(self):
        with tempfile.TemporaryDirectory() as d:
            sae = ls.load_topk_sae(write(d))
            self.assertEqual((sae.cfg.d_in, sae.cfg.d_sae, sae.k), (D_IN, D_SAE, K)); self.assertEqual(sae.cfg.hook_name, "blocks.12.hook_resid_pre")

    def test_a_hook_alias_replaces_the_name_the_frozen_code_sees_and_keeps_the_original(self):
        with tempfile.TemporaryDirectory() as d:
            sae = ls.load_topk_sae(write(d), hook_alias="blocks.11.hook_resid_post")
            self.assertEqual(sae.cfg.hook_name, "blocks.11.hook_resid_post"); self.assertEqual(sae.cfg.hook_name_original, "blocks.12.hook_resid_pre")

    def test_an_expected_hook_that_differs_from_the_config_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "hook"):
                ls.load_topk_sae(write(d), expected_hook="blocks.6.hook_resid_pre")
            ls.load_topk_sae(write(Path(d) / "ok"), expected_hook="blocks.12.hook_resid_pre")

    def test_unsupported_configurations_are_refused_not_silently_mis_encoded(self):
        bad = [("architecture", "jumprelu"), ("apply_b_dec_to_input", False), ("normalize_activations", "expected_average_only_in"), ("prepend_bos", False)]
        for key, value in bad:
            with tempfile.TemporaryDirectory() as d:
                with self.assertRaises(ValueError, msg=key):
                    ls.load_topk_sae(write(d, cfg=dict(CFG, **{key: value})))

    def test_a_missing_k_or_a_shape_that_disagrees_with_the_config_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                ls.load_topk_sae(write(d, cfg=dict(CFG, activation_fn_kwargs={})))
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                ls.load_topk_sae(write(d, cfg=dict(CFG, d_sae=D_SAE + 1)))
        with tempfile.TemporaryDirectory() as d:
            w = weights(); del w["b_dec"]
            with self.assertRaises(ValueError):
                ls.load_topk_sae(write(d, w=w))
        with tempfile.TemporaryDirectory() as d:
            w = weights(); w["W_dec"] = w["W_dec"].T.contiguous()                                                  # a transposed decoder
            with self.assertRaises(ValueError):
                ls.load_topk_sae(write(d, w=w))

    def test_k_larger_than_the_dictionary_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                ls.load_topk_sae(write(d, cfg=dict(CFG, activation_fn_kwargs={"k": D_SAE + 1})))

    def test_the_model_is_in_eval_mode_and_has_no_gradients(self):
        with tempfile.TemporaryDirectory() as d:
            sae = ls.load_topk_sae(write(d)); self.assertFalse(sae.training); self.assertTrue(all(not p.requires_grad for p in sae.parameters()))

    def test_a_directory_without_the_files_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                ls.load_topk_sae(Path(d))


class Compat(unittest.TestCase):
    def test_the_frozen_runners_loader_signature_resolves_a_local_directory_and_moves_to_the_device(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); write(root / "folder")
            sae = ls.load_sae_compat_llama(str(root), "folder", "cpu", hook_alias="blocks.11.hook_resid_post")
            self.assertEqual(sae.cfg.hook_name, "blocks.11.hook_resid_post"); self.assertEqual(sae.W_dec.device.type, "cpu")
            self.assertEqual(ls.load_sae_compat_llama(str(root), "folder", "meta").W_dec.device.type, "meta")           # the device argument is honoured

    def test_the_compat_loader_refuses_a_missing_folder(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                ls.load_sae_compat_llama(d, "nothing", "cpu")


if __name__ == "__main__":
    unittest.main()
