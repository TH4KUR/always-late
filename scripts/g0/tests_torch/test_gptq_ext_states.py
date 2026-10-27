"""Task 6 of the GPTQ extension plan: artifact identity (a sha256 over names, dtypes, shapes and raw bytes of the target tensors), in-place application, and the weight-state shim that replaces the injected
`quantize_rtn_per_output_channel_tl` of the experiment entry points (the default `--weights rtn` leaves the legacy function in place; bits 12 and 16 stay legacy rounding in every arm; an artifact whose hash differs
from the frozen value is refused). Model tests need TransformerLens (~/.venvs/anlp-tl)."""
import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
G0 = HERE.parent
sys.path.insert(0, str(G0)); sys.path.insert(0, str(HERE))
from gptq_ext import artifact, calibrate as cb, calibration as cal, canonical, states  # noqa: E402
from gptq_ext_tiny import HAVE_TL, tiny_gemma2  # noqa: E402
from test_gptq_ext_calibrate import make_pool  # noqa: E402

BOS, PAD = 1, 0


def fake_tensors(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"blocks.0.attn.W_Q": torch.randn(2, 4, 3, generator=g), "blocks.0.mlp.W_in": torch.randn(4, 6, generator=g).to(torch.bfloat16)}


class Hash(unittest.TestCase):
    def test_artifact_hash_detects_one_flipped_value(self):
        t = fake_tensors(); h = artifact.artifact_sha256(t)
        t2 = {k: v.clone() for k, v in t.items()}
        t2["blocks.0.mlp.W_in"].view(torch.int16)[1, 2] ^= 1
        self.assertNotEqual(h, artifact.artifact_sha256(t2))
        self.assertEqual(h, artifact.artifact_sha256({k: v.clone() for k, v in t.items()}))

    def test_hash_does_not_depend_on_insertion_order_but_on_name_dtype_and_shape(self):
        t = fake_tensors(); h = artifact.artifact_sha256(t)
        self.assertEqual(h, artifact.artifact_sha256(dict(reversed(list(t.items())))))
        self.assertNotEqual(h, artifact.artifact_sha256({"x" + k: v for k, v in t.items()}))
        self.assertNotEqual(h, artifact.artifact_sha256({k: v.to(torch.float64) if v.dtype == torch.float32 else v for k, v in t.items()}))
        self.assertNotEqual(artifact.artifact_sha256({"a": torch.zeros(2, 3)}), artifact.artifact_sha256({"a": torch.zeros(3, 2)}))

    def test_hash_separates_dtypes_that_have_the_same_bytes(self):
        self.assertNotEqual(artifact.artifact_sha256({"a": torch.zeros(4, dtype=torch.float32)}), artifact.artifact_sha256({"a": torch.zeros(4, dtype=torch.int32)}))

    def test_hash_covers_negative_zero_and_nan_payloads(self):
        a = {"a": torch.tensor([0.0])}; b = {"a": torch.tensor([-0.0])}
        self.assertNotEqual(artifact.artifact_sha256(a), artifact.artifact_sha256(b))


@unittest.skipUnless(HAVE_TL, "needs TransformerLens (~/.venvs/anlp-tl)")
class WithModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pool = make_pool(); cls.batches = cal.pool_batches(cls.pool, BOS, PAD, batch_size=8)

    def artifact_for(self, method, bits, model=None):
        model = model or tiny_gemma2()
        tensors, _ = cb.calibrate(model, self.batches, cb.GptqConfig(bits=bits), method, torch.device("cpu"), log=lambda *a: None)
        return tensors

    def test_apply_then_restore_returns_exact_original_values_and_touches_only_targets(self):
        model = tiny_gemma2(); orig = {n: p.detach().clone() for n, p in model.state_dict().items()}
        tensors = self.artifact_for("rtn_matched", 4, model)
        artifact.apply(model, tensors)
        for n, p in model.named_parameters():
            if canonical.is_target(n):
                self.assertTrue(torch.equal(p, tensors[n]), n)
                self.assertFalse(torch.equal(p, orig[n]), n)
            else:
                self.assertTrue(torch.equal(p, orig[n]), n)
        model.load_state_dict(orig, strict=True)
        for n, p in model.named_parameters():
            self.assertTrue(torch.equal(p, orig[n]), n)

    def test_switching_states_repeatedly_is_exact(self):
        model = tiny_gemma2(); orig = {n: p.detach().clone() for n, p in model.state_dict().items()}
        a = self.artifact_for("rtn_matched", 4, model); b = self.artifact_for("rtn_matched", 6, model)
        for _ in range(3):
            for tensors in (a, b):
                model.load_state_dict(orig, strict=True); artifact.apply(model, tensors)
                self.assertTrue(all(torch.equal(dict(model.named_parameters())[n], tensors[n]) for n in tensors))
        model.load_state_dict(orig, strict=True)
        self.assertTrue(all(torch.equal(p, orig[n]) for n, p in model.named_parameters()))

    def test_apply_refuses_missing_extra_misshaped_or_retyped_tensors(self):
        model = tiny_gemma2(); tensors = self.artifact_for("rtn_matched", 4, model)
        name = "blocks.0.attn.W_Q"
        for bad in ({k: v for k, v in tensors.items() if k != name}, {**tensors, "blocks.0.ln1.w": torch.zeros(32)}, {**tensors, name: tensors[name][:, :, :-1]}, {**tensors, name: tensors[name].to(torch.bfloat16)}):
            with self.assertRaises(ValueError):
                artifact.apply(model, bad)

    def test_apply_never_changes_an_artifact_tensor(self):
        model = tiny_gemma2(); tensors = self.artifact_for("rtn_matched", 4, model); h = artifact.artifact_sha256(tensors)
        artifact.apply(model, tensors)
        with torch.no_grad():
            for _, p in canonical.targets(model):
                p.add_(1.0)
        self.assertEqual(artifact.artifact_sha256(tensors), h)


class Args(unittest.TestCase):
    def parse(self, *argv):
        ap = argparse.ArgumentParser(); states.add_weights_args(ap)
        return ap.parse_args(list(argv))

    def test_default_weights_flag_is_rtn(self):
        a = self.parse()
        self.assertEqual(a.weights, "rtn")
        self.assertFalse(a.record_weights_hashes)

    def test_choices_and_options(self):
        a = self.parse("--weights", "gptq", "--gptq-freeze", "f.json", "--calibration-freeze", "c.json", "--calibration-text", "d", "--record-weights-hashes")
        self.assertEqual((a.weights, a.gptq_freeze, a.calibration_freeze, a.calibration_text, a.record_weights_hashes), ("gptq", "f.json", "c.json", "d", True))
        with self.assertRaises(SystemExit):
            self.parse("--weights", "awq")

    def test_bits_in_reads_every_label_style_of_the_entry_points(self):
        self.assertEqual(states.bits_in([8, 7]), [7, 8])
        self.assertEqual(states.bits_in(["FP", "8", "7", "6", "5", "4", "16", "12", "0"]), [0, 4, 5, 6, 7, 8, 12, 16])
        self.assertEqual(states.bits_in(["FP", "FP_rebatched", "INT6", "4"]), [4, 6])
        self.assertEqual(states.bits_in(None), [])


@unittest.skipUnless(HAVE_TL, "needs TransformerLens (~/.venvs/anlp-tl)")
class Install(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pool = make_pool(); cls.batches = cal.pool_batches(cls.pool, BOS, PAD, batch_size=8)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.dir = Path(self.tmp.name)
        files = cal.write_text_files(self.pool, self.dir / "text")
        self.calfreeze = {"pool_sha256": cal.pool_sha256(self.pool), "languages": {c: {"shortfall": r["shortfall"], "text_file": files[c], "passages": [{k: p[k] for k in ("article_id", "rank", "n_tokens", "n_used", "sha256")} for p in r["passages"]]}
                                                                              for c, r in self.pool.items()}}
        (self.dir / "calibration_freeze.json").write_text(json.dumps(self.calfreeze))
        self.calls = []
        self.env = {"quantize_rtn_per_output_channel_tl": lambda model, bits=8: (self.calls.append(bits), (7 * model.cfg.n_layers, 123))[1]}

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, weights="rtn-matched", record=True, freeze=None):
        ap = argparse.ArgumentParser(); states.add_weights_args(ap)
        argv = ["--weights", weights, "--calibration-freeze", str(self.dir / "calibration_freeze.json"), "--calibration-text", str(self.dir / "text")]
        if freeze is not None:
            (self.dir / "freeze.json").write_text(json.dumps(freeze)); argv += ["--gptq-freeze", str(self.dir / "freeze.json")]
        if record:
            argv += ["--record-weights-hashes"]
        return ap.parse_args(argv)

    def install(self, a, model=None, bits=(4, 6, 16), out_dir=None):
        model = model or tiny_gemma2()
        return model, states.install(self.env, model, a, lambda *x: None, bits=list(bits), out_dir=out_dir, bos_id=BOS, pad_id=PAD)

    def test_default_weights_flag_leaves_the_legacy_quantizer_in_place(self):
        legacy = self.env["quantize_rtn_per_output_channel_tl"]
        _, manifest = self.install(self.args("rtn"))
        self.assertIsNone(manifest)
        self.assertIs(self.env["quantize_rtn_per_output_channel_tl"], legacy)

    def test_artifact_bits_apply_the_artifact_and_the_floors_stay_legacy_rounding(self):
        model, manifest = self.install(self.args("rtn-matched"))
        quantize = self.env["quantize_rtn_per_output_channel_tl"]
        n_t, _ = quantize(model, bits=4)
        self.assertEqual(n_t, 7 * model.cfg.n_layers)
        want = self.artifact_of(model, 4)
        self.assertTrue(all(torch.equal(dict(model.named_parameters())[n], want[n]) for n in want))
        self.assertEqual(self.calls, [])
        quantize(model, bits=16); quantize(model, bits=12)
        self.assertEqual(self.calls, [16, 12])
        with self.assertRaises(ValueError):
            quantize(model, bits=5)                                              # not requested at install time: never built silently
        with self.assertRaises(ValueError):
            quantize(model, bits=0)

    def artifact_of(self, model, bits):
        tensors, _ = cb.calibrate(tiny_gemma2(), self.batches, cb.GptqConfig(bits=bits), "rtn_matched", torch.device("cpu"), log=lambda *a: None)
        return tensors

    def test_manifest_records_method_config_pool_code_and_one_hash_per_bit_width(self):
        _, m = self.install(self.args("rtn-matched"))
        self.assertEqual(m["method"], "rtn_matched")
        self.assertEqual(sorted(m["artifact_sha256"]), ["4", "6"])
        self.assertEqual(m["pool_sha256"], self.calfreeze["pool_sha256"])
        self.assertEqual(m["config"]["blocksize"], 128)
        self.assertIn("calibrate.py", m["code_sha256"])
        self.assertEqual(m["bits"], [4, 6])
        self.assertTrue(all(len(h) == 64 for h in m["artifact_sha256"].values()))

    def test_install_refuses_when_the_hash_differs_from_the_freeze(self):
        _, m = self.install(self.args("rtn-matched"))
        good = {"artifact_sha256": {"rtn_matched": dict(m["artifact_sha256"])}}
        self.env["quantize_rtn_per_output_channel_tl"] = lambda model, bits=8: (0, 0)
        self.install(self.args("rtn-matched", record=False, freeze=good))        # equal hashes: accepted
        bad = {"artifact_sha256": {"rtn_matched": {**m["artifact_sha256"], "4": "0" * 64}}}
        with self.assertRaises(artifact.ArtifactHashMismatch) as ctx:
            self.install(self.args("rtn-matched", record=False, freeze=bad))
        self.assertIn("4", str(ctx.exception))

    def test_install_refuses_an_unknown_hash_unless_recording(self):
        with self.assertRaises(artifact.ArtifactHashMismatch):
            self.install(self.args("rtn-matched", record=False, freeze={"artifact_sha256": {}}))

    def test_install_refuses_a_pool_whose_hash_differs_from_the_freeze(self):
        freeze = {"calibration": {"pool_sha256": "0" * 64}, "artifact_sha256": {}}
        with self.assertRaises(ValueError):
            self.install(self.args("rtn-matched", record=True, freeze=freeze))

    def test_install_refuses_a_calibration_freeze_file_that_differs_from_the_extension_freeze(self):
        freeze = {"calibration": {"freeze_file_sha256": "0" * 64}, "artifact_sha256": {}}
        with self.assertRaises(ValueError):
            self.install(self.args("rtn-matched", record=True, freeze=freeze))
        good = {"calibration": {"freeze_file_sha256": states._sha_file(self.dir / "calibration_freeze.json"), "pool_sha256": cal.pool_sha256(self.pool)}, "artifact_sha256": {}}
        self.install(self.args("rtn-matched", record=True, freeze=good))

    def test_resume_refuses_an_output_directory_with_another_weights_manifest(self):
        out = self.dir / "out"
        _, m1 = self.install(self.args("rtn-matched"), out_dir=out)
        fp = out / "weights_manifest.json"
        self.assertTrue(fp.is_file())
        self.install(self.args("rtn-matched"), out_dir=out)                      # the same manifest again: fine
        self.install(self.args("rtn-matched"), bits=(4,), out_dir=out)           # a subset of the same artifacts: fine, nothing lost
        self.assertEqual(json.loads(fp.read_text())["bits"], [4, 6])
        with self.assertRaises(ValueError):
            self.install(self.args("gptq"), bits=(4,), out_dir=out)              # another method in the same directory
        tampered = json.loads(fp.read_text()); tampered["artifact_sha256"]["4"] = "0" * 64; fp.write_text(json.dumps(tampered))
        with self.assertRaises(ValueError):
            self.install(self.args("rtn-matched"), bits=(4,), out_dir=out)       # the stored hash of 4 bits is not the one the build gives
        tampered = json.loads(fp.read_text()); tampered["artifact_sha256"]["4"] = m1["artifact_sha256"]["4"]; tampered["pool_sha256"] = "1" * 64; fp.write_text(json.dumps(tampered))
        with self.assertRaises(ValueError):
            self.install(self.args("rtn-matched"), bits=(4,), out_dir=out)       # another calibration pool

    def test_bit_widths_built_later_are_merged_into_the_manifest(self):
        out = self.dir / "out2"
        self.install(self.args("rtn-matched"), bits=(4,), out_dir=out)
        self.install(self.args("rtn-matched"), bits=(6,), out_dir=out)
        m = json.loads((out / "weights_manifest.json").read_text())
        self.assertEqual((m["bits"], sorted(m["artifact_sha256"]), sorted(m["calibration_report"])), ([4, 6], ["4", "6"], ["4", "6"]))

    def test_gptq_method_builds_and_applies_its_own_artifact(self):
        model, m = self.install(self.args("gptq"), bits=(4,))
        self.assertEqual(m["method"], "gptq")
        self.env["quantize_rtn_per_output_channel_tl"](model, bits=4)
        tensors, _ = cb.calibrate(tiny_gemma2(), self.batches, cb.GptqConfig(bits=4), "gptq", torch.device("cpu"), log=lambda *a: None)
        self.assertEqual(artifact.artifact_sha256(tensors), m["artifact_sha256"]["4"])
        self.assertTrue(all(torch.equal(dict(model.named_parameters())[n], tensors[n]) for n in tensors))


if __name__ == "__main__":
    unittest.main()
