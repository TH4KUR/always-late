"""Task 8 of the GPTQ extension plan, the CPU-testable core of the build-and-gate job (`gate_run.run_gates`) on a tiny Gemma-2: gates pass for correct artifacts, and each gate fails when the thing it guards is broken."""
import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

HERE = Path(__file__).resolve().parent
G0 = HERE.parent
sys.path.insert(0, str(G0)); sys.path.insert(0, str(HERE))
import their_code  # noqa: E402
from gptq_ext import artifact, calibrate as cb, calibration as cal, gate_run, states  # noqa: E402
from gptq_ext_tiny import DIMS, HAVE_TL, tiny_gemma2  # noqa: E402
from test_gptq_ext_calibrate import make_pool  # noqa: E402

BOS, PAD = 1, 0


@unittest.skipUnless(HAVE_TL, "needs TransformerLens (~/.venvs/anlp-tl)")
class Gates(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); d = Path(self.tmp.name)
        pool = make_pool(); files = cal.write_text_files(pool, d / "text")
        calfreeze = {"pool_sha256": cal.pool_sha256(pool), "languages": {c: {"shortfall": r["shortfall"], "text_file": files[c], "passages": [{k: p[k] for k in ("article_id", "rank", "n_tokens", "n_used", "sha256")} for p in r["passages"]]} for c, r in pool.items()}}
        (d / "cal.json").write_text(json.dumps(calfreeze))
        ap = argparse.ArgumentParser(); states.add_weights_args(ap)
        self.args = ap.parse_args(["--calibration-freeze", str(d / "cal.json"), "--calibration-text", str(d / "text"), "--record-weights-hashes", "--calibration-batch-size", "8"])
        self.model = tiny_gemma2(); self.original = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        ns = their_code.build_namespace(their_code.extract_functions(their_code.PRISTINE, ["target_tl_weight", "quantize_rtn_per_output_channel_tl"]), {"torch": torch})
        self.env = {"quantize_rtn_per_output_channel_tl": ns["quantize_rtn_per_output_channel_tl"], "restore_model_work": lambda: self.model.load_state_dict(self.original, strict=True)}
        g = torch.Generator().manual_seed(3)
        self.probe = {i: [torch.randint(3, DIMS["d_vocab"], (int(torch.randint(8, 20, (1,), generator=g)),), generator=g).tolist() for _ in range(10)] for i in range(2)}

    def tearDown(self):
        self.tmp.cleanup()

    def run_it(self, **kw):
        return gate_run.run_gates(self.env, self.model, self.probe, BOS, PAD, kw.pop("methods", ["gptq", "rtn-matched"]), kw.pop("bits", [4, 6]), self.args, lambda *a: None, **kw)

    def test_all_gates_pass_for_correct_artifacts_and_the_report_is_complete(self):
        r = self.run_it()
        self.assertTrue(r["all_gates_ok"], r["gates"])
        self.assertEqual(sorted(r["nll"]), sorted(["FP", "legacy_rtn_4", "legacy_rtn_6", "gptq_4", "gptq_6", "rtn-matched_4", "rtn-matched_6"]))
        self.assertTrue(r["rebuild"]["equal"])
        self.assertEqual(set(r["manifests"]), {"gptq", "rtn-matched"})
        for k in ("gptq_4", "gptq_6", "rtn-matched_4", "rtn-matched_6"):
            self.assertEqual(r["gates"][k], {"non_targets_unchanged": True, "targets_changed": True, "restore_exact": True})
        self.assertTrue(r["gates"]["fp_logits_bit_identical_after_all_switches"])

    def test_nll_has_one_entry_per_probe_language_and_quantization_does_not_lower_it_to_nonsense(self):
        r = self.run_it(methods=["rtn-matched"], bits=[4])
        for k, v in r["nll"].items():
            self.assertEqual(sorted(v["per_language"]), [0, 1], k)
            self.assertGreater(v["tokens"], 0)
            self.assertTrue(torch.isfinite(torch.tensor(v["mean"])), k)
        self.assertEqual(r["nll"]["FP"]["tokens"], r["nll"]["rtn-matched_4"]["tokens"])

    def test_a_restore_that_leaves_a_residue_fails_each_restore_gate(self):
        calls = [0]
        def leaky_restore():
            self.model.load_state_dict(self.original, strict=True)
            if calls[0] > 0:
                with torch.no_grad():
                    self.model.embed.W_E.add_(1e-3)
            calls[0] += 1
        self.env["restore_model_work"] = leaky_restore
        r = self.run_it(methods=["rtn-matched"], bits=[4])
        self.assertFalse(r["all_gates_ok"])
        self.assertFalse(r["gates"]["rtn-matched_4"]["restore_exact"])
        self.assertFalse(r["gates"]["fp_logits_bit_identical_after_all_switches"])
        self.assertFalse(r["gates"]["fp_state_exact_at_the_end"])

    def test_every_quantized_state_changes_the_loss_relative_to_full_precision(self):
        r = self.run_it(bits=[4])
        fp = r["nll"]["FP"]["mean"]
        for k in ("legacy_rtn_4", "gptq_4", "rtn-matched_4"):
            self.assertNotEqual(r["nll"][k]["mean"], fp, k)

    def test_an_apply_that_also_changes_a_non_target_tensor_fails_the_gate(self):
        real = artifact.apply
        def bad_apply(model, tensors):
            real(model, tensors)
            with torch.no_grad():
                model.embed.W_E.add_(1.0)
        with mock.patch.object(artifact, "apply", bad_apply):
            r = self.run_it(methods=["rtn-matched"], bits=[4])
        self.assertFalse(r["gates"]["rtn-matched_4"]["non_targets_unchanged"])
        self.assertFalse(r["all_gates_ok"])

    def test_an_apply_that_changes_nothing_fails_the_gate(self):
        with mock.patch.object(artifact, "apply", lambda model, tensors: None):
            r = self.run_it(methods=["rtn-matched"], bits=[4])
        self.assertFalse(r["gates"]["rtn-matched_4"]["targets_changed"])
        self.assertFalse(r["all_gates_ok"])

    def test_a_second_build_with_another_hash_fails_the_job(self):
        real, calls = cb.calibrate, []
        def flaky(*a, **k):
            tensors, rep = real(*a, **k); calls.append(1)
            if len(calls) == 2:                                                  # the rebuild (the first call is the install)
                n = next(iter(tensors)); tensors[n] = tensors[n] + 1.0
            return tensors, rep
        with mock.patch.object(cb, "calibrate", flaky):
            r = self.run_it(methods=["rtn-matched"], bits=[4])
        self.assertFalse(r["rebuild"]["equal"])
        self.assertFalse(r["all_gates_ok"])

    def test_the_model_is_exactly_full_precision_when_the_gates_finish(self):
        self.run_it()
        self.assertTrue(all(torch.equal(v, self.original[k]) for k, v in self.model.state_dict().items()))

    def test_manifests_are_written_per_method_when_an_output_directory_is_given(self):
        out = Path(self.tmp.name) / "out"
        self.run_it(methods=["gptq"], bits=[4], out_dir=out)
        self.assertTrue((out / "gptq" / "weights_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
