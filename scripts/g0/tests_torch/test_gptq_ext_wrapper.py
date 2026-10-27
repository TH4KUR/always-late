"""Task 6 of the GPTQ extension plan: `run_with_weights` leaves the old entry points untouched and replaces their injected quantizer lazily; the lazy quantizer builds each artifact once, from the full-precision model,
refuses to build from a model that is not full precision, and merges its manifest."""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

HERE = Path(__file__).resolve().parent
G0 = HERE.parent
sys.path.insert(0, str(G0)); sys.path.insert(0, str(HERE))
import their_code  # noqa: E402
from gptq_ext import artifact, calibrate as cb, calibration as cal, canonical, run_with_weights as rww, states  # noqa: E402
from gptq_ext_tiny import DIMS, HAVE_TL, tiny_gemma2  # noqa: E402
from test_gptq_ext_calibrate import make_pool  # noqa: E402

BOS, PAD = 1, 0


class Tok:
    bos_token_id, pad_token_id, padding_side = BOS, PAD, "right"


def make_fake_entry(record, requests):
    """A stand-in for an entry point: builds env the way the real ones do (their functions through `their_code.build_namespace`), restores, then quantizes the requested (restore_first, bits) sequence."""
    def main(argv):
        ex = their_code.extract_functions(their_code.PRISTINE, ["target_tl_weight", "quantize_rtn_per_output_channel_tl", "clone_state_to_cpu"])
        env = {"torch": torch, "tqdm": lambda it, **k: it}
        their_code.build_namespace(ex, env)
        model = tiny_gemma2(); model.tokenizer = Tok()
        record["events"] = []
        env["model_work"] = model; env["original_state_cpu"] = env["clone_state_to_cpu"](model); record["events"].append("cloned")
        restore = lambda: model.load_state_dict(env["original_state_cpu"], strict=True)
        record["env"], record["model"], record["argv"] = env, model, list(argv)
        for restore_first, bits in requests:
            if restore_first:
                restore()
            record["events"].append(f"quantize {bits}")
            record.setdefault("returns", []).append(env["quantize_rtn_per_output_channel_tl"](model, bits=bits))
        return "entry-result"
    return main


@unittest.skipUnless(HAVE_TL, "needs TransformerLens (~/.venvs/anlp-tl)")
class Wrapper(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.dir = Path(self.tmp.name)
        pool = make_pool(); files = cal.write_text_files(pool, self.dir / "text")
        (self.dir / "cal.json").write_text(json.dumps({"pool_sha256": cal.pool_sha256(pool), "languages": {c: {"shortfall": r["shortfall"], "text_file": files[c], "passages": [{k: p[k] for k in ("article_id", "rank", "n_tokens", "n_used", "sha256")} for p in r["passages"]]} for c, r in pool.items()}}))
        self.weights = ["--weights", "rtn-matched", "--calibration-freeze", str(self.dir / "cal.json"), "--calibration-text", str(self.dir / "text"), "--record-weights-hashes", "--calibration-batch-size", "8"]
        self.orig_bn = their_code.build_namespace; self.record = {}

    def tearDown(self):
        their_code.build_namespace = self.orig_bn; sys.modules.pop("fake_entry", None); self.tmp.cleanup()

    def run_wrapper(self, requests, extra=(), weights=None, out=None, entry_extra=()):
        sys.modules["fake_entry"] = types.SimpleNamespace(main=make_fake_entry(self.record, requests))
        out = out or str(self.dir / "out")
        return rww.main(["--entry", "fake_entry:main"] + (weights if weights is not None else self.weights) + list(extra) + ["--", "--out", out] + list(entry_extra)), out

    def test_the_entry_point_gets_its_own_arguments_unchanged_and_its_result_is_returned(self):
        res, out = self.run_wrapper([(True, 4)], entry_extra=["--states", "4", "6"])
        self.assertEqual(res, "entry-result")
        self.assertEqual(self.record["argv"], ["--out", out, "--states", "4", "6"])

    def test_weights_rtn_patches_nothing(self):
        before = their_code.build_namespace
        res, _ = self.run_wrapper([(True, 4)], weights=["--weights", "rtn"])
        self.assertIs(their_code.build_namespace, before)
        self.assertNotIsInstance(self.record["env"]["quantize_rtn_per_output_channel_tl"], states.LazyQuantizer)

    def test_the_first_request_builds_the_artifact_applies_it_and_writes_the_manifest(self):
        res, out = self.run_wrapper([(True, 4)])
        model = self.record["model"]
        m = json.loads((Path(out) / "weights_manifest.json").read_text())
        self.assertEqual((m["method"], m["bits"]), ("rtn_matched", [4]))
        tensors, _ = cb.calibrate(tiny_gemma2(), cal.pool_batches(make_pool(), BOS, PAD, batch_size=8), cb.GptqConfig(bits=4), "rtn_matched", torch.device("cpu"), log=lambda *a: None)
        self.assertEqual(artifact.artifact_sha256(tensors), m["artifact_sha256"]["4"])
        self.assertTrue(all(torch.equal(dict(model.named_parameters())[n], tensors[n]) for n in tensors))
        self.assertEqual(self.record["returns"][0][0], 7 * model.cfg.n_layers)

    def test_each_artifact_is_built_once_per_job_and_later_bit_widths_merge_into_the_manifest(self):
        calls = []
        real = cb.calibrate
        def counting(*a, **k):
            calls.append(a[1] and k.get("log") is not None); return real(*a, **k)
        with mock.patch.object(cb, "calibrate", counting):
            res, out = self.run_wrapper([(True, 4), (True, 4), (True, 6), (True, 4)])
        self.assertEqual(len(calls), 2)                                          # 4 bits once, 6 bits once, never again
        self.assertEqual(json.loads((Path(out) / "weights_manifest.json").read_text())["bits"], [4, 6])

    def test_the_floors_stay_legacy_rounding_and_other_bit_widths_raise(self):
        res, out = self.run_wrapper([(True, 16), (True, 12)])
        self.assertEqual(self.record["returns"], [(14, self.record["returns"][0][1]), (14, self.record["returns"][0][1])])
        self.assertFalse((Path(out) / "weights_manifest.json").exists())
        with self.assertRaises(ValueError):
            self.run_wrapper([(True, 3)])
        with self.assertRaises(ValueError):
            self.run_wrapper([(True, 0)])

    def test_building_from_a_model_that_is_not_full_precision_is_refused(self):
        with self.assertRaises(RuntimeError):
            self.run_wrapper([(True, 4), (False, 6)])                            # no restore between the states: 6 bits would be calibrated on the 4-bit weights

    def test_the_artifacts_of_the_requested_states_are_built_right_after_the_model_is_cloned_before_any_work(self):
        calls = []
        real = cb.calibrate
        def counting(*a, **k):
            calls.append(self.record["events"][:]); return real(*a, **k)
        sys.modules["fake_entry"] = types.SimpleNamespace(main=make_fake_entry(self.record, [(True, 4), (True, 6), (True, 4)]))
        with mock.patch.object(cb, "calibrate", counting):
            rww.main(["--entry", "fake_entry:main"] + self.weights + ["--", "--out", str(self.dir / "pb"), "--states", "FP", "4", "6", "16"])
        self.assertEqual(len(calls), 2)                                          # one build per artifact bit width, none of the floors
        self.assertEqual(calls, [[], []])                                        # both happened inside clone_state_to_cpu: before 'cloned' was recorded and before any quantize
        self.assertEqual(self.record["events"][0], "cloned")
        m = json.loads((self.dir / "pb" / "weights_manifest.json").read_text())
        self.assertEqual(m["bits"], [4, 6])

    def test_a_frozen_hash_mismatch_stops_the_job_before_the_entry_point_does_any_work(self):
        fz = self.dir / "gfreeze.json"; fz.write_text(json.dumps({"artifact_sha256": {"rtn_matched": {"4": "0" * 64}}}))
        sys.modules["fake_entry"] = types.SimpleNamespace(main=make_fake_entry(self.record, [(True, 4)]))
        with self.assertRaises(artifact.ArtifactHashMismatch):
            rww.main(["--entry", "fake_entry:main"] + self.weights + ["--gptq-freeze", str(fz), "--", "--out", str(self.dir / "mm"), "--states", "4"])
        self.assertEqual(self.record["events"], [])                              # the entry point never got past its model clone

    def test_floor_only_states_build_nothing_and_write_no_manifest(self):
        calls = []
        real = cb.calibrate
        with mock.patch.object(cb, "calibrate", lambda *a, **k: calls.append(1) or real(*a, **k)):
            res, out = self.run_wrapper([(True, 16), (True, 12)], entry_extra=["--states", "FP", "16", "12"])
        self.assertEqual(calls, [])
        self.assertFalse((Path(out) / "weights_manifest.json").exists())

    def test_without_a_state_hint_the_build_stays_lazy(self):
        calls = []
        real = cb.calibrate
        def counting(*a, **k):
            calls.append(self.record["events"][:]); return real(*a, **k)
        sys.modules["fake_entry"] = types.SimpleNamespace(main=make_fake_entry(self.record, [(True, 4)]))
        with mock.patch.object(cb, "calibrate", counting):
            rww.main(["--entry", "fake_entry:main"] + self.weights + ["--", "--out", str(self.dir / "lz")])
        self.assertEqual(calls, [["cloned", "quantize 4"]])

    def test_an_entry_point_without_the_original_state_is_refused(self):
        def main(argv):
            env = {"torch": torch}; their_code.build_namespace(their_code.extract_functions(their_code.PRISTINE, ["target_tl_weight", "quantize_rtn_per_output_channel_tl"]), env)
            model = tiny_gemma2(); model.tokenizer = Tok()
            env["quantize_rtn_per_output_channel_tl"](model, bits=4)
        sys.modules["fake_entry"] = types.SimpleNamespace(main=main)
        with self.assertRaises(RuntimeError):
            rww.main(["--entry", "fake_entry:main"] + self.weights + ["--", "--out", str(self.dir / "o")])

    def test_a_second_job_in_the_same_directory_with_another_pool_is_refused(self):
        _, out = self.run_wrapper([(True, 4)])
        fp = Path(out) / "weights_manifest.json"; m = json.loads(fp.read_text()); m["pool_sha256"] = "2" * 64; fp.write_text(json.dumps(m))
        with self.assertRaises(ValueError):
            self.run_wrapper([(True, 4)], out=out)

    def test_missing_pieces_stop_the_wrapper_before_the_entry_point_runs(self):
        sys.modules["fake_entry"] = types.SimpleNamespace(main=make_fake_entry(self.record, [(True, 4)]))
        with self.assertRaises(SystemExit):
            rww.main(["--entry", "fake_entry:main", "--weights", "gptq", "--", "--out", str(self.dir / "o")])      # no calibration files
        with self.assertRaises(SystemExit):
            rww.main(["--entry", "fake_entry:main", "--weights", "gptq", "--calibration-freeze", "a", "--calibration-text", "b", "--", "--states", "4"])   # no --out
        with self.assertRaises(SystemExit):
            rww.main(["--entry", "fake_entry:main", "--weights", "rtn"])                                                                        # no separator
        self.assertEqual(self.record, {})


class Argv(unittest.TestCase):
    def test_bit_hint_reads_bits_precisions_and_states_of_the_entry_points(self):
        self.assertEqual(rww.bits_hint(["--out", "o", "--bits", "8", "7", "16", "0"]), [0, 7, 8, 16])
        self.assertEqual(rww.bits_hint(["--precisions", "FP", "6", "4", "--split", "eval"]), [4, 6])
        self.assertEqual(rww.bits_hint(["--states", "FP", "4", "INT6", "--chunk-features", "25"]), [4, 6])
        self.assertEqual(rww.bits_hint(["--out", "o"]), [])

    def test_split_and_out(self):
        self.assertEqual(rww.split_argv(["--a", "1", "--", "--out", "x"]), (["--a", "1"], ["--out", "x"]))
        self.assertEqual(rww.out_dir_of(["--flores-root", "r", "--out", "o", "--states", "4"]), "o")
        self.assertIsNone(rww.out_dir_of(["--states", "4"]))


if __name__ == "__main__":
    unittest.main()
