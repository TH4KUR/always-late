"""Validation of a finished functional-study job of the GPTQ extension (loss task E4, rescue E5, Belebele E7): sacct, GPU line, complete meta with its gates true, nothing skipped for time, the planned precision(s), the
groups manifest and the number of configurations per state equal to the old job's, the output file(s), the sha256 of every copied file, and the weights manifest against the recorded hashes. Remote commands and the copy are injected."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
G0 = HERE.parent
sys.path.insert(0, str(G0))
from gptq_ext import validate_func_job as vf  # noqa: E402

RECORDED = {"gptq": {"4": "a" * 64}, "rtn_matched": {"4": "b" * 64}}
GOOD_META = {"task": "lm", "split": "eval", "preset": "eval", "precisions": ["4"], "n_sentences": 300, "gates": {"split_equals_full": True, "scale_one_identity": True, "donor_self_identity": True, "sham_identity": True},
             "manifest_source_sha256": "m" * 64, "summary": {"INT4": {"seconds": 1.0, "configs": 351}}, "skipped_for_time": [], "complete": True, "peak_gpu_gb": 9.4}


def expect(**kw):
    e = {"precisions": ["4"], "preset": "eval", "split": "eval", "n_sentences": 300, "manifest_source_sha256": "m" * 64, "configs": {"INT4": 351}}
    e.update(kw); return e


class Meta(unittest.TestCase):
    def test_a_good_meta_passes(self):
        self.assertTrue(vf.check_meta(GOOD_META, expect())["ok"])

    def test_each_problem_is_reported(self):
        cases = [({**GOOD_META, "complete": False}, "not complete"), ({**GOOD_META, "gates": {**GOOD_META["gates"], "sham_identity": False}}, "gate"), ({**GOOD_META, "skipped_for_time": ["INT4"]}, "skipped"),
                 ({**GOOD_META, "precisions": ["6"]}, "precisions"), ({**GOOD_META, "preset": "floor"}, "preset"), ({**GOOD_META, "split": "pilot"}, "split"), ({**GOOD_META, "n_sentences": 100}, "sentences"),
                 ({**GOOD_META, "manifest_source_sha256": "z" * 64}, "manifest"), ({**GOOD_META, "summary": {"INT4": {"configs": 350}}}, "configs"), ({**GOOD_META, "gates": {}}, "gate")]
        for meta, word in cases:
            r = vf.check_meta(meta, expect())
            self.assertFalse(r["ok"], word); self.assertTrue(any(word in w for w in r["why"]), (word, r["why"]))

    def test_expectations_left_out_are_not_checked(self):
        self.assertTrue(vf.check_meta({**GOOD_META, "summary": {}}, expect(configs=None))["ok"])
        self.assertTrue(vf.check_meta({**GOOD_META, "manifest_source_sha256": "z"}, expect(manifest_source_sha256=None))["ok"])


REF_F5 = {"task": "f5", "model": "gemma-2-2b", "hook": "blocks.12.hook_resid_post", "n_features": 1487, "per_cell": 12, "subset_seed": 0, "M": 16, "min_ctx": 4, "topk": 50, "extra_pad": 3,
          "dev_sentences_per_language": 997, "n_dev": 26919, "n_eval_per_language": 300, "work_dtype": "float32", "languages": ["eng_Latn", "hin_Deva"], "sample_sha256": "s" * 64, "freeze_sha256": "f" * 64}
GOOD_F5 = {**REF_F5, "states": ["FP", "INT6", "INT4"], "gates": {"logits_native_residual": {"top1": 0.95, "max_abs": 0.28, "ok": True}, "logits_float32_residual": {"top1": 0.95, "max_abs": 0.68, "ok": True}},
           "sham_gate": [{"feature": 15, "exact": True}, {"feature": 54, "exact": True}], "direction_control": False, "fixed_amplitude": False, "state_seconds": {"FP": 1.0, "INT6": 1.0, "INT4": 1.0},
           "complete": True, "peak_gpu_gb": 8.9}


def expect_f5(**kw):
    e = {"states": ["FP", "INT6", "INT4"], "reference": REF_F5}; e.update(kw); return e


class MetaF5(unittest.TestCase):
    """The F5 runner (`phase1_func_f5_run:main`) writes `meta_f5.json`: states instead of precisions, gates as dicts with `ok`, an in-run sham gate, no groups manifest."""

    def test_a_good_f5_meta_passes(self):
        r = vf.check_meta_f5(GOOD_F5, expect_f5()); self.assertTrue(r["ok"], r["why"]); self.assertEqual(r["peak_gpu_gb"], 8.9)

    def test_each_problem_is_reported(self):
        cases = [({**GOOD_F5, "complete": False}, "not complete"),
                 ({**GOOD_F5, "gates": {**GOOD_F5["gates"], "logits_native_residual": {"ok": False}}}, "gate"), ({**GOOD_F5, "gates": {}}, "gate"),
                 ({**GOOD_F5, "sham_gate": [{"feature": 15, "exact": False}]}, "sham"), ({**GOOD_F5, "sham_gate": []}, "sham"), ({k: v for k, v in GOOD_F5.items() if k != "sham_gate"}, "sham"),
                 ({**GOOD_F5, "states": ["FP", "INT6"]}, "states"), ({**GOOD_F5, "direction_control": True}, "direction"), ({**GOOD_F5, "fixed_amplitude": True}, "amplitude"),
                 ({**GOOD_F5, "per_cell": 6}, "per_cell"), ({**GOOD_F5, "sample_sha256": "z" * 64}, "sample_sha256"), ({**GOOD_F5, "freeze_sha256": "z" * 64}, "freeze_sha256"),
                 ({**GOOD_F5, "languages": ["eng_Latn"]}, "languages"), ({**GOOD_F5, "n_features": 1000}, "n_features"), ({**GOOD_F5, "work_dtype": "bfloat16"}, "work_dtype"),
                 ({**GOOD_F5, "state_seconds": {"FP": 1.0, "INT6": 1.0}}, "state_seconds"), ({k: v for k, v in GOOD_F5.items() if k != "topk"}, "topk")]
        for meta, word in cases:
            r = vf.check_meta_f5(meta, expect_f5())
            self.assertFalse(r["ok"], word); self.assertTrue(any(word in w for w in r["why"]), (word, r["why"]))

    def test_a_design_field_missing_from_both_metas_is_still_a_failure(self):
        ref = {k: v for k, v in REF_F5.items() if k != "topk"}
        r = vf.check_meta_f5({k: v for k, v in GOOD_F5.items() if k != "topk"}, expect_f5(reference=ref))
        self.assertFalse(r["ok"]); self.assertTrue(any("topk" in w for w in r["why"]))

    def test_the_old_meta_without_the_variant_flags_is_not_accepted_as_a_new_job(self):
        r = vf.check_meta_f5({k: v for k, v in GOOD_F5.items() if k not in ("direction_control", "fixed_amplitude")}, expect_f5())
        self.assertFalse(r["ok"])

    def test_the_f5_expectations_come_from_the_reference_meta_and_the_state_list(self):
        import argparse
        e = vf.expected_f5_from_args(argparse.Namespace(states=["FP", "INT6", "INT4"], reference_meta=None), reference=REF_F5)
        self.assertEqual(e, {"states": ["FP", "INT6", "INT4"], "reference": REF_F5})

    def test_the_command_line_may_name_a_state_by_its_bit_width_as_the_runner_does(self):
        import argparse
        e = vf.expected_f5_from_args(argparse.Namespace(states=["FP", "6", "4"], reference_meta=None), reference=REF_F5)
        self.assertEqual(e["states"], ["FP", "INT6", "INT4"])

    def test_a_whole_f5_job_validates_with_the_f5_meta_and_files(self):
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup); local = Path(d.name) / "n"
        man = {"method": "gptq", "bits": [4], "artifact_sha256": {"4": "a" * 64}, "pool_sha256": "p" * 64}
        files = ["f5_stage_a.pt", "f5_state_FP.pt", "f5_state_INT6.pt", "f5_state_INT4.pt"]
        def remote(cmd):
            if cmd.startswith("sacct"):
                return "1|gptq-f5|COMPLETED|03:08:01|gnode046|0:0\n"
            if cmd.startswith("head"):
                return "GPU 0: NVIDIA GeForce RTX 2080 Ti (UUID: x)\n"
            if "grep -c" in cmd:
                return "0\n"
            if "sha256sum" in cmd:
                return subprocess.run("sha256sum *", shell=True, cwd=local, capture_output=True, text=True).stdout
            return ""
        def fetch(r, l):
            for f in files:
                Path(l, f).write_bytes(b"data")
            Path(l, "meta_f5.json").write_text(json.dumps(GOOD_F5)); Path(l, "weights_manifest.json").write_text(json.dumps(man)); Path(l, "gpu_mem.log").write_text("x")
        rep = vf.validate(job="1", jobname="gptq-f5", name="n", files=files, meta_name="meta_f5.json", expected=expect_f5(), method="gptq", bits=[4], recorded=RECORDED, remote_exec=remote, fetch=fetch, local_root=d.name, task="f5")
        self.assertTrue(rep["ok"], rep["failed"])
        rep = vf.validate(job="1", jobname="gptq-f5", name="n", files=files, meta_name="meta_f5.json", expected=expect_f5(), method="gptq", bits=[4], recorded=RECORDED, remote_exec=remote, fetch=fetch, local_root=d.name)
        self.assertEqual(rep["failed"], ["meta"])


class NoWeights(unittest.TestCase):
    """The Llama F5 job (shared RTN, no calibration) writes no weights_manifest.json: the validator can be told so explicitly; the default still demands it."""
    def test_files_check_without_a_weights_manifest_when_told_so(self):
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup)
        for n in ("f5_stage_a.pt", "meta_f5.json"):
            Path(d.name, n).write_text("x")
        self.assertFalse(vf.check_files(d.name, ["f5_stage_a.pt"], "meta_f5.json")["ok"])                        # default: the manifest is required
        self.assertTrue(vf.check_files(d.name, ["f5_stage_a.pt"], "meta_f5.json", need_weights=False)["ok"])
        self.assertFalse(vf.check_files(d.name, ["f5_stage_a.pt", "f5_state_FP.pt"], "meta_f5.json", need_weights=False)["ok"])   # other files are still required

    def test_a_whole_job_without_weights_validates_only_when_the_option_is_set(self):
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup); local = Path(d.name) / "n"
        files = ["f5_stage_a.pt", "f5_state_FP.pt", "f5_state_INT6.pt", "f5_state_INT4.pt"]
        def remote(cmd):
            if cmd.startswith("sacct"):
                return "1|llama-f5|COMPLETED|01:40:00|gnode046|0:0\n"
            if cmd.startswith("head"):
                return "GPU 0: NVIDIA GeForce RTX 2080 Ti (UUID: x)\n"
            if "grep -c" in cmd:
                return "0\n"
            if "sha256sum" in cmd:
                return subprocess.run("sha256sum *", shell=True, cwd=local, capture_output=True, text=True).stdout
            return ""
        def fetch(r, l):
            for f in files:
                Path(l, f).write_bytes(b"data")
            Path(l, "meta_f5.json").write_text(json.dumps(GOOD_F5)); Path(l, "gpu_mem.log").write_text("x")
        kw = dict(job="1", jobname="llama-f5", name="n", files=files, meta_name="meta_f5.json", expected=expect_f5(), method="gptq", bits=[], recorded=RECORDED, remote_exec=remote, fetch=fetch, local_root=d.name, task="f5")
        rep = vf.validate(check_weights=False, **kw); self.assertTrue(rep["ok"], rep["failed"]); self.assertNotIn("weights", rep["checks"])
        rep = vf.validate(**kw); self.assertEqual(sorted(rep["failed"]), ["files", "weights"])


class CommandLine(unittest.TestCase):
    def test_the_no_weights_flag_reaches_the_validation_and_is_off_by_default(self):
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup); ref = Path(d.name) / "ref.json"; ref.write_text(json.dumps(REF_F5)); seen = []
        fake = lambda *a, **k: seen.append(k.get("check_weights")) or {"ok": True, "failed": [], "checks": {}}
        base = ["--job", "1", "--jobname", "j", "--name", "n", "--method", "rtn-matched", "--bits", "4", "--task", "f5", "--states", "FP", "16", "6", "4", "--reference-meta", str(ref)]
        with mock.patch.object(vf, "validate", fake), mock.patch("builtins.print"):
            vf.main(base + ["--no-weights"]); vf.main(base)
        self.assertEqual(seen, [False, True])


class DefaultFiles(unittest.TestCase):
    def test_lm_files_are_one_per_bit_width_and_f5_files_one_per_state_plus_stage_a(self):
        self.assertEqual(vf.default_files("lm", [4, 6], None), ["func_lm_eval_INT4.pt", "func_lm_eval_INT6.pt"])
        self.assertEqual(vf.default_files("f5", [4], ["FP", "6", "4"]), ["f5_stage_a.pt", "f5_state_FP.pt", "f5_state_INT6.pt", "f5_state_INT4.pt"])


class Args(unittest.TestCase):
    def ns(self, **kw):
        import argparse
        d = dict(precisions=["4"], preset="eval", split="eval", n_sentences=300, manifest_sha256="m" * 64, configs=["INT4=351"]); d.update(kw)
        return argparse.Namespace(**d)

    def test_defaults_become_expectations(self):
        e = vf.expected_from_args(self.ns())
        self.assertEqual((e["n_sentences"], e["manifest_source_sha256"], e["configs"]), (300, "m" * 64, {"INT4": 351}))

    def test_negative_sentences_and_empty_manifest_mean_not_checked_and_state_keys_may_contain_slashes(self):
        e = vf.expected_from_args(self.ns(n_sentences=-1, manifest_sha256="", configs=["letter/INT4=108"]))
        self.assertEqual((e["n_sentences"], e["manifest_source_sha256"], e["configs"]), (None, None, {"letter/INT4": 108}))
        self.assertTrue(vf.check_meta({"complete": True, "gates": {"a": True}, "skipped_for_time": [], "precisions": ["4"], "preset": "eval", "split": "eval", "summary": {"letter/INT4": {"configs": 108}}}, e)["ok"])


class Files(unittest.TestCase):
    def test_expected_files_must_all_be_present(self):
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup)
        for n in ("func_lm_eval_INT4.pt", "meta_lm_eval.json", "weights_manifest.json", "gpu_mem.log"):
            Path(d.name, n).write_text("x")
        self.assertTrue(vf.check_files(d.name, ["func_lm_eval_INT4.pt"], "meta_lm_eval.json")["ok"])
        r = vf.check_files(d.name, ["func_lm_eval_INT4.pt", "func_lm_eval_INT6.pt"], "meta_lm_eval.json")
        self.assertFalse(r["ok"]); self.assertIn("func_lm_eval_INT6.pt", r["missing"])
        Path(d.name, "meta_lm_eval.json").unlink(); Path(d.name, "weights_manifest.json").unlink()
        r = vf.check_files(d.name, ["func_lm_eval_INT4.pt"], "meta_lm_eval.json")
        self.assertEqual(r["missing"], ["meta_lm_eval.json", "weights_manifest.json"])


class Job(unittest.TestCase):
    def run_job(self, meta=None, man=None, state="COMPLETED", gpu=True, expected=None, bad_sums=False):
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup); local = Path(d.name) / "n"
        meta = GOOD_META if meta is None else meta
        man = man or {"method": "gptq", "bits": [4], "artifact_sha256": {"4": "a" * 64}, "pool_sha256": "p" * 64}
        def remote(cmd):
            if cmd.startswith("sacct"):
                return f"1|gptq-fe4|{state}|01:32:47|gnode061|0:0\n"
            if cmd.startswith("head"):
                return "GPU 0: NVIDIA GeForce RTX 2080 Ti (UUID: x)\n" if gpu else "none\n"
            if "grep -c" in cmd:
                return "0\n"
            if "sha256sum" in cmd:
                out = subprocess.run("sha256sum *", shell=True, cwd=local, capture_output=True, text=True).stdout
                return out.replace("a", "b", 1) if bad_sums else out
            return ""
        def fetch(r, l):
            Path(l, "func_lm_eval_INT4.pt").write_bytes(b"data"); Path(l, "meta_lm_eval.json").write_text(json.dumps(meta)); Path(l, "weights_manifest.json").write_text(json.dumps(man)); Path(l, "gpu_mem.log").write_text("g")
        return vf.validate(job="1", jobname="gptq-fe4", name="n", files=["func_lm_eval_INT4.pt"], meta_name="meta_lm_eval.json", expected=expected or expect(), method="gptq", bits=[4], recorded=RECORDED, remote_exec=remote, fetch=fetch, local_root=d.name)

    def test_a_fully_good_job_passes(self):
        rep = self.run_job(); self.assertTrue(rep["ok"], rep["failed"]); self.assertEqual(rep["failed"], [])

    def test_each_failure_is_isolated_to_its_check(self):
        self.assertEqual(self.run_job(state="FAILED")["failed"], ["sacct"])
        self.assertEqual(self.run_job(gpu=False)["failed"], ["log"])
        self.assertEqual(self.run_job(meta={**GOOD_META, "complete": False})["failed"], ["meta"])
        self.assertEqual(self.run_job(man={"method": "gptq", "bits": [4], "artifact_sha256": {"4": "d" * 64}})["failed"], ["weights"])
        self.assertEqual(self.run_job(bad_sums=True)["failed"], ["checksums"])

    def test_the_extension_log_name_is_used(self):
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup); seen = []
        vf.validate(job="7", jobname="gptq-fe4", name="n", files=[], meta_name="meta_lm_eval.json", expected=expect(), method="gptq", bits=[4], recorded=RECORDED, remote_exec=lambda c: seen.append(c) or "", fetch=lambda r, l: None, local_root=d.name)
        self.assertTrue(any("gptq-fe4-7.out" in c for c in seen))


if __name__ == "__main__":
    unittest.main()
