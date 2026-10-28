"""Validation of a finished functional-study job of the GPTQ extension (loss task E4, rescue E5, Belebele E7): sacct, GPU line, complete meta with its gates true, nothing skipped for time, the planned precision(s), the
groups manifest and the number of configurations per state equal to the old job's, the output file(s), the sha256 of every copied file, and the weights manifest against the recorded hashes. Remote commands and the copy are injected."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

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
