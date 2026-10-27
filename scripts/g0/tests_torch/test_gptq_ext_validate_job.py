"""Validation of a finished GPTQ-extension job: the old transfer checklist (`phase1_transfer_validate.validate_job`, unchanged) with the extension's log name, plus a check of the job's `weights_manifest.json` (method, bit
widths, artifact hashes against the recorded ones). Remote commands and the file copy are injected."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
G0 = HERE.parent
sys.path.insert(0, str(G0))
from gptq_ext import validate_job as vj  # noqa: E402

RECORDED = {"gptq": {"4": "a" * 64}, "rtn_matched": {"4": "b" * 64}}


def manifest(method="gptq", bits=(4,), hashes=None):
    hashes = hashes or {str(b): ("a" * 64 if method == "gptq" else "b" * 64) if b == 4 else "c" * 64 for b in bits}
    return {"method": method, "bits": list(bits), "artifact_sha256": hashes, "pool_sha256": "p" * 64}


class Weights(unittest.TestCase):
    def test_a_matching_manifest_passes(self):
        self.assertTrue(vj.check_weights_manifest(manifest(), "gptq", [4], RECORDED)["ok"])

    def test_a_wrong_method_is_refused(self):
        r = vj.check_weights_manifest(manifest("rtn_matched"), "gptq", [4], RECORDED)
        self.assertFalse(r["ok"]); self.assertTrue(any("method" in w for w in r["why"]))

    def test_a_hash_that_differs_from_the_recorded_one_is_refused(self):
        r = vj.check_weights_manifest(manifest(hashes={"4": "d" * 64}), "gptq", [4], RECORDED)
        self.assertFalse(r["ok"]); self.assertTrue(any("recorded" in w for w in r["why"]))

    def test_unrecorded_bit_widths_are_accepted_and_listed(self):
        r = vj.check_weights_manifest(manifest(bits=(4, 5, 6)), "gptq", [4, 5, 6], RECORDED)
        self.assertTrue(r["ok"]); self.assertEqual(sorted(r["unrecorded"]), ["5", "6"])

    def test_missing_or_extra_bit_widths_are_refused(self):
        self.assertFalse(vj.check_weights_manifest(manifest(bits=(4,)), "gptq", [4, 5], RECORDED)["ok"])
        self.assertFalse(vj.check_weights_manifest(manifest(bits=(4, 5)), "gptq", [4], RECORDED)["ok"])


class Job(unittest.TestCase):
    META = {"complete": True, "gate_max_abs": 2.0, "gates": {"split_equals_full": True, "logits_native_residual": {"ok": True}, "logits_float32_residual": {"ok": True}},
            "run": {"gates": {"sham_exact": True}, "states": {"INT4": {"items": 10, "skipped": 0}}}}

    def run_job(self, man, state="COMPLETED", exit_code="0:0", gpu=True):
        """A job that passes every OTHER check (sacct, GPU line, complete meta, equal checksums) so that the verdict isolates the weights check."""
        import subprocess
        calls = []
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup)
        local = Path(d.name) / "n"
        def remote(cmd):
            calls.append(cmd)
            if cmd.startswith("sacct"):
                return f"42954|rtnm-tf4|{state}|03:13:58|gnode080|{exit_code}\n"
            if cmd.startswith("head"):
                return ("GPU 0: NVIDIA GeForce RTX 2080 Ti (UUID: x)\n" if gpu else "no gpu here\n")
            if "grep -c" in cmd:
                return "0\n"
            if "sha256sum" in cmd:
                return subprocess.run("sha256sum *", shell=True, cwd=local, capture_output=True, text=True).stdout
            return ""
        def fetch(r, l):
            Path(l, "weights_manifest.json").write_text(json.dumps(man)); Path(l, "meta_transfer.json").write_text(json.dumps(self.META))
        rep = vj.validate(job="42954", jobname="rtnm-tf4", name="n", planned={"INT4": 10}, gate_max_abs=2.0, method=man["method"], bits=[4], recorded=RECORDED, remote_exec=remote, fetch=fetch, local_root=d.name)
        return rep, calls

    def test_a_fully_good_job_passes_every_check(self):
        rep, _ = self.run_job(manifest("rtn_matched"))
        self.assertTrue(rep["ok"], rep["failed"])
        self.assertEqual(rep["failed"], [])

    def test_the_extension_log_name_replaces_the_old_one_in_every_remote_command(self):
        rep, calls = self.run_job(manifest("rtn_matched"))
        logs = [c for c in calls if ".out" in c]
        self.assertTrue(logs and all("rtnm-tf4-42954.out" in c and "transfer-42954.out" not in c for c in logs))

    def test_the_report_has_the_old_checks_and_the_weights_check(self):
        rep, _ = self.run_job(manifest("rtn_matched"))
        self.assertIn("weights", rep["checks"]); self.assertIn("sacct", rep["checks"]); self.assertIn("log", rep["checks"])
        self.assertTrue(rep["checks"]["weights"]["ok"])

    def test_a_failed_weights_check_fails_the_whole_report(self):
        rep, _ = self.run_job(manifest(hashes={"4": "d" * 64}))
        self.assertFalse(rep["ok"]); self.assertEqual(rep["failed"], ["weights"])

    def test_a_failed_job_state_fails_the_report(self):
        rep, _ = self.run_job(manifest(), state="FAILED", exit_code="1:0")
        self.assertFalse(rep["ok"]); self.assertEqual(rep["failed"], ["sacct"])

    def test_a_node_without_a_gpu_line_fails_the_log_check_only(self):
        rep, _ = self.run_job(manifest(), gpu=False)
        self.assertEqual(rep["failed"], ["log"])

    def test_a_missing_manifest_fails_the_weights_check(self):
        calls = []
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup)
        rep = vj.validate(job="1", jobname="x", name="n", planned={"INT4": 1}, gate_max_abs=2.0, method="gptq", bits=[4], recorded=RECORDED, remote_exec=lambda c: "1|x|COMPLETED|1|n|0:0\n" if c.startswith("sacct") else ("GPU 0: g\n" if c.startswith("head") else ""),
                          fetch=lambda r, l: Path(l, "meta_transfer.json").write_text("{}"), local_root=d.name)
        self.assertFalse(rep["checks"]["weights"]["ok"])


if __name__ == "__main__":
    unittest.main()
