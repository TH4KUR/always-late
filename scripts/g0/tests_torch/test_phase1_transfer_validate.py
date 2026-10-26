import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phase1_transfer_validate as tv

META = {"complete": True, "gate_max_abs": 2.0, "gates": {"split_equals_full": True, "logits_native_residual": {"ok": True, "top1": 0.95}, "logits_float32_residual": {"ok": True, "top1": 0.94}},
        "run": {"gates": {"sham_exact": True, "sham_max_abs": 0.0}, "states": {"INT4": {"chunks": 4, "resumed": 0, "items": 90, "skipped": 10, "seconds": 100.0}}}, "peak_gpu_gb": 10.1, "seconds": 300.0}
LOG_HEAD = "GPU 0: NVIDIA GeForce RTX 2080 Ti (UUID: GPU-x)\nMon Oct 26 04:05:00 PM IST 2026\n[ 9s] their functions verified against the frozen manifest\n"
SACCT = "40399|transfer|COMPLETED|01:05:00|gnode085|0:0"


class Pieces(unittest.TestCase):
    def test_the_sacct_line_must_be_completed_with_exit_zero(self):
        self.assertEqual(tv.check_sacct(SACCT)["ok"], True)
        for bad in ("40399|transfer|FAILED|00:02:00|gnode085|1:0", "40399|transfer|COMPLETED|01:05:00|gnode085|1:0", "40399|transfer|RUNNING|01:05:00|gnode085|0:0", ""):
            self.assertFalse(tv.check_sacct(bad)["ok"], bad)

    def test_the_log_head_needs_a_gpu_line_and_no_cuda_warning_and_a_node_without_a_device_fails(self):
        self.assertTrue(tv.check_log_head(LOG_HEAD, 0)["ok"]); self.assertFalse(tv.check_log_head(LOG_HEAD, 1)["ok"])
        self.assertFalse(tv.check_log_head("No devices found.\nMon Oct 26\n", 0)["ok"]); self.assertFalse(tv.check_log_head("", 0)["ok"]); self.assertFalse(tv.check_log_head(LOG_HEAD + "No devices found.\n", 0)["ok"])

    def test_the_meta_file_must_be_complete_with_all_gates_passed_and_an_exact_sham(self):
        self.assertTrue(tv.check_meta(META, {"INT4": 100}, gate_max_abs=2.0)["ok"])
        for path, value in (("complete", False), ("gate_max_abs", 3.0)):
            m = json.loads(json.dumps(META)); m[path] = value; self.assertFalse(tv.check_meta(m, {"INT4": 100}, gate_max_abs=2.0)["ok"], path)
        m = json.loads(json.dumps(META)); m["gates"]["split_equals_full"] = False; self.assertFalse(tv.check_meta(m, {"INT4": 100}, 2.0)["ok"])
        m = json.loads(json.dumps(META)); m["gates"]["logits_float32_residual"]["ok"] = False; self.assertFalse(tv.check_meta(m, {"INT4": 100}, 2.0)["ok"])
        m = json.loads(json.dumps(META)); m["run"]["gates"]["sham_exact"] = False; self.assertFalse(tv.check_meta(m, {"INT4": 100}, 2.0)["ok"])

    def test_items_plus_skipped_must_equal_the_planned_count_for_every_state_and_a_resumed_chunk_is_reported(self):
        self.assertTrue(tv.check_meta(META, {"INT4": 100}, 2.0)["ok"])
        self.assertFalse(tv.check_meta(META, {"INT4": 101}, 2.0)["ok"]); self.assertFalse(tv.check_meta(META, {"INT4": 100, "INT6": 5}, 2.0)["ok"])
        m = json.loads(json.dumps(META)); m["run"]["states"]["INT4"]["resumed"] = 2; r = tv.check_meta(m, {"INT4": 100}, 2.0); self.assertTrue(r["ok"]); self.assertIn("resumed", " ".join(r["notes"]))

    def test_the_planned_items_of_the_feature_level_are_the_contexts_times_five_directions_and_of_the_group_level_the_sentences_times_the_sets(self):
        core = {"contexts": [[1, 2, 3], [4], []]}; self.assertEqual(tv.planned_feature_items(core), 4 * 5)
        self.assertEqual(tv.planned_group_items(languages=27, sentences=300, controls=3), 27 * 300 * 4); self.assertEqual(tv.planned_group_items(languages=6, sentences=300, controls=3), 7200)

    def test_checksum_lines_are_compared_exactly_and_a_missing_or_different_file_is_reported(self):
        a = "aa  f1\nbb  f2\n"; self.assertEqual(tv.compare_checksums(a, a)["differences"], [])
        r = tv.compare_checksums(a, "aa  f1\ncc  f2\n"); self.assertEqual(r["differences"], ["f2"]); self.assertFalse(r["ok"])
        r = tv.compare_checksums(a, "aa  f1\n"); self.assertEqual(r["differences"], ["f2"]); r = tv.compare_checksums("aa  f1\n", a); self.assertEqual(r["differences"], ["f2"])

    def test_the_report_collects_every_check_and_is_ok_only_when_all_are_ok(self):
        r = tv.summarize({"sacct": {"ok": True}, "log": {"ok": True}, "meta": {"ok": True, "notes": []}, "checksums": {"ok": True, "differences": []}}); self.assertTrue(r["ok"])
        r = tv.summarize({"sacct": {"ok": True}, "log": {"ok": False, "why": "no GPU line"}, "meta": {"ok": True, "notes": []}}); self.assertFalse(r["ok"]); self.assertEqual(r["failed"], ["log"])


class Glue(unittest.TestCase):
    def setUp(self):
        import tempfile, hashlib
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name); self.files = {"meta_transfer.json": json.dumps(META), "group_INT4_c000.pt": "binary-a", "gpu_mem.log": "log"}
        sums = lambda files: "".join(f"{hashlib.sha256(v.encode()).hexdigest()}  {k}\n" for k, v in sorted(files.items())); self.sums = sums; self.remote_files = dict(self.files)
        def fetch(remote, local):
            for k, v in self.remote_files.items():
                (Path(local) / k).write_text(v)
        self.fetch = fetch
        self.responses = {"sacct": SACCT, "head": LOG_HEAD, "grep": "0\n"}

    def tearDown(self):
        self.tmp.cleanup()

    def remote(self, cmd):
        if cmd.startswith("sacct"):
            return self.responses["sacct"]
        if cmd.startswith("head"):
            return self.responses["head"]
        if cmd.startswith("grep"):
            return self.responses["grep"]
        return self.sums(self.files)                                                                                 # cd <dir> && sha256sum *

    def run_job(self):
        return tv.validate_job("40399", "job-x", {"INT4": 100}, 2.0, self.remote, self.fetch, local_root=self.root)

    def test_a_finished_job_with_equal_checksums_a_gpu_and_a_complete_meta_passes(self):
        r = self.run_job(); self.assertTrue(r["ok"], r); self.assertEqual(r["failed"], []); self.assertEqual(r["checks"]["checksums"]["files"], 3)

    def test_each_kind_of_failure_is_reported_under_its_own_check(self):
        self.responses["sacct"] = "40399|transfer|FAILED|00:02:00|gnode076|1:0"; self.assertEqual(self.run_job()["failed"], ["sacct"]); self.responses["sacct"] = SACCT
        self.responses["head"] = "No devices found.\n"; self.assertEqual(self.run_job()["failed"], ["log"]); self.responses["head"] = LOG_HEAD
        self.responses["grep"] = "2\n"; self.assertEqual(self.run_job()["failed"], ["log"]); self.responses["grep"] = "0\n"
        self.files["gpu_mem.log"] = "changed on Ada"; self.assertEqual(self.run_job()["failed"], ["checksums"]); self.files["gpu_mem.log"] = "log"
        bad = json.loads(META and json.dumps(META)); bad["complete"] = False; self.files["meta_transfer.json"] = json.dumps(bad); self.remote_files["meta_transfer.json"] = json.dumps(bad); self.assertEqual(self.run_job()["failed"], ["meta"])

    def test_a_missing_meta_file_fails_the_meta_check(self):
        del self.files["meta_transfer.json"]; del self.remote_files["meta_transfer.json"]; r = self.run_job(); self.assertEqual(r["failed"], ["meta"])


if __name__ == "__main__":
    unittest.main()
