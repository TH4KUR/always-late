"""Task 3 of the Gemma-2-9B three-arm plan: the artifact hashes of a build-and-gate job (`weights_manifest.json` per method) become the frozen hash file of the 9B jobs, in the schema of
`reports/gptq_extension/artifact_hashes.json` (`artifact_sha256: {method: {bits: sha}}`, `calibration`). Two gate directories that disagree about a hash are an error. Stdlib only."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gptq_ext import record_hashes as rh

POOL, CAL = "p" * 64, "c" * 64


def manifest(method, bits_sha, pool=POOL):
    return {"method": method, "bits": sorted(int(b) for b in bits_sha), "artifact_sha256": dict(bits_sha), "pool_sha256": pool, "calibration_freeze_sha256": CAL, "calibration_report": {}}


def write_gate(root, per_method, layout="dirs"):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    for method, bits_sha in per_method.items():
        m = manifest(method, bits_sha)
        if layout == "dirs":
            (root / method).mkdir(exist_ok=True); (root / method / "weights_manifest.json").write_text(json.dumps(m))
        else:
            (root / f"weights_manifest_{method}.json").write_text(json.dumps(m))
    return root


class Collect(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_collect_reads_both_methods_and_the_pool_hash(self):
        g = write_gate(self.d / "g", {"gptq": {"4": "a" * 64, "6": "b" * 64}, "rtn_matched": {"4": "c" * 64, "6": "d" * 64}})
        r = rh.collect([g])
        self.assertEqual(r["artifact_sha256"], {"gptq": {"4": "a" * 64, "6": "b" * 64}, "rtn_matched": {"4": "c" * 64, "6": "d" * 64}})
        self.assertEqual(r["calibration"], {"freeze_file_sha256": CAL, "pool_sha256": POOL})

    def test_the_flat_layout_of_the_committed_2b_gate_is_read_and_gives_the_committed_hashes(self):
        r = rh.collect([ROOT / "reports/gptq_extension/gateA"])
        want = json.loads((ROOT / "reports/gptq_extension/artifact_hashes.json").read_text())
        self.assertEqual(r["artifact_sha256"], want["artifact_sha256"])
        self.assertEqual(r["calibration"], want["calibration"])

    def test_two_gate_dirs_with_equal_hashes_merge_and_extra_bits_are_added(self):
        a = write_gate(self.d / "a", {"gptq": {"4": "a" * 64}}); b = write_gate(self.d / "b", {"gptq": {"4": "a" * 64, "6": "b" * 64}})
        self.assertEqual(rh.collect([a, b])["artifact_sha256"], {"gptq": {"4": "a" * 64, "6": "b" * 64}})

    def test_collect_fails_when_two_gate_dirs_disagree(self):
        a = write_gate(self.d / "a", {"gptq": {"4": "a" * 64}}); b = write_gate(self.d / "b", {"gptq": {"4": "f" * 64}})
        with self.assertRaises(ValueError) as ctx:
            rh.collect([a, b])
        self.assertIn("gptq", str(ctx.exception)); self.assertIn("4", str(ctx.exception))

    def test_collect_fails_when_the_pools_differ(self):
        a = write_gate(self.d / "a", {"gptq": {"4": "a" * 64}}); b = self.d / "b"; b.mkdir()
        (b / "weights_manifest_rtn_matched.json").write_text(json.dumps(manifest("rtn_matched", {"4": "c" * 64}, pool="q" * 64)))
        with self.assertRaises(ValueError) as ctx:
            rh.collect([a, b])
        self.assertIn("pool", str(ctx.exception))

    def test_a_directory_without_manifests_is_an_error(self):
        (self.d / "empty").mkdir()
        with self.assertRaises(FileNotFoundError):
            rh.collect([self.d / "empty"])


class Write(unittest.TestCase):
    def test_write_refuses_an_existing_output_and_writes_sorted_json_otherwise(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); g = write_gate(t / "g", {"gptq": {"4": "a" * 64}}); out = t / "out" / "artifact_hashes.json"
            rh.write([g], out)
            self.assertEqual(json.loads(out.read_text())["artifact_sha256"], {"gptq": {"4": "a" * 64}})
            with self.assertRaises(FileExistsError):
                rh.write([g], out)

    def test_the_command_line_writes_the_file(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); g = write_gate(t / "g", {"rtn_matched": {"6": "d" * 64}}); out = t / "h.json"
            self.assertEqual(rh.main(["--gate-dir", str(g), "--out", str(out)]), 0)
            self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()
