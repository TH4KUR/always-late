"""Tests for scripts/g0/mutcheck.py: a mutation-check harness that can never leave a mutant in the source (try/finally restore, hash check), reports patterns that no
longer match instead of crashing, and treats a failing test command as 'killed'. stdlib only."""
import hashlib, sys, tempfile, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mutcheck as mc   # noqa: E402

SRC = "def f():\n    return 1\n\n\ndef g(x):\n    return x > 0\n"


class Harness(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp()); self.addCleanup(__import__("shutil").rmtree, self.dir, ignore_errors=True)
        self.src = self.dir / "mod.py"; self.src.write_text(SRC)
        self.cmd = [sys.executable, "-B", "-c", f"import sys; sys.path.insert(0, {str(self.dir)!r}); import mod; sys.exit(0 if (mod.f() == 1 and mod.g(1) and not mod.g(0)) else 1)"]

    def digest(self):
        return hashlib.sha256(self.src.read_bytes()).hexdigest()

    def test_killed_survived_and_not_applicable_are_reported_and_the_file_is_restored(self):
        before = self.digest()
        res = mc.run_mutants(self.src, [("return 1", "return 2", "f changed"), ("x > 0", "x >= 0", "boundary"), ("return x > 0", "return x > 0  # same", "comment only"), ("no such text", "z", "missing")], self.cmd)
        self.assertEqual({r["label"]: r["status"] for r in res}, {"f changed": "KILLED", "boundary": "KILLED", "comment only": "SURVIVED", "missing": "NOT_APPLICABLE"}); self.assertEqual(self.digest(), before)

    def test_a_pattern_that_matches_twice_is_not_applicable_not_silently_applied_once(self):
        res = mc.run_mutants(self.src, [("return", "yield", "ambiguous")], self.cmd); self.assertEqual(res[0]["status"], "NOT_APPLICABLE"); self.assertEqual(self.src.read_text(), SRC)

    def test_the_file_is_restored_even_when_the_test_command_cannot_run(self):
        before = self.digest()
        with self.assertRaises(Exception): mc.run_mutants(self.src, [("return 1", "return 2", "x")], ["/no/such/binary"])
        self.assertEqual(self.digest(), before)

    def test_the_baseline_must_pass_before_any_mutant_is_tried(self):
        self.src.write_text(SRC.replace("return 1", "return 3"))
        with self.assertRaises(RuntimeError): mc.run_mutants(self.src, [("return 3", "return 4", "x")], self.cmd)


if __name__ == "__main__":
    unittest.main()
