"""Regression test for scripts/g0/make_duan_runners_v3.py: it must still generate compilable runners (every anchor
matches Duan's originals) and the runners must carry the wandb hooks. Uses the vendored pristine originals.
Set G0_GENERATOR to test another generator file."""
import os, py_compile, subprocess, sys, tempfile, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
GEN = Path(os.environ.get("G0_GENERATOR", HERE.parent / "make_duan_runners_v3.py"))
DUAN = HERE.parents[2] / "third_party" / "duan2026" / "pristine" / "scripts"   # vendored byte-identical originals
O3, O6 = DUAN / "03_gemma_phase3_streaming_sweep.py", DUAN / "06_gemma_sliding_window_ppl_check.py"


class GeneratedRunners(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.o3, cls.o6 = Path(cls.tmp.name, "r03.py"), Path(cls.tmp.name, "r06.py")
        subprocess.run([sys.executable, str(GEN), str(O3), str(O6), str(cls.o3), str(cls.o6)], check=True, capture_output=True)
        cls.s3, cls.s6 = cls.o3.read_text(), cls.o6.read_text()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_both_runners_compile(self):
        py_compile.compile(str(self.o3), doraise=True); py_compile.compile(str(self.o6), doraise=True)

    def test_runner_03_carries_all_wandb_hooks(self):
        for needle in ("from g0_wandb import G0Logger", "_WB = G0Logger.start(", "_WB.heartbeat(", "_WB.log_condition(",
                       "_WB.log_table(", "_WB.log_gates(", "_WB.finish("):
            self.assertIn(needle, self.s3)

    def test_runner_06_carries_wandb_hooks(self):
        for needle in ("from g0_wandb import G0Logger", "_WB = G0Logger.start(", "_WB.log_condition(", "_WB.finish("):
            self.assertIn(needle, self.s6)

    def test_runner_03_uses_the_protected_gpu_memory_helper(self):
        self.assertIn("gpu_mem_gib", self.s3)
        self.assertNotIn("def _g0_mem", self.s3)   # the inline version could raise outside any try/except

    def test_project_defaults_to_anlp_project(self):
        self.assertIn('"WANDB_PROJECT", "anlp-project"', self.s3)


if __name__ == "__main__":
    unittest.main()
