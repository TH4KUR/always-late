import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "jarvis_run_all.sh"
BASH = shutil.which("bash")


def run(env_extra, base):
    env = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "FAKE_GPUS", "FAKE_RAM_GB", "STATES", "SKIP_SMOKE", "MAX_PARALLEL", "FOREGROUND")}
    env.update({"BASE": str(base), "REPO": str(SCRIPT.parents[2]), "DRY_RUN": "1"}); env.update(env_extra)
    return subprocess.run([BASH, str(SCRIPT)], env=env, capture_output=True, text=True, timeout=120)


@unittest.skipUnless(BASH, "bash is not available")
class DryRun(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.base = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_plan_lists_every_step_in_order_and_runs_nothing(self):
        r = run({"FAKE_GPUS": "81920", "FAKE_RAM_GB": "200"}, self.base)
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        out = r.stdout
        order = [out.index(s) for s in ("== env", "== flores", "== access_and_download", "== checks", "== smoke", "== full")]
        self.assertEqual(order, sorted(order))
        self.assertIn("torch==2.14.1", out); self.assertIn("transformer-lens==3.9.0", out); self.assertIn("sae-lens==6.54.5", out)
        self.assertFalse((self.base / "state").exists())                        # nothing was created

    def test_one_process_per_gpu_when_the_ram_allows_it(self):
        r = run({"FAKE_GPUS": "81920 81920 81920", "FAKE_RAM_GB": "256"}, self.base)
        self.assertIn("usable GPUs: 0 1 2", r.stdout); self.assertIn("concurrent state processes: 3", r.stdout)
        r = run({"FAKE_GPUS": "81920 81920 81920", "FAKE_RAM_GB": "100"}, self.base)
        self.assertIn("concurrent state processes: 1", r.stdout)                 # RAM limits it
        r = run({"FAKE_GPUS": "81920 81920 81920", "FAKE_RAM_GB": "256", "MAX_PARALLEL": "2"}, self.base)
        self.assertIn("concurrent state processes: 2", r.stdout)

    def test_small_gpus_are_refused(self):
        r = run({"FAKE_GPUS": "24576 11264", "FAKE_RAM_GB": "256"}, self.base)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("no GPU with >=", r.stderr)

    def test_a_small_gpu_next_to_a_big_one_is_skipped(self):
        r = run({"FAKE_GPUS": "11264 81920", "FAKE_RAM_GB": "256"}, self.base)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("usable GPUs: 1", r.stdout)

    def test_the_token_is_required_for_a_real_run(self):
        env = {k: v for k, v in os.environ.items() if k != "HF_TOKEN"}
        env.update({"BASE": str(self.base), "REPO": str(SCRIPT.parents[2]), "FAKE_GPUS": "81920", "FAKE_RAM_GB": "200"})
        r = subprocess.run([BASH, str(SCRIPT)], env=env, capture_output=True, text=True, timeout=60)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("HF_TOKEN", r.stderr)


if __name__ == "__main__":
    unittest.main()
