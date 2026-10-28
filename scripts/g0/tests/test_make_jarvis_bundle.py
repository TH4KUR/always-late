"""Task 9 of the Gemma-2-9B three-arm plan: `make_jarvis_bundle.sh` packs the committed repository (git archive of HEAD plus a COMMIT file, so nothing has to be pushed) and the calibration pool token files into two
tarballs with a SHA256SUMS file, and prints the commands that unpack them on the Jarvis machine. It refuses a repository with uncommitted or untracked files (the amendment and the code must be committed), a missing
pool source and a missing pool option. A temporary git repository stands in for the project; the real Ada copy of the pool is not touched by these tests."""
import os
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "make_jarvis_bundle.sh"
BASH, GIT = shutil.which("bash"), shutil.which("git")


def git(repo, *args):
    return subprocess.run([GIT, "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args], capture_output=True, text=True, check=True).stdout.strip()


@unittest.skipUnless(BASH and GIT, "needs bash and git")
class Bundle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); t = Path(self.tmp.name)
        self.repo, self.out, self.pool = t / "repo", t / "out", t / "pool"
        self.repo.mkdir(); git(self.repo, "init", "-q"); (self.repo / "x.txt").write_text("x\n"); (self.repo / "sub").mkdir(); (self.repo / "sub/y.txt").write_text("y\n")
        git(self.repo, "add", "."); git(self.repo, "commit", "-q", "-m", "c1")
        self.pool.mkdir(); (self.pool / "am.json").write_text("{}"); (self.pool / "en.json").write_text("{}")

    def tearDown(self):
        self.tmp.cleanup()

    def run_script(self, *args):
        env = {**os.environ, "REPO": str(self.repo)}
        return subprocess.run([BASH, str(SCRIPT), str(self.out), *args], env=env, capture_output=True, text=True, timeout=120)

    def test_the_bundle_holds_the_repository_at_head_with_a_commit_file_and_the_pool_files(self):
        r = self.run_script("--pool-from", str(self.pool))
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        head = git(self.repo, "rev-parse", "HEAD")
        with tarfile.open(self.out / "always-late-gemma9b.tar.gz") as tf:
            names = tf.getnames()
            self.assertIn("always-late/x.txt", names); self.assertIn("always-late/sub/y.txt", names)
            self.assertEqual(tf.extractfile("always-late/COMMIT").read().decode().strip(), head)
        with tarfile.open(self.out / "gptq_calibration_pool.tar.gz") as tf:
            self.assertEqual(sorted(n for n in tf.getnames() if n.endswith(".json")), ["gptq/calibration/am.json", "gptq/calibration/en.json"])

    def test_sha256sums_verifies_and_the_unpack_commands_are_printed(self):
        r = self.run_script("--pool-from", str(self.pool))
        self.assertEqual(r.returncode, 0, r.stderr)
        sums = (self.out / "SHA256SUMS").read_text()
        self.assertIn("always-late-gemma9b.tar.gz", sums); self.assertIn("gptq_calibration_pool.tar.gz", sums)
        self.assertEqual(subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=self.out, capture_output=True, text=True).returncode, 0)
        for k in ("scp", "tar xzf always-late-gemma9b.tar.gz", "tar xzf gptq_calibration_pool.tar.gz", "sha256sum -c SHA256SUMS", "jarvis_run_all.sh"):
            self.assertIn(k, r.stdout)

    def test_a_modified_tracked_file_is_refused(self):
        (self.repo / "x.txt").write_text("changed\n")
        r = self.run_script("--pool-from", str(self.pool))
        self.assertNotEqual(r.returncode, 0); self.assertIn("uncommitted", r.stderr)

    def test_an_untracked_file_is_refused(self):
        (self.repo / "new.json").write_text("{}")
        r = self.run_script("--pool-from", str(self.pool))
        self.assertNotEqual(r.returncode, 0); self.assertIn("uncommitted", r.stderr)

    def test_a_missing_pool_source_or_option_is_refused(self):
        r = self.run_script("--pool-from", str(self.pool / "nope"))
        self.assertNotEqual(r.returncode, 0); self.assertIn("pool", r.stderr)
        r = self.run_script()
        self.assertNotEqual(r.returncode, 0); self.assertIn("--pool-from", r.stderr)

    def test_an_existing_output_is_never_overwritten(self):
        self.assertEqual(self.run_script("--pool-from", str(self.pool)).returncode, 0)
        r = self.run_script("--pool-from", str(self.pool))
        self.assertNotEqual(r.returncode, 0); self.assertIn("exists", r.stderr)


if __name__ == "__main__":
    unittest.main()
