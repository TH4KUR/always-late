"""Tests for scripts/g0/hf_artifacts.py: upload large run artifacts to a Hugging Face dataset repo in ONE commit, with a sha256
manifest and a remote size check. No network: a fake API object stands in for huggingface_hub.HfApi."""
import hashlib, sys, tempfile, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import hf_artifacts as ha  # noqa: E402


def make_tree(root):
    root = Path(root)
    (root / "run" / "wandb").mkdir(parents=True)
    (root / "run" / "a.csv").write_text("abc"); (root / "run" / "snap_x.pt").write_text("12345"); (root / "run" / "wandb" / "junk.txt").write_text("zz")
    (root / "run" / "sub").mkdir(); (root / "run" / "sub" / "b.json").write_text("{}")
    (root / "single.pt").write_text("tok")
    return root


class Op:
    def __init__(self, path_in_repo, path_or_fileobj):
        self.path_in_repo, self.path_or_fileobj = path_in_repo, path_or_fileobj


class FakeEntry:
    def __init__(self, path, size):
        self.path, self.size = path, size


class FakeApi:
    def __init__(self, remote=()):
        self.commits, self.remote = [], list(remote)
    def create_commit(self, repo_id, operations, commit_message, repo_type):
        self.commits.append({"repo_id": repo_id, "ops": list(operations), "message": commit_message, "repo_type": repo_type})
    def list_repo_tree(self, repo_id, repo_type, recursive):
        return iter(self.remote)


class Collect(unittest.TestCase):
    def test_prefix_include_exclude_and_sizes(self):
        with tempfile.TemporaryDirectory() as t:
            root = make_tree(t)
            files = ha.collect_files([{"local": str(root / "run"), "repo": "runs/R9", "exclude": ["wandb/*"]}])
            self.assertEqual([f[1] for f in files], ["runs/R9/a.csv", "runs/R9/snap_x.pt", "runs/R9/sub/b.json"])
            self.assertEqual({f[1]: f[2] for f in files}["runs/R9/snap_x.pt"], 5)

    def test_include_patterns_and_single_files(self):
        with tempfile.TemporaryDirectory() as t:
            root = make_tree(t)
            files = ha.collect_files([{"local": str(root / "run"), "repo": "r", "include": ["*.csv"]}, {"local": str(root / "single.pt"), "repo": "tokens/x.pt"}])
            self.assertEqual([f[1] for f in files], ["r/a.csv", "tokens/x.pt"])

    def test_a_missing_source_is_an_error_not_a_silent_skip(self):
        with self.assertRaises(FileNotFoundError):
            ha.collect_files([{"local": "/nonexistent/dir/for/test", "repo": "x"}])


class Manifest(unittest.TestCase):
    def test_lines_are_sha256_and_repo_path(self):
        with tempfile.TemporaryDirectory() as t:
            root = make_tree(t)
            files = ha.collect_files([{"local": str(root / "run"), "repo": "p", "include": ["a.csv"]}])
            self.assertEqual(ha.manifest_text(files).strip(), f"{hashlib.sha256(b'abc').hexdigest()}  p/a.csv")


class Upload(unittest.TestCase):
    def test_one_commit_with_every_file_plus_the_text_files(self):
        with tempfile.TemporaryDirectory() as t:
            root = make_tree(t); api = FakeApi()
            files = ha.collect_files([{"local": str(root / "run"), "repo": "runs/R9", "exclude": ["wandb/*"]}])
            ha.upload(api, "u/ds", files, "msg", extra_text={"README.md": "card", "MANIFEST.sha256": ha.manifest_text(files)}, op_cls=Op)
            self.assertEqual(len(api.commits), 1); c = api.commits[0]
            self.assertEqual((c["repo_id"], c["repo_type"], c["message"]), ("u/ds", "dataset", "msg"))
            self.assertEqual(sorted(o.path_in_repo for o in c["ops"]), ["MANIFEST.sha256", "README.md", "runs/R9/a.csv", "runs/R9/snap_x.pt", "runs/R9/sub/b.json"])

    def test_refuses_an_empty_upload(self):
        with self.assertRaises(ValueError):
            ha.upload(FakeApi(), "u/ds", [], "m", op_cls=Op)


class Verify(unittest.TestCase):
    def test_reports_missing_and_wrong_size_files(self):
        files = [("/x/a", "p/a", 3), ("/x/b", "p/b", 5), ("/x/c", "p/c", 7)]
        api = FakeApi(remote=[FakeEntry("p/a", 3), FakeEntry("p/b", 4), FakeEntry("other", 1)])
        v = ha.verify_remote(api, "u/ds", files)
        self.assertFalse(v["ok"]); self.assertEqual(v["missing"], ["p/c"]); self.assertEqual(v["size_mismatch"], ["p/b"])

    def test_ok_when_everything_matches(self):
        files = [("/x/a", "p/a", 3)]
        self.assertTrue(ha.verify_remote(FakeApi(remote=[FakeEntry("p/a", 3)]), "u/ds", files)["ok"])


if __name__ == "__main__":
    unittest.main()
