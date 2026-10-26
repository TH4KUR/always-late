"""Tests of scripts/g0/wandb_upload_v2.py (schema v2 of reports/wandb_schema_v2.md, decision D-091): the pure payload builder on a small fake repository, the guards, the uploader on a fake wandb module,
and an integration check of the real committed files."""
import contextlib
import hashlib
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import wandb_upload_v2 as wu

REPO = Path(__file__).resolve().parents[3]
ANALYSIS = {"macro": {"FP": {"point": 0.5, "lo": 0.4, "hi": 0.6, "median": 0.2, "n_positive": 27}, "INT4": {"point": 0.9, "lo": 0.8, "hi": 1.0, "median": 0.4, "n_positive": 27}, "INT0": {"point": 0.5, "lo": 0.4, "hi": 0.6, "median": 0.2, "n_positive": 27}},
            "retention": {"INT4": {"macro": {"ratio": 2.0, "lo": 1.5, "hi": 2.5, "class": "amplified"}}, "INT0": {"macro": {"ratio": 1.0, "lo": 1.0, "hi": 1.0, "class": "preserved"}}},
            "statements": [{"id": "S0", "outcome": "met"}, {"id": "S1", "outcome": "not met"}], "covered": ["a", "b", "c"], "ratios": {"1.1": {"median": -0.001}}}
CSV = "precision,language,effect,note\nFP,eng,0.25,\nINT4,eng,1,x\nINT4,deu,-0.5,y z\n"


def fam(**kw):
    f = {"slug": "demo", "name": "demo/run", "group": "g", "status": "exploratory", "status_text": "exploratory (demo)", "decisions": ["D-001", "D-002"], "title": "a demo run", "model": "m", "sae": "s", "layer": 12, "quantizer": "q",
         "states": ["FP", "INT4"], "jobs": {"job-a": 11, "job-b": 12}, "code_ref": "abc", "freeze": "reports/x/freeze.json", "limits": "a limit",
         "metrics": [("headline/macro_effect", "reports/x/analysis.json", ["macro", "FP", "point"]), ("headline/macro_effect_hi", "reports/x/analysis.json", ["macro", "FP", "hi"]),
                     ("headline/S1_outcome", "reports/x/analysis.json", ["statements", {"id": "S1"}, "outcome"]), ("headline/n_covered", "reports/x/analysis.json", ["covered", "#len"])],
         "tables": [("t", "reports/x/table.csv")], "images": ["reports/x/fig_light.png"], "files": ["reports/x/report.md", "reports/x/analysis.json"], "history": None}
    f.update(kw)
    return f


def make_repo(root):
    root = Path(root); d = root / "reports" / "x"; d.mkdir(parents=True)
    (d / "analysis.json").write_text(json.dumps(ANALYSIS)); (d / "table.csv").write_text(CSV); (d / "fig_light.png").write_bytes(b"\x89PNG" + bytes(100))
    (d / "report.md").write_text("# report\n"); (d / "freeze.json").write_text('{"freeze": 1}')
    return root


class Fixture(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory(); self.root = make_repo(self._t.name)

    def tearDown(self):
        self._t.cleanup()


class Paths(Fixture):
    def test_resolve_keys_indices_selectors_and_length(self):
        d = ANALYSIS
        self.assertEqual(wu.resolve(d, ["macro", "FP", "point"], "f"), 0.5)
        self.assertEqual(wu.resolve(d, ["statements", 1, "id"], "f"), "S1")
        self.assertEqual(wu.resolve(d, ["statements", {"id": "S0"}, "outcome"], "f"), "met")
        self.assertEqual(wu.resolve(d, ["covered", "#len"], "f"), 3)
        self.assertEqual(wu.resolve(d, ["macro", "#len"], "f"), 3)
        self.assertEqual(wu.resolve(d, ["ratios", "1.1", "median"], "f"), -0.001)

    def test_a_path_that_does_not_resolve_names_the_file_and_the_path(self):
        for bad in (["macro", "INT9", "point"], ["statements", 7, "id"], ["statements", {"id": "S9"}, "outcome"], ["covered", "x"], ["macro", "FP", "point", "deeper"]):
            with self.assertRaises(wu.SpecError) as cm:
                wu.resolve(ANALYSIS, bad, "reports/x/analysis.json")
            self.assertIn("reports/x/analysis.json", str(cm.exception))

    def test_a_selector_that_matches_two_elements_is_an_error(self):
        with self.assertRaises(wu.SpecError):
            wu.resolve({"l": [{"id": "S0", "v": 1}, {"id": "S0", "v": 2}]}, ["l", {"id": "S0"}, "v"], "f")


class Csv(Fixture):
    def test_numbers_become_numbers_empty_cells_none_and_text_stays(self):
        cols, rows = wu.read_csv(self.root, "reports/x/table.csv")
        self.assertEqual(cols, ["precision", "language", "effect", "note"])
        self.assertEqual(rows[0], ["FP", "eng", 0.25, None]); self.assertEqual(rows[1], ["INT4", "eng", 1, "x"]); self.assertEqual(rows[2], ["INT4", "deu", -0.5, "y z"])
        self.assertIsInstance(rows[1][2], int); self.assertIsInstance(rows[0][2], float)

    def test_a_table_above_the_row_cap_is_refused(self):
        big = "a\n" + "\n".join(str(i) for i in range(wu.MAX_TABLE_ROWS + 1)) + "\n"
        (self.root / "reports" / "x" / "big.csv").write_text(big)
        with self.assertRaises(wu.SpecError):
            wu.read_csv(self.root, "reports/x/big.csv")

    def test_a_table_with_exactly_the_cap_of_rows_is_accepted(self):
        (self.root / "reports" / "x" / "cap.csv").write_text("a\n" + "\n".join(str(i) for i in range(wu.MAX_TABLE_ROWS)) + "\n")
        self.assertEqual(len(wu.read_csv(self.root, "reports/x/cap.csv")[1]), wu.MAX_TABLE_ROWS)

    def test_ragged_rows_are_refused(self):
        (self.root / "reports" / "x" / "rag.csv").write_text("a,b\n1,2,3\n")
        with self.assertRaises(wu.SpecError):
            wu.read_csv(self.root, "reports/x/rag.csv")


class Guards(Fixture):
    def test_banned_extensions_are_refused(self):
        for ext in (".pt", ".pth", ".bin", ".safetensors", ".npy", ".npz", ".ckpt", ".gguf", ".zip", ".tar", ".gz"):
            (self.root / "reports" / "x" / f"w{ext}").write_bytes(b"0")
            with self.assertRaises(wu.SpecError, msg=ext):
                wu.check_file(self.root, f"reports/x/w{ext}")

    def test_a_file_above_the_size_cap_is_refused(self):
        (self.root / "reports" / "x" / "big.json").write_bytes(b"0" * (wu.MAX_FILE_BYTES + 1))
        with self.assertRaises(wu.SpecError):
            wu.check_file(self.root, "reports/x/big.json")
        (self.root / "reports" / "x" / "ok.json").write_bytes(b"0" * wu.MAX_FILE_BYTES)
        wu.check_file(self.root, "reports/x/ok.json")

    def test_only_paths_below_reports_are_allowed_without_traversal_or_absolute_paths(self):
        for d, name in (("scripts/g0", "x.py"), ("data", "x.csv"), ("third_party", "x.json"), ("scripts", "x.json")):              # the files exist: only the rules of the guard can refuse them
            (self.root / d).mkdir(parents=True, exist_ok=True); (self.root / d / name).write_text("{}")
        (self.root / "reports" / "y.json").write_text("{}")
        for bad in ("scripts/g0/x.py", "../etc/passwd", "/etc/passwd", "reports/../scripts/x.json", "data/x.csv", "third_party/x.json", "reports"):
            with self.assertRaises(wu.SpecError, msg=bad):
                wu.check_file(self.root, bad)

    def test_the_documented_caps(self):
        self.assertEqual((wu.MAX_FILE_BYTES, wu.MAX_RUN_BYTES, wu.MAX_TABLE_ROWS), (2_000_000, 20_000_000, 10_000))

    def test_a_missing_file_is_refused(self):
        with self.assertRaises(wu.SpecError):
            wu.check_file(self.root, "reports/x/nothing.json")

    def test_a_run_above_the_total_size_cap_is_refused(self):
        for i in range(3):
            (self.root / "reports" / "x" / f"b{i}.json").write_bytes(b"0" * (wu.MAX_FILE_BYTES - 10))
        files = [f"reports/x/b{i}.json" for i in range(3)]
        old = wu.MAX_RUN_BYTES
        try:
            wu.MAX_RUN_BYTES = 3 * wu.MAX_FILE_BYTES - 100
            self.assertEqual(wu.MAX_RUN_BYTES, 3 * wu.MAX_FILE_BYTES - 100)
            with self.assertRaises(wu.SpecError):
                wu.build_run(self.root, fam(files=files, images=[], tables=[]), "c")
            wu.MAX_RUN_BYTES = 3 * wu.MAX_FILE_BYTES
            wu.build_run(self.root, fam(files=files, images=[], tables=[]), "c")
            run = wu.build_run(self.root, fam(files=files, images=[], tables=[], freeze=None), "c")
            wu.MAX_RUN_BYTES = run["bytes_total"]                                                                    # exactly at the cap is allowed
            wu.build_run(self.root, fam(files=files, images=[], tables=[], freeze=None), "c")
            wu.MAX_RUN_BYTES = run["bytes_total"] - 1
            with self.assertRaises(wu.SpecError):
                wu.build_run(self.root, fam(files=files, images=[], tables=[], freeze=None), "c")
        finally:
            wu.MAX_RUN_BYTES = old

    def test_images_must_be_png(self):
        (self.root / "reports" / "x" / "f_light.svg").write_text("<svg/>")
        with self.assertRaises(wu.SpecError):
            wu.build_run(self.root, fam(images=["reports/x/f_light.svg"]), "c")


class BuildRun(Fixture):
    def test_identity_tags_and_group(self):
        r = wu.build_run(self.root, fam(), "c0ffee")
        self.assertEqual(r["id"], "v2-demo"); self.assertEqual(r["name"], "demo/run"); self.assertEqual(r["group"], "g"); self.assertEqual(r["job_type"], "analysis-upload")
        for t in ("schema-v2", "post-hoc-upload", "exploratory", "demo"):
            self.assertIn(t, r["tags"])
        r2 = wu.build_run(self.root, fam(status="confirmatory", status_text="confirmatory (frozen)"), "c")
        self.assertIn("confirmatory", r2["tags"]); self.assertNotIn("exploratory", r2["tags"])

    def test_the_config_has_the_fixed_keys_and_values_from_the_spec(self):
        r = wu.build_run(self.root, fam(), "c0ffee"); c = r["config"]
        for k in wu.CONFIG_KEYS:
            self.assertIn(k, c)
        self.assertEqual(c["schema_version"], 2); self.assertEqual(c["study"], "demo"); self.assertEqual(c["decisions"], "D-001, D-002"); self.assertEqual(c["slurm_jobs"], "job-a=11; job-b=12")
        self.assertEqual(c["upload_commit"], "c0ffee"); self.assertEqual(c["source"], "post-hoc-upload-of-committed-results"); self.assertEqual(c["states"], "FP, INT4"); self.assertEqual(c["layer"], 12)
        self.assertEqual(c["status"], "exploratory (demo)"); self.assertEqual(c["limits"], "a limit")

    def test_the_code_reference_can_be_read_from_a_committed_file(self):
        (self.root / "reports/x/freeze.json").write_text('{"commit": "deadbeef", "other": {"c": "x"}}')
        self.assertEqual(wu.build_run(self.root, fam(code_ref={"file": "reports/x/freeze.json", "path": ["commit"]}), "c")["config"]["code_ref"], "deadbeef")
        self.assertEqual(wu.build_run(self.root, fam(code_ref={"file": "reports/x/freeze.json", "path": ["other", "c"]}), "c")["config"]["code_ref"], "x")
        with self.assertRaises(wu.SpecError):
            wu.build_run(self.root, fam(code_ref={"file": "reports/x/freeze.json", "path": ["missing"]}), "c")

    def test_unknown_values_are_n_a(self):
        r = wu.build_run(self.root, fam(freeze=None, code_ref=None, jobs={}, model=None), "c")
        c = r["config"]; self.assertEqual(c["freeze_file"], "n/a"); self.assertEqual(c["freeze_sha256"], "n/a"); self.assertEqual(c["code_ref"], "n/a"); self.assertEqual(c["slurm_jobs"], "n/a"); self.assertEqual(c["model"], "n/a")

    def test_the_freeze_hash_and_the_file_hashes_are_the_real_sha256(self):
        r = wu.build_run(self.root, fam(), "c"); c = r["config"]
        self.assertEqual(c["freeze_file"], "reports/x/freeze.json"); self.assertEqual(c["freeze_sha256"], hashlib.sha256((self.root / "reports/x/freeze.json").read_bytes()).hexdigest())
        fs = json.loads(c["files_sha256"])
        self.assertEqual(set(fs), {"reports/x/report.md", "reports/x/analysis.json", "reports/x/fig_light.png", "reports/x/table.csv", "reports/x/freeze.json"})
        for p, h in fs.items():
            self.assertEqual(h, hashlib.sha256((self.root / p).read_bytes()).hexdigest())

    def test_summary_values_are_read_from_the_files_and_prefixed(self):
        r = wu.build_run(self.root, fam(), "c"); s = r["summary"]
        self.assertEqual(s["headline/macro_effect"], 0.5); self.assertEqual(s["headline/macro_effect_hi"], 0.6); self.assertEqual(s["headline/S1_outcome"], "not met"); self.assertEqual(s["headline/n_covered"], 3)
        self.assertEqual(s["n_tables"], 1); self.assertEqual(s["n_images"], 1); self.assertEqual(s["n_files"], len(r["files"]))
        for k in s:
            self.assertTrue(k.startswith("headline/") or k in ("n_tables", "n_images", "n_files"), k)
        a = json.loads(json.dumps(ANALYSIS)); a["macro"]["FP"]["point"] = 0.77
        (self.root / "reports/x/analysis.json").write_text(json.dumps(a))
        self.assertEqual(wu.build_run(self.root, fam(), "c")["summary"]["headline/macro_effect"], 0.77)

    def test_a_metric_that_does_not_resolve_stops_the_build(self):
        with self.assertRaises(wu.SpecError):
            wu.build_run(self.root, fam(metrics=[("headline/x", "reports/x/analysis.json", ["macro", "INT9", "point"])]), "c")

    def test_a_duplicate_metric_name_or_a_name_without_the_prefix_is_refused(self):
        m = ("headline/a", "reports/x/analysis.json", ["macro", "FP", "point"])
        with self.assertRaises(wu.SpecError):
            wu.build_run(self.root, fam(metrics=[m, m]), "c")
        with self.assertRaises(wu.SpecError):
            wu.build_run(self.root, fam(metrics=[("a", "reports/x/analysis.json", ["macro", "FP", "point"])]), "c")

    def test_a_non_scalar_metric_value_is_refused(self):
        with self.assertRaises(wu.SpecError):
            wu.build_run(self.root, fam(metrics=[("headline/x", "reports/x/analysis.json", ["macro", "FP"])]), "c")

    def test_tables_images_and_files(self):
        r = wu.build_run(self.root, fam(), "c")
        self.assertEqual(r["tables"]["t"]["columns"], ["precision", "language", "effect", "note"]); self.assertEqual(len(r["tables"]["t"]["rows"]), 3)
        self.assertEqual(r["images"], [{"key": "fig/fig_light", "path": "reports/x/fig_light.png", "caption": "fig_light"}])
        paths = [f["path"] for f in r["files"]]
        self.assertEqual(paths, sorted(set(paths)))
        for f in r["files"]:
            self.assertEqual(f["sha256"], hashlib.sha256((self.root / f["path"]).read_bytes()).hexdigest()); self.assertEqual(f["bytes"], (self.root / f["path"]).stat().st_size)
        self.assertEqual(r["bytes_total"], sum(f["bytes"] for f in r["files"]))
        self.assertIn("reports/x/report.md", paths); self.assertIn("reports/x/analysis.json", paths); self.assertIn("reports/x/fig_light.png", paths); self.assertIn("reports/x/freeze.json", paths)

    def test_the_table_files_are_shipped_as_artifact_files_too(self):
        r = wu.build_run(self.root, fam(), "c"); self.assertIn("reports/x/table.csv", [f["path"] for f in r["files"]])

    def test_the_run_is_json_serialisable(self):
        json.dumps(wu.build_run(self.root, fam(), "c"))

    def test_the_status_must_be_exploratory_or_confirmatory(self):
        with self.assertRaises(wu.SpecError):
            wu.build_run(self.root, fam(status="preliminary"), "c")

    def test_the_slug_must_be_a_safe_identifier(self):
        for bad in ("", "Has Space", "a/b", "UPPER", "x" * 100):
            with self.assertRaises(wu.SpecError, msg=bad):
                wu.build_run(self.root, fam(slug=bad), "c")


class History(Fixture):
    def test_a_registered_history_is_attached_and_needs_bits(self):
        old = dict(wu.HISTORIES)
        try:
            wu.HISTORIES["demo"] = lambda root: [{"bits": 32, "effect": 0.5}, {"bits": 4, "effect": 0.9}]
            r = wu.build_run(self.root, fam(history="demo"), "c"); self.assertEqual(r["history"], [{"bits": 32, "effect": 0.5}, {"bits": 4, "effect": 0.9}])
            wu.HISTORIES["demo"] = lambda root: [{"effect": 0.5}]
            with self.assertRaises(wu.SpecError):
                wu.build_run(self.root, fam(history="demo"), "c")
            with self.assertRaises(wu.SpecError):
                wu.build_run(self.root, fam(history="unknown"), "c")
        finally:
            wu.HISTORIES.clear(); wu.HISTORIES.update(old)
        self.assertEqual(wu.build_run(self.root, fam(), "c")["history"], [])

    def test_bits_of_the_precisions(self):
        self.assertEqual({p: wu.BITS[p] for p in ("FP", "INT16", "INT12", "INT8", "INT7", "INT6", "INT5", "INT4", "INT0")}, {"FP": 32, "INT16": 16, "INT12": 12, "INT8": 8, "INT7": 7, "INT6": 6, "INT5": 5, "INT4": 4, "INT0": 0})

    def test_ladder_history_from_an_analysis_json(self):
        rows = wu.ladder_rows(ANALYSIS["macro"], ANALYSIS["retention"])
        by = {r["bits"]: r for r in rows}
        self.assertEqual(sorted(by, reverse=True), [32, 4, 0]); self.assertEqual([r["bits"] for r in rows], [32, 4, 0])
        self.assertEqual(by[32]["effect"], 0.5); self.assertEqual(by[32]["effect_lo"], 0.4); self.assertEqual(by[32]["effect_hi"], 0.6); self.assertNotIn("retention", by[32])
        self.assertEqual(by[4]["retention"], 2.0); self.assertEqual(by[4]["retention_lo"], 1.5); self.assertEqual(by[4]["retention_hi"], 2.5); self.assertEqual(by[4]["effect"], 0.9)
        self.assertEqual(by[0]["retention"], 1.0)

    def test_an_unknown_precision_is_refused(self):
        with self.assertRaises(wu.SpecError):
            wu.ladder_rows({"INT99": {"point": 1, "lo": 0, "hi": 2}}, {})


class Payload(Fixture):
    def test_build_all_selects_families_and_refuses_duplicates_and_unknown_slugs(self):
        fams = [fam(), fam(slug="demo2", name="demo/run2")]
        p = wu.build_all(self.root, fams, upload_commit="c")
        self.assertEqual(p["schema_version"], 2); self.assertEqual([r["id"] for r in p["runs"]], ["v2-demo", "v2-demo2"])
        self.assertEqual([r["id"] for r in wu.build_all(self.root, fams, only=["demo2"], upload_commit="c")["runs"]], ["v2-demo2"])
        with self.assertRaises(wu.SpecError):
            wu.build_all(self.root, fams, only=["nope"], upload_commit="c")
        with self.assertRaises(wu.SpecError):
            wu.build_all(self.root, [fam(), fam()], upload_commit="c")

    def test_the_shipped_files_are_the_union_of_every_run(self):
        p = wu.build_all(self.root, [fam(), fam(slug="d2", files=["reports/x/report.md"], images=[], tables=[])], upload_commit="c")
        self.assertEqual(wu.shipped_files(p), sorted({f["path"] for r in p["runs"] for f in r["files"]}))

    def test_write_payload_creates_json_and_a_tar_of_exactly_the_shipped_files(self):
        p = wu.build_all(self.root, [fam()], upload_commit="c"); out = Path(self._t.name) / "out"
        wu.write_payload(self.root, p, out)
        self.assertEqual(json.loads((out / "payload.json").read_text()), json.loads(json.dumps(p)))
        with tarfile.open(out / "files.tar") as t:
            names = t.getnames()
            self.assertEqual(sorted(names), wu.shipped_files(p))
            for m in t.getmembers():
                self.assertTrue(m.isfile()); self.assertFalse(m.name.startswith("/")); self.assertNotIn("..", m.name.split("/"))
            self.assertEqual(t.extractfile("reports/x/report.md").read(), b"# report\n")

    def test_dry_run_text_lists_runs_and_counts(self):
        p = wu.build_all(self.root, [fam()], upload_commit="c"); text = wu.dry_run_text(p)
        for k in ("v2-demo", "demo/run", "1 runs", "tables 1", "images 1", "metrics 4", "files"):
            self.assertIn(k, text)

    def test_main_build_dry_run_writes_nothing_and_a_real_build_writes_the_payload(self):
        out = Path(self._t.name) / "o2"
        # the CLI runs on the real families only through --root; the fake repository has none, so the families are injected
        old = wu.FAMILIES
        try:
            wu.FAMILIES = [fam()]
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                wu.main(["build", "--root", str(self.root), "--out", str(out), "--commit", "c", "--dry-run"])
            self.assertIn("v2-demo", buf.getvalue()); self.assertFalse(out.exists())
            with contextlib.redirect_stdout(io.StringIO()):
                wu.main(["build", "--root", str(self.root), "--out", str(out), "--commit", "c"])
            self.assertTrue((out / "payload.json").exists()); self.assertTrue((out / "files.tar").exists())
        finally:
            wu.FAMILIES = old


class Dirty(Fixture):
    """The files that are shipped must be the committed ones: a modified or untracked file is refused by the command line."""
    def git(self, *args):
        import subprocess
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.org", *args], cwd=self.root, check=True, capture_output=True)

    def test_dirty_files_are_found(self):
        self.git("init", "-q"); self.git("add", "-A"); self.git("commit", "-q", "-m", "x")
        files = ["reports/x/report.md", "reports/x/analysis.json"]
        self.assertEqual(wu.dirty_files(self.root, files), [])
        (self.root / "reports/x/report.md").write_text("changed\n")
        self.assertEqual(wu.dirty_files(self.root, files), ["reports/x/report.md"])
        (self.root / "reports/x/new.md").write_text("new\n")
        self.assertEqual(wu.dirty_files(self.root, files + ["reports/x/new.md"]), ["reports/x/new.md", "reports/x/report.md"])

    def test_the_build_command_refuses_dirty_files_unless_allowed(self):
        self.git("init", "-q"); self.git("add", "-A"); self.git("commit", "-q", "-m", "x")
        (self.root / "reports/x/report.md").write_text("changed\n"); out = Path(self._t.name) / "o"; old = wu.FAMILIES
        try:
            wu.FAMILIES = [fam()]
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(wu.SpecError):
                    wu.main(["build", "--root", str(self.root), "--out", str(out), "--commit", "c"])
                self.assertFalse(out.exists())
                wu.main(["build", "--root", str(self.root), "--out", str(out), "--commit", "c", "--allow-dirty"])
            self.assertTrue((out / "payload.json").exists())
        finally:
            wu.FAMILIES = old


class FakeTable:
    def __init__(self, columns=None, data=None): self.columns, self.data = columns, data


class FakeImage:
    def __init__(self, path=None, caption=None): self.path, self.caption = path, caption


class FakeArtifact:
    def __init__(self, name, type=None, metadata=None): self.name, self.type, self.metadata, self.files = name, type, metadata, []

    def add_file(self, local_path, name=None): self.files.append((local_path, name))


class FakeRun:
    def __init__(self, step, rid):
        self.step, self.id, self.url = step, rid, f"https://wandb.example/{rid}"
        self.logged, self.summary, self.defined, self.artifacts, self.finished = [], {}, [], [], False

    def log(self, d, step=None, commit=None): self.logged.append(d)

    def define_metric(self, *a, **k): self.defined.append((a, k))

    def log_artifact(self, art): self.artifacts.append(art)

    def finish(self): self.finished = True


class FakeWandb:
    Table, Image, Artifact = FakeTable, FakeImage, FakeArtifact

    def __init__(self, steps=None):
        self.steps, self.inits, self.runs = steps or {}, [], []

    def init(self, **kw):
        self.inits.append(kw); r = FakeRun(self.steps.get(kw["id"], 0), kw["id"]); self.runs.append(r); return r

    def login(self, *a, **k): raise AssertionError("the uploader must not call login: the credentials come from the environment of the cluster")


class Upload(Fixture):
    def setUp(self):
        super().setUp()
        old = dict(wu.HISTORIES); self.addCleanup(lambda: (wu.HISTORIES.clear(), wu.HISTORIES.update(old)))
        wu.HISTORIES["demo"] = lambda root: [{"bits": 32, "effect": 0.5}, {"bits": 4, "effect": 0.9}]
        self.payload = json.loads(json.dumps(wu.build_all(self.root, [fam(history="demo"), fam(slug="d2", name="demo/two")], upload_commit="c")))

    def run_upload(self, wb=None, **kw):
        wb = wb or FakeWandb(); buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = wu.upload(self.payload, self.root, wb, **kw)
        return wb, res, buf.getvalue()

    def test_init_arguments(self):
        wb, res, _ = self.run_upload()
        k = wb.inits[0]
        self.assertEqual((k["project"], k["entity"], k["id"], k["name"], k["group"], k["job_type"], k["resume"]), ("anlp-project", "karma-skz-", "v2-demo", "demo/run", "g", "analysis-upload", "allow"))
        self.assertIn("schema-v2", k["tags"]); self.assertEqual(k["config"]["schema_version"], 2); self.assertEqual([i["id"] for i in wb.inits], ["v2-demo", "v2-d2"])

    def test_summary_tables_images_and_artifact(self):
        wb, res, _ = self.run_upload(); r = wb.runs[0]
        self.assertEqual(r.summary["headline/macro_effect"], 0.5); self.assertEqual(r.summary["n_tables"], 1)
        media = {k: v for d in r.logged for k, v in d.items()}
        self.assertIsInstance(media["table/t"], FakeTable); self.assertEqual(media["table/t"].columns, ["precision", "language", "effect", "note"]); self.assertEqual(len(media["table/t"].data), 3)
        self.assertIsInstance(media["fig/fig_light"], FakeImage); self.assertEqual(media["fig/fig_light"].path, str(self.root / "reports/x/fig_light.png")); self.assertEqual(media["fig/fig_light"].caption, "fig_light")
        art = r.artifacts[0]
        self.assertEqual((art.name, art.type), ("v2-demo-files", "analysis-results")); names = [n for _, n in art.files]
        self.assertIn("reports/x/report.md", names); self.assertIn("MANIFEST.json", names)
        self.assertEqual(art.metadata["upload_commit"], "c")
        self.assertTrue(r.finished)

    def test_history_is_logged_with_bits_as_the_step_metric(self):
        wb, _, _ = self.run_upload(); r = wb.runs[0]
        rows = [d for d in r.logged if "bits" in d]; self.assertEqual([d["bits"] for d in rows], [32, 4]); self.assertTrue(r.defined)
        self.assertTrue(any(a[0] == "bits" for a, _ in r.defined) or any(a == ("bits",) for a, _ in r.defined))
        self.assertEqual([d for d in wb.runs[1].logged if "bits" in d], [])

    def test_a_resumed_run_does_not_get_its_history_again(self):
        wb, _, _ = self.run_upload(FakeWandb({"v2-demo": 7})); r = wb.runs[0]
        self.assertEqual([d for d in r.logged if "bits" in d], []); self.assertEqual(r.summary["headline/macro_effect"], 0.5); self.assertTrue(r.artifacts)
        self.assertEqual(r.logged, [])                                                                          # no history and no tables or images again: a re-run adds no row

    def test_force_media_logs_tables_and_images_again_on_a_resumed_run(self):
        wb, _, _ = self.run_upload(FakeWandb({"v2-demo": 7}), force_media=True); r = wb.runs[0]
        media = {k: v for d in r.logged for k, v in d.items()}
        self.assertIn("table/t", media); self.assertIn("fig/fig_light", media); self.assertEqual([d for d in r.logged if "bits" in d], [])

    def test_only_selects_runs_and_an_unknown_slug_is_an_error(self):
        wb, res, _ = self.run_upload(only=["d2"]); self.assertEqual([i["id"] for i in wb.inits], ["v2-d2"])
        with self.assertRaises(wu.SpecError):
            self.run_upload(only=["zzz"])

    def test_the_result_lists_ids_and_urls_and_the_output_never_mentions_credentials(self):
        _, res, out = self.run_upload()
        self.assertEqual([r["id"] for r in res], ["v2-demo", "v2-d2"]); self.assertEqual(res[0]["url"], "https://wandb.example/v2-demo")
        self.assertIn("v2-demo", out)
        for bad in ("api_key", "netrc", "WANDB_API_KEY", "wandb_v1"):
            self.assertNotIn(bad, out)

    def test_ids_that_are_not_v2_ids_are_refused_so_the_old_runs_cannot_be_touched(self):
        self.payload["runs"][0]["id"] = "c5641dcd"
        with self.assertRaises(wu.SpecError):
            self.run_upload()

    def test_a_wrong_schema_version_or_a_missing_file_is_refused_before_any_run_starts(self):
        wb = FakeWandb(); self.payload["schema_version"] = 1
        with self.assertRaises(wu.SpecError):
            self.run_upload(wb)
        self.assertEqual(wb.inits, [])
        self.payload["schema_version"] = 2; (self.root / "reports/x/report.md").unlink()
        with self.assertRaises(wu.SpecError):
            self.run_upload(wb)
        self.assertEqual(wb.inits, [])

    def test_a_file_whose_hash_differs_from_the_payload_is_refused(self):
        (self.root / "reports/x/report.md").write_text("changed\n"); wb = FakeWandb()
        with self.assertRaises(wu.SpecError):
            self.run_upload(wb)
        self.assertEqual(wb.inits, [])

    def test_the_uploader_never_deletes_or_logs_in(self):
        wb = FakeWandb(); self.run_upload(wb)
        for r in wb.runs:
            self.assertFalse(hasattr(r, "delete"))


class Real(unittest.TestCase):
    """The real committed files: every family builds, ids are unique, nothing is banned, and the values agree with the numbers of RESULTS_SUMMARY.md."""
    @classmethod
    def setUpClass(cls):
        cls.payload = wu.build_all(REPO, wu.FAMILIES, upload_commit="test")
        cls.runs = {r["id"]: r for r in cls.payload["runs"]}

    def test_fifteen_runs_with_unique_v2_ids_and_names(self):
        self.assertEqual(len(self.payload["runs"]), 15); self.assertEqual(len(self.runs), 15)
        self.assertEqual(len({r["name"] for r in self.payload["runs"]}), 15)
        for r in self.payload["runs"]:
            self.assertTrue(r["id"].startswith("v2-")); self.assertGreater(len(r["summary"]), 3, r["id"]); self.assertTrue(r["files"], r["id"])
            for t in ("schema-v2", "post-hoc-upload"):
                self.assertIn(t, r["tags"])
        for slug in ("functional-lm-ladder", "functional-rescue", "functional-f4-association", "association-margin-a2-a3", "belebele-confirmatory-tmc", "belebele-ksens-k-all", "belebele-positive-control", "belebele-prompt-check",
                     "f5-main", "f5-sensitivity-controls", "study-g-main", "study-g-ext-c", "study-g-ext-d", "study-g-ext-b", "llama-consistency-check"):
            self.assertIn("v2-" + slug, self.runs)

    def test_no_run_names_collide_with_the_old_runs(self):
        old = {"baseline/full", "smoke/R0-v3-runner", "smoke/generator-runner", "smoke/vendored", "R2/bos-mask", "R5a/T", "R5a/O", "R5a/T_sdpa", "R5a/T_bf16sae", "R5b/T_rowquant", "R5b/T_mask", "R5b/T_keep", "R5b/O_keep",
               "R1/nobos-floor", "phase1/mask/eq", "phase1/mask/all", "phase1/none/eq", "phase1/none/all"}
        self.assertFalse(old & {r["name"] for r in self.payload["runs"]})

    def test_every_shipped_file_is_a_small_committed_file_below_reports(self):
        for p in wu.shipped_files(self.payload):
            wu.check_file(REPO, p)
        total = sum(f["bytes"] for r in self.payload["runs"] for f in r["files"])
        self.assertLess(total, 60_000_000)

    def test_every_file_is_tracked_by_git(self):
        import subprocess
        tracked = set(subprocess.run(["git", "ls-files", "reports"], cwd=REPO, capture_output=True, text=True, check=True).stdout.split("\n"))
        missing = [p for p in wu.shipped_files(self.payload) if p not in tracked]
        self.assertEqual(missing, [])

    def test_functional_ladder_values_agree_with_the_results_summary(self):
        s = self.runs["v2-functional-lm-ladder"]["summary"]
        self.assertAlmostEqual(s["headline/macro_effect"], 0.704, places=3); self.assertAlmostEqual(s["headline/retention_INT4"], 2.14, places=2); self.assertAlmostEqual(s["headline/retention_INT5"], 1.13, places=2)
        self.assertAlmostEqual(s["headline/retention_INT16"], 1.00, places=2); self.assertEqual(s["headline/n_positive"], 27)
        h = {r["bits"]: r for r in self.runs["v2-functional-lm-ladder"]["history"]}
        self.assertEqual(sorted(h, reverse=True), [32, 16, 12, 8, 7, 6, 5, 4, 0]); self.assertAlmostEqual(h[4]["retention"], 2.14, places=2); self.assertAlmostEqual(h[32]["effect"], 0.704, places=3)

    def test_belebele_confirmatory_is_not_met_and_the_run_is_marked_confirmatory(self):
        r = self.runs["v2-belebele-confirmatory-tmc"]; s = r["summary"]
        self.assertEqual(s["headline/S0_outcome"], "not met"); self.assertAlmostEqual(s["headline/macro_effect"], -0.0001, places=4); self.assertEqual(s["headline/n_covered"], 18)
        self.assertIn("confirmatory", r["tags"]); self.assertEqual(r["config"]["freeze_file"], "reports/functional_validation/evaluation_freeze_mc.json")
        h = {x["bits"]: x for x in r["history"]}; self.assertAlmostEqual(h[32]["accuracy_none"], 0.426, places=3)

    def test_other_headlines(self):
        self.assertAlmostEqual(self.runs["v2-belebele-prompt-check"]["summary"]["headline/macro_diff"], -0.100, places=3)
        self.assertEqual(self.runs["v2-belebele-prompt-check"]["summary"]["headline/macro_label"], "ours better")
        ll = self.runs["v2-llama-consistency-check"]["summary"]
        self.assertEqual(ll["headline/verdict"], "not consistent"); self.assertEqual(ll["headline/P1_count_positive"], 21); self.assertFalse(ll["headline/P1_met"]); self.assertTrue(ll["headline/P2a_met"])
        self.assertAlmostEqual(ll["headline/retention_median_INT4"], 0.856, places=3)
        self.assertEqual(self.runs["v2-study-g-ext-d"]["summary"]["headline/verdict"], "not supported")
        self.assertFalse(self.runs["v2-study-g-ext-b"]["summary"]["headline/H_Gb2_met"]); self.assertTrue(self.runs["v2-study-g-ext-c"]["summary"]["headline/H_Gc1_met"])
        self.assertEqual(self.runs["v2-f5-main"]["summary"]["headline/n_tested"], 1487)
        self.assertAlmostEqual(self.runs["v2-functional-rescue"]["summary"]["headline/rescue_group_INT4"], 0.026, places=3)

    def test_llama_history_carries_the_gemma_reference_and_the_two_estimators(self):
        h = {x["bits"]: x for x in self.runs["v2-llama-consistency-check"]["history"]}
        self.assertEqual(sorted(h), [4, 5, 6, 8]); self.assertAlmostEqual(h[4]["retention_mean"], -3.5706, places=3); self.assertAlmostEqual(h[4]["retention_median"], 0.856, places=3); self.assertAlmostEqual(h[4]["gemma_retention"], 2.136, places=3)

    def test_study_status_words_come_from_results_summary(self):
        for slug, word in (("belebele-confirmatory-tmc", "confirmatory"), ("llama-consistency-check", "exploratory"), ("study-g-main", "exploratory")):
            self.assertIn(word, self.runs["v2-" + slug]["config"]["status"])


if __name__ == "__main__":
    unittest.main()
