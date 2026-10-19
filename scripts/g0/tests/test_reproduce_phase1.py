"""N13 tests for scripts/g0/reproduce_phase1.py (decision D-026; pre-specified in handoff 008 section 4): a clean-room reproduction of every committed Phase 1 result file from the cached run
files with the committed scripts at 4 BLAS threads. The comparison rule is fixed beforehand: JSON compared exactly after dropping keys that hold filesystem paths ('run', at any depth), markdown
byte for byte. The orchestration is tested with a fake runner on a synthetic repository; the real run happens once. Stdlib only."""
import contextlib, io, json, shutil, sys, tempfile, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import reproduce_phase1 as rp   # noqa: E402


class Normalise(unittest.TestCase):
    def test_run_keys_are_dropped_at_any_depth_including_inside_lists(self):
        obj = {"a": 1, "run": "/x", "b": [{"run": "y", "c": 2}], "d": {"run": 3, "e": {"run": 4, "f": 5}}}
        self.assertEqual(rp.normalise_json(obj), {"a": 1, "b": [{"c": 2}], "d": {"e": {"f": 5}}}); self.assertIn("run", obj)          # the input is not modified


class CompareJson(unittest.TestCase):
    def test_equal_documents_have_no_differences_and_run_keys_do_not_count(self):
        self.assertEqual(rp.compare_json({"x": [1.0, 2.0], "run": "/a"}, {"x": [1.0, 2.0], "run": "/b"}), [])

    def test_a_float_differing_by_1e_12_is_a_difference_with_its_path(self):
        d = rp.compare_json({"x": [1.0, 2.0]}, {"x": [1.0, 2.000000000001]}); self.assertEqual(len(d), 1); self.assertEqual(d[0]["path"], "x/1")
        self.assertEqual((d[0]["committed"], d[0]["regenerated"]), (2.0, 2.000000000001))

    def test_missing_extra_keys_length_and_type_differences_are_reported(self):
        self.assertEqual([x["path"] for x in rp.compare_json({"a": 1, "b": 2}, {"a": 1})], ["b"]); self.assertEqual([x["path"] for x in rp.compare_json({"a": 1}, {"a": 1, "c": 3})], ["c"])
        self.assertEqual(len(rp.compare_json({"x": [1, 2]}, {"x": [1, 2, 3]})), 1); self.assertEqual(len(rp.compare_json({"x": 1}, {"x": 1.0})), 1)                    # 1 and 1.0 are different JSON texts
        self.assertEqual(len(rp.compare_json({"x": True}, {"x": 1})), 1)

    def test_nan_equals_nan_because_the_same_code_writes_the_same_text(self):
        self.assertEqual(rp.compare_json({"x": float("nan")}, {"x": float("nan")}), []); self.assertEqual(len(rp.compare_json({"x": float("nan")}, {"x": 1.0})), 1)

    def test_nested_paths_use_slashes_and_list_indices(self):
        d = rp.compare_json({"a": {"b": [{"c": 1}]}}, {"a": {"b": [{"c": 2}]}}); self.assertEqual(d[0]["path"], "a/b/0/c")


class CompareFiles(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, self.d, ignore_errors=True)

    def w(self, name, text):
        p = self.d / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(text); return p

    def test_json_files_differing_only_in_path_valued_run_keys_are_identical(self):
        a = self.w("a.json", json.dumps({"run": "/tmp/x", "v": [1.5]})); b = self.w("b.json", json.dumps({"run": "/home/y", "v": [1.5]})); self.assertEqual(rp.compare_files(a, b)["status"], "identical")

    def test_a_json_difference_reports_the_count_the_first_path_and_the_largest_numeric_difference(self):
        a = self.w("a.json", json.dumps({"x": [1.0, 2.0, 5.0]})); b = self.w("b.json", json.dumps({"x": [1.0, 2.5, 5.000001]})); r = rp.compare_files(a, b)
        self.assertEqual(r["status"], "differs"); self.assertIn("2 differing values", r["detail"]); self.assertIn("first at x/1", r["detail"]); self.assertIn("largest absolute numeric difference 0.5", r["detail"])

    def test_markdown_is_compared_byte_for_byte_and_names_the_first_differing_line(self):
        a = self.w("a.md", "one\ntwo\nthree\n"); b = self.w("b.md", "one\ntwo\nthree\n"); self.assertEqual(rp.compare_files(a, b)["status"], "identical")
        c = self.w("c.md", "one\nTWO\nthree\n"); r = rp.compare_files(a, c); self.assertEqual(r["status"], "differs"); self.assertIn("line 2", r["detail"])
        e = self.w("e.md", "one\ntwo\nthree"); r2 = rp.compare_files(a, e); self.assertEqual(r2["status"], "differs"); self.assertIn("final newline", r2["detail"])   # a missing final newline is a difference

    def test_a_missing_file_is_a_difference_on_either_side(self):
        a = self.w("a.md", "x"); self.assertEqual(rp.compare_files(self.d / "nope.md", a)["status"], "missing regenerated"); self.assertEqual(rp.compare_files(a, self.d / "nope.md")["status"], "missing committed")

    def test_csv_files_are_compared_as_bytes(self):
        a = self.w("a.csv", "x,1\n"); b = self.w("b.csv", "x,1\n"); c = self.w("c.csv", "x,2\n"); self.assertEqual(rp.compare_files(a, b)["status"], "identical"); self.assertEqual(rp.compare_files(a, c)["status"], "differs")


class Steps(unittest.TestCase):
    REPO = Path("/repo"); RUN = Path("/cache/p1-mask/eq"); WORK = Path("/work"); PY = "/venv/bin/python"

    def steps(self, regime="mask", stream="eq"):
        return rp.steps_for(regime, stream, Path(f"/cache/p1-{regime}/{stream}"), self.WORK, self.REPO, self.PY)

    def test_the_mask_streams_have_seven_steps_and_the_bos_free_streams_skip_the_mixed_variants(self):
        self.assertEqual([s.name for s in self.steps("mask", "eq")], ["analysis", "thresholds", "magnitude", "mixed_variants", "language_level", "behavior", "margin"])
        self.assertEqual([s.name for s in self.steps("none", "all")], ["analysis", "thresholds", "magnitude", "language_level", "behavior", "margin"])

    def argv(self, name, regime="mask", stream="eq"):
        return next(s for s in self.steps(regime, stream) if s.name == name).argv

    def test_each_command_has_the_flags_the_original_runs_used(self):
        g = "/repo/scripts/g0/"; w = "/work/mask/eq/"; c = "/repo/reports/phase1/mask/eq/"; run = "/cache/p1-mask/eq"
        self.assertEqual(self.argv("analysis"), [self.PY, "-B", g + "phase1_analysis.py", "--run", run, "--out", w + "analysis", "--seed", "0"])
        self.assertEqual(self.argv("thresholds"), [self.PY, "-B", g + "phase1_threshold.py", "--run", run, "--out", w + "thresholds", "--taus", "2", "3", "5", "--seed", "0"])
        self.assertEqual(self.argv("magnitude"), [self.PY, "-B", g + "phase1_magnitude.py", "--run", run, "--out", w + "magnitude", "--taus", "3", "5", "--bins", "3", "5", "10", "--seed", "0", "--rate-only-json", w + "thresholds/thresholds.json"])
        self.assertEqual(self.argv("mixed_variants"), [self.PY, "-B", g + "phase1_mixed_variants.py", "--run", run, "--out", w + "mixed_variants", "--tau", "5", "--seed", "0", "--analysis-json", w + "analysis/analysis.json"])
        self.assertEqual(self.argv("language_level"), [self.PY, "-B", g + "phase1_language_level.py", "--per-language", c + "per_language.csv", "--meta", c + "meta.json", "--stats", "/repo/reports/phase1/flores_text_stats.json",
                                                         "--survival-pt", run + "/survival.pt", "--out", w + "language_level", "--seed", "0"])
        self.assertEqual(self.argv("behavior"), [self.PY, "-B", g + "phase1_behavior.py", "--run", run, "--stats", "/repo/reports/phase1/flores_text_stats.json", "--out", w + "behavior"])
        self.assertEqual(self.argv("margin"), [self.PY, "-B", g + "phase1_margin.py", "--run", run, "--stream-dir", c.rstrip("/"), "--out", w + "margin", "--seed", "0"])

    def test_outputs_map_each_regenerated_file_to_its_committed_counterpart(self):
        out = {s.name: s.outputs for s in self.steps()}
        self.assertEqual(out["analysis"], [("analysis/analysis.json", "analysis.json"), ("analysis/analysis.md", "analysis.md")])
        self.assertIn(("thresholds/thresholds.json", "thresholds/thresholds.json"), out["thresholds"]); self.assertIn(("thresholds/tau_5.0/analysis.md", "thresholds/tau_5.0/analysis.md"), out["thresholds"])
        self.assertEqual(len(out["thresholds"]), 5)
        self.assertIn(("behavior/behavior.json", "behavior.json"), out["behavior"]); self.assertIn(("behavior/behavior.md", "behavior.md"), out["behavior"])
        self.assertEqual(sorted(c for _, c in out["margin"]), sorted(["margin/margin_summary.json", "margin/margin_summary.md", "margin/magnitude/magnitude.md", "margin/mixed_variants/mixed_variants.md", "margin/language_level/language_level.md"]))
        self.assertEqual(out["mixed_variants"], [("mixed_variants/mixed_variants.json", "mixed_variants/mixed_variants.json"), ("mixed_variants/mixed_variants.md", "mixed_variants/mixed_variants.md")])
        self.assertEqual(out["language_level"], [("language_level/language_level.json", "language_level/language_level.json"), ("language_level/language_level.md", "language_level/language_level.md")])

    def test_input_pairs_are_the_fetched_ada_outputs_against_their_committed_copies(self):
        pairs = rp.input_pairs("none", "all", Path("/cache/p1-none/all"), self.REPO)
        self.assertEqual(pairs, [(Path("/cache/p1-none/all/per_language.csv"), Path("/repo/reports/phase1/none/all/per_language.csv")), (Path("/cache/p1-none/all/meta.json"), Path("/repo/reports/phase1/none/all/meta.json")),
                                 (Path("/cache/p1-none/all/summary.csv"), Path("/repo/reports/phase1/none/all/summary.csv"))])

    def test_the_thread_environment_pins_four_blas_threads_over_whatever_was_set(self):
        e = rp.thread_env({"OMP_NUM_THREADS": "20", "PATH": "/bin"}); self.assertEqual((e["OMP_NUM_THREADS"], e["MKL_NUM_THREADS"], e["PATH"]), ("4", "4", "/bin"))


class Orchestration(unittest.TestCase):
    """A fake runner stands in for the real scripts: it copies each step's committed outputs to the regenerated places, optionally perturbing one of them."""
    def setUp(self):
        self.d = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, self.d, ignore_errors=True); self.repo = self.d / "repo"; self.work = self.d / "work"; self.cache = self.d / "cache"
        c = self.repo / "reports" / "phase1" / "mask" / "eq"
        for rel, text in {"analysis.json": json.dumps({"tests": {"x": 1.5}}), "analysis.md": "# a\n", "per_language.csv": "l,1\n", "meta.json": "{}", "summary.csv": "c,1\n",
                          "thresholds/thresholds.json": json.dumps({"run": "/p", "rows": [1]}), "thresholds/thresholds.md": "t\n", "thresholds/tau_2.0/analysis.md": "2\n", "thresholds/tau_3.0/analysis.md": "3\n", "thresholds/tau_5.0/analysis.md": "5\n",
                          "magnitude/magnitude.json": "{}", "magnitude/magnitude.md": "m\n", "mixed_variants/mixed_variants.json": "{}", "mixed_variants/mixed_variants.md": "v\n", "language_level/language_level.json": "{}", "language_level/language_level.md": "l\n",
                          "behavior.json": "{}", "behavior.md": "b\n", "margin/margin_summary.json": "{}", "margin/margin_summary.md": "s\n", "margin/magnitude/magnitude.md": "mm\n", "margin/mixed_variants/mixed_variants.md": "mv\n",
                          "margin/language_level/language_level.md": "ml\n"}.items():
            p = c / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(text)
        run = self.cache / "p1-mask" / "eq"; run.mkdir(parents=True)
        for f in ("per_language.csv", "meta.json", "summary.csv"): shutil.copyfile(c / f, run / f)

    def runner(self, perturb=None, fail=None):
        def run(argv, env=None, capture_output=True, text=True):
            out = Path(argv[argv.index("--out") + 1]) if "--out" in argv else None
            class R: returncode = 0; stderr = ""; stdout = ""
            if fail and Path(argv[2]).name == fail:
                R.returncode = 1; R.stderr = "boom\nlast line"; return R
            if out is not None:
                name = Path(argv[2]).stem.replace("phase1_", ""); steps = {s.name: s for s in rp.steps_for("mask", "eq", self.cache / "p1-mask" / "eq", self.work, self.repo, "py")}
                step = next(s for k, s in steps.items() if out == self.work / "mask" / "eq" / k)
                for reg, com in step.outputs:
                    src = self.repo / "reports" / "phase1" / "mask" / "eq" / com; dst = self.work / "mask" / "eq" / reg; dst.parent.mkdir(parents=True, exist_ok=True); text = src.read_text()
                    if perturb and com == perturb: text = text.replace("1.5", "1.5000001") if com.endswith(".json") else text + "x"
                    dst.write_text(text)
            return R
        return run

    def test_when_every_regenerated_file_equals_the_committed_one_every_row_is_identical(self):
        rows = rp.reproduce_stream("mask", "eq", self.cache, self.work, self.repo, "py", runner=self.runner())
        self.assertEqual({r["status"] for r in rows}, {"identical"}); files = {r["file"] for r in rows}
        self.assertIn("analysis.json", files); self.assertIn("margin/language_level/language_level.md", files); self.assertIn("per_language.csv", files); self.assertEqual(len(rows), 3 + 2 + 5 + 2 + 2 + 2 + 2 + 5)

    def test_a_perturbed_output_shows_up_as_exactly_one_differing_row_with_its_detail(self):
        rows = rp.reproduce_stream("mask", "eq", self.cache, self.work, self.repo, "py", runner=self.runner(perturb="analysis.json"))
        bad = [r for r in rows if r["status"] != "identical"]; self.assertEqual([r["file"] for r in bad], ["analysis.json"]); self.assertIn("first at tests/x", bad[0]["detail"])

    def test_a_failing_step_is_recorded_with_its_stderr_and_its_outputs_are_missing(self):
        rows = rp.reproduce_stream("mask", "eq", self.cache, self.work, self.repo, "py", runner=self.runner(fail="phase1_threshold.py"))
        failed = [r for r in rows if r["status"] == "step failed"]; self.assertEqual(len(failed), 1); self.assertEqual(failed[0]["step"], "thresholds"); self.assertIn("last line", failed[0]["detail"])
        self.assertTrue(any(r["status"] == "missing regenerated" and r["file"].startswith("thresholds/") for r in rows))

    def test_a_changed_input_copy_is_reported_before_any_step_runs(self):
        (self.cache / "p1-mask" / "eq" / "meta.json").write_text('{"x": 1}'); rows = rp.reproduce_stream("mask", "eq", self.cache, self.work, self.repo, "py", runner=self.runner())
        self.assertEqual([r["status"] for r in rows if r["step"] == "inputs" and r["file"] == "meta.json"], ["differs"])


class Report(unittest.TestCase):
    ROWS = [{"stream": "mask/eq", "step": "analysis", "file": "analysis.json", "status": "identical", "detail": ""}, {"stream": "mask/eq", "step": "analysis", "file": "analysis.md", "status": "identical", "detail": ""}]
    META = {"commit": "abc1234", "threads": 4, "date": "2026-10-19", "python": "3.13.0"}

    def test_the_verdict_is_reproduced_only_if_every_row_is_identical(self):
        self.assertEqual(rp.verdict(self.ROWS), "REPRODUCED"); self.assertEqual(rp.verdict(self.ROWS + [{"stream": "mask/eq", "step": "x", "file": "f", "status": "differs", "detail": "d"}]), "NOT REPRODUCED")
        self.assertEqual(rp.verdict([]), "NOT REPRODUCED")                                                                                  # nothing compared is not a reproduction

    def test_the_report_states_the_rule_the_header_and_lists_every_difference(self):
        md = rp.render_report(self.ROWS, self.META); self.assertIn("**REPRODUCED**", md)
        for needle in ("abc1234", "OMP_NUM_THREADS", "2026-10-19", "D-026", "path-valued", "byte for byte", "2 of 2"): self.assertIn(needle, md)
        bad = self.ROWS + [{"stream": "none/all", "step": "margin", "file": "margin/margin_summary.json", "status": "differs", "detail": "1 differing values; first at readings/i"}]; md2 = rp.render_report(bad, self.META)
        self.assertIn("**NOT REPRODUCED**", md2); self.assertIn("| none/all | margin | margin/margin_summary.json | differs | 1 differing values; first at readings/i |", md2); self.assertIn("2 of 3", md2)


class DryRun(unittest.TestCase):
    def test_dry_run_prints_every_command_with_the_thread_environment_and_runs_nothing(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf): code = rp.main(["--dry-run", "--run-root", "/cache", "--work", "/work", "--python", "/venv/bin/python"])
        out = buf.getvalue(); self.assertEqual(code, 0); self.assertIn("OMP_NUM_THREADS=4", out); self.assertEqual(out.count("phase1_analysis.py"), 4); self.assertEqual(out.count("phase1_mixed_variants.py"), 2)
        self.assertEqual(out.count("phase1_margin.py"), 4); self.assertIn("phase1_summary.py --check", out)


if __name__ == "__main__":
    unittest.main()
