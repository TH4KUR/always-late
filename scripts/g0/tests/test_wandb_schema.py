"""Tests for the fixed wandb schema (pure functions in g0_wandb) and for scripts/g0/wandb_rebuild.py (rebuild runs from local
result files, verify the schema over a whole project, delete old runs only after verification). No network: fake wandb API."""
import csv, json, math, os, sys, tempfile, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import g0_wandb as g  # noqa: E402
import wandb_rebuild as rb  # noqa: E402


class ConditionRow(unittest.TestCase):
    def test_bits_from_the_condition_name_with_fp_distinct_from_int16(self):
        self.assertEqual(g.condition_bits("FP16 baseline"), 32); self.assertEqual(g.condition_bits("RTN INT16"), 16)
        self.assertEqual(g.condition_bits("RTN INT0"), 0); self.assertEqual(g.condition_bits("RTN INT7"), 7)
        self.assertIsNone(g.condition_bits("something else"))

    def test_their_columns_map_to_fixed_names_and_non_metrics_are_dropped(self):
        r = g.condition_row({"condition": "RTN INT6", "layer": 12, "method": "RTN", "bits": 6, "perplexity": 476.8, "ppl_delta_pct": 4.0,
                             "n_active_features": 7134, "mean_corr": 0.89, "median_corr": 0.9, "survived_>0.9_pct": 51.1, "degraded_0.5_0.9_pct": 48.9,
                             "damaged_<0.5_pct": 0.0, "n_tokens": 499968, "n_total_features": 16384, "loss": 6.17, "perplexity_fp32": 1.0, "cell": "T"})
        self.assertEqual(r["survival_pct"], 51.1); self.assertEqual(r["degraded_pct"], 48.9); self.assertEqual(r["damaged_pct"], 0.0)
        self.assertEqual(set(r) - {"condition"}, {"bits", "perplexity", "ppl_delta_pct", "n_active_features", "mean_corr", "median_corr", "survival_pct",
                                                  "degraded_pct", "damaged_pct", "n_tokens", "loss"})
        self.assertTrue(set(r) - {"condition"} <= set(g.METRIC_KEYS))

    def test_non_finite_and_non_numeric_values_are_dropped_and_rel_l2_is_added(self):
        r = g.condition_row({"condition": "RTN INT8", "median_corr": float("nan"), "perplexity": "bad", "loss": 6.0}, rel_l2=0.0109)
        self.assertNotIn("median_corr", r); self.assertNotIn("perplexity", r); self.assertEqual(r["weight_rel_l2"], 0.0109); self.assertEqual(r["loss"], 6.0)


class RunSummary(unittest.TestCase):
    def rows(self):
        return [g.condition_row(x) for x in (
            {"condition": "FP16 baseline", "perplexity": 458.4, "n_active_features": 7134},
            {"condition": "RTN INT8", "median_corr": 0.97, "survived_>0.9_pct": 99.1, "ppl_delta_pct": 2.5},
            {"condition": "RTN INT6", "median_corr": 0.90, "survived_>0.9_pct": 51.1, "ppl_delta_pct": 4.0},
            {"condition": "RTN INT0", "median_corr": 1.0, "survived_>0.9_pct": 100.0})]

    def test_fixed_names_for_every_run_and_missing_rungs_are_simply_absent(self):
        s = g.run_summary(self.rows())
        self.assertEqual(s["fp_perplexity"], 458.4); self.assertEqual(s["survival_pct_int8"], 99.1); self.assertEqual(s["median_corr_int6"], 0.90)
        self.assertNotIn("survival_pct_int5", s); self.assertNotIn("survival_pct_int0", s)     # INT0 is a control, not a summary rung
        self.assertTrue(set(s) <= set(g.SUMMARY_KEYS))

    def test_the_vocabulary_is_the_same_size_whatever_the_runs(self):
        self.assertEqual(len(set(g.SUMMARY_KEYS)), len(g.SUMMARY_KEYS))
        self.assertLess(len(g.SUMMARY_KEYS), 40)


class NormalizeConfig(unittest.TestCase):
    def test_exactly_the_fixed_keys_with_na_for_missing(self):
        c = g.normalize_config({}, env={})
        self.assertEqual(set(c), set(g.CONFIG_KEYS)); self.assertEqual(c["experiment"], "n/a"); self.assertEqual(c["schema_version"], g.SCHEMA_VERSION)

    def test_legacy_runner_meta_is_mapped_and_the_rest_goes_to_config_extra(self):
        meta = {"run_mode": "g0_core", "bos": "prepend_mask", "token_budget": 500000, "bits": [0, 16], "transformer_lens": "3.9.0", "torch": "2.14.1"}
        c = g.normalize_config(meta, env={"SLURM_JOB_ID": "23643", "G0_EXPERIMENT": "R2"})
        self.assertEqual(c["bos_regime"], "mask"); self.assertEqual(c["n_tokens"], 500000); self.assertEqual(c["stack"], "TL")
        self.assertEqual(c["slurm_job"], "23643"); self.assertEqual(c["experiment"], "R2")
        extra = json.loads(c["config_extra"]); self.assertEqual(extra["run_mode"], "g0_core"); self.assertEqual(extra["torch"], "2.14.1")
        self.assertNotIn("bos", extra)

    def test_live_transformerlens_runs_get_the_same_stack_fields_as_rebuilt_ones(self):
        c = g.normalize_config({"transformer_lens": "3.9.0"}, env={})
        self.assertEqual((c["stack"], c["attn_impl"], c["sae_dtype"], c["quantizer"], c["model"], c["layer"], c["source"]),
                         ("TL", "tl", "fp32", "duan", "google/gemma-2-2b", 12, "live"))

    def test_stack_defaults_never_override_explicit_values_and_code_ref_comes_from_the_environment(self):
        c = g.normalize_config({"transformer_lens": "3.9.0", "attn_impl": "eager"}, env={"G0_CODE_REF": "runner abc123"})
        self.assertEqual(c["attn_impl"], "eager"); self.assertEqual(c["code_ref"], "runner abc123")

    def test_schema_names_pass_through_and_bos_aliases(self):
        c = g.normalize_config({"experiment": "x", "cell": "T", "stack": "HF", "bos_regime": "keep", "attn_impl": "eager"}, env={})
        self.assertEqual((c["experiment"], c["cell"], c["stack"], c["bos_regime"], c["attn_impl"]), ("x", "T", "HF", "keep", "eager"))
        self.assertEqual(g.normalize_config({"bos": "none"}, env={})["bos_regime"], "none")
        self.assertEqual(g.normalize_config({"bos": "prepend_keep"}, env={})["bos_regime"], "keep")


# ------------------------------------------------------------------ rebuild / verify / delete (fake wandb)
class FArt:
    def __init__(self, names):
        self._names = names
    def files(self):
        return [type("F", (), {"name": n}) for n in self._names]


class FRun:
    def __init__(self, id, name, summary, config, state="finished", files=()):
        self.id, self.name, self.summary, self.config, self.state, self.deleted = id, name, summary, config, state, False
        self.created_at = "2026-10-16T00:00:00Z"; self.job_type = "g0"; self._files = list(files)
    def logged_artifacts(self):
        return [FArt(self._files)] if self._files else []
    def delete(self, delete_artifacts=False):
        self.deleted = True


class FApi:
    def __init__(self, runs):
        self.runs_ = runs
    def runs(self, path):
        return list(self.runs_)


def good_run(i):
    return FRun(f"new{i}", f"r{i}", {k: 1.0 for k in list(g.SUMMARY_KEYS)[:5]}, {k: "x" for k in g.CONFIG_KEYS})


def bad_run(i):
    return FRun(f"old{i}", f"old-{i}", {"INT7/median_corr": 0.9, "baseline/median_corr": 1.0}, {"torch": "2.14"})


class Verify(unittest.TestCase):
    def test_a_conforming_project_passes(self):
        v = rb.verify_schema(FApi([good_run(1), good_run(2)]))
        self.assertTrue(v["ok"]); self.assertEqual(v["offending_runs"], [])

    def test_offending_summary_and_config_keys_are_reported_per_run(self):
        v = rb.verify_schema(FApi([good_run(1), bad_run(2)]))
        self.assertFalse(v["ok"]); self.assertEqual([o["id"] for o in v["offending_runs"]], ["old2"])
        self.assertIn("INT7/median_corr", v["offending_runs"][0]["bad_summary_keys"])
        self.assertIn("torch", v["offending_runs"][0]["bad_config_keys"])

    def test_system_and_underscore_keys_are_ignored(self):
        r = good_run(1); r.summary.update({"_runtime": 3, "_wandb": {}, "sys/windows_done": 5, "_step": 1})
        self.assertTrue(rb.verify_schema(FApi([r]))["ok"])

    def test_tables_with_fixed_names_are_allowed(self):
        r = good_run(1); r.summary.update({"conditions": {"_type": "table-file"}, "comparison_vs_published": {"_type": "table-file"}})
        self.assertTrue(rb.verify_schema(FApi([r]))["ok"])


class Delete(unittest.TestCase):
    def test_deletes_only_listed_old_runs_when_everything_else_conforms(self):
        old, new, keep = bad_run(1), good_run(1), bad_run(9)
        out = rb.delete_old(FApi([old, new, keep]), old_ids=["old1"], keep_ids=["old9"])
        self.assertEqual(out["deleted"], ["old1"]); self.assertTrue(old.deleted); self.assertFalse(new.deleted); self.assertFalse(keep.deleted)

    def test_refuses_when_a_non_listed_run_does_not_conform(self):
        old, stray = bad_run(1), bad_run(2)
        out = rb.delete_old(FApi([old, stray]), old_ids=["old1"], keep_ids=[])
        self.assertEqual(out["deleted"], []); self.assertIn("old2", out["blocked_by"]); self.assertFalse(old.deleted)

    def test_never_deletes_a_conforming_run_even_if_listed(self):
        new = good_run(1)
        out = rb.delete_old(FApi([new]), old_ids=["new1"], keep_ids=[])
        self.assertEqual(out["deleted"], []); self.assertFalse(new.deleted)

    def test_a_kept_run_is_never_deleted_even_if_it_is_also_listed(self):
        old, running = bad_run(1), bad_run(9)
        out = rb.delete_old(FApi([old, running]), old_ids=["old1", "old9"], keep_ids=["old9"])
        self.assertEqual(out["deleted"], ["old1"]); self.assertTrue(old.deleted); self.assertFalse(running.deleted)

    def test_requires_a_verified_replacement_for_each_old_run_when_asked(self):
        old = bad_run(1)
        out = rb.delete_old(FApi([old]), old_ids=["old1"], keep_ids=[], replacements={"old1": "baseline/full"})
        self.assertEqual(out["deleted"], []); self.assertIn("old1", out["blocked_by"]); self.assertFalse(old.deleted)
        new = good_run(1); new.name = "baseline/full"
        out = rb.delete_old(FApi([old, new]), old_ids=["old1"], keep_ids=[], replacements={"old1": "baseline/full"})
        self.assertEqual(out["deleted"], ["old1"]); self.assertTrue(old.deleted)
        unfinished = good_run(2); unfinished.name = "x"; unfinished.state = "running"; old2 = bad_run(3)
        out = rb.delete_old(FApi([old2, unfinished]), old_ids=["old3"], keep_ids=[], replacements={"old3": "x"})
        self.assertEqual(out["deleted"], [])                      # a replacement that is still running does not count

    def test_artifact_gaps_ignore_the_run_id_file_and_wandb_table_dumps(self):
        old = bad_run(1); old._files = ["a.csv", "b.csv", "wandb_run_id.txt", "summary.table.json"]
        new = good_run(1); new._files = ["a.csv", "conditions.table.json"]
        self.assertEqual(rb.artifact_gaps(old, new), ["b.csv"])
        self.assertEqual(rb.artifact_gaps(old, new, accepted=["b.csv"]), [])

    def test_a_replacement_that_lacks_files_of_the_old_run_blocks_deletion_unless_accepted(self):
        old = bad_run(1); old._files = ["a.csv", "tokens_2d.pt"]
        new = good_run(1); new.name = "n"; new._files = ["a.csv"]
        out = rb.delete_old(FApi([old, new]), old_ids=["old1"], keep_ids=[], replacements={"old1": "n"})
        self.assertEqual(out["deleted"], []); self.assertIn("old1", out["blocked_by"]); self.assertFalse(old.deleted)
        out = rb.delete_old(FApi([old, new]), old_ids=["old1"], keep_ids=[], replacements={"old1": "n"}, accepted_missing={"old1": ["tokens_2d.pt"]})
        self.assertEqual(out["deleted"], ["old1"]); self.assertTrue(old.deleted)

    def test_dry_run_deletes_nothing(self):
        old, new = bad_run(1), good_run(1)
        out = rb.delete_old(FApi([old, new]), old_ids=["old1"], keep_ids=[], dry_run=True)
        self.assertEqual(out["would_delete"], ["old1"]); self.assertFalse(old.deleted)


class RecordingLogger:
    instances = []
    def __init__(self):
        self.calls = []; RecordingLogger.instances.append(self)
    @classmethod
    def start(cls, **kw):
        lg = cls(); lg.kw = kw; return lg
    active = True
    def log_condition(self, row, **k): self.calls.append(("cond", row["condition"]))
    def log_gates(self, rows, out): self.calls.append(("gates", len(rows)))
    def log_table(self, rows, key="conditions"): self.calls.append(("table", key))
    def set_summary(self, d): self.calls.append(("summary", dict(d)))
    def finish(self, out): self.calls.append(("finish", sorted(p.name for p in Path(out).iterdir() if p.is_file())))


class Rebuild(unittest.TestCase):
    def make(self, d):
        d = Path(d); (d / "src").mkdir()
        rows = [{"layer": 12, "condition": "FP16 baseline", "method": "FP16", "bits": 16, "perplexity": 458.4, "ppl_delta_pct": 0.0, "n_active_features": 7134,
                 "mean_corr": 1.0, "median_corr": 1.0, "survived_>0.9_pct": 100.0, "degraded_0.5_0.9_pct": 0.0, "damaged_<0.5_pct": 0.0, "n_tokens": 499968, "loss": 6.1},
                {"layer": 12, "condition": "RTN INT7", "method": "RTN", "bits": 7, "perplexity": 432.7, "ppl_delta_pct": -5.6, "n_active_features": 7134,
                 "mean_corr": 0.94, "median_corr": 0.95, "survived_>0.9_pct": 81.2, "degraded_0.5_0.9_pct": 18.8, "damaged_<0.5_pct": 0.0, "n_tokens": 499968, "loss": 6.07}]
        with open(d / "src" / "phase3_summary_final.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        (d / "src" / "run_meta.json").write_text("{}"); (d / "src" / "weight_perturbation_INT7.json").write_text(json.dumps({"bits": 7, "rel_l2": 0.02}))
        (d / "src" / "snap_x.pt").write_text("x")
        return d, {"name": "baseline/full", "config": {"experiment": "baseline", "cell": "full", "stack": "TL", "bos_regime": "none"},
                   "summary_csv": "src/phase3_summary_final.csv", "files": ["src/run_meta.json", "src/weight_perturbation_*.json", "src/snap_*.pt"], "tags": ["g0"]}

    def test_rebuild_logs_every_row_through_the_schema_logger_and_uploads_small_files_only(self):
        RecordingLogger.instances.clear()
        with tempfile.TemporaryDirectory() as t:
            root, entry = self.make(t)
            rb.rebuild_run(entry, root=root, project="p", entity=None, logger_cls=RecordingLogger)
        lg = RecordingLogger.instances[0]
        self.assertEqual([c for c in lg.calls if c[0] == "cond"], [("cond", "FP16 baseline"), ("cond", "RTN INT7")])
        self.assertIn(("gates", 2), lg.calls); self.assertIn(("table", "conditions"), lg.calls)
        fin = [c for c in lg.calls if c[0] == "finish"][0][1]
        self.assertIn("phase3_summary_final.csv", fin); self.assertIn("weight_perturbation_INT7.json", fin); self.assertNotIn("snap_x.pt", fin)
        self.assertEqual(lg.kw["name"], "baseline/full"); self.assertEqual(lg.kw["config"]["source"], "rebuilt-from-files")

    def test_optional_fixed_summary_values_and_the_comparison_table_are_logged(self):
        RecordingLogger.instances.clear()
        with tempfile.TemporaryDirectory() as t:
            root, entry = self.make(t)
            (Path(root) / "src" / "baseline_report.json").write_text(json.dumps({"comparison": {"rows": [{"condition": "RTN INT7", "delta": 0.1}]}}))
            entry["summary_extra"] = {"q1_pass": True}; entry["comparison_json"] = "src/baseline_report.json"
            rb.rebuild_run(entry, root=root, project="p", entity=None, logger_cls=RecordingLogger)
        calls = RecordingLogger.instances[0].calls
        self.assertIn(("summary", {"q1_pass": True}), calls); self.assertIn(("table", "comparison_vs_published"), calls)

    def test_rebuild_fails_loudly_if_the_summary_file_is_missing(self):
        with tempfile.TemporaryDirectory() as t:
            root, entry = self.make(t); entry["summary_csv"] = "src/nope.csv"
            with self.assertRaises(FileNotFoundError):
                rb.rebuild_run(entry, root=root, project="p", entity=None, logger_cls=RecordingLogger)


class Backup(unittest.TestCase):
    def test_backup_keeps_id_name_config_summary_of_every_run(self):
        r = good_run(1)
        b = rb.backup_runs(FApi([r, bad_run(2)]))
        self.assertEqual([x["id"] for x in b], ["new1", "old2"]); self.assertIn("summary", b[1]); self.assertEqual(b[1]["summary"]["INT7/median_corr"], 0.9)


if __name__ == "__main__":
    unittest.main()
