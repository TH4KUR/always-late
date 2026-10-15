"""Tests for scripts/g0/g0_wandb.py (wandb logging + closure-protocol gate metrics for the G0 runs).
Run:  python3 -m unittest discover -s scripts/g0/tests -v      (stdlib only; no torch/wandb needed)"""
import json, math, os, sys, tempfile, unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import g0_wandb  # noqa: E402  (module under test)


def row(cond, median_corr, survived=None, ppl=None):
    return {"condition": cond, "median_corr": median_corr,
            "survived_>0.9_pct": survived if survived is not None else 100.0 * (median_corr >= 0.9),
            "perplexity": ppl if ppl is not None else 458.5}


REL = {16: 1.7e-3, 12: 2.3e-3, 8: 2.7e-2, 7: 5.4e-2, 6: 1.1e-1}   # relative L2 weight change per width (from the bf16 emulation)


def ladder(one_minus_r):
    rows = [row("FP16 baseline", 1.0, 100.0), row("RTN INT0", 1.0, 100.0)]
    for b in (16, 12, 8, 7, 6):
        rows.append(row(f"RTN INT{b}", 1.0 - one_minus_r(b)))
    return rows


class GateMetrics(unittest.TestCase):
    def test_int0_identity_true_when_restore_only_is_exact(self):
        g = g0_wandb.compute_gate_metrics(ladder(lambda b: 0.7 * REL[b]), REL)
        self.assertTrue(g["gate0_int0_identity"])

    def test_int0_identity_false_when_restore_only_is_not_exact(self):
        rows = ladder(lambda b: 0.7 * REL[b])
        rows[1]["median_corr"] = 0.9999
        g = g0_wandb.compute_gate_metrics(rows, REL)
        self.assertFalse(g["gate0_int0_identity"])

    def test_int0_identity_false_when_perplexity_differs(self):
        rows = ladder(lambda b: 0.7 * REL[b])
        rows[1]["perplexity"] = 470.0
        self.assertFalse(g0_wandb.compute_gate_metrics(rows, REL)["gate0_int0_identity"])

    def test_signal_limited_when_one_minus_r_scales_linearly_with_perturbation(self):
        g = g0_wandb.compute_gate_metrics(ladder(lambda b: 0.7 * REL[b]), REL)
        self.assertAlmostEqual(g["gate1_slope"], 1.0, places=2)
        self.assertGreaterEqual(g["gate1_median_r_int16"], 0.99)
        self.assertEqual(g["gate1_verdict"], "signal_limited")

    def test_noise_dominated_when_median_r_plateaus_below_threshold(self):
        plateau = {16: 0.040, 12: 0.038, 8: 0.041, 7: 0.043, 6: 0.047}
        g = g0_wandb.compute_gate_metrics(ladder(lambda b: plateau[b]), REL)
        self.assertEqual(g["gate1_verdict"], "noise_dominated")
        self.assertLess(g["gate1_slope"], 0.5)

    def test_incomplete_when_the_ladder_has_too_few_rungs_for_a_slope(self):
        rows = [row("FP16 baseline", 1.0, 100.0), row("RTN INT8", 0.99), row("RTN INT6", 0.97)]
        g = g0_wandb.compute_gate_metrics(rows, REL)
        self.assertEqual(g["gate1_verdict"], "incomplete")
        self.assertIsNone(g["gate1_slope"])
        self.assertIsNone(g["gate0_int0_identity"])


# ------------------------------------------------------------------ logger (fake wandb, no network)
class FakeRun:
    def __init__(self, fail_log=False):
        self.summary, self.logged, self.artifacts, self.finished, self.fail_log = {}, [], [], False, fail_log
    def log(self, d, **kw):
        if self.fail_log:
            raise ConnectionError("network down")
        self.logged.append(d)
    def log_artifact(self, a):
        self.artifacts.append(a)
    def finish(self):
        self.finished = True


class FakeArtifact:
    def __init__(self, name, type=None):
        self.name, self.type, self.files = name, type, []
    def add_file(self, path, name=None):
        self.files.append(Path(path).name)


class FakeTable:
    def __init__(self, columns=None, data=None):
        self.columns, self.data = columns, data


class FakeWandb:
    Artifact, Table = FakeArtifact, FakeTable
    def __init__(self, fail_init=False, run=None):
        self.fail_init, self.run, self.init_calls = fail_init, run or FakeRun(), []
    def init(self, **kw):
        if self.fail_init:
            raise RuntimeError("auth failed")
        self.init_calls.append(kw)
        return self.run


class Exploding:
    def __getattr__(self, name):
        raise AssertionError(f"wandb must not be touched when logging is disabled (accessed {name})")


def start(fw, out, **kw):
    return g0_wandb.G0Logger.start(project="anlp-project", name="t", config={"a": 1}, out_dir=Path(out),
                                   wandb_module=fw, enabled=kw.pop("enabled", True), **kw)


class LoggerBehaviour(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(); self.out = self._tmp.name
    def tearDown(self):
        self._tmp.cleanup()

    def test_disabled_logger_never_touches_wandb(self):
        lg = start(Exploding(), self.out, enabled=False)
        self.assertFalse(lg.active)
        lg.log({"x": 1}); lg.log_condition({"condition": "RTN INT8", "bits": 8}); lg.heartbeat("c", 1, 2, 3.0)
        lg.log_gates([], self.out); lg.finish(self.out)

    def test_env_switch_disables_logging(self):
        with mock.patch.dict(os.environ, {"G0_WANDB": "0"}):
            lg = g0_wandb.G0Logger.start(project="p", name="t", config={}, out_dir=Path(self.out), wandb_module=Exploding())
        self.assertFalse(lg.active)

    def test_init_failure_degrades_to_noop_without_raising(self):
        lg = start(FakeWandb(fail_init=True), self.out)
        self.assertFalse(lg.active)
        lg.log({"x": 1}); lg.finish(self.out)

    def test_log_condition_logs_numeric_metrics_with_width_prefix(self):
        fw = FakeWandb(); lg = start(fw, self.out)
        lg.log_condition({"condition": "RTN INT8", "bits": 8, "median_corr": 0.97, "survived_>0.9_pct": 99.0,
                          "perplexity": 470.5, "method": "RTN", "bad": float("nan")})
        d = fw.run.logged[-1]
        self.assertEqual(d["INT8/median_corr"], 0.97)
        self.assertEqual(d["INT8/survived_gt0.9_pct"], 99.0)
        self.assertNotIn("INT8/method", d); self.assertNotIn("INT8/bad", d)
        lg.log_condition({"condition": "FP16 baseline", "bits": 16, "median_corr": 1.0})
        self.assertIn("baseline/median_corr", fw.run.logged[-1])

    def test_heartbeat_logs_progress_and_gpu_memory(self):
        fw = FakeWandb(); lg = start(fw, self.out)
        lg.heartbeat("RTN INT8", 100, 1953, 42.5, {"cuda:0": 7.5, "cuda:1": 6.9})
        d = fw.run.logged[-1]
        self.assertEqual(d["progress/windows_done"], 100); self.assertEqual(d["progress/windows_total"], 1953)
        self.assertEqual(d["progress/elapsed_s"], 42.5); self.assertEqual(d["mem/cuda:0_gib"], 7.5)

    def test_finish_uploads_artifact_without_snapshots_by_default(self):
        for n in ("RTN_INT8_summary.csv", "run_meta.json", "snap_RTN_INT8.pt", "tokens_2d.pt"):
            (Path(self.out) / n).write_text("x")
        fw = FakeWandb(); lg = start(fw, self.out); lg.finish(self.out)
        files = fw.run.artifacts[0].files
        self.assertIn("RTN_INT8_summary.csv", files); self.assertIn("run_meta.json", files)
        self.assertNotIn("snap_RTN_INT8.pt", files); self.assertTrue(fw.run.finished)

    def test_snapshots_are_uploaded_when_requested(self):
        (Path(self.out) / "snap_RTN_INT8.pt").write_text("x")
        fw = FakeWandb()
        with mock.patch.dict(os.environ, {"G0_WANDB_SNAPS": "1"}):
            lg = start(fw, self.out); lg.finish(self.out)
        self.assertIn("snap_RTN_INT8.pt", fw.run.artifacts[0].files)

    def test_run_id_is_persisted_so_a_restart_resumes_the_same_run(self):
        fw1 = FakeWandb(); start(fw1, self.out)
        fw2 = FakeWandb(); start(fw2, self.out)
        a, b = fw1.init_calls[0], fw2.init_calls[0]
        self.assertEqual(a["id"], b["id"]); self.assertEqual(b["resume"], "allow")
        self.assertTrue((Path(self.out) / "wandb_run_id.txt").exists())

    def test_log_gates_logs_metrics_and_sets_summary(self):
        for b, rel in REL.items():
            (Path(self.out) / f"weight_perturbation_INT{b}.json").write_text(json.dumps({"bits": b, "rel_l2": rel}))
        fw = FakeWandb(); lg = start(fw, self.out)
        lg.log_gates(ladder(lambda b: 0.7 * REL[b]), self.out)
        self.assertEqual(fw.run.summary["gate1_verdict"], "signal_limited")
        self.assertTrue(fw.run.summary["gate0_int0_identity"])
        self.assertAlmostEqual(fw.run.logged[-1]["gate1/slope"], 1.0, places=2)

    def test_log_table_sends_all_columns(self):
        fw = FakeWandb(); lg = start(fw, self.out)
        lg.log_table([{"condition": "RTN INT8", "median_corr": 0.97}, {"condition": "RTN INT6", "median_corr": 0.9}])
        t = fw.run.logged[-1]["summary"]
        self.assertEqual(t.columns, ["condition", "median_corr"]); self.assertEqual(len(t.data), 2)

    def test_logging_errors_never_propagate_into_the_experiment(self):
        fw = FakeWandb(run=FakeRun(fail_log=True)); lg = start(fw, self.out)
        lg.log({"x": 1}); lg.log_condition({"condition": "RTN INT8", "median_corr": 0.9})
        lg.heartbeat("c", 1, 2, 1.0); lg.log_table([{"a": 1}]); lg.finish(self.out)


class FakeCuda:
    def __init__(self, available=True, n=2, gib=8.0, boom=False):
        self._a, self._n, self._g, self._boom = available, n, gib, boom
    def is_available(self):
        if self._boom:
            raise RuntimeError("cuda driver error")
        return self._a
    def device_count(self):
        return self._n
    def max_memory_allocated(self, i):
        return int(self._g * 2**30) + i * 2**20


class FakeTorch:
    def __init__(self, **kw):
        self.cuda = FakeCuda(**kw)


class GpuMemHelper(unittest.TestCase):
    def test_reports_peak_allocated_gib_per_device(self):
        m = g0_wandb.gpu_mem_gib(FakeTorch(n=2, gib=8.0))
        self.assertEqual(sorted(m), ["cuda:0", "cuda:1"])
        self.assertAlmostEqual(m["cuda:0"], 8.0, places=3)

    def test_returns_empty_when_cuda_is_unavailable(self):
        self.assertEqual(g0_wandb.gpu_mem_gib(FakeTorch(available=False)), {})

    def test_never_raises_even_if_the_cuda_runtime_fails(self):
        self.assertEqual(g0_wandb.gpu_mem_gib(FakeTorch(boom=True)), {})


if __name__ == "__main__":
    unittest.main()
