import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_transfer_precision_chunks as ch
import phase1_transfer_precision_run as pr
from test_phase1_transfer_precision_run import CORE_SHA, make_core, make_freeze

KINDS = ("real", "matched_swapped")
OLD = {"real": 20.0, "matched_swapped": 10.0}


def make_plan(n_per_quartile=2, chunk_features=4, n_ctx=2):
    fr = make_freeze(n_per_quartile, n_ctx=n_ctx); return fr, pr.plan_audit(fr, make_core(fr), CORE_SHA, chunk_features=chunk_features)


class Fake:
    """A fake `process`: deterministic effect vectors per (item, cell); the state scales the quantized cells, the fp32 mode adds a small fixed perturbation, a second execution (smaller max_rows) another."""
    def __init__(self, skip=(), seconds=None):
        self.calls = []; self.skip = set(skip); self.seconds = seconds

    def __call__(self, chunk, state, mode, max_rows):
        self.calls.append((tuple(chunk), state, mode, max_rows)); out, vec, skipped = {}, {}, []
        plan_ctx = self.ctx
        for f in chunk:
            for s in plan_ctx[f]:
                for kind in KINDS:
                    key = (f, s, kind)
                    if key in self.skip:
                        skipped.append(key); continue
                    for i, c in enumerate(pr.px.CELLS):
                        g = torch.Generator().manual_seed(1000 * (f % 97) + 10 * s % 997 + i); base = torch.randn(6, generator=g)
                        if state != "FP" and c in ("Q/Q", "FP/Q", "Q/Q-native", "Q/FP"):
                            base = base * (1.0 + (0.5 if state == "INT4" else 0.01))
                        pert = (0.02 if mode == "fp32" else 0.0) * torch.ones(6) + (0.001 if max_rows < 256 else 0.0) * torch.arange(6.0)
                        vec[(key, c)] = base + pert
                    out[key] = {"dloss": [-0.01, -0.02, -0.03, -0.04, -0.05]}
        return out, vec, skipped


def run(plan, fake, out_dir, **kw):
    fake.ctx = dict(zip(plan["core"]["feature_ids"], plan["core"]["contexts"])); quart = {f: i % 4 for i, f in enumerate(plan["core"]["feature_ids"])}; clusters = {(f, s): 7000 + s % 50 for f, c in fake.ctx.items() for s in c}
    return ch.run_chunks(plan, fake, out_dir, "sha-of-the-freeze", quart, clusters, log=lambda *a: None, **kw)


class Loop(unittest.TestCase):
    def test_per_chunk_every_state_and_mode_is_processed_and_the_int16_state_a_second_time_with_the_smaller_batch(self):
        fr, plan = make_plan(); fake = Fake()
        with tempfile.TemporaryDirectory() as d:
            res = run(plan, fake, d)
        self.assertEqual(len(plan["chunks"]), 2); per_chunk = len(fake.calls) // 2; self.assertEqual(per_chunk, 6)
        first = [c for c in fake.calls if c[0] == tuple(plan["chunks"][0])]; self.assertEqual(sorted((s, m, r) for _, s, m, r in first), sorted([("INT16", "bf16", 256), ("INT16", "bf16", 192), ("INT16", "fp32", 256), ("INT16", "fp32", 192), ("INT4", "bf16", 256), ("INT4", "fp32", 256)]))
        n_items = plan["n_pairs"] * 2; self.assertEqual(len(res["rows"]), 2 * n_items); self.assertEqual({r["state"] for r in res["rows"]}, {"INT16", "INT4"}); self.assertEqual(res["skipped"], [])

    def test_the_batching_control_is_the_norm_of_the_difference_between_the_two_int16_executions_in_each_mode(self):
        fr, plan = make_plan(); fake = Fake()
        with tempfile.TemporaryDirectory() as d:
            res = run(plan, fake, d)
        want = float((0.001 * torch.arange(6.0)).norm()); key = (plan["core"]["feature_ids"][0], plan["core"]["contexts"][0][0], "real")
        for m in ("bf16", "fp32"):
            self.assertAlmostEqual(res["control"][m][key]["FP/FP"], want, places=6); self.assertAlmostEqual(res["control"][m][key]["Q/Q"], want, places=6)
        self.assertEqual(len(res["control"]["bf16"]), plan["n_pairs"] * 2)

    def test_the_rows_carry_the_cross_mode_metrics_the_quartile_and_the_cluster(self):
        fr, plan = make_plan(); fake = Fake()
        with tempfile.TemporaryDirectory() as d:
            res = run(plan, fake, d)
        f0 = plan["core"]["feature_ids"][0]; r = [x for x in res["rows"] if x["feature"] == f0 and x["state"] == "INT4" and x["kind"] == "real"][0]; self.assertEqual(r["quartile"], 0); self.assertGreaterEqual(r["cluster"], 7000)
        self.assertAlmostEqual(r["arith_ff"], float((0.02 * torch.ones(6)).norm()), places=5); self.assertGreater(r["err_bf16"], 0.0)

    def test_a_skipped_context_is_reported_and_leaves_no_row_and_no_control_entry(self):
        fr, plan = make_plan(); f0 = plan["core"]["feature_ids"][0]; s0 = plan["core"]["contexts"][0][0]; fake = Fake(skip=[(f0, s0, "real"), (f0, s0, "matched_swapped")])
        with tempfile.TemporaryDirectory() as d:
            res = run(plan, fake, d)
        self.assertEqual(sorted(set(res["skipped"])), sorted([(f0, s0, "real"), (f0, s0, "matched_swapped")])); self.assertFalse(any(r["feature"] == f0 and r["sentence"] == s0 for r in res["rows"])); self.assertNotIn((f0, s0, "real"), res["control"]["bf16"])


class Resume(unittest.TestCase):
    def test_a_finished_chunk_is_read_from_its_checkpoint_and_not_processed_again(self):
        fr, plan = make_plan(); fake = Fake()
        with tempfile.TemporaryDirectory() as d:
            a = run(plan, fake, d); n = len(fake.calls); fake2 = Fake(); b = run(plan, fake2, d)
            self.assertEqual(len(fake2.calls), 0); self.assertEqual(a["rows"], b["rows"]); self.assertEqual(a["control"], b["control"]); self.assertEqual(len(fake.calls), n)

    def test_a_checkpoint_of_another_freeze_or_chunk_content_is_not_taken_for_this_one(self):
        fr, plan = make_plan(); fake = Fake()
        with tempfile.TemporaryDirectory() as d:
            run(plan, fake, d); fake.ctx = dict(zip(plan["core"]["feature_ids"], plan["core"]["contexts"])); quart = {f: 0 for f in fake.ctx}; clusters = {(f, s): 1 for f, c in fake.ctx.items() for s in c}; fake2 = Fake(); fake2.ctx = fake.ctx
            ch.run_chunks(plan, fake2, d, "another-freeze-sha", quart, clusters, log=lambda *a: None); self.assertEqual(len(fake2.calls), 12)

    def test_a_checkpoint_of_the_same_features_with_other_contexts_is_not_reused(self):
        fr, plan = make_plan(); fake = Fake()
        with tempfile.TemporaryDirectory() as d:
            run(plan, fake, d); plan2 = {**plan, "core": {**plan["core"], "contexts": [[c + 1 for c in ctx] for ctx in plan["core"]["contexts"]]}}; fake2 = Fake(); run(plan2, fake2, d); self.assertEqual(len(fake2.calls), 12)

    def test_only_the_missing_chunk_is_processed_after_a_partial_run(self):
        fr, plan = make_plan(); fake = Fake()
        with tempfile.TemporaryDirectory() as d:
            run(plan, fake, d, n_chunks_limit=1); self.assertEqual(len(fake.calls), 6); fake2 = Fake(); res = run(plan, fake2, d); self.assertEqual(len(fake2.calls), 6); self.assertEqual({c[0] for c in fake2.calls}, {tuple(plan["chunks"][1])}); self.assertEqual(len(res["rows"]), 2 * plan["n_pairs"] * 2)


class Stop(unittest.TestCase):
    def clock(self, step):
        t = [0.0]
        def now():
            t[0] += step; return t[0]
        return now

    def test_the_instrument_run_processes_one_chunk_and_reports_the_extrapolation_without_stopping(self):
        fr, plan = make_plan(); fake = Fake()
        with tempfile.TemporaryDirectory() as d:
            res = run(plan, fake, d, n_chunks_limit=1, clock=self.clock(100.0), peak_gb=lambda: 9.0)
        self.assertEqual(len(res["rows"]), plan["n_pairs"] * 2 * 2 // 2); self.assertEqual(res["decision"]["peak_gb"], 9.0); self.assertAlmostEqual(res["decision"]["extrapolated_hours"], res["chunk_seconds"][0] * 2 / 3600.0); self.assertTrue(res["decision"]["proceed"])

    def test_only_the_first_processed_chunk_decides_and_an_instrument_run_never_raises(self):
        fr, plan = make_plan(); seq = iter([0.0, 10.0, 10.0, 1e7]); fake = Fake()
        with tempfile.TemporaryDirectory() as d:
            res = run(plan, fake, d, clock=lambda: next(seq), peak_gb=lambda: 9.0)
        self.assertAlmostEqual(res["decision"]["extrapolated_hours"], 10.0 * 2 / 3600.0); self.assertEqual(res["chunk_seconds"], [10.0, 1e7 - 10.0]); fake = Fake()
        with tempfile.TemporaryDirectory() as d:
            res = run(plan, fake, d, n_chunks_limit=1, clock=self.clock(1e6), peak_gb=lambda: 12.0)
        self.assertFalse(res["decision"]["proceed"]); self.assertEqual(len(fake.calls), 6)

    def test_a_full_run_stops_with_the_decision_when_the_first_chunk_projects_beyond_the_allowance_or_the_memory_limit(self):
        fr, plan = make_plan(); fake = Fake()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ch.StopAudit) as cm:
                run(plan, fake, d, clock=self.clock(1e6), peak_gb=lambda: 9.0)
            self.assertFalse(cm.exception.decision["time_ok"]); self.assertEqual(len(fake.calls), 6)                  # the first chunk was processed, the second was not
        fake = Fake()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ch.StopAudit) as cm:
                run(plan, fake, d, clock=self.clock(1.0), peak_gb=lambda: 10.7)
            self.assertFalse(cm.exception.decision["memory_ok"])


class Finish(unittest.TestCase):
    def test_the_stop_decision_and_the_skipped_items_are_written_to_the_readings_file(self):
        fr, plan = make_plan(); f0 = plan["core"]["feature_ids"][0]; s0 = plan["core"]["contexts"][0][0]; fake = Fake(skip=[(f0, s0, "real")])
        with tempfile.TemporaryDirectory() as d:
            res = run(plan, fake, d, n_chunks_limit=1, peak_gb=lambda: 8.5); out = Path(d) / "r"; ch.write_results(res, OLD, out, n_boot=10, seed=1, gates={}, meta={})
            rd = json.loads((out / "readings.json").read_text())
        self.assertEqual(rd["decision"]["peak_gb"], 8.5); self.assertEqual(rd["skipped"], [[f0, s0, "real"]]); self.assertEqual((rd["n_boot"], rd["seed"]), (10, 1))

    def test_the_results_are_assembled_written_and_the_readings_are_serializable(self):
        fr, plan = make_plan(); fake = Fake()
        with tempfile.TemporaryDirectory() as d:
            res = run(plan, fake, d); out = Path(d) / "result"; summary = ch.write_results(res, OLD, out, n_boot=20, seed=1, gates={"G1": []}, meta={"x": 1})
            names = sorted(p.name for p in out.iterdir()); self.assertEqual(names, ["gates.json", "meta.json", "per_item.csv", "readings.json", "summary.csv"]); rd = json.loads((out / "readings.json").read_text()); self.assertEqual(set(rd["readings"]), {"all_real", "eligible_real_old"})
            self.assertIn("A", rd["readings"]["all_real"]); self.assertIn("median_batch_FF_bf16", rd["controls"]); self.assertEqual(json.loads((out / "meta.json").read_text())["x"], 1); self.assertTrue(any("INT4" in k for k in rd["coverage"]))


if __name__ == "__main__":
    unittest.main()
