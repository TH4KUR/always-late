import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_transfer_precision_run as pr
from test_phase1_transfer_precision_analysis import OLD, table

CORE_SHA = "c" * 64


def make_freeze(n_per_quartile=4, n_ctx=4):
    feats = []
    for q in range(4):
        for k in range(n_per_quartile):
            f = 100 * q + k; feats.append({"feature": f, "direction_id": 9000 + f, "language_index": f % 3, "feature_fp_norm": 10.0 * (q + 1) + k, "contexts": [1000 * f + j for j in range(n_ctx)], "quartile": q, "rank_hash": f"{q}{k}"})
    return {"inputs_sha256": {"core": CORE_SHA, "fp_norm_table": "t" * 64}, "features": feats}


def make_core(freeze, extra=3):
    ids = [e["feature"] for e in freeze["features"]] + list(range(500, 500 + extra))
    return {"feature_ids": ids, "direction_ids": [9000 + f for f in ids], "feature_language_index": [f % 3 for f in ids], "contexts": [[1000 * f + j for j in range(16)] for f in ids], "languages": ["a", "b", "c"]}


class Order(unittest.TestCase):
    def test_the_audit_order_is_round_robin_over_the_quartiles_by_rank_inside_the_quartile(self):
        fr = make_freeze(3); order = pr.audit_order(fr); self.assertEqual([e["feature"] for e in order[:8]], [0, 100, 200, 300, 1, 101, 201, 301]); self.assertEqual(len(order), 12)
        self.assertEqual(sorted(e["feature"] for e in order), sorted(e["feature"] for e in fr["features"]))                                          # the same features, only the order of execution differs

    def test_every_chunk_of_the_round_robin_order_holds_all_quartiles_equally(self):
        fr = make_freeze(8); plan = pr.plan_audit(fr, make_core(fr), CORE_SHA, chunk_features=8); q = pr.quartile_of(fr)
        for ch in plan["chunks"]:
            self.assertEqual(sorted(q[f] for f in ch), [0, 0, 1, 1, 2, 2, 3, 3])


class Plan(unittest.TestCase):
    def test_the_plan_holds_the_selected_features_in_audit_order_with_the_core_fields_and_chunks_of_the_given_size(self):
        fr = make_freeze(4); core = make_core(fr); plan = pr.plan_audit(fr, core, CORE_SHA, chunk_features=5); self.assertEqual(plan["n_features"], 16); self.assertEqual(plan["n_pairs"], 64); self.assertEqual([len(c) for c in plan["chunks"]], [5, 5, 5, 1])
        sub = plan["core"]; self.assertEqual(sub["feature_ids"], [e["feature"] for e in pr.audit_order(fr)]); i = sub["feature_ids"].index(101); self.assertEqual(sub["direction_ids"][i], 9101); self.assertEqual(sub["contexts"][i], [101000 + j for j in range(4)]); self.assertEqual(sub["feature_language_index"][i], 101 % 3)

    def test_the_default_chunk_holds_sixteen_features_so_that_the_128_features_make_eight_chunks(self):
        fr = make_freeze(32); plan = pr.plan_audit(fr, make_core(fr), CORE_SHA); self.assertEqual([len(c) for c in plan["chunks"]], [16] * 8); self.assertEqual(plan["n_pairs"], 512)

    def test_a_changed_language_index_is_refused(self):
        fr = make_freeze(2); core = make_core(fr); core["feature_language_index"] = [(x + 1) % 3 for x in core["feature_language_index"]]
        with self.assertRaisesRegex(ValueError, "language"):
            pr.plan_audit(fr, core, CORE_SHA)

    def test_a_core_whose_hash_differs_from_the_freeze_a_missing_feature_a_changed_direction_or_other_contexts_are_refused(self):
        fr = make_freeze(2); core = make_core(fr)
        with self.assertRaisesRegex(ValueError, "core"):
            pr.plan_audit(fr, core, "d" * 64)
        c2 = {**core, "feature_ids": core["feature_ids"][1:], "direction_ids": core["direction_ids"][1:], "feature_language_index": core["feature_language_index"][1:], "contexts": core["contexts"][1:]}
        with self.assertRaisesRegex(ValueError, "not in the core"):
            pr.plan_audit(fr, c2, CORE_SHA)
        c3 = {**core, "direction_ids": [d + 1 for d in core["direction_ids"]]}
        with self.assertRaisesRegex(ValueError, "direction"):
            pr.plan_audit(fr, c3, CORE_SHA)
        c4 = {**core, "contexts": [[7, 8, 9, 10] + c[4:] for c in core["contexts"]]}
        with self.assertRaisesRegex(ValueError, "contexts"):
            pr.plan_audit(fr, c4, CORE_SHA)


def vec(seed, v=6):
    g = torch.Generator().manual_seed(seed); return torch.randn(v, generator=g)


def chunk_data(keys, perturb=0.0):
    """Observer vectors and process_items outputs of two modes for the items `keys`: the fp32 vectors are the bf16 ones plus `perturb` times a fixed vector."""
    vb, vf, ob, of = {}, {}, {}, {}
    for n, key in enumerate(keys):
        for i, c in enumerate(pr.px.CELLS):
            base = vec(100 * n + i); vb[(key, c)] = base; vf[(key, c)] = base + perturb * vec(7)
        ob[key] = {"dloss": [-0.01 * (n + 1), -0.02, -0.03, -0.04 * (n + 1), -0.05]}; of[key] = {"dloss": [-0.011 * (n + 1), -0.02, -0.03, -0.045 * (n + 1), -0.05]}
    return {"bf16": vb, "fp32": vf}, {"bf16": ob, "fp32": of}


class Rows(unittest.TestCase):
    def test_a_row_per_item_carries_the_identity_the_quartile_the_cluster_the_cross_mode_metrics_and_the_signed_loss_changes_of_ff_and_qq(self):
        keys = [(11, 5, "real"), (11, 5, "matched_swapped")]; v, o = chunk_data(keys, perturb=0.1); rows, skipped = pr.rows_from_chunk(keys, v, o, "INT4", {11: 2}, {(11, 5): 123}); self.assertEqual(skipped, []); self.assertEqual(len(rows), 2)
        r = rows[0]; self.assertEqual((r["feature"], r["sentence"], r["kind"], r["state"], r["quartile"], r["cluster"]), (11, 5, "real", "INT4", 2, 123))
        G = pr.px.cross_gram(v, keys[0]); m = pr.px.item_metrics(G); self.assertAlmostEqual(r["err_bf16"], m["err_bf16"]); self.assertAlmostEqual(r["arith_qq"], m["arith_qq"]); self.assertGreater(r["arith_qq"], 0.0)
        self.assertEqual((r["dloss_FF_bf16"], r["dloss_QQ_bf16"], r["dloss_FF_fp32"], r["dloss_QQ_fp32"]), (-0.01, -0.04, -0.011, -0.045))

    def test_an_item_that_is_in_neither_mode_is_reported_skipped_and_one_that_is_in_one_mode_only_is_an_error(self):
        keys = [(11, 5, "real"), (11, 6, "real")]; v, o = chunk_data(keys[:1]); rows, skipped = pr.rows_from_chunk(keys, v, o, "INT4", {11: 0}, {(11, 5): 1, (11, 6): 2}); self.assertEqual(skipped, [(11, 6, "real")]); self.assertEqual(len(rows), 1)
        o2 = {"bf16": o["bf16"], "fp32": {}}
        with self.assertRaisesRegex(ValueError, "one mode"):
            pr.rows_from_chunk(keys[:1], v, o2, "INT4", {11: 0}, {(11, 5): 1})

    def test_a_missing_cluster_or_quartile_is_an_error_and_a_loss_change_that_is_none_stays_none(self):
        keys = [(11, 5, "real")]; v, o = chunk_data(keys)
        with self.assertRaisesRegex(KeyError, "cluster|123|\\(11, 5\\)"):
            pr.rows_from_chunk(keys, v, o, "INT4", {11: 0}, {})
        with self.assertRaises(KeyError):
            pr.rows_from_chunk(keys, v, o, "INT4", {}, {(11, 5): 1})
        o["bf16"][keys[0]]["dloss"] = [None] * 5; r, _ = pr.rows_from_chunk(keys, v, o, "INT4", {11: 0}, {(11, 5): 1}); self.assertIsNone(r[0]["dloss_FF_bf16"]); self.assertIsNotNone(r[0]["dloss_FF_fp32"])


class Stop(unittest.TestCase):
    def test_the_stop_rules_extrapolate_the_first_chunk_to_all_chunks_and_compare_with_the_memory_and_time_limits(self):
        d = pr.stop_decision(first_chunk_seconds=1800.0, n_chunks=8, peak_gb=9.9); self.assertAlmostEqual(d["extrapolated_hours"], 4.0); self.assertTrue(d["proceed"]); self.assertTrue(d["memory_ok"] and d["time_ok"])
        self.assertFalse(pr.stop_decision(1800.0, 8, 10.61)["proceed"]); self.assertTrue(pr.stop_decision(1800.0, 8, 10.6)["proceed"]); self.assertFalse(pr.stop_decision(3600.1, 8, 9.0)["proceed"]); self.assertTrue(pr.stop_decision(3600.0, 8, 9.0)["proceed"])
        w = pr.stop_decision(4000.0, 8, 12.0); self.assertEqual((w["memory_ok"], w["time_ok"]), (False, False)); self.assertIn("memory", w["why"]); self.assertIn("time", w["why"])


class Gates(unittest.TestCase):
    def good(self):
        return {"G1": [{"label": "FP", "ok": True}, {"label": "INT4", "ok": True}], "G2": {"bf16": {"q": "torch.bfloat16"}, "fp32": {"q": "torch.float32", "normalized": "torch.float32"}}, "G3": {"bf16": True, "fp32": True}, "G4": {"ok": True, "top1": 0.97}}

    def test_all_four_gates_must_pass_and_a_failure_names_the_gate(self):
        self.assertEqual(pr.gates_ok(self.good()), {"ok": True, "failed": []}); g = self.good(); g["G1"][1]["ok"] = False; self.assertEqual(pr.gates_ok(g)["failed"], ["G1"])
        g = self.good(); g["G2"]["fp32"]["q"] = "torch.bfloat16"; self.assertEqual(pr.gates_ok(g)["failed"], ["G2"]); g = self.good(); g["G3"]["fp32"] = False; self.assertEqual(pr.gates_ok(g)["failed"], ["G3"])
        g = self.good(); g["G2"] = {"fp32": g["G2"]["fp32"]}; self.assertEqual(pr.gates_ok(g)["failed"], ["G2"])                      # both modes must have been probed
        g = self.good(); g["G3"] = {"bf16": True, "fp32": False}; self.assertEqual(pr.gates_ok(g)["failed"], ["G3"]); g = self.good(); g["G3"] = {"bf16": False, "fp32": True}; self.assertEqual(pr.gates_ok(g)["failed"], ["G3"])
        g = self.good(); g["G4"]["ok"] = False; g["G1"] = []; self.assertEqual(pr.gates_ok(g), {"ok": False, "failed": ["G1", "G4"]})

    def test_a_gate_that_is_missing_counts_as_failed_not_as_passed(self):
        g = self.good(); del g["G3"]; self.assertEqual(pr.gates_ok(g)["failed"], ["G3"])


class Assemble(unittest.TestCase):
    def test_the_eligible_population_is_read_separately_from_the_primary_one(self):
        rows = table(n_features=12, per=5, int4_err=100.0, int16_err=10.0, dcos=0.0)
        for i, r in enumerate(rows):
            r["quartile"] = r["feature"] % 4
            if r["state"] == "INT4":
                r["cos_bf16"] = 0.6
                if r["feature"] < 11:
                    r["fp_norm_bf16"] = 50.0; r["fp_norm_fp32"] = 50.0; r["cos_fp32"] = 0.1                                       # most items are below five floors and change a lot
            elif r["feature"] < 11:
                r["fp_norm_bf16"] = 50.0; r["fp_norm_fp32"] = 50.0
        res = pr.assemble_results(rows, {}, OLD, n_boot=20, seed=1); self.assertEqual(res["readings"]["all_real"]["A"]["cosine"], "changes"); self.assertEqual(res["readings"]["eligible_real_old"]["A"]["cosine"], "not separated from zero")

    def test_the_assembly_applies_the_int16_floors_the_batching_control_the_summaries_and_the_readings(self):
        rows = table(n_features=12, per=5, int4_err=100.0, int16_err=10.0, dcos=0.0)
        for r in rows:
            r["quartile"] = r["feature"] % 4
            if r["state"] == "INT4":
                r["cos_bf16"] = 0.6                                                                                         # no arithmetic change of the cosine
        control = {m: {(r["feature"], r["sentence"], "real"): {"FP/FP": 0.01 * (1 if m == "bf16" else 2), "Q/Q": 0.02} for r in rows if r["state"] == "INT16"} for m in ("bf16", "fp32")}
        res = pr.assemble_results(rows, control, OLD, n_boot=20, seed=1); self.assertEqual(set(res), {"rows", "summary", "readings", "coverage", "controls"}); self.assertEqual(set(res["readings"]), {"all_real", "eligible_real_old"})
        self.assertEqual(res["readings"]["all_real"]["A"]["cosine"], "not separated from zero"); self.assertTrue(res["readings"]["all_real"]["B"]["persists_in_both_modes"]); self.assertTrue(res["readings"]["all_real"]["C"]["depends"] is False)
        self.assertAlmostEqual(res["controls"]["median_batch_FF_bf16"], 0.01); self.assertAlmostEqual(res["controls"]["median_batch_FF_fp32"], 0.02); self.assertTrue(all("eof_mode_bf16" in r for r in res["rows"]))
        self.assertIn(("all_real", "real", "INT4", "err_bf16"), res["summary"]); self.assertEqual(res["coverage"][("real", "INT4", "old", "bf16")]["items"], 60)


if __name__ == "__main__":
    unittest.main()
