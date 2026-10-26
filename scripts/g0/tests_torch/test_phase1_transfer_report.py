import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_transfer_analysis as ta
import phase1_transfer_core as tc
import phase1_transfer_report as rep
import phase1_transfer_run as tr
from test_phase1_transfer_analysis import CORE, SPLIT, gitem, gram_of, group_items_for, pool_gram

KINDS = ("real", "matched_swapped", "random1", "random2", "random3")
CONTEXTS = {10: (1000, 2000), 11: (1001, 2001), 12: (3000, 3001), 13: (3002, 3003)}
CORE = {**CORE, "contexts": [list(CONTEXTS[f]) for f in CORE["feature_ids"]]}                                                      # the planned contexts of the fixture runs (the feature level is checked against them)
ERR = {"INT16": 0.1, "INT6": 5.0, "INT4": 20.0}


def feature_item(err, kind_gap=0.0):
    """An item with FP effect (100, 0), Q/Q effect (100, err), native (100, err / 2)."""
    return {"gram": gram_of([[100.0, 0.0], [100.0, 0.0], [100.0, 0.0], [100.0, err], [100.0, err / 2]]), "cells": list(tc.CELLS), "dloss": [0.1] * 5, "loss_base": [2.0] * 5, "n_rows": 2, "n_target_rows": 5, "n_active_q": 1, "n_positions": 2}


def write_feature_dir(d, state, drop=None, weak=None):
    items, pooled = {}, {}
    for f, ss in CONTEXTS.items():
        for k in KINDS:
            for s in ss:
                items[(f, s, k)] = feature_item(ERR[state] * (1 + 0.01 * f))
            pooled[(f, k)] = {**pool_gram([100.0, 0.0], [100.0, ERR[state]]), "n_contexts": len(ss)}
    if weak:                                                                                                                         # an item whose full-precision effect is tiny in THIS job (the FP/FP cell is recomputed in every job)
        items[weak] = {**items[weak], "gram": gram_of([[0.3, 0.0]] * 3 + [[0.3, ERR[state]], [0.3, ERR[state] / 2]])}
    if drop:
        del items[drop]
    tr.atomic_save({"items": items, "pooled": pooled, "skipped": []}, Path(d) / f"transfer_{state}_c000.pt", f"k-{state}")


def write_group_dir(d, state, langs=("a", "b"), n_sent=6):
    """Two chunk files of the sentences 0-2 and 3-5; both carry the pool of every (language, set) as the real chunks do."""
    items = group_items_for(list(langs), n_sent, lambda l, j: [1.0, 1.9 + 0.01 * j, 0.7, 1.6, 0.6], [lambda l, j: [0.1] * 5] * 3, ntok=lambda l, j: 10 + j)
    pool = {(l, k): {"gram": np.eye(5), "n_rows": 2, "n_contexts": 3} for l in langs for k in ("group", "control0", "control1", "control2")}
    for c, js in enumerate(((0, 1, 2), (3, 4, 5))):
        tr.atomic_save({"items": {k: v for k, v in items.items() if k[1] in js}, "pooled": pool, "skipped": []}, Path(d) / f"group_{state}_c{c:03d}.pt", f"g-{state}-{c}")


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name); self.fdirs, self.gdirs = {}, {}
        for s in ERR:
            self.fdirs[s] = self.root / f"f_{s}"; self.fdirs[s].mkdir(); write_feature_dir(self.fdirs[s], s)
            self.gdirs[s] = self.root / f"g_{s}"; self.gdirs[s].mkdir(); write_group_dir(self.gdirs[s], s)

    def tearDown(self):
        self.tmp.cleanup()


class FeatureLevel(Fixture):
    def test_the_floor_is_fixed_from_the_int16_calibration_items_of_each_kind_and_applied_to_every_state(self):
        res = rep.feature_level(self.fdirs, CORE, SPLIT, 997, n_boot=20, seed=1)
        want = {k: float(np.percentile([ERR["INT16"] * (1 + 0.01 * f) for f in (10, 12) for _ in (0, 1)], 95)) for k in KINDS}
        self.assertEqual(set(res["floors"]), set(KINDS)); self.assertTrue(all(abs(res["floors"][k] - want[k]) < 1e-9 for k in KINDS)); self.assertEqual(res["floor_state"], "INT16")
        for s in ERR:
            self.assertTrue(all(abs(r["floor"] - want[r["kind"]]) < 1e-9 for r in res["states"][s]["rows"].values()))
        self.assertTrue(res["states"]["INT16"]["reference"]); self.assertFalse(res["states"]["INT4"]["reference"]); self.assertEqual(res["states"]["INT4"]["n_items"], 4 * 2 * 5); self.assertEqual(res["states"]["INT4"]["n_skipped"], 0)

    def test_every_table_is_present_per_state_and_the_predictive_results_are_reported_with_their_counts(self):
        res = rep.feature_level(self.fdirs, CORE, SPLIT, 997, n_boot=20, seed=1); st = res["states"]["INT4"]
        self.assertTrue(st["coverage"] and st["components"] and st["specificity"] and st["averaged"] and st["averaged_features"]); self.assertEqual(set(st["predictive"]), {"eligible", "all_items"})
        self.assertEqual(st["predictive"]["eligible"]["result"]["status"], "insufficient"); self.assertEqual(st["predictive"]["eligible"]["n_dropped"], 8)                      # the fixture has no activation or gradient scalars: every item is dropped for the first missing one
        cov = {(r["kind"], r["split"]): r for r in st["coverage"]}; self.assertEqual(cov[("real", "calibration")]["items"], 4); self.assertEqual(cov[("real", "calibration")]["eligible"], 4)

    def test_states_that_do_not_hold_the_same_items_are_an_error_and_so_is_a_missing_floor_state(self):
        write_feature_dir(self.fdirs["INT4"], "INT4"); other = self.root / "f_other"; other.mkdir(); write_feature_dir(other, "INT4", drop=(10, 1000, "real"))
        with self.assertRaisesRegex(ValueError, "same"):
            rep.feature_level({"INT16": self.fdirs["INT16"], "INT4": other}, CORE, SPLIT, 997, n_boot=10, seed=1)
        with self.assertRaisesRegex(ValueError, "floor state"):
            rep.feature_level({"INT4": self.fdirs["INT4"]}, CORE, SPLIT, 997, n_boot=10, seed=1)


def full_item(err):
    """A feature-level item with the stored scalars of the amendments A9 and A10 (activation vectors, gradient predictions), so that the design of the diagnostic keeps it."""
    return {**feature_item(err), "amp_fp": [4.0, 2.0], "amp_q": [3.0, 2.0], "amp_fp_mean": 3.0, "amp_q_mean": 2.5, "resid_norm_fp": 90.0, "resid_norm_q": 91.0, "edit_norm_fp": 30.0, "edit_norm_native": 28.0, "direction_norm": 1.0,
            "g_pred_fp": 0.02, "g_pred_q_fixed": 0.025, "g_pred_q_native": 0.03, "g_norm_fp": 1.0, "g_norm_q": 1.2}


def write_full_feature_dir(d, state):
    items, pooled = {}, {}
    for f, ss in CONTEXTS.items():
        for k in KINDS:
            for s in ss:
                items[(f, s, k)] = full_item(ERR[state] * (1 + 0.01 * f))
            pooled[(f, k)] = {**pool_gram([100.0, 0.0], [100.0, ERR[state]]), "n_contexts": len(ss)}
    tr.atomic_save({"items": items, "pooled": pooled, "skipped": []}, Path(d) / f"transfer_{state}_c000.pt", f"kf-{state}")


def survival_file(n_feat=20):
    corr = {b: torch.tensor([[0.9 - 0.01 * f for f in range(n_feat)], [0.8 - 0.01 * f for f in range(n_feat)]]) for b in (6, 4)}
    return {"languages": ["a", "b"], "bits": [6, 4], "corr": corr, "pooled_corr": {b: torch.tensor([0.85 - 0.01 * f for f in range(n_feat)]) for b in (6, 4)}}


class SurvivalLevel(Fixture):
    def test_the_survival_diagnostic_is_added_per_state_with_both_variants_and_the_p_columns_come_from_the_file(self):
        full = {}
        for s in ERR:
            full[s] = self.root / f"full_{s}"; full[s].mkdir(); write_full_feature_dir(full[s], s)
        core = {**CORE, "languages": ["a", "b"]}; res = rep.feature_level(full, core, SPLIT, 997, n_boot=20, seed=1, survival=survival_file())
        for s in ("INT6", "INT4"):
            st = res["states"][s]; self.assertEqual(set(st["survival"]), {"eligible", "all_items"}); v = st["survival"]["all_items"]
            self.assertEqual(v["n_design"], 8); self.assertEqual(v["n_dropped"], 0); self.assertEqual(v["result"]["status"], "insufficient")                      # eight real items are too few to fit
        d = ta.design(res["states"]["INT6"]["rows"], eligible_only=False, survival=ta.survival_values(survival_file(), core, ("INT6",))["INT6"]); self.assertTrue(np.allclose(d["P"][0], [0.85 - 0.10, 0.9 - 0.10]))
        self.assertNotIn("survival", rep.feature_level(full, core, SPLIT, 997, n_boot=20, seed=1)["states"]["INT6"])

    def test_the_survival_table_and_the_command_line_option_are_written(self):
        full = {}
        for s in ERR:
            full[s] = self.root / f"full_{s}"; full[s].mkdir(); write_full_feature_dir(full[s], s)
        core = {**CORE, "languages": ["a", "b"]}; feat = rep.feature_level(full, core, SPLIT, 997, n_boot=20, seed=1, survival=survival_file()); out = self.root / "surv_out"
        rep.write_outputs(out, feat, None, rep.claim_status(feat, None)); lines = (out / "survival_validation.csv").read_text().strip().split("\n"); head = lines[0].split(",")
        self.assertTrue({"state", "variant", "status", "n_design", "n_calibration", "n_heldout"} <= set(head)); self.assertEqual({l.split(",")[0] for l in lines[1:]}, {"INT6", "INT4"})                  # the survival question is asked at the judged states only (the reference state INT16 may have no entry in the file)
        self.assertEqual(len(lines), 1 + 2 * 2); self.assertNotIn("survival", feat["states"]["INT16"])
        import io, contextlib
        torch.save(survival_file(), self.root / "survival.pt"); (self.root / "freeze.json").write_text(json.dumps({"spec": {"feature_subset": {"dev_per_language": 997}, "group_panel": {"evaluation_sentences": 6, "control_draws": 3}}, "split": SPLIT}))
        (self.root / "core.json").write_text(json.dumps(core)); args = ["--freeze", str(self.root / "freeze.json"), "--core", str(self.root / "core.json"), "--out", str(self.root / "cli_surv"), "--n-boot", "20", "--survival", str(self.root / "survival.pt")]
        for s in ERR:
            args += ["--feature-dir", f"{s}={full[s]}"]
        with contextlib.redirect_stdout(io.StringIO()):
            rep.main(args)
        self.assertTrue((self.root / "cli_surv" / "survival_validation.csv").exists())


class SurvivalTable(Fixture):
    def test_the_reductions_of_a_plus_p_and_a_plus_b_plus_p_are_written_on_their_own_rows(self):
        m = lambda mae: {"lambda": 1, "cv_mae": mae, "heldout_mae": mae}
        res = {"status": "fitted", "n_calibration": 50, "n_heldout": 40, "null_mae": 0.5, "models": {"A": m(0.40), "A+P": m(0.38), "A+B": m(0.30), "A+B+P": m(0.29)},
               "improvement_over_A": {"point": 0.05, "low": 0.01, "high": 0.09}, "improvement_over_diagnostic": {"point": 0.03, "low": -0.01, "high": 0.07}}
        st = {"reference": False, "n_items": 0, "n_skipped": 0, "rows": {}, "coverage": [], "components": [], "specificity": [], "averaged": [], "averaged_features": {}, "predictive": {}, "survival": {"eligible": {"n_design": 90, "n_dropped": 2, "dropped_reasons": {}, "result": res}}}
        out = self.root / "sv_only"; rep.write_outputs(out, {"floors": {}, "floor_state": "INT16", "states": {"INT6": st}}, None, {"claims": {}})
        lines = (out / "survival_validation.csv").read_text().strip().split("\n"); head = lines[0].split(","); rows = {l.split(",")[head.index("model")]: l.split(",") for l in lines[1:]}
        col = lambda name, key: rows[name][head.index(key)]
        self.assertEqual((col("A+P", "reduction_over_A"), col("A+P", "reduction_over_A_low"), col("A+P", "reduction_over_A_high")), ("0.05", "0.01", "0.09")); self.assertEqual(col("A+P", "reduction_over_diagnostic"), "")
        self.assertEqual((col("A+B+P", "reduction_over_diagnostic"), col("A+B+P", "reduction_over_diagnostic_low"), col("A+B+P", "reduction_over_diagnostic_high")), ("0.03", "-0.01", "0.07")); self.assertEqual(col("A+B+P", "reduction_over_A"), "")
        self.assertEqual(col("A", "reduction_over_A"), ""); self.assertEqual(col("A+B", "reduction_over_diagnostic"), ""); self.assertEqual(col("A", "heldout_mae"), "0.4")


class Completeness(Fixture):
    def test_a_state_that_lost_items_fails_the_planned_count_even_when_every_state_lost_the_same(self):
        for s in ERR:
            write_feature_dir(self.fdirs[s], s, drop=(10, 1000, "real"))                                                                 # all states lack the same item: the states agree, the plan does not
        with self.assertRaisesRegex(ValueError, "planned"):
            rep.feature_level(self.fdirs, CORE, SPLIT, 997, n_boot=10, seed=1)

    def test_skipped_contexts_and_skipped_items_count_five_and_one_against_the_plan(self):
        items, pooled = {}, {}
        for f, ss in CONTEXTS.items():
            for k in KINDS:
                for s in ss:
                    if (f, s) != (13, 3003):
                        items[(f, s, k)] = feature_item(1.0)
                pooled[(f, k)] = {**pool_gram([100.0, 0.0], [100.0, 1.0]), "n_contexts": len(ss) - (1 if f == 13 else 0)}
        d = self.root / "skip"; d.mkdir(); tr.atomic_save({"items": items, "pooled": pooled, "skipped": [(13, 3003, "no_fp_active_position")]}, d / "transfer_INT16_c000.pt", "k")
        res = rep.feature_level({"INT16": d}, CORE, SPLIT, 997, n_boot=10, seed=1); self.assertEqual((res["states"]["INT16"]["n_items"], res["states"]["INT16"]["n_skipped"]), (35, 1)); self.assertEqual(res["states"]["INT16"]["planned_items"], 40)
        del items[(10, 1000, "real")]; pooled[(10, "real")] = {**pooled[(10, "real")], "n_contexts": 1}; tr.atomic_save({"items": items, "pooled": pooled, "skipped": [(13, 3003, "no_fp_active_position"), ((10, 1000, "real"), "no_edited_row")]}, d / "transfer_INT16_c000.pt", "k")
        self.assertEqual(rep.feature_level({"INT16": d}, CORE, SPLIT, 997, n_boot=10, seed=1)["states"]["INT16"]["n_items"], 34)                                            # 34 + 5 + 1 = 40

    def test_eligibility_follows_the_reference_state_so_that_every_state_has_the_same_eligible_set(self):
        key = (10, 1000, "real"); write_feature_dir(self.fdirs["INT16"], "INT16", weak=key)                                           # tiny in the INT16 job only
        res = rep.feature_level(self.fdirs, CORE, SPLIT, 997, n_boot=10, seed=1)
        for s in ("INT16", "INT6", "INT4"):
            self.assertFalse(res["states"][s]["rows"][key]["eligible"], s); self.assertTrue(res["states"][s]["rows"][(10, 2000, "real")]["eligible"], s)
        self.assertAlmostEqual(res["states"]["INT4"]["rows"][key]["fp_norm"], 100.0)                                                   # the row keeps its own norm for the normalized error

    def test_the_group_grid_must_be_complete_too(self):
        write_group_dir(self.gdirs["INT6"], "INT6", n_sent=6)
        with self.assertRaisesRegex(ValueError, "planned"):
            rep.group_level({"INT6": self.gdirs["INT6"]}, ["a", "b"], 7, 3, n_boot=10, seed=1)                                          # the plan says seven sentences, the files hold six
        self.assertEqual(rep.group_level({"INT6": self.gdirs["INT6"]}, ["a", "b"], 6, 3, n_boot=10, seed=1)["states"]["INT6"]["planned_items"], 2 * 6 * 4)

    def test_an_external_core_needs_the_cluster_modulus_and_the_languages_on_the_command_line(self):
        freeze = {"spec": {"feature_subset": {"dev_per_language": 997}, "group_panel": {"evaluation_sentences": 6, "control_draws": 3}}, "split": SPLIT}
        (self.root / "freeze.json").write_text(json.dumps(freeze)); (self.root / "core_ext.json").write_text(json.dumps({**CORE, "languages": ["a", "b"], "kind": "intervention-transfer-core-external"}))
        base = ["--freeze", str(self.root / "freeze.json"), "--core", str(self.root / "core_ext.json"), "--out", str(self.root / "o"), "--n-boot", "10", "--feature-dir", f"INT16={self.fdirs['INT16']}"]
        import contextlib, io
        with contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit):
            rep.main(base)
        self.assertIn("--cluster-mod", err.getvalue())
        with contextlib.redirect_stderr(io.StringIO()) as err2, self.assertRaises(SystemExit):
            rep.main(base + ["--cluster-mod", "1000000", "--group-dir", f"INT16={self.gdirs['INT16']}"])
        self.assertIn("--group-languages", err2.getvalue())


class GroupLevel(Fixture):
    def test_per_language_pooled_harms_and_the_decomposition_agree_with_the_analysis_functions(self):
        res = rep.group_level(self.gdirs, ["a", "b"], 6, 3, n_boot=30, seed=1); st = res["states"]["INT6"]
        items, _, _ = ta.load_chunks(self.gdirs["INT6"], "group", "INT6", pools=False); A = ta.group_arrays(items, ["a", "b"], 6, 3); pool = ta.pooled_harms(A); dec = ta.decompose_net(pool["net"])
        row = st["per_language"][1]; self.assertEqual((row["language"], row["n_sentences"], row["n_sentences_total"]), ("b", 6, 6)); self.assertAlmostEqual(row["net_Q_FP"], float(pool["net"][1, 1])); self.assertAlmostEqual(row["C"], float(dec["C"][1])); self.assertAlmostEqual(row["I"], float(dec["I"][1]))
        self.assertAlmostEqual(row["control_FP_FP"], 0.1); self.assertAlmostEqual(row["native_total"], float(dec["native_total"][1])); self.assertEqual(st["a3"], ta.bootstrap_a3(A, 30, 1)); self.assertTrue(res["states"]["INT16"]["reference"])


def a3_result(low, high, medians=True):
    return {"a3": {"median_difference": (low, high), "point": {"holds_medians": medians}}}


def feat_state(sr=(0.6, 0.9), ne=(0.7, 0.9), floor=(2.0, 3.0), imp_a=(0.1, 0.3), crit=True, cos=(0.05, 0.4), eligible=(10, 100), status="fitted", rel=None, rel_only=None, features=(1, 2, 3)):
    mk = lambda metric, lo, hi, split="all": {"kind": "real", "split": split, "metric": metric, "low": lo, "high": hi, "n": 10}
    comp = [mk("scalar_residual_share", *sr), mk("native_error_share", *ne), mk("error_over_floor", *floor)]
    for m in ("C_rel", "W_rel", "I_rel"):
        comp += [mk(m, 0.1, 0.3, "calibration"), mk(m, *((0.2, 0.4) if rel is None or (rel_only and m != rel_only) else rel), "heldout")]
    pred = {"status": status, "improvement_over_A": {"low": imp_a[0], "high": imp_a[1]}, "criterion_met": crit}
    rows = {(f, 1000, "real"): {"kind": "real", "feature": f, "eligible": True} for f in features}; rows.update({(f, 1000, "random1"): {"kind": "random1", "feature": f, "eligible": True} for f in (7, 8)})
    return {"components": comp, "predictive": {"eligible": {"result": pred}}, "averaged": [mk("cos_gap", *cos)], "rows": rows,
            "coverage": [{"kind": "real", "split": "all", "items": eligible[1], "eligible": eligible[0]}, {"kind": "random1", "split": "all", "items": 500, "eligible": 400}]}


def ref_state(sr=(0.95, 1.0), ne=(0.9, 1.0), cos=(-0.02, 0.02)):
    """The INT16 reference state of a feature-level result: rounding only."""
    s = feat_state(sr=sr, ne=ne, floor=(0.4, 0.8), cos=cos); s["reference"] = True; return s


def feats(**states):
    return {"states": {"INT16": ref_state(), **states}}


class Claims(unittest.TestCase):
    def test_a3_needs_the_paired_interval_and_the_median_reading_to_agree_at_each_judged_state_and_overall_only_when_the_states_agree(self):
        g = {"states": {"INT6": a3_result(0.1, 0.4), "INT4": a3_result(0.05, 0.3), "INT16": a3_result(-1, 1)}}; c = rep.claim_status(None, g)["claims"]["A3"]
        self.assertEqual((c["per_state"], c["overall"]), ({"INT6": "supported", "INT4": "supported"}, "supported"))
        g["states"]["INT4"] = a3_result(-0.2, 0.3); c = rep.claim_status(None, g)["claims"]["A3"]; self.assertEqual((c["per_state"]["INT4"], c["overall"]), ("inconclusive", "mixed"))
        g["states"]["INT4"] = a3_result(-0.5, -0.1, medians=False); self.assertEqual(rep.claim_status(None, g)["claims"]["A3"]["per_state"]["INT4"], "reversed")
        g["states"]["INT4"] = a3_result(0.1, 0.4, medians=False); self.assertEqual(rep.claim_status(None, g)["claims"]["A3"]["per_state"]["INT4"], "inconclusive")          # the two readings of the frozen wording disagree
        g["states"]["INT4"] = a3_result(-0.5, -0.1, medians=True); self.assertEqual(rep.claim_status(None, g)["claims"]["A3"]["per_state"]["INT4"], "inconclusive")
        for edge in ((0.0, 0.3), (-0.3, 0.0)):                                                                                          # an interval that touches zero does not decide
            g["states"]["INT4"] = a3_result(*edge, medians=edge[0] == 0.0); self.assertEqual(rep.claim_status(None, g)["claims"]["A3"]["per_state"]["INT4"], "inconclusive")
        del g["states"]["INT4"]; c = rep.claim_status(None, g)["claims"]["A3"]; self.assertEqual((c["per_state"]["INT4"], c["overall"]), ("not evaluated", "not evaluated"))

    def test_h1_needs_both_shares_above_one_half_and_an_error_above_the_floor_and_shows_the_rounding_reference(self):
        f = lambda **kw: feats(INT6=feat_state(**kw), INT4=feat_state(**kw)); st = lambda **kw: rep.claim_status(f(**kw), None)["claims"]["H1"]
        self.assertEqual(st()["overall"], "supported"); self.assertEqual(st(sr=(0.4, 0.9))["overall"], "inconclusive"); self.assertEqual(st(ne=(0.1, 0.4))["overall"], "not supported"); self.assertEqual(st(sr=(0.5, 0.9))["overall"], "inconclusive")
        self.assertEqual(st(floor=(0.5, 0.9))["overall"], "inconclusive")                                                                  # the shares are high but the error does not clear the numerical floor: rounding noise alone gives them
        self.assertEqual(st(floor=(1.0, 3.0))["overall"], "inconclusive")                                                                  # a lower end of exactly one floor does not clear it
        self.assertEqual(st()["reference"], {"scalar_residual_share": (0.95, 1.0), "native_error_share": (0.9, 1.0), "error_over_floor": (0.4, 0.8)})

    def test_h3_and_the_utility_and_the_population(self):
        f = feats(INT6=feat_state(), INT4=feat_state(imp_a=(-0.1, 0.2), crit=False, eligible=(3, 60), features=(1, 2))); c = rep.claim_status(f, None)["claims"]
        self.assertEqual(c["H3"]["per_state"], {"INT6": "supported", "INT4": "inconclusive"}); self.assertEqual(c["H3"]["overall"], "mixed")
        self.assertEqual(c["UTILITY"]["per_state"], {"INT6": "met", "INT4": "not met"}); self.assertEqual(c["UTILITY"]["overall"], "not met")
        self.assertEqual(c["POPULATION"]["INT4"], {"real_items": 60, "eligible_real_items": 3, "eligible_fraction": 0.05, "eligible_features": 2}); self.assertEqual(c["POPULATION"]["INT6"]["eligible_features"], 3)
        f["states"]["INT4"] = feat_state(status="insufficient"); c = rep.claim_status(f, None)["claims"]; self.assertEqual(c["UTILITY"]["per_state"]["INT4"], "insufficient"); self.assertEqual(c["H3"]["per_state"]["INT4"], "inconclusive")

    def test_the_utility_overall_is_met_only_when_every_state_is_met_and_an_unfitted_or_unrun_state_is_not_a_failure(self):
        u = lambda **kw: rep.claim_status(feats(**{k: feat_state(**v) for k, v in kw.items()}), None)["claims"]["UTILITY"]
        self.assertEqual(u(INT6={}, INT4={})["overall"], "met"); self.assertEqual(u(INT6={}, INT4={"crit": False})["overall"], "not met"); self.assertEqual(u(INT6={}, INT4={"status": "insufficient"})["overall"], "insufficient")
        self.assertEqual(u(INT6={})["overall"], "not evaluated"); self.assertEqual(u(INT6={"crit": False}, INT4={"status": "insufficient"})["overall"], "not met"); self.assertEqual(u(INT6={"status": "insufficient"})["overall"], "insufficient")

    def test_h4_needs_an_interval_that_excludes_zero_and_the_rounding_reference_and_is_labelled_as_a_features_only_interval(self):
        f = lambda cos, ref=(-0.02, 0.02): {"states": {"INT16": ref_state(cos=ref), "INT6": feat_state(cos=cos), "INT4": feat_state(cos=cos)}}; h4 = lambda *a, **k: rep.claim_status(f(*a, **k), None)["claims"]["H4"]
        self.assertEqual(h4((0.05, 0.4))["overall"], "supported"); self.assertEqual(h4((-0.5, -0.1))["overall"], "supported"); self.assertEqual(h4((-0.2, 0.3))["overall"], "inconclusive")
        self.assertEqual(h4((0.01, 0.4))["overall"], "inconclusive")                                                                       # the interval excludes zero but overlaps the rounding-only state: noise averaging alone can give it
        self.assertEqual(h4((0.02, 0.4))["overall"], "inconclusive")                                                                      # an interval that only touches the reference does not separate from it
        self.assertEqual(h4((0.05, 0.4), ref=(0.0, 0.2))["overall"], "inconclusive"); self.assertEqual(h4((0.05, 0.4))["interval"], "features only: the pooled fingerprint cannot be resampled by source-sentence cluster")
        self.assertEqual(h4((0.05, 0.4))["reference"], (-0.02, 0.02))
        noref = rep.claim_status({"states": {"INT6": feat_state(), "INT4": feat_state()}}, None)["claims"]["H4"]; self.assertEqual(noref["per_state"], {"INT6": "inconclusive", "INT4": "inconclusive"})

    def test_h2_is_descriptive_and_the_intervals_overlap_when_the_calibration_and_held_out_intervals_do(self):
        c = rep.claim_status(feats(INT6=feat_state(), INT4=feat_state(rel=(0.8, 0.9))), None)["claims"]["H2"]
        one = rep.claim_status(feats(INT6=feat_state(rel=(0.8, 0.9), rel_only="W_rel"), INT4=feat_state()), None)["claims"]["H2"]["per_state"]["INT6"]; self.assertFalse(one["intervals_overlap"]); self.assertEqual(one["overlap_by_component"], {"C_rel": True, "W_rel": False, "I_rel": True})
        self.assertEqual(c["status"], "descriptive"); self.assertTrue(c["per_state"]["INT6"]["intervals_overlap"]); self.assertFalse(c["per_state"]["INT4"]["intervals_overlap"]); self.assertEqual(c["per_state"]["INT4"]["overlap_by_component"], {"C_rel": False, "W_rel": False, "I_rel": False})

    def test_a_state_that_was_not_run_is_not_evaluated(self):
        c = rep.claim_status(feats(INT6=feat_state()), None)["claims"]; self.assertEqual(c["H1"]["per_state"]["INT4"], "not evaluated"); self.assertEqual(c["H1"]["overall"], "not evaluated"); self.assertEqual(c["UTILITY"]["overall"], "not evaluated")

    def test_the_scope_of_the_claims_is_stated_with_the_output(self):
        s = rep.claim_status(feats(INT6=feat_state(), INT4=feat_state()), a3_group())["scope"]
        self.assertIn("feature-held-out", s["held_out"]); self.assertIn("calibration", s["held_out"]); self.assertIn("sentences only", s["group_bootstrap"]); self.assertIn("features and source-sentence clusters", s["feature_bootstrap"])
        self.assertEqual(s["analysis_parameters"], {"min_calibration": 20, "min_heldout": 10, "ridge_strengths": [0.1, 1, 10], "folds": 5, "utility_reduction": 0.10, "h1_floor_multiple": 1.0})


def a3_group():
    return {"states": {"INT6": a3_result(0.1, 0.4), "INT4": a3_result(0.1, 0.4)}}


class Outputs(Fixture):
    def test_the_tables_the_summary_and_the_claim_status_are_written_and_the_json_has_no_nan(self):
        feat = rep.feature_level(self.fdirs, CORE, SPLIT, 997, n_boot=20, seed=1); grp = rep.group_level(self.gdirs, ["a", "b"], 6, 3, n_boot=20, seed=1); out = self.root / "out"
        rep.write_outputs(out, feat, grp, rep.claim_status(feat, grp))
        for name in ("coverage.csv", "component_summary.csv", "specificity.csv", "averaged_fingerprints.csv", "predictive_validation.csv", "per_context_INT4.csv", "group_per_language.csv", "a3.csv", "claim_status.json", "analysis_summary.json"):
            self.assertTrue((out / name).exists() and (out / name).stat().st_size > 0, name)
        text = (out / "analysis_summary.json").read_text(); self.assertNotIn("NaN", text); self.assertNotIn("abs_error", text); json.loads(text); json.loads((out / "claim_status.json").read_text())
        self.assertEqual(len((out / "per_context_INT4.csv").read_text().strip().split("\n")), 1 + 40)
        feats = (out / "averaged_features_INT4.csv").read_text().strip().split("\n"); self.assertEqual(len(feats), 1 + 4 * 5); head = feats[0].split(","); self.assertTrue({"feature", "kind", "split", "pooled_cos", "item_cos", "cos_gap", "coherence_fp"} <= set(head))

    def test_jsonable_turns_numpy_into_plain_values_and_nan_into_null(self):
        x = rep.jsonable({"a": np.float64(1.5), "b": np.array([1, 2]), "c": (np.int64(3), float("nan")), "abs_error": np.ones(3), "d": np.bool_(True), "e": np.float64("nan")})
        self.assertEqual(x, {"a": 1.5, "b": [1, 2], "c": [3, None], "d": True, "e": None}); json.dumps(x)


class Cli(Fixture):
    def test_main_runs_end_to_end_from_the_freeze_the_core_and_the_chunk_directories(self):
        freeze = {"spec": {"feature_subset": {"dev_per_language": 997}, "group_panel": {"evaluation_sentences": 6, "control_draws": 3}}, "split": SPLIT}; core = {**CORE, "languages": ["a", "b"]}
        (self.root / "freeze.json").write_text(json.dumps(freeze)); (self.root / "core.json").write_text(json.dumps(core)); out = self.root / "cli_out"
        args = ["--freeze", str(self.root / "freeze.json"), "--core", str(self.root / "core.json"), "--out", str(out), "--n-boot", "20", "--seed", "1"]
        for s in ERR:
            args += ["--feature-dir", f"{s}={self.fdirs[s]}", "--group-dir", f"{s}={self.gdirs[s]}"]
        rep.main(args); claims = json.loads((out / "claim_status.json").read_text())["claims"]; self.assertIn("A3", claims); self.assertIn("UTILITY", claims); self.assertEqual(claims["A3"]["per_state"]["INT4"] in ("supported", "inconclusive", "reversed"), True)


class CliExternal(Fixture):
    def test_the_external_run_takes_its_own_languages_and_the_cluster_modulus_from_the_command_line(self):
        freeze = {"spec": {"feature_subset": {"dev_per_language": 997}, "group_panel": {"evaluation_sentences": 300, "control_draws": 3}}, "split": SPLIT}; core = {**CORE, "languages": ["a", "b", "c"]}
        (self.root / "freeze.json").write_text(json.dumps(freeze)); (self.root / "core.json").write_text(json.dumps(core)); out = self.root / "ext_out"
        args = ["--freeze", str(self.root / "freeze.json"), "--core", str(self.root / "core.json"), "--out", str(out), "--n-boot", "20", "--seed", "1", "--group-languages", "a", "b", "--cluster-mod", "5", "--group-sentences", "6"]
        for s in ERR:
            args += ["--feature-dir", f"{s}={self.fdirs[s]}", "--group-dir", f"{s}={self.gdirs[s]}"]
        rep.main(args)
        self.assertEqual({r.split(",")[1] for r in (out / "group_per_language.csv").read_text().strip().split("\n")[1:]}, {"a", "b"})
        lines = (out / "group_per_language.csv").read_text().strip().split("\n"); head = lines[0].split(","); self.assertEqual({l.split(",")[head.index("n_sentences_total")] for l in lines[1:]}, {"6"})
        comp = [l for l in (out / "component_summary.csv").read_text().split("\n") if l.startswith("INT4,real,all,normalized_error")][0].split(","); header = (out / "component_summary.csv").read_text().split("\n")[0].split(",")
        self.assertEqual(comp[header.index("n_clusters")], "4")                                                              # the eight sentences of the fixture fall into the clusters 0, 1, 2, 3 modulo 5 (the default 997 gives eight)


if __name__ == "__main__":
    unittest.main()
