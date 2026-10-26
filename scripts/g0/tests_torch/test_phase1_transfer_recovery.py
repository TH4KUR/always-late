import copy
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_transfer_analysis as ta
import phase1_transfer_cross_corpus as cc
import phase1_transfer_recovery as rec
import phase1_transfer_report as rep
from test_phase1_transfer_analysis import prow

FLORES = {"real": 30.0, "matched_swapped": 27.0, "random1": 27.5, "random2": 27.2, "random3": 26.9}
KINDS = ("real", "matched_swapped", "random1", "random2", "random3")
STATES = ("INT16", "INT6", "INT4")


def make_rows(seed, own_floors=FLORES, n_cal=24, n_held=18, per=4, signal=True, noise=0.03):
    """The per-item rows of one state of a corpus with a known structure: the normalized error of the real direction follows the common error level b of the controls of its context (the generic sensitivity)."""
    rng = np.random.default_rng(seed); rows = {}
    for f in list(range(n_cal)) + [1000 + i for i in range(n_held)]:
        sp = "calibration" if f < 1000 else "heldout"
        for j in range(per):
            sc = 1000 * f + j; b = float(rng.uniform(0.1, 1.0)); fp_real = float(rng.uniform(100.0, 400.0))
            for k in KINDS:
                fp = fp_real if k == "real" else float(rng.uniform(20.0, 60.0)); ne = (0.2 + (2.0 * b if signal else 0.0) + float(rng.normal(0, noise))) if k == "real" else b
                rows[(f, sc, k)] = {**prow(f, sc, k, err=ne, split=sp), "cluster": sc, "fp_norm": fp, "error_norm": ne * (fp + own_floors[k]), "eligible": fp > 5 * own_floors[k], "act_cos": float(rng.uniform(0, 1)), "firing": float(rng.uniform(0, 1))}
    return rows


def locked_from(designs, floors=FLORES):
    """The locked-model JSON structure of reports/publication_reassessment_2026-10-26/locked_flores_models.json, built from the designs of the main corpus (calibration fits by the frozen protocol)."""
    out = {"status": "test", "target": "t", "eligibility": "e", "models": {}}
    for s, d in designs.items():
        cal = d["split"] == "calibration"; fits = {}; cv = {}
        for name, groups in ta.MODEL_GROUPS.items():
            X = np.hstack([d[g][cal] for g in groups]); ch = ta.choose_lambda(X, d["y"][cal], d["feature"][cal]); fit = ta.ridge_fit(X, d["y"][cal], ch["lambda"])
            fits[name] = {"mean": fit["mean"].tolist(), "scale": fit["scale"].tolist(), "coef": fit["coef"].tolist(), "intercept": fit["intercept"], "lambda": ch["lambda"]}; cv[name] = ch["cv_mae"][ch["lambda"]]
        out["models"][s] = {"calibration_selected_baseline": min(ta.BASELINES, key=lambda n: cv[n]), "null_calibration_median": float(np.median(d["y"][cal])), "floor_by_kind": dict(floors), "groups": {k: list(v) for k, v in ta.MODEL_GROUPS.items()},
                            "column_order": {k: list(v) for k, v in ta.COLUMNS.items()}, "fits": fits}
    return out


def committed_mae(designs, locked):
    """The rows of results/main/predictive_validation.csv that the locked models must reproduce (state, variant, model, heldout_mae)."""
    out = []
    for s, d in designs.items():
        held = d["split"] == "heldout"
        for name, groups in ta.MODEL_GROUPS.items():
            f = locked["models"][s]["fits"][name]; m = {k: np.array(f[k], dtype=float) for k in ("mean", "scale", "coef")}; m["intercept"] = f["intercept"]
            out.append({"state": s, "variant": "eligible", "model": name, "heldout_mae": float(np.abs(ta.ridge_predict(m, np.hstack([d[g][held] for g in groups])) - d["y"][held]).mean())})
    return out


def sd(rows):
    d = ta.design(rows); d["columns"] = {k: list(v) for k, v in ta.COLUMNS.items()}; return d


class Renormalize(unittest.TestCase):
    def test_the_normalized_error_uses_the_floor_of_the_kind_and_eligibility_the_five_fold_floor_of_the_fp_effect_only(self):
        rows = {(1, 1, "real"): {"kind": "real", "fp_norm": 150.0, "error_norm": 60.0, "normalized_error": 9.9, "eligible": False}, (1, 1, "random2"): {"kind": "random2", "fp_norm": 140.0, "error_norm": 30.0, "normalized_error": 9.9, "eligible": False}}
        out = rec.renormalize(rows, FLORES)
        self.assertAlmostEqual(out[(1, 1, "real")]["normalized_error"], 60.0 / (150.0 + 30.0)); self.assertAlmostEqual(out[(1, 1, "random2")]["normalized_error"], 30.0 / (140.0 + 27.2))
        self.assertFalse(out[(1, 1, "real")]["eligible"]); self.assertTrue(out[(1, 1, "random2")]["eligible"])
        r2 = rec.renormalize({(1, 1, "real"): {"kind": "real", "fp_norm": 150.0001, "error_norm": 1.0, "normalized_error": 0.0, "eligible": False}}, FLORES); self.assertTrue(r2[(1, 1, "real")]["eligible"])

    def test_the_input_is_not_changed_a_missing_error_stays_missing_and_other_fields_are_kept(self):
        rows = {(1, 1, "real"): {"kind": "real", "fp_norm": 200.0, "error_norm": None, "normalized_error": 0.5, "eligible": True, "act_cos": 0.3}}
        before = copy.deepcopy(rows); out = rec.renormalize(rows, FLORES); self.assertEqual(rows, before)
        self.assertIsNone(out[(1, 1, "real")]["normalized_error"]); self.assertEqual(out[(1, 1, "real")]["act_cos"], 0.3); self.assertTrue(out[(1, 1, "real")]["eligible"])          # the FP effect alone decides

    def test_eligibility_never_depends_on_a_quantized_quantity(self):
        rows = make_rows(1); other = {k: {**r, "error_norm": r["error_norm"] * 37 + 5, "normalized_error": 123.0, "cosine": 0.1, "q_norm": 9e3} for k, r in rows.items()}
        a, b = rec.renormalize(rows, FLORES), rec.renormalize(other, FLORES); self.assertEqual({k: r["eligible"] for k, r in a.items()}, {k: r["eligible"] for k, r in b.items()})

    def test_with_the_floors_of_the_corpus_itself_the_main_normalized_error_is_reproduced(self):
        rows = make_rows(2); out = rec.renormalize(rows, FLORES)
        self.assertTrue(all(abs(out[k]["normalized_error"] - rows[k]["normalized_error"]) < 1e-12 and out[k]["eligible"] == rows[k]["eligible"] for k in rows))


class DesignOf(unittest.TestCase):
    def test_control_rows_without_a_normalized_error_are_left_out_and_counted_and_real_rows_without_one_are_dropped_by_the_design(self):
        rows = rec.renormalize(make_rows(5), FLORES); k = next(k for k in rows if k[2] == "random1"); rows[k]["error_norm"] = None; rows[k]["normalized_error"] = None
        kr = next(k for k, r in rows.items() if k[2] == "real" and r["eligible"]); rows[kr]["normalized_error"] = None
        d = rec.design_of(rows); self.assertEqual(d["n_controls_missing"], 1); self.assertEqual(d["dropped_reasons"].get("missing:normalized_error"), 1)
        ref = ta.design({a: b for a, b in rows.items() if a != k}); self.assertTrue(np.array_equal(d["y"], ref["y"])); self.assertTrue(np.array_equal(d["B"], ref["B"]))


class Locked(unittest.TestCase):
    def test_the_file_is_read_into_arrays_with_its_hash_the_floors_the_baseline_and_the_null(self):
        main = {s: sd(make_rows(i)) for i, s in enumerate(STATES)}; lk = locked_from(main)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "locked.json"; p.write_text(json.dumps(lk)); got = rec.load_locked(p)
        self.assertEqual(len(got["sha256"]), 64); self.assertEqual(got["floors"], FLORES); st = got["states"]["INT4"]
        self.assertEqual(st["baseline"], lk["models"]["INT4"]["calibration_selected_baseline"]); self.assertAlmostEqual(st["null"], lk["models"]["INT4"]["null_calibration_median"]); self.assertEqual(set(st["models"]), set(ta.MODEL_GROUPS))
        self.assertIsInstance(st["models"]["A+B"]["coef"], np.ndarray); self.assertEqual(st["models"]["A+B"]["coef"].shape, (6,)); self.assertEqual(st["models"]["A"]["lambda"], lk["models"]["INT4"]["fits"]["A"]["lambda"])

    def test_model_groups_that_differ_from_the_frozen_ones_are_refused(self):
        main = {s: sd(make_rows(i)) for i, s in enumerate(STATES)}; lk = locked_from(main); lk["models"]["INT4"]["groups"]["A+B"] = ["A"]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "l.json"; p.write_text(json.dumps(lk))
            with self.assertRaisesRegex(ValueError, "groups"):
                rec.load_locked(p)

    def test_floors_that_differ_between_states_or_a_missing_state_are_refused(self):
        main = {s: sd(make_rows(i)) for i, s in enumerate(STATES)}; lk = locked_from(main); lk["models"]["INT6"]["floor_by_kind"]["real"] = 31.0
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "l.json"; p.write_text(json.dumps(lk))
            with self.assertRaisesRegex(ValueError, "floors"):
                rec.load_locked(p)


class Verification(unittest.TestCase):
    def setUp(self):
        self.rows = {s: make_rows(i) for i, s in enumerate(STATES)}; self.designs = {s: sd(r) for s, r in self.rows.items()}; self.lk = locked_from(self.designs)
        self.tmp = tempfile.TemporaryDirectory(); self.path = Path(self.tmp.name) / "l.json"

    def tearDown(self):
        self.tmp.cleanup()

    def load(self, lk=None):
        self.path.write_text(json.dumps(lk or self.lk)); return rec.load_locked(self.path)

    def test_consistent_inputs_pass_all_three_checks(self):
        r = rec.verify_locked(self.load(), self.rows, committed_mae(self.designs, self.lk)); self.assertTrue(r["ok"]); self.assertEqual(set(r["checks"]), {"refit_equals_locked", "committed_mae_reproduced", "renormalization_reproduces_main"})
        self.assertTrue(all(c["ok"] for c in r["checks"].values()))

    def test_a_tampered_coefficient_fails_the_refit_check_and_the_committed_mae_check(self):
        lk = json.loads(json.dumps(self.lk)); lk["models"]["INT4"]["fits"]["A+B"]["coef"][5] += 0.01; r = rec.verify_locked(self.load(lk), self.rows, committed_mae(self.designs, self.lk))
        self.assertFalse(r["ok"]); self.assertFalse(r["checks"]["refit_equals_locked"]["ok"]); self.assertFalse(r["checks"]["committed_mae_reproduced"]["ok"])

    def test_a_committed_mae_that_the_locked_models_do_not_give_fails(self):
        c = committed_mae(self.designs, self.lk); c[3]["heldout_mae"] += 1e-6; r = rec.verify_locked(self.load(), self.rows, c); self.assertFalse(r["ok"]); self.assertFalse(r["checks"]["committed_mae_reproduced"]["ok"]); self.assertTrue(r["checks"]["refit_equals_locked"]["ok"])

    def test_a_different_ridge_strength_alone_fails_the_refit_check(self):
        lk = json.loads(json.dumps(self.lk)); old = lk["models"]["INT4"]["fits"]["A"]["lambda"]; lk["models"]["INT4"]["fits"]["A"]["lambda"] = 0.1 if old != 0.1 else 10
        r = rec.verify_locked(self.load(lk), self.rows, committed_mae(self.designs, self.lk)); self.assertFalse(r["checks"]["refit_equals_locked"]["ok"]); self.assertFalse(r["checks"]["refit_equals_locked"]["detail"]["INT4/A"]["ok"])

    def test_a_committed_row_that_is_missing_fails_the_committed_check(self):
        c = [r for r in committed_mae(self.designs, self.lk) if not (r["state"] == "INT6" and r["model"] == "A+B")]; r = rec.verify_locked(self.load(), self.rows, c)
        self.assertFalse(r["checks"]["committed_mae_reproduced"]["ok"]); self.assertFalse(r["checks"]["committed_mae_reproduced"]["detail"]["INT6/A+B"]["ok"])

    def test_main_rows_whose_eligibility_differs_from_the_five_fold_floor_rule_fail_the_renormalization_check(self):
        rows = {s: {k: dict(v) for k, v in r.items()} for s, r in self.rows.items()}; k0 = next(k for k, v in rows["INT4"].items() if k[2] == "real" and v["eligible"]); rows["INT4"][k0]["eligible"] = False
        r = rec.verify_locked(self.load(), rows, committed_mae(self.designs, self.lk)); self.assertFalse(r["checks"]["renormalization_reproduces_main"]["ok"]); self.assertEqual(r["checks"]["renormalization_reproduces_main"]["detail"]["INT4"]["mismatches"], 1)

    def test_main_rows_whose_normalized_error_is_not_the_floored_one_fail_the_renormalization_check(self):
        rows = {s: {k: dict(v) for k, v in r.items()} for s, r in self.rows.items()}; k0 = next(iter(rows["INT6"])); rows["INT6"][k0]["normalized_error"] *= 1.5
        r = rec.verify_locked(self.load(), rows, committed_mae(self.designs, self.lk)); self.assertFalse(r["checks"]["renormalization_reproduces_main"]["ok"])


class CalibrationCV(unittest.TestCase):
    def test_the_out_of_fold_errors_of_the_locked_strengths_and_of_the_median_null_use_the_frozen_feature_folds(self):
        d = sd(make_rows(3)); lk = locked_from({"INT4": d}); lam = {n: f["lambda"] for n, f in lk["models"]["INT4"]["fits"].items()}; cv = rec.calibration_cv(d, lam)
        cal = d["split"] == "calibration"; fold = np.array([ta.feature_fold(f) for f in d["feature"][cal]]); y = d["y"][cal]; errs = {"A": [], "A+G": [], "null": []}
        for k in range(ta.N_FOLDS):
            te = fold == k
            if te.any():
                for n in ("A", "A+G"):
                    X = np.hstack([d[g][cal] for g in ta.MODEL_GROUPS[n]]); errs[n].append(np.abs(ta.ridge_predict(ta.ridge_fit(X[~te], y[~te], lam[n]), X[te]) - y[te]))
                errs["null"].append(np.abs(y[te] - np.median(y[~te])))
        for n in errs:
            self.assertAlmostEqual(cv[n], float(np.concatenate(errs[n]).mean()), places=12)
        self.assertEqual(set(cv), {"A", "A+G", "null"})

    def test_the_held_out_half_never_enters_and_the_best_comparator_is_the_lowest_error_with_ties_to_the_earlier_one(self):
        d = sd(make_rows(4)); lam = {n: 1 for n in ta.MODEL_GROUPS}; a = rec.calibration_cv(d, lam); e = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in d.items()}; h = e["split"] == "heldout"; e["y"][h] = 99.0; e["A"][h] = 1e6
        self.assertEqual(a, rec.calibration_cv(e, lam)); self.assertEqual(rec.choose_comparator({"A": 0.2, "A+G": 0.1, "null": 0.3}), "A+G"); self.assertEqual(rec.choose_comparator({"A": 0.1, "A+G": 0.1, "null": 0.1}), "A"); self.assertEqual(rec.choose_comparator({"A": 0.2, "A+G": 0.2, "null": 0.1}), "null")


class Scoring(unittest.TestCase):
    def setUp(self):
        self.main = {s: sd(make_rows(i)) for i, s in enumerate(STATES)}; self.lk = locked_from(self.main)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "l.json"; p.write_text(json.dumps(self.lk)); self.locked = rec.load_locked(p)
        self.ext = sd(make_rows(10)); self.cv = rec.calibration_cv(self.main["INT4"], {n: f["lambda"] for n, f in self.lk["models"]["INT4"]["fits"].items()})

    def test_the_held_out_items_are_scored_with_the_locked_coefficients_and_nothing_is_refit(self):
        r = rec.score(self.locked["states"]["INT4"], self.ext, self.cv, n_boot=50, seed=1); held = self.ext["split"] == "heldout"; self.assertEqual((r["status"], r["n_heldout"]), ("fitted", int(held.sum())))
        for name, groups in ta.MODEL_GROUPS.items():
            f = self.lk["models"]["INT4"]["fits"][name]; m = {k: np.array(f[k], dtype=float) for k in ("mean", "scale", "coef")}; m["intercept"] = f["intercept"]
            want = np.abs(ta.ridge_predict(m, np.hstack([self.ext[g][held] for g in groups])) - self.ext["y"][held]); self.assertTrue(np.allclose(r["models"][name]["abs_error"], want)); self.assertAlmostEqual(r["models"][name]["heldout_mae"], float(want.mean()))
        self.assertAlmostEqual(r["null_mae"], float(np.abs(self.ext["y"][held] - self.lk["models"]["INT4"]["null_calibration_median"]).mean()))

    def test_the_comparators_are_the_old_locked_baseline_and_the_new_one_chosen_on_the_main_calibration_errors_never_on_the_external_outcomes(self):
        r = rec.score(self.locked["states"]["INT4"], self.ext, self.cv, n_boot=50, seed=1); self.assertEqual(r["comparators"]["old"]["name"], self.lk["models"]["INT4"]["calibration_selected_baseline"]); self.assertEqual(r["comparators"]["new"]["name"], rec.choose_comparator(self.cv))
        e2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in self.ext.items()}; h = e2["split"] == "heldout"; e2["y"][h] = e2["y"][h] * 7 + 3; r2 = rec.score(self.locked["states"]["INT4"], e2, self.cv, n_boot=50, seed=1)
        self.assertEqual((r["comparators"]["old"]["name"], r["comparators"]["new"]["name"]), (r2["comparators"]["old"]["name"], r2["comparators"]["new"]["name"])); self.assertNotEqual(r["models"]["A+B"]["heldout_mae"], r2["models"]["A+B"]["heldout_mae"])

    def test_the_reduction_is_a_b_against_each_comparator_with_the_paired_interval_and_against_a(self):
        r = rec.score(self.locked["states"]["INT4"], self.ext, self.cv, n_boot=100, seed=1); held = self.ext["split"] == "heldout"; fh, ch = self.ext["feature"][held], self.ext["cluster"][held]
        for which in ("old", "new"):
            c = r["comparators"][which]; err = r["null_abs_error"] if c["name"] == "null" else r["models"][c["name"]]["abs_error"]; ref = ta.paired_relative_improvement(err, r["models"]["A+B"]["abs_error"], fh, ch, 100, 1); self.assertEqual(c["improvement"], ref)
        self.assertEqual(r["improvement_over_A"], ta.paired_relative_improvement(r["models"]["A"]["abs_error"], r["models"]["A+B"]["abs_error"], fh, ch, 100, 1)); self.assertGreater(r["improvement_over_A"]["low"], 0.0)

    def test_the_old_comparator_is_the_locked_baseline_even_when_the_calibration_errors_prefer_another(self):
        st = dict(self.locked["states"]["INT4"]); st["baseline"] = "null"; cv = {"A": 0.1, "A+G": 0.2, "null": 0.3}; r = rec.score(st, self.ext, cv, n_boot=30, seed=1)
        self.assertEqual((r["comparators"]["old"]["name"], r["comparators"]["new"]["name"]), ("null", "A")); self.assertNotEqual(r["comparators"]["old"]["improvement"], r["comparators"]["new"]["improvement"])
        self.assertNotEqual(r["comparators"]["old"]["heldout_mae"], r["comparators"]["new"]["heldout_mae"])

    def test_exactly_the_minimum_number_of_held_out_items_is_scored_one_fewer_is_not(self):
        d = self.ext; cal = np.where(d["split"] == "calibration")[0]; held = np.where(d["split"] == "heldout")[0]
        def subset(n):
            idx = np.concatenate([cal, held[:n]]); return {**{k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in d.items() if k not in ("keys",)}}
        self.assertEqual(rec.score(self.locked["states"]["INT4"], subset(10), self.cv, n_boot=10, seed=1)["status"], "fitted"); self.assertEqual(rec.score(self.locked["states"]["INT4"], subset(9), self.cv, n_boot=10, seed=1)["status"], "insufficient")

    def test_too_few_items_are_reported_not_scored_and_the_counts_say_why(self):
        small = sd({k: v for k, v in make_rows(10, n_held=2, per=2).items()}); r = rec.score(self.locked["states"]["INT4"], small, self.cv, n_boot=10, seed=1); self.assertEqual(r["status"], "insufficient"); self.assertNotIn("comparators", r)


class Run(unittest.TestCase):
    def test_the_driver_verifies_first_then_scores_both_protocols_per_state_and_writes_hashes_counts_and_the_dropped_items(self):
        ext_floors = {k: v * 0.6 for k, v in FLORES.items()}
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / "main").mkdir(); (root / "ext").mkdir(); main_rows = {s: make_rows(i) for i, s in enumerate(STATES)}; designs = {s: sd(r) for s, r in main_rows.items()}; lk = locked_from(designs)
            for s in STATES:
                rep.write_csv(root / "main" / f"per_context_{s}.csv", [r for _, r in sorted(main_rows[s].items())]); rep.write_csv(root / "ext" / f"per_context_{s}.csv", [r for _, r in sorted(make_rows(20 + STATES.index(s), own_floors=ext_floors).items())])
            (root / "locked.json").write_text(json.dumps(lk)); rep.write_csv(root / "committed.csv", committed_mae(designs, lk))
            out = rec.run(root / "locked.json", root / "main", root / "ext", root / "committed.csv", STATES, root / "out", n_boot=30, seed=1)
            self.assertEqual({(r["state"], r["protocol"]) for r in out}, {(s, p) for s in STATES for p in ("primary_flores_floors", "secondary_external_floors")})
            with (root / "out" / "cross_corpus_locked.csv").open() as fh:
                table = list(csv.DictReader(fh))
            self.assertEqual(len(table), 6); self.assertEqual(len({r["locked_sha256"] for r in table}), 1)
            for r in table:
                self.assertEqual(r["status"], "fitted"); self.assertIn("old_comparator", r); self.assertIn("new_comparator", r); self.assertIn("n_external_items", r); self.assertIn("n_heldout_eligible", r)
            js = json.loads((root / "out" / "cross_corpus_locked.json").read_text()); self.assertTrue(js["verification"]["ok"]); self.assertEqual(js["locked_sha256"], table[0]["locked_sha256"])
            prim = [r for r in table if r["protocol"] == "primary_flores_floors"][0]; sec = [r for r in table if r["protocol"] == "secondary_external_floors"][0]
            self.assertNotEqual(prim["n_heldout_eligible"], sec["n_heldout_eligible"])                      # the FLORES floors are larger than the 0.6 times floors of the external rows: fewer items are eligible

    def test_the_table_keeps_the_old_and_the_new_comparator_apart(self):
        ext_floors = {k: v * 0.6 for k, v in FLORES.items()}
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / "main").mkdir(); (root / "ext").mkdir(); main_rows = {s: make_rows(i) for i, s in enumerate(STATES)}; designs = {s: sd(r) for s, r in main_rows.items()}; lk = locked_from(designs); choice = {}
            for s in STATES:
                cv = rec.calibration_cv(designs[s], {n: f["lambda"] for n, f in lk["models"][s]["fits"].items()}); choice[s] = rec.choose_comparator(cv); lk["models"][s]["calibration_selected_baseline"] = "null" if choice[s] != "null" else "A"
                rep.write_csv(root / "main" / f"per_context_{s}.csv", [r for _, r in sorted(main_rows[s].items())]); rep.write_csv(root / "ext" / f"per_context_{s}.csv", [r for _, r in sorted(make_rows(30, own_floors=ext_floors).items())])
            (root / "locked.json").write_text(json.dumps(lk)); rep.write_csv(root / "committed.csv", committed_mae(designs, lk)); out = rec.run(root / "locked.json", root / "main", root / "ext", root / "committed.csv", STATES, root / "out", n_boot=30, seed=1)
            for r in out:
                self.assertEqual(r["new_comparator"], choice[r["state"]]); self.assertEqual(r["old_comparator"], lk["models"][r["state"]]["calibration_selected_baseline"]); self.assertNotEqual(r["improvement_new_point"], r["improvement_old_point"])
                self.assertNotEqual(r["new_comparator_mae"], r["old_comparator_mae"])

    def test_a_failed_verification_stops_the_run_before_any_external_item_is_scored(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / "main").mkdir(); (root / "ext").mkdir(); main_rows = {s: make_rows(i) for i, s in enumerate(STATES)}; designs = {s: sd(r) for s, r in main_rows.items()}; lk = locked_from(designs); lk["models"]["INT6"]["fits"]["A"]["intercept"] += 0.5
            for s in STATES:
                rep.write_csv(root / "main" / f"per_context_{s}.csv", [r for _, r in sorted(main_rows[s].items())]); rep.write_csv(root / "ext" / f"per_context_{s}.csv", [r for _, r in sorted(make_rows(20).items())])
            (root / "locked.json").write_text(json.dumps(lk)); rep.write_csv(root / "committed.csv", committed_mae(designs, locked_from(designs)))
            with self.assertRaisesRegex(RuntimeError, "verification"):
                rec.run(root / "locked.json", root / "main", root / "ext", root / "committed.csv", STATES, root / "out", n_boot=10, seed=1)
            self.assertFalse((root / "out" / "cross_corpus_locked.csv").exists())


if __name__ == "__main__":
    unittest.main()
