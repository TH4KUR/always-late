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
import phase1_transfer_report as rep
from test_phase1_transfer_analysis import prow, synthetic_design


def sd(**kw):
    """The synthetic design of the analysis tests with the `columns` entry a real design has."""
    d = synthetic_design(**kw); d["columns"] = {k: list(v) for k, v in ta.COLUMNS.items()}; return d


def copy_design(d):
    return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in d.items()}


class CrossCorpusDiagnostic(unittest.TestCase):
    def test_the_held_out_items_of_the_test_corpus_are_scored_by_models_fitted_on_the_calibration_items_of_the_training_corpus_with_no_refit(self):
        tr, te = sd(signal="B", seed=0), sd(signal="B", seed=11)
        r = cc.cross_corpus_diagnostic(tr, te, n_boot=50, seed=1, lams=(0.1, 1, 10))
        self.assertEqual((r["status"], r["n_train_calibration"], r["n_test_heldout"]), ("fitted", 96, 72)); self.assertEqual(set(r["models"]), {"A", "A+G", "A+B", "A+B+G"})
        cal, held = tr["split"] == "calibration", te["split"] == "heldout"
        for name, groups in ta.MODEL_GROUPS.items():
            Xtr = np.hstack([tr[g][cal] for g in groups]); Xte = np.hstack([te[g][held] for g in groups]); lam = ta.choose_lambda(Xtr, tr["y"][cal], tr["feature"][cal], (0.1, 1, 10))["lambda"]
            want = np.abs(ta.ridge_predict(ta.ridge_fit(Xtr, tr["y"][cal], lam), Xte) - te["y"][held])
            self.assertEqual(r["models"][name]["lambda"], lam); self.assertTrue(np.allclose(r["models"][name]["abs_error"], want)); self.assertAlmostEqual(r["models"][name]["heldout_mae"], float(want.mean()))

    def test_nothing_from_the_training_held_out_half_or_the_test_calibration_half_enters(self):
        tr, te = sd(signal="B", seed=2), sd(signal="B", seed=3); a = cc.cross_corpus_diagnostic(tr, te, n_boot=50, seed=1)
        tr2, te2 = copy_design(tr), copy_design(te); h1, c2 = tr2["split"] == "heldout", te2["split"] == "calibration"
        tr2["y"][h1] = tr2["y"][h1] * 50 + 1000; tr2["B"][h1] = -tr2["B"][h1] * 1e6; te2["y"][c2] = te2["y"][c2] * 50 + 1000; te2["A"][c2] = te2["A"][c2] * 1e6
        b = cc.cross_corpus_diagnostic(tr2, te2, n_boot=50, seed=1)
        self.assertEqual(a["strongest_baseline"], b["strongest_baseline"]); self.assertEqual(a["improvement"], b["improvement"]); self.assertEqual(a["null_mae"], b["null_mae"])
        for name in a["models"]:
            self.assertEqual(a["models"][name]["abs_error"].tolist(), b["models"][name]["abs_error"].tolist())

    def test_a_relation_shared_by_the_two_corpora_transfers_and_a_reversed_one_does_not(self):
        tr = sd(signal="B", seed=0); te = sd(signal="B", seed=4)
        same = cc.cross_corpus_diagnostic(tr, te, n_boot=200, seed=1); self.assertGreater(same["improvement"]["low"], 0.0); self.assertTrue(same["criterion_met"])
        flipped = copy_design(te); flipped["y"] = 6.0 - flipped["y"]; rev = cc.cross_corpus_diagnostic(tr, flipped, n_boot=200, seed=1)
        self.assertLess(rev["improvement"]["point"], 0.0); self.assertFalse(rev["criterion_met"])

    def test_the_strongest_baseline_is_chosen_on_the_training_calibration_cross_validation_not_on_the_test_corpus(self):
        tr = sd(signal="G", seed=5); te = sd(signal="B", seed=6)
        self.assertEqual(cc.cross_corpus_diagnostic(tr, te, n_boot=20, seed=1)["strongest_baseline"], "A+G")
        tr2 = sd(signal="B", seed=5); te2 = sd(signal="G", seed=6); self.assertEqual(cc.cross_corpus_diagnostic(tr2, te2, n_boot=20, seed=1)["strongest_baseline"], "A")

    def test_the_null_model_is_the_training_calibration_median_scored_on_the_test_held_out_items_and_h3_is_against_a(self):
        tr, te = sd(signal="G", seed=7), sd(signal="G", seed=8); r = cc.cross_corpus_diagnostic(tr, te, n_boot=20, seed=1); self.assertEqual(r["strongest_baseline"], "A+G")
        cal, held = tr["split"] == "calibration", te["split"] == "heldout"; self.assertAlmostEqual(r["null_mae"], float(np.abs(te["y"][held] - np.median(tr["y"][cal])).mean()))
        ref = ta.paired_relative_improvement(r["models"]["A"]["abs_error"], r["models"]["A+B"]["abs_error"], te["feature"][held], te["cluster"][held], 20, 1); self.assertEqual(r["improvement_over_A"], ref); self.assertNotEqual(r["improvement_over_A"], r["improvement"])

    def test_designs_with_different_columns_are_refused_and_too_few_items_are_reported_not_fitted(self):
        tr, te = sd(seed=0), sd(seed=1); te["columns"] = {"A": ["x"]}; tr["columns"] = {"A": ["x"], "P": ["p"]}
        with self.assertRaisesRegex(ValueError, "columns"):
            cc.cross_corpus_diagnostic(tr, te)
        r = cc.cross_corpus_diagnostic(sd(n_cal_features=3, per=2), sd(n_held_features=2, per=2)); self.assertEqual(r["status"], "insufficient"); self.assertNotIn("improvement", r)

    def test_exactly_the_minimum_numbers_of_items_are_fitted(self):
        r = cc.cross_corpus_diagnostic(sd(n_cal_features=10, n_held_features=1, per=2), sd(n_cal_features=1, n_held_features=5, per=2), n_boot=10, seed=0)
        self.assertEqual((r["n_train_calibration"], r["n_test_heldout"], r["status"]), (20, 10, "fitted"))
        r2 = cc.cross_corpus_diagnostic(sd(n_cal_features=9, n_held_features=1, per=2), sd(n_cal_features=1, n_held_features=5, per=2), n_boot=10, seed=0); self.assertEqual(r2["status"], "insufficient")
        r3 = cc.cross_corpus_diagnostic(sd(n_cal_features=10, n_held_features=1, per=2), sd(n_cal_features=1, n_held_features=4, per=2), n_boot=10, seed=0); self.assertEqual(r3["status"], "insufficient")

    def test_features_shared_between_the_training_calibration_half_and_the_test_held_out_half_are_refused(self):
        tr, te = sd(seed=0), sd(seed=1); te["feature"] = te["feature"].copy(); te["feature"][te["split"] == "heldout"] = 3
        with self.assertRaisesRegex(ValueError, "share"):
            cc.cross_corpus_diagnostic(tr, te)


class Rows(unittest.TestCase):
    def test_the_design_of_a_csv_round_trip_equals_the_design_of_the_rows(self):
        rows = {}
        for f in (10, 11, 12, 13):
            for s in (1000, 2000, 3000):
                rows[(f, s, "real")] = prow(f, s, "real", err=1.0 + f / 100 + s / 1e5, split="calibration" if f < 12 else "heldout", eligible=(s != 3000 or f == 10), g_shift_native=-0.3 + s / 1e4)
                for k, e in (("matched_swapped", 0.5), ("random1", 0.2), ("random2", 0.3), ("random3", 0.4)):
                    rows[(f, s, k)] = prow(f, s, k, err=e + f / 100, split="calibration" if f < 12 else "heldout")
        with tempfile.TemporaryDirectory() as d:
            rep.write_csv(Path(d) / "per_context_INT4.csv", [r for _, r in sorted(rows.items())]); back = cc.rows_from_csv(Path(d) / "per_context_INT4.csv")
        a, b = ta.design(rows), ta.design(back)
        for k in ("y", "A", "B", "G", "split", "feature", "cluster"):
            self.assertTrue(np.array_equal(a[k], b[k]), k)
        self.assertEqual(a["keys"], b["keys"]); self.assertEqual(ta.design(rows, eligible_only=False)["keys"], ta.design(back, eligible_only=False)["keys"])

    def test_an_item_that_occurs_twice_is_an_error(self):
        rows = [prow(1, 1, "real", err=1.0), prow(1, 1, "real", err=2.0)]
        with tempfile.TemporaryDirectory() as d:
            rep.write_csv(Path(d) / "p.csv", rows)
            with self.assertRaisesRegex(ValueError, "twice"):
                cc.rows_from_csv(Path(d) / "p.csv")

    def test_blank_cells_become_none_and_an_item_with_a_missing_value_is_dropped_by_the_design_not_filled(self):
        rows = {(1, 1, "real"): prow(1, 1, "real", err=1.0, g_norm_fp=None), (1, 1, "random1"): prow(1, 1, "random1", err=0.5)}
        with tempfile.TemporaryDirectory() as d:
            rep.write_csv(Path(d) / "p.csv", list(rows.values())); back = cc.rows_from_csv(Path(d) / "p.csv")
        self.assertIsNone([r for r in back.values() if r["kind"] == "real"][0]["g_norm_fp"]); dd = ta.design(back); self.assertEqual(len(dd["y"]), 0); self.assertEqual(dd["dropped_reasons"], {"missing:g_norm_fp": 1})


class Table(unittest.TestCase):
    def test_the_state_runner_writes_one_row_per_state_with_the_counts_the_models_and_the_interval_and_the_json_keeps_everything(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / "train").mkdir(); (root / "test").mkdir()
            for side, seed in (("train", 0), ("test", 1)):
                rng = np.random.default_rng(seed)
                for state in ("INT6", "INT4"):
                    rows = []
                    for f in range(40):
                        sp = "calibration" if f < 22 else "heldout"
                        for j in range(4):
                            sc = 1000 * f + j; b = float(rng.uniform(0.1, 1.0)); g = float(rng.uniform(0.05, 1.0))
                            rows.append(prow(f, sc, "real", err=float(0.3 + 2 * b + 0.3 * np.log(g + 1e-6) + rng.normal(0, 0.05)), split=sp, act_cos=float(rng.uniform(0, 1)), firing=float(rng.uniform(0, 1)), g_shift_native=g))
                            for k in ("matched_swapped", "random1", "random2", "random3"):
                                rows.append(prow(f, sc, k, err=b, split=sp))
                    rep.write_csv(root / side / f"per_context_{state}.csv", rows)
            out = cc.run(root / "train", root / "test", ("INT6", "INT4"), root / "out", n_boot=30, seed=1)
            self.assertEqual([r["state"] for r in out], ["INT6", "INT4"]); self.assertTrue(all(r["status"] == "fitted" for r in out))
            table = (root / "out" / "cross_corpus.csv").read_text().splitlines(); self.assertEqual(len(table), 3); self.assertIn("improvement_point", table[0]); self.assertIn("strongest_baseline", table[0])
            js = json.loads((root / "out" / "cross_corpus.json").read_text()); self.assertEqual(set(js["states"]), {"INT6", "INT4"}); self.assertIn("lambda", js["states"]["INT4"]["models"]["A+B"]); self.assertNotIn("abs_error", js["states"]["INT4"]["models"]["A+B"])
            self.assertGreater(js["states"]["INT4"]["improvement"]["point"], 0.1)
            dtr, dte = ta.design(cc.rows_from_csv(root / "train" / "per_context_INT4.csv")), ta.design(cc.rows_from_csv(root / "test" / "per_context_INT4.csv")); ref = cc.cross_corpus_diagnostic(dtr, dte, 30, 1); row = out[1]
            self.assertEqual(row["improvement_point"], ref["improvement"]["point"]); self.assertEqual(row["improvement_over_A_low"], ref["improvement_over_A"]["low"]); self.assertEqual(row["heldout_mae_A+B"], ref["models"]["A+B"]["heldout_mae"])
            self.assertEqual(row["strongest_baseline"], "A+G")
            self.assertEqual((row["n_train_items"], row["n_test_items"]), (len(dtr["y"]), len(dte["y"]))); self.assertEqual(row["strongest_baseline"], ref["strongest_baseline"]); self.assertEqual(row["null_mae"], ref["null_mae"])


if __name__ == "__main__":
    unittest.main()
