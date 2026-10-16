"""Tests for scripts/g0/r5_compare.py: the pre-committed reading rules of reports/audit_our_vs_duan.md section 5, on synthetic summaries."""
import csv, sys, tempfile, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import r5_compare as rc  # noqa: E402

COLS = ["condition", "survived_>0.9_pct", "median_corr", "perplexity", "n_active_features"]


def rows(base_ppl=458.0, surv=None, med=None, active=7100):
    surv = surv or {8: 99.0, 7: 81.0, 6: 51.0}; med = med or {8: 0.97, 7: 0.95, 6: 0.90}
    out = [{"condition": "FP16 baseline", "survived_>0.9_pct": 100.0, "median_corr": 1.0, "perplexity": base_ppl, "n_active_features": active}]
    for b in (8, 7, 6):
        out.append({"condition": f"RTN INT{b}", "survived_>0.9_pct": surv[b], "median_corr": med[b], "perplexity": base_ppl * 1.01, "n_active_features": active})
    return out


def write(path, rs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rs)


class Loading(unittest.TestCase):
    def test_load_summary_keys_by_condition_and_parses_floats(self):
        with tempfile.TemporaryDirectory() as t:
            write(Path(t) / "s.csv", rows())
            s = rc.load_summary(Path(t) / "s.csv")
        self.assertEqual(s["RTN INT7"]["survived_>0.9_pct"], 81.0); self.assertIn("FP16 baseline", s)

    def test_find_cells_uses_the_directory_name_and_skips_the_smoke_job(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            for d in ("R5a/T", "R5a/O", "R5b/T_keep", "R5smoke/T"):
                write(t / d / "phase3_summary_final.csv", rows())
            cells = rc.find_cells(t)
        self.assertEqual(sorted(cells), ["O", "T", "T_keep"])


class Rules(unittest.TestCase):
    def test_anchor_passes_inside_the_tolerances_and_fails_outside(self):
        base = {r["condition"]: r for r in rows()}
        near = {r["condition"]: r for r in rows(surv={8: 99.0, 7: 83.5, 6: 49.0}, med={8: 0.97, 7: 0.955, 6: 0.905}, base_ppl=460.0)}
        far = {r["condition"]: r for r in rows(surv={8: 99.0, 7: 90.0, 6: 51.0})}
        self.assertTrue(rc.anchor_check(near, base)["pass"])
        a = rc.anchor_check(far, base)
        self.assertFalse(a["pass"]); self.assertFalse(a["INT7_survival_within_3pp"]); self.assertTrue(a["INT6_survival_within_3pp"])

    def test_tolerances_have_exact_boundaries(self):
        base = {r["condition"]: r for r in rows()}
        at = {r["condition"]: r for r in rows(surv={8: 99.0, 7: 84.0, 6: 54.0}, med={8: 0.97, 7: 0.96, 6: 0.91})}       # exactly 3 pp, exactly 0.01
        over = {r["condition"]: r for r in rows(surv={8: 99.0, 7: 84.01, 6: 54.0}, med={8: 0.97, 7: 0.9601, 6: 0.91})}
        self.assertTrue(rc.anchor_check(at, base)["pass"])
        a = rc.anchor_check(over, base)
        self.assertFalse(a["INT7_survival_within_3pp"]); self.assertFalse(a["INT7_median_within_0.01"])
        ppl = {r["condition"]: r for r in rows(base_ppl=458.0 * 1.0101)}
        self.assertFalse(rc.anchor_check(ppl, base)["perplexity_within_1pct"])
        mk = lambda s7: {r["condition"]: r for r in rows(surv={8: 99.0, 7: s7, 6: 51.0})}
        self.assertTrue(rc.cross_stack(mk(78.0), mk(81.0))["INT7_within_3pp"]); self.assertFalse(rc.cross_stack(mk(77.99), mk(81.0))["INT7_within_3pp"])

    def test_materiality_thresholds_are_5_and_1_pp_at_either_rung(self):
        self.assertEqual(rc.classify(5.0, 0.0), "material"); self.assertEqual(rc.classify(0.0, -7.0), "material")
        self.assertEqual(rc.classify(-4.99, 0.2), "minor"); self.assertEqual(rc.classify(1.0, 0.0), "minor")
        self.assertEqual(rc.classify(0.99, -0.5), "none")

    def test_factor_deltas_are_relative_to_T_at_int7_and_int6(self):
        T = {r["condition"]: r for r in rows()}
        X = {r["condition"]: r for r in rows(surv={8: 99.0, 7: 70.0, 6: 50.5})}
        d = rc.factor_deltas({"T": T, "T_sdpa": X})["T_sdpa"]
        self.assertAlmostEqual(d["INT7"], -11.0); self.assertAlmostEqual(d["INT6"], -0.5); self.assertEqual(d["class"], "material")

    def test_our_pipeline_cells_are_compared_with_their_own_bos_regime(self):
        mk = lambda s7, s6: {r["condition"]: r for r in rows(surv={8: 99.0, 7: s7, 6: s6})}
        cells = {"T": mk(81, 51), "T_mask": mk(99.9, 97.5), "T_keep": mk(99.8, 98.0), "O_mask": mk(99.5, 97.0), "O_keep": mk(99.7, 98.4)}
        d = rc.factor_deltas(cells)
        self.assertEqual(d["O_mask"]["vs"], "T_mask"); self.assertAlmostEqual(d["O_mask"]["INT7"], -0.4); self.assertAlmostEqual(d["O_mask"]["INT6"], -0.5)
        self.assertEqual(d["O_keep"]["vs"], "T_keep"); self.assertEqual(d["O_mask"]["class"], "none")

    def test_additivity_compares_the_sum_of_single_factor_deltas_with_the_joint_cell(self):
        mk = lambda s7, s6: {r["condition"]: r for r in rows(surv={8: 99.0, 7: s7, 6: s6})}
        cells = {"T": mk(81, 51), "T_sdpa": mk(78, 51), "T_bf16sae": mk(80, 49), "T_rowquant": mk(90, 70), "O": mk(97, 94)}
        a = rc.additivity(cells)
        self.assertAlmostEqual(a["INT7"]["sum_single"], -3 - 1 + 9); self.assertAlmostEqual(a["INT7"]["joint"], 16.0)
        self.assertAlmostEqual(a["INT6"]["sum_single"], 0 - 2 + 19); self.assertAlmostEqual(a["INT6"]["joint"], 43.0)

    def test_loop_closure_and_cross_stack_are_differences_at_both_rungs(self):
        mk = lambda s7, s6: {r["condition"]: r for r in rows(surv={8: 99.0, 7: s7, 6: s6})}
        lc = rc.loop_closure(mk(97.0, 90.0), (98.28, 94.31))
        self.assertAlmostEqual(lc["INT7"], 97.0 - 98.28); self.assertAlmostEqual(lc["INT6"], 90.0 - 94.31)
        cs = rc.cross_stack(mk(70.0, 40.0), mk(72.0, 44.5))
        self.assertAlmostEqual(cs["INT7"], -2.0); self.assertTrue(cs["INT7_within_3pp"]); self.assertFalse(cs["INT6_within_3pp"])

    def test_missing_cells_are_reported_not_crashed_on(self):
        T = {r["condition"]: r for r in rows()}
        self.assertEqual(rc.factor_deltas({"T": T}), {})
        self.assertIsNone(rc.additivity({"T": T}))


class Rendering(unittest.TestCase):
    def test_report_mentions_every_cell_and_the_verdicts(self):
        T = {r["condition"]: r for r in rows()}
        md = rc.render({"T": T, "T_sdpa": {r["condition"]: r for r in rows(surv={8: 99.0, 7: 70.0, 6: 51.0})}}, baseline=T, published=T,
                       demo=(98.28, 94.31), r2=None)
        for needle in ("| T |", "| T_sdpa |", "material", "Anchor", "INT7"):
            self.assertIn(needle, md)


if __name__ == "__main__":
    unittest.main()
