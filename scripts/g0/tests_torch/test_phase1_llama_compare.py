import contextlib
import copy
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_func_analysis as fa
import phase1_gain_analysis as ga
import phase1_gain_run as gr
import phase1_llama_compare as lc
from test_phase1_gain_analysis import DOWN, FINE, UP, planted_run, ratios_json, save_state

LANGS = [f"l{i}" for i in range(27)]
STATES = ("FP", "INT8", "INT6", "INT5", "INT4")
FLAT = {"FP": 1.0, "INT8": 1.0, "INT6": 1.0, "INT5": 1.0, "INT4": 1.0}
THRESHOLD = {"FP": 1.0, "INT8": 1.01, "INT6": 1.04, "INT5": 1.13, "INT4": 2.1}               # the Gemma pattern


def run_of(state, f, e_of, bend=None, langs=LANGS):
    def harm(code, a, kind):
        i = int(code[1:]); e = e_of(i); depth = abs(np.log2(a)) if a > 0 else 6.0
        g = e * f * depth / 6.0
        if bend is not None and bend(state, i, a):
            g = 0.0
        return g if kind == "g" else 0.1 * g
    return planted_run(state, harm, langs=langs, n_controls=3)


def world(factors, e_of=lambda i: 0.2 + 0.01 * i, bend=None, langs=LANGS):
    return ga.GainRuns({s: run_of(s, f, e_of, bend, langs) for s, f in factors.items()}, DOWN, UP, FINE, n_controls=3)


def manifest(sizes):
    return {"languages": list(sizes), "groups": {l: {"all": list(range(n))} for l, n in sizes.items()}}


class Tested(unittest.TestCase):
    def test_a_language_is_tested_when_its_group_has_at_least_sixteen_members(self):
        m = manifest({"a": 16, "b": 15, "c": 200, "d": 0})
        self.assertEqual(lc.tested_languages(m), ["a", "c"])
        self.assertEqual(lc.tested_languages(m, min_group=100), ["c"])


class MedianCI(unittest.TestCase):
    def test_the_interval_is_the_seeded_percentile_interval_of_the_bootstrap_median(self):
        v = list(np.random.default_rng(1).normal(0.1, 0.1, 25)); lo, hi = lc.median_ci(v, 800, 0)
        idx = np.random.default_rng(0).integers(0, 25, size=(800, 25)); m = np.median(np.array(v)[idx], axis=1)
        self.assertAlmostEqual(lo, float(np.percentile(m, 2.5))); self.assertAlmostEqual(hi, float(np.percentile(m, 97.5)))
        self.assertNotEqual((lo, hi), lc.median_ci(v, 800, 1)); self.assertEqual((lo, hi), lc.median_ci(v, 800, 0))


class Predictions(unittest.TestCase):
    def evaluate(self, g, tested=None, **kw):
        return lc.evaluate(g, tested or LANGS, n_boot=30, n_boot_languages=400, seed=0, **kw)

    def test_a_graded_flat_then_rising_world_meets_every_prediction(self):
        r = self.evaluate(world(THRESHOLD))
        self.assertTrue(r["P1"]["met"]); self.assertTrue(r["P2a"]["met"]); self.assertTrue(r["P2b"]["met"]); self.assertTrue(r["P3a"]["met"]); self.assertTrue(r["P3b"]["met"]); self.assertEqual(r["verdict"], "consistent")
        self.assertAlmostEqual(r["P2"]["retention"]["INT4"], 2.1); self.assertAlmostEqual(r["P2"]["retention"]["INT6"], 1.04); self.assertEqual(r["n_tested"], 27)

    def test_p1_needs_24_of_27_positive_languages_and_an_interval_above_zero(self):
        for neg, expect in ((3, True), (4, False)):
            r = self.evaluate(world(FLAT, e_of=lambda i, k=neg: -0.1 if i < k else 0.2 + 0.01 * i))
            self.assertEqual(r["P1"]["count_positive"], 27 - neg); self.assertEqual(r["P1"]["met"], expect, msg=str(neg))
        r = self.evaluate(world(FLAT, e_of=lambda i: 0.2 if i % 2 == 0 else -0.2))                                  # 14 positive of 27: neither the count nor the interval
        self.assertFalse(r["P1"]["met"]); self.assertLessEqual(r["P1"]["lo"], 0.0)

    def test_p1_reports_the_median_and_its_interval(self):
        r = self.evaluate(world(FLAT)); e = [0.9 * (0.2 + 0.01 * i) for i in range(27)]
        self.assertAlmostEqual(r["P1"]["median"], float(np.median(e))); self.assertLess(r["P1"]["lo"], r["P1"]["median"]); self.assertGreater(r["P1"]["lo"], 0)

    def test_p2a_needs_int8_and_int6_retention_within_0_85_and_1_15(self):
        for f6, expect in ((1.15, True), (1.16, False), (0.85, True), (0.84, False)):
            self.assertEqual(self.evaluate(world({**FLAT, "INT6": f6}))["P2a"]["met"], expect, msg=str(f6))
        self.assertFalse(self.evaluate(world({**FLAT, "INT8": 1.3}))["P2a"]["met"])

    def test_p2b_needs_int4_at_least_0_30_above_int6_and_the_largest_of_the_ladder(self):
        self.assertTrue(self.evaluate(world({**FLAT, "INT4": 1.31}))["P2b"]["met"]); self.assertFalse(self.evaluate(world({**FLAT, "INT4": 1.29}))["P2b"]["met"])
        self.assertFalse(self.evaluate(world({**FLAT, "INT5": 2.0, "INT4": 1.5}))["P2b"]["met"])                    # INT5 higher than INT4: INT4 is not the largest
        self.assertFalse(self.evaluate(world(FLAT))["P2b"]["met"])

    def test_p2_uses_the_frozen_macro_retention(self):
        g = world(THRESHOLD); r = self.evaluate(g)
        D = fa.draws(g.n_sentences, 30, 0); eff = {s: {c: fa.functional_effect(g.runs[s], c, "K64", D) for c in LANGS} for s in g.states}
        for s in ("INT8", "INT6", "INT5", "INT4"):
            self.assertAlmostEqual(r["P2"]["retention"][s], fa.retention_macro(eff[s], eff["FP"], LANGS)["ratio"])

    def test_p3a_is_the_frozen_h_g1_at_fp_and_at_int4(self):
        r = self.evaluate(world(THRESHOLD)); self.assertTrue(r["P3a"]["FP"]["met"]); self.assertTrue(r["P3a"]["INT4"]["met"])
        bend = lambda s, i, a: s == "INT4" and i < 4 and a == DOWN[-1]                                                # four languages not ordered at INT4: 23 of 27 < 24
        r = self.evaluate(world(THRESHOLD, bend=bend)); self.assertTrue(r["P3a"]["FP"]["met"]); self.assertFalse(r["P3a"]["INT4"]["met"]); self.assertFalse(r["P3a"]["met"]); self.assertEqual(r["verdict"], "limited")
        bend3 = lambda s, i, a: s == "INT4" and i < 3 and a == DOWN[-1]
        self.assertTrue(self.evaluate(world(THRESHOLD, bend=bend3))["P3a"]["met"])                                    # 24 of 27 is enough

    def test_p3b_counts_languages_and_needs_a_median_ratio_above_1_3(self):
        r = self.evaluate(world({**FLAT, "INT4": 1.31})); self.assertTrue(r["P3b"]["met"]); self.assertEqual(r["P3b"]["count_larger"], 27); self.assertAlmostEqual(r["P3b"]["median_ratio"], 1.31)
        r = self.evaluate(world({**FLAT, "INT4": 1.29})); self.assertFalse(r["P3b"]["met"]); self.assertTrue(r["P3b"]["count_larger"] == 27)
        r = self.evaluate(world({**FLAT, "INT4": 0.9})); self.assertFalse(r["P3b"]["met"]); self.assertEqual(r["P3b"]["count_larger"], 0)

    def test_p3b_count_needs_20_of_27(self):
        def e_of(i):
            return 0.2 + 0.01 * i
        g = world({**FLAT, "INT4": 1.5})
        run = run_of("INT4", 1.5, e_of, bend=lambda s, i, a: i < 8 and a == DOWN[-1])                                 # eight languages with zero harm at the deepest down ratio at INT4: 19 larger
        g.runs["INT4"] = run
        r = self.evaluate(g); self.assertEqual(r["P3b"]["count_larger"], 19); self.assertFalse(r["P3b"]["met"])

    def test_untested_languages_do_not_enter_any_prediction_and_the_counts_are_proportional(self):
        tested = LANGS[:9]                                                                                           # nine tested languages: req(24) = 8
        g = world(THRESHOLD, e_of=lambda i: -0.5 if i >= 9 else 0.2 + 0.01 * i)                                      # the untested ones have negative E: they must not count
        r = self.evaluate(g, tested=tested); self.assertTrue(r["P1"]["met"]); self.assertEqual(r["n_tested"], 9); self.assertEqual(r["P1"]["required"], 8)
        g2 = world(FLAT, e_of=lambda i: -0.1 if i == 0 else 0.2 + 0.01 * i); r2 = self.evaluate(g2, tested=tested); self.assertEqual(r2["P1"]["count_positive"], 8); self.assertTrue(r2["P1"]["met"])
        g3 = world(FLAT, e_of=lambda i: -0.1 if i < 2 else 0.2 + 0.01 * i); r3 = self.evaluate(g3, tested=tested); self.assertEqual(r3["P1"]["count_positive"], 7); self.assertFalse(r3["P1"]["met"])

    def test_equal_net_harm_is_not_larger_and_languages_without_a_positive_fp_value_stay_out_of_the_ratio(self):
        r = self.evaluate(world(FLAT)); self.assertEqual(r["P3b"]["count_larger"], 0)                              # equal values at INT4 and FP: nobody is larger
        r = self.evaluate(world({**FLAT, "INT4": 1.5}, e_of=lambda i: 0.0 if i < 2 else 0.2 + 0.01 * i))          # two languages with no effect at all: their ratio is undefined and they stay out
        self.assertAlmostEqual(r["P3b"]["median_ratio"], 1.5); self.assertEqual(r["P3b"]["count_larger"], 25); self.assertTrue(r["P3b"]["met"])

    def test_p3b_count_threshold_is_20_of_27(self):
        g = world({**FLAT, "INT4": 1.5}); g.runs["INT4"] = run_of("INT4", 1.5, lambda i: 0.2 + 0.01 * i, bend=lambda s, i, a: i < 6 and a == DOWN[-1])      # six languages with no harm at INT4: 21 larger
        r = self.evaluate(g); self.assertEqual(r["P3b"]["count_larger"], 21); self.assertTrue(r["P3b"]["met"])

    def test_the_dose_rule_is_evaluated_on_the_tested_languages_only(self):
        tested = LANGS[:9]
        g = world(THRESHOLD, bend=lambda s, i, a: i >= 9 and a == DOWN[-1])                                          # every untested language is disordered at every state
        r = self.evaluate(g, tested=tested); self.assertTrue(r["P3a"]["met"]); self.assertEqual(r["P3a"]["FP"]["n"], 9)
        self.assertFalse(self.evaluate(g)["P3a"]["met"])                                                             # over all 27 languages the same runs fail the rule

    def test_diagnostics_report_the_group_and_control_harm_and_the_perturbation_norms_without_changing_the_verdict(self):
        skew = lambda i: 0.2 + 0.001 * i ** 2                                                                       # mean and median differ
        g = world(THRESHOLD, e_of=skew)
        for s in ("FP", "INT4"):
            for c in LANGS:
                g.runs[s].stats[f"{c}|zero_K64@{c}"] = {"sum_norm_ratio": 30.0, "rows": 100}
                for r in range(3):
                    g.runs[s].stats[f"{c}|ctrl{r}_K64@{c}"] = {"sum_norm_ratio": 10.0 * (r + 1), "rows": 100}
        r = self.evaluate(g); d = r["diagnostics"]
        e = [skew(i) for i in range(27)]; self.assertNotAlmostEqual(float(np.mean(e)), float(np.median(e)), places=3)
        self.assertAlmostEqual(d["FP"]["harm_group_median"], float(np.median(e))); self.assertAlmostEqual(d["FP"]["harm_control_median"], 0.1 * float(np.median(e)))
        self.assertAlmostEqual(d["INT4"]["harm_group_median"], 2.1 * float(np.median(e)))
        self.assertAlmostEqual(d["FP"]["norm_ratio_group_median"], 0.3); self.assertAlmostEqual(d["FP"]["norm_ratio_control_median"], 0.2)             # (10 + 20 + 30) / 300
        self.assertEqual(r["verdict"], "consistent")
        none = self.evaluate(world(FLAT))["diagnostics"]["FP"]; self.assertIsNone(none["norm_ratio_group_median"]); self.assertIsNone(none["norm_ratio_control_median"])

    def test_the_input_languages_must_be_a_subset_of_the_runs_and_not_empty(self):
        g = world(FLAT)
        with self.assertRaises(ValueError):
            lc.evaluate(g, [], n_boot=10, n_boot_languages=10)
        with self.assertRaises(ValueError):
            lc.evaluate(g, ["nope"], n_boot=10, n_boot_languages=10)

    def test_the_fp_run_and_int4_are_needed_for_the_verdict(self):
        with self.assertRaises(ValueError):
            lc.evaluate(world({"FP": 1.0, "INT8": 1.0, "INT6": 1.0, "INT5": 1.0}), LANGS, n_boot=10, n_boot_languages=10)


class Verdict(unittest.TestCase):
    def test_the_four_labels(self):
        t, f = {"met": True}, {"met": False}
        self.assertEqual(lc.verdict({"P1": t, "P2b": t, "P3a": t, "P3b": f, "P2a": f}), "consistent")
        self.assertEqual(lc.verdict({"P1": t, "P2b": f, "P3a": t, "P3b": t, "P2a": t}), "partly consistent")
        self.assertEqual(lc.verdict({"P1": t, "P2b": t, "P3a": f, "P3b": t, "P2a": t}), "limited")
        self.assertEqual(lc.verdict({"P1": t, "P2b": f, "P3a": f, "P3b": f, "P2a": f}), "limited")
        self.assertEqual(lc.verdict({"P1": f, "P2b": t, "P3a": t, "P3b": t, "P2a": t}), "not consistent")

    def test_p2a_and_p3b_never_change_the_verdict(self):
        base = {"P1": {"met": True}, "P2b": {"met": True}, "P3a": {"met": True}}
        for a in (True, False):
            for b in (True, False):
                self.assertEqual(lc.verdict({**base, "P2a": {"met": a}, "P3b": {"met": b}}), "consistent")


def write_gemma_reference(root):
    root = Path(root); (root / "results/eval").mkdir(parents=True); (root / "gain_study/results").mkdir(parents=True)
    with open(root / "results/eval/effects_by_language.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["precision", "language", "effect", "lo", "hi", "harm_group", "harm_control", "norm_ratio_group", "norm_ratio_control"])
        for i, e in enumerate((0.1, 0.3, 0.5)):
            w.writerow(["FP", f"g{i}", e, 0, 0, 0, 0, 0, 0])
        w.writerow(["INT4", "g0", 9, 0, 0, 0, 0, 0, 0])
    with open(root / "results/eval/retention.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["precision", "language", "ratio", "lo", "hi", "class"])
        for s, v in (("INT8", 1.01), ("INT6", 1.04), ("INT5", 1.13), ("INT4", 2.14)):
            w.writerow([s, "MACRO_MEAN_OF_RATIOS", v, 0, 0, "x"]); w.writerow([s, "g0", 99, 0, 0, "x"])
    with open(root / "gain_study/results/curves.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["state", "language", "ratio", "group_harm", "group_lo", "group_hi", "control_harm", "net", "net_lo", "net_hi", "total_change"])
        for s, f in (("FP", 1.0), ("INT4", 2.0)):
            for i in range(3):
                for a, v in ((0.08838834764831845, 0.1 * (i + 1) * f), (0.3535533905932738, 0.01 * f)):
                    w.writerow([s, f"g{i}", a, 0, 0, 0, 0, v, 0, 0, 0])
    return root


class GemmaReference(unittest.TestCase):
    def test_the_gemma_numbers_are_read_from_the_committed_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            ref = lc.gemma_reference(write_gemma_reference(d))
        self.assertAlmostEqual(ref["median_E_FP"], 0.3); self.assertEqual(ref["retention"], {"INT8": 1.01, "INT6": 1.04, "INT5": 1.13, "INT4": 2.14})
        self.assertAlmostEqual(ref["deepest_net_median"]["FP"], 0.2); self.assertAlmostEqual(ref["deepest_net_median"]["INT4"], 0.4); self.assertAlmostEqual(ref["ratio_int4_over_fp_median"], 2.0)

    def test_a_missing_file_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                lc.gemma_reference(Path(d))


class Report(unittest.TestCase):
    def test_the_report_has_the_verdict_the_predictions_the_setting_and_both_models(self):
        r = lc.evaluate(world(THRESHOLD), LANGS, n_boot=30, n_boot_languages=200, seed=0)
        with tempfile.TemporaryDirectory() as d:
            ref = lc.gemma_reference(write_gemma_reference(d))
        text = lc.report_text(r, ref, {"sae": "fineweb42"})
        for k in ("consistent", "P1", "P2a", "P2b", "P3a", "P3b", "Gemma", "Llama", "depth 12 of 16", "TopK", "Not tested", "27 languages", "fineweb42", "Diagnostics", "perturbation norm"):
            self.assertIn(k, text)

    def test_the_verdict_line_follows_the_evaluation(self):
        r = lc.evaluate(world(FLAT, e_of=lambda i: -0.1), LANGS, n_boot=30, n_boot_languages=200, seed=0)
        with tempfile.TemporaryDirectory() as d:
            ref = lc.gemma_reference(write_gemma_reference(d))
        self.assertIn("**not consistent**", lc.report_text(r, ref, {}))


class Files(unittest.TestCase):
    def test_main_runs_on_job_directories_and_writes_the_outputs(self):
        preset = gr.preset_name(DOWN + UP + FINE, 3, "K64")
        g = world(THRESHOLD)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for s in STATES:
                save_state(d / f"job-{s}", g.runs[s], preset)
            (d / "man.json").write_text(json.dumps(manifest({l: 64 if i < 25 else 5 for i, l in enumerate(LANGS)})))
            ratios = ratios_json(d); ref = write_gemma_reference(d / "gemma")
            with contextlib.redirect_stdout(io.StringIO()) as out:
                lc.main(["--runs", *[str(d / f"job-{s}") for s in STATES], "--manifest", str(d / "man.json"), "--ratios-json", str(ratios), "--gemma-root", str(ref), "--out", str(d / "out"),
                         "--n-boot", "30", "--n-boot-languages", "200", "--sae", "fineweb42"])
            for f in ("report.md", "analysis.json"):
                self.assertTrue((d / "out" / f).exists(), f)
            js = json.loads((d / "out" / "analysis.json").read_text())
            self.assertEqual(js["n_tested"], 25); self.assertEqual(js["verdict"], "consistent"); self.assertEqual(js["untested"], ["l25", "l26"]); self.assertIn("consistent", out.getvalue())

    def test_a_manifest_for_other_languages_is_refused(self):
        preset = gr.preset_name(DOWN + UP + FINE, 3, "K64"); g = world(THRESHOLD)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for s in STATES:
                save_state(d / f"job-{s}", g.runs[s], preset)
            (d / "man.json").write_text(json.dumps(manifest({f"x{i}": 64 for i in range(27)}))); ratios = ratios_json(d); ref = write_gemma_reference(d / "gemma")
            with self.assertRaisesRegex(ValueError, "different languages"):
                lc.main(["--runs", *[str(d / f"job-{s}") for s in STATES], "--manifest", str(d / "man.json"), "--ratios-json", str(ratios), "--gemma-root", str(ref), "--out", str(d / "out")])


if __name__ == "__main__":
    unittest.main()
