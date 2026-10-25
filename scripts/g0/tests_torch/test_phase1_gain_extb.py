import contextlib
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
import phase1_gain_analysis as ga
import phase1_gain_extb as eb
from test_phase1_gain_analysis import DOWN, FINE, LANGS, N, PRESET, UP, planted_run, ratios_json, save_state, simple_harm

STATES = ("FP", "INT6", "INT4")


def scaled(factor, base=simple_harm):
    return lambda c, a, k: factor * base(c, a, k)


def gains_of(factors, noise=None, langs=LANGS):
    """GainRuns over FP, INT6, INT4 whose group harm is the simple harm times a factor per state (controls unchanged)."""
    def h(f):
        return lambda c, a, k: (f * simple_harm(c, a, k) if k == "g" else simple_harm(c, a, k))
    rs = {s: planted_run(s, h(f), langs=langs, noise_group=noise, n_controls=3) for s, f in zip(STATES, factors)}
    return ga.GainRuns(rs, DOWN, UP, FINE, n_controls=3)


def tables(g_sd, g_ref, n_boot=100):
    return ga.build_table(g_sd, n_boot=n_boot, seed=0), ga.build_table(g_ref, n_boot=n_boot, seed=0)


def overlap_rows(j6, j4, n=27):
    rows = []
    for i in range(n):
        rows.append({"language": f"l{i}", "state": "INT6", "ref": "FP", "size": 64, "shared": 59, "jaccard": j6(i)})
        rows.append({"language": f"l{i}", "state": "INT4", "ref": "FP", "size": 64, "shared": 42, "jaccard": j4(i)})
    return rows


class HGb1(unittest.TestCase):
    def test_met_when_int4_overlaps_less_than_int6_in_24_of_27_languages(self):
        for k, expected in ((24, True), (23, False)):
            r = eb.h_gb1(overlap_rows(lambda i: 0.85, lambda i: 0.49 if i < k else 0.90))
            self.assertEqual(r["count"], k); self.assertEqual(r["met"], expected, msg=f"{k}")

    def test_equal_overlap_is_not_less(self):
        r = eb.h_gb1(overlap_rows(lambda i: 0.85, lambda i: 0.85))
        self.assertEqual(r["count"], 0)

    def test_medians_are_reported_and_the_proportional_threshold_applies_to_other_sizes(self):
        r = eb.h_gb1(overlap_rows(lambda i: 0.8, lambda i: 0.5, n=4), )
        self.assertAlmostEqual(r["median_int6"], 0.8); self.assertAlmostEqual(r["median_int4"], 0.5); self.assertTrue(r["met"])

    def test_both_states_must_be_in_the_table(self):
        with self.assertRaises(ValueError):
            eb.h_gb1([x for x in overlap_rows(lambda i: 0.8, lambda i: 0.5) if x["state"] == "INT4"])


class HGb2(unittest.TestCase):
    def test_a_state_defined_group_with_more_net_harm_is_met(self):
        g_sd = gains_of((1.0, 1.0, 1.5)); g_ref = gains_of((1.0, 1.0, 1.0))
        t_sd, t_ref = tables(g_sd, g_ref)
        r = eb.h_gb2(t_sd, t_ref, g_sd, g_ref, state="INT4")
        self.assertEqual(r["count_positive"], 4); self.assertGreater(r["median_diff"], 0); self.assertTrue(r["met"])
        self.assertEqual(r["ratio"], g_sd.down[-1])

    def test_fewer_than_two_thirds_positive_languages_is_not_met(self):
        def sd(c, a, k):
            return (0.5 if LANGS.index(c) < 2 else 1.5) * simple_harm(c, a, "g") if k == "g" else simple_harm(c, a, k)
        rs = {s: planted_run(s, simple_harm, n_controls=3) for s in ("FP", "INT6")}; rs["INT4"] = planted_run("INT4", sd, n_controls=3)
        g_sd = ga.GainRuns(rs, DOWN, UP, FINE, n_controls=3); g_ref = gains_of((1.0, 1.0, 1.0))
        t_sd, t_ref = tables(g_sd, g_ref)
        r = eb.h_gb2(t_sd, t_ref, g_sd, g_ref, state="INT4")
        self.assertEqual(r["count_positive"], 2); self.assertFalse(r["met"])                    # 2 of 4 < ceil(18 * 4 / 27) = 3

    def test_a_negative_median_is_not_met(self):
        g_sd = gains_of((1.0, 1.0, 0.5)); g_ref = gains_of((1.0, 1.0, 1.0))
        t_sd, t_ref = tables(g_sd, g_ref)
        r = eb.h_gb2(t_sd, t_ref, g_sd, g_ref, state="INT4")
        self.assertLess(r["median_diff"], 0); self.assertFalse(r["met"])

    def test_the_difference_is_per_language_and_paired(self):
        g_sd = gains_of((1.0, 1.0, 1.5)); g_ref = gains_of((1.0, 1.0, 1.0))
        t_sd, t_ref = tables(g_sd, g_ref)
        r = eb.h_gb2(t_sd, t_ref, g_sd, g_ref, state="INT4")
        d = g_sd.down[-1]
        for l in LANGS:
            self.assertAlmostEqual(r["per_language"][l]["diff"], t_sd.get("INT4", l, d)["net"] - t_ref.get("INT4", l, d)["net"])
            self.assertLessEqual(r["per_language"][l]["lo"], r["per_language"][l]["diff"] + 1e-12)

    def test_the_interval_comes_from_paired_sentence_draws(self):
        e = np.random.default_rng(3).normal(0, 0.05, N)
        g_sd = gains_of((1.0, 1.0, 1.0), noise=e); g_ref = gains_of((1.0, 1.0, 1.0), noise=e)          # identical noise in both runs: the paired difference has no spread
        t_sd, t_ref = tables(g_sd, g_ref, n_boot=200)
        r = eb.h_gb2(t_sd, t_ref, g_sd, g_ref, state="INT4")
        for l in LANGS:
            self.assertAlmostEqual(r["per_language"][l]["lo"], r["per_language"][l]["hi"], places=9)

    def test_runs_that_differ_in_languages_ratios_or_sentences_are_refused(self):
        g_sd = gains_of((1.0, 1.0, 1.0)); g_ref = gains_of((1.0, 1.0, 1.0), langs=LANGS[:3])
        t_sd, t_ref = tables(g_sd, g_ref)
        with self.assertRaises(ValueError):
            eb.h_gb2(t_sd, t_ref, g_sd, g_ref, state="INT4")
        g_other = ga.GainRuns({s: planted_run(s, simple_harm, n_controls=3) for s in STATES}, DOWN, UP, FINE, n_controls=3)
        g_other.down = [0.5, 0.25, 0.2]
        t1, t2 = tables(g_sd, g_sd)
        with self.assertRaises(ValueError):
            eb.h_gb2(t1, t2, g_sd, g_other, state="INT4")


class Evaluate(unittest.TestCase):
    def setUp(self):
        self.g_sd = gains_of((1.0, 1.0, 1.5)); self.g_ref = gains_of((1.0, 1.0, 1.0))
        self.t_sd, self.t_ref = tables(self.g_sd, self.g_ref)
        self.ov = overlap_rows(lambda i: 0.85, lambda i: 0.49, n=4)
        self.res = eb.evaluate(self.t_sd, self.t_ref, self.g_sd, self.g_ref, self.ov, n_boot=100)

    def test_verdicts_cover_all_four_rules_with_the_two_parts_of_the_repair_rule(self):
        self.assertEqual(set(self.res["verdicts"]), {"H-Gb1", "H-Gb2", "H-Gb3", "H-Gb4a", "H-Gb4b"})
        self.assertTrue(self.res["verdicts"]["H-Gb1"]); self.assertTrue(self.res["verdicts"]["H-Gb2"])

    def test_h_gb2_is_reported_at_int4_for_the_verdict_and_at_int6_descriptively(self):
        self.assertEqual(self.res["rules"]["H-Gb2"]["INT4"]["state"], "INT4"); self.assertIn("INT6", self.res["rules"]["H-Gb2"])
        self.assertEqual(self.res["verdicts"]["H-Gb2"], self.res["rules"]["H-Gb2"]["INT4"]["met"])

    def test_h_gb3_needs_the_main_rules_at_every_state_of_the_state_defined_runs(self):
        r = self.res["rules"]["H-Gb3"]
        self.assertEqual(set(r), {"FP", "INT6", "INT4"})
        for s in r.values():
            self.assertEqual(set(s), {"h_g1", "h_g3"})
        self.assertEqual(self.res["verdicts"]["H-Gb3"], all(s["h_g1"] and s["h_g3"] for s in r.values()))

    def test_h_gb4_is_the_repair_rule_of_the_main_study_on_the_state_defined_groups(self):
        direct = ga.h_g5(self.t_sd, self.g_sd, n_boot=100)
        self.assertEqual(self.res["verdicts"]["H-Gb4a"], direct["a"]["met"]); self.assertEqual(self.res["verdicts"]["H-Gb4b"], direct["b"]["met"])

    def test_the_comparison_table_has_a_row_per_state_and_ratio(self):
        rows = eb.compare_rows(self.t_sd, self.t_ref, self.g_sd)
        self.assertEqual(len(rows), 3 * (len(DOWN + UP + FINE) + 1))
        r = [x for x in rows if x["state"] == "INT4" and x["ratio"] == self.g_sd.down[-1]][0]
        self.assertAlmostEqual(r["median_net_state_defined"], float(np.median([self.t_sd.get("INT4", l, r["ratio"])["net"] for l in LANGS])))
        self.assertAlmostEqual(r["median_net_reference"], float(np.median([self.t_ref.get("INT4", l, r["ratio"])["net"] for l in LANGS])))

    def test_the_report_names_each_rule_and_the_two_selection_texts(self):
        text = eb.report_text(self.res, self.t_sd, self.t_ref, self.g_sd, self.g_ref, {"runs": {}})
        for k in ("H-Gb1", "H-Gb2", "H-Gb3", "H-Gb4"):
            self.assertIn(k, text)
        self.assertIn("reserve", text); self.assertIn("FP reference", text)

    def test_the_verdict_text_follows_the_rule(self):
        g_sd = gains_of((1.0, 1.0, 0.5))
        t_sd, t_ref = tables(g_sd, self.g_ref)
        res = eb.evaluate(t_sd, t_ref, g_sd, self.g_ref, self.ov, n_boot=100)
        text = eb.report_text(res, t_sd, t_ref, g_sd, self.g_ref, {"runs": {}})
        line = [l for l in text.split("\n") if l.startswith("| H-Gb2")][0]
        self.assertTrue(line.rstrip().endswith("**not met** |"))


class Files(unittest.TestCase):
    def test_write_outputs_and_main_on_files(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for name, factors in (("sd", (1.0, 1.0, 1.5)), ("ref", (1.0, 1.0, 1.0))):
                for s, f in zip(STATES, factors):
                    run = planted_run(s, (lambda f: (lambda c, a, k: f * simple_harm(c, a, k) if k == "g" else simple_harm(c, a, k)))(f), n_controls=3)
                    save_state(d / f"{name}-{s}", run, PRESET.replace("c2", "c3"))
            ov = overlap_rows(lambda i: 0.85, lambda i: 0.49, n=4)
            with open(d / "overlap.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(ov[0])); w.writeheader(); w.writerows(ov)
            rj = ratios_json(d)
            with contextlib.redirect_stdout(io.StringIO()):
                eb.main(["--state-defined", str(d / "sd-FP"), str(d / "sd-INT6"), str(d / "sd-INT4"), "--reference", str(d / "ref-FP"), str(d / "ref-INT6"), str(d / "ref-INT4"),
                         "--ratios-json", str(rj), "--overlap", str(d / "overlap.csv"), "--out", str(d / "out"), "--n-boot", "100"])
            js = json.loads((d / "out" / "analysis.json").read_text())
            self.assertTrue(js["verdicts"]["H-Gb1"]); self.assertTrue(js["verdicts"]["H-Gb2"])
            for f in ("analysis.json", "compare.csv", "report.md"):
                self.assertTrue((d / "out" / f).exists(), f)
            with open(d / "out" / "compare.csv") as fh:
                self.assertEqual(len(list(csv.DictReader(fh))), 3 * (len(DOWN + UP + FINE) + 1))


class Survivors(unittest.TestCase):
    """Added after the first mutation round of phase1_gain_extb."""
    def test_the_interval_surrounds_the_difference_and_has_the_paired_percentiles(self):
        e = np.random.default_rng(3).normal(0, 0.05, N)
        g_sd = gains_of((1.0, 1.0, 1.5), noise=e); g_ref = gains_of((1.0, 1.0, 1.0))
        t_sd, t_ref = tables(g_sd, g_ref, n_boot=200)
        r = eb.h_gb2(t_sd, t_ref, g_sd, g_ref, state="INT4")
        d = g_sd.down[-1]
        for l in LANGS:
            x = r["per_language"][l]
            self.assertLess(x["lo"], x["diff"]); self.assertGreater(x["hi"], x["diff"])
            s = t_sd.get("INT4", l, d)["net_s"] - t_ref.get("INT4", l, d)["net_s"]
            self.assertAlmostEqual(x["lo"], float(np.percentile(s, 2.5))); self.assertAlmostEqual(x["hi"], float(np.percentile(s, 97.5)))

    def test_languages_with_no_difference_are_neither_positive_nor_above_zero(self):
        g_sd = gains_of((1.0, 1.0, 1.0)); g_ref = gains_of((1.0, 1.0, 1.0))
        t_sd, t_ref = tables(g_sd, g_ref)
        r = eb.h_gb2(t_sd, t_ref, g_sd, g_ref, state="INT4")
        self.assertEqual(r["count_positive"], 0); self.assertEqual(r["count_lower_above_zero"], 0); self.assertFalse(r["met"])

    def test_the_boundary_is_18_of_27_languages(self):
        from test_phase1_gain_analysis import LANGS27, harm27
        def run_set(k_positive):
            def sd(c, a, kind):
                f = 1.5 if int(c[1:]) < k_positive else 0.7
                return f * harm27(c, a, "g") if kind == "g" else harm27(c, a, kind)
            rs = {s: planted_run(s, harm27, langs=LANGS27, n_controls=3) for s in ("FP", "INT6")}; rs["INT4"] = planted_run("INT4", sd, langs=LANGS27, n_controls=3)
            g_sd = ga.GainRuns(rs, DOWN, UP, FINE, n_controls=3)
            g_ref = ga.GainRuns({s: planted_run(s, harm27, langs=LANGS27, n_controls=3) for s in STATES}, DOWN, UP, FINE, n_controls=3)
            return eb.h_gb2(*tables(g_sd, g_ref, n_boot=30), g_sd, g_ref, state="INT4")
        self.assertTrue(run_set(18)["met"]); self.assertEqual(run_set(18)["count_positive"], 18)
        self.assertFalse(run_set(17)["met"]); self.assertEqual(run_set(17)["count_positive"], 17)

    def test_the_comparison_table_counts_the_languages_where_the_state_defined_group_is_larger(self):
        g_sd = gains_of((1.0, 1.0, 1.5)); g_ref = gains_of((1.0, 1.0, 1.0))
        t_sd, t_ref = tables(g_sd, g_ref)
        rows = eb.compare_rows(t_sd, t_ref, g_sd)
        r4 = [x for x in rows if x["state"] == "INT4" and x["ratio"] == g_sd.down[-1]][0]; r6 = [x for x in rows if x["state"] == "INT6" and x["ratio"] == g_sd.down[-1]][0]
        self.assertEqual(r4["n_state_defined_larger"], 4); self.assertEqual(r6["n_state_defined_larger"], 0)

    def test_h_gb3_needs_the_dose_rules_at_every_state(self):
        def bad(c, a, k):                                         # INT6: the group harm at the second down ratio exceeds the one at the deepest, for one language
            if k == "g" and c == "l3" and a == g_sd_down[-1]:
                return 0.0
            return simple_harm(c, a, k)
        g_sd_down = DOWN
        rs = {"FP": planted_run("FP", simple_harm, n_controls=3), "INT6": planted_run("INT6", bad, n_controls=3), "INT4": planted_run("INT4", simple_harm, n_controls=3)}
        g_sd = ga.GainRuns(rs, DOWN, UP, FINE, n_controls=3); g_ref = gains_of((1.0, 1.0, 1.0))
        t_sd, t_ref = tables(g_sd, g_ref)
        res = eb.evaluate(t_sd, t_ref, g_sd, g_ref, overlap_rows(lambda i: 0.85, lambda i: 0.49, n=4), n_boot=50)
        self.assertFalse(res["rules"]["H-Gb3"]["INT6"]["h_g1"]); self.assertTrue(res["rules"]["H-Gb3"]["INT6"]["h_g3"]); self.assertTrue(res["rules"]["H-Gb3"]["INT4"]["h_g1"])
        self.assertFalse(res["verdicts"]["H-Gb3"])

    def test_h_gb4_uses_the_state_defined_runs_not_the_reference(self):
        def repair(c, a, k):
            return (-0.03 if k == "g" else -0.01) if a in (1.1, 1.2) else simple_harm(c, a, k)
        rs = {"FP": planted_run("FP", simple_harm, n_controls=3), "INT6": planted_run("INT6", simple_harm, n_controls=3), "INT4": planted_run("INT4", repair, n_controls=3)}
        g_sd = ga.GainRuns(rs, DOWN, UP, FINE, n_controls=3); g_ref = gains_of((1.0, 1.0, 1.0))
        t_sd, t_ref = tables(g_sd, g_ref)
        res = eb.evaluate(t_sd, t_ref, g_sd, g_ref, overlap_rows(lambda i: 0.85, lambda i: 0.49, n=4), n_boot=100)
        self.assertTrue(res["verdicts"]["H-Gb4b"]); self.assertTrue(res["verdicts"]["H-Gb4a"])
        res_ref = eb.evaluate(t_ref, t_sd, g_ref, g_sd, overlap_rows(lambda i: 0.85, lambda i: 0.49, n=4), n_boot=100)       # roles swapped: the plain runs are now the state-defined ones
        self.assertFalse(res_ref["verdicts"]["H-Gb4b"])

    def test_the_verdict_of_h_gb2_is_the_int4_one_not_the_int6_one(self):
        g_sd = gains_of((1.0, 0.5, 1.5)); g_ref = gains_of((1.0, 1.0, 1.0))                  # INT6: state-defined smaller; INT4: larger
        t_sd, t_ref = tables(g_sd, g_ref)
        res = eb.evaluate(t_sd, t_ref, g_sd, g_ref, overlap_rows(lambda i: 0.85, lambda i: 0.49, n=4), n_boot=50)
        self.assertTrue(res["verdicts"]["H-Gb2"]); self.assertFalse(res["rules"]["H-Gb2"]["INT6"]["met"])

    def test_main_uses_the_same_sentence_draws_for_both_sets(self):
        e = np.random.default_rng(3).normal(0, 0.05, N)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for name in ("sd", "ref"):
                for s in STATES:
                    save_state(d / f"{name}-{s}", planted_run(s, simple_harm, n_controls=3, noise_group=e), PRESET.replace("c2", "c3"))
            ov = overlap_rows(lambda i: 0.85, lambda i: 0.49, n=4)
            with open(d / "overlap.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(ov[0])); w.writeheader(); w.writerows(ov)
            with contextlib.redirect_stdout(io.StringIO()):
                eb.main(["--state-defined", str(d / "sd-FP"), str(d / "sd-INT6"), str(d / "sd-INT4"), "--reference", str(d / "ref-FP"), str(d / "ref-INT6"), str(d / "ref-INT4"),
                         "--ratios-json", str(ratios_json(d)), "--overlap", str(d / "overlap.csv"), "--out", str(d / "out"), "--n-boot", "100"])
            js = json.loads((d / "out" / "analysis.json").read_text())
            for l, x in js["rules"]["H-Gb2"]["INT4"]["per_language"].items():
                self.assertAlmostEqual(x["lo"], x["hi"], places=9)                              # identical runs and identical draws: no spread

if __name__ == "__main__":
    unittest.main()
