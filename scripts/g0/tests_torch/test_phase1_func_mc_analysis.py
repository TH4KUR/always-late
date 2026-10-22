import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phase1_func_mc_analysis as mc


def lse(x):
    x = np.asarray(x, dtype=float)
    return float(x.max() + np.log(np.exp(x - x.max()).sum()))


def margin_scores(answers, margins):
    """Option scores in which the correct option has score `margin` and the other three have 0."""
    s = np.zeros((len(answers), 4))
    for i, (a, m) in enumerate(zip(answers, margins)):
        s[i, a] = m
    return s


def lp_correct(margin):
    return margin - lse([margin, 0, 0, 0])


KEYS = [("p0", 1), ("p0", 2), ("p1", 1), ("p1", 2), ("p2", 1), ("p2", 2)]
ANSWERS = [0, 1, 2, 3, 0, 1]


def planted_run(label, none_m=2.0, zero_m=0.0, ctrl_ms=(2.0, 2.0), sham_m=None, langs=("aaa",), keys=KEYS, answers=ANSWERS, fmt="letter"):
    n = len(answers); scores = {}
    for code in langs:
        scores[(code, "none")] = margin_scores(answers, [none_m] * n)
        scores[(code, "zero_K64@" + code)] = margin_scores(answers, [zero_m] * n)
        for r, cm in enumerate(ctrl_ms):
            scores[(code, f"ctrl{r}_K64@{code}")] = margin_scores(answers, [cm] * n)
        scores[(code, "sham_K64@" + code)] = margin_scores(answers, [none_m if sham_m is None else sham_m] * n)
    return mc.McRun(label, fmt, list(langs), {c: list(keys) for c in langs}, {c: list(answers) for c in langs}, scores)


class FourWayProbability(unittest.TestCase):
    def test_uniform_scores_give_a_quarter(self):
        np.testing.assert_allclose(mc.option_logp_correct(np.zeros((2, 4)), [0, 3]), [np.log(0.25)] * 2)

    def test_a_dominant_correct_option_gives_about_zero_and_the_axis_is_the_options(self):
        self.assertAlmostEqual(float(mc.option_logp_correct(np.array([[50.0, 0, 0, 0]]), [0])[0]), 0.0, places=12)
        s = np.array([[1.0, 2.0, 3.0, 4.0]])
        self.assertAlmostEqual(float(mc.option_logp_correct(s, [2])[0]), 3.0 - lse([1, 2, 3, 4]), places=12)          # asymmetric scores: only the softmax over the four options gives this

    def test_very_negative_scores_do_not_overflow(self):
        v = mc.option_logp_correct(np.array([[-1000.0, -1001.0, -1002.0, -1003.0]]), [3])
        self.assertTrue(np.isfinite(v).all()); self.assertAlmostEqual(float(v[0]), -1003.0 - lse([-1000, -1001, -1002, -1003]), places=9)

    def test_the_cloze_score_of_an_option_is_its_length_normalised_log_likelihood(self):
        item = {"nll": np.array([4.0, 6.0, 2.0, 8.0, 1.0, 1.0, 1.0, 1.0]), "ntok": np.array([2, 3, 1, 4, 1, 1, 1, 1])}
        s = mc.scores_from_item("cloze", item)
        self.assertEqual(s.shape, (2, 4)); np.testing.assert_allclose(s[0], [-2.0, -2.0, -2.0, -2.0]); np.testing.assert_allclose(s[1], [-1.0] * 4)

    def test_the_letter_score_is_the_stored_option_log_probability(self):
        lp = np.log(np.array([[0.1, 0.2, 0.3, 0.1]]))
        np.testing.assert_allclose(mc.scores_from_item("letter", {"cand_logp": lp}), lp)
        with self.assertRaises(ValueError):
            mc.scores_from_item("bogus", {})


class HarmAndEffect(unittest.TestCase):
    def test_harm_is_the_drop_in_the_log_probability_of_the_correct_option(self):
        run = planted_run("FP", none_m=2.0, zero_m=0.0)
        h = run.harm("aaa", "zero_K64@aaa")
        np.testing.assert_allclose(h, [-(lp_correct(0.0) - lp_correct(2.0))] * 6)
        self.assertTrue((h > 0).all())                                                                       # removing the evidence for the answer hurts: positive harm
        np.testing.assert_allclose(run.harm("aaa", "sham_K64@aaa"), np.zeros(6))                            # the sham leaves the scores unchanged: exactly zero

    def test_an_improvement_is_a_negative_harm_not_clipped(self):
        h = planted_run("FP", none_m=0.0, zero_m=2.0).harm("aaa", "zero_K64@aaa")
        self.assertTrue((h < 0).all())

    def test_the_effect_is_the_mean_harm_beyond_the_mean_of_the_control_draws_by_hand(self):
        run = planted_run("FP", none_m=2.0, zero_m=0.0, ctrl_ms=(2.0, 1.0))
        hz = -(lp_correct(0.0) - lp_correct(2.0)); hc = (0.0 + (-(lp_correct(1.0) - lp_correct(2.0)))) / 2
        index = mc.passage_index([run]); W = np.ones((3, len(index)), dtype=int)
        e = mc.mc_effect(run, "aaa", "K64", index, W)
        self.assertAlmostEqual(e["point"], hz - hc, places=12)
        self.assertAlmostEqual(e["harm_group"], hz, places=12); self.assertAlmostEqual(e["harm_control"], hc, places=12)
        self.assertEqual((e["n_items"], e["n_controls"]), (6, 2))
        np.testing.assert_allclose(e["samples"], [e["point"]] * 3)                                           # every passage once reproduces the point

    def test_the_point_is_the_mean_not_the_median_of_the_item_differences(self):
        answers = [0, 0, 0]; keys = [("p0", 1), ("p1", 1), ("p2", 1)]
        run = planted_run("FP", none_m=2.0, ctrl_ms=(2.0,), keys=keys, answers=answers)
        run.scores[("aaa", "zero_K64@aaa")] = margin_scores(answers, [2.0, 2.0, -4.0])                       # harms 0, 0, large: mean and median differ
        e = mc.mc_effect(run, "aaa", "K64", mc.passage_index([run]), np.ones((1, 3), dtype=int))
        self.assertAlmostEqual(e["point"], -(lp_correct(-4.0) - lp_correct(2.0)) / 3, places=12); self.assertGreater(e["point"], 0.0)

    def test_a_planted_null_has_an_interval_around_zero(self):
        run = planted_run("FP", none_m=2.0, zero_m=2.0, ctrl_ms=(2.0, 2.0))
        e = mc.mc_effect(run, "aaa", "K64", mc.passage_index([run]), mc.passage_draws(mc.passage_index([run]), 200, 0))
        self.assertAlmostEqual(e["point"], 0.0); np.testing.assert_allclose(e["samples"], 0.0, atol=1e-12)

    def test_missing_controls_or_zero_ablation_are_errors(self):
        run = planted_run("FP", ctrl_ms=())
        with self.assertRaises(KeyError):
            mc.mc_effect(run, "aaa", "K64", mc.passage_index([run]), np.ones((2, 3), dtype=int))


class Validation(unittest.TestCase):
    def test_scores_must_match_the_items_and_have_four_options(self):
        with self.assertRaises(ValueError):
            mc.McRun("FP", "letter", ["aaa"], {"aaa": KEYS}, {"aaa": ANSWERS}, {("aaa", "none"): np.zeros((5, 4))})
        with self.assertRaises(ValueError):
            mc.McRun("FP", "letter", ["aaa"], {"aaa": KEYS}, {"aaa": ANSWERS}, {("aaa", "none"): np.zeros((6, 3))})


class PassageBootstrap(unittest.TestCase):
    def test_draws_are_multinomial_counts_of_the_passages_and_seeded(self):
        W = mc.passage_draws(["p0", "p1", "p2"], 50, 0)
        self.assertEqual(W.shape, (50, 3)); self.assertTrue((W.sum(1) == 3).all()); self.assertTrue((W >= 0).all())
        np.testing.assert_array_equal(W, mc.passage_draws(["p0", "p1", "p2"], 50, 0)); self.assertFalse(np.array_equal(W, mc.passage_draws(["p0", "p1", "p2"], 50, 1)))

    def test_samples_are_ratios_of_weighted_sums_over_weighted_counts(self):
        v = np.array([1.0, 2.0, 3.0, 4.0]); p = np.array([0, 0, 1, 1]); W = np.array([[1, 1], [2, 0], [0, 2]])
        np.testing.assert_allclose(mc.cluster_samples(v, p, W), [2.5, 1.5, 3.5])

    def test_a_draw_that_leaves_a_language_without_items_is_an_error(self):
        with self.assertRaises(ValueError):
            mc.cluster_samples(np.array([1.0, 2.0]), np.array([0, 0]), np.array([[0, 2]]))                    # the draw picked only passage 1, which this language does not have

    def test_whole_passages_are_resampled_not_items(self):
        # two passages whose items are identical inside a passage: a draw can only give a mean of 0, 0.5 or 1; independent item resampling would give 0.25 or 0.75 as well
        run = planted_run("FP", none_m=2.0, zero_m=0.0, ctrl_ms=(2.0,), keys=[("p0", 1), ("p0", 2), ("p1", 1), ("p1", 2)], answers=[0, 1, 2, 3])
        hz = -(lp_correct(0.0) - lp_correct(2.0))
        run.scores[("aaa", "zero_K64@aaa")][2:] = run.scores[("aaa", "none")][2:]                       # passage p1 is not harmed, p0 is
        idx = mc.passage_index([run]); e = mc.mc_effect(run, "aaa", "K64", idx, mc.passage_draws(idx, 400, 3))
        vals = {round(float(x) / hz, 6) for x in e["samples"]}
        self.assertTrue(vals <= {0.0, 0.5, 1.0}, vals); self.assertEqual(len(vals), 3)

    def test_the_same_draws_serve_every_language_and_precision(self):
        a = planted_run("FP", langs=("aaa", "bbb")); idx = mc.passage_index([a]); W = mc.passage_draws(idx, 30, 0)
        ea, eb = mc.mc_effect(a, "aaa", "K64", idx, W), mc.mc_effect(a, "bbb", "K64", idx, W)
        np.testing.assert_array_equal(ea["samples"], eb["samples"])                                           # identical data under the same draws: identical samples

    def test_the_passage_list_is_the_sorted_union_over_runs_and_languages(self):
        a = planted_run("FP", keys=[("p1", 1), ("p3", 1)], answers=[0, 1]); b = planted_run("INT6", keys=[("p0", 1), ("p1", 1)], answers=[0, 1])
        self.assertEqual(mc.passage_index([a, b]), ["p0", "p1", "p3"])


class Accuracy(unittest.TestCase):
    def test_accuracy_is_derived_from_the_scores_of_the_same_configuration(self):
        run = planted_run("FP", none_m=2.0, zero_m=-2.0)                                                     # none: the correct option is the maximum; zero: it is the minimum
        self.assertEqual(run.accuracy("aaa", "none"), 1.0); self.assertEqual(run.accuracy("aaa", "zero_K64@aaa"), 0.0)
        mixed = planted_run("FP"); mixed.scores[("aaa", "none")][:3] = margin_scores(ANSWERS[:3], [-2.0] * 3)
        self.assertAlmostEqual(mixed.accuracy("aaa", "none"), 0.5)


class Files(unittest.TestCase):
    def test_letter_and_cloze_files_load_into_the_same_scores(self):
        import torch
        d = Path(tempfile.mkdtemp())
        lp = torch.tensor(np.log(np.array([[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]])))
        letter = {"precision": "INT6", "split": "eval", "format": "letter", "languages": ["aaa"], "preset": "mc_eval", "keys": {"aaa": [("p0", 1), ("p0", 2)]}, "dropped": {"aaa": 0}, "answers": {"aaa": [0, 3]},
                  "items": {"aaa|none": {"cand_logp": lp}, "aaa|sham_K64@aaa": {"cand_logp": lp}}, "stats": {}}
        torch.save(letter, d / "func_mc_eval_letter_INT6.pt")
        r = mc.McRun.from_file(d / "func_mc_eval_letter_INT6.pt")
        self.assertEqual((r.label, r.fmt, r.languages), ("INT6", "letter", ["aaa"])); np.testing.assert_allclose(r.scores[("aaa", "none")], lp.numpy()); self.assertEqual(r.keys["aaa"], [("p0", 1), ("p0", 2)])
        cloze = dict(letter, format="cloze", items={"aaa|none": {"nll": torch.tensor([4.0, 6.0, 2.0, 8.0] * 2, dtype=torch.float64), "ntok": torch.tensor([2, 3, 1, 4] * 2)}})
        torch.save(cloze, d / "func_mc_eval_cloze_INT6.pt")
        np.testing.assert_allclose(mc.McRun.from_file(d / "func_mc_eval_cloze_INT6.pt").scores[("aaa", "none")], [[-2.0, -2.0, -2.0, -2.0]] * 2)


def runs_pair(**kw):
    return {"FP": planted_run("FP", **kw), "INT6": planted_run("INT6", **kw)}


class Analyse(unittest.TestCase):
    def setUp(self):
        self.out = Path(tempfile.mkdtemp()) / "res"

    def test_identical_precisions_have_a_retention_of_exactly_one(self):
        res = mc.analyse_mc(runs_pair(), self.out, n_boot=100)
        r = res["retention"]["INT6"]["languages"]["aaa"]
        self.assertEqual((r["ratio"], r["lo"], r["hi"], r["class"]), (1.0, 1.0, 1.0, "preserved"))
        m = res["retention"]["INT6"]["macro"]; self.assertEqual((m["ratio"], m["n_languages"], m["class"]), (1.0, 1, "preserved"))

    def test_the_same_draws_serve_every_precision_so_identical_heterogeneous_data_give_exactly_one(self):
        keys = [(f"p{i}", q) for i in range(8) for q in (1, 2)]; answers = [i % 4 for i in range(16)]

        def hetero(label):                                                                                   # every passage is harmed, passages 0 to 3 strongly and 4 to 7 weakly: the draws matter
            run = planted_run(label, none_m=2.0, zero_m=0.0, ctrl_ms=(2.0,), keys=keys, answers=answers)
            run.scores[("aaa", "zero_K64@aaa")] = margin_scores(answers, [0.0] * 8 + [1.5] * 8)
            return run
        res = mc.analyse_mc({"FP": hetero("FP"), "INT6": hetero("INT6")}, self.out, n_boot=300)
        e = res["effects"]["FP"]["aaa"]; self.assertGreater(e["hi"] - e["lo"], 0.01)                          # the interval is not degenerate, so the check below is meaningful
        r = res["retention"]["INT6"]["languages"]["aaa"]
        self.assertEqual((r["ratio"], r["lo"], r["hi"]), (1.0, 1.0, 1.0))

    def hetero_pair(self):
        keys = [(f"p{i}", q) for i in range(8) for q in (1, 2)]; answers = [i % 4 for i in range(16)]
        def run(label, margins):
            r = planted_run(label, none_m=2.0, zero_m=0.0, ctrl_ms=(2.0,), keys=keys, answers=answers)
            r.scores[("aaa", "zero_K64@aaa")] = margin_scores(answers, margins)
            return r
        return {"FP": run("FP", [0.0] * 8 + [1.5] * 8), "INT6": run("INT6", [0.4] * 8 + [1.2] * 8)}

    def test_the_retention_level_widens_only_the_retention_intervals(self):
        r95 = mc.analyse_mc(self.hetero_pair(), self.out / "a", n_boot=400)
        r99 = mc.analyse_mc(self.hetero_pair(), self.out / "b", n_boot=400, retention_level=0.99)
        self.assertEqual((r95["retention_level"], r99["retention_level"]), (0.95, 0.99))
        for pick in (lambda r: r["retention"]["INT6"]["macro"], lambda r: r["retention"]["INT6"]["languages"]["aaa"]):
            a, b = pick(r95), pick(r99)
            self.assertEqual(a["ratio"], b["ratio"]); self.assertLess(b["lo"], a["lo"]); self.assertGreater(b["hi"], a["hi"])
        a, b = r95["retention"]["INT6"]["macro"], r99["retention"]["INT6"]["macro"]
        self.assertLess(b["median_lo"], a["median_lo"]); self.assertGreater(b["median_hi"], a["median_hi"])
        a, b = r95["retention"]["INT6"]["pooled"], r99["retention"]["INT6"]["pooled"]
        self.assertEqual(a["ratio"], b["ratio"]); self.assertLess(b["lo"], a["lo"]); self.assertGreater(b["hi"], a["hi"])
        for lab in ("FP", "INT6"):                                                                           # statement S0 and the effects keep the 95 % intervals
            self.assertEqual(r95["macro"][lab], r99["macro"][lab]); self.assertEqual(r95["effects"][lab]["aaa"], r99["effects"][lab]["aaa"])
        rb, ra = (self.out / "b" / "report.md").read_text(), (self.out / "a" / "report.md").read_text()
        self.assertIn("using 99 % intervals", rb); self.assertIn("| 99 % interval |", rb); self.assertIn("using 95 % intervals", ra); self.assertIn("| 95 % interval |", ra)

    def test_the_default_level_is_95_percent_and_a_level_outside_the_open_unit_interval_is_refused(self):
        self.assertEqual(mc.analyse_mc(runs_pair(), self.out / "c", n_boot=20)["retention_level"], 0.95)
        for bad in (0.0, 1.0, 1.5, -0.1):
            with self.assertRaises(ValueError):
                mc.analyse_mc(runs_pair(), self.out / "d", n_boot=20, retention_level=bad)

    def test_a_planted_attenuation_and_amplification_are_classified(self):
        runs = {"FP": planted_run("FP", none_m=2.0, zero_m=0.0), "INT6": planted_run("INT6", none_m=2.0, zero_m=1.5), "INT4": planted_run("INT4", none_m=2.0, zero_m=-3.0)}
        res = mc.analyse_mc(runs, self.out, n_boot=100)
        self.assertEqual(res["retention"]["INT6"]["languages"]["aaa"]["class"], "attenuated"); self.assertEqual(res["retention"]["INT4"]["languages"]["aaa"]["class"], "amplified")
        for lab, cls, lo, hi in (("INT6", "attenuated", 0.0, 0.75), ("INT4", "amplified", 1.33, 100.0)):
            m = res["retention"][lab]["macro"]; self.assertEqual(m["class"], cls); self.assertTrue(lo < m["ratio"] < hi, (lab, m["ratio"]))
            pl = res["retention"][lab]["pooled"]; self.assertEqual(pl["class"], cls); self.assertTrue(lo < pl["ratio"] < hi)
        e = res["effects"]
        self.assertGreater(e["FP"]["aaa"]["point"], e["INT6"]["aaa"]["point"]); self.assertGreater(e["INT4"]["aaa"]["point"], e["FP"]["aaa"]["point"])

    def test_an_effect_of_exactly_zero_is_not_counted_as_positive(self):
        res = mc.analyse_mc({"FP": planted_run("FP", none_m=2.0, zero_m=2.0, ctrl_ms=(2.0,))}, self.out, n_boot=20)
        self.assertEqual(res["macro"]["FP"]["n_positive"], 0); self.assertEqual(res["macro"]["FP"]["point"], 0.0)

    def test_the_sham_harm_is_exactly_zero(self):
        res = mc.analyse_mc(runs_pair(), self.out, n_boot=50)
        s = res["sham"]["INT6"]["aaa"]; self.assertEqual((s["point"], s["lo"], s["hi"]), (0.0, 0.0, 0.0))

    def test_the_schema_of_the_loss_analysis_is_kept_so_that_the_figures_can_read_it(self):
        res = mc.analyse_mc(runs_pair(), self.out, n_boot=50, covered=["aaa"])
        for k in ("K", "n_boot", "seed", "bands", "covered", "effects", "macro", "retention", "sham"):
            self.assertIn(k, res)
        e = res["effects"]["FP"]["aaa"]
        for k in ("point", "lo", "hi", "harm_group", "harm_control", "norm_ratio_group", "norm_ratio_control"):
            self.assertIn(k, e)
        self.assertIsNone(e["norm_ratio_group"])
        self.assertEqual(set(res["macro"]["FP"]), {"point", "lo", "hi", "median", "n_positive"})
        with open(self.out / "effects_by_language.csv") as f:
            self.assertEqual(next(csv.reader(f)), ["precision", "language", "effect", "lo", "hi", "harm_group", "harm_control", "norm_ratio_group", "norm_ratio_control"])
        with open(self.out / "retention.csv") as f:
            self.assertEqual(next(csv.reader(f)), ["precision", "language", "ratio", "lo", "hi", "class"])
        self.assertEqual(json.loads((self.out / "analysis.json").read_text())["covered"], ["aaa"])
        self.assertTrue((self.out / "report.md").read_text().startswith("# "))

    def test_the_accuracy_table_lists_every_configuration_family_per_precision_and_language(self):
        mc.analyse_mc(runs_pair(), self.out, n_boot=20)
        with open(self.out / "accuracy.csv") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual({(r["precision"], r["language"]) for r in rows}, {("FP", "aaa"), ("INT6", "aaa")})
        r = rows[0]
        for k in ("acc_none", "acc_zero", "acc_control_mean", "acc_sham", "n_items"):
            self.assertIn(k, r)
        self.assertEqual(float(r["acc_none"]), 1.0); self.assertEqual(r["n_items"], "6")
        self.assertAlmostEqual(float(r["acc_zero"]), 2 / 6)                                                   # all options tie at 0: the first option is taken, which is correct for 2 of the 6 items
        self.assertEqual(float(r["acc_control_mean"]), 1.0); self.assertEqual(float(r["acc_sham"]), 1.0)

    def test_the_covered_languages_restrict_the_macro_only(self):
        runs = {"FP": planted_run("FP", langs=("aaa", "bbb")), "INT6": planted_run("INT6", langs=("aaa", "bbb"))}
        res = mc.analyse_mc(runs, self.out, n_boot=20, covered=["aaa"])
        self.assertEqual(res["covered"], ["aaa"]); self.assertEqual(set(res["effects"]["FP"]), {"aaa", "bbb"}); self.assertEqual(res["retention"]["INT6"]["macro"]["languages"], ["aaa"])

    def test_the_full_precision_run_is_required(self):
        with self.assertRaises(ValueError):
            mc.analyse_mc({"INT6": planted_run("INT6")}, self.out, n_boot=10)

    def test_items_must_be_identical_across_precisions(self):
        other = planted_run("INT6", keys=[("p0", 1), ("p0", 2), ("p1", 1), ("p1", 2), ("p2", 1), ("p9", 2)])
        with self.assertRaises(ValueError):
            mc.analyse_mc({"FP": planted_run("FP"), "INT6": other}, self.out, n_boot=10)

    def test_the_figures_module_reads_the_output_unchanged(self):
        try:
            import phase1_func_figures as ff
        except ImportError:
            self.skipTest("matplotlib not installed")
        mc.analyse_mc(runs_pair(), self.out, n_boot=50)
        an = json.loads((self.out / "analysis.json").read_text())
        eff = [r for r in ff._read_csv(self.out / "effects_by_language.csv", {"effect", "lo", "hi", "harm_group", "harm_control", "norm_ratio_group", "norm_ratio_control"}) if r["precision"] == "FP"]
        self.assertEqual(len(eff), 1)
        ff.fig_effect_per_language(eff, "light", "t-mc"); ff.fig_retention(an["retention"], an["bands"], "light", "t-mc")


class Main(unittest.TestCase):
    def test_main_reads_the_files_of_one_format_and_writes_the_analysis(self):
        import torch
        d = Path(tempfile.mkdtemp()); out = d / "out"
        for lab in ("FP", "INT6"):
            r = planted_run(lab)
            items = {f"{c}|{cfg}": {"cand_logp": torch.tensor(s)} for (c, cfg), s in r.scores.items()}
            torch.save({"precision": lab, "split": "eval", "format": "letter", "languages": ["aaa"], "preset": "mc_eval", "keys": r.keys, "dropped": {"aaa": 0}, "answers": r.answers, "items": items, "stats": {}},
                       d / f"func_mc_eval_letter_{lab}.pt")
        (d / "func_mc_eval_cloze_FP.pt").write_bytes(b"not read")                                           # another format in the same folder must be ignored
        mc.main(["--runs", str(d), "--split", "eval", "--format", "letter", "--out", str(out), "--n-boot", "30"])
        self.assertTrue((out / "analysis.json").exists()); self.assertEqual(sorted(json.loads((out / "analysis.json").read_text())["effects"]), ["FP", "INT6"])
        self.assertIn("Held-out T-MC evaluation", (out / "report.md").read_text())
        mc.main(["--runs", str(d), "--split", "eval", "--format", "letter", "--out", str(d / "out2"), "--n-boot", "10", "--note", "custom status", "--retention-level", "0.99"])
        self.assertIn("custom status", (d / "out2" / "report.md").read_text())
        self.assertEqual(json.loads((d / "out2" / "analysis.json").read_text())["retention_level"], 0.99)


if __name__ == "__main__":
    unittest.main()
