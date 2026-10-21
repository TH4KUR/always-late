import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phase1_func_analysis as fa


def make_run(label, harm_group=0.5, harm_ctrl=0.1, n=200, langs=("aa", "bb"), seed=0, sham=0.0, R=2, K="K64", noise=0.05, extra=None):
    """A planted precision: per-sentence NLL sums with known group/control harm (nats per token) on the target language's own text."""
    g = np.random.default_rng(seed)
    items, stats, fid = {}, {}, {}
    for code in langs:
        ntok = g.integers(8, 30, n)
        base = ntok * g.uniform(2.0, 3.0, n)
        items[f"{code}|none"] = {"nll": torch.tensor(base), "ntok": torch.tensor(ntok)}
        def add(name, per_tok):
            items[f"{code}|{name}"] = {"nll": torch.tensor(base + ntok * (per_tok + g.normal(0, noise, n))), "ntok": torch.tensor(ntok)}
        add(f"zero_{K}@{code}", harm_group)
        for r in range(R):
            add(f"ctrl{r}_{K}@{code}", harm_ctrl)
        add(f"sham_{K}@{code}", sham)
        stats[f"{code}|zero_{K}@{code}"] = {"sum_norm_ratio": 0.4 * n * 20, "sum_active": 3.0 * n * 20, "rows": n * 20}
        stats[f"{code}|ctrl0_{K}@{code}"] = {"sum_norm_ratio": 0.38 * n * 20, "sum_active": 3.0 * n * 20, "rows": n * 20}
        fid[code] = {"sum_err2": 10.0, "sum_x2": 100.0, "sum_l0": 400.0, "rows": 20.0}
    if extra:
        extra(items)
    return fa.Run(label, items, stats, fid, list(langs), {c: list(range(n)) for c in langs})


class Basics(unittest.TestCase):
    def test_pooled_effect_is_the_ratio_of_sums(self):
        d = np.array([1.0, 2.0, 3.0]); n = np.array([1, 1, 4])
        self.assertAlmostEqual(fa.pooled(d, n), 6.0 / 6.0)                       # a mean of ratios would be 1.25
        self.assertAlmostEqual(fa.pooled(d, n, idx=np.array([0, 2])), 4.0 / 5.0)

    def test_draws_are_seeded_and_in_range(self):
        a, b = fa.draws(50, 20, seed=1), fa.draws(50, 20, seed=1)
        self.assertTrue(np.array_equal(a, b))
        self.assertFalse(np.array_equal(a, fa.draws(50, 20, seed=2)))
        self.assertEqual(a.shape, (20, 50))
        self.assertTrue(a.min() >= 0 and a.max() < 50)

    def test_the_run_exposes_paired_deltas(self):
        run = make_run("FP")
        d, n = run.delta("aa", "zero_K64@aa")
        self.assertEqual(d.shape, (200,))
        self.assertTrue(np.array_equal(n, run.ntok("aa")))
        self.assertAlmostEqual(fa.pooled(d, n), 0.5, delta=0.02)
        with self.assertRaises(KeyError):
            run.delta("aa", "nope")

    def test_runs_with_unpaired_token_counts_are_rejected(self):
        def tamper(items):
            items["aa|zero_K64@aa"]["ntok"] = items["aa|zero_K64@aa"]["ntok"] + 1
        with self.assertRaises(ValueError):
            make_run("FP", extra=tamper)


class Effects(unittest.TestCase):
    def setUp(self):
        self.D = fa.draws(200, 500, seed=0)

    def test_planted_effect_is_recovered_and_the_interval_contains_it(self):
        run = make_run("FP")
        e = fa.functional_effect(run, "aa", "K64", self.D)
        self.assertAlmostEqual(e["point"], 0.4, delta=0.02)
        lo, hi = np.percentile(e["samples"], [2.5, 97.5])
        self.assertLess(lo, 0.4); self.assertGreater(hi, 0.4)
        self.assertGreater(lo, 0.3)

    def test_a_planted_null_has_an_interval_around_zero(self):
        run = make_run("FP", harm_group=0.2, harm_ctrl=0.2)
        e = fa.functional_effect(run, "aa", "K64", self.D)
        lo, hi = np.percentile(e["samples"], [2.5, 97.5])
        self.assertLess(lo, 0); self.assertGreater(hi, 0)

    def test_all_control_draws_are_averaged(self):
        def plant(items):
            for code in ("aa", "bb"):
                base = items[f"{code}|none"]; nt = base["ntok"].numpy()
                items[f"{code}|zero_K64@{code}"] = {"nll": base["nll"] + torch.tensor(0.5 * nt), "ntok": base["ntok"]}
                items[f"{code}|ctrl0_K64@{code}"] = {"nll": base["nll"].clone(), "ntok": base["ntok"]}
                items[f"{code}|ctrl1_K64@{code}"] = {"nll": base["nll"] + torch.tensor(0.2 * nt), "ntok": base["ntok"]}
        e = fa.functional_effect(make_run("FP", extra=plant), "aa", "K64", self.D)
        self.assertAlmostEqual(e["point"], 0.4, places=9)                      # 0.5 - mean(0.0, 0.2); the first draw alone would give 0.5
        self.assertEqual(e["n_controls"], 2)

    def test_negative_effects_are_kept(self):
        run = make_run("FP", harm_group=-0.3, harm_ctrl=0.1)
        self.assertAlmostEqual(fa.functional_effect(run, "aa", "K64", self.D)["point"], -0.4, delta=0.02)

    def test_resampling_pairs_languages_through_the_same_sentence_positions(self):
        # language aa is harmed only at positions < 100, language bb only at positions >= 100; every resampled position belongs to exactly one of them
        def plant(items):
            for code, sel in (("aa", lambda i: i < 100), ("bb", lambda i: i >= 100)):
                base = items[f"{code}|none"]; nt = base["ntok"].numpy()
                bump = np.array([1.0 if sel(i) else 0.0 for i in range(200)])
                items[f"{code}|zero_K64@{code}"] = {"nll": base["nll"] + torch.tensor(bump * nt), "ntok": base["ntok"]}
                for r in range(2):
                    items[f"{code}|ctrl{r}_K64@{code}"] = {"nll": base["nll"].clone(), "ntok": base["ntok"]}
        run = make_run("FP", extra=plant)
        a = fa.functional_effect(run, "aa", "K64", self.D)
        b = fa.functional_effect(run, "bb", "K64", self.D)
        # exact paired statement: in every draw the same resampled positions feed both languages
        nt = run.ntok("aa")
        num_a = np.array([(nt[d] * (d < 100)).sum() for d in self.D]); num_b = np.array([(run.ntok("bb")[d] * (d >= 100)).sum() for d in self.D])
        tot_a = np.array([nt[d].sum() for d in self.D]); tot_b = np.array([run.ntok("bb")[d].sum() for d in self.D])
        self.assertTrue(np.allclose(a["samples"], num_a / tot_a)); self.assertTrue(np.allclose(b["samples"], num_b / tot_b))
        self.assertGreater(float(np.std(a["samples"])), 0)

    def test_identical_data_at_two_precisions_gives_a_zero_width_difference(self):
        r1, r2 = make_run("FP"), make_run("INT6")
        e1, e2 = fa.functional_effect(r1, "aa", "K64", self.D), fa.functional_effect(r2, "aa", "K64", self.D)
        self.assertTrue(np.array_equal(e1["samples"], e2["samples"]))
        self.assertEqual(float(np.ptp(e1["samples"] - e2["samples"])), 0.0)

    def test_deterministic_under_the_seed(self):
        run = make_run("FP")
        a = fa.functional_effect(run, "aa", "K64", fa.draws(200, 100, seed=3))
        b = fa.functional_effect(run, "aa", "K64", fa.draws(200, 100, seed=3))
        self.assertTrue(np.array_equal(a["samples"], b["samples"]))

    def test_dose_statistics_are_reported(self):
        e = fa.functional_effect(make_run("FP"), "aa", "K64", self.D)
        self.assertAlmostEqual(e["norm_ratio_group"], 0.4); self.assertAlmostEqual(e["norm_ratio_control"], 0.38)
        self.assertAlmostEqual(e["n_controls"], 2)

    def test_macro_average_over_languages(self):
        run = make_run("FP", langs=("aa", "bb", "cc"))
        per = {c: fa.functional_effect(run, c, "K64", self.D) for c in ("aa", "bb", "cc")}
        m = fa.macro(per, ["aa", "bb"])
        self.assertAlmostEqual(m["point"], (per["aa"]["point"] + per["bb"]["point"]) / 2)
        self.assertTrue(np.allclose(m["samples"], (per["aa"]["samples"] + per["bb"]["samples"]) / 2))


class Retention(unittest.TestCase):
    def setUp(self):
        self.D = fa.draws(400, 500, seed=0)

    def ratio(self, harm_b, harm_fp=0.5, seed_b=1, **kw):
        fp = make_run("FP", harm_group=harm_fp, harm_ctrl=0.1, n=400, seed=1)
        b = make_run("INT6", harm_group=harm_b, harm_ctrl=0.1, n=400, seed=seed_b, **kw)
        efp, eb = fa.functional_effect(fp, "aa", "K64", self.D), fa.functional_effect(b, "aa", "K64", self.D)
        return fa.retention(eb, efp)

    def test_preserved_attenuated_amplified_and_inconclusive(self):
        self.assertEqual(self.ratio(0.5)["class"], "preserved")
        self.assertEqual(self.ratio(0.2)["class"], "attenuated")                      # effect 0.1 against 0.4
        self.assertEqual(self.ratio(1.2)["class"], "amplified")                       # 1.1 against 0.4
        noisy = self.ratio(0.5, noise=1.5, seed_b=2)
        self.assertEqual(noisy["class"], "inconclusive")

    def test_ratio_value_and_interval(self):
        r = self.ratio(0.3)
        self.assertAlmostEqual(r["ratio"], 0.5, delta=0.05)
        self.assertLess(r["lo"], r["ratio"]); self.assertGreater(r["hi"], r["ratio"])

    def test_no_ratio_when_the_full_precision_effect_is_not_clearly_positive(self):
        fp = make_run("FP", harm_group=0.13, harm_ctrl=0.1, n=400, seed=1, noise=0.6)
        b = make_run("INT6", harm_group=0.3, harm_ctrl=0.1, n=400, seed=1)
        efp = fa.functional_effect(fp, "aa", "K64", self.D)
        self.assertGreater(efp["point"], 0)                                      # a positive point estimate whose interval still reaches zero
        self.assertLessEqual(np.percentile(efp["samples"], 2.5), 0)
        r = fa.retention(fa.functional_effect(b, "aa", "K64", self.D), efp)
        self.assertIsNone(r["ratio"]); self.assertEqual(r["class"], "undefined")

    def test_a_point_estimate_inside_the_bands_is_not_enough_for_preserved(self):
        efp = {"point": 0.4, "samples": 0.4 + 0.001 * np.random.default_rng(0).normal(size=500)}
        eb = {"point": 0.4, "samples": 0.4 + 0.12 * np.random.default_rng(1).normal(size=500)}                 # ratio 1.0, interval about [0.4, 1.6]
        r = fa.retention(eb, efp)
        self.assertAlmostEqual(r["ratio"], 1.0)
        self.assertEqual(r["class"], "inconclusive")

    def test_bands_are_parameters(self):
        fp = make_run("FP", n=400, seed=1); b = make_run("INT6", harm_group=0.4, n=400, seed=1)
        efp, eb = fa.functional_effect(fp, "aa", "K64", self.D), fa.functional_effect(b, "aa", "K64", self.D)
        self.assertEqual(fa.retention(eb, efp, bands=(0.95, 1.05))["class"], "attenuated")
        self.assertEqual(fa.retention(eb, efp, bands=(0.5, 1.5))["class"], "preserved")


def make_rescue_runs(gap=0.2, rec=None, n=300, langs=("aa", "bb"), seed=3, noise=0.01, K="K64", ctrl_second=None, exact_gap=False):
    """FP run and a quantized run whose baseline is `gap` nats per token worse; the quantized run also holds rescue configurations with planted recovered amounts (nats per token)."""
    rec = rec or {"group": 0.05, "ctrl": 0.01, "all": 0.15, "error": 0.04, "full": 0.19}
    g = np.random.default_rng(seed)
    fp_items, q_items = {}, {}
    for code in langs:
        ntok = g.integers(8, 30, n); base = ntok * g.uniform(2.0, 3.0, n)
        fp_items[f"{code}|none"] = {"nll": torch.tensor(base), "ntok": torch.tensor(ntok)}
        e = g.normal(0, noise, n)
        if exact_gap:
            e = e - (e * ntok).sum() / ntok.sum()                                     # the pooled gap is exactly `gap`, whatever the noise
        qb = base + ntok * (gap + e)
        q_items[f"{code}|none"] = {"nll": torch.tensor(qb), "ntok": torch.tensor(ntok)}
        def add(name, r):
            q_items[f"{code}|{name}"] = {"nll": torch.tensor(qb - ntok * (r + g.normal(0, noise, n))), "ntok": torch.tensor(ntok)}
        add(f"rescue_{K}@{code}", rec["group"]); add(f"rescuectrl0_{K}@{code}", rec["ctrl"]); add(f"rescuectrl1_{K}@{code}", rec["ctrl"] if ctrl_second is None else ctrl_second)
        add("rescue_all", rec["all"]); add("rescue_error", rec["error"]); add("rescue_full", rec["full"])
    mk = lambda lab, items: fa.Run(lab, items, {}, {}, list(langs), {c: list(range(n)) for c in langs})
    return mk("FP", fp_items), mk("INT4", q_items)


class RescueAnalysis(unittest.TestCase):
    def setUp(self):
        self.D = fa.draws(300, 400, seed=0)

    def test_fractions_and_bounds_are_recovered(self):
        fp, q = make_rescue_runs()
        r = fa.rescue_effect(q, fp, "aa", "K64", self.D)
        self.assertAlmostEqual(r["gap"]["point"], 0.2, delta=0.01)
        for name, want in (("group", 0.05), ("control", 0.01), ("all", 0.15), ("error", 0.04), ("full", 0.19)):
            self.assertAlmostEqual(r["recovered"][name]["point"], want, delta=0.01)
        self.assertAlmostEqual(r["fraction"]["group"]["point"], 0.25, delta=0.02)
        self.assertAlmostEqual(r["fraction"]["group_minus_control"]["point"], 0.20, delta=0.02)
        self.assertAlmostEqual(r["fraction"]["full"]["point"], 0.95, delta=0.02)
        lo, hi = r["fraction"]["group"]["lo"], r["fraction"]["group"]["hi"]
        self.assertLess(lo, 0.25); self.assertGreater(hi, 0.25)

    def test_all_control_draws_are_averaged(self):
        fp, q = make_rescue_runs(rec={"group": 0.05, "ctrl": 0.00, "all": 0.15, "error": 0.04, "full": 0.19}, ctrl_second=0.04)
        r = fa.rescue_effect(q, fp, "aa", "K64", self.D)
        self.assertAlmostEqual(r["recovered"]["control"]["point"], 0.02, delta=0.003)                   # mean of 0.00 and 0.04, not the first draw alone

    def test_a_positive_but_uncertain_gap_gives_no_fraction(self):
        fp, q = make_rescue_runs(gap=0.03, noise=0.6, exact_gap=True)
        r = fa.rescue_effect(q, fp, "aa", "K64", self.D, min_gap=0.02)
        self.assertAlmostEqual(r["gap"]["point"], 0.03, places=9)
        self.assertLessEqual(np.percentile(r["gap"]["samples"], 2.5), 0.02)                             # the point estimate clears the threshold, its interval does not
        self.assertIsNone(r["fraction"]["group"])

    def test_no_fraction_when_the_quantization_gap_is_negligible(self):
        fp, q = make_rescue_runs(gap=0.005)
        r = fa.rescue_effect(q, fp, "aa", "K64", self.D, min_gap=0.02)
        self.assertIsNone(r["fraction"]["group"]); self.assertIn("point", r["recovered"]["group"])        # the raw recovery is still reported

    def test_negative_recovery_is_kept(self):
        fp, q = make_rescue_runs(rec={"group": -0.03, "ctrl": 0.0, "all": 0.1, "error": 0.0, "full": 0.2})
        r = fa.rescue_effect(q, fp, "aa", "K64", self.D)
        self.assertAlmostEqual(r["recovered"]["group"]["point"], -0.03, delta=0.01)
        self.assertLess(r["fraction"]["group"]["point"], 0)

    def test_missing_bounds_are_skipped_not_invented(self):
        fp, q = make_rescue_runs()
        for k in [k for k in list(q._items) if k[1] in ("rescue_all", "rescue_error", "rescue_full")]:
            del q._items[k]
        r = fa.rescue_effect(q, fp, "aa", "K64", self.D)
        self.assertNotIn("all", r["recovered"]); self.assertIn("group", r["recovered"])

    def test_analyse_writes_the_rescue_table_when_rescue_configs_exist(self):
        fp, q = make_rescue_runs()
        for run in (fp, q):                                                                          # the primary configurations are needed by analyse()
            for code in run.languages:
                base, nt = run._items[(code, "none")]
                for name, h in ((f"zero_K64@{code}", 0.5), (f"ctrl0_K64@{code}", 0.1), (f"ctrl1_K64@{code}", 0.1)):
                    run._items[(code, name)] = (base + nt * h, nt)
        for code in fp.languages:                                                                    # even if full precision holds rescue configurations, it is the reference, not a recipient
            fp._items[(code, "rescue_K64@" + code)] = fp._items[(code, "none")]
            fp._items[(code, "rescuectrl0_K64@" + code)] = fp._items[(code, "none")]
        out = Path(tempfile.mkdtemp())
        res = fa.analyse({"FP": fp, "INT4": q}, out, K="K64", n_boot=100, seed=0)
        self.assertTrue((out / "rescue.csv").exists())
        self.assertIn("INT4", res["rescue"]); self.assertNotIn("FP", res["rescue"])
        self.assertIn("rescue", (out / "report.md").read_text().lower())


class Checks(unittest.TestCase):
    def test_sham_and_off_target_and_fidelity(self):
        run = make_run("FP", sham=0.0, noise=0.05)
        D = fa.draws(200, 300, seed=0)
        s = fa.sham_check(run, "aa", "K64", D)
        self.assertLess(abs(s["point"]), 0.02)
        f = fa.fidelity_table({"FP": run})
        row = [r for r in f if r["precision"] == "FP" and r["language"] == "aa"][0]
        self.assertAlmostEqual(row["fvu"], 0.1); self.assertAlmostEqual(row["l0"], 20.0); self.assertAlmostEqual(row["rms_norm"], (100.0 / 20.0) ** 0.5)

    def test_off_target_harm_on_panel_text(self):
        def plant(items):
            base = items["bb|none"]
            items["bb|zero_K64@aa"] = {"nll": base["nll"] + base["ntok"] * 0.01, "ntok": base["ntok"]}
        run = make_run("FP", extra=plant)
        o = fa.off_target(run, "aa", "K64", fa.draws(200, 100, seed=0))
        self.assertEqual(list(o), ["bb"])
        self.assertAlmostEqual(o["bb"]["point"], 0.01, places=6)


class Report(unittest.TestCase):
    def test_full_analysis_writes_tables_and_a_report_with_limits(self):
        runs = {"FP": make_run("FP"), "INT6": make_run("INT6", harm_group=0.3), "INT4": make_run("INT4", harm_group=0.1)}
        out = Path(tempfile.mkdtemp())
        res = fa.analyse(runs, out, K="K64", n_boot=200, seed=0, covered=None)
        self.assertTrue((out / "analysis.json").exists() and (out / "report.md").exists() and (out / "effects_by_language.csv").exists() and (out / "retention.csv").exists())
        data = json.loads((out / "analysis.json").read_text())
        self.assertIn("INT6", data["retention"]); self.assertNotIn("FP", data["retention"])
        text = (out / "report.md").read_text()
        for needle in ("exploratory", "activity-matched", "sham", "preserved"):
            self.assertIn(needle, text.lower())
        self.assertEqual(data["retention"]["INT4"]["macro"]["class"], "attenuated")
        sub = fa.analyse(runs, Path(tempfile.mkdtemp()), K="K64", n_boot=50, seed=0, covered=["aa"])
        self.assertEqual(sub["covered"], ["aa"])
        self.assertAlmostEqual(sub["macro"]["FP"]["point"], sub["effects"]["FP"]["aa"]["point"])
        self.assertEqual(res["n_boot"], 200)


if __name__ == "__main__":
    unittest.main()
