"""N1 tests for scripts/g0/phase1_behavior.py: the behavioural link of the team's evaluation-plan table (paper/main.tex, row "Behavioral Link"): per-language
selective-feature damage fraction against the per-language next-token loss increase, Spearman rho and multivariable OLS with overall feature damage and token
fertility as covariates. Brute-force references and planted effects on synthetic data. Definitions: D-018."""
import json, shutil, sys, tempfile, unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import phase1_behavior as bh   # noqa: E402


def toy():
    # 3 languages x 8 features; feature f is selective with peak language peak[f]
    corr = np.array([[0.95, 0.90, 0.50, 0.30, 0.99, 0.40, 0.97, 0.20],
                     [0.91, 0.60, 0.89, 0.99, 0.10, 0.95, 0.30, 0.92],
                     [0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]])
    fire = np.array([[0.5, 0.5, 0.5, 0.5, 0.001, 0.5, 0.5, 0.0],        # lang 0: feature 4 is exactly at the threshold (not active), feature 7 never fires
                     [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                     [0.0] * 8])                                          # lang 2: nothing is active
    selective = np.array([1, 1, 1, 1, 1, 0, 1, 1], bool)
    peak = np.array([0, 0, 1, 1, 0, 2, 0, 0])
    return corr, fire, selective, peak


class Damage(unittest.TestCase):
    def test_selective_and_overall_damage_fractions_by_hand(self):
        corr, fire, sel, peak = toy(); d = bh.damage_fractions(corr, fire, sel, peak)
        # language 0, selective at peak 0 and active: features 0 (r .95 ok), 1 (r .90: NOT > .9 -> damaged), 6 (.97 ok); 4 (rate == .001) and 7 (rate 0) are not evaluable
        self.assertEqual(d["n_selective"][0], 3); self.assertAlmostEqual(d["selective"][0], 1 / 3)
        # language 1, selective at peak 1: features 2 (.89 damaged), 3 (.99 ok)
        self.assertEqual(d["n_selective"][1], 2); self.assertAlmostEqual(d["selective"][1], 1 / 2)
        # language 2 has no evaluable selective feature (feature 5 is a control, nothing active)
        self.assertEqual(d["n_selective"][2], 0); self.assertTrue(np.isnan(d["selective"][2])); self.assertTrue(np.isnan(d["overall"][2]))
        # overall = every feature active in the language: language 0 has 6 active (0,1,2,3,5,6): r .95 ok, .90 d, .50 d, .30 d, .40 d, .97 ok -> 4/6
        self.assertEqual(d["n_active"][0], 6); self.assertAlmostEqual(d["overall"][0], 4 / 6)
        self.assertEqual(d["n_active"][1], 8); self.assertAlmostEqual(d["overall"][1], 4 / 8)      # r: .91 ok, .60 d, .89 d, .99 ok, .10 d, .95 ok, .30 d, .92 ok -> 4 damaged

    def test_damaged_kind_uses_their_strict_below_half_rule(self):
        corr, fire, sel, peak = toy(); d = bh.damage_fractions(corr, fire, sel, peak, kind="damaged")
        self.assertAlmostEqual(d["selective"][0], 0.0)                 # selective evaluable at language 0: r .95, .90, .97, none below .5
        self.assertAlmostEqual(d["selective"][1], 0.0)                 # r .89 and .99: none below .5
        self.assertAlmostEqual(d["overall"][0], 2 / 6)                 # r .30 and .40 are below .5; r == .50 exactly is not
        self.assertAlmostEqual(d["overall"][1], 2 / 8)                 # r .10 and .30

    def test_a_nan_correlation_among_evaluated_features_is_an_error_not_a_damaged_feature(self):
        corr, fire, sel, peak = toy(); corr[0, 0] = np.nan
        with self.assertRaises(ValueError): bh.damage_fractions(corr, fire, sel, peak)
        corr, fire, sel, peak = toy(); corr[0, 7] = np.nan            # feature 7 is inactive at language 0: its r is never read
        bh.damage_fractions(corr, fire, sel, peak)


class DeltaLoss(unittest.TestCase):
    def test_mean_next_token_loss_increase_per_language(self):
        ls = {"FP": [10.0, 20.0], 6: [12.0, 21.0]}; ln = {"FP": [10, 10], 6: [10, 10]}
        self.assertTrue(np.allclose(bh.delta_loss(ls, ln, 6), [0.2, 0.1]))

    def test_languages_with_different_token_counts_are_compared_per_token_not_by_sums(self):
        ls = {"FP": [10.0, 40.0], 4: [30.0, 100.0]}; ln = {"FP": [10, 20], 4: [10, 20]}
        self.assertTrue(np.allclose(bh.delta_loss(ls, ln, 4), [2.0, 3.0]))          # sums would give 20 and 60

    def test_refuses_mismatched_token_counts_between_conditions(self):
        with self.assertRaises(ValueError): bh.delta_loss({"FP": [10.0], 6: [11.0]}, {"FP": [10], 6: [11]}, 6)


class Fertility(unittest.TestCase):
    STATS = {"eng_Latn": {"sentences": 10, "chars": 400, "words": 80}, "kan_Knda": {"sentences": 10, "chars": 500, "words": 50}, "zho_Hans": {"sentences": 10, "chars": 120, "words": 10}}
    TOK = {"eng_Latn": 100, "kan_Knda": 400, "zho_Hans": 90}

    def test_character_fertility_multiplier_is_the_ratio_to_english_of_tokens_per_character(self):
        f = bh.fertility_multiplier(self.TOK, self.STATS, ["eng_Latn", "kan_Knda", "zho_Hans"])
        self.assertTrue(np.allclose(f, [1.0, (400 / 500) / (100 / 400), (90 / 120) / (100 / 400)]))

    def test_word_fertility_multiplier_for_space_delimited_languages(self):
        f = bh.fertility_multiplier(self.TOK, self.STATS, ["eng_Latn", "kan_Knda"], unit="word")
        self.assertTrue(np.allclose(f, [1.0, (400 / 50) / (100 / 80)]))

    def test_sentence_counts_must_match_because_the_text_is_parallel(self):
        bad = {**self.STATS, "kan_Knda": {"sentences": 9, "chars": 500, "words": 50}}
        with self.assertRaises(ValueError): bh.fertility_multiplier(self.TOK, bad, ["eng_Latn", "kan_Knda"])

    def test_unknown_unit_is_refused(self):
        with self.assertRaises(ValueError): bh.fertility_multiplier(self.TOK, self.STATS, ["eng_Latn"], unit="byte")


def planted(effect, n=27, seed=0, noise=0.01):
    rng = np.random.default_rng(seed)
    dmg = rng.uniform(0.0, 0.4, n); overall = 0.5 * dmg + rng.uniform(0, 0.2, n); fert = rng.uniform(1.0, 5.0, n)
    delta = effect * dmg + 0.0 * overall + 0.002 * fert + rng.normal(0, noise, n)
    return delta, dmg, overall, fert


class Link(unittest.TestCase):
    def test_spearman_equals_the_rank_correlation_computed_by_hand_and_one_sided_p_is_half_the_two_sided_for_a_positive_rho(self):
        d, s, o, f = planted(0.5); r = bh.behavioural_link(d, s, o, f)
        rank = lambda x: np.argsort(np.argsort(x)).astype(float)
        self.assertAlmostEqual(r["spearman"]["rho"], np.corrcoef(rank(s), rank(d))[0, 1], places=10)
        self.assertGreater(r["spearman"]["rho"], 0.5); self.assertAlmostEqual(r["spearman"]["p_one_sided"], r["spearman"]["p_two_sided"] / 2, places=10)

    def test_a_planted_positive_effect_is_found_and_a_planted_zero_effect_is_not(self):
        yes = bh.behavioural_link(*planted(0.5)); self.assertLess(yes["spearman"]["p_one_sided"], 1e-3)
        no = bh.behavioural_link(*planted(0.0, noise=0.05, seed=3)); self.assertGreater(no["spearman"]["p_one_sided"], 0.05)

    def test_a_negative_association_gives_a_one_sided_p_near_one(self):
        r = bh.behavioural_link(*planted(-0.5)); self.assertLess(r["spearman"]["rho"], 0); self.assertGreater(r["spearman"]["p_one_sided"], 0.99)

    def test_ols_matches_least_squares_by_hand_and_recovers_the_planted_slope_within_three_se(self):
        d, s, o, f = planted(0.5, noise=0.02); r = bh.behavioural_link(d, s, o, f)
        X = np.column_stack([np.ones(len(d)), s, o, f]); beta = np.linalg.lstsq(X, d, rcond=None)[0]
        self.assertTrue(np.allclose([r["ols"]["coef"][k] for k in ("intercept", "selective_damage", "overall_damage", "fertility")], beta, atol=1e-8))
        self.assertLess(abs(r["ols"]["coef"]["selective_damage"] - 0.5), 3 * r["ols"]["se"]["selective_damage"])
        self.assertEqual(r["ols"]["n"], 27); self.assertEqual(r["ols"]["df_resid"], 27 - 4)

    def test_variance_inflation_factors_are_reported_for_the_three_predictors(self):
        r = bh.behavioural_link(*planted(0.5)); v = r["ols"]["vif"]
        self.assertEqual(set(v), {"selective_damage", "overall_damage", "fertility"}); self.assertTrue(all(x >= 1.0 for x in v.values()))
        s = np.linspace(0, 1, 27); o = s * 2.0 + 1e-3 * np.random.default_rng(0).normal(size=27)       # nearly collinear predictors: large VIF
        r2 = bh.behavioural_link(0.3 * s + 0.01 * np.random.default_rng(1).normal(size=27), s, o, np.random.default_rng(2).uniform(1, 3, 27))
        self.assertGreater(r2["ols"]["vif"]["selective_damage"], 100)

    def test_languages_with_an_undefined_damage_fraction_are_dropped_and_listed(self):
        d, s, o, f = planted(0.5); s = s.copy(); s[4] = np.nan; names = [f"l{i}" for i in range(27)]
        r = bh.behavioural_link(d, s, o, f, names=names)
        self.assertEqual(r["n_languages"], 26); self.assertEqual(r["dropped"], ["l4"]); self.assertEqual(r["ols"]["n"], 26)
        keep = np.arange(27) != 4; ref = bh.behavioural_link(d[keep], s[keep], o[keep], f[keep]); self.assertAlmostEqual(r["spearman"]["rho"], ref["spearman"]["rho"], places=12)

    def test_constant_damage_or_loss_is_refused_because_rank_correlation_is_undefined(self):
        d, s, o, f = planted(0.5)
        with self.assertRaises(ValueError): bh.behavioural_link(d, np.full(27, 0.1), o, f)
        with self.assertRaises(ValueError): bh.behavioural_link(np.full(27, 0.1), s, o, f)

    def test_too_few_languages_is_refused(self):
        d, s, o, f = planted(0.5, n=6)
        with self.assertRaises(ValueError): bh.behavioural_link(d, s, o, f)


class Run(unittest.TestCase):
    """End to end on the files phase1_run.save_outputs writes (survival.pt, selectivity.pt, meta.json) plus the FLORES text statistics."""
    def build(self, n_lang=12, F=40, seed=0, english=True):
        import torch
        rng = np.random.default_rng(seed); names = [f"l{i}" for i in range(n_lang)]; names[0] = "eng_Latn" if english else "xxx_Latn"
        corr = {b: torch.tensor(rng.uniform(0.3, 1.0, (n_lang, F)), dtype=torch.float32) for b in (7, 6, 5)}
        fire = torch.tensor(rng.uniform(0.002, 0.01, (n_lang, F)), dtype=torch.float32)
        fp_s, fp_n = [10.0 * (i + 1) for i in range(n_lang)], [100] * n_lang
        loss_sum = {"FP": fp_s, **{b: [x + 0.5 * i for i, x in enumerate(fp_s)] for b in (7, 6, 5)}}; loss_n = {k: list(fp_n) for k in loss_sum}
        peak = rng.integers(0, n_lang, F); selective = rng.uniform(size=F) < 0.8
        surv = {"languages": names, "bits": [7, 6, 5], "corr": corr, "firing_rate": fire, "loss_sum": loss_sum, "loss_n": loss_n, "n": [50] * n_lang}
        selp = {"selective": torch.tensor(selective), "peak": torch.tensor(peak)}
        meta = {"stream": "eq", "regime": "mask", "corpus_report": {"devtest": [{"code": c, "sentences": 10, "tokens": 100 + 5 * i} for i, c in enumerate(names)]}}
        stats = {c: {"devtest": {"sentences": 10, "chars": 400 + i, "words": 80}} for i, c in enumerate(names)}
        return surv, selp, meta, stats

    def test_run_link_reads_the_run_files_and_reports_every_width_with_both_damage_kinds(self):
        import torch
        surv, selp, meta, stats = self.build(); out = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, out, ignore_errors=True)
        torch.save(surv, out / "survival.pt"); torch.save(selp, out / "selectivity.pt"); (out / "meta.json").write_text(json.dumps(meta)); (out / "stats.json").write_text(json.dumps(stats))
        res = bh.run_link(out, out / "stats.json", out / "res", widths=(7, 6, 5))
        self.assertEqual(res["stream"], "eq"); self.assertEqual(set(res["widths"]), {"7", "6", "5"})
        for w in res["widths"].values():
            for kind in ("survival", "damaged"): self.assertIn("spearman", w[kind]); self.assertIn("ols", w[kind])
        self.assertTrue((out / "res" / "behavior.json").exists()); self.assertTrue((out / "res" / "behavior.md").exists())
        # the delta-loss values reaching the test are the per-token mean increases (0.5 * i / 100)
        self.assertTrue(np.allclose(res["widths"]["6"]["delta_loss"], [0.5 * i / 100 for i in range(12)]))

    def test_a_width_where_nothing_is_damaged_is_recorded_as_undefined_and_does_not_abort_the_other_cells(self):
        import torch
        surv, selp, meta, stats = self.build(); surv["corr"][7] = torch.full_like(surv["corr"][7], 0.8)       # at INT7 no feature is below 0.5: the 'damaged' fraction is 0 in every language
        out = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, out, ignore_errors=True)
        torch.save(surv, out / "survival.pt"); torch.save(selp, out / "selectivity.pt"); (out / "meta.json").write_text(json.dumps(meta)); (out / "stats.json").write_text(json.dumps(stats))
        res = bh.run_link(out, out / "stats.json", out / "res", widths=(7, 6))
        self.assertIn("undefined", res["widths"]["7"]["damaged"]); self.assertIn("constant", res["widths"]["7"]["damaged"]["undefined"]); self.assertNotIn("spearman", res["widths"]["7"]["damaged"])
        self.assertIn("undefined", res["widths"]["7"]["survival"])                                              # all r = 0.8 (not above 0.9): the survival damage is 1.0 everywhere, also constant
        self.assertIn("spearman", res["widths"]["6"]["damaged"]); self.assertIn("spearman", res["widths"]["6"]["survival"])
        md = (out / "res" / "behavior.md").read_text(); self.assertIn("undefined", md)

    def test_the_per_language_damage_values_in_the_result_are_the_ones_computed_independently(self):
        import torch
        surv, selp, meta, stats = self.build(); out = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, out, ignore_errors=True)
        torch.save(surv, out / "survival.pt"); torch.save(selp, out / "selectivity.pt"); (out / "meta.json").write_text(json.dumps(meta)); (out / "stats.json").write_text(json.dumps(stats))
        res = bh.run_link(out, out / "stats.json", out / "res", widths=(6,))
        for kind in ("survival", "damaged"):
            ref = bh.damage_fractions(surv["corr"][6].numpy(), surv["firing_rate"].numpy(), selp["selective"].numpy(), selp["peak"].numpy(), kind=kind)
            got = res["widths"]["6"][kind]["per_language"]
            for key in ("selective", "overall"):
                self.assertTrue(np.allclose([np.nan if v is None else v for v in got[key]], ref[key], equal_nan=True))
            self.assertEqual(got["n_selective"], ref["n_selective"].tolist()); self.assertEqual(got["n_active"], ref["n_active"].tolist())

    def test_a_nan_correlation_is_a_data_error_that_aborts_the_run_and_is_not_recorded_as_an_undefined_cell(self):
        import torch
        surv, selp, meta, stats = self.build(); surv["corr"][6][0, :] = float("nan")            # every feature is active (firing rate 0.002-0.01), so language 0 reads NaN values
        out = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, out, ignore_errors=True)
        torch.save(surv, out / "survival.pt"); torch.save(selp, out / "selectivity.pt"); (out / "meta.json").write_text(json.dumps(meta)); (out / "stats.json").write_text(json.dumps(stats))
        with self.assertRaises(ValueError) as cm: bh.run_link(out, out / "stats.json", out / "res", widths=(6,))
        self.assertNotIsInstance(cm.exception, bh.UndefinedLink)

    def test_the_command_line_prints_a_summary_even_when_a_cell_is_undefined(self):
        import contextlib, io, torch
        surv, selp, meta, stats = self.build(); surv["corr"][7] = torch.full_like(surv["corr"][7], 0.8)
        out = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, out, ignore_errors=True)
        torch.save(surv, out / "survival.pt"); torch.save(selp, out / "selectivity.pt"); (out / "meta.json").write_text(json.dumps(meta)); (out / "stats.json").write_text(json.dumps(stats))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf): bh.main(["--run", str(out), "--stats", str(out / "stats.json"), "--out", str(out / "res"), "--widths", "7", "6"])
        text = buf.getvalue(); self.assertIn("INT7", text); self.assertIn("undefined", text); self.assertIn("INT6", text); self.assertIn("rho", text)

    def test_english_is_required_as_the_fertility_reference(self):
        import torch
        surv, selp, meta, stats = self.build(english=False); out = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, out, ignore_errors=True)
        torch.save(surv, out / "survival.pt"); torch.save(selp, out / "selectivity.pt"); (out / "meta.json").write_text(json.dumps(meta)); (out / "stats.json").write_text(json.dumps(stats))
        with self.assertRaises(ValueError): bh.run_link(out, out / "stats.json", out / "res")

    def test_the_report_states_the_definitions_and_their_sources(self):
        import torch
        surv, selp, meta, stats = self.build(); out = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, out, ignore_errors=True)
        torch.save(surv, out / "survival.pt"); torch.save(selp, out / "selectivity.pt"); (out / "meta.json").write_text(json.dumps(meta)); (out / "stats.json").write_text(json.dumps(stats))
        bh.run_link(out, out / "stats.json", out / "res"); md = (out / "res" / "behavior.md").read_text()
        for needle in ("D-018", "Srivastava", "arXiv:2607.24276", "Spearman", "one-sided", "character"): self.assertIn(needle, md)


if __name__ == "__main__":
    unittest.main()
