"""N8 tests for scripts/g0/phase1_language_level.py (decision D-022; pre-specified in handoff 004 section 4, written when only the HEADER of the per-language files had been read).
Question: does the survival of ALL active features of a language depend on the language's resource tier (proposal RQ2; paper/main.tex RQ2 row: per-language outcome, permutation test,
script-family covariate, one-sided), and is it explained by character fertility (Srivastava 2026, arXiv:2607.24276; N1)? Planted effects on the real 27-language roster."""
import csv, json, shutil, sys, tempfile, unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import phase1_analysis as an   # noqa: E402
import phase1_language_level as ll   # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
ROSTER = ROOT / "reports/g0/phase1_languages.json"
LANGS = json.loads(ROSTER.read_text())["languages"]
HEADER = "language,bits,n_tokens,perplexity,n_active_features,mean_corr,median_corr,survived_>0.9_pct,degraded_0.5_0.9_pct,damaged_<0.5_pct".split(",")


class FreedmanLaneCovariates(unittest.TestCase):
    def data(self, seed=0):
        rng = np.random.default_rng(seed); tier = np.repeat([0, 1, 2], 9); script = np.tile(np.arange(3), 9)
        fert = 3.0 - tier + rng.normal(0, 0.3, 27); y = 0.9 - 0.05 * fert + rng.normal(0, 0.005, 27)             # the tier effect runs ENTIRELY through fertility
        return y, tier, script, fert

    def test_the_default_call_is_unchanged_by_the_new_argument(self):
        y, tier, script, _ = self.data()
        a = an.freedman_lane_tier_test(y, tier, script, n_perm=200, seed=1); b = an.freedman_lane_tier_test(y, tier, script, n_perm=200, seed=1, covariates=None)
        self.assertEqual(a, b)

    def test_a_covariate_that_carries_the_tier_effect_removes_the_slope(self):
        y, tier, script, fert = self.data()
        without = an.freedman_lane_tier_test(y, tier, script, n_perm=500, seed=0); with_f = an.freedman_lane_tier_test(y, tier, script, n_perm=500, seed=0, covariates=fert)
        self.assertGreater(without["slope"], 0.02); self.assertLess(without["p"], 0.01)
        self.assertLess(abs(with_f["slope"]), 0.25 * without["slope"]); self.assertGreater(with_f["p"], 0.05)

    def test_covariates_may_be_given_as_a_matrix_and_degrees_of_freedom_account_for_them(self):
        y, tier, script, fert = self.data(); one = an.freedman_lane_tier_test(y, tier, script, n_perm=50, seed=0, covariates=fert)
        two = an.freedman_lane_tier_test(y, tier, script, n_perm=50, seed=0, covariates=np.column_stack([fert, fert ** 2]))
        self.assertEqual(one["n"], 27); self.assertNotEqual(one["t"], two["t"])                                   # a second covariate changes the fit

    def test_the_covariate_must_have_one_value_per_language(self):
        y, tier, script, fert = self.data()
        with self.assertRaisesRegex(ValueError, "26 covariate rows for 27 languages"): an.freedman_lane_tier_test(y, tier, script, n_perm=10, covariates=fert[:-1])


class LanguageTable(unittest.TestCase):
    def write(self, rows, name="per_language.csv"):
        d = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, d, ignore_errors=True); p = d / name
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=HEADER); w.writeheader(); w.writerows(rows)
        return p

    def row(self, lang, bits, surv, med=0.99, active=100):
        return {"language": lang, "bits": bits, "n_tokens": 1000, "perplexity": 20.0, "n_active_features": active, "mean_corr": 0.98, "median_corr": med, "survived_>0.9_pct": surv,
                "degraded_0.5_0.9_pct": 100 - surv, "damaged_<0.5_pct": 0.0}

    def test_values_are_read_per_language_and_width_in_the_order_of_the_roster_and_percent_becomes_a_fraction(self):
        rows = [self.row("deu_Latn", 7, 90.0, 0.95, 80), self.row("eng_Latn", 7, 99.0, 0.99, 120), self.row("deu_Latn", 6, 70.0, 0.9, 80), self.row("eng_Latn", 6, 95.0, 0.97, 120), self.row("eng_Latn", 5, 50.0, 0.8, 120), self.row("deu_Latn", 5, 40.0, 0.7, 80)]
        t = ll.language_table(self.write(rows), ["eng_Latn", "deu_Latn"], widths=(7, 6, 5))
        self.assertEqual(t["survival"][7].tolist(), [0.99, 0.90]); self.assertEqual(t["survival"][6].tolist(), [0.95, 0.70]); self.assertEqual(t["median_corr"][5].tolist(), [0.8, 0.7]); self.assertEqual(t["n_active"][7].tolist(), [120, 80])

    def test_a_missing_language_or_width_and_a_duplicate_row_are_errors(self):
        rows = [self.row("eng_Latn", 7, 99.0), self.row("deu_Latn", 7, 90.0)]
        with self.assertRaises(ValueError): ll.language_table(self.write(rows), ["eng_Latn", "fra_Latn"], widths=(7,))
        with self.assertRaises(ValueError): ll.language_table(self.write(rows), ["eng_Latn", "deu_Latn"], widths=(6,))
        with self.assertRaises(ValueError): ll.language_table(self.write(rows + [self.row("eng_Latn", 7, 98.0)]), ["eng_Latn", "deu_Latn"], widths=(7,))


class Reading(unittest.TestCase):
    """Rule fixed beforehand: (1) survival depends on tier if the INT6 slope is positive with Holm < 0.05; (2) if so and with fertility added the INT6 p is >= 0.05 (strict) the effect is
    explained by tokenization cost, otherwise it is beyond fertility; (3) if (1) fails nothing is claimed (no equivalence)."""
    def test_cases(self):
        r = ll.reading_n8
        self.assertEqual(r(0.05, 0.001, 0.30), "depends on tier; explained by tokenization cost (fertility)")
        self.assertEqual(r(0.05, 0.001, 0.0499), "depends on tier; beyond fertility")
        self.assertEqual(r(0.05, 0.001, 0.05), "depends on tier; explained by tokenization cost (fertility)")          # p exactly 0.05 is not significant
        self.assertEqual(r(0.05, 0.05, 0.001), "no language-level tier dependence detected (no equivalence claimed)")   # Holm exactly 0.05 is not significant
        self.assertEqual(r(-0.05, 0.001, 0.001), "no language-level tier dependence detected (no equivalence claimed)")  # significant but in the opposite direction: not the pre-specified positive branch
        self.assertEqual(r(0.0, 0.5, 0.5), "no language-level tier dependence detected (no equivalence claimed)")


def synth(case, seed=0, widths=(7, 6, 5)):
    """Files for run_language_level: per_language.csv, meta.json (tokens per language), stats json (characters), for the 27 roster languages."""
    rng = np.random.default_rng(seed); d = Path(tempfile.mkdtemp()); L = len(LANGS)
    tier = np.array([an.tier_score(l["tier"]) for l in LANGS]); eng = [i for i, l in enumerate(LANGS) if l["code"] == "eng_Latn"][0]
    if case == "fertility":
        fert = 1.0 + 1.5 * (2 - tier) + rng.normal(0, 0.2, L)
    else:
        fert = 1.0 + rng.uniform(0, 3, L)
    fert[eng] = 1.0
    chars = 100000; base = 25000; tokens = np.round(fert * base).astype(int)                 # character fertility multiplier = tokens / chars relative to English = fert (equal characters)
    rows = []
    for j, b in enumerate(widths):
        level = {7: 0.99, 6: 0.95, 5: 0.80}[b]
        for i, l in enumerate(LANGS):
            if case == "tier":
                s = level - 0.06 * (2 - tier[i]) + rng.normal(0, 0.008)
            elif case == "tier7only":
                s = level - (0.06 * (2 - tier[i]) if b == 7 else 0.0) + rng.normal(0, 0.008)
            elif case == "fertility":
                s = level - 0.025 * (fert[i] - 1.0) + rng.normal(0, 0.003)
            else:
                s = level + rng.normal(0, 0.02)
            s = float(np.clip(s, 0, 1)); rows.append({"language": l["code"], "bits": b, "n_tokens": 1000, "perplexity": 20.0, "n_active_features": 1000, "mean_corr": 0.98, "median_corr": 0.5 + s / 2,
                                                       "survived_>0.9_pct": 100 * s, "degraded_0.5_0.9_pct": 100 * (1 - s), "damaged_<0.5_pct": 0.0})
    with open(d / "per_language.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER); w.writeheader(); w.writerows(rows)
    (d / "meta.json").write_text(json.dumps({"stream": "eq", "regime": "mask", "corpus_report": {"devtest": [{"code": l["code"], "sentences": 1012, "tokens": int(tokens[i])} for i, l in enumerate(LANGS)]}}))
    (d / "stats.json").write_text(json.dumps({l["code"]: {"devtest": {"sentences": 1012, "chars": chars, "words": 15000 + 137 * i}} for i, l in enumerate(LANGS)}))
    return d


class EndToEnd(unittest.TestCase):
    def go(self, case, **kw):
        d = synth(case); self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        res = ll.run_language_level(d / "per_language.csv", d / "meta.json", d / "stats.json", ROSTER, d / "out", n_perm=kw.pop("n_perm", 500), seed=0, **kw); return d, res

    def test_a_planted_tier_effect_is_detected_at_both_confirmatory_widths_and_survives_adding_fertility(self):
        d, res = self.go("tier")
        for b in ("7", "6"):
            w = res["widths"][b]; self.assertGreater(w["slope"], 0.03); self.assertLess(w["holm_p"], 0.05); self.assertLess(w["p_fertility"], 0.05)
        self.assertEqual(res["reading"], "depends on tier; beyond fertility")

    def test_a_fertility_driven_effect_is_a_tier_effect_that_vanishes_once_fertility_is_a_covariate(self):
        d, res = self.go("fertility")
        w = res["widths"]["6"]; self.assertGreater(w["slope"], 0.02); self.assertLess(w["holm_p"], 0.05); self.assertGreater(w["p_fertility"], 0.05)
        self.assertEqual(res["reading"], "depends on tier; explained by tokenization cost (fertility)")

    def test_a_null_is_not_rejected_and_no_equivalence_is_claimed(self):
        d, res = self.go("null"); self.assertEqual(res["reading"], "no language-level tier dependence detected (no equivalence claimed)")
        self.assertGreater(res["widths"]["6"]["holm_p"], 0.05)

    def test_holm_is_over_the_two_confirmatory_widths_and_int5_is_descriptive(self):
        d, res = self.go("tier"); p = [res["widths"][b]["p"] for b in ("7", "6")]; self.assertEqual([res["widths"][b]["holm_p"] for b in ("7", "6")], [float(x) for x in an.holm(p)])
        self.assertNotIn("holm_p", res["widths"]["5"]); self.assertIn("slope", res["widths"]["5"])

    def test_holm_order_is_right_when_only_int7_carries_an_effect(self):
        d, res = self.go("tier7only"); p7, p6 = res["widths"]["7"]["p"], res["widths"]["6"]["p"]; self.assertLess(p7, 0.01); self.assertGreater(p6, 2 * p7)          # Holm values differ only if the larger p exceeds twice the smaller
        adj = an.holm([p7, p6]); self.assertEqual(res["widths"]["7"]["holm_p"], float(adj[0])); self.assertEqual(res["widths"]["6"]["holm_p"], float(adj[1]))
        self.assertLess(res["widths"]["7"]["holm_p"], 0.05); self.assertGreater(res["widths"]["6"]["holm_p"], 0.05); self.assertEqual(res["reading"], "no language-level tier dependence detected (no equivalence claimed)")

    def test_every_reported_number_equals_an_independent_computation_from_the_files(self):
        d, res = self.go("fertility", n_perm=300)
        tier = np.array([an.tier_score(l["tier"]) for l in LANGS]); fam = sorted({l["script_family"] for l in LANGS}); script = np.array([fam.index(l["script_family"]) for l in LANGS]); joshi = np.array([l["joshi_class"] for l in LANGS], float)
        rows = {(r["language"], int(r["bits"])): r for r in csv.DictReader(open(d / "per_language.csv"))}
        for b in (7, 6, 5):
            y = np.array([float(rows[(l["code"], b)]["survived_>0.9_pct"]) / 100 for l in LANGS]); med = np.array([float(rows[(l["code"], b)]["median_corr"]) for l in LANGS]); w = res["widths"][str(b)]
            base = an.freedman_lane_tier_test(y, tier, script, 300, 0); wf = an.freedman_lane_tier_test(y, tier, script, 300, 0, covariates=np.array(res["fertility"])); jo = an.freedman_lane_tier_test(y, joshi, script, 300, 0)
            sec = an.freedman_lane_tier_test(med, tier, script, 300, 0)
            self.assertEqual((w["slope"], w["p"], w["high_minus_low"]), (base["slope"], base["p"], base["high_minus_low"])); self.assertEqual((w["slope_fertility"], w["p_fertility"]), (wf["slope"], wf["p"]))
            self.assertEqual((w["joshi_slope"], w["joshi_p"]), (jo["slope"], jo["p"])); self.assertEqual((w["secondary_median_r"]["slope"], w["secondary_median_r"]["p"]), (sec["slope"], sec["p"]))
            for k in (0, 1, 2): self.assertAlmostEqual(w["mean_by_tier"][str(k)], float(y[tier == k].mean()), places=12)

    def test_the_confirmatory_widths_are_required(self):
        d = synth("tier"); self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        with self.assertRaises(ValueError): ll.run_language_level(d / "per_language.csv", d / "meta.json", d / "stats.json", ROSTER, d / "out", widths=(7, 5), n_perm=10)

    def test_sensitivities_and_the_descriptive_means_by_tier_are_reported_for_every_width(self):
        d, res = self.go("tier")
        for b in ("7", "6", "5"):
            w = res["widths"][b]; self.assertIn("high_minus_low", w); self.assertIn("joshi_slope", w); self.assertIn("joshi_p", w); self.assertEqual(sorted(w["mean_by_tier"]), ["0", "1", "2"])
            self.assertGreater(w["mean_by_tier"]["2"], w["mean_by_tier"]["0"]); self.assertIn("secondary_median_r", w)
        self.assertEqual(res["n_languages"], 27)

    def test_the_fertility_covariate_is_the_character_fertility_multiplier_of_the_behavioural_link(self):
        d, res = self.go("fertility"); import phase1_behavior as bh
        meta = json.loads((d / "meta.json").read_text()); tokens = {r["code"]: r["tokens"] for r in meta["corpus_report"]["devtest"]}; stats = {k: v["devtest"] for k, v in json.loads((d / "stats.json").read_text()).items()}
        self.assertTrue(np.allclose(res["fertility"], bh.fertility_multiplier(tokens, stats, [l["code"] for l in LANGS], unit="char")))

    def test_the_report_states_the_pre_specification_and_the_rule(self):
        d, res = self.go("tier"); md = (d / "out" / "language_level.md").read_text()
        for needle in ("D-022", "pre-specified", "one-sided", "Holm", "fertility", "Srivastava", "Reading rule fixed beforehand", "Reading (by the rule fixed beforehand)", "script", "Freedman"): self.assertIn(needle, md)
        self.assertTrue((d / "out" / "language_level.json").exists())

    def test_the_input_files_are_untouched(self):
        d = synth("tier"); self.addCleanup(shutil.rmtree, d, ignore_errors=True); before = {f: (d / f).read_bytes() for f in ("per_language.csv", "meta.json", "stats.json")}
        ll.run_language_level(d / "per_language.csv", d / "meta.json", d / "stats.json", ROSTER, d / "out", n_perm=50, seed=0)
        for f, b in before.items(): self.assertEqual((d / f).read_bytes(), b)


if __name__ == "__main__":
    unittest.main()
