"""N9 tests (decision D-023; pre-specified in handoff 005 section 4 before it was run): does per-language ACTIVATION SCALE carry the remaining language-level tier effect of N8, alone
and together with character fertility? Scale of a language = log10 of the MEDIAN, over its active features (full-precision firing rate in that language above 0.001, strict), of the
feature's maximum activation in that language (Duan's own predictor aggregated per language; secondary: the mean activation). Planted effects on the real 27-language roster."""
import csv, json, shutil, sys, tempfile, unittest
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_analysis as an   # noqa: E402
import phase1_language_level as ll   # noqa: E402
import test_phase1_language_level as tll   # noqa: E402   (imported as a module so its test classes are not collected a second time)

ROOT = Path(__file__).resolve().parents[3]
ROSTER = ROOT / "reports/g0/phase1_languages.json"
LANGS = tll.LANGS


class ActivationScale(unittest.TestCase):
    def surv(self):
        fr = torch.tensor([[0.5, 0.001, 0.2, 0.3], [0.0, 0.9, 0.9, 0.9], [0.5, 0.5, 0.0, 0.0]])
        mx = torch.tensor([[10.0, 1000.0, 20.0, 40.0], [5.0, 100.0, 200.0, 400.0], [10.0, 1000.0, 7.0, 7.0]]); mn = mx / 10
        return {"languages": ["a", "b", "c"], "firing_rate": fr, "max_activation": mx, "mean_activation": mn}

    def test_log10_of_the_median_over_the_active_features_of_the_languages_own_row(self):
        s = ll.language_activation_scale(self.surv())
        # a: active 0, 2, 3 (rate exactly 0.001 is not active) -> values 10, 20, 40, median 20; b: active 1, 2, 3 -> median 200; c: active 0, 1 -> median of 10 and 1000 = 505 (median first, then log10)
        self.assertTrue(np.allclose(s, [np.log10(20.0), np.log10(200.0), np.log10(505.0)]))

    def test_the_mean_activation_key_is_the_secondary_definition(self):
        self.assertTrue(np.allclose(ll.language_activation_scale(self.surv(), key="mean_activation"), [np.log10(2.0), np.log10(20.0), np.log10(50.5)]))

    def test_the_firing_rate_threshold_is_strict_in_the_dtype_of_the_stored_rates(self):
        # float32 run files: a stored 0.001 is not above 0.001 (compared as float32, as in the rest of the pipeline); float64 data: 0.001 is not active, 0.0010000001 is
        s = self.surv(); self.assertEqual(s["firing_rate"].dtype, torch.float32); self.assertAlmostEqual(float(ll.language_activation_scale(s)[0]), np.log10(20.0))
        s64 = {k: (v.double() if torch.is_tensor(v) else v) for k, v in self.surv().items()}; s64["firing_rate"][0, 1] = 0.001
        self.assertAlmostEqual(float(ll.language_activation_scale(s64)[0]), np.log10(20.0))
        s64["firing_rate"][0, 1] = 0.0010000001; self.assertAlmostEqual(float(ll.language_activation_scale(s64)[0]), np.log10(np.median([1000.0, 10.0, 20.0, 40.0])))

    def test_a_language_without_any_active_feature_is_an_error(self):
        s = self.surv(); s["firing_rate"][1] = 0.0
        with self.assertRaisesRegex(ValueError, "no active feature"): ll.language_activation_scale(s)

    def test_the_result_follows_the_order_of_the_requested_codes(self):
        s = ll.language_activation_scale(self.surv(), codes=["c", "a", "b"]); self.assertTrue(np.allclose(s, [np.log10(505.0), np.log10(20.0), np.log10(200.0)]))
        with self.assertRaisesRegex(ValueError, "not in the survival file"): ll.language_activation_scale(self.surv(), codes=["a", "zzz"])


class Reading(unittest.TestCase):
    """Rule fixed beforehand: with both covariates the INT6 slope p >= 0.05 (strict) -> explained jointly by tokenization cost and activation scale; p < 0.05 -> a tier effect remains beyond
    fertility and activation scale; scale alone with p >= 0.05 -> activation scale alone explains it. Only meaningful if N8 found a tier dependence at INT6."""
    def test_cases(self):
        r = ll.reading_n9; dep = ll.EXPLAINED
        self.assertEqual(r(dep, 0.30, 0.30), {"joint": "explained jointly by tokenization cost and activation scale", "scale_alone_explains": True})
        self.assertEqual(r(dep, 0.001, 0.30), {"joint": "explained jointly by tokenization cost and activation scale", "scale_alone_explains": False})
        self.assertEqual(r(dep, 0.30, 0.001), {"joint": "a tier effect remains beyond fertility and activation scale", "scale_alone_explains": True})
        self.assertEqual(r(ll.BEYOND, 0.001, 0.001), {"joint": "a tier effect remains beyond fertility and activation scale", "scale_alone_explains": False})
        self.assertEqual(r(dep, 0.05, 0.05)["joint"], "explained jointly by tokenization cost and activation scale"); self.assertTrue(r(dep, 0.05, 0.05)["scale_alone_explains"])    # p exactly 0.05 is not significant
        self.assertEqual(r(dep, 0.0499, 0.0499)["joint"], "a tier effect remains beyond fertility and activation scale")
        self.assertEqual(r(ll.NO, 0.3, 0.3), {"joint": "not applicable: no tier dependence at INT6 to explain", "scale_alone_explains": None})


def synth9(case, seed=0, F=40, widths=(7, 6, 5)):
    """per_language.csv / meta.json / stats.json (as tll.synth) plus survival.pt with per-language activation scales. case 'scale': the tier effect on survival runs through activation scale only;
    'fertility': through fertility only; 'neither': a direct tier effect independent of both."""
    rng = np.random.default_rng(seed); d = Path(tempfile.mkdtemp()); L = len(LANGS); tier = np.array([an.tier_score(l["tier"]) for l in LANGS]); eng = [i for i, l in enumerate(LANGS) if l["code"] == "eng_Latn"][0]
    fert = (1.0 + 1.5 * (2 - tier) + rng.normal(0, 0.2, L)) if case == "fertility" else 1.0 + rng.uniform(0, 3, L); fert[eng] = 1.0
    scale = (1.0 + 0.15 * tier + rng.normal(0, 0.03, L)) if case == "scale" else 1.0 + rng.uniform(0, 0.3, L)
    tokens = np.round(fert * 25000).astype(int); rows = []
    for b in widths:
        level = {7: 0.99, 6: 0.95, 5: 0.80}[b]
        for i, l in enumerate(LANGS):
            if case == "scale": s = level + 0.25 * (scale[i] - 1.15) + rng.normal(0, 0.004)             # at INT6 the range is 0.91..0.99: no clipping at 1
            elif case == "fertility": s = level - 0.025 * (fert[i] - 1.0) + rng.normal(0, 0.003)
            else: s = level - 0.06 * (2 - tier[i]) + rng.normal(0, 0.008)
            s = float(np.clip(s, 0, 1)); rows.append({"language": l["code"], "bits": b, "n_tokens": 1000, "perplexity": 20.0, "n_active_features": F, "mean_corr": 0.98, "median_corr": 0.5 + s / 2,
                                                       "survived_>0.9_pct": 100 * s, "degraded_0.5_0.9_pct": 100 * (1 - s), "damaged_<0.5_pct": 0.0})
    with open(d / "per_language.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=tll.HEADER); w.writeheader(); w.writerows(rows)
    (d / "meta.json").write_text(json.dumps({"stream": "eq", "regime": "mask", "corpus_report": {"devtest": [{"code": l["code"], "sentences": 1012, "tokens": int(tokens[i])} for i, l in enumerate(LANGS)]}}))
    (d / "stats.json").write_text(json.dumps({l["code"]: {"devtest": {"sentences": 1012, "chars": 100000, "words": 15000 + 137 * i}} for i, l in enumerate(LANGS)}))
    mx = 10 ** (scale[:, None] + rng.normal(0, 0.05, (L, F))); torch.save({"languages": [l["code"] for l in LANGS], "firing_rate": torch.full((L, F), 0.01), "max_activation": torch.tensor(mx, dtype=torch.float32),
                                                                            "mean_activation": torch.tensor(10 ** (rng.uniform(0.5, 1.5, (L, 1)) + rng.normal(0, 0.05, (L, F))), dtype=torch.float32)}, d / "survival.pt")       # its own per-language scale: not a monotone transform of the maximum
    return d, tier, scale


class EndToEnd(unittest.TestCase):
    def go(self, case, **kw):
        d, tier, scale = synth9(case); self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        res = ll.run_language_level(d / "per_language.csv", d / "meta.json", d / "stats.json", ROSTER, d / "out", n_perm=kw.pop("n_perm", 500), seed=0, survival_pt=d / "survival.pt", **kw); return d, res, tier, scale

    def test_an_effect_that_runs_through_activation_scale_vanishes_with_the_scale_covariate_and_not_with_fertility_alone(self):
        d, res, tier, scale = self.go("scale"); w = res["widths"]["6"]; a = res["activation_scale"]["widths"]["6"]
        self.assertGreater(w["slope"], 0.025); self.assertLess(w["p"], 0.01); self.assertLess(w["p_fertility"], 0.05)                       # fertility is unrelated to the planted mechanism
        self.assertGreater(a["p_scale"], 0.05); self.assertLess(abs(a["slope_scale"]), 0.3 * w["slope"]); self.assertGreater(a["p_both"], 0.05)
        self.assertEqual(res["activation_scale"]["reading"]["joint"], "explained jointly by tokenization cost and activation scale"); self.assertTrue(res["activation_scale"]["reading"]["scale_alone_explains"])

    def test_an_effect_that_runs_through_fertility_vanishes_with_both_and_not_with_the_scale_covariate_alone(self):
        d, res, tier, scale = self.go("fertility"); a = res["activation_scale"]["widths"]["6"]
        self.assertLess(a["p_scale"], 0.05); self.assertGreater(a["p_both"], 0.05)
        self.assertEqual(res["activation_scale"]["reading"], {"joint": "explained jointly by tokenization cost and activation scale", "scale_alone_explains": False})

    def test_a_direct_tier_effect_is_not_removed_by_either_covariate(self):
        d, res, tier, scale = self.go("neither"); a = res["activation_scale"]["widths"]["6"]
        self.assertLess(a["p_scale"], 0.05); self.assertLess(a["p_both"], 0.05)
        self.assertEqual(res["activation_scale"]["reading"], {"joint": "a tier effect remains beyond fertility and activation scale", "scale_alone_explains": False})

    def test_the_scale_vector_and_the_spearman_descriptives_equal_independent_computations(self):
        d, res, tier, scale = self.go("scale"); sv = torch.load(d / "survival.pt"); m = np.log10(np.median(sv["max_activation"].numpy(), axis=1))
        self.assertTrue(np.allclose(res["activation_scale"]["scale"], m)); self.assertTrue(np.allclose(res["activation_scale"]["scale_secondary"], np.log10(np.median(sv["mean_activation"].numpy(), axis=1))))
        self.assertAlmostEqual(res["activation_scale"]["spearman_tier_scale"], spearmanr(tier, m).statistic, places=12)
        rows = {(r["language"], int(r["bits"])): r for r in csv.DictReader(open(d / "per_language.csv"))}
        for b in (7, 6, 5):
            y = np.array([float(rows[(l["code"], b)]["survived_>0.9_pct"]) / 100 for l in LANGS]); self.assertAlmostEqual(res["activation_scale"]["widths"][str(b)]["spearman_scale_survival"], spearmanr(m, y).statistic, places=12)

    def test_every_covariate_test_equals_an_independent_freedman_lane_computation(self):
        d, res, tier, scale = self.go("scale", n_perm=300); fam = sorted({l["script_family"] for l in LANGS}); script = np.array([fam.index(l["script_family"]) for l in LANGS])
        m = np.array(res["activation_scale"]["scale"]); m2 = np.array(res["activation_scale"]["scale_secondary"]); fert = np.array(res["fertility"])
        rows = {(r["language"], int(r["bits"])): r for r in csv.DictReader(open(d / "per_language.csv"))}
        for b in (7, 6, 5):
            y = np.array([float(rows[(l["code"], b)]["survived_>0.9_pct"]) / 100 for l in LANGS]); a = res["activation_scale"]["widths"][str(b)]
            A = an.freedman_lane_tier_test(y, tier, script, 300, 0, covariates=m); B = an.freedman_lane_tier_test(y, tier, script, 300, 0, covariates=np.column_stack([fert, m]))
            A2 = an.freedman_lane_tier_test(y, tier, script, 300, 0, covariates=m2); B2 = an.freedman_lane_tier_test(y, tier, script, 300, 0, covariates=np.column_stack([fert, m2]))
            self.assertEqual((a["slope_scale"], a["p_scale"], a["slope_both"], a["p_both"]), (A["slope"], A["p"], B["slope"], B["p"])); self.assertEqual((a["secondary"]["slope_scale"], a["secondary"]["p_both"]), (A2["slope"], B2["p"]))

    def test_the_survival_file_may_list_the_languages_in_another_order(self):
        d, tier, scale = synth9("scale"); self.addCleanup(shutil.rmtree, d, ignore_errors=True); sv = torch.load(d / "survival.pt"); perm = np.random.default_rng(3).permutation(len(LANGS))
        a = ll.run_language_level(d / "per_language.csv", d / "meta.json", d / "stats.json", ROSTER, d / "a", n_perm=100, seed=0, survival_pt=d / "survival.pt")
        shuffled = {"languages": [sv["languages"][i] for i in perm], **{k: sv[k][torch.tensor(perm)] for k in ("firing_rate", "max_activation", "mean_activation")}}; torch.save(shuffled, d / "shuffled.pt")
        b = ll.run_language_level(d / "per_language.csv", d / "meta.json", d / "stats.json", ROSTER, d / "b", n_perm=100, seed=0, survival_pt=d / "shuffled.pt")
        self.assertEqual(a["activation_scale"], b["activation_scale"])

    def test_without_the_survival_file_the_n8_output_is_exactly_what_it_was(self):
        d, tier, scale = synth9("scale"); self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        plain = ll.run_language_level(d / "per_language.csv", d / "meta.json", d / "stats.json", ROSTER, d / "plain", n_perm=100, seed=0)
        full = ll.run_language_level(d / "per_language.csv", d / "meta.json", d / "stats.json", ROSTER, d / "full", n_perm=100, seed=0, survival_pt=d / "survival.pt")
        self.assertNotIn("activation_scale", plain); self.assertEqual({k: v for k, v in full.items() if k != "activation_scale"}, plain)
        md_plain = (d / "plain" / "language_level.md").read_text(); md_full = (d / "full" / "language_level.md").read_text(); self.assertTrue(md_full.startswith(md_plain.rstrip("\n"))); self.assertNotIn("activation scale", md_plain)

    def test_the_report_states_the_rule_and_the_covariate_definitions(self):
        d, res, tier, scale = self.go("scale"); md = (d / "out" / "language_level.md").read_text()
        for needle in ("D-023", "log10", "median", "activation scale", "p exactly 0.05", "jointly", "issues.md", "C6"): self.assertIn(needle, md)


if __name__ == "__main__":
    unittest.main()
