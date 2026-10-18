"""N7 tests for scripts/g0/phase1_mixed_variants.py (decision D-021; pre-specified in handoff 003 section 4 before it was run). Question (A-034 (c)): the exploratory mixed-effects
logistic model has a POSITIVE S~ coefficient while the matched contrast is negative (N2), the magnitude-matched contrast is not negative (N6) and the threshold-free within-stratum
Spearman is about zero. The SAME model (phase1_analysis.mixed_logit) is refitted on the SAME data in variants: (i) unchanged (must reproduce analysis.json), (ii) without the tier terms,
(iii) with log10 maximum activation added, (iv) restricted to the matched sets of N2 and N6 at one threshold."""
import json, shutil, sys, tempfile, unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_analysis as an   # noqa: E402
import phase1_magnitude as mg   # noqa: E402
import phase1_mixed_variants as mv   # noqa: E402
import phase1_threshold as th   # noqa: E402
import test_phase1_magnitude as tm   # noqa: E402   (imported as a module so its test classes are not collected a second time)

ROOT = Path(__file__).resolve().parents[3]
ROSTER = ROOT / "reports/g0/phase1_languages.json"


def coef(c, p, se=0.1):
    return {"coef": c, "p": p, "se": se}


def variants(no_tier, log_m, n2, n6):
    return {"no_tier_all": {"S_rank": coef(*no_tier)}, "log_m_all": {"S_rank": coef(*log_m)}, "full_matched_N2": {"S_rank": coef(*n2)}, "full_matched_N6": {"S_rank": coef(*n6)}}


class Reading(unittest.TestCase):
    """The rule fixed beforehand: if the S~ coefficient is <= 0 or not significant in variant (ii) or (iii), the conflict is explained by that term (report which); if it stays significantly
    positive in all variants it is UNRESOLVED and no further variants are tried. The matched-set variants are part of 'all variants'; a pattern outside the rule is labelled as such."""
    def test_explained_by_the_term_whose_variant_loses_the_positive_coefficient(self):
        r = mv.reading_n7(variants((-0.02, 0.6), (0.9, 0.001), (1.0, 0.01), (1.0, 0.01))); self.assertEqual(r["label"], "explained by the tier terms"); self.assertEqual(r["terms"], ["no_tier_all"])
        r = mv.reading_n7(variants((0.8, 0.001), (0.3, 0.2), (1.0, 0.01), (1.0, 0.01))); self.assertEqual(r["label"], "explained by activation magnitude"); self.assertEqual(r["terms"], ["log_m_all"])
        r = mv.reading_n7(variants((-0.1, 0.3), (-0.2, 0.01), (1.0, 0.01), (1.0, 0.01))); self.assertEqual(r["label"], "explained by the tier terms and activation magnitude")

    def test_a_negative_coefficient_that_is_significant_still_counts_as_not_positive(self):
        self.assertEqual(mv.reading_n7(variants((-0.9, 0.0001), (0.9, 0.001), (1.0, 0.01), (1.0, 0.01)))["terms"], ["no_tier_all"])

    def test_significance_boundary_is_strict_like_everywhere_else(self):
        self.assertEqual(mv.reading_n7(variants((0.8, 0.05), (0.9, 0.001), (1.0, 0.01), (1.0, 0.01)))["terms"], ["no_tier_all"])                  # p exactly 0.05 is not significant
        self.assertEqual(mv.reading_n7(variants((0.8, 0.0499), (0.9, 0.001), (1.0, 0.01), (1.0, 0.01)))["label"], "unresolved")

    def test_unresolved_when_every_variant_stays_significantly_positive(self):
        r = mv.reading_n7(variants((0.8, 0.001), (0.9, 0.001), (1.0, 0.01), (1.0, 0.01))); self.assertEqual(r["label"], "unresolved"); self.assertEqual(r["terms"], [])

    def test_positive_in_the_full_sample_but_not_in_the_matched_sets_is_labelled_as_outside_the_rule(self):
        r = mv.reading_n7(variants((0.8, 0.001), (0.9, 0.001), (-0.5, 0.01), (1.0, 0.01)))
        self.assertEqual(r["label"], "unresolved in the full sample; matched sets differ (outside the pre-specified rule)")


class MatchedFeatures(unittest.TestCase):
    def selp(self):
        g = torch.Generator().manual_seed(0); F = 300
        return {"S": torch.rand(F, generator=g) * 8, "supported": torch.rand(F, generator=g) < 0.8, "stable_count": torch.randint(60, 101, (F,), generator=g), "min_stable_needed": 80,
                "strata": torch.randint(0, 5, (F,), generator=g), "null_cutoff": 1.8}

    def test_the_matched_set_is_the_union_of_the_selective_and_the_control_features_of_the_threshold(self):
        s = self.selp(); alt = th.alternative_selection(s, 5.0, 0); ids = mv.matched_features(s, 5.0, 0)
        self.assertEqual(ids.tolist(), sorted(set(alt["selective"].nonzero().squeeze(1).tolist()) | set(alt["controls"].tolist())))

    def test_a_replacement_of_the_strata_changes_only_the_controls(self):
        s = self.selp(); alt_strata = (s["strata"] * 3 + torch.randint(0, 3, s["strata"].shape, generator=torch.Generator().manual_seed(1)))
        a = mv.matched_features(s, 5.0, 0); b = mv.matched_features(s, 5.0, 0, strata=alt_strata)
        sel = th.alternative_selection(s, 5.0, 0)["selective"].nonzero().squeeze(1).tolist()
        self.assertTrue(set(sel) <= set(a.tolist()) and set(sel) <= set(b.tolist()))                    # the selective set does not depend on the strata
        self.assertNotEqual(a.tolist(), b.tolist())


class EndToEnd(unittest.TestCase):
    def setUp(self):
        self.helper, self.out = tm.build_run("magnitude"); self.addCleanup(self.helper.doCleanups)
        self.base = an.run_analysis(self.out, ROSTER, Path(self.out) / "base", n_perm=30, n_boot=10, seed=0)
        self.res = mv.run_variants(self.out, ROSTER, Path(self.out) / "mv", tau=3.0, seed=0, analysis_json=Path(self.out) / "base" / "analysis.json")

    def test_variant_i_reproduces_the_saved_model_exactly(self):
        full = self.res["variants"]["full_all"]; saved = self.base["exploratory"]["mixed_logit"]["mixed"]
        self.assertTrue(self.res["reproduces_analysis_json"]); self.assertEqual(full["names"], saved["names"]); self.assertEqual(full["coef"], [float(x) for x in saved["coef"]])

    def test_all_five_variants_are_reported_with_their_formulas_and_the_covariates(self):
        v = self.res["variants"]; self.assertEqual(list(v), ["full_all", "no_tier_all", "log_m_all", "full_matched_N2", "full_matched_N6"])
        self.assertNotIn("tier", v["no_tier_all"]["names"]); self.assertIn("log10_m", v["log_m_all"]["names"]); self.assertIn("tier", v["full_matched_N2"]["names"])
        for k in v: self.assertIn("S_rank", v[k]["names"]); self.assertIn("gee", v[k]); self.assertTrue(v[k]["converged"])

    def test_the_matched_variants_use_exactly_the_matched_evaluable_features(self):
        s = torch.load(Path(self.out) / "selectivity.pt"); sv = torch.load(Path(self.out) / "survival.pt")
        ids = mv.matched_features(s, 3.0, 0); peak = s["peak"].numpy(); fire = sv["firing_rate"].numpy()
        n_eval = int((fire[peak[ids], ids] > 0.001).sum()); self.assertEqual(self.res["variants"]["full_matched_N2"]["n_features"], n_eval)
        self.assertLess(self.res["variants"]["full_matched_N2"]["n_features"], self.res["variants"]["full_all"]["n_features"])

    def test_adding_log_magnitude_removes_the_planted_magnitude_driven_selectivity_coefficient(self):
        # planted: survival depends on magnitude only and selective features have low magnitude, so the full model's S~ coefficient is negative and shrinks once log10 m is in the model
        v = self.res["variants"]; c = lambda k: v[k]["coef"][v[k]["names"].index("S_rank")]
        self.assertLess(c("full_all"), -1.0); self.assertLess(abs(c("log_m_all")), 0.5 * abs(c("full_all")))
        self.assertGreater(v["log_m_all"]["coef"][v["log_m_all"]["names"].index("log10_m")], 0.5)

    def test_the_reading_and_the_report_follow_the_rule_and_name_the_decision(self):
        self.assertIn(self.res["reading"]["label"].split(" ")[0], ("explained", "unresolved")); md = (Path(self.out) / "mv" / "mixed_variants.md").read_text()
        for needle in ("D-021", "A-034", "reading rule", "no_tier_all", "log_m_all", "full_matched_N6", "reproduces", "p exactly", "thread count"): self.assertIn(needle, md)
        self.assertTrue((Path(self.out) / "mv" / "mixed_variants.json").exists())

    def test_reproduction_is_reported_false_when_the_saved_coefficients_differ(self):
        d = json.loads((Path(self.out) / "base" / "analysis.json").read_text()); d["exploratory"]["mixed_logit"]["mixed"]["coef"][1] += 1e-9
        fake = Path(self.out) / "fake_analysis.json"; fake.write_text(json.dumps(d))
        r = mv.run_variants(self.out, ROSTER, Path(self.out) / "mv3", tau=3.0, seed=0, analysis_json=fake, gee=False)
        self.assertFalse(r["reproduces_analysis_json"]); self.assertAlmostEqual(r["max_abs_coef_diff"], 1e-9, delta=1e-12)           # the size of the difference is reported, not only a boolean
        self.assertEqual(self.res["max_abs_coef_diff"], 0.0)                                                                       # the genuine reproduction is exact at a fixed thread count
        r4 = mv.run_variants(self.out, ROSTER, Path(self.out) / "mv4", tau=3.0, seed=0, gee=False); self.assertIsNone(r4["reproduces_analysis_json"]); self.assertIsNone(r4["max_abs_coef_diff"])

    def test_variant_iii_and_the_n6_matched_set_equal_an_independent_computation_with_language_dependent_magnitudes(self):
        helper, out = tm.build_run("magnitude"); self.addCleanup(helper.doCleanups); sv = torch.load(Path(out) / "survival.pt"); L = sv["max_activation"].shape[0]
        sv["max_activation"] = sv["max_activation"] * torch.arange(1, L + 1, dtype=torch.float32)[:, None]; torch.save(sv, Path(out) / "survival.pt")      # magnitude now depends on the language: reading the wrong one is visible
        res = mv.run_variants(out, ROSTER, Path(out) / "mv", tau=3.0, seed=0)
        selp = torch.load(Path(out) / "selectivity.pt"); roster = {l["code"]: l for l in json.loads(ROSTER.read_text())["languages"]}; lm = [roster[c] for c in sv["languages"]]
        corr = {b: sv["corr"][b].numpy() for b in sv["bits"]}; fire = sv["firing_rate"].numpy(); sel = {k: selp[k].numpy() for k in ("selective", "supported", "peak", "S_rank", "strata", "controls")}
        st = an.feature_table(corr, fire, sel, lm, [7, 6], features=np.flatnonzero(sel["supported"])); maxa = sv["max_activation"].numpy().astype(float); peak = sel["peak"]
        expected = an.mixed_logit(st, [7, 6], gee=False, formula=mv.LOG_M, extra={"log10_m": np.log10(maxa[peak[st["feature"]], st["feature"]])})["mixed"]
        self.assertEqual(res["variants"]["log_m_all"]["coef"], [float(x) for x in expected["coef"]])
        idx = np.arange(len(peak)); ev = sel["supported"].astype(bool) & (fire[peak, idx] > 0.001)
        ids6 = mv.matched_features(selp, 3.0, 0, strata=mg.magnitude_strata(selp["strata"], selp["supported"], ev, maxa[peak, idx], 10))
        self.assertEqual(res["variants"]["full_matched_N6"]["n_features"], int(ev[ids6].sum()))

    def test_the_original_files_are_untouched(self):
        before = {f: (Path(self.out) / f).read_bytes() for f in ("survival.pt", "selectivity.pt", "meta.json")}
        mv.run_variants(self.out, ROSTER, Path(self.out) / "mv2", tau=3.0, seed=0)
        for f, b in before.items(): self.assertEqual((Path(self.out) / f).read_bytes(), b)


if __name__ == "__main__":
    unittest.main()
