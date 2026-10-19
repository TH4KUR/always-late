"""N11 step 2 tests (decision D-025; pre-specified in handoff 007 section 4). The JumpReLU margin of an activation is max_activation / theta_f (Gemma Scope, Lieberum et al. 2024,
arXiv:2408.05147). It is introduced by re-expressing `max_activation` in a DERIVED survival file, so that the existing, tested estimators of N6 (magnitude-matched contrast), N7 (mixed-model
variants) and N9 (language-level scale covariate) run unchanged with 'magnitude' meaning 'margin'. Planted effects on synthetic data in which theta varies independently of the magnitude."""
import json, shutil, sys, tempfile, unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_sae_thresholds as ex   # noqa: E402
import phase1_magnitude as mg   # noqa: E402
import phase1_margin as pm   # noqa: E402
import test_phase1_magnitude as tm   # noqa: E402   (imported as modules so their test classes are not collected a second time)
import test_phase1_language_level as tll   # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
ROSTER = ROOT / "reports/g0/phase1_languages.json"


class MarginSurvival(unittest.TestCase):
    def surv(self):
        return {"languages": ["a", "b"], "firing_rate": torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]), "max_activation": torch.tensor([[10.0, 20.0, 30.0], [40.0, 80.0, 150.0]]),
                "mean_activation": torch.tensor([[1.0, 2.0, 3.0], [4.0, 8.0, 15.0]]), "corr": {6: torch.tensor([[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]])}, "n": [5, 5]}

    def test_both_activation_arrays_are_divided_by_the_features_threshold_along_the_feature_axis(self):
        s = self.surv(); d = pm.margin_survival(s, np.array([2.0, 4.0, 5.0], dtype=np.float32))
        self.assertTrue(torch.allclose(d["max_activation"], torch.tensor([[5.0, 5.0, 6.0], [20.0, 20.0, 30.0]]))); self.assertTrue(torch.allclose(d["mean_activation"], torch.tensor([[0.5, 0.5, 0.6], [2.0, 2.0, 3.0]])))

    def test_everything_else_is_unchanged_and_the_input_is_not_modified(self):
        s = self.surv(); before = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in s.items() if k != "corr"}; d = pm.margin_survival(s, np.ones(3, dtype=np.float32) * 2)
        self.assertTrue(torch.equal(d["firing_rate"], s["firing_rate"])); self.assertTrue(torch.equal(d["corr"][6], s["corr"][6])); self.assertEqual(d["languages"], ["a", "b"]); self.assertEqual(d["n"], [5, 5])
        self.assertTrue(torch.equal(s["max_activation"], before["max_activation"])); self.assertTrue(torch.equal(s["mean_activation"], before["mean_activation"]))

    def test_a_threshold_vector_of_the_wrong_length_is_refused(self):
        with self.assertRaisesRegex(ValueError, "2 thresholds for 3 features"): pm.margin_survival(self.surv(), np.array([2.0, 4.0], dtype=np.float32))


class MakeRun(unittest.TestCase):
    def test_the_derived_run_directory_has_the_ratio_survival_file_and_untouched_copies_of_the_rest(self):
        helper, out = tm.build_run("selectivity"); self.addCleanup(helper.doCleanups); F = 4000; rng = np.random.default_rng(0); theta = rng.uniform(2.6, 16.0, F).astype(np.float32)
        csv_path = Path(out) / "thr.csv"; ex.write_thresholds_csv(csv_path, theta, {})
        before = {f: (Path(out) / f).read_bytes() for f in ("survival.pt", "selectivity.pt", "meta.json")}
        run = pm.make_margin_run(out, csv_path, Path(out) / "margin_run")
        for f in ("selectivity.pt", "meta.json"): self.assertEqual((run / f).read_bytes(), before[f])
        orig = torch.load(Path(out) / "survival.pt"); der = torch.load(run / "survival.pt")
        self.assertTrue(torch.allclose(der["max_activation"], orig["max_activation"] / torch.tensor(theta)[None, :])); self.assertTrue(torch.equal(der["corr"][6], orig["corr"][6]))
        for f, b in before.items(): self.assertEqual((Path(out) / f).read_bytes(), b)
        self.assertTrue((Path(out) / "margin_run" / "PROVENANCE.txt").exists())

    def test_a_threshold_file_for_another_sae_width_is_refused(self):
        helper, out = tm.build_run("selectivity"); self.addCleanup(helper.doCleanups); csv_path = Path(out) / "thr.csv"; ex.write_thresholds_csv(csv_path, np.full(100, 3.0, dtype=np.float32), {})
        with self.assertRaisesRegex(ValueError, "100 thresholds for 4000 features"): pm.make_margin_run(out, csv_path, Path(out) / "margin_run")


class Readings(unittest.TestCase):
    """Rules fixed beforehand. (i) the A-035 residual is 'explained by margin' if the margin-matched INT6 contrast at S > 5 (finest resolution with enough support) is not significant or within
    +-1 pp (the 'explained by magnitude' class of D-020), else it 'remains'. (ii) margin 'carries more than raw magnitude' at the language level if with the margin scale alone the INT5 tier
    slope has p >= 0.05 while it is < 0.05 with the max-activation scale (N9), else 'no more than raw activation'."""
    def test_reading_i(self):
        self.assertEqual(pm.reading_i("explained by magnitude"), "explained by margin")
        for r in ("reversed", "supported", "partly explained"): self.assertEqual(pm.reading_i(r), "remains")
        with self.assertRaises(ValueError): pm.reading_i("something else")

    def test_reading_ii(self):
        self.assertEqual(pm.reading_ii(0.30, 0.002), "margin carries more than raw magnitude")
        self.assertEqual(pm.reading_ii(0.05, 0.002), "margin carries more than raw magnitude")             # p exactly 0.05 is not significant
        self.assertEqual(pm.reading_ii(0.0499, 0.002), "no more than raw activation")
        self.assertEqual(pm.reading_ii(0.30, 0.05), "no more than raw activation")                         # the raw scale already leaves no INT5 effect: nothing for the margin to add
        self.assertEqual(pm.reading_ii(0.30, 0.30), "no more than raw activation")


def build_margin_run(seed=0, F=4000):
    """Planted: survival depends on the MARGIN m / theta only; the magnitude m has the same distribution for selective and control features, but selective features have a 2.5 times higher threshold (a 4 times ratio leaves too little common support)."""
    helper, out = tm.build_run("selectivity"); rng = np.random.default_rng(seed); sel = torch.load(Path(out) / "selectivity.pt"); sv = torch.load(Path(out) / "survival.pt"); L = sv["corr"][7].shape[0]
    selective = sel["selective"].numpy(); m = np.exp(rng.normal(0, 0.5, F)); theta = np.where(selective, 7.5, 3.0) * np.exp(rng.normal(0, 0.05, F))
    sv["max_activation"] = torch.tensor(np.tile(m, (L, 1)), dtype=torch.float32); sv["mean_activation"] = sv["max_activation"] / 8          # positive, so that the language-level secondary scale is defined
    for b, base in ((7, 0.985), (6, 0.95), (5, 0.9), (4, 0.8)):
        sv["corr"][b] = torch.tensor(np.clip(base + 0.05 * np.log(m / theta)[None, :] + 0.05 + rng.normal(0, 0.02, (L, F)), -1, 1), dtype=torch.float32)   # +0.05 recentres the mean margin effect
    torch.save(sv, Path(out) / "survival.pt"); csv_path = Path(out) / "thr.csv"; ex.write_thresholds_csv(csv_path, theta.astype(np.float32), {}); return helper, out, csv_path


def with_language_scale(out):
    """Give every language its own activation scale (the language-level analyses are undefined when all languages are identical: a constant scale has no rank correlation)."""
    sv = torch.load(Path(out) / "survival.pt"); L = sv["max_activation"].shape[0]; f = torch.linspace(0.7, 1.6, L)[:, None]
    sv["max_activation"] = sv["max_activation"] * f; sv["mean_activation"] = sv["mean_activation"] * f * 1.1; torch.save(sv, Path(out) / "survival.pt")


class ReadingInputs(unittest.TestCase):
    """Which numbers feed the two readings: the S > 5 row of the margin-matched contrast, the margin scale's INT5 p and N9's max-activation INT5 p. Hand-made inputs in which the neighbouring entries differ."""
    def test_the_headline_is_the_s_gt_5_row_and_the_p_values_are_the_int5_entries(self):
        mag = {"readings": {"3.0": {"final": {"reading": "reversed", "bins": 5}}, "5.0": {"final": {"reading": "explained by magnitude", "bins": 10}}}}
        lev = {"activation_scale": {"widths": {"5": {"p_scale": 0.11}, "6": {"p_scale": 0.22}}}}; n9 = {"activation_scale": {"widths": {"5": {"p_scale": 0.33}, "6": {"p_scale": 0.44}}}}
        fin, mar_p, raw_p = pm.reading_inputs(mag, lev, n9); self.assertEqual(fin, {"reading": "explained by magnitude", "bins": 10}); self.assertEqual((mar_p, raw_p), (0.11, 0.33))

    def test_a_missing_headline_is_an_error(self):
        mag = {"readings": {"5.0": {"final": None}}}; lev = {"activation_scale": {"widths": {"5": {"p_scale": 0.1}}}}; n9 = {"activation_scale": {"widths": {"5": {"p_scale": 0.3}}}}
        with self.assertRaisesRegex(ValueError, "no margin-matched reading at S > 5"): pm.reading_inputs(mag, lev, n9)


class FeatureLevel(unittest.TestCase):
    def test_a_margin_driven_effect_survives_raw_magnitude_matching_and_vanishes_under_margin_matching(self):
        helper, out, csv_path = build_margin_run(); self.addCleanup(helper.doCleanups)
        raw = mg.run_magnitude(out, ROSTER, Path(out) / "raw", [3.0], bins=(10,), n_perm=60, n_boot=20, seed=0)
        res = pm.run_feature_level(out, csv_path, ROSTER, Path(out) / "margin", [3.0], bins=(10,), n_perm=60, n_boot=20, seed=0)
        rate = raw["rate_only"]["rows"][1]["tests"]["T2_int6"]["delta"]; raw10 = raw["matched"]["10"]["rows"][1]["tests"]["T2_int6"]["delta"]; mar10 = res["matched"]["10"]["rows"][1]["tests"]["T2_int6"]["delta"]
        self.assertLess(rate, -0.3); self.assertLess(raw10, -0.25)                 # neither firing rate nor raw magnitude can see the threshold
        self.assertLess(abs(mar10), 0.06)                                          # margin matching removes it

    def test_the_report_is_relabelled_so_that_it_cannot_be_read_as_a_raw_magnitude_report(self):
        helper, out, csv_path = build_margin_run(); self.addCleanup(helper.doCleanups)
        pm.run_feature_level(out, csv_path, ROSTER, Path(out) / "margin", [3.0], bins=(3,), n_perm=20, n_boot=10, seed=0); md = (Path(out) / "margin" / "magnitude.md").read_text()
        self.assertTrue(md.startswith("> **N11 (D-025): in this report every activation is divided by its feature's learned JumpReLU threshold")); self.assertIn("read 'maximum activation' as 'margin ratio'", md)


class EndToEnd(unittest.TestCase):
    def test_run_margin_writes_the_three_analyses_and_the_two_pre_specified_readings(self):
        helper, out, csv_path = build_margin_run(); self.addCleanup(helper.doCleanups); with_language_scale(out)
        d = tll.synth("tier"); self.addCleanup(shutil.rmtree, d, ignore_errors=True)                                  # per-language csv / meta / stats for the same 27 languages
        raw = mg.run_magnitude(out, ROSTER, Path(out) / "raw", [3.0, 5.0], bins=(3, 10), n_perm=30, n_boot=10, seed=0)
        ll9 = json.loads(json.dumps({"activation_scale": {"widths": {"5": {"p_scale": 0.002}}}})); (Path(out) / "n9.json").write_text(json.dumps(ll9)); (Path(out) / "raw" / "n2.json").write_text(json.dumps(raw["rate_only"]))
        res = pm.run_margin(out, csv_path, ROSTER, Path(out) / "m", per_language_csv=d / "per_language.csv", meta_json=d / "meta.json", stats_json=d / "stats.json", n2_json=Path(out) / "raw" / "n2.json",
                            n9_json=Path(out) / "n9.json", taus=[3.0, 5.0], bins=(3, 10), n_perm=30, n_boot=10, seed=0)
        self.assertEqual(sorted(res["readings"]), ["i", "ii"]); self.assertIn(res["readings"]["i"], ("explained by margin", "remains")); self.assertIn(res["readings"]["ii"], ("margin carries more than raw magnitude", "no more than raw activation"))
        for f in ("magnitude/magnitude.json", "mixed_variants/mixed_variants.json", "language_level/language_level.json", "margin_summary.json", "margin_summary.md"): self.assertTrue((Path(out) / "m" / f).exists(), f)
        md = (Path(out) / "m" / "margin_summary.md").read_text()
        for needle in ("D-025", "Reading rule", "explained by margin", "carries more than raw magnitude", "Lieberum"): self.assertIn(needle, md)
        indep = pm.run_feature_level(out, csv_path, ROSTER, Path(out) / "indep", [3.0, 5.0], bins=(3, 10), n_perm=30, n_boot=10, seed=0, rate_only_json=Path(out) / "raw" / "n2.json")
        self.assertEqual(res["feature_level"]["final"], indep["readings"]["5.0"]["final"]); self.assertEqual(res["readings"]["i"], pm.reading_i(indep["readings"]["5.0"]["final"]["reading"]))     # the headline is the S > 5 row
        lev = res["language_level"]; self.assertEqual(lev["margin_p_int5"], lev["activation_scale"]["widths"]["5"]["p_scale"]); self.assertEqual(lev["raw_p_int5"], 0.002)                         # INT5 entries
        self.assertEqual(json.loads((Path(out) / "m" / "mixed_variants" / "mixed_variants.json").read_text())["tau"], 5.0)                                                                 # the mixed variants use S > 5
        self.assertTrue((Path(out) / "m" / "magnitude" / "magnitude.md").read_text().startswith("> **N11 (D-025)")); self.assertTrue((Path(out) / "m" / "language_level" / "language_level.md").read_text().startswith("> **N11 (D-025)"))
        self.assertTrue((Path(out) / "m" / "mixed_variants" / "mixed_variants.md").read_text().startswith("> **N11 (D-025)"))

    def test_the_language_level_part_equals_a_direct_run_on_the_derived_survival_file(self):
        helper, out, csv_path = build_margin_run(); self.addCleanup(helper.doCleanups); with_language_scale(out); d = tll.synth("tier"); self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        run = pm.make_margin_run(out, csv_path, Path(out) / "run"); import phase1_language_level as ll
        direct = ll.run_language_level(d / "per_language.csv", d / "meta.json", d / "stats.json", ROSTER, Path(out) / "direct", n_perm=40, seed=0, survival_pt=run / "survival.pt")
        got = pm.run_language_level_margin(out, csv_path, ROSTER, Path(out) / "ll", per_language_csv=d / "per_language.csv", meta_json=d / "meta.json", stats_json=d / "stats.json", n_perm=40, seed=0)
        self.assertEqual(got["activation_scale"], direct["activation_scale"])
        flat = json.dumps(got["activation_scale"]); self.assertNotIn("NaN", flat); self.assertNotIn("Infinity", flat)             # a degenerate fixture would make the equality meaningless (NaN != NaN)
        raw = pm.ll.run_language_level(d / "per_language.csv", d / "meta.json", d / "stats.json", ROSTER, Path(out) / "rawll", n_perm=40, seed=0, survival_pt=Path(out) / "survival.pt")
        self.assertNotEqual(got["activation_scale"]["scale"], raw["activation_scale"]["scale"])                                     # the margin scale really differs from the raw scale


if __name__ == "__main__":
    unittest.main()
