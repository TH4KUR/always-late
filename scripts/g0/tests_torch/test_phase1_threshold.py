"""N2 tests for scripts/g0/phase1_threshold.py: sensitivity of the binary 'selective' split to FIXED S thresholds (proposal section 4.1 asks for threshold sensitivity;
decision D-019, a labelled deviation because the shuffled-null samples were not saved, so other null percentiles cannot be recomputed). The selection rule is exactly
phase1_run.selectivity_report's, `supported & (stable >= need) & (S > cutoff)`, with the cutoff replaced by tau, and the controls are drawn by the same matched_sample."""
import json, shutil, sys, tempfile, unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_lib as p1   # noqa: E402
import phase1_run as run   # noqa: E402
import phase1_threshold as th   # noqa: E402
import phase1_analysis as an   # noqa: E402
import test_phase1_analysis as tpa   # noqa: E402   (its synthetic run-directory builder; imported as a module so its test class is not collected a second time)

ROOT = Path(__file__).resolve().parents[3]


def hand_selp():
    # features:      0    1    2    3    4    5
    S =              [1.0, 2.0, 3.0, 5.0, 2.5, 6.0]
    supported =      [1,   1,   1,   1,   1,   0]
    stable =         [100, 100, 100, 100, 50,  100]
    strata =         [0,   0,   0,   0,   0,   -1]
    return {"S": torch.tensor(S), "supported": torch.tensor(supported, dtype=torch.bool), "stable_count": torch.tensor(stable), "min_stable_needed": 80,
            "strata": torch.tensor(strata), "null_cutoff": 1.8}


class Selection(unittest.TestCase):
    def test_selective_means_supported_stable_and_strictly_above_tau(self):
        alt = th.alternative_selection(hand_selp(), 2.0)
        # S == tau (feature 1) is not selective; feature 4 has S > tau but only 50 of 100 stable draws (need 80); feature 5 is above tau but unsupported
        self.assertEqual(alt["selective"].tolist(), [False, False, True, True, False, False])

    def test_the_stability_requirement_is_inclusive_exactly_at_the_needed_count(self):
        s = {"S": torch.tensor([3.0, 3.0, 3.0]), "supported": torch.ones(3, dtype=torch.bool), "stable_count": torch.tensor([80, 79, 81]), "min_stable_needed": 80,
             "strata": torch.zeros(3, dtype=torch.long)}                       # the run keeps a feature whose peak is stable in at least `need` draws (stable >= need)
        self.assertEqual(th.alternative_selection(s, 2.0)["selective"].tolist(), [True, False, True])

    def test_tau_is_recorded_and_the_other_keys_of_the_selectivity_dict_are_kept(self):
        s = hand_selp(); s["peak"] = torch.arange(6); alt = th.alternative_selection(s, 2.0)
        self.assertEqual(alt["tau"], 2.0); self.assertTrue(torch.equal(alt["peak"], s["peak"])); self.assertEqual(alt["null_cutoff"], 1.8)
        self.assertEqual(s.get("tau"), None)                                # the input is not modified

    def test_controls_are_drawn_by_the_runs_own_matched_sample_from_the_supported_non_selective_pool(self):
        s = hand_selp(); alt = th.alternative_selection(s, 2.0, seed=3)
        pool = s["supported"] & ~alt["selective"]; ref, short = p1.matched_sample(s["strata"], alt["selective"], pool, 3)
        self.assertTrue(torch.equal(alt["controls"], ref)); self.assertEqual(alt["control_strata_short"], short)
        self.assertTrue(set(alt["controls"].tolist()) <= set(pool.nonzero().squeeze(1).tolist()))
        self.assertFalse(bool(alt["selective"][alt["controls"]].any()))                # controls are never selective
        self.assertEqual(len(alt["controls"]), 2)                                      # as many controls as selective features in the stratum (the pool has 3)

    def test_a_larger_tau_gives_a_nested_smaller_selective_set(self):
        g = torch.Generator().manual_seed(0); F = 500
        s = {"S": torch.rand(F, generator=g) * 8, "supported": torch.rand(F, generator=g) < 0.8, "stable_count": torch.randint(0, 101, (F,), generator=g),
             "min_stable_needed": 80, "strata": torch.randint(-1, 5, (F,), generator=g), "null_cutoff": 1.8}
        sets = [th.alternative_selection(s, t)["selective"] for t in (1.8, 2.0, 3.0, 5.0)]
        for a, b in zip(sets, sets[1:]):
            self.assertTrue(bool((b & ~a).sum() == 0)); self.assertTrue(int(b.sum()) < int(a.sum()))

    def test_a_tau_above_every_score_selects_nothing_and_draws_no_controls(self):
        alt = th.alternative_selection(hand_selp(), 100.0); self.assertEqual(int(alt["selective"].sum()), 0); self.assertEqual(len(alt["controls"]), 0)

    def test_tau_equal_to_the_runs_null_cutoff_reproduces_the_saved_selection_exactly(self):
        # the identity that ties this module to the run: same rule, same pool, same seed -> same selective set and same controls
        NL, SP, NT, F = 3, 60, 20, 80; S_ = NL * SP; counts = torch.zeros(S_, F, dtype=torch.int16); lang = torch.arange(S_) // SP; odd = torch.arange(S_) % 2
        for j in range(4): counts[lang == j % NL, j] = 3
        for j in range(4, 74): counts[:, j] = 1 + (j % 3) * odd
        for j in range(74, 78): counts[(lang == 1) & (torch.arange(S_) % SP < 10), j] = 4
        t = p1.TagStats(S_, F); t.counts, t.sums, t.ntok = counts, counts.float() * 0.5, torch.full((S_,), NT, dtype=torch.long)
        rep = run.selectivity_report(t, lang, NL, n_boot=20, n_perm=10, seed=0)
        saved = {**rep, "S": rep["S"].float()}                                       # save_outputs stores S as float32
        alt = th.alternative_selection(saved, rep["null_cutoff"], seed=0)
        self.assertTrue(torch.equal(alt["selective"], rep["selective"])); self.assertTrue(torch.equal(alt["controls"], rep["controls"]))
        self.assertEqual(alt["control_strata_short"], rep["control_strata_short"]); self.assertGreater(int(rep["selective"].sum()), 0)


class Thresholds(unittest.TestCase):
    """End to end: derived selection files next to the unchanged survival file, the unchanged analysis (D-016/D-017), one result per tau."""
    def setUp(self):
        self.helper = tpa.RunAnalysis("test_a_planted_tier_effect_gives_a_small_p_and_a_null_does_not"); self.addCleanup(self.helper.doCleanups)       # owner of the synthetic run directories

    def make_run(self, F=4000, **kw):
        out = self.helper.synth(F=F, selective_penalty=0.05, **kw); sel = torch.load(out / "selectivity.pt")
        S = torch.ones(F); S[sel["selective"]] = torch.linspace(2.1, 9.0, int(sel["selective"].sum()))            # selective features have S in (2.1, 9]; every other feature 1.0
        S[~sel["selective"] & sel["supported"]] = 1.0
        sel.update(S=S, null_cutoff=2.0, min_stable_needed=80, stable_count=torch.full((F,), 100)); torch.save(sel, out / "selectivity.pt"); return out

    def go(self, out, taus, **kw):
        return th.run_thresholds(out, ROOT / "reports/g0/phase1_languages.json", Path(out) / "thr", taus, n_perm=50, n_boot=20, seed=0, **kw)

    def test_one_result_per_tau_with_group_sizes_and_the_three_confirmatory_tests(self):
        out = self.make_run(); res = self.go(out, [3.0, 5.0])
        self.assertEqual([r["label"] for r in res["rows"]], ["null", "3.0", "5.0"])
        n = [r["n_selective"] for r in res["rows"]]; self.assertTrue(n[0] > n[1] > n[2] > 0)
        for r in res["rows"]:
            self.assertEqual(set(r["tests"]), {"T1_int7", "T2_int6", "T3_tier"}); self.assertEqual(set(r["holm"]), {"T1_int7", "T2_int6", "T3_tier"}); self.assertIn("delta", r["tests"]["T1_int7"])
        self.assertTrue((Path(out) / "thr" / "thresholds.json").exists()); self.assertTrue((Path(out) / "thr" / "thresholds.md").exists())

    def test_the_null_row_equals_the_unchanged_analysis_of_the_original_run(self):
        out = self.make_run(); base = an.run_analysis(out, ROOT / "reports/g0/phase1_languages.json", Path(out) / "base", n_perm=50, n_boot=20, seed=0)
        row = self.go(out, [3.0])["rows"][0]
        for k in ("T1_int7", "T2_int6"):
            self.assertEqual(row["tests"][k]["delta"], base["tests"][k]["delta"]); self.assertEqual(row["tests"][k]["p"], base["tests"][k]["p"]); self.assertEqual(list(row["tests"][k]["ci"]), list(base["tests"][k]["ci"]))
        self.assertEqual(row["tests"]["T3_tier"]["slope"], base["tests"]["T3_tier"]["slope"]); self.assertEqual(row["holm"], base["holm"])

    def test_a_tau_that_selects_nothing_is_recorded_as_undefined_and_does_not_stop_the_other_rows(self):
        out = self.make_run(); res = self.go(out, [3.0, 50.0]); last = res["rows"][-1]
        self.assertEqual(last["n_selective"], 0); self.assertIn("undefined", last); self.assertNotIn("tests", last)
        self.assertIn("tests", res["rows"][1]); self.assertIn("undefined", (Path(out) / "thr" / "thresholds.md").read_text())

    def test_selective_features_without_any_control_pool_are_undefined_not_a_crash(self):
        out = self.make_run(); sel = torch.load(out / "selectivity.pt"); sel["supported"] = sel["selective"].clone(); torch.save(sel, out / "selectivity.pt")      # every supported feature is selective: no universal pool
        res = self.go(out, [3.0]); null, t3 = res["rows"]
        self.assertGreater(null["n_selective"], 0); self.assertEqual(null["n_controls"], 0); self.assertIn("undefined", null); self.assertNotIn("tests", null)    # at the run's own cutoff the pool is empty
        self.assertGreater(t3["n_controls"], 0)                      # at tau = 3 the features with S between 2.1 and 3 are supported and no longer selective: a pool exists, the run went on

    def test_the_original_files_are_untouched_and_no_derived_directory_is_left_behind(self):
        out = self.make_run(); before = {f: (Path(out) / f).read_bytes() for f in ("survival.pt", "selectivity.pt", "meta.json")}
        self.go(out, [3.0])
        for f, b in before.items(): self.assertEqual((Path(out) / f).read_bytes(), b)
        self.assertEqual([p.name for p in (Path(out) / "thr").iterdir() if p.name.startswith("_derived")], [])

    def test_the_report_states_the_deviation_and_the_threshold_rule(self):
        out = self.make_run(); self.go(out, [3.0]); md = (Path(out) / "thr" / "thresholds.md").read_text()
        for needle in ("D-019", "fixed", "deviation", "S >", "stable", "null"): self.assertIn(needle, md)

    def test_taus_must_be_positive_and_unique(self):
        out = self.make_run()
        with self.assertRaises(ValueError): self.go(out, [3.0, 3.0])
        with self.assertRaises(ValueError): self.go(out, [-1.0])
        with self.assertRaises(ValueError): self.go(out, [0.0])


if __name__ == "__main__":
    unittest.main()
