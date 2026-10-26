import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_transfer_analysis as ta
import phase1_transfer_native_fixed as nf
import phase1_transfer_run as tr
from test_phase1_transfer_analysis import group_items_for

LANGS = ["a", "b", "c"]
STATES = ("INT16", "INT6", "INT4")


def cells_of(ff, qf, fq, qq, qn):
    return [ff, qf, fq, qq, qn]


def planted(n_sent=40, seed=0, scale=1.0, remainder=0.1, amp=0.2, drift=1.0):
    """Group items of three languages with per-sentence noise in the cells FF and FQ, a constant native remainder (QN = FQ + remainder, so the paired difference has no sentence variance) and QQ = QN + amp."""
    rng = np.random.default_rng(seed); table = {}
    for l in LANGS:
        for j in range(n_sent):
            ff = float(rng.normal(1.0, 0.3 * scale)); fq = ff + drift + float(rng.normal(0, 0.5 * scale)); qn = fq + remainder; qq = qn + amp; qf = ff + 0.3
            table[(l, j)] = cells_of(ff, qf, fq, qq, qn)
    return table


def items_of(table, n_sent=40):
    return group_items_for(LANGS, n_sent, lambda l, j: table[(l, j)], [lambda l, j: [0.0] * 5, lambda l, j: [0.0] * 5])


class Contrasts(unittest.TestCase):
    def test_the_quantities_follow_the_definitions_and_the_native_total_is_the_weights_effect_plus_the_native_remainder(self):
        net = np.array([[1.0, 2.0, 4.0, 8.0, 5.0], [0.5, 0.6, 0.9, 1.5, 1.0]]); c = nf.contrasts(net)
        self.assertTrue(np.allclose(c["C"], [1.0, 0.1])); self.assertTrue(np.allclose(c["W"], [3.0, 0.4])); self.assertTrue(np.allclose(c["I"], [8 - 2 - 4 + 1, 1.5 - 0.6 - 0.9 + 0.5])); self.assertTrue(np.allclose(c["F"], [7.0, 1.0])); self.assertTrue(np.allclose(c["N"], [4.0, 0.5]))
        self.assertTrue(np.allclose(c["R"], [1.0, 0.1])); self.assertTrue(np.allclose(c["D"], [-3.0, -0.5])); self.assertTrue(np.allclose(c["N"], c["W"] + c["R"])); self.assertTrue(np.allclose(c["N"], c["F"] + c["D"])); self.assertEqual(nf.identity_error(net), 0.0)

    def test_the_identity_error_is_the_largest_violation_of_both_identities_and_nan_languages_do_not_hide_it(self):
        net = np.array([[1.0, 2.0, 4.0, 8.0, 5.0], [np.nan] * 5]); self.assertLess(nf.identity_error(net), 1e-12)
        orig = nf.contrasts
        try:
            nf.contrasts = lambda n: {**orig(n), "D": orig(n)["D"] + np.array([0.0, 0.0]) + 0.5}                              # an inconsistent D: N = F + D is violated by 0.5
            self.assertAlmostEqual(nf.identity_error(net[:1]), 0.5)
            nf.contrasts = lambda n: {**orig(n), "R": orig(n)["R"] + 0.25}                                                     # an inconsistent R: N = W + R is violated by 0.25
            self.assertAlmostEqual(nf.identity_error(net[:1]), 0.25)
        finally:
            nf.contrasts = orig


class Summary(unittest.TestCase):
    def test_medians_of_paired_values_the_counts_and_the_nan_languages_are_left_out_and_counted(self):
        c = {"R": np.array([0.1, -0.2, 0.3, np.nan]), "D": np.array([-1.0, -2.0, 0.5, np.nan]), "N": np.array([1.0, 2.0, 3.0, np.nan]), "F": np.array([2.0, 4.0, 3.0, np.nan]), "W": np.array([0.5, -0.5, 0.1, np.nan])}
        s = nf.summarize(c); self.assertEqual(s["n_languages"], 3); self.assertEqual(s["n_missing_languages"], 1)
        self.assertEqual(s["median_R"], ta.weighted_median(c["R"][:3], np.ones(3))); self.assertEqual(s["median_D"], -1.0); self.assertEqual(s["median_N"], 2.0); self.assertEqual(s["median_F"], 3.0); self.assertEqual(s["median_W"], 0.1)
        self.assertEqual(s["n_abs_R_gt_abs_W"], 1); self.assertEqual(s["n_D_negative"], 2); self.assertEqual(s["n_R_positive"], 2); self.assertEqual(s["median_abs_R_minus_abs_W"], ta.weighted_median(np.abs(c["R"][:3]) - np.abs(c["W"][:3]), np.ones(3)))

    def test_a_language_with_only_the_weights_effect_missing_is_left_out(self):
        c = {"R": np.array([0.1, 0.2]), "D": np.array([1.0, 2.0]), "N": np.array([1.0, 2.0]), "F": np.array([1.0, 2.0]), "W": np.array([0.5, np.nan])}; s = nf.summarize(c); self.assertEqual((s["n_languages"], s["n_missing_languages"]), (1, 1))

    def test_ties_and_zeros_are_not_counted_as_larger_negative_or_positive(self):
        c = {"R": np.array([0.0, 0.5, -0.5]), "D": np.array([0.0, -1.0, 1.0]), "N": np.array([0.0] * 3), "F": np.array([0.0] * 3), "W": np.array([0.0, 0.5, 0.2])}; s = nf.summarize(c)
        self.assertEqual(s["n_abs_R_gt_abs_W"], 1); self.assertEqual(s["n_D_negative"], 1); self.assertEqual(s["n_R_positive"], 1)

    def test_no_language_with_a_value_gives_nan_summaries(self):
        s = nf.summarize({k: np.array([np.nan]) for k in ("R", "D", "N", "F", "W")}); self.assertEqual(s["n_languages"], 0); self.assertTrue(np.isnan(s["median_R"]))


class Bootstrap(unittest.TestCase):
    def setUp(self):
        self.table = planted(); self.A = ta.group_arrays(items_of(self.table), LANGS, 40, 2)

    def test_the_point_is_the_contrast_of_the_pooled_data_and_the_per_language_interval_brackets_it(self):
        r = nf.bootstrap(self.A, n_boot=100, seed=1); pool = ta.pooled_harms(self.A); c = nf.contrasts(pool["net"])
        for k in ("W", "F", "N", "R", "D"):
            self.assertTrue(np.allclose(r["point"][k], c[k])); lo, hi = r["per_language"][k]; self.assertTrue(np.all(lo <= c[k] + 1e-9) and np.all(hi >= c[k] - 1e-9))
        self.assertEqual(r["n_finite"], 100)

    def test_the_replicates_use_the_scheme_and_seed_of_the_a3_bootstrap_with_one_multiplicity_vector_for_all_languages_and_cells(self):
        r = nf.bootstrap(self.A, n_boot=60, seed=7); rng = np.random.default_rng(7); S = 40; meds = []
        for _ in range(60):
            mult = np.bincount(rng.integers(0, S, S), minlength=S); meds.append(nf.summarize(nf.contrasts(ta.pooled_harms(self.A, mult=mult)["net"]))["median_N"])
        self.assertAlmostEqual(r["summary_ci"]["median_N"][0], float(np.percentile(meds, 2.5)), places=12); self.assertAlmostEqual(r["summary_ci"]["median_N"][1], float(np.percentile(meds, 97.5)), places=12)

    def test_a_paired_difference_that_is_constant_over_sentences_has_a_degenerate_interval_while_the_cells_it_joins_do_not(self):
        r = nf.bootstrap(self.A, n_boot=100, seed=2); lo, hi = r["per_language"]["R"]; self.assertTrue(np.allclose(lo, 0.1) and np.allclose(hi, 0.1)); lo, hi = r["per_language"]["D"]; self.assertTrue(np.allclose(lo, -0.2) and np.allclose(hi, -0.2))
        flo, fhi = r["per_language"]["W"]; self.assertTrue(np.all(fhi - flo > 0.05))                                                      # the weights effect varies between sentences: pairing is what makes R exact
        self.assertAlmostEqual(r["summary_ci"]["median_R"][0], 0.1); self.assertAlmostEqual(r["summary_ci"]["median_R"][1], 0.1)

    def test_the_same_seed_gives_the_same_intervals_and_another_seed_different_ones(self):
        a, b, c = nf.bootstrap(self.A, 40, 3), nf.bootstrap(self.A, 40, 3), nf.bootstrap(self.A, 40, 4); self.assertEqual(a["summary_ci"], b["summary_ci"]); self.assertNotEqual(a["summary_ci"]["median_N"], c["summary_ci"]["median_N"])

    def test_a_sentence_with_a_missing_control_set_is_left_out_and_counted_per_language(self):
        items = items_of(self.table); del items[("b", 3, "control1")]; A = ta.group_arrays(items, LANGS, 40, 2); r = nf.bootstrap(A, n_boot=20, seed=1); self.assertEqual(r["n_sentences"].tolist(), [40, 39, 40])


class Run(unittest.TestCase):
    def write(self, root, side, languages, drift_by_state, n_sent=12, drop=None, skip=None):
        for i, s in enumerate(STATES):
            d = root / side / f"group-{s}"; d.mkdir(parents=True)
            table = {}
            rng = np.random.default_rng(10 + i)
            for l in languages:
                for j in range(n_sent):
                    ff = float(rng.normal(1.0, 0.2)); fq = ff + drift_by_state[s] + float(rng.normal(0, 0.2)); table[(l, j)] = cells_of(ff, ff + 0.2, fq, fq + 0.5 + drift_by_state[s], fq + 0.05)
            items = group_items_for(languages, n_sent, lambda l, j: table[(l, j)], [lambda l, j: [0.0] * 5, lambda l, j: [0.0] * 5, lambda l, j: [0.0] * 5])
            if drop and drop[0] == (side, s):
                del items[drop[1]]
            skipped = []
            if skip and skip[0] == (side, s):
                del items[skip[1]]; skipped = [(skip[1], "no_active_group_feature")]
            half = len(items) // 2; keys = sorted(items)
            tr.atomic_save({"items": {k: items[k] for k in keys[:half]}, "pooled": {}, "skipped": skipped}, d / f"group_{s}_c000.pt", "k0"); tr.atomic_save({"items": {k: items[k] for k in keys[half:]}, "pooled": {}, "skipped": []}, d / f"group_{s}_c001.pt", "k1")
            (d / "meta_transfer.json").write_text(json.dumps({"complete": True}))

    def test_the_driver_checks_the_planned_counts_loads_both_corpora_and_writes_per_language_summary_and_json_with_the_identity_check(self):
        ext_langs = ["x", "y"]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); drift = {"INT16": 0.0, "INT6": 0.1, "INT4": 1.0}; self.write(root, "main", LANGS, drift); self.write(root, "ext", ext_langs, drift)
            dirs = {"flores_main": {s: root / "main" / f"group-{s}" for s in STATES}, "wikipedia_external": {s: root / "ext" / f"group-{s}" for s in STATES}}
            per, summ = nf.run(dirs, {"flores_main": LANGS, "wikipedia_external": ext_langs}, 12, 3, STATES, root / "out", n_boot=30, seed=1)
            self.assertEqual(len(per), (3 + 2) * 3); self.assertEqual({r["corpus"] for r in summ}, {"flores_main", "wikipedia_external"}); self.assertEqual(len(summ), 2 * 3 * len(nf.SUMMARY_METRICS))
            js = json.loads((root / "out" / "native_fixed.json").read_text()); self.assertLess(js["max_identity_error"], 1e-12); self.assertGreater(js["max_identity_error"], -1.0); self.assertEqual(js["n_boot"], 30); self.assertIn("flores_main/INT4/meta_transfer.json", js["inputs_meta_sha256"])
            with (root / "out" / "native_fixed_per_language.csv").open() as fh:
                rows = list(csv.DictReader(fh))
            r4 = [r for r in rows if r["corpus"] == "flores_main" and r["state"] == "INT4" and r["language"] == "a"][0]; self.assertAlmostEqual(float(r4["R"]), 0.05, delta=1e-9 + 0.0); self.assertAlmostEqual(float(r4["D"]), 0.05 - 0.5 - 1.0, delta=1e-9)
            self.assertLessEqual(float(r4["R_low"]), float(r4["R"]) + 1e-12); self.assertGreaterEqual(float(r4["R_high"]), float(r4["R"]) - 1e-12)
            m = [r for r in summ if r["corpus"] == "flores_main" and r["state"] == "INT4" and r["metric"] == "median_R"][0]; self.assertAlmostEqual(float(m["point"]), 0.05, delta=1e-9)
            w4 = [r for r in rows if r["corpus"] == "flores_main" and r["state"] == "INT4" and r["language"] == "a"][0]; self.assertLess(float(w4["W_low"]), float(w4["W_high"]))                       # the weights effect varies between sentences: a proper interval
            self.assertEqual(r4["n_sentences"], "12")
            mw = [r for r in summ if r["corpus"] == "flores_main" and r["state"] == "INT4" and r["metric"] == "median_W"][0]; self.assertLess(float(mw["low"]), float(mw["high"])); self.assertLessEqual(float(mw["low"]), float(mw["point"]) + 0.2); self.assertGreaterEqual(float(mw["high"]), float(mw["point"]) - 0.2)
            self.assertTrue(all(float(r["low"]) <= float(r["high"]) for r in summ if r["metric"] in ("median_N", "median_F", "median_W")))

    def test_a_skipped_set_counts_toward_the_planned_items_and_leaves_its_sentence_out_of_that_language(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); drift = {"INT16": 0.0, "INT6": 0.1, "INT4": 1.0}; self.write(root, "main", LANGS, drift, skip=(("main", "INT4"), ("a", 3, "control1"))); dirs = {"flores_main": {s: root / "main" / f"group-{s}" for s in STATES}}
            per, summ = nf.run(dirs, {"flores_main": LANGS}, 12, 3, STATES, root / "out", n_boot=10, seed=1)
            n = {(r["state"], r["language"]): r["n_sentences"] for r in per}; self.assertEqual((n[("INT4", "a")], n[("INT4", "b")], n[("INT6", "a")]), (11, 12, 12))

    def test_the_largest_identity_error_over_corpora_and_states_is_what_the_json_records(self):
        calls = []; orig = nf.identity_error
        def fake(net):
            calls.append(1); return 0.001 * len(calls)
        nf.identity_error = fake
        try:
            with tempfile.TemporaryDirectory() as d:
                root = Path(d); drift = {"INT16": 0.0, "INT6": 0.1, "INT4": 1.0}; self.write(root, "main", LANGS, drift); dirs = {"flores_main": {s: root / "main" / f"group-{s}" for s in STATES}}
                nf.run(dirs, {"flores_main": LANGS}, 12, 3, STATES, root / "out", n_boot=5, seed=1); self.assertAlmostEqual(json.loads((root / "out" / "native_fixed.json").read_text())["max_identity_error"], 0.003)
        finally:
            nf.identity_error = orig

    def test_an_incomplete_run_is_refused_like_the_report_driver_does(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); drift = {"INT16": 0.0, "INT6": 0.1, "INT4": 1.0}; self.write(root, "main", LANGS, drift, drop=(("main", "INT6"), ("a", 3, "group")))
            dirs = {"flores_main": {s: root / "main" / f"group-{s}" for s in STATES}}
            with self.assertRaisesRegex(ValueError, "planned"):
                nf.run(dirs, {"flores_main": LANGS}, 12, 3, STATES, root / "out", n_boot=10, seed=1)


if __name__ == "__main__":
    unittest.main()
