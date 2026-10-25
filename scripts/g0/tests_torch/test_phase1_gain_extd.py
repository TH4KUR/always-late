import contextlib
import csv
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_gain_extc as ec
import phase1_gain_extd as ed
from test_phase1_gain_extc import IDS, N, NTOK, RATIOS, Loading, planted_run

GRID = [1.0] + RATIOS
LANGS27 = [f"l{i}" for i in range(27)]


def quad(xstar, k=0.1):
    """Group harm that is 0 at ratio 1 and lowest at ratio exp(xstar): k x^2 - 2 k xstar x with x = ln a."""
    return lambda a: k * math.log(a) ** 2 - 2 * k * xstar * math.log(a)


class Estimators(unittest.TestCase):
    def test_the_grid_estimator_is_the_ratio_with_the_lowest_harm_among_one_and_the_ratios(self):
        self.assertEqual(ed.alpha_grid({1.1: 0.001, 1.2: -0.002, 1.4: -0.001, 1.7: 0.01}), 1.2)
        self.assertEqual(ed.alpha_grid({1.1: 0.001, 1.2: 0.002, 1.4: 0.003, 1.7: 0.01}), 1.0)                 # nothing helps: leave the group alone

    def test_equal_harms_go_to_the_ratio_nearest_one(self):
        self.assertEqual(ed.alpha_grid({1.1: -0.001, 1.2: -0.001, 1.4: 0.0, 1.7: 0.0}), 1.1)
        self.assertEqual(ed.alpha_grid({1.1: 0.0, 1.2: 0.0, 1.4: 0.0, 1.7: 0.0}), 1.0)                         # a tie with the unedited model is the unedited model

    def test_the_smooth_estimator_recovers_the_vertex_of_a_quadratic_through_the_origin(self):
        for a_star in (1.15, 1.2, 1.5):
            q = quad(math.log(a_star)); harm = {a: q(a) for a in RATIOS}
            self.assertAlmostEqual(ed.alpha_smooth(harm), a_star, places=9)

    def test_the_smooth_estimator_is_clipped_to_the_range_of_the_ratios(self):
        q = quad(math.log(3.0)); self.assertAlmostEqual(ed.alpha_smooth({a: q(a) for a in RATIOS}), 1.7)
        q = quad(-0.3); self.assertEqual(ed.alpha_smooth({a: q(a) for a in RATIOS}), 1.0)

    def test_a_concave_or_straight_curve_goes_to_the_end_with_the_lower_fitted_harm(self):
        self.assertAlmostEqual(ed.alpha_smooth({a: -0.01 * math.log(a) for a in RATIOS}), 1.7)             # straight, falling
        self.assertEqual(ed.alpha_smooth({a: 0.01 * math.log(a) for a in RATIOS}), 1.0)                    # straight, rising
        self.assertAlmostEqual(ed.alpha_smooth({a: -0.05 * math.log(a) ** 2 - 0.02 * math.log(a) for a in RATIOS}), 1.7)   # concave, lower at 1.7 than at 1
        self.assertEqual(ed.alpha_smooth({a: -0.05 * math.log(a) ** 2 + 0.2 * math.log(a) for a in RATIOS}), 1.0)         # concave, higher at 1.7 than at 1

    def test_the_fit_uses_exactly_the_given_ratios_and_no_point_at_one(self):
        harm = {1.1: -0.001, 1.2: -0.003, 1.4: 0.002, 1.7: 0.02}
        x = np.log([1.1, 1.2, 1.4, 1.7]); y = np.array([harm[a] for a in (1.1, 1.2, 1.4, 1.7)])
        c1, c2 = np.linalg.lstsq(np.column_stack([x, x ** 2]), y, rcond=None)[0]
        self.assertAlmostEqual(ed.alpha_smooth(harm), math.exp(min(max(-c1 / (2 * c2), 0.0), math.log(1.7))), places=12)


class Statistics(unittest.TestCase):
    def test_spearman_equals_scipy_and_the_p_value_is_one_sided_with_the_plus_one_rule(self):
        x = np.arange(27.0); y = x + np.random.default_rng(1).normal(0, 6, 27)
        r = ed.spearman_perm(x, y, n_perm=2000, seed=0)
        self.assertAlmostEqual(r["rho"], stats.spearmanr(x, y)[0], places=12)
        r = ed.spearman_perm(x, x, n_perm=100, seed=0)
        self.assertAlmostEqual(r["p"], 1 / 101); self.assertAlmostEqual(r["rho"], 1.0)
        r = ed.spearman_perm(x, -x, n_perm=100, seed=0)
        self.assertEqual(r["p"], 1.0)                                                                       # every permutation is at least as large as -1

    def test_ties_use_average_ranks(self):
        x = np.array([1, 1, 1, 2, 2, 3, 3, 3, 4.0]); y = np.array([1, 2, 1, 3, 2, 3, 4, 5, 5.0])
        self.assertAlmostEqual(ed.spearman_perm(x, y, n_perm=50, seed=0)["rho"], stats.spearmanr(x, y)[0], places=12)

    def test_a_constant_vector_has_no_correlation_and_is_flagged(self):
        r = ed.spearman_perm(np.ones(10), np.arange(10.0), n_perm=50, seed=0)
        self.assertEqual((r["rho"], r["p"], r["constant"]), (0.0, 1.0, True))
        self.assertFalse(ed.spearman_perm(np.arange(10.0), np.arange(10.0), n_perm=50, seed=0)["constant"])

    def test_the_permutation_test_is_seeded_and_its_p_value_estimates_the_exact_one(self):
        x = np.arange(27.0); y = x + np.random.default_rng(2).normal(0, 30, 27)
        a = ed.spearman_perm(x, y, 3000, 0); b = ed.spearman_perm(x, y, 3000, 0); c = ed.spearman_perm(x, y, 3000, 1)
        self.assertEqual(a["p"], b["p"]); self.assertNotEqual(a["p"], c["p"])
        # the t approximation of a Spearman correlation: close to the permutation p for n = 27
        rho = a["rho"]; t = rho * math.sqrt(25 / (1 - rho ** 2)); p_t = stats.t.sf(t, 25)
        self.assertLess(abs(a["p"] - p_t), 0.02)

    def test_permutations_that_equal_the_observed_correlation_count_and_the_stated_seed_is_the_one_used(self):
        r = ed.spearman_perm(np.array([1.0, 2, 3]), np.array([1.0, 2, 3]), n_perm=6000, seed=0)
        self.assertLess(abs(r["p"] - 1 / 6), 0.03)                                                          # the identity permutation gives rho = 1 again: one of the six
        x = np.arange(12.0); y = x + np.random.default_rng(9).normal(0, 6, 12)
        got = ed.spearman_perm(x, y, n_perm=500, seed=0)
        rx = stats.rankdata(x) - 6.5; ry = stats.rankdata(y)
        perms = np.random.default_rng(0).permuted(np.tile(ry, (500, 1)), axis=1)                           # an independent replication of the documented draw
        rhos = np.array([stats.spearmanr(x, q)[0] for q in perms])
        self.assertAlmostEqual(got["p"], (1 + np.sum(rhos >= got["rho"] - 1e-12)) / 501)

    def test_the_bootstrap_interval_is_the_95_percent_interval_of_the_seeded_resamples(self):
        tau = np.linspace(1.07, 1.35, 12); alpha = np.round(tau + np.random.default_rng(3).normal(0, 0.06, 12), 1); tau_c = np.random.default_rng(4).permutation(tau)
        got = ed.boot_diff(alpha, tau, tau_c, n_boot=400, seed=0)
        idx = np.random.default_rng(0).integers(0, 12, size=(400, 12)); vals = []
        for i in idx:
            a = alpha[i]
            if np.ptp(a) == 0 or np.ptp(tau[i]) == 0 or np.ptp(tau_c[i]) == 0:
                continue
            vals.append(stats.spearmanr(a, tau[i])[0] - stats.spearmanr(a, tau_c[i])[0])
        self.assertAlmostEqual(got["lo"], float(np.percentile(vals, 2.5))); self.assertAlmostEqual(got["hi"], float(np.percentile(vals, 97.5))); self.assertEqual(got["n_nan"], 400 - len(vals))

    def test_the_split_is_by_the_parity_of_the_position(self):
        even, odd = ed.split_halves(7)
        self.assertEqual(list(even), [0, 2, 4, 6]); self.assertEqual(list(odd), [1, 3, 5])

    def test_the_reliability_is_the_spearman_brown_of_the_half_correlation_and_never_negative(self):
        a = np.arange(10.0); b = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 8.0])
        r = ed.reliability(a, b); rho_h = stats.spearmanr(a, b)[0]
        self.assertAlmostEqual(r["rho_half"], rho_h); self.assertAlmostEqual(r["sb"], 2 * rho_h / (1 + rho_h))
        r = ed.reliability(a, a[::-1].copy()); self.assertEqual(r["sb"], 0.0); self.assertLess(r["rho_half"], 0)
        r = ed.reliability(np.ones(10), a); self.assertEqual((r["rho_half"], r["sb"]), (0.0, 0.0))             # a constant half carries no information

    def test_the_corrected_correlation_divides_by_the_root_of_the_reliability_and_is_capped_at_one(self):
        self.assertAlmostEqual(ed.corrected(0.3, 0.25), 0.6); self.assertEqual(ed.corrected(0.9, 0.25), 1.0)
        self.assertIsNone(ed.corrected(0.3, 0.0))
        self.assertAlmostEqual(ed.corrected(-0.3, 0.25), -0.6)

    def test_the_bootstrap_difference_is_positive_when_the_group_target_is_the_informative_one(self):
        tau = np.linspace(1.07, 1.35, 27); alpha = np.round(tau, 1) + np.random.default_rng(3).normal(0, 0.02, 27)
        tau_c = np.random.default_rng(4).permutation(tau)
        d = ed.boot_diff(alpha, tau, tau_c, n_boot=2000, seed=0)
        self.assertGreater(d["diff"], 0.5); self.assertGreater(d["lo"], 0.0)
        d2 = ed.boot_diff(alpha, tau_c, tau, n_boot=2000, seed=0)
        self.assertLess(d2["diff"], -0.5); self.assertLess(d2["hi"], 0.0)
        self.assertAlmostEqual(d["diff"], stats.spearmanr(alpha, tau)[0] - stats.spearmanr(alpha, tau_c)[0], places=12)
        self.assertEqual(d, ed.boot_diff(alpha, tau, tau_c, n_boot=2000, seed=0))                           # seeded

    def test_resamples_with_a_constant_vector_are_counted_and_left_out(self):
        alpha = np.array([1.1, 1.1, 1.1, 1.1, 1.2]); tau = np.arange(5.0); d = ed.boot_diff(alpha, tau, tau[::-1].copy(), n_boot=500, seed=0)
        self.assertGreater(d["n_nan"], 0); self.assertTrue(np.isfinite(d["lo"]) and np.isfinite(d["hi"]))


class Verdict(unittest.TestCase):
    def test_the_boundaries_are_at_least_0_30_strictly_below_0_05_and_at_least_0_30(self):
        self.assertEqual(ed.verdict(0.30, 0.049, 0.30), "supported")
        self.assertEqual(ed.verdict(0.2999, 0.01, 0.9), "not supported")
        self.assertEqual(ed.verdict(0.5, 0.05, 0.9), "not supported")
        self.assertEqual(ed.verdict(0.5, 0.01, 0.2999), "uninformative")

    def test_an_unreliable_estimator_is_uninformative_whatever_the_correlation(self):
        self.assertEqual(ed.verdict(0.9, 0.0001, 0.1), "uninformative"); self.assertEqual(ed.verdict(-0.5, 0.9, 0.1), "uninformative")


def planted_langs(xstar_of, noise=0.0, seed=0, langs=LANGS27):
    """A Run of INT4 whose group harm in language l is quad(xstar_of(l)); per-sentence noise on the group harm."""
    rng = np.random.default_rng(seed); table = {}
    for c in langs:
        for a in RATIOS:
            table[(c, a)] = quad(xstar_of(c))(a) + rng.normal(0.0, noise, N)
    def harm(code, a, kind):
        return table[(code, a)] if kind == "g" else 0.0
    return planted_run("INT4", harm, langs=langs)


def baseline_rows(langs=LANGS27, spread=True, controls=0.86):
    out = {}
    for i, c in enumerate(langs):
        tau = 1.07 + 0.28 * i / (len(langs) - 1) if spread else 1.17
        out[c] = {"g_ratio": 1.0 / tau, "controls_ratio": controls}
    return out


def ext_of(run, langs=LANGS27):
    return ec.ExtRuns({"INT4": run}, RATIOS, n_controls=5, label="K64")


class Evaluate(unittest.TestCase):
    def test_harm_per_language_has_the_full_sample_and_the_two_halves(self):
        run = planted_langs(lambda c: math.log(1.2), noise=0.004, seed=1); e = ext_of(run)
        h = ed.per_language_harm(e, "INT4")
        self.assertEqual(list(h), LANGS27)
        c = LANGS27[3]
        dg, nt = run.delta(c, f"scale{1.2:g}_K64@{c}")
        self.assertAlmostEqual(h[c]["all"][1.2], float(dg.sum() / nt.sum()))
        self.assertAlmostEqual(h[c]["even"][1.2], float(dg[0::2].sum() / nt[0::2].sum())); self.assertAlmostEqual(h[c]["odd"][1.2], float(dg[1::2].sum() / nt[1::2].sum()))

    def test_an_optimum_that_follows_the_shrinkage_is_supported(self):
        xs = lambda c: math.log(1.07 + 0.28 * int(c[1:]) / 26)
        e = ext_of(planted_langs(xs)); res = ed.evaluate(ed.per_language_harm(e), baseline_rows(), e.languages, n_perm=2000, n_boot=500, seed=0)
        self.assertEqual(res["verdict"], "supported"); self.assertGreater(res["rd1"]["rho"], 0.8); self.assertLess(res["rd1"]["p"], 0.01)
        self.assertGreater(res["rd2"]["rho"], 0.95)                                                          # the smooth estimator follows it more closely than the grid
        self.assertAlmostEqual(res["rd4"]["sb"], 1.0)                                                        # no noise: the halves agree

    def test_an_optimum_unrelated_to_the_shrinkage_is_not_supported_when_the_estimator_is_reliable(self):
        perm = np.random.default_rng(7).permutation(27)
        xs = lambda c: math.log(1.07 + 0.28 * perm[int(c[1:])] / 26)
        e = ext_of(planted_langs(xs)); res = ed.evaluate(ed.per_language_harm(e), baseline_rows(), e.languages, n_perm=2000, n_boot=500, seed=0)
        self.assertEqual(res["verdict"], "not supported"); self.assertGreaterEqual(res["rd4"]["sb"], 0.3)

    def test_pure_noise_makes_the_estimator_unreliable_and_the_verdict_uninformative(self):
        e = ext_of(planted_langs(lambda c: 0.0, noise=0.03, seed=5)); res = ed.evaluate(ed.per_language_harm(e), baseline_rows(), e.languages, n_perm=500, n_boot=200, seed=0)
        self.assertEqual(res["verdict"], "uninformative"); self.assertLess(res["rd4"]["sb"], 0.3)

    def test_the_placebo_target_is_the_inverse_controls_ratio_and_the_corrected_correlation_is_reported(self):
        xs = lambda c: math.log(1.07 + 0.28 * int(c[1:]) / 26)
        base = baseline_rows(); [v.update(controls_ratio=0.80 + 0.001 * i) for i, v in enumerate(base.values())]
        e = ext_of(planted_langs(xs, noise=0.002, seed=2)); res = ed.evaluate(ed.per_language_harm(e), base, e.languages, n_perm=500, n_boot=300, seed=0)
        alpha = np.array([res["languages"][c]["alpha"] for c in e.languages]); tau_c = np.array([1.0 / base[c]["controls_ratio"] for c in e.languages])
        self.assertAlmostEqual(res["rd3"]["rho_placebo"], stats.spearmanr(alpha, tau_c)[0], places=12)
        self.assertAlmostEqual(res["rd3"]["diff"], res["rd1"]["rho"] - res["rd3"]["rho_placebo"], places=12)
        self.assertEqual(set(res["rd4"]), {"rho_half", "sb", "corrected"})
        ev = [res["languages"][c]["alpha_even"] for c in e.languages]; od = [res["languages"][c]["alpha_odd"] for c in e.languages]
        self.assertAlmostEqual(res["rd4"]["rho_half"], stats.spearmanr(ev, od)[0], places=12)               # the reliability is that of the two halves, not of a half against the whole

    def test_a_language_missing_from_the_baseline_is_an_error(self):
        e = ext_of(planted_langs(lambda c: 0.1)); base = baseline_rows(); del base["l4"]
        with self.assertRaisesRegex(KeyError, "l4"):
            ed.evaluate(ed.per_language_harm(e), base, e.languages, n_perm=50, n_boot=50, seed=0)

    def test_the_report_names_every_rule_the_verdict_and_the_limits(self):
        xs = lambda c: math.log(1.07 + 0.28 * int(c[1:]) / 26)
        e = ext_of(planted_langs(xs, noise=0.001, seed=3)); res = ed.evaluate(ed.per_language_harm(e), baseline_rows(), e.languages, n_perm=300, n_boot=200, seed=0)
        text = ed.report_text(res, e, {})
        for k in ("R-D1", "R-D2", "R-D3", "R-D4", "Post hoc", "27 languages", f"**{res['verdict']}**"):
            self.assertIn(k, text)

    def test_a_placebo_target_without_variation_is_reported_as_undefined_not_as_nan(self):
        xs = lambda c: math.log(1.07 + 0.28 * int(c[1:]) / 26)
        e = ext_of(planted_langs(xs, noise=0.001, seed=3)); res = ed.evaluate(ed.per_language_harm(e), baseline_rows(), e.languages, n_perm=100, n_boot=100, seed=0)    # controls_ratio equal for all languages
        self.assertEqual(res["rd3"]["n_nan"], 100)
        text = ed.report_text(res, e, {})
        self.assertNotIn("nan", text); self.assertIn("interval undefined", text)
        base = baseline_rows(); [v.update(controls_ratio=0.80 + 0.003 * i) for i, v in enumerate(base.values())]
        res = ed.evaluate(ed.per_language_harm(e), base, e.languages, n_perm=100, n_boot=100, seed=0)
        self.assertNotIn("interval undefined", ed.report_text(res, e, {}))


class Files(unittest.TestCase):
    def test_the_baseline_reader_takes_the_int4_rows(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "baseline.csv"
            rows = [{"language": "a", "state": "FP", "g_ratio": 1.0, "controls_ratio": 1.0}, {"language": "a", "state": "INT4", "g_ratio": 0.8, "controls_ratio": 0.85},
                    {"language": "b", "state": "INT4", "g_ratio": 0.9, "controls_ratio": 0.86}]
            with open(p, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
            b = ed.load_baseline(p, "INT4")
            self.assertEqual(b, {"a": {"g_ratio": 0.8, "controls_ratio": 0.85}, "b": {"g_ratio": 0.9, "controls_ratio": 0.86}})
            with self.assertRaises(ValueError):
                ed.load_baseline(p, "INT6")

    def test_main_runs_on_files_and_writes_the_outputs(self):
        xs = lambda c: math.log(1.07 + 0.28 * int(c[1:]) / 26)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            Loading().save(d / "INT4", planted_langs(xs, noise=0.001, seed=4))
            base = baseline_rows()
            with open(d / "baseline.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["language", "state", "g_ratio", "controls_ratio"]); w.writeheader()
                for c, v in base.items():
                    w.writerow({"language": c, "state": "INT4", **v})
            with contextlib.redirect_stdout(io.StringIO()) as out:
                ed.main(["--runs", str(d / "INT4"), "--baseline-csv", str(d / "baseline.csv"), "--out", str(d / "out"), "--n-perm", "300", "--n-boot", "200", "--expect-ids", ",".join(map(str, IDS)), "--forbid-ids", "0,1,2"])
            for f in ("report.md", "analysis.json", "alpha.csv"):
                self.assertTrue((d / "out" / f).exists(), f)
            js = json.loads((d / "out" / "analysis.json").read_text())
            self.assertEqual(js["verdict"], "supported"); self.assertIn("INT4", js["meta"]["runs"])
            with open(d / "out" / "alpha.csv") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 27); self.assertEqual(list(rows[0]), ["language", "tau", "tau_controls", "alpha", "alpha_smooth", "alpha_even", "alpha_odd", "harm_1.1", "harm_1.2", "harm_1.4", "harm_1.7"])
            self.assertIn("supported", out.getvalue())

    def test_a_run_without_int4_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            run = planted_run("INT6", lambda c, a, k: 0.0, langs=LANGS27); Loading().save(d / "INT6", run)
            with open(d / "baseline.csv", "w", newline="") as fh:
                fh.write("language,state,g_ratio,controls_ratio\n")
            with self.assertRaisesRegex(ValueError, "INT4"):
                ed.main(["--runs", str(d / "INT6"), "--baseline-csv", str(d / "baseline.csv"), "--out", str(d / "out"), "--expect-ids", ",".join(map(str, IDS)), "--forbid-ids", "0"])


if __name__ == "__main__":
    unittest.main()
