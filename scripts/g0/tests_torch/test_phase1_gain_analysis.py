import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_func_analysis as fa
import phase1_gain_analysis as ga
import phase1_gain_ratios as rt

LANGS = ["l0", "l1", "l2", "l3"]
DOWN, UP, FINE = [0.5, 0.25, 0.125], [2.0, 4.0, 8.0], [1.1, 1.2]
NTOK = np.array([10, 12, 8, 15, 11, 9, 14, 10])
N = len(NTOK)


def planted_run(label, harm, langs=LANGS, n_controls=2, noise=None, base=3.0, drop=(), noise_group=None):
    """A Run with planted pooled harms. harm(code, alpha, kind) -> harm in nats per token: kind "g" the group scaled by alpha (alpha 0 = zero-ablation), "c0".."c{R-1}" the control draws.
    noise: optional per-sentence additive harm noise array [N] (the same for every configuration, so differences stay exact). drop: configuration names left out."""
    e = np.zeros(N) if noise is None else np.asarray(noise)
    eg = np.zeros(N) if noise_group is None else np.asarray(noise_group)               # added to the group configurations only (the controls stay noise-free)
    items = {}
    def put(code, name, h, group=False):
        if name not in drop:
            items[f"{code}|{name}"] = {"nll": base * NTOK + (h + e + (eg if group else 0.0)) * NTOK, "ntok": NTOK}
    for code in langs:
        items[f"{code}|none"] = {"nll": base * NTOK, "ntok": NTOK}
        put(code, rt.zero_name("K64", code), harm(code, 0.0, "g"), True)
        for r in range(n_controls):
            put(code, rt.zeroed_control_name(r, "K64", code), harm(code, 0.0, f"c{r}"))
        for a in DOWN + UP + FINE:
            put(code, rt.group_name(a, "K64", code), harm(code, a, "g"), True)
            for r in range(n_controls):
                put(code, rt.control_name(a, r, "K64", code), harm(code, a, f"c{r}"))
    return fa.Run(label, items, {}, {}, list(langs), list(range(N)))


def simple_harm(code, a, kind):
    """Group harm grows with the depth of the turn-down and with the turn-up; the controls move a tenth as much."""
    depth = abs(np.log2(a)) if a > 0 else 6.0
    h = 0.01 * depth * (1.0 + 0.1 * LANGS.index(code))
    return h if kind == "g" else 0.1 * h


def runs(states=("FP", "INT4"), harm=simple_harm, **kw):
    return {s: planted_run(s, harm, **kw) for s in states}


def gains(rs=None, **kw):
    return ga.GainRuns(rs or runs(), DOWN, UP, FINE, n_controls=2, label="K64", **kw)


class Construct(unittest.TestCase):
    def test_states_are_ordered_along_the_ladder(self):
        g = gains(runs(("INT4", "FP", "INT6")))
        self.assertEqual(g.states, ["FP", "INT6", "INT4"])
        self.assertEqual(g.languages, LANGS)

    def test_the_full_precision_run_is_required(self):
        with self.assertRaises(ValueError):
            gains(runs(("INT4", "INT6")))

    def test_an_unknown_state_is_refused(self):
        with self.assertRaises(ValueError):
            gains(runs(("FP", "INT3")))

    def test_a_missing_configuration_is_named(self):
        rs = runs(); rs["INT4"] = planted_run("INT4", simple_harm, drop=(rt.control_name(4.0, 1, "K64", "l2"),))
        with self.assertRaises(KeyError) as cm:
            gains(rs)
        self.assertIn("cscale4.1_K64@l2", str(cm.exception)); self.assertIn("INT4", str(cm.exception))

    def test_a_missing_zero_ablation_is_named(self):
        rs = runs(); rs["FP"] = planted_run("FP", simple_harm, drop=(rt.zero_name("K64", "l0"),))
        with self.assertRaises(KeyError):
            gains(rs)

    def test_different_sentence_ids_are_refused(self):
        rs = runs(); r = rs["INT4"]; rs["INT4"] = fa.Run("INT4", {f"{c}|{n}": {"nll": v[0], "ntok": v[1]} for (c, n), v in r._items.items()}, {}, {}, LANGS, list(range(1, N + 1)))
        with self.assertRaises(ValueError):
            gains(rs)

    def test_different_languages_are_refused(self):
        rs = runs(); rs["INT4"] = planted_run("INT4", simple_harm, langs=LANGS[:3])
        with self.assertRaises(ValueError):
            gains(rs)

    def test_bad_ratio_lists_are_refused(self):
        with self.assertRaises(ValueError):
            ga.GainRuns(runs(), [0.5, 0.25], [2.0, 4.0, 8.0], FINE, n_controls=2)       # not three down ratios
        with self.assertRaises(ValueError):
            ga.GainRuns(runs(), DOWN, [0.9, 4.0, 8.0], FINE, n_controls=2)               # an up ratio below 1
        with self.assertRaises(ValueError):
            ga.GainRuns(runs(), [1.5, 0.25, 0.125], UP, FINE, n_controls=2)              # a down ratio above 1

    def test_all_ratios_lists_every_configuration_ratio_once(self):
        self.assertEqual(sorted(gains().all_ratios), sorted(DOWN + UP + FINE))


class TableTest(unittest.TestCase):
    def setUp(self):
        self.g = gains()
        self.t = ga.build_table(self.g, n_boot=50, seed=0)

    def test_point_values_are_the_planted_harms(self):
        for s in ("FP", "INT4"):
            for c in LANGS:
                for a in DOWN + UP + FINE + [0.0]:
                    cell = self.t.get(s, c, a)
                    self.assertAlmostEqual(cell["g"], simple_harm(c, a, "g")); self.assertAlmostEqual(cell["c"], simple_harm(c, a, "c0"))
                    self.assertAlmostEqual(cell["net"], cell["g"] - cell["c"])

    def test_alpha_zero_uses_the_zero_ablation_and_the_zeroed_controls(self):
        h = lambda code, a, k: (0.7 if k == "g" else 0.1 + 0.2 * int(k[1:])) if a == 0.0 else simple_harm(code, a, k)
        t = ga.build_table(gains(runs(harm=h)), n_boot=20, seed=0)
        cell = t.get("FP", "l1", 0.0)
        self.assertAlmostEqual(cell["g"], 0.7); self.assertAlmostEqual(cell["c"], 0.2)             # mean of 0.1 and 0.3

    def test_samples_have_one_value_per_draw_and_are_seeded(self):
        a = self.t.get("FP", "l0", 0.5)
        self.assertEqual(a["g_s"].shape, (50,))
        t2 = ga.build_table(self.g, n_boot=50, seed=0)
        np.testing.assert_array_equal(t2.get("FP", "l0", 0.5)["net_s"], a["net_s"])
        e = np.random.default_rng(3).normal(0, 0.05, N)
        tn = ga.build_table(gains(runs(noise=e)), n_boot=50, seed=0); tn1 = ga.build_table(gains(runs(noise=e)), n_boot=50, seed=1)
        self.assertFalse(np.array_equal(tn.get("FP", "l0", 0.5)["g_s"], tn1.get("FP", "l0", 0.5)["g_s"]))

    def test_the_same_draws_serve_every_state_so_equal_data_gives_equal_samples(self):
        e = np.random.default_rng(3).normal(0, 0.05, N)
        t = ga.build_table(gains(runs(noise=e)), n_boot=50, seed=0)
        np.testing.assert_array_equal(t.get("FP", "l2", 4.0)["g_s"], t.get("INT4", "l2", 4.0)["g_s"])

    def test_the_bootstrap_interval_contains_the_point_for_noisy_data(self):
        e = np.random.default_rng(3).normal(0, 0.05, N)
        t = ga.build_table(gains(runs(noise=e)), n_boot=400, seed=0)
        cell = t.get("FP", "l0", 0.5)
        lo, hi = np.percentile(cell["g_s"], [2.5, 97.5])
        self.assertLess(lo, cell["g"]); self.assertGreater(hi, cell["g"])

    def test_base_loss_is_the_pooled_unedited_loss_of_the_state(self):
        self.assertAlmostEqual(self.t.base_loss("FP", "l0"), 3.0)

    def test_total_change_is_the_state_baseline_plus_harm_minus_the_fp_baseline(self):
        rs = runs()
        rs["INT4"] = planted_run("INT4", simple_harm, base=3.5)
        t = ga.build_table(gains(rs), n_boot=20, seed=0)
        self.assertAlmostEqual(t.total_change("INT4", "l0", 0.5), 3.5 + simple_harm("l0", 0.5, "g") - 3.0)
        self.assertAlmostEqual(t.total_change("FP", "l0", 0.5), simple_harm("l0", 0.5, "g"))

    def test_ratio_one_is_the_unedited_model(self):
        cell = self.t.get("FP", "l0", 1.0)
        self.assertEqual(cell["g"], 0.0); self.assertEqual(cell["net"], 0.0)
        self.assertTrue(np.all(cell["net_s"] == 0.0))

    def test_an_unknown_ratio_is_an_error(self):
        with self.assertRaises(KeyError):
            self.t.get("FP", "l0", 0.3)


def table_of(rs=None, noise=None, n_boot=200, **kw):
    g = gains(rs if rs is not None else runs(**({"noise": noise} if noise is not None else {})), **kw)
    return g, ga.build_table(g, n_boot=n_boot, seed=0)


class Required(unittest.TestCase):
    def test_the_threshold_is_the_design_count_at_27_languages_and_proportional_otherwise(self):
        self.assertEqual(ga.required(27, 24), 24); self.assertEqual(ga.required(27, 20), 20); self.assertEqual(ga.required(27, 14), 14)
        self.assertEqual(ga.required(4, 24), 4); self.assertEqual(ga.required(4, 14), 3)


class HG1(unittest.TestCase):
    def test_all_languages_strictly_ordered_is_met(self):
        g, t = table_of()
        r = ga.h_g1(t, g, "FP")
        self.assertEqual(r["agree"], 4); self.assertTrue(r["median_ordered"]); self.assertTrue(r["met"])
        self.assertTrue(all(v == 1.0 for v in r["spearman"].values()))
        self.assertEqual(len(r["median_net"]), 4)

    def test_one_language_out_of_order_fails_at_four_of_four(self):
        h = lambda c, a, k: (0.5 if (c == "l3" and a == 0.25 and k == "g") else simple_harm(c, a, k))
        g, t = table_of(runs(harm=h))
        r = ga.h_g1(t, g, "FP")
        self.assertEqual(r["agree"], 3); self.assertFalse(r["per_language"]["l3"]); self.assertFalse(r["met"])

    def test_zero_ablation_weaker_than_the_deepest_ratio_breaks_the_chain(self):
        h = lambda c, a, k: (0.001 if (c == "l1" and a == 0.0 and k == "g") else simple_harm(c, a, k))
        g, t = table_of(runs(harm=h))
        self.assertFalse(ga.h_g1(t, g, "FP")["per_language"]["l1"])

    def test_equal_net_harms_are_not_strictly_ordered(self):
        h = lambda c, a, k: (0.02 if (a in (0.5, 0.25, 0.125) and k == "g") else (0.0 if a in (0.5, 0.25, 0.125) else simple_harm(c, a, k)))
        g, t = table_of(runs(harm=h))
        r = ga.h_g1(t, g, "FP")
        self.assertEqual(r["agree"], 0); self.assertFalse(r["median_ordered"]); self.assertFalse(r["met"])

    def test_it_can_be_evaluated_at_another_state(self):
        g, t = table_of()
        self.assertEqual(ga.h_g1(t, g, "INT4")["state"], "INT4")


class HG2(unittest.TestCase):
    def test_net_harm_with_an_interval_above_zero_at_the_deepest_ratio(self):
        g, t = table_of()
        r = ga.h_g2(t, g, "FP")
        self.assertEqual(r["count"], 4); self.assertTrue(r["met"])

    def test_a_language_without_net_harm_is_not_counted(self):
        h = lambda c, a, k: (0.0 if (c == "l2" and a == 0.125 and k == "g") else simple_harm(c, a, k))
        g, t = table_of(runs(harm=h))
        r = ga.h_g2(t, g, "FP")
        self.assertEqual(r["count"], 3); self.assertFalse(r["per_language"]["l2"]); self.assertFalse(r["met"])

    def test_an_interval_that_reaches_below_zero_is_not_counted(self):
        e = np.random.default_rng(5).normal(0, 0.2, N)
        h = lambda c, a, k: (0.1 * simple_harm(c, a, "g") + 0.001 if (c == "l0" and a == 0.125 and k == "g") else simple_harm(c, a, k))        # net harm 0.001, group noise 0.2
        g, t = table_of(runs(harm=h, noise_group=e), n_boot=400)
        self.assertFalse(ga.h_g2(t, g, "FP")["per_language"]["l0"])


class HG3(unittest.TestCase):
    def test_up_harm_increasing_and_beyond_the_controls(self):
        g, t = table_of()
        r = ga.h_g3(t, g, "FP")
        self.assertEqual(r["count_increasing"], 4); self.assertEqual(r["count_net_positive"], 4); self.assertTrue(r["met"])

    def test_a_decreasing_curve_is_not_increasing(self):
        h = lambda c, a, k: (0.05 - 0.01 * np.log2(a) if (c == "l0" and a > 1.5 and k == "g") else simple_harm(c, a, k))
        g, t = table_of(runs(harm=h))
        r = ga.h_g3(t, g, "FP")
        self.assertFalse(r["per_language_increasing"]["l0"]); self.assertEqual(r["count_increasing"], 3); self.assertFalse(r["met"])

    def test_controls_as_harmful_as_the_group_at_the_largest_ratio_fail_the_second_part(self):
        h = lambda c, a, k: (simple_harm(c, a, "g") if (a == 8.0 and c in ("l0", "l1")) else simple_harm(c, a, k))
        g, t = table_of(runs(harm=h))
        r = ga.h_g3(t, g, "FP")
        self.assertEqual(r["count_increasing"], 4); self.assertEqual(r["count_net_positive"], 2); self.assertFalse(r["met"])


def ladder_runs(factors, states=("FP", "INT6", "INT4"), **kw):
    """Runs whose harms are the simple ones times a factor per state (the per-language modifier `tweak(code, state, h)` is applied last)."""
    tweak = kw.pop("tweak", lambda c, s, a, k, h: h)
    out = {}
    for s, f in zip(states, factors):
        out[s] = planted_run(s, lambda c, a, k, s=s, f=f: tweak(c, s, a, k, f * simple_harm(c, a, k)), **kw)
    return out


class HG4(unittest.TestCase):
    def test_harm_growing_along_the_ladder_is_met(self):
        g, t = table_of(ladder_runs((1.0, 1.1, 1.6)))
        r = ga.h_g4(t, g)
        self.assertEqual(r["count_positive"], 4); self.assertTrue(r["median_ok"]); self.assertTrue(r["met"])
        self.assertEqual(set(r["per_ratio"]), set(DOWN))
        self.assertTrue(all(v == 1.0 for v in r["per_ratio"][0.125]["tau"].values()))

    def test_one_language_whose_harm_falls_along_the_ladder_fails(self):
        tw = lambda c, s, a, k, h: (h / (1.0 + ("FP", "INT6", "INT4").index(s)) if (c == "l2" and k == "g") else h)
        g, t = table_of(ladder_runs((1.0, 1.1, 1.6), tweak=tw))
        r = ga.h_g4(t, g)
        self.assertEqual(r["count_positive"], 3); self.assertFalse(r["met"])

    def test_a_dip_of_the_median_larger_than_the_bootstrap_error_fails_a_smaller_one_does_not(self):
        e = np.random.default_rng(5).normal(0, 0.05, N)
        g, t = table_of(ladder_runs((1.0, 0.2, 1.6), noise=e), n_boot=400)
        self.assertFalse(ga.h_g4(t, g)["median_ok"])
        g2, t2 = table_of(ladder_runs((1.0, 0.99, 1.6), noise=e), n_boot=400)
        self.assertTrue(ga.h_g4(t2, g2)["median_ok"])

    def test_with_only_two_states_the_rule_still_runs(self):
        g, t = table_of(ladder_runs((1.0, 1.5), states=("FP", "INT4")))
        self.assertTrue(ga.h_g4(t, g)["met"])

    def test_se_is_reported_for_every_state(self):
        g, t = table_of(ladder_runs((1.0, 1.1, 1.6)))
        r = ga.h_g4(t, g)
        self.assertEqual(set(r["se_by_state"]), {"FP", "INT6", "INT4"}); self.assertEqual(set(r["median_by_state"]), {"FP", "INT6", "INT4"})

def two_states(h_fp, h_int4, **kw):
    return {"FP": planted_run("FP", h_fp, **kw), "INT4": planted_run("INT4", h_int4, **kw)}


def repair(g_val, c_val, ratios=(1.1, 1.2)):
    """INT4 harm function: the group at the fine ratios has harm g_val, every control c_val; otherwise the simple harms."""
    def h(c, a, k):
        if a in ratios:
            return g_val if k == "g" else c_val
        return simple_harm(c, a, k)
    return h


class HG5(unittest.TestCase):
    def test_generic_repair_without_a_group_effect(self):
        g, t = table_of(two_states(simple_harm, repair(-0.02, -0.02)))
        r = ga.h_g5(t, g, n_boot=200)
        self.assertTrue(r["testable"]); self.assertTrue(r["a"]["met"]); self.assertFalse(r["b"]["met"]); self.assertEqual(r["b"]["count"], 0)
        self.assertAlmostEqual(r["a"]["per_ratio"][1.1]["median"], -0.02)

    def test_group_specific_repair(self):
        g, t = table_of(two_states(simple_harm, repair(-0.03, -0.01)))
        r = ga.h_g5(t, g, n_boot=200)
        self.assertTrue(r["a"]["met"]); self.assertTrue(r["b"]["met"]); self.assertEqual(r["b"]["count"], 4)

    def test_no_repair_when_amplification_hurts(self):
        g, t = table_of(two_states(simple_harm, repair(0.02, 0.01)))
        r = ga.h_g5(t, g, n_boot=200)
        self.assertFalse(r["a"]["met"]); self.assertFalse(r["b"]["met"])

    def test_the_better_of_the_two_fine_ratios_is_used_per_language(self):
        def h(c, a, k):
            if k == "g" and a == 1.2:
                return -0.03
            if k == "g" and a == 1.1:
                return 0.02
            return 0.0 if a in (1.1, 1.2) else simple_harm(c, a, k)
        g, t = table_of(two_states(simple_harm, h))
        r = ga.h_g5(t, g, n_boot=200)
        self.assertEqual(set(r["b"]["chosen"].values()), {1.2})

    def test_generic_repair_needs_the_interval_to_exclude_zero(self):
        g, t = table_of(two_states(simple_harm, repair(0.0, 0.0)))
        self.assertFalse(ga.h_g5(t, g, n_boot=200)["a"]["met"])

    def test_the_optimal_gain_is_the_ratio_with_the_lowest_loss(self):
        g, t = table_of(two_states(simple_harm, repair(-0.05, 0.0)))
        r = ga.h_g5(t, g, n_boot=50)
        self.assertEqual(set(r["alpha_star"]["INT4"].values()), {1.1})            # equal harm at 1.1 and 1.2: the one nearer to 1
        self.assertEqual(set(r["alpha_star"]["FP"].values()), {1.0})              # nothing beats leaving the group alone
        self.assertEqual(r["alpha_up_count"], {"FP": 0, "INT4": 4})

    def test_a_single_best_ratio_is_found(self):
        def h(c, a, k):
            return -0.05 if (k == "g" and a == 1.2) else (-0.01 if (k == "g" and a == 1.1) else (0.0 if a in (1.1, 1.2) else simple_harm(c, a, k)))
        g, t = table_of(two_states(simple_harm, h))
        self.assertEqual(set(ga.h_g5(t, g, n_boot=50)["alpha_star"]["INT4"].values()), {1.2})

    def test_alpha_star_against_the_inverse_activation_ratio(self):
        def h(c, a, k):
            best = {"l0": 1.2, "l1": 1.2, "l2": 1.1, "l3": 1.1}[c]
            if k == "g" and a == best:
                return -0.05
            return 0.0 if a in (1.1, 1.2) else simple_harm(c, a, k)
        g, t = table_of(two_states(simple_harm, h))
        r = ga.h_g5(t, g, baseline_g_ratio={"l0": 0.7, "l1": 0.8, "l2": 0.9, "l3": 1.0}, n_boot=50)
        self.assertGreater(r["corr"]["rho"], 0.3); self.assertTrue(r["corr"]["met"]); self.assertEqual(r["corr"]["n"], 4)
        r2 = ga.h_g5(t, g, baseline_g_ratio={"l0": 1.0, "l1": 0.9, "l2": 0.8, "l3": 0.7}, n_boot=50)
        self.assertLess(r2["corr"]["rho"], 0); self.assertFalse(r2["corr"]["met"])
        self.assertIsNone(ga.h_g5(t, g, n_boot=50)["corr"])

    def test_without_the_int4_run_the_rule_is_not_testable(self):
        g, t = table_of(runs(("FP", "INT6")))
        self.assertFalse(ga.h_g5(t, g, n_boot=20)["testable"])


class HG6(unittest.TestCase):
    def test_same_sign_in_every_language_and_the_same_ranking_is_met(self):
        g, t = table_of()
        r = ga.h_g6(t, g)
        self.assertEqual(r["per_state"]["FP"]["count_positive"], 4); self.assertEqual(r["per_state"]["INT4"]["count_positive"], 4)
        self.assertAlmostEqual(r["rank_corr"], 1.0); self.assertTrue(r["met"])

    def test_a_reversed_ranking_between_fp_and_int4_fails(self):
        h_int4 = lambda c, a, k: simple_harm(c, a, k) * (1.0 - 0.2 * LANGS.index(c)) if k == "g" else simple_harm(c, a, k)
        g, t = table_of(two_states(simple_harm, h_int4))
        r = ga.h_g6(t, g)
        self.assertLess(r["rank_corr"], 0.5); self.assertFalse(r["met"])

    def test_one_language_without_net_harm_fails_the_sign_count(self):
        h = lambda c, a, k: (-0.01 if (c == "l1" and k == "g" and a == 0.125) else simple_harm(c, a, k))
        g, t = table_of(runs(harm=h))
        r = ga.h_g6(t, g)
        self.assertEqual(r["per_state"]["FP"]["count_positive"], 3); self.assertFalse(r["met"])

    def test_the_spread_of_the_deepest_net_harm_over_zero_ablation_is_reported(self):
        h = lambda c, a, k: (2 * simple_harm(c, a, k) if (c == "l0" and a == 0.0) else simple_harm(c, a, k))
        g, t = table_of(runs(harm=h))
        r = ga.h_g6(t, g)
        self.assertAlmostEqual(r["iqr_fraction"]["FP"], float(np.subtract(*np.percentile([0.25, 0.5, 0.5, 0.5], [75, 25]))))

import contextlib
import csv
import io
import json
import tempfile

import phase1_gain_run as gr


def save_state(d, run, preset, state=None, meta=None, complete=True, skipped=(), gates=None):
    """Write func_lm_eval_<state>.pt and meta_lm_eval.json of one toy job into directory d."""
    import torch
    d = Path(d); d.mkdir(parents=True, exist_ok=True)
    state = state or run.label
    items = {f"{c}|{n}": {"nll": torch.tensor(v[0]), "ntok": torch.tensor(v[1])} for (c, n), v in run._items.items()}
    torch.save({"precision": state, "split": "eval", "ids": list(run.ids), "languages": list(run.languages), "preset": preset, "items": items, "stats": {}, "fidelity": {}}, d / f"func_lm_eval_{state}.pt")
    (d / "meta_lm_eval.json").write_text(json.dumps(meta or {"complete": complete, "skipped_for_time": list(skipped), "preset": preset,
                                                             "gates": gates or {"split_equals_full": True, "scale_one_identity": True, "donor_self_identity": True, "sham_identity": True}}))


PRESET = gr.preset_name(DOWN + UP + FINE, 2, "K64")


def ratios_json(d, down=DOWN, up=UP, fine=FINE):
    p = Path(d) / "ratios.json"
    p.write_text(json.dumps({"down": down, "up": up, "fine_up": fine}))
    return p


class Loading(unittest.TestCase):
    def test_runs_in_separate_job_directories_are_collected(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for s in ("FP", "INT4"):
                save_state(d / s, planted_run(s, simple_harm), PRESET)
            g, info = ga.load_runs([d / "FP", d / "INT4"], ratios_json(d))
            self.assertEqual(g.states, ["FP", "INT4"]); self.assertEqual(g.n_controls, 2); self.assertEqual(g.label, "K64")
            self.assertEqual(sorted(info), ["FP", "INT4"]); self.assertEqual(len(info["FP"]["sha256"]), 64)

    def test_a_preset_whose_ratios_differ_from_the_decision_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            save_state(d / "FP", planted_run("FP", simple_harm), gr.preset_name(DOWN + UP + [1.1, 1.3], 2, "K64"))
            with self.assertRaises(ValueError):
                ga.load_runs([d / "FP"], ratios_json(d))

    def test_states_with_different_presets_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            save_state(d / "FP", planted_run("FP", simple_harm), PRESET)
            save_state(d / "INT4", planted_run("INT4", simple_harm), gr.preset_name(DOWN + UP + FINE, 3, "K64"))
            with self.assertRaises(ValueError):
                ga.load_runs([d / "FP", d / "INT4"], ratios_json(d))

    def test_an_incomplete_job_a_time_skip_or_a_failed_gate_is_refused(self):
        for kw in ({"complete": False}, {"skipped": ["INT4"]}, {"gates": {"split_equals_full": True, "scale_one_identity": False}}):
            with tempfile.TemporaryDirectory() as d:
                d = Path(d)
                save_state(d / "FP", planted_run("FP", simple_harm), PRESET, **kw)
                with self.assertRaises(ValueError, msg=str(kw)):
                    ga.load_runs([d / "FP"], ratios_json(d))

    def test_no_result_file_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                ga.load_runs([Path(d)], ratios_json(d))

    def test_the_baseline_ratio_is_read_for_int4(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "baseline.csv"
            p.write_text("language,state,g_ratio\nl0,FP,1.0\nl0,INT4,0.85\nl1,FP,1.0\nl1,INT4,0.9\n")
            self.assertEqual(ga.load_baseline_g_ratio(p), {"l0": 0.85, "l1": 0.9})


class Outputs(unittest.TestCase):
    def setUp(self):
        self.g, self.t = table_of(two_states(simple_harm, repair(-0.03, -0.01)))
        self.res = ga.evaluate(self.t, self.g, baseline_g_ratio={"l0": 0.8, "l1": 0.85, "l2": 0.9, "l3": 0.95}, n_boot=100)

    def test_evaluate_returns_every_rule_and_a_summary_of_verdicts(self):
        for k in ("H-G1", "H-G2", "H-G3", "H-G4", "H-G5", "H-G6"):
            self.assertIn(k, self.res["rules"])
        v = self.res["verdicts"]
        self.assertEqual(set(v), {"H-G1", "H-G2", "H-G3", "H-G4", "H-G5a", "H-G5b", "H-G5-alpha", "H-G6"})
        self.assertTrue(all(isinstance(x, (bool, type(None))) for x in v.values()))
        self.assertTrue(v["H-G1"]); self.assertTrue(v["H-G5a"]); self.assertTrue(v["H-G5b"])

    def test_the_rules_for_one_state_only_use_the_stated_state_for_the_verdict(self):
        self.assertEqual(self.res["rules"]["H-G1"]["FP"]["state"], "FP"); self.assertIn("INT4", self.res["rules"]["H-G1"])

    def test_curve_rows_cover_every_state_language_and_ratio(self):
        rows = ga.curve_rows(self.t, self.g)
        self.assertEqual(len(rows), 2 * 4 * (len(DOWN + UP + FINE) + 1))
        r = [x for x in rows if x["state"] == "FP" and x["language"] == "l1" and x["ratio"] == 0.5][0]
        self.assertAlmostEqual(r["group_harm"], simple_harm("l1", 0.5, "g")); self.assertLess(r["net_lo"], r["net_hi"] + 1e-12)
        self.assertIn("total_change", r)

    def test_report_names_every_hypothesis_and_its_verdict(self):
        text = ga.report_text(self.res, self.t, self.g, {"runs": {}})
        for k in ("H-G1", "H-G2", "H-G3", "H-G4", "H-G5a", "H-G5b", "H-G6"):
            self.assertIn(k, text)
        self.assertIn("met", text); self.assertIn("| FP |", text); self.assertIn("l3", text)

    def test_the_report_says_not_met_for_a_failed_rule(self):
        g, t = table_of(two_states(simple_harm, repair(0.02, 0.01)))
        res = ga.evaluate(t, g, n_boot=50)
        self.assertIn("**not met**", ga.report_text(res, t, g, {"runs": {}}))

    def test_write_outputs_files(self):
        with tempfile.TemporaryDirectory() as d:
            ga.write_outputs(d, self.res, self.t, self.g, {"runs": {"FP": {"sha256": "a"}}})
            for f in ("analysis.json", "curves.csv", "alpha_star.csv", "report.md"):
                self.assertTrue((Path(d) / f).exists(), f)
            js = json.loads((Path(d) / "analysis.json").read_text())
            self.assertEqual(js["verdicts"]["H-G1"], True); self.assertEqual(js["meta"]["runs"], {"FP": {"sha256": "a"}})
            with open(Path(d) / "alpha_star.csv") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 2 * 4)

    def test_main_runs_on_files_and_writes_the_analysis(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for s in ("FP", "INT4"):
                save_state(d / s, planted_run(s, simple_harm), PRESET)
            with contextlib.redirect_stdout(io.StringIO()):
                ga.main(["--runs", str(d / "FP"), str(d / "INT4"), "--ratios-json", str(ratios_json(d)), "--out", str(d / "out"), "--n-boot", "50"])
            js = json.loads((d / "out" / "analysis.json").read_text())
            self.assertTrue(js["verdicts"]["H-G1"]); self.assertEqual(js["meta"]["n_boot"], 50); self.assertEqual(set(js["meta"]["runs"]), {"FP", "INT4"})

class MoreConstruct(unittest.TestCase):
    def test_a_ratio_of_exactly_one_is_not_a_down_or_up_ratio(self):
        with self.assertRaises(ValueError):
            ga.GainRuns(runs(), [1.0, 0.25, 0.125], UP, FINE, n_controls=2)
        with self.assertRaises(ValueError):
            ga.GainRuns(runs(), DOWN, [1.0, 4.0, 8.0], FINE, n_controls=2)
        with self.assertRaises(ValueError):
            ga.GainRuns(runs(), DOWN, UP, [1.0, 1.2], n_controls=2)


class Edges(unittest.TestCase):
    def test_a_net_harm_of_exactly_zero_is_not_positive_nor_negative(self):
        same = lambda c, a, k: (0.1 * simple_harm(c, a, "g") if (c == "l1" and a in (0.125, 8.0)) else simple_harm(c, a, k))
        hh = lambda c, a, k: (0.1 * simple_harm(c, a, "g") if (c == "l1" and a in (0.125, 8.0) and k == "g") else simple_harm(c, a, k))
        g, t = table_of(runs(harm=hh))
        self.assertFalse(ga.h_g2(t, g, "FP")["per_language"]["l1"])
        self.assertFalse(ga.h_g3(t, g, "FP")["per_language_net_positive"]["l1"])
        self.assertEqual(ga.h_g6(t, g)["per_state"]["FP"]["count_positive"], 3)

    def test_equal_group_harm_at_two_up_ratios_is_not_increasing(self):
        h = lambda c, a, k: (0.04 if (c == "l0" and a in (2.0, 4.0) and k == "g") else simple_harm(c, a, k))
        g, t = table_of(runs(harm=h))
        self.assertFalse(ga.h_g3(t, g, "FP")["per_language_increasing"]["l0"])

    def test_a_kendall_tau_of_exactly_zero_is_not_positive(self):
        hump = lambda c, s, a, k, h: ({"FP": 1.0, "INT6": 2.0, "INT4": 1.0}[s] * 0.03 if (c == "l1" and k == "g" and a == 0.125) else h)
        g, t = table_of(ladder_runs((1.0, 1.1, 1.6), tweak=hump))
        r = ga.h_g4(t, g)
        self.assertEqual(r["per_ratio"][0.125]["tau"]["l1"], 0.0); self.assertEqual(r["count_positive"], 3)

    def test_the_tolerance_is_the_larger_of_the_two_standard_errors(self):
        e = np.random.default_rng(5).normal(0, 0.05, N)
        rs = {"FP": planted_run("FP", simple_harm), "INT6": planted_run("INT6", lambda c, a, k: 1.12 * simple_harm(c, a, k), noise=e),
              "INT4": planted_run("INT4", lambda c, a, k: 1.6 * simple_harm(c, a, k))}
        g, t = table_of(rs, n_boot=400)
        r = ga.h_g4(t, g)
        self.assertLess(r["se_by_state"]["FP"], 1e-9); self.assertGreater(r["se_by_state"]["INT6"], 0.01)
        self.assertTrue(r["median_ok"])           # the noise shifts INT6 by about -0.012, so its median (0.0265) is 0.008 below FP (0.0345): within the larger error (0.0138), not within the smaller (0)


LANGS27 = [f"l{i}" for i in range(27)]


def harm27(c, a, k):
    depth = abs(np.log2(a)) if a > 0 else 6.0
    h = 0.01 * depth * (1.0 + 0.01 * int(c[1:]))
    return h if k == "g" else 0.1 * h


class Boundaries(unittest.TestCase):
    """The counts of the design at 27 languages: 24 passes, 23 does not (20 and 14 for the other two rules)."""
    def table27(self, harm, states=("FP",), **kw):
        rs = {s: planted_run(s, harm, langs=LANGS27, **kw) for s in states}
        g = ga.GainRuns(rs, DOWN, UP, FINE, n_controls=2)
        return g, ga.build_table(g, n_boot=50, seed=0)

    def break_languages(self, k, kind_a, new):
        """A harm function that replaces the group harm (ratio kind_a) of the first k languages by `new`."""
        return lambda c, a, kind: (new if (int(c[1:]) < k and kind == "g" and a == kind_a) else harm27(c, a, kind))

    def test_h_g1_needs_24_of_27(self):
        for k, expected in ((3, True), (4, False)):
            g, t = self.table27(self.break_languages(k, 0.25, 0.9))                 # net harm at 0.25 above that at 0.125: out of order
            self.assertEqual(ga.h_g1(t, g, "FP")["met"], expected, msg=f"{k} broken")

    def test_h_g2_needs_24_of_27(self):
        for k, expected in ((3, True), (4, False)):
            g, t = self.table27(self.break_languages(k, 0.125, 0.0))
            self.assertEqual(ga.h_g2(t, g, "FP")["met"], expected, msg=f"{k} broken")

    def test_h_g3_needs_24_for_the_increase_and_20_for_the_excess(self):
        for k, expected in ((3, True), (4, False)):
            g, t = self.table27(self.break_languages(k, 4.0, 0.0))                 # group harm at 4 below that at 2: not increasing
            self.assertEqual(ga.h_g3(t, g, "FP")["met"], expected, msg=f"increase {k}")
        for k, expected in ((7, True), (8, False)):
            g, t = self.table27(lambda c, a, kind: (harm27(c, a, "g") if (int(c[1:]) < k and a == 8.0) else harm27(c, a, kind)))     # the controls as harmful as the group at 8
            self.assertEqual(ga.h_g3(t, g, "FP")["met"], expected, msg=f"excess {k}")

    def test_h_g4_needs_24_of_27(self):
        for k, expected in ((3, True), (4, False)):
            h = lambda c, s, a, kind, hh: (hh / (1.0 + ("FP", "INT6", "INT4").index(s)) if (int(c[1:]) < k and kind == "g") else hh)
            rs = {s: planted_run(s, lambda c, a, kind, s=s, f=f: h(c, s, a, kind, f * harm27(c, a, kind)), langs=LANGS27) for s, f in zip(("FP", "INT6", "INT4"), (1.0, 1.1, 1.6))}
            gg = ga.GainRuns(rs, DOWN, UP, FINE, n_controls=2); tt = ga.build_table(gg, n_boot=50, seed=0)
            self.assertEqual(ga.h_g4(tt, gg)["met"], expected, msg=f"{k} broken")

    def test_h_g5b_needs_14_of_27(self):
        for k, expected in ((14, True), (13, False)):
            def h(c, a, kind):
                if a in (1.1, 1.2):
                    return (-0.03 if kind == "g" else -0.01) if int(c[1:]) < k else (0.01 if kind == "g" else 0.0)
                return harm27(c, a, kind)
            g, t = self.table27(harm27, states=("FP",))
            rs = {"FP": planted_run("FP", harm27, langs=LANGS27), "INT4": planted_run("INT4", h, langs=LANGS27)}
            gg = ga.GainRuns(rs, DOWN, UP, FINE, n_controls=2); tt = ga.build_table(gg, n_boot=50, seed=0)
            self.assertEqual(ga.h_g5(tt, gg, n_boot=50)["b"]["met"], expected, msg=f"{k} repaired")

    def test_h_g6_needs_24_of_27_positive(self):
        for k, expected in ((3, True), (4, False)):
            g, t = self.table27(self.break_languages(k, 0.125, -0.5))
            self.assertEqual(ga.h_g6(t, g)["met"], expected, msg=f"{k} negative")

class Survivors(unittest.TestCase):
    """Tests added after the first mutation round of phase1_gain_analysis (each kills a mutant that survived)."""
    def test_two_up_ratios_are_not_three(self):
        with self.assertRaises(ValueError):
            ga.GainRuns(runs(), DOWN, [2.0, 4.0], FINE, n_controls=2)

    def test_h_g1_needs_the_median_ordered_even_when_24_languages_agree(self):
        n = 27
        def h(c, a, k):
            i = int(c[1:])
            if k != "g":
                return 0.0
            if i < 24:
                return {0.5: 0.0, 0.25: 0.0001, 0.125: 0.0002, 0.0: 0.0003}[a] + i * 0.001 if a in (0.5, 0.25, 0.125, 0.0) else 0.0
            return {0.5: 5.0, 0.25: -5.0, 0.125: -4.0, 0.0: -3.0}.get(a, 0.0)        # three languages: high at the mildest ratio, low after
        rs = {"FP": planted_run("FP", h, langs=LANGS27)}
        g = ga.GainRuns(rs, DOWN, UP, FINE, n_controls=2); t = ga.build_table(g, n_boot=20, seed=0)
        r = ga.h_g1(t, g, "FP")
        self.assertEqual(r["agree"], 24); self.assertFalse(r["median_ordered"]); self.assertFalse(r["met"])

    def test_the_interval_of_the_net_harm_decides_h_g5b_not_the_point(self):
        e = np.random.default_rng(5).normal(0, 0.2, N)
        g, t = table_of({"FP": planted_run("FP", simple_harm), "INT4": planted_run("INT4", repair(-0.011, -0.01), noise_group=e)}, n_boot=400)
        self.assertEqual(ga.h_g5(t, g, n_boot=50)["b"]["count"], 0)

    def test_equal_group_harm_at_two_up_ratios_is_not_increasing_with_a_higher_third(self):
        h = lambda c, a, k: (0.015 if (c == "l0" and a in (2.0, 4.0) and k == "g") else simple_harm(c, a, k))
        g, t = table_of(runs(harm=h))
        self.assertFalse(ga.h_g3(t, g, "FP")["per_language_increasing"]["l0"])
        self.assertTrue(ga.h_g3(t, g, "FP")["per_language_increasing"]["l1"])

    def test_a_dipping_median_fails_the_whole_rule(self):
        e = np.random.default_rng(5).normal(0, 0.05, N)
        g, t = table_of(ladder_runs((1.0, 0.2, 1.6), noise=e), n_boot=400)
        r = ga.h_g4(t, g)
        self.assertEqual(r["count_positive"], 4); self.assertFalse(r["met"])

    def test_generic_repair_needs_the_interval_below_zero_at_one_of_the_two_ratios_only(self):
        def h(c, a, k):
            if k != "g" and a == 1.1:
                return -0.02
            if k != "g" and a == 1.2:
                return 0.02
            return 0.0 if a in (1.1, 1.2) else simple_harm(c, a, k)
        g, t = table_of(two_states(simple_harm, h))
        r = ga.h_g5(t, g, n_boot=100)
        self.assertTrue(r["a"]["met"]); self.assertLess(r["a"]["per_ratio"][1.1]["hi"], 0); self.assertGreater(r["a"]["per_ratio"][1.2]["lo"], 0)

    def test_the_alpha_star_correlation_of_zero_is_not_enough(self):
        def h(c, a, k):
            best = {"l0": 1.1, "l1": 1.2, "l2": 1.2, "l3": 1.1}[c]
            return -0.05 if (k == "g" and a == best) else (0.0 if a in (1.1, 1.2) else simple_harm(c, a, k))
        g, t = table_of(two_states(simple_harm, h))
        r = ga.h_g5(t, g, baseline_g_ratio={"l0": 0.7, "l1": 0.8, "l2": 0.9, "l3": 1.0}, n_boot=50)
        self.assertAlmostEqual(r["corr"]["rho"], 0.0); self.assertFalse(r["corr"]["met"])

    def test_a_weak_rank_correlation_between_fp_and_int4_is_not_consistency(self):
        value = {"l0": 1.0, "l1": 3.0, "l2": 4.0, "l3": 2.0}                       # ranks 1, 3, 4, 2 against 1, 2, 3, 4 at FP: Spearman 0.4
        h_int4 = lambda c, a, k: (0.01 * value[c] if (k == "g" and a == 0.125) else (0.0 if a == 0.125 else simple_harm(c, a, k)))
        g, t = table_of(two_states(simple_harm, h_int4))
        r = ga.h_g6(t, g)
        self.assertAlmostEqual(r["rank_corr"], 0.4); self.assertEqual(r["per_state"]["INT4"]["count_positive"], 4); self.assertFalse(r["met"])

    def test_the_verdict_text_follows_each_rule(self):
        g, t = table_of(two_states(simple_harm, repair(0.02, 0.01)))
        text = ga.report_text(ga.evaluate(t, g, n_boot=50), t, g, {"runs": {}})
        line = lambda key: [l for l in text.split("\n") if l.startswith(f"| {key}")][0]
        self.assertTrue(line("H-G1").rstrip().endswith("**met** |")); self.assertFalse(line("H-G1").rstrip().endswith("**not met** |"))
        self.assertTrue(line("H-G5a").rstrip().endswith("**not met** |"))

    def test_without_an_int4_run_the_int4_rules_are_not_testable_not_failed(self):
        g, t = table_of({"FP": planted_run("FP", simple_harm)})
        res = ga.evaluate(t, g, n_boot=20)
        for k in ("H-G4", "H-G5a", "H-G5b", "H-G5-alpha"):
            self.assertIsNone(res["verdicts"][k], k)

    def test_alpha_star_values_are_written(self):
        g, t = table_of(two_states(simple_harm, repair(-0.03, -0.01)))
        res = ga.evaluate(t, g, n_boot=50)
        with tempfile.TemporaryDirectory() as d:
            ga.write_outputs(d, res, t, g, {"runs": {}})
            with open(Path(d) / "alpha_star.csv") as fh:
                rows = list(csv.DictReader(fh))
        self.assertEqual({r["alpha_star"] for r in rows if r["state"] == "INT4"}, {"1.1"}); self.assertEqual({r["alpha_star"] for r in rows if r["state"] == "FP"}, {"1.0"})

    def test_the_baseline_ratio_of_the_asked_state_wins_whatever_the_row_order(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "baseline.csv"
            p.write_text("language,state,g_ratio\nl0,INT4,0.85\nl0,FP,1.0\nl1,INT4,0.9\nl1,FP,1.0\n")
            self.assertEqual(ga.load_baseline_g_ratio(p), {"l0": 0.85, "l1": 0.9})

if __name__ == "__main__":
    unittest.main()
