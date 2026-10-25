import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_func_analysis as fa
import phase1_gain_extc as ec
import phase1_gain_ratios as rt
import phase1_gain_run as gr

LANGS = ["l0", "l1", "l2", "l3"]
RATIOS = [1.1, 1.2, 1.4, 1.7]
NTOK = np.array([10, 12, 8, 15, 11, 9, 14, 10])
N = len(NTOK)
IDS = list(range(100, 100 + N))                        # stand-in for the reserve sentence ids
PRESET = gr.preset_name(RATIOS, 5, "K64", zero=False)


def planted_run(label, harm, langs=LANGS, n_controls=5, noise=None, ids=IDS, drop=()):
    """A Run of an extension C job: scaled groups, scaled controls (draws 0..n-1) and the sham, no zero-ablation. harm(code, ratio, kind) with kind "g" or "c<r>"."""
    e = np.zeros(N) if noise is None else np.asarray(noise)
    items = {}
    def put(code, name, h):
        if name not in drop:
            items[f"{code}|{name}"] = {"nll": 3.0 * NTOK + (h + e) * NTOK, "ntok": NTOK}
    for code in langs:
        items[f"{code}|none"] = {"nll": 3.0 * NTOK, "ntok": NTOK}
        put(code, f"sham_K64@{code}", 0.0)
        for a in RATIOS:
            put(code, rt.group_name(a, "K64", code), harm(code, a, "g"))
            for r in range(n_controls):
                put(code, rt.control_name(a, r, "K64", code), harm(code, a, f"c{r}"))
    return fa.Run(label, items, {}, {}, list(langs), {c: list(ids) for c in langs})


def repair_harm(code, a, kind):
    """INT4-like: the group lowers the loss a little, the controls do not move."""
    return -0.002 * (a - 1.0) * 10 if kind == "g" else 0.0001 * int(kind[1:])


def cost_harm(code, a, kind):
    """FP-like: the group costs a little."""
    return 0.002 * (a - 1.0) * 10 if kind == "g" else 0.0001 * int(kind[1:])


def runs(states=None, **kw):
    states = states or {"FP": cost_harm, "INT4": repair_harm}
    return {s: planted_run(s, h, **kw) for s, h in states.items()}


def ext(rs=None, **kw):
    return ec.ExtRuns(rs or runs(), RATIOS, n_controls=5, label="K64", **kw)


class Construct(unittest.TestCase):
    def test_states_are_on_the_ladder_and_fp_is_not_required_but_int4_and_fp_are_what_the_rules_need(self):
        e = ext(runs({"INT4": repair_harm, "FP": cost_harm, "INT6": cost_harm}))
        self.assertEqual(e.states, ["FP", "INT6", "INT4"]); self.assertEqual(e.languages, LANGS); self.assertEqual(e.n_sentences, N)

    def test_a_missing_configuration_is_named(self):
        rs = runs(); rs["INT4"] = planted_run("INT4", repair_harm, drop=(rt.control_name(1.4, 4, "K64", "l2"),))
        with self.assertRaises(KeyError) as cm:
            ext(rs)
        self.assertIn("cscale1.4.4_K64@l2", str(cm.exception))

    def test_a_missing_sham_is_named(self):
        rs = runs(); rs["FP"] = planted_run("FP", cost_harm, drop=("sham_K64@l0",))
        with self.assertRaises(KeyError):
            ext(rs)

    def test_different_ids_between_states_or_languages_are_refused(self):
        rs = runs(); rs["INT4"] = planted_run("INT4", repair_harm, ids=list(range(200, 200 + N)))
        with self.assertRaises(ValueError):
            ext(rs)
        r = planted_run("FP", cost_harm); items = {f"{c}|{n}": {"nll": v[0], "ntok": v[1]} for (c, n), v in r._items.items()}
        ids = {c: list(IDS) for c in LANGS}; ids["l1"] = list(range(300, 300 + N))
        with self.assertRaises(ValueError):
            ec.ExtRuns({"FP": fa.Run("FP", items, {}, {}, LANGS, ids)}, RATIOS, n_controls=5)

    def test_the_sentences_must_equal_the_expected_ones_and_never_touch_the_forbidden_ones(self):
        ext(expect_ids=IDS, forbid_ids=[0, 1, 2])
        with self.assertRaises(ValueError):
            ext(expect_ids=list(range(N)))
        with self.assertRaises(ValueError):
            ext(forbid_ids=[IDS[3]])

    def test_ratio_lists_must_be_above_one(self):
        with self.assertRaises(ValueError):
            ec.ExtRuns(runs(), [0.9, 1.2], n_controls=5)
        with self.assertRaises(ValueError):
            ec.ExtRuns(runs(), [1.1, 1.1], n_controls=5)

    def test_the_run_must_have_at_least_the_sample_floor_of_draws(self):
        with self.assertRaises(ValueError):
            ec.ExtRuns(runs(), RATIOS, n_controls=2)


class Cells(unittest.TestCase):
    def setUp(self):
        self.e = ext()
        self.c = ec.build_cells(self.e, n_boot=100, seed=0)

    def test_point_values_and_the_net_harm_of_each_draw(self):
        cell = self.c[("INT4", "l1", 1.2)]
        self.assertAlmostEqual(cell["g"], repair_harm("l1", 1.2, "g"))
        self.assertAlmostEqual(cell["c"], float(np.mean([repair_harm("l1", 1.2, f"c{r}") for r in range(5)])))
        self.assertAlmostEqual(cell["net"], cell["g"] - cell["c"])
        self.assertEqual(len(cell["net_draws"]), 5); self.assertAlmostEqual(float(np.mean(cell["net_draws"])), cell["net"])

    def test_the_sentence_bootstrap_is_seeded_and_has_one_value_per_draw(self):
        e = np.random.default_rng(3).normal(0, 0.02, N)
        c1 = ec.build_cells(ext(runs(noise=e)), n_boot=100, seed=0); c2 = ec.build_cells(ext(runs(noise=e)), n_boot=100, seed=0)
        self.assertEqual(c1[("FP", "l0", 1.1)]["net_s"].shape, (100,))
        np.testing.assert_array_equal(c1[("FP", "l0", 1.1)]["net_s"], c2[("FP", "l0", 1.1)]["net_s"])

    def test_the_number_of_sentences_is_read_from_the_id_lists(self):
        noise = np.zeros(N); noise[N // 2:] = np.random.default_rng(1).normal(0, 0.05, N - N // 2)       # in the group configurations only, second half of the sentences
        run = planted_run("FP", lambda code, a, kind: 0.01)
        for (code, name), (nll, ntok) in list(run._items.items()):
            if name.startswith("scale"):
                run._items[(code, name)] = (nll + noise * ntok, ntok)
        cells = ec.build_cells(ec.ExtRuns({"FP": run}, RATIOS, n_controls=5), n_boot=100, seed=0)
        self.assertEqual(ec.ExtRuns({"FP": run}, RATIOS, n_controls=5).n_sentences, N)
        self.assertGreater(float(np.std(cells[("FP", "l0", 1.1)]["net_s"])), 0.0)                    # the draws cover positions beyond the number of languages


def boundary_cells(neg, n=27, ratio=1.1, state="INT4", value=-0.001):
    """Language-level cells for 27 languages with `neg` negative net harms at INT4 and FP positive."""
    cells = {}
    for i in range(n):
        for s in ("FP", "INT4"):
            for a in RATIOS:
                v = (value if i < neg else +0.0005) if s == "INT4" else 0.001
                cells[(s, f"l{i}", a)] = {"g": v, "c": 0.0, "net": v, "net_draws": [v] * 5, "net_s": np.full(10, v)}
    return cells, [f"l{i}" for i in range(n)]


class Rules(unittest.TestCase):
    def test_h_gc1_with_20_or_more_negative_languages_is_met_and_with_18_it_is_not(self):
        for neg, expected in ((27, True), (20, True), (18, False), (10, False)):
            cells, langs = boundary_cells(neg)
            r = ec.h_gc1(cells, langs, "INT4", (1.1, 1.2), n_boot=2000, seed=0)
            self.assertEqual(r["per_ratio"][1.1]["n_negative"], neg)
            self.assertEqual(r["met"], expected, msg=f"{neg} negative")

    def test_the_count_condition_is_19_of_27_by_itself(self):
        cells, langs = boundary_cells(19)
        r = ec.h_gc1(cells, langs, "INT4", (1.1, 1.2), n_boot=500, seed=0)["per_ratio"][1.1]
        self.assertTrue(r["count_ok"])
        cells, langs = boundary_cells(18)
        self.assertFalse(ec.h_gc1(cells, langs, "INT4", (1.1, 1.2), n_boot=500, seed=0)["per_ratio"][1.1]["count_ok"])

    def test_the_median_interval_is_below_zero_exactly_when_at_least_20_of_27_values_are_negative(self):
        for neg, expected in ((21, True), (20, True), (17, False)):
            lo, hi = ec.median_interval([-1.0] * neg + [1.0] * (27 - neg), n_boot=5000, seed=0)
            self.assertEqual(hi < 0, expected, msg=f"{neg} negative")

    def test_h_gc1_is_met_by_either_primary_ratio_but_each_ratio_needs_both_parts(self):
        cells, langs = boundary_cells(27)
        for l in langs:
            cells[("INT4", l, 1.2)]["net"] = +0.001                                    # 1.2 fails
        r = ec.h_gc1(cells, langs, "INT4", (1.1, 1.2), n_boot=500, seed=0)
        self.assertTrue(r["met"]); self.assertTrue(r["ratio_met"][1.1]); self.assertFalse(r["ratio_met"][1.2])

    def test_the_sign_test_p_value_is_one_sided(self):
        cells, langs = boundary_cells(19)
        p = ec.h_gc1(cells, langs, "INT4", (1.1, 1.2), n_boot=200, seed=0)["per_ratio"][1.1]["sign_p"]
        self.assertAlmostEqual(p, 0.026119492948055267)

    def test_the_interval_is_at_97_5_percent(self):
        cells, langs = boundary_cells(27)
        for i, l in enumerate(langs):
            cells[("INT4", l, 1.1)]["net"] = -0.001 * (1 + i)
        r = ec.h_gc1(cells, langs, "INT4", (1.1, 1.2), n_boot=2000, seed=0)["per_ratio"][1.1]
        nets = np.array([-0.001 * (1 + i) for i in range(27)])
        boot = np.median(nets[np.random.default_rng(0).integers(0, 27, (2000, 27))], axis=1)
        self.assertAlmostEqual(r["lo"], float(np.percentile(boot, 1.25))); self.assertAlmostEqual(r["hi"], float(np.percentile(boot, 98.75)))

    def test_h_gc2_the_paired_difference_must_be_below_zero_at_every_primary_ratio(self):
        cells, langs = boundary_cells(27)
        r = ec.h_gc2(cells, langs, ("INT4", "FP"), (1.1, 1.2), n_boot=500, seed=0)
        self.assertTrue(r["met"]); self.assertTrue(all(v["hi"] < 0 for v in r["per_ratio"].values()))
        for l in langs:
            cells[("INT4", l, 1.2)]["net"] = 0.002                                     # INT4 above FP at 1.2
        r2 = ec.h_gc2(cells, langs, ("INT4", "FP"), (1.1, 1.2), n_boot=500, seed=0)
        self.assertFalse(r2["met"])

    def test_h_gc2_is_a_paired_difference_over_languages(self):
        cells, langs = boundary_cells(27)
        for i, l in enumerate(langs):
            cells[("FP", l, 1.1)]["net"] = 0.001 + 0.0001 * i
            cells[("INT4", l, 1.1)]["net"] = -0.001 + 0.0001 * i
        r = ec.h_gc2(cells, langs, ("INT4", "FP"), (1.1, 1.2), n_boot=500, seed=0)["per_ratio"][1.1]
        self.assertAlmostEqual(r["median_diff"], -0.002)

    def test_h_gc3_is_group_specific_only_if_h_gc1_is_met_and_the_controls_do_not_fall_too(self):
        cells, langs = boundary_cells(27)
        h1 = ec.h_gc1(cells, langs, "INT4", (1.1, 1.2), n_boot=500, seed=0)
        r = ec.h_gc3(cells, langs, "INT4", h1, n_boot=500, seed=0)
        self.assertTrue(r["specific"])
        for l in langs:
            cells[("INT4", l, 1.1)]["c"] = -0.002; cells[("INT4", l, 1.2)]["c"] = -0.002
        self.assertFalse(ec.h_gc3(cells, langs, "INT4", h1, n_boot=500, seed=0)["specific"])
        cells2, _ = boundary_cells(5)
        h1b = ec.h_gc1(cells2, langs, "INT4", (1.1, 1.2), n_boot=500, seed=0)
        self.assertIsNone(ec.h_gc3(cells2, langs, "INT4", h1b, n_boot=500, seed=0)["specific"])

    def test_the_optimum_table_has_one_row_per_state_and_ratio(self):
        cells, langs = boundary_cells(27)
        rows = ec.h_gc4_rows(cells, langs, ["FP", "INT4"], RATIOS)
        self.assertEqual(len(rows), 2 * 4)
        r = [x for x in rows if x["state"] == "INT4" and x["ratio"] == 1.1][0]
        self.assertAlmostEqual(r["median_net"], -0.001); self.assertEqual(r["n_negative_net"], 27)
        self.assertIn("n_alpha_star", r)

    def test_alpha_star_counts_the_ratio_with_the_lowest_group_harm_including_one(self):
        cells, langs = boundary_cells(27)
        for l in langs:
            for a, v in ((1.1, -0.001), (1.2, -0.003), (1.4, -0.002), (1.7, 0.01)):
                cells[("INT4", l, a)]["g"] = v
            cells[("FP", l, 1.1)]["g"] = 0.002
        rows = ec.h_gc4_rows(cells, langs, ["FP", "INT4"], RATIOS)
        r = {(x["state"], x["ratio"]): x for x in rows}
        self.assertEqual(r[("INT4", 1.2)]["n_alpha_star"], 27); self.assertEqual(r[("INT4", 1.1)]["n_alpha_star"], 0)
        self.assertEqual(r[("INT4", 1.2)]["best_ratio_by_median_group_harm"], True)


class Loading(unittest.TestCase):
    def save(self, d, run, state=None, preset=PRESET, complete=True, skipped=(), gates=None):
        import torch
        d = Path(d); d.mkdir(parents=True, exist_ok=True)
        state = state or run.label
        items = {f"{c}|{n}": {"nll": torch.tensor(v[0]), "ntok": torch.tensor(v[1])} for (c, n), v in run._items.items()}
        torch.save({"precision": state, "split": "eval", "ids": {k: list(v) for k, v in run.ids.items()}, "languages": list(run.languages), "preset": preset, "items": items, "stats": {}, "fidelity": {}}, d / f"func_lm_eval_{state}.pt")
        (d / "meta_lm_eval.json").write_text(json.dumps({"complete": complete, "skipped_for_time": list(skipped), "preset": preset, "gates": gates or {"split_equals_full": True, "scale_one_identity": True, "donor_self_identity": True, "sham_identity": True}}))

    def test_runs_of_two_states_are_collected_and_checked(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            self.save(d / "FP", planted_run("FP", cost_harm)); self.save(d / "INT4", planted_run("INT4", repair_harm))
            e, info = ec.load_runs([d / "FP", d / "INT4"], RATIOS, expect_ids=IDS, forbid_ids=[0, 1])
            self.assertEqual(e.states, ["FP", "INT4"]); self.assertEqual(e.n_controls, 5); self.assertEqual(len(info["FP"]["sha256"]), 64)

    def test_a_preset_with_zero_ablation_or_other_ratios_or_fewer_draws_is_refused(self):
        for preset in (gr.preset_name(RATIOS, 5, "K64"), gr.preset_name([1.1, 1.2], 5, "K64", zero=False), gr.preset_name(RATIOS, 3, "K64", zero=False)):
            with tempfile.TemporaryDirectory() as d:
                self.save(Path(d) / "FP", planted_run("FP", cost_harm), preset=preset)
                with self.assertRaises(ValueError, msg=preset):
                    ec.load_runs([Path(d) / "FP"], RATIOS, expect_ids=IDS)

    def test_an_incomplete_job_a_skipped_state_or_a_failed_gate_is_refused(self):
        for kw in ({"complete": False}, {"skipped": ["INT4"]}, {"gates": {"split_equals_full": False}}):
            with tempfile.TemporaryDirectory() as d:
                self.save(Path(d) / "FP", planted_run("FP", cost_harm), **kw)
                with self.assertRaises(ValueError, msg=str(kw)):
                    ec.load_runs([Path(d) / "FP"], RATIOS, expect_ids=IDS)

    def test_no_result_file_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                ec.load_runs([Path(d)], RATIOS, expect_ids=IDS)


class Outputs(unittest.TestCase):
    def setUp(self):
        self.e = ext(runs({"FP": cost_harm, "INT4": repair_harm}))
        self.cells = ec.build_cells(self.e, n_boot=100, seed=0)
        self.res = ec.evaluate(self.cells, self.e, n_boot=300, seed=0)

    def test_evaluate_gives_the_three_rules_and_the_table(self):
        for k in ("H-Gc1", "H-Gc2", "H-Gc3", "H-Gc4"):
            self.assertIn(k, self.res["rules"])
        self.assertEqual(set(self.res["verdicts"]), {"H-Gc1", "H-Gc2", "H-Gc3"})
        self.assertTrue(self.res["verdicts"]["H-Gc1"]); self.assertTrue(self.res["verdicts"]["H-Gc2"])

    def test_without_int4_or_fp_the_rules_that_need_them_are_not_testable(self):
        e = ext(runs({"INT6": cost_harm}))
        res = ec.evaluate(ec.build_cells(e, n_boot=50, seed=0), e, n_boot=100, seed=0)
        self.assertIsNone(res["verdicts"]["H-Gc1"]); self.assertIsNone(res["verdicts"]["H-Gc2"]); self.assertIsNone(res["verdicts"]["H-Gc3"])

    def test_the_report_names_every_rule_the_sample_and_the_reliability_table(self):
        text = ec.report_text(self.res, self.cells, self.e, {"runs": {}})
        for k in ("H-Gc1", "H-Gc2", "H-Gc3", "H-Gc4"):
            self.assertIn(k, text)
        self.assertIn(f"{N} sentences per language", text); self.assertIn("5 matched-control draws", text); self.assertIn("Reliability of the sample", text)

    def test_the_verdict_text_follows_the_rules(self):
        e = ext(runs({"FP": cost_harm, "INT4": cost_harm}))
        res = ec.evaluate(ec.build_cells(e, n_boot=50, seed=0), e, n_boot=200, seed=0)
        text = ec.report_text(res, ec.build_cells(e, n_boot=50, seed=0), e, {"runs": {}})
        line = [l for l in text.split("\n") if l.startswith("| H-Gc1")][0]
        self.assertTrue(line.rstrip().endswith("**not met** |"))

    def test_the_bands_are_the_median_intervals_of_the_rules_in_the_order_of_the_languages_of_the_analysis(self):
        rows = self.res["bands"]
        self.assertEqual(len(rows), 2 * 4)                                                      # states x ratios
        for r in rows:
            nets = [self.cells[(r["state"], l, r["ratio"])]["net"] for l in self.e.languages]
            lo, hi = ec.median_interval(nets, 300, 0)
            self.assertEqual((r["lo"], r["hi"]), (lo, hi)); self.assertAlmostEqual(r["median"], float(np.median(nets)))
            self.assertEqual(r["n_negative"], sum(1 for x in nets if x < 0)); self.assertEqual(r["n_languages"], 4)
        h1 = self.res["rules"]["H-Gc1"]["per_ratio"]
        for r in rows:
            if r["state"] == "INT4" and r["ratio"] in h1:
                self.assertEqual((r["lo"], r["hi"], r["median"]), (h1[r["ratio"]]["lo"], h1[r["ratio"]]["hi"], h1[r["ratio"]]["median"]))

    def test_the_interval_depends_on_the_order_of_the_languages_so_the_roster_order_is_the_one_used(self):
        langs = [f"l{i}" for i in reversed(range(27))]                                           # a roster that is not in alphabetical order
        vals = np.random.default_rng(1).normal(-0.0005, 0.003, 27); vals[5] = 0.0                  # one distinct value per language, one language exactly at zero
        def harm(code, a, kind):
            return float(vals[int(code[1:])]) * (a - 1.0) * 10 if kind == "g" else 0.0
        e = ec.ExtRuns({"INT4": planted_run("INT4", harm, langs=langs)}, RATIOS, n_controls=5, label="K64")
        cells = ec.build_cells(e, n_boot=20, seed=0)
        rows = ec.band_rows(cells, e.languages, e.states, e.ratios, 2000, 0)
        other = ec.band_rows(cells, sorted(e.languages), e.states, e.ratios, 2000, 0)
        self.assertEqual(e.languages, langs)
        self.assertTrue(any((a["lo"], a["hi"]) != (b["lo"], b["hi"]) for a, b in zip(rows, other)))
        res = ec.evaluate(cells, e, n_boot=2000, seed=0)
        self.assertEqual([(r["lo"], r["hi"]) for r in res["bands"]], [(r["lo"], r["hi"]) for r in rows])
        for r in res["bands"]:                                                                      # every number of every row, recomputed with the stated draws, seed and order
            nets = [cells[(r["state"], l, r["ratio"])]["net"] for l in langs]
            self.assertEqual((r["lo"], r["hi"]), ec.median_interval(nets, 2000, 0)); self.assertAlmostEqual(r["median"], float(np.median(nets)))
            self.assertEqual(r["n_negative"], sum(1 for x in nets if x < 0)); self.assertEqual(r["n_languages"], 27)
        self.assertTrue(any(sum(1 for l in langs if cells[("INT4", l, a)]["net"] == 0.0) for a in e.ratios))        # the language at zero is really at zero
        self.assertTrue(any(abs(r["median"] - float(np.mean([cells[(r["state"], l, r["ratio"])]["net"] for l in langs]))) > 1e-6 for r in res["bands"]))   # median and mean differ here

    def test_write_outputs_and_main(self):
        with tempfile.TemporaryDirectory() as d:
            ec.write_outputs(d, self.res, self.cells, self.e, {"runs": {}})
            for f in ("analysis.json", "cells.csv", "reliability.csv", "optimum.csv", "bands.csv", "report.md"):
                self.assertTrue((Path(d) / f).exists(), f)
            js = json.loads((Path(d) / "analysis.json").read_text())
            self.assertEqual(js["verdicts"]["H-Gc1"], True)
            with open(Path(d) / "cells.csv") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 2 * 4 * 4)                                        # states x languages x ratios
            with open(Path(d) / "bands.csv") as fh:
                b = list(csv.DictReader(fh))
            self.assertEqual(list(b[0]), ["state", "ratio", "median", "lo", "hi", "n_negative", "n_languages"]); self.assertEqual(len(b), 2 * 4)
            self.assertAlmostEqual(float(b[0]["lo"]), self.res["bands"][0]["lo"])

    def test_main_runs_on_files(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            load = Loading();
            load.save(d / "FP", planted_run("FP", cost_harm)); load.save(d / "INT4", planted_run("INT4", repair_harm))
            with contextlib.redirect_stdout(io.StringIO()):
                ec.main(["--runs", str(d / "FP"), str(d / "INT4"), "--out", str(d / "out"), "--n-boot", "100", "--boot-languages", "300", "--expect-ids", ",".join(map(str, IDS)), "--forbid-ids", "0,1,2"])
            js = json.loads((d / "out" / "analysis.json").read_text())
            self.assertTrue(js["verdicts"]["H-Gc1"]); self.assertEqual(set(js["meta"]["runs"]), {"FP", "INT4"})


class Survivors(unittest.TestCase):
    """Added after the first mutation round of phase1_gain_extc."""
    def test_no_run_an_unknown_state_ratio_one_and_four_draws_are_refused(self):
        with self.assertRaises(ValueError):
            ec.ExtRuns({}, RATIOS, n_controls=5)
        with self.assertRaises(ValueError):
            ec.ExtRuns({"INT3": planted_run("INT3", cost_harm)}, RATIOS, n_controls=5)
        with self.assertRaises(ValueError):
            ec.ExtRuns(runs(), [1.0, 1.2], n_controls=5)
        with self.assertRaises(ValueError):
            ec.ExtRuns(runs(), RATIOS, n_controls=4)

    def test_the_net_harm_of_the_draws_differ_when_the_controls_differ(self):
        c = ec.build_cells(ext(), n_boot=50, seed=0)[("INT4", "l0", 1.2)]
        self.assertEqual(len({round(x, 9) for x in c["net_draws"]}), 5)

    def test_the_seed_of_the_median_interval_matters_and_is_reproducible(self):
        v = np.random.default_rng(9).normal(0, 1, 27)
        self.assertEqual(ec.median_interval(v, 100, 0), ec.median_interval(v, 100, 0))
        self.assertNotEqual(ec.median_interval(v, 100, 0), ec.median_interval(v, 100, 1))

    def test_an_interval_that_ends_exactly_at_zero_is_not_below_zero(self):
        cells, langs = boundary_cells(17)
        for l in langs[17:]:
            for a in (1.1, 1.2):
                cells[("INT4", l, a)]["net"] = 0.0
        r = ec.h_gc1(cells, langs, "INT4", (1.1, 1.2), n_boot=2000, seed=0)["per_ratio"][1.1]
        self.assertEqual(r["hi"], 0.0); self.assertFalse(r["interval_ok"])
        r2 = ec.h_gc2(*self._flat_pair(), n_boot=500, seed=0)
        self.assertFalse(r2["per_ratio"][1.1]["ok"])

    def _flat_pair(self):
        cells, langs = boundary_cells(27)
        for l in langs:
            for a in (1.1, 1.2):
                cells[("INT4", l, a)]["net"] = 0.001; cells[("FP", l, a)]["net"] = 0.001             # no difference: the paired interval is exactly zero
        return cells, langs, ("INT4", "FP"), (1.1, 1.2)

    def test_19_negative_languages_are_borderline_and_the_pinned_seed_decides(self):
        # P(bootstrap median >= 0) = 0.0127 against 0.0125: the 98.75th percentile of the seeded bootstrap decides (characterisation, 5000 draws, tied values)
        cells, langs = boundary_cells(19)
        at0 = ec.h_gc1(cells, langs, "INT4", (1.1, 1.2), n_boot=5000, seed=0)
        at1 = ec.h_gc1(cells, langs, "INT4", (1.1, 1.2), n_boot=5000, seed=1)
        self.assertTrue(at0["per_ratio"][1.1]["count_ok"]); self.assertTrue(at0["per_ratio"][1.1]["interval_ok"]); self.assertTrue(at0["met"])          # seed 0 is the pre-registered seed
        self.assertTrue(at1["per_ratio"][1.1]["count_ok"]); self.assertFalse(at1["per_ratio"][1.1]["interval_ok"]); self.assertFalse(at1["met"])      # both conditions are needed

    def test_a_ratio_that_fails_does_not_count_for_the_specificity(self):
        cells, langs = boundary_cells(27)
        for l in langs:
            cells[("INT4", l, 1.2)]["net"] = +0.001                                   # 1.2 does not repair
            cells[("INT4", l, 1.2)]["c"] = -0.005                                    # its controls fall: irrelevant for the verdict
            cells[("INT4", l, 1.1)]["c"] = 0.0
        h1 = ec.h_gc1(cells, langs, "INT4", (1.1, 1.2), n_boot=500, seed=0)
        self.assertTrue(ec.h_gc3(cells, langs, "INT4", h1, n_boot=500, seed=0)["specific"])

    def test_alpha_star_is_one_when_no_ratio_lowers_the_loss_and_the_tie_goes_to_the_ratio_nearest_one(self):
        cells, langs = boundary_cells(27)
        for l in langs:
            for a in RATIOS:
                cells[("INT4", l, a)]["g"] = -0.001 if a in (1.1, 1.2) else 0.01
        r = {(x["state"], x["ratio"]): x for x in ec.h_gc4_rows(cells, langs, ["FP", "INT4"], RATIOS)}
        for a in RATIOS:
            self.assertEqual(r[("FP", a)]["n_alpha_star"], 0); self.assertFalse(r[("FP", a)]["best_ratio_by_median_group_harm"])      # FP: every group harm is positive
        self.assertEqual(r[("INT4", 1.1)]["n_alpha_star"], 27); self.assertTrue(r[("INT4", 1.1)]["best_ratio_by_median_group_harm"])
        self.assertFalse(r[("INT4", 1.2)]["best_ratio_by_median_group_harm"])

    def test_the_draw_error_enters_the_combined_count(self):
        def h(code, a, kind):
            return 0.01 if kind == "g" else (0.0 if kind in ("c0", "c1") else (0.01 if kind == "c2" else 0.0)) + (0.1 if kind == "c4" else 0.0)
        e = ext(runs({"FP": h}))
        cells = ec.build_cells(e, n_boot=100, seed=0)
        r = ec.reliability_rows(cells, e)[0]
        ses = [float(np.std(cells[("FP", l, 1.1)]["net_draws"], ddof=1)) / np.sqrt(5) for l in LANGS]
        self.assertAlmostEqual(r["median_se_draws"], float(np.median(ses)))
        self.assertLess(r["n_clear_both"], r["n_clear_sentences"] + 1)
        nets = np.array([cells[("FP", l, 1.1)]["net"] for l in LANGS])
        se_s = np.array([float(np.std(cells[("FP", l, 1.1)]["net_s"], ddof=1)) for l in LANGS])
        self.assertEqual(r["n_clear_both"], int((np.abs(nets) > 2 * np.sqrt(se_s ** 2 + np.array(ses) ** 2)).sum()))

    def test_h_gc2_and_the_rules_need_the_states_they_name(self):
        e = ext(runs({"INT4": repair_harm}))
        res = ec.evaluate(ec.build_cells(e, n_boot=50, seed=0), e, n_boot=100, seed=0)
        self.assertIsNone(res["verdicts"]["H-Gc2"]); self.assertTrue(res["verdicts"]["H-Gc1"])

    def test_states_run_with_different_presets_are_refused(self):
        load = Loading()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            load.save(d / "FP", planted_run("FP", cost_harm)); load.save(d / "INT4", planted_run("INT4", repair_harm), preset=gr.preset_name(RATIOS, 6, "K64", zero=False))
            with self.assertRaises(ValueError):
                ec.load_runs([d / "FP", d / "INT4"], RATIOS, expect_ids=IDS)

    def test_the_csv_carries_the_95_percent_sentence_interval(self):
        e = np.random.default_rng(3).normal(0, 0.02, N)
        run = planted_run("INT4", lambda c, a, k: 0.01)
        for (code, name), (nll, ntok) in list(run._items.items()):
            if name.startswith("scale"):
                run._items[(code, name)] = (nll + e * ntok, ntok)
        ex = ec.ExtRuns({"INT4": run}, RATIOS, n_controls=5); cells = ec.build_cells(ex, n_boot=200, seed=0)
        res = ec.evaluate(cells, ex, n_boot=100, seed=0)
        with tempfile.TemporaryDirectory() as d:
            ec.write_outputs(d, res, cells, ex, {"runs": {}})
            with open(Path(d) / "cells.csv") as fh:
                row = [r for r in csv.DictReader(fh) if r["language"] == "l0" and float(r["ratio"]) == 1.1][0]
        s = cells[("INT4", "l0", 1.1)]["net_s"]
        self.assertAlmostEqual(float(row["net_lo"]), float(np.percentile(s, 2.5))); self.assertAlmostEqual(float(row["net_hi"]), float(np.percentile(s, 97.5)))

    def test_the_report_states_the_sample_at_the_start(self):
        e = ext()
        res = ec.evaluate(ec.build_cells(e, n_boot=50, seed=0), e, n_boot=100, seed=0)
        self.assertIn(f"Sample: {N} sentences per language", ec.report_text(res, ec.build_cells(e, n_boot=50, seed=0), e, {"runs": {}}))

if __name__ == "__main__":
    unittest.main()
