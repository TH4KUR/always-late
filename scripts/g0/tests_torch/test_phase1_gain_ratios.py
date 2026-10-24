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
import phase1_func_run as fr
import phase1_gain_ratios as rt
import phase1_gain_run as gr
from test_phase1_gain_run import CODES, toy_manifest


class Grid(unittest.TestCase):
    def test_grid_is_log_symmetric_with_eight_steps(self):
        down, up = rt.grid()
        self.assertEqual(len(down), 8); self.assertEqual(len(up), 8)
        self.assertAlmostEqual(down[0], 2 ** -0.5); self.assertEqual(down[1], 0.5); self.assertEqual(down[3], 0.25); self.assertEqual(down[-1], 0.0625)
        self.assertEqual(up, [1.0 / d for d in down])
        self.assertEqual(down, sorted(down, reverse=True))

    def test_calibration_ratios_add_the_two_fine_up_ratios(self):
        r = rt.calibration_ratios()
        self.assertEqual(len(r), 18)
        self.assertIn(1.1, r); self.assertIn(1.2, r)
        self.assertEqual(len(set(r)), 18)
        gr.validate_ratios(r)                                          # the wrapper accepts them: finite, distinct names, none equal to 1


def curve(**kw):
    """Net harm per down-grid ratio: defaults to a far-away value, overrides given as {k: value} for grid index k (1..8)."""
    down, _ = rt.grid()
    out = {g: 100.0 for g in down}
    for k, v in kw.items():
        out[down[int(k[1:]) - 1]] = v
    return out


class Select(unittest.TestCase):
    def test_values_exactly_at_the_targets_are_chosen_and_the_up_ratios_are_reciprocals(self):
        res = rt.select_ratios(curve(k2=0.1, k4=0.3, k6=0.6), 1.0)
        down, _ = rt.grid()
        self.assertEqual(res["down"], [down[1], down[3], down[5]])
        self.assertEqual(res["up"], [1.0 / d for d in res["down"]])
        self.assertEqual(res["branch"], "targets")

    def test_the_nearest_value_is_chosen_not_an_exact_one(self):
        res = rt.select_ratios(curve(k2=0.02, k3=0.12, k4=0.2, k5=0.35, k6=0.5, k7=0.9), 1.0)
        down, _ = rt.grid()
        # targets 0.1 -> k3 (0.12), 0.3 -> k5 (0.35), 0.6 -> k6 (0.5, nearer than 0.9)
        self.assertEqual(res["down"], [down[2], down[4], down[5]])

    def test_two_targets_with_the_same_nearest_value_take_the_next_nearest_for_the_later_one(self):
        # 0.3 and 0.6 are both nearest to k5 (0.45); the later target (0.6) takes the next nearest not yet chosen: k6 (0.9 is farther than 0.8)
        res = rt.select_ratios(curve(k2=0.1, k5=0.45, k6=0.8, k7=0.9), 1.0)
        down, _ = rt.grid()
        self.assertEqual(res["down"], [down[1], down[4], down[5]])

    def test_equally_near_values_go_to_the_milder_ratio(self):
        res = rt.select_ratios(curve(k2=1.0, k3=2.0, k4=4.0, k6=6.0), 10.0)       # the target 3.0 is equally near k3 (2.0) and k4 (4.0): the larger ratio, k3
        down, _ = rt.grid()
        self.assertEqual(res["down"], [down[1], down[2], down[5]])

    def test_a_flat_response_falls_back_to_the_three_deepest_quarters(self):
        res = rt.select_ratios({g: 0.01 for g in rt.grid()[0]}, 1.0)
        self.assertEqual(res["down"], [0.25, 0.125, 0.0625]); self.assertEqual(res["up"], [4.0, 8.0, 16.0]); self.assertEqual(res["branch"], "flat")

    def test_exactly_ten_percent_is_not_flat(self):
        c = {g: 0.01 for g in rt.grid()[0]}; c[rt.grid()[0][7]] = 0.1
        res = rt.select_ratios(c, 1.0)
        self.assertEqual(res["branch"], "targets")

    def test_scaling_the_net_harms_and_z_together_changes_nothing(self):
        a = rt.select_ratios(curve(k2=0.1, k4=0.3, k6=0.6), 1.0)["down"]
        b = rt.select_ratios({g: v * 5 for g, v in curve(k2=0.1, k4=0.3, k6=0.6).items()}, 5.0)["down"]
        self.assertEqual(a, b)

    def test_input_order_does_not_matter(self):
        c = curve(k2=0.1, k4=0.3, k6=0.6)
        rev = dict(reversed(list(c.items())))
        self.assertEqual(rt.select_ratios(c, 1.0), rt.select_ratios(rev, 1.0))

    def test_a_missing_grid_value_or_a_non_positive_z_is_an_error(self):
        c = curve(k2=0.1); c.pop(rt.grid()[0][0])
        with self.assertRaises(ValueError):
            rt.select_ratios(c, 1.0)
        for z in (0.0, -1.0, float("nan")):
            with self.assertRaises(ValueError):
                rt.select_ratios(curve(), z)


def planted_run(label, per_lang, ratios=(0.5,), n_controls=2, zero=None, ctrl_zero=None, ntok=(10, 10), langs=("l0", "l1", "l2")):
    """A Run with planted pooled harms: per_lang[code] = (group harm, [control harms]) at every ratio; zero = {code: group zero harm}; ctrl_zero = {code: [zeroed control harms]}."""
    items = {}
    nt = np.array(ntok)
    for code in langs:
        items[f"{code}|none"] = {"nll": 2.0 * nt, "ntok": nt}
        g, cs = per_lang[code]
        for a in ratios:
            items[f"{code}|scale{a:g}_K64@{code}"] = {"nll": 2.0 * nt + g * nt, "ntok": nt}
            for r, h in enumerate(cs):
                items[f"{code}|cscale{a:g}.{r}_K64@{code}"] = {"nll": 2.0 * nt + h * nt, "ntok": nt}
        if zero is not None:
            items[f"{code}|zero_K64@{code}"] = {"nll": 2.0 * nt + zero[code] * nt, "ntok": nt}
            for r, h in enumerate(ctrl_zero[code]):
                items[f"{code}|ctrl{r}_K64@{code}"] = {"nll": 2.0 * nt + h * nt, "ntok": nt}
    return fa.Run(label, items, {}, {}, list(langs), list(range(len(ntok))))


PL = {"l0": (0.5, [0.1, 0.3]), "l1": (0.2, [0.0, 0.1]), "l2": (1.0, [0.2, 0.2])}


class Table(unittest.TestCase):
    def test_net_harm_is_group_minus_the_mean_control_pooled_over_tokens(self):
        run = planted_run("FP", PL)
        rows = rt.calibration_rows(run, ["l0", "l1", "l2"], [0.5], label="K64", n_controls=2)
        r = [x for x in rows if x["ratio"] == 0.5][0]
        self.assertAlmostEqual(r["net_median"], 0.3)                  # nets 0.3, 0.15, 0.8: median 0.3
        self.assertAlmostEqual(r["group_median"], 0.5); self.assertAlmostEqual(r["control_median"], 0.2)
        self.assertEqual(r["languages_net_positive"], 3)

    def test_the_median_is_over_languages_not_over_tokens(self):
        run = planted_run("FP", PL, ntok=(1, 99))
        r = rt.calibration_rows(run, ["l0", "l1", "l2"], [0.5], label="K64", n_controls=2)[0]
        self.assertAlmostEqual(r["net_median"], 0.3)

    def test_the_zero_row_uses_the_zero_ablation_and_the_zeroed_controls(self):
        run = planted_run("FP", PL, zero={"l0": 2.0, "l1": 1.0, "l2": 3.0}, ctrl_zero={"l0": [0.2, 0.2], "l1": [0.1, 0.3], "l2": [0.5, 0.5]})
        rows = rt.calibration_rows(run, ["l0", "l1", "l2"], [0.5], label="K64", n_controls=2)
        z = [x for x in rows if x["ratio"] == 0.0][0]
        self.assertAlmostEqual(z["net_median"], 1.8)                  # nets 1.8, 0.8, 2.5
        self.assertEqual(rt.zero_net_median(rows), z["net_median"])

    def test_languages_with_zero_or_negative_net_harm_are_not_counted(self):
        pl = {"l0": (0.5, [0.1, 0.3]), "l1": (0.2, [0.2, 0.2]), "l2": (0.1, [0.3, 0.3])}          # nets 0.3, 0.0, -0.2
        r = rt.calibration_rows(planted_run("FP", pl), ["l0", "l1", "l2"], [0.5], label="K64", n_controls=2)[0]
        self.assertEqual(r["languages_net_positive"], 1)

    def test_net_over_z_divides_by_the_zero_ablation_net_harm(self):
        run = planted_run("FP", PL, zero={"l0": 2.0, "l1": 1.0, "l2": 3.0}, ctrl_zero={"l0": [0.2, 0.2], "l1": [0.1, 0.3], "l2": [0.5, 0.5]})
        rows = rt.calibration_rows(run, ["l0", "l1", "l2"], [0.5], label="K64", n_controls=2)
        self.assertAlmostEqual([x for x in rows if x["ratio"] == 0.5][0]["net_over_z"], 0.3 / 1.8)
        self.assertAlmostEqual([x for x in rows if x["ratio"] == 0.0][0]["net_over_z"], 1.0)

    def test_without_a_zero_row_there_is_no_z(self):
        rows = rt.calibration_rows(planted_run("FP", PL), ["l0", "l1", "l2"], [0.5], label="K64", n_controls=2)
        self.assertTrue(np.isnan(rows[0]["net_over_z"]))
        with self.assertRaises(KeyError):
            rt.zero_net_median(rows)

    def test_a_missing_configuration_is_an_error(self):
        run = planted_run("FP", PL, ratios=(0.5,))
        with self.assertRaises(KeyError):
            rt.calibration_rows(run, ["l0", "l1", "l2"], [0.5, 0.25], label="K64", n_controls=2)
        with self.assertRaises(KeyError):
            rt.calibration_rows(run, ["l0", "l1", "l2"], [0.5], label="K64", n_controls=3)

    def test_names_agree_with_what_the_runner_and_the_wrapper_build(self):
        man = toy_manifest()
        self.addCleanup(gr.unregister)
        name = gr.register([0.5, 1.2], n_controls=3, label="K64")
        names = {c["name"] for c in fr.build_configs(man, CODES, CODES, [], fr.PRESETS[name], "K64")}
        for a in (0.5, 1.2):
            self.assertIn(rt.group_name(a, "K64", "l1"), names)
            for r in range(3):
                self.assertIn(rt.control_name(a, r, "K64", "l1"), names)
        self.assertIn(rt.zero_name("K64", "l1"), names); self.assertIn(rt.zeroed_control_name(2, "K64", "l1"), names)


class Outputs(unittest.TestCase):
    def setUp(self):
        self.run = planted_run("FP", PL, ratios=tuple(rt.calibration_ratios()), zero={"l0": 2.0, "l1": 1.0, "l2": 3.0}, ctrl_zero={"l0": [0.2, 0.2], "l1": [0.1, 0.3], "l2": [0.5, 0.5]})
        self.rows = {"FP": rt.calibration_rows(self.run, ["l0", "l1", "l2"], rt.calibration_ratios(), "K64", 2)}

    def test_write_outputs_files_and_the_chosen_ratios_in_the_json(self):
        sel = rt.select_ratios({g: 0.5 for g in rt.grid()[0]}, 1.8)
        with tempfile.TemporaryDirectory() as d:
            rt.write_outputs(d, self.rows, sel, {"runs": {"FP": "abc"}, "z": 1.8})
            js = json.loads((Path(d) / "ratios.json").read_text())
            self.assertEqual(js["down"], sel["down"]); self.assertEqual(js["up"], sel["up"]); self.assertEqual(js["fine_up"], [1.1, 1.2]); self.assertEqual(js["branch"], sel["branch"])
            self.assertEqual(js["meta"]["runs"], {"FP": "abc"})
            with open(Path(d) / "calibration.csv") as fh:
                got = list(csv.DictReader(fh))
            self.assertEqual(len(got), len(self.rows["FP"])); self.assertEqual(got[0]["state"], "FP")
            text = (Path(d) / "calibration.md").read_text()
            self.assertIn("rule", text); self.assertIn(str(sel["down"][0])[:6], text)

    def _save(self, d, run, state):
        import torch
        items = {f"{text}|{name}": {"nll": torch.tensor(nll), "ntok": torch.tensor(ntok)} for (text, name), (nll, ntok) in run._items.items()}
        torch.save({"precision": state, "split": "pilot", "ids": [0, 1], "languages": ["l0", "l1", "l2"], "preset": "x", "items": items, "stats": {}, "fidelity": {}}, d / f"func_lm_pilot_{state}.pt")

    def test_main_reads_pilot_files_and_writes_the_rule_result(self):
        other = planted_run("x", PL, ratios=tuple(rt.calibration_ratios()), zero={"l0": 4.0, "l1": 4.0, "l2": 4.0}, ctrl_zero={"l0": [0.2, 0.2], "l1": [0.1, 0.3], "l2": [0.5, 0.5]})
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            self._save(d, self.run, "FP"); self._save(d, other, "ABC")
            with contextlib.redirect_stdout(io.StringIO()):
                rt.main(["--runs", str(d), "--out", str(d / "out"), "--n-controls", "2"])
            js = json.loads((d / "out" / "ratios.json").read_text())
            self.assertEqual(len(js["down"]), 3); self.assertEqual(len(js["up"]), 3)
            self.assertEqual(js["meta"]["states"], ["FP", "ABC"])                  # FP first whatever the other labels sort to
            self.assertAlmostEqual(js["meta"]["z"], 1.8)                           # from the FP run, not from the other state (z 3.7)
            with open(d / "out" / "calibration.csv") as fh:
                rows = [r for r in csv.DictReader(fh) if r["state"] == "FP" and float(r["ratio"]) == 0.5]
            self.assertAlmostEqual(float(rows[0]["control_median"]), 0.2)          # the mean of two control draws, as asked by --n-controls 2

    def test_main_needs_the_full_precision_run(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            self._save(d, self.run, "INT4")
            with self.assertRaises(ValueError):
                rt.main(["--runs", str(d), "--out", str(d / "out"), "--n-controls", "2"])


if __name__ == "__main__":
    unittest.main()
