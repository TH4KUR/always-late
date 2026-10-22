"""Rehearsal of the whole Belebele evaluation chain on synthetic item files (written 2026-10-22 while the two real evaluation jobs were running): item files of the format that the runner writes ->
phase1_func_mc_analysis.py with the flags of the STATUS commands -> phase1_func_mc_statements.py against the REAL freeze -> phase1_func_figures.py --macro-effect --statements. The planted data have 18 covered languages
(read from the real freeze), the frozen 385 items per language, six precisions and known effects, so that every step is checked against the numbers that were planted. Nothing here is a result."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phase1_func_figures as ff
import phase1_func_mc_analysis as mc
import phase1_func_mc_statements as ms

ROOT = Path(__file__).resolve().parents[3]
FREEZE_PATH = ROOT / "reports/functional_validation/evaluation_freeze_mc.json"
FREEZE = json.loads(FREEZE_PATH.read_text())
COV = list(FREEZE["analysis"]["covered"]); N = FREEZE["items"][COV[0]]["n_items"]
PRECS = ["FP", "INT8", "INT7", "INT6", "INT5", "INT4"]
DELTA = {"FP": 0.5, "INT8": 0.5, "INT7": 0.5, "INT6": 0.3, "INT5": 0.5, "INT4": 1.5}          # margin that zero-ablating the group removes: ratios 1, 1, 0.6, 1, about 3 against full precision
EXPECTED = {"S0": "met", "S1": "preserved", "S2": "preserved", "S3": "attenuated", "S4": "preserved", "S5": "amplified"}


def margins_to_scores(answers, margins):
    s = np.zeros((len(answers), 4))
    for i, (a, m) in enumerate(zip(answers, margins)):
        s[i, a] = m
    return s


def write_files(d, precs, seed=0):
    """Item files as the runner writes them: one per precision, items {'<language>|<config>': {'cand_logp': (n_items, 4)}}, the same keys and answers for every language (translations of the same items)."""
    import torch
    rng = np.random.default_rng(seed); keys = [(f"https://example.org/passage{i // 5}", i % 5 + 1) for i in range(N)]
    answers = [int(a) for a in rng.integers(0, 4, N)]; base = 2.0 + rng.normal(0, 0.4, (len(COV), N))                  # item difficulty per language, shared by all precisions
    for p in precs:
        items = {}
        for k, lang in enumerate(COV):
            m = base[k]; noise = rng.normal(0, 0.03, N)
            items[f"{lang}|none"] = {"cand_logp": torch.tensor(margins_to_scores(answers, m), dtype=torch.float64)}
            items[f"{lang}|zero_K64@{lang}"] = {"cand_logp": torch.tensor(margins_to_scores(answers, m - DELTA[p] + noise), dtype=torch.float64)}
            for r in range(3):
                items[f"{lang}|ctrl{r}_K64@{lang}"] = {"cand_logp": torch.tensor(margins_to_scores(answers, m + rng.normal(0, 0.03, N)), dtype=torch.float64)}
            items[f"{lang}|sham_K64@{lang}"] = {"cand_logp": torch.tensor(margins_to_scores(answers, m), dtype=torch.float64)}
            items[f"{lang}|res_K64@{lang}"] = {"cand_logp": torch.tensor(margins_to_scores(answers, m - DELTA[p] / 2), dtype=torch.float64)}
        torch.save({"precision": p, "split": "eval", "format": "letter", "languages": list(COV), "preset": "mc_eval", "keys": {l: list(keys) for l in COV}, "dropped": {l: 0 for l in COV},
                    "answers": {l: list(answers) for l in COV}, "items": items, "stats": {}}, d / f"func_mc_eval_letter_{p}.pt")


def analyse(runs_dir, out, n_boot=2000):
    mc.main(["--runs", str(runs_dir), "--split", "eval", "--format", "letter", "--out", str(out), "--n-boot", str(n_boot), "--retention-level", "0.99", "--covered", *COV])


class Chain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = Path(tempfile.mkdtemp()); cls.runs = cls.d / "runs"; cls.runs.mkdir(); write_files(cls.runs, PRECS)
        cls.out = cls.d / "mc_eval"; analyse(cls.runs, cls.out)
        ms.main(["--freeze", str(FREEZE_PATH), "--analysis", str(cls.out / "analysis.json"), "--out", str(cls.out)])
        cls.stm = json.loads((cls.out / "statements.json").read_text())

    def test_the_analysis_has_the_frozen_provenance_and_the_six_precisions(self):
        a = json.loads((self.out / "analysis.json").read_text())
        self.assertEqual((a["task"], a["format"], a["K"], a["n_boot"], a["seed"], a["retention_level"]), ("T-MC", "letter", "K64", 2000, 0, 0.99)); self.assertEqual(a["covered"], COV)
        self.assertEqual(sorted(a["macro"]), sorted(PRECS));                                              # the analysis orders its precisions by file name; every consumer orders them by the ladder itself self.assertEqual({r["n_items"] for r in a["accuracy"]}, {N}); self.assertNotIn("calibration", a["note"].lower())
        for p in PRECS:
            self.assertEqual(max(abs(v["point"]) for v in a["sham"][p].values()), 0.0)                      # the sham path is exactly zero

    def test_the_statements_pass_the_provenance_check_against_the_real_freeze_and_recover_the_planted_outcomes(self):
        self.assertTrue(self.stm["complete"]); self.assertEqual(self.stm["freeze_commit"], FREEZE["commit"])
        self.assertEqual({s["id"]: s["outcome"] for s in self.stm["statements"]}, EXPECTED)
        for s in self.stm["statements"][1:]:
            self.assertEqual(s["n_languages"], len(COV), s["id"])                                           # every covered language has a clearly positive full-precision effect
        self.assertTrue((self.out / "statements.md").read_text().startswith("# ")); self.assertEqual(len((self.out / "statements.csv").read_text().splitlines()), 7)

    def test_the_planted_macro_effects_and_ratios_are_recovered(self):
        a = json.loads((self.out / "analysis.json").read_text())
        e = {p: a["macro"][p]["point"] for p in PRECS}
        self.assertAlmostEqual(e["FP"], e["INT8"], places=2); self.assertAlmostEqual(e["INT7"], e["INT8"], places=2); self.assertLess(e["INT6"], 0.75 * e["FP"]); self.assertGreater(e["INT4"], 1.33 * e["FP"])
        r = {p: a["retention"][p]["macro"]["ratio"] for p in PRECS[1:]}
        self.assertAlmostEqual(r["INT8"], 1.0, delta=0.05); self.assertLess(r["INT6"], 0.75); self.assertGreater(r["INT4"], 1.33)

    def test_the_figures_run_with_the_documented_flags_and_write_figure_10_with_the_statement_outcomes(self):
        figs = self.d / "figs"
        ff.main(["--results", str(self.out), "--out", str(figs), "--note", "synthetic rehearsal, not a result", "--unit", "nats per item", "--macro-effect", "--statements", str(self.out / "statements.json"),
                 "--themes", "light", "--formats", "png"])
        for name in ("fig1_effect_per_language_light.png", "fig2_retention_light.png", "fig10_macro_effect_light.png", "table_macro_effect.md", "table_retention.md"):
            self.assertTrue((figs / name).exists(), name)
        self.assertFalse((figs / "figures_skipped.txt").exists())                                           # 18 languages back every macro retention
        md = (figs / "table_macro_effect.md").read_text()
        for sid, out in EXPECTED.items():
            self.assertIn(f"| {sid} | {out} |", md)

    def test_a_missing_precision_is_no_data_not_an_error_and_figure_10_still_runs(self):
        d = Path(tempfile.mkdtemp()); runs = d / "runs"; runs.mkdir(); write_files(runs, [p for p in PRECS if p != "INT7"]); out = d / "out"; analyse(runs, out, n_boot=2000)
        ms.main(["--freeze", str(FREEZE_PATH), "--analysis", str(out / "analysis.json"), "--out", str(out)])
        stm = json.loads((out / "statements.json").read_text()); got = {s["id"]: s["outcome"] for s in stm["statements"]}
        self.assertEqual(got["S2"], "no data"); self.assertEqual({k: v for k, v in got.items() if k != "S2"}, {k: v for k, v in EXPECTED.items() if k != "S2"})
        self.assertFalse(stm["complete"]); self.assertEqual(stm["missing_precisions"], ["INT7"])
        self.assertIn("INCOMPLETE", (out / "statements.md").read_text())
        figs = d / "figs"; ff.main(["--results", str(out), "--out", str(figs), "--note", "x", "--unit", "nats per item", "--macro-effect", "--statements", str(out / "statements.json"), "--themes", "light", "--formats", "png"])
        self.assertTrue((figs / "fig10_macro_effect_light.png").exists())

    def test_an_analysis_with_other_settings_is_refused_by_the_statements_tool(self):
        d = Path(tempfile.mkdtemp()); out = d / "out"; analyse(self.runs, out, n_boot=200)                   # fewer draws than frozen
        with self.assertRaises(SystemExit) as cm:
            ms.main(["--freeze", str(FREEZE_PATH), "--analysis", str(out / "analysis.json"), "--out", str(d / "stm")])
        self.assertIn("n_boot", str(cm.exception)); self.assertFalse((d / "stm").exists())

    def test_forgetting_the_retention_level_flag_of_the_documented_command_is_refused_not_silently_accepted(self):
        d = Path(tempfile.mkdtemp()); out = d / "out"
        mc.main(["--runs", str(self.runs), "--split", "eval", "--format", "letter", "--out", str(out), "--n-boot", "2000", "--covered", *COV])        # the default level is 95 %, the statements use 99 %
        with self.assertRaises(SystemExit) as cm:
            ms.main(["--freeze", str(FREEZE_PATH), "--analysis", str(out / "analysis.json"), "--out", str(d / "stm")])
        self.assertIn("retention_level", str(cm.exception)); self.assertNotIn("n_boot", str(cm.exception))                      # only that setting differs


if __name__ == "__main__":
    unittest.main()
