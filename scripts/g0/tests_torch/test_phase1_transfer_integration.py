"""End to end on a fake model: the REAL runner (run_transfer, run_group) writes its chunk files, the report driver reads them, writes its tables, and the figure module draws from those tables. The report tests use
hand-made chunk files; the first real group output showed that the formats can drift (group chunks repeat the pools), so this test feeds the runner's own output through the whole analysis."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_transfer_analysis as ta
import phase1_transfer_figures as tf
import phase1_transfer_report as rep
from test_phase1_transfer_run import FakeModel, fake_core, run_fake, run_fake_group

SCALE = {"FP": (1.0, 1.0), "INT16": (1.0001, 1.0001), "INT6": (1.05, 1.1), "INT4": (1.2, 1.35)}                    # prefix and suffix scale per state: INT16 is rounding only, the others differ


class ScaledModel(FakeModel):
    def lower(self, tokens):
        return self.E[tokens] * SCALE[self.state][0]

    def suffix(self, x):
        return torch.tanh(x @ (self.Ws * SCALE[self.state][1]))


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); root = Path(cls.tmp.name); cls.root = root; cls.fdirs, cls.gdirs = {}, {}; grad = lambda h, t: 0.01 * h
        for s in ("INT16", "INT6", "INT4"):
            cls.fdirs[s] = root / f"f_{s}"; run_fake(ScaledModel(), cls.fdirs[s], states=(s,), grad_fn=grad)
            cls.gdirs[s] = root / f"g_{s}"; run_fake_group(ScaledModel(), cls.gdirs[s], states=(s,), grad_fn=grad)
        cls.core = {**fake_core(), "languages": ["l0", "l1"]}; cls.split = {"calibration": [0, 2], "heldout": [1, 3]}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_the_feature_level_report_reads_the_runner_output_for_every_state(self):
        res = rep.feature_level(self.fdirs, self.core, self.split, 997, n_boot=30, seed=1)
        for s in ("INT16", "INT6", "INT4"):
            st = res["states"][s]; self.assertGreater(st["n_items"], 0); self.assertTrue(st["components"] and st["specificity"] and st["averaged"] and st["averaged_features"])
            self.assertTrue(all(r["eligible"] is not None for r in st["rows"].values())); self.assertEqual(st["predictive"]["all_items"]["dropped_reasons"], {})                 # every runner item carries the scalars of the design
        self.assertGreater(sum(r["eligible"] for r in res["states"]["INT4"]["rows"].values()), 0)                                                                                # the fake INT16 floor is tiny, so something clears it
        r4 = [r for r in res["states"]["INT4"]["rows"].values() if r["kind"] == "real"]; r6 = [r for r in res["states"]["INT6"]["rows"].values() if r["kind"] == "real"]
        self.assertGreater(np.median([r["error_norm"] for r in r4]), np.median([r["error_norm"] for r in r6]))                                                                  # a larger perturbation, a larger error

    def test_the_group_level_report_reads_the_runner_output_with_its_repeated_pools(self):
        res = rep.group_level(self.gdirs, ["l0", "l1"], 3, 3, n_boot=30, seed=1)
        for s in ("INT16", "INT6", "INT4"):
            st = res["states"][s]; self.assertEqual(len(st["per_language"]), 2); self.assertTrue(all(np.isfinite(r["C"]) and np.isfinite(r["W"]) for r in st["per_language"]))
        c16 = max(abs(r["C"]) + abs(r["W"]) for r in res["states"]["INT16"]["per_language"]); c4 = max(abs(r["C"]) + abs(r["W"]) for r in res["states"]["INT4"]["per_language"]); self.assertLess(c16, c4)

    def test_the_tables_the_claims_and_the_four_figures_come_out_of_the_runner_output(self):
        feat = rep.feature_level(self.fdirs, self.core, self.split, 997, n_boot=30, seed=1); grp = rep.group_level(self.gdirs, ["l0", "l1"], 3, 3, n_boot=30, seed=1); out = self.root / "report"
        claims = rep.claim_status(feat, grp); rep.write_outputs(out, feat, grp, claims); json.loads((out / "claim_status.json").read_text())
        self.assertIn(claims["claims"]["A3"]["overall"], ("supported", "reversed", "inconclusive", "mixed")); self.assertEqual(claims["claims"]["UTILITY"]["overall"], "not met")                  # twelve real items per state are too few to fit
        tables = tf.load_tables(out); paths = tf.write_figures(tables, self.root / "figs", "fake model, plumbing only", themes=("light",), formats=("svg",))
        self.assertEqual(len(paths), 4); self.assertTrue(all(Path(p).stat().st_size > 1000 for p in paths))


if __name__ == "__main__":
    unittest.main()
