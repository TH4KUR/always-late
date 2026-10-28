"""Tests for scripts/g0/survival_effect_report.py: the Stage 4 report is generated from the result JSON of survival_effect.py; the numbers and the overall statement in the text must equal the JSON, and the wording
follows the decision rules of the freeze for each of the three possible overall statements. Synthetic fixture; numpy not needed."""
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import survival_effect_report as sr  # noqa: E402

ARMS = ("legacy RTN", "matched RTN", "GPTQ"); STATES = ("INT6", "INT4"); POPS = ("primary", "noisy_stratum", "S_all", "S_loss")


def cell(met, imp=0.2, lo=0.1, n=1400):
    return {"n": n, "n_languages": 27, "spearman": {"rho": 0.55, "lo": 0.5, "hi": 0.6}, "partial_spearman_given_A": {"rho": 0.31, "lo": 0.2, "hi": 0.4},
            "mae": {"A": 0.020, "B": 0.016, "C": 0.015, "R1": 0.021, "null": 0.022},
            "improvement": {"B_over_A": {"improvement": imp, "lo": lo, "hi": 0.3}, "C_over_B": {"improvement": 0.05, "lo": -0.01, "hi": 0.1}, "A_over_null": {"improvement": 0.09, "lo": 0.0, "hi": 0.2}, "R1_over_null": {"improvement": 0.02, "lo": -0.02, "hi": 0.06}},
            "median_chosen_lambda": {"A": 1.0, "B": 1.0, "C": 1.0, "R1": 10.0}, "dichotomy": {"survives_event": 900, "survives_no_event": 30, "fails_event": 300, "fails_no_event": 170},
            "auroc_R1_for_event": {"auroc": 0.71, "lo": 0.66, "hi": 0.76, "n_event": 1200, "n_no_event": 200}, "cell_met": met}


def fixture(mets):
    cells = {f"{a}|{s}|{p}": (cell(mets[(a, s)]) if p == "primary" else cell(False, imp=0.07, lo=0.01)) for a in ARMS for s in STATES for p in POPS}
    ctrl = [{"arm": a, "state": s, "rho": 0.5, "lo": 0.4, "hi": 0.6, "n": 1487, "committed_rho": 0.5, "committed_lo": 0.4, "committed_hi": 0.6, "reproduced": True} for a in ARMS for s in STATES]
    n = sum(mets.values())
    stmt = "adds predictive value" if n == 6 else "no clear predictive value beyond baselines" if n == 0 else "mixed"
    return {"freeze": "x", "tau": 0.94888916015625, "eps_dnll": 0.0014654942796007557, "n_features": 1487, "population_counts": {"primary": 1412, "noisy_stratum": 75, "S_all": 1487, "S_loss": 842},
            "n_boot": 2000, "seed": 20261028, "base_columns": ["a", "b"], "cells": cells, "control": ctrl, "overall_statement": stmt, "cells_met_primary": n}


class Report(unittest.TestCase):
    def text(self, k):
        mets = {(a, s): i < k for i, (a, s) in enumerate([(a, s) for a in ARMS for s in STATES])}
        self.res = fixture(mets); return sr.render(self.res)

    def test_the_overall_statement_and_the_cell_count_in_the_text_are_those_of_the_json(self):
        for k in (0, 3, 6):
            t = self.text(k)
            self.assertIn(f"{self.res['overall_statement']}**", t); self.assertIn(f"{k} of 6", t); self.assertIn("Spearman correlation of R1 with W1 is 0.550", t)

    def test_every_primary_cell_row_carries_the_json_numbers_and_the_met_flag(self):
        t = self.text(2); t = t[t.index("## 3. Primary"):t.index("## 4. Secondary")]
        for a in ARMS:
            for s in STATES:
                c = self.res["cells"][f"{a}|{s}|primary"]
                row = [l for l in t.splitlines() if l.startswith(f"| {a} | {s} |")]
                self.assertTrue(row, (a, s)); r = row[0]
                self.assertIn(f"{c['improvement']['B_over_A']['improvement'] * 100:.1f} %", r); self.assertIn("**met**" if c["cell_met"] else "not met", r)
                self.assertIn(f"{c['spearman']['rho']:.3f} [{c['spearman']['lo']:.3f}, {c['spearman']['hi']:.3f}]", r)

    def test_the_result_paragraph_reads_the_primary_cells_only(self):
        t = self.text(0); para = t[t.index("## 1. Result"):t.index("## 2. Question")]
        self.assertIn("by 20.0 %", para); self.assertNotIn("7.0 %", para)

    def test_the_wording_depends_on_the_overall_statement_and_never_claims_more_than_it_allows(self):
        t0, t3, t6 = self.text(0), self.text(3), self.text(6)
        self.assertIn("stops here", t0); self.assertNotIn("stops here", t6)
        self.assertIn("not a mechanism", t6); self.assertIn("predictive evidence", t6)
        self.assertIn("only some cells", t3)
        for t in (t0, t3, t6):
            self.assertIn("post hoc", t); self.assertNotIn("proves", t); self.assertNotIn("causes", t.replace("not a causal", ""))

    def test_the_control_and_the_population_counts_are_reported(self):
        t = self.text(1)
        self.assertIn("1412", t); self.assertIn("842", t); self.assertIn("reproduced", t)


if __name__ == "__main__":
    unittest.main()
