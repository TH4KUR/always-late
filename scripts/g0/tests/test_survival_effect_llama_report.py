"""Tests for scripts/g0/survival_effect_llama_report.py: the Llama-versus-Gemma report is generated from the two result JSON files; numbers, gate results and labels in the text must equal the JSON, and an inadequate
instrument must produce a report that says so and reports no relation. Synthetic fixtures."""
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import survival_effect_llama_report as lr  # noqa: E402

POPS = ("primary", "noisy_stratum", "S_all", "S_loss")


def cell(partial=(0.2, 0.1, 0.3), imp=(0.05, 0.01, 0.09), met=False, rho=(0.4, 0.3, 0.5), n=600):
    return {"n": n, "n_languages": 27, "spearman": {"rho": rho[0], "lo": rho[1], "hi": rho[2]}, "partial_spearman_given_A": {"rho": partial[0], "lo": partial[1], "hi": partial[2]},
            "mae": {"A": 0.02, "B": 0.019, "C": 0.019, "R1": 0.02, "null": 0.021}, "improvement": {"B_over_A": {"improvement": imp[0], "lo": imp[1], "hi": imp[2]}, "C_over_B": {"improvement": 0.0, "lo": -0.01, "hi": 0.01},
            "A_over_null": {"improvement": 0.01, "lo": 0.0, "hi": 0.02}, "R1_over_null": {"improvement": 0.0, "lo": -0.01, "hi": 0.01}}, "cell_met": met}


def llama(ok=True, **kw):
    gate = {"ok": ok, "why": [] if ok else ["median INT16 cosine 0.400 is below 0.7"], "floor": 0.91, "median_cos16": 0.98 if ok else 0.4, "n_reliable": 590, "n_features": 643}
    if not ok:
        return {"gate": gate, "cells": {}, "overall_statement": "instrument inadequate: no relation reported"}
    cells = {f"{s}|{p}": cell(**kw) for s in ("INT6", "INT4") for p in POPS}
    return {"gate": gate, "tau": 0.91, "eps_dnll": 0.002, "n_features": 643, "population_counts": {"primary": 590, "noisy_stratum": 53, "S_all": 643, "S_loss": 300}, "n_boot": 2000, "seed": 20261028, "cells": cells,
            "cells_met_primary": sum(c["cell_met"] for k, c in cells.items() if k.endswith("primary")), "overall_statement": "no clear predictive value beyond baselines"}


def gemma():
    cells = {"legacy RTN|INT6|primary": cell(partial=(0.36, 0.30, 0.41), imp=(0.044, 0.026, 0.062), rho=(0.47, 0.42, 0.51), n=1412),
             "legacy RTN|INT4|primary": cell(partial=(0.27, 0.20, 0.32), imp=(0.037, 0.023, 0.052), rho=(0.34, 0.28, 0.38), n=1412)}
    return {"tau": 0.94888916015625, "population_counts": {"primary": 1412}, "cells": cells}


class Report(unittest.TestCase):
    def test_the_gate_numbers_and_the_overall_statement_are_those_of_the_json(self):
        t = lr.render(llama(), gemma())
        self.assertIn("0.910", t); self.assertIn("590", t); self.assertIn("no clear predictive value beyond baselines", t); self.assertIn("1412", t)

    def test_labels_follow_the_defined_rules(self):
        rows = lambda t: [l for l in t.splitlines() if l.startswith("| Llama-3.2-1B | INT")]
        t = lr.render(llama(partial=(0.2, 0.1, 0.3)), gemma()); self.assertTrue(all("| repeats |" in l for l in rows(t))); self.assertEqual(len(rows(t)), 2)
        t = lr.render(llama(partial=(0.0, -0.1, 0.1)), gemma()); self.assertTrue(all("| cannot be resolved |" in l for l in rows(t)))
        t = lr.render(llama(partial=(-0.3, -0.4, -0.1)), gemma()); self.assertTrue(all("| reverses |" in l for l in rows(t)))
        t = lr.render(llama(met=True, imp=(0.15, 0.05, 0.25)), gemma()); self.assertTrue(all(l.rstrip().endswith("| **met** |") for l in rows(t)))
        t = lr.render(llama(met=False), gemma()); self.assertTrue(all(l.rstrip().endswith("| not met |") for l in rows(t)))
        # the Gemma rows follow the same rule on the Gemma numbers (4.4 % is below 10 %): never 'met'
        self.assertTrue(all(l.rstrip().endswith("| not met |") for l in t.splitlines() if l.startswith("| Gemma-2-2B | INT")))

    def test_the_gemma_rows_carry_the_gemma_numbers_of_their_own_state(self):
        t = lr.render(llama(), gemma()); g = {l.split("|")[2].strip(): l for l in t.splitlines() if l.startswith("| Gemma-2-2B | INT")}
        self.assertIn("0.470 [0.420, 0.510]", g["INT6"]); self.assertIn("0.340 [0.280, 0.380]", g["INT4"]); self.assertIn("| repeats |", g["INT6"]); self.assertIn("| repeats |", g["INT4"])

    def test_the_gate_line_names_the_own_floor_and_the_outcome(self):
        t = lr.render(llama(), gemma()); self.assertIn("this system's floor (5th percentile of the INT16 cosine): 0.910", t); self.assertIn("Gate: **passed**", t)
        t = lr.render(llama(ok=False), gemma()); self.assertIn("Gate: **FAILED**", t)

    def test_an_inadequate_instrument_gives_a_report_without_relations(self):
        t = lr.render(llama(ok=False), gemma())
        self.assertIn("instrument inadequate", t); self.assertIn("median INT16 cosine 0.400", t); self.assertNotIn("Spearman R1-W1", t)

    def test_the_text_keeps_the_claim_boundaries_of_the_addendum(self):
        t = lr.render(llama(), gemma())
        for needle in ("post hoc", "no feature identities", "two systems", "not equivalence"):
            self.assertIn(needle, t)
        self.assertNotIn("proves", t)


if __name__ == "__main__":
    unittest.main()
