"""Tests for scripts/g0/make_project_results_tables.py (the per-language appendix of reports/PROJECT_RESULTS.md). The tables must be generated from the committed result files; the checks below pin
headline numbers that the project reports elsewhere (RESULTS_SUMMARY.md, the Study G README, the association-retention report) and the marker handling of the document. Stdlib only."""
import re
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
import make_project_results_tables as mt  # noqa: E402


def section(text, title_start):
    m = re.search(r"^### " + re.escape(title_start) + r".*?(?=^### |\Z)", text, re.S | re.M)
    return m.group(0)


class Tables(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.L = mt.roster()
        cls.text = mt.build(cls.L)

    def test_the_roster_has_27_languages_nine_per_tier(self):
        self.assertEqual(len(self.L), 27)
        self.assertEqual([sum(1 for l in self.L if l[2] == t) for t in ("high", "mid", "low")], [9, 9, 9])

    def test_every_section_has_one_row_per_language_plus_its_summary_row(self):
        for title, n in (("A1.", 27), ("A2.", 28), ("A3.", 28), ("A4. ", 28), ("A6.", 28), ("A7.", 55), ("A8.", 27), ("A10.", 28)):
            rows = [l for l in section(self.text, title).splitlines() if l.startswith("| ") and "---" not in l]
            self.assertEqual(len(rows) - (2 if title == "A7." else 1), n, title)

    def test_the_pooled_association_row_matches_the_report(self):
        row = [l for l in section(self.text, "A3.").splitlines() if "all 27 pooled" in l][0]
        self.assertIn("| 6663 | 99.2 | 98.1 | 97.5 | 94.7 | 89.1 | 70.8 |", row)

    def test_the_function_medians_match_the_functional_summary(self):
        row = [l for l in section(self.text, "A4. ").splitlines() if "median of the 27" in l][0]
        self.assertIn("| 0.262 |", row)
        self.assertTrue(row.rstrip().endswith("1.54 |"))

    def test_the_zero_ablation_medians_of_study_g_match_its_readme(self):
        row = [l for l in section(self.text, "A7.").splitlines() if "median of the 27" in l][0]
        self.assertIn("+0.262 | +0.268 | +0.266 | +0.271 | +0.281 | +0.432", row)

    def test_belebele_means_match_the_frozen_statements(self):
        row = [l for l in section(self.text, "A6.").splitlines() if "mean of the covered" in l][0]
        self.assertIn("0.426 | 0.426 | 0.425", row)
        self.assertTrue(row.rstrip().endswith("0.236 |"))
        self.assertEqual(sum(1 for l in section(self.text, "A6.").splitlines() if l.startswith("| ") and "not covered" in l), 9)

    def test_the_external_panel_has_six_languages(self):
        rows = [l for l in section(self.text, "A9.").splitlines() if l.startswith("| ") and "---" not in l]
        self.assertEqual(len(rows) - 1, 6)

    def test_the_document_is_updated_only_between_its_markers(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "doc.md"
            p.write_text(f"before\n{mt.BEGIN}\nold\n{mt.END}\nafter\n")
            mt.main_for(str(p), self.L)
            out = p.read_text()
        self.assertTrue(out.startswith("before\n" + mt.BEGIN))
        self.assertTrue(out.endswith(mt.END + "\nafter\n"))
        self.assertNotIn("\nold\n", out)
        self.assertIn("### A1. The 27 languages", out)

    def test_the_committed_document_is_current(self):
        doc = (ROOT / "reports/PROJECT_RESULTS.md").read_text()
        i, j = doc.index(mt.BEGIN), doc.index(mt.END)
        self.assertEqual(doc[i + len(mt.BEGIN):j].strip(), self.text.strip())


if __name__ == "__main__":
    unittest.main()
