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

    def gptq_rows(self, title):
        return [l for l in section(self.text, title).splitlines() if l.startswith("| ") and "---" not in l]

    def test_the_gptq_extension_sections_have_one_row_per_language_plus_the_median_row(self):
        for title in ("A11.", "A12.", "A13.", "A14.", "A15."):
            self.assertEqual(len(self.gptq_rows(title)) - 1, 28, title)

    def test_gptq_survival_legacy_column_equals_the_phase1_table_of_a2(self):
        a2 = [l for l in self.gptq_rows("A2.") if "median of the 27" in l][0].split("|")
        a11 = [l for l in self.gptq_rows("A11.") if "median of the 27" in l][0].split("|")
        self.assertEqual(a11[2].strip(), a2[4].strip())  # legacy INT6 = the A2 INT6 median (90.4)
        self.assertEqual(a11[5].strip(), a2[6].strip())  # legacy INT4 = the A2 INT4 median

    def test_gptq_survival_columns_are_ordered_legacy_matched_gptq_and_gptq_is_best_at_int4_in_every_language(self):
        body = [l.split("|") for l in self.gptq_rows("A11.")[1:-1]]
        for r in body:
            self.assertGreater(float(r[7]), float(r[6]) - 1e-9, r[1])  # GPTQ INT4 >= matched RTN INT4 in each language

    def test_gptq_loss_increase_median_matches_d104(self):
        row = [l for l in self.gptq_rows("A12.") if "median of the 27" in l][0]
        self.assertEqual(len(row.split("|")), 9)
        c = [x.strip() for x in row.split("|")]
        self.assertTrue(float(c[5]) > float(c[6]) > float(c[7]))  # INT4: legacy > matched > GPTQ

    def test_gptq_e4_median_ratios_match_d107(self):
        row = [l for l in self.gptq_rows("A13.") if "median of the 27" in l][0]
        self.assertIn("| 1.54 | 1.36 | 1.22 |", row)

    def test_gptq_transfer_legacy_columns_equal_the_a8_int4_columns(self):
        a8 = {l.split("|")[1].strip(): [x.strip() for x in l.split("|")] for l in self.gptq_rows("A8.")[1:]}
        a14 = {l.split("|")[1].strip(): [x.strip() for x in l.split("|")] for l in self.gptq_rows("A14.")[1:-1]}
        self.assertEqual(set(a8), set(a14))
        for k in a8:
            self.assertEqual((a14[k][2], a14[k][3]), (a8[k][11], a8[k][12]), k)  # legacy INT4 C and W

    def test_gptq_external_panel_has_six_languages_plus_the_median_row(self):
        self.assertEqual(len(self.gptq_rows("A16.")) - 1, 6 + 1)

    def test_gptq_external_legacy_columns_equal_the_a9_int4_columns(self):
        a9 = {l.split("|")[1].strip(): [x.strip() for x in l.split("|")] for l in self.gptq_rows("A9.")[1:]}
        a16 = {l.split("|")[1].strip(): [x.strip() for x in l.split("|")] for l in self.gptq_rows("A16.")[1:-1]}
        self.assertEqual(set(a9), set(a16))
        for k in a9:
            self.assertEqual((a16[k][2], a16[k][3]), (a9[k][11], a9[k][12]), k)

    def test_gptq_external_paired_differences_match_d111(self):
        rows = [[x.strip() for x in l.split("|")] for l in self.gptq_rows("A16.")[1:-1]]
        mean = lambda i, j: sum(float(r[i]) - float(r[j]) for r in rows) / len(rows)
        self.assertAlmostEqual(mean(6, 4), -0.1610, delta=0.003)  # GPTQ C minus matched RTN C (D-111)
        self.assertAlmostEqual(mean(7, 5), -0.2996, delta=0.003)  # GPTQ W minus matched RTN W

    def test_gptq_study_g_medians_match_d109(self):
        row = [l for l in self.gptq_rows("A15.") if "median of the 27" in l][0]
        self.assertIn("| +0.169 | +0.323 | +0.281 | +0.187 |", row)

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
