"""N10 tests for scripts/g0/phase1_summary.py (decision D-024; pre-specified in handoff 006 section 4). The summary's NUMBERS are generated from the committed JSON/CSV so that they cannot
drift from the files; the static prose around them contains no result numbers. Stdlib only; synthetic fixture trees with distinctive values."""
import csv, json, re, shutil, sys, tempfile, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
import phase1_summary as ps   # noqa: E402

CSV_HEADER = "layer,condition,method,bits,perplexity,ppl_delta_pct,n_active_features,mean_corr,median_corr,survived_>0.9_pct,degraded_0.5_0.9_pct,damaged_<0.5_pct,n_tokens,n_total_features,loss".split(",")


def test(delta, p, ci=(-0.02, 0.01), n=100):
    return {"delta": delta, "p": p, "ci": list(ci), "survival_selective": 0.9, "survival_control": 0.92, "n_selective": n, "n_control": n}


def write_stream(root, regime, k, with_mixed=True):
    """k shifts every distinctive value so that the two streams differ."""
    d = Path(root) / regime / "eq"; (d / "thresholds").mkdir(parents=True); (d / "magnitude").mkdir(); (d / "language_level").mkdir(); (d / "mixed_variants").mkdir(); (d / "margin").mkdir()
    rows = [("FP16 baseline", 32, 20.0, 0.0, 9000, 1.0, 1.0, 100.0), ("RTN INT8", 8, 20.1, 0.08 + k, 9000, 0.99, 0.9906 + k / 1000, 99.99 - k), ("RTN INT7", 7, 20.2, 0.4 + k, 9000, 0.98, 0.9829 + k / 1000, 99.97 - k),
            ("RTN INT6", 6, 20.5, 2.67 + k, 9000, 0.96, 0.9629 + k / 1000, 94.94 - k), ("RTN INT5", 5, 21.7, 11.5 + k, 9000, 0.9, 0.9143 + k / 1000, 59.0 - k), ("RTN INT4", 4, 39.7, 103.9 + k, 9000, 0.7, 0.7495 + k / 1000, 13.75 - k),
            ("RTN INT16", 16, 20.0, 0.003, 9000, 0.99, 0.9957, 100.0), ("RTN INT12", 12, 20.0, 0.035, 9000, 0.99, 0.9957, 100.0), ("restore-only control (INT0)", 0, 20.0, 0.0, 9000, 1.0, 1.0, 100.0)]
    with open(d / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER); w.writeheader()
        for cond, bits, ppl, dppl, na, mc, med, surv in rows:
            w.writerow({"layer": 12, "condition": cond, "method": "RTN", "bits": bits, "perplexity": ppl, "ppl_delta_pct": dppl, "n_active_features": na, "mean_corr": mc, "median_corr": med, "survived_>0.9_pct": surv,
                        "degraded_0.5_0.9_pct": 100 - surv, "damaged_<0.5_pct": 0.0, "n_tokens": 1000, "n_total_features": 16384, "loss": 3.0})
    (d / "analysis.json").write_text(json.dumps({"tests": {"T1_int7": test(-0.0028 - k / 100, 0.2571, (-0.0063, 0.0013), 8062), "T2_int6": test(-0.0144 - k / 100, 0.0427, (-0.0265, -0.0013), 8062),
                                                           "T3_tier": {"slope": -0.0048 + k / 100, "p": 0.9902, "high_minus_low": -0.0082, "n_languages": 27, "tost_slope": {"interval": [-0.0075, -0.0016], "equivalent": True, "bound": 0.02},
                                                                       "tost_high_minus_low": {"interval": [-0.0129, -0.0031], "equivalent": k == 0, "bound": 0.02}}},
                                                  "holm": {"T1_int7": 0.5141, "T2_int6": 0.1281, "T3_tier": 0.9902}}))
    def trow(label, tau, d6, h6):
        return {"label": label, "tau": tau, "n_selective": 10, "n_controls": 10, "control_strata_short": [], "tests": {"T1_int7": test(-0.001, 0.5), "T2_int6": test(d6, 0.001), "T3_tier": {"slope": 0.0, "p": 0.5}}, "holm": {"T1_int7": 1.0, "T2_int6": h6, "T3_tier": 1.0}}
    (d / "thresholds" / "thresholds.json").write_text(json.dumps({"rows": [trow("null", 1.83, -0.0144, 0.1281), trow("2.0", 2.0, -0.0102, 0.4721), trow("3.0", 3.0, -0.0214, 0.0165), trow("5.0", 5.0, -0.0716 - k / 100, 0.0003 + k / 1000)]}))
    by = lambda dl, h, r: {"reading": r, "retained": 1.0, "rate_only_delta": -0.0716, "matched_delta": dl, "matched_holm": h}
    (d / "magnitude" / "magnitude.json").write_text(json.dumps({"readings": {"5.0": {"by_bins": {"3": by(0.008, 1.0, "explained by magnitude"), "5": by(0.0413, 0.0048, "reversed"), "10": by(0.0458 + k / 100, 0.0012, "reversed")},
                                                                          "final": {"bins": 10, "reading": "reversed", "retained": 1.0, "all_agree": False, "note": ""}}}}))
    mb = lambda dl, h: {"matched_delta": dl, "matched_holm": h, "retained": 1.0, "reading": "reversed"}
    (d / "margin" / "margin_summary.json").write_text(json.dumps({"feature_level": {"by_bins": {"3": mb(0.0095, 1.0), "10": mb(0.0532 + k / 100, 0.0006 + k / 1000)}}, "readings": {"i": "remains", "ii": "no more than raw activation" if k == 0 else "margin carries more than raw magnitude"}}))
    if with_mixed:
        v = lambda c, p: {"S_rank": {"coef": c, "se": 0.3, "p": p}}
        (d / "mixed_variants" / "mixed_variants.json").write_text(json.dumps({"variants": {"full_all": v(1.194, 0.00043), "no_tier_all": v(-0.812, 0.0044)}, "reading": {"label": "explained by the tier terms", "terms": ["no_tier_all"]}}))
    (d / "language_level" / "language_level.json").write_text(json.dumps({"widths": {"7": {"mean_by_tier": {"0": 0.965, "1": 0.996, "2": 0.998}, "slope": 0.0165, "p": 0.0001, "holm_p": 0.0002, "slope_fertility": 0.0056, "p_fertility": 0.0713},
                                                                                  "6": {"mean_by_tier": {"0": 0.694 - k / 100, "1": 0.906, "2": 0.951}, "slope": 0.1234 - k / 100, "p": 0.0001, "holm_p": 0.0002, "slope_fertility": 0.0626, "p_fertility": 0.0032}},
                                                                    "reading": "depends on tier; beyond fertility",
                                                                    "activation_scale": {"spearman_tier_scale": 0.833, "widths": {"7": {"slope_scale": -0.0074, "p_scale": 0.977, "slope_both": -0.0065, "p_both": 0.971, "spearman_scale_survival": 0.957},
                                                                                                                                "6": {"slope_scale": -0.0081 + k / 1000, "p_scale": 0.667, "slope_both": -0.003, "p_both": 0.577, "spearman_scale_survival": 0.966}},
                                                                                        "reading": {"joint": "explained jointly by tokenization cost and activation scale", "scale_alone_explains": True}}}))
    sp = lambda r, p: {"spearman": {"rho": r, "p_one_sided": p, "p_two_sided": 2 * p}, "ols": {}}
    (d / "behavior.json").write_text(json.dumps({"widths": {"6": {"survival": sp(0.458 + k / 100, 0.00816), "damaged": sp(0.172, 0.196)}, "7": {"survival": sp(0.539, 0.00186), "damaged": {"undefined": "constant"}}}}))
    return d


def tree(with_none_mixed=False):
    t = Path(tempfile.mkdtemp()); write_stream(t, "mask", 0, True); write_stream(t, "none", 1, with_none_mixed); return t


class Cells(unittest.TestCase):
    def setUp(self):
        self.t = tree(); self.addCleanup(shutil.rmtree, self.t, ignore_errors=True); self.out = ps.render_generated(self.t)

    def has(self, *needles):
        for n in needles: self.assertIn(n, self.out, n)

    def test_ladder_cells_come_from_summary_csv_for_both_streams(self):
        self.has("| INT8 | 99.99 % | 0.9906 | +0.08 % | 98.99 % | 0.9916 | +1.08 % |", "| INT6 | 94.94 % | 0.9629 | +2.67 % | 93.94 % | 0.9639 | +3.67 % |", "| INT5 | 59.00 % | 0.9143 | +11.50 % | 58.00 % | 0.9153 | +12.50 % |")
        self.has("| INT16 (noise floor) | 100.00 % | 0.9957 | +0.00 % |", "| INT0 (restore only) | 100.00 % | 1.0000 | +0.00 % |")

    def test_confirmatory_family_cells_come_from_analysis_json(self):
        self.has("-0.28 pp [-0.63, +0.13]", "p 0.257", "Holm 0.514", "-1.44 pp [-2.65, -0.13]", "p 0.0427", "Holm 0.128", "-0.0048", "p 0.99", "Holm 0.99")
        self.has("equivalent within +-0.02: yes", "equivalent within +-0.02: no")                    # the two TOST outcomes of the fixture (mask yes/yes, none yes/no)

    def test_each_additional_analysis_row_has_its_headline_number_its_label_and_its_ids(self):
        self.has("N1", "+0.46", "0.00816")                                                              # behavioural link: INT6 Spearman, one-sided p
        self.has("N2", "-7.16 pp", "Holm 0.0003")                                                       # INT6 contrast at S > 5, firing rate only
        self.has("N6", "+4.58 pp", "Holm 0.0012", "reversed")                                           # matched on 10 magnitude bins at S > 5, headline reading
        self.has("N7", "-0.81", "+1.19", "explained by the tier terms")                                 # S~ without the tier terms and in the full model
        self.has("N8", "69.4 / 90.6 / 95.1", "+12.3 pp", "Holm 0.0002", "+6.3 pp")                      # INT6 mean survival by tier and the slopes
        self.has("N9", "-0.8 pp", "p 0.667", "+0.97", "explained jointly by tokenization cost and activation scale")
        self.has("N11", "+5.32 pp", "Holm 0.0006", "(i) remains", "(ii) no more than raw activation")                                   # margin-matched INT6 contrast at S > 5 (10 bins) and the two readings

    def test_the_bos_free_column_uses_that_streams_values_and_a_stream_without_a_file_says_not_run(self):
        self.has("-8.16 pp", "+5.58 pp", "+11.3 pp", "not run", "+6.32 pp", "(ii) margin carries more than raw magnitude")                   # none stream (k = 1): N11 +6.32 pp and the other reading                                         # none stream (k = 1): N2 -8.16, N6 +5.58, N8 slope 0.1134 -> +11.3, N7 absent

    def test_status_labels_ids_and_the_legend_are_present_and_the_confirmatory_family_is_exactly_t1_t2_t3(self):
        self.has("confirmatory (proposal section 6; D-014)", "additional, post hoc thresholds (D-019)", "additional, pre-specified before running (D-020)", "exploratory (D-014); variants pre-specified (D-021)",
                 "additional, motivated post hoc and pre-specified before running (D-022)", "additional, pre-specified before running (D-023)", "planned in the team's table, definitions provisional (D-018)",
                 "additional, pre-specified before running (D-025)")
        self.assertEqual(len(re.findall(r"\| T[123] ", self.out.split("Additional analyses")[0])), 3)
        self.has("reports/phase1/mask/eq/language_level/language_level.json", "A-038", "A-039", "A-035", "A-037", "A-040", "reports/phase1/mask/eq/margin/margin_summary.json")


class Errors(unittest.TestCase):
    def test_a_missing_file_is_an_error_naming_it(self):
        t = tree(); self.addCleanup(shutil.rmtree, t, ignore_errors=True); (t / "mask" / "eq" / "magnitude" / "magnitude.json").unlink()
        with self.assertRaisesRegex(ps.SummaryError, "magnitude.json"): ps.render_generated(t)

    def test_a_missing_key_is_an_error_naming_the_path(self):
        t = tree(); self.addCleanup(shutil.rmtree, t, ignore_errors=True); p = t / "mask" / "eq" / "analysis.json"; d = json.loads(p.read_text()); del d["tests"]["T2_int6"]["delta"]; p.write_text(json.dumps(d))
        with self.assertRaisesRegex(ps.SummaryError, r"analysis.json.*tests/T2_int6/delta"): ps.render_generated(t)

    def test_the_bos_free_stream_may_lack_the_mixed_variants_but_the_primary_stream_may_not(self):
        t = tree(); self.addCleanup(shutil.rmtree, t, ignore_errors=True); ps.render_generated(t)
        (t / "mask" / "eq" / "mixed_variants" / "mixed_variants.json").unlink()
        with self.assertRaisesRegex(ps.SummaryError, "mixed_variants.json"): ps.render_generated(t)


class Markers(unittest.TestCase):
    def setUp(self):
        self.t = tree(); self.addCleanup(shutil.rmtree, self.t, ignore_errors=True); self.md = self.t / "summary.md"

    def test_only_the_text_between_the_markers_is_replaced_and_the_rest_stays_byte_identical(self):
        before, after = "# Title\n\nstatic text before\n\n", "\n\nstatic text after\n"; self.md.write_text(before + ps.BEGIN + "\nOLD\n" + ps.END + after)
        ps.write_summary(self.t, self.md); new = self.md.read_text()
        self.assertTrue(new.startswith(before + ps.BEGIN + "\n")); self.assertTrue(new.endswith(ps.END + after)); self.assertNotIn("OLD", new); self.assertIn("| INT6 |", new)

    def test_missing_or_duplicated_markers_and_a_missing_file_are_errors(self):
        self.md.write_text("no markers here")
        with self.assertRaisesRegex(ps.SummaryError, "marker"): ps.write_summary(self.t, self.md)
        self.md.write_text(ps.BEGIN + ps.END + ps.BEGIN + ps.END)
        with self.assertRaisesRegex(ps.SummaryError, "marker"): ps.write_summary(self.t, self.md)
        with self.assertRaisesRegex(ps.SummaryError, "does not exist"): ps.write_summary(self.t, self.t / "absent.md")

    def test_check_is_true_after_writing_and_false_after_a_number_in_the_block_is_edited(self):
        self.md.write_text("x\n" + ps.BEGIN + "\n" + ps.END + "\n"); ps.write_summary(self.t, self.md); self.assertTrue(ps.check(self.t, self.md))
        self.md.write_text(self.md.read_text().replace("94.94", "94.95")); self.assertFalse(ps.check(self.t, self.md))

    def test_init_creates_the_static_text_with_exactly_one_pair_of_markers_and_never_overwrites(self):
        p = self.t / "fresh.md"; ps.init_summary(p); text = p.read_text()
        self.assertEqual(text.count(ps.BEGIN), 1); self.assertEqual(text.count(ps.END), 1); self.assertIn("not paper prose", text)
        with self.assertRaisesRegex(ps.SummaryError, "already exists"): ps.init_summary(p)

    def test_writing_twice_gives_the_same_file(self):
        self.md.write_text("x\n" + ps.BEGIN + "\n" + ps.END + "\n"); ps.write_summary(self.t, self.md); a = self.md.read_text(); ps.write_summary(self.t, self.md); self.assertEqual(a, self.md.read_text())


class Committed(unittest.TestCase):
    """The committed summary against the committed result files: the numbers cannot drift."""
    MD = ROOT / "reports" / "phase1_summary.md"; FORBIDDEN = re.compile(r"%|\bpp\b|\d\.\d|\bp\s*[=<]")

    def test_the_committed_summary_equals_what_the_script_regenerates_from_the_committed_files(self):
        self.assertTrue(self.MD.exists(), "reports/phase1_summary.md has not been generated"); self.assertTrue(ps.check(ROOT / "reports" / "phase1", self.MD))

    def test_the_static_prose_has_no_result_numbers_and_says_it_is_not_paper_prose(self):
        text = self.MD.read_text(); head, rest = text.split(ps.BEGIN); tail = rest.split(ps.END)[1]; prose = head + tail
        self.assertEqual(self.FORBIDDEN.findall(prose), [], "result-like numbers in the static prose"); self.assertIn("not paper prose", prose)
        self.assertEqual(self.FORBIDDEN.findall(ps.STATIC_TEXT), [])


if __name__ == "__main__":
    unittest.main()
