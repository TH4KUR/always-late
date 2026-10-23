import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_assoc_function_figures as ff
import phase1_assoc_function_link as fl
import phase1_assoc_margin as am
from test_phase1_assoc_function_link import world


def tagged(fig, gid):
    return [a for a in fig.findobj() if a.get_gid() == gid]


def texts(fig):
    return [t.get_text() for t in fig.findobj(lambda a: hasattr(a, "get_text") and a.get_text())]


def legend_texts(fig):
    return [x.get_text() for x in fig.legends[0].get_texts()]


def roundtrip(res):
    return json.loads(json.dumps(am._clean(res), allow_nan=False))


class Figure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        feats, wf, states, sample = world(21, n_per_lang=150, assoc_beta=1.8, w1_beta=0.01)
        res = fl.analyse(feats, wf, states, sample, n_boot=40, seed=0)
        cls.res = roundtrip(res)
        cls.res["by_precision"]["INT4"]["auroc_function"] = {"mean": None, "lo": None, "hi": None, "n_languages": 0, "per_language": {}}                      # as in the real run: no contrast at INT4
        cls.P = ["INT8", "INT6", "INT5", "INT4"]

    def test_the_markers_sit_at_the_auroc_means_with_the_first_precision_on_the_top_row(self):
        fig = ff.fig_margin_auroc(self.res, "light", "exploratory"); a = tagged(fig, "assoc")[0]; f = tagged(fig, "function")[0]
        np.testing.assert_allclose(a.get_xdata(), [self.res["by_precision"][p]["auroc_association"]["mean"] for p in self.P])
        np.testing.assert_allclose(f.get_xdata(), [self.res["by_precision"][p]["auroc_function"]["mean"] for p in self.P[:3]])                                   # INT4 has no function marker
        ya = list(a.get_ydata()); self.assertEqual(ya, sorted(ya, reverse=True)); self.assertEqual(len(set(ya)), 4)
        self.assertEqual([t.get_text() for t in fig.axes[0].get_yticklabels()][::-1], self.P)

    def test_the_function_markers_are_on_the_rows_of_their_precisions(self):
        fig = ff.fig_margin_auroc(self.res, "light", "x"); a = tagged(fig, "assoc")[0]; f = tagged(fig, "function")[0]; np.testing.assert_allclose(f.get_ydata(), list(a.get_ydata())[:3])

    def test_the_interval_bars_run_from_the_lower_to_the_upper_limit_for_each_series(self):
        fig = ff.fig_margin_auroc(self.res, "light", "x")
        segs = tagged(fig, "ci:assoc")[0].get_segments(); np.testing.assert_allclose([s[0][0] for s in segs], [self.res["by_precision"][p]["auroc_association"]["lo"] for p in self.P]); np.testing.assert_allclose([s[1][0] for s in segs], [self.res["by_precision"][p]["auroc_association"]["hi"] for p in self.P])
        segs = tagged(fig, "ci:function")[0].get_segments(); np.testing.assert_allclose([s[0][0] for s in segs], [self.res["by_precision"][p]["auroc_function"]["lo"] for p in self.P[:3]]); np.testing.assert_allclose([s[1][0] for s in segs], [self.res["by_precision"][p]["auroc_function"]["hi"] for p in self.P[:3]])

    def test_a_gray_link_joins_the_two_markers_of_a_row_and_an_undefined_row_has_none(self):
        fig = ff.fig_margin_auroc(self.res, "light", "x"); segs = tagged(fig, "link")[0].get_segments(); self.assertEqual(len(segs), 3)
        for s, p in zip(segs, self.P[:3]):
            self.assertAlmostEqual(min(s[0][0], s[1][0]), min(self.res["by_precision"][p]["auroc_association"]["mean"], self.res["by_precision"][p]["auroc_function"]["mean"])); self.assertEqual(s[0][1], s[1][1])

    def test_the_reference_line_at_one_half_is_dashed_and_inside_the_axis(self):
        fig = ff.fig_margin_auroc(self.res, "light", "x"); ref = tagged(fig, "ref")[0]; self.assertAlmostEqual(float(ref.get_xdata()[0]), 0.5); self.assertEqual(ref.get_linestyle(), "--"); lo, hi = fig.axes[0].get_xlim(); self.assertLess(lo, 0.5); self.assertGreater(hi, 0.5)

    def test_a_row_without_a_function_contrast_says_so_instead_of_drawing_a_marker(self):
        fig = ff.fig_margin_auroc(self.res, "light", "x"); lab = tagged(fig, "undefined:INT4"); self.assertEqual(len(lab), 1); self.assertIn("no function-event contrast", lab[0].get_text()); self.assertEqual(tagged(fig, "undefined:INT6"), [])

    def test_the_undefined_text_stays_inside_the_axes_on_whichever_side_has_room(self):
        for theme in ("light", "dark"):
            for assoc in ((0.74, 0.70, 0.78), (0.93, 0.90, 0.97), (0.55, 0.52, 0.58)):                                                                           # room on the right, on the left only, on both
                r = copy.deepcopy(self.res); r["by_precision"]["INT4"]["auroc_association"] = {"mean": assoc[0], "lo": assoc[1], "hi": assoc[2], "n_languages": 5, "per_language": {}}
                fig = ff.fig_margin_auroc(r, theme, "x"); fig.canvas.draw(); ax = fig.axes[0]; bb = tagged(fig, "undefined:INT4")[0].get_window_extent(fig.canvas.get_renderer()); ab = ax.get_window_extent()
                self.assertGreaterEqual(bb.x0, ab.x0 - 1); self.assertLessEqual(bb.x1, ab.x1 + 1)

    def test_the_legend_names_both_series_and_the_reference(self):
        leg = legend_texts(ff.fig_margin_auroc(self.res, "light", "x")); self.assertEqual(len(leg), 3); self.assertIn("loss of the D-035 association", leg[0]); self.assertIn("function event", leg[1]); self.assertIn("0.5", leg[2]); self.assertEqual(leg[1], "AUROC for the function event (W1 below the floor)")

    def test_the_two_series_have_different_colours_and_shapes_in_each_theme(self):
        for theme in ("light", "dark"):
            fig = ff.fig_margin_auroc(self.res, theme, "x"); a, f = tagged(fig, "assoc")[0], tagged(fig, "function")[0]; self.assertNotEqual(a.get_color(), f.get_color()); self.assertNotEqual(a.get_marker(), f.get_marker())

    def test_the_series_are_named_on_the_top_row(self):
        fig = ff.fig_margin_auroc(self.res, "light", "x"); a, f = tagged(fig, "label:assoc")[0], tagged(fig, "label:function")[0]; self.assertEqual(a.get_text(), "association loss"); self.assertEqual(f.get_text(), "function event")
        self.assertAlmostEqual(a.get_position()[0], self.res["by_precision"]["INT8"]["auroc_association"]["mean"]); self.assertAlmostEqual(f.get_position()[0], self.res["by_precision"]["INT8"]["auroc_function"]["mean"])

    def test_the_axis_contains_every_interval_and_does_not_exceed_one_by_more_than_a_margin(self):
        ax = ff.fig_margin_auroc(self.res, "light", "x").axes[0]; lo, hi = ax.get_xlim(); lows = [self.res["by_precision"][p][k]["lo"] for p in self.P for k in ("auroc_association", "auroc_function") if self.res["by_precision"][p][k]["lo"] is not None]
        highs = [self.res["by_precision"][p][k]["hi"] for p in self.P for k in ("auroc_association", "auroc_function") if self.res["by_precision"][p][k]["hi"] is not None]; self.assertLessEqual(lo, min(lows)); self.assertGreaterEqual(hi, max(highs)); self.assertLessEqual(hi, 1.06); self.assertGreater(lo, 0.0)

    def test_the_axis_has_room_below_the_lowest_interval_and_keeps_the_reference_in_view_when_every_auroc_is_high(self):
        ax = ff.fig_margin_auroc(self.res, "light", "x").axes[0]; lows = [self.res["by_precision"][p][k]["lo"] for p in self.P for k in ("auroc_association", "auroc_function") if self.res["by_precision"][p][k]["lo"] is not None]
        self.assertLess(ax.get_xlim()[0], min(lows))
        r = copy.deepcopy(self.res)
        for p in self.P:
            r["by_precision"][p]["auroc_association"] = {"mean": 0.9, "lo": 0.85, "hi": 0.95, "n_languages": 9, "per_language": {}}
            if p != "INT4":
                r["by_precision"][p]["auroc_function"] = {"mean": 0.8, "lo": 0.75, "hi": 0.85, "n_languages": 9, "per_language": {}}
        lo, hi = ff.fig_margin_auroc(r, "light", "x").axes[0].get_xlim(); self.assertLess(lo, 0.5)                                                              # the line of no information stays on the axis

    def test_the_number_of_features_has_a_thousands_separator(self):
        r = copy.deepcopy(self.res); r["n_joined"] = 1449; self.assertIn("n = 1,449", " ".join(" ".join(texts(ff.fig_margin_auroc(r, "light", "x"))).split()))

    def test_the_subtitle_states_the_status_the_number_of_features_the_same_features_and_the_interval_basis(self):
        tx = " ".join(" ".join(texts(ff.fig_margin_auroc(self.res, "light", "exploratory, test"))).split())
        self.assertIn("exploratory, test", tx); self.assertIn(f"n = {self.res['n_joined']:,}", tx); self.assertIn("the same features", tx); self.assertIn("95 % intervals over languages", tx); self.assertIn("0.5 = no information", tx)

    def test_a_cohort_without_a_language_says_so(self):
        r = copy.deepcopy(self.res); r["excluded"] = ["aaa"]; self.assertIn("without aaa", " ".join(" ".join(texts(ff.fig_margin_auroc(r, "light", "x"))).split()))

    def test_a_precision_absent_from_the_results_is_not_drawn_and_is_named(self):
        r = copy.deepcopy(self.res); del r["by_precision"]["INT5"]; fig = ff.fig_margin_auroc(r, "light", "x")
        self.assertEqual([t.get_text() for t in fig.axes[0].get_yticklabels()][::-1], ["INT8", "INT6", "INT4"]); self.assertTrue(any("not drawn: INT5" in " ".join(x.split()) for x in texts(fig)))

    def test_no_precision_at_all_is_an_error(self):
        r = copy.deepcopy(self.res); r["by_precision"] = {}
        with self.assertRaises(ValueError):
            ff.fig_margin_auroc(r, "light", "x")

    def test_nothing_is_clipped_in_either_theme_and_the_files_are_written(self):
        for theme in ("light", "dark"):
            fig = ff.fig_margin_auroc(self.res, theme, "exploratory; the F5 main run joined with the D-035 statuses, Gemma-2-2B layer 12"); fig.canvas.draw(); r = fig.canvas.get_renderer()
            shown = list(fig.texts) + [x for ax in fig.axes for x in list(ax.texts) + [ax.title, ax.xaxis.label, ax.yaxis.label]] + [x for lg in fig.legends for x in lg.get_texts()]
            for x in [s for s in shown if s.get_text()]:
                bb = x.get_window_extent(r); self.assertGreaterEqual(bb.x0, -1.0, x.get_text()); self.assertLessEqual(bb.x1, fig.bbox.width + 1.0, x.get_text()); self.assertGreaterEqual(bb.y0, -1.0, x.get_text()); self.assertLessEqual(bb.y1, fig.bbox.height + 1.0, x.get_text())
        with tempfile.TemporaryDirectory() as d:
            paths = ff.write_margin_auroc_figure(self.res, d, "fig_margin_auroc", "exploratory", themes=("light", "dark"), formats=("png", "svg"))
            self.assertEqual(sorted(Path(p).name for p in paths), ["fig_margin_auroc_dark.png", "fig_margin_auroc_dark.svg", "fig_margin_auroc_light.png", "fig_margin_auroc_light.svg"]); self.assertTrue(all(Path(p).stat().st_size > 2000 for p in paths))

    def test_the_writer_passes_its_keyword_arguments_and_the_command_line_draws_the_chosen_cohort(self):
        with tempfile.TemporaryDirectory() as d:
            paths = ff.write_margin_auroc_figure(self.res, d, "f", "x", themes=("light",), formats=("svg",), precisions=("INT6",)); svg = Path(paths[0]).read_text(); self.assertIn("INT6", svg); self.assertNotIn(">INT8<", svg)
            doc = {"all": self.res, "without_excluded": {**self.res, "excluded": ["aaa"]}}; a = Path(d) / "analysis.json"; a.write_text(json.dumps(doc))
            ff.main(["--analysis", str(a), "--out", str(Path(d) / "o"), "--note", "exploratory, test"]); ff.main(["--analysis", str(a), "--out", str(Path(d) / "o2"), "--note", "exploratory, test", "--cohort", "without_excluded", "--stem", "fig_ex"])
            self.assertTrue((Path(d) / "o" / "fig_margin_auroc_light.png").exists()); self.assertNotIn("without aaa", (Path(d) / "o" / "fig_margin_auroc_light.svg").read_text()); self.assertIn("without aaa", (Path(d) / "o2" / "fig_ex_light.svg").read_text())


if __name__ == "__main__":
    unittest.main()
