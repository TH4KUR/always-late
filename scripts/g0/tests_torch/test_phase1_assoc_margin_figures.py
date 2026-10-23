import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_assoc_margin as am
import phase1_assoc_margin_figures as mf
from test_phase1_assoc_margin import planted


def tagged(fig, gid):
    return [a for a in fig.findobj() if a.get_gid() == gid]


def texts(fig):
    return [t.get_text() for t in fig.findobj(lambda a: hasattr(a, "get_text") and a.get_text())]


def roundtrip(res):
    """As read from analysis.json: numpy types converted, NaN as None."""
    return json.loads(json.dumps(am._clean(res), allow_nan=False))


BETA = {"INT16": 0.0, "INT12": 0.0, "INT8": 0.5, "INT7": 0.7, "INT6": 1.5, "INT5": 1.8, "INT4": 2.0}
BASE = {"INT16": -4.0, "INT12": -4.0, "INT8": -3.2, "INT7": -3.0, "INT6": -2.4, "INT5": -1.5, "INT4": -0.4}


class Figure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = roundtrip(am.analyse(planted(11, n_per_lang=150, beta=BETA, base=BASE), n_boot=40, seed=0))

    def test_one_line_per_precision_with_the_lost_fractions_at_the_five_quintiles(self):
        fig = mf.fig_loss_by_margin(self.res, "light", "exploratory")
        for p in ("INT6", "INT5", "INT4"):
            line = tagged(fig, f"line:{p}")[0]; np.testing.assert_allclose(line.get_xdata(), [1, 2, 3, 4, 5]); np.testing.assert_allclose(line.get_ydata(), [b["fraction"] for b in self.res["by_precision"][p]["bins_m0"]])
        self.assertEqual(tagged(fig, "line:INT8"), [])                                                                                                          # only the three precisions of the story and the floor

    def test_the_floor_is_a_dashed_reference_line_with_the_int16_fractions(self):
        fig = mf.fig_loss_by_margin(self.res, "light", "x"); line = tagged(fig, "floor")[0]
        np.testing.assert_allclose(line.get_ydata(), [b["fraction"] for b in self.res["by_precision"]["INT16"]["bins_m0"]]); self.assertEqual(line.get_linestyle(), "--")

    def test_the_interval_bars_are_the_wilson_bounds(self):
        fig = mf.fig_loss_by_margin(self.res, "light", "x"); bins = self.res["by_precision"]["INT4"]["bins_m0"]; segs = tagged(fig, "ci:INT4")[0].get_segments()
        np.testing.assert_allclose([s[0][1] for s in segs], [b["lo"] for b in bins]); np.testing.assert_allclose([s[1][1] for s in segs], [b["hi"] for b in bins]); np.testing.assert_allclose([s[0][0] for s in segs], [1, 2, 3, 4, 5])

    def test_the_legend_gives_the_auroc_of_the_margin_for_each_precision_and_names_the_floor(self):
        leg = [t.get_text() for t in fig_legend(mf.fig_loss_by_margin(self.res, "light", "x"))]
        for p in ("INT6", "INT5", "INT4"):
            self.assertIn(f"{p}, AUROC {self.res['by_precision'][p]['auroc']['m0']['mean']:.2f}", leg)
        self.assertEqual(len(leg), 4); self.assertTrue(any("INT16" in x and "rounding only" in x for x in leg))

    def test_the_axis_starts_at_zero_and_contains_every_interval(self):
        ax = mf.fig_loss_by_margin(self.res, "light", "x").axes[0]; lo, hi = ax.get_ylim(); top = max(b["hi"] for p in ("INT6", "INT5", "INT4", "INT16") for b in self.res["by_precision"][p]["bins_m0"])
        self.assertEqual(lo, 0.0); self.assertGreaterEqual(hi, top); self.assertLess(hi, top * 1.3)

    def test_the_ticks_are_the_five_quintiles_and_the_axis_label_says_which_end_is_borderline(self):
        ax = mf.fig_loss_by_margin(self.res, "light", "x").axes[0]; self.assertEqual([t.get_text() for t in ax.get_xticklabels()], ["1", "2", "3", "4", "5"]); self.assertIn("most borderline", ax.get_xlabel())

    def test_each_precision_is_named_at_the_left_end_of_its_line(self):
        fig = mf.fig_loss_by_margin(self.res, "light", "x")
        for p in ("INT6", "INT5", "INT4"):
            lab = tagged(fig, f"label:{p}")[0]; x, y = lab.get_position(); self.assertEqual(lab.get_text(), p); self.assertLess(x, 1.0); self.assertAlmostEqual(y, self.res["by_precision"][p]["bins_m0"][0]["fraction"])

    def test_an_undefined_auroc_is_written_as_n_a_and_the_precision_is_still_drawn(self):
        import copy
        r = copy.deepcopy(self.res); r["by_precision"]["INT4"]["auroc"]["m0"]["mean"] = None; fig = mf.fig_loss_by_margin(r, "light", "x")
        self.assertIn("INT4, AUROC n/a", [x.get_text() for x in fig_legend(fig)]); self.assertEqual(len(tagged(fig, "line:INT4")), 1)

    def test_the_number_of_features_has_a_thousands_separator(self):
        import copy
        r = copy.deepcopy(self.res); r["n_established"] = 6663; self.assertIn("n = 6,663", " ".join(" ".join(texts(mf.fig_loss_by_margin(r, "light", "x"))).split()))

    def test_the_three_precisions_have_three_different_colours_in_each_theme(self):
        for theme in ("light", "dark"):
            fig = mf.fig_loss_by_margin(self.res, theme, "x"); cols = {tagged(fig, f"line:{p}")[0].get_color() for p in ("INT6", "INT5", "INT4")}; self.assertEqual(len(cols), 3)

    def test_the_writer_passes_its_keyword_arguments_to_the_figure(self):
        with tempfile.TemporaryDirectory() as d:
            paths = mf.write_margin_figure(self.res, d, "f", "x", themes=("light",), formats=("svg",), precisions=("INT4",)); svg = Path(paths[0]).read_text()
        self.assertIn("INT4, AUROC", svg); self.assertNotIn("INT6, AUROC", svg)

    def test_the_direct_labels_are_inside_the_axis_limits(self):
        ax = mf.fig_loss_by_margin(self.res, "light", "x").axes[0]; lo, hi = ax.get_xlim()
        for lab in ax.texts:
            if (lab.get_gid() or "").startswith("label:"):
                self.assertGreater(lab.get_position()[0], lo + 0.3)                                                                          # room for the text itself to the left of its anchor

    def test_a_precision_missing_from_the_results_is_not_drawn_and_is_named_in_the_subtitle(self):
        import copy
        r = copy.deepcopy(self.res); del r["by_precision"]["INT5"]; fig = mf.fig_loss_by_margin(r, "light", "x")
        self.assertEqual(tagged(fig, "line:INT5"), []); self.assertTrue(any("not drawn: INT5" in " ".join(x.split()) for x in texts(fig))); self.assertFalse(any("not drawn" in x for x in texts(mf.fig_loss_by_margin(self.res, "light", "x"))))

    def test_no_precision_at_all_is_an_error(self):
        import copy
        r = copy.deepcopy(self.res); r["by_precision"] = {"INT16": r["by_precision"]["INT16"]}
        with self.assertRaises(ValueError):
            mf.fig_loss_by_margin(r, "light", "x")

    def test_the_subtitle_states_the_status_the_number_of_features_and_the_direction_of_the_axis(self):
        tx = " ".join(" ".join(texts(mf.fig_loss_by_margin(self.res, "light", "exploratory, test"))).split())
        self.assertIn("exploratory, test", tx); self.assertIn(f"n = {self.res['n_established']:,}", tx); self.assertIn("1 = the most borderline fifth", tx); self.assertIn("Wilson 95 % intervals", tx)

    def test_a_cohort_without_a_language_says_so_in_the_subtitle(self):
        import copy
        r = copy.deepcopy(self.res); r["excluded"] = ["aaa"]; tx = " ".join(" ".join(texts(mf.fig_loss_by_margin(r, "light", "x"))).split()); self.assertIn("without aaa", tx)

    def test_nothing_is_clipped_in_either_theme_and_the_files_are_written(self):
        for theme in ("light", "dark"):
            fig = mf.fig_loss_by_margin(self.res, theme, "exploratory; the D-035 statuses of the eq stream, 27 languages"); fig.canvas.draw(); r = fig.canvas.get_renderer()
            shown = list(fig.texts) + [x for ax in fig.axes for x in list(ax.texts) + [ax.title, ax.xaxis.label, ax.yaxis.label]] + [x for lg in fig.legends for x in lg.get_texts()]
            for x in [s for s in shown if s.get_text()]:
                bb = x.get_window_extent(r); self.assertGreaterEqual(bb.x0, -1.0, x.get_text()); self.assertLessEqual(bb.x1, fig.bbox.width + 1.0, x.get_text()); self.assertGreaterEqual(bb.y0, -1.0, x.get_text()); self.assertLessEqual(bb.y1, fig.bbox.height + 1.0, x.get_text())
        with tempfile.TemporaryDirectory() as d:
            paths = mf.write_margin_figure(self.res, d, "fig_loss_by_margin", "exploratory", themes=("light", "dark"), formats=("png", "svg"))
            self.assertEqual(sorted(Path(p).name for p in paths), ["fig_loss_by_margin_dark.png", "fig_loss_by_margin_dark.svg", "fig_loss_by_margin_light.png", "fig_loss_by_margin_light.svg"]); self.assertTrue(all(Path(p).stat().st_size > 2000 for p in paths))

    def test_the_command_line_draws_the_chosen_cohort_from_analysis_json(self):
        doc = {"all": self.res, "without_excluded": {**self.res, "excluded": ["aaa"]}}
        with tempfile.TemporaryDirectory() as d:
            a = Path(d) / "analysis.json"; a.write_text(json.dumps(doc)); mf.main(["--analysis", str(a), "--out", str(Path(d) / "o"), "--note", "exploratory, test"])
            self.assertTrue((Path(d) / "o" / "fig_loss_by_margin_light.png").exists()); self.assertTrue((Path(d) / "o" / "fig_loss_by_margin_dark.svg").exists())
            mf.main(["--analysis", str(a), "--out", str(Path(d) / "o2"), "--note", "exploratory, test", "--cohort", "without_excluded", "--stem", "fig_ex"]); self.assertTrue((Path(d) / "o2" / "fig_ex_light.png").exists())
            self.assertNotIn("without aaa", (Path(d) / "o" / "fig_loss_by_margin_light.svg").read_text()); self.assertIn("without aaa", (Path(d) / "o2" / "fig_ex_light.svg").read_text())                  # the cohort option reaches the figure


def fig_legend(fig):
    return fig.legends[0].get_texts()


if __name__ == "__main__":
    unittest.main()
