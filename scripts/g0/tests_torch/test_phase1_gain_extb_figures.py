import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_gain_extb as eb
import phase1_gain_extb_figures as ef
import phase1_gain_figures as gf
from test_phase1_gain_analysis import DOWN, FINE, UP
from test_phase1_gain_extb import gains_of, tables
from test_phase1_gain_figures import overlaps, texts_inside


def compare(f_int6=1.0, f_int4=1.5):
    g_sd = gains_of((1.0, f_int6, f_int4)); g_ref = gains_of((1.0, 1.0, 1.0))
    t_sd, t_ref = tables(g_sd, g_ref)
    return eb.compare_rows(t_sd, t_ref, g_sd)


def string_rows(rows):
    return [{k: str(v) for k, v in r.items()} for r in rows]


class Data(unittest.TestCase):
    def test_the_loader_converts_numbers_and_keeps_the_state_names(self):
        rows = compare()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "compare.csv"
            with open(p, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
            got = ef.load_compare(p)
        self.assertEqual(len(got), len(rows)); self.assertIsInstance(got[0]["state"], str)
        self.assertIsInstance(got[0]["ratio"], float); self.assertIsInstance(got[0]["n_languages"], float)
        self.assertAlmostEqual(got[5]["median_net_state_defined"], rows[5]["median_net_state_defined"])

    def test_the_panels_are_the_states_other_than_fp_in_ladder_order(self):
        rows = compare()
        self.assertEqual(ef.panel_states(rows), ["INT6", "INT4"])
        self.assertEqual(ef.panel_states(list(reversed(rows))), ["INT6", "INT4"])
        self.assertEqual(ef.panel_states([r for r in rows if r["state"] != "INT6"]), ["INT4"])

    def test_without_a_state_other_than_fp_there_is_nothing_to_draw(self):
        with self.assertRaises(ValueError):
            ef.panel_states([r for r in compare() if r["state"] == "FP"])

    def test_the_series_of_a_state_are_ordered_by_ratio_with_zero_ablation_first(self):
        s = ef.series(compare(), "INT4")
        self.assertEqual(s["ratios"], sorted(s["ratios"])); self.assertEqual(s["ratios"][0], 0.0)
        self.assertEqual(len(s["state_defined"]), len(s["ratios"])); self.assertEqual(len(s["reference"]), len(s["ratios"]))


class Figure(unittest.TestCase):
    def setUp(self):
        self.rows = compare()

    def test_one_panel_per_state_with_the_two_groups_as_lines(self):
        for theme in ("light", "dark"):
            fig = ef.fig_state_defined(self.rows, theme, "exploratory test")
            panels = [a for a in fig.axes if a.get_gid() and a.get_gid().startswith("panel:")]
            self.assertEqual([a.get_gid() for a in panels], ["panel:INT6", "panel:INT4"])
            for a in panels:
                self.assertEqual({l.get_gid() for l in a.lines if l.get_gid() in ("state_defined", "reference")}, {"state_defined", "reference"})
            self.assertEqual(texts_inside(fig), []); self.assertEqual(overlaps(fig), [])

    def test_the_lines_carry_the_medians_at_the_x_positions_of_the_ratios(self):
        fig = ef.fig_state_defined(self.rows, "light", "x")
        a = [a for a in fig.axes if a.get_gid() == "panel:INT4"][0]
        s = ef.series(self.rows, "INT4"); x = gf.x_positions(s["ratios"])
        sd = [l for l in a.lines if l.get_gid() == "state_defined"][0]; ref = [l for l in a.lines if l.get_gid() == "reference"][0]
        np.testing.assert_allclose(sd.get_xdata(), [x[r] for r in s["ratios"]]); np.testing.assert_allclose(sd.get_ydata(), s["state_defined"])
        np.testing.assert_allclose(ref.get_ydata(), s["reference"])
        self.assertGreater(float(np.max(sd.get_ydata())), float(np.max(ref.get_ydata())))          # the planted INT4 state-defined group harms more

    def test_the_two_lines_differ_in_colour_and_the_reference_is_the_muted_one(self):
        fig = ef.fig_state_defined(self.rows, "light", "x")
        a = fig.axes[0]
        sd = [l for l in a.lines if l.get_gid() == "state_defined"][0]; ref = [l for l in a.lines if l.get_gid() == "reference"][0]
        self.assertNotEqual(sd.get_color(), ref.get_color())

    def test_the_vertical_scale_is_symmetric_log_and_shared_by_the_panels(self):
        fig = ef.fig_state_defined(self.rows, "light", "x")
        panels = [a for a in fig.axes if a.get_gid() and a.get_gid().startswith("panel:")]
        self.assertTrue(all(a.get_yscale() == "symlog" for a in panels))
        self.assertEqual(len({tuple(round(v, 9) for v in a.get_ylim()) for a in panels}), 1)

    def test_the_vertical_axis_has_plain_decimal_ticks_inside_the_range_with_a_zero_and_never_crowded(self):
        fig = ef.fig_state_defined(self.rows, "light", "x")
        for a in [a for a in fig.axes if a.get_gid() and a.get_gid().startswith("panel:")]:
            fig.canvas.draw()
            lo, hi = a.get_ylim(); ticks = list(a.get_yticks()); labels = [t.get_text() for t in a.get_yticklabels()]
            self.assertTrue(all(lo <= y <= hi for y in ticks)); self.assertIn(0.0, ticks); self.assertGreaterEqual(len(ticks), 3); self.assertEqual(len(ticks), len(labels))
            self.assertTrue(all("e" not in l and "^" not in l and "{" not in l for l in labels), labels)
            self.assertEqual([float(l.replace("\u2212", "-")) for l in labels], ticks)
            pix = sorted(a.transData.transform((0, y))[1] for y in ticks)
            self.assertGreaterEqual(min(q - p for p, q in zip(pix, pix[1:])), 0.2 * fig.dpi)
            self.assertEqual(len(a.yaxis.get_minorticklocs()), 0)

    def test_the_x_ticks_are_the_ratios_of_the_study(self):
        fig = ef.fig_state_defined(self.rows, "light", "x")
        a = fig.axes[0]
        labels = [t.get_text() for t in a.get_xticklabels()]
        self.assertEqual(labels[0], "0"); self.assertNotIn("1", labels); self.assertIn(gf.ratio_label(DOWN[-1]), labels); self.assertIn(gf.ratio_label(UP[-1]), labels)

    def test_a_legend_names_both_groups(self):
        fig = ef.fig_state_defined(self.rows, "light", "x")
        texts = [t.get_text() for lg in fig.legends for t in lg.get_texts()]
        self.assertEqual(texts, ["state-defined group (selected in that precision)", "FP reference group (selected at full precision)"])

    def test_the_subtitle_carries_the_note_and_never_touches_a_panel_title(self):
        long_note = "exploratory, a long note about the status of the data that makes the subtitle wrap onto a third line of the figure"
        for theme in ("light", "dark"):
            fig = ef.fig_state_defined(self.rows, theme, long_note)
            self.assertIn("exploratory", " ".join(t.get_text() for t in fig.texts))
            self.assertEqual(overlaps(fig), [])

    def test_a_single_panel_figure_keeps_every_text_inside_and_clear_of_the_subtitle(self):
        rows = [r for r in self.rows if r["state"] != "INT6"]
        long_note = "exploratory, a long note about the status of the data that makes the subtitle wrap onto a third line of the figure"
        for theme in ("light", "dark"):
            fig = ef.fig_state_defined(rows, theme, long_note)
            self.assertEqual([a.get_gid() for a in fig.axes], ["panel:INT4"]); self.assertEqual(texts_inside(fig), []); self.assertEqual(overlaps(fig), [])

    def test_the_vertical_range_always_contains_zero_and_each_panel_draws_the_zero_line(self):
        rows = [dict(r, median_net_state_defined=abs(r["median_net_state_defined"]) + 0.1, median_net_reference=abs(r["median_net_reference"]) + 0.1) for r in self.rows]
        fig = ef.fig_state_defined(rows, "light", "x")
        for a in [a for a in fig.axes if a.get_gid() and a.get_gid().startswith("panel:")]:
            self.assertLessEqual(a.get_ylim()[0], 0.0)
            self.assertTrue(any(l.get_gid() is None and list(l.get_ydata()) == [0.0, 0.0] for l in a.lines), "zero line")

    def test_the_note_closes_the_subtitle_and_the_legend_text_wears_the_text_colour(self):
        import phase1_func_figures as base
        fig = ef.fig_state_defined(self.rows, "dark", "exploratory, end marker")
        sub = [x for x in fig.texts if x.get_gid() == "subtitle"][0].get_text().replace("\n", " ")
        self.assertTrue(sub.endswith("end marker")); self.assertTrue(sub.startswith("median over the languages"))
        ink = base.theme_tokens("dark")["ink2"]
        self.assertTrue(all(tx.get_color() == ink for lg in fig.legends for tx in lg.get_texts()))

    def test_a_state_with_a_missing_row_is_an_error_not_a_zero(self):
        rows = [r for r in self.rows if not (r["state"] == "INT4" and r["ratio"] == DOWN[0])]
        with self.assertRaisesRegex(ValueError, "lacks ratios"):
            ef.fig_state_defined(rows, "light", "x")

    def test_write_figures_creates_svg_and_png_in_both_themes(self):
        with tempfile.TemporaryDirectory() as d:
            paths = ef.write_figures(self.rows, d, "exploratory test")
            self.assertEqual(len(paths), 4)
            names = sorted(Path(p).name for p in paths)
            self.assertEqual(names, ["fig_gb1_state_defined_dark.png", "fig_gb1_state_defined_dark.svg", "fig_gb1_state_defined_light.png", "fig_gb1_state_defined_light.svg"])
            self.assertTrue(all(Path(p).stat().st_size > 1000 for p in paths))

    def test_the_command_line_reads_compare_csv_and_writes_the_figures(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "compare.csv"
            with open(p, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(self.rows[0])); w.writeheader(); w.writerows(self.rows)
            ef.main(["--compare", str(p), "--out", str(Path(d) / "figs"), "--note", "exploratory", "--formats", "svg", "--themes", "light"])
            self.assertEqual([x.name for x in (Path(d) / "figs").iterdir()], ["fig_gb1_state_defined_light.svg"])


if __name__ == "__main__":
    unittest.main()
