import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_func_figures as base
import phase1_gain_extc as ec
import phase1_gain_extc_figures as xf
import phase1_gain_figures as gf
from test_phase1_gain_extc import LANGS, RATIOS, cost_harm, ext, repair_harm, runs
from test_phase1_gain_figures import overlaps, texts_inside


def mid_harm(code, a, kind):
    return -0.0005 * (a - 1.0) * 10 if kind == "g" else 0.0001 * int(kind[1:])


def skewed(h):
    """The harm of the group differs between the languages (so that a wrong language order shows)."""
    return lambda c, a, k: h(c, a, k) + (0.0003 * int(c[1:]) * (a - 1.0) if k == "g" else 0.0)


def cell_rows(states=None):
    """The rows of cells.csv (numbers converted by the loader) of a planted extension C study, with the cells they came from."""
    states = states or {"FP": cost_harm, "INT6": cost_harm, "INT5": mid_harm, "INT4": repair_harm}
    e = ext(runs({s: skewed(h) for s, h in states.items()})); cells = ec.build_cells(e, n_boot=20, seed=0)
    with tempfile.TemporaryDirectory() as d:
        res = ec.evaluate(cells, e, n_boot=50, seed=0)
        ec.write_outputs(d, res, cells, e, {})
        return xf.load_cells(Path(d) / "cells.csv"), cells, e


class Data(unittest.TestCase):
    def setUp(self):
        self.rows, self.cells, self.e = cell_rows()

    def test_the_loader_converts_numbers_and_keeps_the_names(self):
        r = self.rows[0]
        self.assertIsInstance(r["state"], str); self.assertIsInstance(r["language"], str)
        self.assertIsInstance(r["ratio"], float); self.assertIsInstance(r["net"], float)
        self.assertEqual(len(self.rows), 4 * len(LANGS) * len(RATIOS))

    def test_values_are_the_net_harm_of_each_language_in_the_order_of_the_languages(self):
        v = xf.net_by_state_ratio(self.rows)
        self.assertEqual(list(v), ["FP", "INT6", "INT5", "INT4"])                           # ladder order, whatever the row order
        self.assertEqual(list(xf.net_by_state_ratio(list(reversed(self.rows)))), ["FP", "INT6", "INT5", "INT4"])
        for s in v:
            self.assertEqual(list(v[s]), RATIOS)
            for a in RATIOS:
                np.testing.assert_allclose(v[s][a], [self.cells[(s, l, a)]["net"] for l in LANGS])

    def test_the_band_is_the_97_5_percent_interval_of_the_analysis(self):
        v = xf.net_by_state_ratio(self.rows)
        b = xf.band(v["INT4"][1.1])
        lo, hi = ec.median_interval(v["INT4"][1.1], 5000, 0)
        self.assertAlmostEqual(b["median"], float(np.median(v["INT4"][1.1]))); self.assertAlmostEqual(b["lo"], lo); self.assertAlmostEqual(b["hi"], hi)

    def test_the_band_of_many_values_is_the_seeded_5000_draw_interval(self):
        vals = list(np.random.default_rng(5).normal(0.0, 1.0, 40))
        b = xf.band(vals); lo, hi = ec.median_interval(vals, 5000, 0)
        self.assertEqual((b["lo"], b["hi"]), (lo, hi))
        self.assertNotEqual((lo, hi), ec.median_interval(vals, 5000, 1)); self.assertNotEqual((lo, hi), ec.median_interval(vals, 100, 0))

    def test_the_negative_count_counts_languages_below_zero_strictly(self):
        self.assertEqual(xf.n_negative([-1.0, 0.0, 2.0, -0.1]), 2)

    def test_a_missing_cell_is_an_error(self):
        rows = [r for r in self.rows if not (r["state"] == "INT4" and r["language"] == "l1" and r["ratio"] == 1.4)]
        with self.assertRaises(ValueError):
            xf.net_by_state_ratio(rows)


class Figure(unittest.TestCase):
    def setUp(self):
        self.rows, self.cells, self.e = cell_rows()

    def test_two_panels_with_a_line_and_a_band_per_state(self):
        for theme in ("light", "dark"):
            fig = xf.fig_repair_c(self.rows, theme, "exploratory test")
            panels = [a for a in fig.axes if a.get_gid() and a.get_gid().startswith("panel:")]
            self.assertEqual([a.get_gid() for a in panels], ["panel:median", "panel:negative"])
            med = panels[0]
            self.assertEqual({l.get_gid() for l in med.lines if l.get_gid() and l.get_gid().startswith("state:")}, {f"state:{s}" for s in ("FP", "INT6", "INT5", "INT4")})
            self.assertEqual({c.get_gid() for c in med.collections if c.get_gid() and c.get_gid().startswith("band:")}, {f"band:{s}" for s in ("FP", "INT6", "INT5", "INT4")})
            self.assertEqual(texts_inside(fig), []); self.assertEqual(overlaps(fig), [])

    def test_the_median_lines_carry_the_medians_at_the_study_ratios(self):
        fig = xf.fig_repair_c(self.rows, "light", "x")
        a = [a for a in fig.axes if a.get_gid() == "panel:median"][0]
        v = xf.net_by_state_ratio(self.rows)
        for s in v:
            ln = [l for l in a.lines if l.get_gid() == f"state:{s}"][0]
            np.testing.assert_allclose(ln.get_ydata(), [float(np.median(v[s][r])) for r in RATIOS]); np.testing.assert_allclose(ln.get_xdata(), range(len(RATIOS)))

    def test_the_negative_panel_shows_counts_and_the_half_line(self):
        fig = xf.fig_repair_c(self.rows, "light", "x")
        a = [a for a in fig.axes if a.get_gid() == "panel:negative"][0]
        v = xf.net_by_state_ratio(self.rows)
        ln = [l for l in a.lines if l.get_gid() == "state:INT4"][0]
        np.testing.assert_allclose(ln.get_ydata(), [xf.n_negative(v["INT4"][r]) for r in RATIOS])
        half = [l for l in a.lines if l.get_gid() == "half"]
        self.assertEqual(len(half), 1); self.assertEqual(list(half[0].get_ydata()), [len(LANGS) / 2.0] * 2)
        lo, hi = a.get_ylim()
        self.assertLess(lo, 0.0); self.assertGreater(hi, float(len(LANGS))); self.assertLess(hi, len(LANGS) + 0.5)                  # room for the markers at 0 and at n, no more
        a.figure.canvas.draw()
        self.assertTrue(all(float(y).is_integer() for y in a.get_yticks() if 0 <= y <= len(LANGS)))                                 # counts of languages: whole numbers

    def test_the_zero_line_is_drawn_and_the_range_contains_zero(self):
        rows = [dict(r, net=abs(r["net"]) + 0.001) for r in self.rows]
        fig = xf.fig_repair_c(rows, "light", "x")
        a = [a for a in fig.axes if a.get_gid() == "panel:median"][0]
        self.assertLessEqual(a.get_ylim()[0], 0.0)
        self.assertTrue(any(l.get_gid() is None and list(l.get_ydata()) == [0.0, 0.0] for l in a.lines))

    def test_the_x_ticks_are_the_ratios_of_the_study(self):
        fig = xf.fig_repair_c(self.rows, "light", "x")
        for a in [a for a in fig.axes if a.get_gid() and a.get_gid().startswith("panel:")]:
            self.assertEqual([t.get_text() for t in a.get_xticklabels()], [f"{r:g}" for r in RATIOS])

    def test_the_states_wear_the_ordinal_colours_in_ladder_order(self):
        fig = xf.fig_repair_c(self.rows, "light", "x")
        a = fig.axes[0]
        t = base.theme_tokens("light"); want = base._precision_colors(t, 4)
        got = [[l for l in a.lines if l.get_gid() == f"state:{s}"][0].get_color() for s in ("FP", "INT6", "INT5", "INT4")]
        self.assertEqual(got, list(want))

    def test_the_legend_names_the_states_in_ladder_order(self):
        fig = xf.fig_repair_c(self.rows, "light", "x")
        self.assertEqual([t.get_text() for lg in fig.legends for t in lg.get_texts()], ["FP", "INT6", "INT5", "INT4"])

    def test_a_state_without_one_of_the_ratios_is_an_error_naming_it(self):
        rows = [r for r in self.rows if not (r["state"] == "INT4" and r["ratio"] == 1.4)]
        with self.assertRaisesRegex(ValueError, "lacks ratios"):
            xf.fig_repair_c(rows, "light", "x")

    def test_the_band_is_translucent_the_legend_has_one_column_per_state_and_its_text_wears_the_text_colour(self):
        fig = xf.fig_repair_c(self.rows, "dark", "x")
        a = fig.axes[0]
        self.assertTrue(all(c.get_alpha() is not None and c.get_alpha() < 0.5 for c in a.collections if c.get_gid() and c.get_gid().startswith("band:")))
        lg = fig.legends[0]
        self.assertEqual(lg._ncols, 4)
        ink = base.theme_tokens("dark")["ink2"]
        self.assertTrue(all(tx.get_color() == ink for tx in lg.get_texts()))

    def test_the_subtitle_closes_with_the_note_and_never_touches_a_panel_title(self):
        long_note = "exploratory, a long note about the status of the data that makes the subtitle wrap onto a third line of the figure"
        for theme in ("light", "dark"):
            fig = xf.fig_repair_c(self.rows, theme, long_note)
            sub = [x for x in fig.texts if x.get_gid() == "subtitle"][0].get_text().replace("\n", " ")
            self.assertTrue(sub.endswith("third line of the figure")); self.assertEqual(overlaps(fig), [])

    def test_a_single_state_still_fits(self):
        rows = [r for r in self.rows if r["state"] == "INT4"]
        for theme in ("light", "dark"):
            fig = xf.fig_repair_c(rows, theme, "exploratory test")
            self.assertEqual(texts_inside(fig), []); self.assertEqual(overlaps(fig), [])

    def test_write_figures_creates_svg_and_png_in_both_themes(self):
        with tempfile.TemporaryDirectory() as d:
            paths = xf.write_figures(self.rows, d, "exploratory test")
            self.assertEqual(sorted(Path(p).name for p in paths), ["fig_gc1_repair_dark.png", "fig_gc1_repair_dark.svg", "fig_gc1_repair_light.png", "fig_gc1_repair_light.svg"])
            self.assertTrue(all(Path(p).stat().st_size > 1000 for p in paths))

    def test_the_command_line_reads_cells_csv_and_writes_the_figures(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cells.csv"
            with open(p, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(self.rows[0])); w.writeheader(); w.writerows(self.rows)
            xf.main(["--cells", str(p), "--out", str(Path(d) / "figs"), "--note", "exploratory", "--formats", "svg", "--themes", "light"])
            self.assertEqual([x.name for x in (Path(d) / "figs").iterdir()], ["fig_gc1_repair_light.svg"])


if __name__ == "__main__":
    unittest.main()
