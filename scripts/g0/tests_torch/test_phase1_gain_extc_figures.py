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
from test_phase1_gain_extc import LANGS, RATIOS, cost_harm, ext, repair_harm, runs
from test_phase1_gain_figures import overlaps, texts_inside


def mid_harm(code, a, kind):
    return -0.0005 * (a - 1.0) * 10 if kind == "g" else 0.0001 * int(kind[1:])


def skewed(h):
    """The harm of the group differs between the languages (so that a wrong language order shows)."""
    return lambda c, a, k: h(c, a, k) + (0.0003 * int(c[1:]) * (a - 1.0) if k == "g" else 0.0)


def steep(c, a, k):
    """Cost that grows fast with the ratio: about 0.0013 at 1.2 and 0.014 at 1.7."""
    return 0.0065 * (a - 1.0) ** 2 * 10 * 0.9 + 0.0002 * int(c[1:]) if k == "g" else 0.0


def band_rows(states=None):
    """The rows of bands.csv of a planted extension C study, as the loader returns them (numbers converted), with the cells they came from."""
    states = states or {"FP": cost_harm, "INT6": cost_harm, "INT5": mid_harm, "INT4": repair_harm}
    e = ext(runs({s: skewed(h) for s, h in states.items()})); cells = ec.build_cells(e, n_boot=20, seed=0)
    with tempfile.TemporaryDirectory() as d:
        res = ec.evaluate(cells, e, n_boot=300, seed=0)
        ec.write_outputs(d, res, cells, e, {})
        return xf.load_bands(Path(d) / "bands.csv"), res, e


class Data(unittest.TestCase):
    def setUp(self):
        self.rows, self.res, self.e = band_rows()

    def test_the_loader_converts_numbers_and_keeps_the_names(self):
        r = self.rows[0]
        self.assertIsInstance(r["state"], str)
        for k in ("ratio", "median", "lo", "hi", "n_negative", "n_languages"):
            self.assertIsInstance(r[k], float, k)
        self.assertEqual(len(self.rows), 4 * len(RATIOS))
        self.assertAlmostEqual(self.rows[3]["hi"], self.res["bands"][3]["hi"])

    def test_the_series_are_in_ladder_order_and_ascending_ratio_whatever_the_row_order(self):
        for rows in (self.rows, list(reversed(self.rows))):
            s = xf.series(rows)
            self.assertEqual(list(s), ["FP", "INT6", "INT5", "INT4"])
            for st in s:
                self.assertEqual(list(s[st]), RATIOS)
                self.assertEqual(s[st][1.1]["median"], [r for r in self.rows if r["state"] == st and r["ratio"] == 1.1][0]["median"])

    def test_a_state_without_one_of_the_ratios_is_an_error_naming_it(self):
        rows = [r for r in self.rows if not (r["state"] == "INT4" and r["ratio"] == 1.4)]
        with self.assertRaisesRegex(ValueError, "lacks ratios"):
            xf.series(rows)

    def test_a_state_with_the_wrong_number_of_languages_is_an_error(self):
        rows = [dict(r, n_languages=26.0) if (r["state"] == "INT4" and r["ratio"] == 1.1) else r for r in self.rows]
        with self.assertRaisesRegex(ValueError, "languages"):
            xf.series(rows)

    def test_no_rows_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "no rows"):
            xf.series([])


class Figure(unittest.TestCase):
    def setUp(self):
        self.rows, self.res, self.e = band_rows()

    def panel(self, fig, gid):
        return [a for a in fig.axes if a.get_gid() == gid][0]

    def test_two_panels_with_a_line_and_a_band_per_state(self):
        for theme in ("light", "dark"):
            fig = xf.fig_repair_c(self.rows, theme, "exploratory test")
            panels = [a for a in fig.axes if a.get_gid() and a.get_gid().startswith("panel:")]
            self.assertEqual([a.get_gid() for a in panels], ["panel:median", "panel:negative"])
            med = panels[0]; states = ("FP", "INT6", "INT5", "INT4")
            self.assertEqual({l.get_gid() for l in med.lines if l.get_gid() and l.get_gid().startswith("state:")}, {f"state:{s}" for s in states})
            self.assertEqual({c.get_gid() for c in med.collections if c.get_gid() and c.get_gid().startswith("band:")}, {f"band:{s}" for s in states})
            self.assertEqual(texts_inside(fig), []); self.assertEqual(overlaps(fig), [])

    def test_the_lines_and_bands_carry_the_numbers_of_the_file_and_do_not_recompute_them(self):
        rows = [dict(r, median=r["median"] + 0.001, lo=r["lo"] - 0.002, hi=r["hi"] + 0.003) for r in self.rows]      # numbers no recomputation could give
        fig = xf.fig_repair_c(rows, "light", "x")
        a = self.panel(fig, "panel:median"); s = xf.series(rows)
        for st in s:
            ln = [l for l in a.lines if l.get_gid() == f"state:{st}"][0]
            np.testing.assert_allclose(ln.get_ydata(), [s[st][r]["median"] for r in RATIOS]); np.testing.assert_allclose(ln.get_xdata(), range(len(RATIOS)))
            poly = [c for c in a.collections if c.get_gid() == f"band:{st}"][0]
            verts = poly.get_paths()[0].vertices
            for k, r in enumerate(RATIOS):
                ys = verts[np.isclose(verts[:, 0], k), 1]
                self.assertAlmostEqual(float(ys.min()), s[st][r]["lo"]); self.assertAlmostEqual(float(ys.max()), s[st][r]["hi"])

    def test_the_negative_panel_shows_the_counts_of_the_file_and_the_half_line(self):
        fig = xf.fig_repair_c(self.rows, "light", "x")
        a = self.panel(fig, "panel:negative"); s = xf.series(self.rows); n = len(LANGS)
        ln = [l for l in a.lines if l.get_gid() == "state:INT4"][0]
        np.testing.assert_allclose(ln.get_ydata(), [s["INT4"][r]["n_negative"] for r in RATIOS])
        half = [l for l in a.lines if l.get_gid() == "half"]
        self.assertEqual(len(half), 1); self.assertEqual(list(half[0].get_ydata()), [n / 2.0] * 2)
        lo, hi = a.get_ylim()
        self.assertLess(lo, 0.0); self.assertGreater(hi, float(n)); self.assertLess(hi, n + 0.5)                      # room for the markers at 0 and at n, no more
        a.figure.canvas.draw()
        self.assertTrue(all(float(y).is_integer() for y in a.get_yticks() if 0 <= y <= n))                            # counts of languages: whole numbers

    def test_the_zero_line_is_drawn_and_the_range_contains_zero(self):
        rows = [dict(r, median=abs(r["median"]) + 0.001, lo=abs(r["lo"]) + 0.0005, hi=abs(r["hi"]) + 0.002) for r in self.rows]
        a = self.panel(xf.fig_repair_c(rows, "light", "x"), "panel:median")
        self.assertLessEqual(a.get_ylim()[0], 0.0)
        self.assertTrue(any(l.get_gid() is None and list(l.get_ydata()) == [0.0, 0.0] for l in a.lines))

    def test_the_x_ticks_are_the_ratios_of_the_study(self):
        fig = xf.fig_repair_c(self.rows, "light", "x")
        for a in [a for a in fig.axes if a.get_gid() and a.get_gid().startswith("panel:")]:
            self.assertEqual([t.get_text() for t in a.get_xticklabels()], [f"{r:g}" for r in RATIOS])

    def test_the_states_wear_the_ordinal_colours_in_ladder_order(self):
        fig = xf.fig_repair_c(self.rows, "light", "x")
        want = base._precision_colors(base.theme_tokens("light"), 4)
        got = [[l for l in fig.axes[0].lines if l.get_gid() == f"state:{s}"][0].get_color() for s in ("FP", "INT6", "INT5", "INT4")]
        self.assertEqual(got, list(want))

    def test_the_legend_names_the_states_in_ladder_order(self):
        fig = xf.fig_repair_c(self.rows, "light", "x")
        self.assertEqual([t.get_text() for lg in fig.legends for t in lg.get_texts()], ["FP", "INT6", "INT5", "INT4"])

    def test_the_median_panel_is_symmetric_log_so_that_the_small_primary_ratios_stay_readable(self):
        rows, res, e = band_rows({"FP": steep, "INT4": repair_harm})
        a = self.panel(xf.fig_repair_c(rows, "light", "x"), "panel:median")
        self.assertEqual(a.get_yscale(), "symlog")
        s = xf.series(rows); lo_, hi_ = s["FP"][1.2]["median"], s["FP"][1.7]["median"]
        self.assertGreater(hi_, 5 * lo_)                                                                              # the planted data really span a wide range
        tr = a.transData; y0, ylo, yhi = (tr.transform((0, y))[1] for y in (0.0, lo_, hi_))
        self.assertGreater((ylo - y0) / (yhi - y0), 0.25)                                                             # on a linear axis this fraction would be below 0.2

    def test_the_median_axis_has_plain_decimal_ticks_inside_the_range_and_a_zero(self):
        fig = xf.fig_repair_c(self.rows, "light", "x")
        a = self.panel(fig, "panel:median"); fig.canvas.draw()
        lo, hi = a.get_ylim(); ticks = list(a.get_yticks()); labels = [t.get_text() for t in a.get_yticklabels()]
        self.assertTrue(all(lo <= y <= hi for y in ticks)); self.assertIn(0.0, ticks); self.assertEqual(len(ticks), len(labels))
        self.assertTrue(all("e" not in l and "^" not in l and "{" not in l for l in labels), labels)
        self.assertEqual([float(l.replace("−", "-")) for l in labels], ticks)
        self.assertGreaterEqual(len(ticks), 3)

    def test_the_ticks_of_the_median_axis_are_never_crowded(self):
        for rows in (self.rows, band_rows({"FP": steep, "INT4": repair_harm})[0]):
            fig = xf.fig_repair_c(rows, "light", "x")
            a = self.panel(fig, "panel:median"); fig.canvas.draw()
            pix = sorted(a.transData.transform((0, y))[1] for y in a.get_yticks())
            gaps = [q - p for p, q in zip(pix, pix[1:])]
            self.assertGreaterEqual(min(gaps), 0.2 * fig.dpi, gaps)                                                   # at least a fifth of an inch between two tick labels

    def test_the_vertical_range_is_the_range_of_the_bands_with_six_percent_room_and_the_threshold_is_the_stated_one(self):
        fig = xf.fig_repair_c(self.rows, "light", "x")
        a = self.panel(fig, "panel:median")
        lo = min(0.0, min(r["lo"] for r in self.rows)); hi = max(0.0, max(r["hi"] for r in self.rows)); pad = 0.06 * (hi - lo)
        np.testing.assert_allclose(a.get_ylim(), (lo - pad, hi + pad))
        self.assertEqual(xf.LINTHRESH, 0.002); self.assertEqual(a.yaxis.get_transform().linthresh, 0.002)

    def test_the_subtitle_states_the_scale_and_the_number_of_languages_and_closes_with_the_note(self):
        fig = xf.fig_repair_c(self.rows, "light", "exploratory, end marker")
        sub = [x for x in fig.texts if x.get_gid() == "subtitle"][0].get_text().replace("\n", " ")
        self.assertIn("linear up to 0.002 and logarithmic beyond", sub); self.assertIn(f"{len(LANGS)} languages", sub)
        self.assertTrue(sub.endswith("end marker")); self.assertTrue(sub.startswith("net harm = loss change"))

    def test_the_subtitle_never_touches_a_panel_title_and_the_band_and_legend_are_styled(self):
        long_note = "exploratory, a long note about the status of the data that makes the subtitle wrap onto a third line of the figure"
        for theme in ("light", "dark"):
            fig = xf.fig_repair_c(self.rows, theme, long_note)
            self.assertEqual(overlaps(fig), [])
        fig = xf.fig_repair_c(self.rows, "dark", "x")
        self.assertTrue(all(c.get_alpha() is not None and c.get_alpha() < 0.5 for c in fig.axes[0].collections if c.get_gid() and c.get_gid().startswith("band:")))
        lg = fig.legends[0]
        self.assertEqual(lg._ncols, 4)
        ink = base.theme_tokens("dark")["ink2"]
        self.assertTrue(all(tx.get_color() == ink for tx in lg.get_texts()))

    def test_a_state_without_one_of_the_ratios_is_an_error_in_the_figure_too(self):
        rows = [r for r in self.rows if not (r["state"] == "INT4" and r["ratio"] == 1.4)]
        with self.assertRaisesRegex(ValueError, "lacks ratios"):
            xf.fig_repair_c(rows, "light", "x")

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

    def test_the_command_line_reads_bands_csv_and_writes_the_figures(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bands.csv"
            with open(p, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(self.res["bands"][0])); w.writeheader(); w.writerows(self.res["bands"])
            xf.main(["--bands", str(p), "--out", str(Path(d) / "figs"), "--note", "exploratory", "--formats", "svg", "--themes", "light"])
            self.assertEqual([x.name for x in (Path(d) / "figs").iterdir()], ["fig_gc1_repair_light.svg"])


if __name__ == "__main__":
    unittest.main()
