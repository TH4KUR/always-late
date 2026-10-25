import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_func_figures as base
import phase1_gain_extd_figures as df
from test_phase1_gain_figures import overlaps, texts_inside

GRID = [1.0, 1.1, 1.2, 1.4, 1.7]


def planted_rows(n=27, seed=0):
    rng = np.random.default_rng(seed); rows = []
    for i in range(n):
        tau = 1.07 + 0.28 * i / (n - 1)
        a = float(rng.choice(GRID)); rows.append({"language": f"l{i}", "tau": tau, "tau_controls": 1.16 + 0.002 * i, "alpha": a, "alpha_smooth": float(np.clip(a + rng.normal(0, 0.05), 1.0, 1.7)),
                                                  "alpha_even": float(rng.choice(GRID)), "alpha_odd": a})
    return rows


ANALYSIS = {"verdict": "not supported", "rd1": {"rho": -0.17017, "p": 0.80362, "constant": False}, "rd2": {"rho": -0.15122, "p": 0.77584}, "rd3": {"rho_placebo": 0.18584, "diff": -0.356, "lo": -0.95, "hi": 0.28, "n_nan": 0},
            "rd4": {"rho_half": 0.8462, "sb": 0.91670, "corrected": -0.1777}, "n_languages": 27, "n_sentences": 562, "ratios": [1.1, 1.2, 1.4, 1.7], "n_perm": 10000}


def panel(fig, gid):
    return [a for a in fig.axes if a.get_gid() == gid][0]


class Data(unittest.TestCase):
    def test_the_loader_converts_numbers_and_keeps_the_names(self):
        rows = planted_rows()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "alpha.csv"
            with open(p, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]) + ["harm_1.1"]); w.writeheader(); w.writerows([dict(r, **{"harm_1.1": 0.001}) for r in rows])
            got = df.load_alpha(p)
        self.assertEqual(len(got), 27); self.assertIsInstance(got[0]["language"], str)
        for k in ("tau", "alpha", "alpha_smooth", "alpha_even", "alpha_odd", "harm_1.1"):
            self.assertIsInstance(got[0][k], float, k)
        self.assertEqual(got[3]["alpha"], rows[3]["alpha"])

    def test_points_with_equal_values_are_spread_deterministically_and_single_points_stay(self):
        y = [1.1, 1.2, 1.1, 1.1, 1.4]
        off = df.jitter(y, 0.02)
        self.assertEqual(off[1], 0.0); self.assertEqual(off[4], 0.0)                                       # alone at their value
        three = sorted(off[i] for i in (0, 2, 3))
        self.assertEqual(len(set(three)), 3); self.assertAlmostEqual(sum(three), 0.0); self.assertAlmostEqual(three[0], -0.02); self.assertAlmostEqual(three[-1], 0.02)
        self.assertEqual(df.jitter(y, 0.02), off)                                                           # no randomness
        self.assertEqual(list(df.jitter([1.0, 1.0], 0.02)), [-0.02, 0.02])

    def test_the_jitter_is_in_log_units_so_that_it_is_symmetric_on_the_log_axis(self):
        y = np.array([1.1, 1.1]); off = df.jitter(list(y), 0.02)
        shifted = y * np.exp(off)
        self.assertAlmostEqual(float(np.log(shifted[0] / 1.1)), -0.02); self.assertAlmostEqual(float(np.log(shifted[1] / 1.1)), 0.02)

    def test_the_analysis_numbers_are_taken_from_the_file_and_a_missing_one_is_an_error(self):
        s = df.summary_text(ANALYSIS)
        self.assertIn("-0.17", s); self.assertIn("0.80", s); self.assertIn("0.92", s); self.assertIn("not supported", s)
        with self.assertRaises(KeyError):
            df.summary_text({"verdict": "x", "rd1": {"rho": 0.1}, "rd4": {"sb": 0.9}})


class Figure(unittest.TestCase):
    def setUp(self):
        self.rows = planted_rows()

    def test_two_panels_with_one_point_per_language(self):
        for theme in ("light", "dark"):
            fig = df.fig_alpha(self.rows, ANALYSIS, theme, "exploratory test")
            self.assertEqual([a.get_gid() for a in fig.axes if a.get_gid() and a.get_gid().startswith("panel:")], ["panel:optimum", "panel:halves"])
            a = panel(fig, "panel:optimum")
            for gid in ("grid", "smooth"):
                self.assertEqual(len([c for c in a.collections if c.get_gid() == gid][0].get_offsets()), 27, gid)
            self.assertEqual(len([c for c in panel(fig, "panel:halves").collections if c.get_gid() == "halves"][0].get_offsets()), 27)
            self.assertEqual(texts_inside(fig), []); self.assertEqual(overlaps(fig), [])

    def test_the_points_carry_the_values_of_the_file_with_only_a_small_vertical_offset_for_ties(self):
        fig = df.fig_alpha(self.rows, ANALYSIS, "light", "x")
        a = panel(fig, "panel:optimum")
        g = [c for c in a.collections if c.get_gid() == "grid"][0].get_offsets()
        np.testing.assert_allclose(g[:, 0], [r["tau"] for r in self.rows])
        ratio = g[:, 1] / np.array([r["alpha"] for r in self.rows])
        self.assertTrue(np.all(np.abs(np.log(ratio)) <= df.JITTER + 1e-12))
        sm = [c for c in a.collections if c.get_gid() == "smooth"][0].get_offsets()
        np.testing.assert_allclose(sm[:, 1], [r["alpha_smooth"] for r in self.rows]); np.testing.assert_allclose(sm[:, 0], [r["tau"] for r in self.rows])
        h = [c for c in panel(fig, "panel:halves").collections if c.get_gid() == "halves"][0].get_offsets()
        self.assertTrue(np.all(np.abs(np.log(h[:, 0] / np.array([r["alpha_even"] for r in self.rows]))) <= df.JITTER + 1e-12))
        self.assertTrue(np.all(np.abs(np.log(h[:, 1] / np.array([r["alpha_odd"] for r in self.rows]))) <= df.JITTER + 1e-12))

    def test_the_identity_lines_are_drawn_on_both_panels(self):
        fig = df.fig_alpha(self.rows, ANALYSIS, "light", "x")
        for gid in ("panel:optimum", "panel:halves"):
            lines = [l for l in panel(fig, gid).lines if l.get_gid() == "identity"]
            self.assertEqual(len(lines), 1)
            self.assertTrue(np.allclose(lines[0].get_xdata(), lines[0].get_ydata()))

    def test_the_vertical_axis_is_logarithmic_with_the_grid_ratios_as_ticks(self):
        fig = df.fig_alpha(self.rows, ANALYSIS, "light", "x")
        for gid in ("panel:optimum", "panel:halves"):
            a = panel(fig, gid); fig.canvas.draw()
            self.assertEqual(a.get_yscale(), "log")
            self.assertEqual([t.get_text() for t in a.get_yticklabels() if t.get_text()], ["1", "1.1", "1.2", "1.4", "1.7"])
            lo, hi = a.get_ylim(); self.assertLess(lo, 1.0); self.assertGreater(hi, 1.7)

    def test_the_two_estimators_have_different_markers_and_a_legend_names_them(self):
        fig = df.fig_alpha(self.rows, ANALYSIS, "light", "x")
        a = panel(fig, "panel:optimum")
        g = [c for c in a.collections if c.get_gid() == "grid"][0]; s = [c for c in a.collections if c.get_gid() == "smooth"][0]
        self.assertNotEqual(str(g.get_paths()[0].vertices.tolist()), str(s.get_paths()[0].vertices.tolist()))
        self.assertEqual([t.get_text() for lg in fig.legends for t in lg.get_texts()], ["grid optimum (ratio with the lowest harm)", "smooth optimum (quadratic fit)"])

    def test_the_subtitle_has_the_analysis_numbers_and_closes_with_the_note(self):
        fig = df.fig_alpha(self.rows, ANALYSIS, "light", "exploratory, end marker")
        sub = [x for x in fig.texts if x.get_gid() == "subtitle"][0].get_text().replace("\n", " ")
        self.assertIn(df.summary_text(ANALYSIS), " ".join(sub.split())); self.assertTrue(sub.endswith("end marker"))

    def test_the_number_of_sentences_comes_from_the_analysis(self):
        sub = lambda an: " ".join([x for x in df.fig_alpha(self.rows, an, "light", "x").texts if x.get_gid() == "subtitle"][0].get_text().split())
        self.assertIn("on 562 held out sentences per language", sub(ANALYSIS)); self.assertIn("on 777 held out sentences per language", sub(dict(ANALYSIS, n_sentences=777)))
        bad = {k: v for k, v in ANALYSIS.items() if k != "n_sentences"}
        with self.assertRaises(KeyError):
            df.fig_alpha(self.rows, bad, "light", "x")

    def test_the_spread_is_the_stated_small_amount_and_really_separates_tied_points(self):
        self.assertEqual(df.JITTER, 0.02)
        fig = df.fig_alpha(self.rows, ANALYSIS, "light", "x")
        a = panel(fig, "panel:optimum"); y = [c for c in a.collections if c.get_gid() == "grid"][0].get_offsets()[:, 1]
        alpha = np.array([r["alpha"] for r in self.rows])
        for v in set(alpha):
            self.assertEqual(len(set(np.round(y[alpha == v], 9))), int((alpha == v).sum()), v)                  # every point at a tied value has its own height

    def test_every_point_lies_inside_its_axes_and_there_are_no_minor_tick_marks_on_the_log_axes(self):
        fig = df.fig_alpha(self.rows, ANALYSIS, "light", "x"); fig.canvas.draw()
        for gid, gids in (("panel:optimum", ("grid", "smooth")), ("panel:halves", ("halves",))):
            a = panel(fig, gid); (x0, x1), (y0, y1) = a.get_xlim(), a.get_ylim()
            for c in a.collections:
                if c.get_gid() in gids:
                    off = c.get_offsets()
                    self.assertTrue(np.all((off[:, 0] > x0) & (off[:, 0] < x1) & (off[:, 1] > y0) & (off[:, 1] < y1)), c.get_gid())
            self.assertEqual(len(a.yaxis.get_minorticklocs()), 0); self.assertEqual(len(a.xaxis.get_minorticklocs()), 0)

    def test_the_legend_has_two_columns_and_the_text_colour_and_the_subtitle_ends_above_the_panels(self):
        fig = df.fig_alpha(self.rows, ANALYSIS, "dark", "a note that makes the subtitle a little longer than the shortest possible one")
        lg = fig.legends[0]; self.assertEqual(lg._ncols, 2)
        ink = base.theme_tokens("dark")["ink2"]; self.assertTrue(all(tx.get_color() == ink for tx in lg.get_texts()))
        fig.canvas.draw(); r = fig.canvas.get_renderer()
        sub = [x for x in fig.texts if x.get_gid() == "subtitle"][0].get_window_extent(r)
        for a in [a for a in fig.axes if a.get_gid() and a.get_gid().startswith("panel:")]:
            self.assertGreater(sub.y0, a.title.get_window_extent(r).y1 if a.title.get_text() else a.get_window_extent(r).y1)
            self.assertGreater(sub.y0, a._left_title.get_window_extent(r).y1)

    def test_the_subtitle_never_touches_a_panel_title(self):
        long_note = "exploratory, a long note about the status of the data that makes the subtitle wrap onto a third line of the figure"
        for theme in ("light", "dark"):
            self.assertEqual(overlaps(df.fig_alpha(self.rows, ANALYSIS, theme, long_note)), [])

    def test_a_wrong_number_of_languages_in_the_analysis_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "27"):
            df.fig_alpha(self.rows[:20], ANALYSIS, "light", "x")

    def test_write_figures_creates_svg_and_png_in_both_themes(self):
        with tempfile.TemporaryDirectory() as d:
            paths = df.write_figures(self.rows, ANALYSIS, d, "exploratory test")
            self.assertEqual(sorted(Path(p).name for p in paths), ["fig_gd1_alpha_dark.png", "fig_gd1_alpha_dark.svg", "fig_gd1_alpha_light.png", "fig_gd1_alpha_light.svg"])
            self.assertTrue(all(Path(p).stat().st_size > 1000 for p in paths))

    def test_the_command_line_reads_alpha_csv_and_analysis_json(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            with open(d / "alpha.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(self.rows[0])); w.writeheader(); w.writerows(self.rows)
            (d / "analysis.json").write_text(json.dumps(ANALYSIS))
            df.main(["--alpha", str(d / "alpha.csv"), "--analysis", str(d / "analysis.json"), "--out", str(d / "figs"), "--note", "exploratory", "--formats", "svg", "--themes", "light"])
            self.assertEqual([x.name for x in (d / "figs").iterdir()], ["fig_gd1_alpha_light.svg"])


if __name__ == "__main__":
    unittest.main()
