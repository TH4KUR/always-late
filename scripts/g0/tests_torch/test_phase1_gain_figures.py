import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_gain_analysis as ga
import phase1_gain_figures as gf
from test_phase1_gain_analysis import DOWN, FINE, LANGS, UP, LANGS27, harm27, planted_run, simple_harm, table_of, two_states, repair, ladder_runs

STATES = ("FP", "INT6", "INT4")


def curve_rows(langs=LANGS, states=STATES, harm=simple_harm):
    """The rows of curves.csv of a planted study (state harms grow with the ladder position)."""
    rs = {s: planted_run(s, lambda c, a, k, f=1.0 + 0.3 * i: f * harm(c, a, k), langs=langs) for i, s in enumerate(states)}
    g = ga.GainRuns(rs, DOWN, UP, FINE, n_controls=2)
    t = ga.build_table(g, n_boot=30, seed=0)
    return ga.curve_rows(t, g), g


def string_rows(rows):
    """What csv.DictReader returns for the rows (every value a string): the loader must convert them."""
    return [{k: str(v) for k, v in r.items()} for r in rows]


class Data(unittest.TestCase):
    def setUp(self):
        self.rows, self.g = curve_rows()

    def test_the_loader_converts_numbers_and_keeps_the_names(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "curves.csv"
            with open(p, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(self.rows[0])); w.writeheader(); w.writerows(self.rows)
            got = gf.load_curves(p)
        self.assertEqual(len(got), len(self.rows))
        self.assertIsInstance(got[0]["ratio"], float); self.assertIsInstance(got[0]["state"], str)
        self.assertAlmostEqual(got[3]["net"], self.rows[3]["net"])

    def test_x_positions_are_log2_with_zero_ablation_to_the_left_of_everything(self):
        ratios = [0.0, 0.125, 0.5, 1.0, 2.0, 8.0]
        x = gf.x_positions(ratios)
        self.assertAlmostEqual(x[0.5], -1.0); self.assertAlmostEqual(x[1.0], 0.0); self.assertAlmostEqual(x[8.0], 3.0)
        self.assertLess(x[0.0], x[0.125] - 0.5)
        self.assertEqual(len(set(x.values())), len(ratios))

    def test_macro_curves_are_medians_over_languages_with_zero_ablation_first(self):
        m = gf.macro_by_state(self.rows)
        self.assertEqual(list(m), list(STATES))
        fp = m["FP"]
        self.assertEqual(fp["ratios"][0], 0.0); self.assertEqual(fp["ratios"], sorted(fp["ratios"]))
        i = fp["ratios"].index(0.5)
        self.assertAlmostEqual(fp["group"][i], float(np.median([simple_harm(c, 0.5, "g") for c in LANGS])))
        self.assertAlmostEqual(fp["net"][i], float(np.median([simple_harm(c, 0.5, "g") - simple_harm(c, 0.5, "c0") for c in LANGS])))
        self.assertIn(1.0, fp["ratios"]); self.assertAlmostEqual(fp["group"][fp["ratios"].index(1.0)], 0.0)

    def test_the_state_order_follows_the_ladder_whatever_the_row_order(self):
        rows = list(reversed(self.rows))
        self.assertEqual(list(gf.macro_by_state(rows)), list(STATES))

    def test_normalised_net_is_the_fraction_of_the_fp_zero_ablation_net_harm(self):
        n = gf.normalised_net(self.rows, "FP")
        self.assertAlmostEqual(n["l1"][0.0], 1.0)
        z = simple_harm("l1", 0.0, "g") - simple_harm("l1", 0.0, "c0")
        self.assertAlmostEqual(n["l1"][0.5], (simple_harm("l1", 0.5, "g") - simple_harm("l1", 0.5, "c0")) / z)
        n4 = gf.normalised_net(self.rows, "INT4")                          # still divided by the FP zero-ablation net harm
        self.assertAlmostEqual(n4["l1"][0.0], 1.0 + 0.3 * 2)

    def test_the_heat_matrix_is_sorted_by_the_fp_value_and_labelled(self):
        langs, states, mat = gf.heat_matrix(self.rows, 0.125)
        self.assertEqual(states, list(STATES)); self.assertEqual(mat.shape, (4, 3))
        fp = mat[:, 0]
        self.assertTrue(all(fp[i] >= fp[i + 1] for i in range(len(fp) - 1)))
        self.assertEqual(len(set(langs)), 4)

    def test_a_missing_cell_is_nan_not_zero(self):
        rows = [r for r in self.rows if not (r["state"] == "INT6" and r["language"] == "l2" and r["ratio"] == 0.125)]
        langs, states, mat = gf.heat_matrix(rows, 0.125)
        self.assertTrue(np.isnan(mat[langs.index("l2"), states.index("INT6")]))
        self.assertFalse(np.isnan(mat[langs.index("l1"), states.index("INT6")]))

    def test_an_unknown_ratio_is_an_error(self):
        with self.assertRaises(ValueError):
            gf.heat_matrix(self.rows, 0.3)


def texts_inside(fig):
    """Every visible text of the figure lies inside the figure canvas."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer(); box = fig.bbox
    out = []
    for t in fig.findobj(lambda o: hasattr(o, "get_text") and hasattr(o, "get_window_extent")):
        if not t.get_visible() or not t.get_text().strip():
            continue
        bb = t.get_window_extent(r)
        if bb.x0 < box.x0 - 1 or bb.x1 > box.x1 + 1 or bb.y0 < box.y0 - 1 or bb.y1 > box.y1 + 1:
            out.append(t.get_text())
    return out


class Figures(unittest.TestCase):
    def setUp(self):
        self.rows, self.g = curve_rows(LANGS27, STATES, harm27)

    def test_language_panels_one_per_language_with_a_curve_each(self):
        for theme in ("light", "dark"):
            fig = gf.fig_language_curves(self.rows, theme, "exploratory test")
            axes = [a for a in fig.axes if a.get_gid() and a.get_gid().startswith("panel:")]
            self.assertEqual(len(axes), 27)
            for a in axes:
                self.assertEqual(len([l for l in a.lines if l.get_gid() == "net"]), 1)
            self.assertEqual(texts_inside(fig), [])

    def test_the_language_panels_share_one_vertical_scale(self):
        fig = gf.fig_language_curves(self.rows, "light", "x")
        lims = {a.get_ylim() for a in fig.axes if a.get_gid() and a.get_gid().startswith("panel:")}
        self.assertEqual(len(lims), 1)

    def test_macro_figure_has_a_line_per_state_in_each_panel(self):
        for theme in ("light", "dark"):
            fig = gf.fig_macro(self.rows, theme, "exploratory test")
            panels = [a for a in fig.axes if a.get_gid() and a.get_gid().startswith("panel:")]
            self.assertEqual(len(panels), 2)
            for a in panels:
                self.assertEqual({l.get_gid() for l in a.lines if l.get_gid() and l.get_gid().startswith("state:")}, {f"state:{s}" for s in STATES})
            self.assertEqual(texts_inside(fig), [])

    def test_macro_lines_have_distinct_colours_in_ladder_order(self):
        fig = gf.fig_macro(self.rows, "light", "x")
        a = [a for a in fig.axes if a.get_gid() == "panel:group"][0]
        cols = [l.get_color() for l in a.lines if l.get_gid() and l.get_gid().startswith("state:")]
        self.assertEqual(len(set(cols)), 3)

    def test_heatmap_has_two_panels_with_every_cell_annotated(self):
        for theme in ("light", "dark"):
            fig = gf.fig_heatmap(self.rows, theme, "exploratory test")
            panels = [a for a in fig.axes if a.get_gid() and a.get_gid().startswith("panel:")]
            self.assertEqual(len(panels), 2)
            for a in panels:
                self.assertEqual(len([t for t in a.texts if t.get_gid() == "cell"]), 27 * 3)
            self.assertEqual(texts_inside(fig), [])

    def test_a_missing_cell_is_written_as_a_dash_in_the_heatmap(self):
        rows = [r for r in self.rows if not (r["state"] == "INT6" and r["language"] == "l5" and r["ratio"] == 0.125)]
        fig = gf.fig_heatmap(rows, "light", "x")
        a = [a for a in fig.axes if a.get_gid() and a.get_gid().startswith("panel:")][0]
        self.assertIn("-", [t.get_text() for t in a.texts if t.get_gid() == "cell"])

    def test_write_figures_creates_svg_and_png_in_both_themes(self):
        with tempfile.TemporaryDirectory() as d:
            paths = gf.write_figures(self.rows, d, "exploratory test", themes=("light", "dark"), formats=("svg", "png"))
            names = sorted(Path(p).name for p in paths)
            for stem in ("fig_g1_language_curves", "fig_g2_macro", "fig_g3_heatmap"):
                for theme in ("light", "dark"):
                    for ext in ("svg", "png"):
                        self.assertIn(f"{stem}_{theme}.{ext}", names)


class Repair(unittest.TestCase):
    def setUp(self):
        rs = two_states(simple_harm, repair(-0.03, -0.01), langs=LANGS)
        self.g = ga.GainRuns(rs, DOWN, UP, FINE, n_controls=2)
        self.t = ga.build_table(self.g, n_boot=30, seed=0)
        self.rows = ga.curve_rows(self.t, self.g)
        self.star = ga.alpha_star(self.t, self.g)
        self.baseline = {"l0": 0.8, "l1": 0.85, "l2": 0.9, "l3": 0.95}

    def test_repair_figure_shows_the_fine_ratios_by_state_and_the_alpha_star_scatter(self):
        for theme in ("light", "dark"):
            fig = gf.fig_repair(self.rows, self.star, self.baseline, theme, "exploratory test")
            panels = {a.get_gid(): a for a in fig.axes if a.get_gid() and a.get_gid().startswith("panel:")}
            self.assertEqual(set(panels), {"panel:fine", "panel:alpha"})
            self.assertEqual(len(panels["panel:alpha"].collections[0].get_offsets()), 4)
            self.assertEqual(texts_inside(fig), [])

    def test_the_alpha_star_scatter_uses_the_inverse_activation_ratio_on_x(self):
        fig = gf.fig_repair(self.rows, self.star, self.baseline, "light", "x")
        a = [a for a in fig.axes if a.get_gid() == "panel:alpha"][0]
        xs = sorted(a.collections[0].get_offsets()[:, 0])
        self.assertAlmostEqual(xs[0], 1 / 0.95); self.assertAlmostEqual(xs[-1], 1 / 0.8)

    def test_without_the_baseline_the_scatter_panel_says_so(self):
        fig = gf.fig_repair(self.rows, self.star, None, "light", "x")
        a = [a for a in fig.axes if a.get_gid() == "panel:alpha"][0]
        self.assertTrue(any("baseline" in t.get_text() for t in a.texts))


def overlaps(fig):
    """Pairs of visible texts (subtitle against every axes title and the panel texts) whose boxes intersect."""
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    subs = [x for x in fig.texts if x.get_gid() == "subtitle"]
    others = []
    for a in fig.axes:
        for tt in (a._left_title, a.title, a._right_title):
            if tt.get_visible() and tt.get_text().strip():
                others.append(tt)
    bad = []
    for s in subs:
        sb = s.get_window_extent(r)
        for o in others:
            if sb.overlaps(o.get_window_extent(r)):
                bad.append((s.get_text()[:20], o.get_text()))
    return bad


class Layout(unittest.TestCase):
    def test_the_subtitle_never_touches_a_panel_title(self):
        rows, g = curve_rows(LANGS27, ("FP", "INT8", "INT7", "INT6", "INT5", "INT4"), harm27)
        two = two_states(simple_harm, repair(-0.03, -0.01), langs=LANGS)
        gg = ga.GainRuns(two, DOWN, UP, FINE, n_controls=2); tt = ga.build_table(gg, n_boot=20, seed=0)
        rows4 = ga.curve_rows(tt, gg); star = ga.alpha_star(tt, gg)
        long_note = "exploratory, a long note about the status of the data that makes the subtitle wrap onto a third line of the figure"
        for theme in ("light", "dark"):
            self.assertEqual(overlaps(gf.fig_language_curves(rows, theme, long_note)), [], "language curves")
            self.assertEqual(overlaps(gf.fig_macro(rows, theme, long_note)), [], "macro")
            self.assertEqual(overlaps(gf.fig_heatmap(rows, theme, long_note)), [], "heatmap")
            self.assertEqual(overlaps(gf.fig_repair(rows4, star, {"l0": .8, "l1": .85, "l2": .9, "l3": .95}, theme, long_note)), [], "repair")

class Survivors(unittest.TestCase):
    def test_the_macro_curve_is_a_median_so_one_outlier_language_does_not_move_it(self):
        h = lambda c, a, k: (100.0 if (c == "l3" and k == "g") else simple_harm(c, a, k))
        rows, g = curve_rows(LANGS, STATES, h)
        m = gf.macro_by_state(rows)["FP"]
        i = m["ratios"].index(0.5)
        self.assertAlmostEqual(m["group"][i], float(np.median([simple_harm(c, 0.5, "g") if c != "l3" else 100.0 for c in LANGS])))
        self.assertLess(m["group"][i], 1.0)

    def test_the_27_language_panels_are_laid_out_in_six_columns_and_five_rows(self):
        rows, g = curve_rows(LANGS27, STATES, harm27)
        fig = gf.fig_language_curves(rows, "light", "x")
        panels = [a for a in fig.axes if a.get_gid() and a.get_gid().startswith("panel:")]
        self.assertEqual(len({round(a.get_position().x0, 4) for a in panels}), 6)
        self.assertEqual(len({round(a.get_position().y0, 4) for a in panels}), 5)

    def test_one_vertical_scale_covers_the_largest_language_curve(self):
        h = lambda c, a, k: (5.0 * harm27(c, a, k) if (c == "l7" and a == 8.0 and k == "g") else harm27(c, a, k))
        rows, g = curve_rows(LANGS27, STATES, h)
        norm = gf.normalised_net(rows, "FP")
        fig = gf.fig_language_curves(rows, "light", "x")
        panels = [a for a in fig.axes if a.get_gid() and a.get_gid().startswith("panel:")]
        self.assertEqual(len({a.get_ylim() for a in panels}), 1)
        top = max(v for l in norm for v in norm[l].values())
        self.assertGreaterEqual(panels[0].get_ylim()[1], top)

    def test_the_heatmap_numbers_change_colour_with_the_cell_shade(self):
        rows, g = curve_rows(LANGS27, ("FP", "INT8", "INT7", "INT6", "INT5", "INT4"), harm27)           # FP cells are light blue, INT4 cells dark blue
        fig = gf.fig_heatmap(rows, "light", "x")
        a = [a for a in fig.axes if a.get_gid() == "panel:down"][0]
        self.assertGreater(len({t.get_color() for t in a.texts if t.get_gid() == "cell"}), 1)

class RealDataFindings(unittest.TestCase):
    """Defects found when the figures were first drawn from the real FP and INT4 runs (2026-10-24): heavy-tailed values, ticks at ratios that are not in the study, a scatter stretched by one language."""
    def heavy_rows(self):
        h = lambda c, a, k: (harm27(c, a, k) * (1.0 + (60.0 if (a == 8.0 and c in ("l3", "l9")) else 0.0)) if k == "g" else harm27(c, a, k))
        return curve_rows(LANGS27, ("FP", "INT4"), h)

    def test_the_language_panels_and_the_macro_panels_use_a_symmetric_log_vertical_scale(self):
        rows, g = self.heavy_rows()
        panels = [a for a in gf.fig_language_curves(rows, "light", "x").axes if a.get_gid() and a.get_gid().startswith("panel:")]
        self.assertTrue(all(a.get_yscale() == "symlog" for a in panels))
        panels = [a for a in gf.fig_macro(rows, "light", "x").axes if a.get_gid() and a.get_gid().startswith("panel:")]
        self.assertTrue(all(a.get_yscale() == "symlog" for a in panels))

    def test_the_x_ticks_are_the_ratios_of_the_study_not_fixed_numbers(self):
        rows, g = self.heavy_rows()                                  # down 0.5 0.25 0.125, up 2 4 8
        fig = gf.fig_language_curves(rows, "light", "x")
        last = [a for a in fig.axes if a.get_gid() == "panel:l26"][0]
        labels = [t.get_text() for t in last.get_xticklabels()]
        self.assertIn("0.125", labels); self.assertIn("8", labels); self.assertIn("0", labels); self.assertIn("1", labels)
        macro = [a for a in gf.fig_macro(rows, "light", "x").axes if a.get_gid() == "panel:group"][0]
        ml = [t.get_text() for t in macro.get_xticklabels()]
        for want in ("0", "0.125", "0.25", "0.5", "1", "2", "4", "8"):
            self.assertIn(want, ml)
        self.assertNotIn("1.1", ml)                                  # the fine ratios are drawn but not labelled

    def test_heatmap_titles_show_the_ratio_to_three_significant_digits(self):
        rows, g = curve_rows(LANGS27, ("FP", "INT4"), harm27)
        rows = [dict(r, ratio=0.08838834764831845) if r["ratio"] == 0.125 else r for r in rows]          # a ratio of the real study
        fig = gf.fig_heatmap(rows, "light", "x")
        titles = [a.get_title(loc="left") for a in fig.axes if a.get_gid() and a.get_gid().startswith("panel:")]
        self.assertTrue(any(t.startswith("ratio 0.0884 ") for t in titles)); self.assertTrue(any(t.startswith("ratio 8 ") for t in titles))

    def test_heatmap_shades_stay_distinct_when_one_language_is_an_outlier(self):
        rows, g = self.heavy_rows()
        fig = gf.fig_heatmap(rows, "light", "x")
        a = [a for a in fig.axes if a.get_gid() == "panel:up"][0]
        pos = [float(a.images[0].norm(v)) for v in (0.2, 0.5, 1.0, 3.0, 20.0)]
        self.assertTrue(all(b - a_ > 0.05 for a_, b in zip(pos, pos[1:])), pos)                    # each step is a visible step of the colour scale

    def test_the_alpha_star_scatter_is_not_stretched_by_an_outlier_on_the_other_axis(self):
        rows, g = curve_rows(LANGS, ("FP", "INT4"), simple_harm)
        star = {"FP": {l: 1.0 for l in LANGS}, "INT4": {"l0": 1.2, "l1": 1.1, "l2": 1.1, "l3": 2.83}}
        base = {"l0": 0.85, "l1": 0.9, "l2": 0.88, "l3": 0.92}
        a = [a for a in gf.fig_repair(rows, star, base, "light", "x").axes if a.get_gid() == "panel:alpha"][0]
        self.assertLess(a.get_xlim()[1], 1.4); self.assertGreater(a.get_ylim()[1], 2.83)

if __name__ == "__main__":
    unittest.main()
