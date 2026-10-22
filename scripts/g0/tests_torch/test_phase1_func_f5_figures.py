import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_func_f5_analysis as an
import phase1_func_f5_figures as ff
from test_phase1_func_f5_analysis import build_varying_world, build_world, make_world


def tagged(fig, gid):
    return [a for a in fig.findobj() if a.get_gid() == gid]


def texts(fig):
    return [t.get_text() for t in fig.findobj(lambda a: hasattr(a, "get_text") and a.get_text())]


class Data(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sa, cls.st, cls.sm, cls.T = build_varying_world(); cls.res = an.analyse_f5(cls.sa, cls.st, cls.sm, meta=None, n_boot=60, seed=0)
        cls.arrays = ff.figure_arrays(cls.sa, cls.st, cls.sm, cls.T)

    def test_the_arrays_of_a_state_hold_the_write_similarity_the_read_correlation_and_the_label_of_the_tested_features(self):
        a = self.arrays["INT4"]; rows = self.T["rows"]
        np.testing.assert_array_equal(a["cos"], self.st["INT4"]["write"]["cos"]); np.testing.assert_array_equal(a["corr"], self.st["INT4"]["read"]["corr"][rows])
        np.testing.assert_array_equal(a["retained"], self.T["status"]["INT4"] == "retained"); self.assertEqual(len(a["control"]), len(a["cos"]))

    def test_full_precision_is_not_an_array_and_the_repeat_measurement_has_no_status(self):
        self.assertNotIn("FP", self.arrays); self.assertIn("FP_rebatched", self.arrays); self.assertNotIn("retained", self.arrays["FP_rebatched"])


class Figures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sa, cls.st, cls.sm, cls.T = build_varying_world(); cls.res = an.analyse_f5(cls.sa, cls.st, cls.sm, meta=None, n_boot=60, seed=0); cls.arrays = ff.figure_arrays(cls.sa, cls.st, cls.sm, cls.T)

    def test_the_distribution_figure_has_one_histogram_per_state_the_floor_and_the_control_band(self):
        for theme in ("light", "dark"):
            fig = ff.fig_w1_distribution(self.arrays, self.res["floor"], theme, "exploratory")
            labels = [g.get_gid() for g in fig.findobj() if (g.get_gid() or "").startswith("hist:")]
            self.assertEqual(sorted(labels), sorted(f"hist:{s}" for s in self.arrays)); self.assertEqual(len(tagged(fig, "floor")), 1)
            self.assertEqual(len(tagged(fig, "control_band")), len(self.arrays)); self.assertEqual(len(tagged(fig, "median")), len(self.arrays))
            self.assertAlmostEqual(float(tagged(fig, "floor")[0].get_xdata()[0]), self.res["floor"]["tau"])

    def test_each_histogram_peaks_where_the_planted_similarities_are(self):
        fig = ff.fig_w1_distribution(self.arrays, self.res["floor"], "light", "x"); h = [g for g in fig.findobj() if g.get_gid() == "hist:INT4"][0]
        v, edges, base = h.get_data(); k = int(np.argmax(v - base)); c = 0.5 * (edges[k] + edges[k + 1]); self.assertTrue(0.9 <= c <= 1.0 or 0.2 <= c <= 0.5)                       # the planted modes

    def test_undefined_similarities_are_not_drawn_and_are_reported_in_the_row_label(self):
        arr = {k: {kk: (vv.copy() if hasattr(vv, "copy") else vv) for kk, vv in v.items()} for k, v in self.arrays.items()}; arr["INT4"]["cos"][:10] = np.nan
        fig = ff.fig_w1_distribution(arr, self.res["floor"], "light", "x"); self.assertTrue(any("10 undefined" in s for s in texts(fig)))

    def test_the_scatter_has_one_panel_per_state_with_the_two_status_groups_and_the_two_cutoffs(self):
        fig = ff.fig_read_vs_write(self.arrays, self.res["floor"]["tau"], "light", "exploratory", states=("INT6", "INT5", "INT4"))
        for s in ("INT6", "INT5", "INT4"):
            ret, lost = tagged(fig, f"retained:{s}")[0], tagged(fig, f"lost:{s}")[0]; a = self.arrays[s]
            self.assertEqual(len(ret.get_offsets()), int(a["retained"].sum())); self.assertEqual(len(lost.get_offsets()), int((~a["retained"]).sum()))
        self.assertEqual(len(tagged(fig, "duan_cutoff")), 3); self.assertEqual(len(tagged(fig, "floor")), 3)
        pts = tagged(fig, "retained:INT4")[0].get_offsets(); np.testing.assert_allclose(pts[:, 1], self.arrays["INT4"]["cos"][self.arrays["INT4"]["retained"]])

    def test_the_statement_figure_draws_each_statement_with_its_threshold_and_its_verdict(self):
        fig = ff.fig_statements(self.res["validity"], "light", "exploratory"); s = self.res["validity"]["statements"]
        pts = {g.get_gid(): g for g in fig.findobj() if (g.get_gid() or "").startswith("point:")}; self.assertEqual(sorted(pts), ["point:S1", "point:S2", "point:S3"])
        self.assertAlmostEqual(float(pts["point:S1"].get_offsets()[0][0]), s["S1"]["share"]); self.assertAlmostEqual(float(pts["point:S2"].get_offsets()[0][0]), s["S2"]["auroc"])
        for k in ("S1", "S2", "S3"):
            self.assertTrue(any(s[k]["verdict"] in t for t in texts(fig)))
        self.assertTrue(any(self.res["validity"]["overall"] in t for t in texts(fig)))

    def test_an_undefined_statement_is_listed_and_not_drawn(self):
        sa, st, sm = build_world("sensible")[:3]; st2 = {k: v for k, v in st.items() if k in ("FP", "FP_rebatched", "INT4")}
        v = an.analyse_f5(sa, st2, sm, meta=None, n_boot=30, seed=0)["validity"]; fig = ff.fig_statements(v, "light", "x")
        self.assertEqual(sorted(g.get_gid() for g in fig.findobj() if (g.get_gid() or "").startswith("point:")), ["point:S2"]); self.assertTrue(any("undefined" in t for t in texts(fig)))

    def test_the_retrieval_figure_has_the_overall_the_within_language_and_the_chance_level(self):
        fig = ff.fig_retrieval(self.res["tables"], "light", "x"); self.assertEqual(len(tagged(fig, "top1")), 1); self.assertEqual(len(tagged(fig, "within_top1")), 1); self.assertEqual(len(tagged(fig, "chance")), 1)
        qs = [s for s in ff.STATE_ORDER if s in self.res["tables"] and s not in ff.REFERENCE_STATES]; y = tagged(fig, "top1")[0].get_ydata(); self.assertEqual(len(y), len(qs))
        self.assertAlmostEqual(float(y[0]), self.res["tables"][qs[0]]["all"]["retrieval"]["top1"])
        rep = tagged(fig, "top1_repeat")[0]; self.assertEqual(len(rep.get_ydata()), len(ff.REFERENCE_STATES)); self.assertEqual(rep.get_linestyle(), "None")                          # the repeat measurement is a reference, not a step of the precision ladder

    def test_the_median_ladder_shows_read_and_write_medians_with_intervals_and_the_floor(self):
        fig = ff.fig_median_ladder(self.res["tables"], self.res["floor"], "light", "x"); r1, w1 = tagged(fig, "R1")[0], tagged(fig, "W1")[0]
        qs = [s for s in ff.STATE_ORDER if s in self.res["tables"] and s not in ff.REFERENCE_STATES]; self.assertEqual(len(r1.get_ydata()), len(qs))
        self.assertAlmostEqual(float(r1.get_ydata()[0]), self.res["tables"][qs[0]]["all"]["R1"]["median"]); self.assertEqual(len(tagged(fig, "floor")), 1)
        self.assertEqual(len(w1.get_ydata()), len(qs)); rep = tagged(fig, "W1_repeat")[0]; self.assertEqual(len(rep.get_ydata()), len(ff.REFERENCE_STATES)); self.assertEqual(rep.get_linestyle(), "None")
        np.testing.assert_allclose(rep.get_ydata(), [self.res["tables"][s]["all"]["W1"]["median"] for s in ff.REFERENCE_STATES])

    def test_the_control_band_is_inside_the_axis_even_when_the_control_reaches_below_the_floor_of_the_axis(self):
        arr = {k: {kk: (vv.copy() if hasattr(vv, "copy") else vv) for kk, vv in v.items()} for k, v in self.arrays.items()}; arr["INT4"]["control"][:] = -0.7; arr["INT4"]["control"][::2] = -0.9
        fig = ff.fig_w1_distribution(arr, self.res["floor"], "light", "x"); ax = fig.axes[0]; lo, hi = ax.get_xlim(); self.assertLessEqual(lo, -0.9)

    def test_the_paired_figure_has_one_dot_per_language_and_precision_and_a_zero_line(self):
        fig = ff.fig_paired(self.res["validity"], "light", "x"); p = self.res["validity"]["paired"]
        for lab in ("INT6", "INT5", "INT4"):
            d = tagged(fig, f"paired:{lab}")[0]; self.assertEqual(len(d.get_offsets()), len(p[lab]["by_language"]))
        self.assertEqual(len(tagged(fig, "zero")), 1)

    def test_every_figure_renders_in_both_themes_and_the_files_are_written(self):
        with tempfile.TemporaryDirectory() as d:
            paths = ff.write_figures(self.sa, self.st, self.sm, self.T, self.res, d, note="planted worlds", themes=("light", "dark"), formats=("png", "svg"))
            names = sorted(Path(p).name for p in paths); self.assertEqual(len(names), 6 * 2 * 2)
            for stem in ("f5_fig1_w1_distribution", "f5_fig2_read_vs_write", "f5_fig3_statements", "f5_fig4_retrieval", "f5_fig5_median_ladder", "f5_fig6_paired"):
                for theme in ("light", "dark"):
                    self.assertIn(f"{stem}_{theme}.png", names)
            self.assertTrue(all(Path(p).stat().st_size > 2000 for p in paths))

    def test_every_title_states_that_the_data_are_exploratory(self):
        fig = ff.fig_statements(self.res["validity"], "light", "exploratory, planted"); self.assertTrue(any("exploratory" in t for t in texts(fig)))

class Constants(unittest.TestCase):
    def test_the_noise_only_states_are_the_references_and_come_before_the_ladder(self):
        self.assertEqual(ff.REFERENCE_STATES, ("FP_rebatched", "INT16")); self.assertEqual(ff.STATE_ORDER, ["FP_rebatched", "INT16", "INT8", "INT6", "INT5", "INT4"])
        self.assertEqual(sorted(ff.LABELS), sorted(ff.STATE_ORDER)); self.assertEqual(sorted(ff.PLAIN), sorted(ff.STATE_ORDER))


class DrawnValues(unittest.TestCase):
    """What is drawn equals the data: positions, colours, lines and thresholds."""
    @classmethod
    def setUpClass(cls):
        cls.sa, cls.st, cls.sm, cls.T = build_varying_world(); cls.res = an.analyse_f5(cls.sa, cls.st, cls.sm, meta=None, n_boot=60, seed=0); cls.arrays = ff.figure_arrays(cls.sa, cls.st, cls.sm, cls.T)
        cls.tau = cls.res["floor"]["tau"]

    def test_the_read_correlation_is_taken_at_the_tested_rows_when_features_are_untested(self):
        sa, st, sm = make_world(); rng = np.random.default_rng(2)
        for s in st.values():
            s["read"]["corr"] = rng.uniform(0.0, 1.0, len(s["read"]["corr"]))
        T = an.assemble(sa, sm); a = ff.figure_arrays(sa, st, sm, T)
        np.testing.assert_array_equal(a["INT4"]["corr"], st["INT4"]["read"]["corr"][T["rows"]]); self.assertEqual(len(a["INT4"]["corr"]), len(T["rows"]))

    def test_median_line_and_control_band_are_the_median_and_the_fifth_to_ninety_fifth_percentile(self):
        fig = ff.fig_w1_distribution(self.arrays, self.res["floor"], "light", "x"); order = [s for s in ff.STATE_ORDER if s in self.arrays]
        med = sorted(float(g.get_xdata()[0]) for g in tagged(fig, "median")); want = sorted(float(np.median(self.arrays[s]["cos"][~np.isnan(self.arrays[s]["cos"])])) for s in order)
        np.testing.assert_allclose(med, want)
        bands = sorted((float(g.get_segments()[0][0][0]), float(g.get_segments()[0][1][0])) for g in tagged(fig, "control_band"))
        want = sorted((float(np.percentile(self.arrays[s]["control"][~np.isnan(self.arrays[s]["control"])], 5)), float(np.percentile(self.arrays[s]["control"][~np.isnan(self.arrays[s]["control"])], 95))) for s in order)
        np.testing.assert_allclose(bands, want)

    def test_the_scatter_colours_lines_and_thresholds(self):
        from phase1_func_figures import theme_tokens
        fig = ff.fig_read_vs_write(self.arrays, self.tau, "light", "x", states=("INT6", "INT5", "INT4")); t = theme_tokens("light")
        from matplotlib.colors import to_rgb
        for s in ("INT6", "INT5", "INT4"):
            self.assertEqual(tuple(tagged(fig, f"retained:{s}")[0].get_facecolor()[0][:3]), to_rgb(t["series"][0])); self.assertEqual(tuple(tagged(fig, f"lost:{s}")[0].get_facecolor()[0][:3]), to_rgb(t["series"][1]))
        for g in tagged(fig, "duan_cutoff"):
            self.assertEqual(float(g.get_xdata()[0]), 0.9)
        for g in tagged(fig, "floor"):
            self.assertAlmostEqual(float(g.get_ydata()[0]), self.tau)

    def test_the_rules_are_drawn_at_their_thresholds(self):
        fig = ff.fig_statements(self.res["validity"], "light", "x"); r = {g.get_gid(): g for g in fig.findobj() if (g.get_gid() or "").startswith("rule")}
        self.assertEqual([float(r[f"rule:{k}"].get_offsets()[0][0]) for k in ("S1", "S2", "S3")], [0.8, 0.6, 0.8]); self.assertEqual(float(r["rule_lower:S2"].get_segments()[0][0][0]), 0.5)

    def test_the_verdict_texts_are_the_verdicts_in_the_order_of_the_statements(self):
        for world in (build_varying_world(), build_world("independent")):
            sa, st, sm = world[:3]; v = an.analyse_f5(sa, st, sm, meta=None, n_boot=30, seed=0)["validity"]; fig = ff.fig_statements(v, "light", "x")
            ver = [t for t in fig.findobj(lambda a: hasattr(a, "get_text") and getattr(a, "get_fontweight", lambda: "")() == "bold")]
            self.assertEqual(sorted((round(x.get_position()[1]), x.get_text()) for x in ver), sorted((2 - i, v["statements"][k]["verdict"]) for i, k in enumerate(("S1", "S2", "S3"))))

    def test_the_chance_line_and_the_ladder_floor_carry_their_values(self):
        fig = ff.fig_retrieval(self.res["tables"], "light", "x"); order = [s for s in ff.STATE_ORDER if s in self.res["tables"]]
        np.testing.assert_allclose(tagged(fig, "chance")[0].get_ydata(), [self.res["tables"][s]["all"]["retrieval"]["within"]["chance"] for s in order])
        fig = ff.fig_median_ladder(self.res["tables"], self.res["floor"], "light", "x"); self.assertAlmostEqual(float(tagged(fig, "floor")[0].get_ydata()[0]), self.tau)

    def test_the_paired_dots_are_the_differences_and_the_languages_are_sorted_by_int4(self):
        fig = ff.fig_paired(self.res["validity"], "light", "x"); p = self.res["validity"]["paired"]["INT4"]["by_language"]; off = tagged(fig, "paired:INT4")[0].get_offsets()
        np.testing.assert_allclose(sorted(off[:, 0]), sorted(p.values())); order = np.argsort(off[:, 1]); self.assertEqual(list(off[order, 0]), sorted(off[:, 0]))                      # y grows with the INT4 difference
        labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]; self.assertEqual(labels, sorted(p, key=p.get))


if __name__ == "__main__":
    unittest.main()
