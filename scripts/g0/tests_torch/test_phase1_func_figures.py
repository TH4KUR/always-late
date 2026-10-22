import json
import sys
import unittest

import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import matplotlib
    HAVE_MPL = True
except ImportError:                                                     # matplotlib lives in the torch venv, not in the stdlib environment
    HAVE_MPL = False
if HAVE_MPL:                                                            # outside the try: a missing or broken module must fail the file, not skip it
    from matplotlib.colors import to_hex
    import phase1_func_figures as ff


def by_gid(ax, gid):
    return [a for a in list(ax.collections) + list(ax.lines) + list(ax.patches) + list(ax.texts) if a.get_gid() == gid]


def all_text(fig):
    out = [t.get_text() for t in fig.texts]
    for ax in fig.axes:
        out += [ax.get_title(loc="left"), ax.get_xlabel(), ax.get_ylabel()] + [t.get_text() for t in ax.texts]
    return " | ".join(out)


def assert_text_inside(test, fig):
    """Render and check that no visible text (title, subtitle, ticks, legend, notes) is clipped by the canvas edge."""
    fig.canvas.draw(); r = fig.canvas.get_renderer(); box = fig.bbox
    for tx in fig.findobj(matplotlib.text.Text):
        if not tx.get_visible() or not tx.get_text().strip():
            continue
        b = tx.get_window_extent(r)
        test.assertGreaterEqual(b.x0, box.x0 - 1, tx.get_text()); test.assertLessEqual(b.x1, box.x1 + 1, tx.get_text())
        test.assertGreaterEqual(b.y0, box.y0 - 1, tx.get_text()); test.assertLessEqual(b.y1, box.y1 + 1, tx.get_text())


@unittest.skipUnless(HAVE_MPL, "matplotlib not installed")
class Themes(unittest.TestCase):
    def test_both_themes_have_the_palette_tokens(self):
        for mode in ("light", "dark"):
            t = ff.theme_tokens(mode)
            for k in ("surface", "ink", "ink2", "muted", "grid", "axis", "series", "ordinal", "ordinal4", "ramp"):
                self.assertIn(k, t)
            self.assertEqual(len(t["series"]), 3); self.assertEqual(len(t["ordinal"]), 3); self.assertEqual(len(t["ordinal4"]), 4)

    def test_the_documented_palette_values(self):
        self.assertEqual(ff.theme_tokens("light")["surface"], "#fcfcfb"); self.assertEqual(ff.theme_tokens("dark")["surface"], "#1a1a19")
        self.assertEqual(ff.theme_tokens("light")["series"][0], "#2a78d6"); self.assertEqual(ff.theme_tokens("dark")["series"][0], "#3987e5")
        self.assertEqual(ff.theme_tokens("light")["series"][1], "#eb6834"); self.assertEqual(ff.theme_tokens("dark")["series"][1], "#d95926")

    def test_the_validated_ordinal_ramps_are_pinned(self):
        # these hex values passed validate_palette.js --ordinal in both modes (2026-10-22); changing one means validating again
        self.assertEqual(ff.theme_tokens("light")["ordinal"], ["#86b6ef", "#2a78d6", "#104281"]); self.assertEqual(ff.theme_tokens("dark")["ordinal"], ["#184f95", "#3987e5", "#9ec5f4"])
        self.assertEqual(ff.theme_tokens("light")["ordinal4"], ["#0d366b", "#1c5cab", "#3987e5", "#86b6ef"]); self.assertEqual(ff.theme_tokens("dark")["ordinal4"], ["#b7d3f6", "#6da7ec", "#2a78d6", "#184f95"])

    def test_unknown_theme_is_an_error(self):
        with self.assertRaises(ValueError):
            ff.theme_tokens("sepia")


@unittest.skipUnless(HAVE_MPL, "matplotlib not installed")
class EffectPerLanguage(unittest.TestCase):
    ROWS = [{"language": "aaa", "effect": 0.30, "lo": 0.20, "hi": 0.40, "harm_control": 0.02},
            {"language": "bbb", "effect": 7.00, "lo": 6.00, "hi": 8.00, "harm_control": 0.05},
            {"language": "ccc", "effect": 0.005, "lo": -0.01, "hi": 0.02, "harm_control": 0.01}]

    def setUp(self):
        self.fig = ff.fig_effect_per_language(self.ROWS, "light", note="pilot, exploratory")
        self.ax = self.fig.axes[0]

    def test_languages_are_sorted_by_the_effect_with_the_largest_on_top(self):
        self.assertEqual([t.get_text() for t in self.ax.get_yticklabels()], ["ccc", "aaa", "bbb"])         # bottom to top

    def test_effect_points_and_intervals_are_the_input_numbers(self):
        pts = by_gid(self.ax, "effect")[0].get_offsets()
        self.assertEqual([tuple(p) for p in pts.tolist()], [(0.005, 0.0), (0.30, 1.0), (7.00, 2.0)])
        segs = [[tuple(p) for p in s.tolist()] for s in by_gid(self.ax, "interval")[0].get_segments()]
        self.assertEqual(segs, [[(-0.01, 0.0), (0.02, 0.0)], [(0.20, 1.0), (0.40, 1.0)], [(6.00, 2.0), (8.00, 2.0)]])      # an interval that crosses zero is drawn, not clipped

    def test_control_harm_is_the_reference_on_the_same_rows(self):
        pts = by_gid(self.ax, "control")[0].get_offsets()
        self.assertEqual([tuple(p) for p in pts.tolist()], [(0.01, 0.0), (0.02, 1.0), (0.05, 2.0)])

    def test_axis_is_symlog_because_the_effects_span_orders_of_magnitude_and_can_cross_zero(self):
        self.assertEqual(self.ax.get_xscale(), "symlog")
        self.assertIn("nats per token", self.ax.get_xlabel())

    def test_ticks_the_room_for_an_interval_below_zero_and_labels_only_the_extremes(self):
        self.assertEqual([t.get_text() for t in self.ax.get_xticklabels()], ["0", "0.01", "0.1", "1", "10"])
        self.assertLessEqual(self.ax.get_xlim()[0], -0.01)                                                  # the interval that crosses zero is inside the axes
        self.assertEqual(sorted(t.get_text() for t in by_gid(self.ax, "label")), ["0.005", "7"])

    def test_legend_names_both_series_and_the_note_states_the_status(self):
        texts = [t.get_text() for t in self.ax.get_legend().get_texts()]
        self.assertEqual(len(texts), 2); self.assertIn("95 %", texts[0]); self.assertIn("control", texts[1])
        self.assertIn("pilot, exploratory", all_text(self.fig))

    def test_surface_follows_the_theme(self):
        self.assertEqual(to_hex(self.fig.get_facecolor()), "#fcfcfb")
        self.assertEqual(to_hex(ff.fig_effect_per_language(self.ROWS, "dark", note="x").get_facecolor()), "#1a1a19")

    def test_no_rows_is_an_error(self):
        with self.assertRaises(ValueError):
            ff.fig_effect_per_language([], "light", note="x")

    def test_the_unit_of_the_effect_axis_is_a_parameter(self):
        ax = ff.fig_effect_per_language(self.ROWS, "light", note="x", unit="nats per item").axes[0]
        self.assertIn("nats per item", ax.get_xlabel()); self.assertNotIn("per token", ax.get_xlabel())

    def test_no_text_is_clipped_in_either_theme(self):
        for theme in ("light", "dark"):
            assert_text_inside(self, ff.fig_effect_per_language(self.ROWS, theme, note="pilot, exploratory"))
        many = [{"language": f"lang_{i:02d}_Latn", "effect": 0.01 * (i + 1), "lo": 0.005 * (i + 1), "hi": 0.02 * (i + 1), "harm_control": 0.01} for i in range(27)]
        assert_text_inside(self, ff.fig_effect_per_language(many, "light", note="pilot, exploratory"))

    def test_intervals_carry_end_caps_so_that_narrow_intervals_stay_visible(self):
        segs = sorted(tuple(map(tuple, s.tolist())) for s in by_gid(self.ax, "interval_cap")[0].get_segments())
        want = sorted(((x, y - 0.14), (x, y + 0.14)) for y, (lo, hi) in enumerate([(-0.01, 0.02), (0.20, 0.40), (6.00, 8.00)]) for x in (lo, hi))
        self.assertEqual(segs, want)

    def test_the_subtitle_carries_the_group_size_and_the_status(self):
        txt = all_text(ff.fig_effect_per_language(self.ROWS, "light", note="pilot, exploratory", K="K64"))
        self.assertIn("K64", txt); self.assertIn("3 languages", txt); self.assertIn("pilot, exploratory", txt)


def planted_retention():
    def lang(r, lo, hi, c="preserved"):
        return {"ratio": r, "lo": lo, "hi": hi, "class": c}
    return {"INT8": {"macro": {"ratio": 1.01, "lo": 1.00, "hi": 1.03, "class": "preserved", "n_languages": 2, "median_ratio": 1.00, "median_lo": 0.99, "median_hi": 1.01},
                     "pooled": {"ratio": 1.0, "lo": 0.99, "hi": 1.01, "class": "preserved"},
                     "languages": {"aaa": lang(1.02, 0.9, 1.1), "bbb": lang(0.98, 0.9, 1.0), "vie": lang(None, None, None, "undefined")}},
            "INT4": {"macro": {"ratio": 2.14, "lo": 2.02, "hi": 2.55, "class": "amplified", "n_languages": 2, "median_ratio": 1.43, "median_lo": 1.40, "median_hi": 1.61},
                     "pooled": {"ratio": 1.34, "lo": 1.3, "hi": 1.4, "class": "inconclusive"},
                     "languages": {"aaa": lang(3.0, 2.5, 3.5, "amplified"), "bbb": lang(-0.5, -0.9, 0.1, "attenuated"), "vie": lang(None, None, None, "undefined")}}}


@unittest.skipUnless(HAVE_MPL, "matplotlib not installed")
class Retention(unittest.TestCase):
    def setUp(self):
        self.fig = ff.fig_retention(planted_retention(), (0.75, 1.33), "light", note="pilot, exploratory")
        self.ax = self.fig.axes[0]

    def test_precisions_are_ordered_from_the_numerical_floor_to_the_coarsest_and_only_those_present(self):
        self.assertEqual([t.get_text() for t in self.ax.get_xticklabels()], ["INT8", "INT4"])
        r = planted_retention(); r["INT16"] = r["INT8"]; r["INT6"] = r["INT4"]
        ax = ff.fig_retention(r, (0.75, 1.33), "light", note="x").axes[0]
        self.assertEqual([t.get_text() for t in ax.get_xticklabels()], ["INT16", "INT8", "INT6", "INT4"])

    def test_macro_and_median_markers_and_intervals_are_the_input_numbers(self):
        macro = [tuple(p) for p in by_gid(self.ax, "macro")[0].get_offsets().tolist()]
        self.assertEqual([round(x, 6) for x, _ in macro], [-0.1, 0.9]); self.assertEqual([y for _, y in macro], [1.01, 2.14])
        med = [tuple(p) for p in by_gid(self.ax, "median")[0].get_offsets().tolist()]
        self.assertEqual([round(x, 6) for x, _ in med], [0.1, 1.1]); self.assertEqual([y for _, y in med], [1.00, 1.43])
        seg = [[tuple(p) for p in s.tolist()] for s in by_gid(self.ax, "macro_interval")[0].get_segments()]
        self.assertEqual([(round(s[0][0], 6), s[0][1], s[1][1]) for s in seg], [(-0.1, 1.00, 1.03), (0.9, 2.02, 2.55)])

    def test_language_ratios_are_context_dots_around_their_precision_and_undefined_ones_are_not_plotted(self):
        pts = by_gid(self.ax, "language")[0].get_offsets().tolist()
        self.assertEqual(len(pts), 3)                                                                       # aaa and bbb at INT8, aaa at INT4 (the negative ratio is on the floor, below)
        self.assertEqual(sorted(round(y, 6) for _, y in pts), [0.98, 1.02, 3.0])
        for x, y in pts:
            self.assertTrue(abs(x - round(x)) <= 0.25 and round(x) in (0, 1))

    def test_a_ratio_below_the_log_axis_floor_is_drawn_on_the_floor_and_counted(self):
        floor = by_gid(self.ax, "language_floor")[0].get_offsets().tolist()
        self.assertEqual(len(floor), 1); self.assertEqual(floor[0][1], self.ax.get_ylim()[0])
        self.assertIn("1 value", all_text(self.fig)); self.assertIn("below the axis", all_text(self.fig))

    def test_the_axis_floor_is_a_quarter_so_a_ratio_of_one_fifth_is_drawn_on_it(self):
        r = planted_retention(); r["INT4"]["languages"]["bbb"] = {"ratio": 0.2, "lo": 0.1, "hi": 0.3, "class": "attenuated"}
        ax = ff.fig_retention(r, (0.75, 1.33), "light", note="x").axes[0]
        self.assertEqual(ax.get_ylim()[0], 0.25); self.assertEqual(by_gid(ax, "language_floor")[0].get_offsets().tolist()[0][1], 0.25)

    def test_an_interval_that_reaches_below_the_floor_is_cut_at_the_floor(self):
        r = planted_retention(); r["INT4"]["macro"]["lo"] = 0.1; r["INT4"]["macro"]["median_lo"] = 0.05
        ax = ff.fig_retention(r, (0.75, 1.33), "light", note="x").axes[0]
        lows = sorted(s.tolist()[0][1] for s in by_gid(ax, "macro_interval")[0].get_segments()); self.assertEqual(lows, [0.25, 1.0])
        lows = sorted(s.tolist()[0][1] for s in by_gid(ax, "median_interval")[0].get_segments()); self.assertEqual(lows, [0.25, 0.99])

    def test_the_value_axis_is_labelled_with_the_reference_the_band_edges_and_round_numbers(self):
        labels = [t.get_text() for t in self.ax.get_yticklabels() if t.get_text()]
        for want in ("0.25", "0.75", "1", "1.33", "2"):
            self.assertIn(want, labels)

    def test_language_dots_are_the_gray_context_and_the_two_summaries_wear_the_accent_and_the_second_hue(self):
        tok = ff.theme_tokens("light")
        self.assertEqual(to_hex(by_gid(self.ax, "language")[0].get_facecolor()[0][:3]), tok["muted"])
        self.assertEqual(to_hex(by_gid(self.ax, "macro")[0].get_facecolor()[0][:3]), tok["series"][0])
        self.assertEqual(to_hex(by_gid(self.ax, "median")[0].get_facecolor()[0][:3]), tok["series"][1])

    def test_undefined_languages_are_reported_not_zeroed(self):
        self.assertIn("undefined", all_text(self.fig)); self.assertIn("1 language", all_text(self.fig))

    def test_band_reference_line_and_log_axis(self):
        self.assertEqual(self.ax.get_yscale(), "log")
        band = by_gid(self.ax, "band")[0]
        self.assertAlmostEqual(band.get_y(), 0.75); self.assertAlmostEqual(band.get_y() + band.get_height(), 1.33)
        self.assertTrue(any(abs(l.get_ydata()[0] - 1.0) < 1e-12 for l in self.ax.lines if l.get_gid() == "one"))

    def test_legend_has_the_primary_and_the_robust_summary(self):
        texts = " | ".join(t.get_text() for t in self.ax.get_legend().get_texts())
        for s in ("macro mean of ratios", "median ratio", "preserved band", "0.75", "1.33"):
            self.assertIn(s, texts)

    def test_no_precision_is_an_error(self):
        with self.assertRaises(ValueError):
            ff.fig_retention({}, (0.75, 1.33), "light", note="x")

    def test_no_text_is_clipped_in_either_theme(self):
        for theme in ("light", "dark"):
            assert_text_inside(self, ff.fig_retention(planted_retention(), (0.75, 1.33), theme, note="pilot, exploratory"))
        r = planted_retention()
        for p in ("INT16", "INT12", "INT7", "INT6", "INT5"):
            r[p] = r["INT8"]
        assert_text_inside(self, ff.fig_retention(r, (0.75, 1.33), "light", note="pilot, exploratory"))

    def test_intervals_carry_end_caps(self):
        segs = sorted((round(s[0][0], 6), round(s[1][0], 6), s[0][1]) for s in (x.tolist() for c in by_gid(self.ax, "macro_cap") for x in c.get_segments()))
        want = sorted((round(x - 0.05, 6), round(x + 0.05, 6), y) for x in (-0.1, 0.9) for y in ((1.00, 1.03) if x < 0 else (2.02, 2.55)))
        self.assertEqual(segs, want)


def planted_families():
    def row(p, l, f, h):
        return {"precision": p, "language": l, "family": f, "harm": h, "lo": h - 0.05, "hi": h + 0.05}
    rows = []
    for p, (a16, a64, aall, b16, b64, ball) in {"FP": (0.1, 0.2, 0.3, 0.3, 0.4, 0.5), "INT8": (0.1, 0.2, 0.3, 0.3, 0.4, 0.5), "INT4": (0.2, 0.4, 0.5, 0.6, 0.8, 0.9)}.items():
        for l, vals in (("aaa", (a16, a64, aall)), ("bbb", (b16, b64, ball))):
            for f, h in zip(("E_K16", "E_K64", "E_all"), vals):
                rows.append(row(p, l, f, h))
            rows.append(row(p, l, "ctrl_K64", 0.02))                                                      # other families are ignored by the size figure
    return rows


@unittest.skipUnless(HAVE_MPL, "matplotlib not installed")
class SizeCurves(unittest.TestCase):
    def setUp(self):
        self.fig = ff.fig_size_curves(planted_families(), "light", note="pilot, exploratory")
        self.ax = self.fig.axes[0]

    def test_median_over_languages_per_precision_with_full_precision_first(self):
        self.assertEqual([t.get_text() for t in self.ax.get_xticklabels()], ["FP", "INT8", "INT4"])
        self.assertEqual([round(v, 6) for v in by_gid(self.ax, "size_K16")[0].get_ydata()], [0.2, 0.2, 0.4])
        self.assertEqual([round(v, 6) for v in by_gid(self.ax, "size_K64")[0].get_ydata()], [0.3, 0.3, 0.6])
        self.assertEqual([round(v, 6) for v in by_gid(self.ax, "size_all")[0].get_ydata()], [0.4, 0.4, 0.7])

    def test_group_size_is_an_ordinal_ramp_in_the_theme_colours_with_direct_end_labels(self):
        tokens = ff.theme_tokens("light")["ordinal"]
        self.assertEqual([to_hex(by_gid(self.ax, g)[0].get_color()) for g in ("size_K16", "size_K64", "size_all")], tokens)
        self.assertEqual(sorted(t.get_text() for t in by_gid(self.ax, "end_label")), ["K16", "K64", "all features"])

    def test_the_summary_is_the_median_so_one_outlier_language_does_not_move_it(self):
        rows = [{"precision": "FP", "language": l, "family": "E_K64", "harm": h, "lo": 0.0, "hi": 1.0} for l, h in (("a", 0.1), ("b", 0.2), ("c", 9.0))]
        self.assertEqual(list(by_gid(ff.fig_size_curves(rows, "light", note="x").axes[0], "size_K64")[0].get_ydata()), [0.2])

    def test_axis_starts_at_zero_and_says_what_the_number_is(self):
        self.assertEqual(self.ax.get_ylim()[0], 0.0); self.assertIn("nats per token", self.ax.get_ylabel())

    def test_a_missing_family_is_skipped_and_no_data_is_an_error(self):
        rows = [r for r in planted_families() if r["family"] != "E_all"]
        ax = ff.fig_size_curves(rows, "light", note="x").axes[0]
        self.assertEqual(by_gid(ax, "size_all"), []); self.assertEqual(len(by_gid(ax, "size_K16")), 1)
        with self.assertRaises(ValueError):
            ff.fig_size_curves([r for r in planted_families() if not r["family"].startswith("E_")], "light", note="x")

    def test_no_text_is_clipped_in_either_theme(self):
        for theme in ("light", "dark"):
            assert_text_inside(self, ff.fig_size_curves(planted_families(), theme, note="pilot, exploratory"))


ASSOC = {"INT8": {"macro": 0.98, "pooled": 0.97, "n_languages": 2}, "INT4": {"macro": 0.70, "pooled": 0.71, "n_languages": 2}}


@unittest.skipUnless(HAVE_MPL, "matplotlib not installed")
class AssociationVersusFunction(unittest.TestCase):
    def setUp(self):
        self.fig = ff.fig_assoc_vs_function(ASSOC, planted_retention(), "light", note="pilot, exploratory")
        self.ax = self.fig.axes[0]

    def test_both_series_share_one_axis_indexed_to_full_precision(self):
        self.assertEqual(len(self.fig.axes), 1)                                                            # never a second y axis
        self.assertEqual(self.ax.get_yscale(), "log")
        self.assertEqual([t.get_text() for t in self.ax.get_xticklabels()], ["FP", "INT8", "INT4"])
        self.assertEqual(list(by_gid(self.ax, "function")[0].get_ydata()), [1.0, 1.01, 2.14])
        self.assertEqual(list(by_gid(self.ax, "association")[0].get_ydata()), [1.0, 0.98, 0.70])
        self.assertTrue(any(abs(l.get_ydata()[0] - 1.0) < 1e-12 for l in self.ax.lines if l.get_gid() == "one"))

    def test_function_has_its_interval_and_association_is_descriptive_only(self):
        seg = [s.tolist() for s in by_gid(self.ax, "function_interval")[0].get_segments()]
        self.assertEqual([(s[0][1], s[1][1]) for s in seg], [(1.0, 1.0), (1.00, 1.03), (2.02, 2.55)])      # full precision is the reference: a zero-width interval
        self.assertEqual(by_gid(self.ax, "association_interval"), [])

    def test_legend_names_both_series_and_the_reference_each_in_its_own_colour(self):
        texts = [t.get_text() for t in self.ax.get_legend().get_texts()]
        self.assertEqual(len(texts), 2); self.assertIn("association", texts[0] + texts[1]); self.assertIn("function", texts[0] + texts[1])
        tok = ff.theme_tokens("light")["series"]
        self.assertEqual(to_hex(by_gid(self.ax, "function")[0].get_color()), tok[0]); self.assertEqual(to_hex(by_gid(self.ax, "association")[0].get_color()), tok[1])

    def test_only_precisions_present_in_both_inputs_are_drawn_and_nothing_in_common_is_an_error(self):
        a = dict(ASSOC); a["INT6"] = {"macro": 0.9, "pooled": 0.9, "n_languages": 2}
        ax = ff.fig_assoc_vs_function(a, planted_retention(), "light", note="x").axes[0]
        self.assertEqual([t.get_text() for t in ax.get_xticklabels()], ["FP", "INT8", "INT4"])
        with self.assertRaises(ValueError):
            ff.fig_assoc_vs_function({"INT5": {"macro": 0.9, "pooled": 0.9, "n_languages": 2}}, planted_retention(), "light", note="x")

    def test_no_text_is_clipped_in_either_theme(self):
        for theme in ("light", "dark"):
            assert_text_inside(self, ff.fig_assoc_vs_function(ASSOC, planted_retention(), theme, note="pilot, exploratory"))


def planted_rank_rows():
    def row(c, b, n_est, n_ret):
        f = n_ret / n_est if n_est else None
        return {"condition": c, "bin": b, "n_established": n_est, "n_retained": n_ret, "fraction": f, "lo": None if f is None else max(0.0, f - 0.05), "hi": None if f is None else min(1.0, f + 0.05), "n_languages": 3}
    rows = []
    for c, fr in (("INT8", (0.99, 0.98, 0.97, 0.96)), ("INT4", (0.85, 0.80, 0.66, 0.69))):
        for b, f in zip(("1-16", "17-64", "65-256", "257+"), fr):
            rows.append(row(c, b, 100, round(100 * f)))
    return rows


@unittest.skipUnless(HAVE_MPL, "matplotlib not installed")
class RankBins(unittest.TestCase):
    BINS = ("1-16", "17-64", "65-256", "257+")

    def setUp(self):
        self.fig = ff.fig_rank_bins(planted_rank_rows(), "light", note="descriptive")
        self.ax = self.fig.axes[0]

    def test_one_line_per_rank_bin_from_full_precision_at_one_to_the_precisions_present(self):
        self.assertEqual([t.get_text() for t in self.ax.get_xticklabels()], ["FP", "INT8", "INT4"])
        want = {"1-16": [1.0, 0.99, 0.85], "17-64": [1.0, 0.98, 0.80], "65-256": [1.0, 0.97, 0.66], "257+": [1.0, 0.96, 0.69]}
        for b in self.BINS:
            self.assertEqual([round(v, 6) for v in by_gid(self.ax, f"rank_{b}")[0].get_ydata()], want[b])

    def test_the_four_bins_wear_the_validated_ordinal_ramp_and_every_bin_is_named_in_the_legend_and_at_the_line_end(self):
        tok = ff.theme_tokens("light")["ordinal4"]
        self.assertEqual([to_hex(by_gid(self.ax, f"rank_{b}")[0].get_color()) for b in self.BINS], tok)
        self.assertEqual(len(self.ax.get_legend().get_texts()), 4)
        self.assertEqual(sorted(t.get_text() for t in by_gid(self.ax, "end_label")), sorted(["ranks 1-16", "ranks 17-64", "ranks 65-256", "ranks 257+"]))

    def test_intervals_are_a_wash_spanning_the_wilson_bounds_and_the_axis_leaves_room_below_them(self):
        want = {"1-16": (0.80, 1.0), "17-64": (0.75, 1.0), "65-256": (0.61, 1.0), "257+": (0.64, 1.0)}                # INT4 lower bound and the full-precision anchor; upper bounds are capped at 1
        for b in self.BINS:
            fill = by_gid(self.ax, f"band_{b}"); self.assertEqual(len(fill), 1)
            ys = fill[0].get_paths()[0].vertices[:, 1]
            self.assertAlmostEqual(float(ys.min()), want[b][0], places=6); self.assertAlmostEqual(float(ys.max()), want[b][1], places=6)
            self.assertTrue(any(abs(float(y) - {"1-16": 0.90, "17-64": 0.85, "65-256": 0.71, "257+": 0.74}[b]) < 1e-6 for y in ys), b)    # the upper Wilson bound at INT4 is on the boundary
        self.assertAlmostEqual(self.ax.get_ylim()[0], 0.61 - 0.04, places=6); self.assertAlmostEqual(self.ax.get_ylim()[1], 1.02, places=6)

    def test_an_empty_bin_is_a_gap_not_a_zero(self):
        rows = [r for r in planted_rank_rows() if not (r["condition"] == "INT4" and r["bin"] == "257+")]
        rows.append({"condition": "INT4", "bin": "257+", "n_established": 0, "n_retained": 0, "fraction": None, "lo": None, "hi": None, "n_languages": 0})
        y = list(by_gid(ff.fig_rank_bins(rows, "light", note="x").axes[0], "rank_257+")[0].get_ydata())
        self.assertEqual(y[:2], [1.0, 0.96]); self.assertTrue(np.isnan(y[2]))
        ax = ff.fig_rank_bins(rows, "light", note="x").axes[0]
        lab = [x for x in by_gid(ax, "end_label") if x.get_text() == "ranks 257+"][0]
        self.assertEqual((round(lab.get_position()[0], 6), lab.get_position()[1]), (1.12, 0.96))                  # at the last measured point, not at the empty one

    def test_the_title_and_text_are_inside_in_both_themes_and_no_rows_is_an_error(self):
        for theme in ("light", "dark"):
            assert_text_inside(self, ff.fig_rank_bins(planted_rank_rows(), theme, note="evaluation split, frozen settings, confirmatory analysis of the pre-registered kind"))
        with self.assertRaises(ValueError):
            ff.fig_rank_bins([], "light", note="x")
        with self.assertRaises(ValueError):
            ff.fig_rank_bins([dict(r, condition="INT0") for r in planted_rank_rows()], "light", note="x")      # INT0 (the identity control) is not on the ladder


NAN = float("nan")


def planted_calibration(chance=0.25, margin=0.05):
    """A calibration record as phase1_func_mc_calibration.py writes it: per language and format {acc, lo, hi, n}; the chosen format is cloze; `ddd` has no cloze items (n = 0, NaN)."""
    def cell(acc, lo, hi, n):
        return {"acc": acc, "lo": lo, "hi": hi, "n": n}
    langs = {"aaa": {"letter": cell(0.40, 0.34, 0.46, 250), "cloze": cell(0.60, 0.51, 0.69, 120)},
             "bbb": {"letter": cell(0.22, 0.17, 0.28, 250), "cloze": cell(0.28, 0.20, 0.37, 120)},
             "ccc": {"letter": cell(0.33, 0.27, 0.39, 250), "cloze": cell(0.45, 0.36, 0.54, 120)},
             "ddd": {"letter": cell(0.30, 0.24, 0.37, 250), "cloze": cell(NAN, 0.0, 1.0, 0)},
             "eee": {"letter": cell(0.36, 0.30, 0.42, 250), "cloze": cell(0.34, 0.26, 0.43, 120)}}                  # accuracy above the threshold, lower bound below it: not covered
    thr = chance + margin
    for c, row in langs.items():
        row["covered"] = bool(row["cloze"]["lo"] > thr)
    return {"format": "cloze", "chance": chance, "margin": margin, "languages": langs, "covered": [c for c in langs if langs[c]["covered"]], "uncovered": [c for c in langs if not langs[c]["covered"]]}


@unittest.skipUnless(HAVE_MPL, "matplotlib not installed")
class Coverage(unittest.TestCase):
    def setUp(self):
        self.cal = planted_calibration()
        self.fig = ff.fig_coverage(self.cal, "light", note="calibration split, a pipeline check")
        self.ax = self.fig.axes[0]

    def test_languages_with_items_are_sorted_by_accuracy_highest_on_top_and_languages_without_items_are_not_drawn(self):
        self.assertEqual([t.get_text() for t in self.ax.get_yticklabels()], ["bbb", "eee", "ccc", "aaa"])
        self.assertNotIn("ddd", [t.get_text() for t in self.ax.get_yticklabels()])

    def test_the_plotted_numbers_are_the_chosen_formats_accuracies_and_intervals(self):
        cov = by_gid(self.ax, "coverage_covered")[0].get_offsets().tolist(); unc = by_gid(self.ax, "coverage_uncovered")[0].get_offsets().tolist()
        self.assertEqual(sorted(map(tuple, cov)), [(0.45, 2.0), (0.60, 3.0)]); self.assertEqual(sorted(tuple(u) for u in unc), [(0.28, 0.0), (0.34, 1.0)])      # not the letter numbers (0.33, 0.40, 0.22, 0.36)
        segs = sorted((s.tolist()[0][0], s.tolist()[1][0]) for g in ("interval_covered", "interval_uncovered") for c in by_gid(self.ax, g) for s in c.get_segments())
        self.assertEqual(segs, [(0.20, 0.37), (0.26, 0.43), (0.36, 0.54), (0.51, 0.69)])

    def test_covered_and_uncovered_wear_two_encodings_the_accent_circle_and_the_gray_square(self):
        tok = ff.theme_tokens("light")
        self.assertEqual(to_hex(by_gid(self.ax, "coverage_covered")[0].get_facecolor()[0][:3]), tok["series"][0])
        self.assertEqual(to_hex(by_gid(self.ax, "coverage_uncovered")[0].get_facecolor()[0][:3]), tok["muted"])
        self.assertNotEqual(by_gid(self.ax, "coverage_covered")[0].get_paths()[0].vertices.tolist(), by_gid(self.ax, "coverage_uncovered")[0].get_paths()[0].vertices.tolist())   # different marker shapes
        self.assertEqual(len(self.ax.get_legend().get_texts()), 2)

    def test_the_threshold_is_chance_plus_margin_read_from_the_record(self):
        thr = by_gid(self.ax, "threshold")[0]; ch = by_gid(self.ax, "chance")[0]
        self.assertEqual((list(thr.get_xdata())[0], list(ch.get_xdata())[0]), (0.30, 0.25))
        other = planted_calibration(chance=0.20, margin=0.10)
        ax = ff.fig_coverage(other, "light", note="x").axes[0]
        self.assertEqual((round(list(by_gid(ax, "threshold")[0].get_xdata())[0], 9), list(by_gid(ax, "chance")[0].get_xdata())[0]), (0.30, 0.20))
        other2 = planted_calibration(chance=0.25, margin=0.0)
        self.assertEqual(list(by_gid(ff.fig_coverage(other2, "light", note="x").axes[0], "threshold")[0].get_xdata())[0], 0.25)

    def test_the_subtitle_states_the_format_the_covered_count_and_the_languages_without_items(self):
        txt = all_text(self.fig)
        self.assertIn("cloze", txt); self.assertIn("2 of 4 languages covered", txt); self.assertIn("1 language without items", txt); self.assertIn("calibration split, a pipeline check", txt)

    def test_a_record_that_contradicts_its_own_rule_is_refused(self):
        bad = planted_calibration(); bad["languages"]["bbb"]["covered"] = True; bad["covered"] = ["aaa", "bbb", "ccc"]; bad["uncovered"] = []
        with self.assertRaises(ValueError):
            ff.fig_coverage(bad, "light", note="x")
        bad2 = planted_calibration(); bad2["covered"] = ["aaa"]                                              # ccc is covered by its flag and by the rule but missing from the list
        with self.assertRaises(ValueError):
            ff.fig_coverage(bad2, "light", note="x")                                                          # ccc is covered by the rule and by its flag but missing from the list

    def test_a_wrong_per_language_flag_is_refused_even_when_the_covered_list_agrees_with_the_rule(self):
        bad = planted_calibration(); bad["languages"]["aaa"]["covered"] = False                               # aaa is covered by the rule and in the list; only its own flag is wrong
        with self.assertRaises(ValueError) as cm:
            ff.fig_coverage(bad, "light", note="x")
        self.assertIn("aaa", str(cm.exception))

    def test_a_record_without_any_language_with_items_is_an_error(self):
        empty = planted_calibration()
        for row in empty["languages"].values():
            row["cloze"] = {"acc": NAN, "lo": 0.0, "hi": 1.0, "n": 0}; row["covered"] = False
        empty["covered"] = []; empty["uncovered"] = list(empty["languages"])
        with self.assertRaises(ValueError):
            ff.fig_coverage(empty, "light", note="x")

    def test_the_letter_format_is_plotted_when_it_was_chosen(self):
        rec = planted_calibration(); rec["format"] = "letter"
        for row in rec["languages"].values():
            row["covered"] = bool(row["letter"]["lo"] > 0.30)
        rec["covered"] = [c for c, r in rec["languages"].items() if r["covered"]]; rec["uncovered"] = [c for c, r in rec["languages"].items() if not r["covered"]]
        ax = ff.fig_coverage(rec, "light", note="x").axes[0]
        self.assertEqual(sorted(t.get_text() for t in ax.get_yticklabels()), ["aaa", "bbb", "ccc", "ddd", "eee"])

    def test_no_text_is_clipped_in_either_theme(self):
        for theme in ("light", "dark"):
            assert_text_inside(self, ff.fig_coverage(self.cal, theme, note="calibration split, dev-origin passages, a pipeline check and the basis of the coverage rule, not a result"))

    def test_the_table_twin_lists_every_language_and_leaves_missing_numbers_empty(self):
        rows = {r["language"]: r for r in ff.coverage_table(self.cal)}
        self.assertEqual(list(rows), ["aaa", "bbb", "ccc", "ddd", "eee"])
        self.assertEqual((rows["aaa"]["items"], rows["aaa"]["accuracy"], rows["aaa"]["lo"], rows["aaa"]["covered"]), (120, 0.60, 0.51, True))
        self.assertEqual((rows["ddd"]["items"], rows["ddd"]["accuracy"], rows["ddd"]["lo"], rows["ddd"]["covered"]), (0, None, None, False))


def planted_fidelity():
    return [{"precision": "FP", "language": "aaa", "fvu": 0.1}, {"precision": "FP", "language": "bbb", "fvu": 0.3}, {"precision": "FP", "language": "ccc", "fvu": 0.2},
            {"precision": "INT4", "language": "aaa", "fvu": 0.2}, {"precision": "INT4", "language": "bbb", "fvu": 0.4}]


@unittest.skipUnless(HAVE_MPL, "matplotlib not installed")
class Fidelity(unittest.TestCase):
    def setUp(self):
        self.fig = ff.fig_fidelity(planted_fidelity(), "light", note="pilot, exploratory")
        self.ax = self.fig.axes[0]

    def test_rows_are_languages_sorted_by_full_precision_fvu_and_missing_cells_are_masked(self):
        self.assertEqual([t.get_text() for t in self.ax.get_yticklabels()], ["aaa", "ccc", "bbb"])
        self.assertEqual([t.get_text() for t in self.ax.get_xticklabels()], ["FP", "INT4"])
        arr = self.ax.images[0].get_array()
        self.assertEqual(arr.shape, (3, 2)); self.assertTrue(bool(arr.mask[1, 1])); self.assertEqual(float(arr[2, 1]), 0.4); self.assertEqual(float(arr[0, 0]), 0.1)

    def test_colorbar_names_the_quantity_and_follows_the_sequential_ramp(self):
        self.assertEqual(len(self.fig.axes), 2)
        self.assertIn("variance unexplained", self.fig.axes[1].get_ylabel())
        self.assertEqual(self.ax.images[0].cmap(0.0), matplotlib.colors.to_rgba(ff.theme_tokens("light")["ramp"][0]))
        self.assertEqual(self.ax.images[0].cmap(1.0), matplotlib.colors.to_rgba(ff.theme_tokens("light")["ramp"][-1]))

    def test_blank_cells_wear_the_surface_colour_in_both_themes(self):
        for theme in ("light", "dark"):
            im = ff.fig_fidelity(planted_fidelity(), theme, note="x").axes[0].images[0]
            self.assertEqual(matplotlib.colors.to_hex(im.cmap.get_bad()), ff.theme_tokens(theme)["surface"])

    def test_missing_cells_are_reported_not_zeroed(self):
        self.assertIn("1 cell", all_text(self.fig)); self.assertIn("not measured", all_text(self.fig))

    def test_no_rows_is_an_error_and_text_is_inside(self):
        with self.assertRaises(ValueError):
            ff.fig_fidelity([], "light", note="x")
        for theme in ("light", "dark"):
            assert_text_inside(self, ff.fig_fidelity(planted_fidelity(), theme, note="pilot, exploratory"))


@unittest.skipUnless(HAVE_MPL, "matplotlib not installed")
class LongNotes(unittest.TestCase):
    NOTE = "evaluation split of 300 held-out devtest sentences per language, frozen settings (freeze of 2026-10-22), confirmatory analysis"

    def builders(self):
        return {"effect": lambda th: ff.fig_effect_per_language(EffectPerLanguage.ROWS, th, self.NOTE, K="K64"), "retention": lambda th: ff.fig_retention(planted_retention(), (0.75, 1.33), th, self.NOTE),
                "size": lambda th: ff.fig_size_curves(planted_families(), th, self.NOTE), "assoc": lambda th: ff.fig_assoc_vs_function(ASSOC, planted_retention(), th, self.NOTE),
                "fidelity": lambda th: ff.fig_fidelity(planted_fidelity(), th, self.NOTE), "rank": lambda th: ff.fig_rank_bins(planted_rank_rows(), th, self.NOTE),
                "coverage": lambda th: ff.fig_coverage(planted_calibration(), th, self.NOTE)}

    def test_every_figure_wraps_a_long_status_note_instead_of_clipping_it(self):
        for name, make in self.builders().items():
            for theme in ("light", "dark"):
                with self.subTest(figure=name, theme=theme):
                    fig = make(theme); assert_text_inside(self, fig)
                    self.assertIn("confirmatory analysis", all_text(fig).replace("\n", " "))

    def test_titles_describe_what_is_plotted_and_never_state_a_result(self):
        # the prose and the claims are the team's: a title that says "lost", "is not", "amplified" would assert a finding from exploratory data
        banned = ("lost", " not ", "amplif", "attenuat", "forget", "fails", "proves", "confirm")
        for name, make in self.builders().items():
            title = make("light").axes[0].get_title(loc="left").lower()
            self.assertTrue(title.strip(), name)
            for w in banned:
                self.assertNotIn(w, f" {title} ", f"{name}: {title}")

    def test_the_association_series_is_named_as_a_macro_average_like_the_function_series(self):
        legend = " | ".join(x.get_text() for x in ff.fig_assoc_vs_function(ASSOC, planted_retention(), "light", note="x").axes[0].get_legend().get_texts())
        self.assertIn("macro over languages", legend)

    def test_the_subtitle_and_the_title_do_not_overlap(self):
        for name, make in self.builders().items():
            with self.subTest(figure=name):
                fig = make("light"); fig.canvas.draw(); r = fig.canvas.get_renderer(); ax = fig.axes[0]
                title = ax._left_title.get_window_extent(r); sub = by_gid(ax, "subtitle")[0].get_window_extent(r)
                self.assertGreaterEqual(title.y0, sub.y1 - 1)


@unittest.skipUnless(HAVE_MPL, "matplotlib not installed")
class AssociationLoader(unittest.TestCase):
    HEADER = "group,condition,language,n_selective,fp_established,retained,lost,retention_fraction,retention_lo,retention_hi,loss_fraction,loss_lo,loss_hi,conditional_fraction,n_languages\n"

    def write(self, rows):
        import tempfile
        d = Path(tempfile.mkdtemp()); f = d / "language_summary.csv"
        f.write_text(self.HEADER + "".join(f"{g},{c},{l},{ns},{fe},{re},{ns - re},0,0,0,0,0,0,0,\n" for g, c, l, ns, fe, re in rows))
        return f

    def test_macro_is_the_mean_of_per_language_fractions_and_pooled_is_the_ratio_of_sums(self):
        f = self.write([("selective", "INT4", "aaa", 100, 100, 50), ("selective", "INT4", "bbb", 10, 10, 10), ("selective", "INT8", "aaa", 100, 100, 100), ("selective", "INT8", "bbb", 10, 10, 9)])
        a = ff.load_association(f)
        self.assertAlmostEqual(a["INT4"]["macro"], (0.5 + 1.0) / 2); self.assertAlmostEqual(a["INT4"]["pooled"], 60 / 110); self.assertEqual(a["INT4"]["n_languages"], 2)
        self.assertAlmostEqual(a["INT8"]["macro"], (1.0 + 0.9) / 2)

    def test_the_aggregate_rows_of_the_real_file_are_not_languages(self):
        # the D-035 language_summary.csv carries POOLED and MACRO_LANGUAGE_MEAN rows next to the 27 languages (all capitals); counting them would bias the macro and double the pooled sums
        f = self.write([("selective", "INT4", "aaa", 100, 100, 50), ("selective", "INT4", "bbb", 10, 10, 10), ("selective", "INT4", "POOLED", 110, 110, 60), ("selective", "INT4", "MACRO_LANGUAGE_MEAN", 10, 10, 8)])
        a = ff.load_association(f)["INT4"]
        self.assertEqual(a["n_languages"], 2); self.assertAlmostEqual(a["macro"], (0.5 + 1.0) / 2); self.assertAlmostEqual(a["pooled"], 60 / 110)
        self.assertEqual(ff.load_association(f)["INT4"]["n_languages"], 2)

    def test_the_real_file_has_27_languages_and_its_pooled_row_is_reproduced(self):
        root = Path(__file__).resolve().parents[3]; f = root / "reports/association_retention/results/eq/language_summary.csv"
        if not f.exists():
            self.skipTest("D-035 summary not present")
        a = ff.load_association(f)
        self.assertEqual({v["n_languages"] for v in a.values()}, {27})
        self.assertAlmostEqual(a["INT4"]["pooled"], 4718 / 6663, places=6)                                 # the headline of association_retention.md: 0.708

    def test_macro_is_the_mean_not_the_median(self):
        f = self.write([("selective", "INT4", "aaa", 100, 100, 50), ("selective", "INT4", "bbb", 10, 10, 10), ("selective", "INT4", "ccc", 10, 10, 10)])
        self.assertAlmostEqual(ff.load_association(f)["INT4"]["macro"], (0.5 + 1.0 + 1.0) / 3)

    def test_only_the_selective_group_counts_and_languages_without_an_established_feature_are_skipped(self):
        f = self.write([("selective", "INT4", "aaa", 100, 100, 80), ("selective", "INT4", "bbb", 10, 0, 0), ("controls", "INT4", "aaa", 100, 100, 10)])
        a = ff.load_association(f)
        self.assertAlmostEqual(a["INT4"]["macro"], 0.8); self.assertEqual(a["INT4"]["n_languages"], 1); self.assertAlmostEqual(a["INT4"]["pooled"], 0.8)


def planted_analysis():
    return {"K": "K64", "bands": [0.75, 1.33], "n_boot": 10, "seed": 0, "retention": planted_retention(),
            "macro": {"FP": {"point": 0.7, "lo": 0.6, "hi": 0.8, "median": 0.3, "n_positive": 3}, "INT8": {"point": 0.71, "lo": 0.6, "hi": 0.8, "median": 0.31, "n_positive": 3},
                      "INT4": {"point": 0.93, "lo": 0.8, "hi": 1.0, "median": 0.46, "n_positive": 3}},
            "fidelity": planted_fidelity()}


def planted_manifest():
    return {"sizes": {"aaa": {"K16": 16, "K64": 64, "all": 120}, "bbb": {"K16": 16, "K64": 64, "all": 73}},
            "controls": {"aaa": {"K16": [[1]] * 5, "K64": [[1]] * 5, "all": [[1]] * 3}, "bbb": {"K16": [[1]] * 5, "K64": [[1]] * 5, "all": [[1]] * 3}}}


@unittest.skipUnless(HAVE_MPL, "matplotlib not installed")
class Tables(unittest.TestCase):
    def test_retention_table_has_one_row_per_precision_with_class_counts_and_the_d035_numbers(self):
        rows = {r["precision"]: r for r in ff.retention_table(planted_analysis(), ASSOC)}
        self.assertEqual(list(rows), ["FP", "INT8", "INT4"])
        r = rows["INT4"]
        self.assertEqual((r["mean_E"], r["median_E"], r["macro_ratio"], r["macro_lo"], r["macro_hi"], r["median_ratio"]), (0.93, 0.46, 2.14, 2.02, 2.55, 1.43))
        self.assertEqual((r["n_amplified"], r["n_attenuated"], r["n_preserved"], r["n_undefined"]), (1, 1, 0, 1))
        self.assertEqual((r["assoc_macro"], r["assoc_pooled"]), (0.70, 0.71))
        self.assertIsNone(rows["FP"]["macro_ratio"]); self.assertEqual(rows["FP"]["mean_E"], 0.7)

    def test_effects_table_joins_sizes_effects_and_ratios_per_language(self):
        eff = [{"language": "aaa", "effect": 0.3, "lo": 0.2, "hi": 0.4, "harm_control": 0.02}, {"language": "bbb", "effect": 7.0, "lo": 6.0, "hi": 8.0, "harm_control": 0.05}]
        rows = {r["language"]: r for r in ff.effects_table(eff, planted_analysis(), planted_manifest())}
        self.assertEqual((rows["aaa"]["size_all"], rows["aaa"]["size_K64"], rows["bbb"]["size_all"]), (120, 64, 73))
        self.assertEqual((rows["bbb"]["effect"], rows["bbb"]["harm_control"]), (7.0, 0.05))
        self.assertEqual((rows["aaa"]["ratio_INT4"], rows["aaa"]["ratio_INT8"]), (3.0, 1.02))

    def test_config_table_counts_control_draws(self):
        rows = {r["language"]: r for r in ff.config_table(planted_manifest())}
        self.assertEqual((rows["aaa"]["size_K16"], rows["aaa"]["draws_K16"], rows["aaa"]["draws_all"]), (16, 5, 3))

    def test_markdown_renders_headers_formats_and_marks_missing_values(self):
        md = ff.to_markdown([{"a": 1.23456, "b": None, "c": 3}], [("a", "alpha", "{:.2f}"), ("b", "beta", "{:.2f}"), ("c", "gamma", "{}")])
        self.assertEqual(md.splitlines(), ["| alpha | beta | gamma |", "|---|---|---|", "| 1.23 | - | 3 |"])

    def test_csv_writes_missing_values_as_empty_fields_and_the_header_from_the_columns(self):
        import tempfile
        f = Path(tempfile.mkdtemp()) / "t.csv"
        ff.write_csv([{"a": 1.5, "b": None}], [("a", "alpha", "{}"), ("b", "beta", "{}")], f)
        self.assertEqual(f.read_text().splitlines(), ["a,b", "1.5,"])

    def test_undefined_ratios_are_written_as_missing_not_zero(self):
        rows = {r["language"]: r for r in ff.effects_table([{"language": "vie", "effect": 0.005, "lo": -0.01, "hi": 0.02, "harm_control": 0.01}], planted_analysis(), {"sizes": {"vie": {"K16": 1, "K64": 2, "all": 3}}, "controls": {}})}
        self.assertIsNone(rows["vie"]["ratio_INT8"])


@unittest.skipUnless(HAVE_MPL, "matplotlib not installed")
class Main(unittest.TestCase):
    def setUp(self):
        import csv, json, tempfile
        self.d = Path(tempfile.mkdtemp()); self.res = self.d / "res"; self.res.mkdir(); self.out = self.d / "out"
        (self.res / "analysis.json").write_text(json.dumps(planted_analysis()))
        with open(self.res / "effects_by_language.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["precision", "language", "effect", "lo", "hi", "harm_group", "harm_control", "norm_ratio_group", "norm_ratio_control"])
            for l, e in (("aaa", 0.3), ("bbb", 7.0)):
                w.writerow(["FP", l, e, e - 0.1, e + 0.1, e + 0.02, 0.02, 0.1, 0.1]); w.writerow(["INT4", l, e * 2, e, e * 3, e * 2 + 0.02, 0.03, 0.1, 0.1])
        with open(self.res / "families.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["precision", "language", "family", "harm", "lo", "hi"])
            for r in planted_families():
                w.writerow([r["precision"], r["language"], r["family"], r["harm"], r["lo"], r["hi"]])
        self.assoc = self.d / "language_summary.csv"
        self.assoc.write_text(AssociationLoader.HEADER + "selective,INT4,aaa,100,100,70,30,0,0,0,0,0,0,0,\nselective,INT8,aaa,100,100,98,2,0,0,0,0,0,0,0,\n")
        import json as _j
        self.man = self.d / "groups_manifest.json"; self.man.write_text(_j.dumps(planted_manifest()))
        self.cal = self.d / "belebele_calibration.json"; self.cal.write_text(json.dumps(planted_calibration()))
        self.rank = self.d / "rank_table.csv"
        with open(self.rank, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["condition", "bin", "n_established", "n_retained", "fraction", "lo", "hi", "n_languages"])
            for r in planted_rank_rows():
                w.writerow([r["condition"], r["bin"], r["n_established"], r["n_retained"], r["fraction"], r["lo"], r["hi"], r["n_languages"]])

    def run_main(self, *extra):
        return ff.main(["--results", str(self.res), "--out", str(self.out), "--note", "pilot, exploratory", "--assoc", str(self.assoc), "--manifest", str(self.man), "--rank-table", str(self.rank), "--calibration", str(self.cal), *extra])

    def test_writes_every_figure_in_both_themes_and_formats_and_the_tables(self):
        written = [Path(p).name for p in self.run_main()]
        for stem in ("fig1_effect_per_language", "fig2_retention", "fig3_group_size", "fig5_association_vs_function", "fig6_sae_fidelity", "fig7_belebele_coverage", "fig8_association_by_rank"):
            for theme in ("light", "dark"):
                for ext in ("svg", "png"):
                    self.assertIn(f"{stem}_{theme}.{ext}", written); self.assertTrue((self.out / f"{stem}_{theme}.{ext}").stat().st_size > 1000)
        for name in ("table_retention", "table_effects", "table_configurations", "table_coverage"):
            self.assertIn(f"{name}.csv", written); self.assertIn(f"{name}.md", written)

    def test_svg_keeps_text_as_text_and_png_is_a_png(self):
        self.run_main()
        svg = (self.out / "fig2_retention_light.svg").read_text()
        self.assertIn("<text", svg); self.assertIn("macro mean of ratios", svg)
        self.assertEqual((self.out / "fig2_retention_light.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_png_resolution_is_200_dpi(self):
        self.run_main()
        png = (self.out / "fig2_retention_light.png").read_bytes()
        self.assertEqual(int.from_bytes(png[16:20], "big"), round(7.6 * 200))

    def test_without_the_association_file_the_other_figures_and_tables_are_still_written(self):
        self.assoc.unlink()
        written = [Path(p).name for p in self.run_main()]
        self.assertNotIn("fig5_association_vs_function_light.png", written); self.assertIn("fig2_retention_light.png", written); self.assertIn("table_retention.md", written)

    def test_tables_contain_the_numbers(self):
        self.run_main()
        md = (self.out / "table_retention.md").read_text()
        self.assertIn("INT4", md); self.assertIn("2.14", md)
        csv_text = (self.out / "table_effects.csv").read_text().splitlines()
        self.assertTrue(csv_text[0].startswith("language,")); self.assertEqual(len(csv_text), 3)

    def test_regenerating_gives_byte_identical_files(self):
        import shutil, time
        self.run_main(); first = {p.name: p.read_bytes() for p in self.out.iterdir()}
        shutil.rmtree(self.out)
        time.sleep(1.1)                                                                                      # a date stamp inside a file would differ by now
        self.run_main(); second = {p.name: p.read_bytes() for p in self.out.iterdir()}
        self.assertEqual(sorted(first), sorted(second))
        for name in first:
            self.assertEqual(first[name], second[name], name)

    def test_a_results_folder_without_families_and_fidelity_still_gives_the_other_figures_and_the_unit_is_passed_on(self):
        (self.res / "families.csv").unlink()
        an = json.loads((self.res / "analysis.json").read_text()); del an["fidelity"]; (self.res / "analysis.json").write_text(json.dumps(an))
        written = [Path(p).name for p in self.run_main("--unit", "nats per item")]
        for name in ("fig1_effect_per_language_light.png", "fig2_retention_light.png", "fig5_association_vs_function_light.png", "table_retention.md"):
            self.assertIn(name, written)
        for name in ("fig3_group_size_light.png", "fig6_sae_fidelity_light.png"):
            self.assertNotIn(name, written)
        self.assertIn("nats per item", (self.out / "fig1_effect_per_language_light.svg").read_text())

    def test_figure_8_carries_its_own_status_note_not_the_one_of_the_other_figures(self):
        self.run_main()
        svg = (self.out / "fig8_association_by_rank_light.svg").read_text()
        self.assertIn("descriptive", svg); self.assertNotIn("pilot, exploratory", svg)
        self.run_main("--rank-note", "customnote8")                                                          # one word: the wrapped subtitle may break a longer phrase over two lines
        self.assertIn("customnote8", (self.out / "fig8_association_by_rank_light.svg").read_text())

    def test_figure_7_carries_the_calibration_note_and_is_skipped_without_the_record(self):
        self.run_main()
        svg = (self.out / "fig7_belebele_coverage_light.svg").read_text()
        self.assertIn("calibration", svg); self.assertNotIn("pilot, exploratory", svg)
        self.run_main("--calibration-note", "customnote7"); self.assertIn("customnote7", (self.out / "fig7_belebele_coverage_light.svg").read_text())
        self.cal.unlink()
        written = [Path(p).name for p in self.run_main()]
        self.assertNotIn("fig7_belebele_coverage_light.png", written); self.assertNotIn("table_coverage.md", written); self.assertIn("fig2_retention_light.png", written)

    def test_without_a_rank_table_figure_8_is_skipped(self):
        self.rank.unlink()
        written = [Path(p).name for p in self.run_main()]
        self.assertNotIn("fig8_association_by_rank_light.png", written); self.assertIn("fig2_retention_light.png", written)

    def test_themes_and_formats_can_be_restricted(self):
        written = [Path(p).name for p in self.run_main("--themes", "dark", "--formats", "png")]
        self.assertIn("fig1_effect_per_language_dark.png", written)
        self.assertNotIn("fig1_effect_per_language_light.png", written); self.assertFalse(any(n.endswith(".svg") for n in written))


if __name__ == "__main__":
    unittest.main()
