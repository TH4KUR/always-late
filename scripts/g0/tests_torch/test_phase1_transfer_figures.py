import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_transfer_figures as tf
import phase1_transfer_report as rep
from test_phase1_gain_figures import overlaps, texts_inside

STATES = ("INT6", "INT4")
KINDS = ("real", "matched_swapped", "random1", "random2", "random3")
LANGS = [f"l{i:02d}" for i in range(27)]


def planted(root, insufficient=("INT4",)):
    """The tables of a report run with a known structure (the file names and columns of phase1_transfer_report.write_outputs)."""
    root = Path(root); rng = np.random.default_rng(0)
    for s in STATES:
        rows = []
        for f in range(8):
            for j in range(5):
                for k in KINDS:
                    drift = float(rng.uniform(0.0, 1.4)); rows.append({"feature": f, "sentence": 1000 * f + j, "kind": k, "split": "calibration" if f % 2 == 0 else "heldout", "eligible": (f + j) % 2 == 0,
                                                                        "normalized_error": float(0.2 + abs(1 - drift) * 0.8 + rng.normal(0, 0.05)), "amp_fp_mean": 4.0, "amp_q_mean": 4.0 * drift, "act_cos": float(rng.uniform(0, 1))})
        rep.write_csv(root / f"per_context_{s}.csv", rows)
        rep.write_csv(root / f"averaged_features_{s}.csv", [{"feature": f, "kind": k, "split": "calibration", "n_contexts": 5, "pooled_cos": (None if (f, k) == (0, "real") else float(rng.uniform(0, 1))), "item_cos": float(rng.uniform(0, 1)),
                                                              "cos_gap": 0.1, "coherence_fp": 0.5, "coherence_qq": 0.5} for f in range(8) for k in KINDS])
    comp = []
    for s in STATES:
        for k in KINDS:
            for m, base in (("C_rel", 0.6), ("W_rel", 0.3), ("I_rel", 0.2)):
                v = base * (1.0 if k == "real" else 0.5) * (1.4 if s == "INT4" else 1.0); comp.append({"state": s, "kind": k, "split": "all", "metric": m, "point": v, "low": v * 0.8, "high": v * 1.2, "n": 20, "n_items": 100})
    rep.write_csv(root / "component_summary.csv", comp)
    rep.write_csv(root / "group_per_language.csv", [{"state": s, "language": l, "C": (-1) ** i * (0.5 + 0.02 * i), "W": 0.2 + 0.01 * i, "I": 0.1, "total": 0.5, "n_sentences": 300, "n_sentences_total": 300} for s in ("INT16",) + STATES for i, l in enumerate(LANGS)])
    rep.write_csv(root / "a3.csv", [{"state": s, "n_languages": 27, "c_gt_w": 20, "median_abs_c": 0.7, "median_abs_w": 0.3, "median_difference": 0.4, "holds": True, "median_difference_ci": (0.2, 0.6), "share_holds": 0.99} for s in ("INT16",) + STATES])
    pv = []
    for s in STATES:
        base = {"state": s, "variant": "eligible", "n_design": 300, "n_dropped": 0, "n_calibration": 160, "n_heldout": 140}
        if s in insufficient:
            pv.append({**base, "status": "insufficient"}); continue
        for name, mae in (("A", 0.40), ("A+G", 0.39), ("A+B", 0.30), ("A+B+G", 0.29)):
            pv.append({**base, "status": "fitted", "model": name, "lambda": 1, "cv_mae": mae, "heldout_mae": mae, "null_mae": 0.5, "strongest_baseline": "A+G", "improvement_diagnostic": 0.23 if name == "A+B" else None,
                       "improvement_low": 0.12 if name == "A+B" else None, "improvement_high": 0.31 if name == "A+B" else None, "criterion_met": True if name == "A+B" else None})
    rep.write_csv(root / "predictive_validation.csv", pv)


def panel(fig, gid):
    return [a for a in fig.axes if a.get_gid() == gid][0]


def collection(ax, gid):
    return [c for c in ax.collections if c.get_gid() == gid][0]


class Loader(unittest.TestCase):
    def test_values_are_converted_blanks_are_none_and_pairs_become_tuples(self):
        with tempfile.TemporaryDirectory() as d:
            planted(d); t = tf.load_tables(d)
            r = t["per_context"]["INT6"][0]; self.assertIsInstance(r["feature"], int); self.assertIsInstance(r["normalized_error"], float); self.assertIsInstance(r["eligible"], bool); self.assertEqual(r["kind"], "real")
            self.assertIsNone([x for x in t["averaged"]["INT6"] if x["feature"] == 0 and x["kind"] == "real"][0]["pooled_cos"]); self.assertEqual(t["a3"][1]["median_difference_ci"], (0.2, 0.6))
            self.assertEqual({r["state"] for r in t["components"]}, set(STATES)); self.assertEqual(len(t["group"]), 3 * 27); self.assertEqual({r["status"] for r in t["predictive"]}, {"fitted", "insufficient"})

    def test_a_missing_table_is_an_error_that_names_it(self):
        with tempfile.TemporaryDirectory() as d:
            planted(d); (Path(d) / "a3.csv").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "a3.csv"):
                tf.load_tables(d)


class Figures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); planted(cls.tmp.name); cls.t = tf.load_tables(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_figure_1_draws_the_eligible_real_items_with_their_activation_ratio_and_error(self):
        fig = tf.fig_activation_effect(self.t, "light", "exploratory")
        for s in STATES:
            rows = [r for r in self.t["per_context"][s] if r["kind"] == "real" and r["eligible"]]; off = collection(panel(fig, f"panel:{s}"), "items").get_offsets()
            self.assertEqual(len(off), len(rows)); self.assertTrue(np.allclose(sorted(map(tuple, off)), sorted((r["amp_q_mean"] / r["amp_fp_mean"], r["normalized_error"]) for r in rows)))

    def test_figure_2_shows_the_relative_norms_of_c_w_and_i_with_their_intervals_for_every_kind_and_the_group_level_c_against_w(self):
        fig = tf.fig_crossed_contrasts(self.t, "light", "exploratory"); ax = panel(fig, "panel:components:INT4")
        for k in KINDS:
            off = collection(ax, f"est:{k}").get_offsets(); self.assertEqual(len(off), 3)
            want = [r["point"] for m in ("C_rel", "W_rel", "I_rel") for r in self.t["components"] if r["state"] == "INT4" and r["kind"] == k and r["metric"] == m]
            self.assertTrue(np.allclose(off[:, 1], want)); self.assertEqual(sorted(set(np.round(off[:, 0]))), [0.0, 1.0, 2.0])
            segs = [s for lc in ax.collections if lc.get_gid() == f"ci:{k}" for s in lc.get_segments()]; self.assertEqual(len(segs), 3); self.assertTrue(np.allclose(sorted(s[0][1] for s in segs), sorted(np.array(want) * 0.8)))
        g = panel(fig, "panel:group"); off4 = collection(g, "group:INT4").get_offsets(); self.assertEqual(len(off4), 27)
        rows = [r for r in self.t["group"] if r["state"] == "INT4"]; self.assertTrue(np.allclose(sorted(map(tuple, off4)), sorted((abs(r["W"]), abs(r["C"])) for r in rows)))
        self.assertEqual(len([c for c in g.collections if c.get_gid() in ("group:INT6", "group:INT4")]), 2); self.assertEqual([l.get_gid() for l in g.lines if l.get_gid() == "identity"], ["identity"])

    def test_figure_3_compares_the_cosine_of_the_averaged_fingerprint_with_the_mean_cosine_of_the_contexts_per_feature(self):
        fig = tf.fig_averaged_vs_context(self.t, "light", "exploratory")
        for s in STATES:
            ax = panel(fig, f"panel:{s}"); rows = [r for r in self.t["averaged"][s] if r["kind"] == "real" and r["pooled_cos"] is not None]
            self.assertEqual(len(collection(ax, "real").get_offsets()), len(rows)); self.assertTrue(np.allclose(sorted(map(tuple, collection(ax, "real").get_offsets())), sorted((r["item_cos"], r["pooled_cos"]) for r in rows)))
            ctrl = [r for r in self.t["averaged"][s] if r["kind"] != "real"]; self.assertEqual(len(collection(ax, "controls").get_offsets()), len(ctrl)); self.assertEqual([l.get_gid() for l in ax.lines if l.get_gid() == "identity"], ["identity"])

    def test_figure_4_shows_the_held_out_error_of_the_models_the_null_and_the_reduction_with_its_interval_and_says_so_when_there_were_too_few_items(self):
        fig = tf.fig_predictive(self.t, "light", "exploratory"); ax = panel(fig, "panel:mae:INT6"); off = collection(ax, "models").get_offsets()
        self.assertTrue(np.allclose(sorted(off[:, 0]), [0.29, 0.30, 0.39, 0.40])); self.assertEqual([l.get_xdata()[0] for l in ax.lines if l.get_gid() == "null"], [0.5])
        imp = panel(fig, "panel:improvement"); est = collection(imp, "reduction").get_offsets(); self.assertEqual(len(est), 1); self.assertAlmostEqual(float(est[0][0]), 0.23)
        segs = [s for lc in imp.collections if lc.get_gid() == "reduction-ci" for s in lc.get_segments()]; self.assertEqual(len(segs), 1); self.assertTrue(np.allclose([segs[0][0][0], segs[0][1][0]], [0.12, 0.31]))
        self.assertEqual(sorted(l.get_xdata()[0] for l in imp.lines if l.get_gid() in ("zero", "criterion")), [0.0, 0.10])
        self.assertIn("too few", " ".join(x.get_text() for x in panel(fig, "panel:mae:INT4").texts)); self.assertEqual(len(panel(fig, "panel:mae:INT4").collections), 0)

    def test_every_text_is_inside_the_canvas_and_the_subtitle_never_overlaps_a_panel_title_in_both_themes(self):
        for theme in ("light", "dark"):
            for make in (tf.fig_activation_effect, tf.fig_crossed_contrasts, tf.fig_averaged_vs_context, tf.fig_predictive):
                fig = make(self.t, theme, "exploratory; planted data"); self.assertEqual(texts_inside(fig), [], (theme, make.__name__)); self.assertEqual(overlaps(fig), [], (theme, make.__name__))

    def test_the_files_are_written_for_every_figure_theme_and_format_and_the_svg_is_deterministic(self):
        with tempfile.TemporaryDirectory() as d:
            paths = tf.write_figures(self.t, d, "exploratory", themes=("light", "dark"), formats=("svg",)); names = sorted(Path(p).name for p in paths)
            self.assertEqual(names, sorted(f"fig_tr{i}_{n}_{th}.svg" for i, n in ((1, "activation_effect"), (2, "crossed_contrasts"), (3, "averaged_vs_context"), (4, "predictive")) for th in ("light", "dark")))
            first = {p: Path(p).read_bytes() for p in paths}
            again = tf.write_figures(self.t, d, "exploratory", themes=("light", "dark"), formats=("svg",)); self.assertTrue(all(first[p] == Path(p).read_bytes() for p in again))


if __name__ == "__main__":
    unittest.main()
