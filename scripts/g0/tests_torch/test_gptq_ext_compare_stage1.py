"""The stage-1 comparison of the GPTQ extension (E1 ordinary loss and E2 survival, paired per language, arm against arm; freeze endpoints E1 and E2). Synthetic arms with known differences, the bridge check (FP and the legacy floors
must be identical across arms), and a check that the loader reproduces the real legacy `per_language.csv` from its `survival.pt` when the cached legacy run is present."""
import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
G0 = HERE.parent
ROOT = G0.parents[1]
sys.path.insert(0, str(G0))
from gptq_ext import compare_stage1 as cs  # noqa: E402

LANGS = ["eng_Latn", "hin_Deva", "khm_Khmr", "lao_Laoo"]
LEGACY_DIR = Path.home() / ".cache/anlp-always-late/phase1_runs/p1-mask/eq"


def make_arm(d, nll_shift=0.0, corr_shift=0.0, bits=(8, 6, 4, 16, 12, 0), n_feat=50, seed=0, fp_loss=3.0, floors_shift=0.0):
    """A survival.pt like save_outputs writes it: corr per rung [n_lang, n_feat], FP firing rate, per-language loss sums and counts."""
    g = torch.Generator().manual_seed(seed)
    n_lang = len(LANGS)
    fire = torch.rand(n_lang, n_feat, generator=g) * 0.01 + 0.0005           # most features active (> 0.001)
    base_corr = 0.85 + 0.15 * torch.rand(n_lang, n_feat, generator=g)
    corr, loss_sum, loss_n = {}, {"FP": torch.full((n_lang,), fp_loss * 1000.0, dtype=torch.float64)}, {"FP": torch.full((n_lang,), 1000, dtype=torch.long)}
    for b in bits:
        floor = b in (16, 12, 0)
        shift = floors_shift if floor else corr_shift
        c = (base_corr - (0.0 if floor else (8 - b) * 0.005) + shift).clamp(max=1.0)
        corr[b] = c
        loss_sum[b] = torch.full((n_lang,), (fp_loss + (0.0 if floor else (8 - b) * 0.05 + nll_shift)) * 1000.0, dtype=torch.float64); loss_n[b] = torch.full((n_lang,), 1000, dtype=torch.long)
    Path(d).mkdir(parents=True, exist_ok=True)
    torch.save({"languages": LANGS, "bits": list(bits), "n": [1000] * n_lang, "corr": corr, "firing_rate": fire, "mean_activation": fire, "max_activation": fire, "loss_sum": loss_sum, "loss_n": loss_n,
                "pooled_corr": {b: c.mean(0) for b, c in corr.items()}, "pooled_firing_rate": fire.mean(0)}, Path(d) / "survival.pt")
    return Path(d)


def roster_file(d):
    rows = [{"code": c, "name": c[:3], "tier": t, "joshi_class": 3, "script": "x", "script_family": "x"} for c, t in zip(LANGS, ["high", "mid", "low", "low"])]
    p = Path(d) / "roster.json"; p.write_text(json.dumps({"languages": rows})); return p


class Load(unittest.TestCase):
    def test_the_arm_table_has_one_entry_per_language_and_state(self):
        with tempfile.TemporaryDirectory() as d:
            arm = cs.load_arm(make_arm(Path(d) / "a"))
            self.assertEqual(arm["languages"], LANGS)
            self.assertEqual(sorted(arm["bits"]), [0, 4, 6, 8, 12, 16])
            self.assertAlmostEqual(arm["nll_delta"]["eng_Latn"][4], (8 - 4) * 0.05, places=9)
            self.assertAlmostEqual(arm["nll_delta"]["eng_Latn"][16], 0.0, places=9)
            self.assertTrue(0.0 <= arm["survived_pct"]["eng_Latn"][8] <= 100.0)

    def test_survival_uses_the_projects_own_definition(self):
        import phase1_run as pr
        with tempfile.TemporaryDirectory() as d:
            p = make_arm(Path(d) / "a"); arm = cs.load_arm(p); s = torch.load(p / "survival.pt", weights_only=False)
            want = pr.survival_summary(s["corr"][6][1].double(), s["firing_rate"][1].double(), 0.001)
            self.assertEqual(arm["survived_pct"]["hin_Deva"][6], want["survived_>0.9_pct"])
            self.assertEqual(arm["median_corr"]["hin_Deva"][6], want["median_corr"])
            self.assertEqual(arm["n_active"]["hin_Deva"], want["n_active_features"])

    @unittest.skipUnless((LEGACY_DIR / "survival.pt").exists(), "the cached legacy run is not on this machine")
    def test_the_loader_reproduces_the_real_legacy_per_language_table(self):
        arm = cs.load_arm(LEGACY_DIR)
        rows = list(csv.DictReader(open(LEGACY_DIR / "per_language.csv")))
        self.assertEqual(len(rows), 27 * 8)
        for r in rows:
            b = int(r["bits"]); lang = r["language"]
            self.assertAlmostEqual(arm["survived_pct"][lang][b], float(r["survived_>0.9_pct"]), places=9, msg=(lang, b))
            self.assertAlmostEqual(arm["median_corr"][lang][b], float(r["median_corr"]), places=9, msg=(lang, b))
            self.assertAlmostEqual(math.exp(arm["nll_abs"][lang][b]), float(r["perplexity"]), places=6, msg=(lang, b))


class Paired(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); d = Path(self.tmp.name)
        self.legacy = cs.load_arm(make_arm(d / "legacy", seed=1))
        self.matched = cs.load_arm(make_arm(d / "matched", seed=1, nll_shift=0.02, corr_shift=-0.01))          # same features, slightly worse
        self.gptq = cs.load_arm(make_arm(d / "gptq", seed=1, nll_shift=-0.03, corr_shift=0.02))                # same features, better
        self.roster = cs.load_roster(roster_file(d))

    def tearDown(self):
        self.tmp.cleanup()

    def test_differences_are_per_language_and_signed_arm_a_minus_arm_b(self):
        t = cs.paired_table(self.gptq, self.matched, self.roster, bits=[8, 6, 4])
        r = [x for x in t if x["language"] == "eng_Latn" and x["bits"] == 4][0]
        self.assertAlmostEqual(r["nll_delta_diff"], -0.05, places=9)                                          # (-0.03) - (+0.02)
        self.assertGreater(r["survived_pct_diff"], 0.0)
        self.assertEqual(len(t), len(LANGS) * 3)
        self.assertEqual({x["tier"] for x in t}, {"high", "mid", "low"})

    def test_summary_counts_languages_and_gives_a_bootstrap_interval_that_contains_the_mean(self):
        t = cs.paired_table(self.gptq, self.matched, self.roster, bits=[4])
        s = cs.summarize(t, "nll_delta_diff", n_boot=500, seed=0)
        self.assertEqual(s["n_languages"], 4)
        self.assertEqual(s["n_negative"], 4)
        self.assertLessEqual(s["ci_low"], s["mean"] + 1e-12); self.assertGreaterEqual(s["ci_high"], s["mean"] - 1e-12)
        self.assertAlmostEqual(s["mean"], -0.05, places=9); self.assertAlmostEqual(s["median"], -0.05, places=9)

    def test_the_interval_is_ordered_for_varying_differences_and_counts_signs_exactly(self):
        s = cs.summarize([{"x": v} for v in (-1.0, 2.0, 3.0, 0.0, 5.0, -2.0)], "x", n_boot=500, seed=1)
        self.assertLess(s["ci_low"], s["mean"]); self.assertLess(s["mean"], s["ci_high"])
        self.assertEqual((s["n_negative"], s["n_positive"], s["n_languages"]), (2, 3, 6))
        self.assertAlmostEqual(s["mean"], 7 / 6, places=9); self.assertAlmostEqual(s["median"], 1.0, places=9)

    def test_summary_is_deterministic_for_a_seed_and_by_tier(self):
        t = cs.paired_table(self.gptq, self.matched, self.roster, bits=[4])
        a, b = cs.summarize(t, "survived_pct_diff", 300, 7), cs.summarize(t, "survived_pct_diff", 300, 7)
        self.assertEqual(a, b)
        bt = cs.summarize_by_tier(t, "survived_pct_diff")
        self.assertEqual(set(bt), {"high", "mid", "low"}); self.assertEqual(bt["low"]["n_languages"], 2)

    def test_the_bridge_is_clean_for_identical_fp_and_floors_and_flags_a_difference(self):
        d = Path(self.tmp.name)
        same = cs.load_arm(make_arm(d / "same", seed=1))
        self.assertTrue(cs.bridge(self.legacy, same)["ok"])
        other_fp = cs.load_arm(make_arm(d / "o1", seed=1, fp_loss=3.0001))
        r = cs.bridge(self.legacy, other_fp)
        self.assertFalse(r["ok"]); self.assertGreater(r["fp_loss_max_abs_diff"], 0)
        other_floor = cs.load_arm(make_arm(d / "o2", seed=1, floors_shift=-0.001))
        r = cs.bridge(self.legacy, other_floor)
        self.assertFalse(r["ok"]); self.assertEqual(sorted(r["floor_corr_differs"]), [0, 12, 16])

    def test_a_changed_fp_firing_rate_alone_breaks_the_bridge(self):
        d = Path(self.tmp.name)
        other = cs.load_arm(make_arm(d / "o3", seed=1))
        other["fp_firing_rate"] = other["fp_firing_rate"] + 1e-6
        r = cs.bridge(self.legacy, other)
        self.assertFalse(r["ok"]); self.assertGreater(r["fp_firing_rate_max_abs_diff"], 0.0); self.assertEqual(r["fp_loss_max_abs_diff"], 0.0)

    def test_the_report_shows_a_broken_bridge(self):
        d = Path(self.tmp.name)
        broken = cs.load_arm(make_arm(d / "o4", seed=1, nll_shift=0.02, corr_shift=-0.01, fp_loss=3.5))
        out = d / "out2"
        b = cs.write_report(out, {"legacy RTN": self.legacy, "matched RTN": broken, "GPTQ": self.gptq}, self.roster, bits=[4], n_boot=50, seed=0)
        self.assertFalse(b["matched RTN"]["ok"]); self.assertTrue(b["GPTQ"]["ok"])
        self.assertIn("**NO**", (out / "stage1_comparison.md").read_text())
        self.assertFalse(json.loads((out / "bridge.json").read_text())["matched RTN"]["ok"])

    def test_arms_with_another_language_order_are_refused(self):
        other = dict(self.gptq); other["languages"] = list(reversed(self.gptq["languages"]))
        with self.assertRaises(ValueError):
            cs.paired_table(other, self.matched, self.roster, bits=[4])

    def test_missing_bit_width_is_reported_not_silently_skipped(self):
        d = Path(self.tmp.name)
        few = cs.load_arm(make_arm(d / "few", seed=1, bits=(8, 16, 12, 0)))
        with self.assertRaises(ValueError):
            cs.paired_table(few, self.gptq, self.roster, bits=[4])

    def test_the_report_names_arms_bits_and_all_three_contrasts(self):
        out = Path(self.tmp.name) / "out"
        cs.write_report(out, {"legacy RTN": self.legacy, "matched RTN": self.matched, "GPTQ": self.gptq}, self.roster, bits=[8, 6, 4], n_boot=200, seed=0)
        text = (out / "stage1_comparison.md").read_text()
        for needle in ("matched RTN minus legacy RTN", "GPTQ minus matched RTN", "GPTQ minus legacy RTN", "INT4", "Exploratory"):
            self.assertIn(needle, text)
        self.assertTrue((out / "stage1_paired_per_language.csv").is_file()); self.assertTrue((out / "stage1_summary.csv").is_file()); self.assertTrue((out / "bridge.json").is_file())
        rows = list(csv.DictReader(open(out / "stage1_summary.csv")))
        self.assertEqual({r["contrast"] for r in rows}, {"matched RTN minus legacy RTN", "GPTQ minus matched RTN", "GPTQ minus legacy RTN"})


if __name__ == "__main__":
    unittest.main()
