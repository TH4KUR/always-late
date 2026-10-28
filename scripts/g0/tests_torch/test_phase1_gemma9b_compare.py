"""Tasks 6 and 7 of the Gemma-2-9B three-arm plan: the analysis of the frozen predictions (spec section 5). Tier 1: Q0 to Q3 across the legacy, matched-RTN and GPTQ arms of Study G, computed with the frozen Study G
code (`phase1_llama_compare.evaluate`, `phase1_gain_analysis`). Synthetic worlds with planted harms (the helpers of the Gain and Llama tests); the group harm of language i in state s at ratio a is
E(i) * f(s, i) * depth(a) / 6, the matched-control harm a tenth of it, so the net harm at the deepest down ratio of a state is proportional to f and the INT4 over FP ratio equals f(INT4) / f(FP)."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_gain_analysis as ga
import phase1_gain_run as gr
import phase1_gemma9b_compare as gc
from test_gptq_ext_compare_stage1 import LANGS as L4, make_arm, roster_file
from test_phase1_gain_analysis import DOWN, FINE, UP, planted_run, ratios_json, save_state

ROOT = Path(__file__).resolve().parents[3]

LANGS = [f"l{i}" for i in range(27)]
E = lambda i: 0.2 + 0.01 * i


def arm(factors, langs=LANGS, shape=None, e_of=E):
    """GainRuns for one arm. factors: {state: number or callable(language index) -> factor}; shape(state, i, a) -> multiplier of the group harm (default 1)."""
    runs = {}
    for state, f in factors.items():
        fn = f if callable(f) else (lambda i, f=f: f)

        def harm(code, a, kind, state=state, fn=fn):
            i = int(code[1:]); depth = abs(np.log2(a)) if a > 0 else 6.0
            g = e_of(i) * fn(i) * depth / 6.0 * (shape(state, i, a) if shape else 1.0)
            return g if kind == "g" else 0.1 * g
        runs[state] = planted_run(state, harm, langs=langs, n_controls=3)
    return ga.GainRuns(runs, DOWN, UP, FINE, n_controls=3)


def three(legacy=1.9, matched=1.7, gptq=1.1, int6=(1.0, 1.0, 1.0), **kw):
    """Arms with the 2B pattern: legacy {FP 1, INT8 1, INT6, INT4}, the other two {FP 1, INT6, INT4}."""
    return {"legacy": arm({"FP": 1.0, "INT8": 1.0, "INT6": int6[0], "INT4": legacy}, **kw), "rtn-matched": arm({"FP": 1.0, "INT6": int6[1], "INT4": matched}, **kw),
            "gptq": arm({"FP": 1.0, "INT6": int6[2], "INT4": gptq}, **kw)}


def evaluate(arms, tested=None):
    return gc.evaluate_arms(arms, tested or LANGS, n_boot=30, n_boot_languages=400, seed=0)


class TierOne(unittest.TestCase):
    def test_world_with_amplified_legacy_and_flatter_gptq_is_consistent(self):
        r = evaluate(three())
        self.assertTrue(r["Q0"]["met"]); self.assertTrue(r["Q1"]["evaluable"]); self.assertTrue(r["Q1"]["met"]); self.assertTrue(r["Q2"]["met"])
        self.assertEqual(r["verdict_tier1"], "consistent")
        self.assertAlmostEqual(r["Q1"]["r"]["legacy"], 1.9, places=6); self.assertAlmostEqual(r["Q1"]["r"]["rtn-matched"], 1.7, places=6); self.assertAlmostEqual(r["Q1"]["r"]["gptq"], 1.1, places=6)
        self.assertEqual(r["Q1"]["count_lower"], 27); self.assertEqual(r["Q1"]["required"], 20)
        self.assertEqual(r["n_tested"], 27)

    def test_q1_fails_when_gptq_is_not_lower_than_legacy_by_0_3(self):
        r = evaluate(three(legacy=1.9, gptq=1.7, matched=1.8))
        self.assertTrue(r["Q1"]["evaluable"]); self.assertFalse(r["Q1"]["met"])
        self.assertEqual(r["verdict_tier1"], "partly consistent")

    def test_q1_fails_when_gptq_is_above_matched_rtn_even_if_far_below_legacy(self):
        r = evaluate(three(legacy=2.5, matched=1.0, gptq=1.4))
        self.assertFalse(r["Q1"]["met"]); self.assertEqual(r["verdict_tier1"], "partly consistent")

    def test_q1_is_not_evaluable_when_legacy_shows_no_amplification_and_the_verdict_is_partly_consistent(self):
        r = evaluate(three(legacy=1.0, matched=1.0, gptq=1.0))
        self.assertFalse(r["Q1"]["evaluable"]); self.assertFalse(r["Q1"]["met"])
        self.assertTrue(r["Q0"]["met"]); self.assertTrue(r["Q2"]["met"])
        self.assertEqual(r["verdict_tier1"], "partly consistent")

    def test_q2_fails_when_an_arm_leaves_the_band_at_int6(self):
        r = evaluate(three(int6=(1.0, 1.0, 1.3)))
        self.assertFalse(r["Q2"]["met"]); self.assertAlmostEqual(r["Q2"]["retention_int6"]["gptq"], 1.3, places=6)
        self.assertEqual(r["verdict_tier1"], "different")

    def test_q0_needs_p1_in_both_new_arms(self):
        r = evaluate(three(e_of=lambda i: -0.1 if i < 5 else E(i)))                       # 22 of 27 positive: below req(24)
        self.assertFalse(r["Q0"]["met"]); self.assertEqual(r["verdict_tier1"], "different")

    def test_q0_needs_the_dose_rule_at_int4_in_both_new_arms(self):
        broken = lambda s, i, a: 0.0 if (s == "INT4" and a == DOWN[-1]) else 1.0           # no harm at the deepest down ratio: H-G1 cannot hold
        arms = three(); arms["gptq"] = arm({"FP": 1.0, "INT6": 1.0, "INT4": 1.1}, shape=broken)
        r = evaluate(arms)
        self.assertTrue(r["Q0"]["per_arm"]["rtn-matched"]["P3a_INT4"]); self.assertFalse(r["Q0"]["per_arm"]["gptq"]["P3a_INT4"])
        self.assertFalse(r["Q0"]["met"])

    def test_per_language_count_for_q1_is_20_of_27(self):
        for k, expect in ((20, True), (19, False)):
            arms = three(); arms["gptq"] = arm({"FP": 1.0, "INT6": 1.0, "INT4": lambda i, k=k: 1.1 if i < k else 1.9})
            r = evaluate(arms)
            self.assertEqual(r["Q1"]["count_lower"], k); self.assertEqual(r["Q1"]["met"], expect, msg=str(k))

    def test_p2a_is_restricted_to_int6_for_an_arm_without_int8_and_whole_for_the_legacy_arm(self):
        r = evaluate(three())
        self.assertEqual(r["arms"]["gptq"]["P2a"]["restricted_to"], ["INT6"]); self.assertTrue(r["arms"]["gptq"]["P2a"]["met"])
        self.assertNotIn("restricted_to", r["arms"]["legacy"]["P2a"])
        self.assertEqual(sorted(r["arms"]["legacy"]["P2a"]["values"]), ["INT6", "INT8"])

    def test_q3_reports_h_g5b_and_the_count_of_best_factor_above_one_without_changing_the_verdict(self):
        r = evaluate(three())
        for name in ("legacy", "rtn-matched", "gptq"):
            q = r["Q3"][name]
            self.assertEqual(set(q["H-G5b"]), {"count", "n", "met"}); self.assertEqual(q["H-G5b"]["n"], 27)
            self.assertIsInstance(q["best_factor_above_one"], int)
        base = {"Q0": {"met": True}, "Q1": {"met": True}, "Q2": {"met": True}}
        self.assertEqual(gc.verdict_tier1({**base, "Q3": {"x": 1}}), gc.verdict_tier1({**base, "Q3": {"x": 2}}))

    def test_the_verdict_words(self):
        v = gc.verdict_tier1
        t, f = {"met": True}, {"met": False}
        self.assertEqual(v({"Q0": t, "Q1": t, "Q2": t}), "consistent")
        self.assertEqual(v({"Q0": t, "Q1": f, "Q2": t}), "partly consistent")
        self.assertEqual(v({"Q0": f, "Q1": t, "Q2": t}), "different")
        self.assertEqual(v({"Q0": t, "Q1": t, "Q2": f}), "different")
        self.assertEqual(v({"Q0": f, "Q1": f, "Q2": f}), "different")

    def test_a_missing_language_raises_with_the_arm_and_the_language_name(self):
        arms = three(); arms["gptq"] = arm({"FP": 1.0, "INT6": 1.0, "INT4": 1.1}, langs=[l for l in LANGS if l != "l5"])
        with self.assertRaises(ValueError) as ctx:
            evaluate(arms)
        self.assertIn("gptq", str(ctx.exception)); self.assertIn("l5", str(ctx.exception))

    def test_a_missing_state_raises_with_the_arm_and_the_state_name(self):
        arms = three(); arms["gptq"] = arm({"FP": 1.0, "INT4": 1.1})
        with self.assertRaises(ValueError) as ctx:
            evaluate(arms)
        self.assertIn("gptq", str(ctx.exception)); self.assertIn("INT6", str(ctx.exception))

    def test_a_missing_arm_raises(self):
        arms = three(); del arms["rtn-matched"]
        with self.assertRaises(ValueError) as ctx:
            evaluate(arms)
        self.assertIn("rtn-matched", str(ctx.exception))

    def test_a_legacy_arm_without_amplification_is_labelled_not_evaluable_with_its_numbers_in_the_report(self):
        r = evaluate(three(legacy=1.0, matched=1.0, gptq=1.0))
        res = {"tier1": r, "tier2": None, "verdict_tier1": r["verdict_tier1"], "verdict_tier2": "not run", "s4": gc.s4({}), "reference_2b": gc.reference_2b(ROOT)}
        text = gc.report_text(res)
        self.assertIn("**not evaluable**", text); self.assertNotIn("**not met** (not evaluable)", text)
        self.assertIn("legacy arm meets P3b", text); self.assertIn("1.000", text)
        ok = evaluate(three()); res["tier1"] = ok; res["verdict_tier1"] = ok["verdict_tier1"]
        self.assertNotIn("not evaluable", gc.report_text(res))


# ------------------------------------------------------------------------------------------------------------------------------ tier 2 (Task 7)
L27 = [f"l{i}" for i in range(27)]
ROSTER27 = {l: {"tier": ("high", "mid", "low")[i % 3]} for i, l in enumerate(L27)}


def s1_arm(surv, nll, langs=L27, bits=(8, 6, 4)):
    """The part of `compare_stage1.load_arm` that the tier-2 predictions read."""
    return {"languages": list(langs), "bits": list(bits), "survived_pct": {l: {b: surv(l, b) for b in bits} for l in langs}, "nll_delta": {l: {b: nll(l, b) for b in bits} for l in langs}}


def flat(table):
    """surv or nll function from {bits: value} (the same for every language)."""
    return lambda l, b: table.get(b, 0.0)


def pattern_2b(legacy_surv=None):
    return {"legacy": s1_arm(flat(legacy_surv or {8: 100.0, 6: 85.0, 4: 12.7}), flat({8: 0.0, 6: 0.0254, 4: 0.590})),
            "rtn-matched": s1_arm(flat({8: 100.0, 6: 89.7, 4: 18.7}), flat({8: 0.0, 6: 0.0197, 4: 0.385})),
            "gptq": s1_arm(flat({8: 100.0, 6: 97.5, 4: 30.5}), flat({8: 0.0, 6: 0.0058, 4: 0.130}))}


class TierTwo(unittest.TestCase):
    def test_s1_s2_orders_and_counts_on_planted_survival_and_loss(self):
        s = gc.evaluate_stage1(pattern_2b(), ROSTER27)
        self.assertTrue(s["S1"]["met"]); self.assertTrue(s["S2"]["met"])
        self.assertEqual(s["S1"]["required"], 24); self.assertEqual(s["S1"]["bits"]["6"]["count_gptq_better_than_legacy"], 27)
        self.assertAlmostEqual(s["S1"]["bits"]["4"]["mean"]["gptq"], 30.5); self.assertAlmostEqual(s["S2"]["bits"]["6"]["mean"]["legacy"], 0.0254)

    def test_s1_fails_when_the_order_breaks_at_one_bit_width(self):
        a = pattern_2b(); a["rtn-matched"] = s1_arm(flat({8: 100.0, 6: 89.7, 4: 35.0}), flat({8: 0.0, 6: 0.0197, 4: 0.385}))
        s = gc.evaluate_stage1(a, ROSTER27)
        self.assertFalse(s["S1"]["met"]); self.assertFalse(s["S1"]["bits"]["4"]["ordered"]); self.assertTrue(s["S1"]["bits"]["6"]["ordered"])

    def test_s2_fails_when_gptq_loses_more_than_matched_rtn_at_one_bit_width(self):
        a = pattern_2b(); a["gptq"] = s1_arm(flat({8: 100.0, 6: 97.5, 4: 30.5}), flat({8: 0.0, 6: 0.0058, 4: 0.5}))
        s = gc.evaluate_stage1(a, ROSTER27)
        self.assertTrue(s["S1"]["met"]); self.assertFalse(s["S2"]["met"])

    def test_the_per_language_count_needs_24_of_27(self):
        for k, expect in ((24, True), (23, False)):
            a = pattern_2b(); a["gptq"] = s1_arm(lambda l, b, k=k: ({8: 100.0, 6: 97.5, 4: 30.5}[b] if int(l[1:]) < k else {8: 100.0, 6: 85.0, 4: 12.7}[b]), flat({8: 0.0, 6: 0.0058, 4: 0.130}))
            s = gc.evaluate_stage1(a, ROSTER27)
            self.assertEqual(s["S1"]["bits"]["6"]["count_gptq_better_than_legacy"], k); self.assertEqual(s["S1"]["met"], expect, msg=str(k))

    def test_s3_the_tier_gap_at_int6_is_smaller_under_gptq_and_mid_languages_do_not_enter(self):
        high_low = lambda hi, lo, mid: (lambda l, b: {"high": hi, "mid": mid, "low": lo}[ROSTER27[l]["tier"]] if b == 6 else 50.0)
        a = pattern_2b(); a["legacy"] = s1_arm(high_low(95.0, 69.0, 10.0), flat({8: 0.0, 6: 0.0254, 4: 0.590})); a["gptq"] = s1_arm(high_low(99.0, 94.0, 0.0), flat({8: 0.0, 6: 0.0058, 4: 0.130}))
        s = gc.evaluate_stage1(a, ROSTER27)
        self.assertAlmostEqual(s["S3"]["gap"]["legacy"], 26.0); self.assertAlmostEqual(s["S3"]["gap"]["gptq"], 5.0); self.assertTrue(s["S3"]["met"])
        a["gptq"] = s1_arm(high_low(99.0, 60.0, 0.0), flat({8: 0.0, 6: 0.0058, 4: 0.130}))
        self.assertFalse(gc.evaluate_stage1(a, ROSTER27)["S3"]["met"])

    def test_the_tier2_verdict_words(self):
        v = gc.verdict_tier2; t, f = {"met": True}, {"met": False}
        self.assertEqual(v({"S1": t, "S2": t, "S3": t}), "consistent")
        self.assertEqual(v({"S1": t, "S2": t, "S3": f}), "partly consistent")
        self.assertEqual(v({"S1": f, "S2": t, "S3": t}), "different")
        self.assertEqual(v({"S1": t, "S2": f, "S3": t}), "different")

    def test_a_missing_bit_width_or_language_raises_naming_the_arm(self):
        a = pattern_2b(); a["gptq"] = s1_arm(flat({6: 97.5}), flat({6: 0.0058}), bits=(6,))
        with self.assertRaises(ValueError) as ctx:
            gc.evaluate_stage1(a, ROSTER27)
        self.assertIn("gptq", str(ctx.exception)); self.assertIn("4", str(ctx.exception))
        b = pattern_2b(); b["legacy"] = s1_arm(flat({8: 100.0, 6: 85.0, 4: 12.7}), flat({8: 0.0, 6: 0.0254, 4: 0.590}), langs=L27[:-1])
        with self.assertRaises(ValueError) as ctx:
            gc.evaluate_stage1(b, ROSTER27)
        self.assertIn("legacy", str(ctx.exception))

    def test_s4_reports_the_holm_p_values_per_arm_and_whether_the_arms_agree(self):
        an = {"legacy": {"holm": {"T1_int7": 1.0, "T2_int6": 1.0, "T3_tier": 1.0}}, "rtn-matched": {"holm": {"T1_int7": 0.042, "T2_int6": 0.038, "T3_tier": 0.98}}, "gptq": {"holm": {"T1_int7": 1.0, "T2_int6": 1.0, "T3_tier": 1.0}}}
        s4 = gc.s4(an)
        self.assertEqual(s4["words"], {"legacy": "null", "rtn-matched": "below 0.05: T1_int7, T2_int6", "gptq": "null"}); self.assertFalse(s4["agree"])
        self.assertTrue(gc.s4({"legacy": an["legacy"], "gptq": an["gptq"]})["agree"])
        self.assertEqual(gc.s4({"legacy": None, "rtn-matched": None, "gptq": None})["words"]["legacy"], "not provided")

    def test_the_arms_must_cover_the_whole_roster(self):
        small = {l: ROSTER27[l] for l in L27[:20]}
        a = {n: s1_arm(lambda l, b: 50.0, lambda l, b: 0.1, langs=L27[:20]) for n in gc.ARMS}
        with self.assertRaises(ValueError) as ctx:
            gc.evaluate_stage1(a, ROSTER27)
        self.assertIn("20", str(ctx.exception)); self.assertIn("27", str(ctx.exception)); self.assertIn("roster", str(ctx.exception))
        roster_with_extra = {**ROSTER27, "extra": {"tier": "high"}}
        with self.assertRaises(ValueError):
            gc.evaluate_stage1(pattern_2b(), roster_with_extra)
        gc.evaluate_stage1({n: s1_arm(flat({8: 1.0, 6: 90.0 + k, 4: 20.0 + k}), flat({8: 0.0, 6: 0.01 - 0.001 * k, 4: 0.1 - 0.01 * k}), langs=L27[:20]) for k, n in enumerate(("legacy", "rtn-matched", "gptq"))}, small)   # the roster is what counts: a roster of 20 is covered


class Reference(unittest.TestCase):
    """The committed 2B outputs through the same code: the thresholds of the amendment were derived from them, so the 2B arms must satisfy them."""

    @classmethod
    def setUpClass(cls):
        cls.ref = gc.reference_2b(ROOT)

    def test_the_2b_control_gives_the_known_tier_1_numbers(self):
        r = self.ref["tier1"]
        self.assertAlmostEqual(r["r"]["legacy"], 1.911, delta=0.005); self.assertAlmostEqual(r["r"]["rtn-matched"], 1.575, delta=0.005); self.assertAlmostEqual(r["r"]["gptq"], 1.239, delta=0.005)
        self.assertEqual((r["count_gptq_below_legacy"], r["n"]), (25, 27))

    def test_q1_is_met_on_the_2b_data(self):
        r = self.ref["tier1"]
        stubs = {a: {"P3b": {"median_ratio": r["r"][a], "met": True}} for a in gc.ARMS}
        q = gc.q1(stubs, r["ratios"], list(r["ratios"]["legacy"]))
        self.assertTrue(q["met"]); self.assertEqual(q["count_lower"], 25)

    def test_the_2b_control_gives_the_known_tier_2_numbers_and_the_tier_2_verdict(self):
        s = self.ref["tier2"]
        for arm, want in (("legacy", 85.0), ("rtn-matched", 89.7), ("gptq", 97.5)):
            self.assertAlmostEqual(s["S1"]["bits"]["6"]["mean"][arm], want, delta=0.15, msg=arm)
        for arm, want in (("legacy", 12.7), ("rtn-matched", 18.7), ("gptq", 30.5)):
            self.assertAlmostEqual(s["S1"]["bits"]["4"]["mean"][arm], want, delta=0.15, msg=arm)
        for arm, want in (("legacy", 0.590), ("rtn-matched", 0.385), ("gptq", 0.130)):
            self.assertAlmostEqual(s["S2"]["bits"]["4"]["mean"][arm], want, delta=0.001, msg=arm)
        self.assertTrue(s["S1"]["met"]); self.assertTrue(s["S2"]["met"]); self.assertTrue(s["S3"]["met"]); self.assertEqual(s["verdict_tier2"], "consistent")


class Cli(unittest.TestCase):
    def tier_one_dirs(self, d):
        preset = gr.preset_name(DOWN + UP + FINE, 3, "K64"); arms = three(); out = {}
        for name, g in arms.items():
            out[name] = []
            for s in g.states:
                if s == "FP" and name != "legacy":
                    continue
                dd = d / f"job-{name}-{s}"; save_state(dd, g.runs[s], preset); out[name].append(str(dd))
        (d / "man.json").write_text(json.dumps({"languages": LANGS, "groups": {l: {"all": list(range(64))} for l in LANGS}}))
        return out, ratios_json(d)

    def tier_two_dirs(self, d):
        return {"legacy": str(make_arm(d / "p1-legacy")), "rtn-matched": str(make_arm(d / "p1-matched", corr_shift=0.03, nll_shift=-0.05)), "gptq": str(make_arm(d / "p1-gptq", corr_shift=0.06, nll_shift=-0.10))}, roster_file(d)

    def run_cli(self, argv):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            gc.main(argv)
        return out.getvalue()

    def test_cli_writes_both_files_and_both_verdicts(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); g, ratios = self.tier_one_dirs(d); p1, roster = self.tier_two_dirs(d)
            text = self.run_cli(["--legacy-g", *g["legacy"], "--rtnm-g", *g["rtn-matched"], "--gptq-g", *g["gptq"], "--manifest", str(d / "man.json"), "--ratios-json", str(ratios),
                                 "--legacy-p1", p1["legacy"], "--rtnm-p1", p1["rtn-matched"], "--gptq-p1", p1["gptq"], "--roster", str(roster), "--repo-root", str(ROOT), "--out", str(d / "out"),
                                 "--n-boot", "30", "--n-boot-languages", "200"])
            js = json.loads((d / "out/analysis.json").read_text()); report = (d / "out/report.md").read_text()
            self.assertEqual(js["verdict_tier1"], "consistent")
            self.assertEqual(js["verdict_tier2"], gc.verdict_tier2(js["tier2"]))
            for k in ("Q0", "Q1", "Q2", "Q3", "S1", "S2", "S3", "S4", "consistent", "one model", "RTX PRO 6000", "2B"):
                self.assertIn(k, report)
            self.assertIn("tier 1", text.lower()); self.assertIn("tier 2", text.lower())

    def test_cli_with_tier_two_absent_says_not_run_and_gives_no_verdict(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); g, ratios = self.tier_one_dirs(d)
            self.run_cli(["--legacy-g", *g["legacy"], "--rtnm-g", *g["rtn-matched"], "--gptq-g", *g["gptq"], "--manifest", str(d / "man.json"), "--ratios-json", str(ratios), "--repo-root", str(ROOT),
                          "--out", str(d / "out"), "--n-boot", "30", "--n-boot-languages", "200"])
            js = json.loads((d / "out/analysis.json").read_text())
            self.assertIsNone(js["tier2"]); self.assertEqual(js["verdict_tier2"], "not run")
            self.assertIn("tier 2: not run", (d / "out/report.md").read_text().lower())

    def test_cli_with_tier_one_absent_says_not_run(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); p1, roster = self.tier_two_dirs(d)
            self.run_cli(["--legacy-p1", p1["legacy"], "--rtnm-p1", p1["rtn-matched"], "--gptq-p1", p1["gptq"], "--roster", str(roster), "--repo-root", str(ROOT), "--out", str(d / "out")])
            js = json.loads((d / "out/analysis.json").read_text())
            self.assertIsNone(js["tier1"]); self.assertEqual(js["verdict_tier1"], "not run")

    def test_cli_refuses_a_partial_tier(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); p1, roster = self.tier_two_dirs(d)
            with self.assertRaises(SystemExit):
                self.run_cli(["--legacy-p1", p1["legacy"], "--rtnm-p1", p1["rtn-matched"], "--roster", str(roster), "--out", str(d / "out")])

    def test_cli_refuses_a_manifest_for_other_languages(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); g, ratios = self.tier_one_dirs(d)
            (d / "man.json").write_text(json.dumps({"languages": ["x1"], "groups": {"x1": {"all": list(range(64))}}}))
            with self.assertRaisesRegex(ValueError, "different languages"):
                self.run_cli(["--legacy-g", *g["legacy"], "--rtnm-g", *g["rtn-matched"], "--gptq-g", *g["gptq"], "--manifest", str(d / "man.json"), "--ratios-json", str(ratios), "--out", str(d / "out")])


if __name__ == "__main__":
    unittest.main()
