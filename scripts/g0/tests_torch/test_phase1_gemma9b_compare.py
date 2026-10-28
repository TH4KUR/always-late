"""Tasks 6 and 7 of the Gemma-2-9B three-arm plan: the analysis of the frozen predictions (spec section 5). Tier 1: Q0 to Q3 across the legacy, matched-RTN and GPTQ arms of Study G, computed with the frozen Study G
code (`phase1_llama_compare.evaluate`, `phase1_gain_analysis`). Synthetic worlds with planted harms (the helpers of the Gain and Llama tests); the group harm of language i in state s at ratio a is
E(i) * f(s, i) * depth(a) / 6, the matched-control harm a tenth of it, so the net harm at the deepest down ratio of a state is proportional to f and the INT4 over FP ratio equals f(INT4) / f(FP)."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_gain_analysis as ga
import phase1_gemma9b_compare as gc
from test_phase1_gain_analysis import DOWN, FINE, UP, planted_run

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


if __name__ == "__main__":
    unittest.main()
