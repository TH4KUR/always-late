import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phase1_func_mc_calibration as mc


def letter_run(acc_by_lang, n=100, seed=0, keys_offset=0):
    """A planted letter-format calibration file: item i of language l is answered correctly with probability acc."""
    g = np.random.default_rng(seed)
    items, keys, answers = {}, {}, {}
    for code, acc in acc_by_lang.items():
        ans = g.integers(0, 4, n)
        lp = g.normal(0, 1, (n, 4))
        right = g.random(n) < acc
        for i in range(n):
            if right[i]:
                lp[i, ans[i]] += 20.0
            else:
                lp[i, (ans[i] + 1) % 4] += 20.0
        items[f"{code}|none"] = {"cand_logp": torch.tensor(lp - 3.0)}              # log-probabilities over the vocabulary are lower than the 4-way normalised ones
        keys[code] = [(f"p{i + keys_offset}", 1) for i in range(n)]
        answers[code] = ans.tolist()
    return {"precision": "FP", "split": "cal", "format": "letter", "languages": list(acc_by_lang), "keys": keys, "answers": answers, "items": items, "dropped": {c: 0 for c in acc_by_lang}}


def cloze_run(acc_by_lang, n=60, seed=1):
    g = np.random.default_rng(seed)
    items, keys, answers = {}, {}, {}
    for code, acc in acc_by_lang.items():
        ans = g.integers(0, 4, n); nll = np.zeros((n, 4)); ntok = g.integers(2, 6, (n, 4))
        for i in range(n):
            right = g.random() < acc
            best = ans[i] if right else (ans[i] + 1) % 4
            nll[i] = ntok[i] * 3.0
            nll[i, best] = ntok[i, best] * 1.0
        items[f"{code}|none"] = {"nll": torch.tensor(nll.reshape(-1)), "ntok": torch.tensor(ntok.reshape(-1))}
        keys[code] = [(f"p{i}", 1) for i in range(n)]
        answers[code] = ans.tolist()
    return {"precision": "FP", "split": "cal", "format": "cloze", "languages": list(acc_by_lang), "keys": keys, "answers": answers, "items": items, "dropped": {c: 0 for c in acc_by_lang}}


class Scoring(unittest.TestCase):
    def test_wilson_interval(self):
        lo, hi = mc.wilson(50, 100)
        self.assertAlmostEqual(lo, 0.4038, places=3); self.assertAlmostEqual(hi, 0.5962, places=3)
        lo, hi = mc.wilson(0, 10)
        self.assertEqual(lo, 0.0); self.assertGreater(hi, 0.2)

    def test_letter_correctness_uses_the_four_candidates_only(self):
        run = letter_run({"aa": 1.0}, n=50)
        c = mc.correct_letter(run, "aa")
        self.assertEqual(c.shape, (50,)); self.assertTrue(bool(c.all()))
        run0 = letter_run({"aa": 0.0}, n=50)
        self.assertFalse(bool(mc.correct_letter(run0, "aa").any()))

    def test_cloze_uses_length_normalised_likelihood(self):
        # option 0 (the answer) has the highest per-token likelihood but the lowest total likelihood because it is long
        nll = torch.tensor([9.0, 6.0, 6.0, 6.0]); ntok = torch.tensor([9, 2, 2, 2])                 # per token: 1.0 against 3.0
        run = {"languages": ["aa"], "keys": {"aa": [("p", 1)]}, "answers": {"aa": [0]}, "items": {"aa|none": {"nll": nll, "ntok": ntok}}, "format": "cloze"}
        self.assertTrue(bool(mc.correct_cloze(run, "aa")[0]))
        self.assertFalse(bool(mc.correct_cloze(run, "aa", normalise=False)[0]))


class Calibration(unittest.TestCase):
    def test_coverage_rule_and_macro_accuracy(self):
        run = letter_run({"hi": 0.9, "lo": 0.3, "mid": 0.55}, n=200)
        res = mc.calibrate({"letter": run}, chance=0.25, margin=0.05)
        cov = res["languages"]
        self.assertTrue(cov["hi"]["covered"]); self.assertTrue(cov["mid"]["covered"]); self.assertFalse(cov["lo"]["covered"])
        self.assertAlmostEqual(cov["hi"]["letter"]["acc"], 0.9, delta=0.06)
        self.assertEqual(res["covered"], ["hi", "mid"])
        self.assertEqual(res["format"], "letter")

    def test_format_choice_compares_on_the_shared_items_and_ties_go_to_letter(self):
        letter = letter_run({"aa": 0.4, "bb": 0.4}, n=100)
        cloze = cloze_run({"aa": 0.9, "bb": 0.9}, n=60)
        res = mc.calibrate({"letter": letter, "cloze": cloze})
        self.assertEqual(res["format"], "cloze")
        self.assertEqual(res["shared_items"], {"aa": 60, "bb": 60})
        same = mc.calibrate({"letter": letter_run({"aa": 0.5}, n=60, seed=3), "cloze": letter_to_cloze_same(letter_run({"aa": 0.5}, n=60, seed=3))})
        self.assertEqual(same["format"], "letter")

    def test_coverage_uses_the_chosen_format(self):
        letter = letter_run({"aa": 0.3}, n=300)
        cloze = cloze_run({"aa": 0.9}, n=300)
        res = mc.calibrate({"letter": letter, "cloze": cloze})
        self.assertEqual(res["format"], "cloze"); self.assertTrue(res["languages"]["aa"]["covered"])

    def test_report_files(self):
        out = Path(tempfile.mkdtemp())
        res = mc.calibrate({"letter": letter_run({"aa": 0.9, "bb": 0.26}, n=100)}, out_dir=out)
        self.assertTrue((out / "belebele_calibration.json").exists())
        text = (out / "belebele_calibration.md").read_text()
        for needle in ("Coverage", "aa", "bb", "chance", "format"):
            self.assertIn(needle, text)
        self.assertEqual(json.loads((out / "belebele_calibration.json").read_text())["covered"], res["covered"])


def letter_to_cloze_same(run):
    """A cloze file whose per-item correctness equals the given letter file's (to test the tie rule)."""
    items = {}
    for code in run["languages"]:
        c = mc.correct_letter(run, code); ans = run["answers"][code]
        nll = np.full((len(c), 4), 3.0)
        for i in range(len(c)):
            nll[i, ans[i] if c[i] else (ans[i] + 1) % 4] = 1.0
        items[f"{code}|none"] = {"nll": torch.tensor(nll.reshape(-1)), "ntok": torch.ones(len(c) * 4, dtype=torch.long)}
    return {**run, "format": "cloze", "items": items}


if __name__ == "__main__":
    unittest.main()
