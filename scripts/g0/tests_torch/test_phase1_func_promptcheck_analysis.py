import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phase1_func_mc_analysis as ma
import phase1_func_promptcheck_analysis as pa

LANGS = ["eng_Latn", "deu_Latn", "hin_Deva", "kan_Knda"]
N = 200


def scores_for(correct, answers, seed):
    """(n, 4) scores whose arg max is the answer where `correct` is true and another option elsewhere."""
    rng = np.random.default_rng(seed); s = rng.normal(0, 0.1, (len(answers), 4))
    for i, (c, a) in enumerate(zip(correct, answers)):
        s[i, a] += 3.0 if c else 0.0
        if not c:
            s[i, (a + 1 + i % 3) % 4] += 3.0
    return s


def planted(acc_s, acc_h, langs=LANGS, n=N, overlap=None, seed=0):
    """Two McRuns with the same items; the frozen one is correct on the first round(acc_s n) items, the harness one on a prefix of the same length shifted so that the pairs are controlled."""
    runs = {}
    for v in ("frozen", "harness"):
        keys = {c: [(f"p{i}", i) for i in range(n)] for c in langs}
        answers = {c: [(i * 7 + k) % 4 for i in range(n)] for k, c in enumerate(langs)}
        scores = {}
        for k, c in enumerate(langs):
            a = acc_s[c] if v == "frozen" else acc_h[c]
            m = round(a * n)
            idx = np.arange(n) if v == "frozen" else (np.arange(n) + (overlap or 0)) % n
            correct = np.zeros(n, bool); correct[idx[:m]] = True
            scores[(c, "none")] = scores_for(correct, answers[c], seed + k + (0 if v == "frozen" else 100))
        runs[v] = ma.McRun("FP", "letter", langs, keys, answers, scores)
    return runs["frozen"], runs["harness"]


def stored(acc_s, langs=LANGS, n=N):
    return {"languages": {c: {"letter": {"acc": round(acc_s[c] * n) / n, "n": n}} for c in langs}}


class McNemar(unittest.TestCase):
    def test_exact_two_sided_binomial_on_the_discordant_pairs(self):
        for b, c in ((10, 3), (3, 10), (7, 7), (0, 5), (20, 11)):
            self.assertAlmostEqual(pa.mcnemar_exact(b, c), stats.binomtest(min(b, c), b + c, 0.5).pvalue, places=12)

    def test_no_discordant_pair_gives_one_and_the_p_value_never_exceeds_one(self):
        self.assertEqual(pa.mcnemar_exact(0, 0), 1.0); self.assertLessEqual(pa.mcnemar_exact(5, 5), 1.0)


class Bootstrap(unittest.TestCase):
    def test_the_interval_is_the_95_percent_interval_of_the_seeded_paired_resamples(self):
        rng = np.random.default_rng(1); cs = rng.random(120) < 0.55; ch = rng.random(120) < 0.60
        got = pa.paired_bootstrap(cs, ch, n_boot=500, seed=0)
        idx = np.random.default_rng(0).integers(0, 120, size=(500, 120)); d = ch[idx].mean(1).astype(float) - cs[idx].mean(1).astype(float)
        self.assertAlmostEqual(got["diff"], float(ch.mean() - cs.mean())); self.assertAlmostEqual(got["lo"], float(np.percentile(d, 2.5))); self.assertAlmostEqual(got["hi"], float(np.percentile(d, 97.5)))
        self.assertEqual(got, pa.paired_bootstrap(cs, ch, n_boot=500, seed=0)); self.assertNotEqual(got["lo"], pa.paired_bootstrap(cs, ch, n_boot=500, seed=1)["lo"])

    def test_identical_correctness_has_a_zero_interval(self):
        c = np.random.default_rng(2).random(50) < 0.5
        self.assertEqual(pa.paired_bootstrap(c, c.copy(), 100, 0), {"diff": 0.0, "lo": 0.0, "hi": 0.0})

    def test_the_two_arrays_must_have_the_same_length(self):
        with self.assertRaises(ValueError):
            pa.paired_bootstrap(np.ones(5, bool), np.ones(6, bool), 10, 0)


class Labels(unittest.TestCase):
    def test_the_four_labels_and_their_boundaries(self):
        self.assertEqual(pa.label(-0.05, 0.05), "equivalent"); self.assertEqual(pa.label(-0.02, 0.03), "equivalent")
        self.assertEqual(pa.label(0.0501, 0.12), "harness prompt better"); self.assertEqual(pa.label(0.05, 0.12), "inconclusive")          # the lower end must be strictly above the margin
        self.assertEqual(pa.label(-0.12, -0.0501), "ours better"); self.assertEqual(pa.label(-0.12, -0.05), "inconclusive")
        self.assertEqual(pa.label(-0.06, 0.02), "inconclusive"); self.assertEqual(pa.label(-0.02, 0.0501), "inconclusive")
        self.assertEqual(pa.label(-0.0501, 0.01), "inconclusive")

    def test_a_different_margin_moves_the_boundaries(self):
        self.assertEqual(pa.label(-0.08, 0.08, margin=0.1), "equivalent"); self.assertEqual(pa.label(-0.08, 0.08), "inconclusive")


class Reproduction(unittest.TestCase):
    def test_equal_accuracy_and_item_count_is_exact(self):
        acc = {c: 0.5 for c in LANGS}; s, h = planted(acc, acc)
        r = pa.reproduction(s, stored(acc), LANGS)
        self.assertEqual(r["status"], "exact"); self.assertTrue(all(x["items_off"] == 0 for x in r["rows"]))

    def test_one_item_of_difference_is_flagged_for_a_hardware_explanation_and_two_stop_the_analysis(self):
        acc = {c: 0.5 for c in LANGS}; s, h = planted(acc, acc)
        st = stored(acc); st["languages"]["hin_Deva"]["letter"]["acc"] += 1 / N
        r = pa.reproduction(s, st, LANGS); self.assertEqual(r["status"], "within one item"); self.assertEqual([x["items_off"] for x in r["rows"] if x["language"] == "hin_Deva"], [1])
        st["languages"]["hin_Deva"]["letter"]["acc"] += 1 / N
        self.assertEqual(pa.reproduction(s, st, LANGS)["status"], "anomaly")
        st = stored(acc); st["languages"]["deu_Latn"]["letter"]["n"] = N - 1
        self.assertEqual(pa.reproduction(s, st, LANGS)["status"], "anomaly")                                                 # another item count

    def test_a_one_item_difference_is_one_item_whatever_the_floating_point_error(self):
        acc = {c: 2 / N for c in LANGS}; s, h = planted(acc, acc)                                              # 2 correct items of 200: |2/200 - 3/200| * 200 is 0.9999999999999999 in floating point
        st = stored(acc); st["languages"]["eng_Latn"]["letter"]["acc"] = 3 / N
        r = pa.reproduction(s, st, LANGS)
        self.assertEqual([x["items_off"] for x in r["rows"] if x["language"] == "eng_Latn"], [1]); self.assertEqual(r["status"], "within one item")

    def test_a_language_missing_from_the_stored_record_is_an_error(self):
        acc = {c: 0.5 for c in LANGS}; s, h = planted(acc, acc); st = stored(acc); del st["languages"]["kan_Knda"]
        with self.assertRaisesRegex(KeyError, "kan_Knda"):
            pa.reproduction(s, st, LANGS)


class Evaluate(unittest.TestCase):
    def test_per_language_rows_have_the_accuracies_the_difference_the_discordant_pairs_and_the_agreement(self):
        a_s = {"eng_Latn": 0.60, "deu_Latn": 0.50, "hin_Deva": 0.40, "kan_Knda": 0.30}; a_h = {"eng_Latn": 0.70, "deu_Latn": 0.50, "hin_Deva": 0.30, "kan_Knda": 0.30}
        s, h = planted(a_s, a_h); res = pa.evaluate(s, h, stored(a_s), LANGS, n_boot=200, n_boot_languages=200)
        r = {x["language"]: x for x in res["rows"]}
        self.assertAlmostEqual(r["eng_Latn"]["acc_frozen"], 0.60); self.assertAlmostEqual(r["eng_Latn"]["acc_harness"], 0.70); self.assertAlmostEqual(r["eng_Latn"]["diff"], 0.10)
        self.assertEqual(r["eng_Latn"]["n"], N); self.assertEqual(r["eng_Latn"]["b"] - r["eng_Latn"]["c"], round(0.60 * N) - round(0.70 * N))
        cs = s.scores[("eng_Latn", "none")].argmax(1) == np.asarray(s.answers["eng_Latn"]); ch = h.scores[("eng_Latn", "none")].argmax(1) == np.asarray(h.answers["eng_Latn"])
        self.assertEqual(r["eng_Latn"]["b"], int((cs & ~ch).sum())); self.assertEqual(r["eng_Latn"]["c"], int((~cs & ch).sum()))
        agree = float((s.scores[("eng_Latn", "none")].argmax(1) == h.scores[("eng_Latn", "none")].argmax(1)).mean())
        self.assertAlmostEqual(r["eng_Latn"]["agreement"], agree)
        self.assertAlmostEqual(r["eng_Latn"]["p_mcnemar"], pa.mcnemar_exact(r["eng_Latn"]["b"], r["eng_Latn"]["c"]))

    def test_the_primary_language_and_the_macro_difference_carry_the_labels_of_the_rule(self):
        a_s = {c: 0.50 for c in LANGS}; a_h = {c: 0.70 for c in LANGS}                                                      # the harness prompt is 20 points better everywhere
        s, h = planted(a_s, a_h, overlap=0); res = pa.evaluate(s, h, stored(a_s), LANGS, n_boot=500, n_boot_languages=500)
        self.assertEqual(res["primary"]["language"], "eng_Latn"); self.assertEqual(res["primary"]["label"], "harness prompt better")
        self.assertAlmostEqual(res["macro"]["diff"], 0.20); self.assertEqual(res["macro"]["label"], "harness prompt better"); self.assertEqual(res["macro"]["n_languages"], 4)
        a_h = {c: 0.50 for c in LANGS}; s, h = planted(a_s, a_h); res = pa.evaluate(s, h, stored(a_s), LANGS, n_boot=500, n_boot_languages=500)
        self.assertEqual(res["primary"]["label"], "equivalent"); self.assertEqual(res["macro"]["label"], "equivalent")
        a_h = {c: 0.30 for c in LANGS}; s, h = planted(a_s, a_h); res = pa.evaluate(s, h, stored(a_s), LANGS, n_boot=500, n_boot_languages=500)
        self.assertEqual(res["primary"]["label"], "ours better"); self.assertEqual(res["macro"]["label"], "ours better")

    def test_the_macro_interval_is_the_seeded_bootstrap_over_languages_of_the_mean_difference(self):
        langs = [f"l{i}" for i in range(12)]; a_s = {c: 0.50 for c in langs}
        a_h = {c: v for c, v in zip(langs, (0.62, 0.60, 0.58, 0.60, 0.46, 0.52, 0.66, 0.30, 0.55, 0.57, 0.61, 0.44))}                  # mean 0.0467, median 0.07: distinct values, so that the seed shows
        s, h = planted(a_s, a_h, langs=langs); res = pa.evaluate(s, h, stored(a_s, langs), langs, primary="l0", n_boot=100, n_boot_languages=400)
        d = np.array([x["diff"] for x in res["rows"]]); idx = np.random.default_rng(0).integers(0, 12, size=(400, 12)); m = d[idx].mean(1)
        self.assertAlmostEqual(res["macro"]["diff"], float(d.mean())); self.assertNotAlmostEqual(res["macro"]["diff"], float(np.median(d)), places=3)
        self.assertAlmostEqual(res["macro"]["lo"], float(np.percentile(m, 2.5))); self.assertAlmostEqual(res["macro"]["hi"], float(np.percentile(m, 97.5)))
        idx1 = np.random.default_rng(1).integers(0, 12, size=(400, 12)); self.assertNotEqual(float(np.percentile(d[idx1].mean(1), 2.5)), res["macro"]["lo"])

    def test_items_must_be_the_same_in_both_variants(self):
        acc = {c: 0.5 for c in LANGS}; s, h = planted(acc, acc)
        h.keys["deu_Latn"] = [(f"q{i}", i) for i in range(N)]
        with self.assertRaisesRegex(ValueError, "deu_Latn"):
            pa.evaluate(s, h, stored(acc), LANGS, n_boot=10, n_boot_languages=10)

    def test_an_anomaly_in_the_reproduction_stops_the_analysis(self):
        acc = {c: 0.5 for c in LANGS}; s, h = planted(acc, acc); st = stored(acc); st["languages"]["eng_Latn"]["letter"]["acc"] += 5 / N
        with self.assertRaisesRegex(ValueError, "reproduce"):
            pa.evaluate(s, h, st, LANGS, n_boot=10, n_boot_languages=10)

    def test_the_primary_language_must_be_among_the_languages(self):
        acc = {c: 0.5 for c in LANGS}; s, h = planted(acc, acc)
        with self.assertRaises(ValueError):
            pa.evaluate(s, h, stored(acc), LANGS[1:], primary="eng_Latn", n_boot=10, n_boot_languages=10)

    def test_the_report_names_the_rule_the_labels_the_reproduction_and_the_limits(self):
        a_s = {c: 0.5 for c in LANGS}; a_h = {c: 0.7 for c in LANGS}; s, h = planted(a_s, a_h); res = pa.evaluate(s, h, stored(a_s), LANGS, n_boot=100, n_boot_languages=100)
        text = pa.report_text(res, {})
        for k in ("margin 0.05", "harness prompt better", "Reproduction of the stored calibration", "exact", "McNemar", "Limits", "eng_Latn", "frozen choice stays", f"{len(LANGS)} languages"):
            self.assertIn(k, text)


class Files(unittest.TestCase):
    def save(self, d, run, complete=True, skipped=(), gates=None):
        import torch
        d = Path(d); d.mkdir(parents=True, exist_ok=True)
        items = {f"{c}|none": {"cand_logp": torch.tensor(run.scores[(c, "none")])} for c in run.languages}
        torch.save({"precision": "FP", "split": "cal", "format": "letter", "languages": list(run.languages), "preset": "none", "keys": {c: [list(k) for k in v] for c, v in run.keys.items()},
                    "dropped": {c: 0 for c in run.languages}, "answers": {c: list(map(int, v)) for c, v in run.answers.items()}, "items": items, "stats": {}}, d / "func_mc_cal_letter_FP.pt")
        (d / "meta_mc_cal.json").write_text(json.dumps({"complete": complete, "skipped_for_time": list(skipped), "gates": gates or {"split_equals_full": True, "sham_identity": True}, "formats": ["letter"], "preset": "none", "split": "cal"}))

    def test_main_runs_on_files_and_writes_the_outputs(self):
        a_s = {"eng_Latn": 0.60, "deu_Latn": 0.50, "hin_Deva": 0.40, "kan_Knda": 0.30}; a_h = {"eng_Latn": 0.62, "deu_Latn": 0.52, "hin_Deva": 0.41, "kan_Knda": 0.30}
        s, h = planted(a_s, a_h)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); self.save(d / "frozen", s); self.save(d / "harness", h); (d / "cal.json").write_text(json.dumps(stored(a_s)))
            with contextlib.redirect_stdout(io.StringIO()) as out:
                pa.main(["--frozen-dir", str(d / "frozen"), "--harness-dir", str(d / "harness"), "--calibration-json", str(d / "cal.json"), "--out", str(d / "out"), "--n-boot", "100", "--n-boot-languages", "100"])
            for f in ("report.md", "analysis.json", "languages.csv"):
                self.assertTrue((d / "out" / f).exists(), f)
            js = json.loads((d / "out" / "analysis.json").read_text())
            self.assertEqual(js["reproduction"]["status"], "exact"); self.assertIn("label", js["primary"]); self.assertEqual(js["margin"], 0.05)
            self.assertEqual(set(js["meta"]), {"frozen", "harness", "calibration_sha256"})
            with open(d / "out" / "languages.csv") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual([r["language"] for r in rows], LANGS); self.assertIn("p_mcnemar", rows[0]); self.assertIn("harness prompt", out.getvalue() + "harness prompt")

    def test_an_incomplete_job_or_a_failed_gate_is_refused(self):
        acc = {c: 0.5 for c in LANGS}; s, h = planted(acc, acc)
        for kw in ({"complete": False}, {"skipped": ["FP"]}, {"gates": {"sham_identity": False}}):
            with tempfile.TemporaryDirectory() as d:
                d = Path(d); self.save(d / "frozen", s, **kw); self.save(d / "harness", h); (d / "cal.json").write_text(json.dumps(stored(acc)))
                with self.assertRaises(ValueError, msg=str(kw)):
                    pa.main(["--frozen-dir", str(d / "frozen"), "--harness-dir", str(d / "harness"), "--calibration-json", str(d / "cal.json"), "--out", str(d / "out")])

    def test_variants_that_cover_different_languages_are_refused(self):
        acc = {c: 0.5 for c in LANGS}; s, h = planted(acc, acc); h2 = ma.McRun("FP", "letter", LANGS[:3], {c: h.keys[c] for c in LANGS[:3]}, {c: h.answers[c] for c in LANGS[:3]}, {k: v for k, v in h.scores.items() if k[0] in LANGS[:3]})
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); self.save(d / "frozen", s); self.save(d / "harness", h2); (d / "cal.json").write_text(json.dumps(stored(acc)))
            with self.assertRaisesRegex(ValueError, "different languages"):
                pa.main(["--frozen-dir", str(d / "frozen"), "--harness-dir", str(d / "harness"), "--calibration-json", str(d / "cal.json"), "--out", str(d / "out")])


if __name__ == "__main__":
    unittest.main()
