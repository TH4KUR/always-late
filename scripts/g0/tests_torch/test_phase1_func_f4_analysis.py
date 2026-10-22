import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_func_analysis as fa
import phase1_func_f4_analysis as f4a
from test_phase1_func_analysis import make_run


def merged(label, specs, n=400, seed=0, noise=0.05):
    """A planted precision with, per language, the harm of the LOST subset (the group) and of the matched RETAINED subsets (the controls): specs {code: (harm_lost, harm_retained)}."""
    items, stats, fid, ids = {}, {}, {}, {}
    for k, (code, (lost, ret)) in enumerate(specs.items()):
        r = make_run(label, harm_group=lost, harm_ctrl=ret, n=n, langs=(code,), seed=seed + 7 * k, noise=noise)
        for (text, name), (nll, ntok) in r._items.items():
            items[f"{text}|{name}"] = {"nll": torch.tensor(nll), "ntok": torch.tensor(ntok)}
        stats.update(r.stats); fid.update(r.fidelity); ids.update(r.ids)
    return fa.Run(label, items, stats, fid, list(specs), ids)


def save_run(run, path):
    items = {f"{text}|{name}": {"nll": torch.tensor(nll), "ntok": torch.tensor(ntok)} for (text, name), (nll, ntok) in run._items.items()}
    torch.save({"precision": run.label, "items": items, "stats": run.stats, "fidelity": run.fidelity, "languages": run.languages, "ids": run.ids}, path)


class Effects(unittest.TestCase):
    def setUp(self):
        self.out = Path(tempfile.mkdtemp()) / "res"

    def runs(self, fp=(0.1, 0.3), b=(0.05, 0.4), langs=("aa", "bb")):
        return {"FP": merged("FP", {c: fp for c in langs}, seed=1), "INT4": merged("INT4", {c: b for c in langs}, seed=100)}

    def test_the_planted_contrast_is_lost_minus_retained_and_is_recovered(self):
        res = f4a.analyse_f4(self.runs(), self.out, n_boot=300)
        e = res["per_language"]["aa"]
        self.assertAlmostEqual(e["E_fp"]["point"], -0.2, delta=0.02); self.assertAlmostEqual(e["E_b"]["point"], -0.35, delta=0.02)
        self.assertAlmostEqual(e["harm_lost_b"], 0.05, delta=0.02); self.assertAlmostEqual(e["harm_retained_b"], 0.4, delta=0.02)
        self.assertAlmostEqual(e["harm_lost_fp"], 0.1, delta=0.02); self.assertAlmostEqual(e["harm_retained_fp"], 0.3, delta=0.02)
        self.assertLess(e["E_b"]["lo"], e["E_b"]["point"]); self.assertGreater(e["E_b"]["hi"], e["E_b"]["point"])
        self.assertEqual(res["condition"], "INT4"); self.assertEqual(res["languages"], ["aa", "bb"])

    def test_delta_is_the_native_contrast_minus_the_full_precision_contrast_with_paired_draws(self):
        res = f4a.analyse_f4(self.runs(), self.out, n_boot=300)
        d = res["per_language"]["aa"]["delta"]
        self.assertAlmostEqual(d["point"], res["per_language"]["aa"]["E_b"]["point"] - res["per_language"]["aa"]["E_fp"]["point"], places=12)
        self.assertAlmostEqual(d["point"], -0.15, delta=0.03)
        self.assertLess(d["lo"], d["point"]); self.assertGreater(d["hi"], d["point"]); self.assertLess(d["hi"], 0.0)       # the interval is around the point (a mirrored interval would be around +0.15)
        same = {"FP": merged("FP", {"aa": (0.1, 0.3)}, seed=1), "INT4": merged("INT4", {"aa": (0.1, 0.3)}, seed=1)}
        z = f4a.analyse_f4(same, self.out / "same", n_boot=200)["per_language"]["aa"]["delta"]
        self.assertEqual((z["point"], z["lo"], z["hi"]), (0.0, 0.0, 0.0))                                   # identical data under the same draws: a zero-width zero

    def test_a_planted_null_has_an_interval_around_zero_and_is_inconclusive(self):
        # a 95 % interval misses a planted null in about 1 of 40 seeds (scanned: 39 of 40 cover it), so the seeds are fixed; the property is "within sampling error of zero, interval covers it"
        res = f4a.analyse_f4({"FP": merged("FP", {"aa": (0.2, 0.2)}, seed=7, noise=0.15), "INT4": merged("INT4", {"aa": (0.2, 0.2)}, seed=1007, noise=0.15)}, self.out, n_boot=300)
        e = res["per_language"]["aa"]["E_b"]
        self.assertLess(abs(e["point"]), 0.03); self.assertLess(e["lo"], 0.0); self.assertGreater(e["hi"], 0.0); self.assertEqual(res["per_language"]["aa"]["sign_b"], "inconclusive")

    def test_signs_are_decided_by_the_interval_and_counted(self):
        specs = {"aa": (0.05, 0.40), "bb": (0.40, 0.05), "cc": (0.2, 0.2)}
        res = f4a.analyse_f4({"FP": merged("FP", specs, seed=1, noise=0.15), "INT4": merged("INT4", specs, seed=2, noise=0.15)}, self.out, n_boot=300)
        self.assertEqual([res["per_language"][c]["sign_b"] for c in ("aa", "bb", "cc")], ["negative", "positive", "inconclusive"])
        self.assertEqual(res["counts"]["E_b"], {"negative": 1, "positive": 1, "inconclusive": 1})

    def test_the_full_precision_signs_are_counted_separately_from_the_native_ones(self):
        fp = {"aa": (0.2, 0.2), "bb": (0.4, 0.05), "cc": (0.05, 0.4)}                                       # null, positive, negative at full precision
        b = {"aa": (0.05, 0.4), "bb": (0.05, 0.4), "cc": (0.2, 0.2)}                                        # negative, negative, null at the native precision
        res = f4a.analyse_f4({"FP": merged("FP", fp, seed=1, noise=0.15), "INT4": merged("INT4", b, seed=2, noise=0.15)}, self.out, n_boot=300)
        self.assertEqual([res["per_language"][c]["sign_fp"] for c in ("aa", "bb", "cc")], ["inconclusive", "positive", "negative"])
        self.assertEqual([res["per_language"][c]["sign_b"] for c in ("aa", "bb", "cc")], ["negative", "negative", "inconclusive"])
        self.assertEqual(res["counts"]["E_fp"], {"negative": 1, "positive": 1, "inconclusive": 1}); self.assertEqual(res["counts"]["E_b"], {"negative": 2, "positive": 0, "inconclusive": 1})

    def test_the_macro_is_the_unweighted_mean_over_the_languages_with_its_interval_and_median(self):
        specs = {"aa": (0.05, 0.40), "bb": (0.40, 0.05), "cc": (0.2, 0.3)}
        res = f4a.analyse_f4({"FP": merged("FP", specs, seed=1), "INT4": merged("INT4", specs, seed=2)}, self.out, n_boot=300)
        pts = [res["per_language"][c]["E_b"]["point"] for c in ("aa", "bb", "cc")]
        m = res["macro"]["E_b"]
        self.assertAlmostEqual(m["point"], float(np.mean(pts)), places=12); self.assertAlmostEqual(m["median"], float(np.median(pts)), places=12)
        self.assertLess(m["lo"], m["point"]); self.assertGreater(m["hi"], m["point"])
        self.assertNotAlmostEqual(m["point"], m["median"], places=3)
        dm = res["macro"]["delta"]; self.assertAlmostEqual(dm["point"], float(np.mean([res["per_language"][c]["delta"]["point"] for c in ("aa", "bb", "cc")])), places=12)

    def test_the_full_precision_run_is_required_and_exactly_one_other_precision(self):
        with self.assertRaises(ValueError):
            f4a.analyse_f4({"INT4": merged("INT4", {"aa": (0.1, 0.3)})}, self.out, n_boot=10)
        with self.assertRaises(ValueError):
            f4a.analyse_f4({"FP": merged("FP", {"aa": (0.1, 0.3)}), "INT4": merged("INT4", {"aa": (0.1, 0.3)}), "INT5": merged("INT5", {"aa": (0.1, 0.3)})}, self.out, n_boot=10)
        with self.assertRaises(ValueError):
            f4a.analyse_f4({"FP": merged("FP", {"aa": (0.1, 0.3)})}, self.out, n_boot=10)

    def test_the_languages_of_the_two_runs_must_agree(self):
        with self.assertRaises(ValueError):
            f4a.analyse_f4({"FP": merged("FP", {"aa": (0.1, 0.3), "bb": (0.1, 0.3)}), "INT4": merged("INT4", {"aa": (0.1, 0.3)})}, self.out, n_boot=10)


class Outputs(unittest.TestCase):
    def setUp(self):
        self.out = Path(tempfile.mkdtemp()) / "res"
        specs = {"aa": (0.05, 0.40), "bb": (0.1, 0.3)}
        self.res = f4a.analyse_f4({"FP": merged("FP", specs, seed=1), "INT4": merged("INT4", specs, seed=2)}, self.out, n_boot=100)

    def test_the_report_states_the_contrast_the_hypothesis_and_never_uses_retention_wording(self):
        txt = (self.out / "report.md").read_text()
        self.assertTrue(txt.startswith("# ")); self.assertIn("lost minus retained", txt); self.assertIn("H-F4", txt); self.assertIn("exploratory", txt.lower()); self.assertIn("negative", txt)
        for word in ("retention", "preserved", "amplified", "attenuated"):
            self.assertNotIn(word, txt.lower())
        self.assertIn("INT4", txt); self.assertIn("aa", txt)

    def test_the_json_and_the_csv_have_the_documented_schema(self):
        d = json.loads((self.out / "f4_results.json").read_text())
        for k in ("condition", "K", "n_boot", "seed", "languages", "per_language", "macro", "counts"):
            self.assertIn(k, d)
        with open(self.out / "f4_per_language.csv") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 2)
        for k in ("language", "E_b", "E_b_lo", "E_b_hi", "E_fp", "E_fp_lo", "E_fp_hi", "delta", "delta_lo", "delta_hi", "harm_lost_b", "harm_retained_b", "harm_lost_fp", "harm_retained_fp", "sign_b", "sign_fp"):
            self.assertIn(k, rows[0])
        self.assertEqual(rows[0]["language"], "aa"); self.assertAlmostEqual(float(rows[0]["E_b"]), self.res["per_language"]["aa"]["E_b"]["point"])
        p = self.res["per_language"]["aa"]
        for col, want in (("E_fp", p["E_fp"]["point"]), ("E_fp_lo", p["E_fp"]["lo"]), ("delta", p["delta"]["point"]), ("delta_hi", p["delta"]["hi"]), ("harm_lost_b", p["harm_lost_b"]), ("harm_retained_fp", p["harm_retained_fp"])):
            self.assertAlmostEqual(float(rows[0][col]), want, msg=col)
        self.assertEqual((rows[0]["sign_b"], rows[0]["sign_fp"]), (p["sign_b"], p["sign_fp"]))

    def test_main_reads_the_files_of_the_condition_and_writes_the_outputs(self):
        d = Path(tempfile.mkdtemp()); out = d / "o"
        specs = {"aa": (0.05, 0.40)}
        save_run(merged("FP", specs, seed=1), d / "func_lm_eval_FP.pt"); save_run(merged("INT4", specs, seed=2), d / "func_lm_eval_INT4.pt"); save_run(merged("INT5", specs, seed=3), d / "func_lm_eval_INT5.pt")
        f4a.main(["--runs", str(d), "--split", "eval", "--condition", "INT4", "--out", str(out), "--n-boot", "50"])
        r = json.loads((out / "f4_results.json").read_text())
        self.assertEqual(r["condition"], "INT4"); self.assertEqual(r["languages"], ["aa"])                  # INT5 in the same folder is ignored


if __name__ == "__main__":
    unittest.main()
