"""Stage-2 comparison of the GPTQ extension (freeze endpoint E6): the unchanged `phase1_transfer_report.py` is run once per arm; this tool puts the three arms' outputs side by side: the claim verdicts (UTILITY and the others),
the feature-level medians with their crossed-bootstrap intervals, and the group-level quantities per language, paired by language. The real legacy results (`reports/intervention_transfer/results/main`) are one arm in the tests;
the others are perturbed copies."""
import csv
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
G0 = HERE.parent
ROOT = G0.parents[1]
sys.path.insert(0, str(G0))
from gptq_ext import compare_transfer as ct  # noqa: E402

LEGACY = ROOT / "reports/intervention_transfer/results/main"
ROSTER = ROOT / "reports/g0/phase1_languages.json"


def perturbed_copy(dst, group_shift=0.0, point_shift=0.0, utility=None, drop_state=None):
    dst = Path(dst); dst.mkdir(parents=True, exist_ok=True)
    for name in ("claim_status.json", "component_summary.csv", "group_per_language.csv"):
        shutil.copy(LEGACY / name, dst / name)
    with open(dst / "group_per_language.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if r["state"] != "INT16":
            for k in ("C", "W", "I", "total", "native_total", "net_FP_FP", "native_minus_fixed"):
                r[k] = repr(float(r[k]) + group_shift)
    if drop_state:
        rows = [r for r in rows if r["state"] != drop_state]
    with open(dst / "group_per_language.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(dst / "component_summary.csv", newline="") as f:
        crows = list(csv.DictReader(f))
    for r in crows:
        if r["state"] != "INT16" and r["kind"] == "real":
            for k in ("point", "low", "high"):
                r[k] = repr(float(r[k]) + point_shift)
    with open(dst / "component_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(crows[0])); w.writeheader(); w.writerows(crows)
    if utility:
        c = json.loads((dst / "claim_status.json").read_text()); c["claims"]["UTILITY"]["per_state"] = dict(utility); c["claims"]["UTILITY"]["overall"] = "met" if all(v == "met" for v in utility.values()) else "not met"
        (dst / "claim_status.json").write_text(json.dumps(c))
    return dst


class Load(unittest.TestCase):
    def test_the_real_legacy_files_load_into_the_three_tables(self):
        arm = ct.load_arm(LEGACY)
        self.assertEqual(arm["claims"]["UTILITY"]["overall"], "not met")
        self.assertEqual(arm["claims"]["UTILITY"]["per_state"], {"INT4": "met", "INT6": "not met"})
        self.assertEqual(sorted({s for s, *_ in arm["feature"]}), ["INT4", "INT6"])
        self.assertEqual(len(arm["group"]), 2 * 27)
        self.assertAlmostEqual(arm["feature"][("INT4", "normalized_error")]["point"], 0.66411322, places=6)
        self.assertEqual(arm["feature"][("INT4", "normalized_error")]["n"], 2203)

    def test_the_floor_state_is_not_compared(self):
        arm = ct.load_arm(LEGACY)
        self.assertNotIn("INT16", {k[0] for k in arm["feature"]})
        self.assertNotIn("INT16", {k[0] for k in arm["group"]})


class Compare(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); d = Path(self.tmp.name)
        self.legacy = ct.load_arm(LEGACY)
        self.better = ct.load_arm(perturbed_copy(d / "better", group_shift=-0.01, point_shift=-0.05, utility={"INT4": "met", "INT6": "met"}))
        self.worse = ct.load_arm(perturbed_copy(d / "worse", group_shift=0.02, point_shift=0.1))
        self.roster = ct.load_roster(ROSTER)

    def tearDown(self):
        self.tmp.cleanup()

    def test_group_differences_are_per_language_a_minus_b(self):
        rows = ct.group_paired(self.better, self.legacy, self.roster)
        self.assertEqual(len(rows), 2 * 27 * len(ct.GROUP_COLUMNS))
        r = [x for x in rows if x["language"] == "eng_Latn" and x["state"] == "INT4" and x["column"] == "total"][0]
        self.assertAlmostEqual(r["diff"], -0.01, places=9)
        self.assertEqual({x["tier"] for x in rows}, {"high", "mid", "low"})

    def test_group_summary_has_counts_and_a_bootstrap_interval(self):
        rows = ct.group_paired(self.worse, self.legacy, self.roster)
        s = ct.group_summary(rows, n_boot=300, seed=0)
        r = [x for x in s if x["state"] == "INT6" and x["column"] == "W"][0]
        self.assertAlmostEqual(r["mean"], 0.02, places=9); self.assertEqual((r["n_languages"], r["n_positive"]), (27, 27))
        self.assertLessEqual(r["ci_low"], r["mean"] + 1e-12); self.assertGreaterEqual(r["ci_high"], r["mean"] - 1e-12)

    def test_feature_level_table_gives_points_intervals_overlap_and_the_difference(self):
        rows = ct.feature_side_by_side({"legacy RTN": self.legacy, "matched RTN": self.worse, "GPTQ": self.better})
        r = [x for x in rows if x["state"] == "INT4" and x["metric"] == "normalized_error"][0]
        self.assertAlmostEqual(r["legacy RTN"], 0.66411322, places=6)
        self.assertAlmostEqual(r["GPTQ"], 0.66411322 - 0.05, places=6); self.assertAlmostEqual(r["matched RTN"], 0.66411322 + 0.1, places=6)
        self.assertAlmostEqual(r["GPTQ minus legacy RTN"], -0.05, places=6)
        self.assertFalse(r["intervals_overlap GPTQ vs matched RTN"])
        self.assertTrue(all(x["n_equal"] for x in rows))

    def test_a_different_eligible_item_count_is_flagged_not_hidden(self):
        other = ct.load_arm(perturbed_copy(Path(self.tmp.name) / "o", point_shift=0.0))
        other["feature"][("INT4", "normalized_error")]["n"] = 2000
        rows = ct.feature_side_by_side({"legacy RTN": self.legacy, "matched RTN": other, "GPTQ": self.better})
        r = [x for x in rows if x["state"] == "INT4" and x["metric"] == "normalized_error"][0]
        self.assertFalse(r["n_equal"])

    def test_claims_are_listed_side_by_side_with_per_state_verdicts(self):
        rows = ct.claims_side_by_side({"legacy RTN": self.legacy, "matched RTN": self.worse, "GPTQ": self.better})
        u = [r for r in rows if r["claim"] == "UTILITY" and r["scope"] == "overall"][0]
        self.assertEqual((u["legacy RTN"], u["matched RTN"], u["GPTQ"]), ("not met", "not met", "met"))
        u6 = [r for r in rows if r["claim"] == "UTILITY" and r["scope"] == "INT6"][0]
        self.assertEqual((u6["legacy RTN"], u6["GPTQ"]), ("not met", "met"))

    def test_a_missing_feature_level_row_in_one_arm_is_refused(self):
        other = ct.load_arm(perturbed_copy(Path(self.tmp.name) / "mf"))
        other["feature"].pop(("INT6", "cosine"))
        with self.assertRaises(ValueError):
            ct.feature_side_by_side({"legacy RTN": self.legacy, "matched RTN": other, "GPTQ": self.better})

    def test_a_duplicated_feature_level_row_is_refused(self):
        d = perturbed_copy(Path(self.tmp.name) / "dup")
        with open(d / "component_summary.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        extra = [r for r in rows if r["state"] == "INT4" and r["kind"] == "real" and r["split"] == "all" and r["metric"] == "cosine"][0]
        rows.append(dict(extra))
        with open(d / "component_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        with self.assertRaises(ValueError):
            ct.load_arm(d)

    def test_arms_with_different_states_or_languages_are_refused(self):
        d = Path(self.tmp.name)
        missing = ct.load_arm(perturbed_copy(d / "m", drop_state="INT6"))
        with self.assertRaises(ValueError):
            ct.group_paired(missing, self.legacy, self.roster)

    def test_the_report_names_the_arms_the_contrasts_and_the_claims_and_writes_the_tables(self):
        out = Path(self.tmp.name) / "out"
        ct.write_report(out, {"legacy RTN": self.legacy, "matched RTN": self.worse, "GPTQ": self.better}, self.roster, n_boot=100, seed=0)
        text = (out / "transfer_comparison.md").read_text()
        for needle in ("GPTQ minus matched RTN", "matched RTN minus legacy RTN", "GPTQ minus legacy RTN", "UTILITY", "Exploratory", "INT4", "INT6"):
            self.assertIn(needle, text)
        for f in ("transfer_claims.csv", "transfer_feature_level.csv", "transfer_group_paired.csv", "transfer_group_summary.csv"):
            self.assertTrue((out / f).is_file(), f)


if __name__ == "__main__":
    unittest.main()
