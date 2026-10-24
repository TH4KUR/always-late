import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phase1_gain_baseline as gb

# Toy panel: 3 languages (a, b, c), 8 features, 10,000 tokens per language.
#   f0: fires only on a (selective for a)          f1: fires only on b (selective for b)
#   f2: fires everywhere equally (general)         f3: never fires (unsupported)
#   f4: tied peak a = b, nothing on c              f5: selective for a, second member of a's group
#   f6: fires on c only                            f7: weak, 10 positives only (low support)
LANGS = ["a", "b", "c"]
NTOK = np.array([10000, 10000, 10000])
COUNTS = np.array([
    [900, 0, 500, 0, 300, 400, 0, 10],
    [0, 800, 500, 0, 300, 0, 0, 0],
    [0, 0, 500, 0, 0, 0, 700, 0]])
MEAN = np.array([
    [2.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.1],
    [0.0, 3.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 4.0, 0.0]])
SUMS = MEAN * NTOK[:, None]
CUTOFF = 1.4


def classes(counts=COUNTS):
    return gb.state_classes(counts, NTOK, CUTOFF)


class StateClasses(unittest.TestCase):
    def test_every_feature_has_exactly_one_label_per_language(self):
        cls = classes()
        for l in range(3):
            lab = gb.labels_for_language(cls, l)
            self.assertEqual(lab.shape, (8,))
            self.assertTrue(set(lab) <= {"own", "other_selective", "general", "unsupported"})

    def test_labels_of_the_toy_panel(self):
        cls = classes()
        a = list(gb.labels_for_language(cls, 0))
        self.assertEqual(a[0], "own"); self.assertEqual(a[1], "other_selective"); self.assertEqual(a[2], "general")
        self.assertEqual(a[3], "unsupported")                       # never fires
        self.assertEqual(a[5], "own")
        self.assertEqual(a[7], "unsupported")                       # 10 positives: below the support rule

    def test_a_tied_peak_is_selective_for_nobody(self):
        cls = classes()
        for l in range(3):
            self.assertNotEqual(gb.labels_for_language(cls, l)[4], "own")
        self.assertFalse(cls["selective"][4])
        self.assertEqual(gb.labels_for_language(cls, 0)[4], "general")   # supported, no unique selective peak

    def test_cutoff_is_strict(self):
        S = classes()["S"][5]                                       # 3 in exact arithmetic (one language of three); a cutoff equal to S does not select the feature
        self.assertTrue(gb.state_classes(COUNTS, NTOK, np.nextafter(S, 0.0))["selective"][5])
        self.assertFalse(gb.state_classes(COUNTS, NTOK, S)["selective"][5])

    def test_partition_counts(self):
        cls = classes()
        self.assertEqual(int(cls["own"][0].sum()), 2)               # f0, f5
        self.assertEqual(int(cls["own"][1].sum()), 1)               # f1
        self.assertEqual(int(cls["own"][2].sum()), 1)               # f6
        self.assertEqual(int(cls["supported"].sum()), 6)            # f0 f1 f2 f4 f5 f6


GROUPS = {"a": {"all": [0, 5]}, "b": {"all": [1]}, "c": {"all": [6]}}
SEL = {"selective": np.array([1, 1, 0, 0, 0, 1, 1, 0], bool), "controls": np.array([2, 4]), "null_cutoff": CUTOFF}


def profile(sums=SUMS, counts=COUNTS):
    return {"n_tokens": NTOK, "positive_counts": counts, "activation_sums": sums}


class Rows(unittest.TestCase):
    def rows(self, sums=SUMS, counts=COUNTS, K=1):
        fp = profile()
        return gb.baseline_rows({"FP": fp, "INT4": profile(sums, counts)}, SEL, GROUPS, LANGS, K=K)

    def test_one_row_per_language_and_state(self):
        rows = self.rows()
        self.assertEqual(len(rows), 6)
        self.assertEqual({(r["language"], r["state"]) for r in rows}, {(l, s) for l in LANGS for s in ("FP", "INT4")})

    def test_fp_against_itself_has_ratio_one(self):
        for r in self.rows():
            if r["state"] == "FP":
                self.assertAlmostEqual(r["g_ratio"], 1.0)

    def test_halved_group_activation_gives_ratio_half(self):
        s = SUMS.copy(); s[0, 0] *= 0.5                              # group of a (K=1) is f0
        r = [x for x in self.rows(sums=s) if x["state"] == "INT4" and x["language"] == "a"][0]
        self.assertAlmostEqual(r["g_ratio"], 0.5)
        self.assertAlmostEqual(r["group_mass"], 1.0)

    def test_total_ratio_is_the_whole_language_mass_over_fp(self):
        rows = self.rows(sums=SUMS * 0.5)
        r = [x for x in rows if x["state"] == "INT4" and x["language"] == "a"][0]
        self.assertAlmostEqual(r["total_ratio"], 0.5); self.assertAlmostEqual(r["g_rel"], 1.0)
        fp = [x for x in rows if x["state"] == "FP" and x["language"] == "a"][0]
        self.assertAlmostEqual(fp["total_ratio"], 1.0); self.assertAlmostEqual(fp["controls_ratio"], 1.0)

    def test_controls_ratio_follows_the_control_pool_only(self):
        s = SUMS.copy(); s[0, 2] *= 0.5; s[0, 4] *= 0.5              # the pool is f2 and f4
        r = [x for x in self.rows(sums=s) if x["state"] == "INT4" and x["language"] == "a"][0]
        self.assertAlmostEqual(r["controls_ratio"], 0.5)
        self.assertAlmostEqual(r["g_ratio"], 1.0)
        self.assertAlmostEqual(r["total_ratio"], (5.1 - 1.0) / 5.1)
        self.assertAlmostEqual(r["g_rel"], 1.0 / ((5.1 - 1.0) / 5.1))

    def test_group_is_the_first_k_members_in_manifest_order(self):
        r1 = [x for x in self.rows(K=1) if x["state"] == "FP" and x["language"] == "a"][0]
        r2 = [x for x in self.rows(K=2) if x["state"] == "FP" and x["language"] == "a"][0]
        self.assertAlmostEqual(r1["group_mass"], 2.0)
        self.assertAlmostEqual(r2["group_mass"], 3.0)

    def test_mass_shares_sum_to_one(self):
        for r in self.rows():
            tot = r["share_group"] + r["share_cohort_rest"] + r["share_other_selective"] + r["share_controls"] + r["share_rest"]
            self.assertAlmostEqual(tot, 1.0)

    def test_mass_shares_of_the_toy_language_a(self):
        r = [x for x in self.rows(K=1) if x["state"] == "FP" and x["language"] == "a"][0]
        total = MEAN[0].sum()                                        # 2 + 1 + 1 + 1 + 0.1 = 5.1
        self.assertAlmostEqual(r["share_group"], 2.0 / total)        # f0
        self.assertAlmostEqual(r["share_cohort_rest"], 1.0 / total)  # f5
        self.assertAlmostEqual(r["share_other_selective"], 0.0)      # f1, f6 do not fire on a
        self.assertAlmostEqual(r["share_controls"], 2.0 / total)     # f2 and f4
        self.assertAlmostEqual(r["share_rest"], 0.1 / total)         # f7 (unsupported at dev)

    def test_shares_sum_to_one_even_if_the_control_list_overlaps_a_cohort(self):
        sel = dict(SEL, controls=np.array([2, 4, 0]))                # f0 is a's group member: it must be counted once
        rows = gb.baseline_rows({"FP": profile()}, sel, GROUPS, LANGS, K=1)
        for r in rows:
            self.assertAlmostEqual(r["share_group"] + r["share_cohort_rest"] + r["share_other_selective"] + r["share_controls"] + r["share_rest"], 1.0)

    def test_state_counts_and_overlap(self):
        r = [x for x in self.rows(K=1) if x["state"] == "FP" and x["language"] == "a"][0]
        self.assertEqual(r["n_selective_state"], 2)
        self.assertEqual(r["n_cohort_fp"], 2)
        self.assertEqual(r["n_overlap"], 2)
        self.assertAlmostEqual(r["cohort_kept"], 1.0)
        self.assertEqual(r["n_new"], 0)

    def test_a_group_member_whose_peak_moves_is_not_retained(self):
        c = COUNTS.copy(); c[1, 0] = 2000                            # f0 now peaks on b in the quantized state
        r = [x for x in self.rows(counts=c, K=1) if x["state"] == "INT4" and x["language"] == "a"][0]
        self.assertAlmostEqual(r["group_peak_retained"], 0.0)
        self.assertEqual(r["n_overlap"], 1)                          # only f5 of a's cohort is still its own
        self.assertAlmostEqual(r["cohort_kept"], 0.5)

    def test_new_selective_features_are_counted(self):
        c = COUNTS.copy(); c[0, 4] = 900                             # f4 (general in FP) now peaks on a alone
        r = [x for x in self.rows(counts=c, K=1) if x["state"] == "INT4" and x["language"] == "a"][0]
        self.assertEqual(r["n_new"], 1)
        self.assertEqual(r["n_selective_state"], 3)

    def test_zero_fp_mass_gives_nan_ratio(self):
        fp = profile(SUMS * 0.0)
        rows = gb.baseline_rows({"FP": fp, "INT4": profile()}, SEL, GROUPS, LANGS, K=1)
        r = [x for x in rows if x["state"] == "INT4" and x["language"] == "a"][0]
        self.assertTrue(np.isnan(r["g_ratio"]))

    def test_fp_state_is_required(self):
        with self.assertRaises(ValueError):
            gb.baseline_rows({"INT4": profile()}, SEL, GROUPS, LANGS, K=1)

    def test_language_order_must_match_the_groups(self):
        with self.assertRaises(ValueError):
            gb.baseline_rows({"FP": profile()}, SEL, {"a": {"all": [0]}}, LANGS, K=1)


class Hg7(unittest.TestCase):
    def rows(self, g, kept):
        out = []
        for s, gg, kk in zip(("FP", "INT8", "INT4"), g, kept):
            for l in range(27):
                out.append({"language": f"l{l}", "state": s, "g_ratio": gg, "group_peak_retained": kk, "share_group": 0.1 * gg})
        return out

    def test_ordered_and_declining_passes(self):
        res = gb.hg7(self.rows((1.0, 0.98, 0.7), (0.8, 0.79, 0.5)), order=("FP", "INT8", "INT4"))
        self.assertTrue(res["met"])
        self.assertEqual(res["languages_declining_at_last_state"], {"g_ratio": 27, "group_peak_retained": 27, "share_group": 27})

    def test_a_rise_is_not_met(self):
        res = gb.hg7(self.rows((1.0, 1.1, 1.3), (0.8, 0.79, 0.5)), order=("FP", "INT8", "INT4"))
        self.assertFalse(res["met"])
        self.assertEqual(res["languages_declining_at_last_state"]["g_ratio"], 0)

    def test_median_not_ordered_is_not_met(self):
        res = gb.hg7(self.rows((1.0, 0.7, 0.9), (0.8, 0.79, 0.5)), order=("FP", "INT8", "INT4"))
        self.assertFalse(res["met"])

    def test_tied_medians_are_still_ordered(self):
        res = gb.hg7(self.rows((1.0, 1.0, 0.7), (0.8, 0.8, 0.5)), order=("FP", "INT8", "INT4"))
        self.assertTrue(res["met"])

    def test_equal_values_are_not_a_decline(self):
        rows = self.rows((1.0, 0.98, 0.7), (0.8, 0.79, 0.5)); n = 0
        for r in rows:
            if r["state"] == "INT4" and n < 4:
                r["share_group"] = 0.1; n += 1                       # equal to FP: not below it
        res = gb.hg7(rows, order=("FP", "INT8", "INT4"))
        self.assertEqual(res["languages_declining_at_last_state"]["share_group"], 23)
        self.assertFalse(res["met"])

    def test_a_language_without_a_value_does_not_spoil_the_median(self):
        rows = self.rows((1.0, 0.98, 0.7), (0.8, 0.79, 0.5))
        for r in rows:
            if r["language"] == "l0":
                r["share_group"] = float("nan"); r["g_ratio"] = float("nan")
        res = gb.hg7(rows, order=("FP", "INT8", "INT4"))
        self.assertTrue(res["met"])
        self.assertEqual(res["languages_declining_at_last_state"]["share_group"], 26)

    def test_the_reported_ratio_does_not_enter_the_rule(self):
        rows = self.rows((1.0, 0.98, 0.7), (0.8, 0.79, 0.5))
        for r in rows:
            if r["state"] == "INT4":
                r["g_ratio"] = 1.3                                   # rises, while share and retained fraction decline
        res = gb.hg7(rows, order=("FP", "INT8", "INT4"))
        self.assertTrue(res["met"]); self.assertFalse(res["ordered"]["g_ratio"])

    def test_threshold_is_24_of_27(self):
        rows = self.rows((1.0, 0.98, 0.7), (0.8, 0.79, 0.5))
        bad = 0
        for r in rows:
            if r["state"] == "INT4" and bad < 4:
                r["g_ratio"] = 1.2; r["share_group"] = 0.12; bad += 1
        self.assertFalse(gb.hg7(rows, order=("FP", "INT8", "INT4"))["met"])      # 23 of 27 decline
        rows2 = self.rows((1.0, 0.98, 0.7), (0.8, 0.79, 0.5)); bad = 0
        for r in rows2:
            if r["state"] == "INT4" and bad < 3:
                r["g_ratio"] = 1.2; r["share_group"] = 0.12; bad += 1
        self.assertTrue(gb.hg7(rows2, order=("FP", "INT8", "INT4"))["met"])      # 24 of 27 decline


class Outputs(unittest.TestCase):
    def setUp(self):
        fp = profile(); q = profile(SUMS * 0.5)
        self.rows = gb.baseline_rows({"FP": fp, "INT4": q}, SEL, GROUPS, LANGS, K=1)
        self.res = dict(gb.hg7(self.rows, order=("FP", "INT4"), need=1), met=True)

    def test_report_names_the_verdict_the_states_and_the_inputs(self):
        text = gb.report_text(self.rows, self.res, {"profiles_sha256": "abc123", "selectivity_sha256": "def456", "K": 1})
        self.assertIn("H-G7", text); self.assertIn("abc123", text); self.assertIn("def456", text)
        self.assertIn("| FP |", text); self.assertIn("| INT4 |", text)
        self.assertIn("met", text)
        self.assertNotIn("not met", text)
        self.assertIn("not met", gb.report_text(self.rows, dict(self.res, met=False), {"profiles_sha256": "a", "selectivity_sha256": "b", "K": 1}))

    def test_report_lists_every_language(self):
        text = gb.report_text(self.rows, self.res, {"profiles_sha256": "a", "selectivity_sha256": "b", "K": 1})
        for l in LANGS:
            self.assertIn(f"| {l} |", text)

    def test_write_outputs_csv_and_json(self):
        import csv, json, tempfile
        with tempfile.TemporaryDirectory() as d:
            gb.write_outputs(d, self.rows, self.res, {"profiles_sha256": "a", "selectivity_sha256": "b", "K": 1})
            got = list(csv.DictReader(open(Path(d) / "baseline.csv")))
            self.assertEqual(len(got), len(self.rows))
            self.assertEqual(set(got[0]), set(self.rows[0]))
            js = json.loads((Path(d) / "baseline.json").read_text())
            self.assertEqual(js["hg7"]["met"], True); self.assertEqual(len(js["rows"]), len(self.rows))
            self.assertTrue((Path(d) / "report.md").exists())

    def test_nan_is_written_as_null_in_json(self):
        import json, tempfile
        rows = [dict(r) for r in self.rows]; rows[0]["g_ratio"] = float("nan")
        with tempfile.TemporaryDirectory() as d:
            gb.write_outputs(d, rows, self.res, {"profiles_sha256": "a", "selectivity_sha256": "b", "K": 1})
            js = json.loads((Path(d) / "baseline.json").read_text())
            self.assertIsNone(js["rows"][0]["g_ratio"])


class CommandLine(unittest.TestCase):
    def test_main_reads_the_files_and_writes_the_outputs(self):
        import json, tempfile, torch
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            torch.save({"schema_version": 1, "languages": LANGS, "feature_ids": torch.arange(8),
                        "conditions": {s: {"n_tokens": torch.tensor(NTOK), "positive_counts": torch.tensor(COUNTS), "activation_sums": torch.tensor(SUMS), "activation_max": torch.tensor(SUMS)} for s in ("FP", "INT4")}},
                       d / "profiles.pt")
            torch.save({"selective": torch.tensor(SEL["selective"]), "controls": torch.tensor(SEL["controls"]), "null_cutoff": CUTOFF, "languages": LANGS}, d / "selectivity.pt")
            (d / "manifest.json").write_text(json.dumps({"languages": LANGS, "groups": GROUPS}))
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()):
                gb.main(["--profiles", str(d / "profiles.pt"), "--selectivity", str(d / "selectivity.pt"), "--manifest", str(d / "manifest.json"), "--out", str(d / "out"), "--K", "1", "--states", "FP", "INT4"])
            self.assertTrue((d / "out" / "baseline.csv").exists())
            js = json.loads((d / "out" / "baseline.json").read_text())
            self.assertEqual(len(js["rows"]), 6)
            self.assertEqual(len(js["meta"]["profiles_sha256"]), 64)

    def test_main_refuses_profiles_in_another_language_order(self):
        import json, tempfile, torch
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            torch.save({"schema_version": 1, "languages": ["b", "a", "c"], "feature_ids": torch.arange(8),
                        "conditions": {"FP": {"n_tokens": torch.tensor(NTOK), "positive_counts": torch.tensor(COUNTS), "activation_sums": torch.tensor(SUMS), "activation_max": torch.tensor(SUMS)}}}, d / "profiles.pt")
            torch.save({"selective": torch.tensor(SEL["selective"]), "controls": torch.tensor(SEL["controls"]), "null_cutoff": CUTOFF, "languages": LANGS}, d / "selectivity.pt")
            (d / "manifest.json").write_text(json.dumps({"languages": LANGS, "groups": GROUPS}))
            with self.assertRaises(ValueError):
                gb.main(["--profiles", str(d / "profiles.pt"), "--selectivity", str(d / "selectivity.pt"), "--manifest", str(d / "manifest.json"), "--out", str(d / "out"), "--K", "1", "--states", "FP"])


    def _files(self, d, sel_langs):
        import json, torch
        torch.save({"schema_version": 1, "languages": LANGS, "feature_ids": torch.arange(8),
                    "conditions": {s: {"n_tokens": torch.tensor(NTOK), "positive_counts": torch.tensor(COUNTS), "activation_sums": torch.tensor(SUMS), "activation_max": torch.tensor(SUMS)} for s in ("FP", "INT4")}}, d / "profiles.pt")
        torch.save({"selective": torch.tensor(SEL["selective"]), "controls": torch.tensor(SEL["controls"]), "null_cutoff": CUTOFF, "languages": sel_langs}, d / "selectivity.pt")
        (d / "manifest.json").write_text(json.dumps({"languages": LANGS, "groups": GROUPS}))
        return ["--profiles", str(d / "profiles.pt"), "--selectivity", str(d / "selectivity.pt"), "--manifest", str(d / "manifest.json"), "--out", str(d / "out"), "--K", "1"]

    def test_main_refuses_a_selectivity_file_in_another_language_order(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                gb.main(self._files(Path(d), ["b", "a", "c"]) + ["--states", "FP", "INT4"])

    def test_main_refuses_a_state_that_the_profiles_lack(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                gb.main(self._files(Path(d), LANGS) + ["--states", "FP", "INT5"])

if __name__ == "__main__":
    unittest.main()
