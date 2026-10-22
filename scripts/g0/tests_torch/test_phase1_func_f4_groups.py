import csv
import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phase1_func_f4_groups as f4


def feat(fid, lang, status, x1, x2, fp="retained", cond="INT4"):
    """A status row; the activity of the feature at the precision is rate = 10**x1 and mean = 10**x2 (None = inactive: rate and mean 0)."""
    inactive = x1 is None
    return {"language": lang, "feature_id": fid, "condition": cond, "fp_status": fp, "status": status, "rate": 0.0 if inactive else 10.0 ** x1, "mean": 0.0 if inactive else 10.0 ** x2}


def pool(lang, n_lost, n_ret, seed=0, spread=1.0, start=0, cond="INT4"):
    """n_lost lost and n_ret retained features of one language with similar activity (uniform random coordinates)."""
    g = np.random.default_rng(seed); rows = []
    for k in range(n_lost):
        rows.append(feat(start + k, lang, "different_peak", *g.uniform(-3, -3 + spread, 2), cond=cond))
    for k in range(n_ret):
        rows.append(feat(start + 10000 + k, lang, "retained", *g.uniform(-3, -3 + spread, 2), cond=cond))
    return rows


class Match(unittest.TestCase):
    def test_nearest_neighbour_without_replacement(self):
        pl = {10: (0.0, 0.0), 11: (1.0, 0.0), 12: (5.0, 5.0)}
        got = f4.match([(1, (0.1, 0.0)), (2, (0.2, 0.0))], pl, (1.0, 1.0), caliper=10.0)
        self.assertEqual(got, {1: 10, 2: 11})                                                                # the second lost feature cannot reuse 10

    def test_the_caliper_leaves_far_features_unmatched(self):
        self.assertEqual(f4.match([(1, (0.0, 0.0))], {10: (3.0, 0.0)}, (1.0, 1.0), caliper=1.0), {})
        self.assertEqual(f4.match([(1, (0.0, 0.0))], {10: (0.9, 0.0)}, (1.0, 1.0), caliper=1.0), {1: 10})

    def test_the_distance_is_standardised_per_axis(self):
        pl = {1: (1.0, 0.0), 2: (0.0, 1.0)}
        self.assertEqual(f4.match([(9, (0.0, 0.0))], pl, (10.0, 1.0), caliper=9.0), {9: 1})                  # axis 1 is wide: a unit step there is small
        self.assertEqual(f4.match([(9, (0.0, 0.0))], pl, (1.0, 10.0), caliper=9.0), {9: 2})
        mirror = {2: (1.0, 0.0), 1: (0.0, 1.0)}                                                             # the same geometry with the ids swapped: an unstandardised tie would go to the wrong id
        self.assertEqual(f4.match([(9, (0.0, 0.0))], mirror, (10.0, 1.0), caliper=9.0), {9: 2})
        self.assertEqual(f4.match([(9, (0.0, 0.0))], mirror, (1.0, 10.0), caliper=9.0), {9: 1})

    def test_ties_go_to_the_smaller_feature_id_and_order_is_the_given_order(self):
        pl = {5: (1.0, 0.0), 3: (-1.0, 0.0)}
        self.assertEqual(f4.match([(9, (0.0, 0.0))], pl, (1.0, 1.0), caliper=9.0), {9: 3})


class BuildCondition(unittest.TestCase):
    def build(self, rows, **kw):
        args = dict(cap=64, R=3, min_pairs=8, caliper=3.0, seed=0); args.update(kw)                       # a wide caliper unless a test is about the caliper: random planted data must not make tests flaky
        return f4.build_condition(rows, "INT4", ["aa"], **args)

    def test_only_established_features_enter_the_pools(self):
        rows = pool("aa", 12, 60)
        rows += [feat(500, "aa", "different_peak", -2.5, -2.5, fp="different_peak"), feat(501, "aa", "retained", -2.5, -2.5, fp="low_support")]       # never established in full precision
        c = self.build(rows)["aa"]
        used = set(c["lost"]) | {i for d in c["draws"] for i in d}
        self.assertNotIn(500, used); self.assertNotIn(501, used); self.assertEqual(c["n_lost_total"], 12)

    def test_lost_features_that_are_inactive_at_the_precision_are_excluded_and_counted(self):
        rows = pool("aa", 12, 60) + [feat(700 + k, "aa", "low_support", None, None) for k in range(5)]
        c = self.build(rows)["aa"]
        self.assertEqual((c["n_lost_total"], c["n_inactive_lost"]), (17, 5)); self.assertTrue(all(i < 700 for i in c["lost"]))

    def test_the_draws_are_disjoint_from_each_other_and_from_the_lost_features_and_have_the_size_of_the_lost_subset(self):
        c = self.build(pool("aa", 10, 80))["aa"]
        self.assertEqual(c["n_selected"], len(c["lost"])); self.assertEqual(len(c["draws"]), 3)
        allr = [i for d in c["draws"] for i in d]
        self.assertEqual(len(set(allr)), len(allr)); self.assertFalse(set(allr) & set(c["lost"]))
        self.assertTrue(all(len(d) == c["n_selected"] for d in c["draws"]))

    def test_a_draw_that_cannot_match_every_lost_feature_is_dropped(self):
        c = self.build(pool("aa", 10, 15), caliper=50.0)["aa"]                                              # everything is within the caliper; 15 retained give one full draw of 10, not a second
        self.assertEqual(c["n_selected"], 10); self.assertEqual(len(c["draws"]), 1)

    def test_the_cap_is_taken_by_a_seeded_hash_of_the_feature_id_not_by_id_or_outcome(self):
        rows = pool("aa", 100, 400, spread=0.6)
        c = self.build(rows, cap=20)["aa"]; c2 = self.build(rows, cap=20)["aa"]; c3 = self.build(rows, cap=20, seed=1)["aa"]
        self.assertEqual(len(c["lost"]), 20); self.assertEqual(c["lost"], c2["lost"]); self.assertNotEqual(sorted(c["lost"]), sorted(c3["lost"]))
        self.assertNotEqual(sorted(c["lost"]), list(range(20)))
        key = lambda i: hashlib.sha256(f"0:INT4:aa:{i}".encode()).hexdigest()
        self.assertEqual(c["n_matched"], 100)                                                               # all 100 were matchable, so the candidates are the ids 0 to 99
        self.assertEqual(sorted(c["lost"]), sorted(sorted(range(100), key=key)[:20]))                       # the first 20 of the hash order

    def test_lost_features_are_processed_in_decreasing_activity_so_the_more_active_one_gets_the_only_partner(self):
        rows = [feat(1, "aa", "different_peak", 1.0, 1.0), feat(2, "aa", "different_peak", 0.0, 1.0), feat(50, "aa", "retained", 0.5, 1.0)]
        c = self.build(rows, min_pairs=1, R=1)["aa"]
        self.assertEqual(c["lost"], [1]); self.assertEqual(c["draws"], [[50]])

    def test_exactly_min_pairs_matched_features_are_enough(self):
        self.assertEqual(self.build(pool("aa", 8, 60))["aa"]["n_selected"], 8); self.assertIsNone(self.build(pool("aa", 8, 60))["aa"]["excluded"])
        self.assertIsNotNone(self.build(pool("aa", 7, 60))["aa"]["excluded"])

    def test_every_non_retained_status_counts_as_lost(self):
        rows = [feat(k, "aa", s, -3.0 + 0.01 * k, -3.0) for k, s in enumerate(["different_peak", "low_support", "ambiguous_peak", "inactive"] * 3)] + [feat(100 + k, "aa", "retained", -3.0 + 0.01 * k, -3.0) for k in range(60)]
        self.assertEqual(self.build(rows)["aa"]["n_lost_total"], 12)

    def test_a_language_with_too_few_matched_pairs_is_excluded_with_a_reason(self):
        out = self.build(pool("aa", 5, 60))
        self.assertEqual(out["aa"]["lost"], []); self.assertEqual(out["aa"]["draws"], []); self.assertIn("fewer than 8", out["aa"]["excluded"])

    def test_balance_is_the_standardised_mean_difference_of_the_native_activity(self):
        rows = [feat(k, "aa", "different_peak", -3 + 0.01 * k, -3) for k in range(10)] + [feat(100 + k, "aa", "retained", -3 + 0.01 * k, -3) for k in range(10)]
        rows += [feat(300 + k, "aa", "retained", 1.0 + k, 1.0) for k in range(30)]                          # far-away retained features that must not be matched (they widen the SD only)
        c = self.build(rows, caliper=0.5, min_pairs=5)["aa"]
        self.assertEqual(c["n_selected"], 10)
        self.assertLess(abs(c["smd_x1"]), 1e-9); self.assertLess(abs(c["smd_x2"]), 1e-9)                    # identical coordinates: perfectly balanced
        shifted = [feat(k, "aa", "different_peak", -3.0, -3.0) for k in range(10)] + [feat(100 + k, "aa", "retained", -2.9, -3.0) for k in range(10)] + [feat(300 + k, "aa", "retained", 1.0 + k, 1.0) for k in range(30)]
        c2 = self.build(shifted, caliper=0.5, min_pairs=5)["aa"]
        x1 = np.array([-3.0] * 10 + [-2.9] * 10 + [1.0 + k for k in range(30)])                              # the pooled SD is that of ALL established features with native activity
        self.assertAlmostEqual(c2["smd_x1"], (-3.0 - (-2.9)) / float(x1.std()), places=9)                   # sign: lost minus retained is negative
        self.assertLess(c2["smd_x1"], 0.0); self.assertLess(abs(c2["smd_x2"]), 1e-9)

    def test_each_language_is_built_separately(self):
        rows = pool("aa", 12, 60) + pool("bb", 12, 60, seed=5, start=2000)
        out = f4.build_condition(rows, "INT4", ["aa", "bb"], cap=64, R=3, min_pairs=8, caliper=3.0, seed=0)
        self.assertEqual(set(out), {"aa", "bb"}); self.assertFalse(set(out["aa"]["lost"]) & set(out["bb"]["lost"]))

    def test_only_rows_of_the_requested_condition_are_used(self):
        rows = pool("aa", 12, 60, cond="INT4") + pool("aa", 12, 60, cond="INT6", seed=3)
        c4 = self.build(rows)["aa"]
        self.assertEqual(c4["n_lost_total"], 12)


def roster_rows():
    return pool("aa", 12, 60) + pool("bb", 3, 60, seed=5, start=2000)


class Manifest(unittest.TestCase):
    def manifest(self, **kw):
        args = dict(condition="INT4", cap=64, R=3, min_pairs=8, caliper=3.0, seed=0); args.update(kw)
        return f4.build_manifest(roster_rows(), ["aa", "bb"], source_sha="abc", **args)

    def test_schema_of_the_groups_manifest_so_that_the_runner_reads_it_unchanged(self):
        m = self.manifest()
        self.assertEqual(m["languages"], ["aa", "bb"]); self.assertEqual(m["source"], {"sha256": "abc"}); self.assertEqual(m["schema"], 1)
        n = len(m["groups"]["aa"]["all"])
        self.assertEqual(m["sizes"]["aa"], {"K16": min(16, n), "K64": n, "all": n})
        self.assertEqual(len(m["controls"]["aa"]["K64"]), 3); self.assertTrue(all(len(d) == n for d in m["controls"]["aa"]["K64"]))
        self.assertTrue(all(isinstance(i, int) for i in m["groups"]["aa"]["all"]))
        json.dumps(m)                                                                                       # plain JSON

    def test_an_excluded_language_has_empty_lists_and_is_reported_in_the_f4_block(self):
        m = self.manifest()
        self.assertEqual(m["groups"]["bb"]["all"], []); self.assertEqual(m["controls"]["bb"]["K64"], []); self.assertEqual(m["sizes"]["bb"]["K64"], 0)
        self.assertEqual(m["f4"]["included"], ["aa"]); self.assertIn("bb", m["f4"]["excluded"]); self.assertEqual(m["f4"]["condition"], "INT4")
        self.assertIn("fewer than 8", m["f4"]["excluded"]["bb"])

    def test_the_manifest_is_deterministic(self):
        self.assertEqual(json.dumps(self.manifest(), sort_keys=True), json.dumps(self.manifest(), sort_keys=True))

    def test_the_unchanged_runner_builds_the_f4_contrast_from_it(self):
        import phase1_func_run as fr
        m = self.manifest()
        cfgs = {c["name"]: c for c in fr.build_configs(m, ["aa"], ["aa"], [], fr.PRESETS["floor"], off_target_label="K64")}
        self.assertEqual(cfgs["zero_K64@aa"]["idx"].tolist(), m["groups"]["aa"]["all"])                      # the group is the LOST subset
        for r in range(3):
            self.assertEqual(cfgs[f"ctrl{r}_K64@aa"]["idx"].tolist(), m["controls"]["aa"]["K64"][r])         # the controls are the matched RETAINED draws
        self.assertIn("sham_K64@aa", cfgs)

    def test_the_condition_is_recorded_and_rows_of_other_conditions_are_ignored(self):
        rows = roster_rows() + pool("aa", 12, 60, cond="INT6", seed=9, start=7000)
        m = f4.build_manifest(rows, ["aa", "bb"], condition="INT4", source_sha="abc", cap=64, R=3, min_pairs=8, caliper=3.0, seed=0)
        self.assertTrue(all(i < 7000 for i in m["groups"]["aa"]["all"]))


class Files(unittest.TestCase):
    def test_the_status_file_is_read_with_native_activity_and_only_for_the_selective_group(self):
        d = Path(tempfile.mkdtemp()); f = d / "feature_status.csv"
        with open(f, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["feature_id", "group", "original_language", "condition", "fp_status", "status", "target_rate", "target_mean"]); w.writeheader()
            w.writerow({"feature_id": 7, "group": "selective", "original_language": "aa", "condition": "INT4", "fp_status": "retained", "status": "different_peak", "target_rate": "0.002", "target_mean": "0.05"})
            w.writerow({"feature_id": 8, "group": "controls", "original_language": "aa", "condition": "INT4", "fp_status": "retained", "status": "retained", "target_rate": "0.1", "target_mean": "0.2"})
        rows = f4.read_status(f)
        self.assertEqual(len(rows), 1); self.assertEqual((rows[0]["feature_id"], rows[0]["rate"], rows[0]["mean"], rows[0]["status"]), (7, 0.002, 0.05, "different_peak"))

    def test_main_writes_a_manifest_per_condition_and_a_report(self):
        d = Path(tempfile.mkdtemp()); f = d / "feature_status.csv"; out = d / "out"; groups = d / "groups_manifest.json"
        groups.write_text(json.dumps({"languages": ["aa", "bb"]}))
        rows = pool("aa", 12, 60, cond="INT4") + pool("aa", 12, 60, cond="INT6", seed=2, start=0) + pool("bb", 12, 60, cond="INT4", seed=3, start=3000) + pool("bb", 12, 60, cond="INT6", seed=4, start=3000)
        with open(f, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["feature_id", "group", "original_language", "condition", "fp_status", "status", "target_rate", "target_mean"]); w.writeheader()
            for r in rows:
                w.writerow({"feature_id": r["feature_id"], "group": "selective", "original_language": r["language"], "condition": r["condition"], "fp_status": r["fp_status"], "status": r["status"], "target_rate": r["rate"], "target_mean": r["mean"]})
        f4.main(["--status", str(f), "--groups-manifest", str(groups), "--out-dir", str(out), "--conditions", "INT6", "INT4", "--caliper", "3.0"])
        for b in ("INT6", "INT4"):
            m = json.loads((out / f"f4_manifest_{b}.json").read_text()); self.assertEqual(m["f4"]["condition"], b); self.assertEqual(m["f4"]["included"], ["aa", "bb"])
        self.assertFalse((out / "f4_manifest_INT5.json").exists())
        rep = (out / "f4_matching.md").read_text(); self.assertIn("H-F4", rep); self.assertIn("INT4", rep); self.assertIn("balance", rep.lower())


if __name__ == "__main__":
    unittest.main()
