import copy
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phase1_func_f5_sample as fs

ROOT = Path(__file__).resolve().parents[3]
CONDS = ["INT8", "INT7", "INT6", "INT5", "INT4"]


def groups_manifest():
    """aa has K64 = features 1..4 (size 4 here), bb has 11..13; the lists are rankings, K64 is the prefix of the size K64."""
    return {"schema": 1, "languages": ["aa", "bb"], "groups": {"aa": {"all": [1, 2, 3, 4, 5, 6]}, "bb": {"all": [11, 12, 13, 14]}},
            "sizes": {"aa": {"K16": 4, "K64": 4, "all": 6}, "bb": {"K16": 3, "K64": 3, "all": 4}}, "controls": {}, "source": {"sha256": "g" * 8}}


def f4_manifest(cond, included, lost, draws):
    """lost {lang: [ids]}, draws {lang: [[ids], ...]}: the schema of phase1_func_f4_groups."""
    return {"schema": 1, "kind": "f4-subgroups", "languages": ["aa", "bb"], "groups": {l: {"all": list(lost.get(l, []))} for l in ("aa", "bb")},
            "controls": {l: {"K64": [list(d) for d in draws.get(l, [])]} for l in ("aa", "bb")}, "sizes": {l: {"K16": 1, "K64": len(lost.get(l, [])), "all": len(lost.get(l, []))} for l in ("aa", "bb")},
            "f4": {"condition": cond, "included": list(included), "excluded": {}, "records": {}}}


def status_rows(ids_by_lang, lost_at=None):
    """A row per feature and condition; lost_at {feature: [conditions where it is lost]}; rate and mean encode the feature id so that tests can see which row was read."""
    lost_at = lost_at or {}
    rows = []
    for lang, ids in ids_by_lang.items():
        for fid in ids:
            for c in CONDS:
                rows.append({"language": lang, "feature_id": fid, "condition": c, "fp_status": "retained", "status": "different_peak" if c in lost_at.get(fid, []) else "retained",
                             "rate": 0.001 * fid, "mean": 0.01 * fid})
    return rows


ALL = {"aa": [1, 2, 3, 4, 5, 6, 21, 22, 23], "bb": [11, 12, 13, 14, 31, 32]}


def sample(**kw):
    g = groups_manifest()
    f4 = {"INT4": f4_manifest("INT4", ["aa", "bb"], {"aa": [3, 5], "bb": [13]}, {"aa": [[21, 22]], "bb": [[31]]}),
          "INT5": f4_manifest("INT5", ["aa"], {"aa": [4]}, {"aa": [[23]]})}
    rows = status_rows(ALL, {3: ["INT4"], 5: ["INT4", "INT5"], 13: ["INT4"], 4: ["INT5"]})
    return fs.build_sample(kw.get("g", g), kw.get("f4", f4), kw.get("rows", rows), {"groups": "g", "f4": {"INT4": "a", "INT5": "b"}, "status": "s"})


class Build(unittest.TestCase):
    def test_every_feature_appears_once_with_all_of_its_roles(self):
        s = sample()
        self.assertEqual(sorted(map(int, s["features"])), [1, 2, 3, 4, 5, 11, 12, 13, 21, 22, 23, 31])
        self.assertEqual(s["features"]["3"]["roles"], ["k64", "lost@INT4"])                                      # in the K64 group of aa and in its lost subset at INT4
        self.assertEqual(s["features"]["4"]["roles"], ["k64", "lost@INT5"])
        self.assertEqual(s["features"]["5"]["roles"], ["lost@INT4"])                                             # 5 is not in the K64 prefix (the group has size 4)
        self.assertEqual(s["features"]["21"]["roles"], ["retained@INT4"]); self.assertEqual(s["features"]["23"]["roles"], ["retained@INT5"]); self.assertEqual(s["features"]["1"]["roles"], ["k64"])

    def test_the_k64_group_is_the_prefix_of_the_size_k64_not_the_whole_ranking(self):
        s = sample()
        self.assertNotIn("6", s["features"]); self.assertNotIn("14", s["features"])                              # ranked 5th/6th of aa and 4th of bb: beyond K64

    def test_the_language_comes_from_the_status_file_and_must_agree_with_the_manifests(self):
        s = sample(); self.assertEqual((s["features"]["3"]["language"], s["features"]["13"]["language"]), ("aa", "bb"))
        rows = status_rows(ALL); rows = [dict(r, language="bb") if r["feature_id"] == 3 else r for r in rows]
        with self.assertRaises(ValueError) as cm:
            sample(rows=rows)
        self.assertIn("3", str(cm.exception)); self.assertIn("language", str(cm.exception))

    def test_the_status_at_every_condition_and_the_native_activity_are_copied_from_the_rows_of_that_feature(self):
        s = sample(); f = s["features"]["5"]
        self.assertEqual(f["status"], {"INT8": "retained", "INT7": "retained", "INT6": "retained", "INT5": "different_peak", "INT4": "different_peak"})
        self.assertEqual(f["native"]["INT6"], [0.005, 0.05]); self.assertTrue(f["fp_established"])

    def test_a_feature_without_status_rows_is_an_error_and_so_is_an_inconsistent_fp_status(self):
        with self.assertRaises(ValueError) as cm:
            sample(rows=[r for r in status_rows(ALL) if r["feature_id"] != 12])
        self.assertIn("12", str(cm.exception))
        rows = status_rows(ALL); rows[0] = dict(rows[0], fp_status="different_peak")
        with self.assertRaises(ValueError):
            sample(rows=rows)

    def test_lost_and_retained_subsets_of_a_language_and_precision_must_not_overlap(self):
        g = groups_manifest(); f4 = {"INT4": f4_manifest("INT4", ["aa"], {"aa": [3, 5]}, {"aa": [[5, 21]]})}
        with self.assertRaises(ValueError) as cm:
            fs.build_sample(g, f4, status_rows(ALL), {})
        self.assertIn("overlap", str(cm.exception))

    def test_every_retained_draw_counts_not_only_the_first(self):
        g = groups_manifest(); f4 = {"INT4": f4_manifest("INT4", ["aa"], {"aa": [3]}, {"aa": [[21], [22]]})}
        s = fs.build_sample(g, f4, status_rows(ALL), {})
        self.assertEqual(s["counts"]["retained@INT4"], {"aa": 2}); self.assertEqual(s["f4"]["INT4"]["aa"]["retained"], [[21], [22]])

    def test_a_feature_that_was_not_established_in_full_precision_is_flagged(self):
        rows = [dict(r, fp_status="different_peak") if r["feature_id"] == 22 else r for r in status_rows(ALL)]
        g = groups_manifest(); f4 = {"INT4": f4_manifest("INT4", ["aa"], {"aa": [3]}, {"aa": [[21, 22]]})}
        s = fs.build_sample(g, f4, rows, {})
        self.assertFalse(s["features"]["22"]["fp_established"]); self.assertTrue(s["features"]["21"]["fp_established"])

    def test_an_excluded_language_contributes_no_f4_role(self):
        s = sample(); self.assertNotIn("lost@INT5", s["features"]["13"]["roles"]); self.assertNotIn("retained@INT5", s["features"]["31"]["roles"])

    def test_the_counts_equal_the_manifests(self):
        c = sample()["counts"]
        self.assertEqual(c["k64"], {"aa": 4, "bb": 3}); self.assertEqual(c["lost@INT4"], {"aa": 2, "bb": 1}); self.assertEqual(c["retained@INT4"], {"aa": 2, "bb": 1})
        self.assertEqual(c["lost@INT5"], {"aa": 1}); self.assertEqual(c["retained@INT5"], {"aa": 1})
        s = sample(); self.assertEqual(s["n_features"], 12); self.assertEqual(s["n_multi_role"], 3)                  # features 3 and 4 (aa) and 13 (bb) are in the K64 prefix and in a lost subset

    def test_the_f4_pairs_of_a_language_and_precision_are_listed_for_the_paired_analysis(self):
        p = sample()["f4"]["INT4"]["aa"]
        self.assertEqual((p["lost"], p["retained"]), ([3, 5], [[21, 22]]))

    def test_the_output_is_deterministic_and_the_input_is_not_modified(self):
        g = groups_manifest(); g0 = copy.deepcopy(g)
        self.assertEqual(json.dumps(sample(g=g), sort_keys=True), json.dumps(sample(), sort_keys=True)); self.assertEqual(g, g0)

    def test_the_sources_are_recorded_and_the_precisions_come_from_the_manifests(self):
        s = sample(); self.assertEqual(s["source"]["status"], "s"); self.assertEqual(sorted(s["f4"]), ["INT4", "INT5"]); self.assertEqual(s["languages"], ["aa", "bb"])

    def test_a_manifest_of_another_condition_than_its_key_is_an_error(self):
        g = groups_manifest(); f4 = {"INT4": f4_manifest("INT5", ["aa"], {"aa": [3]}, {"aa": [[21]]})}
        with self.assertRaises(ValueError):
            fs.build_sample(g, f4, status_rows(ALL), {})


class Report(unittest.TestCase):
    def test_the_report_states_the_roles_counts_and_the_multi_role_features(self):
        txt = fs.report_text(sample())
        self.assertTrue(txt.startswith("# ")); self.assertIn("k64", txt); self.assertIn("lost@INT4", txt); self.assertIn("12 features", txt); self.assertIn("3 in more than one role", txt)


class Main(unittest.TestCase):
    def test_main_writes_the_sample_and_the_report_from_files(self):
        d = Path(tempfile.mkdtemp())
        (d / "groups.json").write_text(json.dumps(groups_manifest()))
        (d / "f4").mkdir()
        for cond, inc, lost, draws in (("INT4", ["aa", "bb"], {"aa": [3, 5], "bb": [13]}, {"aa": [[21, 22]], "bb": [[31]]}), ("INT5", ["aa"], {"aa": [4]}, {"aa": [[23]]})):
            (d / "f4" / f"f4_manifest_{cond}.json").write_text(json.dumps(f4_manifest(cond, inc, lost, draws)))
        rows = status_rows(ALL, {3: ["INT4"]})
        with open(d / "status.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["feature_id", "group", "original_language", "condition", "fp_status", "status", "target_rate", "target_mean"])
            for r in rows:
                w.writerow([r["feature_id"], "selective", r["language"], r["condition"], r["fp_status"], r["status"], r["rate"], r["mean"]])
            w.writerow([999, "non_selective", "aa", "INT4", "retained", "retained", 0.1, 0.1])                       # not a selective feature: ignored
        fs.main(["--status", str(d / "status.csv"), "--groups-manifest", str(d / "groups.json"), "--f4-dir", str(d / "f4"), "--conditions", "INT4", "INT5", "--out-dir", str(d / "out")])
        j = json.loads((d / "out" / "f5_sample.json").read_text()); self.assertEqual(j["n_features"], 12); self.assertNotIn("999", j["features"])
        self.assertTrue((d / "out" / "f5_sample.md").read_text().startswith("# "))
        import hashlib
        self.assertEqual(j["source"]["status"], hashlib.sha256((d / "status.csv").read_bytes()).hexdigest()); self.assertEqual(j["source"]["groups"], hashlib.sha256((d / "groups.json").read_bytes()).hexdigest())


class RealManifests(unittest.TestCase):
    """The roles and counts of the committed F4 manifests are reproduced exactly (synthetic status rows: the D-035 status file is not in the repository)."""
    def test_the_counts_of_the_real_f4_manifests_are_reproduced(self):
        g = json.loads((ROOT / "reports/functional_validation/groups_manifest.json").read_text())
        f4 = {c: json.loads((ROOT / f"reports/functional_validation/f4/f4_manifest_{c}.json").read_text()) for c in ("INT6", "INT5", "INT4")}
        ids = {}
        for l in g["languages"]:
            members = set(g["groups"][l]["all"][:g["sizes"][l]["K64"]])
            for m in f4.values():
                members |= set(m["groups"][l]["all"][:m["sizes"][l]["K64"]])
                for d in m["controls"][l]["K64"]:
                    members |= set(d)
            ids[l] = sorted(members)
        s = fs.build_sample(g, f4, status_rows(ids), {})
        for c, m in f4.items():
            for l in m["f4"]["included"]:
                self.assertEqual(s["counts"][f"lost@{c}"][l], m["sizes"][l]["K64"], (c, l)); self.assertEqual(sorted(s["f4"][c][l]["lost"]), sorted(m["groups"][l]["all"][:m["sizes"][l]["K64"]]))
            self.assertEqual(sorted(s["f4"][c]), sorted(m["f4"]["included"]))
        self.assertEqual(sum(s["counts"]["k64"].values()), sum(g["sizes"][l]["K64"] for l in g["languages"]))
        self.assertGreater(s["n_features"], 2000)


if __name__ == "__main__":
    unittest.main()
