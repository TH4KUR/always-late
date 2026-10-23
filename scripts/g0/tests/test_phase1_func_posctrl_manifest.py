import copy
import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_func_posctrl_manifest as pm
from test_phase1_func_ksens_manifest import source

SHA = "cd" * 32


def pool(n=40, start=500000):
    return list(range(start, start + n))


def sel_union(src):
    return {f for l in src["languages"] for f in src["groups"][l]["all"]}


class Derive(unittest.TestCase):
    def setUp(self):
        self.src = source(); self.pool = pool(); self.out = pm.derive(self.src, self.pool, seed=0, doses=(8, 20), n_draws=3, source_sha256=SHA)

    def test_the_edited_set_is_the_same_language_independent_random_order_of_the_pool_in_every_language(self):
        first = self.out["groups"]["aa"]["all"]
        self.assertEqual(sorted(first), sorted(self.pool)); self.assertNotEqual(first, sorted(self.pool))                                 # a permutation, not the sorted list
        for l in self.src["languages"]:
            self.assertEqual(self.out["groups"][l]["all"], first)

    def test_the_three_labels_are_nested_prefixes_of_that_order_with_the_declared_sizes(self):
        for l in self.src["languages"]:
            self.assertEqual(self.out["sizes"][l], {"K16": 8, "K64": 20, "all": 40})
            g = self.out["groups"][l]["all"]
            self.assertEqual(g[:self.out["sizes"][l]["K16"]], g[:8]); self.assertEqual(set(g[:8]) <= set(g[:20]), True)                 # the runner formula groups[l]['all'][:sizes[l][label]]

    def test_controls_are_random_draws_of_the_size_of_the_label_from_the_selective_features_of_the_other_languages(self):
        union = sel_union(self.src)
        for l in self.src["languages"]:
            own = set(self.src["groups"][l]["all"])
            for label in ("K16", "K64", "all"):
                draws = self.out["controls"][l][label]; self.assertEqual(len(draws), 3)
                for d in draws:
                    self.assertEqual(len(d), self.out["sizes"][l][label]); self.assertEqual(d, sorted(d)); self.assertEqual(len(set(d)), len(d))
                    self.assertTrue(set(d) <= union - own); self.assertFalse(set(d) & set(self.pool))
            self.assertNotEqual(self.out["controls"][l]["K64"][0], self.out["controls"][l]["K64"][1])                                      # the draws differ

    def test_the_draws_depend_on_the_language_the_label_and_the_seed(self):
        a, b = self.out["controls"]["aa"]["K64"][0], self.out["controls"]["bb"]["K64"][0]; self.assertNotEqual(a, b)
        other = pm.derive(self.src, self.pool, seed=1, doses=(8, 20), n_draws=3, source_sha256=SHA); self.assertNotEqual(other["groups"]["aa"]["all"], self.out["groups"]["aa"]["all"]); self.assertNotEqual(other["controls"]["aa"]["K64"], self.out["controls"]["aa"]["K64"])

    def test_two_languages_with_candidate_lists_of_the_same_length_get_different_index_patterns(self):
        same = source(n_all=(30, 30, 30)); o = pm.derive(same, pool(20), seed=0, doses=(4, 8), n_draws=2, source_sha256=SHA); u = sel_union(same); pat = {}
        for l in ("aa", "bb"):
            cand = sorted(u - set(same["groups"][l]["all"])); pat[l] = [cand.index(f) for f in o["controls"][l]["K64"][0]]
        self.assertEqual(len(pat["aa"]), 8); self.assertNotEqual(pat["aa"], pat["bb"])                                                                # the seed depends on the language, not only on the list

    def test_the_output_is_deterministic_and_the_input_is_not_modified(self):
        src = copy.deepcopy(self.src); again = pm.derive(src, self.pool, seed=0, doses=(8, 20), n_draws=3, source_sha256=SHA)
        self.assertEqual(json.dumps(again, sort_keys=True), json.dumps(self.out, sort_keys=True)); self.assertEqual(src, self.src)

    def test_the_schema_of_the_source_is_kept_and_the_block_records_the_derivation(self):
        for k in ("schema", "languages", "Ks", "Rs", "seed", "source"):
            self.assertEqual(self.out[k], self.src[k])
        self.assertEqual(self.out["kind"], "posctrl"); b = self.out["posctrl"]
        self.assertEqual(b["source_manifest_sha256"], SHA); self.assertEqual(b["doses"], [8, 20, 40]); self.assertEqual(b["n_pool"], 40); self.assertEqual(b["n_selective"], len(sel_union(self.src))); self.assertEqual(b["seed"], 0)
        self.assertEqual(b["pool_sha256"], hashlib.sha256(json.dumps(sorted(self.pool)).encode()).hexdigest()); self.assertIn("K64", b["derivation"])
        self.assertEqual(self.out["controls"]["aa"]["gap"], {})                                                                            # the matching diagnostic of the source does not apply

    def test_inconsistent_inputs_are_errors(self):
        with self.assertRaises(ValueError):
            pm.derive(self.src, self.pool + [self.src["groups"]["aa"]["all"][0]], seed=0, doses=(8, 20), n_draws=3, source_sha256=SHA)          # a selective feature in the pool
        with self.assertRaises(ValueError):
            pm.derive(self.src, self.pool + [self.pool[0]], seed=0, doses=(8, 20), n_draws=3, source_sha256=SHA)                                 # a duplicate
        with self.assertRaises(ValueError):
            pm.derive(self.src, self.pool, seed=0, doses=(8, 40), n_draws=3, source_sha256=SHA)                                                  # a dose as large as the pool
        with self.assertRaises(ValueError):
            pm.derive(self.src, self.pool, seed=0, doses=(20, 8), n_draws=3, source_sha256=SHA)                                                  # doses not ascending
        with self.assertRaises(ValueError):
            pm.derive(self.src, self.pool, seed=0, doses=(8, 8), n_draws=3, source_sha256=SHA)                                                    # equal doses
        with self.assertRaises(ValueError):
            pm.derive(self.src, self.pool, seed=0, doses=(8, 20), n_draws=0, source_sha256=SHA)
        small = source(n_all=(10, 10, 10))
        with self.assertRaises(ValueError):
            pm.derive(small, pool(60), seed=0, doses=(8, 20), n_draws=3, source_sha256=SHA)                                                      # the 'all' dose (60) is larger than the other languages' selective features (20)

    def test_boundaries_a_zero_dose_is_an_error_and_exactly_enough_candidates_is_accepted(self):
        with self.assertRaises(ValueError):
            pm.derive(self.src, self.pool, seed=0, doses=(0, 20), n_draws=3, source_sha256=SHA)
        three = source(n_all=(10, 10, 10)); ok = pm.derive(three, pool(20), seed=0, doses=(4, 8), n_draws=2, source_sha256=SHA)                       # the others have exactly 20 selective features: the 'all' dose is 20
        self.assertEqual(len(ok["controls"]["aa"]["all"][0]), 20); self.assertEqual(sorted(ok["controls"]["aa"]["all"][0]), sorted(sel_union(three) - set(three["groups"]["aa"]["all"])))

    def test_the_report_states_the_status_the_labels_and_every_language(self):
        md = pm.report_text(self.out)
        self.assertIn("positive control", md); self.assertIn("K16", md); self.assertIn("K64", md); self.assertIn("all", md); self.assertIn("Exploratory", md)
        for l in self.src["languages"]:
            self.assertIn(f"| {l} |", md)


class Pool(unittest.TestCase):
    def test_the_pool_is_the_set_of_control_features_of_the_d035_status_file(self):
        cols = ["feature_id", "group", "original_language", "condition", "fp_status", "status"]
        rows = [{"feature_id": 7, "group": "controls", "original_language": "aa", "condition": "FP"}, {"feature_id": 7, "group": "controls", "original_language": "aa", "condition": "INT4"},
                {"feature_id": 3, "group": "controls", "original_language": "bb", "condition": "FP"}, {"feature_id": 99, "group": "selective", "original_language": "aa", "condition": "FP"}, {"feature_id": 5, "group": "controls", "original_language": "aa", "condition": "INT4"}]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.csv"
            with open(p, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); [w.writerow({c: r.get(c, "") for c in cols}) for r in rows]
            r = pm.read_pool(p)
        self.assertEqual(r["ids"], [3, 7]); self.assertEqual(len(r["sha256"]), 64)                                                                           # only the full-precision rows define the pool: id 5 appears at INT4 only

    def test_a_file_without_control_features_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.csv"; p.write_text("feature_id,group,original_language,condition\n1,selective,aa,FP\n")
            with self.assertRaises(ValueError):
                pm.read_pool(p)


class SelectDose(unittest.TestCase):
    def setUp(self):
        self.src = source(); self.m = pm.derive(self.src, pool(), seed=0, doses=(8, 20), n_draws=3, source_sha256=SHA)

    def test_the_label_K64_stands_for_the_chosen_dose_in_sizes_and_controls_and_the_other_labels_are_kept(self):
        for label, size in (("K16", 8), ("K64", 20), ("all", 40)):
            s = pm.select_dose(self.m, label)
            for l in self.src["languages"]:
                self.assertEqual(s["sizes"][l]["K64"], size); self.assertEqual(s["controls"][l]["K64"], self.m["controls"][l][label]); self.assertEqual(s["sizes"][l]["K16"], 8); self.assertEqual(s["sizes"][l]["all"], 40)
                self.assertEqual(s["groups"][l]["all"][:s["sizes"][l]["K64"]], self.m["groups"][l]["all"][:size])                                   # the runner formula returns the dose
            self.assertEqual(s["posctrl"]["k64_stands_for"], label); self.assertEqual(s["posctrl"]["k64_size"], size)

    def test_the_input_is_not_modified_and_an_unknown_label_is_an_error(self):
        before = json.dumps(self.m, sort_keys=True); pm.select_dose(self.m, "all"); self.assertEqual(json.dumps(self.m, sort_keys=True), before)
        with self.assertRaises(ValueError):
            pm.select_dose(self.m, "K32")

    def test_the_report_of_a_selected_dose_says_what_the_label_stands_for(self):
        self.assertIn("the label K64 stands for the dose `all` (40 features)", pm.report_text(pm.select_dose(self.m, "all"))); self.assertNotIn("In THIS manifest", pm.report_text(self.m))


class Command(unittest.TestCase):
    def test_the_command_line_writes_the_spec_the_report_and_three_manifests_and_a_rebuild_from_the_spec_is_identical(self):
        src = source(); pl = pool()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); (d / "src.json").write_text(json.dumps(src)); st = d / "f.csv"
            with open(st, "w", newline="") as fh:
                w = csv.writer(fh); w.writerow(["feature_id", "group", "original_language", "condition"]); [w.writerow([i, "controls", "aa", "FP"]) for i in pl]
            pm.main(["--src", str(d / "src.json"), "--status", str(st), "--out-dir", str(d / "o"), "--manifest-dir", str(d / "m"), "--doses", "8", "20"])
            spec = json.loads((d / "o" / "posctrl_spec.json").read_text()); self.assertEqual(spec["pool"], sorted(pl)); self.assertEqual(spec["doses"], [8, 20]); self.assertTrue((d / "o" / "posctrl_report.md").exists())
            self.assertEqual(sorted(spec["manifest_sha256"]), ["K16", "K64", "all"]); self.assertEqual(spec["status_sha256"], hashlib.sha256(st.read_bytes()).hexdigest())
            for label in ("K16", "K64", "all"):
                f = d / "m" / f"posctrl_{label}_manifest.json"; self.assertEqual(spec["manifest_sha256"][label], hashlib.sha256(f.read_bytes()).hexdigest()); self.assertEqual(json.loads(f.read_text())["posctrl"]["k64_stands_for"], label)
            pm.main(["--src", str(d / "src.json"), "--from-spec", str(d / "o" / "posctrl_spec.json"), "--manifest-dir", str(d / "m2")])
            for label in ("K16", "K64", "all"):
                self.assertEqual((d / "m" / f"posctrl_{label}_manifest.json").read_bytes(), (d / "m2" / f"posctrl_{label}_manifest.json").read_bytes())
            other = copy.deepcopy(src); other["seed"] = 9; (d / "other_src.json").write_text(json.dumps(other))
            with self.assertRaises(ValueError):
                pm.main(["--src", str(d / "other_src.json"), "--from-spec", str(d / "o" / "posctrl_spec.json"), "--manifest-dir", str(d / "m4")])                  # the spec belongs to another source manifest
            bad = dict(spec); bad["seed"] = 5; (d / "bad.json").write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                pm.main(["--src", str(d / "src.json"), "--from-spec", str(d / "bad.json"), "--manifest-dir", str(d / "m3")])                                  # the rebuild must equal the recorded hashes


if __name__ == "__main__":
    unittest.main()
