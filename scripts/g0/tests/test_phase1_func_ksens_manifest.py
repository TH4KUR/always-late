import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phase1_func_ksens_manifest as km

ROOT = Path(__file__).resolve().parents[3]


def source(languages=("aa", "bb", "cc"), n_all=(70, 90, 64)):
    """A small manifest in the schema of groups_manifest.json: K16 and K64 are prefixes of the ranking `all`; the controls of a label are draws of the size of that label (`gap` holds an unrelated reference)."""
    groups, controls, sizes = {}, {}, {}
    for k, (l, n) in enumerate(zip(languages, n_all)):
        ids = list(range(1000 * k, 1000 * k + n))
        groups[l] = {"all": ids}
        n64 = min(64, n)
        sizes[l] = {"K16": 16, "K64": n64, "all": n}
        controls[l] = {"gap": [[5, 6, 7]] * 3, "K16": [[11] * 16] * 5, "K64": [[22 + d] * n64 for d in range(5)], "all": [[33 + d] * n for d in range(3)]}
    return {"schema": 1, "source": {"sha256": "ab" * 32}, "languages": list(languages), "Ks": [16, 64, None], "Rs": [5, 5, 3], "seed": 0, "groups": groups, "controls": controls, "sizes": sizes}


SHA = "cd" * 32


class Derive(unittest.TestCase):
    def setUp(self):
        self.src = source(); self.out = km.derive(copy.deepcopy(self.src), SHA)

    def test_the_label_K64_stands_for_the_whole_group_in_sizes_and_controls(self):
        for l, n in zip(("aa", "bb", "cc"), (70, 90, 64)):
            self.assertEqual(self.out["sizes"][l]["K64"], n)
            self.assertEqual(self.out["controls"][l]["K64"], self.src["controls"][l]["all"])
            self.assertEqual(len(self.out["controls"][l]["K64"]), 3)                                          # the three draws of the full size, not the five of K64
            self.assertTrue(all(len(d) == n for d in self.out["controls"][l]["K64"]))

    def test_the_runner_formula_returns_the_whole_group_and_its_matched_controls_under_the_label_K64(self):
        # the slicing that phase1_func_run.py applies to a manifest (pinned by a text test below)
        for l, n in zip(("aa", "bb", "cc"), (70, 90, 64)):
            members = self.out["groups"][l]["all"][:self.out["sizes"][l]["K64"]]
            self.assertEqual(members, self.src["groups"][l]["all"]); self.assertEqual(len(members), n)
            self.assertEqual(self.out["controls"][l]["K64"][:3], self.src["controls"][l]["all"])

    def test_everything_else_is_unchanged(self):
        for k in ("schema", "source", "languages", "Ks", "Rs", "seed", "groups"):
            self.assertEqual(self.out[k], self.src[k], k)
        for l in self.src["languages"]:
            self.assertEqual(self.out["sizes"][l]["K16"], 16); self.assertEqual(self.out["sizes"][l]["all"], self.src["sizes"][l]["all"])
            for lab in ("gap", "K16", "all"):
                self.assertEqual(self.out["controls"][l][lab], self.src["controls"][l][lab], lab)
        self.assertEqual(list(self.out["languages"]), ["aa", "bb", "cc"])
        self.assertEqual(sorted(set(self.out) - set(self.src)), ["kind", "ksens"])

    def test_the_input_is_not_modified(self):
        before = copy.deepcopy(self.src); km.derive(self.src, SHA)
        self.assertEqual(self.src, before)

    def test_the_ksens_block_records_the_source_and_the_derivation(self):
        k = self.out["ksens"]
        self.assertEqual(k["source_file_sha256"], SHA); self.assertEqual(k["source"], {"sha256": "ab" * 32}); self.assertEqual(self.out["kind"], "ksens-all")
        self.assertEqual(k["sizes"], {"aa": 70, "bb": 90, "cc": 64}); self.assertEqual(k["n_draws"], {"aa": 3, "bb": 3, "cc": 3})
        self.assertEqual(k["k64_equals_all"], ["cc"])                                                         # where the original K64 already was the whole group the two coincide
        self.assertIn("K64", k["derivation"])

    def test_the_output_is_deterministic(self):
        self.assertEqual(json.dumps(km.derive(copy.deepcopy(self.src), SHA)), json.dumps(km.derive(copy.deepcopy(self.src), SHA)))

    def test_inconsistent_inputs_are_errors(self):
        s = source(); del s["sizes"]["bb"]["all"]
        with self.assertRaises(ValueError):
            km.derive(s, SHA)
        s = source(); del s["controls"]["bb"]["all"]
        with self.assertRaises(ValueError):
            km.derive(s, SHA)
        s = source(); s["controls"]["aa"]["all"][1] = [1, 2, 3]
        with self.assertRaises(ValueError):
            km.derive(s, SHA)                                                                                 # a draw that is not of the size of the group
        s = source(); s["groups"]["aa"]["all"] = s["groups"]["aa"]["all"][:-1]
        with self.assertRaises(ValueError):
            km.derive(s, SHA)                                                                                 # the size (70) does not match the list of the group (69); the draws are still of size 70, so only this check can fire
        s = source(); s["controls"]["aa"]["all"] = []
        with self.assertRaises(ValueError):
            km.derive(s, SHA)                                                                                 # no draw at all: no matched control

    def test_the_report_states_the_status_the_label_and_every_language(self):
        txt = km.report_text(self.out)
        self.assertTrue(txt.startswith("# ")); self.assertIn("exploratory", txt.lower()); self.assertIn("K64 stands for the whole group", txt)
        for l in ("aa", "bb", "cc"):
            self.assertIn(l, txt)
        self.assertIn("| aa | 70 | 3 |", txt); self.assertIn("| bb | 90 | 3 |", txt)
        self.assertIn("(the two coincide): cc.", txt)


class Main(unittest.TestCase):
    def test_main_writes_the_manifest_and_the_report_and_hashes_the_source_file_bytes(self):
        d = Path(tempfile.mkdtemp()); src = d / "groups_manifest.json"; text = json.dumps(source()); src.write_text(text)
        km.main(["--src", str(src), "--out-dir", str(d / "out")])
        m = json.loads((d / "out" / "ksens_manifest.json").read_text())
        self.assertEqual(m["ksens"]["source_file_sha256"], hashlib.sha256(text.encode()).hexdigest())
        self.assertEqual(m["sizes"]["aa"]["K64"], 70); self.assertTrue((d / "out" / "ksens_report.md").read_text().startswith("# "))
        km.main(["--src", str(src), "--out-dir", str(d / "out2")])
        self.assertEqual((d / "out" / "ksens_manifest.json").read_bytes(), (d / "out2" / "ksens_manifest.json").read_bytes())


class RealManifest(unittest.TestCase):
    """The premise of the device, checked against the real frozen files: the unchanged runner reads a group as groups[l]["all"][:sizes[l][label]] and the controls as controls[l][label][:cap]."""
    def test_the_runner_still_slices_a_manifest_the_way_the_device_assumes(self):
        run = (ROOT / "scripts/g0/phase1_func_run.py").read_text()
        self.assertIn('man["groups"][code]["all"][:man["sizes"][code][label]]', run)
        self.assertIn('man["controls"][code][label][:caps.get(label)]', run)

    def test_the_real_frozen_manifest_gives_the_whole_group_and_three_matched_draws_for_every_language(self):
        path = ROOT / "reports/functional_validation/groups_manifest.json"
        src = json.loads(path.read_text()); out = km.derive(copy.deepcopy(src), hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(len(out["languages"]), 27)
        for l in out["languages"]:
            n = len(src["groups"][l]["all"])
            self.assertEqual(out["sizes"][l]["K64"], n); self.assertEqual(len(out["groups"][l]["all"][:out["sizes"][l]["K64"]]), n)
            self.assertEqual(len(out["controls"][l]["K64"]), 3); self.assertTrue(all(len(x) == n for x in out["controls"][l]["K64"]))
        self.assertEqual(out["ksens"]["k64_equals_all"], [])                                                  # every language has more than 64 selective features


if __name__ == "__main__":
    unittest.main()
