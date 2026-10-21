import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phase1_func_freeze as ff

PRESETS = {"eval": {"zero": ["K64"], "controls": ["K64"], "n_controls": {"K64": 3}, "sham": ["K64"]}, "floor": {"zero": ["K64"], "controls": ["K64"], "n_controls": {"K64": 2}}}
PRECISIONS = {"FP": "eval", "INT6": "eval", "INT16": "floor"}
ANALYSIS = {"bands": [0.75, 1.33], "macro_rule": "mean of per-language ratios (D-041)", "n_boot": 2000, "seed": 0, "primary_K": "K64", "sensitivity_K": ["K16", "all"]}


class Freeze(unittest.TestCase):
    def setUp(self):
        d = Path(tempfile.mkdtemp())
        self.files = {}
        for name in ("design", "plan", "groups_manifest"):
            p = d / f"{name}.txt"; p.write_text(f"content of {name}\n"); self.files[name] = p
        self.ids = {"n_total": 1012, "n_pilot": 150, "n_eval": 300, "seed": 0}
        self.freeze = ff.build_freeze(self.files, PRESETS, PRECISIONS, self.ids, ANALYSIS, commit="abc123", created="2026-10-22", root=d)

    def verify(self, **over):
        kw = {"files": self.files, "presets": PRESETS, "ids": self.ids, "analysis": ANALYSIS}
        kw.update(over)
        return ff.verify_freeze(self.freeze, **kw)

    def test_a_fresh_freeze_verifies_and_is_json_serialisable(self):
        self.assertTrue(self.verify())
        self.assertEqual(json.loads(json.dumps(self.freeze)), self.freeze)
        self.assertEqual(self.freeze["commit"], "abc123"); self.assertEqual(self.freeze["part"], "T-LM")

    def test_ids_are_disjoint_from_the_pilot_ids_and_a_subset_of_the_larger_set(self):
        blk = self.freeze["ids"]
        self.assertEqual(len(blk["eval_ids"]), 300)
        from phase1_func_tasks import lm_ids
        self.assertFalse(set(blk["eval_ids"]) & set(lm_ids(1012, 150, 450, 0)["pilot"]))
        self.assertTrue(set(blk["eval_ids"]) <= set(lm_ids(1012, 150, 450, 0)["eval"]))
        self.assertEqual(len(blk["eval_sha256"]), 64)

    def test_a_changed_file_is_named(self):
        self.files["plan"].write_text("edited\n")
        with self.assertRaisesRegex(ValueError, "plan"):
            self.verify()

    def test_a_missing_or_extra_file_entry_is_an_error(self):
        extra = dict(self.files); extra["new"] = self.files["plan"]
        with self.assertRaisesRegex(ValueError, "new"):
            self.verify(files=extra)
        fewer = {k: v for k, v in self.files.items() if k != "design"}
        with self.assertRaisesRegex(ValueError, "design"):
            self.verify(files=fewer)

    def test_changed_presets_ids_or_analysis_constants_are_errors(self):
        p2 = copy.deepcopy(PRESETS); p2["eval"]["n_controls"]["K64"] = 5
        with self.assertRaisesRegex(ValueError, "preset"):
            self.verify(presets=p2)
        with self.assertRaisesRegex(ValueError, "ids"):
            self.verify(ids={**self.ids, "n_eval": 301})
        with self.assertRaisesRegex(ValueError, "analysis"):
            self.verify(analysis={**ANALYSIS, "bands": [0.8, 1.25]})

    def test_a_tampered_id_list_is_an_error(self):
        self.freeze["ids"]["eval_ids"][0] += 1
        with self.assertRaisesRegex(ValueError, "ids"):
            self.verify()

    def test_unknown_preset_in_the_precision_map_is_rejected_at_build(self):
        with self.assertRaises(ValueError):
            ff.build_freeze(self.files, PRESETS, {"FP": "nonexistent"}, self.ids, ANALYSIS, "abc", "2026-10-22")

    def test_all_problems_are_listed_together(self):
        self.files["plan"].write_text("edited\n")
        with self.assertRaises(ValueError) as cm:
            self.verify(ids={**self.ids, "n_eval": 301}, analysis={**ANALYSIS, "n_boot": 10})
        for needle in ("plan", "ids", "analysis"):
            self.assertIn(needle, str(cm.exception))


if __name__ == "__main__":
    unittest.main()
