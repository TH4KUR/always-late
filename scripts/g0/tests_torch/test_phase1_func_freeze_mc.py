import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phase1_func_freeze as ff

PRESET = {"zero": ["K64"], "controls": ["K64"], "n_controls": {"K64": 3}, "sham": ["K64"], "resample": ["K64"]}
EVAL_KEYS = {"aaa": [("p0", 1), ("p0", 2), ("p1", 1)], "bbb": [("p0", 1), ("p2", 1)]}
CAL_KEYS = {"aaa": [("q0", 1)], "bbb": [("q0", 1)]}
SHARED = ("code_run", "code_tasks", "code_lib")


class Env:
    """A planted repository: code files, Belebele manifest, origin file, calibration record and a T-LM freeze that hashes the shared code."""
    def __init__(self):
        self.d = Path(tempfile.mkdtemp())
        self.paths = {}
        for name in ("code_run", "code_tasks", "code_lib", "code_intervene", "code_groups", "code_mc_analysis", "code_analysis", "code_calibration"):
            self.write(name, f"# {name} v1\n")
        self.write("belebele_manifest", json.dumps({"files": {"aaa": {"sha256": "x"}}}))
        self.write("passage_origin", json.dumps({"origin": {"p0": "devtest"}}))
        self.calibration = {"format": "cloze", "covered": ["aaa", "bbb"], "uncovered": [], "chance": 0.25, "margin": 0.05}
        self.write("calibration_record", json.dumps(self.calibration))
        self.mc = {"preset": "mc_eval", "preset_definition": PRESET, "split": "eval", "format": "cloze", "n_items": 200, "item_seed": 0, "max_len": 1024, "max_tokens": 1024, "batch_size": 16}
        self.precisions = {"FP": "mc_eval", "INT8": "mc_eval", "INT7": "mc_eval", "INT6": "mc_eval", "INT5": "mc_eval", "INT4": "mc_eval"}
        self.analysis = ff.mc_analysis_constants(["aaa", "bbb"], "cloze", [0.75, 1.33])
        self.lm_sha = "a" * 64
        self.lm_freeze = {"files": {n: {"path": n, "sha256": ff.sha256_file(self.paths[n])} for n in SHARED}}

    def write(self, name, text):
        p = self.d / f"{name}.txt"; p.write_text(text); self.paths[name] = p
        return p

    def build(self, **kw):
        args = dict(files=self.paths, lm_freeze=self.lm_freeze, lm_freeze_sha256=self.lm_sha, calibration=self.calibration, mc=self.mc, eval_keys=EVAL_KEYS, cal_keys=CAL_KEYS,
                    precisions=self.precisions, analysis=self.analysis, commit="abc123", created="2026-10-23", root=self.d)
        args.update(kw)
        return ff.build_mc_freeze(**args)

    def verify(self, freeze, **kw):
        args = dict(files=self.paths, lm_freeze=self.lm_freeze, lm_freeze_sha256=self.lm_sha, calibration=self.calibration, mc=self.mc, eval_keys=EVAL_KEYS, analysis=self.analysis)
        args.update(kw)
        return ff.verify_mc_freeze(freeze, **args)


class Statements(unittest.TestCase):
    def test_the_default_statements_of_q19(self):
        a = ff.mc_analysis_constants(["aaa", "bbb"], "cloze", [0.75, 1.33])
        st = {s["id"]: s for s in a["statements"]}
        self.assertEqual(list(st), ["S0", "S1", "S2", "S3", "S4", "S5"])
        self.assertEqual((st["S0"]["precision"], st["S0"]["sidedness"], st["S0"]["interval_level"]), ("FP", "one-sided", 0.95))
        self.assertEqual([st[f"S{i}"]["precision"] for i in range(1, 6)], ["INT8", "INT7", "INT6", "INT5", "INT4"])
        self.assertTrue(all(st[f"S{i}"]["interval_level"] == 0.99 and "Bonferroni" in st[f"S{i}"]["correction"] for i in range(1, 6)))
        self.assertEqual((a["covered"], a["format"], a["bands"], a["cluster"]), (["aaa", "bbb"], "cloze", [0.75, 1.33], "passage"))
        self.assertIn("exploratory", a["everything_else"])

    def test_the_constants_do_not_alias_the_caller_lists(self):
        covered = ["aaa"]; a = ff.mc_analysis_constants(covered, "letter", [0.75, 1.33]); covered.append("zzz")
        self.assertEqual(a["covered"], ["aaa"])


class ItemsBlock(unittest.TestCase):
    def test_counts_and_order_sensitive_hashes(self):
        b = ff.items_block(EVAL_KEYS)
        self.assertEqual({c: v["n_items"] for c, v in b.items()}, {"aaa": 3, "bbb": 2})
        swapped = dict(EVAL_KEYS, aaa=[("p0", 2), ("p0", 1), ("p1", 1)])
        self.assertNotEqual(ff.items_block(swapped)["aaa"]["keys_sha256"], b["aaa"]["keys_sha256"])
        self.assertEqual(ff.items_block(EVAL_KEYS)["bbb"], b["bbb"])

    def test_overlap_with_the_calibration_items_and_duplicates_are_refused(self):
        with self.assertRaises(ValueError):
            ff.items_block(EVAL_KEYS, {"aaa": [("p1", 1)]})
        with self.assertRaises(ValueError):
            ff.items_block({"aaa": [("p0", 1), ("p0", 1)]})


class Build(unittest.TestCase):
    def setUp(self):
        self.env = Env()

    def test_structure_and_hashes(self):
        fz = self.env.build()
        self.assertEqual((fz["kind"], fz["part"], fz["created"], fz["commit"], fz["lm_freeze_sha256"]), ("functional-validation-freeze", "T-MC", "2026-10-23", "abc123", "a" * 64))
        for name, p in self.env.paths.items():
            self.assertEqual(fz["files"][name]["sha256"], ff.sha256_file(p)); self.assertEqual(fz["files"][name]["path"], p.name)
        self.assertEqual(fz["calibration"], {"format": "cloze", "covered": ["aaa", "bbb"]})
        self.assertEqual(fz["items"]["aaa"]["n_items"], 3); self.assertEqual(fz["precisions"], self.env.precisions); self.assertEqual(fz["deviations"], {})
        self.assertEqual(fz["mc"], self.env.mc); self.assertEqual(fz["analysis"], self.env.analysis)

    def test_an_empty_covered_list_cannot_be_frozen_even_when_the_items_agree_with_it(self):
        with self.assertRaises(ValueError) as cm:
            self.env.build(calibration={**self.env.calibration, "covered": []}, eval_keys={}, cal_keys={})
        self.assertIn("no covered language", str(cm.exception))

    def test_a_missing_or_unknown_format_cannot_be_frozen_even_when_the_run_agrees_with_it(self):
        for bad in (None, "bogus"):
            with self.assertRaises(ValueError) as cm:
                self.env.build(calibration={**self.env.calibration, "format": bad}, mc={**self.env.mc, "format": bad})
            self.assertIn("no chosen format", str(cm.exception))

    def test_the_format_of_the_run_must_be_the_calibrated_one(self):
        with self.assertRaises(ValueError):
            self.env.build(mc={**self.env.mc, "format": "letter"})

    def test_the_items_must_be_exactly_those_of_the_covered_languages(self):
        with self.assertRaises(ValueError):
            self.env.build(eval_keys={"aaa": EVAL_KEYS["aaa"]})
        with self.assertRaises(ValueError):
            self.env.build(eval_keys={**EVAL_KEYS, "ccc": [("p9", 1)]})

    def test_an_overlap_with_the_calibration_split_is_refused(self):
        with self.assertRaises(ValueError):
            self.env.build(cal_keys={"aaa": [("p0", 2)], "bbb": []})

    def test_every_precision_must_use_the_frozen_preset(self):
        with self.assertRaises(ValueError):
            self.env.build(precisions={**self.env.precisions, "INT4": "eval"})

    def test_a_shared_code_file_that_differs_from_the_loss_freeze_needs_a_recorded_deviation(self):
        self.env.write("code_run", "# changed after the T-LM freeze\n")
        with self.assertRaises(ValueError) as cm:
            self.env.build()
        self.assertIn("code_run", str(cm.exception))
        fz = self.env.build(deviations={"code_run": "D-043"})
        self.assertEqual(fz["deviations"], {"code_run": "D-043"})
        with self.assertRaises(ValueError) as cm:
            self.env.build(deviations={"code_run": "because"})                                              # a deviation is a decision-register id
        self.assertIn("decision-register id", str(cm.exception))

    def test_a_deviation_for_a_file_that_is_not_frozen_is_refused(self):
        with self.assertRaises(ValueError) as cm:
            self.env.build(deviations={"code_unknown": "D-043"})                                            # the code is unchanged here, so no other check can fire
        self.assertIn("not a frozen file", str(cm.exception))

    def test_files_that_are_not_in_the_loss_freeze_are_not_cross_checked(self):
        self.env.write("code_mc_analysis", "# another version\n")
        self.assertEqual(self.env.build()["files"]["code_mc_analysis"]["sha256"], ff.sha256_file(self.env.paths["code_mc_analysis"]))


class Verify(unittest.TestCase):
    def setUp(self):
        self.env = Env(); self.fz = self.env.build()

    def test_passes_on_the_unchanged_state(self):
        self.assertTrue(self.env.verify(self.fz))

    def assertDetects(self, text, **kw):
        with self.assertRaises(ValueError) as cm:
            self.env.verify(self.fz, **kw)
        self.assertIn(text, str(cm.exception))

    def test_a_changed_code_file_is_named(self):
        self.env.write("code_mc_analysis", "# edited\n")
        self.assertDetects("code_mc_analysis")

    def test_a_changed_shared_code_file_is_a_difference_from_the_loss_freeze_too(self):
        self.env.write("code_tasks", "# edited\n")
        with self.assertRaises(ValueError) as cm:
            self.env.verify(self.fz)
        self.assertIn("code_tasks", str(cm.exception))

    def test_a_recorded_deviation_is_allowed_to_differ_from_the_loss_freeze(self):
        self.env.write("code_run", "# changed with a D-entry\n")
        fz = self.env.build(deviations={"code_run": "D-043"})
        self.assertTrue(self.env.verify(fz))
        self.env.write("code_run", "# changed again\n")
        with self.assertRaises(ValueError):
            self.env.verify(fz)                                                                              # still frozen at the recorded hash

    def test_changed_belebele_inputs_and_calibration_are_detected(self):
        self.env.write("belebele_manifest", json.dumps({"files": {"aaa": {"sha256": "y"}}}))
        self.assertDetects("belebele_manifest")

    def test_a_changed_calibration_record_format_or_covered_list_is_detected(self):
        self.assertDetects("calibration", calibration={**self.env.calibration, "covered": ["aaa"]})
        self.assertDetects("calibration", calibration={**self.env.calibration, "format": "letter"})

    def test_a_changed_item_selection_is_detected_even_by_reordering(self):
        self.assertDetects("items", eval_keys={**EVAL_KEYS, "aaa": [("p0", 2), ("p0", 1), ("p1", 1)]})
        self.assertDetects("items", eval_keys={**EVAL_KEYS, "bbb": EVAL_KEYS["bbb"][:1]})
        self.assertDetects("items", eval_keys={"aaa": EVAL_KEYS["aaa"]})                                    # a covered language that vanished from the selection
        self.assertDetects("items", eval_keys={**EVAL_KEYS, "ccc": [("p9", 1)]})                            # and one that appeared

    def test_a_changed_preset_cap_or_seed_is_detected(self):
        self.assertDetects("mc", mc={**self.env.mc, "n_items": 150})
        self.assertDetects("mc", mc={**self.env.mc, "preset_definition": {**PRESET, "n_controls": {"K64": 5}}})

    def test_changed_statements_or_constants_are_detected(self):
        a = copy.deepcopy(self.env.analysis); a["statements"][1]["interval_level"] = 0.95
        self.assertDetects("analysis", analysis=a)
        b = copy.deepcopy(self.env.analysis); b["bands"] = [0.8, 1.25]
        self.assertDetects("analysis", analysis=b)

    def test_a_changed_loss_freeze_file_is_detected(self):
        self.assertDetects("T-LM freeze", lm_freeze_sha256="b" * 64)

    def test_every_difference_is_listed(self):
        self.env.write("code_mc_analysis", "# edited\n")
        with self.assertRaises(ValueError) as cm:
            self.env.verify(self.fz, mc={**self.env.mc, "n_items": 1}, calibration={**self.env.calibration, "covered": ["aaa"]})
        msg = str(cm.exception)
        for part in ("code_mc_analysis", "mc", "calibration"):
            self.assertIn(part, msg)


ROOT = Path(__file__).resolve().parents[3]


@unittest.skipUnless((ROOT / "data/belebele/passage_origin.json").exists() and (ROOT / "data/belebele/eng_Latn.jsonl").exists(), "local Belebele data not present")
class RepoStateMc(unittest.TestCase):
    """The plumbing on the real local Belebele files with a planted calibration record: the freeze records the items the runner would score."""
    def setUp(self):
        self.d = Path(tempfile.mkdtemp()); self.cal = self.d / "belebele_calibration.json"
        self.cal.write_text(json.dumps({"format": "cloze", "covered": ["eng_Latn", "hin_Deva"], "uncovered": []}))

    def state(self, n_items=40):
        return ff.repo_state_mc(ROOT, self.cal, ROOT / "data/belebele", ROOT / "data/belebele/passage_origin.json", n_items=n_items)

    def test_the_evaluation_items_are_devtest_origin_passages_and_disjoint_from_the_calibration_items(self):
        import json as _j
        st = self.state(); origin = _j.loads((ROOT / "data/belebele/passage_origin.json").read_text())["origin"]
        for code in ("eng_Latn", "hin_Deva"):
            keys = st["eval_keys"][code]
            self.assertEqual(len(keys), 40); self.assertTrue(all(origin[k[0]] == "devtest" for k in keys), code)
            self.assertFalse({tuple(k) for k in keys} & {tuple(k) for k in st["cal_keys"][code]})
            self.assertTrue(all(origin[k[0]] == "dev" for k in st["cal_keys"][code]))
        self.assertEqual(set(st["eval_keys"]), {"eng_Latn", "hin_Deva"})

    def test_a_build_and_a_verify_round_trip_on_the_real_state_and_a_changed_cap_is_detected(self):
        st = self.state(); lm = json.loads((ROOT / "reports/functional_validation/evaluation_freeze_lm.json").read_text())
        fz = ff.build_mc_freeze(files=st["files"], lm_freeze=lm, lm_freeze_sha256=ff.sha256_file(ROOT / "reports/functional_validation/evaluation_freeze_lm.json"), calibration=st["calibration"], mc=st["mc"],
                                eval_keys=st["eval_keys"], cal_keys=st["cal_keys"], precisions=st["precisions"], analysis=st["analysis"], commit="x", created="t", root=ROOT)
        import phase1_func_run as fr
        self.assertEqual(fz["part"], "T-MC"); self.assertEqual(fz["mc"]["preset"], "mc_eval"); self.assertEqual(sorted(fz["items"]), ["eng_Latn", "hin_Deva"])
        self.assertEqual(fz["mc"]["preset_definition"], json.loads(json.dumps(fr.MC_PRESETS["mc_eval"])))                           # the runner's own definition, not a copy that can drift
        self.assertEqual((fz["mc"]["split"], fz["mc"]["n_items"], fz["mc"]["item_seed"], fz["mc"]["max_len"]), ("eval", 40, 0, 1024))
        for name in ("code_run", "code_tasks", "code_lib", "code_mc_analysis", "code_analysis", "code_calibration", "belebele_manifest", "passage_origin", "calibration_record"):
            self.assertIn(name, fz["files"])
        self.assertTrue(ff.verify_mc_freeze(fz, st["files"], lm, ff.sha256_file(ROOT / "reports/functional_validation/evaluation_freeze_lm.json"), st["calibration"], st["mc"], st["eval_keys"], st["analysis"]))
        other = self.state(n_items=41)
        with self.assertRaises(ValueError):
            ff.verify_mc_freeze(fz, st["files"], lm, ff.sha256_file(ROOT / "reports/functional_validation/evaluation_freeze_lm.json"), st["calibration"], other["mc"], other["eval_keys"], st["analysis"])


if __name__ == "__main__":
    unittest.main()
