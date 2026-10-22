import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phase1_func_mc_statements as ms

ROOT = Path(__file__).resolve().parents[3]
LANGS = ["aa", "bb", "cc"]
PRECS = ["FP", "INT8", "INT7", "INT6", "INT5", "INT4"]


def freeze():
    """The part of evaluation_freeze_mc.json that the statements tool reads."""
    st = [{"id": "S0", "precision": "FP", "claim": "the macro effect E over the covered languages is greater than 0", "sidedness": "one-sided", "interval_level": 0.95,
           "decision": "the lower bound of the percentile interval of the paired passage bootstrap is above 0"}]
    for k, p in enumerate(PRECS[1:], 1):
        st.append({"id": f"S{k}", "precision": p, "claim": "the macro retention of E is classified preserved, attenuated, amplified or inconclusive against the bands", "sidedness": "two-sided", "interval_level": 0.99,
                   "correction": "Bonferroni over the five precision statements (Q19 default a)", "decision": "preserved if the interval lies inside the bands, attenuated if entirely below, amplified if entirely above, inconclusive otherwise"})
    return {"schema": 1, "kind": "functional-validation-freeze", "part": "T-MC", "commit": "ff9d230", "precisions": {p: "mc_eval" for p in PRECS},
            "items": {l: {"n_items": 385, "keys_sha256": "x"} for l in LANGS},
            "analysis": {"bands": [0.75, 1.33], "n_boot": 2000, "seed": 0, "primary_K": "K64", "format": "letter", "covered": list(LANGS), "statements": st}}


def macro_ret(cls="preserved", ratio=1.02, lo=0.9, hi=1.1, n=3):
    if cls == "undefined":
        return {"ratio": None, "lo": None, "hi": None, "class": "undefined", "n_languages": 0, "languages": []}
    return {"ratio": ratio, "lo": lo, "hi": hi, "class": cls, "n_languages": n, "languages": LANGS[:n], "median_ratio": ratio, "median_lo": lo, "median_hi": hi}


def analysis(precs=PRECS, e_fp=(0.010, 0.004, 0.016, 2)):
    macro = {"FP": {"point": e_fp[0], "lo": e_fp[1], "hi": e_fp[2], "median": 0.01, "n_positive": e_fp[3]}}
    ret = {}
    for p in precs:
        if p != "FP":
            macro[p] = {"point": 0.009, "lo": 0.002, "hi": 0.015, "median": 0.009, "n_positive": 2}
            ret[p] = {"macro": macro_ret(), "pooled": {}, "languages": {}}
    macro = {p: macro[p] for p in precs}
    acc = [{"precision": p, "language": l, "n_items": 385, "acc_none": 0.5, "acc_zero": 0.5, "acc_control_mean": 0.5, "acc_sham": 0.5} for p in precs for l in LANGS]
    return {"task": "T-MC", "format": "letter", "K": "K64", "n_boot": 2000, "seed": 0, "bands": [0.75, 1.33], "covered": list(LANGS), "n_passages": 109, "n_positions": 109, "note": "Held-out evaluation split.",
            "retention_level": 0.99, "effects": {}, "macro": macro, "retention": ret, "sham": {}, "accuracy": acc}


def outcome(res, sid):
    return next(s for s in res["statements"] if s["id"] == sid)


class Outcomes(unittest.TestCase):
    def test_s0_is_met_only_when_the_lower_bound_of_the_95_percent_interval_is_above_zero(self):
        self.assertEqual(outcome(ms.evaluate(freeze(), analysis(e_fp=(0.01, 0.004, 0.016, 2))), "S0")["outcome"], "met")
        self.assertEqual(outcome(ms.evaluate(freeze(), analysis(e_fp=(0.05, -0.001, 0.1, 2))), "S0")["outcome"], "not met")          # a large point estimate does not decide
        self.assertEqual(outcome(ms.evaluate(freeze(), analysis(e_fp=(0.05, 0.0, 0.1, 2))), "S0")["outcome"], "not met")             # strictly above 0
        self.assertEqual(outcome(ms.evaluate(freeze(), analysis(e_fp=(-0.05, -0.1, -0.01, 0))), "S0")["outcome"], "not met")

    def test_s0_reports_its_numbers_and_the_frozen_wording(self):
        s = outcome(ms.evaluate(freeze(), analysis()), "S0")
        self.assertEqual((s["precision"], s["point"], s["lo"], s["hi"], s["interval_level"], s["sidedness"]), ("FP", 0.010, 0.004, 0.016, 0.95, "one-sided"))
        self.assertIn("greater than 0", s["claim"]); self.assertIn("+0.0100", s["detail"]); self.assertIn("[+0.0040, +0.0160]", s["detail"])

    def test_the_retention_statements_give_the_class_of_the_analysis_with_the_number_of_languages_and_the_99_percent_interval(self):
        a = analysis()
        for p, c in (("INT8", "preserved"), ("INT7", "attenuated"), ("INT6", "amplified"), ("INT5", "inconclusive")):
            a["retention"][p]["macro"] = macro_ret(c, 1.5, 1.2, 1.9, 2)
        res = ms.evaluate(freeze(), a)
        for sid, c in (("S1", "preserved"), ("S2", "attenuated"), ("S3", "amplified"), ("S4", "inconclusive")):
            s = outcome(res, sid); self.assertEqual(s["outcome"], c); self.assertEqual(s["n_languages"], 2); self.assertEqual(s["interval_level"], 0.99)
            self.assertIn("2 languages", s["detail"]); self.assertIn("99 %", s["detail"]); self.assertIn("[1.20, 1.90]", s["detail"])
        self.assertEqual([outcome(res, f"S{k}")["precision"] for k in range(1, 6)], ["INT8", "INT7", "INT6", "INT5", "INT4"])

    def test_no_language_with_a_clearly_positive_full_precision_effect_is_undefined_not_inconclusive(self):
        a = analysis(); a["retention"]["INT4"]["macro"] = macro_ret("undefined")
        s = outcome(ms.evaluate(freeze(), a), "S5")
        self.assertEqual(s["outcome"], "undefined"); self.assertEqual(s["n_languages"], 0); self.assertIn("no covered language", s["detail"])

    def test_a_precision_that_is_not_in_the_analysis_has_no_data_and_the_result_is_marked_incomplete(self):
        res = ms.evaluate(freeze(), analysis(precs=["FP", "INT8", "INT4"]))
        self.assertEqual(outcome(res, "S2")["outcome"], "no data"); self.assertEqual(outcome(res, "S5")["outcome"], "preserved")
        self.assertFalse(res["complete"]); self.assertEqual(res["missing_precisions"], ["INT7", "INT6", "INT5"])
        self.assertTrue(ms.evaluate(freeze(), analysis())["complete"])

    def test_the_context_has_the_macro_effect_of_every_precision_in_the_analysis(self):
        res = ms.evaluate(freeze(), analysis())
        self.assertEqual([c["precision"] for c in res["context"]], PRECS)
        c = res["context"][0]; self.assertEqual((c["precision"], c["point"], c["lo"], c["hi"], c["n_positive"], c["n_covered"]), ("FP", 0.010, 0.004, 0.016, 2, 3))

    def test_the_input_is_not_modified(self):
        f, a = freeze(), analysis(); f0, a0 = copy.deepcopy(f), copy.deepcopy(a); ms.evaluate(f, a)
        self.assertEqual((f, a), (f0, a0))


class Refusals(unittest.TestCase):
    """Each mismatch is its own test, so that no other check can mask it."""
    def refused(self, mutate_analysis=None, mutate_freeze=None, word=None):
        f, a = freeze(), analysis()
        if mutate_analysis:
            mutate_analysis(a)
        if mutate_freeze:
            mutate_freeze(f)
        with self.assertRaises(ms.ProvenanceError) as cm:
            ms.evaluate(f, a)
        if word:
            self.assertIn(word, str(cm.exception))
        return str(cm.exception)

    def test_the_task_the_format_and_the_group_size(self):
        self.refused(lambda a: a.update(task="T-LM"), word="task")
        self.refused(lambda a: a.update(format="cloze"), word="format")
        self.refused(lambda a: a.update(K="K16"), word="K")

    def test_the_bootstrap_settings_and_the_bands(self):
        self.refused(lambda a: a.update(n_boot=500), word="n_boot")
        self.refused(lambda a: a.update(seed=1), word="seed")
        self.refused(lambda a: a.update(bands=[0.8, 1.25]), word="bands")

    def test_the_covered_languages_and_their_order(self):
        self.refused(lambda a: a.update(covered=["aa", "bb"]), word="covered")
        self.refused(lambda a: a.update(covered=["bb", "aa", "cc"]), word="covered")

    def test_the_retention_level_must_be_the_one_of_the_statements(self):
        self.refused(lambda a: a.update(retention_level=0.95), word="retention_level")

    def test_the_calibration_split_is_refused_by_its_note(self):
        self.refused(lambda a: a.update(note="Calibration-split run (dev-origin passages), a pipeline check, not a result."), word="calibration")

    def test_item_counts_must_equal_the_frozen_ones_in_every_row(self):
        def one(a):
            a["accuracy"][4]["n_items"] = 120
        msg = self.refused(one, word="n_items")                                                              # row 4 of the planted accuracy list is INT8 / bb
        self.assertIn("n_items: INT8: 1 language(s) differ from the freeze (bb has 120 against 385)", msg); self.assertEqual(msg.count("n_items"), 1)      # one line per precision, with the count and the first examples
        def short(a):
            a["accuracy"] = [r for r in a["accuracy"] if not (r["precision"] == "INT5" and r["language"] == "cc")]
        self.refused(short, word="cc")                                                                      # a covered language without items at a precision

    def test_an_unknown_precision_and_a_missing_full_precision_run(self):
        def extra(a):                                                                                       # a complete extra precision (effect, retention block, item rows): only "not in the freeze" can fire
            a["macro"]["INT3"] = a["macro"]["INT4"]; a["retention"]["INT3"] = a["retention"]["INT4"]
            a["accuracy"] += [dict(r, precision="INT3") for r in a["accuracy"] if r["precision"] == "INT4"]
        self.assertIn("INT3: precision is not in the freeze", self.refused(extra, word="INT3"))
        def nofp(a):
            del a["macro"]["FP"]
        self.refused(nofp, word="FP")

    def test_a_precision_with_an_effect_but_no_retention_block_is_refused(self):
        def noret(a):
            del a["retention"]["INT6"]
        self.refused(noret, word="retention: INT6")

    def test_a_freeze_of_another_part_is_refused(self):
        self.refused(mutate_freeze=lambda f: f.update(part="T-LM"), word="part")

    def test_every_mismatch_is_listed_in_one_error(self):
        def many(a):
            a.update(seed=3, format="cloze", n_boot=10)
        msg = self.refused(many)
        for w in ("seed", "format", "n_boot"):
            self.assertIn(w, msg)


class Outputs(unittest.TestCase):
    def setUp(self):
        self.res = ms.evaluate(freeze(), analysis())

    def test_the_report_states_that_the_statements_are_the_frozen_ones_and_lists_all_of_them(self):
        txt = ms.report_text(self.res)
        self.assertIn("their wording, sidedness, levels and decision rules are not changed here", txt)
        self.assertTrue(txt.startswith("# ")); self.assertIn("as frozen", txt); self.assertIn("exploratory", txt.lower()); self.assertIn("ff9d230", txt)
        for sid in ("S0", "S1", "S2", "S3", "S4", "S5"):
            self.assertIn(sid, txt)
        for w in ("undefined", "no data"):
            self.assertIn(w, txt)                                                                           # the legend explains both
        for w in ("confirmed", "confirms", "proves", "demonstrates", "shows that"):                      # the report states outcomes, it makes no claim ("confirmatory" is the name of the family)
            self.assertNotIn(w, txt.lower())

    def test_the_report_has_the_outcomes_and_the_context_table(self):
        txt = ms.report_text(self.res)
        self.assertIn("| S0 | FP |", txt); self.assertIn("| met |", txt); self.assertIn("| INT8 |", txt)
        a = analysis(); a["retention"]["INT4"]["macro"] = macro_ret("undefined"); self.assertIn("| undefined |", ms.report_text(ms.evaluate(freeze(), a)))

    def test_an_incomplete_analysis_says_so_at_the_top(self):
        txt = ms.report_text(ms.evaluate(freeze(), analysis(precs=["FP", "INT8", "INT4"])))
        self.assertIn("INCOMPLETE", txt.split("\n\n")[1] + txt.split("\n\n")[2]); self.assertIn("INT7, INT6, INT5", txt)
        self.assertNotIn("INCOMPLETE", ms.report_text(self.res))

    def test_main_writes_json_markdown_and_csv_and_nothing_when_the_provenance_fails(self):
        d = Path(tempfile.mkdtemp()); (d / "f.json").write_text(json.dumps(freeze())); (d / "a.json").write_text(json.dumps(analysis()))
        ms.main(["--freeze", str(d / "f.json"), "--analysis", str(d / "a.json"), "--out", str(d / "out")])
        j = json.loads((d / "out" / "statements.json").read_text()); self.assertEqual([s["id"] for s in j["statements"]], ["S0", "S1", "S2", "S3", "S4", "S5"])
        self.assertTrue((d / "out" / "statements.md").read_text().startswith("# "))
        rows = (d / "out" / "statements.csv").read_text().splitlines(); self.assertEqual(len(rows), 7); self.assertTrue(rows[0].startswith("id,precision"))
        bad = analysis(); bad["seed"] = 9; (d / "b.json").write_text(json.dumps(bad))
        with self.assertRaises(SystemExit):
            ms.main(["--freeze", str(d / "f.json"), "--analysis", str(d / "b.json"), "--out", str(d / "out2")])
        self.assertFalse((d / "out2").exists())


class RealFiles(unittest.TestCase):
    """Schema coupling against the committed freeze and the committed Belebele pilot analysis."""
    def setUp(self):
        self.freeze = json.loads((ROOT / "reports/functional_validation/evaluation_freeze_mc.json").read_text())
        self.pilot = json.loads((ROOT / "reports/functional_validation/results/mc_pilot/analysis.json").read_text())

    def test_the_calibration_pilot_is_refused_for_what_it_is(self):
        with self.assertRaises(ms.ProvenanceError) as cm:
            ms.evaluate(self.freeze, self.pilot)
        msg = str(cm.exception); self.assertIn("calibration", msg.lower()); self.assertIn("n_items", msg)                 # the note and the item counts (120 against 385)

    def test_with_only_its_provenance_fields_replaced_the_pilot_is_read_without_error_so_the_schema_matches(self):
        a = copy.deepcopy(self.pilot); a["note"] = "Held-out evaluation split."
        for r in a["accuracy"]:
            r["n_items"] = 385
        res = ms.evaluate(self.freeze, a)
        self.assertEqual(outcome(res, "S0")["outcome"], "not met")                                                   # the pilot's full-precision effect is about zero
        self.assertEqual(outcome(res, "S5")["precision"], "INT4"); self.assertEqual(outcome(res, "S5")["n_languages"], 1)
        self.assertEqual([outcome(res, f"S{k}")["outcome"] for k in range(1, 5)], ["no data"] * 4); self.assertFalse(res["complete"])


if __name__ == "__main__":
    unittest.main()
