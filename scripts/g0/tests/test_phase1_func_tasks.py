import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phase1_func_tasks as ft


def item(link, qn, passage, q="Q?", opts=("o1", "o2", "o3", "o4"), ans="2", code="eng_Latn"):
    return {"link": link, "question_number": qn, "flores_passage": passage, "question": q, "mc_answer1": opts[0], "mc_answer2": opts[1],
            "mc_answer3": opts[2], "mc_answer4": opts[3], "correct_answer_num": ans, "dialect": code, "ds": "2023-05-03"}


DEV = ["Alpha one.", "Beta two is here.", "Gamma three.", "Delta four stays."]
DEVTEST = ["Epsilon five.", "Zeta six goes.", "Eta seven.", "Theta eight ends."]


class WordTokenizer:
    """tokenize(text) -> ids: one id per whitespace-separated piece, a leading space is a separate piece ' ' (SentencePiece-like for the tests)."""
    def __init__(self):
        self.vocab = {}

    def __call__(self, text):
        out, cur = [], ""
        for ch in text:
            if ch == " ":
                if cur:
                    out.append(self._id(cur)); cur = ""
                cur = " "
            elif cur == " ":
                cur = " " + ch
            else:
                cur += ch
        if cur:
            out.append(self._id(cur))
        return out

    def _id(self, s):
        return self.vocab.setdefault(s, len(self.vocab) + 10)


class LoadBelebele(unittest.TestCase):
    def write(self, rows, name="eng_Latn"):
        d = Path(tempfile.mkdtemp())
        (d / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        return d

    def test_loads_and_validates(self):
        rows = [item("l1", 1, "p"), item("l1", 2, "p", ans="4")]
        got = ft.load_belebele(self.write(rows), "eng_Latn", expect=2)
        self.assertEqual(len(got), 2)
        self.assertEqual(got[1]["answer"], 3)            # zero-based index of the correct option
        self.assertEqual(got[0]["options"], ["o1", "o2", "o3", "o4"])
        self.assertEqual(got[0]["passage_key"], "l1")

    def test_wrong_count_is_an_error(self):
        with self.assertRaises(ValueError):
            ft.load_belebele(self.write([item("l1", 1, "p")]), "eng_Latn", expect=900)

    def test_bad_answer_number_is_an_error(self):
        for bad in ("0", "5", "x", ""):
            with self.assertRaises(ValueError):
                ft.load_belebele(self.write([item("l1", 1, "p", ans=bad)]), "eng_Latn", expect=1)

    def test_missing_field_is_an_error(self):
        r = item("l1", 1, "p"); del r["mc_answer3"]
        with self.assertRaises(ValueError):
            ft.load_belebele(self.write([r]), "eng_Latn", expect=1)

    def test_empty_option_is_an_error(self):
        with self.assertRaises(ValueError):
            ft.load_belebele(self.write([item("l1", 1, "p", opts=("a", "", "c", "d"))]), "eng_Latn", expect=1)


class Origin(unittest.TestCase):
    """Belebele passages are built from FLORES sentences but may skip sentences and carry CSV-doubled quotation marks, so the origin is decided by which sentences of which file the passage CONTAINS."""
    def test_dev_devtest_none_and_mixed(self):
        items = [item("a", 1, " ".join(DEV[0:2])), item("b", 1, " ".join(DEVTEST[1:4])), item("c", 1, "Some other text entirely, with no FLORES sentence in it."),
                 item("d", 1, DEV[3])]
        origin = ft.passage_origin(items, DEV, DEVTEST, min_chars=5)
        self.assertEqual(origin, {"a": "dev", "b": "devtest", "c": "none", "d": "dev"})
        self.assertEqual(ft.passage_origin([item("e", 1, DEV[0] + " " + DEVTEST[0])], DEV, DEVTEST, min_chars=5), {"e": "mixed"})
        self.assertEqual(ft.passage_origin([item("e", 1, "Same line here.")], ["Same line here."], ["Same line here."], min_chars=5), {"e": "mixed"})

    def test_skipped_sentences_and_extra_text_are_tolerated(self):
        origin = ft.passage_origin([item("a", 1, DEV[0] + " " + DEV[2]), item("b", 1, "Prefix " + DEV[0] + " and some suffix")], DEV, DEVTEST, min_chars=5)
        self.assertEqual(origin, {"a": "dev", "b": "dev"})

    def test_whitespace_and_quotation_marks_are_normalised(self):
        dev = ['He said "Alpha one is here" and left.', "Beta two is here."]
        passage = '"He said ""Alpha one is here"" and left.\n Beta  two is here."'
        self.assertEqual(ft.passage_origin([item("a", 1, passage)], dev, DEVTEST, min_chars=5)["a"], "dev")
        self.assertEqual(ft.passage_origin([item("a", 1, "He said \u201cAlpha one is here\u201d and left.")], dev, DEVTEST, min_chars=5)["a"], "dev")

    def test_a_passage_that_only_matches_after_whitespace_normalisation(self):
        self.assertEqual(ft.passage_origin([item("a", 1, "Beta  two\n is   here.")], DEV, DEVTEST, min_chars=5), {"a": "dev"})

    def test_short_lines_do_not_decide_the_origin(self):
        self.assertEqual(ft.passage_origin([item("a", 1, "Yes. It was the best one.")], ["Yes."], ["It was the best one."], min_chars=10), {"a": "devtest"})
        self.assertEqual(ft.passage_origin([item("a", 1, "Yes. Something else.")], ["Yes."], DEVTEST, min_chars=10), {"a": "none"})

    def test_a_line_of_exactly_min_chars_counts(self):
        self.assertEqual(ft.passage_origin([item("a", 1, "xx abcdefghij yy")], ["abcdefghij"], DEVTEST, min_chars=10), {"a": "dev"})
        self.assertEqual(ft.passage_origin([item("a", 1, "xx abcdefghij yy")], ["abcdefghij"], DEVTEST, min_chars=11), {"a": "none"})

    def test_hit_counts_are_reported(self):
        origin, hits = ft.passage_origin_detail([item("a", 1, DEV[0] + " " + DEV[1] + " " + DEVTEST[2])], DEV, DEVTEST, min_chars=5)
        self.assertEqual(origin["a"], "mixed")
        self.assertEqual(hits["a"], {"dev": 2, "devtest": 1})

    def test_inconsistent_items_of_one_passage_are_an_error(self):
        with self.assertRaises(ValueError):
            ft.passage_origin([item("a", 1, DEV[0]), item("a", 2, DEV[1])], DEV, DEVTEST, min_chars=5)


class CrossLanguage(unittest.TestCase):
    def test_agreement_across_languages_and_alignment_by_key(self):
        en = [item("a", 1, "x"), item("b", 1, "y"), item("c", 1, "z")]
        hi = [item("c", 1, "zz"), item("a", 1, "xx"), item("b", 1, "yy")]             # another order, same keys
        ok, bad = ft.agree_origin({"eng": {"a": "dev", "b": "devtest", "c": "none"}, "hin": {"a": "dev", "b": "devtest", "c": "none"}})
        self.assertEqual((ok, bad), ({"a": "dev", "b": "devtest", "c": "none"}, []))
        ok, bad = ft.agree_origin({"eng": {"a": "dev", "b": "devtest"}, "hin": {"a": "dev", "b": "dev"}})
        self.assertEqual(ok["b"], "excluded"); self.assertEqual(bad, ["b"])
        self.assertEqual(ok["a"], "dev")
        self.assertEqual([i["link"] for i in ft.align_by_key(hi, [("a", 1), ("b", 1), ("c", 1)])], ["a", "b", "c"])
        with self.assertRaises(KeyError):
            ft.align_by_key(hi, [("a", 1), ("zzz", 1)])


class Split(unittest.TestCase):
    def test_split_by_origin(self):
        items = [item("a", 1, "x"), item("a", 2, "x"), item("b", 1, "y"), item("c", 1, "z"), item("d", 1, "w")]
        origin = {"a": "dev", "b": "devtest", "c": "mixed", "d": "none"}   # "excluded" (cross-language disagreement) is handled like mixed and none
        s = ft.split_items(items, origin)
        self.assertEqual([i["question_number"] for i in s["cal"]], [1, 2])
        self.assertEqual([i["passage_key"] if "passage_key" in i else i["link"] for i in s["eval"]], ["b"])
        self.assertEqual(len(s["excluded"]), 2)
        self.assertEqual(len(s["cal"]) + len(s["eval"]) + len(s["excluded"]), len(items))

    def test_split_is_language_independent(self):
        origin = {"a": "dev", "b": "devtest"}
        en = [item("a", 1, "x"), item("b", 1, "y")]
        hi = [item("a", 1, "xx", code="hin_Deva"), item("b", 1, "yy", code="hin_Deva")]
        se, sh = ft.split_items(en, origin), ft.split_items(hi, origin)
        self.assertEqual([i["link"] for i in se["eval"]], [i["link"] for i in sh["eval"]])

    def test_unknown_passage_is_an_error(self):
        with self.assertRaises(KeyError):
            ft.split_items([item("zzz", 1, "x")], {"a": "dev"})


class LmIds(unittest.TestCase):
    def test_disjoint_sorted_and_deterministic(self):
        a = ft.lm_ids(1012, 150, 450, seed=7)
        self.assertEqual((len(a["pilot"]), len(a["eval"]), len(a["reserve"])), (150, 450, 412))
        allids = a["pilot"] + a["eval"] + a["reserve"]
        self.assertEqual(sorted(allids), list(range(1012)))
        for k in a:
            self.assertEqual(a[k], sorted(a[k]))
        self.assertEqual(a, ft.lm_ids(1012, 150, 450, seed=7))
        self.assertNotEqual(a["eval"], ft.lm_ids(1012, 150, 450, seed=8)["eval"])

    def test_too_many_requested_is_an_error(self):
        with self.assertRaises(ValueError):
            ft.lm_ids(100, 60, 60, seed=0)

    def test_pilot_and_eval_do_not_change_when_the_other_size_changes(self):
        a, b = ft.lm_ids(1012, 150, 450, seed=3), ft.lm_ids(1012, 150, 300, seed=3)
        self.assertEqual(a["pilot"], b["pilot"])
        self.assertTrue(set(b["eval"]) <= set(a["eval"]))


class Prompts(unittest.TestCase):
    def setUp(self):
        self.it = {"passage": "P text.", "question": "What?", "options": ["red", "green", "blue", "black"], "answer": 2}

    def test_letter_prompt_has_options_in_order_and_no_key(self):
        text = ft.format_prompt(self.it, "letter")
        lines = text.split("\n")
        self.assertEqual(lines[-1], "Answer:")
        self.assertEqual(lines[2:6], ["A. red", "B. green", "C. blue", "D. black"])
        self.assertNotIn("correct", text.lower())

    def test_cloze_context_has_no_options(self):
        text = ft.format_prompt(self.it, "cloze")
        for o in self.it["options"]:
            self.assertNotIn(o, text)
        self.assertTrue(text.endswith("Answer:"))

    def test_unknown_format_is_an_error(self):
        with self.assertRaises(ValueError):
            ft.format_prompt(self.it, "other")

    def test_encode_letter_gives_four_distinct_single_token_candidates(self):
        tok = WordTokenizer()
        e = ft.encode_mc(tok, self.it, "letter")
        self.assertEqual(len(e["cand_ids"]), 4)
        self.assertEqual(len(set(e["cand_ids"])), 4)
        self.assertEqual(e["answer"], 2)
        self.assertEqual(e["tokens"], tok(ft.format_prompt(self.it, "letter")))

    def test_encode_cloze_gives_option_token_lists(self):
        tok = WordTokenizer()
        e = ft.encode_mc(tok, self.it, "cloze")
        self.assertEqual(len(e["option_ids"]), 4)
        self.assertTrue(all(len(o) >= 1 for o in e["option_ids"]))
        self.assertEqual(e["tokens"], tok(ft.format_prompt(self.it, "cloze")))
        full = tok(ft.format_prompt(self.it, "cloze") + " green")
        self.assertEqual(full, e["tokens"] + e["option_ids"][1])

    def test_letters_that_share_a_token_are_an_error(self):
        class Collapsing:
            def __call__(self, text):
                return [1, 2, 3] + ([7] if text[-2:] in (" A", " B", " C", " D") else [])
        with self.assertRaises(ValueError):
            ft.encode_mc(Collapsing(), self.it, "letter")

    def test_inconsistent_tokenisation_is_an_error(self):
        class Bad:
            def __call__(self, text):
                return [1, 2, 3] if text.endswith("Answer:") else [9, 9, 9, 9]
        with self.assertRaises(ValueError):
            ft.encode_mc(Bad(), self.it, "cloze")

    def test_build_drops_overlong_items_and_counts_them(self):
        tok = WordTokenizer()
        n = len(tok(ft.format_prompt(self.it, "letter")))
        long_it = dict(self.it, passage=" ".join(["w"] * 50))
        kept, dropped = ft.build_mc_items(tok, [self.it, long_it], "letter", max_len=n + 2)
        self.assertEqual((len(kept), dropped), (1, 1))
        self.assertEqual(len(ft.build_mc_items(tok, [self.it], "letter", max_len=n)[0]), 1)         # a prompt of exactly max_len tokens is kept
        self.assertEqual(ft.build_mc_items(tok, [self.it], "letter", max_len=n - 1), ([], 1))


if __name__ == "__main__":
    unittest.main()
