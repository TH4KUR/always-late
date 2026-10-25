import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phase1_func_promptcheck as pc
import phase1_func_run as fr
import phase1_func_tasks as ft

ITEM = {"passage_key": "p1", "question_number": 3, "passage": "The cat sat. It was warm.", "question": "Where did the cat sit?", "options": ["on the mat", "in the car", "under the bed", "on the roof"], "answer": 0, "code": "eng_Latn"}


def toy_tokenize(text):
    """Words with their leading space are one token, newlines and lone spaces are tokens: ' A' is one token and appending ' A' extends the tokenisation of a prompt that ends in 'Answer:'."""
    table = pc_table
    return [table.setdefault(t, len(table)) for t in re.findall(r" ?[^ \n]+|\n| ", text)]


pc_table = {}


class HarnessText(unittest.TestCase):
    def test_the_text_is_the_harness_template_with_the_fields_of_the_item(self):
        want = "P: The cat sat. It was warm.\nQ: Where did the cat sit?\nA: on the mat\nB: in the car\nC: under the bed\nD: on the roof\nAnswer:"
        self.assertEqual(pc.harness_prompt(ITEM), want)

    def test_the_question_is_stripped_and_nothing_else_is_changed(self):
        it = dict(ITEM, question="  Where did the cat sit?  \n", passage="A  b\nc", options=["x  y", "z", "w", "v"])
        t = pc.harness_prompt(it)
        self.assertIn("\nQ: Where did the cat sit?\nA: x  y\n", t); self.assertTrue(t.startswith("P: A  b\nc\nQ:"))      # the passage and the options are not touched

    def test_it_differs_from_the_frozen_letter_prompt_in_the_scaffolding_only(self):
        frozen = ft.format_prompt(ITEM, "letter"); h = pc.harness_prompt(ITEM)
        self.assertNotEqual(frozen, h)
        for text in ("The cat sat. It was warm.", "Where did the cat sit?", "on the mat", "in the car", "under the bed", "on the roof"):
            self.assertIn(text, frozen); self.assertIn(text, h)
        self.assertTrue(frozen.startswith("Passage: ") and frozen.endswith("Answer:") and h.endswith("Answer:"))
        self.assertIn("\nA. on the mat\n", frozen); self.assertIn("\nA: on the mat\n", h)

    def test_the_item_must_have_four_options(self):
        with self.assertRaises(ValueError):
            pc.harness_prompt(dict(ITEM, options=["a", "b", "c"]))


class Patch(unittest.TestCase):
    def test_the_harness_variant_changes_the_letter_prompt_only_inside_the_block(self):
        before = ft.format_prompt
        with pc.use_variant("harness"):
            self.assertEqual(ft.format_prompt(ITEM, "letter"), pc.harness_prompt(ITEM))
        self.assertIs(ft.format_prompt, before); self.assertTrue(ft.format_prompt(ITEM, "letter").startswith("Passage: "))

    def test_the_cloze_prompt_is_never_changed(self):
        want = ft.format_prompt(ITEM, "cloze")
        with pc.use_variant("harness"):
            self.assertEqual(ft.format_prompt(ITEM, "cloze"), want)
            with self.assertRaises(ValueError):
                ft.format_prompt(ITEM, "other")                                                              # the frozen error for an unknown format stays

    def test_the_frozen_variant_changes_nothing(self):
        before = ft.format_prompt
        with pc.use_variant("frozen"):
            self.assertIs(ft.format_prompt, before)

    def test_the_function_is_restored_after_an_error_and_an_unknown_variant_is_refused(self):
        before = ft.format_prompt
        with self.assertRaises(RuntimeError):
            with pc.use_variant("harness"):
                raise RuntimeError("boom")
        self.assertIs(ft.format_prompt, before)
        with self.assertRaises(ValueError):
            with pc.use_variant("other"):
                pass
        self.assertIs(ft.format_prompt, before)

    def test_the_patch_reaches_the_encoder_of_the_frozen_task_module(self):
        e0 = ft.encode_mc(toy_tokenize, ITEM, "letter")
        with pc.use_variant("harness"):
            e1 = ft.encode_mc(toy_tokenize, ITEM, "letter")
        self.assertNotEqual(e0["tokens"], e1["tokens"]); self.assertEqual(e0["answer"], e1["answer"]); self.assertEqual(len(e1["cand_ids"]), 4)
        self.assertEqual(ft.encode_mc(toy_tokenize, ITEM, "letter")["tokens"], e0["tokens"])                  # restored: the frozen tokens again
        self.assertEqual(toy_tokenize(pc.harness_prompt(ITEM)), e1["tokens"])

    def test_the_candidates_are_the_same_four_letter_tokens_in_both_variants(self):
        e0 = ft.encode_mc(toy_tokenize, ITEM, "letter")
        with pc.use_variant("harness"):
            e1 = ft.encode_mc(toy_tokenize, ITEM, "letter")
        self.assertEqual(e0["cand_ids"], e1["cand_ids"])                                                      # ' A' to ' D' score the same tokens


class Main(unittest.TestCase):
    def test_main_runs_the_frozen_runner_inside_the_variant_and_passes_the_other_arguments(self):
        seen = {}
        def fake(argv):
            seen["argv"] = list(argv); seen["prompt"] = ft.format_prompt(ITEM, "letter")
        old = fr.main_mc; fr.main_mc = fake
        try:
            pc.main(["--variant", "harness", "--out", "o", "--languages", "eng_Latn"])
            self.assertEqual(seen["argv"], ["--out", "o", "--languages", "eng_Latn"]); self.assertEqual(seen["prompt"], pc.harness_prompt(ITEM))
            pc.main(["--variant", "frozen", "--out", "o2"])
            self.assertTrue(seen["prompt"].startswith("Passage: ")); self.assertEqual(seen["argv"], ["--out", "o2"])
        finally:
            fr.main_mc = old
        self.assertTrue(ft.format_prompt(ITEM, "letter").startswith("Passage: "))

    def test_the_variant_is_required_and_the_runner_is_not_called_without_it(self):
        called = []
        old = fr.main_mc; fr.main_mc = lambda argv: called.append(argv)
        try:
            with self.assertRaises(SystemExit):
                pc.main(["--out", "o"])
        finally:
            fr.main_mc = old
        self.assertEqual(called, [])

    def test_the_wrapper_refuses_a_cloze_run_because_only_the_letter_prompt_is_replaced(self):
        old = fr.main_mc; fr.main_mc = lambda argv: None
        try:
            with self.assertRaises(SystemExit):
                pc.main(["--variant", "harness", "--formats", "letter", "cloze", "--out", "o"])
            pc.main(["--variant", "harness", "--formats", "letter", "--out", "o"])                          # letter only is fine
        finally:
            fr.main_mc = old


class Freeze(unittest.TestCase):
    def test_the_frozen_runner_and_task_module_are_not_the_wrapper(self):
        self.assertTrue(hasattr(fr, "main_mc")); self.assertFalse(hasattr(ft, "harness_prompt")); self.assertFalse(hasattr(fr, "use_variant"))


if __name__ == "__main__":
    unittest.main()
