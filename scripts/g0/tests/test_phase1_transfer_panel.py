import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phase1_transfer_panel as tp


def words(text):
    return len(text.split())


class Sentences(unittest.TestCase):
    def test_sentences_end_at_final_punctuation_of_several_scripts(self):
        self.assertEqual(tp.split_sentences("One two. Three four! Five?"), ["One two.", "Three four!", "Five?"])
        self.assertEqual(tp.split_sentences("这是第一句。这是第二句！第三句？"), ["这是第一句。", "这是第二句！", "第三句？"])
        self.assertEqual(tp.split_sentences("यह पहला वाक्य है। यह दूसरा है।"), ["यह पहला वाक्य है।", "यह दूसरा है।"])

    def test_closing_quotes_stay_with_their_sentence_and_a_tail_without_punctuation_is_kept(self):
        self.assertEqual(tp.split_sentences('He said "stop." Then left'), ['He said "stop."', "Then left"])

    def test_headings_and_blank_lines_are_not_sentences_and_paragraph_breaks_do_not_join_sentences(self):
        text = "Title line\n\nFirst sentence here. Second one.\n\nSection heading\nThird sentence."
        self.assertEqual(tp.split_sentences(text), ["First sentence here.", "Second one.", "Third sentence."])

    def test_a_full_stop_inside_a_number_or_an_abbreviation_without_space_does_not_end_a_sentence(self):
        self.assertEqual(tp.split_sentences("The value is 3.5 percent. Next one."), ["The value is 3.5 percent.", "Next one."])

    def test_empty_text_has_no_sentences(self):
        self.assertEqual(tp.split_sentences(""), []); self.assertEqual(tp.split_sentences("\n\n"), [])


class Normalise(unittest.TestCase):
    def test_case_spacing_punctuation_and_width_are_removed(self):
        self.assertEqual(tp.normalize("  Hello,   World! "), tp.normalize("hello world")); self.assertEqual(tp.normalize("ＡＢＣ"), "abc")
        self.assertNotEqual(tp.normalize("hello world"), tp.normalize("hello there"))

    def test_scripts_without_case_survive(self):
        self.assertEqual(tp.normalize("你好，世界。"), "你好世界"); self.assertEqual(tp.normalize("Привет, мир"), "приветмир")


class Script(unittest.TestCase):
    def test_the_fraction_of_letters_in_the_expected_script(self):
        self.assertEqual(tp.script_fraction("Hello World 123", "latin"), 1.0); self.assertEqual(tp.script_fraction("Привет hello", "cyrillic"), 6 / 11)
        self.assertEqual(tp.script_fraction("你好 world", "han"), 2 / 7); self.assertEqual(tp.script_fraction("नमस्ते", "devanagari"), 1.0)

    def test_yoruba_letters_with_dot_below_are_latin_and_text_without_letters_is_zero(self):
        self.assertEqual(tp.script_fraction("Ọmọ ẹ̀ṣọ́", "latin"), 1.0); self.assertEqual(tp.script_fraction("1234 !!", "latin"), 0.0)

    def test_an_unknown_script_is_an_error(self):
        with self.assertRaises(ValueError):
            tp.script_fraction("abc", "klingon")


class Order(unittest.TestCase):
    def test_articles_are_ordered_by_the_hash_of_their_id_and_the_salt(self):
        ids = ["7", "3", "9", "1"]
        got = tp.article_order(ids, "s")
        self.assertEqual(got, sorted(ids, key=lambda i: hashlib.sha256(f"s:{i}".encode()).hexdigest()))
        self.assertEqual(tp.article_order(list(reversed(ids)), "s"), got); self.assertNotEqual(tp.article_order([str(i) for i in range(20)], "s"), tp.article_order([str(i) for i in range(20)], "t"))


class Cut(unittest.TestCase):
    def test_sentences_are_added_from_the_start_while_they_fit(self):
        s = ["a b c.", "d e f g.", "h i."]
        out = tp.cut_passage(s, words, 4, 8)
        self.assertEqual(out["text"], "a b c. d e f g."); self.assertEqual((out["n_tokens"], out["n_sentences"]), (7, 2))

    def test_a_passage_below_the_minimum_is_refused_and_a_first_sentence_above_the_maximum_too(self):
        self.assertIsNone(tp.cut_passage(["a b."], words, 4, 8))
        self.assertIsNone(tp.cut_passage(["a b c d e f g h i j."], words, 4, 8))
        self.assertIsNone(tp.cut_passage([], words, 4, 8))

    def test_the_boundary_values_are_inclusive(self):
        self.assertEqual(tp.cut_passage(["a b c d."], words, 4, 8)["n_tokens"], 4); self.assertEqual(tp.cut_passage(["a b c d e f g h."], words, 4, 8)["n_tokens"], 8)

    def test_the_token_count_is_of_the_whole_joined_text_not_the_sum_of_sentence_counts(self):
        count = lambda t: len(t.replace(".", " . ").split())                                                          # a tokenizer that splits the full stop off
        out = tp.cut_passage(["a b.", "c d."], count, 3, 7)
        self.assertEqual(out["n_tokens"], count("a b. c d.")); self.assertEqual(out["text"], "a b. c d.")


class Language(unittest.TestCase):
    CFG = {"code": "eng_Latn", "script": "latin", "filter_characters": ""}

    def rows(self, n, prefix="Alpha beta gamma delta. Epsilon zeta eta theta."):
        return [(str(i), f"{prefix} Article {i} here.") for i in range(n)]

    def test_the_first_acceptable_passages_in_hash_order_are_split_into_calibration_then_evaluation(self):
        res = tp.build_language(self.rows(50), self.CFG, set(), words, n_calibration=3, n_evaluation=5, min_tokens=4, max_tokens=60, salt="s", min_script_fraction=0.8)
        self.assertEqual(len(res["passages"]), 8); self.assertEqual([p["split"] for p in res["passages"]], ["calibration"] * 3 + ["evaluation"] * 5)
        order = tp.article_order([str(i) for i in range(50)], "s"); self.assertEqual([p["article_id"] for p in res["passages"]], order[:8])
        self.assertFalse(res["missing"]); self.assertEqual(len({p["article_id"] for p in res["passages"]}), 8)

    def test_a_passage_with_a_sentence_of_flores_is_dropped_and_counted(self):
        flores = {tp.normalize("Alpha beta gamma delta.")}
        res = tp.build_language(self.rows(10), self.CFG, flores, words, 2, 2, 4, 60, "s", 0.8)
        self.assertEqual(len(res["passages"]), 0); self.assertTrue(res["missing"]); self.assertEqual(res["rejected"]["flores_duplicate"], 10)

    def test_the_traditional_character_filter_drops_passages_for_simplified_chinese(self):
        cfg = {"code": "zho_Hans", "script": "han", "filter_characters": "個們"}
        rows = [("1", "这是一个很长的句子，用来测试。我们喜欢这个。"), ("2", "這是一個很長的句子，用來測試。我們喜歡這個。個們。"), ("3", "他们说了很多话。她们也来了。")]
        res = tp.build_language(rows, cfg, set(), lambda t: len(t) // 2, 2, 1, 4, 60, "s", 0.8)
        self.assertEqual(sorted(p["article_id"] for p in res["passages"]), ["1", "3"]); self.assertEqual(res["rejected"]["filter_character"], 1)
        self.assertEqual({p["article_id"]: p["text"] for p in res["passages"]}["1"], "这是一个很长的句子，用来测试。我们喜欢这个。")      # no spaces are inserted between Chinese sentences

    def test_wrong_script_short_and_long_passages_are_counted_under_their_reason(self):
        rows = [("1", "Привет мир, это длинный текст на другом языке. И еще предложение."), ("2", "Tiny."), ("3", " ".join(["word"] * 100) + "."), ("4", "Good text with enough words here. Another good sentence follows.")]
        res = tp.build_language(rows, self.CFG, set(), words, 2, 0, 4, 60, "s", 0.8)
        self.assertEqual([p["article_id"] for p in res["passages"]], ["4"]); self.assertEqual(res["rejected"]["script"], 1); self.assertEqual(res["rejected"]["length"], 2)

    def test_too_few_passages_mark_the_language_missing_and_never_pad_it(self):
        res = tp.build_language(self.rows(3), self.CFG, set(), words, 2, 3, 4, 60, "s", 0.8)
        self.assertTrue(res["missing"]); self.assertEqual(len(res["passages"]), 3); self.assertEqual(res["needed"], 5)

    def test_a_document_is_used_once_even_if_listed_twice(self):
        rows = self.rows(4) + [("0", "DUPLICATE row of the same article. Nothing else here.")]
        res = tp.build_language(rows, self.CFG, set(), words, 3, 2, 4, 60, "s", 0.8)
        self.assertEqual(len(res["passages"]), 4); self.assertEqual(len({p["article_id"] for p in res["passages"]}), 4)
        self.assertNotIn("DUPLICATE", " ".join(p["text"] for p in res["passages"]))                          # the first row of an id wins

    def test_every_passage_carries_its_text_hash_token_count_and_position_in_the_order(self):
        res = tp.build_language(self.rows(10), self.CFG, set(), words, 1, 1, 4, 60, "s", 0.8); p = res["passages"][0]
        self.assertEqual(p["sha256"], hashlib.sha256(p["text"].encode()).hexdigest()); self.assertEqual(p["rank"], 0); self.assertEqual(res["passages"][1]["rank"], 1)
        self.assertEqual(p["n_tokens"], words(p["text"]))


class Records(unittest.TestCase):
    def test_the_freeze_record_has_hashes_and_no_text(self):
        res = tp.build_language([("1", "Alpha beta gamma delta. Epsilon zeta eta theta.")], {"code": "eng_Latn", "script": "latin", "filter_characters": ""}, set(), words, 1, 0, 4, 60, "s", 0.8)
        rec = tp.freeze_record({"eng_Latn": res}, {"eng_Latn": {"file": "x", "sha256": "ab"}})
        self.assertNotIn("Alpha", json.dumps(rec)); self.assertEqual(rec["languages"]["eng_Latn"]["passages"][0]["article_id"], "1")
        self.assertEqual(rec["languages"]["eng_Latn"]["n_calibration"], 1); self.assertEqual(rec["languages"]["eng_Latn"]["text_file"], {"file": "x", "sha256": "ab"})
        self.assertEqual(rec["languages"]["eng_Latn"]["missing"], False); self.assertEqual(rec["languages"]["eng_Latn"]["needed"], 1)        # one calibration passage was needed and found

    def test_the_split_counts_of_a_record_with_both_splits(self):
        rows = [(str(i), f"Alpha beta gamma delta. Epsilon zeta eta theta. Article {i} here.") for i in range(10)]
        res = tp.build_language(rows, {"code": "eng_Latn", "script": "latin", "filter_characters": ""}, set(), words, 2, 3, 4, 60, "s", 0.8)
        rec = tp.freeze_record({"eng_Latn": res}, {})["languages"]["eng_Latn"]
        self.assertEqual((rec["n_calibration"], rec["n_evaluation"], rec["needed"], rec["missing"]), (2, 3, 5, False)); self.assertEqual([p["split"] for p in rec["passages"]], ["calibration"] * 2 + ["evaluation"] * 3)

    def test_the_frozen_constants(self):
        self.assertEqual((tp.MIN_TOKENS, tp.MAX_TOKENS, tp.N_CALIBRATION, tp.N_EVALUATION, tp.MIN_SCRIPT_FRACTION), (64, 128, 100, 300, 0.8)); self.assertEqual(tp.SALT, "intervention-transfer-external-v1")

    def test_the_languages_of_the_panel_and_their_shards_are_the_frozen_ones(self):
        self.assertEqual(list(tp.PANEL), ["eng_Latn", "spa_Latn", "rus_Cyrl", "hin_Deva", "zho_Hans", "yor_Latn"])
        self.assertEqual([v["config"] for v in tp.PANEL.values()], ["en", "es", "ru", "hi", "zh", "yo"])
        self.assertEqual(tp.PANEL["zho_Hans"]["filter_characters"], "這個們來說為與對會時學國將後開長關點從還麼種現發裡當問應頭見無動經讓樣實體電機車東書氣門馬歲親邊運進")
        self.assertEqual(tp.SHARDS["yo"]["file"], "20231101.yo/train-00000-of-00001.parquet"); self.assertEqual(tp.REVISION, "b04c8d1ceb2f5cd4588862100d08de323dccfbaa")
        self.assertEqual(tp.SHARDS["en"]["sha256"], "382e7f6f09e488b24793a7f7cfc659879d5a22da2cf2efec6491665f0c019677")


if __name__ == "__main__":
    unittest.main()
