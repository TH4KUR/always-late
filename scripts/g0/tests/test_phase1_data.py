"""S3 tests for scripts/g0/phase1_data.py: the 27-language roster (tiers, Joshi classes, scripts) and the FLORES-200 reader. Stdlib only."""
import json, sys, tempfile, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
import phase1_data as pd_  # noqa: E402

ROSTER = ROOT / "reports" / "g0" / "phase1_languages.json"


class Roster(unittest.TestCase):
    def test_the_real_roster_file_satisfies_the_proposals_rules(self):
        r = pd_.load_roster(ROSTER)
        self.assertEqual(len(r), 27)
        for tier in ("high", "mid", "low"):
            langs = [x for x in r if x["tier"] == tier]
            self.assertEqual(len(langs), 9); self.assertGreaterEqual(len({x["script"] for x in langs}), 4)      # >= 4 scripts per tier (proposal section 5)

    def test_joshi_classes_match_the_authors_file_values_recorded_in_V5(self):
        r = {x["code"]: x for x in pd_.load_roster(ROSTER)}
        self.assertEqual([r[c]["joshi_class"] for c in ("eng_Latn", "por_Latn", "kor_Hang", "hin_Deva", "ben_Beng", "kan_Knda", "yor_Latn", "lao_Laoo")], [5, 4, 4, 4, 3, 1, 2, 2])
        self.assertEqual(r["zho_Hans"]["name"].lower(), "mandarin")              # FLORES-200 code is zho_Hans (the proposal's cmn_Hans alias does not exist there)

    def test_validation_rejects_bad_rosters(self):
        good = pd_.load_roster(ROSTER)
        with tempfile.TemporaryDirectory() as t:
            def write(rows):
                p = Path(t) / "r.json"; p.write_text(json.dumps({"languages": rows})); return p
            with self.assertRaisesRegex(ValueError, "26 languages"): pd_.load_roster(write(good[:-1]))              # the count check itself, not a side effect
            few = [dict(x) for x in good]
            for x in few:
                if x["tier"] == "high": x["script"] = "Latn"
            with self.assertRaisesRegex(ValueError, "four scripts"): pd_.load_roster(write(few))                  # a tier with fewer than four scripts
            dup = [dict(x) for x in good]; dup[1]["code"] = dup[0]["code"]
            with self.assertRaises(ValueError): pd_.load_roster(write(dup))                                       # duplicate code
            skew = [dict(x) for x in good]; skew[0]["tier"] = "low"
            with self.assertRaises(ValueError): pd_.load_roster(write(skew))                                      # tiers no longer 9 / 9 / 9
            bad = [dict(x) for x in good]; bad[0]["joshi_class"] = 9
            with self.assertRaises(ValueError): pd_.load_roster(write(bad))                                       # class outside 0-5


def fake_flores(root, codes, n_dev=3, n_dt=4):
    for split, n in (("dev", n_dev), ("devtest", n_dt)):
        d = Path(root) / "flores200_dataset" / split; d.mkdir(parents=True)
        for c in codes:
            (d / f"{c}.{split}").write_text("".join(f"{c} {split} sentence {i}\n" for i in range(n)), encoding="utf-8")


class FakeTok:
    def __init__(self): self.calls = []
    def encode(self, text, add_special_tokens=True):
        self.calls.append(add_special_tokens); return [10 + len(w) for w in text.split()]


class Flores(unittest.TestCase):
    def test_reads_lines_and_checks_the_expected_count(self):
        with tempfile.TemporaryDirectory() as t:
            fake_flores(t, ["eng_Latn"])
            self.assertEqual(len(pd_.read_flores_split(t, "eng_Latn", "devtest", expect=4)), 4)
            with self.assertRaises(ValueError): pd_.read_flores_split(t, "eng_Latn", "devtest", expect=1012)       # wrong size is an error, not a silent truncation
            with self.assertRaises(FileNotFoundError): pd_.read_flores_split(t, "xxx_Latn", "dev", expect=3)

    def test_build_corpus_tokenizes_without_special_tokens_and_reports_sizes(self):
        roster = [{"code": "eng_Latn"}, {"code": "deu_Latn"}]
        with tempfile.TemporaryDirectory() as t:
            fake_flores(t, ["eng_Latn", "deu_Latn"]); tok = FakeTok()
            data, rep = pd_.build_corpus(t, roster, "dev", tok, expect=3)
            self.assertEqual(sorted(data), [0, 1]); self.assertEqual(len(data[0]), 3)
            self.assertTrue(tok.calls and not any(tok.calls))                              # add_special_tokens=False every time (BOS is added by the regime, not the tokenizer)
            self.assertEqual(rep[0]["sentences"], 3); self.assertEqual(rep[0]["tokens"], sum(len(s) for s in data[0]))
            self.assertEqual(rep[1]["max_len"], max(len(s) for s in data[1])); self.assertEqual(rep[0]["code"], "eng_Latn")


if __name__ == "__main__":
    unittest.main()


class TextStats(unittest.TestCase):
    """Characters and whitespace words of the raw FLORES text, the denominators of fertility (Srivastava 2026, arXiv:2607.24276, section 3: word fertility = tokens per
    whitespace-delimited word, character fertility = tokens per character; V-11)."""
    def write(self, lines, code="eng_Latn", split="devtest", trailing=True):
        t = tempfile.TemporaryDirectory(); self.addCleanup(t.cleanup)
        d = Path(t.name) / "flores200_dataset" / split; d.mkdir(parents=True)
        (d / f"{code}.{split}").write_text("\n".join(lines) + ("\n" if trailing else ""), encoding="utf-8"); return t.name

    def test_counts_sentences_code_points_and_whitespace_words(self):
        root = self.write(["ab cd", "e", "  x   y  z "])
        s = pd_.text_stats(root, "eng_Latn", "devtest")
        self.assertEqual(s, {"sentences": 3, "chars": 5 + 1 + 11, "words": 2 + 1 + 3})      # the newline is not a character; split() collapses runs of spaces

    def test_characters_are_unicode_code_points_not_bytes(self):
        root = self.write(["ಕನ್ನಡ", "日本語です"], code="kan_Knda", split="dev")
        s = pd_.text_stats(root, "kan_Knda", "dev")
        self.assertEqual(s["chars"], len("ಕನ್ನಡ") + 5); self.assertEqual(s["words"], 2); self.assertTrue(s["chars"] < len("ಕನ್ನಡ".encode()) + len("日本語です".encode()))

    def test_no_trailing_newline_and_the_expected_count_check_agree_with_the_reader(self):
        root = self.write(["a b", "c"], trailing=False); self.assertEqual(pd_.text_stats(root, "eng_Latn", "devtest")["sentences"], 2)
        with self.assertRaises(ValueError): pd_.text_stats(root, "eng_Latn", "devtest", expect=1012)

    def test_all_languages_helper_returns_one_entry_per_code_and_split(self):
        root = self.write(["a b"]); self.assertEqual(pd_.text_stats_all(root, ["eng_Latn"], ("devtest",))["eng_Latn"]["devtest"]["words"], 2)
