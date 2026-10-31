"""Task 2 of the Gemma-2-9B three-arm plan: the frozen GPTQ calibration pool holds token ids of the Gemma-2-2B tokenizer; it may be used for Gemma-2-9B only if the two tokenizers are the same. The check compares the
vocabulary size, the special ids, a hash of the whole vocabulary and the encodings of probe strings in several scripts; any difference raises `TokenizerMismatch` naming the first differing field (stdlib only, fake tokenizers)."""
import sys
import unicodedata
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gptq_ext import tokenizer_check as tc


class Fake:
    """The part of a Hugging Face tokenizer the check uses."""
    def __init__(self, vocab=None, bos=2, eos=1, pad=0, unk=3, encode=None):
        self.vocab = vocab or {f"t{i}": i for i in range(50)}
        self.bos_token_id, self.eos_token_id, self.pad_token_id, self.unk_token_id = bos, eos, pad, unk
        self._encode = encode or (lambda s: [len(s) % 50, (sum(map(ord, s)) % 47) + 1])

    def get_vocab(self):
        return dict(self.vocab)

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(self._encode(text))

    def __len__(self):
        return len(self.vocab)


class Check(unittest.TestCase):
    def test_equal_tokenizers_pass_and_report_both_fingerprints(self):
        r = tc.check_same(Fake(), Fake())
        self.assertTrue(r["equal"])
        self.assertEqual(r["a"], r["b"])
        self.assertEqual(r["a"]["vocab_size"], 50)
        self.assertEqual(len(r["a"]["probe_ids"]), len(tc.PROBES))

    def test_a_different_vocab_size_raises_naming_vocab_size(self):
        small = Fake(vocab={f"t{i}": i for i in range(49)})
        with self.assertRaises(tc.TokenizerMismatch) as ctx:
            tc.check_same(Fake(), small)
        self.assertIn("vocab_size", str(ctx.exception))

    def test_a_swapped_token_id_changes_the_vocab_hash_and_raises(self):
        v = {f"t{i}": i for i in range(50)}; v["t4"], v["t5"] = 5, 4
        with self.assertRaises(tc.TokenizerMismatch) as ctx:
            tc.check_same(Fake(), Fake(vocab=v))
        self.assertIn("vocab_sha256", str(ctx.exception))

    def test_a_different_probe_encoding_raises_naming_probe_ids(self):
        other = Fake(encode=lambda s: [7, 7])
        with self.assertRaises(tc.TokenizerMismatch) as ctx:
            tc.check_same(Fake(), other)
        self.assertIn("probe_ids", str(ctx.exception))

    def test_different_special_ids_raise_naming_special_ids(self):
        with self.assertRaises(tc.TokenizerMismatch) as ctx:
            tc.check_same(Fake(), Fake(bos=9))
        self.assertIn("special_ids", str(ctx.exception))

    def test_the_probes_cover_the_scripts_of_the_roster_and_a_digit_string(self):
        names = {unicodedata.name(c, "") for c in "".join(tc.PROBES)}
        for script in ("LATIN", "CYRILLIC", "DEVANAGARI", "THAI", "KHMER", "CJK UNIFIED IDEOGRAPH", "ETHIOPIC", "DIGIT"):
            self.assertTrue(any(n.startswith(script) for n in names), script)
        self.assertEqual(len(tc.PROBES), 8)

    def test_the_fingerprint_does_not_depend_on_the_dict_order_of_the_vocab(self):
        v = {f"t{i}": i for i in range(50)}; w = dict(reversed(list(v.items())))
        self.assertEqual(tc.fingerprint(Fake(vocab=v)), tc.fingerprint(Fake(vocab=w)))


if __name__ == "__main__":
    unittest.main()
