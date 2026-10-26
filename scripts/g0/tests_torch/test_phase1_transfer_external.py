import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_transfer_external as te


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def make_panel(root, langs=("aaa_Latn", "bbb_Cyrl"), n_cal=2, n_eval=3, missing=()):
    """A tiny panel: jsonl text files and the freeze record of the real format."""
    freeze = {"languages": {}}
    for code in langs:
        if code in missing:
            freeze["languages"][code] = {"missing": True, "passages": [], "text_file": None, "n_calibration": 0, "n_evaluation": 0}; continue
        rows = []
        for r in range(n_cal + n_eval):
            text = " ".join([code, "passage", str(r), "text"] + ["x"] * r); rows.append({"article_id": str(100 + r), "rank": r, "split": "calibration" if r < n_cal else "evaluation", "text": text, "sha256": sha(text), "n_tokens": 4 + r})
        body = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows); (Path(root) / f"passages_{code}.jsonl").write_text(body)
        freeze["languages"][code] = {"missing": False, "n_calibration": n_cal, "n_evaluation": n_eval, "text_file": {"file": f"passages_{code}.jsonl", "sha256": hashlib.sha256(body.encode()).hexdigest()},
                                     "passages": [{k: x[k] for k in ("article_id", "rank", "split", "sha256", "n_tokens")} for x in rows]}
    return freeze


class Loader(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name); self.freeze = make_panel(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_evaluation_passages_come_back_in_rank_order_with_their_token_counts_and_the_language_order_of_the_freeze(self):
        p = te.load_passages(self.freeze, self.root)
        self.assertEqual(p["codes"], ["aaa_Latn", "bbb_Cyrl"]); self.assertEqual(p["texts"]["aaa_Latn"], ["aaa_Latn passage 2 text x x", "aaa_Latn passage 3 text x x x", "aaa_Latn passage 4 text x x x x"]); self.assertEqual(p["n_tokens"]["bbb_Cyrl"], [6, 7, 8])
        c = te.load_passages(self.freeze, self.root, split="calibration"); self.assertEqual(c["texts"]["aaa_Latn"], ["aaa_Latn passage 0 text", "aaa_Latn passage 1 text x"])

    def test_a_changed_text_file_a_changed_passage_or_a_changed_rank_is_an_error(self):
        f = self.root / "passages_aaa_Latn.jsonl"; good = f.read_text()
        f.write_text(good.replace("passage 3 ", "passage X "))
        with self.assertRaisesRegex(ValueError, "sha256"):
            te.load_passages(self.freeze, self.root)
        f.write_text(good); freeze = json.loads(json.dumps(self.freeze)); freeze["languages"]["aaa_Latn"]["passages"][3]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "passage"):
            te.load_passages(freeze, self.root)
        freeze = json.loads(json.dumps(self.freeze)); freeze["languages"]["aaa_Latn"]["passages"][3]["rank"] = 9
        with self.assertRaisesRegex(ValueError, "passage"):
            te.load_passages(freeze, self.root)

    def test_an_extra_row_or_an_edited_text_with_a_matching_file_hash_is_still_an_error(self):
        f = self.root / "passages_aaa_Latn.jsonl"; good = f.read_text()
        def with_file(body):
            fr = json.loads(json.dumps(self.freeze)); f.write_text(body); fr["languages"]["aaa_Latn"]["text_file"]["sha256"] = hashlib.sha256(body.encode()).hexdigest(); return fr
        extra = good + json.dumps({"article_id": "999", "rank": 5, "split": "evaluation", "text": "x", "sha256": sha("x"), "n_tokens": 1}) + "\n"
        with self.assertRaisesRegex(ValueError, "passages in the file"):
            te.load_passages(with_file(extra), self.root)
        edited = good.replace("passage 3 text x x x", "passage 3 TEXT x x x")                                                          # the file hash is made to match, the text no longer matches its own hash
        with self.assertRaisesRegex(ValueError, "passage of rank"):
            te.load_passages(with_file(edited), self.root)

    def test_a_missing_language_is_left_out_and_listed_and_never_replaced(self):
        freeze = make_panel(self.root, langs=("aaa_Latn", "bbb_Cyrl", "ccc_Hani"), missing=("bbb_Cyrl",)); p = te.load_passages(freeze, self.root)
        self.assertEqual(p["codes"], ["aaa_Latn", "ccc_Hani"]); self.assertEqual(p["missing"], ["bbb_Cyrl"]); self.assertNotIn("bbb_Cyrl", p["texts"])

    def test_the_passage_count_of_a_language_must_match_the_freeze(self):
        freeze = json.loads(json.dumps(self.freeze)); freeze["languages"]["aaa_Latn"]["n_evaluation"] = 4
        with self.assertRaisesRegex(ValueError, "evaluation"):
            te.load_passages(freeze, self.root)


class TokenCounts(unittest.TestCase):
    def test_every_passage_must_have_the_frozen_number_of_tokens(self):
        te.check_token_counts({"aaa": [[1, 2, 3], [4]]}, {"aaa": [3, 1]})
        with self.assertRaisesRegex(ValueError, "aaa"):
            te.check_token_counts({"aaa": [[1, 2, 3], [4]]}, {"aaa": [3, 2]})
        with self.assertRaisesRegex(ValueError, "number"):
            te.check_token_counts({"aaa": [[1, 2, 3]]}, {"aaa": [3, 3]})


CORE = {"feature_ids": [10, 11, 12, 13], "direction_ids": [11, 10, 13, 12], "feature_language_index": [0, 0, 1, 1], "languages": ["aaa_Latn", "bbb_Cyrl"], "contexts": [[0], [0], [0], [0]], "excluded_features": [], "kind": "intervention-transfer-core", "parameters": {"M": 16}, "source": {}, "schema": 1}


class Prepare(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name); self.freeze = make_panel(self.root); self.calls = []
        def encode(texts):
            self.calls.append(len(texts)); return [[hash(w) % 1000 + 1 for w in x.split()] for x in texts]
        self.encode = encode

    def tearDown(self):
        self.tmp.cleanup()

    def test_tokens_per_language_in_the_freeze_order_flattened_with_offsets_and_checked_against_the_frozen_counts(self):
        e = te.prepare_external(self.freeze, self.root, self.encode, max_len=256)
        self.assertEqual(e["codes"], ["aaa_Latn", "bbb_Cyrl"]); self.assertEqual([len(t) for t in e["tokens"]["aaa_Latn"]], [6, 7, 8]); self.assertEqual(len(e["flat"]), 6)
        self.assertEqual(e["offsets"], {"aaa_Latn": 0, "bbb_Cyrl": 3}); self.assertEqual(e["flat"][3], e["tokens"]["bbb_Cyrl"][0]); self.assertEqual(self.calls, [3, 3])

    def test_a_tokenizer_that_disagrees_with_the_freeze_or_a_passage_longer_than_the_limit_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "token count"):
            te.prepare_external(self.freeze, self.root, lambda texts: [[len(w) for w in x.split()[:-1]] for x in texts], max_len=256)
        with self.assertRaisesRegex(ValueError, "max_len"):
            te.prepare_external(self.freeze, self.root, self.encode, max_len=7)


class FromModel(unittest.TestCase):
    def test_the_activation_columns_are_the_features_of_the_core_in_the_order_of_the_core(self):
        from test_phase1_func_f5_run import ToyStack, ToySAE, sentences, batches_of, F
        import phase1_func_f5_run as r5
        stack, sae, sents = ToyStack(), ToySAE(), sentences(30)
        core = {**CORE, "feature_ids": [3, 1, 4, 0], "contexts": [[0]] * 4}; full = r5.max_activation_per_sentence(stack.lower, sae.encode, batches_of(sents), torch.arange(F), len(sents))
        out, ma = te.external_core_from_model(core, sents, stack.lower, sae.encode, batches_of, None, M=4, min_ctx=2)
        self.assertTrue(torch.equal(ma, full[:, [3, 1, 4, 0]])); self.assertEqual(out["parameters"]["n_passages"], 30); self.assertTrue(set(out["feature_ids"]) <= {3, 1, 4, 0})
        sel = r5.select_contexts(ma, 4, 2); kept = [k for k in range(4) if not bool(sel["excluded"][k])]; self.assertEqual(out["contexts"], [[int(c) for c in sel["contexts"][k].tolist() if c >= 0] for k in kept])


class Subset(unittest.TestCase):
    def test_the_external_core_must_be_a_subset_of_the_main_core_with_the_same_directions_and_languages(self):
        ext = {**CORE, "feature_ids": [10, 12], "direction_ids": [11, 13], "feature_language_index": [0, 1], "contexts": [[0], [1]]}; te.check_external_core_subset(CORE, ext)
        for key, bad in (("feature_ids", [10, 99]), ("direction_ids", [11, 12]), ("feature_language_index", [0, 0])):
            with self.assertRaisesRegex(ValueError, "main core"):
                te.check_external_core_subset(CORE, {**ext, key: bad})
        with self.assertRaisesRegex(ValueError, "languages"):
            te.check_external_core_subset(CORE, {**ext, "languages": ["bbb_Cyrl", "aaa_Latn"]})


class ExternalCore(unittest.TestCase):
    def test_contexts_follow_the_f5_rule_on_the_external_passages_and_rare_features_are_excluded_and_listed(self):
        ma = torch.zeros(6, 4)                                                                                       # 6 passages, 4 features
        ma[:, 0] = torch.tensor([0.0, 3.0, 5.0, 1.0, 5.0, 2.0])                                                      # feature 10: five positives, top 3 are passages 2 and 4 (tie: smaller index first), then 1
        ma[1:3, 1] = 1.0                                                                                              # feature 11: two positives only: excluded at min_ctx 3
        ma[:, 2] = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0]); ma[:, 3] = 0.5
        out = te.external_core(CORE, ma, M=3, min_ctx=3)
        self.assertEqual(out["feature_ids"], [10, 12, 13]); self.assertEqual(out["direction_ids"], [11, 13, 12]); self.assertEqual(out["feature_language_index"], [0, 1, 1])
        self.assertEqual(out["contexts"][0], [2, 4, 1]); self.assertEqual(out["contexts"][1], [0, 1, 2]); self.assertEqual(out["contexts"][2], [0, 1, 2])
        few = te.external_core(CORE, ma, M=3, min_ctx=2); self.assertEqual(few["contexts"][1], [1, 2]); self.assertEqual(few["feature_ids"], [10, 11, 12, 13])                  # two positives are enough at min_ctx 2 and the -1 padding is dropped
        self.assertEqual(out["excluded_features"], [{"feature": 11, "n_active": 2}]); self.assertEqual(out["languages"], CORE["languages"]); self.assertEqual(out["kind"], "intervention-transfer-core-external")
        self.assertEqual(out["parameters"]["M"], 3); self.assertEqual(out["parameters"]["min_ctx"], 3); self.assertEqual(out["parameters"]["n_passages"], 6)

    def test_the_activation_matrix_must_have_one_column_per_feature_of_the_core(self):
        with self.assertRaisesRegex(ValueError, "columns"):
            te.external_core(CORE, torch.zeros(5, 3), M=3, min_ctx=3)


class CoreRecord(unittest.TestCase):
    def test_the_record_pins_the_file_the_ids_and_the_contexts_and_a_changed_core_is_refused(self):
        core = te.external_core(CORE, torch.ones(6, 4), M=3, min_ctx=3); text = json.dumps(core, indent=1, sort_keys=True) + "\n"; rec = te.core_record(core, text)
        te.check_external_core(rec, core, text)
        other = json.loads(json.dumps(core)); other["contexts"][0] = [5, 4, 3]
        with self.assertRaisesRegex(ValueError, "contexts"):
            te.check_external_core(rec, other, json.dumps(other, indent=1, sort_keys=True) + "\n")
        with self.assertRaisesRegex(ValueError, "file"):
            te.check_external_core(rec, core, text + " ")
        shifted = json.loads(json.dumps(core)); shifted["feature_ids"] = shifted["feature_ids"][::-1]
        with self.assertRaisesRegex(ValueError, "feature"):
            te.check_external_core(rec, shifted, text)


if __name__ == "__main__":
    unittest.main()
