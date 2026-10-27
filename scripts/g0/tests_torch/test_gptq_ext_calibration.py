"""Task 4 of the GPTQ extension plan: the multilingual calibration pool (8,192 content tokens per language, sequences of at most 128 tokens with their own BOS, independent of FLORES and of the external-panel articles).
A fake character-level tokenizer stands in for Gemma's; the real build runs on the Ada login node (`calibration.main`)."""
import json
import sys
import unittest
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
G0 = HERE.parent
ROOT = G0.parents[1]
sys.path.insert(0, str(G0))
from gptq_ext import calibration as cal  # noqa: E402

BOS, PAD = 2, 0


def encode(text):
    return [ord(c) + 3 for c in text]


def decode(ids):
    return "".join(chr(i - 3) for i in ids)


def make_rows(n, length=300, prefix="art", letters="abcdefghij"):
    return [(f"{prefix}{i}", "".join(letters[(i + j) % len(letters)] for j in range(length))) for i in range(n)]


def build(rows, budget=1000, exclude=frozenset(), flores=frozenset(), cfg=None, **kw):
    cfg = cfg or {"config": "xx", "script": "Latn", "filter_characters": ""}
    return cal.build_language(rows, cfg, flores, set(exclude), encode, decode, budget_tokens=budget, salt=kw.get("salt", cal.SALT), max_len=128, min_tokens=64, min_script_fraction=0.8)


class Build(unittest.TestCase):
    def test_budget_is_content_tokens_and_the_last_sequence_is_truncated_to_fit(self):
        res = build(make_rows(40), budget=1000)
        self.assertEqual(sum(p["n_used"] for p in res["passages"]), 1000)
        self.assertEqual(res["shortfall"], 0)
        self.assertTrue(all(1 <= p["n_used"] <= 128 for p in res["passages"]))
        self.assertEqual([p["n_used"] for p in res["passages"]][:-1], [128] * (len(res["passages"]) - 1))
        self.assertEqual(len(res["passages"]), 8)                       # 7 x 128 + 104

    def test_pool_reports_shortfall_and_never_fills(self):
        res = build(make_rows(3), budget=1000)
        self.assertEqual(len(res["passages"]), 3)
        self.assertEqual(res["shortfall"], 1000 - 3 * 128)
        short = build([("a", "x" * 10), ("b", "y" * 200)], budget=1000)
        self.assertEqual(short["rejected"]["length"], 1)

    def test_pool_excludes_external_article_ids_and_normalised_flores_duplicates(self):
        rows = make_rows(40)
        res = build(rows, budget=1000, exclude={"art3", "art4"})
        self.assertFalse({"art3", "art4"} & {p["article_id"] for p in res["passages"]})
        dup = cal.normalize(rows[5][1][10:90])                          # a FLORES line occurring inside article 5
        res2 = build(rows, budget=1000, flores={dup})
        self.assertNotIn("art5", [p["article_id"] for p in res2["passages"]])
        self.assertGreaterEqual(res2["rejected"]["flores_duplicate"], 1)

    def test_an_article_needs_at_least_sixty_four_tokens_and_a_passage_is_cut_at_128(self):
        res = build([("a", "x" * 64), ("b", "y" * 63), ("c", "z" * 5000)], budget=100000)
        by = {p["article_id"]: p for p in res["passages"]}
        self.assertEqual(sorted(by), ["a", "c"])
        self.assertEqual((by["a"]["n_tokens"], by["c"]["n_tokens"]), (64, 128))
        self.assertEqual(res["rejected"]["length"], 1)

    def test_excluded_article_counts_are_reported(self):
        res = build(make_rows(40), budget=100000, exclude={"art3", "art4"})
        self.assertEqual(res["rejected"]["excluded_article"], 2)

    def test_pool_is_deterministic_and_the_salt_changes_the_order(self):
        a, b = build(make_rows(40)), build(make_rows(40))
        self.assertEqual(cal.pool_sha256({"x": a}), cal.pool_sha256({"x": b}))
        c = build(make_rows(40), salt="another")
        self.assertNotEqual([p["article_id"] for p in a["passages"]], [p["article_id"] for p in c["passages"]])

    def test_script_filter_and_the_traditional_chinese_filter(self):
        rows = [("a", "Привет мир " * 40), ("b", "hello world " * 40)]
        res = build(rows, budget=100000)
        self.assertEqual([p["article_id"] for p in res["passages"]], ["b"])
        self.assertEqual(res["rejected"]["script"], 1)
        zh = {"config": "zh", "script": "Hans", "filter_characters": "這個們"}
        res = build([("a", "这是简体中文文本" * 30), ("b", "這個們是傳統" * 40)], budget=100000, cfg=zh)
        self.assertEqual([p["article_id"] for p in res["passages"]], ["a"])
        self.assertEqual(res["rejected"]["filter_character"], 1)

    def test_every_script_of_the_roster_has_a_prefix_and_recognises_its_own_letters(self):
        sample = {"Latn": "a", "Cyrl": "я", "Hans": "中", "Jpan": "の", "Hang": "한", "Deva": "क", "Beng": "ক", "Taml": "க", "Knda": "ಕ", "Telu": "క", "Thai": "ก", "Khmr": "ក", "Laoo": "ກ", "Ethi": "አ"}
        for s, ch in sample.items():
            self.assertEqual(cal.script_fraction(ch * 10, s), 1.0, s)
        self.assertEqual(cal.script_fraction("a" * 10, "Cyrl"), 0.0)
        self.assertEqual(cal.script_fraction("1234", "Latn"), 0.0)


class Table(unittest.TestCase):
    def test_the_language_table_covers_the_roster_and_the_pinned_shards(self):
        roster = json.loads((ROOT / "reports/g0/phase1_languages.json").read_text())["languages"]
        self.assertEqual([r["code"] for r in roster], list(cal.LANGS))
        shards = json.loads((ROOT / "reports/gptq_extension/wikipedia_first_shards.json").read_text())
        for r in roster:
            cfg = cal.LANGS[r["code"]]
            self.assertEqual(cfg["script"], r["script"], r["code"])
            self.assertIn(cfg["config"], shards, r["code"])
            self.assertTrue(shards[cfg["config"]]["file"].startswith(f"20231101.{cfg['config']}/train-00000-of-"))
            self.assertEqual(len(shards[cfg["config"]]["sha256"]), 64)

    def test_the_six_panel_shards_are_the_ones_the_external_panel_pinned(self):
        import phase1_transfer_panel as pp
        shards = json.loads((ROOT / "reports/gptq_extension/wikipedia_first_shards.json").read_text())
        for c, v in pp.SHARDS.items():
            self.assertEqual((v["file"], v["sha256"]), (shards[c]["file"], shards[c]["sha256"]))

    def test_a_language_missing_from_the_external_panel_has_no_excluded_articles(self):
        ids = cal.external_article_ids({"languages": {"a": {"missing": False, "passages": [{"article_id": 1}, {"article_id": "2"}]}, "b": {"missing": True, "passages": []}}})
        self.assertEqual(ids, {"a": {"1", "2"}})

    def test_the_external_panel_article_ids_are_loaded_per_language(self):
        ids = cal.external_article_ids(json.loads((ROOT / "reports/intervention_transfer/freeze_external.json").read_text()))
        self.assertEqual(set(ids), {"eng_Latn", "spa_Latn", "rus_Cyrl", "hin_Deva", "zho_Hans", "yor_Latn"} & set(ids))
        self.assertTrue(all(len(v) > 0 for v in ids.values()))
        self.assertTrue(all(isinstance(i, str) for v in ids.values() for i in v))


class Freeze(unittest.TestCase):
    def freeze_and_dir(self, d):
        pool = {"aaa_Latn": build(make_rows(40), budget=300), "bbb_Latn": build(make_rows(40, prefix="b"), budget=200)}
        files = cal.write_text_files(pool, d)
        freeze = {"pool_sha256": cal.pool_sha256(pool), "languages": {c: {"shortfall": r["shortfall"], "text_file": files[c], "passages": [{k: p[k] for k in ("article_id", "rank", "n_tokens", "n_used", "sha256")} for p in r["passages"]]}
                                                                          for c, r in pool.items()}}
        return pool, freeze

    def test_load_pool_round_trips_and_verifies_everything(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            pool, freeze = self.freeze_and_dir(d)
            back = cal.load_pool(freeze, d)
            self.assertEqual(cal.pool_sha256(back), cal.pool_sha256(pool))
            self.assertEqual({c: r["tokens"] for c, r in back.items()}, {"aaa_Latn": 300, "bbb_Latn": 200})

    def test_load_pool_detects_a_tampered_file_a_changed_hash_and_a_missing_passage(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            pool, freeze = self.freeze_and_dir(d)
            fp = Path(d) / "calibration_aaa_Latn.jsonl"; good = fp.read_text()
            fp.write_text(good.replace("[", "[7,", 1))
            with self.assertRaises(ValueError):
                cal.load_pool(freeze, d)
            fp.write_text(good)
            bad = json.loads(json.dumps(freeze)); bad["languages"]["aaa_Latn"]["passages"] = bad["languages"]["aaa_Latn"]["passages"][:-1]
            with self.assertRaises(ValueError):
                cal.load_pool(bad, d)
            bad = json.loads(json.dumps(freeze)); bad["pool_sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                cal.load_pool(bad, d)

    def test_load_pool_detects_a_changed_file_even_when_its_content_parses_the_same(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            pool, freeze = self.freeze_and_dir(d)
            fp = Path(d) / "calibration_bbb_Latn.jsonl"; fp.write_text(fp.read_text() + "\n")
            with self.assertRaises(ValueError):
                cal.load_pool(freeze, d)

    def test_load_pool_detects_an_inconsistent_count_even_when_the_file_hash_is_refreshed(self):
        import tempfile
        import phase1_transfer_panel as pp
        with tempfile.TemporaryDirectory() as d:
            pool, freeze = self.freeze_and_dir(d)
            fp = Path(d) / "calibration_aaa_Latn.jsonl"
            rows = [json.loads(line) for line in fp.read_text().splitlines()]; rows[0]["n_used"] = 5
            fp.write_text("".join(json.dumps(r) + "\n" for r in rows)); freeze["languages"]["aaa_Latn"]["text_file"]["sha256"] = pp.sha256_file(fp)
            with self.assertRaises(ValueError):
                cal.load_pool(freeze, d)


class Batches(unittest.TestCase):
    def pool(self):
        return {"aaa_Latn": build(make_rows(40), budget=300), "bbb_Latn": build(make_rows(40, prefix="b"), budget=200)}

    def test_sequences_have_one_leading_bos_padding_is_masked_and_bos_is_kept(self):
        batches = cal.pool_batches(self.pool(), bos_id=BOS, pad_id=PAD, batch_size=4)
        for b in batches:
            self.assertTrue((b["tokens"][:, 0] == BOS).all())
            self.assertTrue(b["mask"][:, 0].all())
            self.assertTrue(((b["tokens"][:, 1:] == BOS).sum() == 0).item())
            self.assertTrue(((b["tokens"] == PAD) == ~b["mask"]).all())
            self.assertTrue((b["mask"].sum(1) - 1 <= 128).all())
        self.assertEqual(sum(int(b["mask"].sum()) - b["tokens"].shape[0] for b in batches), 300 + 200)

    def test_batches_follow_the_pool_order_and_are_deterministic(self):
        p = self.pool()
        a, b = cal.pool_batches(p, BOS, PAD, 4), cal.pool_batches(p, BOS, PAD, 4)
        self.assertTrue(all(torch.equal(x["tokens"], y["tokens"]) for x, y in zip(a, b)))
        self.assertEqual([x["lang"] for x in a][0], "aaa_Latn")

    def test_pool_sha256_changes_when_one_token_changes(self):
        p = self.pool(); h = cal.pool_sha256(p)
        p["aaa_Latn"]["passages"][0]["ids"][3] += 1
        self.assertNotEqual(h, cal.pool_sha256(p))

    def test_pool_batches_refuse_a_pool_with_a_shortfall_unless_allowed(self):
        p = {"aaa_Latn": build(make_rows(2), budget=1000)}
        with self.assertRaises(ValueError):
            cal.pool_batches(p, BOS, PAD, 4)
        self.assertTrue(cal.pool_batches(p, BOS, PAD, 4, allow_shortfall=True))


if __name__ == "__main__":
    unittest.main()
