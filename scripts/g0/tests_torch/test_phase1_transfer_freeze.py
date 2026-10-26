import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phase1_transfer_freeze as tf

LANGS = ["aa", "bb", "cc"]
DEV = 5          # dev sentences per language in the synthetic pool
M = 3


def sample(per_lang=(6, 4, 3)):
    """A synthetic F5-style sample: feature ids 10, 11, ... in language order, every feature with one role."""
    feats, i = {}, 10
    for lang, n in zip(LANGS, per_lang):
        for _ in range(n):
            feats[str(i)] = {"language": lang, "roles": ["r1"]}; i += 1
    return {"languages": LANGS, "features": feats}


def stage_a(smp, per_cell=None, seed=0, n_active=None, contexts=None, excluded=None, dev_lang=None, direction=None):
    """A stage-A file consistent with the sample: contexts are the first M dev sentences by feature id, direction = another feature of the same language."""
    import phase1_func_f5_run as fr
    ids, groups = fr.sample_arrays(smp, per_cell=per_cell, seed=seed); ids = ids.numpy(); groups = groups.numpy(); n = len(ids)
    ctx = np.full((n, M), -1, dtype=np.int64) if contexts is None else np.array(contexts)
    if contexts is None:
        for k in range(n):
            ctx[k] = [(k + j) % (DEV * len(LANGS)) for j in range(M)]
    if direction is None:
        direction = np.zeros(n, dtype=np.int64)
        for k in range(n):
            same = [int(x) for x in ids[groups == groups[k]] if int(x) != int(ids[k])]; direction[k] = same[0]
    return {"feat_ids": ids, "feat_lang": groups, "contexts": ctx, "n_ctx": np.full(n, M), "n_active": np.full(n, 99) if n_active is None else np.array(n_active),
            "excluded": np.zeros(n, dtype=bool) if excluded is None else np.array(excluded), "dev_lang": np.repeat(np.arange(len(LANGS)), DEV) if dev_lang is None else np.array(dev_lang),
            "direction_ids": direction}


class Core(unittest.TestCase):
    def setUp(self):
        self.smp = sample(); self.sa = stage_a(self.smp)

    def derive(self, sa=None, smp=None, **kw):
        return tf.derive_core(smp or self.smp, sa or self.sa, per_cell=kw.get("per_cell"), subset_seed=kw.get("seed", 0), dev_per_language=DEV, M=M, min_ctx=2)

    def test_the_core_lists_the_features_their_languages_directions_and_contexts_of_the_stage_file(self):
        c = self.derive()
        self.assertEqual(c["feature_ids"], list(range(10, 23))); self.assertEqual(c["languages"], LANGS)
        self.assertEqual(c["feature_language_index"], [0] * 6 + [1] * 4 + [2] * 3)
        self.assertEqual(len(c["direction_ids"]), 13); self.assertEqual(c["contexts"][0], [0, 1, 2]); self.assertEqual(c["contexts"][1], [1, 2, 3])
        self.assertEqual(c["excluded_features"], [])

    def test_a_stage_file_of_another_feature_subset_is_refused(self):
        sa = dict(self.sa); sa["feat_ids"] = self.sa["feat_ids"][::-1].copy()
        with self.assertRaises(ValueError):
            self.derive(sa)
        sa = stage_a(self.smp, per_cell=2, seed=0)
        with self.assertRaises(ValueError):                                                      # the stage file was built with per_cell 2, the core is asked for all features
            self.derive(sa)
        self.derive(sa, per_cell=2)                                                               # and it is accepted when the subset rule is the same

    def test_a_wrong_language_of_a_feature_or_a_dev_pool_that_is_not_language_major_is_refused(self):
        sa = dict(self.sa); sa["feat_lang"] = self.sa["feat_lang"][::-1].copy()
        with self.assertRaises(ValueError):
            self.derive(sa)
        sa = dict(self.sa); sa["dev_lang"] = np.roll(self.sa["dev_lang"], 1)
        with self.assertRaises(ValueError):
            self.derive(sa)

    def test_the_direction_must_be_another_feature_of_the_same_language(self):
        d = self.sa["direction_ids"].copy(); d[0] = 20                                          # 20 is a feature of language bb
        with self.assertRaises(ValueError):
            self.derive(stage_a(self.smp, direction=d))
        d = self.sa["direction_ids"].copy(); d[0] = int(self.sa["feat_ids"][0])
        with self.assertRaises(ValueError):
            self.derive(stage_a(self.smp, direction=d))

    def test_contexts_must_be_valid_dev_indices_without_repeats_inside_a_feature(self):
        c = self.sa["contexts"].copy(); c[2, 1] = DEV * len(LANGS)
        with self.assertRaises(ValueError):
            self.derive(stage_a(self.smp, contexts=c))
        c = self.sa["contexts"].copy(); c[2, 1] = c[2, 0]
        with self.assertRaises(ValueError):
            self.derive(stage_a(self.smp, contexts=c))

    def test_padding_marks_the_end_of_the_contexts_and_an_excluded_feature_is_listed_and_dropped(self):
        c = self.sa["contexts"].copy(); c[0, 2] = -1
        core = self.derive(stage_a(self.smp, contexts=c)); self.assertEqual(core["contexts"][0], [0, 1])
        c2 = self.sa["contexts"].copy(); c2[0, 1] = -1                                           # a hole before a valid entry
        with self.assertRaises(ValueError):
            self.derive(stage_a(self.smp, contexts=c2))
        core = self.derive(stage_a(self.smp, excluded=[True] + [False] * 12))
        self.assertEqual(core["excluded_features"], [10]); self.assertNotIn(10, core["feature_ids"]); self.assertEqual(len(core["contexts"]), 12)
        self.assertEqual(len(core["direction_ids"]), 12); self.assertEqual(len(core["feature_language_index"]), 12)

    def test_the_source_cluster_of_a_dev_sentence_is_its_position_in_the_language(self):
        self.assertEqual([tf.source_cluster(i, 997) for i in (0, 996, 997, 1993, 26918)], [0, 996, 0, 996, 996])
        self.assertEqual(tf.source_cluster(7, DEV), 2)
        c = self.derive(); self.assertEqual(c["context_clusters"][1], [1, 2, 3])
        self.assertEqual(self.derive()["context_clusters"][4], [tf.source_cluster(i, DEV) for i in c["contexts"][4]])


class Split(unittest.TestCase):
    def test_every_language_has_both_halves_the_calibration_half_is_the_larger_one_and_the_halves_are_disjoint(self):
        ids = list(range(10, 23)); langs = [0] * 6 + [1] * 4 + [2] * 3
        s = tf.split_features(ids, langs, "salt")
        self.assertEqual(sorted(s["calibration"] + s["heldout"]), ids); self.assertEqual(set(s["calibration"]) & set(s["heldout"]), set())
        for lang, n in ((0, 6), (1, 4), (2, 3)):
            members = [i for i, l in zip(ids, langs) if l == lang]
            self.assertEqual(sum(i in s["calibration"] for i in members), (n + 1) // 2); self.assertEqual(sum(i in s["heldout"] for i in members), n // 2)

    def test_the_split_follows_the_hash_order_not_the_order_of_the_input_and_depends_on_the_salt(self):
        ids = list(range(10, 23)); langs = [0] * 6 + [1] * 4 + [2] * 3
        a = tf.split_features(ids, langs, "salt"); perm = list(reversed(range(13)))
        b = tf.split_features([ids[i] for i in perm], [langs[i] for i in perm], "salt")
        self.assertEqual(a, b)
        c = tf.split_features(ids, langs, "another salt"); self.assertNotEqual(a["calibration"], c["calibration"])
        first = sorted((i for i, l in zip(ids, langs) if l == 0), key=lambda i: hashlib.sha256(f"salt:{i}".encode()).hexdigest())[:3]
        self.assertEqual(sorted(first), sorted(i for i in a["calibration"] if i in range(10, 16)))

    def test_a_language_with_one_feature_cannot_be_split_and_is_an_error(self):
        with self.assertRaises(ValueError):
            tf.split_features([1, 2, 3], [0, 0, 1], "salt")

    def test_lists_are_sorted_and_plain_integers(self):
        s = tf.split_features([5, 3, 9, 4], [0, 0, 1, 1], "x")
        self.assertEqual(s["calibration"], sorted(s["calibration"])); self.assertTrue(all(type(i) is int for i in s["calibration"] + s["heldout"]))
        json.dumps(s)


class Spec(unittest.TestCase):
    def test_the_spec_encodes_the_choices_written_in_the_ledger_before_any_output(self):
        s = tf.SPEC
        self.assertEqual(s["states"], ["FP", "INT16", "INT6", "INT4"]); self.assertEqual(s["quantized_states"], ["INT6", "INT4"]); self.assertEqual(s["floor_state"], "INT16")
        self.assertEqual(s["cells"], ["FP/FP", "Q/FP", "FP/Q", "Q/Q", "Q/Q-native"]); self.assertEqual(s["direction_kinds"], ["real", "matched_swapped", "random1", "random2", "random3"])
        self.assertEqual(s["random_direction_seeds"], [1, 2, 3]); self.assertEqual(s["bootstrap"], {"replicates": 2000, "seed": 20261026})
        self.assertEqual(s["floor"], {"percentile": 95, "eligibility_multiple": 5}); self.assertEqual(s["ridge_strengths"], [0.1, 1, 10])
        self.assertEqual(s["utility_rule"]["relative_mae_reduction"], 0.10); self.assertEqual(s["budget"], {"elapsed_hours": 48, "gpu_hours": 32, "concurrent_jobs": 2})
        self.assertEqual(s["models"], ["gemma-2-2b"]); self.assertIn("gradient", s["baselines"]); self.assertEqual(s["stored_statistics"]["gram_cells"], 5)
        self.assertEqual(s["feature_subset"], {"per_cell": 3, "subset_seed": 0, "M": 16, "min_ctx": 4, "dev_per_language": 997})
        self.assertEqual(s["group_panel"], {"languages": 27, "evaluation_sentences": 300, "group_size": "K64", "control_draws": 3})

    def test_the_expectation_that_the_context_term_dominates_is_written_as_a_hypothesis_before_the_data(self):
        text = json.dumps(tf.SPEC["hypotheses"])
        for k in ("H1", "H2", "H3", "H4", "A3"):
            self.assertIn(k, tf.SPEC["hypotheses"])
        self.assertIn("|C| > |W|", tf.SPEC["hypotheses"]["A3"]); self.assertIn("INT4", text)

    def test_the_traditional_character_filter_has_fifty_distinct_cjk_characters(self):
        chars = tf.TRADITIONAL_ONLY
        self.assertEqual(len(chars), 50); self.assertEqual(len(set(chars)), 50); self.assertTrue(all("一" <= c <= "鿿" for c in chars))
        self.assertEqual(tf.SPEC["external_panel"]["zho_Hans_filter_characters"], "".join(chars))

    def test_the_external_panel_is_pending_with_its_rules_and_not_yet_frozen_with_hashes(self):
        e = tf.SPEC["external_panel"]
        self.assertEqual(e["status"], "pending"); self.assertEqual(e["languages"], ["eng_Latn", "spa_Latn", "rus_Cyrl", "hin_Deva", "zho_Hans", "yor_Latn"])
        self.assertEqual((e["calibration_passages"], e["evaluation_passages"], e["max_tokens"]), (100, 300, 128))
        self.assertEqual(e["corpus"], {"dataset": "wikimedia/wikipedia", "config_date": "20231101", "revision": "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"})


class Build(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); d = Path(self.tmp.name); self.dir = d
        self.files = {}
        for name, body in (("sample", "S"), ("eval_freeze", "E"), ("groups", "G"), ("plan", "P")):
            (d / name).write_text(body); self.files[name] = d / name
        smp = sample(); self.core = tf.derive_core(smp, stage_a(smp), per_cell=None, subset_seed=0, dev_per_language=DEV, M=M, min_ctx=2)
        self.core["source"] = {"stage_a_sha256": "ab" * 32, "sample_sha256": "cd" * 32}

    def tearDown(self):
        self.tmp.cleanup()

    def build(self):
        return tf.build_freeze(self.files, self.core, commit="abc", created="2026-10-26 10:00 IST", root=self.dir)

    def test_the_freeze_records_the_file_hashes_the_core_hashes_the_split_and_the_spec(self):
        f = self.build()
        self.assertEqual(f["files"]["sample"]["sha256"], hashlib.sha256(b"S").hexdigest()); self.assertEqual(f["kind"], "intervention-transfer-freeze")
        self.assertEqual(f["core"]["n_features"], 13); self.assertEqual(f["core"]["feature_ids_sha256"], tf.sha256_obj(self.core["feature_ids"]))
        self.assertEqual(f["core"]["contexts_sha256"], tf.sha256_obj(self.core["contexts"])); self.assertEqual(f["core"]["direction_ids_sha256"], tf.sha256_obj(self.core["direction_ids"]))
        self.assertEqual(f["core"]["n_feature_context_pairs"], 39); self.assertEqual(f["spec"], json.loads(json.dumps(tf.SPEC)))
        self.assertEqual(sorted(f["split"]["calibration"] + f["split"]["heldout"]), self.core["feature_ids"]); self.assertEqual(f["split"]["salt"], tf.SPLIT_SALT)
        self.assertEqual(f["core"]["source"], self.core["source"])

    def test_a_freeze_of_an_unchanged_state_verifies(self):
        f = self.build(); self.assertTrue(tf.verify_freeze(f, self.files, self.core))

    def test_every_change_after_the_freeze_is_reported(self):
        f = self.build()
        (self.dir / "plan").write_text("P2")
        with self.assertRaisesRegex(ValueError, "plan"):
            tf.verify_freeze(f, self.files, self.core)
        (self.dir / "plan").write_text("P")
        core = json.loads(json.dumps(self.core)); core["contexts"][0][0] += 1
        with self.assertRaisesRegex(ValueError, "contexts"):
            tf.verify_freeze(f, self.files, core)
        core = json.loads(json.dumps(self.core)); core["direction_ids"][0] += 1
        with self.assertRaisesRegex(ValueError, "direction"):
            tf.verify_freeze(f, self.files, core)
        spec_backup = json.loads(json.dumps(tf.SPEC)); tf.SPEC["bootstrap"]["replicates"] = 1000
        try:
            with self.assertRaisesRegex(ValueError, "spec"):
                tf.verify_freeze(f, self.files, self.core)
        finally:
            tf.SPEC["bootstrap"]["replicates"] = spec_backup["bootstrap"]["replicates"]

    def test_a_missing_or_extra_file_is_reported(self):
        f = self.build(); files = dict(self.files); del files["groups"]
        with self.assertRaisesRegex(ValueError, "groups"):
            tf.verify_freeze(f, files, self.core)
        extra = dict(self.files); (self.dir / "x").write_text("x"); extra["x"] = self.dir / "x"
        with self.assertRaisesRegex(ValueError, "not in the freeze"):
            tf.verify_freeze(f, extra, self.core)

    def test_the_freeze_refuses_a_core_with_a_floor_or_calibration_set_that_overlaps_the_heldout_features(self):
        f = self.build(); f["split"]["heldout"] = f["split"]["heldout"] + [f["split"]["calibration"][0]]
        with self.assertRaisesRegex(ValueError, "split"):
            tf.verify_freeze(f, self.files, self.core)


class Files(unittest.TestCase):
    def test_main_build_and_verify_roundtrip_on_synthetic_inputs(self):
        import contextlib
        import io
        import torch
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); smp = sample(); sa = stage_a(smp)
            (d / "sample.json").write_text(json.dumps(smp))
            torch.save({k: (torch.tensor(v) if isinstance(v, np.ndarray) else v) for k, v in sa.items()}, d / "stage_a.pt")
            for n in ("evalfreeze.json", "groups.json", "plan.md", "spec.md", "verification.md"):
                (d / n).write_text(n)
            argv = ["--sample", str(d / "sample.json"), "--stage-a", str(d / "stage_a.pt"), "--eval-freeze", str(d / "evalfreeze.json"), "--groups-manifest", str(d / "groups.json"),
                    "--plan", str(d / "plan.md"), "--spec", str(d / "spec.md"), "--verification", str(d / "verification.md"), "--commit", "abc123", "--dev-per-language", str(DEV), "--M", str(M), "--min-ctx", "2",
                    "--per-cell", "0", "--out-dir", str(d / "out")]
            with contextlib.redirect_stdout(io.StringIO()):
                tf.main(["build"] + argv)
            self.assertTrue((d / "out" / "freeze.json").exists()); self.assertTrue((d / "out" / "core_sample.json").exists())
            core = json.loads((d / "out" / "core_sample.json").read_text()); self.assertEqual(len(core["feature_ids"]), 13)
            with contextlib.redirect_stdout(io.StringIO()) as out:
                tf.main(["verify", "--freeze", str(d / "out" / "freeze.json")] + argv[:-2] + ["--out-dir", str(d / "out")])
            self.assertIn("verified", out.getvalue())
            sa2 = dict(sa); c2 = sa["contexts"].copy(); c2[0, 0] = (c2[0, 0] + 1) % (DEV * len(LANGS)); c2[0, 1] = (c2[0, 1] + 1) % (DEV * len(LANGS)); c2[0, 2] = (c2[0, 2] + 1) % (DEV * len(LANGS)); sa2["contexts"] = c2
            torch.save({k: (torch.tensor(v) if isinstance(v, np.ndarray) else v) for k, v in sa2.items()}, d / "stage_b.pt")
            argv_b = [x if x != str(d / "stage_a.pt") else str(d / "stage_b.pt") for x in argv[:-2]]
            with self.assertRaisesRegex(ValueError, "fresh derivation"):                                                                              # the committed core must equal a fresh derivation from the stage file
                tf.main(["verify", "--freeze", str(d / "out" / "freeze.json")] + argv_b + ["--out-dir", str(d / "out")])


if __name__ == "__main__":
    unittest.main()
