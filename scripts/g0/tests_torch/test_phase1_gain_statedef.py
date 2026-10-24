import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_func_run as fr
import phase1_gain_baseline as gb
import phase1_gain_statedef as sd

# Toy sidecar: 3 languages, 8 sentences, a universe of 40 features with ids 3i + 1 inside 130 features.
LANGS = ["l0", "l1", "l2"]
S, U, NF = 8, 40, 130
FEAT_IDS = np.array([3 * i + 1 for i in range(U)])
TOK = 2500                                           # selected tokens per sentence
CUTOFF = 1.4


def dense(lang_index, scale=1.0):
    """Counts and sums [S, U] of one language: features 6l..6l+5 fire only on that language (50 positives per sentence, mean 1.0 + 0.1 k), features 18..39 fire on every language alike."""
    c = np.zeros((S, U)); m = np.zeros((S, U))
    for k in range(6):
        f = 6 * lang_index + k
        c[:, f] = 50 * scale; m[:, f] = (1.0 + 0.1 * k) * 50 * scale
    for j, f in enumerate(range(18, 40)):
        c[:, f] = 40; m[:, f] = (0.5 + 0.02 * j) * 40
    return c, m


def shard(c, m, sentence_ids=None, feature_ids=None, n_tokens=None):
    rows, cols = np.nonzero(c)
    row_ptr = np.zeros(S + 1, dtype=np.int64)
    for r in rows:
        row_ptr[r + 1] += 1
    row_ptr = np.cumsum(row_ptr)
    return {"schema_version": 1, "condition": "X", "stream": "eq", "language": "x",
            "sentence_ids": torch.tensor(np.arange(S) if sentence_ids is None else sentence_ids), "n_tokens": torch.full((S,), TOK, dtype=torch.int64) if n_tokens is None else n_tokens,
            "feature_ids": torch.tensor(FEAT_IDS if feature_ids is None else feature_ids), "row_ptr": torch.tensor(row_ptr), "col": torch.tensor(cols.astype(np.int32)),
            "counts": torch.tensor(c[rows, cols].astype(np.int16)), "sums": torch.tensor(m[rows, cols].astype(np.float32))}


def toy_shards(**kw):
    return {l: shard(*dense(i), **kw) for i, l in enumerate(LANGS)}


EVAL = [0, 1, 2]


class Profile(unittest.TestCase):
    def test_excluded_sentences_never_enter_the_profile(self):
        p = sd.profile_from_shards(toy_shards(), EVAL, NF)
        self.assertEqual(p["n_tokens"].tolist(), [5 * TOK] * 3)
        c0, _ = dense(0)
        f = int(FEAT_IDS[2])                                  # a feature of language 0
        self.assertEqual(int(p["positive_counts"][0, f]), int(c0[3:, 2].sum()))
        self.assertEqual(int(p["positive_counts"][1, f]), 0)

    def test_the_same_profile_for_any_content_of_an_excluded_sentence(self):
        a = toy_shards()
        c, m = dense(0); c2 = c.copy(); c2[0, :] = 7; m2 = m.copy(); m2[0, :] = 99
        b = dict(a); b["l0"] = shard(c2, m2)
        pa, pb = sd.profile_from_shards(a, EVAL, NF), sd.profile_from_shards(b, EVAL, NF)
        np.testing.assert_array_equal(pa["positive_counts"], pb["positive_counts"]); np.testing.assert_allclose(pa["activation_sums"], pb["activation_sums"])

    def test_features_outside_the_universe_stay_zero_and_ids_are_mapped(self):
        p = sd.profile_from_shards(toy_shards(), EVAL, NF)
        self.assertEqual(p["positive_counts"].shape, (3, NF))
        outside = np.setdiff1d(np.arange(NF), FEAT_IDS)
        self.assertEqual(int(p["positive_counts"][:, outside].sum()), 0); self.assertEqual(float(p["activation_sums"][:, outside].sum()), 0.0)
        c1, m1 = dense(1)
        self.assertAlmostEqual(float(p["activation_sums"][1, int(FEAT_IDS[7])]), float(m1[3:, 7].sum()), places=3)

    def test_dtypes_and_shapes(self):
        p = sd.profile_from_shards(toy_shards(), EVAL, NF)
        self.assertEqual(p["positive_counts"].dtype, np.int64); self.assertEqual(p["activation_sums"].dtype, np.float64); self.assertEqual(p["n_tokens"].dtype, np.int64)

    def test_a_shard_with_other_sentence_ids_is_refused(self):
        with self.assertRaises(ValueError):
            sd.profile_from_shards(toy_shards(sentence_ids=np.arange(1, S + 1)), EVAL, NF)

    def test_shards_with_different_feature_ids_are_refused(self):
        sh = toy_shards(); sh["l1"] = shard(*dense(1), feature_ids=FEAT_IDS + 1)
        with self.assertRaises(ValueError):
            sd.profile_from_shards(sh, EVAL, NF)

    def test_an_evaluation_id_outside_the_sentences_is_refused(self):
        with self.assertRaises(ValueError):
            sd.profile_from_shards(toy_shards(), [0, 1, 99], NF)

    def test_excluding_nothing_keeps_every_sentence(self):
        p = sd.profile_from_shards(toy_shards(), [], NF)
        self.assertEqual(p["n_tokens"].tolist(), [S * TOK] * 3)


class Selectivity(unittest.TestCase):
    def setUp(self):
        self.p = sd.profile_from_shards(toy_shards(), EVAL, NF)
        self.sel = sd.state_selectivity(self.p, CUTOFF, LANGS)

    def test_the_fields_of_the_frozen_dev_view_are_there_as_tensors(self):
        for k in ("rates", "mean_activation", "peak", "selective", "supported", "languages"):
            self.assertIn(k, self.sel)
        self.assertEqual(tuple(self.sel["rates"].shape), (3, NF)); self.assertEqual(self.sel["selective"].dtype, torch.bool); self.assertEqual(self.sel["languages"], LANGS)

    def test_own_features_are_selective_and_general_ones_are_supported_but_not(self):
        sel = self.sel["selective"]; sup = self.sel["supported"]
        own = [int(FEAT_IDS[f]) for f in range(18)]; gen = [int(FEAT_IDS[f]) for f in range(18, 40)]
        self.assertTrue(bool(sel[own].all())); self.assertFalse(bool(sel[gen].any())); self.assertTrue(bool(sup[gen].all()))
        outside = np.setdiff1d(np.arange(NF), FEAT_IDS)
        self.assertFalse(bool(sup[outside].any()))

    def test_peaks_rates_and_mean_activation(self):
        f = int(FEAT_IDS[2])
        self.assertEqual(int(self.sel["peak"][f]), 0)
        self.assertAlmostEqual(float(self.sel["rates"][0, f]), 250 / (5 * TOK), places=6)
        c0, m0 = dense(0)
        self.assertAlmostEqual(float(self.sel["mean_activation"][0, f]), float(m0[3:, 2].sum()) / (5 * TOK), places=5)

    def test_it_agrees_with_the_baseline_classification(self):
        cls = gb.state_classes(self.p["positive_counts"], self.p["n_tokens"], CUTOFF)
        np.testing.assert_array_equal(self.sel["selective"].numpy(), cls["selective"])


class Manifest(unittest.TestCase):
    def setUp(self):
        self.p = sd.profile_from_shards(toy_shards(), EVAL, NF)
        self.sel = sd.state_selectivity(self.p, CUTOFF, LANGS)
        self.man = sd.build_state_manifest(self.sel, K=4, R=3, seed=0, source_sha256="abc")

    def test_the_group_is_the_strongest_own_features_by_mean_activation(self):
        for i, l in enumerate(LANGS):
            ids = self.man["groups"][l]["all"]
            expected = [int(FEAT_IDS[6 * i + k]) for k in (5, 4, 3, 2, 1, 0)]           # mean activation rises with k
            self.assertEqual(ids, expected)
            self.assertEqual(self.man["sizes"][l]["K4"], 4)

    def test_controls_are_supported_non_selective_features_of_the_asked_size_and_number(self):
        gen = {int(FEAT_IDS[f]) for f in range(18, 40)}
        for l in LANGS:
            draws = self.man["controls"][l]["K4"]
            self.assertEqual(len(draws), 3)
            for d in draws:
                self.assertEqual(len(d), 4); self.assertTrue(set(d) <= gen); self.assertEqual(len(set(d)), 4)

    def test_it_is_deterministic_and_the_seed_matters(self):
        again = sd.build_state_manifest(self.sel, K=4, R=3, seed=0, source_sha256="abc")
        self.assertEqual(json.dumps(again, sort_keys=True), json.dumps(self.man, sort_keys=True))
        other = sd.build_state_manifest(self.sel, K=4, R=3, seed=5, source_sha256="abc")
        self.assertNotEqual(other["controls"], self.man["controls"])

    def test_the_manifest_has_the_frozen_format_and_records_its_source(self):
        for k in ("schema", "source", "languages", "Ks", "Rs", "seed", "groups", "controls", "sizes"):
            self.assertIn(k, self.man)
        self.assertEqual(self.man["source"]["sha256"], "abc"); self.assertEqual(self.man["languages"], LANGS); self.assertEqual(self.man["Ks"], [4]); self.assertEqual(self.man["Rs"], [3])
        json.dumps(self.man)                                                     # serialisable

    def test_the_frozen_runner_builds_its_configurations_from_it(self):
        cfgs = fr.build_configs(self.man, LANGS, LANGS, [], {"zero": ["K4"], "controls": ["K4"], "n_controls": {"K4": 3}}, "K4")
        self.assertEqual(len(cfgs), 3 * (1 + 3))
        z = [c for c in cfgs if c["name"] == "zero_K4@l1"][0]
        self.assertEqual(z["idx"].tolist(), self.man["groups"]["l1"]["all"][:4])

    def test_a_language_with_fewer_own_features_than_asked_is_refused(self):
        with self.assertRaises(ValueError) as cm:
            sd.build_state_manifest(self.sel, K=7, R=3, seed=0, source_sha256="abc")
        self.assertIn("l0", str(cm.exception))

    def test_the_controls_are_those_of_the_frozen_matching_with_the_same_seed(self):
        import phase1_func_groups as fg
        for i, l in enumerate(LANGS):
            grp = fg.top_k(fg.group_members(self.sel, i), 4)
            self.assertEqual(self.man["controls"][l]["K4"], fg.draw_controls(self.sel, i, grp, 3, 0)["sets"])

    def test_a_group_as_large_as_the_whole_own_set_is_allowed(self):
        man = sd.build_state_manifest(self.sel, K=6, R=3, seed=0, source_sha256="abc")
        self.assertEqual(man["sizes"]["l0"]["K6"], 6)

    def test_ties_in_mean_activation_are_broken_by_feature_id(self):
        sel = {k: (v.clone() if hasattr(v, "clone") else v) for k, v in self.sel.items()}
        ids = [int(FEAT_IDS[k]) for k in range(6)]
        sel["mean_activation"][0, ids] = 1.0
        man = sd.build_state_manifest(sel, K=4, R=3, seed=0, source_sha256="abc")
        self.assertEqual(man["groups"]["l0"]["all"], sorted(ids))


class Overlap(unittest.TestCase):
    def test_jaccard_of_the_groups_between_states(self):
        a = {"languages": ["x", "y"], "groups": {"x": {"all": [1, 2, 3, 4]}, "y": {"all": [5, 6, 7, 8]}}, "sizes": {"x": {"K4": 4}, "y": {"K4": 4}}}
        b = {"languages": ["x", "y"], "groups": {"x": {"all": [3, 4, 5, 6]}, "y": {"all": [5, 6, 7, 8]}}, "sizes": {"x": {"K4": 4}, "y": {"K4": 4}}}
        rows = sd.overlap_table({"FP": a, "INT4": b}, "K4", ref="FP")
        r = {(x["language"], x["state"]): x for x in rows}
        self.assertAlmostEqual(r[("x", "INT4")]["jaccard"], 2 / 6); self.assertAlmostEqual(r[("y", "INT4")]["jaccard"], 1.0)
        self.assertEqual(r[("x", "INT4")]["shared"], 2); self.assertNotIn(("x", "FP"), r)

    def test_the_group_size_label_decides_how_many_members_count(self):
        a = {"languages": ["x"], "groups": {"x": {"all": [1, 2, 3, 4, 5]}}, "sizes": {"x": {"K2": 2}}}
        b = {"languages": ["x"], "groups": {"x": {"all": [2, 1, 9, 9, 9]}}, "sizes": {"x": {"K2": 2}}}
        self.assertAlmostEqual(sd.overlap_table({"FP": a, "INT4": b}, "K2", ref="FP")[0]["jaccard"], 1.0)

    def test_languages_must_agree(self):
        a = {"languages": ["x"], "groups": {"x": {"all": [1]}}, "sizes": {"x": {"K1": 1}}}
        b = {"languages": ["y"], "groups": {"y": {"all": [1]}}, "sizes": {"y": {"K1": 1}}}
        with self.assertRaises(ValueError):
            sd.overlap_table({"FP": a, "INT4": b}, "K1", ref="FP")


class Files(unittest.TestCase):
    def test_shards_are_read_from_the_sidecar_layout(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for l, sh in toy_shards().items():
                (d / "eq" / "sentences" / "INT4").mkdir(parents=True, exist_ok=True)
                torch.save(sh, d / "eq" / "sentences" / "INT4" / f"{l}.pt")
            shards, shas = sd.load_shards(d, "eq", "INT4", LANGS)
            self.assertEqual(list(shards), LANGS); self.assertEqual(set(shas), set(LANGS)); self.assertEqual(len(shas["l0"]), 64)
            p = sd.profile_from_shards(shards, EVAL, NF)
            self.assertEqual(p["n_tokens"].tolist(), [5 * TOK] * 3)

    def test_a_missing_shard_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                sd.load_shards(Path(d), "eq", "INT4", LANGS)


if __name__ == "__main__":
    unittest.main()
