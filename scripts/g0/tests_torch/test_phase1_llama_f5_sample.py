"""Tests for scripts/g0/phase1_llama_f5_sample.py: the feature sample of the Llama-3.2-1B F5 check (Stage 5 of the survival-and-intervention plan). Roles: `k64` (the K64 group of a language: the first 64 of its ranked
selective list) and `selective_other` (the rest of its selective list). Synthetic inputs only."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import phase1_llama_f5_sample as ls  # noqa: E402
import phase1_func_f5_run as fr  # noqa: E402

LANGS = ["aaa_Latn", "bbb_Latn", "ccc_Latn"]


def fixture(n_feat=60):
    rates = torch.zeros(3, n_feat); mean_act = torch.zeros(3, n_feat); peak = torch.full((n_feat,), -1, dtype=torch.long); selective = torch.zeros(n_feat, dtype=torch.bool)
    groups = {}
    lists = {"aaa_Latn": [5, 3, 9, 1, 7, 11], "bbb_Latn": [20, 21, 22], "ccc_Latn": [40, 41, 42, 43]}
    for li, (lang, ids) in enumerate(lists.items()):
        groups[lang] = {"all": ids}
        for f in ids:
            peak[f] = li; selective[f] = True; rates[li, f] = 0.01 * (f + 1); mean_act[li, f] = 0.5 + f
    sel = {"rates": rates, "mean_activation": mean_act, "peak": peak, "selective": selective}
    man = {"languages": LANGS, "groups": groups, "Ks": [16, 64], "sizes": {l: {"K16": 16, "K64": 64} for l in LANGS}}
    return sel, man


class Build(unittest.TestCase):
    def test_roles_are_the_first_k_of_the_ranked_list_and_the_rest(self):
        sel, man = fixture(); s = ls.build_llama_sample(sel, man, LANGS, {"x": "h"}, k=2)
        self.assertEqual(s["features"]["5"]["roles"], ["k64"]); self.assertEqual(s["features"]["3"]["roles"], ["k64"])
        self.assertEqual(s["features"]["9"]["roles"], ["selective_other"]); self.assertEqual(s["features"]["11"]["roles"], ["selective_other"])
        self.assertEqual(s["features"]["22"]["roles"], ["selective_other"]); self.assertEqual(s["features"]["20"]["roles"], ["k64"])

    def test_every_listed_feature_appears_once_with_its_language_and_native_activity(self):
        sel, man = fixture(); s = ls.build_llama_sample(sel, man, LANGS, {}, k=2)
        self.assertEqual(len(s["features"]), 6 + 3 + 4); self.assertEqual(s["languages"], LANGS)
        self.assertEqual(s["features"]["42"]["language"], "ccc_Latn")
        self.assertAlmostEqual(s["features"]["42"]["native"]["FP"][0], 0.43); self.assertAlmostEqual(s["features"]["42"]["native"]["FP"][1], 42.5)
        self.assertEqual(s["counts"]["aaa_Latn"], {"k64": 2, "selective_other": 4})

    def test_a_feature_in_two_languages_or_with_another_peak_is_refused(self):
        sel, man = fixture(); man["groups"]["bbb_Latn"]["all"] = [20, 21, 5]
        with self.assertRaises(ValueError):
            ls.build_llama_sample(sel, man, LANGS, {}, k=2)
        sel, man = fixture(); sel["peak"][21] = 2
        with self.assertRaises(ValueError):
            ls.build_llama_sample(sel, man, LANGS, {}, k=2)
        sel, man = fixture(); sel["selective"][22] = False
        with self.assertRaises(ValueError):
            ls.build_llama_sample(sel, man, LANGS, {}, k=2)

    def test_a_feature_listed_twice_in_a_consistent_way_is_refused(self):
        sel, man = fixture(); man["groups"]["aaa_Latn"]["all"] = [5, 3, 5, 1]   # same language, same peak: only the duplicate check can catch it
        with self.assertRaises(ValueError):
            ls.build_llama_sample(sel, man, LANGS, {}, k=2)

    def test_the_roster_must_be_the_manifest_language_order(self):
        sel, man = fixture()
        with self.assertRaises(ValueError):
            ls.build_llama_sample(sel, man, LANGS[::-1], {}, k=2)

    def test_the_frozen_runner_keeps_at_most_per_cell_features_per_language_and_role(self):
        sel, man = fixture(); s = ls.build_llama_sample(sel, man, LANGS, {}, k=2)
        ids, lang = fr.sample_arrays(s, per_cell=1, seed=0)
        cells = {}
        for i, l in zip(ids.tolist(), lang.tolist()):
            cells.setdefault((LANGS[l], tuple(s["features"][str(i)]["roles"])), []).append(i)
        self.assertTrue(all(len(v) == 1 for v in cells.values())); self.assertEqual(len(cells), 6)   # three languages x two roles, every cell represented
        self.assertEqual(sorted(ids.tolist()), sorted(set(ids.tolist())))


if __name__ == "__main__":
    unittest.main()
