import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phase1_func_groups as fg


def make_cohort(F=300, n_lang=4, sizes=(40, 12, 25, 9), n_pool=120, seed=0):
    """Synthetic frozen-cohort fields (dev only): selective features with a peak language, a pool of supported non-selective features, the rest unsupported."""
    F = max(F, sum(sizes) + n_pool + 20)
    g = torch.Generator().manual_seed(seed)
    rates = torch.rand(n_lang, F, generator=g) * 0.02
    mean_act = rates * (0.5 + 4 * torch.rand(n_lang, F, generator=g))
    peak = torch.zeros(F, dtype=torch.long)
    selective = torch.zeros(F, dtype=torch.bool)
    supported = torch.zeros(F, dtype=torch.bool)
    pos = 0
    for l, n in enumerate(sizes):
        ids = torch.arange(pos, pos + n); pos += n
        selective[ids] = True; supported[ids] = True; peak[ids] = l
        rates[l, ids] += 0.03; mean_act[l, ids] = rates[l, ids] * (1 + 3 * torch.rand(n, generator=g))
    pool_ids = torch.arange(pos, pos + n_pool)
    supported[pool_ids] = True
    rates[:, pool_ids] = torch.rand(n_lang, n_pool, generator=g) * 0.09                 # a pool whose activity overlaps the groups' range
    mean_act[:, pool_ids] = rates[:, pool_ids] * (0.5 + 4 * torch.rand(n_lang, n_pool, generator=g))
    return {"rates": rates, "mean_activation": mean_act, "peak": peak, "selective": selective, "supported": supported,
            "languages": [f"l{i}" for i in range(n_lang)]}


class Groups(unittest.TestCase):
    def setUp(self):
        self.sel = make_cohort()

    def test_members_are_the_selective_features_of_that_peak_sorted_by_dev_mass(self):
        for l in range(4):
            m = fg.group_members(self.sel, l)
            expect = ((self.sel["peak"] == l) & self.sel["selective"]).nonzero().squeeze(1)
            self.assertEqual(sorted(m.tolist()), sorted(expect.tolist()))
            mass = self.sel["mean_activation"][l, m]
            self.assertTrue(bool((mass[:-1] >= mass[1:]).all()))

    def test_ties_are_broken_by_feature_id(self):
        sel = copy.deepcopy(self.sel)
        ids = fg.group_members(sel, 1)
        sel["mean_activation"][1, ids] = 1.0
        m = fg.group_members(sel, 1)
        self.assertEqual(m.tolist(), sorted(ids.tolist()))

    def test_top_k_is_nested_and_clipped(self):
        m = fg.group_members(self.sel, 0)
        k16, k32 = fg.top_k(m, 16), fg.top_k(m, 32)
        self.assertEqual((k16.numel(), k32.numel()), (16, 32))
        self.assertEqual(k32[:16].tolist(), k16.tolist())
        self.assertEqual(fg.top_k(m, None).tolist(), m.tolist())
        self.assertEqual(fg.top_k(fg.group_members(self.sel, 3), 64).numel(), 9)       # fewer members than K: the whole group

    def test_dev_view_keeps_only_dev_fields_and_demands_all_of_them(self):
        sel = make_cohort(); sel["devtest_rates"] = torch.zeros(1)
        self.assertEqual(set(fg.dev_view(sel)), set(fg.DEV_FIELDS))
        del sel["peak"]
        with self.assertRaises(ValueError):
            fg.dev_view(sel)

    def test_pool_is_supported_and_not_selective(self):
        pool = fg.pool_ids(self.sel)
        self.assertEqual(pool.numel(), 120)
        self.assertFalse(bool(self.sel["selective"][pool].any()))
        self.assertTrue(bool(self.sel["supported"][pool].all()))


class Controls(unittest.TestCase):
    def setUp(self):
        self.sel = make_cohort()

    def test_controls_are_size_matched_disjoint_from_selective_and_from_the_pool_only(self):
        pool = set(fg.pool_ids(self.sel).tolist())
        for l in range(4):
            grp = fg.top_k(fg.group_members(self.sel, l), 12)
            draws = fg.draw_controls(self.sel, l, grp, R=4, seed=3)
            self.assertEqual(len(draws["sets"]), 4)
            for s in draws["sets"]:
                self.assertEqual(len(s), len(grp))
                self.assertEqual(len(set(s)), len(s))                               # without replacement within a draw
                self.assertTrue(set(s) <= pool)
                self.assertFalse(set(s) & set(grp.tolist()))

    def test_nearest_available_pool_feature_is_chosen_when_m_is_one(self):
        # one language, group of two, pool of four with hand-set statistics (distance = |dlog mu| + |dlog rate|)
        F = 8
        rates = torch.full((1, F), 1e-3); mean_act = torch.full((1, F), 1e-3)
        peak = torch.zeros(F, dtype=torch.long); selective = torch.zeros(F, dtype=torch.bool); supported = torch.zeros(F, dtype=torch.bool)
        selective[[0, 1]] = True; supported[:6] = True
        rates[0, [0, 1]] = torch.tensor([0.10, 0.01]); mean_act[0, [0, 1]] = torch.tensor([1.0, 0.1])
        rates[0, 2:6] = torch.tensor([0.011, 0.098, 0.5, 0.0001]); mean_act[0, 2:6] = torch.tensor([0.11, 1.05, 3.0, 0.001])
        sel = {"rates": rates, "mean_activation": mean_act, "peak": peak, "selective": selective, "supported": supported, "languages": ["l0"]}
        grp = fg.group_members(sel, 0)                                               # [0, 1] (feature 0 has the larger mass)
        self.assertEqual(grp.tolist(), [0, 1])
        d = fg.draw_controls(sel, 0, grp, R=1, seed=0, M=1)
        self.assertEqual(d["sets"][0], [2, 3])                                       # feature 0 -> 3, feature 1 -> 2 (sorted ids)

    def test_greedy_order_gives_the_most_active_member_the_first_choice(self):
        F = 6
        rates = torch.full((1, F), 0.01); mean_act = torch.full((1, F), 0.1)
        peak = torch.zeros(F, dtype=torch.long); selective = torch.zeros(F, dtype=torch.bool); supported = torch.zeros(F, dtype=torch.bool)
        selective[[0, 1]] = True; supported[:5] = True
        mean_act[0, 0] = 0.7; mean_act[0, 1] = 1.0           # both members prefer the pool feature with mass 0.85; the more active member has the LARGER id
        mean_act[0, 2] = 0.85; mean_act[0, 3] = 0.3; mean_act[0, 4] = 2.5
        sel = {"rates": rates, "mean_activation": mean_act, "peak": peak, "selective": selective, "supported": supported, "languages": ["l0"]}
        d = fg.draw_controls(sel, 0, fg.group_members(sel, 0), R=1, seed=0, M=1)
        # descending mass: member 1 (1.0) takes feature 2, member 0 (0.7) then prefers feature 3 (0.3) to feature 4 (2.5); matching in id order would give {2, 4}
        self.assertEqual(d["sets"][0], [2, 3])
        d2 = fg.draw_controls(sel, 0, torch.tensor([0, 1]), R=1, seed=0, M=1)               # the same set of members in another order gives the same result
        self.assertEqual(d2["sets"][0], [2, 3])

    def test_draws_differ_and_are_reproducible(self):
        grp = fg.top_k(fg.group_members(self.sel, 0), 15)
        a = fg.draw_controls(self.sel, 0, grp, R=5, seed=1)
        b = fg.draw_controls(self.sel, 0, grp, R=5, seed=1)
        c = fg.draw_controls(self.sel, 0, grp, R=5, seed=2)
        self.assertEqual(a["sets"], b["sets"])
        self.assertNotEqual(a["sets"], c["sets"])
        self.assertEqual(len({tuple(s) for s in a["sets"]}), 5)

    def test_controls_use_the_languages_own_statistics_only(self):
        grp = fg.top_k(fg.group_members(self.sel, 1), 8)
        a = fg.draw_controls(self.sel, 1, grp, R=3, seed=1)
        sel2 = copy.deepcopy(self.sel)
        sel2["rates"][[0, 2, 3]] = torch.rand_like(sel2["rates"][[0, 2, 3]]); sel2["mean_activation"][[0, 2, 3]] = torch.rand_like(sel2["mean_activation"][[0, 2, 3]])
        self.assertEqual(a["sets"], fg.draw_controls(sel2, 1, grp, R=3, seed=1)["sets"])
        sel3 = copy.deepcopy(self.sel)
        sel3["mean_activation"][1] = sel3["mean_activation"][1].flip(0)
        self.assertNotEqual(a["sets"], fg.draw_controls(sel3, 1, grp, R=3, seed=1)["sets"])

    def test_matching_quality_on_a_pool_with_enough_candidates(self):
        sel = make_cohort(n_pool=900, seed=2)
        grp = fg.top_k(fg.group_members(sel, 0), 20)
        d = fg.draw_controls(sel, 0, grp, R=4, seed=0)
        g_mean = float(sel["mean_activation"][0, grp].mean())
        for s in d["sets"]:
            self.assertTrue(0.8 < g_mean / float(sel["mean_activation"][0, torch.tensor(s)].mean()) < 1.25)
        self.assertEqual(len(d["gap"]), 4)
        self.assertTrue(all(x >= 0 for x in d["gap"]))

    def test_pool_smaller_than_the_group_is_an_error(self):
        sel = make_cohort(n_pool=10)
        with self.assertRaisesRegex(ValueError, "control pool"):
            fg.draw_controls(sel, 0, fg.group_members(sel, 0), R=1, seed=0)

    def test_pool_exactly_as_large_as_the_group_is_used_up(self):
        sel = make_cohort(sizes=(9, 12, 25, 9), n_pool=9)
        d = fg.draw_controls(sel, 0, fg.group_members(sel, 0), R=2, seed=0)
        for s in d["sets"]:
            self.assertEqual(s, sorted(fg.pool_ids(sel).tolist()))

    def test_the_rate_term_matters_when_the_activations_tie(self):
        F = 5
        rates = torch.full((1, F), 0.01); mean_act = torch.full((1, F), 0.5)
        peak = torch.zeros(F, dtype=torch.long); selective = torch.zeros(F, dtype=torch.bool); supported = torch.zeros(F, dtype=torch.bool)
        selective[0] = True; supported[:3] = True
        rates[0, 0] = 0.02; rates[0, 1] = 0.2; rates[0, 2] = 0.021                       # equal mean activation: only the rate separates the pool features
        sel = {"rates": rates, "mean_activation": mean_act, "peak": peak, "selective": selective, "supported": supported, "languages": ["l0"]}
        self.assertEqual(fg.draw_controls(sel, 0, torch.tensor([0]), R=1, seed=0, M=1)["sets"][0], [2])

    def test_invalid_m_is_an_error(self):
        with self.assertRaises(ValueError):
            fg.draw_controls(self.sel, 0, fg.top_k(fg.group_members(self.sel, 0), 5), R=1, seed=0, M=0)


class Manifest(unittest.TestCase):
    def write_source(self, sel):
        d = Path(tempfile.mkdtemp()); p = d / "selectivity.pt"
        torch.save(sel, p)
        return p

    def test_round_trip_and_tamper_detection(self):
        sel = make_cohort()
        p = self.write_source(sel)
        m = fg.build_manifest(p, Ks=(8, None), Rs=(2, 2), seed=0)
        fg.verify_manifest(m, p)
        self.assertEqual(m["languages"], sel["languages"])
        for l in sel["languages"]:
            self.assertEqual(len(m["groups"][l]["all"]), int(((sel["peak"] == sel["languages"].index(l)) & sel["selective"]).sum()))
            self.assertEqual(len(m["controls"][l]["K8"]), 2)
        bad = json.loads(json.dumps(m))
        bad["controls"]["l0"]["K8"][0][0] = (bad["controls"]["l0"]["K8"][0][0] + 1) % 300
        with self.assertRaisesRegex(ValueError, "fresh rebuild"):
            fg.verify_manifest(bad, p)
        sel2 = copy.deepcopy(sel); sel2["rates"][0, 0] += 1.0
        with self.assertRaisesRegex(ValueError, "selectivity file differs"):
            fg.verify_manifest(m, self.write_source(sel2))

    def test_manifest_depends_on_dev_fields_only(self):
        sel = make_cohort()
        sel["devtest_rates"] = torch.rand(4, 300)
        p1 = self.write_source(sel)
        m1 = fg.build_manifest(p1, Ks=(8,), Rs=(2,), seed=0)
        sel2 = copy.deepcopy(sel); sel2["devtest_rates"] = torch.rand(4, 300)
        p2 = self.write_source(sel2)
        m2 = fg.build_manifest(p2, Ks=(8,), Rs=(2,), seed=0)
        self.assertEqual(m1["groups"], m2["groups"])
        self.assertEqual(m1["controls"], m2["controls"])

    def test_manifest_is_json_serialisable_and_records_matching_gaps(self):
        sel = make_cohort(n_pool=30)
        m = fg.build_manifest(self.write_source(sel), Ks=(20,), Rs=(2,), seed=0)
        json.dumps(m)
        self.assertIn("gap", m["controls"]["l0"])


class MatchingTable(unittest.TestCase):
    def test_rows_report_group_and_control_means_per_language_and_size(self):
        sel = make_cohort(n_pool=400, seed=1)
        d = Path(tempfile.mkdtemp()); p = d / "selectivity.pt"; torch.save(sel, p)
        m = fg.build_manifest(p, Ks=(8, None), Rs=(2, 2), seed=0)
        rows = fg.matching_table(dev_view_of(sel), m)
        self.assertEqual(len(rows), 4 * 2)
        r0 = [r for r in rows if r["language"] == "l0" and r["size"] == "K8"][0]
        grp = fg.top_k(fg.group_members(sel, 0), 8)
        self.assertEqual(r0["n"], 8)
        self.assertAlmostEqual(r0["group_mean_activation"], float(sel["mean_activation"][0, grp].mean()), places=9)
        ctrl = torch.tensor(m["controls"]["l0"]["K8"], dtype=torch.long)
        self.assertAlmostEqual(r0["control_mean_activation"], float(sel["mean_activation"][0][ctrl].mean()), places=9)
        self.assertAlmostEqual(r0["mean_log_mu_gap"], sum(m["controls"]["l0"]["gap"]["K8"]) / 2, places=9)
        self.assertAlmostEqual(r0["activation_ratio"], r0["group_mean_activation"] / r0["control_mean_activation"], places=9)


def dev_view_of(sel):
    return fg.dev_view(sel)


if __name__ == "__main__":
    unittest.main()
