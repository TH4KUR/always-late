import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phase1_func_f5_analysis as an

LANGS = ["l0", "l1", "l2", "l3"]


def make_world(seed=0, per_lang=30, cos_by_state=None, status_by_cond=None, n_excluded_per_lang=2):
    """A synthetic F5 run: per language `per_lang` sample features, the last `n_excluded_per_lang` of each language untested (rare); stage-A arrays, states with the planted similarity (the diagonal of S) and read-side arrays,
    and the sample JSON (roles, statuses, native rates). cos_by_state: {label: function(rng, n) -> array of W1} (default: nothing drifts)."""
    rng = np.random.default_rng(seed); n_f = per_lang * len(LANGS)
    feat_ids = np.arange(100, 100 + 3 * n_f, 3)[:n_f]; feat_lang = np.repeat(np.arange(len(LANGS)), per_lang)
    excluded = np.zeros(n_f, dtype=bool)
    for l in range(len(LANGS)):
        excluded[(l + 1) * per_lang - n_excluded_per_lang:(l + 1) * per_lang] = True
    write_rows = np.flatnonzero(~excluded); n_w = len(write_rows)
    stage_a = {"feat_ids": feat_ids, "feat_lang": feat_lang, "write_rows": write_rows.tolist(), "write_features": [int(feat_ids[k]) for k in write_rows], "excluded": excluded,
               "sham_gate": [{"feature": int(feat_ids[write_rows[0]]), "exact": True}], "dev_lang": np.repeat(np.arange(len(LANGS)), 10), "contexts": np.zeros((n_f, 4), dtype=int),
               "n_active": np.full(n_f, 9), "n_ctx": np.full(n_f, 4)}
    cos_by_state = cos_by_state or {}
    states = {}
    for label in ("FP", "FP_rebatched", "INT16", "INT8", "INT6", "INT5", "INT4"):
        cos = np.ones(n_w) if label == "FP" else cos_by_state.get(label, lambda r, n: np.full(n, 0.995))(rng, n_w)
        S = rng.uniform(-0.3, 0.3, (n_w, n_w)); S[np.arange(n_w), np.arange(n_w)] = cos
        states[label] = {"read": {"n": 1000, "corr": np.full(n_f, 0.99), "jaccard": np.full(n_f, 0.95), "scale": np.full(n_f, 1.05),
                                  "n_active_fp": np.full(n_f, 50), "n_active_q": np.full(n_f, 50), "n_both": np.full(n_f, 48)},
                        "write": {"S": S.astype(np.float16), "cos": cos, "zero_fp": np.zeros(n_w, dtype=bool), "zero_q": np.zeros(n_w, dtype=bool), "topk_overlap": np.clip(cos, 0, 1) * 0.5, "magnitude_ratio": np.full(n_w, 0.8),
                                  "n_pos": np.full(n_w, 40), "n_active_q": np.full(n_w, 40), "dnll": np.full(n_w, 0.01),
                                  "retrieval": {"top1": 0.9, "mrr": 0.95, "n_queries": n_w, "n_undefined": 0, "n_ties": 0, "ranks": np.ones(n_w), "within": {"top1": 0.95, "mrr": 0.97, "n_queries": n_w, "n_undefined": 0, "n_singletons": 0, "n_ties": 0, "chance": 0.1}}}}
    status_by_cond = status_by_cond or {}
    features = {}
    for k in range(n_f):
        fid = int(feat_ids[k]); lang = LANGS[int(feat_lang[k])]; rate = float(0.001 * (1 + (k % 7)))
        features[str(fid)] = {"language": lang, "roles": ["k64"], "fp_established": True, "status": {c: status_by_cond.get(c, lambda kk: "retained")(k) for c in ("FP", "INT8", "INT6", "INT5", "INT4")},
                              "native": {c: [rate * (1 + 0.1 * j), 10 * rate] for j, c in enumerate(("FP", "INT8", "INT6", "INT5", "INT4"))}}
    sample = {"languages": LANGS, "n_features": n_f, "features": features, "kind": "f5-sample", "schema": 1}
    return stage_a, states, sample


class Assemble(unittest.TestCase):
    def test_the_table_is_aligned_to_the_tested_rows_with_their_language_roles_status_and_rate(self):
        sa, st, sm = make_world(); T = an.assemble(sa, sm)
        self.assertEqual(T["fid"].tolist(), sa["write_features"]); self.assertEqual(len(T["fid"]), 4 * 28)
        self.assertEqual(T["lang"][0], "l0"); self.assertEqual(T["lang"][-1], "l3"); self.assertTrue(all("k64" in r for r in T["roles"]))
        self.assertEqual(T["status"]["INT6"][0], "retained"); self.assertAlmostEqual(float(T["rate_fp"][0]), 0.001)
        self.assertEqual(T["n_sample"], 4 * 30); self.assertEqual(T["n_excluded"], 4 * 2)

    def test_a_language_that_disagrees_between_the_run_and_the_sample_is_an_error(self):
        sa, st, sm = make_world(); sm["features"][str(int(sa["feat_ids"][0]))]["language"] = "l3"
        with self.assertRaises(ValueError):
            an.assemble(sa, sm)

    def test_write_features_that_are_not_the_features_of_the_write_rows_are_an_error(self):
        sa, st, sm = make_world(); wf = list(sa["write_features"]); wf[0], wf[1] = wf[1], wf[0]; sa["write_features"] = wf
        with self.assertRaises(ValueError):
            an.assemble(sa, sm)

    def test_a_tested_feature_that_is_not_in_the_sample_is_an_error(self):
        sa, st, sm = make_world(); del sm["features"][str(int(sa["write_features"][3]))]
        with self.assertRaises(ValueError):
            an.assemble(sa, sm)


class Summaries(unittest.TestCase):
    def test_the_summary_counts_undefined_values_and_uses_only_defined_ones(self):
        v = np.array([0.1, 0.2, np.nan, 0.4, 0.9, np.nan]); s = an.summarize(v)
        self.assertEqual((s["n_total"], s["n_defined"]), (6, 4)); self.assertAlmostEqual(s["median"], 0.3); self.assertAlmostEqual(s["q25"], float(np.percentile([0.1, 0.2, 0.4, 0.9], 25)))
        self.assertAlmostEqual(s["q75"], float(np.percentile([0.1, 0.2, 0.4, 0.9], 75))); self.assertAlmostEqual(s["mean"], 0.4)

    def test_an_empty_or_all_undefined_sample_gives_nan_with_counts(self):
        s = an.summarize(np.array([np.nan, np.nan])); self.assertEqual((s["n_total"], s["n_defined"]), (2, 0)); self.assertTrue(np.isnan(s["median"]))
        self.assertEqual(an.summarize(np.array([]))["n_total"], 0)

    def test_the_cluster_interval_of_the_median_covers_the_point_and_resamples_whole_languages(self):
        rng = np.random.default_rng(3); v = rng.normal(0.9, 0.05, 200); cl = np.repeat(np.arange(8), 25); s = an.summarize(v, clusters=cl, n_boot=300, seed=1)
        self.assertLessEqual(s["median_lo"], s["median"]); self.assertGreaterEqual(s["median_hi"], s["median"]); self.assertLess(s["median_hi"] - s["median_lo"], 0.1)
        # a world in which the languages differ strongly has a wider interval than the same values with rows shuffled over the languages
        v2 = np.repeat(np.linspace(0.2, 0.95, 8), 25); wide = an.summarize(v2, clusters=np.repeat(np.arange(8), 25), n_boot=300, seed=1)
        shuf = an.summarize(rng.permutation(v2), clusters=np.repeat(np.arange(8), 25), n_boot=300, seed=1)
        self.assertGreater(wide["median_hi"] - wide["median_lo"], 2 * (shuf["median_hi"] - shuf["median_lo"]))


class Floor(unittest.TestCase):
    def test_the_floor_is_the_lower_of_the_two_noise_only_fifth_percentiles_and_both_are_reported(self):
        sa, st, sm = make_world(cos_by_state={"FP_rebatched": lambda r, n: np.linspace(0.95, 1.0, n), "INT16": lambda r, n: np.linspace(0.90, 1.0, n)})
        f = an.floor_report(st); n = len(st["FP"]["write"]["cos"])
        self.assertAlmostEqual(f["primary"]["tau"], float(np.percentile(np.linspace(0.95, 1.0, n), 5))); self.assertAlmostEqual(f["secondary"]["tau"], float(np.percentile(np.linspace(0.90, 1.0, n), 5)))
        self.assertAlmostEqual(f["tau"], f["secondary"]["tau"]); self.assertEqual(f["source"], "INT16"); self.assertTrue(f["adequate"])

    def test_the_floor_is_the_repeat_measurement_when_it_is_the_lower_one(self):
        sa, st, sm = make_world(cos_by_state={"FP_rebatched": lambda r, n: np.linspace(0.80, 1.0, n), "INT16": lambda r, n: np.linspace(0.95, 1.0, n)})
        f = an.floor_report(st); self.assertEqual(f["source"], "FP_rebatched"); self.assertAlmostEqual(f["tau"], f["primary"]["tau"])

    def test_int8_is_reported_but_does_not_enter_the_floor(self):
        sa, st, sm = make_world(cos_by_state={"FP_rebatched": lambda r, n: np.linspace(0.95, 1.0, n), "INT16": lambda r, n: np.linspace(0.93, 1.0, n), "INT8": lambda r, n: np.linspace(0.10, 0.2, n)})
        f = an.floor_report(st); n = len(st["FP"]["write"]["cos"]); self.assertEqual(f["source"], "INT16"); self.assertAlmostEqual(f["tau"], float(np.percentile(np.linspace(0.93, 1.0, n), 5)))
        self.assertAlmostEqual(f["int8"]["tau"], float(np.percentile(np.linspace(0.10, 0.2, n), 5))); self.assertAlmostEqual(f["int8"]["median"], 0.15)

    def test_an_instrument_is_inadequate_when_either_noise_only_state_is_not_similar_enough(self):
        for bad in ("FP_rebatched", "INT16"):
            sa, st, sm = make_world(cos_by_state={bad: lambda r, n: np.full(n, 0.5)}); f = an.floor_report(st)
            self.assertFalse(f["adequate"], bad); self.assertAlmostEqual(f[{"FP_rebatched": "primary", "INT16": "secondary"}[bad]]["median"], 0.5)
        for v, ok in ((0.7, True), (0.6999, False)):
            sa, st, sm = make_world(cos_by_state={"FP_rebatched": lambda r, n: np.full(n, v)}); self.assertEqual(an.floor_report(st)["adequate"], ok)
            sa, st, sm = make_world(cos_by_state={"INT16": lambda r, n: np.full(n, v)}); self.assertEqual(an.floor_report(st)["adequate"], ok)

    def test_with_only_one_noise_only_state_the_other_is_none_and_adequacy_follows_the_one_present(self):
        sa, st, sm = make_world(); del st["FP_rebatched"]; f = an.floor_report(st)
        self.assertIsNone(f["primary"]); self.assertEqual(f["source"], "INT16"); self.assertTrue(f["adequate"])
        sa, st, sm = make_world(); del st["INT16"]; f = an.floor_report(st); self.assertIsNone(f["secondary"]); self.assertEqual(f["source"], "FP_rebatched")

    def test_without_any_noise_only_state_int8_stands_in_and_adequacy_is_unknown(self):
        sa, st, sm = make_world(); del st["FP_rebatched"]; del st["INT16"]; f = an.floor_report(st)
        self.assertEqual(f["source"], "INT8"); self.assertIsNone(f["adequate"]); self.assertIsNone(f["primary"]); self.assertIsNone(f["secondary"])

    def test_undefined_similarities_are_left_out_of_the_floor_and_counted(self):
        sa, st, sm = make_world(); c = st["FP_rebatched"]["write"]["cos"]; c[:5] = np.nan; f = an.floor_report(st)
        self.assertEqual(f["primary"]["n_undefined"], 5); self.assertFalse(np.isnan(f["tau"]))

    def test_no_reference_state_at_all_is_an_error(self):
        sa, st, sm = make_world(); del st["FP_rebatched"]; del st["INT16"]; del st["INT8"]
        with self.assertRaises(ValueError):
            an.floor_report(st)


class NegativeControl(unittest.TestCase):
    def test_the_control_feature_is_the_same_language_feature_with_the_nearest_native_rate(self):
        rate = np.array([1e-3, 2e-3, 1.1e-3, 5e-3, 5e-3, 3e-3]); lang = np.array([0, 0, 0, 1, 1, 2])
        S = np.arange(36, dtype=np.float64).reshape(6, 6) / 100.0
        nc = an.negative_control(S.astype(np.float16), lang, rate)
        self.assertEqual(nc["match"].tolist(), [2, 2, 0, 4, 3, -1])                                                  # row 3 and 4 tie with themselves excluded; row 5 is alone in its language
        want = [S[0, 2], S[1, 2], S[2, 0], S[3, 4], S[4, 3]]
        np.testing.assert_allclose(nc["values"][:5], np.array(want).astype(np.float16).astype(np.float64)); self.assertTrue(np.isnan(nc["values"][5]))

    def test_the_nearest_rate_is_nearest_on_the_log_scale(self):
        rate = np.array([10.0, 4.0, 20.0]); nc = an.negative_control(np.zeros((3, 3)), np.zeros(3, dtype=int), rate)
        self.assertEqual(nc["match"][0], 2)                                                                          # 20 is a factor 2 from 10, 4 a factor 2.5 (linear distances would say 4)

    def test_ties_in_the_rate_go_to_the_smaller_index_and_a_feature_never_controls_itself(self):
        rate = np.array([1e-3, 1e-3, 1e-3, 1e-3]); lang = np.zeros(4, dtype=int); nc = an.negative_control(np.zeros((4, 4)), lang, rate)
        self.assertEqual(nc["match"].tolist(), [1, 0, 0, 0])

    def test_a_feature_without_a_positive_native_rate_has_no_control(self):
        rate = np.array([0.0, 1e-3, 2e-3]); nc = an.negative_control(np.zeros((3, 3)), np.zeros(3, dtype=int), rate)
        self.assertEqual(nc["match"][0], -1); self.assertTrue(np.isnan(nc["values"][0])); self.assertEqual(nc["match"][1], 2); self.assertEqual(nc["match"][2], 1)


class Groups(unittest.TestCase):
    def test_role_groups_are_all_k64_and_the_lost_and_retained_subsets_of_the_precision(self):
        roles = [frozenset({"k64"}), frozenset({"k64", "lost@INT6"}), frozenset({"lost@INT6"}), frozenset({"retained@INT6"}), frozenset({"retained@INT4"})]
        g = an.group_masks(roles, "INT6")
        self.assertEqual(sorted(g), ["all", "k64", "lost@INT6", "retained@INT6"])
        self.assertEqual(g["all"].tolist(), [True] * 5); self.assertEqual(g["k64"].tolist(), [True, True, False, False, False])
        self.assertEqual(g["lost@INT6"].tolist(), [False, True, True, False, False]); self.assertEqual(g["retained@INT6"].tolist(), [False, False, False, True, False])

    def test_a_subset_that_does_not_exist_for_the_precision_is_not_listed(self):
        g = an.group_masks([frozenset({"k64"})], "INT8"); self.assertEqual(sorted(g), ["all", "k64"])


class StateTable(unittest.TestCase):
    def setUp(self):
        self.sa, self.st, self.sm = make_world(cos_by_state={"INT6": lambda r, n: np.where(np.arange(n) % 2 == 0, 0.97, 0.6)})
        for fid, f in self.sm["features"].items():
            f["roles"] = ["k64"]
        self.T = an.assemble(self.sa, self.sm); self.floor = {"tau": 0.9}

    def tab(self, label="INT6", **kw):
        w = self.st[label]["write"]; nc = an.negative_control(w["S"], self.T["lang_idx"], self.T["rate_fp"])
        return an.state_table(label, self.st, self.T, self.sa, tau=self.floor["tau"], nc=nc, n_boot=50, seed=0, **kw)

    def test_the_read_side_uses_every_sample_feature_and_the_write_side_only_the_tested_ones(self):
        self.st["INT6"]["read"]["corr"][self.sa["excluded"]] = 0.1
        r = self.tab()["all"]; self.assertEqual(r["n_features"], 120); self.assertEqual(r["n_tested"], 112)
        self.assertEqual(r["R1"]["n_total"], 120); self.assertEqual(r["W1"]["n_total"], 112)
        self.assertAlmostEqual(r["R1"]["median"], 0.99); self.assertAlmostEqual(r["R1"]["mean"], (112 * 0.99 + 8 * 0.1) / 120)

    def test_the_write_side_summaries_and_the_share_above_the_floor_and_the_control(self):
        r = self.tab()["all"]; self.assertAlmostEqual(r["W1"]["median"], float(np.median(np.where(np.arange(112) % 2 == 0, 0.97, 0.6))))
        self.assertAlmostEqual(r["W1_above_floor"]["share"], 0.5); self.assertEqual(r["W1_above_floor"]["n_defined"], 112); self.assertLessEqual(r["W1_above_floor"]["lo"], 0.5); self.assertGreaterEqual(r["W1_above_floor"]["hi"], 0.5)
        nc = an.negative_control(self.st["INT6"]["write"]["S"], self.T["lang_idx"], self.T["rate_fp"]); p95 = float(np.nanpercentile(nc["values"], 95))
        self.assertAlmostEqual(r["control"]["p95"], p95); self.assertAlmostEqual(r["W1_above_control"]["share"], float((np.where(np.arange(112) % 2 == 0, 0.97, 0.6) >= p95).mean()))

    def test_features_that_do_nothing_in_the_quantized_model_are_counted_with_similarity_zero(self):
        w = self.st["INT6"]["write"]; w["zero_q"][:7] = True; w["cos"][:7] = 0.0; r = self.tab()["all"]
        self.assertEqual(r["n_zero_q"], 7); self.assertAlmostEqual(r["share_zero_q"], 7 / 112); self.assertEqual(r["W1"]["n_defined"], 112)                       # similarity zero is a defined value

    def test_features_without_a_full_precision_effect_are_undefined_and_counted(self):
        w = self.st["INT6"]["write"]; w["zero_fp"][:4] = True; w["cos"][:4] = np.nan; r = self.tab()["all"]
        self.assertEqual(r["n_zero_fp"], 4); self.assertEqual(r["W1"]["n_defined"], 108); self.assertEqual(r["W1"]["n_total"], 112)

    def test_the_loss_ratio_the_magnitude_ratio_and_the_inactive_share(self):
        w = self.st["INT6"]["write"]; w["dnll"][:] = 0.02; w["magnitude_ratio"][:] = 0.8; w["n_active_q"][:] = 30
        fp = self.st["FP"]["write"]; fp["dnll"][:] = 0.01; fp["dnll"][:3] = 0.0
        r = self.tab()["all"]; self.assertAlmostEqual(r["W3_loss_ratio"]["median"], 2.0); self.assertEqual(r["W3_loss_ratio"]["n_defined"], 109)                    # no effect in FP: undefined
        self.assertAlmostEqual(r["W3_magnitude"]["median"], 0.8); self.assertAlmostEqual(r["inactive_share"]["median"], 0.25)

    def test_the_role_groups_are_tabulated_separately(self):
        for k, (fid, f) in enumerate(self.sm["features"].items()):
            f["roles"] = ["k64", "lost@INT6"] if k % 4 == 0 else ["retained@INT6"] if k % 4 == 1 else ["k64"]
        self.T = an.assemble(self.sa, self.sm); t = self.tab()
        self.assertEqual(sorted(t), ["all", "k64", "lost@INT6", "retained@INT6"]); self.assertEqual(t["lost@INT6"]["n_features"], 30); self.assertEqual(t["retained@INT6"]["n_features"], 30); self.assertEqual(t["k64"]["n_features"], 90)

    def test_the_retrieval_numbers_of_the_run_are_passed_through(self):
        r = self.tab()["all"]["retrieval"]; self.assertEqual(r["top1"], 0.9); self.assertEqual(r["within"]["chance"], 0.1)

def build_world(kind, seed=0, per_lang=40):
    """Planted validity worlds. 'sensible': the D-035 status carries the function (retained features keep W1 high, lost ones do not) and the Duan survivors keep it; 'independent': W1 does not depend on the status;
    'inadequate': the repeat measurement of the same weights is not similar. Every feature of the sample is tested; 20 % are 'lost' at INT6, INT5 and INT4 (roles lost@p and retained@p for the F4 subsets)."""
    sa, st, sm = make_world(per_lang=per_lang, n_excluded_per_lang=0, seed=seed); rng = np.random.default_rng(seed + 1)
    for k, f in enumerate(sm["features"].values()):
        lost = k % 5 == 0
        for c in ("INT6", "INT5", "INT4"):
            f["status"][c] = "different_peak" if lost else "retained"
        f["roles"] = ["k64"] + [f"{'lost' if lost else 'retained'}@{c}" for c in ("INT6", "INT5", "INT4")]
    T = an.assemble(sa, sm); lost = T["status"]["INT4"] != "retained"; n = len(lost)
    floor_cos = (lambda: np.full(n, 0.5)) if kind == "inadequate" else (lambda: rng.uniform(0.5, 1.0, n))
    for label in ("FP_rebatched", "INT16"):
        st[label]["write"]["cos"] = floor_cos()
    for label in ("INT6", "INT5", "INT4"):
        st[label]["write"]["cos"] = np.where(lost, rng.uniform(0.2, 0.5, n), rng.uniform(0.9, 1.0, n)) if kind != "independent" else rng.uniform(0.2, 1.0, n)
        corr = np.where(T["all"]["status"][label] != "retained", 0.5, 0.95) + rng.uniform(-0.03, 0.03, len(T["all"]["fid"])) if kind != "independent" else rng.uniform(0.3, 1.0, len(T["all"]["fid"]))
        st[label]["read"]["corr"] = corr.astype(float)
    for s in st.values():                                                                                        # the diagonal of S carries the planted similarity, as in the runner
        w = s["write"]; S = np.asarray(w["S"]).astype(np.float64); S[np.arange(n), np.arange(n)] = w["cos"]; w["S"] = S.astype(np.float16)
    return sa, st, sm, T


def build_varying_world(seed=0, per_lang=40):
    """Like the sensible world, but the lost sets differ by precision (lost at INT6: k % 5 == 0, at INT5: k % 5 == 1, at INT4: k % 5 == 2) and every state's W1 and token correlation follow ITS OWN status, so that a
    statement that reads the wrong precision gives another number."""
    sa, st, sm = make_world(per_lang=per_lang, n_excluded_per_lang=0, seed=seed); rng = np.random.default_rng(seed + 1); res = {"INT6": 0, "INT5": 1, "INT4": 2}
    for k, f in enumerate(sm["features"].values()):
        f["roles"] = ["k64"]
        for c, r in res.items():
            lost = k % 5 == r; f["status"][c] = "different_peak" if lost else "retained"; f["roles"].append(f"{'lost' if lost else 'retained'}@{c}")
    T = an.assemble(sa, sm); n = len(T["fid"])
    for label in ("FP_rebatched", "INT16"):
        st[label]["write"]["cos"] = rng.uniform(0.5, 1.0, n)
    for label in res:
        lost = T["status"][label] != "retained"; weak = lost | (rng.random(n) < 0.1)                                    # 10 % of the retained features have lost their function anyway
        st[label]["write"]["cos"] = np.where(weak, rng.uniform(0.2, 0.5, n), rng.uniform(0.9, 1.0, n))
        st[label]["read"]["corr"] = np.where(T["all"]["status"][label] != "retained", 0.5, 0.95).astype(float)
    for s in st.values():
        w = s["write"]; S = np.asarray(w["S"]).astype(np.float64); S[np.arange(n), np.arange(n)] = w["cos"]; w["S"] = S.astype(np.float16)
    return sa, st, sm, T


class VaryingWorld(unittest.TestCase):
    def setUp(self):
        self.sa, self.st, self.sm, self.T = build_varying_world(); self.floor = an.floor_report(self.st); self.tau = self.floor["tau"]
        import phase1_func_f5_metrics as met
        self.met = met

    def test_each_statement_reads_its_own_precision(self):
        v = an.validity(self.st, self.T, self.floor, n_boot=50, seed=0)["statements"]; T, st, tau = self.T, self.st, self.tau
        ret6 = T["status"]["INT6"] == "retained"; self.assertAlmostEqual(v["S1"]["share"], float((st["INT6"]["write"]["cos"][ret6] >= tau).mean())); self.assertEqual(v["S1"]["n_defined"], int(ret6.sum()))
        lost4, kept4 = np.array(["lost@INT4" in r for r in T["roles"]]), np.array(["retained@INT4" in r for r in T["roles"]]); c4 = st["INT4"]["write"]["cos"]; pair = lost4 | kept4
        a = self.met.auroc(kept4[pair].astype(float), c4[pair] >= tau); self.assertAlmostEqual(v["S2"]["auroc"], a["auroc"]); self.assertEqual((v["S2"]["n_pos"], v["S2"]["n_neg"]), (a["n_pos"], a["n_neg"]))
        duan5 = st["INT5"]["read"]["corr"][T["rows"]] >= 0.9; self.assertAlmostEqual(v["S3"]["share"], float((st["INT5"]["write"]["cos"][duan5] >= tau).mean())); self.assertEqual(v["S3"]["n_defined"], int(duan5.sum()))

    def test_a_similarity_exactly_at_the_floor_counts_as_at_or_above_in_the_event(self):
        T, st, tau = self.T, self.st, self.tau; kept4 = np.array(["retained@INT4" in r for r in T["roles"]]); j = int(np.flatnonzero(kept4)[0]); c4 = st["INT4"]["write"]["cos"]; c4[j] = tau
        v = an.validity(st, T, self.floor, n_boot=50, seed=0)["statements"]; lost4 = np.array(["lost@INT4" in r for r in T["roles"]]); pair = lost4 | kept4
        self.assertEqual(v["S2"]["n_pos"], int((c4[pair] >= tau).sum()))
        c6 = st["INT6"]["write"]["cos"]; ret6 = T["status"]["INT6"] == "retained"; c6[np.flatnonzero(ret6)[0]] = tau
        self.assertAlmostEqual(an.validity(st, T, self.floor, n_boot=50, seed=0)["statements"]["S1"]["share"], float((c6[ret6] >= tau).mean()))


class Guards(unittest.TestCase):
    """A statement whose data cannot support a verdict is undefined, not 'not met' (D-057)."""
    def test_the_minimum_count_is_ten(self):
        self.assertEqual(an.MIN_N, 10)

    def test_s2_is_undefined_when_fewer_than_ten_features_reach_the_floor_even_if_the_auroc_is_low(self):
        sa, st, sm, T = build_world("sensible"); floor = an.floor_report(st); tau = floor["tau"]; c = st["INT4"]["write"]["cos"]; c[:] = 0.3; c[:3] = 0.99                      # three events only
        s = an.validity(st, T, floor, n_boot=50, seed=0)["statements"]["S2"]
        self.assertEqual(s["n_pos"], 3); self.assertEqual(s["verdict"], "undefined"); self.assertFalse(np.isnan(s["auroc"]))                                                    # the number is still reported

    def test_s2_is_undefined_when_fewer_than_ten_features_miss_the_floor(self):
        sa, st, sm, T = build_world("sensible"); floor = an.floor_report(st); c = st["INT4"]["write"]["cos"]; c[:] = 0.99; c[:4] = 0.1
        self.assertEqual(an.validity(st, T, floor, n_boot=50, seed=0)["statements"]["S2"]["verdict"], "undefined")

    def test_s2_with_exactly_ten_events_and_ten_non_events_is_decided(self):
        sa, st, sm, T = build_varying_world(); floor = an.floor_report(st); tau = floor["tau"]; kept = np.array(["retained@INT4" in r for r in T["roles"]]); lost = np.array(["lost@INT4" in r for r in T["roles"]]); pair = np.flatnonzero(kept | lost)
        c = st["INT4"]["write"]["cos"]; c[:] = 0.0; c[pair[:10]] = 0.99; c[pair[10:20]] = 0.1
        s = an.validity(st, T, floor, n_boot=50, seed=0)["statements"]["S2"]; self.assertEqual((s["n_pos"], s["n_neg"]), (10, int((kept | lost).sum()) - 10)); self.assertNotEqual(s["verdict"], "undefined")

    def test_s1_and_s3_are_undefined_below_ten_defined_features(self):
        sa, st, sm, T = build_world("sensible"); floor = an.floor_report(st)
        for lab, key in (("INT6", "S1"), ("INT5", "S3")):
            T2 = dict(T); T2["status"] = dict(T["status"]); T2["status"][lab] = np.where(np.arange(len(T["fid"])) < 9, "retained", "different_peak")
            st2 = copy.deepcopy(st); st2["INT5"]["read"]["corr"] = np.where(np.arange(len(st["INT5"]["read"]["corr"])) < 9, 0.95, 0.2)
            T2["rows"] = T["rows"]
            s = an.validity(st2, T2, floor, n_boot=50, seed=0)["statements"][key]; self.assertEqual(s["n_defined"], 9, key); self.assertEqual(s["verdict"], "undefined", key); self.assertFalse(np.isnan(s["share"]), key)

    def test_ten_defined_features_are_enough_for_s1(self):
        sa, st, sm, T = build_world("sensible"); floor = an.floor_report(st); T2 = dict(T); T2["status"] = dict(T["status"]); T2["status"]["INT6"] = np.where(np.arange(len(T["fid"])) < 10, "retained", "different_peak")
        s = an.validity(st, T2, floor, n_boot=50, seed=0)["statements"]["S1"]; self.assertEqual(s["n_defined"], 10); self.assertIn(s["verdict"], ("met", "not met"))


    def test_ten_defined_features_are_enough_for_s3(self):
        sa, st, sm, T = build_world("sensible"); floor = an.floor_report(st); st["INT5"]["read"]["corr"] = np.where(np.arange(len(st["INT5"]["read"]["corr"])) < 10, 0.95, 0.2)
        s = an.validity(st, T, floor, n_boot=50, seed=0)["statements"]["S3"]; self.assertEqual(s["n_defined"], 10); self.assertIn(s["verdict"], ("met", "not met"))


class DirectionControlRuns(unittest.TestCase):
    def test_a_control_run_and_an_ordinary_run_are_never_merged(self):
        w = build_world("sensible")
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            save_run(w, a, ["FP", "INT8"], meta={"complete": True, "work_dtype": "float32", "direction_control": True}); save_run(w, b, ["FP", "INT6"], meta={"complete": True, "work_dtype": "float32", "direction_control": False})
            with self.assertRaises(ValueError):
                an.load_runs([a, b])

    def test_two_control_runs_merge_and_carry_the_flag(self):
        w = build_world("sensible")
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            save_run(w, a, ["FP", "INT8"], meta={"complete": True, "work_dtype": "float32", "direction_control": True}); save_run(w, b, ["FP", "INT6"], meta={"complete": True, "work_dtype": "float32", "direction_control": True})
            self.assertTrue(an.load_runs([a, b])[2]["direction_control"])
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            save_run(w, a, ["FP", "INT8"], meta={"complete": True, "work_dtype": "float32"}); save_run(w, b, ["FP", "INT6"], meta={"complete": True, "work_dtype": "float32"})
            self.assertFalse(an.load_runs([a, b])[2]["direction_control"])                                              # runs written before the option existed are ordinary runs

    def test_the_report_of_a_control_run_says_so_at_the_top_of_the_first_section(self):
        sa, st, sm, T = build_varying_world()
        ctrl = an.analyse_f5(sa, st, sm, meta={"direction_control": True}, n_boot=30, seed=0); self.assertTrue(ctrl["instrument"]["direction_control"])
        md = an.report_text(ctrl); sec1 = md.split("## 2.")[0]; self.assertIn("DIRECTION-SWAP CONTROL RUN", sec1); self.assertIn("not the effect of the features themselves", sec1)
        plain = an.report_text(an.analyse_f5(sa, st, sm, meta=None, n_boot=30, seed=0)); self.assertNotIn("DIRECTION-SWAP", plain)


class FixedAmplitudeRuns(unittest.TestCase):
    def test_a_fixed_amplitude_run_is_never_merged_with_an_ordinary_run_or_a_differently_varied_one(self):
        w = build_world("sensible"); base = {"complete": True, "work_dtype": "float32"}
        for a_meta, b_meta in (({**base, "fixed_amplitude": True}, base), ({**base, "fixed_amplitude": True}, {**base, "direction_control": True}), ({**base, "fixed_amplitude": True, "direction_control": True}, {**base, "fixed_amplitude": True})):
            with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
                save_run(w, a, ["FP", "INT8"], meta=a_meta); save_run(w, b, ["FP", "INT6"], meta=b_meta)
                with self.assertRaises(ValueError):
                    an.load_runs([a, b])

    def test_matching_variants_merge_and_carry_both_flags(self):
        w = build_world("sensible"); m = {"complete": True, "work_dtype": "float32", "fixed_amplitude": True, "direction_control": True}
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            save_run(w, a, ["FP", "INT8"], meta=m); save_run(w, b, ["FP", "INT6"], meta=m); meta = an.load_runs([a, b])[2]; self.assertTrue(meta["fixed_amplitude"]); self.assertTrue(meta["direction_control"])

    def test_the_report_says_that_the_amplitudes_were_fixed(self):
        sa, st, sm, T = build_varying_world(); res = an.analyse_f5(sa, st, sm, meta={"fixed_amplitude": True}, n_boot=30, seed=0); self.assertTrue(res["instrument"]["fixed_amplitude"])
        sec1 = an.report_text(res).split("## 2.")[0]; self.assertIn("FIXED-AMPLITUDE RUN", sec1); self.assertIn("full-precision amplitudes", sec1)
        self.assertNotIn("FIXED-AMPLITUDE", an.report_text(an.analyse_f5(sa, st, sm, meta=None, n_boot=30, seed=0)))
        both = an.report_text(an.analyse_f5(sa, st, sm, meta={"fixed_amplitude": True, "direction_control": True}, n_boot=30, seed=0)).split("## 2.")[0]; self.assertIn("FIXED-AMPLITUDE RUN", both); self.assertIn("DIRECTION-SWAP CONTROL RUN", both)


class ContinuousAuroc(unittest.TestCase):
    def test_the_descriptive_auroc_is_the_probability_that_a_retained_feature_keeps_more_function_than_a_lost_one(self):
        import phase1_func_f5_metrics as met
        sa, st, sm, T = build_varying_world(); floor = an.floor_report(st); v = an.validity(st, T, floor, n_boot=60, seed=0); d = v["descriptive"]["S2c"]
        kept = np.array(["retained@INT4" in r for r in T["roles"]]); lost = np.array(["lost@INT4" in r for r in T["roles"]]); pair = kept | lost; c = st["INT4"]["write"]["cos"]
        want = met.auroc(c[pair], kept[pair])
        self.assertAlmostEqual(d["auroc"], want["auroc"]); self.assertEqual((d["n_pos"], d["n_neg"]), (want["n_pos"], want["n_neg"])); self.assertLessEqual(d["lo"], d["auroc"]); self.assertGreaterEqual(d["hi"], d["auroc"])
        self.assertNotIn("S2c", v["statements"]); self.assertNotIn("verdict", d)                                                                       # descriptive: no verdict, not part of the overall

    def test_the_descriptive_auroc_is_below_one_half_when_lost_features_keep_more(self):
        sa, st, sm, T = build_world("sensible"); lost = T["status"]["INT4"] != "retained"; rng = np.random.default_rng(4); n = len(lost)
        st["INT4"]["write"]["cos"] = np.where(lost, rng.uniform(0.9, 1.0, n), rng.uniform(0.2, 0.5, n)); d = an.validity(st, T, an.floor_report(st), n_boot=60, seed=0)["descriptive"]["S2c"]
        self.assertLess(d["auroc"], 0.05); self.assertLess(d["hi"], 0.5)

    def test_it_is_undefined_without_pairs_and_the_report_lists_it(self):
        sa, st, sm, T = build_varying_world(); res = an.analyse_f5(sa, st, sm, meta=None, n_boot=40, seed=0); md = an.report_text(res)
        self.assertIn("descriptive, no rule", md); self.assertIn(an._f3(res["validity"]["descriptive"]["S2c"]["auroc"]), md)
        for f in sm["features"].values():
            f["roles"] = ["k64"]
        d = an.validity(st, an.assemble(sa, sm), an.floor_report(st), n_boot=30, seed=0)["descriptive"]["S2c"]; self.assertTrue(np.isnan(d["auroc"]))


class Rules(unittest.TestCase):
    def test_the_share_rule_is_at_least_eighty_percent(self):
        self.assertEqual(an.rule_share(0.80), "met"); self.assertEqual(an.rule_share(0.7999), "not met"); self.assertEqual(an.rule_share(float("nan")), "undefined")

    def test_the_auroc_rule_needs_sixty_percent_and_a_lower_bound_strictly_above_one_half(self):
        self.assertEqual(an.rule_auroc(0.60, 0.5001), "met"); self.assertEqual(an.rule_auroc(0.60, 0.5), "not met"); self.assertEqual(an.rule_auroc(0.5999, 0.55), "not met")
        self.assertEqual(an.rule_auroc(float("nan"), float("nan")), "undefined"); self.assertEqual(an.rule_auroc(0.9, float("nan")), "undefined")

    def test_the_overall_verdict_is_supported_only_when_all_three_are_met(self):
        self.assertEqual(an.overall(["met", "met", "met"]), "supported"); self.assertEqual(an.overall(["met", "not met", "met"]), "not supported")
        self.assertEqual(an.overall(["met", "undefined", "met"]), "undetermined"); self.assertEqual(an.overall(["undefined", "not met", "met"]), "not supported")


class Validity(unittest.TestCase):
    def run_kind(self, kind, **kw):
        sa, st, sm, T = build_world(kind, **kw); floor = an.floor_report(st)
        return an.validity(st, T, floor, n_boot=200, seed=0), floor, st, T

    def test_a_world_in_which_the_status_carries_the_function_satisfies_all_three_statements(self):
        v, floor, st, T = self.run_kind("sensible")
        self.assertTrue(floor["adequate"]); s = v["statements"]
        self.assertEqual([s["S1"]["verdict"], s["S2"]["verdict"], s["S3"]["verdict"]], ["met", "met", "met"]); self.assertEqual(v["overall"], "supported")
        self.assertGreaterEqual(s["S1"]["share"], 0.8); self.assertAlmostEqual(s["S2"]["auroc"], 1.0); self.assertGreater(s["S2"]["lo"], 0.5); self.assertGreaterEqual(s["S3"]["share"], 0.8)

    def test_a_world_in_which_the_function_does_not_depend_on_the_status_fails_the_statements(self):
        v, floor, st, T = self.run_kind("independent"); s = v["statements"]
        self.assertTrue(floor["adequate"]); self.assertEqual(s["S2"]["verdict"], "not met"); self.assertLess(abs(s["S2"]["auroc"] - 0.5), 0.2); self.assertEqual(s["S1"]["verdict"], "not met"); self.assertEqual(v["overall"], "not supported")

    def test_an_inadequate_instrument_makes_every_statement_undefined_but_keeps_the_numbers(self):
        v, floor, st, T = self.run_kind("inadequate"); s = v["statements"]
        self.assertFalse(floor["adequate"]); self.assertEqual([s["S1"]["verdict"], s["S2"]["verdict"], s["S3"]["verdict"]], ["undefined"] * 3); self.assertEqual(v["overall"], "undetermined")
        self.assertFalse(np.isnan(s["S2"]["auroc"])); self.assertIn("inadequate", v["note"])

    def test_s1_uses_the_d035_label_retained_at_int6_and_s3_the_duan_cutoff_at_int5(self):
        sa, st, sm, T = build_world("sensible"); floor = an.floor_report(st); tau = floor["tau"]
        cos6, cos5 = st["INT6"]["write"]["cos"], st["INT5"]["write"]["cos"]; ret6 = T["status"]["INT6"] == "retained"; duan5 = st["INT5"]["read"]["corr"][T["rows"]] >= 0.9
        v = an.validity(st, T, floor, n_boot=50, seed=0)["statements"]
        self.assertAlmostEqual(v["S1"]["share"], float((cos6[ret6] >= tau).mean())); self.assertEqual(v["S1"]["n_defined"], int(ret6.sum()))
        self.assertAlmostEqual(v["S3"]["share"], float((cos5[duan5] >= tau).mean())); self.assertEqual(v["S3"]["n_defined"], int(duan5.sum()))
        # the other way round: change the labels and the cutoff set changes
        st["INT5"]["read"]["corr"][:] = 0.5; v2 = an.validity(st, T, floor, n_boot=50, seed=0)["statements"]; self.assertEqual(v2["S3"]["n_defined"], 0); self.assertEqual(v2["S3"]["verdict"], "undefined")

    def test_s2_is_computed_in_the_f4_pairs_at_int4_only(self):
        sa, st, sm, T = build_world("sensible"); floor = an.floor_report(st)
        for f in sm["features"].values():
            f["roles"] = ["k64"]                                                                              # no F4 role at all
        T2 = an.assemble(sa, sm); v = an.validity(st, T2, floor, n_boot=50, seed=0)["statements"]
        self.assertEqual(v["S2"]["verdict"], "undefined"); self.assertEqual(v["S2"]["n_pos"] + v["S2"]["n_neg"], 0)

    def test_the_confusion_tables_count_status_against_the_event_and_the_duan_cutoff_against_the_event(self):
        v, floor, st, T = self.run_kind("sensible"); c = v["confusion"]["INT4"]["status"]; tau = floor["tau"]
        ret = T["status"]["INT4"] == "retained"; ev = st["INT4"]["write"]["cos"] >= tau
        self.assertEqual(c, {"retained_event": int((ret & ev).sum()), "retained_no_event": int((ret & ~ev).sum()), "lost_event": int((~ret & ev).sum()), "lost_no_event": int((~ret & ~ev).sum())})
        d = v["confusion"]["INT5"]["duan"]; duan = st["INT5"]["read"]["corr"][T["rows"]] >= 0.9; ev5 = st["INT5"]["write"]["cos"] >= tau
        self.assertEqual(d["survives_event"], int((duan & ev5).sum())); self.assertEqual(d["fails_no_event"], int((~duan & ~ev5).sum()))

    def test_the_spearman_correlation_of_read_and_write_side_is_reported_per_state_with_an_interval(self):
        v, floor, st, T = self.run_kind("independent"); sp = v["spearman"]["INT6"]
        import phase1_func_f5_metrics as met
        self.assertAlmostEqual(sp["rho"], met.spearman(st["INT6"]["read"]["corr"][T["rows"]], st["INT6"]["write"]["cos"])["rho"]); self.assertLessEqual(sp["lo"], sp["rho"]); self.assertGreaterEqual(sp["hi"], sp["rho"])

    def test_the_paired_difference_of_retained_and_lost_by_language_has_the_right_sign(self):
        v, floor, st, T = self.run_kind("sensible"); p = v["paired"]["INT4"]
        self.assertGreater(p["median"], 0.4); self.assertEqual(p["sign"], "positive"); self.assertEqual(len(p["by_language"]), 4)
        v, floor, st, T = self.run_kind("independent"); self.assertLess(abs(v["paired"]["INT4"]["median"]), 0.2)                       # no function difference was planted between the statuses

def save_run(world, directory, labels, meta=None):
    """Write the files of a runner job for the given states (S as a torch half tensor, as the runner saves it)."""
    sa, st, sm = world[:3]; d = Path(directory); d.mkdir(parents=True, exist_ok=True)
    torch.save(dict(sa), d / "f5_stage_a.pt")
    for label in labels:
        res = copy.deepcopy(st[label]); res["write"]["S"] = torch.from_numpy(np.asarray(res["write"]["S"])); torch.save({"label": label, **res}, d / f"f5_state_{label}.pt")
    (d / "meta_f5.json").write_text(json.dumps(meta or {"complete": True, "work_dtype": "float32", "gates": {"logits_native_residual": {"ok": True}}, "states": list(labels)}))


class Checks(unittest.TestCase):
    def test_a_consistent_run_passes(self):
        sa, st, sm, T = build_world("sensible"); an.check_inputs(sa, st)

    def test_inconsistent_inputs_are_refused(self):
        for mutate, why in ((lambda sa, st: st["INT6"]["write"].update({"S": st["INT6"]["write"]["S"][:, :-1]}), "S not square"),
                            (lambda sa, st: st["INT6"]["write"]["S"].__setitem__((3, 3), np.float16(0.123)), "diagonal differs from the similarity"),
                            (lambda sa, st: (st["FP"]["write"]["cos"].__setitem__(5, 0.5), st["FP"]["write"]["S"].__setitem__((5, 5), np.float16(0.5))), "FP is not the identity"),
                            (lambda sa, st: st["INT6"]["write"].update({"S": np.pad(np.asarray(st["INT6"]["write"]["S"]), ((0, 0), (0, 1)))}), "S with an extra column but the right diagonal"),
                            (lambda sa, st: st["INT6"]["write"].update({"dnll": st["INT6"]["write"]["dnll"][:-1]}), "write-side length"),
                            (lambda sa, st: st["INT6"]["write"].update({"cos": st["INT6"]["write"]["cos"][:-1]}), "length"),
                            (lambda sa, st: st.pop("FP"), "no FP"),
                            (lambda sa, st: st["INT6"]["read"].update({"corr": st["INT6"]["read"]["corr"][:-1]}), "read length")):
            sa, st, sm, T = build_world("sensible"); mutate(sa, st)
            with self.assertRaises(ValueError, msg=why):
                an.check_inputs(sa, st)


class Pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sa, cls.st, cls.sm, cls.T = build_world("sensible")
        cls.meta = {"complete": True, "work_dtype": "float32", "gates": {"logits_native_residual": {"ok": True, "top1": 1.0, "max_abs": 0.1}}, "sham_gate": cls.sa["sham_gate"], "states": list(cls.st)}
        cls.res = an.analyse_f5(cls.sa, cls.st, cls.sm, meta=cls.meta, n_boot=100, seed=0)

    def test_every_state_except_full_precision_has_tables_and_the_instrument_checks_are_reported(self):
        r = self.res; self.assertEqual(sorted(r["tables"]), ["FP_rebatched", "INT16", "INT4", "INT5", "INT6", "INT8"])
        i = r["instrument"]; self.assertTrue(i["sham_gate_exact"]); self.assertTrue(i["logit_gates_ok"]); self.assertEqual(i["work_dtype"], "float32"); self.assertTrue(i["complete"]); self.assertTrue(i["fp_identity"])
        self.assertEqual(r["counts"]["n_sample"], 160); self.assertEqual(r["counts"]["n_tested"], 160); self.assertEqual(r["counts"]["n_excluded"], 0)

    def test_the_verdict_of_the_planted_world_is_supported_and_comes_with_the_floor(self):
        r = self.res; self.assertEqual(r["validity"]["overall"], "supported"); self.assertAlmostEqual(r["floor"]["tau"], r["validity"]["tau"]); self.assertTrue(r["floor"]["adequate"])

    def test_missing_meta_is_reported_as_unknown_not_as_ok(self):
        r = an.analyse_f5(self.sa, self.st, self.sm, meta=None, n_boot=20, seed=0)["instrument"]
        self.assertIsNone(r["logit_gates_ok"]); self.assertIsNone(r["complete"]); self.assertIsNone(r["work_dtype"]); self.assertTrue(r["sham_gate_exact"])                  # the sham gate is in stage A, not in the metadata
        sa = dict(self.sa); del sa["sham_gate"]; self.assertIsNone(an.analyse_f5(sa, self.st, self.sm, meta=None, n_boot=20, seed=0)["instrument"]["sham_gate_exact"])

    def test_a_job_with_only_some_states_still_gives_the_statements_it_can_and_leaves_the_others_undefined(self):
        st = {k: v for k, v in self.st.items() if k in ("FP", "FP_rebatched", "INT4")}; r = an.analyse_f5(self.sa, st, self.sm, meta=self.meta, n_boot=50, seed=0)
        s = r["validity"]["statements"]; self.assertEqual((s["S1"]["verdict"], s["S3"]["verdict"]), ("undefined", "undefined")); self.assertEqual(s["S2"]["verdict"], "met"); self.assertEqual(r["validity"]["overall"], "undetermined")
        self.assertEqual(sorted(r["tables"]), ["FP_rebatched", "INT4"])

    def test_the_descriptive_hypotheses_are_derived_from_the_tables(self):
        h = self.res["hypotheses"]; tb = self.res["tables"]
        self.assertAlmostEqual(h["H5a"]["median_R1"]["INT6"], tb["INT6"]["all"]["R1"]["median"]); self.assertAlmostEqual(h["H5b"]["median_W1"]["INT4"], tb["INT4"]["all"]["W1"]["median"])
        self.assertEqual(h["H5b"]["W1_below_R1"]["INT4"], bool(tb["INT4"]["all"]["W1"]["median"] < tb["INT4"]["all"]["R1"]["median"]))
        self.assertEqual(h["H5c"]["sign"]["INT4"], self.res["validity"]["paired"]["INT4"]["sign"]); self.assertAlmostEqual(h["H5b"]["share_below_floor"]["INT4"], 1.0 - tb["INT4"]["all"]["W1_above_floor"]["share"])

    def test_the_outputs_are_strict_json_a_flat_table_and_a_statement_file(self):
        with tempfile.TemporaryDirectory() as d:
            an.write_outputs(self.res, d)
            j = json.loads((Path(d) / "f5_results.json").read_text(), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))                   # no NaN or Infinity tokens
            self.assertEqual(j["validity"]["overall"], "supported")
            rows = (Path(d) / "f5_tables.csv").read_text().strip().split("\n"); head = rows[0].split(",")
            for col in ("state", "group", "n_features", "n_tested", "R1_median", "W1_median", "W1_above_floor", "W1_above_floor_lo", "W1_above_floor_hi", "share_zero_q"):
                self.assertIn(col, head)
            self.assertEqual(len(rows) - 1, sum(len(g) for g in self.res["tables"].values()))
            md = (Path(d) / "statements.md").read_text()
            for needle in ("S1", "S2", "S3", "supported", "exploratory", "numerical floor", "INT16"):
                self.assertIn(needle, md)
            s = self.res["validity"]["statements"]; rows = {r.split("|")[1].strip()[:2]: r for r in md.split("\n") if r.startswith("| S")}
            self.assertIn(an._f3(s["S1"]["share"]), rows["S1"]); self.assertIn(an._f3(s["S2"]["auroc"]), rows["S2"]); self.assertIn(an._f3(s["S3"]["share"]), rows["S3"])
            for k in ("S1", "S2", "S3"):
                self.assertTrue(rows[k].rstrip().endswith(f"| {s[k]['verdict']} |"))
            self.assertIn(f"{self.res['floor']['tau']:.4f}", md); self.assertIn(f"not part of the floor: median {an._f3(self.res['floor']['int8']['median'])}", md)
            for banned in ("confirmed", "proved", "demonstrated"):
                self.assertNotIn(banned, md)

    def test_nan_becomes_null_in_the_json(self):
        self.assertEqual(an.to_jsonable({"a": float("nan"), "b": np.float64(1.5), "c": np.array([1, 2]), "d": [np.int64(3), float("inf")]}), {"a": None, "b": 1.5, "c": [1, 2], "d": [3, None]})


class Loading(unittest.TestCase):
    def test_a_run_directory_is_loaded_back_to_numpy(self):
        w = build_world("sensible")
        with tempfile.TemporaryDirectory() as d:
            save_run(w, d, ["FP", "INT6"]); sa, st, meta = an.load_run(d)
            self.assertEqual(sorted(st), ["FP", "INT6"]); self.assertIsInstance(st["INT6"]["write"]["S"], np.ndarray); self.assertEqual(st["INT6"]["write"]["S"].dtype, np.float16)
            np.testing.assert_array_equal(st["INT6"]["write"]["cos"], w[1]["INT6"]["write"]["cos"]); self.assertEqual(sa["write_features"], w[0]["write_features"]); self.assertTrue(meta["complete"])

    def test_two_jobs_are_merged_when_they_agree_on_the_tested_features_and_the_contexts(self):
        w = build_world("sensible")
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            save_run(w, a, ["FP", "FP_rebatched", "INT8"]); save_run(w, b, ["FP", "INT6", "INT5"])
            sa, st, meta = an.load_runs([a, b]); self.assertEqual(sorted(st), ["FP", "FP_rebatched", "INT5", "INT6", "INT8"]); self.assertEqual(meta["runs"], [str(a), str(b)])

    def test_jobs_with_different_tested_features_are_refused(self):
        w = build_world("sensible"); w2 = build_world("sensible", seed=3)
        w2[0]["write_features"] = list(reversed(w2[0]["write_features"]))
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            save_run(w, a, ["FP", "INT8"]); save_run(w2, b, ["FP", "INT6"])
            with self.assertRaises(ValueError):
                an.load_runs([a, b])

    def test_a_state_in_two_jobs_is_refused(self):
        w = build_world("sensible")
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            save_run(w, a, ["FP", "INT8"]); save_run(w, b, ["FP", "INT8"])
            with self.assertRaises(ValueError):
                an.load_runs([a, b])

class Main(unittest.TestCase):
    def test_the_command_line_merges_jobs_and_writes_the_three_files(self):
        import contextlib
        import io
        w = build_world("sensible")
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b, tempfile.TemporaryDirectory() as o:
            save_run(w, a, ["FP", "FP_rebatched", "INT8"]); save_run(w, b, ["FP", "INT6", "INT5", "INT4"]); (Path(o) / "sample.json").write_text(json.dumps(w[2]))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                an.main(["--run", a, "--run", b, "--sample", str(Path(o) / "sample.json"), "--out", str(Path(o) / "res"), "--n-boot", "40", "--figures", "../figures/f5/f5_fig3_statements_light.png"])
            self.assertEqual(sorted(p.name for p in (Path(o) / "res").iterdir()), ["f5_results.json", "f5_tables.csv", "report.md", "statements.md"])
            self.assertIn("(../figures/f5/f5_fig3_statements_light.png)", (Path(o) / "res" / "report.md").read_text()); self.assertIn("supported", buf.getvalue()); j = json.loads((Path(o) / "res" / "f5_results.json").read_text()); self.assertEqual(j["instrument"]["runs"], [str(a), str(b)])

class Sharpening(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sa, cls.st, cls.sm, cls.T = build_world("sensible"); cls.res = an.analyse_f5(cls.sa, cls.st, cls.sm, meta=None, n_boot=60, seed=0)

    def test_every_table_quantity_comes_from_its_own_source(self):
        g = self.res["tables"]["INT6"]["all"]; cos = self.st["INT6"]["write"]["cos"]
        self.assertAlmostEqual(g["R2"]["median"], 0.95); self.assertAlmostEqual(g["R3"]["median"], 1.05); self.assertAlmostEqual(g["W3_magnitude"]["median"], 0.8)
        self.assertAlmostEqual(g["W2"]["median"], float(np.median(self.st["INT6"]["write"]["topk_overlap"]))); self.assertAlmostEqual(g["R1_duan"]["share"], float((self.st["INT6"]["read"]["corr"] >= 0.9).mean()))

    def test_every_csv_column_reads_its_own_field(self):
        row = next(r for r in an.table_rows(self.res) if r["state"] == "INT6" and r["group"] == "all"); g = self.res["tables"]["INT6"]["all"]
        self.assertEqual((row["R1_median_lo"], row["R1_median_hi"]), (g["R1"]["median_lo"], g["R1"]["median_hi"])); self.assertEqual((row["W1_median_lo"], row["W1_median_hi"]), (g["W1"]["median_lo"], g["W1"]["median_hi"]))
        self.assertEqual((row["W1_above_floor"], row["W1_above_floor_lo"], row["W1_above_floor_hi"]), (g["W1_above_floor"]["share"], g["W1_above_floor"]["lo"], g["W1_above_floor"]["hi"]))
        self.assertEqual(row["W1_above_control"], g["W1_above_control"]["share"]); self.assertEqual(row["retrieval_within_chance"], g["retrieval"]["within"]["chance"]); self.assertEqual(row["retrieval_within_top1"], g["retrieval"]["within"]["top1"])
        self.assertNotEqual(row["W1_above_control"], row["W1_above_floor"]); self.assertLessEqual(row["R1_median_lo"], row["R1_median_hi"])

    def test_formatting_helpers(self):
        self.assertEqual(an._fmt(float("nan")), ""); self.assertEqual(an._fmt(None), ""); self.assertEqual(an._fmt(0.123456789), "0.123457"); self.assertEqual(an._fmt(7), "7")
        self.assertEqual(an._f3(0.123456789), "0.123"); self.assertEqual(an._f3(float("nan")), "n/a"); self.assertEqual(an._f3(None), "n/a"); self.assertEqual(an._f3(1.0), "1.000")
        self.assertEqual(an._ci(0.1, 0.2), " [0.100, 0.200]"); self.assertEqual(an._ci(float("nan"), 0.2), "")

    def test_the_sign_of_an_interval_is_decided_by_its_open_bounds(self):
        self.assertEqual([an._sign(*x) for x in ((0.1, 0.3), (-0.3, -0.1), (-0.1, 0.2), (0.0, 0.2), (-0.2, 0.0))], ["positive", "negative", "inconclusive", "inconclusive", "inconclusive"])

    def test_the_paired_difference_is_negative_when_the_lost_features_keep_more_function(self):
        sa, st, sm, T = build_world("sensible"); lost = T["status"]["INT4"] != "retained"; n = len(lost); rng = np.random.default_rng(5)
        st["INT4"]["write"]["cos"] = np.where(lost, rng.uniform(0.9, 1.0, n), rng.uniform(0.2, 0.5, n)); p = an.validity(st, T, an.floor_report(st), n_boot=60, seed=0)["paired"]["INT4"]
        self.assertEqual(p["sign"], "negative"); self.assertLess(p["median"], -0.3)

    def test_the_instrument_report_needs_every_gate_and_every_sham_to_pass(self):
        sa = dict(self.sa); sa["sham_gate"] = [{"feature": 1, "exact": True}, {"feature": 2, "exact": False}]
        meta = {"gates": {"a": {"ok": True}, "b": {"ok": False}}, "complete": True}; i = an.analyse_f5(sa, self.st, self.sm, meta=meta, n_boot=20, seed=0)["instrument"]
        self.assertFalse(i["sham_gate_exact"]); self.assertFalse(i["logit_gates_ok"])

    def test_the_counts_of_a_world_with_untested_features_by_language(self):
        sa, st, sm = make_world(); r = an.analyse_f5(sa, st, sm, meta=None, n_boot=20, seed=0)["counts"]
        self.assertEqual((r["n_sample"], r["n_tested"], r["n_excluded"]), (120, 112, 8)); self.assertEqual(r["excluded_by_language"], {l: 2 for l in LANGS})

    def test_the_hypothesis_flags_need_all_conditions(self):
        def mk(r1, w1):
            return {lab: {"all": {"R1": {"median": r1[lab]}, "W1": {"median": w1[lab]}, "W1_above_floor": {"share": 0.7}}} for lab in r1}
        val = {"paired": {"INT4": {"sign": "positive", "median": 0.3}}}
        r1 = {"INT8": 0.99, "INT6": 0.95, "INT5": 0.9, "INT4": 0.8}
        self.assertTrue(an.hypotheses(mk(r1, {"INT8": 0.9, "INT6": 0.9, "INT5": 0.8, "INT4": 0.7}), val, {})["H5a"]["holds"])
        for bad in ({**r1, "INT8": 0.85}, {**r1, "INT6": 0.85}, {**r1, "INT4": 0.97}):
            self.assertFalse(an.hypotheses(mk(bad, {k: 0.1 for k in r1}), val, {})["H5a"]["holds"])
        self.assertTrue(an.hypotheses(mk(r1, {k: 0.5 for k in r1}), val, {})["H5b"]["holds"])
        self.assertFalse(an.hypotheses(mk(r1, {**{k: 0.5 for k in r1}, "INT5": 0.97}), val, {})["H5b"]["holds"])
        self.assertAlmostEqual(an.hypotheses(mk(r1, {k: 0.5 for k in r1}), val, {})["H5b"]["share_below_floor"]["INT4"], 0.3)

    def test_the_statement_file_shows_each_statement_with_its_own_numbers(self):
        sa, st, sm, T = build_varying_world(); res = an.analyse_f5(sa, st, sm, meta=None, n_boot=40, seed=0); s = res["validity"]["statements"]
        rows = {r.split("|")[1].strip()[:2]: r for r in an.statements_text(res).split("\n") if r.startswith("| S")}
        for k, key in (("S1", "share"), ("S2", "auroc"), ("S3", "share")):
            self.assertIn(an._f3(s[k][key]) + " |", rows[k]); self.assertIn(an._ci(s[k]["lo"], s[k]["hi"]).strip(), rows[k])
        self.assertNotEqual(an._f3(s["S1"]["share"]), an._f3(s["S2"]["auroc"])); self.assertNotEqual(an._f3(s["S2"]["auroc"]), an._f3(s["S3"]["share"]))


class MergeDetails(unittest.TestCase):
    def test_merged_metadata_reports_incompleteness_mixed_dtypes_and_all_sham_gates(self):
        w = build_world("sensible"); w2 = build_world("sensible")
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            save_run(w, a, ["FP", "INT8"], meta={"complete": True, "work_dtype": "float32", "gates": {"g": {"ok": True}}}); save_run(w2, b, ["FP", "INT6"], meta={"complete": False, "work_dtype": "native", "gates": {"g": {"ok": True}}})
            sa, st, meta = an.load_runs([a, b]); self.assertFalse(meta["complete"]); self.assertEqual(meta["work_dtype"], "mixed"); self.assertEqual(sorted(meta["gates"]), ["run0:g", "run1:g"])
            self.assertEqual(len(sa["sham_gate"]), 2 * len(w[0]["sham_gate"]))

    def test_the_subset_cap_of_the_jobs_is_carried_into_the_merged_metadata(self):
        w = build_world("sensible")
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            save_run(w, a, ["FP", "INT8"], meta={"complete": True, "work_dtype": "float32", "per_cell": 12, "subset_seed": 3}); save_run(w, b, ["FP", "INT6"], meta={"complete": True, "work_dtype": "float32", "per_cell": 12, "subset_seed": 3})
            meta = an.load_runs([a, b])[2]; self.assertEqual((meta["per_cell"], meta["subset_seed"]), (12, 3))

    def test_full_precision_comes_from_the_first_job(self):
        w = build_world("sensible"); w2 = build_world("sensible"); w2[1]["FP"]["read"]["jaccard"] = np.full(len(w2[1]["FP"]["read"]["jaccard"]), 0.123)
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            save_run(w, a, ["FP", "INT8"]); save_run(w2, b, ["FP", "INT6"]); sa, st, meta = an.load_runs([a, b])
            self.assertAlmostEqual(float(st["FP"]["read"]["jaccard"][0]), 0.95)

class Report(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sa, st, sm, T = build_varying_world(); cls.res = an.analyse_f5(sa, st, sm, meta={"complete": True, "work_dtype": "float32", "gates": {"g": {"ok": True}}, "per_cell": 12, "subset_seed": 0}, n_boot=60, seed=0)
        cls.md = an.report_text(cls.res, figures=["f5_fig1_w1_distribution_light.png"])

    def test_the_report_has_its_sections_in_order(self):
        heads = [l for l in self.md.split("\n") if l.startswith("## ")]
        self.assertEqual([h.split(".")[0].replace("## ", "") for h in heads], ["1", "2", "3", "4", "5", "6", "7"])
        for needle in ("Instrument", "Main table", "lost and retained", "Statements", "Descriptive hypotheses", "Limits", "Figures"):
            self.assertTrue(any(needle in h for h in heads), needle)

    def test_the_main_table_has_a_row_per_state_with_the_numbers_of_the_tables(self):
        rows = {l.split("|")[1].strip(): l for l in self.md.split("\n") if l.startswith("| ") and l.split("|")[1].strip() in self.res["tables"]}
        self.assertEqual(sorted(rows), sorted(self.res["tables"]))
        for s, g in ((s, self.res["tables"][s]["all"]) for s in self.res["tables"]):
            for v in (g["R1"]["median"], g["W1"]["median"], g["W1_above_floor"]["share"], g["retrieval"]["top1"]):
                self.assertIn(f"{float(v):.3f}", rows[s])
            self.assertIn(str(g["n_tested"]), rows[s])

    def test_every_cell_of_a_main_table_row_is_its_own_quantity(self):
        res = copy.deepcopy(self.res); g = res["tables"]["INT6"]["all"]; g["n_zero_q"] = 7; g["n_zero_fp"] = 3; g["share_zero_q"] = 0.0625; md = an.report_text(res)
        row = [l for l in md.split("\n") if l.startswith("| INT6 ")][0]; cells = [c.strip() for c in row.split("|")[1:-1]]; rt = g["retrieval"]
        iv = lambda m, k="median", lo="median_lo", hi="median_hi": f"{m[k]:.3f} [{m[lo]:.3f}, {m[hi]:.3f}]"
        want = ["INT6", str(g["n_tested"]), iv(g["R1"]), f"{g['R1_duan']['share']:.3f}", f"{g['R2']['median']:.3f}", f"{g['R3']['median']:.3f}", iv(g["W1"]), iv(g["W1_above_floor"], "share", "lo", "hi"),
                f"{g['W1_above_control']['share']:.3f}", "7 (0.062)", f"{rt['top1']:.3f}", f"{rt['within']['top1']:.3f} ({rt['within']['chance']:.3f})"]
        self.assertEqual(cells, want); self.assertNotEqual(cells[4], cells[5]); self.assertNotEqual(cells[7], cells[8]); self.assertNotEqual(cells[10], cells[11].split(" ")[0])

    def test_the_role_table_puts_lost_before_retained_and_every_limit_and_the_subset_note_are_present(self):
        lines = self.md.split("\n")
        for lab in ("INT6", "INT5", "INT4"):
            self.assertLess([i for i, l in enumerate(lines) if l.startswith(f"| lost@{lab} ")][0], [i for i, l in enumerate(lines) if l.startswith(f"| retained@{lab} ")][0])
        for x in an.LIMITS:
            self.assertIn(x, self.md)
        sec1 = self.md.split("## 2.")[0]; self.assertIn("Stratified subset: a per-cell cap of 12 features per language and role (seed 0)", sec1)

    def test_the_role_table_lists_lost_and_retained_by_precision(self):
        for lab in ("INT6", "INT5", "INT4"):
            for kind in ("lost", "retained"):
                g = self.res["tables"][lab][f"{kind}@{lab}"]; line = [l for l in self.md.split("\n") if l.startswith(f"| {kind}@{lab} ")]
                self.assertEqual(len(line), 1); self.assertIn(f"{float(g['W1']['median']):.3f}", line[0]); self.assertIn(str(g["n_tested"]), line[0])

    def test_the_limits_the_subset_and_the_exploratory_status_are_stated(self):
        for needle in ("exploratory", "single-feature zero ablation", "one layer", "round-to-nearest", "evaluation sentences", "per-cell cap of 12", "a verdict on a metric, not on the features"):
            self.assertIn(needle, self.md)
        for banned in ("confirmed", "proved", "demonstrated"):
            self.assertNotIn(banned, self.md)

    def test_the_figures_are_linked_and_the_verdict_is_in_the_report(self):
        self.assertIn("(f5_fig1_w1_distribution_light.png)", self.md); self.assertIn(self.res["validity"]["overall"], self.md)
        self.assertIn(f"{self.res['floor']['tau']:.4f}", self.md)

    def test_an_inadequate_instrument_puts_its_note_and_the_undefined_verdicts_into_the_report(self):
        sa, st, sm, T = build_world("inadequate"); res = an.analyse_f5(sa, st, sm, meta=None, n_boot=30, seed=0); md = an.report_text(res)
        self.assertIn("**Note:**", md); self.assertIn("instrument is inadequate", md); self.assertEqual(md.count("| undefined |"), 3); self.assertIn("**Overall: undetermined.**", md)
        self.assertEqual([l.split(".")[0] for l in md.split("\n") if l.startswith("## ")], [f"## {i}" for i in range(1, 8)])

    def test_the_statement_table_is_one_contiguous_markdown_table_including_the_descriptive_row(self):
        lines = self.md.split("\n"); i = next(k for k, l in enumerate(lines) if l.startswith("| statement |")); j = next(k for k, l in enumerate(lines) if l.startswith("| descriptive S2c"))
        self.assertTrue(all(l.startswith("|") for l in lines[i:j + 1])); self.assertEqual(j - i, 5)

    def test_the_hypothesis_numbers_are_formatted_for_reading(self):
        self.assertNotRegex(self.md, r"0\.\d{6,}")                                                                    # no 16-digit floats in the report

    def test_write_outputs_writes_the_report_too(self):
        with tempfile.TemporaryDirectory() as d:
            an.write_outputs(self.res, d); self.assertTrue((Path(d) / "report.md").read_text().startswith("# F5 report"))


if __name__ == "__main__":
    unittest.main()
