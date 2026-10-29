"""Tests for scripts/g0/survival_sidecar.py (D-117 Stage 2, design note reports/survival_measurement/stage2_design.md): the SurvivalSink streams 21 float64/int64
statistics per (condition, language, feature) over the equal-budget stream, on top of the unchanged phase1_run.run_survival(profile_sink=...). Hand-computable toy
dictionary first, then a plain-Python reference on random data, then the sink inside run_survival on the toy harness of test_phase1_run, then the gates G1 to G4, G6
and the saved files. CPU only; no sae_lens, no model."""
import contextlib, copy, io, json, shutil, sys, tempfile, unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_lib as p1   # noqa: E402
import survival_sidecar as sc   # noqa: E402
from test_phase1_run import Fake, F, BOS, PAD, V, D, corpus, call   # noqa: E402

COUNTS = ("n_both", "n_a", "n_b", "n_on_off", "n_off_on", "n_exceed2", "n_exceed1")
THETA = torch.tensor([0.5, 1.0, 0.2], dtype=torch.float32)


def sae_acts(pre, theta):
    return pre * (pre > theta)                                     # the JumpReLU arithmetic in float32 (acts == pre * (pre > theta))


class Feeder:
    """Hands batches to a sink the way run_survival does: the condition's features, plus the pre-activations through the sink's pre_source."""
    def __init__(self, n_lang, theta, **kw):
        self.cur = {"pre": None}
        self.theta = theta
        self.sink = sc.SurvivalSink(n_lang, theta.numel(), theta, lambda: self.cur["pre"], **kw)

    def condition(self, label, items):
        """items: list of (batch, pre [B, L, d]); the features are formed with the SAE arithmetic from the pre-activations."""
        self.sink.start_condition(label)
        for batch, pre in items:
            flat = pre.reshape(-1, pre.shape[-1]).float()
            self.cur["pre"] = flat
            self.sink.update(batch, sae_acts(flat, self.theta).reshape(pre.shape))
        self.sink.finish_condition()


def toy_batch(lang, valid, eq, tokens=None):
    valid, eq = torch.tensor(valid, dtype=torch.bool), torch.tensor(eq, dtype=torch.bool)
    tokens = torch.arange(valid.numel()).reshape(valid.shape) + 5 if tokens is None else tokens
    return {"lang": lang, "tokens": tokens, "valid": valid, "eq": eq, "sent": torch.arange(valid.shape[0]), "n_truncated": 0}


# the hand-computable case: one sentence, position 0 is BOS, position 3 is valid but not in the equal-budget stream; positions 0 and 3 carry wild values that must not count
PRE_FP = torch.tensor([[[50., 50., 50.], [1.0, 0.5, 0.3], [0.2, 2.0, 0.1], [3.0, 3.0, 3.0], [0.8, 1.5, 0.6]]], dtype=torch.float32)
PRE_Q = torch.tensor([[[-9., -9., -9.], [0.9, 0.7, 0.1], [0.6, 1.8, 0.4], [0.0, 0.0, 0.0], [0.3, 1.4, 0.7]]], dtype=torch.float32)
BATCH0 = toy_batch(0, [[0, 1, 1, 1, 1]], [[0, 1, 1, 0, 1]])


def reference(pre_fp, pre_q, theta, rows):
    """Plain-Python loops over the selected tokens, doubles of the float32 inputs. -> {stat: [d] list}."""
    d = len(theta); out = {k: [0.0] * d for k in sc.STAT_NAMES}
    for r in rows:
        for j in range(d):
            th, pf, pq = float(theta[j]), float(pre_fp[r][j]), float(pre_q[r][j])
            a = pf if pf > th else 0.0
            b = pq if pq > th else 0.0
            out["Sa"][j] += a; out["Sb"][j] += b; out["Saa"][j] += a * a; out["Sbb"][j] += b * b; out["Sab"][j] += a * b
            if a > 0 and b > 0:
                out["n_both"][j] += 1; out["Sa_both"][j] += a; out["Sb_both"][j] += b; out["Saa_both"][j] += a * a; out["Sbb_both"][j] += b * b; out["Sab_both"][j] += a * b
            out["n_a"][j] += a > 0; out["n_b"][j] += b > 0
            out["n_on_off"][j] += (a > 0) and not (b > 0); out["n_off_on"][j] += (not a > 0) and (b > 0)
            if a > 0:
                dpre, dd = pq - pf, pf - th
                out["S_dpre"][j] += dpre; out["S_dpre2"][j] += dpre * dpre; out["S_d"][j] += dd; out["S_d2"][j] += dd * dd
                out["n_exceed2"][j] += abs(dpre) > dd; out["n_exceed1"][j] += not (pq > th)
    return out


class HandComputed(unittest.TestCase):
    def sums(self):
        f = Feeder(2, THETA)
        f.condition("FP", [(BATCH0, PRE_FP)]); f.condition("INT6", [(BATCH0, PRE_Q)])
        return f.sink.sums["INT6"]

    def test_the_statistics_are_the_agreed_21_with_the_agreed_dtypes(self):
        self.assertEqual(len(sc.STAT_NAMES), 21); self.assertEqual(len(set(sc.STAT_NAMES)), 21)
        s = self.sums()
        self.assertEqual(set(s["stats"]), set(sc.STAT_NAMES))
        for k, v in s["stats"].items():
            self.assertEqual(tuple(v.shape), (2, 3)); self.assertEqual(v.dtype, torch.int64 if k in COUNTS else torch.float64, k)
        self.assertEqual(s["n"].dtype, torch.int64)

    def test_pinned_values_computed_by_hand(self):
        s = self.sums(); st = s["stats"]
        self.assertEqual(s["n"].tolist(), [3, 0])                       # tokens 1, 2 and 4 of language 0; BOS and the unselected token do not count
        j0 = lambda k: float(st[k][0, 0]); j2 = lambda k: float(st[k][0, 2])
        self.assertAlmostEqual(j0("Sa"), 1.8, places=6); self.assertAlmostEqual(j0("Sb"), 1.5, places=6)   # a = (1.0, 0, 0.8), b = (0.9, 0.6, 0)
        self.assertEqual(int(st["n_both"][0, 0]), 1); self.assertEqual(int(st["n_on_off"][0, 0]), 1); self.assertEqual(int(st["n_off_on"][0, 0]), 1)
        self.assertAlmostEqual(j0("S_dpre"), -0.6, places=6)           # FP-active tokens of feature 0: -0.1 and -0.5
        self.assertAlmostEqual(j0("S_d"), 0.8, places=6)               # margins 0.5 and 0.3
        self.assertEqual(int(st["n_exceed2"][0, 0]), 1); self.assertEqual(int(st["n_exceed1"][0, 0]), 1)     # token 4: |dpre| 0.5 > margin 0.3, and it is switched off
        self.assertEqual(int(st["n_exceed2"][0, 2]), 1); self.assertEqual(int(st["n_exceed1"][0, 2]), 1)     # feature 2, token 1: |dpre| 0.2 > margin 0.1
        self.assertAlmostEqual(j2("S_dpre"), -0.1, places=6)
        self.assertEqual(int(st["n_off_on"][0, 2]), 1)                  # token 2 of feature 2: a = 0, b = 0.4

    def test_every_statistic_equals_a_plain_python_reference(self):
        s = self.sums(); ref = reference(PRE_FP[0], PRE_Q[0], THETA, rows=[1, 2, 4])
        for k in sc.STAT_NAMES:
            self.assertTrue(np.allclose(s["stats"][k][0].double().numpy(), np.array(ref[k]), rtol=1e-12, atol=1e-12), k)
            self.assertEqual(float(s["stats"][k][1].abs().sum()), 0.0, k)         # the language without a token has all-zero sums

    def test_positions_outside_the_equal_budget_stream_are_ignored(self):
        a = self.sums()
        f = Feeder(2, THETA)
        wild_fp, wild_q = PRE_FP.clone(), PRE_Q.clone()
        wild_fp[0, [0, 3]] = 77.0; wild_q[0, [0, 3]] = -5.0
        f.condition("FP", [(BATCH0, wild_fp)]); f.condition("INT6", [(BATCH0, wild_q)])
        for k in sc.STAT_NAMES:
            self.assertTrue(torch.equal(a["stats"][k], f.sink.sums["INT6"]["stats"][k]), k)

    def test_a_tie_at_the_threshold_is_off_exactly_as_in_the_sae(self):
        fp = torch.tensor([[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]]]); q = torch.tensor([[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]])   # quantized pre == theta_0 == 0.5 exactly
        batch = toy_batch(0, [[0, 1]], [[0, 1]]); f = Feeder(2, THETA)
        f.condition("FP", [(batch, fp)]); f.condition("INT4", [(batch, q)])
        st = f.sink.sums["INT4"]["stats"]
        self.assertEqual(int(st["n_on_off"][0, 0]), 1); self.assertEqual(int(st["n_exceed1"][0, 0]), 1)       # off by the SAE comparison (pre > theta)
        self.assertEqual(int(st["n_exceed2"][0, 0]), 0)                                                         # |dpre| == margin exactly: not strictly greater
        sc.gate_g4(f.sink.sums["INT4"])

    def test_the_correlation_rebuilt_from_the_sums_equals_langpearson(self):
        g = torch.Generator().manual_seed(3); items_fp, items_q = [], []
        for lang, L in ((0, 7), (1, 5), (0, 6)):
            pre = torch.randn(2, L, 3, generator=g); preq = pre + 0.3 * torch.randn(2, L, 3, generator=g)
            batch = toy_batch(lang, torch.ones(2, L, dtype=torch.bool).tolist(), torch.ones(2, L, dtype=torch.bool).tolist())
            items_fp.append((batch, pre)); items_q.append((batch, preq))
        f = Feeder(2, THETA); f.condition("FP", items_fp); f.condition("INT4", items_q)
        lp = p1.LangPearson(2, 3, "cpu")
        for (batch, pre), (_, preq) in zip(items_fp, items_q):
            x = sae_acts(pre.reshape(-1, 3), THETA); y = sae_acts(preq.reshape(-1, 3), THETA)
            lp.update(batch["lang"], x, y)
        s = f.sink.sums["INT4"]; corr = sc.corr_from_sums(s)
        for lang in (0, 1):
            self.assertTrue(torch.allclose(corr[lang], lp.result(lang)["corr"], atol=1e-8, rtol=0)); self.assertEqual(int(s["n"][lang]), lp.result(lang)["n"])


class RandomData(unittest.TestCase):
    """Several batches, three languages (one never has a selected token), against the plain-Python reference and against one single batch."""
    def build(self):
        g = torch.Generator().manual_seed(11); d = 5; theta = torch.rand(d, generator=g) * 0.6 + 0.1
        specs = [(0, 4, 6), (0, 3, 5), (1, 2, 7), (2, 3, 4)]                      # (language, sentences, length); language 2 gets no selected token
        fp, q, batches = [], [], []
        for lang, B, L in specs:
            valid = torch.ones(B, L, dtype=torch.bool); valid[:, 0] = False        # position 0 plays BOS
            for r in range(B):
                valid[r, L - 1 - r % 2:] = valid[r, L - 1 - r % 2:] & (torch.rand(1, generator=g) > 0.5)   # ragged right edge
            eq = valid & (torch.rand(B, L, generator=g) > (0.4 if lang < 2 else 2.0))
            tokens = torch.randint(3, V, (B, L), generator=g)
            batches.append({"lang": lang, "tokens": tokens, "valid": valid, "eq": eq, "sent": torch.arange(B), "n_truncated": 0})
            pre = torch.randn(B, L, d, generator=g); fp.append(pre); q.append(pre + 0.25 * torch.randn(B, L, d, generator=g))
        return d, theta, batches, fp, q

    def test_against_the_plain_python_reference_per_language(self):
        d, theta, batches, fp, q = self.build(); f = Feeder(3, theta)
        f.condition("FP", list(zip(batches, fp))); f.condition("INT4", list(zip(batches, q)))
        s = f.sink.sums["INT4"]
        for lang in range(3):
            rows_fp, rows_q = [], []
            for b, pf, pq in zip(batches, fp, q):
                if b["lang"] != lang:
                    continue
                m = b["eq"].reshape(-1); rows_fp.append(pf.reshape(-1, d)[m]); rows_q.append(pq.reshape(-1, d)[m])
            if not rows_fp or sum(len(x) for x in rows_fp) == 0:
                self.assertEqual(int(s["n"][lang]), 0); self.assertEqual(float(sum(float(v[lang].abs().sum()) for v in s["stats"].values())), 0.0); continue
            pf, pq = torch.cat(rows_fp), torch.cat(rows_q); ref = reference(pf, pq, theta, range(len(pf)))
            self.assertEqual(int(s["n"][lang]), len(pf))
            for k in sc.STAT_NAMES:
                self.assertTrue(np.allclose(s["stats"][k][lang].double().numpy(), np.array(ref[k]), rtol=1e-10, atol=1e-10), (lang, k))

    def test_accumulating_over_batches_equals_one_batch_per_language(self):
        d, theta, batches, fp, q = self.build(); f = Feeder(3, theta)
        f.condition("FP", list(zip(batches, fp))); f.condition("INT4", list(zip(batches, q)))
        merged_b, merged_fp, merged_q = [], [], []
        for lang in (0, 1, 2):
            idx = [i for i, b in enumerate(batches) if b["lang"] == lang]; L = max(batches[i]["valid"].shape[1] for i in idx)
            pad = lambda x, v: torch.nn.functional.pad(x, (0, 0, 0, L - x.shape[1]), value=v) if x.dim() == 3 else torch.nn.functional.pad(x, (0, L - x.shape[1]), value=v)
            merged_b.append({"lang": lang, "tokens": torch.cat([pad(batches[i]["tokens"], 0) for i in idx]), "valid": torch.cat([pad(batches[i]["valid"], False) for i in idx]),
                             "eq": torch.cat([pad(batches[i]["eq"], False) for i in idx]), "sent": torch.arange(sum(batches[i]["valid"].shape[0] for i in idx)), "n_truncated": 0})
            merged_fp.append(torch.cat([pad(fp[i], 0.0) for i in idx])); merged_q.append(torch.cat([pad(q[i], 0.0) for i in idx]))
        g = Feeder(3, theta); g.condition("FP", list(zip(merged_b, merged_fp))); g.condition("INT4", list(zip(merged_b, merged_q)))
        a, b = f.sink.sums["INT4"], g.sink.sums["INT4"]
        self.assertTrue(torch.equal(a["n"], b["n"]))
        for k in sc.STAT_NAMES:
            self.assertTrue(torch.allclose(a["stats"][k].double(), b["stats"][k].double(), rtol=1e-10, atol=1e-10), k)

    def test_chunking_the_rows_does_not_change_the_sums(self):
        d, theta, batches, fp, q = self.build()
        a = Feeder(3, theta); a.condition("FP", list(zip(batches, fp))); a.condition("INT4", list(zip(batches, q)))
        b = Feeder(3, theta, chunk=3); b.condition("FP", list(zip(batches, fp))); b.condition("INT4", list(zip(batches, q)))
        for k in sc.STAT_NAMES:
            self.assertTrue(torch.allclose(a.sink.sums["INT4"]["stats"][k].double(), b.sink.sums["INT4"]["stats"][k].double(), rtol=1e-10, atol=1e-10), k)


class Gates(unittest.TestCase):
    def feeder(self):
        f = Feeder(2, THETA); f.condition("FP", [(BATCH0, PRE_FP)]); f.condition("INT6", [(BATCH0, PRE_Q)]); f.condition("INT0", [(BATCH0, PRE_FP)])
        return f

    def test_g1_int0_equals_fp_and_a_perturbation_fails_it(self):
        f = self.feeder(); sc.gate_int0(f.sink.sums["INT0"])
        with self.assertRaises(sc.SidecarError):
            sc.gate_int0(f.sink.sums["INT6"])
        s = copy.deepcopy(f.sink.sums["INT0"]); s["stats"]["Sab"][0, 0] *= 1.0 + 1e-9
        with self.assertRaises(sc.SidecarError):
            sc.gate_int0(s)

    def test_the_rebuilt_correlation_of_the_identity_is_one_and_never_above_one(self):
        corr = sc.corr_from_sums(self.feeder().sink.sums["INT0"]); active = self.feeder().sink.sums["INT0"]["stats"]["n_a"] > 1
        self.assertLessEqual(float(corr.max()), 1.0); self.assertGreaterEqual(float(corr.min()), -1.0)
        nz = self.feeder().sink.sums["INT0"]["stats"]["Saa"] > 0
        self.assertTrue(bool((corr[nz & active] > 1 - 1e-6).all()))

    def test_every_count_and_perturbation_of_the_identity_is_checked_by_g1_alone(self):
        for k in ("n_on_off", "n_off_on", "n_exceed1", "n_exceed2", "S_dpre", "S_dpre2"):
            s = copy.deepcopy(self.feeder().sink.sums["INT0"]); s["stats"][k][0, 0] += 1
            with self.assertRaises(sc.SidecarError, msg=k):
                sc.gate_int0(s)

    def test_the_rebuilt_correlation_is_clamped_to_one(self):
        z = lambda v: torch.full((1, 1), v, dtype=torch.float64)
        stats = {k: z(0.0) for k in sc.STAT_NAMES}; stats.update(Saa=z(1.0), Sbb=z(1.0), Sab=z(1.5)); stats["n_a"] = stats["n_a"].long()
        self.assertEqual(float(sc.corr_from_sums({"stats": stats, "n": torch.tensor([2])})), 1.0)
        stats["Sab"] = z(-1.5); self.assertEqual(float(sc.corr_from_sums({"stats": stats, "n": torch.tensor([2])})), -1.0)

    def test_g3_accepts_the_sae_arithmetic_and_rejects_anything_else(self):
        flat = PRE_FP.reshape(-1, 3); acts = sae_acts(flat, THETA)
        self.assertEqual(sc.gate_g3(acts, flat, THETA), 0.0)
        with self.assertRaises(sc.SidecarError):
            sc.gate_g3(sae_acts(flat * 1.1, THETA), flat, THETA)                          # acts formed from another pre than the one captured
        with self.assertRaises(sc.SidecarError):
            sc.gate_g3(acts, flat, THETA + 0.15)                                          # another threshold (row 1 of feature 2 is on at 0.2, off at 0.35)

    def test_the_sink_runs_g3_on_its_first_condition_batch_and_stops_a_wrong_pre_source(self):
        f = Feeder(2, THETA)
        f.sink.start_condition("FP"); f.cur["pre"] = (PRE_FP * 1.1).reshape(-1, 3)
        with self.assertRaises(sc.SidecarError):
            f.sink.update(BATCH0, sae_acts(PRE_FP.reshape(-1, 3), THETA).reshape(1, 5, 3))
        f2 = self.feeder(); self.assertEqual(f2.sink.g3_max_abs_diff, 0.0)

    def test_g4_is_exact_and_rejects_a_mismatch_and_a_wrong_firing_count(self):
        f = self.feeder(); s = f.sink.sums["INT6"]; sc.gate_g4(s)
        bad = copy.deepcopy(s); bad["stats"]["n_on_off"][0, 1] += 1
        with self.assertRaises(sc.SidecarError):
            sc.gate_g4(bad)
        fr = torch.zeros(2, 3, dtype=torch.float64); fr[0] = s["stats"]["n_a"][0].double() / s["n"][0]
        sc.gate_g4(s, committed_firing_rate=fr)
        fr[0, 0] += 2.0 / float(s["n"][0])
        with self.assertRaises(sc.SidecarError):
            sc.gate_g4(s, committed_firing_rate=fr)

    def test_g2_internal_tolerance(self):
        f = self.feeder(); s = f.sink.sums["INT6"]; corr = sc.corr_from_sums(s)
        self.assertEqual(sc.gate_g2_internal(s, corr), 0.0)
        with self.assertRaises(sc.SidecarError):
            sc.gate_g2_internal(s, corr + 1e-6)

    def test_g2_committed_is_pre_specified(self):
        corr = torch.zeros(2, 4, dtype=torch.float64); corr[:, 0] = 0.95; corr[:, 1] = 0.4; fr = torch.full((2, 4), 0.01, dtype=torch.float64); fr[:, 3] = 0.0
        ok = sc.gate_g2_committed(corr, corr + 5e-5, fr)
        self.assertTrue(ok["passed"]); self.assertTrue(ok["counts_identical"]); self.assertLessEqual(ok["max_abs_diff_active"], 1e-3)
        moved = corr.clone(); moved[0, 0] = 0.8999                                                   # a feature crosses 0.9 -> the counts differ
        self.assertFalse(sc.gate_g2_committed(moved, corr, fr)["passed"])
        wide = corr + 2e-3                                                                           # larger than 1e-3 on active features
        self.assertFalse(sc.gate_g2_committed(wide, corr, fr)["passed"])
        many = torch.zeros(2, 5000, dtype=torch.float64); many[:, :10] += 3e-4                         # 0.2 % of the cells differ by 3e-4: the 99.9th percentile over all cells exceeds 1e-4
        r = sc.gate_g2_committed(many, torch.zeros_like(many), torch.zeros(2, 5000, dtype=torch.float64))
        self.assertFalse(r["passed"]); self.assertGreater(r["p999_abs_diff_all"], 1e-4); self.assertTrue(r["counts_identical"])      # reported, and it is the only reason to fail
        self.assertIn("per_language_survived", r)

    def test_g2_committed_each_criterion_alone_decides(self):
        base = torch.zeros(2, 10000, dtype=torch.float64); base[:, :5] = 0.95; fr = torch.full((2, 10000), 0.01, dtype=torch.float64)
        self.assertTrue(sc.gate_g2_committed(base, base.clone(), fr)["passed"])
        cross = base.clone(); cross[0, 0] = 0.8999; other = base.clone(); other[0, 0] = 0.9001                 # one active feature crosses 0.9 by 2e-4: only the counts differ
        r = sc.gate_g2_committed(cross, other, fr); self.assertFalse(r["counts_identical"]); self.assertFalse(r["passed"]); self.assertLess(r["max_abs_diff_active"], 1e-3); self.assertLessEqual(r["p999_abs_diff_all"], 1e-4)
        big = base.clone(); big[0, 7] += 5e-3                                                                     # one active feature moves by 5e-3 (below 0.9 on both sides): only the maximum fails
        r = sc.gate_g2_committed(big, base, fr); self.assertTrue(r["counts_identical"]); self.assertFalse(r["passed"]); self.assertLessEqual(r["p999_abs_diff_all"], 1e-4)
        quiet = base.clone(); quiet[0, 8000] += 5e-2                                                              # an INACTIVE cell may move by any amount as long as the 99.9th percentile stays under 1e-4
        frq = fr.clone(); frq[0, 8000] = 0.0; self.assertTrue(sc.gate_g2_committed(quiet, base, frq)["passed"])


class OrderHash(unittest.TestCase):
    def items(self):
        d, theta, batches, fp, q = RandomData().build()
        return theta, batches, fp, q

    def test_the_expected_hash_from_the_batches_equals_the_fp_pass_hash(self):
        theta, batches, fp, q = self.items(); f = Feeder(3, theta)
        f.condition("FP", list(zip(batches, fp))); f.condition("INT4", list(zip(batches, q)))
        self.assertEqual(f.sink.hash["FP"], sc.batches_hash(batches)); self.assertEqual(f.sink.hash["INT4"], f.sink.hash["FP"])

    def test_the_hash_depends_on_tokens_order_language_and_mask(self):
        theta, batches, fp, q = self.items(); h = sc.batches_hash(batches)
        self.assertNotEqual(h, sc.batches_hash(batches[::-1]))
        for edit in ("tokens", "lang", "eq"):
            bb = copy.deepcopy(batches)
            if edit == "tokens":
                r, c = bb[0]["eq"].nonzero()[0].tolist(); bb[0]["tokens"][r, c] += 1
            elif edit == "lang":
                bb[0]["lang"] = 2
            else:
                r, c = bb[0]["eq"].nonzero()[0].tolist(); bb[0]["eq"][r, c] = False
            self.assertNotEqual(h, sc.batches_hash(bb), edit)

    def test_a_condition_fed_in_another_order_or_with_another_token_is_stopped(self):
        theta, batches, fp, q = self.items(); f = Feeder(3, theta); f.condition("FP", list(zip(batches, fp)))
        with self.assertRaises(sc.SidecarError):
            f.condition("INT4", list(zip(batches[::-1], q[::-1])))
        f2 = Feeder(3, theta); f2.condition("FP", list(zip(batches, fp))); bb = copy.deepcopy(batches); r, c = bb[1]["eq"].nonzero()[0].tolist(); bb[1]["tokens"][r, c] += 1
        with self.assertRaises(sc.SidecarError):
            f2.condition("INT4", list(zip(bb, q)))

    def test_a_condition_with_fewer_batches_is_stopped_at_finish(self):
        theta, batches, fp, q = self.items(); f = Feeder(3, theta); f.condition("FP", list(zip(batches, fp)))
        with self.assertRaises(sc.SidecarError):
            f.condition("INT4", list(zip(batches[:-1], q[:-1])))


class Protocol(unittest.TestCase):
    def test_protocol_errors(self):
        f = Feeder(2, THETA)
        with self.assertRaises(sc.SidecarError):                                                    # update outside a condition
            f.sink.update(BATCH0, PRE_FP)
        with self.assertRaises(sc.SidecarError):                                                    # a quantized condition before FP
            f.sink.start_condition("INT6")
        f.condition("FP", [(BATCH0, PRE_FP)])
        with self.assertRaises(sc.SidecarError):                                                    # FP twice
            f.sink.start_condition("FP")
        f.sink.start_condition("INT6")
        with self.assertRaises(sc.SidecarError):                                                    # unfinished condition
            f.sink.start_condition("INT4")

    def test_no_pre_activation_for_the_batch_is_an_error_not_a_stale_value(self):
        f = Feeder(2, THETA); f.sink.start_condition("FP"); f.cur["pre"] = None
        with self.assertRaises(sc.SidecarError):
            f.sink.update(BATCH0, sae_acts(PRE_FP.reshape(-1, 3), THETA).reshape(1, 5, 3))
        f.cur["pre"] = torch.zeros(3, 3)                                                            # wrong number of rows
        with self.assertRaises(sc.SidecarError):
            f.sink.update(BATCH0, sae_acts(PRE_FP.reshape(-1, 3), THETA).reshape(1, 5, 3))


class FakePre(Fake):
    """The toy harness whose encoder also remembers the pre-activation of the last call (the role of PreCapture on the real SAE)."""
    def encode(self, acts):
        pre = acts @ self.We + self.be; self.last_pre = pre
        return (pre > self.thr) * torch.relu(pre)


def toy_keep(data):
    return p1.keep_masks(data, p1.budget_for(data), 0, split_offset=2)


class InsideRunSurvival(unittest.TestCase):
    BITS = (8, 4, 0)

    def run_both(self, batch_size=4):
        data = corpus(); keep = toy_keep(data); n_lang = len(data)
        base = call(Fake(), data, "mask", bits=self.BITS, keep=keep, batch_size=batch_size)
        fk = FakePre(); sink = sc.SurvivalSink(n_lang, F, fk.thr, lambda: fk.last_pre)
        res = call(fk, data, "mask", bits=self.BITS, keep=keep, batch_size=batch_size, profile_sink=sink)
        return data, keep, base, res, sink

    def test_results_with_the_sink_are_identical_to_results_without(self):
        _, _, base, res, _ = self.run_both()
        for key in base:
            for fld in ("corr", "firing_rate", "mean_activation", "max_activation"):
                self.assertTrue(torch.equal(base[key][fld], res[key][fld]), (key, fld)); self.assertTrue(torch.equal(base[key]["eq"][fld], res[key]["eq"][fld]), (key, fld, "eq"))
            self.assertEqual(base[key]["n"], res[key]["n"]); self.assertEqual(base[key]["eq"]["n"], res[key]["eq"]["n"])

    def test_g2_the_correlation_rebuilt_from_the_sums_equals_the_run_survival_eq_correlation(self):
        _, _, _, res, sink = self.run_both()
        for bits in self.BITS:
            s = sink.sums[f"INT{bits}"]
            self.assertLess(sc.gate_g2_internal(s, res[bits]["eq"]["corr"]), 1e-7)       # run_survival stores float32 correlations (_pack)
            self.assertEqual(s["n"].tolist(), res[bits]["eq"]["n"])

    def test_g1_g4_and_the_firing_counts_on_the_run(self):
        _, _, _, res, sink = self.run_both()
        sc.gate_int0(sink.sums["INT0"])
        for bits in (8, 4):
            sc.gate_g4(sink.sums[f"INT{bits}"], committed_firing_rate=res["FP"]["eq"]["firing_rate"].double())
        self.assertGreater(float(sink.sums["INT4"]["stats"]["n_on_off"].sum() + sink.sums["INT4"]["stats"]["n_off_on"].sum()), 0.0)   # the toy rung does flip features

    def test_the_sums_do_not_depend_on_the_batching(self):
        a = self.run_both(batch_size=1)[4]; b = self.run_both(batch_size=64)[4]
        for key in ("INT8", "INT4", "INT0"):
            self.assertTrue(torch.equal(a.sums[key]["n"], b.sums[key]["n"]))
            for k in sc.STAT_NAMES:                                                                  # float32 forward passes of different batch shapes differ at 1e-7; counts are exact
                self.assertTrue(torch.allclose(a.sums[key]["stats"][k].double(), b.sums[key]["stats"][k].double(), rtol=1e-4, atol=1e-4), (key, k))

    def test_the_fp_pass_hash_equals_the_hash_of_make_batches(self):
        data, keep, _, _, sink = self.run_both()
        self.assertEqual(sink.hash["FP"], sc.batches_hash(p1.make_batches(data, BOS, "mask", PAD, 256, 4, None, keep)))

    def test_the_encoder_side_sums_equal_a_brute_force_pass_per_sentence(self):
        data, keep, _, _, sink = self.run_both()
        fk = FakePre(); s = sink.sums["INT4"]; lang = 1
        sums = {k: torch.zeros(F, dtype=torch.float64) for k in ("S_dpre", "S_dpre2", "S_d")}; n_ex1 = torch.zeros(F, dtype=torch.int64)
        for i, sent in enumerate(data[lang]):
            toks = torch.tensor([[BOS] + sent]); fk.restore(); r, _ = fk.forward(toks); a = fk.encode(r[0])[1:]; pf = fk.last_pre[1:]
            fk.quantize(4); r2, _ = fk.forward(toks); fk.encode(r2[0]); pq = fk.last_pre[1:]
            m = keep[lang][i]; act = (a > 0)[m]; dpre = (pq - pf)[m].double() * act; d = (pf - fk.thr)[m].double() * act
            sums["S_dpre"] += dpre.sum(0); sums["S_dpre2"] += (dpre ** 2).sum(0); sums["S_d"] += d.sum(0); n_ex1 += (act & ~(pq[m] > fk.thr)).sum(0)
        for k in sums:
            self.assertTrue(torch.allclose(s["stats"][k][lang], sums[k], rtol=1e-4, atol=1e-4), k)
        self.assertTrue(torch.equal(s["stats"]["n_exceed1"][lang], n_ex1))


class Files(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        data = corpus(); keep = toy_keep(data); fk = FakePre(); self.sink = sc.SurvivalSink(len(data), F, fk.thr, lambda: fk.last_pre)
        call(fk, data, "mask", bits=(8, 4, 0), keep=keep, batch_size=4, profile_sink=self.sink)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write(self, labels=("INT8", "INT4", "INT0")):
        for lab in labels:
            sc.save_sums(self.tmp, lab, self.sink)

    def test_saved_sums_round_trip_and_validate(self):
        self.write(); out = sc.validate_dir(self.tmp, ["INT8", "INT4", "INT0"], n_lang=3, n_feat=F)
        self.assertEqual(out["conditions"], ["INT8", "INT4", "INT0"])
        back = sc.load_sums(self.tmp / "sums_INT4.pt")
        for k in sc.STAT_NAMES:
            self.assertTrue(torch.equal(back["stats"][k], self.sink.sums["INT4"]["stats"][k]))
        self.assertTrue(torch.equal(back["n"], self.sink.sums["INT4"]["n"]))

    def test_the_validator_rejects_a_missing_file_a_missing_statistic_a_wrong_dtype_a_wrong_shape_and_disagreeing_counts(self):
        self.write(("INT8", "INT4"))
        with self.assertRaises(sc.SidecarError):
            sc.validate_dir(self.tmp, ["INT8", "INT4", "INT0"], n_lang=3, n_feat=F)                 # INT0 missing
        self.write(("INT0",))
        d = sc.load_sums(self.tmp / "sums_INT4.pt")
        for edit in ("drop", "dtype", "shape", "n"):
            bad = copy.deepcopy(d)
            if edit == "drop":
                del bad["stats"]["Sab"]
            elif edit == "dtype":
                bad["stats"]["n_a"] = bad["stats"]["n_a"].double()
            elif edit == "shape":
                bad["stats"]["Sa"] = bad["stats"]["Sa"][:, :-1]
            else:
                bad["n"] = bad["n"] + 1
            torch.save(bad, self.tmp / "sums_INT4.pt")
            with self.assertRaises(sc.SidecarError, msg=edit):
                sc.validate_dir(self.tmp, ["INT8", "INT4", "INT0"], n_lang=3, n_feat=F)
            torch.save(d, self.tmp / "sums_INT4.pt")


class FakeSAEHook(torch.nn.Module):
    def __init__(self):
        super().__init__(); g = torch.Generator().manual_seed(5)
        self.W_enc = torch.randn(D, F, generator=g); self.b_enc = torch.randn(F, generator=g); self.b_dec = torch.randn(D, generator=g)
        self.threshold = torch.rand(F, generator=g) * 0.3; self.hook_sae_acts_pre = torch.nn.Identity()
        self.cfg = type("Cfg", (), {"apply_b_dec_to_input": True})()

    def encode(self, x):
        pre = self.hook_sae_acts_pre((x - self.b_dec) @ self.W_enc + self.b_enc)
        return pre * (pre > self.threshold)


class FakeSAENoHook(FakeSAEHook):
    def __init__(self):
        super().__init__(); del self.hook_sae_acts_pre

    def encode(self, x):
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        return pre * (pre > self.threshold)


class PreCaptureTests(unittest.TestCase):
    def test_the_hook_path_returns_the_pre_activation_of_the_last_encode_and_never_a_stale_one(self):
        sae = FakeSAEHook(); cap = sc.PreCapture(sae); enc = cap.wrap(lambda x: sae.encode(x))
        x = torch.randn(6, D); acts = enc(x)
        pre = (x - sae.b_dec) @ sae.W_enc + sae.b_enc
        self.assertTrue(torch.equal(cap.get(), pre)); self.assertTrue(torch.equal(acts, sae_acts(pre, sae.threshold)))
        self.assertEqual(cap.source, "sae.hook_sae_acts_pre"); self.assertEqual(cap.dtype, "torch.float32")
        cap.last = None
        with self.assertRaises(sc.SidecarError):
            cap.get()
        cap.close(); enc2 = lambda x: sae.encode(x); enc2(x)
        with self.assertRaises(sc.SidecarError):
            cap.get()

    def test_an_encode_that_skips_the_hook_leaves_nothing_to_read_instead_of_the_previous_batch(self):
        sae = FakeSAEHook(); cap = sc.PreCapture(sae); x = torch.randn(4, D)
        cap.wrap(lambda x: sae.encode(x))(x); cap.get()
        cap.wrap(lambda x: x)(x)
        with self.assertRaises(sc.SidecarError):
            cap.get()

    def test_the_fallback_computes_x_w_enc_plus_b_enc_with_the_sae_input_processing(self):
        sae = FakeSAENoHook(); cap = sc.PreCapture(sae); enc = cap.wrap(lambda x: sae.encode(x))
        x = torch.randn(6, D); enc(x); pre = (x - sae.b_dec) @ sae.W_enc + sae.b_enc
        self.assertTrue(torch.allclose(cap.get(), pre, atol=1e-6)); self.assertEqual(cap.source, "x@W_enc+b_enc")
        sc.gate_g3(sae.encode(x), cap.get(), sae.threshold)                                          # G3 would catch a wrong fallback


class StubCap:
    """Stands in for PreCapture around the toy encoder (which remembers its own pre-activation)."""
    source, dtype = "toy encoder", "torch.float32"
    def __init__(self, fk): self.fk = fk
    def get(self): return self.fk.last_pre
    def close(self): pass


class DriftingRestore(FakePre):
    """restore() is exact the first time (before the FP pass) and leaves slightly different weights afterwards: INT0 is then not the identity."""
    calls = 0
    def restore(self):
        DriftingRestore.calls += 1
        self.W = self.W0.clone() * (1.0 if DriftingRestore.calls == 1 else 1.02)


class Job(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()); self.data = corpus(); self.names = [f"lang{i}" for i in self.data]

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_job(self, fk=None, bits=(8, 4, 0), extra=(), committed_dir=None):
        fk = fk or FakePre()
        a = sc.parse_args(["--out", str(self.tmp / "out"), "--bits", *map(str, bits), "--flores-root", "unused", "--batch-size", "4", *extra]); a.thresholds_csv = None
        if committed_dir is not None:
            a.committed_run = str(committed_dir)
        return sc.run_job(a, self.names, self.data, BOS, PAD, F, fk.forward, fk.encode, fk.quantize, fk.restore, fk.thr, StubCap(fk), {"device": "cpu", "model": "toy"}, lambda *m: None, 0.0)

    def committed(self, shift=0.0, rate_shift=0.0):
        keep = toy_keep(self.data); res = call(Fake(), self.data, "mask", bits=(8, 4, 0), keep=keep, batch_size=4); eq = {k: v["eq"] for k, v in res.items()}
        corr = {b: eq[b]["corr"].float() + (shift if b == 4 else 0.0) for b in (8, 4, 0)}
        d = self.tmp / "committed"; d.mkdir(); torch.save({"languages": self.names, "bits": [8, 4, 0], "n": eq["FP"]["n"], "corr": corr, "firing_rate": eq["FP"]["firing_rate"].float() + rate_shift}, d / "survival.pt")
        return d

    def test_the_job_writes_validated_files_and_every_gate_passes(self):
        meta = self.run_job()
        self.assertEqual(meta["gate_failures"], []); self.assertEqual(meta["gate_warnings"], [])
        self.assertTrue(all(g["passed"] for g in meta["gates"].values()), meta["gates"])
        self.assertEqual(meta["saved_conditions"], ["INT8", "INT4"])                                  # INT0 is a gate only
        self.assertEqual(sorted(p.name for p in (self.tmp / "out").iterdir()), ["meta.json", "sums_INT4.pt", "sums_INT8.pt"])
        on_disk = json.loads((self.tmp / "out" / "meta.json").read_text())
        self.assertEqual(on_disk["tokens_per_language"], [p1.budget_for(self.data)] * 3); self.assertEqual(on_disk["pre_activation_source"], "toy encoder")
        self.assertEqual(on_disk["stat_names"], list(sc.STAT_NAMES)); self.assertEqual(len(on_disk["theta"]), F); self.assertEqual(len(on_disk["batch_hash"]), 64)
        self.assertEqual(sc.validate_dir(self.tmp / "out", ["INT8", "INT4"], n_lang=3, n_feat=F)["n"], [p1.budget_for(self.data)] * 3)

    def test_an_int0_that_is_not_the_identity_fails_g1_writes_its_files_and_ends_non_zero(self):
        DriftingRestore.calls = 0
        with self.assertRaises(SystemExit) as cm:
            self.run_job(fk=DriftingRestore())
        self.assertIn("G1", str(cm.exception)); meta = json.loads((self.tmp / "out" / "meta.json").read_text())
        self.assertEqual(meta["gate_failures"], ["G1"]); self.assertFalse(meta["gates"]["G1"]["passed"]); self.assertTrue(meta["gates"]["G2_internal"]["passed"])

    def test_the_committed_comparison_passes_when_the_run_matches_and_warns_without_failing_when_it_does_not(self):
        meta = self.run_job(committed_dir=self.committed())
        self.assertTrue(meta["gates"]["G2_committed"]["passed"]); self.assertEqual(meta["gate_warnings"], [])
        shutil.rmtree(self.tmp / "out"); shutil.rmtree(self.tmp / "committed")
        meta = self.run_job(committed_dir=self.committed(shift=0.05))
        self.assertFalse(meta["gates"]["G2_committed"]["passed"]); self.assertEqual(meta["gate_warnings"], ["G2_committed"]); self.assertEqual(meta["gate_failures"], [])

    def test_a_committed_firing_rate_that_disagrees_with_the_token_counts_fails_g4(self):
        with self.assertRaises(SystemExit) as cm:
            self.run_job(committed_dir=self.committed(rate_shift=0.2))                                   # 0.2 of 24 tokens is about 5 tokens: far beyond the one-token rounding allowance
        self.assertIn("G4", str(cm.exception))

    def test_check_committed_on_the_finished_files_agrees_with_the_in_job_comparison(self):
        self.run_job(); ok = sc.check_committed(self.tmp / "out", self.committed())
        self.assertTrue(ok["languages_match"]); self.assertTrue(ok["G2_committed"]["passed"]); self.assertEqual(ok["G4_committed"], {"INT8": True, "INT4": True})
        shutil.rmtree(self.tmp / "committed"); bad = sc.check_committed(self.tmp / "out", self.committed(shift=0.05))
        self.assertFalse(bad["G2_committed"]["passed"]); self.assertTrue(bad["G2_committed"]["per_condition"]["INT8"]["passed"]); self.assertFalse(bad["G2_committed"]["per_condition"]["INT4"]["passed"])
        shutil.rmtree(self.tmp / "committed"); bad = sc.check_committed(self.tmp / "out", self.committed(rate_shift=0.2))
        self.assertEqual(bad["G4_committed"], {"INT8": False, "INT4": False})
        meta = json.loads((self.tmp / "out" / "meta.json").read_text()); meta["languages"] = meta["languages"][::-1]; (self.tmp / "out" / "meta.json").write_text(json.dumps(meta))
        shutil.rmtree(self.tmp / "committed"); self.assertFalse(sc.check_committed(self.tmp / "out", self.committed())["G2_committed"]["passed"])      # another language order is never compared

    def test_check_committed_skips_a_rung_the_committed_file_does_not_have(self):
        self.run_job(); d = self.committed(); c = torch.load(d / "survival.pt", weights_only=True); del c["corr"][4]; torch.save(c, d / "survival.pt")
        r = sc.check_committed(self.tmp / "out", d)
        self.assertEqual(list(r["G2_committed"]["per_condition"]), ["INT8"]); self.assertEqual(r["G4_committed"], {"INT8": True}); self.assertTrue(r["G2_committed"]["passed"])

    def test_a_subset_run_skips_the_committed_comparison(self):
        meta = self.run_job(extra=("--languages", "x"), committed_dir=self.committed(shift=0.5))
        self.assertNotIn("G2_committed", meta["gates"]); self.assertTrue(meta["subset"]); self.assertIsNone(meta["committed_run"])

    def test_a_wrong_expected_hash_fails_g6_in_run_gates(self):
        data = corpus(); keep = toy_keep(data); fk = FakePre(); sink = sc.SurvivalSink(3, F, fk.thr, lambda: fk.last_pre)
        res = call(fk, data, "mask", bits=(8, 0), keep=keep, batch_size=4, profile_sink=sink); res_eq = {k: v["eq"] for k, v in res.items()}
        gates, failures, warnings = sc.run_gates(sink, res_eq, (8, 0), sink.hash["FP"])
        self.assertEqual((failures, warnings), ([], [])); self.assertTrue(gates["G6"]["passed"])
        gates, failures, _ = sc.run_gates(sink, res_eq, (8, 0), "0" * 64)
        self.assertEqual(failures, ["G6"])
        sink.hash["INT8"] = "1" * 64                                                                     # a condition that hashed differently from the FP pass (the sink stops this itself; run_gates re-checks)
        self.assertEqual(sc.run_gates(sink, res_eq, (8, 0), sink.hash["FP"])[1], ["G6"])

    def test_the_bits_must_be_distinct_and_required(self):
        for bad in (["--out", "o", "--flores-root", "x", "--bits", "4", "4"], ["--out", "o", "--flores-root", "x"], ["--out", "o", "--flores-root", "x", "--bits", "-1"]):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                sc.parse_args(bad)


class Thresholds(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()); self.theta = torch.tensor([3.2026248, 2.9936347, 0.5, 1e-3], dtype=torch.float32)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def csv(self, values, header="feature,threshold"):
        f = self.tmp / "t.csv"; f.write_text("# release: x\n# d_sae: 4\n" + header + "\n" + "".join(f"{i},{v:.8g}\n" for i, v in enumerate(values))); return f

    def test_a_matching_table_passes_and_any_difference_or_shape_problem_fails(self):
        self.assertLess(sc.check_thresholds(self.theta, self.csv(self.theta.tolist())), 1e-6)
        for values in ([3.2026248, 2.9936347, 0.5, 2e-3], [3.2026248, 2.9936347, 0.5], [3.2026248, 2.9936347, 0.5, 1e-3, 9.0]):
            with self.assertRaises(sc.SidecarError):
                sc.check_thresholds(self.theta, self.csv(values))
        with self.assertRaises(sc.SidecarError):
            sc.check_thresholds(self.theta, self.csv(self.theta.tolist(), header="feature,thresh"))

    def test_the_committed_table_of_the_project_parses(self):
        root = Path(__file__).resolve().parents[3]; rows = [ln.split(",") for ln in (root / "reports/phase1/sae_thresholds.csv").read_text().splitlines() if ln and not ln.startswith("#")][1:]
        theta = torch.tensor([float(r[1]) for r in rows], dtype=torch.float32)
        self.assertEqual(theta.numel(), 16384); self.assertLess(sc.check_thresholds(theta, root / "reports/phase1/sae_thresholds.csv"), 1e-6)


if __name__ == "__main__":
    unittest.main()
