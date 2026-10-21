"""D-035 Task 4 tests for scripts/g0/phase1_association_analysis.py (plan: docs/superpowers/plans/2026-10-21-language-association-retention.md, Task 4; design section 7; resampling after
Koehn 2004 and Berg-Kirkpatrick et al. 2012, register M01 and M02). Sidecars come from the real collector run on the fake causal model of test_phase1_run.py, or from the hand-made
`Planted` model, so the analysis is tested on exactly the files the Ada run will write. Thresholds are lowered (min_positives) because the fixtures are tiny."""
import json, shutil, sys, tempfile, unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import phase1_association_analysis as an   # noqa: E402
import phase1_association_collect as ac   # noqa: E402
import phase1_association_metrics as am   # noqa: E402
import phase1_run as run   # noqa: E402
from test_phase1_run import Fake, F, BOS, PAD   # noqa: E402
from test_phase1_association_collect import make_planted, CODES   # noqa: E402

MINPOS = 3


def corpus_sym(seed=1, lens=((5, 9, 4, 7), (6, 6, 10, 2), (8, 5, 11, 3))):
    g = torch.Generator().manual_seed(seed)
    return {l: [torch.randint(3, 40, (n,), generator=g).tolist() for n in ls] for l, ls in enumerate(lens)}


def write_baseline(d, peak, selective, controls, codes=CODES, n_feat=F, extra=None):
    d = Path(d); d.mkdir(parents=True, exist_ok=True)
    sel = {"languages": list(codes), "peak": torch.as_tensor(peak, dtype=torch.long), "selective": torch.as_tensor(selective, dtype=torch.bool), "controls": torch.as_tensor(controls, dtype=torch.long),
           "supported": torch.ones(n_feat, dtype=torch.bool), "rates": torch.zeros(len(codes), n_feat), "mean_activation": torch.zeros(len(codes), n_feat), "nu": torch.zeros(len(codes), n_feat),
           "S": torch.ones(n_feat), "S_rank": torch.ones(n_feat)}
    torch.save(sel, d / "selectivity.pt")
    torch.save({"languages": list(codes), "bits": [4, 8], "corr": {b: torch.arange(len(codes) * n_feat, dtype=torch.float32).reshape(len(codes), n_feat) / (len(codes) * n_feat) / b for b in (4, 8)}, **(extra or {})}, d / "survival.pt")
    return d


def read_csv(path):
    import csv
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def make_run(self, data=None, bits=(8, 4, 0), name="run", streams=("all",), keep=None):
        data = data or corpus_sym()
        sink = ac.ProfileSink(self.tmp / name, CODES, F, conditions=["FP"] + [f"INT{b}" for b in bits], sentence_features=torch.arange(F), streams=streams)
        run.run_survival(Fake().forward, Fake().encode, Fake().quantize, Fake().restore, data, "mask", BOS, PAD, F, list(bits), profile_sink=sink, keep=keep, batch_size=4); sink.finalize()
        return self.tmp / name

    def baseline_from_fp(self, run_dir, name="base", stream="all", min_pos=MINPOS, swap=False):
        """Frozen cohort chosen from the FP counts: every feature that fires is 'selective' at its FP peak language, silent ones are controls."""
        prof = ac.read_profiles(run_dir, stream); c = prof["conditions"]["FP"]
        rates = c["positive_counts"].double() / c["n_tokens"].double()[:, None]; peak = rates.argmax(0); fires = c["positive_counts"].sum(0) > 0
        ctrl = (~fires).nonzero().squeeze(1)
        return write_baseline(self.tmp / name, peak, fires, ctrl)


class SidecarLoading(Base):
    def test_the_sidecar_loads_with_shapes_and_the_unit_weight_totals_equal_the_saved_aggregates(self):
        d = self.make_run(); side = an.load_sidecar(d, "all")
        self.assertEqual(side.languages, CODES); self.assertEqual(side.conditions, ["FP", "INT8", "INT4", "INT0"]); self.assertEqual(side.n_sentences, 4); self.assertEqual(side.n_features, F)
        prof = ac.read_profiles(d, "all")["conditions"]
        for cond in side.conditions:
            cnt, sm, nt = side.totals(cond, np.ones((side.n_sentences, 1)))
            self.assertTrue(np.array_equal(cnt[0], prof[cond]["positive_counts"].numpy())); self.assertTrue(np.array_equal(nt[0], prof[cond]["n_tokens"].numpy()))
            self.assertTrue(np.allclose(sm[0], prof[cond]["activation_sums"].numpy(), rtol=1e-9, atol=1e-12))

    def test_only_full_precision_and_the_condition_in_use_are_held_in_memory(self):
        d = self.make_run(); side = an.load_sidecar(d, "all"); ones = np.ones((side.n_sentences, 1))
        side.totals("INT4", ones); self.assertEqual(sorted(side.loaded_conditions()), ["FP", "INT4"] if "FP" in side.loaded_conditions() else ["INT4"])
        side.totals("INT8", ones); self.assertNotIn("INT4", side.loaded_conditions()); self.assertIn("INT8", side.loaded_conditions()); self.assertLessEqual(len(side.loaded_conditions()), 2)
        a = side.totals("INT4", ones)[0]; b = side.totals("INT4", ones)[0]; self.assertTrue(np.array_equal(a, b))                  # reloading gives the same totals

    def test_totals_with_integer_weights_equal_explicit_replication_of_sentences(self):
        d = self.make_run(); side = an.load_sidecar(d, "all"); w = np.array([[2.0], [0.0], [1.0], [3.0]])
        cnt, sm, nt = side.totals("INT4", w)
        for i, l in enumerate(CODES):
            sh = ac.read_shard(d, "all", "INT4", l); dense = ac.dense_counts(sh).numpy()
            self.assertTrue(np.array_equal(cnt[0][i], (dense * w).sum(0))); self.assertEqual(int(nt[0][i]), int((sh["n_tokens"].numpy() * w[:, 0]).sum()))

    def test_languages_with_different_sentence_ids_are_rejected(self):
        d = self.make_run(data=corpus_sym(lens=((5, 9, 4, 7), (6, 6, 10), (8, 5, 11, 3))))
        with self.assertRaisesRegex(an.AnalysisError, "sentence"):
            an.load_sidecar(d, "all")

    def test_a_sidecar_whose_aggregates_disagree_with_its_shards_is_rejected(self):
        d = self.make_run(); p = d / "association" / "all" / "profiles.pt"; prof = torch.load(p, weights_only=True)
        prof["conditions"]["INT4"]["positive_counts"][0, 0] += 1; torch.save(prof, p)
        m = json.loads((d / "association" / "manifest.json").read_text()); import hashlib; m["profiles"]["all"]["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
        (d / "association" / "manifest.json").write_text(json.dumps(m))
        with self.assertRaisesRegex(an.AnalysisError, "aggregate"):
            an.load_sidecar(d, "all")

    def test_conditions_that_saw_different_token_counts_are_rejected(self):
        d = self.make_run(); import hashlib
        shard = d / "association" / "all" / "sentences" / "INT4" / f"{CODES[0]}.pt"; sh = torch.load(shard, weights_only=True); sh["n_tokens"][0] += 1; torch.save(sh, shard)
        m = json.loads((d / "association" / "manifest.json").read_text()); m["shards"]["all"]["INT4"][CODES[0]]["sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
        (d / "association" / "manifest.json").write_text(json.dumps(m))
        with self.assertRaisesRegex(an.AnalysisError, "token counts"):
            an.load_sidecar(d, "all")

    def test_a_draw_that_selects_no_token_is_rejected(self):
        d = self.make_run(); side = an.load_sidecar(d, "all"); coh = an.load_cohort(self.baseline_from_fp(d), side)
        with self.assertRaisesRegex(an.AnalysisError, "no token"):
            an.statistics_for_weights(side, coh, np.zeros((side.n_sentences, 1)), min_positives=MINPOS)

    def test_an_incomplete_sidecar_is_rejected(self):
        data = corpus_sym(); sink = ac.ProfileSink(self.tmp / "inc", CODES, F, conditions=["FP", "INT8", "INT4"], sentence_features=torch.arange(F), streams=("all",))
        run.run_survival(Fake().forward, Fake().encode, Fake().quantize, Fake().restore, data, "mask", BOS, PAD, F, [8], profile_sink=sink); sink.finalize()
        with self.assertRaisesRegex(ac.CollectError, "incomplete"):
            an.load_sidecar(self.tmp / "inc", "all")


class CohortMapping(Base):
    def test_the_frozen_cohort_is_mapped_by_language_code_not_by_index(self):
        d = self.make_run(); side = an.load_sidecar(d, "all"); base = self.baseline_from_fp(d)
        a = an.load_cohort(base, side)
        sel = torch.load(base / "selectivity.pt", weights_only=True); order = [2, 0, 1]
        rev = write_baseline(self.tmp / "rev", torch.tensor([order.index(int(p)) for p in sel["peak"]]), sel["selective"], sel["controls"], codes=[CODES[i] for i in order])
        b = an.load_cohort(rev, side)
        self.assertTrue(np.array_equal(a.peak, b.peak)); self.assertTrue(np.array_equal(a.selective, b.selective))

    def test_a_baseline_with_other_languages_is_rejected(self):
        d = self.make_run(); side = an.load_sidecar(d, "all"); base = write_baseline(self.tmp / "other", np.zeros(F, dtype=int), np.ones(F, dtype=bool), [], codes=["x", "y", "z"])
        with self.assertRaisesRegex(an.AnalysisError, "language"):
            an.load_cohort(base, side)

    def test_a_subset_run_restricts_the_cohort_to_features_whose_original_language_is_in_the_run(self):
        d = self.make_run(); side = an.load_sidecar(d, "all"); prof = ac.read_profiles(d, "all")["conditions"]["FP"]
        rates = prof["positive_counts"].double() / prof["n_tokens"].double()[:, None]; fires = (prof["positive_counts"].sum(0) > 0).numpy()
        peak_base = rates.argmax(0).numpy().copy(); outside = np.nonzero(fires)[0][::3]; peak_base[outside] = 3                          # these features belong to a language the run does not contain
        base = write_baseline(self.tmp / "sub", peak_base, fires, [], codes=CODES + ["l3_Latn"])
        with self.assertRaisesRegex(an.AnalysisError, "language"):
            an.load_cohort(base, side)
        coh = an.load_cohort(base, side, allow_subset=True)
        self.assertEqual(coh.n_selective, int(fires.sum()) - len(outside)); self.assertFalse(bool(coh.selective[outside].any()))
        self.assertTrue(np.array_equal(coh.peak[coh.selective], rates.argmax(0).numpy()[coh.selective]))                                   # mapped by code to the run's language order

    def test_the_feature_count_must_match(self):
        d = self.make_run(); side = an.load_sidecar(d, "all"); base = write_baseline(self.tmp / "nf", np.zeros(F + 1, dtype=int), np.ones(F + 1, dtype=bool), [], n_feat=F + 1)
        with self.assertRaisesRegex(an.AnalysisError, "feature count"):
            an.load_cohort(base, side)


class PointEstimates(Base):
    def test_restore_only_control_retains_every_established_association_and_has_zero_distance(self):
        d = self.make_run(); side = an.load_sidecar(d, "all"); coh = an.load_cohort(self.baseline_from_fp(d), side)
        r = an.point_estimate(side, coh, min_positives=MINPOS)["INT0"]
        self.assertEqual(r["retained_of_established"], r["fp_established"]); self.assertEqual(r["conditional_fraction"], 1.0); self.assertEqual(r["median_tv"], 0.0)
        self.assertAlmostEqual(r["retention_fraction"], r["fp_established"] / r["n_selective"]); self.assertAlmostEqual(r["fp_establishment_fraction"], r["fp_established"] / r["n_selective"])
        self.assertGreater(r["fp_established"], 0)                                                                                    # non-vacuous
        self.assertEqual(r["n_selective"], int(coh.selective.sum()))
        self.assertEqual(sum(r["status_counts"].values()), r["fp_established"]); self.assertEqual(sum(r["fp_status_counts"].values()), r["n_selective"])        # fixed denominators
        self.assertEqual(r["lost_of_established"], 0); self.assertEqual(r["loss_fraction"], 0.0)
        self.assertLess(r["fp_established"], r["n_selective"])                                                                                                  # some selective features are not established in FP

    def test_planted_broadening_inactivity_and_switch_are_counted_with_fixed_denominators(self):
        g = torch.Generator().manual_seed(3)
        data = {l: [torch.randint(3 + 7 * l, 10 + 7 * l, (n,), generator=g).tolist() for n in (4, 6, 5)] for l in range(3)}
        sink = ac.ProfileSink(self.tmp / "pl", CODES, F, conditions=["FP", "INT4"], sentence_features=torch.arange(F), streams=("all",))
        fwd, enc, qz, rs = make_planted(); run.run_survival(fwd, enc, qz, rs, data, "mask", BOS, PAD, F, [4], profile_sink=sink); sink.finalize()
        side = an.load_sidecar(self.tmp / "pl", "all")
        sel = np.zeros(F, dtype=bool); sel[:3] = True; peak = np.zeros(F, dtype=int); peak[:3] = [0, 1, 2]
        base = write_baseline(self.tmp / "plb", peak, sel, [3, 4]); coh = an.load_cohort(base, side)
        r = an.point_estimate(side, coh, min_positives=5)["INT4"]
        self.assertEqual((r["n_selective"], r["fp_established"], r["retained_of_established"]), (3, 3, 0))
        self.assertEqual(r["status_counts"], {"retained": 0, "ambiguous_peak": 1, "different_peak": 1, "low_support": 0, "inactive": 1})      # f0 broadened (tie), f2 switched, f1 inactive
        self.assertEqual(sum(r["status_counts"].values()), r["fp_established"])
        self.assertEqual(r["retention_fraction"], 0.0); self.assertEqual(r["fp_establishment_fraction"], 1.0)
        self.assertEqual(r["lost_of_established"], 3); self.assertEqual(r["loss_fraction"], 1.0)                                  # established in FP and not retained, over the fixed denominator
        self.assertEqual(r["by_language"]["l0_Latn"], {"n_selective": 1, "fp_established": 1, "retained": 0, "retention_fraction": 0.0})
        self.assertAlmostEqual(r["median_delta_share"], -0.75)       # f0: 1.0 -> 0.5; f2: 1.0 -> 0; f1 has no share after quantization (all-zero profile) and is left out
        self.assertAlmostEqual(r["median_delta_margin"], -1.0)      # margins 1 -> 0 (f0), 1 -> 0 (f1, margin 0 when silent), 1 -> -1 (f2): changes -1, -1, -2

    def test_loss_fraction_uses_the_fixed_denominator_including_features_never_established(self):
        g = torch.Generator().manual_seed(3)
        data = {l: [torch.randint(3 + 7 * l, 10 + 7 * l, (n,), generator=g).tolist() for n in (4, 6, 5)] for l in range(3)}
        sink = ac.ProfileSink(self.tmp / "fd", CODES, F, conditions=["FP", "INT4"], sentence_features=torch.arange(F), streams=("all",))
        fwd, enc, qz, rs = make_planted(); run.run_survival(fwd, enc, qz, rs, data, "mask", BOS, PAD, F, [4], profile_sink=sink); sink.finalize()
        side = an.load_sidecar(self.tmp / "fd", "all"); sel = np.zeros(F, dtype=bool); sel[[0, 1, 2, 5]] = True; peak = np.zeros(F, dtype=int); peak[:3] = [0, 1, 2]       # feature 5 never fires
        r = an.point_estimate(side, an.load_cohort(write_baseline(self.tmp / "fdb", peak, sel, []), side), min_positives=5)["INT4"]
        self.assertEqual((r["n_selective"], r["fp_established"], r["lost_of_established"]), (4, 3, 3)); self.assertEqual(r["loss_fraction"], 0.75); self.assertAlmostEqual(r["fp_establishment_fraction"], 0.75)

    def test_languages_without_selective_features_have_no_fraction_not_zero(self):
        d = self.make_run(); side = an.load_sidecar(d, "all"); prof = ac.read_profiles(d, "all")["conditions"]["FP"]
        peak = np.zeros(F, dtype=int); sel = (prof["positive_counts"].sum(0) > 0).numpy()
        coh = an.load_cohort(write_baseline(self.tmp / "z", peak, sel, []), side)                                                      # every selective feature peaks in language 0
        r = an.point_estimate(side, coh, min_positives=MINPOS)["INT8"]["by_language"]
        self.assertEqual(r["l1_Latn"]["n_selective"], 0); self.assertTrue(np.isnan(r["l1_Latn"]["retention_fraction"])); self.assertTrue(np.isnan(r["l2_Latn"]["retention_fraction"]))
        self.assertGreater(r["l0_Latn"]["n_selective"], 0)

    def test_undefined_distances_are_excluded_from_the_median_not_set_to_zero(self):
        g = torch.Generator().manual_seed(3)
        data = {l: [torch.randint(3 + 7 * l, 10 + 7 * l, (n,), generator=g).tolist() for n in (4, 6, 5)] for l in range(3)}
        sink = ac.ProfileSink(self.tmp / "pl2", CODES, F, conditions=["FP", "INT4"], sentence_features=torch.arange(F), streams=("all",))
        fwd, enc, qz, rs = make_planted(); run.run_survival(fwd, enc, qz, rs, data, "mask", BOS, PAD, F, [4], profile_sink=sink); sink.finalize()
        side = an.load_sidecar(self.tmp / "pl2", "all"); sel = np.zeros(F, dtype=bool); sel[:3] = True; peak = np.zeros(F, dtype=int); peak[:3] = [0, 1, 2]
        coh = an.load_cohort(write_baseline(self.tmp / "pl2b", peak, sel, []), side)
        r = an.point_estimate(side, coh, min_positives=5)["INT4"]
        self.assertEqual(r["n_tv_undefined"], 1)                                                                                         # f1 is inactive after quantization: no distance
        self.assertAlmostEqual(r["median_tv"], 0.75 * 0 + float(np.median([0.5, 1.0])), places=6)                                       # median of the two defined distances (f0: 0.5, f2: 1.0), not of three with a 0


class Groups(Base):
    def test_controls_are_summarised_separately_with_their_own_denominator(self):
        d = self.make_run(); side = an.load_sidecar(d, "all"); prof = ac.read_profiles(d, "all")["conditions"]["FP"]
        rates = prof["positive_counts"].double() / prof["n_tokens"].double()[:, None]; peak = rates.argmax(0); fires = (prof["positive_counts"].sum(0) > 0).numpy()
        idx = np.nonzero(fires)[0]; sel = np.zeros(F, dtype=bool); sel[idx[::2]] = True; ctrl = idx[1::2][:max(1, len(idx) // 5)]
        coh = an.load_cohort(write_baseline(self.tmp / "g", peak, sel, ctrl), side)
        a = an.point_estimate(side, coh, min_positives=MINPOS)["INT4"]; b = an.point_estimate(side, coh, min_positives=MINPOS, group="controls")["INT4"]
        self.assertEqual(a["n_selective"], int(sel.sum())); self.assertEqual(b["n_selective"], len(ctrl)); self.assertNotEqual(len(ctrl), int(sel.sum()))
        self.assertEqual(sum(b["fp_status_counts"].values()), len(ctrl))


class Bootstrap(Base):
    def setUp(self):
        super().setUp(); self.d = self.make_run(data=corpus_sym(lens=((5, 9, 4, 7, 6, 8, 3, 9), (6, 6, 10, 2, 7, 5, 8, 4), (8, 5, 11, 3, 6, 9, 4, 5))))
        self.side = an.load_sidecar(self.d, "all"); self.coh = an.load_cohort(self.baseline_from_fp(self.d), self.side)

    def test_the_same_multiplicities_are_used_for_every_language_and_condition(self):
        res = an.bootstrap(self.side, self.coh, n_boot=40, seed=0, min_positives=MINPOS)
        r0 = res["INT0"]
        self.assertTrue(np.array_equal(r0["retained_of_established"], r0["fp_established"]))                                           # INT0 == FP in every draw (paired)
        self.assertTrue(np.all(r0["median_tv"] == 0.0)); self.assertGreater(float(np.std(r0["fp_established"])), 0.0)                  # ... while the baseline itself varies across draws
        self.assertTrue(np.all(r0["loss_fraction"] == 0.0)); self.assertGreater(float(np.std(r0["retention_fraction"])), 0.0)           # the paired loss does not inherit the threshold noise of the baseline

    def test_results_are_reproducible_for_a_seed_and_differ_between_seeds(self):
        a = an.bootstrap(self.side, self.coh, 30, 0, min_positives=MINPOS); b = an.bootstrap(self.side, self.coh, 30, 0, min_positives=MINPOS); c = an.bootstrap(self.side, self.coh, 30, 1, min_positives=MINPOS)
        self.assertTrue(np.array_equal(a["INT4"]["retention_fraction"], b["INT4"]["retention_fraction"], equal_nan=True))
        self.assertFalse(np.array_equal(a["INT4"]["fp_established"], c["INT4"]["fp_established"]))

    def test_draws_do_not_depend_on_the_batch_size(self):
        a = an.bootstrap(self.side, self.coh, 25, 3, min_positives=MINPOS, batch=1); b = an.bootstrap(self.side, self.coh, 25, 3, min_positives=MINPOS, batch=10)
        for k in ("retention_fraction", "fp_established", "median_tv"):
            self.assertTrue(np.array_equal(a["INT4"][k], b["INT4"][k], equal_nan=True), k)

    def test_unit_weights_reproduce_the_point_estimate(self):
        w = np.ones((self.side.n_sentences, 1)); r = an.statistics_for_weights(self.side, self.coh, w, min_positives=MINPOS)["INT4"]; p = an.point_estimate(self.side, self.coh, min_positives=MINPOS)["INT4"]
        self.assertEqual(int(r["retained_of_established"][0]), p["retained_of_established"]); self.assertEqual(int(r["fp_established"][0]), p["fp_established"])

    def test_draws_resample_sentences_with_replacement_so_multiplicities_sum_to_the_sentence_count(self):
        w = an.draw_weights(8, 5, seed=0, start=0)
        self.assertEqual(w.shape, (8, 5)); self.assertTrue(np.all(w.sum(0) == 8)); self.assertTrue(np.all(w >= 0)); self.assertTrue(np.all(w == np.round(w)))
        self.assertFalse(np.array_equal(w[:, 0], w[:, 1]))

    def test_percentile_intervals_ignore_undefined_draws(self):
        x = np.arange(1.0, 101.0); x[:10] = np.nan
        lo, hi = an.percentile_interval(x); self.assertAlmostEqual(lo, float(np.percentile(np.arange(11.0, 101.0), 2.5))); self.assertAlmostEqual(hi, float(np.percentile(np.arange(11.0, 101.0), 97.5)))
        lo2, hi2 = an.percentile_interval(np.full(5, np.nan)); self.assertTrue(np.isnan(lo2) and np.isnan(hi2))


class Outputs(Base):
    def planted(self, name="pl"):
        g = torch.Generator().manual_seed(3)
        data = {l: [torch.randint(3 + 7 * l, 10 + 7 * l, (n,), generator=g).tolist() for n in (4, 6, 5, 3)] for l in range(3)}
        sink = ac.ProfileSink(self.tmp / name, CODES, F, conditions=["FP", "INT4", "INT8"], sentence_features=torch.arange(F), streams=("all",))
        fwd, enc, qz, rs = make_planted(); run.run_survival(fwd, enc, qz, rs, data, "mask", BOS, PAD, F, [4, 8], profile_sink=sink); sink.finalize()
        sel = np.zeros(F, dtype=bool); sel[:3] = True; peak = np.zeros(F, dtype=int); peak[:3] = [0, 1, 2]
        return self.tmp / name, write_baseline(self.tmp / (name + "_base"), peak, sel, [3, 4])

    def analyse(self, out="out", n_boot=20, **kw):
        d, b = self.planted()
        return an.analyse_association(d, b, self.tmp / out, stream="all", n_boot=n_boot, seed=0, min_positives=5, **kw)

    def test_four_outputs_are_written_and_the_json_is_strict(self):
        res = self.analyse(); out = self.tmp / "out"
        for f in ("analysis.json", "feature_status.csv", "language_summary.csv", "association_retention.md"):
            self.assertTrue((out / f).is_file(), f)
        j = json.loads((out / "analysis.json").read_text())                                                  # strict JSON: no NaN literals
        self.assertEqual(j["config"]["n_boot"], 20); self.assertEqual(j["config"]["seed"], 0); self.assertIn("percentile", j["config"]["interval"])
        c = j["groups"]["selective"]["INT4"]; self.assertEqual(c["point"]["n_selective"], 3); lo, hi = c["interval"]["retention_fraction"]; self.assertLessEqual(lo, hi)
        self.assertEqual(c["interval"]["loss_fraction"], [1.0, 1.0]); self.assertEqual(c["diagnostics"]["loss_fraction"], {"bootstrap_mean": 1.0, "point_in_interval": True})
        self.assertEqual(c["point"]["status_counts"]["ambiguous_peak"], 1)
        self.assertEqual(sorted(j["groups"]), ["controls", "selective"]); self.assertEqual(j["provenance"]["languages"], CODES)
        self.assertIsNone(j["groups"]["controls"]["INT4"]["point"]["median_tv"])                       # undefined stays null in the JSON, never 0

    def test_reports_are_reproducible_from_the_same_sidecar_into_fresh_directories(self):
        d, b = self.planted()
        for name in ("o1", "o2"):
            an.analyse_association(d, b, self.tmp / name, stream="all", n_boot=15, seed=0, min_positives=5)
        for f in ("analysis.json", "feature_status.csv", "language_summary.csv", "association_retention.md"):
            self.assertEqual((self.tmp / "o1" / f).read_bytes(), (self.tmp / "o2" / f).read_bytes(), f)

    def test_a_nonempty_output_directory_is_refused(self):
        d, b = self.planted(); (self.tmp / "occupied").mkdir(); (self.tmp / "occupied" / "x").write_text("x")
        with self.assertRaisesRegex(an.AnalysisError, "not empty"):
            an.analyse_association(d, b, self.tmp / "occupied", stream="all", n_boot=5, seed=0, min_positives=5)

    def test_feature_status_has_one_row_per_cohort_feature_and_condition_with_planted_statuses(self):
        import csv
        self.analyse(); rows = read_csv(self.tmp / "out" / "feature_status.csv")
        self.assertEqual(len(rows), 5 * 3)                                                                   # 3 selective + 2 control features x (FP, INT4, INT8)
        by = {(r["feature_id"], r["condition"]): r for r in rows}
        self.assertEqual(by[("0", "INT4")]["status"], "ambiguous_peak"); self.assertEqual(by[("1", "INT4")]["status"], "inactive"); self.assertEqual(by[("2", "INT4")]["status"], "different_peak")
        self.assertEqual(by[("1", "INT4")]["tv_distance"], "")                                               # undefined distance stays blank, not 0
        self.assertEqual(by[("0", "FP")]["status"], "retained"); self.assertEqual(by[("0", "INT4")]["fp_status"], "retained")
        self.assertEqual(by[("0", "INT4")]["original_language"], "l0_Latn"); self.assertEqual(by[("0", "INT4")]["group"], "selective"); self.assertEqual(by[("3", "INT4")]["group"], "controls")
        self.assertEqual([(r["group"], r["feature_id"], r["condition"]) for r in rows[:4]], [("selective", "0", "FP"), ("selective", "0", "INT4"), ("selective", "0", "INT8"), ("selective", "1", "FP")])      # feature-major order, selective before controls
        self.assertEqual([r["group"] for r in rows[-6:]], ["controls"] * 6)
        self.assertEqual(by[("2", "INT4")]["peak_set"], "l0_Latn"); self.assertAlmostEqual(float(by[("0", "INT4")]["target_share"]), 0.5)
        self.assertAlmostEqual(float(by[("0", "INT4")]["tv_distance"]), 0.5)

    def test_the_historical_correlation_is_attached_at_the_original_language_when_available(self):
        import csv
        self.analyse(); rows = {(r["feature_id"], r["condition"]): r for r in read_csv(self.tmp / "out" / "feature_status.csv")}
        n = len(CODES) * F; self.assertAlmostEqual(float(rows[("1", "INT4")]["historical_corr"]), (1 * F + 1) / n / 4, places=5)      # corr[4][language 1, feature 1]
        self.assertEqual(rows[("1", "FP")]["historical_corr"], "")

    def test_the_historical_correlation_follows_the_baseline_language_index_not_the_run_index(self):
        d, _ = self.planted(); order = [2, 0, 1]
        sel = np.zeros(F, dtype=bool); sel[:3] = True; peak_run = np.zeros(F, dtype=int); peak_run[:3] = [0, 1, 2]
        rev = write_baseline(self.tmp / "revb", np.array([order.index(int(p)) for p in peak_run]), sel, [3, 4], codes=[CODES[i] for i in order])
        an.analyse_association(d, rev, self.tmp / "orev", stream="all", n_boot=5, seed=0, min_positives=5)
        rows = {(r["feature_id"], r["condition"]): r for r in read_csv(self.tmp / "orev" / "feature_status.csv")}
        n = len(CODES) * F; self.assertAlmostEqual(float(rows[("1", "INT4")]["historical_corr"]), (order.index(1) * F + 1) / n / 4, places=5)

    def test_language_summary_keeps_fixed_denominators_and_blank_fractions_for_empty_languages(self):
        import csv
        d, b = self.planted(); sel = np.zeros(F, dtype=bool); sel[:2] = True; peak = np.zeros(F, dtype=int); peak[:2] = [0, 1]                  # language 2 has no selective feature
        b2 = write_baseline(self.tmp / "b2", peak, sel, [3]); an.analyse_association(d, b2, self.tmp / "o3", stream="all", n_boot=10, seed=0, min_positives=5)
        rows = [r for r in read_csv(self.tmp / "o3" / "language_summary.csv") if r["group"] == "selective" and r["condition"] == "INT4"]
        by = {r["language"]: r for r in rows}
        self.assertEqual(by["l2_Latn"]["n_selective"], "0"); self.assertEqual(by["l2_Latn"]["retention_fraction"], ""); self.assertEqual((by["l2_Latn"]["retention_lo"], by["l2_Latn"]["retention_hi"]), ("", "")); self.assertEqual(by["l0_Latn"]["n_selective"], "1")
        self.assertEqual((by["l0_Latn"]["lost"], by["l0_Latn"]["loss_fraction"]), ("1", "1")); self.assertEqual((by["POOLED"]["lost"], by["POOLED"]["loss_fraction"]), ("2", "1"))
        self.assertEqual((by["l2_Latn"]["lost"], by["l2_Latn"]["loss_fraction"], by["l2_Latn"]["loss_lo"]), ("0", "", ""))
        self.assertEqual((by["l0_Latn"]["loss_lo"], by["l0_Latn"]["loss_hi"]), ("1", "1")); self.assertEqual((by["POOLED"]["loss_lo"], by["POOLED"]["loss_hi"]), ("1", "1"))
        self.assertEqual(sum(int(by[c]["n_selective"]) for c in CODES), 2); self.assertEqual(by["POOLED"]["n_selective"], "2"); self.assertEqual(by["MACRO_LANGUAGE_MEAN"]["n_languages"], "2")

    def test_the_markdown_report_lists_every_condition_and_states_its_limits(self):
        self.analyse(); md = (self.tmp / "out" / "association_retention.md").read_text()
        for c in ("INT4", "INT8"):
            self.assertIn(c, md)
        self.assertIn("pulled toward the middle", md); self.assertIn("loss", md.lower()); self.assertIn("pointwise", md); self.assertIn("exploratory", md.lower()); self.assertIsNone(__import__("re").search(r"\bnan\b", md.lower())); self.assertIn("fixed denominator", md.lower())

    def test_the_cli_runs_and_refuses_a_missing_baseline(self):
        d, b = self.planted()
        self.assertEqual(an.main(["--run", str(d), "--baseline", str(b), "--out", str(self.tmp / "cli"), "--stream", "all", "--n-boot", "5", "--min-positives", "5"]), 0)
        self.assertTrue((self.tmp / "cli" / "analysis.json").is_file())
        with self.assertRaises(SystemExit):
            an.main(["--run", str(d), "--out", str(self.tmp / "cli2")])


class AggregateOnlyStream(Base):
    """The all-token stream of the full run has aggregate profiles but no sentence shards: its point estimates come from the aggregates, without intervals."""
    def make_both(self):
        data = corpus_sym(); sink = ac.ProfileSink(self.tmp / "agg", CODES, F, conditions=["FP", "INT8", "INT4", "INT0"], sentence_features=torch.arange(F), streams=("eq", "all"), sentence_streams=("eq",))
        keep = __import__("phase1_lib").keep_masks(data, __import__("phase1_lib").budget_for(data) - 2, 0, split_offset=2)
        run.run_survival(Fake().forward, Fake().encode, Fake().quantize, Fake().restore, data, "mask", BOS, PAD, F, [8, 4, 0], profile_sink=sink, keep=keep, batch_size=4); sink.finalize()
        full = ac.ProfileSink(self.tmp / "agg_full", CODES, F, conditions=["FP", "INT8", "INT4", "INT0"], sentence_features=torch.arange(F), streams=("all",))
        run.run_survival(Fake().forward, Fake().encode, Fake().quantize, Fake().restore, data, "mask", BOS, PAD, F, [8, 4, 0], profile_sink=full, batch_size=4); full.finalize()
        return self.tmp / "agg", self.tmp / "agg_full"

    def test_aggregate_sidecar_point_estimates_equal_those_from_sentence_shards(self):
        d, full = self.make_both(); agg = an.load_aggregate_sidecar(d, "all"); shards = an.load_sidecar(full, "all")
        base = self.baseline_from_fp(full); a = an.point_estimate(agg, an.load_cohort(base, agg), min_positives=MINPOS); b = an.point_estimate(shards, an.load_cohort(base, shards), min_positives=MINPOS)
        self.assertEqual(a.keys(), b.keys())
        for cond in a:
            for k in ("n_selective", "fp_established", "retained_of_established", "lost_of_established", "status_counts", "by_language"):
                self.assertEqual(a[cond][k], b[cond][k], (cond, k))
            for k in ("retention_fraction", "median_tv", "median_delta_share", "median_delta_margin"):
                self.assertTrue(np.allclose(a[cond][k], b[cond][k], equal_nan=True), (cond, k))

    def test_analyse_association_handles_a_stream_without_sentence_shards_and_reports_no_intervals(self):
        d, _ = self.make_both(); base = self.baseline_from_fp(d, stream="all")
        out = an.analyse_association(d, base, self.tmp / "oagg", stream="all", n_boot=50, seed=0, min_positives=MINPOS)
        j = json.loads((self.tmp / "oagg" / "analysis.json").read_text())
        self.assertEqual(j["config"]["n_boot"], 0); self.assertTrue(j["config"]["aggregate_only"]); c = j["groups"]["selective"]["INT4"]
        self.assertIsNone(c["interval"]["retention_fraction"][0]); self.assertIsNotNone(c["point"]["retention_fraction"])
        md = (self.tmp / "oagg" / "association_retention.md").read_text(); self.assertIn("no sentence-level shards", md.lower()); self.assertNotIn("paired bootstrap draws", md)
        self.assertTrue((self.tmp / "oagg" / "feature_status.csv").is_file())


if __name__ == "__main__":
    unittest.main()
