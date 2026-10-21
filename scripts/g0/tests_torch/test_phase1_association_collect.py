"""D-035 Task 2 tests for scripts/g0/phase1_association_collect.py and the optional `profile_sink` of phase1_run.run_survival (plan: docs/superpowers/plans/
2026-10-21-language-association-retention.md, Task 2). The collector must save each condition's OWN language-response profiles (positive counts z > 0, activation sums, maxima, content
tokens per language) and per-sentence counts and sums for a fixed feature subset, from the encoder outputs of that condition, never from the reference side (the legacy
LangPearson statistics). Fake causal model fixtures of test_phase1_run.py; brute force = sentence by sentence, no padding."""
import json, shutil, sys, tempfile, unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import phase1_association_collect as ac   # noqa: E402
import phase1_lib as p1   # noqa: E402
import phase1_run as run   # noqa: E402
from test_phase1_run import Fake, corpus, brute, call, F, BOS, PAD   # noqa: E402

CODES = ["l0_Latn", "l1_Latn", "l2_Latn"]
SUB = torch.tensor([1, 4, 5, 9, 13, 20])                                              # the feature subset kept per sentence


def conds(bits):
    return ["FP"] + [f"INT{b}" for b in bits]


def expected(data, regime, bits, keep=None):
    """Per language: n, counts[F], sums[F], max[F] over the selected content positions, and per sentence (n, counts[K], sums[K]); bits None = full precision, 0 = restore only."""
    off = 1 if regime == "mask" else 0; out = {}
    for l, sents in data.items():
        n, cnt, sm, mx, rows = 0, torch.zeros(F, dtype=torch.int64), torch.zeros(F, dtype=torch.float64), torch.zeros(F, dtype=torch.float64), []
        for i, s in enumerate(sents):
            fake = Fake(); toks = ([BOS] if regime == "mask" else []) + s
            if bits:
                fake.quantize(bits)
            y = fake.encode(fake.forward(torch.tensor([toks]))[0][0])[off:]
            sel = keep[l][i] if keep is not None else torch.ones(len(s), dtype=torch.bool)
            y = y[sel].double()
            n += len(y); cnt += (y > 0).sum(0); sm += y.sum(0)
            if len(y):
                mx = torch.maximum(mx, y.max(0).values)
            rows.append((len(y), (y[:, SUB] > 0).sum(0), y[:, SUB].sum(0)))
        out[l] = {"n": n, "counts": cnt, "sums": sm, "max": mx, "rows": rows}
    return out


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def collect(self, data=None, regime="mask", bits=(8, 4, 0), keep=None, streams=("eq", "all"), name="out"):
        data = data or corpus()
        sink = ac.ProfileSink(self.tmp / name, CODES, F, conditions=conds(bits), sentence_features=SUB, streams=streams)
        res = call(Fake(), data, regime, bits=bits, keep=keep, profile_sink=sink, batch_size=4)
        return sink, sink.finalize(), res, data


class AggregateProfiles(Base):
    def test_all_token_profiles_of_every_condition_equal_brute_force(self):
        sink, manifest, res, data = self.collect(streams=("all",))
        prof = ac.read_profiles(self.tmp / "out", "all")
        for cond, bits in (("FP", None), ("INT8", 8), ("INT4", 4), ("INT0", 0)):
            ref = expected(data, "mask", bits)
            for i, l in enumerate(sorted(data)):
                c = prof["conditions"][cond]
                self.assertEqual(int(c["n_tokens"][i]), ref[l]["n"]); self.assertTrue(torch.equal(c["positive_counts"][i], ref[l]["counts"]))
                self.assertTrue(torch.allclose(c["activation_sums"][i], ref[l]["sums"], rtol=1e-5, atol=1e-6)); self.assertTrue(torch.allclose(c["activation_max"][i], ref[l]["max"], atol=1e-6))

    def test_quantized_profiles_are_the_quantized_side_not_the_reference_side(self):
        sink, manifest, res, data = self.collect(streams=("all",), bits=(4,))
        prof = ac.read_profiles(self.tmp / "out", "all")["conditions"]
        legacy_counts = (res["FP"]["firing_rate"].double() * torch.tensor(res["FP"]["n"], dtype=torch.float64)[:, None]).round().long()      # the reference-side statistic of the old schema
        self.assertTrue(torch.equal(prof["FP"]["positive_counts"], legacy_counts))                                                           # FP: same quantity
        self.assertFalse(torch.equal(prof["INT4"]["positive_counts"], legacy_counts))                                                         # a rung: genuinely different
        self.assertFalse(torch.allclose(prof["INT4"]["activation_sums"], prof["FP"]["activation_sums"]))

    def test_restore_only_control_equals_full_precision_exactly(self):
        self.collect(streams=("all",), bits=(0,)); c = ac.read_profiles(self.tmp / "out", "all")["conditions"]
        for k in ("n_tokens", "positive_counts", "activation_sums", "activation_max"):
            self.assertTrue(torch.equal(c["INT0"][k], c["FP"][k]), k)

    def test_equal_budget_stream_counts_only_the_selected_positions(self):
        data = corpus(); keep = p1.keep_masks(data, p1.budget_for(data) - 2, 0, split_offset=2)
        sink, manifest, res, data = self.collect(data=data, keep=keep, bits=(8,))
        eq = ac.read_profiles(self.tmp / "out", "eq")["conditions"]["INT8"]; allt = ac.read_profiles(self.tmp / "out", "all")["conditions"]["INT8"]
        ref = expected(data, "mask", 8, keep=keep)
        for i, l in enumerate(sorted(data)):
            self.assertEqual(int(eq["n_tokens"][i]), ref[l]["n"]); self.assertTrue(torch.equal(eq["positive_counts"][i], ref[l]["counts"]))
        self.assertTrue(bool((eq["n_tokens"] < allt["n_tokens"]).any()))                                                                      # non-vacuous: some positions were dropped

    def test_results_do_not_depend_on_batching(self):
        data = corpus(); outs = []
        for name, kw in (("a", {"batch_size": 1}), ("b", {"batch_size": 64}), ("c", {"batch_size": 64, "max_tokens": 24})):
            sink = ac.ProfileSink(self.tmp / name, CODES, F, conditions=conds((8, 4)), sentence_features=SUB, streams=("all",))
            call(Fake(), data, "mask", bits=(8, 4), profile_sink=sink, **kw); sink.finalize(); outs.append(name)
        ref = ac.read_profiles(self.tmp / "a", "all")["conditions"]
        for name in ("b", "c"):
            got = ac.read_profiles(self.tmp / name, "all")["conditions"]
            for cond in ref:
                self.assertTrue(torch.equal(ref[cond]["positive_counts"], got[cond]["positive_counts"])); self.assertTrue(torch.equal(ref[cond]["n_tokens"], got[cond]["n_tokens"]))
                self.assertTrue(torch.allclose(ref[cond]["activation_sums"], got[cond]["activation_sums"], rtol=1e-5, atol=1e-9))
            s_ref, s_got = ac.read_shard(self.tmp / "a", "all", "INT4", CODES[0]), ac.read_shard(self.tmp / name, "all", "INT4", CODES[0])
            self.assertTrue(torch.equal(ac.dense_counts(s_ref), ac.dense_counts(s_got)))


class SentenceShards(Base):
    def test_shards_equal_brute_force_per_sentence_and_conserve_the_aggregates(self):
        data = corpus(); keep = p1.keep_masks(data, p1.budget_for(data) - 1, 0, split_offset=2)
        sink, manifest, res, data = self.collect(data=data, keep=keep, bits=(4,))
        for stream, kp in (("eq", keep), ("all", None)):
            ref = expected(data, "mask", 4, keep=kp); prof = ac.read_profiles(self.tmp / "out", stream)["conditions"]["INT4"]
            for i, l in enumerate(sorted(data)):
                sh = ac.read_shard(self.tmp / "out", stream, "INT4", CODES[i]); cnt, sums = ac.dense_counts(sh), ac.dense_sums(sh)
                self.assertEqual(sh["sentence_ids"].tolist(), list(range(len(data[l]))))
                self.assertEqual(sh["feature_ids"].tolist(), SUB.tolist())
                for s, (n_s, c_s, m_s) in enumerate(ref[l]["rows"]):
                    self.assertEqual(int(sh["n_tokens"][s]), n_s); self.assertTrue(torch.equal(cnt[s].long(), c_s)); self.assertTrue(torch.allclose(sums[s], m_s, rtol=1e-6, atol=1e-7))
                self.assertEqual(int(sh["n_tokens"].sum()), int(prof["n_tokens"][i]))                                                          # conservation: tokens
                self.assertTrue(torch.equal(cnt.long().sum(0), prof["positive_counts"][i][SUB]))                                               # conservation: counts on the subset
                self.assertTrue(torch.allclose(sums.sum(0), prof["activation_sums"][i][SUB], rtol=1e-6, atol=1e-7))

    def test_a_sentence_with_no_selected_token_keeps_its_row_with_zero_tokens(self):
        data = corpus(); keep = p1.keep_masks(data, p1.budget_for(data), 0, split_offset=2)
        keep[0][1] = torch.zeros_like(keep[0][1])                                                                                       # nothing selected in sentence 1 of language 0
        self.collect(data=data, keep=keep, bits=(8,), streams=("eq",))
        sh = ac.read_shard(self.tmp / "out", "eq", "INT8", CODES[0])
        self.assertEqual(sh["sentence_ids"].tolist(), list(range(len(data[0])))); self.assertEqual(int(sh["n_tokens"][1]), 0)
        self.assertEqual(int(ac.dense_counts(sh)[1].sum()), 0)

    def test_rows_are_stored_in_sentence_order_whatever_the_batch_order(self):
        sink, manifest, res, data = self.collect(streams=("all",), bits=(8,))
        for i in range(3):
            self.assertEqual(ac.read_shard(self.tmp / "out", "all", "INT8", CODES[i])["sentence_ids"].tolist(), list(range(len(data[i]))))

    def test_shard_row_pointer_is_a_valid_csr_index(self):
        self.collect(streams=("all",), bits=(8,)); sh = ac.read_shard(self.tmp / "out", "all", "INT8", CODES[0])
        self.assertEqual(int(sh["row_ptr"][0]), 0); self.assertEqual(int(sh["row_ptr"][-1]), len(sh["col"])); self.assertTrue(bool((sh["row_ptr"][1:] >= sh["row_ptr"][:-1]).all()))
        self.assertTrue(bool((sh["counts"] > 0).all())); self.assertEqual(sh["counts"].dtype, torch.int16)


class SinkUnits(Base):
    def batch(self, valid, eq=None, lang=0, sent=(0,)):
        valid = torch.tensor(valid, dtype=torch.bool); return {"lang": lang, "valid": valid, "eq": valid.clone() if eq is None else torch.tensor(eq, dtype=torch.bool), "sent": torch.tensor(sent), "tokens": torch.zeros_like(valid, dtype=torch.long)}

    def test_positive_means_strictly_greater_than_zero_and_invalid_positions_are_ignored(self):
        sink = ac.ProfileSink(self.tmp / "u", CODES, 3, conditions=["FP"], sentence_features=torch.tensor([0, 1, 2]), streams=("all",))
        feats = torch.tensor([[[9.0, 9.0, 9.0],                                                 # position 0: BOS (not valid)
                               [0.0, 1e-30, 0.5],                                               # content: exact zero does not count, a tiny positive does
                               [-0.0, 0.0, 2.0],                                                # content
                               [7.0, 7.0, 7.0]]])                                               # padding (not valid)
        sink.start_condition("FP"); sink.update(self.batch([[False, True, True, False]]), feats); sink.finish_condition()
        sink.finalize(); c = ac.read_profiles(self.tmp / "u", "all")["conditions"]["FP"]
        self.assertEqual(c["positive_counts"][0].tolist(), [0, 1, 2]); self.assertEqual(int(c["n_tokens"][0]), 2)
        self.assertAlmostEqual(float(c["activation_sums"][0][2]), 2.5); self.assertAlmostEqual(float(c["activation_max"][0][2]), 2.0); self.assertEqual(float(c["activation_max"][0][0]), 0.0)

    def test_duplicate_sentence_id_in_a_language_is_rejected(self):
        sink = ac.ProfileSink(self.tmp / "d", CODES, 3, conditions=["FP"], sentence_features=torch.tensor([0]), streams=("all",))
        feats = torch.ones(1, 2, 3); sink.start_condition("FP")
        sink.update(self.batch([[True, True]], sent=(5,)), feats); sink.update(self.batch([[True, True]], sent=(5,)), feats)
        with self.assertRaisesRegex(ac.CollectError, "duplicate sentence"):
            sink.finish_condition()

    def test_batches_of_one_language_must_be_contiguous(self):
        sink = ac.ProfileSink(self.tmp / "c", CODES, 3, conditions=["FP"], sentence_features=torch.tensor([0]), streams=("all",)); sink.start_condition("FP"); feats = torch.ones(1, 1, 3)
        sink.update(self.batch([[True]], lang=0, sent=(0,)), feats); sink.update(self.batch([[True]], lang=1, sent=(0,)), feats)
        with self.assertRaisesRegex(ac.CollectError, "contiguous"):
            sink.update(self.batch([[True]], lang=0, sent=(1,)), feats)

    def test_update_outside_a_condition_and_unknown_condition_are_rejected(self):
        sink = ac.ProfileSink(self.tmp / "o", CODES, 3, conditions=["FP", "INT4"], sentence_features=torch.tensor([0]), streams=("all",))
        with self.assertRaises(ac.CollectError):
            sink.update(self.batch([[True]]), torch.ones(1, 1, 3))
        with self.assertRaises(ac.CollectError):
            sink.start_condition("INT5")
        sink.start_condition("FP")
        with self.assertRaises(ac.CollectError):
            sink.start_condition("INT4")                                                       # the previous condition was not finished

    def test_an_existing_nonempty_output_directory_is_refused(self):
        (self.tmp / "x" / "association").mkdir(parents=True); (self.tmp / "x" / "association" / "old.txt").write_text("old")
        with self.assertRaisesRegex(ac.CollectError, "not empty"):
            ac.ProfileSink(self.tmp / "x", CODES, 3, conditions=["FP"], sentence_features=torch.tensor([0]))

    def test_feature_count_mismatch_between_sink_and_batch_is_rejected(self):
        sink = ac.ProfileSink(self.tmp / "m", CODES, 3, conditions=["FP"], sentence_features=torch.tensor([0]), streams=("all",)); sink.start_condition("FP")
        with self.assertRaises(ac.CollectError):
            sink.update(self.batch([[True]]), torch.ones(1, 1, 5))


class Completeness(Base):
    def test_a_complete_run_validates_and_records_hashes_and_sizes(self):
        sink, manifest, res, data = self.collect(streams=("all",), bits=(8, 0))
        self.assertTrue(manifest["complete"]); m = ac.validate_run(self.tmp / "out", expected_conditions=conds((8, 0)), expected_languages=CODES, expected_sentences={c: len(data[i]) for i, c in enumerate(CODES)})
        self.assertEqual(m["bytes_total"], sum(p.stat().st_size for p in (self.tmp / "out" / "association").rglob("*") if p.is_file() and p.name != "manifest.json"))
        self.assertEqual(m["languages"], CODES); self.assertEqual(m["n_features"], F)

    def test_finalize_without_every_condition_is_marked_incomplete_and_does_not_validate(self):
        data = corpus(); sink = ac.ProfileSink(self.tmp / "inc", CODES, F, conditions=conds((8, 4)), sentence_features=SUB, streams=("all",))
        call(Fake(), data, "mask", bits=(8,), profile_sink=sink)                                                                 # INT4 never ran
        m = sink.finalize(); self.assertFalse(m["complete"]); self.assertIn("INT4", m["missing_conditions"])
        with self.assertRaisesRegex(ac.CollectError, "incomplete"):
            ac.validate_run(self.tmp / "inc", expected_conditions=conds((8, 4)), expected_languages=CODES)

    def test_a_missing_language_or_a_wrong_sentence_count_fails_validation(self):
        sink, manifest, res, data = self.collect(streams=("all",), bits=(8,))
        with self.assertRaisesRegex(ac.CollectError, "language"):
            ac.validate_run(self.tmp / "out", expected_conditions=conds((8,)), expected_languages=CODES + ["l3_Latn"])
        with self.assertRaisesRegex(ac.CollectError, "sentence"):
            ac.validate_run(self.tmp / "out", expected_conditions=conds((8,)), expected_languages=CODES, expected_sentences={CODES[0]: 999, CODES[1]: len(data[1]), CODES[2]: len(data[2])})

    def test_a_modified_shard_fails_its_hash_check(self):
        self.collect(streams=("all",), bits=(8,)); shard = next((self.tmp / "out" / "association" / "all" / "sentences" / "INT8").glob("*.pt"))
        shard.write_bytes(shard.read_bytes() + b"x")
        with self.assertRaisesRegex(ac.CollectError, "hash"):
            ac.validate_run(self.tmp / "out", expected_conditions=conds((8,)), expected_languages=CODES)

    def test_the_sidecar_is_recognised_by_the_inventory_as_complete_profiles(self):
        import phase1_association_inventory as inv
        self.collect(streams=("all",), bits=(8, 4, 16, 12, 0, 6, 5, 7)); p = self.tmp / "out" / "association" / "all" / "profiles.pt"
        r = inv.build_inventory([p], expected_languages=CODES, required_conditions=conds((8, 4, 16, 12, 0, 6, 5, 7)))
        self.assertEqual(r["quantities"]["quantized_language_profiles"]["verdict"], "available"); self.assertEqual(r["n_features"], F)


class DefaultPathUnchanged(Base):
    def test_run_survival_results_are_identical_with_and_without_a_sink(self):
        data = corpus(); plain = call(Fake(), data, "mask", bits=(8, 4, 0))
        sink = ac.ProfileSink(self.tmp / "s", CODES, F, conditions=conds((8, 4, 0)), sentence_features=SUB, streams=("all",)); with_sink = call(Fake(), data, "mask", bits=(8, 4, 0), profile_sink=sink)
        self.assertEqual(plain.keys(), with_sink.keys())
        for k in plain:
            for f in ("corr", "firing_rate", "mean_activation", "max_activation"):
                self.assertTrue(torch.equal(plain[k][f], with_sink[k][f]), (k, f))
            self.assertEqual(plain[k]["n"], with_sink[k]["n"]); self.assertEqual(plain[k]["loss_sum"], with_sink[k]["loss_sum"])

    def test_the_model_is_restored_after_collection(self):
        fake = Fake(); sink = ac.ProfileSink(self.tmp / "r", CODES, F, conditions=conds((8, 4)), sentence_features=SUB, streams=("all",))
        call(fake, corpus(), "mask", bits=(8, 4), profile_sink=sink); self.assertTrue(torch.equal(fake.W, fake.W0))


class SetupHelpers(Base):
    def test_condition_names(self):
        self.assertEqual(ac.condition_name("FP"), "FP"); self.assertEqual(ac.condition_name(6), "INT6"); self.assertEqual(ac.condition_name(0), "INT0")

    def test_the_none_regime_is_refused(self):
        with self.assertRaisesRegex(ac.CollectError, "mask"):
            ac.make_sink(self.tmp / "n", regime="none", languages=CODES, n_feat=F, bits=[8], baseline_dir=self.tmp, equal_budget=True)

    def test_a_baseline_is_required_and_supplies_the_feature_union(self):
        with self.assertRaisesRegex(ac.CollectError, "baseline"):
            ac.make_sink(self.tmp / "b", regime="mask", languages=CODES, n_feat=F, bits=[8], baseline_dir=None, equal_budget=True)
        base = self.tmp / "base"; base.mkdir()
        sel = {"languages": CODES, "selective": torch.tensor([True, False, True, False] + [False] * (F - 4)), "controls": torch.tensor([3, 7]), "peak": torch.zeros(F, dtype=torch.long),
               "nu": torch.zeros(3, F), "supported": torch.ones(F, dtype=torch.bool), "rates": torch.zeros(3, F), "mean_activation": torch.zeros(3, F), "S": torch.ones(F), "S_rank": torch.ones(F)}
        torch.save(sel, base / "selectivity.pt")
        sink = ac.make_sink(self.tmp / "b2", regime="mask", languages=CODES, n_feat=F, bits=[8, 0], baseline_dir=base, equal_budget=False)
        self.assertEqual(sink.sentence_features.tolist(), [0, 2, 3, 7]); self.assertEqual(sink.conditions, ["FP", "INT8", "INT0"]); self.assertEqual(sink.streams, ("all",))

    def test_baseline_languages_must_match_unless_a_subset_is_declared(self):
        base = self.tmp / "base2"; base.mkdir()
        torch.save({"languages": CODES[::-1], "selective": torch.ones(F, dtype=torch.bool), "controls": torch.tensor([], dtype=torch.long)}, base / "selectivity.pt")
        with self.assertRaisesRegex(ac.CollectError, "language"):
            ac.make_sink(self.tmp / "b3", regime="mask", languages=CODES, n_feat=F, bits=[8], baseline_dir=base, equal_budget=True)


# the hand-made model of the Planted test, importable by other test modules
def make_planted():
    """(forward, encode, quantize, restore) of the planted model: see class Planted."""
    state = {"q": False}
    rng = {0: range(3, 10), 1: range(10, 17), 2: range(17, 24)}
    def forward(tokens):
        return torch.nn.functional.one_hot(tokens, 40).float(), torch.zeros(*tokens.shape, 40)
    def encode(acts):
        tok = acts.argmax(-1); out = torch.zeros(acts.shape[0], F); inl = lambda l: torch.tensor([int(x) in rng[l] for x in tok])
        if not state["q"]:
            out[:, 0] = inl(0).float(); out[:, 1] = inl(1).float() * 2; out[:, 2] = inl(2).float() * 3
        else:
            out[:, 0] = (inl(0) | inl(1)).float(); out[:, 2] = inl(0).float() * 3
        return out
    return forward, encode, (lambda bits: state.__setitem__("q", True)), (lambda: state.__setitem__("q", False))



class Planted(Base):
    """A hand-made model where quantization changes WHICH languages a feature responds to (the plan's planted cases), so the saved quantized counts can be written down by hand.
    Language 0 uses tokens 3-9, language 1 tokens 10-16, language 2 tokens 17-23. Full precision: f0 fires only on language 0, f1 only on language 1, f2 only on language 2.
    Quantized: f0 also fires on language 1 (broadening), f1 never fires (inactive), f2 fires on language 0 instead of language 2 (switch)."""
    make = staticmethod(lambda: make_planted())

    def test_saved_quantized_counts_show_broadening_inactivity_and_a_switch(self):
        g = torch.Generator().manual_seed(3)
        data = {l: [torch.randint(3 + 7 * l, 10 + 7 * l, (n,), generator=g).tolist() for n in (4, 6, 5)] for l in range(3)}
        n_tok = [sum(len(s) for s in data[l]) for l in range(3)]
        sink = ac.ProfileSink(self.tmp / "p", CODES, F, conditions=["FP", "INT4"], sentence_features=torch.tensor([0, 1, 2]), streams=("all",))
        fwd, enc, qz, rs = make_planted(); run.run_survival(fwd, enc, qz, rs, data, "mask", BOS, PAD, F, [4], profile_sink=sink); sink.finalize()
        c = ac.read_profiles(self.tmp / "p", "all")["conditions"]
        self.assertEqual(c["FP"]["positive_counts"][:, :3].tolist(), [[n_tok[0], 0, 0], [0, n_tok[1], 0], [0, 0, n_tok[2]]])
        self.assertEqual(c["INT4"]["positive_counts"][:, :3].tolist(), [[n_tok[0], 0, n_tok[0]], [n_tok[1], 0, 0], [0, 0, 0]])
        self.assertAlmostEqual(float(c["INT4"]["activation_sums"][0, 2]), 3.0 * n_tok[0]); self.assertEqual(float(c["INT4"]["activation_max"][2, 2]), 0.0)
        sh = ac.read_shard(self.tmp / "p", "all", "INT4", CODES[1]); self.assertEqual(ac.dense_counts(sh)[:, 0].tolist(), [len(s) for s in data[1]])                                           # f0 now fires on every token of language 1
        self.assertEqual(int(ac.dense_counts(sh)[:, 1].sum()), 0)                                                                 # f1 inactive in every sentence after quantization


class StorageOptions(Base):
    def test_sentence_shards_can_be_limited_to_the_equal_budget_stream(self):
        data = corpus(); keep = p1.keep_masks(data, p1.budget_for(data) - 2, 0, split_offset=2)
        sink = ac.ProfileSink(self.tmp / "e", CODES, F, conditions=conds((8,)), sentence_features=SUB, streams=("eq", "all"), sentence_streams=("eq",))
        call(Fake(), data, "mask", bits=(8,), keep=keep, profile_sink=sink, batch_size=4); m = sink.finalize()
        self.assertTrue(m["complete"]); self.assertTrue((self.tmp / "e" / "association" / "eq" / "sentences").is_dir()); self.assertFalse((self.tmp / "e" / "association" / "all" / "sentences").exists())
        self.assertEqual(sorted(m["shards"]), ["eq"]); self.assertEqual(m["config"]["sentence_streams"], ["eq"])
        ref = expected(data, "mask", 8)                                                                                              # the aggregate profile of the all-token stream is still saved
        allp = ac.read_profiles(self.tmp / "e", "all")["conditions"]["INT8"]
        for i, l in enumerate(sorted(data)):
            self.assertTrue(torch.equal(allp["positive_counts"][i], ref[l]["counts"]))
        ac.validate_run(self.tmp / "e", expected_conditions=conds((8,)), expected_languages=CODES)

    def test_float32_sentence_sums_are_smaller_and_close_while_aggregates_stay_float64(self):
        data = corpus(); sizes = {}
        for name, dt in (("d64", "float64"), ("d32", "float32")):
            sink = ac.ProfileSink(self.tmp / name, CODES, F, conditions=conds((8,)), sentence_features=SUB, streams=("all",), sums_dtype=dt)
            call(Fake(), data, "mask", bits=(8,), profile_sink=sink, batch_size=4); sink.finalize()
            sizes[name] = ac.read_shard(self.tmp / name, "all", "INT8", CODES[0])
        self.assertEqual(sizes["d64"]["sums"].dtype, torch.float64); self.assertEqual(sizes["d32"]["sums"].dtype, torch.float32)
        self.assertTrue(torch.allclose(ac.dense_sums(sizes["d32"]), ac.dense_sums(sizes["d64"]), rtol=1e-6, atol=1e-7)); self.assertTrue(torch.equal(sizes["d32"]["counts"], sizes["d64"]["counts"]))
        self.assertEqual(ac.read_profiles(self.tmp / "d32", "all")["conditions"]["INT8"]["activation_sums"].dtype, torch.float64)

    def test_invalid_storage_options_are_rejected(self):
        with self.assertRaises(ac.CollectError):
            ac.ProfileSink(self.tmp / "i1", CODES, F, conditions=["FP"], sentence_features=SUB, streams=("all",), sentence_streams=("eq",))
        with self.assertRaises(ac.CollectError):
            ac.ProfileSink(self.tmp / "i2", CODES, F, conditions=["FP"], sentence_features=SUB, sums_dtype="float16")


class CliWiring(Base):
    def run_main(self, *argv):
        import contextlib, io
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
            run.main(["--out", str(self.tmp / "o"), *argv])
        self.assertEqual(cm.exception.code, 2); return err.getvalue()

    def test_association_profiles_are_refused_for_the_none_regime(self):
        self.assertIn("defined for the mask regime only", self.run_main("--corpus", "flores", "--regime", "none", "--association-profiles", "--association-baseline", str(self.tmp)))

    def test_association_profiles_require_a_baseline_directory(self):
        self.assertIn("--association-baseline is required", self.run_main("--corpus", "flores", "--regime", "mask", "--association-profiles"))

    def test_association_profiles_are_refused_for_wikitext(self):
        self.assertIn("needs --corpus flores", self.run_main("--corpus", "wikitext", "--regime", "mask", "--association-profiles", "--association-baseline", str(self.tmp)))

    def test_invalid_sentence_stream_option_is_refused(self):
        self.assertIn("must be eq, all or eq,all", self.run_main("--corpus", "flores", "--regime", "mask", "--association-profiles", "--association-baseline", str(self.tmp), "--association-sentence-streams", "bogus"))

    def test_a_baseline_without_the_profiles_flag_is_refused(self):
        self.assertIn("--association-baseline needs --association-profiles", self.run_main("--corpus", "flores", "--regime", "mask", "--association-baseline", str(self.tmp)))


if __name__ == "__main__":
    unittest.main()
