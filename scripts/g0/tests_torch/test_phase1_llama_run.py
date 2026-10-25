import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phase1_func_groups as fg
import phase1_func_run as fr
import phase1_gain_run as gr
import phase1_llama_run as lr
import phase1_run as pr
import their_code as tc

BOS, PAD = 1, 0
N_LANG, PER, COMMON = 3, 10, 10                 # tokens 2..: language l owns PER tokens, then COMMON tokens shared by all languages
OWN = lambda l: list(range(2 + l * PER, 2 + (l + 1) * PER)); SHARED = list(range(2 + N_LANG * PER, 2 + N_LANG * PER + COMMON)); VOCAB = 2 + N_LANG * PER + COMMON


def corpus(n_sent=100, length=20, seed=0):
    g = torch.Generator().manual_seed(seed); data = {}
    for l in range(N_LANG):
        pool = OWN(l) * 4 + SHARED * 8                   # an own token fires about 67 times in 2,000 tokens of its language (the support rule needs 50), a shared token about 133 times in every language
        data[l] = [[pool[int(i)] for i in torch.randint(0, len(pool), (length,), generator=g)] for _ in range(n_sent)]
    return data


def forward_acts(tokens):                                 # the "residual" is the one-hot of the token: the feature of a token fires exactly on that token
    return torch.nn.functional.one_hot(tokens, VOCAB).float(), None


encode = lambda resid: resid


class Setting(unittest.TestCase):
    def test_the_constants_of_the_frozen_runners_are_set_inside_and_restored_after(self):
        before = (pr.MODEL_CANDIDATES, pr.SAE_RELEASE, pr.SAE_ID, pr.LAYER, fr.LAYER, tc.build_namespace)
        with lr.llama_setting("fineweb42"):
            self.assertEqual(pr.MODEL_CANDIDATES, ["meta-llama/Llama-3.2-1B"]); self.assertEqual(pr.SAE_RELEASE, "seonglae/Llama-3.2-1B-sae")
            self.assertTrue(pr.SAE_ID.endswith("_42_fineweb_512")); self.assertEqual((pr.LAYER, fr.LAYER), (11, 11)); self.assertIsNot(tc.build_namespace, before[5])
        self.assertEqual((pr.MODEL_CANDIDATES, pr.SAE_RELEASE, pr.SAE_ID, pr.LAYER, fr.LAYER, tc.build_namespace), before)

    def test_everything_is_restored_after_an_error(self):
        before = (pr.MODEL_CANDIDATES, pr.SAE_RELEASE, pr.SAE_ID, pr.LAYER, fr.LAYER, tc.build_namespace)
        with self.assertRaises(RuntimeError):
            with lr.llama_setting("faithful42"):
                self.assertTrue(pr.SAE_ID.endswith("faithful-llama3.2-1b_512")); raise RuntimeError("boom")
        self.assertEqual((pr.MODEL_CANDIDATES, pr.SAE_RELEASE, pr.SAE_ID, pr.LAYER, fr.LAYER, tc.build_namespace), before)

    def test_an_unknown_sae_key_is_refused(self):
        with self.assertRaises(ValueError):
            with lr.llama_setting("other"):
                pass

    def test_the_namespace_builder_keeps_their_functions_and_replaces_only_the_sae_loader(self):
        seen = {}
        def fake_build(ex, env):
            env["load_sae_compat"] = "gemma loader"; env["quantize_rtn_per_output_channel_tl"] = "their quantizer"; seen["called"] = True
        old = tc.build_namespace; tc.build_namespace = fake_build
        try:
            with lr.llama_setting("fineweb42"):
                env = {"DEVICE": "cpu"}; tc.build_namespace({}, env)
                self.assertTrue(seen["called"]); self.assertEqual(env["quantize_rtn_per_output_channel_tl"], "their quantizer"); self.assertNotEqual(env["load_sae_compat"], "gemma loader"); self.assertTrue(callable(env["load_sae_compat"]))
        finally:
            tc.build_namespace = old

    def test_the_injected_loader_asks_for_the_original_hook_and_gives_the_alias(self):
        import phase1_llama_sae as ls
        got = {}
        old_load = ls.load_sae_compat_llama; ls.load_sae_compat_llama = lambda release, sae_id, device, hook_alias=None, expected_hook=None: got.update(release=release, sae_id=sae_id, device=device, alias=hook_alias, expected=expected_hook) or "sae"
        old = tc.build_namespace; tc.build_namespace = lambda ex, env: None
        try:
            with lr.llama_setting("fineweb42"):
                env = {"DEVICE": "cpu"}; tc.build_namespace({}, env)
                self.assertEqual(env["load_sae_compat"]("seonglae/Llama-3.2-1B-sae", pr.SAE_ID, "cpu"), "sae")
        finally:
            tc.build_namespace = old; ls.load_sae_compat_llama = old_load
        self.assertEqual((got["alias"], got["expected"]), ("blocks.11.hook_resid_post", "blocks.12.hook_resid_pre")); self.assertEqual(got["release"], "seonglae/Llama-3.2-1B-sae")

    def test_the_hook_alias_is_the_same_residual_as_the_sae_hook(self):
        self.assertEqual((lr.HOOK, lr.HOOK_ALIAS, lr.LAYER), ("blocks.12.hook_resid_pre", "blocks.11.hook_resid_post", 11))


class Tagging(unittest.TestCase):
    def run_pipeline(self, **kw):
        return lr.tag_pipeline(forward_acts, encode, corpus(), BOS, PAD, VOCAB, seed=0, n_boot=20, n_perm=10, batch_size=16, **kw)

    def test_language_owned_features_are_selective_with_their_own_language_as_the_peak(self):
        rep, info = self.run_pipeline()
        for l in range(N_LANG):
            own = torch.tensor(OWN(l))
            self.assertTrue(bool(rep["selective"][own].all()), l); self.assertTrue(bool((rep["peak"][own] == l).all()))
        self.assertFalse(bool(rep["selective"][torch.tensor(SHARED)].any())); self.assertTrue(bool(rep["supported"][torch.tensor(SHARED)].all()))

    def test_the_equal_budget_stream_is_used_and_its_budget_is_reported(self):
        rep, info = self.run_pipeline()
        self.assertEqual(info["stream"], "eq"); self.assertEqual(info["budget_tokens_per_language"], 100 * 20); self.assertEqual(info["n_languages"], N_LANG)

    def test_the_frozen_functions_are_called_with_the_mask_regime_the_equal_budget_stream_the_split_offset_and_the_seed(self):
        calls = {}
        class FakeTag:
            eq = "the equal-budget statistics"
        old_run, old_rep, old_keep = pr.run_tagging, pr.selectivity_report, lr.p1.keep_masks
        pr.run_tagging = lambda forward, enc, data, regime, bos, pad, n_feat, **kw: calls.update(regime=regime, keep=kw["keep"], n_feat=n_feat) or (FakeTag(), None, "lang_of")
        pr.selectivity_report = lambda tag, lang_of, n_lang, **kw: calls.update(tag=tag, lang_of=lang_of, n_lang=n_lang, kw=kw) or {"rates": torch.zeros(N_LANG, 2)}
        lr.p1.keep_masks = lambda data, budget, seed, split_offset=0, **kw: calls.update(budget=budget, seed=seed, offset=split_offset) or "keep masks"
        try:
            lr.tag_pipeline(forward_acts, encode, corpus(), BOS, PAD, 7, seed=5, n_boot=11, n_perm=13)
        finally:
            pr.run_tagging, pr.selectivity_report, lr.p1.keep_masks = old_run, old_rep, old_keep
        self.assertEqual((calls["regime"], calls["keep"], calls["n_feat"]), ("mask", "keep masks", 7)); self.assertEqual((calls["seed"], calls["offset"], calls["budget"]), (5, 1, 2000))
        self.assertEqual((calls["tag"], calls["lang_of"], calls["n_lang"]), ("the equal-budget statistics", "lang_of", N_LANG)); self.assertEqual(calls["kw"], {"n_boot": 11, "n_perm": 13, "seed": 5})

    def test_the_special_tokens_never_enter_the_statistics(self):
        rep, _ = self.run_pipeline()
        self.assertTrue(bool((rep["rates"][:, BOS] == 0).all()) if BOS < VOCAB else True)                    # BOS (token 1) is prepended by the batching and excluded as a position

    def test_the_saved_file_has_the_fields_the_frozen_group_builder_reads(self):
        rep, info = self.run_pipeline()
        with tempfile.TemporaryDirectory() as d:
            path = lr.save_selectivity(d, rep, ["aaa", "bbb", "ccc"])
            sel = fg.dev_view(torch.load(path, weights_only=False)); self.assertEqual(list(sel["languages"]), ["aaa", "bbb", "ccc"])
            man = fg.build_manifest(path, Ks=(5, None), Rs=(2, 2))
            self.assertEqual(man["sizes"]["aaa"]["all"], PER); self.assertEqual(man["sizes"]["aaa"]["K5"], 5)
            self.assertEqual(sorted(man["groups"]["bbb"]["all"]), OWN(1))

    def test_a_language_with_a_different_number_of_sentences_is_handled_and_the_names_must_match(self):
        data = corpus(); data[2] = data[2][:40]
        rep, info = lr.tag_pipeline(forward_acts, encode, data, BOS, PAD, VOCAB, seed=0, n_boot=20, n_perm=10, batch_size=16)
        self.assertEqual(info["budget_tokens_per_language"], 40 * 20)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                lr.save_selectivity(d, rep, ["only", "two"])

    def test_the_selectivity_floats_are_stored_as_float32(self):
        rep, _ = self.run_pipeline()
        with tempfile.TemporaryDirectory() as d:
            sel = torch.load(lr.save_selectivity(d, rep, ["a", "b", "c"]), weights_only=False)
            for k in ("rates", "mean_activation", "S"):
                self.assertEqual(sel[k].dtype, torch.float32, k)


class Lm(unittest.TestCase):
    def test_the_gain_wrapper_is_called_with_the_study_g_down_ratios_three_draws_and_the_llama_setting_active(self):
        seen = {}
        def fake(argv):
            seen["argv"] = list(argv); seen["layer"] = (pr.LAYER, fr.LAYER); seen["model"] = list(pr.MODEL_CANDIDATES)
        old = gr.main; gr.main = fake
        try:
            lr.main(["lm", "--out", "o", "--split", "eval"])
        finally:
            gr.main = old
        a = seen["argv"]
        self.assertEqual(a[a.index("--ratios") + 1:a.index("--ratios") + 9], [str(r) for r in lr.STUDY_RATIOS]); self.assertEqual(a[a.index("--n-controls") + 1], "3"); self.assertEqual(a[a.index("--label") + 1], "K64")
        self.assertEqual(a[a.index("--ratios") + 9], "--n-controls")                                           # exactly eight ratios
        self.assertEqual(a[-4:], ["--out", "o", "--split", "eval"]) if "--out" in a[-4:] else None
        self.assertIn("--out", a); self.assertEqual(seen["layer"], (11, 11)); self.assertEqual(seen["model"], ["meta-llama/Llama-3.2-1B"])
        self.assertEqual((pr.LAYER, fr.LAYER), (12, 12))                                                      # restored (the Gemma values)

    def test_the_ratios_are_the_study_g_decision_in_the_order_of_its_preset_name(self):
        self.assertEqual(lr.STUDY_RATIOS, (0.3535533905932738, 0.1767766952966369, 0.08838834764831845, 2.82842712474619, 5.65685424949238, 11.31370849898476, 1.1, 1.2))
        import json
        dec = json.loads((Path(__file__).resolve().parents[3] / "reports/functional_validation/gain_study/calibration/ratios.json").read_text())
        self.assertEqual(list(lr.STUDY_RATIOS), dec["down"] + dec["up"] + dec["fine_up"])                         # the committed decision of D-071

    def test_the_sae_choice_is_passed_to_the_setting(self):
        seen = {}
        old = gr.main; gr.main = lambda argv: seen.update(sae=pr.SAE_ID)
        try:
            lr.main(["lm", "--sae", "faithful42", "--out", "o"])
        finally:
            gr.main = old
        self.assertTrue(seen["sae"].endswith("faithful-llama3.2-1b_512"))

    def test_the_ratios_cannot_be_overridden_from_the_command_line(self):
        old = gr.main; gr.main = lambda argv: None
        try:
            with self.assertRaises(SystemExit):
                lr.main(["lm", "--ratios", "0.5", "--out", "o"])
        finally:
            gr.main = old

    def test_an_unknown_task_is_refused_and_nothing_is_run(self):
        called = []
        old_tag, old_gr = lr.tag_main, gr.main; lr.tag_main = lambda argv, sae: called.append("tag"); gr.main = lambda argv: called.append("lm")
        try:
            with self.assertRaises(SystemExit):
                lr.main(["other"])
        finally:
            lr.tag_main, gr.main = old_tag, old_gr
        self.assertEqual(called, [])


if __name__ == "__main__":
    unittest.main()
