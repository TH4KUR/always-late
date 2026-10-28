import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phase1_func_groups as fg
import phase1_func_run as fr
import phase1_gain_run as gr
import phase1_gemma9b_run as g9
import phase1_run as pr
import their_code as tc

BOS, PAD = 1, 0
N_LANG, PER, COMMON = 3, 10, 10
OWN = lambda l: list(range(2 + l * PER, 2 + (l + 1) * PER)); SHARED = list(range(2 + N_LANG * PER, 2 + N_LANG * PER + COMMON)); VOCAB = 2 + N_LANG * PER + COMMON


def corpus(n_sent=100, length=20, seed=0):
    g = torch.Generator().manual_seed(seed); data = {}
    for l in range(N_LANG):
        pool = OWN(l) * 4 + SHARED * 8
        data[l] = [[pool[int(i)] for i in torch.randint(0, len(pool), (length,), generator=g)] for _ in range(n_sent)]
    return data


def forward_acts(tokens):
    return torch.nn.functional.one_hot(tokens, VOCAB).float(), None


encode = lambda resid: resid


class Setting(unittest.TestCase):
    def test_the_constants_of_the_frozen_runners_are_set_inside_and_restored_after(self):
        before = (pr.MODEL_CANDIDATES, pr.SAE_RELEASE, pr.SAE_ID, pr.LAYER, fr.LAYER)
        with g9.gemma9b_setting():
            self.assertEqual(pr.MODEL_CANDIDATES, ["google/gemma-2-9b"]); self.assertEqual(pr.SAE_RELEASE, "gemma-scope-9b-pt-res-canonical")
            self.assertEqual(pr.SAE_ID, "layer_20/width_16k/canonical"); self.assertEqual((pr.LAYER, fr.LAYER), (20, 20))
        self.assertEqual((pr.MODEL_CANDIDATES, pr.SAE_RELEASE, pr.SAE_ID, pr.LAYER, fr.LAYER), before)

    def test_everything_is_restored_after_an_error(self):
        before = (pr.MODEL_CANDIDATES, pr.SAE_RELEASE, pr.SAE_ID, pr.LAYER, fr.LAYER)
        with self.assertRaises(RuntimeError):
            with g9.gemma9b_setting():
                self.assertEqual(pr.LAYER, 20); raise RuntimeError("boom")
        self.assertEqual((pr.MODEL_CANDIDATES, pr.SAE_RELEASE, pr.SAE_ID, pr.LAYER, fr.LAYER), before)

    def test_the_namespace_builder_is_not_replaced(self):
        original = tc.build_namespace
        with g9.gemma9b_setting():
            self.assertIs(tc.build_namespace, original)
        self.assertIs(tc.build_namespace, original)

    def test_the_hook_is_natively_resid_post_with_no_alias(self):
        self.assertEqual(g9.LAYER, 20)
        self.assertFalse(hasattr(g9, "HOOK_ALIAS"))

    def test_the_constants_match_the_9b_model_specification(self):
        self.assertEqual(g9.MODEL, "google/gemma-2-9b")
        self.assertEqual(g9.SAE_RELEASE, "gemma-scope-9b-pt-res-canonical")
        self.assertEqual(g9.SAE_ID, "layer_20/width_16k/canonical")
        self.assertEqual(g9.LAYER, 20)


class Tagging(unittest.TestCase):
    def run_pipeline(self, **kw):
        return g9.tag_pipeline(forward_acts, encode, corpus(), BOS, PAD, VOCAB, seed=0, n_boot=20, n_perm=10, batch_size=16, **kw)

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
        old_run, old_rep, old_keep = pr.run_tagging, pr.selectivity_report, g9.p1.keep_masks
        pr.run_tagging = lambda forward, enc, data, regime, bos, pad, n_feat, **kw: calls.update(regime=regime, keep=kw["keep"], n_feat=n_feat) or (FakeTag(), None, "lang_of")
        pr.selectivity_report = lambda tag, lang_of, n_lang, **kw: calls.update(tag=tag, lang_of=lang_of, n_lang=n_lang, kw=kw) or {"rates": torch.zeros(N_LANG, 2)}
        g9.p1.keep_masks = lambda data, budget, seed, split_offset=0, **kw: calls.update(budget=budget, seed=seed, offset=split_offset) or "keep masks"
        try:
            g9.tag_pipeline(forward_acts, encode, corpus(), BOS, PAD, 7, seed=5, n_boot=11, n_perm=13)
        finally:
            pr.run_tagging, pr.selectivity_report, g9.p1.keep_masks = old_run, old_rep, old_keep
        self.assertEqual((calls["regime"], calls["keep"], calls["n_feat"]), ("mask", "keep masks", 7)); self.assertEqual((calls["seed"], calls["offset"], calls["budget"]), (5, 1, 2000))
        self.assertEqual((calls["tag"], calls["lang_of"], calls["n_lang"]), ("the equal-budget statistics", "lang_of", N_LANG)); self.assertEqual(calls["kw"], {"n_boot": 11, "n_perm": 13, "seed": 5})

    def test_the_special_tokens_never_enter_the_statistics(self):
        rep, _ = self.run_pipeline()
        self.assertTrue(bool((rep["rates"][:, BOS] == 0).all()) if BOS < VOCAB else True)

    def test_the_saved_file_has_the_fields_the_frozen_group_builder_reads(self):
        rep, info = self.run_pipeline()
        with tempfile.TemporaryDirectory() as d:
            path = g9.save_selectivity(d, rep, ["aaa", "bbb", "ccc"])
            sel = fg.dev_view(torch.load(path, weights_only=False)); self.assertEqual(list(sel["languages"]), ["aaa", "bbb", "ccc"])
            man = fg.build_manifest(path, Ks=(5, None), Rs=(2, 2))
            self.assertEqual(man["sizes"]["aaa"]["all"], PER); self.assertEqual(man["sizes"]["aaa"]["K5"], 5)
            self.assertEqual(sorted(man["groups"]["bbb"]["all"]), OWN(1))

    def test_a_language_with_a_different_number_of_sentences_is_handled_and_the_names_must_match(self):
        data = corpus(); data[2] = data[2][:40]
        rep, info = g9.tag_pipeline(forward_acts, encode, data, BOS, PAD, VOCAB, seed=0, n_boot=20, n_perm=10, batch_size=16)
        self.assertEqual(info["budget_tokens_per_language"], 40 * 20)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                g9.save_selectivity(d, rep, ["only", "two"])

    def test_the_selectivity_floats_are_stored_as_float32(self):
        rep, _ = self.run_pipeline()
        with tempfile.TemporaryDirectory() as d:
            sel = torch.load(g9.save_selectivity(d, rep, ["a", "b", "c"]), weights_only=False)
            for k in ("rates", "mean_activation", "S"):
                self.assertEqual(sel[k].dtype, torch.float32, k)


class Lm(unittest.TestCase):
    def test_the_gain_wrapper_is_called_with_the_study_g_down_ratios_three_draws_and_the_gemma9b_setting_active(self):
        seen = {}
        def fake(argv):
            seen["argv"] = list(argv); seen["layer"] = (pr.LAYER, fr.LAYER); seen["model"] = list(pr.MODEL_CANDIDATES)
        old = gr.main; gr.main = fake
        try:
            g9.main(["lm", "--out", "o", "--split", "eval"])
        finally:
            gr.main = old
        a = seen["argv"]
        self.assertEqual(a[a.index("--ratios") + 1:a.index("--ratios") + 9], [str(r) for r in g9.STUDY_RATIOS]); self.assertEqual(a[a.index("--n-controls") + 1], "3"); self.assertEqual(a[a.index("--label") + 1], "K64")
        self.assertEqual(a[a.index("--ratios") + 9], "--n-controls")
        self.assertIn("--out", a); self.assertEqual(seen["layer"], (20, 20)); self.assertEqual(seen["model"], ["google/gemma-2-9b"])
        self.assertEqual((pr.LAYER, fr.LAYER), (12, 12))

    def test_the_ratios_are_the_study_g_decision_in_the_order_of_its_preset_name(self):
        self.assertEqual(g9.STUDY_RATIOS, (0.3535533905932738, 0.1767766952966369, 0.08838834764831845, 2.82842712474619, 5.65685424949238, 11.31370849898476, 1.1, 1.2))
        dec = json.loads((Path(__file__).resolve().parents[3] / "reports/functional_validation/gain_study/calibration/ratios.json").read_text())
        self.assertEqual(list(g9.STUDY_RATIOS), dec["down"] + dec["up"] + dec["fine_up"])

    def test_the_ratios_cannot_be_overridden_from_the_command_line(self):
        old = gr.main; gr.main = lambda argv: None
        try:
            with self.assertRaises(SystemExit):
                g9.main(["lm", "--ratios", "0.5", "--out", "o"])
        finally:
            gr.main = old

    def test_an_unknown_task_is_refused_and_nothing_is_run(self):
        called = []
        old_tag, old_gr = g9.tag_main, gr.main; g9.tag_main = lambda argv, _sae_key=None: called.append("tag"); gr.main = lambda argv: called.append("lm")
        try:
            with self.assertRaises(SystemExit):
                g9.main(["other"])
        finally:
            g9.tag_main, gr.main = old_tag, old_gr
        self.assertEqual(called, [])


class Manifest(unittest.TestCase):
    def selectivity(self, d, n_sel0=70, n_pool=120, n_sel1=20):
        F = n_sel0 + n_sel1 + n_pool + 10
        g = torch.Generator().manual_seed(3)
        rates = torch.rand(2, F, generator=g) * 0.01 + 0.001; mean_act = torch.rand(2, F, generator=g) + 0.1
        peak = torch.zeros(F, dtype=torch.long); peak[n_sel0:n_sel0 + n_sel1] = 1
        selective = torch.zeros(F, dtype=torch.bool); selective[:n_sel0 + n_sel1] = True
        supported = torch.ones(F, dtype=torch.bool); supported[-10:] = False
        path = Path(d) / "selectivity.pt"; torch.save({"rates": rates, "mean_activation": mean_act, "peak": peak, "selective": selective, "supported": supported, "languages": ["aaa", "bbb"]}, path)
        return path

    def test_the_default_builder_includes_the_all_family_unlike_llama(self):
        with tempfile.TemporaryDirectory() as d:
            path = self.selectivity(d)
            out = Path(d) / "man.json"; g9.build_gemma9b_manifest(path, out)
            man = json.loads(out.read_text())
            self.assertEqual(man["Ks"], [16, 64, None]); self.assertEqual(man["Rs"], [5, 5, 5])
            self.assertIn("all", man["sizes"]["aaa"]); self.assertEqual(man["sizes"]["aaa"]["all"], 70)
            self.assertEqual(man["sizes"]["aaa"]["K16"], 16); self.assertEqual(man["sizes"]["aaa"]["K64"], 64)

    def test_it_is_the_frozen_builder_with_three_families_and_the_frozen_serialisation(self):
        with tempfile.TemporaryDirectory() as d:
            path = self.selectivity(d); out = Path(d) / "man.json"; g9.build_gemma9b_manifest(path, out)
            want = fg.build_manifest(path, Ks=(16, 64, None), Rs=(5, 5, 5), seed=0)
            self.assertEqual(out.read_text(), json.dumps(want, separators=(",", ":")) + "\n")

    def test_the_seed_is_passed_on_and_changes_the_control_draws(self):
        with tempfile.TemporaryDirectory() as d:
            path = self.selectivity(d)
            a, b = Path(d) / "a.json", Path(d) / "b.json"; g9.build_gemma9b_manifest(path, a, seed=0); g9.build_gemma9b_manifest(path, b, seed=1)
            self.assertNotEqual(json.loads(a.read_text())["controls"], json.loads(b.read_text())["controls"]); self.assertEqual(json.loads(b.read_text())["seed"], 1)

    def test_the_task_manifest_runs_without_the_model_setting(self):
        with tempfile.TemporaryDirectory() as d:
            path = self.selectivity(d); out = Path(d) / "man.json"
            before = (pr.MODEL_CANDIDATES, pr.LAYER)
            g9.main(["manifest", "--selectivity", str(path), "--manifest", str(out)])
            self.assertTrue(out.exists()); self.assertEqual((pr.MODEL_CANDIDATES, pr.LAYER), before)
            g9.main(["manifest", "--selectivity", str(path), "--manifest", str(out), "--seed", "4"]); self.assertEqual(json.loads(out.read_text())["seed"], 4)


if __name__ == "__main__":
    unittest.main()
