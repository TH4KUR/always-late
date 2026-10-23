"""Rehearsal of `phase1_func_f5_run.main` on a toy model: the heavy modules (TransformerLens, SAE Lens, their_code, phase1_data, phase1_run, tqdm) are replaced by small stand-ins so that the whole wiring (arguments, set-up,
gates, state loop, files) runs in a second. What it cannot show is that the real Gemma-2-2B behaves like the toy: the smoke job on Ada does that."""
import json
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_func_f5_run as r5
import phase1_func_tasks as ft
from test_phase1_func_f5_run import ToyStack
from test_phase1_func_intervene import ToySAE
from test_phase1_func_run import V, D, F, BOS


class ToyTL:
    """The parts of a HookedTransformer that main() uses, on the toy stack: model(tokens, stop_at_layer=13) -> residual, model(resid, start_at_layer=13, stop_at_layer=26) -> final hidden state,
    model(resid, start_at_layer=13) -> the model's own logits (final norm, unembedding, soft cap). The bit width of the quantizer stand-in changes the upper map."""
    def __init__(self):
        g = torch.Generator().manual_seed(7); self.stack = ToyStack(linear=True); self.bits = None; R = torch.randn(D, D, generator=g)
        self.stack.perturb = lambda A: A if self.bits is None else A + (8 - self.bits) * 0.03 * R
        self.cfg = types.SimpleNamespace(n_layers=26, d_model=D, d_vocab=V, eps=1e-6, output_logits_soft_cap=30.0)
        self.ln_final = types.SimpleNamespace(w=torch.rand(D, generator=g) + 0.5); self.unembed = types.SimpleNamespace(W_U=torch.randn(D, V, generator=g) * 2, b_U=torch.zeros(V))
        self.tokenizer = types.SimpleNamespace(bos_token_id=BOS, pad_token_id=0)

    def __call__(self, x, start_at_layer=None, stop_at_layer=None):
        if start_at_layer is None:
            assert stop_at_layer == 13, stop_at_layer
            return self.stack.lower(x)
        assert start_at_layer == 13
        h = self.stack.upper_final(x)
        if stop_at_layer is not None:
            assert stop_at_layer == 26, stop_at_layer
            return h
        n = h / torch.sqrt(h.pow(2).mean(-1, keepdim=True) + self.cfg.eps) * self.ln_final.w
        z = n @ self.unembed.W_U + self.unembed.b_U
        return 30.0 * torch.tanh(z / 30.0)


def line(code, split, i):
    rng = random.Random(f"{code}/{split}/{i}")
    return "".join(rng.choice("abcdefghijklmnop") for _ in range(rng.randint(3, 9)))


class MainRehearsal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); tmp = Path(cls.tmp.name); cls.out = tmp / "out"
        cls.codes = ["l0", "l1", "l2"]; cls.model = ToyTL(); cls.sae = ToySAE(); cls.sae.cfg = types.SimpleNamespace(hook_name="blocks.12.hook_resid_post", d_in=D, d_sae=F)
        (tmp / "roster.json").write_text(json.dumps({"codes": cls.codes}))
        cls.ids = list(range(0, F, 2)); (tmp / "sample.json").write_text(json.dumps({"languages": cls.codes, "features": {str(f): {"language": cls.codes[f % 3], "roles": ["k64"], "native": {"FP": [0.001 * (1 + (f % 7)), 0.01]}} for f in cls.ids}}))
        (tmp / "manifest.json").write_text(json.dumps({"functions": {}}))
        (tmp / "freeze.json").write_text(json.dumps({"ids": {"n_total": 1012, "n_pilot": 150, "n_eval": 300, "seed": 0, "eval_ids": ft.lm_ids(1012, 150, 300, 0)["eval"]}}))
        cls.stubs = cls.make_stubs(tmp, cls.model, cls.sae)
        cls.args = ["--out", str(cls.out), "--flores-root", "x", "--roster", str(tmp / "roster.json"), "--sample", str(tmp / "sample.json"), "--freeze", str(tmp / "freeze.json"),
                    "--states", "FP", "FP_rebatched", "6", "4", "--M", "5", "--min-ctx", "3", "--topk", "3", "--dev-sentences", "40", "--max-eval", "20", "--batch-size", "6", "--commit", "abc123"]
        cls.run_main(cls.args, cls.stubs)
        cls.meta = json.loads((cls.out / "meta_f5.json").read_text())
        cls.stage = torch.load(cls.out / "f5_stage_a.pt", weights_only=False); cls.st = {s: torch.load(cls.out / f"f5_state_{s}.pt", weights_only=False) for s in ("FP", "FP_rebatched", "INT6", "INT4")}

    @staticmethod
    def make_stubs(tmp, model, sae):
        def build_namespace(ex, env):
            env["load_gemma_tl_model"] = lambda candidates: (model, "toy-gemma"); env["load_sae_compat"] = lambda release, sae_id, device: sae
            env["clone_state_to_cpu"] = lambda m: {}; env["restore_model_work"] = lambda: setattr(model, "bits", None)
            env["quantize_rtn_per_output_channel_tl"] = lambda m, bits: setattr(m, "bits", bits); env["encode_sae"] = lambda s, x: s.encode(x.float())
        tc = types.ModuleType("their_code"); tc.PRISTINE = "p"; tc.THEIR_FUNCTIONS = []; tc.MANIFEST = tmp / "manifest.json"; tc.extract_functions = lambda p, f: {}; tc.verify_hashes = lambda ex, m: None
        tc.build_namespace = build_namespace
        pdm = types.ModuleType("phase1_data"); pdm.load_roster = lambda path: [{"code": c} for c in json.loads(Path(path).read_text())["codes"]]
        pdm.read_flores_split = lambda root, code, split, expect=None: [line(code, split, i) for i in range(expect)]
        pdm.encode_sentences = lambda tok, lines: [[2 + (ord(c) % (V - 2)) for c in s] for s in lines]
        prm = types.ModuleType("phase1_run"); prm.MODEL_CANDIDATES = ["toy"]; prm.SAE_RELEASE = "r"; prm.SAE_ID = "i"; prm.package_versions = lambda names: {n: "0" for n in names}
        tq = types.ModuleType("tqdm"); tqa = types.ModuleType("tqdm.auto"); tqa.tqdm = lambda x, **k: x; tq.auto = tqa
        return {"their_code": tc, "phase1_data": pdm, "phase1_run": prm, "tqdm": tq, "tqdm.auto": tqa, "transformer_lens": types.SimpleNamespace(HookedTransformer=object), "sae_lens": types.SimpleNamespace(SAE=object)}

    @staticmethod
    def run_main(args, stubs):
        try:
            with mock.patch.dict(sys.modules, stubs):
                r5.main(args)
        finally:
            torch.set_grad_enabled(True)                                                                     # main switches gradients off for the process

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_the_files_of_every_state_and_the_metadata_are_written(self):
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), ["f5_stage_a.pt", "f5_state_FP.pt", "f5_state_FP_rebatched.pt", "f5_state_INT4.pt", "f5_state_INT6.pt", "meta_f5.json"])
        m = self.meta; self.assertTrue(m["complete"]); self.assertEqual(m["states"], ["FP", "FP_rebatched", "INT6", "INT4"]); self.assertEqual(sorted(m["state_seconds"]), sorted(m["states"]))
        self.assertEqual(m["commit"], "abc123"); self.assertEqual(m["n_features"], len(self.ids)); self.assertEqual(m["work_dtype"], "float32"); self.assertEqual(m["n_eval_per_language"], 20); self.assertEqual(m["n_dev"], 120); self.assertEqual(sorted(m["gates"]), ["logits_float32_residual", "logits_native_residual"])
        self.assertEqual(len(m["sample_sha256"]), 64); self.assertEqual(len(m["freeze_sha256"]), 64); self.assertTrue(all(g["ok"] for g in m["gates"].values())); self.assertTrue(all(g["exact"] for g in m["sham_gate"]))

    def test_stage_a_files_carry_the_feature_ids_languages_and_the_candidate_languages(self):
        s = self.stage; self.assertEqual(s["feat_ids"].tolist(), self.ids); self.assertEqual(s["feat_lang"].tolist(), [f % 3 for f in self.ids])
        self.assertEqual(len(s["dev_lang"]), 3 * 40); self.assertEqual(sorted(set(s["dev_lang"].tolist())), [0, 1, 2]); self.assertEqual(s["contexts"].shape, (len(self.ids), 5))
        self.assertEqual(s["write_features"], [int(self.ids[k]) for k in s["write_rows"]])
        self.assertGreaterEqual(len(s["write_rows"]), 6)

    def test_full_precision_is_the_identity_and_the_floor_state_is_numerically_the_same(self):
        fp, rb = self.st["FP"]["write"], self.st["FP_rebatched"]["write"]
        torch.testing.assert_close(torch.tensor(fp["cos"]), torch.ones(len(fp["cos"])).double(), atol=1e-6, rtol=0); torch.testing.assert_close(torch.tensor(rb["cos"]), torch.ones(len(rb["cos"])).double(), atol=1e-5, rtol=0)
        r = self.st["FP"]["read"]; self.assertTrue(((r["corr"] > 0.999999) | torch.tensor(r["corr"]).isnan().numpy()).all())

    def test_the_planted_drift_of_the_quantized_states_grows_with_the_bit_loss(self):
        import numpy as np
        c6, c4 = np.nanmedian(self.st["INT6"]["write"]["cos"]), np.nanmedian(self.st["INT4"]["write"]["cos"])
        self.assertLess(c6, 0.9999); self.assertLess(c4, c6)

    def test_the_within_group_retrieval_uses_the_languages_of_the_tested_features(self):
        import numpy as np
        g = self.stage["feat_lang"][self.stage["write_rows"]]; sizes = np.array([int((g == x).sum()) for x in g if int((g == x).sum()) > 1], dtype=float)
        self.assertAlmostEqual(self.st["FP"]["write"]["retrieval"]["within"]["chance"], float((1.0 / sizes).mean()), places=12)

    def test_the_read_side_is_computed_for_the_whole_sample_in_every_state(self):
        for s in self.st.values():
            self.assertEqual(len(s["read"]["corr"]), len(self.ids))

    def test_a_first_state_other_than_FP_is_refused_before_anything_is_computed(self):
        argv = list(self.args); i = argv.index("--states"); argv[i + 1:i + 5] = ["6", "FP"]; argv[argv.index("--out") + 1] = str(Path(self.tmp.name) / "out2")
        with self.assertRaises(SystemExit):
            self.run_main(argv, self.stubs)
        self.assertFalse((Path(self.tmp.name) / "out2" / "f5_stage_a.pt").exists())

    def test_logits_that_disagree_with_the_models_own_stop_the_run_before_any_state_is_measured(self):
        class Broken(ToyTL):
            def __call__(self, x, start_at_layer=None, stop_at_layer=None):
                z = super().__call__(x, start_at_layer, stop_at_layer)
                return -z if (start_at_layer is not None and stop_at_layer is None) else z
        argv = list(self.args); argv[argv.index("--out") + 1] = str(Path(self.tmp.name) / "out3")
        with self.assertRaises(r5.GateError):
            self.run_main(argv, self.make_stubs(Path(self.tmp.name), Broken(), self.sae))
        self.assertFalse((Path(self.tmp.name) / "out3" / "f5_stage_a.pt").exists())

    def test_the_native_work_dtype_skips_the_float32_residual_gate_and_is_recorded(self):
        argv = list(self.args); argv[argv.index("--out") + 1] = str(Path(self.tmp.name) / "out4"); i = argv.index("--states"); argv[i + 1:i + 5] = ["FP", "4"]; argv += ["--work-dtype", "native"]
        self.run_main(argv, self.stubs); m = json.loads((Path(self.tmp.name) / "out4" / "meta_f5.json").read_text())
        self.assertEqual(m["work_dtype"], "native"); self.assertEqual(sorted(m["gates"]), ["logits_native_residual"]); self.assertEqual(m["states"], ["FP", "INT4"])


    def test_a_feature_limit_keeps_evenly_spaced_features_of_the_sample(self):
        argv = list(self.args); argv[argv.index("--out") + 1] = str(Path(self.tmp.name) / "out5"); i = argv.index("--states"); argv[i + 1:i + 5] = ["FP"]; argv += ["--max-features", "8"]
        self.run_main(argv, self.stubs); o = Path(self.tmp.name) / "out5"; st = torch.load(o / "f5_stage_a.pt", weights_only=False)
        self.assertEqual(json.loads((o / "meta_f5.json").read_text())["n_features"], 8); self.assertEqual(len(st["feat_ids"]), 8)
        self.assertEqual(int(st["feat_ids"][0]), self.ids[0]); self.assertEqual(int(st["feat_ids"][-1]), self.ids[-1])


    def test_without_a_state_list_the_default_states_run_including_the_weight_rounding_floor(self):
        argv = list(self.args); i = argv.index("--states"); del argv[i:i + 5]; argv[argv.index("--out") + 1] = str(Path(self.tmp.name) / "out6"); argv += ["--max-features", "6"]
        self.run_main(argv, self.stubs); m = json.loads((Path(self.tmp.name) / "out6" / "meta_f5.json").read_text()); self.assertEqual(m["states"], ["FP", "FP_rebatched", "INT16", "INT8", "INT6", "INT5", "INT4"])
        self.assertTrue((Path(self.tmp.name) / "out6" / "f5_state_INT16.pt").exists())


    def test_a_per_cell_cap_gives_a_stratified_subset_and_is_recorded(self):
        argv = list(self.args); argv[argv.index("--out") + 1] = str(Path(self.tmp.name) / "out7"); i = argv.index("--states"); argv[i + 1:i + 5] = ["FP"]; argv += ["--per-cell", "2", "--subset-seed", "5"]
        self.run_main(argv, self.stubs); o = Path(self.tmp.name) / "out7"; m = json.loads((o / "meta_f5.json").read_text()); st = torch.load(o / "f5_stage_a.pt", weights_only=False)
        self.assertEqual((m["n_features"], m["per_cell"], m["subset_seed"]), (6, 2, 5)); self.assertEqual(len(st["feat_ids"]), 6)
        sample = json.loads((Path(self.tmp.name) / "sample.json").read_text()); self.assertEqual(st["feat_ids"].tolist(), r5.sample_arrays(sample, None, 2, 5)[0].tolist())
        self.assertNotEqual(st["feat_ids"].tolist(), r5.sample_arrays(sample, None, 2, 0)[0].tolist())
        self.assertEqual(sorted(np.bincount(st["feat_lang"]).tolist()), [2, 2, 2])                                    # two per language (each language is one cell: the toy features have the role k64 only)


    def test_the_direction_control_option_records_the_matched_direction_features_and_is_off_by_default(self):
        import phase1_func_f5_metrics as met
        o = Path(self.tmp.name) / "out8"; argv = list(self.args); argv[argv.index("--out") + 1] = str(o); i = argv.index("--states"); argv[i + 1:i + 5] = ["FP", "4"]; argv += ["--direction-control"]
        self.run_main(argv, self.stubs); m = json.loads((o / "meta_f5.json").read_text()); st = torch.load(o / "f5_stage_a.pt", weights_only=False)
        rates = np.array([0.001 * (1 + (f % 7)) for f in self.ids]); want = np.array(self.ids)[met.match_controls(np.array([f % 3 for f in self.ids]), rates)]
        self.assertTrue(m["direction_control"]); np.testing.assert_array_equal(st["direction_ids"], want); self.assertTrue((st["direction_ids"] != st["feat_ids"]).all())
        self.assertFalse(self.meta["direction_control"]); self.assertIsNone(self.stage["direction_ids"])

    def test_a_language_with_a_single_sample_feature_makes_the_direction_control_impossible_and_is_refused(self):
        argv = list(self.args); argv[argv.index("--out") + 1] = str(Path(self.tmp.name) / "out9"); i = argv.index("--states"); argv[i + 1:i + 5] = ["FP"]; argv += ["--direction-control", "--per-cell", "1"]
        with self.assertRaises(ValueError):
            self.run_main(argv, self.stubs)
        self.assertFalse((Path(self.tmp.name) / "out9" / "f5_stage_a.pt").exists())


    def test_the_fixed_amplitude_option_is_recorded_and_off_by_default(self):
        o = Path(self.tmp.name) / "out10"; argv = list(self.args); argv[argv.index("--out") + 1] = str(o); i = argv.index("--states"); argv[i + 1:i + 5] = ["FP", "4"]; argv += ["--fixed-amplitude"]
        self.run_main(argv, self.stubs); m = json.loads((o / "meta_f5.json").read_text()); st = torch.load(o / "f5_stage_a.pt", weights_only=False)
        self.assertTrue(m["fixed_amplitude"]); self.assertTrue(st["fixed_amplitude"]); self.assertFalse(self.meta["fixed_amplitude"]); self.assertFalse(self.stage["fixed_amplitude"])
        c = torch.load(o / "f5_state_INT4.pt", weights_only=False)["write"]["cos"]; self.assertTrue(np.isfinite(c).all())


if __name__ == "__main__":
    unittest.main()
