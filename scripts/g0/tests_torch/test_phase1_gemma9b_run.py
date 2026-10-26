import os
import sys
import unittest
import torch
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phase1_func_run as fr
import phase1_gemma9b_run as g9
import phase1_llama_run as lr
import phase1_run as pr
import their_code as tc


def snapshot():
    return (pr.MODEL_CANDIDATES, pr.SAE_RELEASE, pr.SAE_ID, pr.LAYER, fr.LAYER, tc.build_namespace)


class Setting(unittest.TestCase):
    def test_constants_are_set_inside_and_restored_after(self):
        before = snapshot()
        with g9.gemma9b_setting():
            self.assertEqual(pr.MODEL_CANDIDATES, ["google/gemma-2-9b"])
            self.assertEqual(pr.SAE_RELEASE, "gemma-scope-9b-pt-res-canonical")
            self.assertEqual(pr.SAE_ID, "layer_20/width_16k/canonical")
            self.assertEqual((pr.LAYER, fr.LAYER), (20, 20))
            self.assertIs(tc.build_namespace, before[5])          # no loader replacement: the frozen load_sae_compat is used
        self.assertEqual(snapshot(), before)

    def test_everything_is_restored_after_an_error(self):
        before = snapshot()
        with self.assertRaises(RuntimeError):
            with g9.gemma9b_setting():
                raise RuntimeError("boom")
        self.assertEqual(snapshot(), before)

    def test_the_llama_setting_still_works_after_a_9b_run(self):
        before = snapshot()
        with g9.gemma9b_setting():
            pass
        with lr.llama_setting("fineweb42"):
            self.assertEqual((pr.LAYER, fr.LAYER), (11, 11))
        self.assertEqual(snapshot(), before)


class Checks(unittest.TestCase):
    def test_hook_and_width_assertion(self):
        g9.check_sae_fits("blocks.20.hook_resid_post", 3584, 3584)
        with self.assertRaises(AssertionError):
            g9.check_sae_fits("blocks.12.hook_resid_post", 3584, 3584)
        with self.assertRaises(AssertionError):
            g9.check_sae_fits("blocks.20.hook_resid_post", 2304, 3584)

    def test_the_2b_ratios_are_used_unchanged(self):
        self.assertEqual(g9.STUDY_RATIOS, lr.STUDY_RATIOS)
        self.assertEqual(len(g9.STUDY_RATIOS), 8)

    def test_families(self):
        self.assertEqual(g9.FAMILIES["full"], ((16, 64, None), (5, 5, 3)))
        self.assertEqual(g9.FAMILIES["k"], ((16, 64), (5, 5)))
        with self.assertRaises(ValueError):
            g9.build_manifest("x", "y", families="nope")


class Routing(unittest.TestCase):
    def test_lm_passes_the_study_g_preset_inside_the_setting(self):
        seen = {}

        def fake_main(args):
            seen["args"], seen["layer"] = args, (pr.LAYER, fr.LAYER)
        with mock.patch.object(g9.gr, "main", fake_main):
            g9.main(["lm", "--out", "o", "--precisions", "FP", "4"])
        self.assertEqual(seen["layer"], (20, 20))
        self.assertEqual(seen["args"][0], "--ratios")
        self.assertEqual(seen["args"][1:9], [str(r) for r in g9.STUDY_RATIOS])
        self.assertEqual(seen["args"][9:13], ["--n-controls", "3", "--label", "K64"])
        self.assertEqual(seen["args"][13:], ["--out", "o", "--precisions", "FP", "4"])
        self.assertEqual(pr.LAYER, 12)

    def test_lm_refuses_overriding_the_preset(self):
        with self.assertRaises(SystemExit):
            g9.main(["lm", "--ratios", "1"])

    def test_tag_runs_inside_the_setting(self):
        seen = {}
        with mock.patch.object(g9, "tag_main", lambda rest: seen.update(layer=pr.LAYER, rest=rest)):
            g9.main(["tag", "--out", "o"])
        self.assertEqual(seen, {"layer": 20, "rest": ["--out", "o"]})

    def test_manifest_forwards_families(self):
        with mock.patch.object(g9, "build_manifest", return_value={"languages": ["a"], "sizes": {"a": {}}}) as bm:
            g9.main(["manifest", "--selectivity", "s", "--manifest", "m", "--families", "k"])
        bm.assert_called_once_with("s", "m", families="k", seed=0)


class FakeHT:
    """HookedTransformer stand-in: records the load call; its forward returns a CPU tensor."""
    calls = []

    class Cfg:
        n_layers, d_model = 42, 3584

    @classmethod
    def from_pretrained_no_processing(cls, name, **kw):
        cls.calls.append((name, kw))
        if name == "bad":
            raise OSError("no such model")
        m = cls()
        m.cfg = cls.Cfg()
        return m

    def eval(self):
        return self

    def forward(self, tokens, stop_at_layer=None):
        return torch.ones(2, 3)


class MultiDevice(unittest.TestCase):
    def setUp(self):
        FakeHT.calls = []

    def test_the_loader_passes_n_devices_and_bf16_and_tries_every_candidate(self):
        load = g9.make_multi_device_loader(4, FakeHT, device="cpu")
        model, name = load(["bad", "google/gemma-2-9b"])
        self.assertEqual(name, "google/gemma-2-9b")
        self.assertEqual([c[0] for c in FakeHT.calls], ["bad", "google/gemma-2-9b"])
        kw = FakeHT.calls[1][1]
        self.assertEqual((kw["n_devices"], kw["dtype"], kw["device"]), (4, torch.bfloat16, "cpu"))

    def test_the_loader_fails_loudly_when_no_candidate_loads(self):
        with self.assertRaises(RuntimeError):
            g9.make_multi_device_loader(2, FakeHT, device="cpu")(["bad"])

    def test_a_single_device_is_not_a_multi_device_load(self):
        with self.assertRaises(ValueError):
            g9.make_multi_device_loader(1, FakeHT)

    def test_outputs_are_returned_on_the_first_device(self):
        m = FakeHT(); g9.first_device_outputs(m, "meta")
        self.assertEqual(m.forward(torch.zeros(1), stop_at_layer=21).device.type, "meta")

    def test_non_tensor_outputs_are_untouched(self):
        m = FakeHT(); m.forward = lambda *a, **k: ("x", 1)
        g9.first_device_outputs(m, "meta")
        self.assertEqual(m.forward(), ("x", 1))

    def test_the_setting_replaces_only_the_model_loader_when_n_devices_is_above_one(self):
        before = tc.build_namespace
        seen = {}

        def fake_build(ex, env):
            env["load_gemma_tl_model"] = "frozen loader"; env["load_sae_compat"] = "frozen sae loader"
        with mock.patch.object(tc, "build_namespace", fake_build):
            outer = tc.build_namespace
            with g9.gemma9b_setting(3):
                env = {"HookedTransformer": FakeHT, "DEVICE": "cpu"}
                tc.build_namespace(None, env)
                seen["loader"], seen["sae"] = env["load_gemma_tl_model"], env["load_sae_compat"]
            self.assertIs(tc.build_namespace, outer)
        self.assertNotEqual(seen["loader"], "frozen loader"); self.assertTrue(callable(seen["loader"]))
        self.assertEqual(seen["sae"], "frozen sae loader")
        self.assertIs(tc.build_namespace, before)

    def test_n_devices_one_leaves_the_frozen_loader_alone(self):
        before = tc.build_namespace
        with g9.gemma9b_setting(1):
            self.assertIs(tc.build_namespace, before)

    def test_main_routes_n_devices_and_the_environment_default(self):
        seen = {}
        with mock.patch.object(g9.gr, "main", lambda args: seen.update(patched=tc.build_namespace is not seen.get("orig"))):
            seen["orig"] = tc.build_namespace
            g9.main(["lm", "--n-devices", "4", "--out", "o"])
        self.assertTrue(seen["patched"])
        with mock.patch.dict(os.environ, {"GEMMA9B_N_DEVICES": "4"}):
            self.assertEqual(g9.default_n_devices(), 4)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GEMMA9B_N_DEVICES", None)
            self.assertEqual(g9.default_n_devices(), 1)


if __name__ == "__main__":
    unittest.main()
