import contextlib
import io
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_func_run as fr
import phase1_gain_run as gr
from test_phase1_func_intervene import ToySAE
from test_phase1_func_run import F, D, ToyModel, brute_nll, corpus
import phase1_lib as p1

CODES = ["l0", "l1", "l2"]                       # the toy corpus has three languages, indices 0 to 2
RATIOS = [0.5, 0.25, 1.5, 2.0]


def toy_manifest(n_draws=3, size=4):
    g = torch.Generator().manual_seed(7)
    groups, controls, sizes = {}, {}, {}
    for l, code in enumerate(CODES):
        pool = torch.randperm(F, generator=g).tolist()
        groups[code] = {"all": pool[:6]}
        controls[code] = {"K64": [sorted(pool[6 + size * r:6 + size * (r + 1)]) for r in range(n_draws)], "gap": {"K64": [0.0] * n_draws}}
        sizes[code] = {"K64": size}
    return {"languages": CODES, "groups": groups, "controls": controls, "sizes": sizes}


class Ratios(unittest.TestCase):
    def test_a_valid_list_is_returned_as_floats_in_the_given_order(self):
        self.assertEqual(gr.validate_ratios([0.5, 2, 0.25]), [0.5, 2.0, 0.25])

    def test_rejected_values(self):
        for bad in ([], [0.5, 0.5], [0.0], [-1.0], [float("nan")], [float("inf")], [1.0], [0.5, 1.0]):
            with self.assertRaises(ValueError, msg=str(bad)):
                gr.validate_ratios(bad)

    def test_two_ratios_that_format_alike_in_the_names_are_rejected(self):
        with self.assertRaises(ValueError):
            gr.validate_ratios([0.35355339, 0.35355341])

    def test_exact_values_survive_the_preset_name(self):
        ratios = [2 ** -0.5, 0.25, 2 ** 0.5, 1.1]
        name = gr.preset_name(ratios, 3, "K64")
        self.assertEqual(gr.parse_preset(name), (ratios, 3, "K64"))

    def test_a_name_that_is_not_a_gain_preset_is_refused(self):
        with self.assertRaises(ValueError):
            gr.parse_preset("eval")


class Configs(unittest.TestCase):
    def setUp(self):
        self.man = toy_manifest()

    def test_control_configs_cover_every_draw_and_ratio_of_every_target(self):
        cfgs = gr.control_scale_configs(self.man, CODES, CODES, RATIOS, "K64", 3)
        self.assertEqual(len(cfgs), 3 * 3 * 4)
        self.assertEqual(len({c["name"] for c in cfgs}), len(cfgs))
        c = [x for x in cfgs if x["name"] == "cscale0.5.2_K64@l1"][0]
        self.assertEqual(c["kind"], "scale"); self.assertEqual(c["kw"], {"alpha": 0.5}); self.assertEqual(c["langs"], {1})
        self.assertEqual(c["idx"].tolist(), self.man["controls"]["l1"]["K64"][2])

    def test_only_the_asked_number_of_draws_is_used(self):
        cfgs = gr.control_scale_configs(toy_manifest(n_draws=5), CODES, ["l0"], [0.5], "K64", 3)
        self.assertEqual([c["name"] for c in cfgs], [f"cscale0.5.{r}_K64@l0" for r in range(3)])

    def test_fewer_draws_than_asked_is_an_error(self):
        with self.assertRaises(ValueError):
            gr.control_scale_configs(toy_manifest(n_draws=2), CODES, CODES, RATIOS, "K64", 3)

    def test_unknown_target_is_an_error(self):
        with self.assertRaises(ValueError):
            gr.control_scale_configs(self.man, CODES, ["zzz"], RATIOS, "K64", 3)

    def test_index_checks(self):
        ok = [{"name": "x", "kind": "scale", "idx": torch.tensor([1, 2, 3]), "langs": {0}, "kw": {"alpha": 2.0}}]
        gr.check_indices(ok, F)
        for idx in ([1, 2, 2], [0, F], [-1, 3]):
            bad = [dict(ok[0], idx=torch.tensor(idx))]
            with self.assertRaises(ValueError, msg=str(idx)):
                gr.check_indices(bad, F)


class Registered(unittest.TestCase):
    def setUp(self):
        self.man = toy_manifest()
        self.original = fr.build_configs
        self.addCleanup(gr.unregister)
        self.name = gr.register(RATIOS, n_controls=3, label="K64")

    def test_register_adds_a_preset_with_the_study_families(self):
        fam = fr.PRESETS[self.name]
        self.assertEqual(fam["zero"], ["K64"]); self.assertEqual(fam["controls"], ["K64"]); self.assertEqual(fam["n_controls"], {"K64": 3})
        self.assertEqual(fam["sham"], ["K64"]); self.assertEqual(fam["scale"], {"K64": RATIOS})

    def test_unregister_restores_the_frozen_function_and_presets(self):
        gr.unregister()
        self.assertIs(fr.build_configs, self.original)
        self.assertEqual(set(fr.PRESETS), {"none", "smoke", "primary", "eval", "floor"})

    def test_the_wrapped_builder_makes_the_expected_configuration_set(self):
        cfgs = fr.build_configs(self.man, CODES, CODES, [], fr.PRESETS[self.name], "K64")
        names = [c["name"] for c in cfgs]
        self.assertEqual(len(names), len(set(names)))
        per_language = 1 + 3 + 1 + len(RATIOS) + 3 * len(RATIOS)            # zero, three zeroed controls, sham, group scales, scaled controls
        self.assertEqual(len(cfgs), 3 * per_language)
        for code in CODES:
            self.assertIn(f"scale0.5_K64@{code}", names); self.assertIn(f"cscale2.0_K64@{code}", names); self.assertIn(f"ctrl2_K64@{code}", names)
            self.assertIn(f"zero_K64@{code}", names); self.assertIn(f"sham_K64@{code}", names)

    def test_no_configuration_runs_on_another_language_text_without_a_panel(self):
        for c in fr.build_configs(self.man, CODES, CODES, [], fr.PRESETS[self.name], "K64"):
            self.assertEqual(len(c["langs"]), 1)
            self.assertEqual(c["langs"], {CODES.index(c["name"].split("@")[1])})

    def test_the_wrapper_refuses_a_set_with_a_repeated_index(self):
        man = toy_manifest(); man["controls"]["l0"]["K64"][1][1] = man["controls"]["l0"]["K64"][1][0]
        with self.assertRaises(ValueError):
            fr.build_configs(man, CODES, CODES, [], fr.PRESETS[self.name], "K64")

    def test_the_feature_count_of_the_run_bounds_the_indices(self):
        man = toy_manifest(); man["controls"]["l2"]["K64"][0][0] = F + 3
        gr_ok = fr.build_configs(man, CODES, CODES, [], fr.PRESETS[self.name], "K64")      # default bound 16,384: accepted
        self.assertTrue(gr_ok)
        with self.assertRaises(ValueError):
            fr.build_configs(man, CODES, CODES, [], fr.PRESETS[self.name], "K64", n_features=F)

    def test_registering_twice_does_not_stack_wrappers(self):
        gr.register(RATIOS, n_controls=3, label="K64")
        cfgs = fr.build_configs(self.man, CODES, CODES, [], fr.PRESETS[self.name], "K64")
        self.assertEqual(len({c["name"] for c in cfgs}), len(cfgs))      # a stacked wrapper would add the scaled controls twice and the frozen validator would see duplicates
        gr.unregister()
        self.assertIs(fr.build_configs, self.original)

    def test_frozen_families_still_work_without_the_gain_key(self):
        cfgs = fr.build_configs(self.man, CODES, CODES, [], {"zero": ["K64"]}, "K64")
        self.assertEqual([c["name"] for c in cfgs], [f"zero_K64@{c}" for c in CODES])


class Engine(unittest.TestCase):
    """The scaled configurations through the frozen engine on the toy model: identity, and equality with a brute-force run of one sentence at a time."""
    def setUp(self):
        self.model, self.sae = ToyModel(), ToySAE(d=D, F=F)
        self.data = corpus()
        self.batches = p1.make_batches(self.data, 1, "mask", 0, 256, batch_size=3)
        self.n_sent = {l: len(v) for l, v in self.data.items()}
        self.man = toy_manifest()

    def run_engine(self, configs):
        return fr.evaluate_precision(self.model.lower, self.model.upper, self.sae.encode, self.sae.W_dec, self.batches, self.n_sent, configs, decode=self.sae.decode)

    def test_a_control_scaled_by_a_ratio_equals_the_brute_force_edit(self):
        cfgs = gr.control_scale_configs(self.man, CODES, CODES, [0.5, 2.0], "K64", 3)
        out = self.run_engine(cfgs)
        for c in cfgs:
            l = next(iter(c["langs"]))
            for i, s in enumerate(self.data[l]):
                nll, _ = brute_nll(self.model, self.sae, s, idx=c["idx"], kind="scale", alpha=c["kw"]["alpha"])
                self.assertAlmostEqual(float(out["items"][(l, c["name"])]["nll"][i]), nll, places=4, msg=c["name"])

    def test_ratio_one_is_the_identity_for_a_control_and_for_a_group(self):
        ctrl = gr.control_scale_configs(self.man, CODES, CODES, [0.5], "K64", 1)[0]
        ctrl = dict(ctrl, name="id_ctrl", kw={"alpha": 1.0})
        grp = {"name": "id_group", "kind": "scale", "idx": torch.tensor(self.man["groups"]["l0"]["all"][:4]), "langs": {0}, "kw": {"alpha": 1.0}}
        out = self.run_engine([ctrl, grp])
        for name, l in (("id_ctrl", 0), ("id_group", 0)):
            self.assertTrue(torch.equal(out["items"][(l, name)]["nll"], out["items"][(l, "none")]["nll"]), name)

    def test_a_control_only_changes_the_loss_through_its_own_features(self):
        cfgs = gr.control_scale_configs(self.man, CODES, CODES, [2.0], "K64", 1)
        c = cfgs[0]; l = next(iter(c["langs"]))
        out = self.run_engine([c])
        self.assertFalse(torch.equal(out["items"][(l, c["name"])]["nll"], out["items"][(l, "none")]["nll"]))
        other = dict(c, name="other_language", langs={(l + 1) % 3})
        out2 = self.run_engine([c, other])
        self.assertTrue(torch.equal(out2["items"][(l, c["name"])]["nll"], out["items"][(l, c["name"])]["nll"]))


class Main(unittest.TestCase):
    def setUp(self):
        self.addCleanup(gr.unregister)
        self.calls = []
        self.real_main = fr.main
        fr.main = lambda argv=None: self.calls.append((list(argv), set(fr.PRESETS)))
        self.addCleanup(lambda: setattr(fr, "main", self.real_main))

    def test_main_registers_then_forwards_the_rest_with_the_preset(self):
        gr.main(["--ratios", "0.5", "0.25", "2", "--n-controls", "2", "--out", "o", "--precisions", "FP", "4"])
        argv, presets = self.calls[0]
        name = gr.preset_name([0.5, 0.25, 2.0], 2, "K64")
        self.assertEqual(argv, ["--out", "o", "--precisions", "FP", "4", "--preset", name])
        self.assertIn(name, presets)

    def test_three_control_draws_are_the_default(self):
        gr.main(["--ratios", "0.5", "--out", "o"])
        self.assertEqual(self.calls[0][0][-1], gr.preset_name([0.5], 3, "K64"))

    def test_a_flag_of_the_frozen_runner_is_not_taken_for_an_abbreviation_of_ours(self):
        gr.main(["--ratios", "0.5", "--n", "150", "--out", "o"])
        self.assertEqual(self.calls[0][0][:2], ["--n", "150"])

    def test_main_refuses_a_preset_argument_and_missing_ratios(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            gr.main(["--ratios", "0.5", "--preset", "eval", "--out", "o"])
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            gr.main(["--out", "o"])
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
