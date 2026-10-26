import sys
import unittest
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


if __name__ == "__main__":
    unittest.main()
