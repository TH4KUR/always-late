"""Task 4 of the Gemma-2-9B three-arm plan: `phase1_gemma9b_ext.main` runs the frozen entry points under the adapter's 9B setting without editing the pinned adapter: task `lm` is the adapter's own Study G run (unchanged),
`p1` is `phase1_run.main` (tier 2) and `gate` is `gptq_ext.gate_run.main` (the build-and-gate job), both inside `gemma9b_setting`. Through `gptq_ext/run_with_weights.py --entry phase1_gemma9b_ext:main` they get the GPTQ or
matched-RTN weight sets. Fakes replace the entry points; the frozen-code check uses the committed freeze."""
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import make_gemma9b_freeze as mf
import phase1_func_run as fr
import phase1_gemma9b_ext as ext
import phase1_gemma9b_run as g9
import phase1_run as pr
from gptq_ext import gate_run, run_with_weights as rww

ROOT = Path(__file__).resolve().parents[3]


def constants():
    return (pr.MODEL_CANDIDATES, pr.SAE_RELEASE, pr.SAE_ID, pr.LAYER, fr.LAYER)


NINE_B = (["google/gemma-2-9b"], "gemma-scope-9b-pt-res-canonical", "layer_20/width_16k/canonical", 20, 20)


class Tasks(unittest.TestCase):
    def test_p1_runs_with_the_9b_constants_and_restores_them(self):
        before, seen = constants(), {}
        with mock.patch.object(pr, "main", side_effect=lambda argv: seen.update(c=constants(), argv=list(argv))):
            ext.main(["p1", "--corpus", "flores", "--regime", "mask", "--bits", "7", "6", "4"])
        self.assertEqual(seen["c"], NINE_B)
        self.assertEqual(seen["argv"], ["--corpus", "flores", "--regime", "mask", "--bits", "7", "6", "4"])
        self.assertEqual(constants(), before)

    def test_gate_runs_with_the_9b_constants_and_restores_them(self):
        before, seen = constants(), {}
        with mock.patch.object(gate_run, "main", side_effect=lambda argv: seen.update(c=constants(), argv=list(argv))):
            ext.main(["gate", "--methods", "gptq", "--bits", "6", "4"])
        self.assertEqual(seen["c"], NINE_B)
        self.assertEqual(seen["argv"], ["--methods", "gptq", "--bits", "6", "4"])
        self.assertEqual(constants(), before)

    def test_constants_are_restored_when_the_entry_point_raises(self):
        before = constants()
        with mock.patch.object(pr, "main", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                ext.main(["p1", "--corpus", "flores", "--regime", "mask"])
        self.assertEqual(constants(), before)

    def test_lm_delegates_unchanged_arguments_to_the_adapter(self):
        with mock.patch.object(g9, "main") as m:
            ext.main(["lm", "--out", "o", "--precisions", "6", "4", "--split", "eval"])
        (argv,), _ = m.call_args
        self.assertEqual(argv[0], "lm")
        self.assertEqual(argv[1:3], ["--n-devices", "1"])
        self.assertEqual(argv[3:], ["--out", "o", "--precisions", "6", "4", "--split", "eval"])

    def test_lm_forwards_n_devices(self):
        with mock.patch.object(g9, "main") as m:
            ext.main(["lm", "--n-devices", "2", "--out", "o"])
        self.assertEqual(m.call_args[0][0][:3], ["lm", "--n-devices", "2"])

    def test_p1_and_gate_refuse_a_multi_device_split(self):
        for task in ("p1", "gate"):
            with self.assertRaises(SystemExit):
                ext.main([task, "--n-devices", "2"])

    def test_unknown_task_is_refused(self):
        with self.assertRaises(SystemExit):
            ext.main(["tag", "--out", "o"])


class ThroughTheWeightsWrapper(unittest.TestCase):
    def test_the_precisions_and_bits_are_seen_by_the_prebuild_hint(self):
        self.assertEqual(rww.bits_hint(["lm", "--precisions", "6", "4", "--split", "eval"]), [4, 6])
        self.assertEqual(rww.bits_hint(["p1", "--bits", "7", "6", "4", "16", "12", "0"]), [0, 4, 6, 7, 12, 16])

    def test_the_wrapper_calls_the_entry_with_the_arguments_after_the_separator(self):
        with mock.patch.object(ext, "main") as m:
            rww.main(["--entry", "phase1_gemma9b_ext:main", "--", "p1", "--corpus", "flores", "--regime", "mask"])
        m.assert_called_once_with(["p1", "--corpus", "flores", "--regime", "mask"])


class FrozenFiles(unittest.TestCase):
    def test_the_pinned_files_are_unchanged(self):
        self.assertEqual(mf.verify(ROOT / "reports/functional_validation/gemma9b/freeze.json"), [])


if __name__ == "__main__":
    unittest.main()
