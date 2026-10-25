import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phase1_gain_commands as gc
import phase1_gain_ratios as rt


def write(d, down=(0.5, 0.25, 0.125), up=(2.0, 4.0, 8.0)):
    p = Path(d) / "ratios.json"
    p.write_text(json.dumps({"down": list(down), "up": list(up), "fine_up": [1.1, 1.2]}))
    return p


class Main(unittest.TestCase):
    def test_the_main_command_names_the_state_the_ratios_and_the_frozen_arguments(self):
        with tempfile.TemporaryDirectory() as d:
            cmd = gc.main_command("INT4", write(d), "repo_abc")
        self.assertIn("-J gain-4", cmd); self.assertIn("run_gain.sbatch gain-4 ", cmd); self.assertIn("-t 04:00:00", cmd)
        self.assertIn("--ratios 0.5 0.25 0.125 2.0 4.0 8.0 1.1 1.2", cmd)
        self.assertIn("--n-controls 3", cmd); self.assertIn("--precisions 4 ", cmd + " "); self.assertIn("--split eval --n-eval 300 --panel", cmd)
        self.assertIn("repo_abc", cmd); self.assertIn("REPO=", cmd)

    def test_the_full_precision_state_is_fp(self):
        with tempfile.TemporaryDirectory() as d:
            cmd = gc.main_command("FP", write(d), "repo_abc")
        self.assertIn("-J gain-FP", cmd); self.assertIn("--precisions FP", cmd)

    def test_ratios_are_written_with_their_exact_values(self):
        with tempfile.TemporaryDirectory() as d:
            cmd = gc.main_command("INT6", write(d, down=(2 ** -0.5, 0.25, 0.0625), up=(2 ** 0.5, 4.0, 16.0)), "r")
        self.assertIn("0.7071067811865476", cmd); self.assertIn("1.4142135623730951", cmd)

    def test_an_unknown_state_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            for bad in ("INT3", "INT16", "fp", ""):
                with self.assertRaises(ValueError, msg=bad):
                    gc.main_command(bad, write(d), "r")

    def test_a_ratios_file_that_is_not_a_decision_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ratios.json"; p.write_text(json.dumps({"down": [0.5], "up": [2.0, 4.0, 8.0], "fine_up": [1.1, 1.2]}))
            with self.assertRaises(ValueError):
                gc.main_command("FP", p, "r")


class Calibration(unittest.TestCase):
    def test_the_calibration_command_uses_the_eighteen_ratios_and_the_pilot_split(self):
        cmd = gc.calibration_command("INT4", "repo_abc")
        for r in rt.calibration_ratios():
            self.assertIn(repr(r), cmd)
        self.assertIn("-J gain-cal-INT4", cmd); self.assertIn("--precisions 4 ", cmd + " "); self.assertIn("--split pilot --max-sentences 100", cmd); self.assertIn("--n-controls 2", cmd)
        self.assertIn("-t 03:00:00", cmd); self.assertIn("--panel", cmd)

    def test_the_two_calibration_states(self):
        self.assertIn("-J gain-cal-FP", gc.calibration_command("FP", "r"))
        with self.assertRaises(ValueError):
            gc.calibration_command("INT6", "r")


class Cli(unittest.TestCase):
    def test_main_prints_the_command(self):
        import contextlib, io
        with tempfile.TemporaryDirectory() as d:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                gc.main(["main", "--state", "INT5", "--ratios-json", str(write(d)), "--repo", "repo_abc"])
        self.assertIn("-J gain-5", out.getvalue())
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            gc.main(["calibration", "--state", "FP", "--repo", "repo_abc"])
        self.assertIn("-J gain-cal-FP", out.getvalue())


class ExtensionC(unittest.TestCase):
    def test_the_extension_command_has_the_larger_sample_and_no_zero_ablation(self):
        cmd = gc.extc_command("INT4", "repo_abc")
        self.assertIn("-J gainc-4", cmd); self.assertIn("run_gain.sbatch gainc-4 ", cmd); self.assertIn("-t 05:00:00", cmd)
        self.assertIn("--ratios 1.1 1.2 1.4 1.7", cmd); self.assertIn("--n-controls 5", cmd); self.assertIn("--no-zero", cmd)
        self.assertIn("--precisions 4 ", cmd + " "); self.assertIn("--n-pilot 450 --n-eval 562 --split eval --panel", cmd); self.assertIn("repo_abc", cmd)

    def test_the_full_precision_state_and_an_unknown_state(self):
        self.assertIn("-J gainc-FP", gc.extc_command("FP", "r")); self.assertIn("--precisions FP", gc.extc_command("FP", "r"))
        with self.assertRaises(ValueError):
            gc.extc_command("INT3", "r")

    def test_the_sample_is_never_below_the_main_study(self):
        cmd = gc.extc_command("INT6", "r")
        self.assertGreater(562, 300); self.assertIn("--n-eval 562", cmd); self.assertIn("--n-controls 5", cmd)

    def test_the_reserve_split_arguments_give_the_unused_sentences(self):
        import phase1_func_tasks as ft
        main_ids = ft.lm_ids(1012, 150, 300, 0); ext = ft.lm_ids(1012, 450, 562, 0)
        self.assertEqual(ext["eval"], main_ids["reserve"])
        self.assertFalse(set(ext["eval"]) & (set(main_ids["eval"]) | set(main_ids["pilot"])))

    def test_main_prints_the_extension_command(self):
        import contextlib, io
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            gc.main(["extc", "--state", "INT5", "--repo", "repo_abc"])
        self.assertIn("-J gainc-5", out.getvalue())

class MislabelledNode(unittest.TestCase):
    """D-079: node gnode068 is labelled 2080ti but holds at least one RTX 4070; every Study G job excludes it."""
    def test_every_printed_command_excludes_the_node(self):
        with tempfile.TemporaryDirectory() as d:
            cmds = [gc.main_command("INT4", write(d), "r"), gc.calibration_command("FP", "r"), gc.extc_command("INT4", "r")]
        for c in cmds:
            self.assertIn("--exclude=gnode068", c)
            self.assertTrue(c.startswith("sbatch --exclude=gnode068 -t "), c[:60])

    def test_the_node_list_is_a_module_constant(self):
        self.assertEqual(gc.EXCLUDED_NODES, ("gnode068",))

if __name__ == "__main__":
    unittest.main()
