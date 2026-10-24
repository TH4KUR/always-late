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


if __name__ == "__main__":
    unittest.main()
