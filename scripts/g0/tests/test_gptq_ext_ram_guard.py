"""Task 3 of the Gemma-2-9B three-arm plan: a pre-job estimate of the host RAM a weight-set job needs (the full-precision CPU copy of the model, the calibrator's host copy of the original target matrices, and one
artifact per bit width; each artifact is the target matrices in bf16, 17.6 GB for Gemma-2-9B) and a check against the RAM that is there. Stdlib only."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gptq_ext import ram_guard as rg


class Estimate(unittest.TestCase):
    def test_estimate_for_one_artifact_is_about_58_gb_with_the_9b_numbers(self):
        e = rg.estimate_gb(18.5, 17.6, 1)
        self.assertTrue(55 <= e <= 60, e)

    def test_two_artifacts_need_exactly_one_more_artifact_plus_margin(self):
        one, two = rg.estimate_gb(18.5, 17.6, 1), rg.estimate_gb(18.5, 17.6, 2)
        self.assertAlmostEqual(two - one, 17.6 * 1.1, places=9)

    def test_the_margin_scales_the_whole_estimate(self):
        self.assertAlmostEqual(rg.estimate_gb(10, 5, 2, margin=1.0), 10 + 5 + 10)
        self.assertAlmostEqual(rg.estimate_gb(10, 5, 2, margin=2.0), 50)

    def test_zero_artifacts_is_a_legacy_rtn_job_without_the_artifact_terms(self):
        self.assertAlmostEqual(rg.estimate_gb(18.5, 17.6, 0, margin=1.0), 18.5 + 0.0)


class Check(unittest.TestCase):
    def test_check_returns_the_estimate_when_it_fits(self):
        self.assertAlmostEqual(rg.check(160, 18.5, 17.6, 1), rg.estimate_gb(18.5, 17.6, 1))

    def test_check_raises_with_both_numbers_in_the_message(self):
        with self.assertRaises(rg.RamShortfall) as ctx:
            rg.check(50, 18.5, 17.6, 1)
        msg = str(ctx.exception)
        self.assertIn("59.1", msg); self.assertIn("50", msg)

    def test_three_artifacts_do_not_fit_160_gb_with_two_processes_but_one_artifact_does(self):
        self.assertLessEqual(2 * rg.estimate_gb(18.5, 17.6, 1), 160)
        self.assertGreater(2 * rg.estimate_gb(18.5, 17.6, 3), 160)

    def test_the_command_line_exits_4_on_a_shortfall_and_0_otherwise(self):
        self.assertEqual(rg.main(["--model-gb", "18.5", "--target-gb", "17.6", "--n-artifacts", "1", "--ram-gb", "160"]), 0)
        self.assertEqual(rg.main(["--model-gb", "18.5", "--target-gb", "17.6", "--n-artifacts", "3", "--ram-gb", "80"]), 4)


if __name__ == "__main__":
    unittest.main()
