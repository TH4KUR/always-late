"""Tests for scripts/g0/survival_lambda.py (Stage 1 of the survival measurement-validity study, D-117; rules in reports/survival_measurement/stage1_freeze.json).
Hand-computable cases: the lambda transform and its cut-off, the reproduction of Duan's survival share, the shift model and the decision rules."""
import math, sys, unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import survival_lambda as sl


class Lambda(unittest.TestCase):
    def test_values(self):
        lam, clipped = sl.lam(np.array([0.9, 0.6, 1 / math.sqrt(2)]))
        self.assertAlmostEqual(lam[0], math.log(1 / 0.81 - 1), places=12)
        self.assertAlmostEqual(lam[1], math.log(1 / 0.36 - 1), places=12)
        self.assertAlmostEqual(lam[2], 0.0, places=12)
        self.assertFalse(clipped.any())

    def test_cutoff_matches_duan(self):
        self.assertAlmostEqual(sl.LAMBDA_CUT, -1.4500102, places=6)   # log(0.2345679)
        r = np.array([0.8999, 0.9001])
        lam, _ = sl.lam(r)
        self.assertEqual(list(lam < sl.LAMBDA_CUT), list(r > 0.9))

    def test_clipping(self):
        lam, clipped = sl.lam(np.array([1.0, -0.3, 0.0, 0.5]))
        self.assertEqual(list(clipped), [True, True, True, False])
        self.assertTrue(np.isfinite(lam).all())
        self.assertAlmostEqual(lam[0], math.log(1 / (1 - 1e-6) ** 2 - 1), places=9)
        self.assertAlmostEqual(lam[1], math.log(1 / 0.001 ** 2 - 1), places=9)
        self.assertEqual(lam[1], lam[2])

    def test_monotone_decreasing(self):
        lam, _ = sl.lam(np.linspace(0.01, 0.99, 50))
        self.assertTrue((np.diff(lam) < 0).all())


class Reproduction(unittest.TestCase):
    def test_survival_share_and_median(self):
        corr = np.array([0.95, 0.85, 0.99, 0.2, 0.91])
        fr = np.array([0.01, 0.002, 0.0005, 0.5, 0.001])   # third below 0.001, last exactly 0.001: both inactive (strict >)
        pct, med, n = sl.survival_share(corr, fr)
        self.assertEqual(n, 3)
        self.assertAlmostEqual(pct, 100 * 1 / 3)
        self.assertAlmostEqual(med, 0.85)

    def test_median_is_lower_middle_like_torch(self):
        corr = np.array([0.1, 0.2, 0.3, 0.4]); fr = np.ones(4)
        self.assertAlmostEqual(sl.survival_share(corr, fr)[1], 0.2)   # torch.median returns the lower of the two middle values


class H1(unittest.TestCase):
    def test_pure_shift(self):
        rng = np.random.default_rng(0); a = rng.normal(size=500); b = a + 1.3
        rho, slope = sl.h1_stats(a, b, np.zeros(500, bool), np.zeros(500, bool))
        self.assertAlmostEqual(rho, 1.0); self.assertAlmostEqual(slope, 1.0)

    def test_scaled(self):
        a = np.arange(10.0); b = 2 * a
        self.assertAlmostEqual(sl.h1_stats(a, b, np.zeros(10, bool), np.zeros(10, bool))[1], 2.0)

    def test_slope_excludes_clipped(self):
        a = np.array([0.0, 1, 2, 3, 100]); b = np.array([0.0, 1, 2, 3, -100])
        ca = np.array([0, 0, 0, 0, 1], bool)
        self.assertAlmostEqual(sl.h1_stats(a, b, ca, np.zeros(5, bool))[1], 1.0)

    def test_slope_excludes_arm_clipped(self):
        a = np.array([0.0, 1, 2, 3, -100]); b = np.array([0.0, 1, 2, 3, 100])
        cb = np.array([0, 0, 0, 0, 1], bool)
        self.assertAlmostEqual(sl.h1_stats(a, b, np.zeros(5, bool), cb)[1], 1.0)

    def test_rule_boundary_rho(self):
        self.assertTrue(sl.h1_met([0.8], [1.0]))

    def test_rule(self):
        self.assertTrue(sl.h1_met([0.8, 0.9, 0.85], [1.0, 1.2, 0.81]))
        self.assertFalse(sl.h1_met([0.79, 0.7, 0.9], [1.0, 1.0, 1.0]))
        self.assertFalse(sl.h1_met([0.9, 0.9, 0.9], [1.3, 1.3, 1.0]))
        self.assertTrue(sl.h1_met([0.9], [1.25])); self.assertTrue(sl.h1_met([0.9], [0.8]))


class H2(unittest.TestCase):
    def test_exact_shift_predicts_perfectly(self):
        rng = np.random.default_rng(1)
        leg = [rng.normal(-2, 1, 300) for _ in range(4)]
        arm = [x - 0.7 for x in leg]
        out = sl.shift_model(leg, arm)
        self.assertAlmostEqual(out["delta"], -0.7)
        self.assertAlmostEqual(out["mae"], 0.0)
        self.assertGreater(out["mae_no_shift"], 0.0)

    def test_hand_case(self):
        leg = [np.array([-3.0, -1.0, 0.0, 1.0])]; arm = [np.array([-4.0, -2.0, -1.0, 0.0])]
        out = sl.shift_model(leg, arm)
        self.assertAlmostEqual(out["delta"], -1.0)
        self.assertAlmostEqual(out["predicted"][0], 50.0)    # -4, -2 below -1.4499
        self.assertAlmostEqual(out["observed"][0], 50.0)
        self.assertAlmostEqual(out["mae_no_shift"], 25.0)    # legacy share 25

    def test_no_shift_reference(self):
        out = sl.shift_model([np.array([-2.0, -1.0])], [np.array([-2.0, -1.0])])
        self.assertAlmostEqual(out["mae_no_shift"], 0.0)   # -2.0 survives (lambda < -1.45)

    def test_rule(self):
        self.assertTrue(sl.h2_met(2.0)); self.assertFalse(sl.h2_met(2.01))


class H3(unittest.TestCase):
    def test_rule(self):
        self.assertTrue(sl.h3_lambda_met(-0.51)); self.assertFalse(sl.h3_lambda_met(0.5))

    def test_r2_perfect(self):
        x = np.random.default_rng(2).normal(size=(50, 3)); y = x @ np.array([1.0, -2, 0.5]) + 3
        self.assertAlmostEqual(sl.ols_r2(y, x), 1.0)


if __name__ == "__main__":
    unittest.main()
