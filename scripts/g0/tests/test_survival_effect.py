"""Tests for scripts/g0/survival_effect.py (Stage 3 of the survival-and-intervention plan; freeze reports/survival_effect_rescue/stage3_analysis_freeze_2026-10-28.md): quantile transform,
partial Spearman, ridge with a grouped lambda choice, leave-one-language-out prediction, the cluster bootstrap of the improvement, the strata and the decision rules. Hand-computable cases; numpy only."""
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import survival_effect as se  # noqa: E402


class Transform(unittest.TestCase):
    def test_quantile_transform_uses_the_training_ecdf_with_mid_ranks_for_ties(self):
        q = se.quantile_transform([1, 2, 3, 4], [0, 1, 2.5, 5])
        np.testing.assert_allclose(q, [0.0, 0.125, 0.5, 1.0])
        np.testing.assert_allclose(se.quantile_transform([1, 1, 1, 1], [1]), [0.5])

    def test_nan_is_filled_with_the_training_median_before_the_transform(self):
        tr = np.array([[1.0], [2.0], [3.0], [np.nan]]); te = np.array([[np.nan], [3.0]])
        a, b = se.prepare(tr, te)
        # training median of [1,2,3] is 2: the filled training column is [1,2,3,2]; the test NaN becomes 2
        np.testing.assert_allclose(a[:, 0], se.quantile_transform([1, 2, 3, 2], [1, 2, 3, 2]))
        np.testing.assert_allclose(b[:, 0], se.quantile_transform([1, 2, 3, 2], [2, 3]))


class Association(unittest.TestCase):
    def test_partial_spearman_removes_a_shared_cause(self):
        rng = np.random.default_rng(0); z = rng.normal(size=3000); x = z + 0.3 * rng.normal(size=3000); y = z + 0.3 * rng.normal(size=3000)
        raw = se.met.spearman(x, y)["rho"]; part = se.partial_spearman(x, y, z[:, None])
        self.assertGreater(raw, 0.8); self.assertLess(abs(part), 0.1)

    def test_partial_spearman_keeps_a_relation_that_the_covariate_does_not_explain(self):
        rng = np.random.default_rng(1); z = rng.normal(size=3000); x = rng.normal(size=3000); y = x + 0.5 * z + 0.2 * rng.normal(size=3000)
        self.assertGreater(se.partial_spearman(x, y, z[:, None]), 0.8)

    def test_partial_spearman_is_nan_without_variation(self):
        z = np.arange(10.0)[:, None]; self.assertTrue(np.isnan(se.partial_spearman(z[:, 0], 2 * z[:, 0], z)))


class Ridge(unittest.TestCase):
    def test_noiseless_linear_target_is_recovered(self):
        rng = np.random.default_rng(2); X = rng.uniform(size=(200, 2)); y = 2 * X[:, 0] - X[:, 1] + 1
        m = se.ridge_fit(X, y, 1e-9); np.testing.assert_allclose(se.ridge_predict(m, X), y, atol=1e-6)

    def test_a_large_penalty_shrinks_to_the_mean(self):
        rng = np.random.default_rng(3); X = rng.uniform(size=(100, 2)); y = X[:, 0] + 5
        m = se.ridge_fit(X, y, 1e9); np.testing.assert_allclose(se.ridge_predict(m, X), np.full(100, y.mean()), atol=1e-3)


class Folds(unittest.TestCase):
    def test_every_group_lies_in_exactly_one_fold_and_folds_are_balanced_by_group_count(self):
        groups = np.repeat(np.arange(10), 3); f = se.grouped_folds(groups, 5)
        for g in range(10):
            self.assertEqual(len(set(f[groups == g])), 1)
        self.assertEqual(sorted(np.bincount(f)), [6] * 5)

    def test_lambda_choice_ties_go_to_the_larger_penalty(self):
        X = np.random.default_rng(4).uniform(size=(60, 2)); y = np.zeros(60); groups = np.repeat(np.arange(6), 10)
        self.assertEqual(se.choose_lambda(X, y, groups, se.GRID, 5), max(se.GRID))

    def test_lambda_choice_prefers_a_small_penalty_for_a_noiseless_signal(self):
        rng = np.random.default_rng(5); X = rng.uniform(size=(300, 2)); y = 3 * X[:, 0] - 2 * X[:, 1]; groups = np.repeat(np.arange(15), 20)
        self.assertEqual(se.choose_lambda(X, y, groups, se.GRID, 5), min(se.GRID))


class LeaveLanguageOut(unittest.TestCase):
    def data(self):
        rng = np.random.default_rng(6); lang = np.repeat(np.arange(6), 40); X = rng.uniform(size=(240, 2)); y = 2 * X[:, 0] - X[:, 1] + 1
        return X, y, lang

    def test_a_noiseless_signal_is_predicted_for_unseen_languages(self):
        X, y, lang = self.data(); pred, _ = se.loo_language(X, y, lang)
        self.assertLess(np.mean(np.abs(pred - y)), 0.05)

    def test_the_held_out_language_is_not_used_in_its_own_fit(self):
        X, y, lang = self.data(); y = y.copy(); y[lang == 0] = 100.0   # an outlier language that no model of the other languages can know
        pred, _ = se.loo_language(X, y, lang)
        self.assertGreater(np.min(np.abs(pred[lang == 0] - 100.0)), 50.0)

    def test_no_call_sees_the_held_out_language_in_training_or_in_the_choice_of_lambda(self):
        X, y, lang = self.data(); X = np.column_stack([lang, X])   # the first column is a marker of the language
        lam_calls, fit_overlaps = [], []
        real_cl, real_fp = se.choose_lambda, se.fit_predict
        def cl(Xa, ya, groups, grid=se.GRID, k=5):
            lam_calls.append(len(np.unique(groups))); return real_cl(Xa, ya, groups, grid, k)
        def fp(Xtr, ytr, Xte, lam):
            fit_overlaps.append(set(Xtr[:, 0].tolist()) & set(Xte[:, 0].tolist())); return real_fp(Xtr, ytr, Xte, lam)
        with mock.patch.object(se, "choose_lambda", cl), mock.patch.object(se, "fit_predict", fp):
            se.loo_language(X, y, lang)
        self.assertEqual(set(lam_calls), {5})                                  # lambda is chosen on the five other languages only
        self.assertTrue(fit_overlaps and all(len(o) == 0 for o in fit_overlaps))   # no fit is ever tested on a language it was trained on

    def test_the_null_prediction_is_the_training_median_of_the_other_languages(self):
        y = np.array([1.0, 1.0, 5.0, 5.0, 9.0, 9.0]); lang = np.array([0, 0, 1, 1, 2, 2])
        np.testing.assert_allclose(se.loo_null(y, lang), [7.0, 7.0, 5.0, 5.0, 3.0, 3.0])


class Improvement(unittest.TestCase):
    def test_constant_errors_give_a_degenerate_interval(self):
        a = np.full(60, 2.0); b = np.full(60, 1.0); lang = np.repeat(np.arange(6), 10)
        r = se.improvement(a, b, lang, n_boot=200, seed=0)
        self.assertAlmostEqual(r["improvement"], 0.5); self.assertAlmostEqual(r["lo"], 0.5); self.assertAlmostEqual(r["hi"], 0.5)

    def test_an_improvement_in_only_some_languages_has_an_interval_that_reaches_zero(self):
        lang = np.repeat(np.arange(6), 10); a = np.ones(60); b = np.ones(60); b[lang < 2] = 0.2
        r = se.improvement(a, b, lang, n_boot=500, seed=0)
        self.assertGreater(r["improvement"], 0.2); self.assertLessEqual(r["lo"], r["improvement"]); self.assertGreaterEqual(r["hi"], r["improvement"])
        self.assertLess(r["lo"], 0.1)


class Rules(unittest.TestCase):
    def test_the_cell_rule_is_ten_percent_and_a_positive_lower_bound(self):
        self.assertTrue(se.cell_met(0.10, 0.0001)); self.assertFalse(se.cell_met(0.0999, 0.05)); self.assertFalse(se.cell_met(0.30, 0.0)); self.assertFalse(se.cell_met(0.30, -0.01))
        self.assertFalse(se.cell_met(float("nan"), 0.1)); self.assertFalse(se.cell_met(0.3, float("nan")))

    def test_the_overall_statement_needs_all_six_cells_for_value_and_none_for_no_value(self):
        self.assertEqual(se.overall_statement([True] * 6), "adds predictive value")
        self.assertEqual(se.overall_statement([False] * 6), "no clear predictive value beyond baselines")
        self.assertEqual(se.overall_statement([True, False, True, True, True, True]), "mixed")
        self.assertEqual(se.overall_statement([False, False, False, False, False, True]), "mixed")

    def test_strata_use_greater_or_equal_and_the_reference_noise_percentile(self):
        c16 = np.array([0.9, 0.94888916015625, 0.99]); np.testing.assert_array_equal(se.reliable_mask(c16, 0.94888916015625), [False, True, True])
        np.testing.assert_array_equal(se.loss_mask(np.array([-0.002, 0.0014, 0.0015]), 0.0015), [True, False, True])
        d_fp = np.zeros(101); d16 = np.arange(101) / 1000.0
        self.assertAlmostEqual(se.reference_noise(d_fp, d16), 0.095)   # 95th percentile (linear) of |d16 - d_fp| = 0.000 ... 0.100


if __name__ == "__main__":
    unittest.main()
