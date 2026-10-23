import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phase1_func_f5_metrics as m


def acc_run(x, y, batch=7):
    a = m.ReadAccumulator(x.shape[1])
    for i in range(0, len(x), batch):
        a.add(x[i:i + batch], y[i:i + batch])
    return a.finish()


class ReadSide(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.x = np.maximum(rng.normal(0, 1, (200, 5)), 0) * (rng.random((200, 5)) < 0.5)               # sparse non-negative activations
        self.y = np.maximum(self.x * 0.7 + rng.normal(0, 0.1, (200, 5)), 0) * (self.x > 0)

    def test_the_streaming_correlation_equals_the_one_shot_correlation_whatever_the_batch_size(self):
        want = np.array([np.corrcoef(self.x[:, j], self.y[:, j])[0, 1] for j in range(5)])
        for batch in (1, 7, 200):
            np.testing.assert_allclose(acc_run(self.x, self.y, batch)["corr"], want, atol=1e-12)

    def test_identical_activations_give_correlation_jaccard_and_scale_exactly_one(self):
        r = acc_run(self.x, self.x.copy())
        np.testing.assert_allclose(r["corr"], 1.0, atol=1e-12); np.testing.assert_array_equal(r["jaccard"], 1.0); np.testing.assert_allclose(r["scale"], 1.0, atol=1e-12)

    def test_jaccard_is_the_overlap_of_the_firing_sets(self):
        x = np.array([[1.0], [1.0], [1.0], [0.0], [0.0]]); y = np.array([[1.0], [1.0], [0.0], [1.0], [0.0]])      # active sets {0,1,2} and {0,1,3}: 2 of 4
        self.assertAlmostEqual(float(acc_run(x, y, 2)["jaccard"][0]), 0.5)
        self.assertEqual(float(acc_run(np.array([[1.0], [0.0]]), np.array([[0.0], [1.0]]))["jaccard"][0]), 0.0)    # disjoint

    def test_the_scale_ratio_uses_only_the_positions_active_in_both(self):
        x = np.array([[2.0], [4.0], [10.0], [0.0]]); y = np.array([[1.0], [2.0], [0.0], [50.0]])               # both active at the first two positions: y is half of x there
        r = acc_run(x, y, 3)
        self.assertAlmostEqual(float(r["scale"][0]), 0.5); self.assertEqual(int(r["n_both"][0]), 2)

    def test_undefined_values_are_nan_and_counted_never_zero(self):
        x = np.array([[1.0, 0.0, 3.0], [2.0, 0.0, 3.0], [3.0, 0.0, 3.0]]); y = np.array([[1.0, 0.0, 1.0], [2.0, 0.0, 2.0], [3.0, 0.0, 5.0]])
        r = acc_run(x, y)
        self.assertTrue(np.isnan(r["corr"][2]))                                                              # zero variance in x
        self.assertTrue(np.isnan(r["corr"][1]) and np.isnan(r["jaccard"][1]) and np.isnan(r["scale"][1]))     # a feature that never fires
        self.assertEqual(int(r["n_active_fp"][1]), 0)
        self.assertTrue(np.isnan(acc_run(np.array([[1.0], [0.0]]), np.array([[0.0], [1.0]]))["scale"][0]))   # no position active in both

    def test_the_counts_are_returned(self):
        r = acc_run(self.x, self.y)
        np.testing.assert_array_equal(r["n_active_fp"], (self.x > 0).sum(0)); np.testing.assert_array_equal(r["n_active_q"], (self.y > 0).sum(0)); self.assertEqual(r["n"], 200)

    def test_shapes_must_match(self):
        with self.assertRaises(ValueError):
            m.ReadAccumulator(3).add(np.zeros((4, 3)), np.zeros((4, 2)))
        with self.assertRaises(ValueError):
            m.ReadAccumulator(3).add(np.zeros((4, 4)), np.zeros((4, 4)))


class WriteSide(unittest.TestCase):
    def test_cosine_of_identical_orthogonal_and_opposite_fingerprints(self):
        U = np.array([[1.0, 0, 0], [1.0, 0, 0], [1.0, 2.0, 3.0]]); V = np.array([[2.0, 0, 0], [0, 5.0, 0], [-1.0, -2.0, -3.0]])
        sim, flags = m.cosine_rows(U, V)
        np.testing.assert_allclose(sim, [1.0, 0.0, -1.0], atol=1e-12); self.assertFalse(flags["zero_fp"].any()); self.assertFalse(flags["zero_q"].any())

    def test_no_effect_in_the_quantized_model_is_similarity_zero_and_flagged_but_no_effect_in_fp_is_undefined(self):
        U = np.array([[1.0, 2.0], [0.0, 0.0], [1.0, 2.0]]); V = np.array([[0.0, 0.0], [1.0, 2.0], [1.0, 2.0]])
        sim, flags = m.cosine_rows(U, V)
        self.assertEqual(sim[0], 0.0); self.assertTrue(flags["zero_q"][0] and not flags["zero_fp"][0])        # f does nothing in the quantized model
        self.assertTrue(np.isnan(sim[1]) and flags["zero_fp"][1])                                              # f has no effect to compare with
        self.assertAlmostEqual(float(sim[2]), 1.0)

    def test_the_cosine_does_not_depend_on_the_scale_of_either_fingerprint(self):
        rng = np.random.default_rng(1); U = rng.normal(size=(4, 9)); V = rng.normal(size=(4, 9))
        np.testing.assert_allclose(m.cosine_rows(U * 7.0, V * 0.01)[0], m.cosine_rows(U, V)[0], atol=1e-12)

    def test_topk_overlap_counts_the_shared_most_promoted_tokens(self):
        U = np.array([[9.0, 8.0, 1.0, 0.0, -5.0]]); V = np.array([[9.0, 0.0, 8.5, 1.0, -5.0]])               # top 2 of U: {0,1}; of V: {0,2}
        self.assertAlmostEqual(float(m.topk_overlap(U, V, 2)[0]), 0.5)
        self.assertAlmostEqual(float(m.topk_overlap(U, V, 1)[0]), 1.0); self.assertAlmostEqual(float(m.topk_overlap(U, U, 3)[0]), 1.0)
        self.assertAlmostEqual(float(m.topk_overlap(U, -U, 1)[0]), 0.0)                                         # promoted means the largest values, not the largest magnitudes

    def test_topk_overlap_of_a_zero_fingerprint_is_undefined_and_k_must_fit(self):
        self.assertTrue(np.isnan(m.topk_overlap(np.zeros((1, 4)), np.ones((1, 4)), 2)[0])); self.assertTrue(np.isnan(m.topk_overlap(np.ones((1, 4)), np.zeros((1, 4)), 2)[0]))
        with self.assertRaises(ValueError):
            m.topk_overlap(np.ones((1, 4)), np.ones((1, 4)), 5)

    def test_the_magnitude_ratio_is_quantized_over_full_precision(self):
        U = np.array([[3.0, 4.0], [0.0, 0.0]]); V = np.array([[6.0, 8.0], [1.0, 1.0]])
        r = m.magnitude_ratio(U, V); self.assertAlmostEqual(float(r[0]), 2.0); self.assertTrue(np.isnan(r[1]))


class Retrieval(unittest.TestCase):
    def test_a_diagonal_dominant_similarity_matrix_finds_every_identity(self):
        S = np.eye(4) + 0.1; r = m.retrieval(S)
        self.assertEqual((r["top1"], r["mrr"], r["n_queries"], r["n_ties"]), (1.0, 1.0, 4, 0))

    def test_a_permuted_identity_is_never_found_and_the_reciprocal_rank_follows_the_rank(self):
        S = np.array([[0.1, 0.9, 0.5], [0.9, 0.1, 0.5], [0.5, 0.6, 0.2]])                                       # true candidates on the diagonal
        r = m.retrieval(S)
        self.assertEqual(r["top1"], 0.0); self.assertAlmostEqual(r["mrr"], (1 / 3 + 1 / 3 + 1 / 3) / 3)

    def test_the_rank_of_the_true_candidate_counts_the_candidates_that_beat_it(self):
        S = np.array([[0.5, 0.9, 0.7, 0.1]]); r = m.retrieval(S, true=np.array([0]))                            # true candidate 0: two candidates beat it
        self.assertAlmostEqual(r["mrr"], 1 / 3); self.assertEqual(r["top1"], 0.0)

    def test_ties_are_resolved_against_the_true_candidate_and_counted(self):
        S = np.array([[0.5, 0.5, 0.1], [0.2, 0.9, 0.1]])
        r = m.retrieval(S, true=np.array([0, 1]))
        self.assertEqual(r["top1"], 0.5); self.assertEqual(r["n_ties"], 1); self.assertAlmostEqual(r["mrr"], (1 / 2 + 1) / 2)    # the tied candidate counts as ahead

    def test_queries_without_a_defined_true_similarity_are_excluded_and_counted(self):
        S = np.array([[np.nan, 0.1], [0.1, 0.9]]); r = m.retrieval(S)
        self.assertEqual((r["n_queries"], r["n_undefined"], r["top1"]), (1, 1, 1.0))

    def test_within_group_retrieval_only_competes_with_the_same_group_and_reports_the_chance_level(self):
        S = np.array([[0.9, 0.1, 0.95, 0.0], [0.1, 0.9, 0.0, 0.0], [0.5, 0.0, 0.9, 0.2], [0.0, 0.0, 0.1, 0.9]])
        g = np.array([0, 0, 1, 1]); r = m.retrieval(S, groups=g)
        self.assertEqual(r["top1"], 0.75)                                                                       # globally query 0 loses to candidate 2 (0.95 against 0.9)
        w = r["within"]; self.assertEqual((w["top1"], w["n_queries"]), (1.0, 4)); self.assertAlmostEqual(w["chance"], 0.5)

    def test_a_group_of_one_has_nothing_to_compete_with_and_is_excluded_from_the_within_group_numbers(self):
        S = np.eye(3); w = m.retrieval(S, groups=np.array([0, 0, 1]))["within"]
        self.assertEqual((w["n_queries"], w["n_singletons"]), (2, 1))

    def test_shapes_and_true_indices_are_checked(self):
        with self.assertRaises(ValueError):
            m.retrieval(np.ones((2, 3)))                                                                        # no diagonal: the true candidates must be given
        with self.assertRaises(ValueError):
            m.retrieval(np.ones((2, 3)), true=np.array([0, 5]))


class Validity(unittest.TestCase):
    def test_the_floor_is_a_low_percentile_of_the_reference_ignoring_undefined_values(self):
        ref = np.concatenate([np.arange(1, 101, dtype=float), [np.nan] * 5])
        self.assertAlmostEqual(m.floor_threshold(ref, 0.05), float(np.percentile(np.arange(1, 101), 5)))
        self.assertGreater(m.floor_threshold(ref, 0.5), m.floor_threshold(ref, 0.05))

    def test_the_share_above_the_floor_counts_defined_and_total(self):
        s = m.share_above(np.array([0.9, 0.5, np.nan, 0.95, 0.2]), 0.6)
        self.assertEqual((s["share"], s["n_defined"], s["n_total"]), (0.5, 4, 5))
        self.assertTrue(np.isnan(m.share_above(np.array([np.nan]), 0.5)["share"]))
        self.assertEqual(m.share_above(np.array([0.6, 0.6, 0.5]), 0.6)["share"], 2 / 3)                         # a value exactly at the floor counts as above it

    def test_auroc_of_a_perfect_reversed_and_uninformative_score(self):
        ev = np.array([1, 1, 0, 0])
        self.assertEqual(m.auroc(np.array([0.9, 0.8, 0.2, 0.1]), ev)["auroc"], 1.0); self.assertEqual(m.auroc(np.array([0.1, 0.2, 0.8, 0.9]), ev)["auroc"], 0.0)
        self.assertEqual(m.auroc(np.array([1, 1, 1, 1]), ev)["auroc"], 0.5)

    def test_auroc_counts_ties_as_half_and_equals_the_pairwise_definition(self):
        rng = np.random.default_rng(3); score = rng.integers(0, 4, 60).astype(float); ev = rng.random(60) < 0.4
        pos, neg = score[ev], score[~ev]
        want = np.mean([(1.0 if p > q else 0.5 if p == q else 0.0) for p in pos for q in neg])
        self.assertAlmostEqual(m.auroc(score, ev)["auroc"], want, places=12)

    def test_auroc_of_a_binary_status_by_hand(self):
        status = np.array([1] * 6 + [0] * 4, dtype=float); ev = np.array([1, 1, 1, 1, 0, 0, 1, 0, 0, 0])
        # event rows 0,1,2,3,6 have status 1,1,1,1,0 (P(status=1 | event) = 0.8); non-event rows 4,5,7,8,9 have status 1,1,0,0,0 (0.4)
        # AUROC = P(win) + 0.5 P(tie) = 0.8 * 0.6 + 0.5 * (0.8 * 0.4 + 0.2 * 0.6) = 0.48 + 0.22
        self.assertAlmostEqual(m.auroc(status, ev)["auroc"], 0.70, places=12)

    def test_auroc_ignores_undefined_pairs_and_reports_the_counts_and_needs_both_classes(self):
        r = m.auroc(np.array([0.9, np.nan, 0.1, 0.5]), np.array([1, 1, 0, 0]))
        self.assertEqual((r["auroc"], r["n_pos"], r["n_neg"], r["n_undefined"]), (1.0, 1, 2, 1))
        self.assertTrue(np.isnan(m.auroc(np.array([0.1, 0.2]), np.array([1, 1]))["auroc"]))

    def test_spearman_is_one_for_a_monotone_relation_and_handles_ties_and_undefined_pairs(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, np.nan]); y = np.array([10.0, 20.0, 35.0, 50.0, 1.0])
        r = m.spearman(x, y); self.assertAlmostEqual(r["rho"], 1.0); self.assertEqual(r["n"], 4)
        self.assertAlmostEqual(m.spearman(np.array([1.0, 2, 3, 4]), np.array([4.0, 3, 2, 1]))["rho"], -1.0)
        from scipy.stats import spearmanr
        a = np.array([1.0, 1, 2, 3, 3, 4]); b = np.array([2.0, 1, 1, 4, 5, 5]); self.assertAlmostEqual(m.spearman(a, b)["rho"], float(spearmanr(a, b)[0]), places=12)


class ClusterBootstrap(unittest.TestCase):
    def test_whole_clusters_are_resampled_not_rows(self):
        clusters = np.array(["a"] * 3 + ["b"] * 5); sizes = []
        m.cluster_bootstrap(clusters, lambda idx: sizes.append(len(idx)) or 0.0, n_boot=200, seed=0)
        self.assertTrue(set(sizes) <= {6, 8, 10})                                                               # two draws of clusters of 3 and 5 rows: 3+3, 3+5, 5+5
        self.assertEqual(len(set(sizes)), 3)

    def test_a_cluster_drawn_twice_contributes_all_its_rows_twice(self):
        clusters = np.array([0, 0, 1, 1]); seen = []
        m.cluster_bootstrap(clusters, lambda idx: seen.append(tuple(sorted(idx))) or 0.0, n_boot=100, seed=1)
        self.assertTrue({(0, 0, 1, 1), (2, 2, 3, 3), (0, 1, 2, 3)} >= set(seen))

    def test_the_interval_brackets_the_point_for_a_symmetric_planted_effect_and_is_reproducible(self):
        rng = np.random.default_rng(2); clusters = np.repeat(np.arange(20), 10); vals = rng.normal(0.5, 1, 200) + np.repeat(rng.normal(0, 0.3, 20), 10)
        point, lo, hi = m.cluster_bootstrap(clusters, lambda idx: float(vals[idx].mean()), n_boot=500, seed=0)
        self.assertAlmostEqual(point, float(vals.mean())); self.assertLess(lo, point); self.assertGreater(hi, point)
        self.assertEqual((point, lo, hi), m.cluster_bootstrap(clusters, lambda idx: float(vals[idx].mean()), n_boot=500, seed=0))

    def test_identical_clusters_give_a_zero_width_interval_and_one_cluster_is_an_error(self):
        clusters = np.repeat(np.arange(5), 4); vals = np.tile([1.0, 2.0, 3.0, 4.0], 5)
        point, lo, hi = m.cluster_bootstrap(clusters, lambda idx: float(vals[idx].mean()), n_boot=100, seed=0)
        self.assertEqual((lo, hi), (point, point))
        with self.assertRaises(ValueError):
            m.cluster_bootstrap(np.zeros(5, dtype=int), lambda idx: 0.0, n_boot=10)

    def test_a_statistic_that_is_undefined_in_some_draws_is_skipped_and_counted(self):
        clusters = np.repeat(np.arange(6), 2); calls = {"n": 0}
        def fn(idx):
            calls["n"] += 1
            return float("nan") if calls["n"] % 3 == 0 else 1.0
        r = m.cluster_bootstrap(clusters, fn, n_boot=30, seed=0, return_counts=True)
        self.assertEqual(r["n_boot"], 30); self.assertGreater(r["n_undefined"], 0); self.assertLess(r["n_undefined"], 30)

    def test_the_median_difference_is_taken_per_cluster_between_sets_of_different_size_then_across_clusters(self):
        a = np.array([5.0, 6.0, 7.0, 1.0, 2.0, 9.0, 0.0, 0.0, 0.0]); ca = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        b = np.array([1.0, 2.0, 3.0, 5.0, 3.0, 4.0, 2.0, 1.0, 1.0, 1.0]); cb = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])         # cluster 1 has four values on the b side
        d = m.paired_median_difference(a, ca, b, cb)
        self.assertEqual(d["by_cluster"], {0: 4.0, 1: -1.5, 2: -1.0})                                          # medians 6 against 2; 2 against 3.5; 0 against 1
        self.assertEqual(d["median"], -1.0)                                                                    # the median of the cluster differences (their mean would be 0.5)

    def test_a_cluster_present_on_one_side_only_or_without_defined_values_is_left_out(self):
        a = np.array([1.0, 2.0, np.nan, np.nan]); ca = np.array([0, 0, 1, 1]); b = np.array([0.0, 0.0, 5.0, 6.0, 7.0]); cb = np.array([0, 0, 1, 1, 2])
        d = m.paired_median_difference(a, ca, b, cb)
        self.assertEqual(list(d["by_cluster"]), [0]); self.assertEqual(d["median"], 1.5)                         # cluster 1 has no defined a, cluster 2 no a at all


class MatchControls(unittest.TestCase):
    """The matched different feature: same language, nearest native rate on the log scale (shared by the negative control of the analysis and the direction-swap control of the runner)."""
    def test_the_match_is_the_same_language_feature_with_the_nearest_log_rate(self):
        rate = np.array([1e-3, 2e-3, 1.1e-3, 5e-3, 5e-3, 3e-3]); lang = np.array([0, 0, 0, 1, 1, 2])
        self.assertEqual(m.match_controls(lang, rate).tolist(), [2, 2, 0, 4, 3, -1])

    def test_the_log_scale_decides_not_the_linear_distance(self):
        self.assertEqual(m.match_controls(np.zeros(3, dtype=int), np.array([10.0, 4.0, 20.0]))[0], 2)

    def test_ties_go_to_the_smaller_index_and_a_feature_never_matches_itself(self):
        self.assertEqual(m.match_controls(np.zeros(4, dtype=int), np.full(4, 1e-3)).tolist(), [1, 0, 0, 0])

    def test_a_feature_without_a_positive_rate_has_no_match_and_is_not_anyones_match(self):
        r = m.match_controls(np.zeros(3, dtype=int), np.array([0.0, 1e-3, 2e-3])); self.assertEqual(r.tolist(), [-1, 2, 1])

    def test_matches_never_cross_languages(self):
        lang = np.array([0, 1, 0, 1]); r = m.match_controls(lang, np.array([1.0, 1.0, 5.0, 5.0]))
        self.assertTrue(all(lang[i] == lang[j] for i, j in enumerate(r) if j >= 0)); self.assertEqual(r.tolist(), [2, 3, 0, 1])

if __name__ == "__main__":
    unittest.main()
