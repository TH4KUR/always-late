"""D-035 Task 6 robustness views (scripts/g0/phase1_association_robustness.py), descriptive and exploratory: (1) how the established and lost counts change with the support rule (minimum
positive tokens at the peak), (2) how the loss of the association relates to baseline properties of a feature (target share, margin, firing rate, activation scale): loss fraction by
quantile bin and a rank correlation. Pure-function tests on synthetic arrays, an integration test on a collector sidecar, and the CLI."""
import json, shutil, sys, tempfile, unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import phase1_association_analysis as an   # noqa: E402
import phase1_association_collect as ac   # noqa: E402
import phase1_association_robustness as rb   # noqa: E402
import phase1_run as run   # noqa: E402
from test_phase1_run import F, BOS, PAD   # noqa: E402
from test_phase1_association_collect import make_planted, CODES   # noqa: E402
from test_phase1_association_analysis import write_baseline   # noqa: E402


class Bins(unittest.TestCase):
    def test_quantile_bins_partition_the_masked_values_only(self):
        x = np.arange(100, dtype=float); mask = x >= 20
        b = rb.quantile_bins(x, mask, 4)
        self.assertTrue(np.all(b[~mask] == -1)); self.assertEqual(sorted(set(b[mask].tolist())), [0, 1, 2, 3])
        self.assertEqual([int((b == k).sum()) for k in range(4)], [20, 20, 20, 20])

    def test_binned_loss_reports_counts_and_fractions_per_bin(self):
        x = np.arange(40, dtype=float); established = np.ones(40, dtype=bool); lost = x < 10                  # loss concentrated in the lowest quarter
        t = rb.binned_loss(x, established, lost, 4)
        self.assertEqual([r["n"] for r in t], [10, 10, 10, 10]); self.assertEqual([r["lost"] for r in t], [10, 0, 0, 0]); self.assertEqual(t[0]["loss_fraction"], 1.0)
        self.assertEqual((t[0]["lo"], t[-1]["hi"]), (0.0, 39.0))

    def test_features_not_established_are_left_out(self):
        x = np.arange(40, dtype=float); established = x >= 10; lost = x < 20
        t = rb.binned_loss(x, established, lost, 3)
        self.assertEqual(sum(r["n"] for r in t), 30); self.assertEqual(sum(r["lost"] for r in t), 10)

    def test_a_constant_covariate_gives_one_bin_and_few_values_fewer_bins(self):
        t = rb.binned_loss(np.zeros(20), np.ones(20, dtype=bool), np.arange(20) < 5, 10)
        self.assertEqual(len(t), 1); self.assertEqual(t[0]["n"], 20); self.assertEqual(t[0]["lost"], 5)
        t2 = rb.binned_loss(np.array([1.0, 1.0, 2.0, 2.0] * 5), np.ones(20, dtype=bool), np.zeros(20, dtype=bool), 10)
        self.assertLessEqual(len(t2), 2)

    def test_rank_correlation_of_a_monotone_relation_and_the_empty_case(self):
        x = np.arange(100, dtype=float); lost = x < 30; est = np.ones(100, dtype=bool)
        self.assertLess(rb.rank_correlation(x, lost, est), -0.5)
        self.assertGreater(rb.rank_correlation(-x, lost, est), 0.5)
        self.assertTrue(np.isnan(rb.rank_correlation(x, np.zeros(100, dtype=bool), est)))                  # nobody lost: undefined, not zero

    def test_only_established_features_enter_the_correlation(self):
        x = np.arange(100, dtype=float); est = x >= 50; lost = (x < 50) | (x >= 90)                       # among the established, loss sits at the high end; over everybody it would not
        self.assertGreater(rb.rank_correlation(x, lost, est), 0.2)

    def test_nan_covariates_are_excluded(self):
        x = np.arange(30, dtype=float); x[:5] = np.nan; est = np.ones(30, dtype=bool); lost = np.arange(30) % 2 == 0
        t = rb.binned_loss(x, est, lost, 3); self.assertEqual(sum(r["n"] for r in t), 25)


class Integration(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def make(self):
        g = torch.Generator().manual_seed(3)
        data = {l: [torch.randint(3 + 7 * l, 10 + 7 * l, (n,), generator=g).tolist() for n in (4, 6, 5, 3)] for l in range(3)}
        sink = ac.ProfileSink(self.tmp / "run", CODES, F, conditions=["FP", "INT4", "INT0"], sentence_features=torch.arange(F), streams=("all",))
        fwd, enc, qz, rs = make_planted(); run.run_survival(fwd, enc, qz, rs, data, "mask", BOS, PAD, F, [4, 0], profile_sink=sink); sink.finalize()
        sel = np.zeros(F, dtype=bool); sel[:3] = True; peak = np.zeros(F, dtype=int); peak[:3] = [0, 1, 2]
        return self.tmp / "run", write_baseline(self.tmp / "base", peak, sel, [3, 4])

    def test_threshold_sensitivity_changes_established_counts_with_the_support_rule(self):
        d, base = self.make(); side = an.load_aggregate_sidecar(d, "all"); coh = an.load_cohort(base, side)
        out = rb.threshold_sensitivity(side, coh, thresholds=(5, 18, 19))                                  # every planted feature fires on all 18 tokens of its language
        self.assertEqual(sorted(out), [5, 18, 19])
        self.assertEqual(out[5]["INT4"]["fp_established"], 3)
        self.assertEqual(out[18]["INT4"]["fp_established"], 3); self.assertEqual(out[19]["INT4"]["fp_established"], 0)                 # 'at least 18' holds, 'at least 19' does not
        for thr in out:
            self.assertEqual(out[thr]["INT0"]["lost_of_established"], 0)

    def test_fragility_on_a_sidecar_has_the_requested_covariates(self):
        d, base = self.make(); side = an.load_aggregate_sidecar(d, "all"); coh = an.load_cohort(base, side)
        out = rb.fragility(d, side, coh, "INT4", stream="all", min_positives=5, n_bins=2)
        self.assertEqual(sorted(out["covariates"]), ["margin", "max_activation", "target_mean", "target_rate", "target_share"])
        self.assertEqual(out["n_established"], 3); self.assertEqual(out["n_lost"], 3)
        for v in out["covariates"].values():
            self.assertTrue(all("loss_fraction" in b for b in v["bins"]))

    def test_activation_scale_is_read_at_the_original_language(self):
        d, base = self.make(); side = an.load_aggregate_sidecar(d, "all"); coh = an.load_cohort(base, side)
        out = rb.fragility(d, side, coh, "INT4", stream="all", min_positives=5, n_bins=3)
        self.assertEqual([b["lo"] for b in out["covariates"]["max_activation"]["bins"]], [1.0, 2.0, 3.0])        # the planted features fire with values 1, 2 and 3 in their own languages

    def test_cli_writes_json_and_markdown(self):
        d, base = self.make()
        self.assertEqual(rb.main(["--run", str(d), "--baseline", str(base), "--out", str(self.tmp / "o"), "--stream", "all", "--thresholds", "5,19", "--conditions", "INT4", "--bins", "2", "--min-positives", "5"]), 0)
        j = json.loads((self.tmp / "o" / "robustness.json").read_text()); md = (self.tmp / "o" / "robustness.md").read_text()
        self.assertEqual(sorted(j["thresholds"]), ["19", "5"]); self.assertIn("INT4", j["fragility"]); self.assertIn("exploratory", md.lower())


if __name__ == "__main__":
    unittest.main()
