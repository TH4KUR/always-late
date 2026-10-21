"""D-035 Task 6 cross-checks (scripts/g0/phase1_association_crosscheck.py): (1) does the activation-mean view of a feature's peak language agree with the firing-rate view used for the status, and
(2) does the status agree with the historical Pearson correlation at the original language? Pure-function tests on hand-made arrays, an integration test on a collector sidecar, and the CLI."""
import json, shutil, sys, tempfile, unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import phase1_association_collect as ac   # noqa: E402
import phase1_association_crosscheck as cc   # noqa: E402
import phase1_run as run   # noqa: E402
from test_phase1_run import F, BOS, PAD   # noqa: E402
from test_phase1_association_collect import make_planted, CODES   # noqa: E402
from test_phase1_association_analysis import write_baseline   # noqa: E402

L = 3
NT = np.array([1000, 1000, 1000])


def prof(counts, sums):
    """columns = features, rows = languages"""
    return {"counts": np.array(counts, dtype=np.int64).T, "sums": np.array(sums, dtype=np.float64).T, "ntok": NT}


# six features; f5 is not in the cohort
FP = prof([[200, 20, 0], [0, 300, 10], [100, 80, 0], [150, 0, 0], [60, 10, 0], [5, 5, 5]], [[200, 20, 0], [0, 300, 10], [100, 400, 0], [150, 0, 0], [60, 10, 0], [5, 5, 5]])
Q = prof([[200, 20, 0], [0, 100, 300], [100, 80, 0], [150, 160, 0], [60, 10, 0], [5, 5, 5]], [[200, 20, 0], [0, 100, 300], [400, 80, 0], [150, 16, 0], [60, 600, 0], [5, 5, 5]])
PEAK = np.array([0, 1, 0, 0, 0, 0]); MASK = np.array([True, True, True, True, True, False])


class ViewAgreement(unittest.TestCase):
    def test_two_by_two_of_rate_and_mean_retention_among_established_features(self):
        r = cc.view_agreement(FP, Q, PEAK, MASK, min_positives=50)
        self.assertEqual(r["n_established"], 5)
        self.assertEqual((r["both"], r["rate_only"], r["mean_only"], r["neither"]), (2, 1, 1, 1))
        self.assertEqual(r["rate_retained"], 3); self.assertEqual(r["mean_retained"], 3)
        self.assertEqual(r["fp_mean_agrees"], 4)                                                   # f2: the full-precision MEAN peak is language 1, not the original language

    def test_identical_conditions_agree_with_themselves(self):
        r = cc.view_agreement(FP, FP, PEAK, MASK, min_positives=50)
        self.assertEqual(r["rate_retained"], r["n_established"])
        self.assertEqual((r["both"], r["rate_only"], r["mean_only"], r["neither"]), (4, 1, 0, 0))                      # f2 keeps its rate peak but its full-precision mean peak was never the original language

    def test_features_outside_the_group_are_ignored(self):
        r = cc.view_agreement(FP, Q, PEAK, np.array([True, False, False, False, False, False]), min_positives=50)
        self.assertEqual((r["n_established"], r["both"]), (1, 1))

    def test_the_mean_view_requires_support_like_the_rate_view(self):
        q = prof([[200, 20, 0], [0, 100, 300], [100, 80, 0], [150, 160, 0], [10, 0, 0], [5, 5, 5]], [[200, 20, 0], [0, 100, 300], [400, 80, 0], [150, 16, 0], [10, 0, 0], [5, 5, 5]])
        r = cc.view_agreement(FP, q, PEAK, MASK, min_positives=50)                              # f4 keeps its mean peak but has only 10 positives: low support in both views
        self.assertEqual((r["rate_only"], r["both"]), (0, 2)); self.assertEqual(r["neither"], 2)

    def test_a_tie_in_the_mean_view_is_not_retained(self):
        q = prof([[200, 20, 0], [0, 100, 300], [100, 80, 0], [150, 160, 0], [60, 10, 0], [5, 5, 5]], [[200, 20, 0], [0, 100, 300], [400, 80, 0], [150, 16, 0], [60, 60, 0], [5, 5, 5]])
        r = cc.view_agreement(FP, q, PEAK, MASK, min_positives=50)                              # f4: means 0.06 and 0.06 are tied, so the original language is not the UNIQUE mean peak
        self.assertEqual((r["rate_only"], r["both"]), (1, 2))

    def test_invalid_arrays_are_rejected(self):
        with self.assertRaises(ValueError):
            cc.view_agreement(FP, {**Q, "ntok": NT[:2]}, PEAK, MASK)


class CorrelationSplit(unittest.TestCase):
    def test_retained_and_lost_split_by_the_correlation_threshold(self):
        est = np.array([True, True, True, True, True, False]); ret = np.array([True, False, True, False, True, False]); r = np.array([0.99, 0.2, 0.95, 0.5, 0.3, 0.9])
        s = cc.correlation_split(ret, est, r, threshold=0.9)
        self.assertEqual((s["retained_high"], s["retained_low"], s["lost_high"], s["lost_low"]), (2, 1, 0, 2))
        self.assertAlmostEqual(s["median_r_retained"], 0.95); self.assertAlmostEqual(s["median_r_lost"], 0.35); self.assertEqual(s["n_nan"], 0)

    def test_threshold_is_strict_and_nan_is_counted_apart(self):
        est = np.array([True, True, True]); ret = np.array([True, True, False]); r = np.array([0.9, np.nan, 0.91])
        s = cc.correlation_split(ret, est, r, threshold=0.9)
        self.assertEqual((s["retained_high"], s["retained_low"], s["lost_high"], s["lost_low"], s["n_nan"]), (0, 1, 1, 0, 1))
        self.assertAlmostEqual(s["median_r_retained"], 0.9)                                         # the NaN is left out of the median


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
        base = write_baseline(self.tmp / "base", peak, sel, [3, 4])
        corr = torch.zeros(3, F); corr[0, 0], corr[1, 1], corr[2, 2] = 0.99, 0.4, 0.2
        torch.save({"languages": CODES, "bits": [4, 0], "corr": {4: corr, 0: torch.ones(3, F)}}, base / "survival.pt")
        return self.tmp / "run", base

    def test_crosscheck_on_a_collector_sidecar(self):
        d, base = self.make(); out = cc.crosscheck(d, base, stream="all", min_positives=5)
        c = out["conditions"]["INT4"]; self.assertEqual(c["views"]["n_established"], 3); self.assertEqual(c["views"]["rate_retained"], 0)
        self.assertEqual(c["correlation"]["lost_high"], 1); self.assertEqual(c["correlation"]["lost_low"], 2)                       # f0 lost although r = 0.99 (peak broadened, ties), f1 and f2 lost with low r
        i = out["conditions"]["INT0"]; self.assertEqual(i["views"]["rate_retained"], 3); self.assertEqual(i["correlation"]["retained_high"], 3)

    def test_established_controls_are_not_counted_in_the_selective_cross_check(self):
        d, _ = self.make(); sel = np.zeros(F, dtype=bool); sel[[1, 2]] = True; peak = np.zeros(F, dtype=int); peak[:3] = [0, 1, 2]
        base = write_baseline(self.tmp / "ctl", peak, sel, [0])                                    # feature 0 fires and is established, but it is a control here
        torch.save({"languages": CODES, "bits": [4, 0], "corr": {4: torch.zeros(3, F), 0: torch.ones(3, F)}}, base / "survival.pt")
        out = cc.crosscheck(d, base, stream="all", min_positives=5)
        self.assertEqual(out["n_established"], 2); self.assertEqual(out["conditions"]["INT0"]["correlation"]["retained_high"], 2)

    def test_the_correlation_is_read_at_the_baseline_language_index(self):
        d, _ = self.make(); order = [2, 0, 1]
        sel = np.zeros(F, dtype=bool); sel[:3] = True; peak_run = np.zeros(F, dtype=int); peak_run[:3] = [0, 1, 2]
        rev = write_baseline(self.tmp / "rev", np.array([order.index(int(p)) for p in peak_run]), sel, [3, 4], codes=[CODES[i] for i in order])
        corr = torch.zeros(3, F)
        for run_lang, f in ((0, 0), (1, 1), (2, 2)):
            corr[order.index(run_lang), f] = (0.99, 0.4, 0.2)[f]                                  # the value sits at the baseline index of the feature's original language
        torch.save({"languages": [CODES[i] for i in order], "bits": [4], "corr": {4: corr}}, rev / "survival.pt")
        out = cc.crosscheck(d, rev, stream="all", min_positives=5)
        self.assertEqual((out["conditions"]["INT4"]["correlation"]["lost_high"], out["conditions"]["INT4"]["correlation"]["lost_low"]), (1, 2))

    def test_cli_writes_json_and_markdown(self):
        d, base = self.make()
        self.assertEqual(cc.main(["--run", str(d), "--baseline", str(base), "--out", str(self.tmp / "o"), "--stream", "all", "--min-positives", "5"]), 0)
        j = json.loads((self.tmp / "o" / "crosscheck.json").read_text()); md = (self.tmp / "o" / "crosscheck.md").read_text()
        self.assertIn("INT4", j["conditions"]); self.assertIn("INT4", md); self.assertIn("firing-rate", md)


if __name__ == "__main__":
    unittest.main()
