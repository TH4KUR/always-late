"""Tests for scripts/g0/gptq_ext/wandb_headlines.py (design reports/wandb_schema_v2_gptq_extension.md, D-115): the headline file of the wandb extension is derived from committed result files and must be reproduced
byte for byte; its values are pinned to the numbers written in the decision entries D-104 to D-111 (so the generator is checked against the written record, not only against itself)."""
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "gptq_ext")); sys.path.insert(0, str(HERE.parent))
import wandb_headlines as wh  # noqa: E402

ARMS = ("legacy", "matched", "gptq")


class Headlines(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h = wh.build(ROOT)

    def vals(self, slug, base, nd=3):
        return [round(self.h[slug][f"{base}_{a}"], nd) for a in ARMS]

    def test_the_committed_file_is_exactly_what_the_generator_writes(self):
        self.assertEqual((ROOT / "reports/gptq_extension/wandb_headlines.json").read_text(), wh.render(self.h))

    def test_every_run_has_every_quantity_for_every_arm_and_only_scalars(self):
        self.assertEqual(sorted(self.h), sorted(wh.SLUGS))
        for slug, d in self.h.items():
            self.assertTrue(d, slug)
            for k, v in d.items():
                self.assertIsInstance(v, (int, float, bool, str), (slug, k)); self.assertTrue(k.endswith(tuple("_" + a for a in ARMS)), (slug, k))
            bases = {k.rsplit("_", 1)[0] for k in d}
            for b in bases:
                self.assertTrue(all(f"{b}_{a}" in d for a in ARMS), (slug, b))

    def test_stage1_matches_d104_and_d106(self):
        # D-104 printed the matched-RTN INT4 survival as 18.7 (12.7 + the rounded difference 6.0); the mean of the 27 per-language values is 18.649, so 18.6 (correction note of 2026-10-29)
        s = "gptq-ext-stage1-survival-loss"
        self.assertEqual(self.vals(s, "survival_mean_INT4", 1), [12.7, 18.6, 30.5]); self.assertEqual(self.vals(s, "survival_mean_INT6", 1), [85.0, 89.7, 97.5])
        self.assertEqual(self.vals(s, "loss_increase_INT4"), [0.590, 0.385, 0.130]); self.assertEqual(self.vals(s, "loss_increase_INT6", 4), [0.0254, 0.0197, 0.0058])
        self.assertEqual(self.vals(s, "holm_T1"), [0.514, 0.042, 1.0]); self.assertEqual(self.vals(s, "holm_T2"), [0.128, 0.038, 1.0])

    def test_transfer_main_matches_d105(self):
        s = "gptq-ext-transfer-main"
        self.assertEqual([self.h[s][f"utility_overall_{a}"] for a in ARMS], ["not met"] * 3)
        self.assertEqual([self.h[s][f"utility_INT4_{a}"] for a in ARMS], ["met"] * 3); self.assertEqual([self.h[s][f"utility_INT6_{a}"] for a in ARMS], ["not met"] * 3)
        self.assertEqual(self.vals(s, "error_over_floor_INT4", 2), [7.69, 6.71, 5.54]); self.assertEqual(self.vals(s, "error_over_floor_INT6", 2), [1.88, 1.75, 1.52])

    def test_transfer_external_matches_d111(self):
        s = "gptq-ext-transfer-external"
        self.assertEqual([self.h[s][f"utility_overall_{a}"] for a in ARMS], ["met", "met", "not met"])
        self.assertEqual(self.vals(s, "heldout_improvement_INT4"), [0.194, 0.287, 0.296]); self.assertEqual(self.vals(s, "heldout_improvement_INT6"), [0.234, 0.250, 0.026])
        self.assertEqual(self.vals(s, "cosine_INT4"), [0.754, 0.808, 0.876])

    def test_loss_task_and_rescue_match_d107(self):
        s = "gptq-ext-loss-task-rescue"
        self.assertEqual(self.vals(s, "retention_median_INT4", 2), [1.54, 1.36, 1.22]); self.assertEqual(self.vals(s, "retention_mean_INT4", 2), [2.14, 1.75, 1.31]); self.assertEqual(self.vals(s, "retention_mean_INT6", 2), [1.04, 1.06, 0.99])
        self.assertEqual(self.vals(s, "rescue_group_share_INT4", 3), [0.026, 0.023, 0.020]); self.assertEqual(self.vals(s, "rescue_residual_share_INT4", 3), [0.393, 0.352, 0.414])

    def test_the_retention_median_uses_the_27_language_rows_only_not_the_aggregate_rows(self):
        import csv, statistics
        langs = {l["code"] for l in json.loads((ROOT / "reports/g0/phase1_languages.json").read_text())["languages"]}
        paths = {"legacy": "reports/functional_validation/results/eval/retention.csv", "matched": "reports/gptq_extension/results/stage3/analysis_rtnm_eval/retention.csv", "gptq": "reports/gptq_extension/results/stage3/analysis_gptq_eval/retention.csv"}
        for a, p in paths.items():
            rows = [r for r in csv.DictReader(open(ROOT / p)) if r["precision"] == "INT4"]
            self.assertEqual(len([r for r in rows if r["language"] in langs]), 27); self.assertGreater(len(rows), 27)   # the file also holds aggregate rows
            want = statistics.median(float(r["ratio"]) for r in rows if r["language"] in langs)
            self.assertAlmostEqual(self.h["gptq-ext-loss-task-rescue"][f"retention_median_INT4_{a}"], want, places=12)

    def test_belebele_matches_d107(self):
        # D-107 printed the GPTQ INT4 no-edit accuracy as 0.319; the mean over the 18 languages is 0.31847, so 0.318 (correction note of 2026-10-29)
        s = "gptq-ext-belebele"
        self.assertEqual(self.vals(s, "accuracy_noedit_INT4"), [0.236, 0.291, 0.318]); self.assertEqual(self.vals(s, "accuracy_noedit_INT6"), [0.423, 0.424, 0.423])
        self.assertEqual(self.vals(s, "group_effect_INT4", 4), [0.0042, -0.0009, -0.0024])

    def test_f5_matches_d108(self):
        s = "gptq-ext-f5"
        self.assertEqual(self.vals(s, "w1_median_INT4"), [0.839, 0.875, 0.923]); self.assertEqual(self.vals(s, "w1_median_INT6"), [0.988, 0.989, 0.992])
        self.assertEqual(self.vals(s, "share_at_floor_INT4"), [0.003, 0.007, 0.117]); self.assertEqual(self.vals(s, "spearman_r1_w1_INT6"), [0.533, 0.520, 0.575]); self.assertEqual(self.vals(s, "spearman_r1_w1_INT4"), [0.413, 0.378, 0.381])
        self.assertEqual(self.vals(s, "s2_auroc"), [0.256, 0.381, 0.490])

    def test_study_g_matches_d109(self):
        s = "gptq-ext-study-g"
        self.assertEqual(self.vals(s, "net_harm_x0088_INT4"), [0.323, 0.281, 0.187]); self.assertEqual(self.vals(s, "net_harm_x0088_INT6"), [0.166, 0.170, 0.166])
        self.assertEqual([int(self.h[s][f"alpha_above_1_INT4_{a}"]) for a in ARMS], [21, 13, 7]); self.assertEqual([self.h[s][f"hg4_{a}"] for a in ARMS], [False, False, True])


if __name__ == "__main__":
    unittest.main()
