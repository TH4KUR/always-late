import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phase1_assoc_rank_link as rl


def groups_for(**langs):
    """groups[lang]['all'] = feature ids in dev-mass order; the rank of an id is its position + 1."""
    return {code: {"all": list(ids)} for code, ids in langs.items()}


def rows_for(code, ids, retained, conditions=("INT4",), fp_status=None):
    """Status rows of one language: every feature is established in FP; at each condition `retained(rank)` says whether it is still retained."""
    out = []
    for r, fid in enumerate(ids, start=1):
        fps = (fp_status or {}).get(r, "retained")
        out.append({"language": code, "feature_id": fid, "condition": "FP", "fp_status": fps, "status": fps})
        for c in conditions:
            out.append({"language": code, "feature_id": fid, "condition": c, "fp_status": fps, "status": "retained" if retained(r) else "different_peak"})
    return out


IDS_A = list(range(1000, 1300))                                                                              # 300 features, rank = position + 1


def pattern_a(rank):                                                                                         # fractions 1.0, 0.5, 0.25, 0.0 in the four bins
    return rank <= 16 or (17 <= rank <= 64 and rank % 2 == 0) or (65 <= rank <= 256 and rank % 4 == 0)


def pattern_b(rank):                                                                                         # the opposite: the top is lost, the marginal features are kept
    return rank > 64


class Bins(unittest.TestCase):
    def test_bin_edges(self):
        for rank, want in ((1, "1-16"), (16, "1-16"), (17, "17-64"), (64, "17-64"), (65, "65-256"), (256, "65-256"), (257, "257+"), (1719, "257+")):
            self.assertEqual(rl.bin_of(rank), want, rank)

    def test_the_four_bins_are_in_rank_order(self):
        self.assertEqual(rl.BIN_LABELS, ["1-16", "17-64", "65-256", "257+"])

    def test_ranks_come_from_the_position_in_the_group_list(self):
        rank = rl.rank_lookup(groups_for(aaa=[30, 20, 10], bbb=[10, 30]))
        self.assertEqual((rank[("aaa", 30)], rank[("aaa", 10)], rank[("bbb", 10)], rank[("bbb", 30)]), (1, 3, 1, 2))


class RankTable(unittest.TestCase):
    def setUp(self):
        self.rank = rl.rank_lookup(groups_for(aaa=IDS_A)); self.rows = rows_for("aaa", IDS_A, pattern_a)

    def table(self, rows=None, **kw):
        return {(r["condition"], r["bin"]): r for r in rl.rank_table(rows if rows is not None else self.rows, self.rank, **kw)}

    def test_counts_and_fractions_by_hand(self):
        t = self.table()
        got = [(t[("INT4", b)]["n_established"], t[("INT4", b)]["n_retained"]) for b in rl.BIN_LABELS]
        self.assertEqual(got, [(16, 16), (48, 24), (192, 48), (44, 0)])
        self.assertEqual([t[("INT4", b)]["fraction"] for b in rl.BIN_LABELS], [1.0, 0.5, 0.25, 0.0])
        self.assertEqual({t[("INT4", b)]["n_languages"] for b in rl.BIN_LABELS}, {1})

    def test_wilson_intervals_bracket_the_fraction_and_are_wider_for_fewer_features(self):
        t = self.table()
        for b in rl.BIN_LABELS:
            r = t[("INT4", b)]; self.assertLessEqual(r["lo"], r["fraction"] + 1e-12); self.assertGreaterEqual(r["hi"], r["fraction"] - 1e-12)
        self.assertGreater(t[("INT4", "1-16")]["hi"] - t[("INT4", "1-16")]["lo"], t[("INT4", "65-256")]["hi"] - t[("INT4", "65-256")]["lo"] - 1.0)
        self.assertLess(t[("INT4", "1-16")]["lo"], 1.0)                                                      # 16 of 16 is not certainty

    def test_features_not_established_in_full_precision_are_not_counted_even_if_they_look_retained_later(self):
        rows = rows_for("aaa", IDS_A, pattern_a, fp_status={1: "different_peak", 2: "low_support"})           # ranks 1 and 2 were never established; at INT4 they are "retained" again
        t = self.table(rows)
        self.assertEqual((t[("INT4", "1-16")]["n_established"], t[("INT4", "1-16")]["n_retained"]), (14, 14))

    def test_full_precision_is_not_a_condition_and_the_condition_list_filters(self):
        rows = rows_for("aaa", IDS_A, pattern_a, conditions=("INT4", "INT8"))
        self.assertEqual({c for c, _ in self.table(rows)}, {"INT4", "INT8"})
        self.assertEqual({c for c, _ in self.table(rows, conditions=["INT8"])}, {"INT8"})

    def test_an_empty_bin_has_no_fraction_not_zero(self):
        short = IDS_A[:20]; rows = rows_for("aaa", short, pattern_a); rank = rl.rank_lookup(groups_for(aaa=short))
        t = {(r["condition"], r["bin"]): r for r in rl.rank_table(rows, rank)}
        self.assertEqual(t[("INT4", "257+")]["n_established"], 0); self.assertIsNone(t[("INT4", "257+")]["fraction"]); self.assertIsNone(t[("INT4", "257+")]["lo"])

    def test_a_feature_missing_from_the_group_ranking_is_an_error(self):
        with self.assertRaises(KeyError):
            rl.rank_table(rows_for("aaa", IDS_A + [9999], pattern_a), self.rank)

    def test_languages_are_counted_per_bin(self):
        rank = rl.rank_lookup(groups_for(aaa=IDS_A, bbb=IDS_A[:100]))
        rows = rows_for("aaa", IDS_A, pattern_a) + rows_for("bbb", IDS_A[:100], pattern_a)
        t = {(r["condition"], r["bin"]): r for r in rl.rank_table(rows, rank)}
        self.assertEqual((t[("INT4", "1-16")]["n_languages"], t[("INT4", "257+")]["n_languages"]), (2, 1))
        self.assertEqual(t[("INT4", "1-16")]["n_established"], 32)


class LanguageContrast(unittest.TestCase):
    def setUp(self):
        self.rank = rl.rank_lookup(groups_for(aaa=IDS_A, bbb=IDS_A, ccc=IDS_A[:66]))
        self.rows = rows_for("aaa", IDS_A, pattern_a) + rows_for("bbb", IDS_A, pattern_b) + rows_for("ccc", IDS_A[:66], pattern_a)

    def test_difference_per_language_by_hand_and_the_mean_over_languages(self):
        c = rl.language_contrast(self.rows, self.rank, K=64, min_n=3, n_boot=300, seed=0)["INT4"]
        self.assertAlmostEqual(c["per_language"]["aaa"]["top"], 40 / 64); self.assertAlmostEqual(c["per_language"]["aaa"]["rest"], 48 / 236)
        self.assertAlmostEqual(c["per_language"]["aaa"]["diff"], 40 / 64 - 48 / 236)
        self.assertAlmostEqual(c["per_language"]["bbb"]["diff"], -1.0)
        self.assertEqual(c["n_languages"], 2)                                                                # ccc has only 2 features beyond rank 64: below min_n in the rest
        self.assertAlmostEqual(c["mean_diff"], ((40 / 64 - 48 / 236) + (-1.0)) / 2)
        self.assertEqual((c["n_top_better"], c["n_top_worse"]), (1, 1))

    def test_a_language_needs_min_n_established_features_in_both_parts(self):
        c = rl.language_contrast(self.rows, self.rank, K=64, min_n=3, n_boot=50, seed=0)["INT4"]
        self.assertNotIn("ccc", c["per_language"])
        c2 = rl.language_contrast(self.rows, self.rank, K=64, min_n=2, n_boot=50, seed=0)["INT4"]
        self.assertIn("ccc", c2["per_language"]); self.assertEqual(c2["n_languages"], 3)                     # with two features beyond rank 64 it enters once min_n <= 2

    def test_the_interval_comes_from_a_bootstrap_over_languages_and_is_seeded(self):
        a = rl.language_contrast(self.rows, self.rank, K=64, min_n=3, n_boot=500, seed=0)["INT4"]; b = rl.language_contrast(self.rows, self.rank, K=64, min_n=3, n_boot=500, seed=0)["INT4"]
        self.assertEqual((a["lo"], a["hi"]), (b["lo"], b["hi"]))
        d = [a["per_language"][c]["diff"] for c in ("aaa", "bbb")]
        self.assertAlmostEqual(a["lo"], min(d)); self.assertAlmostEqual(a["hi"], max(d))                      # with two languages the draws can only give the two diffs or their mean
        self.assertLessEqual(a["lo"], a["mean_diff"]); self.assertGreaterEqual(a["hi"], a["mean_diff"])

    def test_the_interval_is_the_2_5_and_97_5_percentiles_of_language_level_draws_and_the_point_is_their_mean(self):
        codes = ["aaa", "bbb", "ccc", "ddd", "eee", "fff"]
        rule = {c: (lambda r, m=m: (r <= 64 and r % m == 0) or (r > 64 and r % 7 == 0)) for m, c in enumerate(codes, start=1)}      # top rate 1/m, rest rate 1/7: six different differences
        rank = rl.rank_lookup(groups_for(**{c: IDS_A for c in codes})); rows = [x for c in codes for x in rows_for(c, IDS_A, rule[c])]
        c = rl.language_contrast(rows, rank, K=64, min_n=3, n_boot=400, seed=7)["INT4"]
        d = np.array([c["per_language"][k]["diff"] for k in codes])
        self.assertEqual(len(set(np.round(d, 6))), 6)
        self.assertAlmostEqual(c["mean_diff"], d.mean()); self.assertNotAlmostEqual(float(d.mean()), float(np.median(d)), places=3)
        draws = np.random.default_rng(7).integers(0, 6, size=(400, 6)); lo, hi = np.percentile(d[draws].mean(axis=1), [2.5, 97.5])
        self.assertAlmostEqual(c["lo"], lo); self.assertAlmostEqual(c["hi"], hi)
        self.assertGreater(c["lo"], d.min() + 0.05); self.assertLess(c["hi"], d.max() - 0.05)                # a mean over six languages is narrower than the spread of single languages

    def test_ties_count_for_neither_side(self):
        rows = rows_for("aaa", IDS_A, lambda r: True) + rows_for("bbb", IDS_A, pattern_b)                    # aaa: every established feature retained in both parts, a difference of exactly 0
        c = rl.language_contrast(rows, rl.rank_lookup(groups_for(aaa=IDS_A, bbb=IDS_A)), K=64, min_n=3, n_boot=20, seed=0)["INT4"]
        self.assertEqual(c["per_language"]["aaa"]["diff"], 0.0); self.assertEqual((c["n_top_better"], c["n_top_worse"]), (0, 1))

    def test_a_different_top_size_moves_the_split(self):
        c16 = rl.language_contrast(self.rows, self.rank, K=16, min_n=3, n_boot=50, seed=0)["INT4"]
        self.assertAlmostEqual(c16["per_language"]["aaa"]["top"], 1.0)

    def test_no_language_with_enough_features_gives_an_undefined_contrast_not_zero(self):
        c = rl.language_contrast(self.rows, self.rank, K=64, min_n=1000, n_boot=20, seed=0)["INT4"]
        self.assertEqual(c["n_languages"], 0); self.assertIsNone(c["mean_diff"]); self.assertIsNone(c["lo"])


def write_status(path, rows, extra_group_rows=True):
    fields = ["feature_id", "group", "original_language", "condition", "fp_status", "status", "peak_set", "target_share", "margin"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows:
            w.writerow({"feature_id": r["feature_id"], "group": "selective", "original_language": r["language"], "condition": r["condition"], "fp_status": r["fp_status"], "status": r["status"],
                        "peak_set": "x", "target_share": 0.1, "margin": 0.0})
        if extra_group_rows:
            w.writerow({"feature_id": 1, "group": "controls", "original_language": "aaa", "condition": "INT4", "fp_status": "retained", "status": "retained", "peak_set": "x", "target_share": 0, "margin": 0})


class Files(unittest.TestCase):
    def test_only_the_selective_group_is_read_and_ids_are_integers(self):
        d = Path(tempfile.mkdtemp()); f = d / "feature_status.csv"
        write_status(f, rows_for("aaa", IDS_A[:3], pattern_a))
        rows = rl.read_status(f)
        self.assertEqual(len(rows), 3 * 2); self.assertTrue(all(isinstance(r["feature_id"], int) for r in rows)); self.assertEqual({r["condition"] for r in rows}, {"FP", "INT4"})

    def test_the_d035_headline_check_passes_when_the_pooled_counts_agree_and_raises_when_they_do_not(self):
        rank = rl.rank_lookup(groups_for(aaa=IDS_A)); rows = rows_for("aaa", IDS_A, pattern_a)
        table = rl.rank_table(rows, rank)
        d = Path(tempfile.mkdtemp()); f = d / "language_summary.csv"
        header = "group,condition,language,n_selective,fp_established,retained,lost,retention_fraction,retention_lo,retention_hi,loss_fraction,loss_lo,loss_hi,conditional_fraction,n_languages\n"
        f.write_text(header + "selective,INT4,POOLED,300,300,88,212,0,0,0,0,0,0,0.29,\n")
        rl.check_against_d035(table, f)                                                                      # 16 + 24 + 48 + 0 = 88 retained of 300 established
        f.write_text(header + "selective,INT4,POOLED,300,300,89,211,0,0,0,0,0,0,0.29,\n")
        with self.assertRaises(ValueError):
            rl.check_against_d035(table, f)

    def test_a_summary_without_any_condition_of_the_table_cannot_be_used_to_check(self):
        rank = rl.rank_lookup(groups_for(aaa=IDS_A)); table = rl.rank_table(rows_for("aaa", IDS_A, pattern_a), rank)
        f = Path(tempfile.mkdtemp()) / "language_summary.csv"
        f.write_text("group,condition,language,n_selective,fp_established,retained,lost,retention_fraction,retention_lo,retention_hi,loss_fraction,loss_lo,loss_hi,conditional_fraction,n_languages\n"
                     "selective,INT8,POOLED,300,300,300,0,0,0,0,0,0,0,1,\ncontrols,INT4,POOLED,300,300,88,0,0,0,0,0,0,0,1,\n")
        with self.assertRaises(ValueError):
            rl.check_against_d035(table, f)                                                                  # INT8 is not in the table; the controls row is not the selective group

    def test_main_writes_tables_and_a_report_and_refuses_a_summary_that_disagrees(self):
        d = Path(tempfile.mkdtemp()); status = d / "feature_status.csv"; manifest = d / "groups_manifest.json"; summ = d / "language_summary.csv"; out = d / "out"
        write_status(status, rows_for("aaa", IDS_A, pattern_a) + rows_for("bbb", IDS_A, pattern_b))
        manifest.write_text(json.dumps({"groups": groups_for(aaa=IDS_A, bbb=IDS_A)}))
        header = "group,condition,language,n_selective,fp_established,retained,lost,retention_fraction,retention_lo,retention_hi,loss_fraction,loss_lo,loss_hi,conditional_fraction,n_languages\n"
        summ.write_text(header + f"selective,INT4,POOLED,600,600,{88 + 236},0,0,0,0,0,0,0,0.5,\n")
        rl.main(["--status", str(status), "--manifest", str(manifest), "--d035-summary", str(summ), "--out", str(out), "--n-boot", "100"])
        with open(out / "rank_table.csv") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 4); self.assertEqual({r["condition"] for r in rows}, {"INT4"})
        with open(out / "language_contrast.csv") as f:
            self.assertEqual({r["condition"] for r in csv.DictReader(f)}, {"INT4"})
        txt = (out / "report.md").read_text()
        self.assertTrue(txt.startswith("# ")); self.assertIn("H-A1", txt); self.assertIn("exploratory", txt.lower())
        self.assertIn("confounded", txt); self.assertIn("F4", txt)                                           # the limitation travels with the numbers
        summ.write_text(header + "selective,INT4,POOLED,600,600,1,0,0,0,0,0,0,0,0.5,\n")
        with self.assertRaises(ValueError):
            rl.main(["--status", str(status), "--manifest", str(manifest), "--d035-summary", str(summ), "--out", str(d / "out2"), "--n-boot", "10"])


if __name__ == "__main__":
    unittest.main()
