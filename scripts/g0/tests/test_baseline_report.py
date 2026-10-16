"""Tests for scripts/g0/baseline_report.py: compare a replicated Phase-3 summary with Duan's published one using the
pre-committed bands (reports/duan_codebase_audit.md section 7.2, Q1): INT7 survival within 5 pp, INT6 within 10 pp,
active features within 10%, perplexity signs (INT7 improves, INT6 worsens)."""
import csv, sys, tempfile, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import baseline_report as br  # noqa: E402


class _RecLogger:
    instances = []
    active = True
    def __init__(self): self.calls = []; _RecLogger.instances.append(self)
    @classmethod
    def start(cls, **kw): lg = cls(); lg.kw = kw; return lg
    def log_condition(self, row): self.calls.append(("cond", row["condition"]))
    def log_table(self, rows, key="conditions"): self.calls.append(("table", key))
    def set_summary(self, d): self.calls.append(("summary", dict(d)))
    def finish(self, out): self.calls.append(("finish",))


class WandbUpload(unittest.TestCase):
    def test_upload_uses_the_fixed_schema_only(self):
        _RecLogger.instances.clear()
        rep = [r("FP16 baseline", 100.0, 0.0), r("RTN INT7", 81.2, -5.6)]
        cmp_ = {"rows": [{"condition": "RTN INT7", "d_survived_pp": -0.1}], "q1_pass": True, "bands": {"x": 1}}
        pf = {"n_common_active": 7121, "spearman_corr": 0.9996, "spearman_firing_rate": 0.9999}
        br.upload_to_wandb(Path("/tmp/run-xyz"), rep, cmp_, pf, logger_cls=_RecLogger, project="p", entity=None, name="baseline/full")
        lg = _RecLogger.instances[0]
        self.assertEqual([c for c in lg.calls if c[0] == "cond"], [("cond", "FP16 baseline"), ("cond", "RTN INT7")])
        self.assertIn(("table", "comparison_vs_published"), lg.calls); self.assertIn(("table", "per_feature_int6"), lg.calls)
        self.assertIn(("summary", {"q1_pass": True}), lg.calls); self.assertEqual(lg.calls[-1], ("finish",))
        self.assertEqual(lg.kw["name"], "baseline/full"); self.assertEqual(lg.kw["config"]["experiment"], "baseline")


def r(cond, survived, delta, active=7144, median=0.9, ppl=450.0):
    return {"condition": cond, "survived_>0.9_pct": survived, "ppl_delta_pct": delta, "n_active_features": active,
            "median_corr": median, "perplexity": ppl}


PUB = [r("FP16 baseline", 100.0, 0.0, median=1.0), r("RTN INT8", 99.09, 2.62, median=0.9707), r("RTN INT7", 81.26, -5.65, median=0.9499),
       r("RTN INT6", 51.30, 3.99, median=0.9027)]


def rep(**over):
    rows = [dict(x) for x in PUB]
    for x in rows:
        x.update(over.get(x["condition"], {}))
    return rows


class CompareRows(unittest.TestCase):
    def test_identical_replication_passes_every_band(self):
        c = br.compare_rows(rep(), PUB)
        self.assertTrue(c["q1_pass"])
        self.assertEqual([k for k, v in c["bands"].items() if v is not True], [])
        self.assertAlmostEqual(c["rows"][2]["d_survived_pp"], 0.0)

    def test_int7_survival_off_by_more_than_5pp_fails_that_band_only(self):
        c = br.compare_rows(rep(**{"RTN INT7": {"survived_>0.9_pct": 87.5}}), PUB)
        self.assertFalse(c["bands"]["int7_survival_within_5pp"]); self.assertTrue(c["bands"]["int6_survival_within_10pp"])
        self.assertFalse(c["q1_pass"])

    def test_int6_survival_off_by_9pp_is_still_inside_its_10pp_band(self):
        c = br.compare_rows(rep(**{"RTN INT6": {"survived_>0.9_pct": 60.3}}), PUB)
        self.assertTrue(c["bands"]["int6_survival_within_10pp"]); self.assertTrue(c["q1_pass"])

    def test_active_feature_count_more_than_10pct_off_fails(self):
        c = br.compare_rows(rep(**{"FP16 baseline": {"n_active_features": 6400}, "RTN INT7": {"n_active_features": 6400}}), PUB)
        self.assertFalse(c["bands"]["active_features_within_10pct"]); self.assertFalse(c["q1_pass"])

    def test_perplexity_sign_pattern_is_checked(self):
        c = br.compare_rows(rep(**{"RTN INT7": {"ppl_delta_pct": 1.0}}), PUB)
        self.assertFalse(c["bands"]["ppl_signs_int7_improves_int6_worsens"]); self.assertFalse(c["q1_pass"])

    def test_missing_conditions_make_their_bands_not_evaluable_not_failed(self):
        smoke = [x for x in rep() if x["condition"] in ("FP16 baseline", "RTN INT8", "RTN INT6")]   # like the 3a_test smoke
        c = br.compare_rows(smoke, PUB)
        self.assertIsNone(c["bands"]["int7_survival_within_5pp"]); self.assertIsNone(c["q1_pass"])
        self.assertTrue(c["bands"]["int6_survival_within_10pp"])


def write_pf(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["feature_id", "corr", "firing_rate", "max_activation", "active"]); w.writeheader(); w.writerows(rows)


class PerFeatureAgreement(unittest.TestCase):
    def test_identical_files_give_spearman_one_and_reversed_give_minus_one(self):
        with tempfile.TemporaryDirectory() as d:
            a = [{"feature_id": i, "corr": 0.5 + i / 100, "firing_rate": 0.01 * (i + 1), "max_activation": i + 1.0, "active": True} for i in range(30)]
            b = [dict(x, corr=0.5 + (29 - x["feature_id"]) / 100) for x in a]
            write_pf(Path(d, "a.csv"), a); write_pf(Path(d, "b.csv"), a); write_pf(Path(d, "c.csv"), b)
            same = br.per_feature_agreement(Path(d, "a.csv"), Path(d, "b.csv"))
            rev = br.per_feature_agreement(Path(d, "a.csv"), Path(d, "c.csv"))
        self.assertAlmostEqual(same["spearman_corr"], 1.0); self.assertAlmostEqual(same["spearman_firing_rate"], 1.0)
        self.assertAlmostEqual(rev["spearman_corr"], -1.0); self.assertEqual(same["n_common_active"], 30)

    def test_only_features_active_in_both_files_are_compared(self):
        with tempfile.TemporaryDirectory() as d:
            a = [{"feature_id": i, "corr": 0.9, "firing_rate": 0.01 * (i + 1), "max_activation": 1.0, "active": i < 20} for i in range(30)]
            b = [dict(x, active=i >= 10) for i, x in enumerate(a)]
            write_pf(Path(d, "a.csv"), a); write_pf(Path(d, "b.csv"), b)
            self.assertEqual(br.per_feature_agreement(Path(d, "a.csv"), Path(d, "b.csv"))["n_common_active"], 10)


if __name__ == "__main__":
    unittest.main()
