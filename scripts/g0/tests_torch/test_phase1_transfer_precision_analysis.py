import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phase1_transfer_analysis as ta
import phase1_transfer_precision_analysis as pa

OLD = {"real": 20.0, "matched_swapped": 10.0}


def row(f, s, kind="real", state="INT4", cluster=None, **kw):
    r = {"feature": f, "sentence": s, "kind": kind, "state": state, "cluster": s if cluster is None else cluster, "quartile": 3, "fp_norm_bf16": 200.0, "fp_norm_fp32": 200.0, "qq_norm_bf16": 150.0, "qq_norm_fp32": 150.0, "err_bf16": 100.0, "err_fp32": 100.0,
         "cos_bf16": 0.6, "cos_fp32": 0.6, "arith_ff": 1.0, "arith_qq": 5.0, "arith_qq_over_err_bf16": 0.05, "arith_qq_over_err_fp32": 0.05, "arith_ff_over_fp_norm_bf16": 0.005, "cos_arith_vs_err": 0.1,
         "dloss_FF_bf16": -0.01, "dloss_FF_fp32": -0.01, "dloss_QQ_bf16": -0.02, "dloss_QQ_fp32": -0.02}
    r.update(kw); return r


def table(n_features=10, per=4, seed=0, int4_err=100.0, int16_err=10.0, dcos=0.0):
    rng = np.random.default_rng(seed); rows = []
    for f in range(n_features):
        for j in range(per):
            s = 1000 * f + j; base = float(rng.uniform(0.9, 1.1))
            rows.append(row(f, s, state="INT16", err_bf16=int16_err * base, err_fp32=int16_err * base * 1.1, cos_bf16=0.99, cos_fp32=0.99, fp_norm_bf16=200.0 * base, fp_norm_fp32=200.0 * base))
            rows.append(row(f, s, state="INT4", err_bf16=int4_err * base, err_fp32=int4_err * base, cos_bf16=0.6 + 0.01 * float(rng.normal()), cos_fp32=0.6 + dcos, fp_norm_bf16=200.0 * base, fp_norm_fp32=200.0 * base))
    return rows


class Paired(unittest.TestCase):
    def test_the_paired_differences_are_fp32_minus_bf16_per_item_and_none_propagates(self):
        r = pa.add_paired(row(1, 1, err_bf16=100.0, err_fp32=90.0, cos_bf16=0.6, cos_fp32=0.5, nerr_old_bf16=0.4, nerr_old_fp32=0.45, dloss_FF_bf16=-0.01, dloss_FF_fp32=-0.012, dloss_QQ_bf16=-0.02, dloss_QQ_fp32=-0.03))
        self.assertAlmostEqual(r["d_err"], -10.0); self.assertAlmostEqual(r["d_cos"], -0.1); self.assertAlmostEqual(r["d_nerr_old"], 0.05); self.assertAlmostEqual(r["d_dloss_FF"], -0.002); self.assertAlmostEqual(r["d_dloss_QQ"], -0.01)
        n = pa.add_paired(row(1, 1, cos_bf16=None, nerr_old_bf16=0.4, nerr_old_fp32=0.4)); self.assertIsNone(n["d_cos"]); self.assertEqual(n["d_nerr_old"], 0.0)


class Finalize(unittest.TestCase):
    def test_the_mode_floors_come_from_the_int16_items_only_and_apply_to_both_states_with_the_old_floors_next_to_them(self):
        rows = table(); out = pa.finalize_rows(rows, OLD); r16 = [r for r in rows if r["state"] == "INT16"]
        want = {m: float(np.percentile([r[f"err_{m}"] for r in r16], 95)) for m in ("bf16", "fp32")}; r4 = [r for r in out if r["state"] == "INT4"][0]
        self.assertAlmostEqual(r4["eof_mode_bf16"], r4["err_bf16"] / want["bf16"]); self.assertAlmostEqual(r4["eof_mode_fp32"], r4["err_fp32"] / want["fp32"]); self.assertAlmostEqual(r4["nerr_old_bf16"], r4["err_bf16"] / (r4["fp_norm_bf16"] + 20.0))
        self.assertTrue(all("d_cos" in r and "eligible_old_bf16" in r for r in out)); self.assertEqual(len(out), len(rows))
        # changing an INT4 error does not change the floors
        rows2 = table(int4_err=500.0); out2 = pa.finalize_rows(rows2, OLD); self.assertAlmostEqual([r for r in out2 if r["state"] == "INT16"][0]["eof_mode_bf16"], [r for r in out if r["state"] == "INT16"][0]["eof_mode_bf16"])

    def test_floors_are_per_kind_and_a_kind_without_int16_items_is_an_error(self):
        rows = table() + [row(1, 99, kind="matched_swapped", state="INT4")]
        with self.assertRaisesRegex(ValueError, "matched_swapped"):
            pa.finalize_rows(rows, OLD)

    def test_the_batching_control_values_are_attached_to_the_int16_item_of_each_mode_and_nothing_else(self):
        rows = table(n_features=2, per=1); batch = {(0, 0, "real"): {"FP/FP": 0.3, "Q/Q": 0.5}, (1, 1000, "real"): {"FP/FP": 0.2, "Q/Q": 0.1}}
        ctrl = {"bf16": batch, "fp32": {k: {c: v * 2 for c, v in d.items()} for k, d in batch.items()}}; out = pa.attach_batching(rows, ctrl); r16 = [r for r in out if r["state"] == "INT16" and r["feature"] == 0][0]
        self.assertEqual((r16["batch_FF_bf16"], r16["batch_QQ_bf16"], r16["batch_FF_fp32"], r16["batch_QQ_fp32"]), (0.3, 0.5, 0.6, 1.0)); self.assertTrue(all(r.get("batch_FF_bf16") is None for r in out if r["state"] == "INT4"))


class Summary(unittest.TestCase):
    def setUp(self):
        self.rows = pa.finalize_rows(table(n_features=12, per=5, int4_err=100.0, int16_err=10.0, dcos=0.05), OLD)

    def test_each_median_is_the_crossed_bootstrap_of_the_item_values_by_population_kind_state_and_quantity(self):
        s = pa.summarize(self.rows, n_boot=40, seed=3); r4 = [r for r in self.rows if r["state"] == "INT4"]
        ref = ta.crossed_bootstrap([r["err_bf16"] for r in r4], [r["feature"] for r in r4], [r["cluster"] for r in r4], "median", 40, 3); got = s[("all_real", "real", "INT4", "err_bf16")]
        self.assertEqual((got["point"], got["low"], got["high"], got["n"]), (ref["point"], ref["low"], ref["high"], ref["n"]))
        d = s[("all_real", "real", "INT4", "d_cos")]; self.assertGreater(d["low"], 0.03); self.assertAlmostEqual(d["point"], 0.05, delta=0.03)

    def test_the_populations_are_all_real_the_eligible_real_items_under_the_old_floors_in_both_modes_and_all_matched_swapped_items(self):
        rows = self.rows + [row(0, 7, kind="matched_swapped", state="INT16", err_bf16=3.0, err_fp32=3.3), row(0, 7, kind="matched_swapped", state="INT4", err_bf16=50.0, err_fp32=50.0)]
        rows = pa.finalize_rows(rows, OLD); rows[0]["fp_norm_bf16"] = 50.0; s = pa.summarize(pa.finalize_rows([{k: v for k, v in r.items() if not k.startswith(("nerr", "eof", "eligible", "d_"))} for r in rows], OLD), n_boot=10, seed=1)
        self.assertEqual({p for p, *_ in s}, {"all_real", "eligible_real_old", "all_matched_swapped"}); self.assertEqual(s[("all_matched_swapped", "matched_swapped", "INT4", "err_bf16")]["n"], 1)
        self.assertEqual(s[("eligible_real_old", "real", "INT16", "err_bf16")]["n"], s[("all_real", "real", "INT16", "err_bf16")]["n"] - 1)                  # the one item with a FP effect norm of 50 is not above five floors of 20

    def test_values_that_are_none_are_left_out_and_counted(self):
        rows = [dict(r) for r in self.rows]; k = 0
        for r in rows:
            if r["state"] == "INT4" and k < 3:
                r["cos_arith_vs_err"] = None; k += 1
        s = pa.summarize(rows, n_boot=10, seed=1); self.assertEqual(s[("all_real", "real", "INT4", "cos_arith_vs_err")]["n_missing"], 3)

    def test_a_quantity_that_no_row_of_the_group_has_is_not_listed(self):
        s = pa.summarize(self.rows, n_boot=10, seed=1); self.assertNotIn(("all_real", "real", "INT4", "batch_FF_bf16"), s); self.assertNotIn(("all_real", "real", "INT16", "batch_QQ_fp32"), s); self.assertIn(("all_real", "real", "INT4", "err_fp32"), s)

    def test_the_coverage_counts_the_eligible_items_under_each_floor_definition_and_mode(self):
        rows = table(n_features=6, per=3)
        for r in [r for r in rows if r["state"] == "INT4"][:4]:
            r["fp_norm_bf16"] = 50.0                                                                  # not above five old floors of 20
        self.rows = pa.finalize_rows(rows, OLD); c = pa.coverage(self.rows); r4 = [r for r in self.rows if r["state"] == "INT4"]; self.assertEqual(c[("real", "INT4", "old", "bf16")]["eligible"], len(r4) - 4)
        self.assertEqual(c[("real", "INT4", "old", "bf16")], {"eligible": sum(r["eligible_old_bf16"] for r in r4), "items": len(r4)}); self.assertEqual(c[("real", "INT16", "mode", "fp32")]["items"], len([r for r in self.rows if r["state"] == "INT16"]))


def summ(**kw):
    """A summary dict of the quantities the readings read: (kind, state, quantity) -> {point, low, high}."""
    base = {("INT4", "d_cos"): (0.0, -0.01, 0.01), ("INT4", "d_nerr_old"): (0.0, -0.01, 0.01), ("INT4", "cos_bf16"): (0.6, 0.55, 0.65), ("INT4", "cos_fp32"): (0.6, 0.55, 0.65), ("INT4", "eof_mode_bf16"): (6.0, 5.0, 7.0), ("INT4", "eof_mode_fp32"): (6.0, 5.0, 7.0),
            ("INT16", "err_bf16"): (10.0, 9.0, 11.0), ("INT16", "err_fp32"): (11.0, 10.0, 12.0), ("INT4", "arith_qq_over_err_bf16"): (0.05, 0.04, 0.06)}
    base.update({(k.split("__")[0], k.split("__")[1]): v for k, v in kw.items()})
    return {("all_real", "real", st, q): {"point": p, "low": lo, "high": hi} for (st, q), (p, lo, hi) in base.items()}


class Readings(unittest.TestCase):
    def test_the_arithmetic_change_of_the_cosine_counts_when_the_interval_excludes_zero_and_the_median_is_at_least_the_margin(self):
        self.assertEqual(pa.readings(summ(INT4__d_cos=(-0.10, -0.12, -0.08)))["A"]["cosine"], "changes"); self.assertEqual(pa.readings(summ(INT4__d_cos=(0.10, 0.02, 0.2)))["A"]["cosine"], "changes")
        self.assertEqual(pa.readings(summ(INT4__d_cos=(-0.0999, -0.12, -0.08)))["A"]["cosine"], "separated but below the margin"); self.assertEqual(pa.readings(summ(INT4__d_cos=(0.2, -0.05, 0.4)))["A"]["cosine"], "not separated from zero")
        self.assertEqual(pa.readings(summ())["A"]["cosine"], "not separated from zero"); self.assertEqual(pa.readings(summ(INT4__d_cos=(0.2, 0.0, 0.4)))["A"]["cosine"], "not separated from zero"); self.assertEqual(pa.readings(summ(INT4__d_cos=(-0.2, -0.4, 0.0)))["A"]["cosine"], "not separated from zero")

    def test_the_normalized_error_uses_the_margin_015_and_any_change_is_reported(self):
        r = pa.readings(summ(INT4__d_nerr_old=(0.15, 0.1, 0.2))); self.assertEqual(r["A"]["normalized_error"], "changes"); self.assertTrue(r["A"]["any_change"]); self.assertFalse(pa.readings(summ(INT4__d_nerr_old=(0.1499, 0.1, 0.2)))["A"]["any_change"])
        self.assertEqual(pa.readings(summ(INT4__d_nerr_old=(0.1499, 0.1, 0.2)))["A"]["normalized_error"], "separated but below the margin")

    def test_the_drift_persists_only_if_in_both_modes_the_error_in_floors_has_a_lower_end_above_one_and_the_cosine_is_below_09(self):
        self.assertTrue(pa.readings(summ())["B"]["persists_in_both_modes"]); r = pa.readings(summ(INT4__eof_mode_fp32=(1.5, 0.9, 2.0))); self.assertFalse(r["B"]["persists_in_both_modes"]); self.assertTrue(r["B"]["bf16"]); self.assertFalse(r["B"]["fp32"])
        r2 = pa.readings(summ(INT4__cos_bf16=(0.95, 0.9, 0.99))); self.assertFalse(r2["B"]["bf16"]); self.assertTrue(r2["B"]["fp32"]); self.assertFalse(pa.readings(summ(INT4__eof_mode_bf16=(6.0, 1.0, 7.0)))["B"]["bf16"])

    def test_the_rounding_reference_depends_on_the_arithmetic_when_the_int16_errors_differ_by_a_factor_of_two_either_way(self):
        self.assertFalse(pa.readings(summ())["C"]["depends"]); self.assertTrue(pa.readings(summ(INT16__err_fp32=(20.0, 19.0, 21.0)))["C"]["depends"]); self.assertTrue(pa.readings(summ(INT16__err_fp32=(5.0, 4.0, 6.0)))["C"]["depends"])
        self.assertFalse(pa.readings(summ(INT16__err_fp32=(19.99, 19.0, 21.0)))["C"]["depends"]); self.assertAlmostEqual(pa.readings(summ(INT16__err_fp32=(5.0, 4.0, 6.0)))["C"]["ratio"], 2.0)

    def test_the_arithmetic_change_relative_to_the_weight_state_change_is_read_in_the_declared_bands(self):
        for v, band in ((0.05, "small"), (0.1999, "small"), (0.2, "intermediate"), (0.5, "intermediate"), (0.5001, "comparable")):
            self.assertEqual(pa.readings(summ(INT4__arith_qq_over_err_bf16=(v, v - 0.01, v + 0.01)))["D"]["band"], band, v)

    def test_a_missing_quantity_gives_no_reading_instead_of_a_guess(self):
        s = summ(); del s[("all_real", "real", "INT4", "d_cos")]; r = pa.readings(s); self.assertIsNone(r["A"]["cosine"])


class Controls(unittest.TestCase):
    def test_the_control_summary_gives_the_median_batching_difference_per_mode_and_its_ratio_to_the_arithmetic_change_of_the_fp_cell(self):
        rows = [row(f, f, state="INT16", batch_FF_bf16=0.1 * (f + 1), batch_FF_fp32=0.2 * (f + 1), batch_QQ_bf16=0.3, batch_QQ_fp32=0.4, arith_ff=1.0, fp_norm_bf16=100.0) for f in range(5)]
        rows.append(row(9, 9, state="INT4", batch_FF_bf16=100.0, batch_FF_fp32=100.0, batch_QQ_bf16=1.0, batch_QQ_fp32=1.0))                                  # a row of another state must not enter the control
        c = pa.controls_summary(rows); self.assertAlmostEqual(c["median_batch_FF_bf16"], 0.3); self.assertAlmostEqual(c["median_batch_FF_fp32"], 0.6); self.assertAlmostEqual(c["median_arith_ff"], 1.0); self.assertAlmostEqual(c["arith_ff_over_batch_FF_bf16"], 1.0 / 0.3)
        self.assertAlmostEqual(c["median_batch_FF_bf16_over_fp_norm"], 0.003); self.assertEqual(c["n"], 5)


class Tables(unittest.TestCase):
    def test_the_item_table_and_the_summary_table_are_written_with_one_row_per_item_and_per_summarized_quantity(self):
        rows = pa.finalize_rows(table(n_features=4, per=2), OLD); s = pa.summarize(rows, n_boot=10, seed=1)
        with tempfile.TemporaryDirectory() as d:
            pa.write_tables(rows, s, d)
            with open(Path(d) / "per_item.csv", newline="") as fh:
                items = list(csv.DictReader(fh))
            with open(Path(d) / "summary.csv", newline="") as fh:
                sm = list(csv.DictReader(fh))
        self.assertEqual(len(items), len(rows)); self.assertEqual(len(sm), len(s)); self.assertIn("eof_mode_fp32", items[0]); self.assertEqual({r["population"] for r in sm}, {"all_real", "eligible_real_old"})
        self.assertEqual(set(sm[0]), {"population", "kind", "state", "quantity", "point", "low", "high", "n", "n_missing", "n_features", "n_clusters"})
        k = ("all_real", "real", "INT4", "err_bf16"); w = [r for r in sm if (r["population"], r["kind"], r["state"], r["quantity"]) == k][0]
        self.assertAlmostEqual(float(w["point"]), s[k]["point"]); self.assertAlmostEqual(float(w["low"]), s[k]["low"]); self.assertAlmostEqual(float(w["high"]), s[k]["high"]); self.assertEqual(int(w["n"]), s[k]["n"])


if __name__ == "__main__":
    unittest.main()
