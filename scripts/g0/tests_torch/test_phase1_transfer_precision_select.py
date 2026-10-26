import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phase1_transfer_precision_select as ps

SALT = "test-salt"


def make_core(n_features=40, n_ctx=16, short=()):
    ids = [100 + 3 * i for i in range(n_features)]
    return {"feature_ids": ids, "direction_ids": [9000 + i for i in ids], "feature_language_index": [i % 5 for i in range(n_features)],
            "contexts": [[1000 * k + j for j in range(2 if k in short else n_ctx)] for k in range(n_features)], "languages": ["a"] * 5}


def make_norms(core, seed=0):
    rng = np.random.default_rng(seed); out = {}
    for f, ctx in zip(core["feature_ids"], core["contexts"]):
        base = float(rng.uniform(50, 500))
        for s in ctx:
            out[(f, s)] = base * float(rng.uniform(0.8, 1.2))
    return out


class Select(unittest.TestCase):
    def setUp(self):
        self.core = make_core(); self.norms = make_norms(self.core)

    def test_the_selection_has_a_fixed_number_of_features_per_fp_effect_quartile_and_the_first_contexts_of_the_core(self):
        r = ps.select(self.core, self.norms, SALT, per_quartile=5, n_contexts=4); self.assertEqual(len(r["features"]), 20); self.assertEqual([sum(1 for f in r["features"] if f["quartile"] == q) for q in range(4)], [5, 5, 5, 5])
        for f in r["features"]:
            k = self.core["feature_ids"].index(f["feature"]); self.assertEqual(f["contexts"], self.core["contexts"][k][:4]); self.assertEqual(f["direction_id"], self.core["direction_ids"][k])

    def test_the_quartile_is_the_rank_of_the_median_fp_effect_norm_over_the_features_with_ties_to_the_smaller_id(self):
        r = ps.select(self.core, self.norms, SALT, per_quartile=10, n_contexts=4); med = {f: float(np.median([self.norms[(f, s)] for s in c])) for f, c in zip(self.core["feature_ids"], self.core["contexts"])}
        order = sorted(med, key=lambda f: (med[f], f)); q = {f: (4 * i) // len(order) for i, f in enumerate(order)}
        self.assertTrue(all(q[f["feature"]] == f["quartile"] for f in r["features"])); self.assertTrue(all(abs(f["feature_fp_norm"] - med[f["feature"]]) < 1e-9 for f in r["features"]))
        ties = {(f, s): 7.0 for (f, s) in self.norms}; rt = ps.select(self.core, ties, SALT, per_quartile=10, n_contexts=4); self.assertTrue(all(f["quartile"] == (4 * sorted(self.core["feature_ids"]).index(f["feature"])) // 40 for f in rt["features"]))

    def test_inside_a_quartile_the_features_with_the_lowest_salted_hash_are_chosen_in_that_order_and_the_salt_changes_the_choice(self):
        r = ps.select(self.core, self.norms, SALT, per_quartile=4, n_contexts=4); hs = lambda f: hashlib.sha256(f"{SALT}:{f}".encode()).hexdigest()
        for q in range(4):
            chosen = [f["feature"] for f in r["features"] if f["quartile"] == q]; members = [f["feature"] for f in r["all_features"] if f["quartile"] == q]
            self.assertEqual(chosen, sorted(members, key=hs)[:4]); self.assertEqual([f["rank_hash"] for f in r["features"] if f["quartile"] == q], [hs(c) for c in chosen])
        other = ps.select(self.core, self.norms, "another-salt", per_quartile=4, n_contexts=4); self.assertNotEqual([f["feature"] for f in r["features"]], [f["feature"] for f in other["features"]])

    def test_the_result_does_not_depend_on_the_order_of_the_inputs(self):
        a = ps.select(self.core, self.norms, SALT, per_quartile=5, n_contexts=4); perm = list(reversed(range(40)))
        core2 = {**self.core, "feature_ids": [self.core["feature_ids"][i] for i in perm], "direction_ids": [self.core["direction_ids"][i] for i in perm], "feature_language_index": [self.core["feature_language_index"][i] for i in perm], "contexts": [self.core["contexts"][i] for i in perm]}
        norms2 = dict(reversed(list(self.norms.items()))); b = ps.select(core2, norms2, SALT, per_quartile=5, n_contexts=4); self.assertEqual(a["features"], b["features"]); self.assertEqual(a["selection_sha256"], b["selection_sha256"])

    def test_a_feature_with_too_few_contexts_or_a_missing_norm_is_excluded_and_listed_with_its_reason(self):
        core = make_core(short=(3,)); norms = make_norms(core); missing_f = core["feature_ids"][7]; del norms[(missing_f, core["contexts"][7][0])]
        r = ps.select(core, norms, SALT, per_quartile=4, n_contexts=4); reasons = {e["feature"]: e["reason"] for e in r["excluded"]}
        self.assertEqual(reasons[core["feature_ids"][3]], "fewer than 4 contexts"); self.assertEqual(reasons[missing_f], "missing FP effect norm"); self.assertNotIn(core["feature_ids"][3], [f["feature"] for f in r["features"]]); self.assertEqual(len(r["all_features"]), 38)

    def test_exactly_the_needed_number_of_contexts_is_enough_one_fewer_is_not_and_non_finite_norms_count_as_missing(self):
        core = make_core(); core["contexts"][2] = core["contexts"][2][:3]; core["contexts"][5] = core["contexts"][5][:4]; norms = make_norms(core); nan_f = core["feature_ids"][9]; norms[(nan_f, core["contexts"][9][1])] = float("nan"); inf_f = core["feature_ids"][11]; norms[(inf_f, core["contexts"][11][0])] = float("inf")
        r = ps.select(core, norms, SALT, per_quartile=4, n_contexts=4); reasons = {e["feature"]: e["reason"] for e in r["excluded"]}
        self.assertEqual(reasons[core["feature_ids"][2]], "fewer than 4 contexts"); self.assertNotIn(core["feature_ids"][5], reasons); self.assertEqual((reasons[nan_f], reasons[inf_f]), ("missing FP effect norm", "missing FP effect norm"))
        self.assertTrue(all(np.isfinite(f["feature_fp_norm"]) for f in r["all_features"]))

    def test_too_few_features_in_a_quartile_is_an_error_not_a_silent_smaller_sample(self):
        with self.assertRaisesRegex(ValueError, "quartile"):
            ps.select(self.core, self.norms, SALT, per_quartile=11, n_contexts=4)

    def test_the_selection_hash_covers_the_chosen_features_and_contexts_and_changes_with_them(self):
        a = ps.select(self.core, self.norms, SALT, per_quartile=5, n_contexts=4); b = ps.select(self.core, self.norms, SALT, per_quartile=5, n_contexts=3); self.assertNotEqual(a["selection_sha256"], b["selection_sha256"])
        payload = json.dumps(a["features"], sort_keys=True, separators=(",", ":")); self.assertEqual(a["selection_sha256"], hashlib.sha256(payload.encode()).hexdigest())


class Source(unittest.TestCase):
    def write(self, path, rows, extra=None):
        keys = ["feature", "sentence", "kind", "fp_norm"] + (["error_norm", "cosine", "q_norm"] if extra else [])
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
            for r in rows:
                w.writerow({**r, **(extra(r) if extra else {})})

    def test_only_the_fp_effect_norm_of_the_real_direction_is_read_and_quantized_columns_cannot_change_the_selection(self):
        core = make_core(); norms = make_norms(core); rows = [{"feature": f, "sentence": s, "kind": "real", "fp_norm": v} for (f, s), v in norms.items()] + [{"feature": 100, "sentence": 5, "kind": "random1", "fp_norm": 1e9}]
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a.csv", Path(d) / "b.csv"; self.write(a, rows); self.write(b, rows, extra=lambda r: {"error_norm": 1e6 * (r["feature"] % 7), "cosine": -1.0, "q_norm": 1e9})
            na, nb = ps.fp_norms_from_csv(a), ps.fp_norms_from_csv(b); self.assertEqual(na, nb); self.assertEqual(len(na), len(norms)); self.assertEqual(ps.select(core, na, SALT, 5, 4)["selection_sha256"], ps.select(core, nb, SALT, 5, 4)["selection_sha256"])

    def test_a_table_without_the_fp_norm_column_or_with_a_duplicate_real_item_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.csv"; p.write_text("feature,sentence,kind,q_norm\n1,2,real,3.0\n")
            with self.assertRaisesRegex(ValueError, "fp_norm"):
                ps.fp_norms_from_csv(p)
            p.write_text("feature,sentence,kind,fp_norm\n1,2,real,3.0\n1,2,real,4.0\n")
            with self.assertRaisesRegex(ValueError, "twice"):
                ps.fp_norms_from_csv(p)


class Freeze(unittest.TestCase):
    def test_the_freeze_record_holds_the_rule_the_salt_the_input_hashes_the_counts_and_the_selected_features(self):
        core = make_core(); norms = make_norms(core)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / "core.json").write_text(json.dumps(core)); rows = [{"feature": f, "sentence": s, "kind": "real", "fp_norm": v} for (f, s), v in norms.items()]
            with open(root / "t.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["feature", "sentence", "kind", "fp_norm"]); w.writeheader(); w.writerows(rows)
            rec = ps.build_freeze(root / "core.json", root / "t.csv", SALT, per_quartile=5, n_contexts=4)
            self.assertEqual(rec["salt"], SALT); self.assertEqual(rec["inputs_sha256"]["core"], hashlib.sha256((root / "core.json").read_bytes()).hexdigest()); self.assertEqual(rec["inputs_sha256"]["fp_norm_table"], hashlib.sha256((root / "t.csv").read_bytes()).hexdigest())
            self.assertEqual(rec["counts"], {"features_in_core": 40, "excluded": 0, "per_quartile": 5, "n_contexts": 4, "selected_features": 20, "selected_pairs": 80}); self.assertEqual(len(rec["features"]), 20); self.assertIn("rule", rec)
            self.assertNotIn("error_norm", json.dumps(rec)); self.assertEqual(rec["selection_sha256"], ps.select(core, norms, SALT, 5, 4)["selection_sha256"])


if __name__ == "__main__":
    unittest.main()
