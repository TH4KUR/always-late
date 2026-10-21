"""D-035 Task 1 tests for scripts/g0/phase1_association_inventory.py (plan: docs/superpowers/plans/2026-10-21-language-association-retention.md, Task 1).
The inventory decides, from the SCHEMA of saved files (keys, shapes, dtypes, language order, hashes; never values), which quantities the association-retention study needs are
available, derivable on the CPU, incomplete, or need a new collection. Stdlib only: the pure logic is tested on dictionaries that describe files; the loader that describes real
.pt files (it needs torch) is tested in scripts/g0/tests_torch/test_phase1_association_inventory_torch.py."""
import sys, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import phase1_association_inventory as inv   # noqa: E402

L = ["eng_Latn", "hin_Deva", "khm_Khmr"]
F = 8
ALL_CONDITIONS = ["FP", "INT8", "INT7", "INT6", "INT5", "INT4", "INT16", "INT12", "INT0"]


def T(*shape, dtype="torch.float32"):
    return {"type": "tensor", "shape": list(shape), "dtype": dtype}


def LANGS(codes):
    return {"type": "list", "len": len(codes), "elem": "str", "values": list(codes)}


def D(**items):
    return {"type": "dict", "items": items}


def S(kind, value):
    return {"type": "scalar", "kind": kind, "value": value}


def rec(name, top, sha="a" * 64):
    return {"path": f"/saved/{name}", "name": name, "sha256": sha, "size": 123, "top": top}


def legacy_survival(langs=L, f=F, bits=(6, 4), with_profiles=True):
    top = {"languages": LANGS(langs), "bits": {"type": "list", "len": len(bits), "elem": "int"}, "n": {"type": "list", "len": len(langs), "elem": "int"},
           "corr": D(**{str(b): T(len(langs), f) for b in bits}),
           "loss_sum": D(FP={"type": "list", "len": len(langs), "elem": "float"}), "loss_n": D(FP={"type": "list", "len": len(langs), "elem": "int"}),
           "pooled_corr": D(**{str(b): T(f) for b in bits}), "pooled_firing_rate": T(f)}
    if with_profiles:
        top.update({"firing_rate": T(len(langs), f), "mean_activation": T(len(langs), f), "max_activation": T(len(langs), f)})
    return rec("survival.pt", top)


def legacy_selectivity(langs=L, f=F, drop=()):
    top = {"languages": LANGS(langs), "rates": T(len(langs), f), "mean_activation": T(len(langs), f), "S": T(f), "S_rank": T(f), "peak": T(f, dtype="torch.int64"),
           "supported": T(f, dtype="torch.bool"), "selective": T(f, dtype="torch.bool"), "controls": T(3, dtype="torch.int64"), "nu": T(len(langs), f),
           "stable_count": T(f, dtype="torch.int64"), "strata": T(f, dtype="torch.int64")}
    for k in drop:
        top.pop(k)
    return rec("selectivity.pt", top)


def sidecar_profiles(langs=L, f=F, conditions=ALL_CONDITIONS, drop=None, bad_shape=None):
    n = len(langs); conds = {}
    for c in conditions:
        fields = {"n_tokens": T(n, dtype="torch.int64"), "positive_counts": T(n, f, dtype="torch.int64"), "activation_sums": T(n, f, dtype="torch.float64"),
                  "activation_max": T(n, f, dtype="torch.float64")}
        if drop and drop[0] == c:
            fields.pop(drop[1])
        if bad_shape and bad_shape[0] == c:
            fields[bad_shape[1]] = T(n, f - 1, dtype="torch.int64")
        conds[c] = D(**fields)
    return rec("profiles.pt", {"schema_version": S("int", 1), "languages": LANGS(langs), "feature_ids": T(f, dtype="torch.int64"), "conditions": D(**conds)}, sha="b" * 64)


def verdicts(result):
    return {q: v["verdict"] for q, v in result["quantities"].items()}


class VerdictTests(unittest.TestCase):
    def test_legacy_files_give_distinct_verdicts(self):
        v = verdicts(inv.inventory_from_records([legacy_survival(), legacy_selectivity()]))
        self.assertEqual(v["fp_language_profiles"], "available")
        self.assertEqual(v["all_language_correlations"], "available")
        self.assertEqual(v["fp_tagging_cohort"], "available")
        self.assertEqual(v["quantized_language_profiles"], "new_collection_required")
        self.assertEqual(v["quantized_peak_and_selectivity"], "new_collection_required")
        self.assertEqual(v["sentence_level_summaries"], "new_collection_required")
        self.assertEqual(v["s_vs_nu_cohort_comparison"], "derivable")
        self.assertEqual(v["fp_baseline_on_devtest"], "derivable")

    def test_correlations_are_never_mistaken_for_response_profiles(self):
        r = inv.inventory_from_records([legacy_survival(with_profiles=False), legacy_selectivity()])
        v = verdicts(r)
        self.assertEqual(v["all_language_correlations"], "available")
        self.assertEqual(v["fp_language_profiles"], "new_collection_required")
        self.assertEqual(v["quantized_language_profiles"], "new_collection_required")
        self.assertIn("correlation", r["quantities"]["quantized_language_profiles"]["evidence"].lower())

    def test_complete_sidecar_makes_quantized_profiles_available_and_peaks_derivable(self):
        r = inv.inventory_from_records([sidecar_profiles()])
        v = verdicts(r)
        self.assertEqual(v["quantized_language_profiles"], "available")
        self.assertEqual(v["quantized_peak_and_selectivity"], "derivable")
        self.assertEqual(v["fp_language_profiles"], "available")
        self.assertEqual(v["sentence_level_summaries"], "new_collection_required")      # sentence shards are a separate artifact (Task 2)
        self.assertIn("profiles.pt", r["quantities"]["quantized_language_profiles"]["evidence"])

    def test_sidecar_missing_a_condition_is_incomplete_not_available(self):
        r = inv.inventory_from_records([sidecar_profiles(conditions=[c for c in ALL_CONDITIONS if c != "INT4"])])
        self.assertEqual(r["quantities"]["quantized_language_profiles"]["verdict"], "incomplete")
        self.assertIn("INT4", r["quantities"]["quantized_language_profiles"]["evidence"])
        self.assertNotEqual(r["quantities"]["quantized_peak_and_selectivity"]["verdict"], "derivable")

    def test_sidecar_missing_a_field_of_one_condition_is_incomplete(self):
        r = inv.inventory_from_records([sidecar_profiles(drop=("INT6", "activation_max"))])
        q = r["quantities"]["quantized_language_profiles"]
        self.assertEqual(q["verdict"], "incomplete")
        self.assertIn("INT6", q["evidence"]); self.assertIn("activation_max", q["evidence"])

    def test_incomplete_fp_condition_does_not_make_fp_profiles_available(self):
        r = inv.inventory_from_records([sidecar_profiles(conditions=[c for c in ALL_CONDITIONS if c != "FP"])])
        self.assertNotEqual(r["quantities"]["fp_language_profiles"]["verdict"], "available")

    def test_selectivity_missing_a_required_key_is_incomplete(self):
        r = inv.inventory_from_records([legacy_survival(), legacy_selectivity(drop=("controls",))])
        q = r["quantities"]["fp_tagging_cohort"]
        self.assertEqual(q["verdict"], "incomplete"); self.assertIn("controls", q["evidence"])
        self.assertEqual(r["quantities"]["s_vs_nu_cohort_comparison"]["verdict"], "derivable")                # does not need `controls`

    def test_unknown_file_is_listed_and_ignored(self):
        r = inv.inventory_from_records([legacy_survival(), rec("other.pt", {"foo": T(2)})])
        kinds = {f["name"]: f["kind"] for f in r["files"]}
        self.assertEqual(kinds["other.pt"], "unknown"); self.assertEqual(kinds["survival.pt"], "legacy_survival")


class ProvenanceTests(unittest.TestCase):
    def test_language_order_mismatch_is_rejected(self):
        with self.assertRaisesRegex(inv.InventoryError, "language"):
            inv.inventory_from_records([legacy_survival(), legacy_selectivity(langs=[L[1], L[0], L[2]])])

    def test_feature_count_mismatch_is_rejected(self):
        with self.assertRaisesRegex(inv.InventoryError, "feature"):
            inv.inventory_from_records([legacy_survival(f=F), legacy_selectivity(f=F + 1)])

    def test_duplicate_language_codes_are_rejected(self):
        with self.assertRaisesRegex(inv.InventoryError, "duplicate"):
            inv.inventory_from_records([legacy_survival(langs=[L[0], L[0], L[2]])])

    def test_expected_language_roster_is_enforced(self):
        with self.assertRaisesRegex(inv.InventoryError, "expected"):
            inv.inventory_from_records([legacy_survival()], expected_languages=L + ["fra_Latn"])
        inv.inventory_from_records([legacy_survival()], expected_languages=L)            # the right roster passes

    def test_sidecar_array_with_wrong_shape_is_rejected(self):
        with self.assertRaisesRegex(inv.InventoryError, "shape"):
            inv.inventory_from_records([sidecar_profiles(bad_shape=("INT5", "positive_counts"))])

    def test_array_with_wrong_language_dimension_is_rejected(self):
        bad = legacy_survival(); bad["top"]["firing_rate"] = T(len(L) + 1, F)
        with self.assertRaisesRegex(inv.InventoryError, "shape"):
            inv.inventory_from_records([bad])

    def test_sidecar_and_legacy_must_agree_on_languages_and_features(self):
        with self.assertRaisesRegex(inv.InventoryError, "language"):
            inv.inventory_from_records([legacy_survival(), sidecar_profiles(langs=L[:2])])
        with self.assertRaisesRegex(inv.InventoryError, "feature"):
            inv.inventory_from_records([legacy_survival(f=F), sidecar_profiles(f=F + 2)])

    def test_schema_version_other_than_one_is_rejected(self):
        bad = sidecar_profiles(); bad["top"]["schema_version"] = S("int", 2)
        with self.assertRaisesRegex(inv.InventoryError, "schema"):
            inv.inventory_from_records([bad])


class ReportTests(unittest.TestCase):
    def test_result_records_provenance_not_values(self):
        r = inv.inventory_from_records([legacy_survival(), legacy_selectivity()])
        self.assertEqual(r["languages"], L); self.assertEqual(r["n_features"], F)
        f = {x["name"]: x for x in r["files"]}
        self.assertEqual(f["survival.pt"]["sha256"], "a" * 64); self.assertEqual(f["survival.pt"]["size"], 123)

    def test_markdown_names_every_quantity_with_its_verdict_and_hash_prefix(self):
        md = inv.render_markdown(inv.inventory_from_records([legacy_survival(), legacy_selectivity()]))
        for q in inv.QUANTITIES:
            self.assertIn(q, md)
        self.assertIn("new_collection_required", md); self.assertIn("derivable", md); self.assertIn("a" * 12, md)

    def test_required_conditions_are_the_plan_ladder(self):
        self.assertEqual(inv.REQUIRED_CONDITIONS, ALL_CONDITIONS)


if __name__ == "__main__":
    unittest.main()
