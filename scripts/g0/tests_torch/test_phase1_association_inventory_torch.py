"""D-035 Task 1 tests (torch side) for scripts/g0/phase1_association_inventory.py: describing real .pt files (schema only, no values, no pickled code) and the frozen cohort
manifest computed from the saved mask/eq files. Hand-computed fixture of 3 languages and 6 features."""
import hashlib, json, shutil, sys, tempfile, unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import phase1_association_inventory as inv   # noqa: E402

LANGS = ["aaa_Latn", "bbb_Latn", "ccc_Latn"]


class Evil:
    """An arbitrary Python object: loading it needs pickled code, which the inventory must refuse."""


def pick_n():
    """A token count for which float32(50 / n) * n is BELOW 50 (the float32 rate of exactly 50 positives): rounding of positive counts must absorb it."""
    for n in range(2000, 6000):
        if float(np.float32(50 / n)) * n < 50:
            return n
    raise AssertionError("no such n")


N = pick_n()


def rate(count, n=N):
    return float(np.float32(count / n))


def write_run(d, with_extra=False):
    """Devtest counts per feature (rows = languages A, B, C): f0 [60,10,0] dev peak A; f1 [5,50,3] dev peak B; f2 [0,0,49] dev peak C; f3 [40,40,0] dev peak A (tie at the
    maximum on devtest AND on dev); f4 [70,20,0] dev peak B; f5 is a control. Selective = f0..f4."""
    d = Path(d); d.mkdir(parents=True, exist_ok=True)
    test_counts = [[60, 10, 0], [5, 50, 3], [0, 0, 49], [40, 40, 0], [70, 20, 0], [1, 1, 1]]           # [feature][language]
    fr = torch.tensor([[rate(test_counts[f][l]) for f in range(6)] for l in range(3)], dtype=torch.float32)
    torch.save({"languages": LANGS, "bits": [6], "n": [N, N, N], "corr": {6: torch.zeros(3, 6)}, "firing_rate": fr, "mean_activation": fr.clone(), "max_activation": fr.clone(),
                "loss_sum": {"FP": [1.0, 1.0, 1.0]}, "loss_n": {"FP": [1, 1, 1]}, "pooled_corr": {6: torch.zeros(6)}, "pooled_firing_rate": torch.zeros(6)}, d / "survival.pt")
    dev_counts = [[60, 10, 0], [5, 60, 3], [0, 0, 60], [60, 60, 0], [10, 80, 0], [1, 1, 1]]
    dev = torch.tensor([[dev_counts[f][l] / 1000 for f in range(6)] for l in range(3)], dtype=torch.float32)
    sel = {"languages": LANGS, "rates": dev, "mean_activation": dev.clone(), "S": torch.ones(6), "S_rank": torch.ones(6), "peak": torch.tensor([0, 1, 2, 0, 1, 0]),
           "supported": torch.ones(6, dtype=torch.bool), "selective": torch.tensor([True, True, True, True, True, False]), "controls": torch.tensor([5]), "nu": dev.clone(),
           "stable_count": torch.full((6,), 100), "strata": torch.zeros(6, dtype=torch.long), "null_cutoff": 1.5}
    torch.save(sel, d / "selectivity.pt")
    (d / "meta.json").write_text(json.dumps({"corpus": "flores", "regime": "mask", "stream": "eq", "seed": 0, "commit": "abc", "model": "gemma-2-2b", "sae_release": "rel", "sae_id": "sid",
                                             "hook": "blocks.12.hook_resid_post", "bos_id": 2, "pad_id": 0, "batch_size": 32, "max_tokens": 1024, "bits": [6], "n_feat": 6,
                                             "equal_budget": {"dev_tokens_per_language": 1000, "devtest_tokens_per_language": N, "seed": 0}}))
    return d


class DescribeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_describe_pt_records_hash_shapes_languages_and_no_values(self):
        p = self.tmp / "x.pt"; torch.save({"languages": ["a", "b"], "rates": torch.full((2, 3), 0.123456), "nested": {"k": torch.zeros(4, dtype=torch.int64)}, "n": [5, 6]}, p)
        r = inv.describe_pt(p)
        self.assertEqual(r["sha256"], hashlib.sha256(p.read_bytes()).hexdigest()); self.assertEqual(r["size"], p.stat().st_size)
        self.assertEqual(r["top"]["languages"]["values"], ["a", "b"])
        self.assertEqual(r["top"]["rates"], {"type": "tensor", "shape": [2, 3], "dtype": "torch.float32"})
        self.assertEqual(r["top"]["nested"]["items"]["k"]["shape"], [4])
        self.assertNotIn("0.123456", json.dumps(r))

    def test_describe_pt_refuses_files_that_need_pickled_code(self):
        p = self.tmp / "evil.pt"; torch.save({"obj": Evil()}, p)
        with self.assertRaises(inv.InventoryError):
            inv.describe_pt(p)

    def test_build_inventory_on_real_files_gives_the_legacy_verdicts(self):
        d = write_run(self.tmp / "run")
        r = inv.build_inventory([d / "survival.pt", d / "selectivity.pt", d / "meta.json"], expected_languages=LANGS)
        v = {q: x["verdict"] for q, x in r["quantities"].items()}
        self.assertEqual(v["fp_language_profiles"], "available"); self.assertEqual(v["quantized_language_profiles"], "new_collection_required")
        self.assertEqual(r["n_features"], 6); self.assertEqual(r["languages"], LANGS)
        self.assertEqual({f["name"]: f["kind"] for f in r["files"]}["meta.json"], "unknown")


class FrozenManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp()); cls.m = inv.frozen_manifest(write_run(cls.tmp / "run"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_cohort_sizes_and_per_language_counts_keyed_by_language_code(self):
        c = self.m["cohort"]
        self.assertEqual((c["supported"], c["selective"], c["controls"], c["union"]), (6, 5, 1, 6))
        self.assertEqual(c["selective_per_language"], {"aaa_Latn": 2, "bbb_Latn": 2, "ccc_Latn": 1})
        self.assertEqual(self.m["languages"], LANGS); self.assertEqual(self.m["n_features"], 6)

    def test_dev_ties_are_counted_among_selective_features_only(self):
        self.assertEqual(self.m["dev_peak_ties_among_selective"], 1)                          # f3: dev rates [0.06, 0.06, 0]

    def test_fp_baseline_on_devtest_uses_rounded_positive_counts(self):
        b = self.m["fp_baseline_on_devtest"]
        self.assertEqual(b["selective"], 5)
        self.assertEqual(b["unique_peak_at_dev_language"], 3)                                  # f0, f1, f2
        self.assertEqual(b["tied_at_maximum"], 1)                                              # f3
        self.assertEqual(b["positives_at_dev_language_ge_min"], 2)                             # f0 (60) and f1 (exactly 50: float32 rate * n is just below 50)
        self.assertEqual(b["established"], 2)
        self.assertEqual(b["per_language"], {"aaa_Latn": [1, 2], "bbb_Latn": [1, 2], "ccc_Latn": [0, 1]})

    def test_provenance_and_hashes(self):
        p = self.m["provenance"]
        self.assertEqual((p["regime"], p["stream"], p["seed"], p["sae_id"], p["bos_id"], p["commit"]), ("mask", "eq", 0, "sid", 2, "abc"))
        self.assertEqual(p["equal_budget"]["devtest_tokens_per_language"], N)
        d = self.tmp / "run"
        self.assertEqual(self.m["files"]["selectivity.pt"]["sha256"], hashlib.sha256((d / "selectivity.pt").read_bytes()).hexdigest())
        self.assertEqual(self.m["devtest_tokens_per_language"], {l: N for l in LANGS})

    def test_manifest_contains_no_feature_id_lists(self):
        self.assertNotIn("feature_ids", json.dumps(self.m))

    def test_language_order_mismatch_between_files_is_rejected(self):
        d = write_run(self.tmp / "bad"); s = torch.load(d / "selectivity.pt", weights_only=True); s["languages"] = [LANGS[1], LANGS[0], LANGS[2]]; torch.save(s, d / "selectivity.pt")
        with self.assertRaisesRegex(inv.InventoryError, "language"):
            inv.frozen_manifest(d)

    def test_token_count_disagreeing_with_the_recorded_budget_is_rejected(self):
        d = write_run(self.tmp / "bad2"); m = json.loads((d / "meta.json").read_text()); m["equal_budget"]["devtest_tokens_per_language"] = N + 1; (d / "meta.json").write_text(json.dumps(m))
        with self.assertRaisesRegex(inv.InventoryError, "budget"):
            inv.frozen_manifest(d)


class CliTests(unittest.TestCase):
    def test_cli_writes_a_report_and_a_manifest_that_agree_with_the_library(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            d = write_run(tmp / "run"); out = tmp / "out"
            rc = inv.main(["--run", str(d), "--out", str(out), "--expected-languages", ",".join(LANGS), "--inspected", "fixture run directory", "--not-inspected", "every other cache"])
            self.assertEqual(rc, 0)
            md = (out / "artifact_inventory.md").read_text(); man = json.loads((out / "frozen_cohort_manifest.json").read_text())
            for q in inv.QUANTITIES:
                self.assertIn(q, md)
            self.assertIn("fixture run directory", md); self.assertIn("every other cache", md); self.assertIn("not a search", md)
            self.assertEqual(man, json.loads(json.dumps(inv.frozen_manifest(d))))
            self.assertEqual(man["fp_baseline_on_devtest"]["established"], 2)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_cli_refuses_a_wrong_language_roster(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            d = write_run(tmp / "run")
            with self.assertRaises(inv.InventoryError):
                inv.main(["--run", str(d), "--out", str(tmp / "out"), "--expected-languages", "x,y,z"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
