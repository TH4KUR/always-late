"""Guards the claim 'we run THEIR code'. third_party/duan2026 holds their repo pruned to what the Gemma Phase-3 replication needs:
 * pristine/  byte-identical retained originals (hashes in MANIFEST.sha256; the manifest also lists every REMOVED upstream file)
 * scripts/03_gemma_phase3_streaming_sweep.py  the file we run = pristine + an explicit whitelist of edits (verify_vendor.EDITS)
The working copy must equal reconstruct(pristine) EXACTLY, over the whole file including top-level statements, and a committed
mutation suite proves that any other change (to their math, constants, config or structure) is rejected."""
import difflib, hashlib, shutil, sys, tempfile, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import verify_vendor as vv  # noqa: E402

ROOT = HERE.parents[2]
VENDOR = ROOT / "third_party" / "duan2026"
PRISTINE03 = VENDOR / "pristine" / "scripts" / "03_gemma_phase3_streaming_sweep.py"
WORKING03 = VENDOR / "scripts" / "03_gemma_phase3_streaming_sweep.py"
UPSTREAM_TREE_SHA256 = "a84e43dac17e5a9c6b1404e4b502c97d2b184b0168e860035432e4c261e44adc"   # computed from a fresh download of their repo (36 files)
EXPECTED_VENDOR_FILES = {
    "UPSTREAM.md", "MANIFEST.sha256", "working.sha256", "patches/03_working_vs_pristine.diff", "scripts/03_gemma_phase3_streaming_sweep.py",
    "pristine/LICENSE", "pristine/README.md", "pristine/scripts/03_gemma_phase3_streaming_sweep.py", "pristine/scripts/06_gemma_sliding_window_ppl_check.py",
    "pristine/results/tables/phase3_summary_final.csv", "pristine/results/tables/gemma_chunked_vs_sliding_ppl_w2048_s512_FULL.csv",
    "pristine/results/per_feature/phase5b_gemma_rtn_int6_vs_pruning_per_feature_overlap.csv"}


class Manifest(unittest.TestCase):
    def test_manifest_lists_all_36_upstream_files_and_reproduces_the_upstream_tree_hash(self):
        entries, tree = vv.parse_manifest(VENDOR / "MANIFEST.sha256")
        self.assertEqual(len(entries), 36)
        self.assertEqual(vv.tree_hash(entries), tree)
        self.assertEqual(tree, UPSTREAM_TREE_SHA256)

    def test_every_retained_file_is_byte_identical_to_upstream_and_every_removed_file_is_gone(self):
        entries, _ = vv.parse_manifest(VENDOR / "MANIFEST.sha256")
        retained = [e for e in entries if e["status"] == "retained"]
        self.assertEqual(len(retained), 7)                        # LICENSE, README, scripts 03+06, 2 tables, 1 per-feature file
        for e in entries:
            p = VENDOR / "pristine" / e["path"]
            if e["status"] == "retained":
                self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), e["sha256"], e["path"])
            else:
                self.assertFalse(p.exists(), f"removed file still present: {e['path']}")

    def test_a_tampered_pristine_file_is_detected(self):
        with tempfile.TemporaryDirectory() as d:
            shutil.copytree(VENDOR, Path(d) / "v")
            p = Path(d) / "v" / "pristine" / "scripts" / "03_gemma_phase3_streaming_sweep.py"
            p.write_text(p.read_text().replace("FIRING_THRESHOLD = 0.001", "FIRING_THRESHOLD = 0.002"))
            self.assertNotEqual(vv.check_pristine(Path(d) / "v"), [])
        self.assertEqual(vv.check_pristine(VENDOR), [])


class WorkingCopy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pristine, cls.working = PRISTINE03.read_text(), WORKING03.read_text()

    def test_working_copy_is_exactly_the_pristine_file_plus_the_whitelisted_edits(self):
        self.assertEqual(self.working, vv.reconstruct(self.pristine))

    def test_the_whole_diff_is_six_edits_and_the_removed_tail(self):
        head = self.pristine.split(vv.TRUNCATE_AT)[0]
        d = [l for l in difflib.unified_diff(head.splitlines(), self.working.splitlines(), lineterm="", n=0)
             if l[:1] in "+-" and not l.startswith(("+++", "---"))]
        added = [l for l in d if l.startswith("+")]; removed = [l for l in d if l.startswith("-")]
        self.assertEqual(len(added), 5)       # WORK_DEVICE line, flag, dtype, load, restore
        self.assertEqual(len(removed), 4)     # flag, dtype, load, restore (the WORK_DEVICE line is a pure insertion)

    def test_the_removed_tail_is_only_cells_that_call_functions_never_defined_in_the_file(self):
        tail = self.pristine.split(vv.TRUNCATE_AT)[1]
        for name in ("corrected_ppl_sweep", "compute_ppl_shifted_manual", "restore_work_from_ref"):
            self.assertIn(name, tail); self.assertNotIn(f"def {name}", self.pristine)

    def test_the_shipped_3b_full_configuration_is_otherwise_untouched(self):
        for needle in ("TOKEN_BUDGET = 500_000", "BITS_TO_TEST = [8, 7, 6, 5, 4]", "FIRING_THRESHOLD = 0.001", "SEQ_LEN = 256", "LAYER = 12",
                       'SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"', 'SAE_ID = "layer_12/width_16k/canonical"', "(active_corrs > 0.9)"):
            self.assertIn(needle, self.working)


class MutationsAreRejected(unittest.TestCase):
    """Committed mutation suite: each mutation of the WORKING copy must be rejected by working_matches()."""
    MUTATIONS = {
        "survival threshold 0.9 -> 0.91": ("(active_corrs > 0.9)", "(active_corrs > 0.91)"),
        "damage threshold 0.5 -> 0.55": ("(active_corrs < 0.5)", "(active_corrs < 0.55)"),
        "quantizer qmax": ("qmax = (2 ** (bits - 1)) - 1\n    qmin = -(2 ** (bits - 1))\n\n    n_tensors = 0", "qmax = (2 ** (bits - 1)) - 2\n    qmin = -(2 ** (bits - 1))\n\n    n_tensors = 0"),
        "token budget (top-level config)": ("TOKEN_BUDGET = 500_000", "TOKEN_BUDGET = 400_000"),
        "bit-width list (top-level config)": ("BITS_TO_TEST = [8, 7, 6, 5, 4]", "BITS_TO_TEST = [8, 7, 6]"),
        "firing threshold (top-level)": ("FIRING_THRESHOLD = 0.001", "FIRING_THRESHOLD = 0.002"),
        "sequence length (top-level)": ("SEQ_LEN = 256", "SEQ_LEN = 128"),
        "SAE id (top-level)": ('SAE_ID = "layer_12/width_16k/canonical"', 'SAE_ID = "layer_12/width_16k/average_l0_176"'),
        "layer (top-level)": ("LAYER = 12", "LAYER = 20"),
        "deleted statement in compute_perplexity": ("        losses.append(float(loss.item()))\n", ""),
        "pearson numerator": ("numerator = sum_xy - (sum_x * sum_y / n)", "numerator = sum_xy"),
        "re-enabled pruning flag": ("    RUN_PRUNING = False  # [G0-EDIT 1]", "    RUN_PRUNING = True  # [G0-EDIT 1]"),
        "appended statement": ("print(\"Original CPU state saved.\")\n", "print(\"Original CPU state saved.\")\nTOKEN_BUDGET = 1\n"),
    }

    def test_each_mutation_is_rejected_and_the_unmutated_copy_is_accepted(self):
        pristine, working = PRISTINE03.read_text(), WORKING03.read_text()
        self.assertTrue(vv.working_matches(working, pristine))
        for name, (old, new) in self.MUTATIONS.items():
            self.assertEqual(working.count(old), 1, f"mutation anchor for {name!r} must match exactly once")
            self.assertFalse(vv.working_matches(working.replace(old, new), pristine), f"mutation not detected: {name}")


class Layout(unittest.TestCase):
    def test_vendor_tree_contains_exactly_the_expected_files(self):
        found = {str(p.relative_to(VENDOR)) for p in VENDOR.rglob("*") if p.is_file()}
        self.assertEqual(found, EXPECTED_VENDOR_FILES)

    def test_working_copy_hash_file_matches_the_working_copy(self):
        line = (VENDOR / "working.sha256").read_text().split()
        self.assertEqual(hashlib.sha256(WORKING03.read_bytes()).hexdigest(), line[0])


if __name__ == "__main__":
    unittest.main()
