"""The baseline replication must run THEIR code. make_faithful_runner.py may only apply edits that are physically
unavoidable (make the notebook export runnable; place the second bf16 model on a second 11 GB card; select one of
their two shipped modes). These tests prove their computation code is untouched. Skipped if Duan's repo copy is absent."""
import ast, difflib, os, py_compile, subprocess, sys, tempfile, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
GEN = HERE.parent / "make_faithful_runner.py"
DUAN = HERE.parents[3] / "duan-sae-feature-survival" / "scripts" / "03_gemma_phase3_streaming_sweep.py"

# every function that computes, quantizes, prunes, tokenizes or aggregates: must be AST-identical to the original
COMPUTATION = ["streaming_feature_metrics", "compute_perplexity", "get_hook_acts", "encode_sae", "quantize_rtn_per_output_channel_tl",
               "target_tl_weight", "magnitude_prune_tl", "pruning_search_match_target", "build_tokens_2d", "clone_state_to_cpu",
               "free_memory", "save_condition_outputs", "load_sae_compat"]
# the only functions allowed to differ (placement / notebook-only display)
ALLOWED_TO_DIFFER = {"load_gemma_tl_model", "restore_model_work", "add_summary"}
EXPERIMENTAL_MARKERS = ["G0_BOS", "SNAP_EVERY", "_WB", "g0_wandb", "G0_RESUME", "weight_perturbation", "_g0_assert_model",
                        "INT0", "g0_core", "prepend_mask", "prepend_keep", "tokens_2d.pt", "run_meta"]


def funcs(src):
    return {n.name: ast.dump(n) for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)}


@unittest.skipUnless(DUAN.exists(), "Duan's repo copy not found next to the repo")
class FaithfulRunner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.orig = DUAN.read_text()
        # Cells 22+ re-define target_tl_weight (WITHOUT W_gate) and others AFTER every result file is written; the code that
        # actually runs is cells 1-21, so that is what the faithful runner must reproduce.
        cls.orig_run = cls.orig.split("# %% Cell 22")[0]
        cls.out = {}
        for mode in ("3a_test", "3b_full"):
            p = Path(cls.tmp.name) / f"faithful_{mode}.py"
            subprocess.run([sys.executable, str(GEN), str(DUAN), str(p), "--mode", mode], check=True, capture_output=True)
            cls.out[mode] = p

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_both_modes_compile(self):
        for p in self.out.values():
            py_compile.compile(str(p), doraise=True)

    def test_every_computation_function_is_ast_identical_to_the_original(self):
        o = funcs(self.orig_run)
        for mode, p in self.out.items():
            f = funcs(p.read_text())
            for name in COMPUTATION:
                self.assertIn(name, o); self.assertIn(name, f)
                self.assertEqual(o[name], f[name], f"{name} differs in faithful {mode} runner")

    def test_only_placement_and_display_functions_differ(self):
        o = funcs(self.orig_run)
        for mode, p in self.out.items():
            f = funcs(p.read_text())
            differing = {n for n in o if n in f and o[n] != f[n]}
            self.assertEqual(differing, ALLOWED_TO_DIFFER, mode)
            self.assertEqual(set(o) - set(f), set(), "no function may be removed")

    def test_no_experimental_feature_leaks_into_the_faithful_runner(self):
        for mode, p in self.out.items():
            s = p.read_text()
            for m in EXPERIMENTAL_MARKERS:
                self.assertNotIn(m, s, f"{m!r} must not appear in the faithful {mode} runner")

    def test_mode_constant_is_one_of_their_two_shipped_values(self):
        self.assertIn('RUN_MODE = "3a_test"', self.out["3a_test"].read_text())
        self.assertIn('RUN_MODE = "3b_full"', self.out["3b_full"].read_text())

    def test_total_diff_before_the_dropped_cells_is_small(self):
        head = self.orig.split("# %% Cell 22")[0]
        for mode, p in self.out.items():
            body = p.read_text().split("\n", 3)[3]          # strip the 3-line provenance header
            body = body.split("# %% Cell 22")[0]
            changed = [l for l in difflib.unified_diff(head.splitlines(), body.splitlines(), lineterm="", n=0)
                       if l[:1] in "+-" and not l.startswith(("+++", "---"))]
            self.assertLessEqual(len(changed), 40, f"{mode}: {len(changed)} changed lines")

    def test_the_redefinition_in_the_dropped_cells_is_the_reason_the_comparison_uses_cells_1_to_21(self):
        tail = self.orig.split("# %% Cell 22")[1]
        self.assertIn("def target_tl_weight", tail)          # redefined without W_gate, only after all results are written
        self.assertIn("W_gate", self.orig_run)
        self.assertNotIn("W_gate", tail.split("def quantize_rtn_per_channel_tl")[0])

    def test_dropped_cells_are_exactly_the_ones_calling_undefined_functions(self):
        tail = self.orig.split("# %% Cell 22")[1]
        for name in ("corrected_ppl_sweep", "compute_ppl_shifted_manual", "restore_work_from_ref"):
            self.assertIn(name, tail)
            self.assertNotIn(f"def {name}", self.orig)


if __name__ == "__main__":
    unittest.main()
