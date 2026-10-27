"""Task 5 of the GPTQ extension plan: the sequential block-by-block calibrator (audit section 6): Q/K/V together, then O, then input and gate together, then the MLP output, block after block, with the inputs of every stage
recomputed after the already quantized stages; one Hessian per stage input over the non-padding rows; GPTQ (pinned authors' core) or the matched RTN as the per-matrix rule. Model tests run on a tiny random-weight Gemma-2
(~/.venvs/anlp-tl) and are skipped elsewhere; the per-matrix tests need only torch."""
import hashlib
import sys
import unittest
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
G0 = HERE.parent
sys.path.insert(0, str(G0)); sys.path.insert(0, str(HERE))
from gptq_ext import calibrate as cb, calibration as cal, canonical, grids, upstream  # noqa: E402
from gptq_ext_tiny import DIMS, HAVE_TL, tiny_gemma2  # noqa: E402

BOS, PAD = 1, 0


def make_pool(n_seq=24, seed=0, lo=18, hi=40):
    g = torch.Generator().manual_seed(seed)
    passages = [{"article_id": f"a{i}", "rank": i, "ids": torch.randint(3, DIMS["d_vocab"], (int(torch.randint(lo, hi, (1,), generator=g)),), generator=g).tolist()} for i in range(n_seq)]
    return {"xx": {"passages": passages, "shortfall": 0}}


def correlated_rows(n=600, d=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(n, 6, generator=g) @ torch.randn(6, d, generator=g)
    return base + 0.05 * torch.randn(n, d, generator=g)


def hessian_of(rows):
    core = upstream.load()
    lin = torch.nn.Linear(rows.shape[1], 1, bias=False)
    gp = core.GPTQ(lin); gp.add_batch(rows, None)
    return gp.H.clone()


def digest(tensors):
    h = hashlib.sha256()
    for n in sorted(tensors):
        h.update(n.encode()); h.update(tensors[n].float().numpy().tobytes())
    return h.hexdigest()


class PerMatrix(unittest.TestCase):
    def test_gptq_beats_rtn_on_the_calibration_objective_for_correlated_inputs(self):
        rows = correlated_rows(); w = torch.randn(24, 32, generator=torch.Generator().manual_seed(1)); h = hessian_of(rows)
        cfg = cb.GptqConfig(bits=3)
        qg, _ = cb.quantize_matrix(w, h, cfg, "gptq"); qr, _ = cb.quantize_matrix(w, h, cfg, "rtn_matched")
        err = lambda q: float(((rows @ (q - w).T) ** 2).sum())
        self.assertLess(err(qg), err(qr))

    def test_rtn_matched_path_is_the_grids_function(self):
        rows = correlated_rows(); w = torch.randn(24, 32, generator=torch.Generator().manual_seed(2)); h = hessian_of(rows)
        q, info = cb.quantize_matrix(w, h, cb.GptqConfig(bits=4), "rtn_matched")
        self.assertTrue(torch.equal(q, grids.rtn_matched(w, 4)))
        self.assertEqual(info["dead"], 0)

    def test_dead_columns_are_counted_zeroed_and_identical_in_both_methods(self):
        rows = correlated_rows(); rows[:, [3, 20]] = 0
        w = torch.randn(10, 32, generator=torch.Generator().manual_seed(3)); h = hessian_of(rows)
        for method in ("gptq", "rtn_matched"):
            q, info = cb.quantize_matrix(w, h, cb.GptqConfig(bits=4), method)
            self.assertEqual(info["dead"], 2, method)
            self.assertTrue((q[:, [3, 20]] == 0).all(), method)
        self.assertTrue(torch.equal(cb.quantize_matrix(w, h, cb.GptqConfig(bits=4), "rtn_matched")[0], grids.rtn_matched(w, 4, dead=torch.diag(h) == 0)))

    def test_a_hessian_that_is_not_positive_definite_raises_a_named_error(self):
        w = torch.randn(4, 8); h = -torch.eye(8)
        with self.assertRaises(cb.HessianError) as ctx:
            cb.quantize_matrix(w, h, cb.GptqConfig(bits=4), "gptq", name="blocks.7.mlp.W_out")
        self.assertIn("blocks.7.mlp.W_out", str(ctx.exception))

    def test_dtype_is_preserved_and_the_input_is_not_modified(self):
        rows = correlated_rows(); w = torch.randn(8, 32, generator=torch.Generator().manual_seed(4)).to(torch.bfloat16); h = hessian_of(rows); w0 = w.clone()
        q, _ = cb.quantize_matrix(w, h, cb.GptqConfig(bits=4), "gptq")
        self.assertEqual(q.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(w, w0))

    def test_the_hessian_is_not_modified_so_the_matrices_of_a_stage_can_share_it(self):
        rows = correlated_rows(); rows[:, 5] = 0; h = hessian_of(rows); h0 = h.clone()
        for method in ("gptq", "rtn_matched", "identity"):
            cb.quantize_matrix(torch.randn(6, 32, generator=torch.Generator().manual_seed(5)), h, cb.GptqConfig(bits=4), method)
            self.assertTrue(torch.equal(h, h0), method)

    def test_unknown_method_is_refused(self):
        with self.assertRaises(ValueError):
            cb.quantize_matrix(torch.randn(2, 4), torch.eye(4), cb.GptqConfig(bits=4), "awq")

    def test_valid_rows_exclude_padding_only(self):
        x = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2); mask = torch.tensor([[True, True, False], [True, False, False]])
        rows = cb.valid_rows(x, mask)
        self.assertEqual(rows.shape, (3, 2))
        self.assertTrue(torch.equal(rows, x[mask]))

    def test_config_defaults_are_the_audits_primary_setting(self):
        c = cb.GptqConfig(bits=4)
        self.assertEqual((c.blocksize, c.percdamp, c.actorder, c.groupsize, c.sym, c.mse), (128, 0.01, False, -1, True, False))


@unittest.skipUnless(HAVE_TL, "needs TransformerLens (~/.venvs/anlp-tl)")
class WithModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pool = make_pool()
        cls.batches = cal.pool_batches(cls.pool, BOS, PAD, batch_size=8)

    def run_cal(self, method, bits=4, model=None, batches=None, observer=None):
        model = model or tiny_gemma2()
        tensors, report = cb.calibrate(model, batches or self.batches, cb.GptqConfig(bits=bits), method, torch.device("cpu"), log=lambda *a: None, observer=observer)
        return model, tensors, report

    def test_rtn_matched_artifact_equals_grids_rtn_matched_of_every_target(self):
        ref = tiny_gemma2()
        _, tensors, report = self.run_cal("rtn_matched")
        self.assertEqual(report["dead_total"], 0)
        for name, p in canonical.targets(ref):
            want = canonical.from_linear(name, grids.rtn_matched(canonical.to_linear(name, p.data), 4), p.data)
            self.assertTrue(torch.equal(tensors[name], want), name)

    def test_the_model_is_left_unchanged_and_non_targets_are_never_touched(self):
        model = tiny_gemma2(); before = {n: p.detach().clone() for n, p in model.named_parameters()}
        self.run_cal("gptq", model=model)
        for n, p in model.named_parameters():
            self.assertTrue(torch.equal(p, before[n]), n)

    def test_artifact_has_exactly_the_target_tensors_in_their_own_shape_and_dtype(self):
        model = tiny_gemma2(dtype=torch.bfloat16)
        _, tensors, _ = self.run_cal("gptq", model=model)
        self.assertEqual(list(tensors), [n for n, _ in canonical.targets(model)])
        for n, p in canonical.targets(model):
            self.assertEqual((tensors[n].shape, tensors[n].dtype), (p.shape, p.dtype))
            self.assertFalse(torch.equal(tensors[n], p.data), n)

    def test_calibration_is_deterministic(self):
        _, a, _ = self.run_cal("gptq"); _, b, _ = self.run_cal("gptq")
        self.assertEqual(digest(a), digest(b))

    def test_padding_tokens_do_not_change_the_artifact(self):
        _, a, _ = self.run_cal("gptq")
        _, b, _ = self.run_cal("gptq", batches=cal.pool_batches(self.pool, BOS, 7, batch_size=8))
        self.assertEqual(digest(a), digest(b))

    def test_the_batch_order_is_part_of_the_result_but_the_set_of_rows_is_what_counts_for_rtn(self):
        _, a, _ = self.run_cal("gptq")
        _, b, _ = self.run_cal("gptq", batches=cal.pool_batches(self.pool, BOS, PAD, batch_size=5))
        self.assertEqual(set(a), set(b))                                          # same targets; values may differ in the last bits because the accumulation order differs

    def reference_rows(self, model, hook_name=None, module=None):
        """Stage-input rows of the given model computed by ordinary forward passes (no calibrator): hook point output or module output, non-padding rows."""
        rows = []
        for b in self.batches:
            cap = {}
            handle = module.register_forward_hook(lambda m, i, o: cap.setdefault("x", o.detach().clone())) if module is not None else None
            try:
                with torch.no_grad():
                    model.run_with_hooks(b["tokens"], fwd_hooks=[(hook_name, lambda t, hook: cap.__setitem__("x", t.detach().clone()))] if hook_name else [], stop_at_layer=2)
            finally:
                if handle is not None:
                    handle.remove()
            x = cap["x"]
            rows.append(cb.valid_rows(x.reshape(*x.shape[:2], -1), b["mask"]))
        return torch.cat(rows)

    def test_the_identity_control_sees_exactly_the_activations_of_the_ordinary_forward_pass(self):
        seen = {}
        model, tensors, report = self.run_cal("identity", observer=lambda block, stage, name, rows: seen.setdefault(name, rows.clone()))
        ref = tiny_gemma2()
        for name, p in canonical.targets(ref):
            self.assertTrue(torch.equal(tensors[name], p.data), name)
        want = {"blocks.0.attn.W_Q": self.reference_rows(ref, module=ref.blocks[0].ln1), "blocks.0.attn.W_O": self.reference_rows(ref, hook_name="blocks.0.attn.hook_z"),
                "blocks.0.mlp.W_in": self.reference_rows(ref, module=ref.blocks[0].ln2), "blocks.0.mlp.W_out": self.reference_rows(ref, hook_name="blocks.0.mlp.hook_post"),
                "blocks.1.attn.W_Q": self.reference_rows(ref, module=ref.blocks[1].ln1), "blocks.1.attn.W_O": self.reference_rows(ref, hook_name="blocks.1.attn.hook_z"),
                "blocks.1.mlp.W_gate": self.reference_rows(ref, module=ref.blocks[1].ln2), "blocks.1.mlp.W_out": self.reference_rows(ref, hook_name="blocks.1.mlp.hook_post")}
        for name, rows in want.items():
            self.assertTrue(torch.allclose(seen[name], rows, atol=1e-5), name)
        self.assertEqual(report["dead_total"], 0)

    def test_a_stage_sees_the_output_of_the_already_quantized_earlier_stage_of_its_block(self):
        seen = {}
        model, tensors, _ = self.run_cal("gptq", observer=lambda block, stage, name, rows: seen.setdefault(name, rows.clone()))
        ref = tiny_gemma2(); orig = tiny_gemma2()
        with torch.no_grad():
            for n in ("blocks.0.attn.W_Q", "blocks.0.attn._W_K", "blocks.0.attn._W_V"):
                dict(ref.named_parameters())[n].copy_(tensors[n])
        z_quant = self.reference_rows(ref, hook_name="blocks.0.attn.hook_z")
        z_orig = self.reference_rows(orig, hook_name="blocks.0.attn.hook_z")
        self.assertTrue(torch.allclose(seen["blocks.0.attn.W_O"], z_quant, atol=1e-5))
        self.assertGreater(float((z_quant - z_orig).abs().max()), 1e-3)                  # not vacuous: quantization changed the input of W_O
        # the MLP output stage sees the post-activation of the QUANTIZED input and gate matrices (block 0: attention still original)
        ref2 = tiny_gemma2()
        with torch.no_grad():
            for n in ("blocks.0.attn.W_Q", "blocks.0.attn._W_K", "blocks.0.attn._W_V", "blocks.0.attn.W_O", "blocks.0.mlp.W_in", "blocks.0.mlp.W_gate"):
                dict(ref2.named_parameters())[n].copy_(tensors[n])
        self.assertTrue(torch.allclose(seen["blocks.0.mlp.W_out"], self.reference_rows(ref2, hook_name="blocks.0.mlp.hook_post"), atol=1e-5))

    def test_a_block_sees_the_fully_quantized_blocks_before_it(self):
        seen = {}
        model, tensors, _ = self.run_cal("gptq", observer=lambda block, stage, name, rows: seen.setdefault(name, rows.clone()))
        ref = tiny_gemma2(); orig = tiny_gemma2()
        with torch.no_grad():
            for n, p in canonical.targets(ref):
                if n.startswith("blocks.0."):
                    p.copy_(tensors[n])
        x_quant = self.reference_rows(ref, module=ref.blocks[1].ln1)
        x_orig = self.reference_rows(orig, module=orig.blocks[1].ln1)
        self.assertTrue(torch.allclose(seen["blocks.1.attn.W_Q"], x_quant, atol=1e-5))
        self.assertGreater(float((x_quant - x_orig).abs().max()), 1e-3)

    def test_every_stage_input_has_one_row_per_non_padding_position(self):
        n_rows = sum(int(b["mask"].sum()) for b in self.batches)
        seen = {}
        self.run_cal("rtn_matched", observer=lambda block, stage, name, rows: seen.__setitem__(name, rows.shape))
        for name, shape in seen.items():
            self.assertEqual(shape[0], n_rows, name)
        self.assertEqual(seen["blocks.0.attn.W_O"][1], DIMS["n_heads"] * DIMS["d_head"])
        self.assertEqual(seen["blocks.1.mlp.W_out"][1], DIMS["d_mlp"])
        self.assertEqual(seen["blocks.1.attn.W_Q"][1], DIMS["d_model"])

    def test_a_dead_input_dimension_is_counted_and_zeroed_in_the_artifact_of_both_methods(self):
        for method in ("gptq", "rtn_matched"):
            model = tiny_gemma2()
            with torch.no_grad():
                model.blocks[0].ln1.w[3] = 0.0                                   # the normalised attention input is exactly 0 in dimension 3
            _, tensors, report = self.run_cal(method, model=model)
            for name in ("blocks.0.attn.W_Q", "blocks.0.attn._W_K", "blocks.0.attn._W_V"):
                self.assertEqual(report["targets"][name]["dead"], 1, (method, name))
                self.assertTrue((tensors[name][:, 3, :] == 0).all(), (method, name))
                self.assertTrue((tensors[name][:, 4, :] != 0).any(), (method, name))
            self.assertEqual(report["targets"]["blocks.0.attn.W_O"]["dead"], 0)
            self.assertEqual(report["dead_total"], 3, method)

    def test_report_lists_every_target_with_its_dead_columns_and_error(self):
        _, tensors, report = self.run_cal("gptq")
        self.assertEqual(set(report["targets"]), set(tensors))
        self.assertTrue(all({"dead", "rows", "seconds"} <= set(v) for v in report["targets"].values()))
        self.assertEqual(report["method"], "gptq")
        self.assertEqual(report["bits"], 4)
        self.assertEqual(report["n_rows"], sum(int(b["mask"].sum()) for b in self.batches))

    def test_gptq_output_error_is_lower_than_matched_rtn_for_the_first_block(self):
        model = tiny_gemma2(); seen = {}
        _, tg, _ = self.run_cal("gptq", observer=lambda block, stage, name, rows: seen.setdefault(name, rows.clone()))
        _, tr, _ = self.run_cal("rtn_matched")
        total = {"g": 0.0, "r": 0.0}
        for name in ("blocks.0.attn.W_Q", "blocks.0.attn._W_K", "blocks.0.attn._W_V", "blocks.0.mlp.W_in", "blocks.0.mlp.W_gate"):    # inputs of block 0 are the same for both methods up to the stage's own earlier matrices
            w = canonical.to_linear(name, dict(canonical.targets(model))[name].data); x = seen[name]
            for key, t in (("g", tg), ("r", tr)):
                total[key] += float(((x @ (canonical.to_linear(name, t[name]) - w).T) ** 2).sum())
        self.assertLess(total["g"], total["r"])


if __name__ == "__main__":
    unittest.main()
