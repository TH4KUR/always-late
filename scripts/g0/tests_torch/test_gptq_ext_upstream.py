"""Task 1 of the GPTQ extension plan (docs/superpowers/plans/2026-10-27-gptq-matched-comparison.md): the authors' GPTQ core (IST-DASLab/gptq, commit 2d65066e) is vendored byte for byte, hash-checked at load time, and
reproduces same-grid nearest rounding when the Hessian is the identity (the audit's CPU check, reports/quantization_gptq_audit_2026-10-27.md section 10)."""
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
G0 = HERE.parent
ROOT = G0.parents[1]
sys.path.insert(0, str(G0))
from gptq_ext import upstream as up  # noqa: E402

PINNED = {"gptq.py": "bb47468cc2c788577562751de414d76eeb9968ff1c6098e21153557ef3d154c7", "quant.py": "211528568b962b868d20fc58b55200b615f41ebd9f36b4d2d7e4a196a095ea5a"}
COMMIT = "2d65066eeb06a5c9ff5184d8cebdf33662c67faf"


class Pinned(unittest.TestCase):
    def test_upstream_files_match_pinned_hashes(self):
        for name, h in PINNED.items():
            self.assertEqual(hashlib.sha256((up.UPSTREAM_DIR / name).read_bytes()).hexdigest(), h, name)
        self.assertEqual(up.PINNED_SHA256, PINNED)

    def test_the_pinned_hashes_are_the_ones_of_the_audit(self):
        ev = json.loads((ROOT / "reports/quantization_gptq_audit_2026-10-27/evidence.json").read_text())
        self.assertEqual(ev["upstream_sha256"], PINNED)
        self.assertEqual(ev["upstream_commit"], COMMIT)

    def test_provenance_records_commit_licence_and_hashes(self):
        prov = json.loads((up.UPSTREAM_DIR / "PROVENANCE.json").read_text())
        self.assertEqual(prov["commit"], COMMIT)
        self.assertEqual(prov["license"], "Apache-2.0")
        self.assertEqual({k: v["sha256"] for k, v in prov["files"].items() if k in PINNED}, PINNED)
        self.assertTrue((up.UPSTREAM_DIR / "LICENSE").is_file())

    def test_load_refuses_a_file_with_one_changed_byte(self):
        with tempfile.TemporaryDirectory() as d:
            for name in PINNED:
                shutil.copy(up.UPSTREAM_DIR / name, d)
            p = Path(d) / "gptq.py"
            body = bytearray(p.read_bytes()); body[100] ^= 1; p.write_bytes(bytes(body))
            with self.assertRaises(up.HashMismatch):
                up.load(d)

    def test_load_refuses_a_missing_file(self):
        with tempfile.TemporaryDirectory() as d:
            shutil.copy(up.UPSTREAM_DIR / "quant.py", d)
            with self.assertRaises(up.HashMismatch):
                up.load(d)


class Substitutions(unittest.TestCase):
    def test_load_turns_tf32_off_as_upstream_does_at_import(self):
        old = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
            up.load()
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertFalse(torch.backends.cudnn.allow_tf32)
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old

    @unittest.skipIf(torch.cuda.is_available(), "the no-op synchronise only exists without CUDA")
    def test_synchronize_is_a_no_op_without_cuda_and_the_class_runs(self):
        core = up.load()
        layer = torch.nn.Linear(8, 4, bias=False)
        gp = core.GPTQ(layer); gp.quantizer = core.Quantizer(); gp.quantizer.configure(4, perchannel=True, sym=True)
        gp.add_batch(torch.randn(2, 5, 8), None)
        gp.fasterquant()


class IdentityHessian(unittest.TestCase):
    def run_identity(self, bits, dtype, rows=8, cols=24, seed=0):
        core = up.load()
        g = torch.Generator().manual_seed(seed)
        layer = torch.nn.Linear(cols, rows, bias=False)
        layer.weight.data = torch.randn(rows, cols, generator=g).to(dtype)
        w0 = layer.weight.data.clone()
        gp = core.GPTQ(layer)
        gp.quantizer = core.Quantizer(); gp.quantizer.configure(bits, perchannel=True, sym=True, mse=False)
        gp.H = torch.eye(cols)
        gp.fasterquant(blocksize=128, percdamp=0.01, groupsize=-1, actorder=False)
        ref = core.Quantizer(); ref.configure(bits, perchannel=True, sym=True, mse=False); ref.find_params(w0.float(), weight=True)
        expected = core.quantize(w0.float(), ref.scale, ref.zero, ref.maxq).to(dtype)
        return layer.weight.data, expected, w0

    def test_identity_hessian_gptq_equals_same_grid_nearest_rounding(self):
        for bits in (4, 6):
            for dtype in (torch.float32, torch.bfloat16):
                got, expected, w0 = self.run_identity(bits, dtype)
                self.assertTrue(torch.equal(got, expected), (bits, dtype))
                self.assertEqual(got.dtype, dtype)
                self.assertFalse(torch.equal(got, w0), "the weights must actually be quantized")

    def test_four_bit_values_use_at_most_sixteen_levels_per_row(self):
        got, _, _ = self.run_identity(4, torch.float32)
        self.assertTrue(all(len(torch.unique(r)) <= 16 for r in got))

    def test_upstream_grid_scale_is_two_fifteenths_of_the_row_maximum_at_four_bits(self):
        core = up.load()
        q = core.Quantizer(); q.configure(4, perchannel=True, sym=True, mse=False)
        q.find_params(torch.tensor([[-1.0, -0.5, 0.0, 0.5, 1.0]]), weight=True)
        self.assertAlmostEqual(float(q.scale), 2 / 15, places=6)
        self.assertEqual(float(q.zero), 8.0)


if __name__ == "__main__":
    unittest.main()
