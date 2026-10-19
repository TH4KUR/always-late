"""N11 step 1 tests (decision D-025; pre-specified in handoff 007 section 4): extract the JumpReLU thresholds theta_f of the SAE (Gemma Scope, Lieberum et al. 2024, arXiv:2408.05147: a feature
is active when its pre-activation exceeds a learned threshold) with THEIR loader, into a small CSV with provenance. A fake SAE object stands in for sae_lens on CPU; the real load happens once on Ada."""
import hashlib, shutil, sys, tempfile, types, unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import extract_sae_thresholds as ex   # noqa: E402


def fake_sae(vals=(0.5, 2.0, 0.01, 7.25)):
    return types.SimpleNamespace(threshold=torch.nn.Parameter(torch.tensor(vals, dtype=torch.float32)), cfg=types.SimpleNamespace(d_sae=len(vals), architecture="jumprelu"))


class FromSae(unittest.TestCase):
    def test_the_threshold_vector_is_returned_in_feature_order_as_float32_values(self):
        t = ex.thresholds_from_sae(fake_sae()); self.assertEqual(t.dtype, np.float32); self.assertEqual(t.tolist(), torch.tensor([0.5, 2.0, 0.01, 7.25], dtype=torch.float32).tolist())

    def test_a_missing_threshold_attribute_is_an_error_naming_it(self):
        with self.assertRaisesRegex(ValueError, "no 'threshold'"): ex.thresholds_from_sae(types.SimpleNamespace(cfg=types.SimpleNamespace(d_sae=4)))

    def test_wrong_length_non_positive_and_non_finite_values_are_refused(self):
        with self.assertRaisesRegex(ValueError, "4 thresholds for 5 features"): ex.thresholds_from_sae(fake_sae(), n_feat=5)
        for bad in ((0.5, 0.0, 1.0), (0.5, -1.0, 1.0), (0.5, float("nan"), 1.0), (0.5, float("inf"), 1.0)):
            with self.assertRaisesRegex(ValueError, "positive and finite"): ex.thresholds_from_sae(fake_sae(bad))

    def test_a_matrix_shaped_threshold_is_refused_rather_than_flattened(self):
        s = fake_sae(); s.threshold = torch.nn.Parameter(torch.ones(2, 2))
        with self.assertRaisesRegex(ValueError, "one-dimensional"): ex.thresholds_from_sae(s)


class Csv(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, self.d, ignore_errors=True); self.p = self.d / "t.csv"

    def test_write_and_read_round_trip_the_float32_values_exactly(self):
        t = np.array([0.5, 2.0, 0.01, 7.25, 1e-7, 123456.0], dtype=np.float32); ex.write_thresholds_csv(self.p, t, {"release": "r", "sae_id": "s"})
        back = ex.read_thresholds(self.p); self.assertEqual(back.dtype, np.float32); self.assertTrue(np.array_equal(back, t)); self.assertEqual(len(back), 6)

    def test_the_header_records_provenance_and_the_vector_hash(self):
        t = np.array([0.5, 2.0], dtype=np.float32); ex.write_thresholds_csv(self.p, t, {"release": "gemma-scope-2b-pt-res-canonical", "sae_id": "layer_12/width_16k/canonical"})
        text = self.p.read_text().splitlines(); self.assertEqual(text[-3], "feature,threshold"); self.assertEqual(text[-2:], ["0,0.5", "1,2.0"])
        head = "\n".join(l for l in text if l.startswith("#")); self.assertIn("release: gemma-scope-2b-pt-res-canonical", head); self.assertIn("sae_id: layer_12/width_16k/canonical", head)
        self.assertIn(f"threshold_sha256: {hashlib.sha256(t.tobytes()).hexdigest()}", head); self.assertIn("d_sae: 2", head)

    def test_read_refuses_a_wrong_count_a_wrong_order_a_duplicate_and_a_missing_header(self):
        ex.write_thresholds_csv(self.p, np.array([1.0, 2.0, 3.0], dtype=np.float32), {})
        with self.assertRaisesRegex(ValueError, "expected 4 features"): ex.read_thresholds(self.p, n_feat=4)
        lines = self.p.read_text().splitlines(); head = [l for l in lines if l.startswith("#")]
        for body, msg in ((["feature,threshold", "0,1.0", "2,2.0", "1,3.0"], "feature order"), (["feature,threshold", "0,1.0", "0,2.0"], "feature order"), (["0,1.0", "1,2.0"], "header")):
            self.p.write_text("\n".join(head + body) + "\n")
            with self.assertRaisesRegex(ValueError, msg): ex.read_thresholds(self.p)

    def test_read_refuses_non_positive_values(self):
        self.p.write_text("feature,threshold\n0,1.0\n1,-2.0\n")
        with self.assertRaisesRegex(ValueError, "positive and finite"): ex.read_thresholds(self.p)


class Extract(unittest.TestCase):
    def test_the_loader_is_called_with_their_signature_on_cpu_and_the_file_carries_release_id_and_size(self):
        d = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, d, ignore_errors=True); calls = []
        def load(release, sae_id, device):
            calls.append((release, sae_id, device)); return fake_sae()
        out = ex.extract(load, "rel", "sid", d / "t.csv"); self.assertEqual(calls, [("rel", "sid", "cpu")])
        head = "\n".join(l for l in (d / "t.csv").read_text().splitlines() if l.startswith("#")); self.assertIn("release: rel", head); self.assertIn("sae_id: sid", head); self.assertIn("d_sae: 4", head)
        self.assertEqual(out.tolist(), ex.read_thresholds(d / "t.csv").tolist())

    def test_a_threshold_vector_that_disagrees_with_the_declared_sae_width_is_refused_before_anything_is_written(self):
        d = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, d, ignore_errors=True); s = fake_sae(); s.cfg.d_sae = 5
        with self.assertRaisesRegex(ValueError, "4 thresholds for 5 features"): ex.extract(lambda r, sid, dev: s, "rel", "sid", d / "t.csv")
        self.assertFalse((d / "t.csv").exists())

    def test_a_non_jumprelu_looking_sae_is_refused_before_anything_is_written(self):
        d = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        with self.assertRaises(ValueError): ex.extract(lambda r, s, dev: types.SimpleNamespace(cfg=types.SimpleNamespace(d_sae=4)), "rel", "sid", d / "t.csv")
        self.assertFalse((d / "t.csv").exists())


if __name__ == "__main__":
    unittest.main()
