"""Tests for scripts/g0/survival_validate.py (validation of a finished sidecar job): the project's remote checks (sacct, log head, checksums) with injected remote commands and file copy, the meta,
the saved sums, the weights manifest and the committed comparison, on a toy job written by survival_sidecar.run_job."""
import json, shutil, subprocess, sys, tempfile, time, unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import survival_sidecar as sc   # noqa: E402
import survival_validate as sv   # noqa: E402
from test_phase1_run import Fake, F, BOS, PAD, corpus, call   # noqa: E402
from test_survival_sidecar import FakePre, StubCap, toy_keep   # noqa: E402

BITS = (6, 4, 0)
RECORDED = {"gptq": {"4": "a" * 64, "6": "b" * 64}, "rtn_matched": {"4": "c" * 64, "6": "d" * 64}}


class RealNamedCap(StubCap):
    source = "sae.hook_sae_acts_pre"                                                              # the toy encoder, reporting the name of the real source so that the toy job is a valid job


class Validate(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()); self.remote = self.tmp / "remote"; self.local_root = self.tmp / "local"; self.data = corpus()
        fk = FakePre(); csv = self.tmp / "t.csv"
        csv.write_text("# toy\nfeature,threshold\n" + "".join(f"{i},{float(v):.9g}\n" for i, v in enumerate(fk.thr)))
        a = sc.parse_args(["--out", str(self.remote), "--bits", *map(str, BITS), "--flores-root", "x", "--batch-size", "4", "--thresholds-csv", str(csv)])
        sc.run_job(a, [f"l{i}" for i in self.data], self.data, BOS, PAD, F, fk.forward, fk.encode, fk.quantize, fk.restore, fk.thr, RealNamedCap(fk), {"device": "cpu", "model": "toy", "weights": "rtn"}, lambda *m: None, time.time())
        self.sacct = "43001|sc-test|COMPLETED|00:10:00|gnode001|0:0"; self.head = "GPU 0: NVIDIA GeForce RTX 2080 Ti (UUID: x)\nThu Oct 29"; self.cuda = "0"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_remote(self, cmd):
        if cmd.startswith("sacct"):
            return self.sacct
        if cmd.startswith("head"):
            return self.head
        if cmd.startswith("grep"):
            return self.cuda
        return subprocess.run("sha256sum *", shell=True, cwd=self.remote, capture_output=True, text=True).stdout

    def fetch(self, remote, local):
        for f in self.remote.iterdir():
            shutil.copy(f, Path(local) / f.name)

    def validate(self, arm="legacy", **kw):
        kw = {"n_lang": 3, "n_feat": F, **kw}
        return sv.validate(43001, "sc-test", "sc-test", arm, list(BITS), self.run_remote, self.fetch, self.local_root, recorded=RECORDED, **kw)

    def committed(self, shift=0.0):
        res = call(Fake(), self.data, "mask", bits=BITS, keep=toy_keep(self.data), batch_size=4); eq = {k: v["eq"] for k, v in res.items()}
        d = self.tmp / "committed"; d.mkdir(exist_ok=True)
        torch.save({"languages": [f"l{i}" for i in self.data], "bits": list(BITS), "n": eq["FP"]["n"], "corr": {b: eq[b]["corr"].float() + (shift if b == 4 else 0.0) for b in BITS},
                    "firing_rate": eq["FP"]["firing_rate"].float()}, d / "survival.pt")
        return d

    def test_a_good_legacy_job_passes_every_check(self):
        rep = self.validate(); self.assertTrue(rep["ok"], rep); self.assertEqual(rep["failed"], []); self.assertNotIn("weights", rep["checks"]); self.assertNotIn("committed", rep)
        self.assertEqual(rep["checks"]["sums"]["conditions"], ["INT6", "INT4"]); self.assertEqual(rep["checks"]["meta"]["pre_activation_source"], "sae.hook_sae_acts_pre")

    def test_the_recorded_weights_must_be_those_of_the_arm(self):
        meta = json.loads((self.remote / "meta.json").read_text())
        self.assertTrue(sv.check_meta(meta, list(BITS), full=False, arm="legacy")["ok"]); self.assertFalse(sv.check_meta(meta, list(BITS), full=False, arm="gptq")["ok"])
        self.assertTrue(sv.check_meta({**meta, "weights": "rtn-matched"}, list(BITS), full=False, arm="rtn-matched")["ok"]); self.assertFalse(sv.check_meta({**meta, "weights": None}, list(BITS), full=False, arm="legacy")["ok"])

    def test_a_failed_gate_a_missing_theta_comparison_and_another_regime_are_each_refused_alone(self):
        meta = json.loads((self.remote / "meta.json").read_text()); self.assertTrue(sv.check_meta(meta, list(BITS), full=False)["ok"])
        failed = json.loads(json.dumps(meta)); failed["gates"]["G4"]["passed"] = False; self.assertFalse(sv.check_meta(failed, list(BITS), full=False)["ok"])           # gate_failures stays empty
        no_theta = json.loads(json.dumps(meta)); del no_theta["gates"]["G3"]["max_abs_diff_theta_vs_csv"]; self.assertFalse(sv.check_meta(no_theta, list(BITS), full=False)["ok"])
        self.assertFalse(sv.check_meta({**meta, "regime": "none"}, list(BITS), full=False)["ok"]); self.assertFalse(sv.check_meta({**meta, "stream": "all"}, list(BITS), full=False)["ok"])

    def test_an_unknown_pre_activation_source_or_dtype_is_refused(self):
        meta = json.loads((self.remote / "meta.json").read_text())
        self.assertTrue(sv.check_meta(meta, list(BITS), full=False)["ok"])
        self.assertFalse(sv.check_meta({**meta, "pre_activation_source": "toy encoder"}, list(BITS), full=False)["ok"])
        self.assertFalse(sv.check_meta({**meta, "pre_activation_dtype": "torch.float16"}, list(BITS), full=False)["ok"])

    def test_remote_failures_fail_their_checks(self):
        self.sacct = "43001|sc-test|FAILED|00:10:00|gnode001|1:0"; self.assertIn("sacct", self.validate()["failed"]); self.sacct = "43001|sc-test|COMPLETED|00:10:00|gnode001|0:0"
        self.head = "no gpu here"; self.assertIn("log", self.validate()["failed"]); self.head = "GPU 0: x"
        self.cuda = "2"; self.assertIn("log", self.validate()["failed"])

    def test_a_file_that_differs_from_the_remote_one_fails_the_checksums(self):
        orig = self.fetch
        def corrupting(remote, local):
            orig(remote, local); (Path(local) / "sums_INT4.pt").write_bytes(b"x")
        self.fetch = corrupting
        self.assertIn("checksums", self.validate()["failed"])

    def test_the_meta_and_the_sums_are_checked(self):
        meta = json.loads((self.remote / "meta.json").read_text()); good = dict(meta)
        meta["gate_failures"] = ["G1"]; (self.remote / "meta.json").write_text(json.dumps(meta)); self.assertIn("meta", self.validate()["failed"])
        meta = dict(good); meta["gates"] = {k: v for k, v in good["gates"].items() if k != "G4"}; (self.remote / "meta.json").write_text(json.dumps(meta)); self.assertIn("meta", self.validate()["failed"])
        meta = dict(good); meta["saved_conditions"] = ["INT6"]; (self.remote / "meta.json").write_text(json.dumps(meta)); self.assertIn("meta", self.validate()["failed"])
        (self.remote / "meta.json").write_text(json.dumps(good))
        d = sc.load_sums(self.remote / "sums_INT6.pt"); del d["stats"]["Sab"]; torch.save(d, self.remote / "sums_INT6.pt"); self.assertIn("sums", self.validate()["failed"])

    def test_a_full_run_must_cover_all_languages_a_smoke_run_need_not(self):
        meta = json.loads((self.remote / "meta.json").read_text())
        self.assertFalse(sv.check_meta({**meta, "subset": True}, list(BITS), full=True, n_lang=3, n_feat=F)["ok"])
        self.assertFalse(sv.check_meta(meta, list(BITS), full=True, n_lang=27)["ok"])                                                                    # 3 languages, 24 features is no full run
        self.assertTrue(sv.check_meta(meta, list(BITS), full=False)["ok"]); self.assertTrue(sv.check_meta(meta, list(BITS), full=True, n_lang=3, n_feat=F)["ok"])

    def test_the_weights_manifest_is_required_for_the_matched_and_gptq_arms_and_checked(self):
        self.assertIn("weights", self.validate("gptq")["failed"])
        man = {"method": "gptq", "bits": [4, 6], "artifact_sha256": {"4": "a" * 64, "6": "b" * 64}, "pool_sha256": "p" * 64}
        (self.remote / "weights_manifest.json").write_text(json.dumps(man)); self.assertNotIn("weights", self.validate("gptq")["failed"])
        self.assertIn("weights", self.validate("rtn-matched")["failed"])                                                                                 # method differs
        man["artifact_sha256"]["4"] = "e" * 64; (self.remote / "weights_manifest.json").write_text(json.dumps(man)); self.assertIn("weights", self.validate("gptq")["failed"])

    def test_the_stage_1_manifest_supplies_the_hashes_of_the_widths_the_freeze_file_does_not_record(self):
        rec = {"gptq": {"4": "a" * 64}, "rtn_matched": {"4": "c" * 64}}
        stage1 = {"method": "gptq", "artifact_sha256": {"4": "z" * 64, "6": "b" * 64, "8": "d" * 64}}
        merged, conflicts = sv.merge_recorded(rec, stage1)
        self.assertEqual(merged["gptq"], {"4": "a" * 64, "6": "b" * 64, "8": "d" * 64})                          # the freeze file wins, the stage-1 manifest fills the gaps
        self.assertEqual(len(conflicts), 1); self.assertIn("gptq 4 bits", conflicts[0])
        self.assertEqual(rec["gptq"], {"4": "a" * 64})                                                           # the input is not modified
        self.assertEqual(sv.merge_recorded(rec, {"method": "rtn_matched", "artifact_sha256": {"6": "e" * 64}})[0]["rtn_matched"], {"4": "c" * 64, "6": "e" * 64})
        man = {"method": "gptq", "bits": [4, 6], "artifact_sha256": {"4": "a" * 64, "6": "b" * 64}, "pool_sha256": "p" * 64}
        (self.remote / "weights_manifest.json").write_text(json.dumps(man))
        self.assertNotIn("weights", sv.validate(43001, "sc-test", "sc-test", "gptq", list(BITS), self.run_remote, self.fetch, self.local_root, recorded=merged, n_lang=3, n_feat=F)["failed"])
        man["artifact_sha256"]["6"] = "f" * 64; (self.remote / "weights_manifest.json").write_text(json.dumps(man))
        self.assertIn("weights", sv.validate(43001, "sc-test", "sc-test", "gptq", list(BITS), self.run_remote, self.fetch, self.local_root, recorded=merged, n_lang=3, n_feat=F)["failed"])  # INT6 differs from stage 1

    def test_the_committed_comparison_is_reported_and_never_fails_the_job(self):
        rep = self.validate(committed_run=self.committed()); self.assertTrue(rep["committed"]["G2_committed"]["passed"])
        shutil.rmtree(self.tmp / "committed"); rep = self.validate(committed_run=self.committed(shift=0.05))
        self.assertFalse(rep["committed"]["G2_committed"]["passed"]); self.assertTrue(rep["ok"])
        rep = self.validate(full=False, committed_run=self.committed()); self.assertNotIn("committed", rep)                                              # a smoke run is never compared


if __name__ == "__main__":
    unittest.main()
