import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "jarvis_run_all.sh"
BASH = shutil.which("bash")


def run(env_extra, base):
    env = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "FAKE_GPUS", "FAKE_RAM_GB", "STATES", "SKIP_SMOKE", "MAX_PARALLEL", "FOREGROUND", "FAKE_CC", "FAKE_DRIVER", "ARMS", "STAGES", "STATES_NEW",
                                                              "PROCS_PER_GPU", "TORCH_INDEX", "P1_BITS_LEGACY", "P1_BITS_NEW")}
    env.update({"BASE": str(base), "REPO": str(SCRIPT.parents[2]), "DRY_RUN": "1"}); env.update(env_extra)
    return subprocess.run([BASH, str(SCRIPT)], env=env, capture_output=True, text=True, timeout=120)


@unittest.skipUnless(BASH, "bash is not available")
class DryRun(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.base = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_plan_lists_every_step_in_order_and_runs_nothing(self):
        r = run({"FAKE_GPUS": "81920", "FAKE_RAM_GB": "200"}, self.base)
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        out = r.stdout
        order = [out.index(s) for s in ("== env", "== flores", "== access_and_download", "== checks", "== smoke", "== full")]
        self.assertEqual(order, sorted(order))
        self.assertIn("torch==2.14.1", out); self.assertIn("transformer-lens==3.9.0", out); self.assertIn("sae-lens==6.54.5", out)
        self.assertFalse((self.base / "state").exists())                        # nothing was created

    def test_one_process_per_gpu_when_the_ram_allows_it(self):
        r = run({"FAKE_GPUS": "81920 81920 81920", "FAKE_RAM_GB": "256"}, self.base)
        self.assertIn("usable GPUs: 0 1 2", r.stdout); self.assertIn("concurrent state processes: 3", r.stdout)
        r = run({"FAKE_GPUS": "81920 81920 81920", "FAKE_RAM_GB": "100"}, self.base)
        self.assertIn("concurrent state processes: 1", r.stdout)                 # RAM limits it
        r = run({"FAKE_GPUS": "81920 81920 81920", "FAKE_RAM_GB": "256", "MAX_PARALLEL": "2"}, self.base)
        self.assertIn("concurrent state processes: 2", r.stdout)

    def test_small_gpus_are_refused(self):
        r = run({"FAKE_GPUS": "24576 11264", "FAKE_RAM_GB": "256"}, self.base)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("no GPU with >=", r.stderr)

    def test_a_small_gpu_next_to_a_big_one_is_skipped(self):
        r = run({"FAKE_GPUS": "11264 81920", "FAKE_RAM_GB": "256"}, self.base)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("usable GPUs: 1", r.stdout)

    def test_the_token_is_required_for_a_real_run(self):
        env = {k: v for k, v in os.environ.items() if k != "HF_TOKEN"}
        env.update({"BASE": str(self.base), "REPO": str(SCRIPT.parents[2]), "FAKE_GPUS": "81920", "FAKE_RAM_GB": "200"})
        r = subprocess.run([BASH, str(SCRIPT)], env=env, capture_output=True, text=True, timeout=60)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("HF_TOKEN", r.stderr)


BLACKWELL = {"FAKE_GPUS": "98304", "FAKE_RAM_GB": "160", "FAKE_CC": "12.0", "FAKE_DRIVER": "580.65"}
SPEC_ORDER = [("g", "legacy", "FP"), ("g", "gptq", "4"), ("g", "gptq", "6"), ("g", "rtn-matched", "4"), ("g", "rtn-matched", "6"), ("g", "legacy", "4"), ("g", "legacy", "6"), ("g", "legacy", "8"), ("g", "legacy", "7"),
              ("g", "legacy", "5"), ("p1", "gptq", "all"), ("p1", "legacy", "all"), ("p1", "rtn-matched", "all")]


def units(stdout):
    """[(stage, arm, token, command line)] of the `+ UNIT` lines of a dry run, in order."""
    out = []
    for line in stdout.splitlines():
        if line.startswith("+ UNIT "):
            head, _, cmd = line[len("+ UNIT "):].partition(" :: ")
            stage, arm, token = head.split()
            out.append((stage, arm, token, cmd))
    return out


@unittest.skipUnless(BASH, "bash is not available")
class ThreeArms(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.base = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def plan(self, **extra):
        r = run({**BLACKWELL, **extra}, self.base)
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        return r.stdout

    def test_dry_run_lists_the_new_steps_in_order(self):
        out = self.plan()
        names = ["env", "flores", "access_and_download", "checks", "tokenizer", "pool", "smoke", "smoke_p1", "gate9b", "smoke_gptq", "full"]
        idx = []
        for n in names:
            m = re.search(rf"== {n}$", out, re.M)
            self.assertIsNotNone(m, n); idx.append(m.start())
        self.assertEqual(idx, sorted(idx), names)

    def test_blackwell_with_a_current_driver_chooses_cu130(self):
        out = self.plan()
        self.assertIn("whl/cu130", out); self.assertNotIn("whl/cu126", out)

    def test_blackwell_with_an_old_driver_is_refused_with_a_message(self):
        r = run({**BLACKWELL, "FAKE_DRIVER": "550.54"}, self.base)
        self.assertNotEqual(r.returncode, 0); self.assertIn("580", r.stderr)

    def test_an_older_gpu_chooses_cu126_and_an_explicit_index_wins(self):
        self.assertIn("whl/cu126", self.plan(FAKE_CC="8.0", FAKE_DRIVER="550.54"))
        out = self.plan(FAKE_DRIVER="470.1", TORCH_INDEX="https://example.org/whl/cu128")
        self.assertIn("https://example.org/whl/cu128", out); self.assertNotIn("whl/cu130", out)

    def test_one_gpu_and_160_gb_give_one_slot_and_two_only_with_procs_per_gpu_when_two_units_fit(self):
        self.assertIn("concurrent state processes: 1", self.plan())
        self.assertIn("concurrent state processes: 2", self.plan(PROCS_PER_GPU="2"))
        self.assertIn("concurrent state processes: 1", self.plan(PROCS_PER_GPU="2", FAKE_RAM_GB="100"))

    def test_every_arm_and_stage_unit_appears_once_with_the_right_flags(self):
        us = units(self.plan())
        self.assertEqual([(s, a, k) for s, a, k, _ in us], SPEC_ORDER)
        for stage, arm, token, cmd in us:
            if arm == "legacy":
                self.assertNotIn("--weights", cmd, (stage, arm, token))
            else:
                self.assertIn(f"--weights {arm}", cmd, (stage, arm, token))
            if stage == "g":
                self.assertIn("lm", cmd); self.assertIn(f"--precisions {token}", cmd)
            else:
                self.assertIn("p1", cmd); self.assertIn("--bits " + ("8 7 6 5 4 16 12 0" if arm == "legacy" else "7 6 4 16 12 0"), cmd)

    def test_gptq_and_matched_units_come_after_the_gate_and_carry_the_frozen_hash_file(self):
        out = self.plan(); gate = out.index("== gate9b")
        for stage, arm, token, cmd in units(out):
            if arm != "legacy":
                self.assertGreater(out.index(f"+ UNIT {stage} {arm} {token} "), gate)
                self.assertIn("--gptq-freeze", cmd); self.assertIn("artifact_hashes.json", cmd); self.assertIn("--calibration-text", cmd)

    def test_the_gate_unit_records_hashes_and_uses_the_conservative_ram_estimate(self):
        out = self.plan()
        self.assertIn("phase1_gemma9b_ext.py gate", out); self.assertIn("--record-weights-hashes", out); self.assertIn("gptq_ext.record_hashes", out)
        self.assertIn("--n-artifacts 4", out)

    def test_ram_estimates_are_printed_per_unit_kind(self):
        out = self.plan()
        self.assertIn("59.1", out); self.assertIn("97.8", out); self.assertIn("20.4", out)

    def test_resume_skips_finished_units(self):
        st = self.base / "state"; st.mkdir()
        for name in ("state_FP.done", "unit_g_gptq_4.done", "unit_p1_legacy.done"):
            (st / name).write_text("")
        out = self.plan(); us = [(s, a, k) for s, a, k, _ in units(out)]
        for u in (("g", "legacy", "FP"), ("g", "gptq", "4"), ("p1", "legacy", "all")):
            self.assertNotIn(u, us); self.assertIn("[skip] unit %s %s %s" % u, out)
        self.assertEqual(len(us), len(SPEC_ORDER) - 3)

    def test_the_old_plan_is_what_arms_legacy_stages_g_gives(self):
        out = self.plan(ARMS="legacy", STAGES="g")
        for m in ("== tokenizer", "== pool", "== gate9b", "== smoke_p1", "== smoke_gptq"):
            self.assertNotIn(m, out)
        self.assertEqual([(s, a, k) for s, a, k, _ in units(out)], [("g", "legacy", s) for s in ("FP", "4", "6", "8", "7", "5")])

    def test_states_and_states_new_filter_the_units(self):
        us = [(s, a, k) for s, a, k, _ in units(self.plan(STATES="6 4", STATES_NEW="4", STAGES="g"))]
        self.assertEqual(us, [("g", "gptq", "4"), ("g", "rtn-matched", "4"), ("g", "legacy", "4"), ("g", "legacy", "6")])

    def test_stage_p1_only_has_no_study_g_units(self):
        us = [(s, a, k) for s, a, k, _ in units(self.plan(STAGES="p1"))]
        self.assertEqual(us, [("p1", "gptq", "all"), ("p1", "legacy", "all"), ("p1", "rtn-matched", "all")])


def gate_bit_sets(stdout):
    """[[bits of one gate command]] from the dry-run lines `+ gate9b: phase1_gemma9b_ext.py gate ... --bits <b> ... --record-weights-hashes`."""
    return [[int(x) for x in m.split()] for m in re.findall(r"\+ gate9b: phase1_gemma9b_ext\.py gate .*? --bits ([0-9 ]+?) --record-weights-hashes", stdout)]


def artifact_bits(stage, cmd):
    """The artifact bit widths (4 to 8) a weight-set unit builds, from its command line."""
    if stage == "g":
        return {int(re.search(r"--precisions (\d+)", cmd).group(1))}
    return {int(x) for x in re.search(r"--bits ([0-9 ]+?) --out", cmd).group(1).split()} & {4, 5, 6, 7, 8}


@unittest.skipUnless(BASH, "bash is not available")
class ReviewFixes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.base = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def plan(self, **extra):
        r = run({**BLACKWELL, **extra}, self.base)
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        return r.stdout

    def test_the_gate_records_every_artifact_bit_width_the_units_need(self):
        for extra, expect in (({}, {4, 6, 7}), ({"STAGES": "g"}, {4, 6}), ({"P1_BITS_NEW": "8 7 6 5 4 16 12 0"}, {4, 5, 6, 7, 8})):
            out = self.plan(**extra); sets = gate_bit_sets(out)
            self.assertEqual({b for s in sets for b in s}, expect, extra)
            self.assertTrue(all(1 <= len(s) <= 2 for s in sets), sets)                         # at most two bit widths per gate job (host RAM)
            for stage, arm, token, cmd in units(out):
                if arm != "legacy":
                    self.assertLessEqual(artifact_bits(stage, cmd), expect, (extra, stage, arm, token))

    def test_every_gate_job_has_its_own_ram_guard_and_the_hashes_come_from_all_gate_directories(self):
        out = self.plan(); sets = gate_bit_sets(out)
        self.assertEqual(len(re.findall(r"\+ gate9b: python -m gptq_ext\.ram_guard", out)), len(sets))
        record = re.search(r"\+ gate9b: python -m gptq_ext\.record_hashes (.*)", out).group(1)
        self.assertEqual(record.count("--gate-dir"), len(sets))

    def test_p1_extra_goes_to_tier_two_units_and_the_tier_two_smoke_only(self):
        out = self.plan(P1_EXTRA="--batch-size 16")
        for stage, arm, token, cmd in units(out):
            self.assertEqual("--batch-size 16" in cmd, stage == "p1", (stage, arm, token))
        self.assertIn("--bits 4 --batch-size 16", out)

    def test_the_frozen_files_are_verified_first_on_every_invocation(self):
        out = self.plan()
        first = re.search(r"== verify_frozen$", out, re.M); env = re.search(r"== env$", out, re.M)
        self.assertIsNotNone(first); self.assertLess(first.start(), env.start())
        self.assertIn("make_gemma9b_freeze.py --verify", out); self.assertIn("make_gemma9b_amendment.py --verify", out)
        legacy = self.plan(ARMS="legacy", STAGES="g")
        self.assertIn("make_gemma9b_freeze.py --verify", legacy); self.assertNotIn("make_gemma9b_amendment.py --verify", legacy)

    def test_a_failing_nvidia_smi_does_not_abort_silently_and_a_missing_gpu_names_the_driver(self):
        bindir = self.base / "bin"; bindir.mkdir(); (bindir / "nvidia-smi").write_text("#!/bin/sh\nexit 2\n"); (bindir / "nvidia-smi").chmod(0o755)
        env = {"PATH": f"{bindir}:{os.environ['PATH']}"}
        r = run({"FAKE_GPUS": "98304", "FAKE_RAM_GB": "160", **env}, self.base)
        self.assertEqual(r.returncode, 0, r.stderr); self.assertIn("whl/cu126", r.stdout)                  # compute capability unknown: the default index, not a silent exit
        r = run(env, self.base)
        self.assertNotEqual(r.returncode, 0); self.assertIn("nvidia-smi", r.stderr); self.assertIn("580", r.stderr)


if __name__ == "__main__":
    unittest.main()
