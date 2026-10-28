"""Rehearsal of the REAL (not dry) path of `jarvis_run_all.sh` with a stub python in place of the venv (the dry-run tests cannot reach the scheduler, the markers, the validation of units or the resume logic). The stub
records every call in a log and creates the files the real programs would create (selectivity, manifest, `func_lm_eval_*.pt`, `survival.pt`, weights manifests, a gate report); `ram_guard` and `record_hashes` run for
real. Nothing here touches a GPU or the network; the repository is represented by a temporary directory that links to the real `scripts/`."""
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "jarvis_run_all.sh"
SCRIPTS = SCRIPT.parents[1]
BASH = shutil.which("bash")

STUB = r'''#!/bin/bash
# stub of the venv python: logs the call, creates the files the real program would create
echo "$*" >> "$STUB_LOG"
arg_after() { local k=$1; shift; while [ $# -gt 0 ]; do if [ "$1" = "$k" ]; then echo "$2"; return; fi; shift; done; }
has() { local k=$1; shift; for a in "$@"; do [ "$a" = "$k" ] && return 0; done; return 1; }
if [ "${1:-}" = "-m" ] && [ "$2" = "gptq_ext.record_hashes" ] && ! has --check "$@" && [ -n "${STUB_FAIL_RECORD:-}" ]; then echo "record_hashes failed on purpose"; exit 1; fi
if [ "${1:-}" = "-m" ] && { [ "$2" = "gptq_ext.ram_guard" ] || [ "$2" = "gptq_ext.record_hashes" ]; }; then exec python3 -B "$@"; fi
if [ "${1:-}" = "-m" ] && [ "$2" = "pip" ] && [ "${3:-}" = "install" ] && [ -n "${STUB_FAIL_PIP:-}" ]; then echo "pip failed on purpose"; exit 1; fi
if [[ "$*" == *make_gemma9b_freeze.py* ]] && [ -n "${STUB_FAIL_VERIFY:-}" ]; then echo "frozen code differs"; exit 1; fi
if [ "${1:-}" = "-B" ] && [ "${2:-}" = "-m" ] && [ "${3:-}" = "unittest" ] && [ -n "${STUB_FAIL_UNITTEST:-}" ]; then echo "tests failed on purpose"; exit 1; fi
if [ "${1:-}" = "-m" ] && [ "$2" = "gptq_ext.tokenizer_check" ]; then echo '{}' > "$(arg_after --out "$@")"; exit 0; fi
if [ "${1:-}" = "-m" ] && [ "$2" = "pip" ]; then [ "${3:-}" = "freeze" ] && echo "stub==0"; exit 0; fi
if [ "${1:-}" = "-" ]; then cat > /dev/null; exit 0; fi
case "$*" in
    *phase1_gemma9b_run.py\ tag*)
        out=$(arg_after --out "$@"); mkdir -p "$out/eq"; : > "$out/eq/selectivity.pt"; echo "tag done"; exit 0 ;;
    *phase1_gemma9b_run.py\ manifest*)
        echo '{}' > "$(arg_after --manifest "$@")"; echo "manifest done"; exit 0 ;;
    *phase1_gemma9b_ext.py\ gate*)
        [ -z "${STUB_FAIL_GATE:-}" ] || { echo "gate failed on purpose"; exit 1; }
        out=$(arg_after --out "$@"); mkdir -p "$out/gptq" "$out/rtn_matched"
        bits=$(python3 - "$@" <<'PY'
import sys
a = sys.argv[1:]; i = a.index("--bits") + 1; out = []
while i < len(a) and not a[i].startswith("--"):
    out.append(a[i]); i += 1
print(" ".join(out))
PY
)
        for m in gptq rtn_matched; do
            python3 - "$m" "$out/$m/weights_manifest.json" $bits <<'PY'
import json, sys
m, path, bits = sys.argv[1], sys.argv[2], sys.argv[3:]
json.dump({"method": m, "bits": sorted(int(b) for b in bits), "artifact_sha256": {b: f"{m}{b}" for b in bits}, "pool_sha256": "pool", "calibration_freeze_sha256": "freeze", "calibration_report": {}}, open(path, "w"))
PY
        done
        echo '{"all_gates_ok": true}' > "$out/gate_report.json"; echo "gates ok"; exit 0 ;;
    *phase1_gemma9b_ext.py\ p1*|*run_with_weights.py*p1*|*phase1_gemma9b_run.py\ lm*|*run_with_weights.py*\ lm\ *)
        out=$(arg_after --out "$@")
        if has --gptq-freeze "$@" && [ ! -f "$(arg_after --gptq-freeze "$@")" ]; then echo "frozen hash file missing"; exit 2; fi
        prec=$(arg_after --precisions "$@")
        if has --gptq-freeze "$@"; then      # the real wrapper refuses an artifact bit width without a frozen hash (states.build_artifacts)
            need=$(python3 - "$(arg_after --gptq-freeze "$@")" "$(arg_after --weights "$@")" "$prec" "$(python3 - "$@" <<'PY2'
import sys
a = sys.argv[1:]; out = []
if "--bits" in a:
    i = a.index("--bits") + 1
    while i < len(a) and not a[i].startswith("--"):
        out.append(a[i]); i += 1
print(" ".join(out))
PY2
)" <<'PY'
import json, sys
fr, weights, prec, bits = json.load(open(sys.argv[1])), sys.argv[2].replace("-", "_"), sys.argv[3], sys.argv[4]
want = {int(b) for b in (prec.split() + bits.split()) if b.isdigit() and 4 <= int(b) <= 8}
have = {int(b) for b in fr["artifact_sha256"].get(weights, {})}
missing = sorted(want - have)
if missing:
    print(f"no frozen sha256 for {weights} at {missing[0]} bits")
PY
)
            if [ -n "$need" ]; then echo "$need"; exit 2; fi
        fi
        if [ -n "${STUB_FAIL_UNIT:-}" ] && [[ "$*" == *"$STUB_FAIL_UNIT"* ]]; then echo "unit failed on purpose"; exit 1; fi
        mkdir -p "$out"
        if has p1 "$@"; then mkdir -p "$out/eq" "$out/all"; : > "$out/eq/survival.pt"; : > "$out/all/survival.pt"
        else : > "$out/func_lm_eval_${prec:-X}.pt"; echo "gates passed"; echo "peak GPU GB 12.3"; fi
        if has --weights "$@"; then echo '{}' > "$out/weights_manifest.json"; fi
        exit 0 ;;
esac
exit 0
'''


@unittest.skipUnless(BASH, "bash is not available")
class Rehearsal(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); t = Path(self.tmp.name)
        self.base, self.repo = t / "base", t / "repo"
        (self.repo / "reports/functional_validation/gemma9b").mkdir(parents=True); (self.repo / "reports/gptq_extension").mkdir(parents=True); (self.repo / "reports/g0").mkdir(parents=True)
        (self.repo / "scripts").symlink_to(SCRIPTS)
        (self.repo / "COMMIT").write_text("deadbeef\n")
        (self.repo / "reports/functional_validation/gemma9b/amendment_A1_three_arm.json").write_text("{}")
        (self.base / "venv/bin").mkdir(parents=True); (self.base / "data/flores200_dataset/dev").mkdir(parents=True); (self.base / "data/flores200_dataset/devtest").mkdir(parents=True)
        (self.base / "gptq/calibration").mkdir(parents=True)
        py = self.base / "venv/bin/python"; py.write_text(STUB); py.chmod(0o755)
        self.log = t / "stub.log"; self.log.write_text("")

    def tearDown(self):
        self.tmp.cleanup()

    def run_driver(self, **extra):
        env = {k: v for k, v in os.environ.items() if k not in ("ARMS", "STAGES", "STATES", "STATES_NEW", "PROCS_PER_GPU", "TORCH_INDEX", "SKIP_SMOKE", "MAX_PARALLEL", "LM_EXTRA")}
        env.update({"BASE": str(self.base), "REPO": str(self.repo), "HF_TOKEN": "hf_stub", "FAKE_GPUS": "98304", "FAKE_RAM_GB": "160", "FAKE_CC": "12.0", "FAKE_DRIVER": "580.65", "FOREGROUND": "1",
                    "STUB_LOG": str(self.log), "MODEL_GB": "0.1", "TARGET_GB": "0.1", "POLL_SECONDS": "0.2"})
        env.update(extra)
        return subprocess.run([BASH, str(SCRIPT)], env=env, capture_output=True, text=True, timeout=300)

    def calls(self):
        return [l for l in self.log.read_text().splitlines() if l]

    def state(self):
        return {p.name for p in (self.base / "state").glob("*")}

    def test_a_full_run_finishes_every_unit_in_the_order_of_the_spec_and_packages_the_results(self):
        r = self.run_driver()
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-2000:])
        self.assertIn("ALL DONE", r.stdout)
        done = {n[:-5] for n in self.state() if n.endswith(".done")}
        for name in ("env", "flores", "access_and_download", "checks", "tokenizer", "pool", "smoke", "smoke_p1", "gate9b", "smoke_gptq", "full",
                     "state_FP", "state_4", "state_6", "state_8", "state_7", "state_5", "unit_g_gptq_4", "unit_g_gptq_6", "unit_g_rtn-matched_4", "unit_g_rtn-matched_6", "unit_p1_gptq", "unit_p1_legacy", "unit_p1_rtn-matched"):
            self.assertIn(name, done, name)
        self.assertFalse([n for n in self.state() if n.endswith(".failed")])
        h = json.loads((self.repo / "reports/functional_validation/gemma9b/artifact_hashes.json").read_text())
        self.assertEqual(h["artifact_sha256"], {"gptq": {"4": "gptq4", "6": "gptq6", "7": "gptq7"}, "rtn_matched": {"4": "rtn_matched4", "6": "rtn_matched6", "7": "rtn_matched7"}})      # bit 7: tier 2 needs it
        self.assertTrue((self.base / "gemma9b_results.tar.gz").exists()); self.assertTrue((self.base / "out/frozen/artifact_hashes.json").exists())
        calls = self.calls()
        idx = lambda needle: next(i for i, c in enumerate(calls) if needle in c)
        self.assertLess(idx("phase1_gemma9b_ext.py gate"), idx("--weights gptq"))                       # the gate comes before every weight-set unit
        self.assertLess(idx("--weights gptq --gptq-freeze"), idx("phase1_gemma9b_run.py lm --out " + str(self.base / "out/phase1/g9-lm-4")))   # tier 1: gptq units before the legacy INT4 unit
        self.assertLess(idx("phase1_gemma9b_run.py lm --out " + str(self.base / "out/phase1/g9-lm-5")), idx("p1 --corpus flores --regime mask --bits 7 6 4 16 12 0"))   # tier 2 after tier 1

    def unit_calls(self, since=0):
        return [c for c in self.calls()[since:] if "--out" in c and ("lm" in c.split() or "p1" in c.split())]

    def test_a_second_run_skips_everything_and_a_removed_unit_marker_runs_only_that_unit(self):
        self.assertEqual(self.run_driver().returncode, 0)
        n = len(self.calls())
        r = self.run_driver()
        self.assertEqual(r.returncode, 0, r.stdout[-2000:]); self.assertIn("[skip] full (already done)", r.stdout); self.assertIn("[skip] gate9b", r.stdout)
        self.assertEqual(self.unit_calls(n), [])
        (self.base / "state/full.done").unlink(); (self.base / "state/unit_g_gptq_6.done").unlink()
        n = len(self.calls()); r = self.run_driver()
        self.assertEqual(r.returncode, 0, r.stdout[-2000:]); self.assertIn("[skip] unit g gptq 4", r.stdout)
        again = self.unit_calls(n)
        self.assertEqual(len(again), 1, again); self.assertIn("--weights gptq", again[0]); self.assertIn("--precisions 6", again[0])

    def test_a_failed_unit_is_reported_marked_and_retried_on_the_next_run(self):
        common = dict(STAGES="g", ARMS="legacy gptq", STATES_NEW="4", SKIP_SMOKE="1")
        r = self.run_driver(STUB_FAIL_UNIT="--weights gptq --gptq-freeze", **common)
        self.assertNotEqual(r.returncode, 0); self.assertIn("UNITS NOT FINISHED", r.stdout); self.assertIn("g:gptq:4", r.stdout)
        self.assertIn("unit_g_gptq_4.failed", self.state()); self.assertIn("state_FP.done", self.state()); self.assertNotIn("full.done", self.state())
        r2 = self.run_driver(**common)
        self.assertEqual(r2.returncode, 0, r2.stdout[-2000:]); self.assertIn("unit_g_gptq_4.done", self.state()); self.assertNotIn("unit_g_gptq_4.failed", self.state())

    def test_a_failed_gate_stops_the_run_before_any_weight_set_unit(self):
        r = self.run_driver(STUB_FAIL_GATE="1")
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse((self.repo / "reports/functional_validation/gemma9b/artifact_hashes.json").exists())
        self.assertFalse([c for c in self.calls() if "--weights" in c])
        self.assertNotIn("unit_g_gptq_4.done", self.state())

    def test_a_legacy_only_run_never_touches_the_weight_machinery(self):
        r = self.run_driver(ARMS="legacy", STAGES="g")
        self.assertEqual(r.returncode, 0, r.stdout[-2000:])
        self.assertFalse([c for c in self.calls() if "--weights" in c or "run_with_weights" in c or "gate" in c.split() or "record_hashes" in c])
        self.assertFalse((self.repo / "reports/functional_validation/gemma9b/artifact_hashes.json").exists())

    def test_the_gate_is_refused_when_the_ram_cannot_hold_its_weight_sets(self):
        r = self.run_driver(MODEL_GB="18.5", TARGET_GB="17.6", FAKE_RAM_GB="100")
        self.assertNotEqual(r.returncode, 0); self.assertIn("not enough host RAM for the gate job", r.stderr)
        self.assertFalse([c for c in self.calls() if "--weights" in c])

    def test_a_unit_whose_weight_sets_do_not_fit_is_refused_while_the_gate_still_runs(self):
        r = self.run_driver(MODEL_GB="18.5", TARGET_GB="17.6", FAKE_RAM_GB="120", ARMS="gptq", STAGES="p1", P1_BITS_NEW="8 7 6 5 4 16 12 0", SKIP_SMOKE="1")
        self.assertNotEqual(r.returncode, 0, r.stdout[-1500:]); self.assertIn("REFUSED: not enough host RAM", r.stdout)
        self.assertIn("unit_p1_gptq.failed", self.state()); self.assertIn("gate9b.done", self.state())
        self.assertFalse([c for c in self.unit_calls() if "p1" in c.split()])

    def test_the_smoke_tags_all_languages_and_only_the_lm_run_is_restricted(self):
        self.assertEqual(self.run_driver(ARMS="legacy", STAGES="g").returncode, 0)
        tag = [c for c in self.calls() if "phase1_gemma9b_run.py tag" in c and "g9-smoke-tag" in c]
        self.assertEqual(len(tag), 1); self.assertNotIn("--languages", tag[0]); self.assertIn("--max-sentences 100", tag[0])
        lm = [c for c in self.calls() if "phase1_gemma9b_run.py lm" in c and "g9-smoke-lm" in c]
        self.assertEqual(len(lm), 1); self.assertIn("--languages eng_Latn hin_Deva khm_Khmr", lm[0])

    def test_tier_two_units_find_the_hash_of_every_bit_width_they_build(self):
        r = self.run_driver(ARMS="gptq rtn-matched", STAGES="p1", SKIP_SMOKE="1")
        self.assertEqual(r.returncode, 0, r.stdout[-2500:])
        h = json.loads((self.repo / "reports/functional_validation/gemma9b/artifact_hashes.json").read_text())["artifact_sha256"]
        self.assertEqual(sorted(h["gptq"]), ["4", "6", "7"]); self.assertEqual(sorted(h["rtn_matched"]), ["4", "6", "7"])
        self.assertTrue({"unit_p1_gptq.done", "unit_p1_rtn-matched.done"} <= self.state())

    def test_a_failing_frozen_hash_check_stops_the_run_at_once_and_on_every_resume(self):
        r = self.run_driver(STUB_FAIL_VERIFY="1")
        self.assertNotEqual(r.returncode, 0); self.assertNotIn("env.done", self.state()); self.assertFalse(self.unit_calls())
        self.assertEqual(self.run_driver().returncode, 0)                                         # the same tree is fine when the check passes
        r = self.run_driver(STUB_FAIL_VERIFY="1")                                                  # a resume re-verifies even though every step is done
        self.assertNotEqual(r.returncode, 0); self.assertIn("frozen", r.stdout + r.stderr)

    def test_a_failing_test_run_or_pip_install_or_hash_recording_is_not_swallowed(self):
        for switch, marker in (("STUB_FAIL_UNITTEST", "checks.done"), ("STUB_FAIL_PIP", "env.done"), ("STUB_FAIL_RECORD", "gate9b.done")):
            with self.subTest(switch):
                self.tearDown(); self.setUp()
                r = self.run_driver(**{switch: "1"})
                self.assertNotEqual(r.returncode, 0, r.stdout[-1500:]); self.assertNotIn(marker, self.state()); self.assertNotIn("ALL DONE", r.stdout)
                self.assertFalse([c for c in self.unit_calls() if "smoke" not in c])               # no real unit started (the smoke runs come before the gate)

    def test_stop_after_pauses_before_the_long_units_and_a_resume_continues(self):
        r = self.run_driver(STOP_AFTER="smoke_gptq")
        self.assertEqual(r.returncode, 0, r.stdout[-1500:]); self.assertIn("STOP_AFTER=smoke_gptq", r.stdout)
        self.assertIn("smoke_gptq.done", self.state()); self.assertNotIn("full.done", self.state()); self.assertFalse([c for c in self.unit_calls() if "g9-lm-" in c and "smoke" not in c])
        self.assertEqual(self.run_driver().returncode, 0); self.assertIn("full.done", self.state()); self.assertIn("unit_p1_gptq.done", self.state())


if __name__ == "__main__":
    unittest.main()
