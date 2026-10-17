"""Tests for scripts/g0/their_code.py: their functions are taken VERBATIM, by AST, from the vendored pristine script (never retyped), and a frozen
hash manifest makes any drift of what we extract visible. stdlib only."""
import json, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "g0"))
import their_code as tc  # noqa: E402

SRC = '''import os
SENTINEL = []
SENTINEL.append("module-level statement ran")

def dup(x):
    return "first"

def uses_global():
    return DEVICE

def dup(x):
    return "second"

class K:
    def nested_name(self):
        return 1

def outer():
    def inner():
        return 2
    return inner
'''


def write(text):
    d = tempfile.mkdtemp(); p = Path(d) / "s.py"; p.write_text(text); return p


class Extract(unittest.TestCase):
    def test_first_module_level_definition_is_taken_with_exact_text_line_and_hash(self):
        p = write(SRC); got = tc.extract_functions(p, ["dup", "uses_global"])
        self.assertEqual(got["dup"]["source"], 'def dup(x):\n    return "first"')
        self.assertEqual(got["dup"]["lineno"], 5); self.assertEqual(len(got["dup"]["sha256"]), 64)
        self.assertEqual(got["uses_global"]["lineno"], 8)

    def test_nested_functions_and_methods_are_not_module_level_and_are_not_found(self):
        p = write(SRC)
        for name in ("inner", "nested_name"):
            with self.assertRaises(KeyError): tc.extract_functions(p, [name])

    def test_missing_name_raises_naming_it(self):
        with self.assertRaises(KeyError) as cm: tc.extract_functions(write(SRC), ["nope"])
        self.assertIn("nope", str(cm.exception))


class Namespace(unittest.TestCase):
    def test_functions_run_against_the_injected_globals_and_module_statements_never_run(self):
        p = write(SRC); env = {"DEVICE": "cuda:7"}
        ex = tc.extract_functions(p, ["uses_global"]); tc.build_namespace(ex, env)
        self.assertEqual(env["uses_global"](), "cuda:7"); self.assertNotIn("SENTINEL", env)


class Frozen(unittest.TestCase):
    def test_verify_hashes_passes_on_identical_and_fails_on_a_changed_body_or_a_missing_entry(self):
        p = write(SRC); ex = tc.extract_functions(p, ["dup", "uses_global"]); frozen = tc.freeze(ex)
        tc.verify_hashes(ex, frozen)
        p2 = write(SRC.replace('"first"', '"changed"')); ex2 = tc.extract_functions(p2, ["dup", "uses_global"])
        with self.assertRaises(ValueError) as cm: tc.verify_hashes(ex2, frozen)
        self.assertIn("dup", str(cm.exception)); self.assertNotIn("uses_global", str(cm.exception))
        with self.assertRaises(ValueError): tc.verify_hashes(ex, {k: v for k, v in frozen.items() if k != "dup"})


class RealFile(unittest.TestCase):
    PRISTINE = ROOT / "third_party/duan2026/pristine/scripts/03_gemma_phase3_streaming_sweep.py"
    MANIFEST = ROOT / "reports/g0/their_functions.json"

    def test_every_function_phase1_uses_is_found_and_matches_the_frozen_manifest(self):
        m = json.loads(self.MANIFEST.read_text())
        ex = tc.extract_functions(self.PRISTINE, tc.THEIR_FUNCTIONS)
        self.assertEqual(sorted(m["functions"]), sorted(tc.THEIR_FUNCTIONS))
        tc.verify_hashes(ex, m["functions"])
        self.assertEqual(m["file_sha256"], tc.sha256_file(self.PRISTINE))

    def test_the_quantizer_taken_is_their_shared_scale_quantizer_not_a_re_implementation(self):
        ex = tc.extract_functions(self.PRISTINE, ["quantize_rtn_per_output_channel_tl"])
        self.assertIn("amax(\n                    dim=tuple(range(w.ndim - 1))", ex["quantize_rtn_per_output_channel_tl"]["source"])


if __name__ == "__main__":
    unittest.main()
