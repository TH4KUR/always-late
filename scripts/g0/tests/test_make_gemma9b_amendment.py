"""Task 5 of the Gemma-2-9B three-arm plan: the dated amendment A1 of the 9B freeze. It freezes, before any 9B number exists, the verbatim text of the new predictions (spec section 5: Q0 to Q3 for tier 1, S1 to S4
for tier 2, the two verdict functions), the arms and states, the sha256 of the new code files and of the unchanged freeze.json. The text is read from the spec between the line that starts `- **Q0` and the closing line
`(The Q0 to Q3 list ...`. Stdlib only; fixtures in a temporary directory."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import make_gemma9b_amendment as ma

SPEC = """# Spec

## 4. Something
text before

## 5. Predictions

intro line

- **Q0 (existence):** P1 holds. (2B: met.)
- **Q1 (ordering):** r(GPTQ) <= r(legacy) - 0.3.
- **Q2 (6 bits):** retention in [0.85, 1.15].
- **Q3 (repair hint, reported):** H-G5b per arm.

**Verdict for the project's updates (the complete function):** *consistent* = Q0 and Q1 and Q2; *partly consistent* = Q0 and Q2 and not Q1; *different* = not Q0 or not Q2.

**Tier 2 (Phase 1 survival and loss), frozen the same way.**

- **S1 (survival ordering):** GPTQ > matched > legacy.
- **S2 (loss ordering):** GPTQ < matched < legacy.
- **S3 (tier gap):** smaller under GPTQ.
- **S4 (original tests, reported):** verdict word per arm.

**Tier 2 verdict:** *consistent* = S1 and S2 and S3; *partly consistent* = S1 and S2 and not S3; *different* = not S1 or not S2.

(The Q0 to Q3 list, the S1 to S4 list and the two verdict paragraphs are the amendment's `text`; its sha256 is stored in the amendment.)

## 6. Analysis tool
after
"""


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        (self.root / "docs").mkdir(); self.spec = self.root / "docs/spec.md"; self.spec.write_text(SPEC)
        for rel in ("a.py", "b.py"):
            (self.root / rel).write_text(f"# {rel}\r\nx = 1\r\n".encode().decode(), newline="")
        (self.root / "freeze.json").write_text('{"frozen": true}\n')
        self.kw = dict(root=self.root, code=("a.py", "b.py"), freeze_path=self.root / "freeze.json")

    def tearDown(self):
        self.tmp.cleanup()

    def build(self):
        return ma.build(self.spec, "2026-10-28 20:00", "abc1234", **self.kw)


class Text(Fixture):
    def test_text_is_the_verbatim_spec_block(self):
        t = ma.extract_text(self.spec)
        self.assertTrue(t.startswith("- **Q0 (existence)"))
        self.assertTrue(t.rstrip().endswith("stored in the amendment.)"))
        self.assertIn("**Tier 2 verdict:**", t)
        self.assertNotIn("## 6.", t); self.assertNotIn("intro line", t)
        self.assertIn(t.rstrip(), SPEC)

    def test_a_missing_start_or_end_marker_is_an_error(self):
        self.spec.write_text(SPEC.replace("- **Q0", "- **Z0"))
        with self.assertRaises(ValueError):
            ma.extract_text(self.spec)
        self.spec.write_text(SPEC.replace("(The Q0 to Q3 list", "(The list"))
        with self.assertRaises(ValueError):
            ma.extract_text(self.spec)

    def test_the_text_names_every_prediction_and_both_verdict_words(self):
        t = ma.extract_text(self.spec)
        for key in ("Q0", "Q1", "Q2", "Q3", "S1", "S2", "S3", "S4", "consistent", "partly consistent", "different"):
            self.assertIn(key, t)


class Build(Fixture):
    def test_the_amendment_carries_the_text_its_hash_the_arms_the_code_and_the_freeze_hash(self):
        a = self.build()
        self.assertEqual((a["kind"], a["id"], a["created"], a["base_commit"]), ("gemma9b_amendment", "A1", "2026-10-28 20:00", "abc1234"))
        self.assertEqual(a["text"], ma.extract_text(self.spec))
        self.assertEqual(a["text_sha256"], ma.hashlib.sha256(a["text"].encode("utf-8")).hexdigest())
        self.assertEqual(sorted(a["code_sha256"]), ["a.py", "b.py"])
        self.assertEqual(a["freeze_sha256"], ma.sha(self.root / "freeze.json"))
        self.assertEqual(a["arms"]["tier1"]["legacy"], ["FP", "INT8", "INT7", "INT6", "INT5", "INT4"])
        self.assertEqual(a["arms"]["tier1"]["rtn-matched"], ["INT6", "INT4"]); self.assertEqual(a["arms"]["tier1"]["gptq"], ["INT6", "INT4"])
        self.assertEqual(a["arms"]["tier2"]["legacy"], [8, 7, 6, 5, 4, 16, 12, 0]); self.assertEqual(a["arms"]["tier2"]["gptq"], [7, 6, 4, 16, 12, 0])

    def test_hash_changes_when_one_threshold_changes(self):
        h1 = self.build()["text_sha256"]
        self.spec.write_text(SPEC.replace("r(legacy) - 0.3", "r(legacy) - 0.2"))
        self.assertNotEqual(self.build()["text_sha256"], h1)

    def test_the_code_hash_ignores_the_line_ending(self):
        h1 = self.build()["code_sha256"]["a.py"]
        (self.root / "a.py").write_bytes((self.root / "a.py").read_bytes().replace(b"\r\n", b"\n"))
        self.assertEqual(self.build()["code_sha256"]["a.py"], h1)


class VerifyAndCli(Fixture):
    def write(self):
        out = self.root / "amend.json"; out.write_text(json.dumps(self.build())); return out

    def test_verify_is_empty_for_an_unchanged_checkout(self):
        self.assertEqual(ma.verify(self.write(), **self.kw, spec_path=self.spec), [])

    def test_verify_lists_a_changed_code_file(self):
        out = self.write(); (self.root / "b.py").write_text("changed\n")
        self.assertEqual(ma.verify(out, **self.kw, spec_path=self.spec), ["code:b.py"])

    def test_verify_lists_a_changed_freeze_and_a_changed_spec_text(self):
        out = self.write(); (self.root / "freeze.json").write_text('{"frozen": false}\n'); self.spec.write_text(SPEC.replace("0.85", "0.80"))
        self.assertEqual(sorted(ma.verify(out, **self.kw, spec_path=self.spec)), ["freeze.json", "spec_text"])

    def test_verify_lists_a_tampered_amendment_text(self):
        out = self.write(); d = json.loads(out.read_text()); d["text"] = d["text"].replace("0.3", "0.1"); out.write_text(json.dumps(d))
        self.assertIn("text_sha256", ma.verify(out, **self.kw, spec_path=self.spec))

    def test_refuses_to_overwrite_and_the_cli_writes_and_verifies(self):
        out = self.root / "out.json"
        argv = ["--spec", str(self.spec), "--base-commit", "abc1234", "--out", str(out), "--root", str(self.root), "--code", "a.py", "b.py", "--freeze", str(self.root / "freeze.json")]
        self.assertEqual(ma.main(argv), 0)
        with self.assertRaises(SystemExit):
            ma.main(argv)
        self.assertEqual(ma.main(["--verify", "--out", str(out), "--spec", str(self.spec), "--root", str(self.root), "--code", "a.py", "b.py", "--freeze", str(self.root / "freeze.json")]), 0)


if __name__ == "__main__":
    unittest.main()
