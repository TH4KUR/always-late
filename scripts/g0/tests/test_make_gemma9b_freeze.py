import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import make_gemma9b_freeze as mf


class Freeze(unittest.TestCase):
    def test_section_is_verbatim_and_bounded(self):
        text = "## 1. a\nx\n## 3. Predictions\n- P1\n\n## 4. next\ny\n"
        self.assertEqual(mf.section(text, 3), "## 3. Predictions\n- P1\n")
        with self.assertRaises(ValueError):
            mf.section(text, 9)

    def test_the_predictions_are_the_llama_text_and_hashed(self):
        fr = mf.build("abc", "base", "2026-10-26 00:00")
        plan = mf.PLAN.read_text(encoding="utf-8")
        self.assertIn(mf.section(plan, 3), fr["predictions"]["text"])
        self.assertIn("P3b (L3, the threshold)", fr["predictions"]["text"])
        self.assertIn("limited", fr["predictions"]["text"])                       # the section 7 amendment is part of the frozen text
        self.assertEqual(fr["predictions"]["text_sha256"], hashlib.sha256(fr["predictions"]["text"].encode("utf-8")).hexdigest())

    def test_setting_and_ratios(self):
        fr = mf.build("abc", "base", "d")
        self.assertEqual((fr["model"], fr["layer"], fr["d_model"]), ("google/gemma-2-9b", 20, 3584))
        self.assertEqual(fr["preset"]["ratios_down"], [0.3535533905932738, 0.1767766952966369, 0.08838834764831845])
        self.assertFalse(fr["preset"]["recalibrated_on_9b"])
        self.assertEqual(len(fr["preset"]["states"]), 6)

    def test_an_existing_freeze_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "freeze.json"
            mf.main(["--flores-sha256", "abc", "--base-commit", "b", "--out", str(out)])
            first = out.read_text(encoding="utf-8")
            with self.assertRaises(SystemExit):
                mf.main(["--flores-sha256", "zzz", "--base-commit", "b", "--out", str(out)])
            self.assertEqual(out.read_text(encoding="utf-8"), first)
            self.assertEqual(json.loads(first)["flores"]["sha256"], "abc")


class Verify(unittest.TestCase):
    def test_a_matching_freeze_verifies_and_a_changed_code_hash_is_reported(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "freeze.json"
            mf.main(["--flores-sha256", "abc", "--base-commit", "b", "--out", str(out)])
            self.assertEqual(mf.verify(out), [])
            mf.main(["--verify", "--out", str(out)])                                  # exits normally
            data = json.loads(out.read_text(encoding="utf-8"))
            data["code_sha256"]["scripts/g0/phase1_gemma9b_run.py"] = "0" * 64
            out.write_text(json.dumps(data), encoding="utf-8")
            self.assertEqual(mf.verify(out), ["scripts/g0/phase1_gemma9b_run.py"])
            with self.assertRaises(SystemExit):
                mf.main(["--verify", "--out", str(out)])

    def test_creating_without_the_sha_arguments_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit):
                mf.main(["--out", str(Path(d) / "f.json")])


if __name__ == "__main__":
    unittest.main()
