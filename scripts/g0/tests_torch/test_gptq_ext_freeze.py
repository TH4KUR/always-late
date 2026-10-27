"""Task 7 of the GPTQ extension plan: the extension freeze names every endpoint, pins the hashes it can pin before any artifact exists, agrees with the code's defaults and the committed calibration freeze, and holds no artifact hash."""
import hashlib
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
G0 = HERE.parent
ROOT = G0.parents[1]
sys.path.insert(0, str(G0))
from gptq_ext import calibrate as cb, calibration as cal, states, upstream  # noqa: E402

FREEZE = json.loads((ROOT / "reports/gptq_extension/freeze.json").read_text())
CAL = json.loads((ROOT / "reports/gptq_extension/calibration_freeze.json").read_text())


class Freeze(unittest.TestCase):
    def test_freeze_names_every_endpoint_and_pins_hashes(self):
        self.assertEqual(sorted(FREEZE["endpoints"]), [f"E{i}" for i in range(1, 8)])
        self.assertEqual(FREEZE["upstream"]["sha256"], upstream.PINNED_SHA256)
        self.assertEqual(FREEZE["calibration"]["freeze_file_sha256"], hashlib.sha256((ROOT / "reports/gptq_extension/calibration_freeze.json").read_bytes()).hexdigest())
        self.assertEqual(FREEZE["calibration"]["pool_sha256"], CAL["pool_sha256"])

    def test_freeze_holds_no_artifact_hash_they_are_recorded_by_amendment(self):
        text = json.dumps(FREEZE)
        self.assertNotIn("artifact_sha256", text)
        self.assertEqual(FREEZE["amendments"], [])

    def test_gptq_config_equals_the_code_defaults(self):
        c = cb.GptqConfig(bits=4); f = FREEZE["gptq_config"]
        self.assertEqual((f["blocksize"], f["percdamp"], f["actorder"], f["groupsize"], f["sym"], f["mse"]), (c.blocksize, c.percdamp, c.actorder, c.groupsize, c.sym, c.mse))

    def test_bits_agree_with_the_shim(self):
        self.assertEqual(sorted(FREEZE["bits"]["priority"] + FREEZE["bits"]["later_if_time"]), sorted(states.ARTIFACT_BITS))
        self.assertEqual(sorted(FREEZE["bits"]["legacy_rounding_floors"]), sorted(states.LEGACY_BITS))

    def test_calibration_numbers_agree_with_the_committed_pool_freeze(self):
        c = FREEZE["calibration"]
        self.assertEqual((c["languages"], c["tokens_per_language"], c["total_tokens"]), (27, cal.BUDGET_TOKENS, 27 * cal.BUDGET_TOKENS))
        self.assertTrue(CAL["complete"])
        self.assertTrue(all(r["shortfall"] == 0 and r["tokens"] == cal.BUDGET_TOKENS for r in CAL["languages"].values()))

    def test_stage_and_drop_order_are_listed_and_the_central_stages_are_not_in_the_drop_list(self):
        self.assertEqual(FREEZE["stage_order"][0], "build and gates")
        self.assertTrue(FREEZE["drop_order_if_time_runs_out"])
        self.assertFalse(any("E1" in d or "E6" in d or "transfer" in d for d in FREEZE["drop_order_if_time_runs_out"]))

    def test_the_wrapper_is_the_documented_way_to_run_a_stage(self):
        self.assertIn("run_with_weights.py", FREEZE["how_a_stage_is_run"])


if __name__ == "__main__":
    unittest.main()
