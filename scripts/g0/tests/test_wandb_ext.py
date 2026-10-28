"""Tests for scripts/g0/gptq_ext/wandb_ext.py (design reports/wandb_schema_v2_gptq_extension.md, D-115): the nine extension runs build with the UNCHANGED v2 builder, their headline values equal the committed headline
file and result JSON, the slugs and names never collide with the v2 runs, every decision id exists, and `verify` reports every difference between a payload and what wandb returns. Synthetic fake API."""
import json
import re
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "gptq_ext")); sys.path.insert(0, str(HERE.parent))
import wandb_ext as ext  # noqa: E402
import wandb_upload_v2 as wu  # noqa: E402

HL = json.loads((ROOT / "reports/gptq_extension/wandb_headlines.json").read_text())


class Families(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = wu.build_all(ROOT, ext.FAMILIES_EXT, upload_commit="test"); cls.runs = {r["id"]: r for r in cls.payload["runs"]}

    def test_nine_runs_build_with_the_v2_builder_and_have_the_v2_ids(self):
        self.assertEqual(len(self.runs), 9)
        for i, r in self.runs.items():
            self.assertTrue(i.startswith("v2-")); self.assertEqual(r["tags"][:3], ["schema-v2", "post-hoc-upload", "exploratory"]); self.assertEqual(r["job_type"], "analysis-upload")
            self.assertTrue(r["files"]); self.assertLess(r["bytes_total"], wu.MAX_RUN_BYTES)

    def test_slugs_and_names_do_not_collide_with_the_v2_runs(self):
        old_slugs = {f["slug"] for f in wu.FAMILIES}; old_names = {f["name"] for f in wu.FAMILIES}
        new_slugs = [f["slug"] for f in ext.FAMILIES_EXT]; new_names = [f["name"] for f in ext.FAMILIES_EXT]
        self.assertEqual(len(set(new_slugs)), 9); self.assertEqual(len(set(new_names)), 9)
        self.assertFalse(old_slugs & set(new_slugs)); self.assertFalse(old_names & set(new_names))

    def test_every_decision_id_exists_in_the_decision_file(self):
        text = (ROOT / "reports/decisions_and_anomalies.md").read_text()
        for f in ext.FAMILIES_EXT:
            self.assertTrue(f["decisions"], f["slug"])
            for d in f["decisions"]:
                self.assertRegex(text, r"(?m)^\| " + re.escape(d) + r" \|", (f["slug"], d))

    def test_no_run_ships_weights_or_item_files_or_a_path_outside_reports(self):
        for r in self.payload["runs"]:
            for x in r["files"]:
                self.assertTrue(x["path"].startswith("reports/")); self.assertNotIn(Path(x["path"]).suffix.lower(), wu.BANNED_EXT)
                self.assertNotIn("per_context", x["path"]); self.assertNotIn("averaged_features", x["path"])

    def test_the_seven_extension_runs_carry_exactly_the_headline_file_and_its_values(self):
        for slug, d in HL.items():
            r = self.runs[f"v2-{slug}"]; got = {k[len("headline/"):]: v for k, v in r["summary"].items() if k.startswith("headline/")}
            self.assertEqual(got, d, slug)

    def test_the_survival_effect_runs_read_the_result_json_of_d113_and_d114(self):
        g = json.loads((ROOT / "reports/survival_effect_rescue/results/survival_effect_results.json").read_text()); s = self.runs["v2-survival-effect-gemma"]["summary"]
        self.assertEqual(s["headline/overall_statement"], g["overall_statement"]); self.assertEqual(s["headline/cells_met"], g["cells_met_primary"])
        for arm, key in (("legacy", "legacy RTN"), ("matched", "matched RTN"), ("gptq", "GPTQ")):
            for st in ("INT6", "INT4"):
                c = g["cells"][f"{key}|{st}|primary"]
                self.assertEqual(s[f"headline/improvement_B_over_A_{st}_{arm}"], c["improvement"]["B_over_A"]["improvement"]); self.assertEqual(s[f"headline/partial_spearman_{st}_{arm}"], c["partial_spearman_given_A"]["rho"])
        l = json.loads((ROOT / "reports/survival_effect_rescue/llama/results/llama_survival_effect_results.json").read_text()); t = self.runs["v2-survival-effect-llama"]["summary"]
        self.assertIs(t["headline/gate_ok"], True); self.assertEqual(t["headline/n_reliable"], l["gate"]["n_reliable"]); self.assertEqual(t["headline/overall_statement"], l["overall_statement"])
        for st in ("INT6", "INT4"):
            self.assertEqual(t[f"headline/improvement_B_over_A_{st}"], l["cells"][f"{st}|primary"]["improvement"]["B_over_A"]["improvement"])

    def test_config_names_the_three_arms_and_keeps_the_limits(self):
        for f in ext.FAMILIES_EXT:
            self.assertTrue(f["limits"]); self.assertTrue(f["title"])
        for slug in HL:
            c = self.runs[f"v2-{slug}"]["config"]; self.assertIn("legacy RTN", c["quantizer"]); self.assertIn("GPTQ", c["quantizer"]); self.assertEqual(c["status"], "exploratory")
        self.assertIn("meta-llama/Llama-3.2-1B", self.runs["v2-survival-effect-llama"]["config"]["model"])

    def test_tables_are_the_committed_csv_files_and_the_arm_prefixes_are_unique(self):
        for r in self.payload["runs"]:
            keys = list(r["tables"]); self.assertEqual(len(keys), len(set(keys)))
        self.assertIn("gptq_retention", self.runs["v2-gptq-ext-loss-task-rescue"]["tables"]); self.assertIn("legacy_accuracy", self.runs["v2-gptq-ext-belebele"]["tables"])


class ArmTables(unittest.TestCase):
    def test_every_arm_prefixed_table_comes_from_that_arms_file(self):
        payload = wu.build_all(ROOT, ext.FAMILIES_EXT, upload_commit="test"); seen = 0
        for r in payload["runs"]:
            for key, t in r["tables"].items():
                f = t["file"]
                if key.startswith("matched_"):
                    self.assertIn("rtnm", f, (r["id"], key)); self.assertNotIn("gptq", f.replace("gptq_extension", ""), (r["id"], key)); seen += 1
                elif key.startswith("gptq_"):
                    self.assertIn("gptq", f.replace("gptq_extension", ""), (r["id"], key)); self.assertNotIn("rtnm", f, (r["id"], key)); seen += 1
                elif key.startswith("legacy_"):
                    self.assertNotIn("gptq_extension", f, (r["id"], key)); seen += 1
        self.assertEqual(seen, 3 * 6)          # six three-arm table families: retention, rescue, accuracy, f5_tables, curves, alpha_star


class FakeRun:
    def __init__(self, summary, config, art=1):
        self.summary, self.config, self.n_art = summary, config, art

    def logged_artifacts(self):
        return [object()] * self.n_art


class FakeApi:
    def __init__(self, runs):
        self.runs = runs

    def run(self, path):
        if path not in self.runs:
            raise LookupError(path)
        return self.runs[path]


class Verify(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = wu.build_all(ROOT, [f for f in ext.FAMILIES_EXT if f["slug"] == "gptq-ext-f5"], upload_commit="test")

    def api(self, mutate=None):
        r = self.payload["runs"][0]; summary = dict(r["summary"]); summary.update({f"table/{k}": {"_type": "table-file"} for k in r["tables"]}); config = dict(r["config"])
        if mutate:
            mutate(summary, config)
        return FakeApi({f"{self.payload['entity']}/{self.payload['project']}/{r['id']}": FakeRun(summary, config)})

    def test_an_identical_run_verifies_clean(self):
        self.assertEqual(ext.verify_runs(self.payload, self.api()), [])

    def test_a_changed_summary_value_is_reported(self):
        def m(s, c): s["headline/w1_median_INT4_gptq"] = 0.5
        probs = ext.verify_runs(self.payload, self.api(m)); self.assertEqual(len(probs), 1); self.assertIn("w1_median_INT4_gptq", probs[0])

    def test_a_missing_summary_key_a_changed_file_hash_and_a_missing_table_are_reported(self):
        def m(s, c):
            del s["headline/s2_auroc_legacy"]; c["files_sha256"] = "{}"; del s["table/gptq_f5_tables"]
        probs = " | ".join(ext.verify_runs(self.payload, self.api(m)))
        self.assertIn("s2_auroc_legacy", probs); self.assertIn("files_sha256", probs); self.assertIn("gptq_f5_tables", probs)

    def test_a_run_that_is_not_there_is_reported(self):
        probs = ext.verify_runs(self.payload, FakeApi({})); self.assertEqual(len(probs), 1); self.assertIn("v2-gptq-ext-f5", probs[0])


if __name__ == "__main__":
    unittest.main()
