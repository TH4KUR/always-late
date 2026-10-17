"""Tests for scripts/g0/phase1_wandb_manifest.py: manifest entries that put the Phase 1 runs (equal-budget and all-token streams) under the fixed wandb schema. stdlib only."""
import json, sys, tempfile, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import g0_wandb as g       # noqa: E402
import phase1_wandb_manifest as pm   # noqa: E402


class Entries(unittest.TestCase):
    def entries(self):
        return pm.entries_for_run("p1-mask", "mask", "25745", "618496f9e67837767c1459876f4965c0bad9dd59", tokens_eq=27036, tokens_all=3_300_000)

    def test_two_entries_one_per_stream_with_distinct_names_and_the_summary_paths(self):
        e = self.entries(); self.assertEqual([x["name"] for x in e], ["phase1/mask/eq", "phase1/mask/all"])
        self.assertEqual([x["summary_csv"] for x in e], ["reports/phase1/mask/eq/summary.csv", "reports/phase1/mask/all/summary.csv"])
        for x in e: self.assertEqual(x["files"], [f"reports/phase1/mask/{x['config']['stream']}/{n}" for n in ("per_language.csv", "meta.json", "analysis.json", "analysis.md")])

    def test_config_maps_onto_the_fixed_schema_with_the_stream_and_budget_in_config_extra(self):
        for x in self.entries():
            cfg = g.normalize_config(x["config"])
            self.assertEqual(set(cfg), set(g.CONFIG_KEYS)); self.assertEqual(cfg["experiment"], "phase1"); self.assertEqual(cfg["bos_regime"], "mask"); self.assertEqual(cfg["slurm_job"], "25745")
            self.assertEqual(cfg["stack"], "TL"); self.assertEqual(cfg["layer"], 12); self.assertTrue(cfg["code_ref"].endswith("@618496f"))
            extra = json.loads(cfg["config_extra"]); self.assertIn(extra["stream"], ("eq", "all")); self.assertIn("budget_rule", extra)
        eq, al = self.entries(); self.assertEqual(eq["config"]["n_tokens"], 27036); self.assertEqual(al["config"]["n_tokens"], 3_300_000)

    def test_tags_name_the_regime_and_the_stream(self):
        for x in self.entries(): self.assertEqual(sorted(x["tags"]), sorted(["phase1", "mask", x["config"]["stream"]]))


class AddToManifest(unittest.TestCase):
    def test_entries_are_appended_once_and_existing_ones_are_kept(self):
        p = Path(tempfile.mkdtemp()) / "m.json"; p.write_text(json.dumps({"schema_version": 1, "runs": [{"name": "old"}]}))
        pm.add_to_manifest(p, pm.entries_for_run("p1-none", "none", "25746", "abc1234", 27036, 3_000_000))
        pm.add_to_manifest(p, pm.entries_for_run("p1-none", "none", "25746", "abc1234", 27036, 3_000_000))             # idempotent
        names = [r["name"] for r in json.loads(p.read_text())["runs"]]; self.assertEqual(names, ["old", "phase1/none/eq", "phase1/none/all"])


if __name__ == "__main__":
    unittest.main()
