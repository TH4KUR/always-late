import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_transfer_precision_main as pm
from test_phase1_transfer_precision_run import make_freeze

KINDS = ("real", "matched_swapped")


class Skipped(unittest.TestCase):
    def test_a_skipped_context_expands_to_its_audited_kinds_and_a_skipped_item_stays_one_key_and_other_kinds_are_dropped(self):
        entries = [(11, 5, "no_fp_active_position"), ((12, 6, "real"), "no_edited_row"), ((12, 6, "random1"), "no_edited_row"), ((12, 6, "real"), "no_edited_row")]
        self.assertEqual(pm.expand_skipped(entries, KINDS), [(11, 5, "real"), (11, 5, "matched_swapped"), (12, 6, "real")])

    def test_an_entry_of_an_unexpected_shape_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "skipped"):
            pm.expand_skipped([(1,)], KINDS)


class Kinds(unittest.TestCase):
    def test_the_direction_kinds_of_a_module_are_restricted_inside_the_block_and_restored_also_after_an_exception(self):
        class M:
            DIRECTION_KINDS = ("real", "matched_swapped", "random1", "random2", "random3")
        with pm.direction_kinds(M, KINDS):
            self.assertEqual(M.DIRECTION_KINDS, KINDS)
        self.assertEqual(len(M.DIRECTION_KINDS), 5)
        with self.assertRaises(KeyError):
            with pm.direction_kinds(M, KINDS):
                raise KeyError("x")
        self.assertEqual(len(M.DIRECTION_KINDS), 5)

    def test_a_kind_that_the_module_does_not_know_is_refused(self):
        class M:
            DIRECTION_KINDS = ("real", "matched_swapped")
        with self.assertRaisesRegex(ValueError, "random9"):
            with pm.direction_kinds(M, ("real", "random9")):
                pass


class Probe(unittest.TestCase):
    def test_the_probe_paths_cover_the_four_tensors_of_the_first_and_the_last_suffix_block(self):
        self.assertEqual(pm.probe_paths(), pm.probe_paths([13, 25])); self.assertEqual(len(pm.probe_paths()), 8); p = pm.probe_paths([13, 25]); self.assertEqual(p["13.normalized"], "blocks.13.ln1.hook_normalized"); self.assertEqual(p["25.q"], "blocks.25.attn.hook_q"); self.assertEqual(p["13.pattern"], "blocks.13.attn.hook_pattern"); self.assertEqual(p["25.mlp_post"], "blocks.25.mlp.hook_post"); self.assertEqual(len(p), 8)


class Aux(unittest.TestCase):
    def write_table(self, path, freeze, extra_rows=()):
        rows = []
        for e in freeze["features"]:
            for s in e["contexts"]:
                for kind in ("real", "matched_swapped"):
                    rows.append({"feature": e["feature"], "sentence": s, "kind": kind, "cluster": 7000 + s % 50, "fp_norm": 1.0})
        rows += list(extra_rows)
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["feature", "sentence", "kind", "cluster", "fp_norm"]); w.writeheader(); w.writerows(rows)

    def test_the_aux_file_holds_the_cluster_of_every_selected_pair_the_old_floors_and_the_input_hashes(self):
        fr = make_freeze(2, n_ctx=2)
        with tempfile.TemporaryDirectory() as d:
            t = Path(d) / "t.csv"; self.write_table(t, fr, extra_rows=[{"feature": 999, "sentence": 1, "kind": "real", "cluster": 1, "fp_norm": 1.0}]); aux = pm.build_aux(fr, t, {"real": 28.0, "matched_swapped": 27.0, "random1": 26.0})
        self.assertEqual(len(aux["clusters"]), 16); e = fr["features"][1]; self.assertEqual(aux["clusters"][f"{e['feature']}|{e['contexts'][0]}"], 7000 + e["contexts"][0] % 50); self.assertEqual(aux["old_floors"], {"real": 28.0, "matched_swapped": 27.0}); self.assertEqual(len(aux["inputs_sha256"]["per_context_table"]), 64)
        self.assertNotIn("999|1", aux["clusters"])

    def test_a_selected_pair_without_a_cluster_or_with_two_different_clusters_is_an_error(self):
        fr = make_freeze(1, n_ctx=2)
        with tempfile.TemporaryDirectory() as d:
            t = Path(d) / "t.csv"; self.write_table(t, fr); lines = t.read_text().splitlines(); t.write_text("\n".join([lines[0]] + lines[2:]) + "\n")                                # the first data row, the real row of the first pair, is removed
            with self.assertRaisesRegex(ValueError, "no cluster"):
                pm.build_aux(fr, t, {"real": 1.0, "matched_swapped": 1.0})
            self.write_table(t, fr); e = fr["features"][0]; self.write_table(t, fr, extra_rows=[{"feature": e["feature"], "sentence": e["contexts"][0], "kind": "real", "cluster": 1, "fp_norm": 1.0}])
            with self.assertRaisesRegex(ValueError, "two clusters"):
                pm.build_aux(fr, t, {"real": 1.0, "matched_swapped": 1.0})

    def test_the_aux_file_is_read_back_into_tuple_keys_and_the_floors(self):
        fr = make_freeze(1, n_ctx=2)
        with tempfile.TemporaryDirectory() as d:
            t = Path(d) / "t.csv"; self.write_table(t, fr); aux = pm.build_aux(fr, t, {"real": 1.5, "matched_swapped": 2.5}); p = Path(d) / "aux.json"; p.write_text(json.dumps(aux)); clusters, floors = pm.load_aux(p)
        e = fr["features"][1]; self.assertEqual(clusters[(e["feature"], e["contexts"][0])], 7000 + e["contexts"][0] % 50); self.assertEqual(floors, {"real": 1.5, "matched_swapped": 2.5})

    def test_a_floor_for_a_kind_that_is_missing_is_an_error(self):
        fr = make_freeze(1, n_ctx=2)
        with tempfile.TemporaryDirectory() as d:
            t = Path(d) / "t.csv"; self.write_table(t, fr)
            with self.assertRaisesRegex(ValueError, "matched_swapped"):
                pm.build_aux(fr, t, {"real": 1.0})


class Args(unittest.TestCase):
    def test_the_defaults_point_at_the_committed_files_and_the_instrument_flag_limits_the_run_to_one_chunk(self):
        a = pm.parse_args(["--out", "o", "--flores-root", "f"]); self.assertTrue(str(a.audit_freeze).endswith("reports/intervention_transfer/precision_audit_freeze.json")); self.assertTrue(str(a.aux).endswith("reports/intervention_transfer/precision_audit_aux.json"))
        self.assertEqual((a.chunk_features, a.n_boot, a.seed, a.instrument), (16, 2000, 20261026, False)); self.assertTrue(pm.parse_args(["--out", "o", "--flores-root", "f", "--instrument"]).instrument)
        import contextlib, io
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            pm.parse_args(["--flores-root", "f"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            pm.parse_args(["--out", "o"])                                                       # --flores-root is needed unless the aux file is built
        self.assertTrue(pm.parse_args(["--out", "o", "--build-aux", "--table", "t.csv"]).build_aux)


if __name__ == "__main__":
    unittest.main()
