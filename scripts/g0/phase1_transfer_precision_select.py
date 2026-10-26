"""Frozen selection of the 512 pairs of the paired arithmetic-precision audit (task 2 of docs/superpowers/plans/2026-10-26-publication-recovery-after-results.md; implementation plan
docs/superpowers/plans/2026-10-27-precision-audit-task2-implementation.md, step 1; POST HOC revision R5).

The rule (written before any GPU code, reading FULL-PRECISION quantities only): the feature-level FP effect norm is the median over the feature's contexts of the FP effect norm of the real direction; the features of
the committed core are ranked by (that norm, feature id) and cut into four quartiles of equal rank; inside a quartile the features with the lowest sha256 of `salt:feature id` are chosen (`per_quartile` of them);
each feature contributes the first `n_contexts` contexts in the order of the frozen core (the original FP ranking). Nothing quantized, no residual, no desired sign and no language enters; a feature with fewer
contexts than needed or a missing norm is excluded and listed.

    python phase1_transfer_precision_select.py --core <core_sample.json> --table <per_context_INT16.csv> --salt <salt> --out <precision_audit_freeze.json>
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

RULE = ("feature FP effect norm = median over the feature's core contexts of the FP effect norm (fp_norm) of the real direction; features ranked by (norm, feature id), quartile = floor(4 * rank / n); inside a quartile the "
        "features with the lowest sha256 of 'salt:feature id' are chosen; each chosen feature contributes the first n_contexts contexts in the order of the frozen core (the original FP ranking). Only FP quantities are read.")


def fp_norms_from_csv(path):
    """{(feature, sentence): fp_norm} of the REAL-direction items of a per-context table; only the columns feature, sentence, kind and fp_norm are read (the FP effect norm is the same in every state table)."""
    out = {}
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh)
        if "fp_norm" not in (rd.fieldnames or []):
            raise ValueError(f"{path} has no fp_norm column")
        for r in rd:
            if r["kind"] != "real":
                continue
            k = (int(r["feature"]), int(r["sentence"]))
            if k in out:
                raise ValueError(f"the real item {k} occurs twice in {path}")
            out[k] = float(r["fp_norm"])
    return out


def _hash(salt, f):
    return hashlib.sha256(f"{salt}:{f}".encode()).hexdigest()


def select(core, norms, salt, per_quartile=32, n_contexts=4):
    """-> {"features": the chosen features (quartile, hash order), "all_features": every eligible feature with its quartile, "excluded": [{"feature", "reason"}], "selection_sha256": sha256 of the canonical JSON of
    `features`}. Raises ValueError if a quartile has fewer than `per_quartile` eligible features (never a silently smaller sample)."""
    elig, excluded = [], []
    for k, f in enumerate(core["feature_ids"]):
        ctx = [int(s) for s in core["contexts"][k]]
        if len(ctx) < n_contexts:
            excluded.append({"feature": int(f), "reason": f"fewer than {n_contexts} contexts"}); continue
        vals = [norms.get((int(f), s)) for s in ctx]
        if any(v is None or not np.isfinite(v) for v in vals):
            excluded.append({"feature": int(f), "reason": "missing FP effect norm"}); continue
        elig.append({"feature": int(f), "direction_id": int(core["direction_ids"][k]), "language_index": int(core["feature_language_index"][k]), "feature_fp_norm": float(np.median(vals)), "contexts": ctx[:n_contexts]})
    order = sorted(elig, key=lambda e: (e["feature_fp_norm"], e["feature"])); n = len(order)
    for i, e in enumerate(order):
        e["quartile"] = (4 * i) // n
    chosen = []
    for q in range(4):
        members = sorted((e for e in order if e["quartile"] == q), key=lambda e: (_hash(salt, e["feature"]), e["feature"]))
        if len(members) < per_quartile:
            raise ValueError(f"quartile {q} has {len(members)} eligible features, {per_quartile} needed")
        chosen += [{**e, "rank_hash": _hash(salt, e["feature"])} for e in members[:per_quartile]]
    payload = json.dumps(chosen, sort_keys=True, separators=(",", ":"))
    return {"features": chosen, "all_features": sorted(order, key=lambda e: e["feature"]), "excluded": sorted(excluded, key=lambda e: e["feature"]), "selection_sha256": hashlib.sha256(payload.encode()).hexdigest()}


def build_freeze(core_path, table_path, salt, per_quartile=32, n_contexts=4):
    """The freeze record: the rule, the salt, the sha256 of both inputs, the counts, the excluded features and the chosen features with their contexts."""
    core = json.loads(Path(core_path).read_text()); norms = fp_norms_from_csv(table_path); r = select(core, norms, salt, per_quartile, n_contexts)
    return {"schema": 1, "kind": "intervention-transfer-precision-audit-freeze", "rule": RULE, "salt": salt, "inputs_sha256": {"core": hashlib.sha256(Path(core_path).read_bytes()).hexdigest(), "fp_norm_table": hashlib.sha256(Path(table_path).read_bytes()).hexdigest()},
            "counts": {"features_in_core": len(core["feature_ids"]), "excluded": len(r["excluded"]), "per_quartile": per_quartile, "n_contexts": n_contexts, "selected_features": len(r["features"]), "selected_pairs": len(r["features"]) * n_contexts},
            "excluded": r["excluded"], "features": r["features"], "selection_sha256": r["selection_sha256"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--core", required=True); ap.add_argument("--table", required=True); ap.add_argument("--salt", required=True)
    ap.add_argument("--per-quartile", type=int, default=32); ap.add_argument("--contexts", type=int, default=4); ap.add_argument("--out", required=True); a = ap.parse_args()
    rec = build_freeze(a.core, a.table, a.salt, a.per_quartile, a.contexts); Path(a.out).write_text(json.dumps(rec, indent=1)); print(rec["counts"], rec["selection_sha256"])


if __name__ == "__main__":
    main()
