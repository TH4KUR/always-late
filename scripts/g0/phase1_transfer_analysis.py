"""Analysis of the intervention-transfer study (D-088; ledger reports/intervention_transfer/STATUS.md; plan Task 3). Written before the full outputs exist. This first part reads the chunk files of the runner,
builds one row per (feature, context, direction) item with the exact norms of the decomposition (from the stored Gram matrices), fixes the numerical floor from the CALIBRATION features' INT16 items only and applies
the frozen eligibility rule (FP effect norm above five times the floor of its direction kind), and tabulates coverage. Later parts (summaries with crossed bootstrap, the group-level decomposition, the
predictive baselines) are added with their own tests.

The floor is computed per direction kind (real, matched swapped, random 1 to 3): the frozen definition says 'at the same features, directions and contexts'; the held-out items never enter it (a test says so)."""
import numpy as np
import torch

import phase1_transfer_core as tc

ELIGIBILITY_MULTIPLE = 5
FLOOR_PERCENTILE = 95


def load_chunks(directory, prefix, state):
    """Merge the chunk files `<prefix>_<state>_c*.pt` of a run directory: (items, pooled, skipped). A key that occurs in two chunk files is an error."""
    import torch as _t
    from pathlib import Path
    files = sorted(Path(directory).glob(f"{prefix}_{state}_c*.pt"))
    if not files:
        raise FileNotFoundError(f"no chunk files {prefix}_{state}_c*.pt in {directory}")
    items, pooled, skipped = {}, {}, []
    for f in files:
        payload = _t.load(f, weights_only=False)["payload"]
        for k, v in payload["items"].items():
            if k in items:
                raise ValueError(f"item {k} occurs twice (second time in {f.name})")
            items[k] = v
        for k, v in payload["pooled"].items():
            if k in pooled:
                raise ValueError(f"pool {k} occurs twice (second time in {f.name})")
            pooled[k] = v
        skipped += list(payload["skipped"])
    return items, pooled, skipped


def item_table(items, core, split, cluster_mod):
    """One row per item {(feature, sentence, kind): row}: the language index of the feature, its calibration or held-out membership, the source cluster (dev index modulo `cluster_mod`), the norms of the FP
    effect, the Q/Q effect and the error, the cosine and norm ratio, the norms of C, W, I and the total, and the stored per-item scalars."""
    lang = dict(zip(core["feature_ids"], core["feature_language_index"])); where = {f: "calibration" for f in split["calibration"]}; where.update({f: "heldout" for f in split["heldout"]})
    rows = {}
    for (f, s, kind), it in items.items():
        if f not in where or f not in lang:
            raise ValueError(f"feature {f} is not in the split or in the core")
        G = torch.tensor(np.asarray(it["gram"])); e = tc.transfer_error(G, 0.0); n = tc.decomposition_norms(G)
        rows[(f, s, kind)] = {"feature": f, "sentence": s, "kind": kind, "language_index": lang[f], "split": where[f], "cluster": int(s) % int(cluster_mod), "fp_norm": e["fp_norm"], "q_norm": e["q_norm"],
                              "error_norm": e["error_norm"], "cosine": e["cosine"], "norm_ratio": e["norm_ratio"], "C_norm": n["C"], "W_norm": n["W"], "I_norm": n["I"], "total_norm": n["total"],
                              "n_rows": it["n_rows"], **{k: it[k] for k in ("edit_norm_fp", "edit_norm_native", "resid_norm_fp", "resid_norm_q", "amp_fp_mean", "amp_q_mean", "direction_norm") if k in it}}
    return rows


def floors_by_kind(rows_int16, percentile=FLOOR_PERCENTILE):
    """The numerical floor of each direction kind: the `percentile`-th percentile of the INT16 error norm over the CALIBRATION items of that kind."""
    by = {}
    for r in rows_int16.values():
        if r["split"] == "calibration":
            by.setdefault(r["kind"], []).append(r["error_norm"])
    if not by:
        raise ValueError("no calibration item among the INT16 items: the floor cannot be fixed")
    return {k: float(np.percentile(v, percentile)) for k, v in by.items()}


def apply_floor(rows, floors, multiple=ELIGIBILITY_MULTIPLE):
    """Rows with the floor of their kind: eligible = FP effect norm above `multiple` times the floor; normalized error = error norm / (FP effect norm + floor); the error in units of the floor."""
    out = {}
    for k, r in rows.items():
        if r["kind"] not in floors:
            raise ValueError(f"no floor for the direction kind {r['kind']}")
        fl = floors[r["kind"]]
        out[k] = {**r, "floor": fl, "eligible": bool(r["fp_norm"] > multiple * fl), "normalized_error": r["error_norm"] / (r["fp_norm"] + fl), "error_over_floor": r["error_norm"] / fl}
    return out


def coverage(rows):
    """{(kind, split): {"items": n, "eligible": m}}: every item is counted, eligible or not."""
    out = {}
    for r in rows.values():
        c = out.setdefault((r["kind"], r["split"]), {"items": 0, "eligible": 0}); c["items"] += 1; c["eligible"] += int(r["eligible"])
    return out
