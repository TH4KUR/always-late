"""Analysis of the intervention-transfer study (D-088; ledger reports/intervention_transfer/STATUS.md; plan Task 3). Written before the full outputs exist. This first part reads the chunk files of the runner,
builds one row per (feature, context, direction) item with the exact norms of the decomposition (from the stored Gram matrices), fixes the numerical floor from the CALIBRATION features' INT16 items only and applies
the frozen eligibility rule (FP effect norm above five times the floor of its direction kind), and tabulates coverage. Later parts (summaries with crossed bootstrap, the group-level decomposition, the
predictive baselines) are added with their own tests. Part 2 (this file, after `coverage`): the group-level arrays, the token-weighted pooled harms, the decomposition of the net harm and the A3 summary with its bootstrap.

The floor is computed per direction kind (real, matched swapped, random 1 to 3): the frozen definition says 'at the same features, directions and contexts'; the held-out items never enter it (a test says so)."""
import numpy as np
import torch

import phase1_transfer_core as tc

ELIGIBILITY_MULTIPLE = 5
FLOOR_PERCENTILE = 95
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261026
BOOTSTRAP_LEVEL = 95


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


def group_arrays(items, languages, n_sentences, n_controls):
    """The group-level items as arrays. items: {(language code, sentence, set): item} with the sets 'group', 'control0', ... as the runner stores them (the loss changes `dloss` of the five cells in the order of
    `tc.CELLS`). -> {"dloss": [L, S, 1 + n_controls, 5] (set 0 is the group; NaN where an item is missing: a set with no active feature on that sentence was skipped by the runner), "ntok": [L, S] target rows of
    the sentence (the pooling weight), "valid": [L, S] all sets present}. A control whose target-row count differs from the group's, or whose cells are not the five of the study, is an error."""
    names = ["group"] + [f"control{r}" for r in range(n_controls)]
    dloss = np.full((len(languages), n_sentences, len(names), len(tc.CELLS)), np.nan); ntok = np.zeros((len(languages), n_sentences))
    for li, lang in enumerate(languages):
        for j in range(n_sentences):
            for si, name in enumerate(names):
                it = items.get((lang, j, name))
                if it is None:
                    continue
                if list(it["cells"]) != list(tc.CELLS):
                    raise ValueError(f"item {(lang, j, name)} has the cells {it['cells']}, expected {list(tc.CELLS)}")
                dloss[li, j, si] = np.array([np.nan if x is None else x for x in it["dloss"]], dtype=float)
                if ntok[li, j] and ntok[li, j] != it["n_target_rows"]:
                    raise ValueError(f"item {(lang, j, name)}: {it['n_target_rows']} target rows, the sentence has {ntok[li, j]}")
                ntok[li, j] = it["n_target_rows"]
    return {"dloss": dloss, "ntok": ntok, "valid": ~np.isnan(dloss).any(axis=(2, 3))}


def pooled_harms(A, mult=None):
    """The signed loss change of every cell pooled over the sentences of a language by token count (target rows), for the group and for the mean of the control draws, and their difference (the NET harm, the
    selective part of the group effect). A sentence with a missing set is left out of all of them (`n_sentences` counts the others); `mult` [S] are the multiplicities of the source sentences in a bootstrap
    replicate (the same for every language). -> {"group", "control", "net": [L, 5], "n_sentences": [L]}; a language without weight gives NaN."""
    dl, valid = A["dloss"], A["valid"]
    if dl.shape[2] < 2:
        raise ValueError("the net harm needs at least one control draw")
    w = A["ntok"] * valid * (1.0 if mult is None else np.asarray(mult, dtype=float)[None, :])
    clean = np.where(valid[:, :, None, None], dl, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        pooled = (w[:, :, None, None] * clean).sum(1) / w.sum(1)[:, None, None]
    group, control = pooled[:, 0], pooled[:, 1:].mean(axis=1)
    return {"group": group, "control": control, "net": group - control, "n_sentences": valid.sum(1)}


def decompose_net(net):
    """C, W, I and the total of the net harm from its five cells [L, 5] (order of `tc.CELLS`: FP/FP, Q/FP, FP/Q, Q/Q, Q/Q-native), the native-amplitude difference and the native total effect."""
    net = np.asarray(net, dtype=float); ff, qf, fq, qq, qn = (net[:, i] for i in range(5))
    return {"C": qf - ff, "W": fq - ff, "I": qq - qf - fq + ff, "total": qq - ff, "native_minus_fixed": qn - qq, "native_total": qn - ff}


def a3_summary(dec):
    """The pre-registered expectation A3 (the context term dominates the weight term) over languages: the number of languages with |C| > |W|, the medians of |C| and |W| and the median of the paired difference
    |C| - |W|. `holds` is the median of the paired difference being positive (the reading used throughout; the two medians are reported next to it)."""
    c, w = np.abs(np.asarray(dec["C"], dtype=float)), np.abs(np.asarray(dec["W"], dtype=float)); diff = c - w
    return {"n_languages": int(len(c)), "c_gt_w": int((c > w).sum()), "median_abs_c": float(np.median(c)), "median_abs_w": float(np.median(w)), "median_difference": float(np.median(diff)), "holds": bool(np.median(diff) > 0)}


def bootstrap_a3(A, n_boot=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED, level=BOOTSTRAP_LEVEL):
    """The A3 summary with percentile intervals from a bootstrap over the SOURCE sentences: the sentences are drawn with replacement once per replicate and the same multiplicities are used for every language
    (FLORES is parallel, so the languages are not independent samples of the sentences). -> {"point": a3_summary of the data, "median_difference", "median_abs_c", "median_abs_w", "c_gt_w": (low, high),
    "share_holds": fraction of replicates in which A3 holds, "n_boot", "seed", "level"}."""
    rng = np.random.default_rng(seed); S = A["dloss"].shape[1]; reps = []
    for _ in range(n_boot):
        mult = np.bincount(rng.integers(0, S, S), minlength=S)
        reps.append(a3_summary(decompose_net(pooled_harms(A, mult=mult)["net"])))
    lo, hi = (100 - level) / 2, 100 - (100 - level) / 2
    out = {"point": a3_summary(decompose_net(pooled_harms(A)["net"])), "share_holds": float(np.mean([r["holds"] for r in reps])), "n_boot": int(n_boot), "seed": int(seed), "level": level}
    for k in ("median_difference", "median_abs_c", "median_abs_w", "c_gt_w"):
        v = [r[k] for r in reps]; out[k] = (float(np.percentile(v, lo)), float(np.percentile(v, hi)))
    return out
