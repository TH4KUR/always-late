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


def _quad(G, coef):
    """Norm of the combination `coef` of the five cell vectors, from the Gram matrix."""
    c = np.asarray(coef, dtype=np.float64); return float(np.sqrt(max(c @ np.asarray(G, dtype=np.float64)[:5, :5] @ c, 0.0)))


def _rescaling_residual(G, cell):
    """Norm of the part of the effect of `cell` (index into tc.CELLS) that is orthogonal to the FP effect: the error that remains after the best rescaling of the FP effect (None when the FP effect is zero)."""
    G = np.asarray(G, dtype=np.float64); ff = G[0, 0]
    return None if ff <= 0 else float(np.sqrt(max(G[cell, cell] - G[0, cell] ** 2 / ff, 0.0)))


def item_table(items, core, split, cluster_mod):
    """One row per item {(feature, sentence, kind): row}: the language index of the feature, its calibration or held-out membership, the source cluster (dev index modulo `cluster_mod`), the norms of the FP
    effect, the Q/Q effect and the error, the cosine and norm ratio, the norms of C, W, I and the total, the error of the native-amplitude cell, the errors that remain after the best rescaling of the FP effect
    (fixed and native amplitude; H1), and the stored per-item scalars. Feature-level items with the amplitude vectors (amendment A10) add the activation cosine (cosine of the FP and the quantized amplitude
    vectors over the edited positions, 0 when the feature does not fire in the quantized state) and the firing share; items with the gradient scalars (amendment A9) add the shifts of the first-order loss
    effect (quantized minus FP, for the fixed and for the native edit)."""
    lang = dict(zip(core["feature_ids"], core["feature_language_index"])); where = {f: "calibration" for f in split["calibration"]}; where.update({f: "heldout" for f in split["heldout"]})
    rows = {}
    for (f, s, kind), it in items.items():
        if f not in where or f not in lang:
            raise ValueError(f"feature {f} is not in the split or in the core")
        G = torch.tensor(np.asarray(it["gram"])); e = tc.transfer_error(G, 1.0); n = tc.decomposition_norms(G); g = np.asarray(it["gram"])         # the floor argument only matters for the normalized error, which is formed in apply_floor
        r = {"feature": f, "sentence": s, "kind": kind, "language_index": lang[f], "split": where[f], "cluster": int(s) % int(cluster_mod), "fp_norm": e["fp_norm"], "q_norm": e["q_norm"],
             "error_norm": e["error_norm"], "cosine": e["cosine"], "norm_ratio": e["norm_ratio"], "C_norm": n["C"], "W_norm": n["W"], "I_norm": n["I"], "total_norm": n["total"],
             "native_error_norm": _quad(g, [-1, 0, 0, 0, 1]), "scalar_residual_norm": _rescaling_residual(g, 3), "native_scalar_residual_norm": _rescaling_residual(g, 4),
             "n_rows": it["n_rows"], "n_active_q": it.get("n_active_q"), "n_positions": it.get("n_positions"),
             **{k: it[k] for k in ("edit_norm_fp", "edit_norm_native", "resid_norm_fp", "resid_norm_q", "amp_fp_mean", "amp_q_mean", "direction_norm", "g_pred_fp", "g_pred_q_fixed", "g_pred_q_native", "g_norm_fp", "g_norm_q") if k in it}}
        if it.get("n_active_q") is not None and it.get("n_positions"):
            r["firing"] = it["n_active_q"] / it["n_positions"]
        if "amp_fp" in it and "amp_q" in it:
            a, b = np.asarray(it["amp_fp"], dtype=np.float64), np.asarray(it["amp_q"], dtype=np.float64); na, nb = np.linalg.norm(a), np.linalg.norm(b)
            r["act_cos"] = float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0
        if "g_pred_fp" in it and "g_pred_q_native" in it and "g_pred_q_fixed" in it:
            r["g_shift_native"] = it["g_pred_q_native"] - it["g_pred_fp"]; r["g_shift_fixed"] = it["g_pred_q_fixed"] - it["g_pred_fp"]
        rows[(f, s, kind)] = r
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


def weighted_median(values, weights):
    """The lower weighted median: the smallest value whose cumulative weight reaches half of the total (NaN for no weight). The same definition serves the point estimate (unit weights) and every bootstrap replicate."""
    v, w = np.asarray(values, dtype=float), np.asarray(weights, dtype=float); total = w.sum()
    if v.size == 0 or total <= 0:
        return float("nan")
    order = np.argsort(v, kind="stable"); return float(v[order][np.searchsorted(np.cumsum(w[order]), 0.5 * total, side="left")])


def bootstrap_multiplicities(n_features, n_clusters, n_boot, seed):
    """The multiplicities of a crossed bootstrap: [n_boot, n_features] and [n_boot, n_clusters] draws with replacement (each row sums to the number of units), features and source-sentence clusters independently
    (the 'pigeonhole' scheme: a feature-context row has the weight of its feature times the weight of its cluster, so reused features and parallel translations of one source sentence are not iid rows)."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_features, np.full(n_features, 1.0 / n_features), size=n_boot), rng.multinomial(n_clusters, np.full(n_clusters, 1.0 / n_clusters), size=n_boot)


def replicate_stats(values, feature_idx, cluster_idx, mult_features, mult_clusters, stat):
    """The statistic ('mean' or 'median', weighted) of the values in every replicate; the weight of a row is the multiplicity of its feature times that of its cluster."""
    values = np.asarray(values, dtype=float); out = np.empty(len(mult_features))
    order = np.argsort(values, kind="stable") if stat == "median" else None
    for b in range(len(mult_features)):
        w = mult_features[b][feature_idx] * mult_clusters[b][cluster_idx]
        if stat == "mean":
            out[b] = (w * values).sum() / w.sum() if w.sum() > 0 else np.nan
        elif stat == "median":
            total = w.sum(); out[b] = values[order][np.searchsorted(np.cumsum(w[order]), 0.5 * total, side="left")] if total > 0 else np.nan
        else:
            raise ValueError(f"unknown statistic {stat}")
    return out


def crossed_bootstrap(values, feature_idx, cluster_idx, stat="median", n_boot=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED, level=BOOTSTRAP_LEVEL):
    """A statistic of per-row values with a percentile interval from the crossed bootstrap over features and source-sentence clusters. NaN values are dropped and counted. -> {"point", "low", "high", "n", "n_missing",
    "n_features", "n_clusters", "stat", "n_boot", "seed", "level"}; no value at all gives NaNs. The multiplicities are drawn for the features and clusters that occur among the kept rows."""
    v = np.asarray(values, dtype=float); keep = ~np.isnan(v); out = {"n": int(keep.sum()), "n_missing": int((~keep).sum()), "stat": stat, "n_boot": int(n_boot), "seed": int(seed), "level": level}
    if not keep.any():
        return {**out, "point": float("nan"), "low": float("nan"), "high": float("nan"), "n_features": 0, "n_clusters": 0}
    v = v[keep]; _, fi = np.unique(np.asarray(feature_idx)[keep], return_inverse=True); _, ci = np.unique(np.asarray(cluster_idx)[keep], return_inverse=True)
    nf, nc = int(fi.max()) + 1, int(ci.max()) + 1; mf, mc = bootstrap_multiplicities(nf, nc, n_boot, seed)
    point = float(v.mean()) if stat == "mean" else weighted_median(v, np.ones(len(v))); reps = replicate_stats(v, fi, ci, mf, mc, stat)
    lo, hi = (100 - level) / 2, 100 - (100 - level) / 2
    return {**out, "point": point, "low": float(np.nanpercentile(reps, lo)), "high": float(np.nanpercentile(reps, hi)), "n_features": nf, "n_clusters": nc}
