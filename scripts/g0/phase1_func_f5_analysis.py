"""Analysis of the F5 study (plan Task 14; design docs/superpowers/specs/2026-10-23-feature-function-equivalence-design.md sections 5, 6 and 11): does a feature do the same thing in a quantized model as in the
full-precision (FP) model, and is the survival metric therefore sensible? CPU only, numpy; reads the files of the runner (`phase1_func_f5_run.py`) and the F5 sample, computes the tables, the numerical floor, the
negative control and the validity analysis, and applies the rule S1 to S3 that was proposed before any number (Q23). Exploratory: every number is reported whatever the verdict.

Row spaces: the READ side has one value per sample feature (`stage_a["feat_ids"]`), the WRITE side one per tested feature (`stage_a["write_rows"]`: the sample features with enough context sentences)."""
import csv
import json
from pathlib import Path

import numpy as np

import phase1_func_f5_metrics as met

DUAN_CUTOFF = 0.9             # the survival cutoff of the baseline work: token-level correlation of the activations at least 0.9 (design section 5, S3)
ADEQUATE_MEDIAN = 0.7          # design section 11: the repeat measurement of the same full-precision weights must have a typical W1 of at least about 0.7, or the instrument cannot separate sameness from noise


def assemble(stage_a, sample):
    """Per-feature table aligned to the tested (write) rows: feature id, language, roles, D-035 statuses, FP native rate; plus the counts of sample and excluded features. Raises ValueError when the run and the sample
    disagree (a feature missing from the sample, another language)."""
    langs = list(sample["languages"]); feats = sample["features"]; ids = np.asarray(stage_a["feat_ids"]).astype(int); fl = np.asarray(stage_a["feat_lang"]).astype(int)
    for k, f in zip(ids.tolist(), fl.tolist()):
        if str(k) not in feats:
            raise ValueError(f"feature {k} of the run is not in the sample")
        if feats[str(k)]["language"] != langs[f]:
            raise ValueError(f"feature {k}: the run says {langs[f]}, the sample says {feats[str(k)]['language']}")
    fid = np.asarray(stage_a["write_features"]).astype(int); rows = np.asarray(stage_a["write_rows"]).astype(int)
    if not np.array_equal(ids[rows], fid):
        raise ValueError("write_features and write_rows of the run disagree")
    conds = sorted({c for f in feats.values() for c in f["status"]})
    everything = {"fid": ids, "lang_idx": fl, "roles": [frozenset(feats[str(k)]["roles"]) for k in ids], "status": {c: np.array([feats[str(k)]["status"].get(c, "missing") for k in ids]) for c in conds}}
    return {"all": everything, "fid": fid, "rows": rows, "lang": np.array([feats[str(k)]["language"] for k in fid]), "lang_idx": fl[rows], "roles": [frozenset(feats[str(k)]["roles"]) for k in fid],
            "status": {c: np.array([feats[str(k)]["status"].get(c, "missing") for k in fid]) for c in conds}, "rate_fp": np.array([feats[str(k)]["native"]["FP"][0] for k in fid], dtype=float),
            "fp_established": np.array([bool(feats[str(k)]["fp_established"]) for k in fid]), "n_sample": int(len(ids)), "n_excluded": int(np.asarray(stage_a["excluded"]).sum()), "languages": langs}


def _med(v):
    return float(np.median(v)) if len(v) else float("nan")


def summarize(values, clusters=None, n_boot=2000, seed=0):
    """Counts of all and of defined (non-NaN) values, median, quartiles and mean of the defined ones; with `clusters` (the language of every value) also the 95 % interval of the median from a seeded bootstrap that
    resamples whole languages."""
    v = np.asarray(values, dtype=np.float64); d = v[~np.isnan(v)]
    out = {"n_total": int(v.size), "n_defined": int(d.size), "median": _med(d), "q25": float(np.percentile(d, 25)) if d.size else float("nan"), "q75": float(np.percentile(d, 75)) if d.size else float("nan"),
           "mean": float(d.mean()) if d.size else float("nan")}
    if clusters is not None:
        lo = hi = float("nan")
        if d.size and len(np.unique(np.asarray(clusters)[~np.isnan(v)])) >= 2:
            def fn(idx):
                x = v[idx]; x = x[~np.isnan(x)]
                return _med(x)
            _, lo, hi = met.cluster_bootstrap(clusters, fn, n_boot=n_boot, seed=seed)
        out.update({"median_lo": lo, "median_hi": hi})
    return out


def _reference(states, label, q):
    if label not in states:
        return None
    c = np.asarray(states[label]["write"]["cos"], dtype=np.float64); d = c[~np.isnan(c)]
    return {"label": label, "tau": met.floor_threshold(c, q), "median": float(np.median(d)), "n_defined": int(d.size), "n_undefined": int(c.size - d.size)}


def floor_report(states, q=0.05):
    """The numerical floor of the write side: what 'the same function' looks like when only numerical noise differs. Two noise-only states: FP-rebatched (primary: the same weights, another batching) and INT16 (secondary:
    the weights rounded at the bf16 noise level; INT16 is NOT identical to FP, the floor job of the loss task found per-token differences of about 0.0065 nats against 0.011 for INT8). The floor tau is the lower of
    the q-quantiles of W1 (cosine of the fingerprints) of the two (D-055), both are reported; INT8 is reported (`int8`) but is a real quantization and does not enter the floor. It stands in only when neither noise-only
    state was run. `adequate`: every noise-only state has a median W1 of at least ADEQUATE_MEDIAN (None when none was run)."""
    prim, sec, i8 = _reference(states, "FP_rebatched", q), _reference(states, "INT16", q), _reference(states, "INT8", q)
    noise = [r for r in (prim, sec) if r is not None]
    if not noise and i8 is None:
        raise ValueError("none of FP_rebatched, INT16 and INT8 is available: there is no numerical floor")
    best = min(noise or [i8], key=lambda r: r["tau"])
    return {"q": q, "primary": prim, "secondary": sec, "int8": i8, "tau": best["tau"], "source": best["label"], "adequate": None if not noise else bool(all(r["median"] >= ADEQUATE_MEDIAN for r in noise))}


def negative_control(S, lang_idx, rate_fp):
    """The matched negative control: for every tested feature i the feature j != i of the same language whose FP native rate is nearest on the log scale (ties: the smaller index); the control value is S[i, j], the
    similarity of the FP fingerprint of i and the quantized fingerprint of j (what 'a different function' looks like). No candidate or no positive rate: match -1 and NaN."""
    S = np.asarray(S, dtype=np.float64); lang = np.asarray(lang_idx); rate = np.asarray(rate_fp, dtype=np.float64); n = len(rate)
    match = np.full(n, -1, dtype=int); values = np.full(n, np.nan)
    for l in np.unique(lang):
        idx = np.flatnonzero((lang == l) & (rate > 0))
        for i in idx:
            cand = idx[idx != i]
            if len(cand):
                match[i] = cand[int(np.argmin(np.abs(np.log(rate[cand]) - np.log(rate[i]))))]; values[i] = S[i, match[i]]
    return {"match": match, "values": values}


def group_masks(roles, label):
    """Boolean masks over the features by role: all, k64 (the K64 groups) and, when it has members, lost@<label> and retained@<label> (the F4 subsets of that precision)."""
    out = {"all": np.ones(len(roles), dtype=bool), "k64": np.array(["k64" in r for r in roles], dtype=bool)}
    for kind in ("lost", "retained"):
        m = np.array([f"{kind}@{label}" in r for r in roles], dtype=bool)
        if m.any():
            out[f"{kind}@{label}"] = m
    return out


def share_ci(values, threshold, clusters, n_boot=2000, seed=0):
    """Share of the defined values at or above the threshold with the cluster-bootstrap 95 % interval (whole languages resampled)."""
    v = np.asarray(values, dtype=np.float64); out = dict(met.share_above(v, threshold)); lo = hi = float("nan")
    if (~np.isnan(v)).any() and len(np.unique(np.asarray(clusters))) >= 2:
        _, lo, hi = met.cluster_bootstrap(clusters, lambda idx: met.share_above(v[idx], threshold)["share"], n_boot=n_boot, seed=seed)
    out.update({"lo": lo, "hi": hi})
    return out


def state_table(label, states, T, stage_a, tau, nc, n_boot=2000, seed=0):
    """The tables of one quantized state by role group: read side R1 (token correlation, with the share at or above the baseline cutoff), R2 (firing-set Jaccard), R3 (scale) over every sample feature of the group;
    write side W1 (cosine of the fingerprints, with the share at or above the numerical floor tau and above the 95th percentile of the negative control), W2 (promoted-token overlap), W3 (magnitude and loss ratios),
    the features that do nothing in the quantized model, the share of ablated positions where the feature is inactive in the quantized model, and (group 'all') the retrieval of the run. Undefined values are counted."""
    r, w, fp = states[label]["read"], states[label]["write"], states["FP"]["write"]; A = T["all"]
    ga, gw = group_masks(A["roles"], label), group_masks(T["roles"], label); cl_a, cl_w = A["lang_idx"], T["lang_idx"]
    nc_vals = np.asarray(nc["values"], dtype=np.float64); p95 = float(np.nanpercentile(nc_vals, 95)) if (~np.isnan(nc_vals)).any() else float("nan")
    cos = np.asarray(w["cos"], dtype=np.float64); dq, dfp = np.asarray(w["dnll"], dtype=np.float64), np.asarray(fp["dnll"], dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        loss_ratio = np.where(np.abs(dfp) > 1e-12, dq / dfp, np.nan); inactive = np.where(np.asarray(w["n_pos"]) > 0, 1.0 - np.asarray(w["n_active_q"]) / np.asarray(w["n_pos"]), np.nan)
    out = {}
    for name in gw:
        ma, mw = ga[name], gw[name]; c = lambda m, cl: cl[m]
        row = {"n_features": int(ma.sum()), "n_tested": int(mw.sum()),
               "R1": summarize(np.asarray(r["corr"], dtype=float)[ma], c(ma, cl_a), n_boot, seed), "R1_duan": share_ci(np.asarray(r["corr"], dtype=float)[ma], DUAN_CUTOFF, c(ma, cl_a), n_boot, seed),
               "R2": summarize(np.asarray(r["jaccard"], dtype=float)[ma], c(ma, cl_a), n_boot, seed), "R3": summarize(np.asarray(r["scale"], dtype=float)[ma], c(ma, cl_a), n_boot, seed),
               "W1": summarize(cos[mw], c(mw, cl_w), n_boot, seed), "W1_above_floor": share_ci(cos[mw], tau, c(mw, cl_w), n_boot, seed), "W1_above_control": share_ci(cos[mw], p95, c(mw, cl_w), n_boot, seed),
               "control": {**summarize(nc_vals[mw]), "p95": p95}, "W2": summarize(np.asarray(w["topk_overlap"], dtype=float)[mw]), "W3_magnitude": summarize(np.asarray(w["magnitude_ratio"], dtype=float)[mw]),
               "W3_loss_ratio": summarize(loss_ratio[mw]), "inactive_share": summarize(inactive[mw]), "n_zero_q": int(np.asarray(w["zero_q"])[mw].sum()), "n_zero_fp": int(np.asarray(w["zero_fp"])[mw].sum())}
        row["share_zero_q"] = row["n_zero_q"] / row["n_tested"] if row["n_tested"] else float("nan")
        if name == "all":
            rt = w["retrieval"]; row["retrieval"] = {k: v for k, v in rt.items() if k != "ranks"}
        out[name] = row
    return out


S1_MIN, S2_AUROC, S2_LOWER, S3_MIN = 0.80, 0.60, 0.5, 0.80       # the rule proposed before any number (design section 5, Q23)


def rule_share(share):
    """S1 and S3: at least 80 % of the group has W1 at or above the numerical floor."""
    return "undefined" if share != share else "met" if share >= S1_MIN else "not met"


def rule_auroc(auroc, lo):
    """S2: AUROC of the survival status for the event 'W1 at or above the floor' of at least 0.60 with a lower cluster-bootstrap bound strictly above 0.5."""
    return "undefined" if (auroc != auroc or lo != lo) else "met" if (auroc >= S2_AUROC and lo > S2_LOWER) else "not met"


def overall(verdicts):
    """'supported' when all three statements are met, 'not supported' when any is not met, else 'undetermined'."""
    return "not supported" if "not met" in verdicts else "undetermined" if "undefined" in verdicts else "supported"


def _sign(lo, hi):
    return "positive" if lo > 0 else "negative" if hi < 0 else "inconclusive"


def _boot(clusters, fn, n_boot, seed):
    if len(np.unique(clusters)) < 2:
        return float("nan"), float("nan")
    _, lo, hi = met.cluster_bootstrap(clusters, fn, n_boot=n_boot, seed=seed)
    return lo, hi


def validity(states, T, floor, n_boot=2000, seed=0):
    """Is the survival metric sensible? The three statements S1 to S3 of the rule proposed in advance and the descriptive analysis behind them. tau is the numerical floor. S1: at INT6, among the features labelled
    'retained' by the D-035 status, the share with W1 >= tau. S2: at INT4, in the F4 pairs (roles lost@INT4 and retained@INT4), the AUROC of the status for the event W1 >= tau. S3: at INT5, among the features that
    survive by the baseline cutoff (token correlation >= 0.9), the share with W1 >= tau. If the instrument is inadequate (the repeat measurement of the same weights is not similar) every statement is undefined.
    Also: the 2 x 2 tables of status and of the baseline cutoff against the event, the Spearman correlation of read side and write side, and the paired (by language) median difference of W1 between retained and lost
    F4 features. Undefined values are counted, never zero."""
    tau, rows, lang = floor["tau"], T["rows"], T["lang_idx"]; n_w = len(rows)
    cos = lambda s: np.asarray(states[s]["write"]["cos"], dtype=np.float64) if s in states else np.full(n_w, np.nan)                          # a state that was not run: undefined, not an error
    corr = lambda s: np.asarray(states[s]["read"]["corr"], dtype=np.float64)[rows] if s in states else np.full(n_w, np.nan)
    inadequate = floor["adequate"] is False
    note = ("the instrument is inadequate: the repeat measurement of the same full-precision weights has a median W1 below %.2f, so a similarity at or above the floor does not mean sameness of function" % ADEQUATE_MEDIAN
            if inadequate else "" if floor["adequate"] else "no repeat-measurement pseudo-state: the adequacy of the instrument is not established")
    st = {}
    m = T["status"]["INT6"] == "retained"; r = share_ci(cos("INT6")[m], tau, lang[m], n_boot, seed)
    st["S1"] = {"text": "At INT6, at least 80 % of the features labelled retained by the D-035 status have W1 at or above the numerical floor.", **r, "verdict": rule_share(r["share"])}
    lost, kept = np.array(["lost@INT4" in x for x in T["roles"]]), np.array(["retained@INT4" in x for x in T["roles"]]); pair = lost | kept; c4 = cos("INT4"); d = pair & ~np.isnan(c4)
    score, event, cl = kept[d].astype(float), c4[d] >= tau, lang[d]; a = met.auroc(score, event)
    lo, hi = _boot(cl, lambda idx: met.auroc(score[idx], event[idx])["auroc"], n_boot, seed) if a["n_pos"] and a["n_neg"] else (float("nan"), float("nan"))
    st["S2"] = {"text": "At INT4, in the F4 pairs, the AUROC of the status (retained against lost) for the event W1 >= floor is at least 0.60 with its lower cluster-bootstrap bound above 0.5.", "auroc": a["auroc"], "lo": lo, "hi": hi,
                "n_pos": a["n_pos"], "n_neg": a["n_neg"], "verdict": rule_auroc(a["auroc"], lo)}
    m = corr("INT5") >= DUAN_CUTOFF; r = share_ci(cos("INT5")[m], tau, lang[m], n_boot, seed)
    st["S3"] = {"text": "At INT5, at least 80 % of the features that survive by the baseline cutoff (token correlation >= 0.9) have W1 at or above the numerical floor.", **r, "verdict": rule_share(r["share"])}
    if inadequate:
        for s in st.values():
            s["verdict"] = "undefined"
    conf, spear, paired = {}, {}, {}
    for label in [s for s in ("INT6", "INT5", "INT4") if s in states]:
        c = cos(label); ok = ~np.isnan(c); ev = c >= tau; ret = T["status"][label] == "retained"; duan = corr(label) >= DUAN_CUTOFF
        conf[label] = {"status": {"retained_event": int((ret & ev & ok).sum()), "retained_no_event": int((ret & ~ev & ok).sum()), "lost_event": int((~ret & ev & ok).sum()), "lost_no_event": int((~ret & ~ev & ok).sum())},
                       "duan": {"survives_event": int((duan & ev & ok).sum()), "survives_no_event": int((duan & ~ev & ok).sum()), "fails_event": int((~duan & ev & ok).sum()), "fails_no_event": int((~duan & ~ev & ok).sum())}}
        ra, rl = np.array([f"retained@{label}" in x for x in T["roles"]]), np.array([f"lost@{label}" in x for x in T["roles"]])
        pm = met.paired_median_difference(c[ra], lang[ra], c[rl], lang[rl]); diffs = np.array(list(pm["by_cluster"].values()))
        lo, hi = _boot(np.arange(len(diffs)), lambda idx: _med(diffs[idx]), n_boot, seed) if len(diffs) >= 2 else (float("nan"), float("nan"))
        paired[label] = {"median": pm["median"], "lo": lo, "hi": hi, "sign": _sign(lo, hi) if lo == lo else "undefined", "by_language": {T["languages"][int(k)]: v for k, v in pm["by_cluster"].items()}}
    for label in ("INT8", "INT6", "INT5", "INT4"):
        if label in states:
            x, y = corr(label), cos(label); sp = met.spearman(x, y); lo, hi = _boot(lang, lambda idx: met.spearman(x[idx], y[idx])["rho"], n_boot, seed) if sp["n"] >= 3 else (float("nan"), float("nan"))
            spear[label] = {**sp, "lo": lo, "hi": hi}
    return {"tau": tau, "statements": st, "overall": overall([s["verdict"] for s in st.values()]), "note": note, "confusion": conf, "spearman": spear, "paired": paired}


def check_inputs(stage_a, states):
    """Consistency of the files of a run: full precision present and the identity (W1 = 1 against itself), every array of the right length, S square with a diagonal that equals the stored similarity (to float16
    precision). Raises ValueError naming the problem."""
    if "FP" not in states:
        raise ValueError("the full-precision state is missing: every comparison is against it")
    n_w, n_f = len(stage_a["write_rows"]), len(stage_a["feat_ids"])
    for label, s in states.items():
        w, r = s["write"], s["read"]; S = np.asarray(w["S"])
        if S.shape != (n_w, n_w):
            raise ValueError(f"{label}: the similarity matrix has the shape {S.shape}, expected {(n_w, n_w)}")
        for key in ("cos", "zero_fp", "zero_q", "topk_overlap", "magnitude_ratio", "n_pos", "n_active_q", "dnll"):
            if len(w[key]) != n_w:
                raise ValueError(f"{label}: write-side array {key} has length {len(w[key])}, expected {n_w}")
        for key in ("corr", "jaccard", "scale", "n_active_fp", "n_active_q", "n_both"):
            if len(r[key]) != n_f:
                raise ValueError(f"{label}: read-side array {key} has length {len(r[key])}, expected {n_f}")
        if not np.allclose(np.diagonal(S.astype(np.float64)), np.asarray(w["cos"], dtype=np.float64), atol=2e-3, equal_nan=True):
            raise ValueError(f"{label}: the diagonal of the similarity matrix differs from the stored similarity")
    c = np.asarray(states["FP"]["write"]["cos"], dtype=np.float64)
    if (~np.isnan(c)).any() and np.nanmin(c) < 0.999:
        raise ValueError("full precision against itself is not the identity on the write side (smallest similarity %.4f)" % np.nanmin(c))


def hypotheses(tables, val, floor):
    """The descriptive checks of H5a to H5c (design section 5), derived from the tables: H5a the median token correlation is at least 0.9 at INT8 and INT6 and lower at INT4 than at INT6; H5b the median W1 is
    below the median token correlation at every quantized precision (and the share below the floor is reported); H5c the sign of the paired difference between retained and lost F4 features."""
    g = lambda lab, key: tables[lab]["all"][key]["median"] if lab in tables else float("nan")
    r1 = {lab: g(lab, "R1") for lab in ("INT8", "INT6", "INT5", "INT4") if lab in tables}; w1 = {lab: g(lab, "W1") for lab in ("INT8", "INT6", "INT5", "INT4") if lab in tables}
    h5a = bool(r1.get("INT8", 0) >= 0.9 and r1.get("INT6", 0) >= 0.9 and r1.get("INT4", 1) < r1.get("INT6", 0)) if {"INT8", "INT6", "INT4"} <= set(r1) else None
    below = {lab: bool(w1[lab] < r1[lab]) for lab in w1}
    return {"H5a": {"median_R1": r1, "holds": h5a}, "H5b": {"median_W1": w1, "W1_below_R1": below, "holds": all(below.values()) if below else None,
                                                        "share_below_floor": {lab: 1.0 - tables[lab]["all"]["W1_above_floor"]["share"] for lab in ("INT6", "INT5", "INT4") if lab in tables}},
            "H5c": {"sign": {lab: v["sign"] for lab, v in val["paired"].items()}, "median": {lab: v["median"] for lab, v in val["paired"].items()}}}


def analyse_f5(stage_a, states, sample, meta=None, n_boot=2000, seed=0):
    """The whole analysis of one set of runner files: checks, the table of every state but FP by role group, the floor, the validity analysis with the statements S1 to S3, the descriptive hypotheses and the
    instrument report (sham gate, logit gates, work dtype, completeness; None where the metadata is missing)."""
    check_inputs(stage_a, states); T = assemble(stage_a, sample); floor = floor_report(states); tables = {}
    for label in states:
        if label != "FP":
            nc = negative_control(states[label]["write"]["S"], T["lang_idx"], T["rate_fp"]); tables[label] = state_table(label, states, T, stage_a, floor["tau"], nc, n_boot, seed)
    val = validity(states, T, floor, n_boot, seed); m = meta or {}; sg = stage_a.get("sham_gate"); gates = m.get("gates")
    fl = np.asarray(stage_a["feat_lang"]).astype(int); exc = np.asarray(stage_a["excluded"]).astype(bool)
    instrument = {"sham_gate_exact": None if sg is None else bool(all(x["exact"] for x in sg)), "logit_gates_ok": None if gates is None else bool(all(x.get("ok") for x in gates.values())), "work_dtype": m.get("work_dtype"),
                  "complete": m.get("complete"), "per_cell": m.get("per_cell"), "subset_seed": m.get("subset_seed"), "fp_identity": True, "gates": gates, "state_seconds": m.get("state_seconds"), "runs": m.get("runs")}
    counts = {"n_sample": T["n_sample"], "n_tested": int(len(T["fid"])), "n_excluded": T["n_excluded"], "excluded_by_language": {T["languages"][l]: int(((fl == l) & exc).sum()) for l in range(len(T["languages"]))}}
    return {"instrument": instrument, "counts": counts, "floor": floor, "tables": tables, "validity": val, "hypotheses": hypotheses(tables, val, floor)}


def to_jsonable(x):
    """Plain Python for strict JSON: numpy scalars and arrays converted, NaN and infinities become null."""
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist())
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    return x


CSV_COLUMNS = ["state", "group", "n_features", "n_tested", "R1_median", "R1_median_lo", "R1_median_hi", "R1_duan_share", "R2_median", "R3_median", "W1_median", "W1_median_lo", "W1_median_hi", "W1_above_floor",
               "W1_above_floor_lo", "W1_above_floor_hi", "W1_above_control", "W2_median", "W3_magnitude_median", "W3_loss_ratio_median", "inactive_share_median", "share_zero_q", "n_zero_fp",
               "retrieval_top1", "retrieval_mrr", "retrieval_within_top1", "retrieval_within_chance"]


def _fmt(v):
    return "" if v is None or (isinstance(v, float) and v != v) else ("%.6g" % v if isinstance(v, float) else str(v))


def table_rows(res):
    """One flat row per (state, role group) for the CSV."""
    rows = []
    for state, groups in res["tables"].items():
        for group, g in groups.items():
            rt = g.get("retrieval", {}); wi = rt.get("within", {})
            rows.append({"state": state, "group": group, "n_features": g["n_features"], "n_tested": g["n_tested"], "R1_median": g["R1"]["median"], "R1_median_lo": g["R1"]["median_lo"], "R1_median_hi": g["R1"]["median_hi"],
                         "R1_duan_share": g["R1_duan"]["share"], "R2_median": g["R2"]["median"], "R3_median": g["R3"]["median"], "W1_median": g["W1"]["median"], "W1_median_lo": g["W1"]["median_lo"],
                         "W1_median_hi": g["W1"]["median_hi"], "W1_above_floor": g["W1_above_floor"]["share"], "W1_above_floor_lo": g["W1_above_floor"]["lo"], "W1_above_floor_hi": g["W1_above_floor"]["hi"],
                         "W1_above_control": g["W1_above_control"]["share"], "W2_median": g["W2"]["median"], "W3_magnitude_median": g["W3_magnitude"]["median"], "W3_loss_ratio_median": g["W3_loss_ratio"]["median"],
                         "inactive_share_median": g["inactive_share"]["median"], "share_zero_q": g["share_zero_q"], "n_zero_fp": g["n_zero_fp"], "retrieval_top1": rt.get("top1", ""), "retrieval_mrr": rt.get("mrr", ""),
                         "retrieval_within_top1": wi.get("top1", ""), "retrieval_within_chance": wi.get("chance", "")})
    return rows


def _f3(v):
    """Three decimals for the text files; 'n/a' for an undefined value."""
    return "n/a" if v is None or v != v else f"{v:.3f}"


def _ci(lo, hi):
    return "" if lo != lo or hi != hi else f" [{lo:.3f}, {hi:.3f}]"


def _floor_paragraph(res):
    f = res["floor"]
    return (f"**Numerical floor.** W1 (cosine of the single-feature effect fingerprints) between full precision and the repeat measurement of the same weights (FP-rebatched): median {_f3(f['primary']['median']) if f['primary'] else 'n/a'}, "
            f"5th percentile {_f3(f['primary']['tau']) if f['primary'] else 'n/a'}; between full precision and INT16 (weights rounded at the bf16 noise level): median {_f3(f['secondary']['median']) if f['secondary'] else 'n/a'}, "
            f"5th percentile {_f3(f['secondary']['tau']) if f['secondary'] else 'n/a'}. The numerical floor used below is {f['tau']:.4f} (the lower of the two, from {f['source']}). "
            f"INT8, a real quantization, is reported but not part of the floor: median {_f3(f['int8']['median']) if f['int8'] else 'n/a'}. Instrument adequate: {f['adequate']}.")


def _statement_block(res):
    """The note on the instrument (when there is one), the table of S1 to S3 with values, intervals and verdicts, the overall verdict and what it does and does not mean."""
    v, s, i = res["validity"], res["validity"]["statements"], res["instrument"]; L = []
    if v["note"]:
        L += [f"**Note:** {v['note']}.", ""]
    L += ["| statement | value | 95 % interval (languages resampled) | defined | verdict |", "|---|---|---|---|---|",
          f"| S1 INT6, retained by the D-035 status: share with W1 >= floor (rule: >= 0.80) | {_f3(s['S1']['share'])} | {_ci(s['S1']['lo'], s['S1']['hi']).strip()} | {s['S1']['n_defined']} | {s['S1']['verdict']} |",
          f"| S2 INT4, F4 pairs: AUROC of the status for W1 >= floor (rule: >= 0.60, lower bound > 0.5) | {_f3(s['S2']['auroc'])} | {_ci(s['S2']['lo'], s['S2']['hi']).strip()} | {s['S2']['n_pos']} event, {s['S2']['n_neg']} no event | {s['S2']['verdict']} |",
          f"| S3 INT5, baseline cutoff (correlation >= 0.9): share with W1 >= floor (rule: >= 0.80) | {_f3(s['S3']['share'])} | {_ci(s['S3']['lo'], s['S3']['hi']).strip()} | {s['S3']['n_defined']} | {s['S3']['verdict']} |", "",
          f"**Overall: {v['overall']}.** Work dtype {i['work_dtype']}; sham gate exact {i['sham_gate_exact']}; logit gates ok {i['logit_gates_ok']}; run complete {i['complete']}.", "",
          "If the three statements hold, they hold for this model, this layer, this quantizer and this sample, nothing more. If they do not hold, the survival label describes the input-side agreement of a coordinate system "
          "and no functional claim may be built on it. The study is exploratory: no confirmatory claim is made from it.", ""]
    return L


def statements_text(res):
    """The statement file: the numerical floor, S1 to S3 with values, intervals and verdicts, the overall verdict and what it does and does not mean."""
    return "\n".join(["# F5 statements: is the survival metric sensible? (exploratory)", "",
                      "Rule proposed before any number (design section 5, Q23); every number is reported whatever the verdict (`f5_tables.csv`, `f5_results.json`).", "", _floor_paragraph(res), ""] + _statement_block(res))


LIMITS = ["This study is exploratory: no confirmatory claim is made from it. The rule S1 to S3 was fixed before the first number (Q23); the sample, the contexts and the analysis are one design among several.",
          "One model (Gemma-2-2B), one layer (the layer-12 residual), one SAE (Gemma Scope, 16,384 features), one quantizer (round-to-nearest per output channel). The ablation is a single-feature zero ablation: a test of the necessity of the feature's term, not of its sufficiency.",
          "The effect fingerprint is the immediate next-token effect at the ablated positions plus what flows through later positions of the same sentence; the contexts are the dev sentences in which the feature is most active in full precision.",
          "Features that are active in fewer than 4 candidate dev sentences cannot be tested on the write side; they are counted per language and have read-side values only.",
          "The D-035 statuses were measured on devtest; the read side uses the 300 frozen evaluation sentences of every language (the same devtest sentences of the frozen loss task).",
          "The survival verdict is a verdict on a metric, not on the features."]


def _row(cells):
    return "| " + " | ".join(cells) + " |"


def _iv(m, key="median", lo="median_lo", hi="median_hi"):
    return f"{m[key]:.3f}" + ("" if m.get(lo) != m.get(lo) or m.get(hi) != m.get(hi) else f" [{m[lo]:.3f}, {m[hi]:.3f}]")


def report_text(res, figures=None):
    """The report of one set of runner files: instrument and floor, the main table by state, the F4 subsets by precision, the statements, the descriptive hypotheses, the limits and the figures."""
    f, i, c, v, h = res["floor"], res["instrument"], res["counts"], res["validity"], res["hypotheses"]; L = ["# F5 report: does a feature do the same thing after quantization? (exploratory)", ""]
    L += ["## 1. Instrument and numerical floor", "",
          f"{c['n_sample']} sample features, {c['n_tested']} tested on the write side, {c['n_excluded']} too rare to test (read side only). Work dtype {i['work_dtype']}; sham ablation exact: {i['sham_gate_exact']}; logit gates ok: {i['logit_gates_ok']}; "
          f"run complete: {i['complete']}." + (f" Stratified subset: a per-cell cap of {i['per_cell']} features per language and role (seed {i['subset_seed']})." if i.get("per_cell") else ""), ""]
    L += [_floor_paragraph(res), ""]
    L += ["## 2. Main table (all tested features, by state)", "",
          "R1 token correlation of the activations with full precision (the survival basis); R2 firing-set Jaccard; R3 scale ratio; W1 cosine of the single-feature effect fingerprints; the share at or above the numerical floor and above the "
          "95th percentile of the matched different-feature control; 'does nothing' = the feature has no effect in the quantized model; top-1 = retrieval of the feature's own identity (within language, chance in brackets). Intervals: 95 %, languages resampled.", "",
          _row(["state", "tested", "R1 median", "R1 >= 0.9", "R2", "R3", "W1 median", "W1 >= floor", "W1 >= control", "does nothing", "top-1", "within top-1 (chance)"]), _row(["---"] * 12)]
    for s, groups in res["tables"].items():
        g = groups["all"]; rt = g.get("retrieval", {}); wi = rt.get("within", {})
        L.append(_row([s, str(g["n_tested"]), _iv(g["R1"]), f"{g['R1_duan']['share']:.3f}", f"{g['R2']['median']:.3f}", f"{g['R3']['median']:.3f}", _iv(g["W1"]), _iv(g["W1_above_floor"], "share", "lo", "hi"),
                       f"{g['W1_above_control']['share']:.3f}", f"{g['n_zero_q']} ({g['share_zero_q']:.3f})", f"{rt.get('top1', float('nan')):.3f}", f"{wi.get('top1', float('nan')):.3f} ({wi.get('chance', float('nan')):.3f})"]))
    L += ["", "## 3. Features whose association was lost and retained (F4 subsets), by precision", "",
          _row(["group", "tested", "W1 median", "W1 >= floor", "R1 median", "R1 >= 0.9"]), _row(["---"] * 6)]
    for lab in ("INT6", "INT5", "INT4"):
        for kind in ("lost", "retained"):
            g = res["tables"].get(lab, {}).get(f"{kind}@{lab}")
            if g:
                L.append(_row([f"{kind}@{lab}", str(g["n_tested"]), _iv(g["W1"]), _iv(g["W1_above_floor"], "share", "lo", "hi"), f"{g['R1']['median']:.3f}", f"{g['R1_duan']['share']:.3f}"]))
    L += ["", "## 4. Statements: is the survival metric sensible?", ""] + _statement_block(res)
    fd = lambda d: ", ".join(f"{k} {x:.3f}" for k, x in d.items()) or "n/a"
    L += ["## 5. Descriptive hypotheses (H5a to H5c)", "",
          f"- H5a (the median token correlation is at least 0.9 at INT8 and INT6 and lower at INT4 than at INT6): {h['H5a']['holds']}; medians {fd(h['H5a']['median_R1'])}.",
          f"- H5b (the median W1 is below the median token correlation at every quantized precision): {h['H5b']['holds']}; median W1 {fd(h['H5b']['median_W1'])}; share below the floor {fd(h['H5b']['share_below_floor'])}.",
          f"- H5c (retained features keep more of their effect than lost ones, paired by language): signs {', '.join(f'{k} {x}' for k, x in h['H5c']['sign'].items()) or 'n/a'}; medians {fd(h['H5c']['median'])}.", "",
          "## 6. Limits", ""] + [f"- {x}" for x in LIMITS] + ([f"- The sample is a stratified subset of the F5 sample (a per-cell cap of {i['per_cell']} per language and role, seed {i['subset_seed']}), chosen for cost before any result."] if i.get("per_cell") else [])
    L += ["", "## 7. Figures", ""] + ([f"- ![{n}]({n})" for n in figures] if figures else ["- none"])
    return "\n".join(L) + "\n"


def write_outputs(res, out_dir, figures=None):
    """f5_results.json (strict JSON), f5_tables.csv (one row per state and role group) and statements.md in out_dir."""
    d = Path(out_dir); d.mkdir(parents=True, exist_ok=True)
    (d / "f5_results.json").write_text(json.dumps(to_jsonable(res), indent=1, allow_nan=False) + "\n")
    with open(d / "f5_tables.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(CSV_COLUMNS)
        for r in table_rows(res):
            w.writerow([_fmt(r[c]) if c not in ("state", "group") else r[c] for c in CSV_COLUMNS])
    (d / "statements.md").write_text(statements_text(res)); (d / "report.md").write_text(report_text(res, figures))


def _numpy(x):
    import torch
    if isinstance(x, torch.Tensor):
        return x.numpy()
    if isinstance(x, dict):
        return {k: _numpy(v) for k, v in x.items()}
    return x


def load_run(directory):
    """(stage_a, states, meta) of one runner job directory, tensors converted to numpy (S stays float16)."""
    import torch
    d = Path(directory); sa = _numpy(torch.load(d / "f5_stage_a.pt", weights_only=False)); states = {}
    for f in sorted(d.glob("f5_state_*.pt")):
        res = _numpy(torch.load(f, weights_only=False)); states[res.pop("label")] = res
    mp = d / "meta_f5.json"
    return sa, states, (json.loads(mp.read_text()) if mp.exists() else None)


def load_runs(directories):
    """Several jobs merged: they must agree on the tested features and their contexts (each job repeats stage A and FP); FP is taken from the first job; any other state may appear in one job only."""
    sa0, states, metas, sham = None, {}, [], []
    for k, d in enumerate(directories):
        sa, st, meta = load_run(d); metas.append(meta or {}); sham += list(sa.get("sham_gate", []))
        if sa0 is None:
            sa0 = sa
        else:
            for key in ("feat_ids", "write_rows", "write_features", "contexts"):
                if not np.array_equal(np.asarray(sa0[key]), np.asarray(sa[key])):
                    raise ValueError(f"{d}: stage A differs from the first job in {key}: the jobs did not test the same features in the same contexts")
        for label, res in st.items():
            if label in states and label != "FP":
                raise ValueError(f"{d}: the state {label} is in more than one job")
            states.setdefault(label, res)
    sa0["sham_gate"] = sham
    dtypes = {m.get("work_dtype") for m in metas}; gates = {}
    for k, m in enumerate(metas):
        gates.update({f"run{k}:{n}": g for n, g in (m.get("gates") or {}).items()})
    return sa0, states, {"runs": [str(d) for d in directories], "complete": all(m.get("complete") for m in metas), "work_dtype": dtypes.pop() if len(dtypes) == 1 else "mixed", "gates": gates or None,
                         "per_cell": next((m.get("per_cell") for m in metas if m.get("per_cell") is not None), None), "subset_seed": next((m.get("subset_seed") for m in metas if m.get("subset_seed") is not None), None),
                         "state_seconds": {k: v for m in metas for k, v in (m.get("state_seconds") or {}).items()}}


def main(argv=None):
    import argparse
    ROOT = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="F5 analysis: tables, numerical floor, negative control and the validity statements S1 to S3 from the files of the runner (one or more job directories).")
    ap.add_argument("--run", action="append", required=True, help="a job directory of phase1_func_f5_run.py (repeat for several jobs)"); ap.add_argument("--sample", default=str(ROOT / "reports/functional_validation/f5/f5_sample.json"))
    ap.add_argument("--out", required=True); ap.add_argument("--n-boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--figures", nargs="*", default=[], help="paths of figures (relative to the report) to link in report.md")
    a = ap.parse_args(argv)
    sa, states, meta = load_runs(a.run); sample = json.loads(Path(a.sample).read_text())
    res = analyse_f5(sa, states, sample, meta=meta, n_boot=a.n_boot, seed=a.seed); write_outputs(res, a.out, a.figures)
    print("wrote", a.out, "| states", sorted(res["tables"]), "| floor", round(res["floor"]["tau"], 4), "from", res["floor"]["source"], "| overall:", res["validity"]["overall"])


if __name__ == "__main__":
    main()
