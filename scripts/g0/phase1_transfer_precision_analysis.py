"""Table, summaries and pre-declared readings of the paired arithmetic-precision audit (protocol revision_R5_precision_audit.json; implementation plan docs/superpowers/plans/2026-10-27-precision-audit-task2-implementation.md,
step 4b; POST HOC).

Per-item rows (one per item and state, with the quantities of `phase1_transfer_precision_cross.item_metrics` and the in-run signed loss changes) get the old-floor and mode-specific-floor quantities, the paired
differences fp32 minus bf16 and the repeated-batching control; the summaries are medians with the crossed bootstrap over features and clusters; `readings` applies the rules A to D of the protocol to a summary.
Populations: all real-direction items (primary), the real-direction items eligible under the old fixed floors in BOTH modes, and all matched-swapped items.
"""
import csv
from pathlib import Path

import numpy as np

import phase1_transfer_analysis as ta
import phase1_transfer_precision_cross as px

MODES = px.MODES
QUANTITIES = (tuple(f"{q}_{m}" for m in MODES for q in ("fp_norm", "qq_norm", "err", "cos", "nerr_old", "eof_old", "nerr_mode", "eof_mode", "dloss_FF", "dloss_QQ"))
              + ("arith_ff", "arith_qq", "arith_qq_over_err_bf16", "arith_qq_over_err_fp32", "arith_ff_over_fp_norm_bf16", "cos_arith_vs_err", "d_err", "d_cos", "d_nerr_old", "d_dloss_FF", "d_dloss_QQ")
              + tuple(f"batch_{c}_{m}" for c in ("FF", "QQ") for m in MODES))
MARGIN_COS, MARGIN_NERR, FACTOR_REFERENCE, BAND_SMALL, BAND_COMPARABLE, COS_CEILING = 0.10, 0.15, 2.0, 0.2, 0.5, 0.9
STATUS_CHANGES, STATUS_SMALL, STATUS_NONE = "changes", "separated but below the margin", "not separated from zero"


def _diff(a, b):
    return None if a is None or b is None else b - a


def add_paired(r):
    """The row with the paired differences fp32 minus bf16: d_err, d_cos, d_nerr_old, d_dloss_FF, d_dloss_QQ (None if either side is missing)."""
    out = dict(r)
    for name, base in (("d_err", "err"), ("d_cos", "cos"), ("d_nerr_old", "nerr_old"), ("d_dloss_FF", "dloss_FF"), ("d_dloss_QQ", "dloss_QQ")):
        out[name] = _diff(r.get(f"{base}_bf16"), r.get(f"{base}_fp32"))
    return out


def finalize_rows(rows, old_floors):
    """Rows of all states -> rows with the mode-specific floors (the 95th percentile of the INT16 error per kind and mode, from the INT16 rows ONLY), the old-floor and mode-floor quantities and the paired
    differences. A kind with rows but no INT16 row is a ValueError."""
    kinds = {r["kind"] for r in rows}; r16 = [r for r in rows if r["state"] == "INT16"]
    missing = sorted(k for k in kinds if not any(r["kind"] == k for r in r16))
    if missing:
        raise ValueError(f"no INT16 item of the kind {missing[0]}: the mode-specific floor cannot be calibrated")
    floors = px.mode_floors(r16)
    return [add_paired(px.with_floors(r, old_floors, floors)) for r in rows]


def attach_batching(rows, control):
    """Attach the repeated-batching control (`px.batching_difference` per mode: control = {mode: {item key: {cell: norm}}}) to the INT16 rows as batch_FF_m and batch_QQ_m; every other row gets None."""
    out = []
    for r in rows:
        q = dict(r); key = (r["feature"], r["sentence"], r["kind"])
        for m in MODES:
            d = control.get(m, {}).get(key, {}) if r["state"] == "INT16" else {}
            q[f"batch_FF_{m}"] = d.get("FP/FP"); q[f"batch_QQ_{m}"] = d.get("Q/Q")
        out.append(q)
    return out


def _populations(rows):
    return {"all_real": [r for r in rows if r["kind"] == "real"], "eligible_real_old": [r for r in rows if r["kind"] == "real" and r.get("eligible_old_bf16") and r.get("eligible_old_fp32")],
            "all_matched_swapped": [r for r in rows if r["kind"] == "matched_swapped"]}


def summarize(rows, n_boot=ta.BOOTSTRAP_REPLICATES, seed=ta.BOOTSTRAP_SEED):
    """{(population, kind, state, quantity): {point, low, high, n, n_missing, n_features, n_clusters}}: the median of the item values with the crossed bootstrap over features and clusters; None values are left
    out and counted; a quantity that no row of the group has is not listed."""
    out = {}
    for pop, members in _populations(rows).items():
        for state in sorted({r["state"] for r in members}):
            sel = [r for r in members if r["state"] == state]
            if not sel:
                continue
            kind = sel[0]["kind"]
            for q in QUANTITIES:
                if not any(r.get(q) is not None for r in sel):
                    continue
                v = np.array([np.nan if r.get(q) is None else float(r[q]) for r in sel], dtype=float); b = ta.crossed_bootstrap(v, [r["feature"] for r in sel], [r["cluster"] for r in sel], "median", n_boot, seed)
                out[(pop, kind, state, q)] = {k: b[k] for k in ("point", "low", "high", "n", "n_missing", "n_features", "n_clusters")}
    return out


def coverage(rows):
    """{(kind, state, floor definition 'old'|'mode', mode): {"eligible", "items"}}: the items whose FP effect norm is above five floors."""
    out = {}
    for r in rows:
        for tag in ("old", "mode"):
            for m in MODES:
                c = out.setdefault((r["kind"], r["state"], tag, m), {"eligible": 0, "items": 0}); c["items"] += 1; c["eligible"] += int(bool(r.get(f"eligible_{tag}_{m}")))
    return out


def _status(e, margin):
    if e is None:
        return None
    if e["low"] > 0 or e["high"] < 0:
        return STATUS_CHANGES if abs(e["point"]) >= margin else STATUS_SMALL
    return STATUS_NONE


def readings(summary, population="all_real", kind="real"):
    """The pre-declared readings A to D of the protocol for one population. A: the INT4 description changes with the arithmetic if the paired median difference (fp32 minus bf16) of the cosine (margin 0.10) or of
    the normalized error under the old floors (margin 0.15) has an interval that excludes zero and is at least the margin in size; B: the drift persists in a mode if the median INT4 error in mode-specific floors has an
    interval lower end above 1 and the median INT4 cosine is below 0.9 (both modes needed); C: the rounding reference depends on the arithmetic if the median INT16 errors differ by a factor of at least 2;
    D: the median ratio of the arithmetic change of Q/Q to the INT4 transfer error is small below 0.2 and comparable above 0.5. A missing quantity gives None."""
    g = lambda state, q: summary.get((population, kind, state, q))
    a_cos, a_err = _status(g("INT4", "d_cos"), MARGIN_COS), _status(g("INT4", "d_nerr_old"), MARGIN_NERR); out = {"A": {"cosine": a_cos, "normalized_error": a_err, "any_change": STATUS_CHANGES in (a_cos, a_err)}}
    b = {}
    for m in MODES:
        e, c = g("INT4", f"eof_mode_{m}"), g("INT4", f"cos_{m}"); b[m] = None if e is None or c is None else bool(e["low"] > 1.0 and c["point"] < COS_CEILING)
    out["B"] = {**b, "persists_in_both_modes": all(v is True for v in b.values())}
    e16 = {m: g("INT16", f"err_{m}") for m in MODES}
    if any(v is None for v in e16.values()):
        out["C"] = {"ratio": None, "depends": None}
    else:
        hi, lo = max(e16["bf16"]["point"], e16["fp32"]["point"]), min(e16["bf16"]["point"], e16["fp32"]["point"]); ratio = hi / lo if lo > 0 else float("inf"); out["C"] = {"ratio": ratio, "depends": bool(ratio >= FACTOR_REFERENCE)}
    d = g("INT4", "arith_qq_over_err_bf16"); out["D"] = {"value": None, "band": None} if d is None else {"value": d["point"], "band": "small" if d["point"] < BAND_SMALL else ("comparable" if d["point"] > BAND_COMPARABLE else "intermediate")}
    return out


def controls_summary(rows):
    """Control C1 on the INT16 rows: the median repeated-batching difference of the FP/FP cell per mode, its median ratio to the FP effect norm, the median arithmetic change of FP/FP and their ratio."""
    r16 = [r for r in rows if r["state"] == "INT16" and r.get("batch_FF_bf16") is not None]; med = lambda xs: float(np.median(xs)); out = {"n": len(r16)}
    if not r16:
        return out
    for m in MODES:
        out[f"median_batch_FF_{m}"] = med([r[f"batch_FF_{m}"] for r in r16]); out[f"median_batch_FF_{m}_over_fp_norm"] = med([r[f"batch_FF_{m}"] / r[f"fp_norm_{m}"] for r in r16 if r[f"fp_norm_{m}"] > 0])
    out["median_arith_ff"] = med([r["arith_ff"] for r in r16])
    for m in MODES:
        out[f"arith_ff_over_batch_FF_{m}"] = None if out[f"median_batch_FF_{m}"] == 0 else out["median_arith_ff"] / out[f"median_batch_FF_{m}"]
    return out


def write_tables(rows, summary, directory):
    """`per_item.csv` (every item row, the union of the keys in first-seen order) and `summary.csv` (one row per summarized quantity)."""
    d = Path(directory); d.mkdir(parents=True, exist_ok=True); keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(d / "per_item.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    cols = ["population", "kind", "state", "quantity", "point", "low", "high", "n", "n_missing", "n_features", "n_clusters"]
    with open(d / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for (pop, kind, state, q), v in sorted(summary.items()):
            w.writerow({"population": pop, "kind": kind, "state": state, "quantity": q, **v})
