"""Common-support analysis of the real direction against its control directions (revision R3, task 1 of docs/superpowers/plans/2026-10-26-publication-recovery-after-results.md; POST HOC; protocol fixed in
reports/intervention_transfer/revision_R3_common_support.json before the code and the run).

The original specificity table compares the transfer error of the eligible real items with the mean of their control directions, which are not eligibility-filtered and have a different full-precision (FP) effect
size (the median real/swapped ratio of the FP effect norms among jointly eligible pairs is about 2). Here the pairs of the real item and ONE control item of the same feature and context are nested into S0 (all
pairs), S1 (both items eligible) and S2 (S1 with the FP effect-norm ratio rho = fp_norm(real) / fp_norm(control) inside the diagnostic overlap band [0.8, 1.25]). The coverage (counts, features, clusters, the
quartiles of rho and of the ratios of the FP edit norms and FP residual norms) is reported first, a contrast only if the set has at least 100 pairs and 20 features. No stored effect vector is rescaled.

    python phase1_transfer_support.py --main <analysis_main> --external <analysis_external> --out <dir> [--states INT16 INT6 INT4]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .tables import sha256_file as _sha, write_csv

import phase1_transfer_analysis as ta
from . import cross_corpus as cc

KINDS = tuple(ta.CONTROL_KINDS)
BAND = (0.8, 1.25)
MIN_PAIRS = 100
MIN_FEATURES = 20
CORPORA = ("flores_main", "wikipedia_external")
SETS = ("S0", "S1", "S2")


def _num(x):
    return float("nan") if x is None else float(x)


def pairs(rows, kind):
    """The pairs of the real item and the control item of `kind` of the same (feature, context), sorted by (feature, sentence): arrays of the feature, cluster, FP effect norm, eligibility, FP edit norm, FP residual
    norm, normalized error, cosine (NaN if missing) and error norm of both items, and `n_dropped`: the existing pairs without a finite positive FP effect norm in both items (never filled)."""
    keys = sorted(k for k in rows if k[2] == "real" and (k[0], k[1], kind) in rows); out = {n: [] for n in ("feature", "cluster", "fp_real", "fp_ctrl", "elig_real", "elig_ctrl", "edit_real", "edit_ctrl", "resid_real", "resid_ctrl",
                                                                                                      "ne_real", "ne_ctrl", "cos_real", "cos_ctrl", "err_real", "err_ctrl")}; dropped = 0
    for k in keys:
        a, b = rows[k], rows[(k[0], k[1], kind)]; fa, fb = _num(a["fp_norm"]), _num(b["fp_norm"])
        if not (np.isfinite(fa) and fa > 0 and np.isfinite(fb) and fb > 0):
            dropped += 1; continue
        out["feature"].append(k[0]); out["cluster"].append(a["cluster"]); out["fp_real"].append(fa); out["fp_ctrl"].append(fb); out["elig_real"].append(bool(a["eligible"])); out["elig_ctrl"].append(bool(b["eligible"]))
        for name, field in (("edit", "edit_norm_fp"), ("resid", "resid_norm_fp"), ("ne", "normalized_error"), ("cos", "cosine"), ("err", "error_norm")):
            out[f"{name}_real"].append(_num(a.get(field))); out[f"{name}_ctrl"].append(_num(b.get(field)))
    res = {n: np.array(v, dtype=(bool if n.startswith("elig") else (np.int64 if n in ("feature", "cluster") else float))) for n, v in out.items()}; res["n_dropped"] = dropped; return res


def sets(p, lo=BAND[0], hi=BAND[1]):
    """The nested sets as masks over the pairs: S0 all, S1 both eligible, S2 S1 with the FP effect-norm ratio inside [lo, hi] (inclusive); also the ratio `rho`."""
    rho = p["fp_real"] / p["fp_ctrl"]; s1 = p["elig_real"] & p["elig_ctrl"]; return {"S0": np.ones(len(rho), dtype=bool), "S1": s1, "S2": s1 & (rho >= lo) & (rho <= hi), "rho": rho}


def _quartiles(x):
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    return [None, None, None] if len(x) == 0 else [float(v) for v in np.percentile(x, [25, 50, 75])]


def coverage(p, s):
    """Per set: the number of pairs, features and clusters and the quartiles of the three ratios real/control (FP effect norm `rho`, FP edit norm, FP residual norm); for S1 the share inside the band."""
    out = {"n_dropped": p["n_dropped"]}
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = {"rho": p["fp_real"] / p["fp_ctrl"], "edit_ratio": p["edit_real"] / p["edit_ctrl"], "resid_ratio": p["resid_real"] / p["resid_ctrl"]}
    for name in SETS:
        m = s[name]; d = {"n_pairs": int(m.sum()), "n_features": int(len(np.unique(p["feature"][m]))), "n_clusters": int(len(np.unique(p["cluster"][m])))}
        for key, ratio in ratios.items():
            q = _quartiles(ratio[m]); d.update({f"{key}_p25": q[0], f"{key}_p50": q[1], f"{key}_p75": q[2]})          # a ratio that is not finite (zero or missing control value) is left out
        if name == "S1":
            d["share_in_band"] = float(s["S2"].sum() / m.sum()) if m.sum() > 0 else None
        out[name] = d
    return out


def all_four(rows):
    """The number of eligible real items whose four control items (matched swapped, random 1 to 3) all exist and are eligible: the joint support of all controls (counted only, no contrast)."""
    n = 0
    for k, r in rows.items():
        if k[2] == "real" and r["eligible"] and all((k[0], k[1], c) in rows and rows[(k[0], k[1], c)]["eligible"] for c in KINDS):
            n += 1
    return n


def contrast(p, mask, n_boot=ta.BOOTSTRAP_REPLICATES, seed=ta.BOOTSTRAP_SEED, min_pairs=MIN_PAIRS, min_features=MIN_FEATURES):
    """The support-matched contrast on the pairs of the mask: the median over pairs of the normalized-error gap (real minus control), the cosine gap, log2 of the error-norm ratio and the share of pairs in which the
    real direction has the larger normalized error, each with the crossed bootstrap over features and clusters (NaN values left out and counted). 'insufficient support' (with the counts) below the minimum coverage."""
    f, c = p["feature"][mask], p["cluster"][mask]; out = {"n": int(mask.sum()), "n_features": int(len(np.unique(f))), "n_clusters": int(len(np.unique(c)))}
    if out["n"] < min_pairs or out["n_features"] < min_features:
        return {**out, "status": "insufficient support"}
    ne_r, ne_c, er, ec = p["ne_real"][mask], p["ne_ctrl"][mask], p["err_real"][mask], p["err_ctrl"][mask]
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.where((er > 0) & (ec > 0), np.log2(er / ec), np.nan); larger = np.where(np.isfinite(ne_r) & np.isfinite(ne_c), (ne_r > ne_c).astype(float), np.nan)
    stats = {"ne_gap": (ne_r - ne_c, "median"), "cos_gap": (p["cos_real"][mask] - p["cos_ctrl"][mask], "median"), "log2_error_ratio": (lr, "median"), "share_real_larger": (larger, "mean")}
    return {**out, "status": "fitted", **{k: ta.crossed_bootstrap(v, f, c, stat, n_boot, seed) for k, (v, stat) in stats.items()}}


def _same(pa, pb):
    return all(np.array_equal(pa[k], pb[k], equal_nan=True) for k in ("feature", "fp_real", "fp_ctrl", "elig_real", "elig_ctrl", "edit_real", "edit_ctrl", "resid_real", "resid_ctrl"))


def run(main_dir, external_dir, states, out_dir, n_boot=ta.BOOTSTRAP_REPLICATES, seed=ta.BOOTSTRAP_SEED):
    """Coverage once per corpus and control kind (from the first state table; the sets depend on FP quantities only, which is checked across the state tables and reported), then the contrast per corpus, state, control
    kind and set (S1, S2). Writes `support_coverage.csv`, `support_contrast.csv` and `support.json`. -> (coverage rows, contrast rows)."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True); cov_rows, con_rows, same, inputs = [], [], {}, {}
    for corpus, directory in zip(CORPORA, (main_dir, external_dir)):
        tabs = {s: cc.rows_from_csv(Path(directory) / f"per_context_{s}.csv") for s in states}; inputs.update({f"{corpus}/per_context_{s}.csv": _sha(Path(directory) / f"per_context_{s}.csv") for s in states}); same[corpus] = True
        for kind in KINDS:
            base = pairs(tabs[states[0]], kind); sb = sets(base); cv = coverage(base, sb)
            for name in SETS:
                cov_rows.append({"corpus": corpus, "control_kind": kind, "set": name, **cv[name], "n_dropped": cv["n_dropped"]})
            for s in states:
                ps = pairs(tabs[s], kind); same[corpus] &= _same(base, ps); ss = sets(ps)
                for name in ("S1", "S2"):
                    r = contrast(ps, ss[name], n_boot, seed); row = {"corpus": corpus, "state": s, "control_kind": kind, "set": name, "status": r["status"], "n": r["n"], "n_features": r["n_features"], "n_clusters": r["n_clusters"]}
                    if r["status"] == "fitted":
                        for k in ("ne_gap", "cos_gap", "log2_error_ratio", "share_real_larger"):
                            row.update({k: r[k]["point"], f"{k}_low": r[k]["low"], f"{k}_high": r[k]["high"], f"{k}_n_missing": r[k]["n_missing"]})
                    con_rows.append(row)
        cov_rows.append({"corpus": corpus, "control_kind": "all_four", "set": "S1_all_four", "n_pairs": all_four(tabs[states[0]])})
    for name, rows in (("support_coverage.csv", cov_rows), ("support_contrast.csv", con_rows)):
        write_csv(out_dir / name, rows)
    (out_dir / "support.json").write_text(json.dumps({"band": list(BAND), "min_pairs": MIN_PAIRS, "min_features": MIN_FEATURES, "sets_identical_across_states": same, "inputs_sha256": inputs, "n_boot": n_boot, "seed": seed,
                                                       "coverage": cov_rows, "contrast": con_rows}, indent=1, default=float))
    return cov_rows, con_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--main", required=True); ap.add_argument("--external", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--states", nargs="*", default=["INT16", "INT6", "INT4"]); a = ap.parse_args(); cov, con = run(a.main, a.external, a.states, a.out)
    for r in cov:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k in ("corpus", "control_kind", "set", "n_pairs", "n_features", "share_in_band", "rho_p50", "edit_ratio_p50")})


if __name__ == "__main__":
    main()
