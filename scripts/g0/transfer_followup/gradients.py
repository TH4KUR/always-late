"""The stored gradient scalars of the intervention-transfer study against the signed negative-log-likelihood changes they approximate (revision R2, task 1b of
docs/superpowers/plans/2026-10-26-publication-recovery-after-results.md; POST HOC; protocol fixed in reports/intervention_transfer/revision_R2_gradient_scalars.json before the code and the run).

The executed gradient baseline (amendment A9) stores, per item, the first-order prediction of the loss change of its edit: minus the dot product of the autograd gradient of the mean NLL and the edit, summed over
the edited rows, in the full-precision model (g_pred_fp), in the quantized model for the fixed full-precision edit (g_pred_q_fixed) and for the edit with the quantized state's own amplitude (g_pred_q_native). The
cells of the study give the actual signed loss changes (dloss_FP_FP, dloss_Q_Q, dloss_Q_Q_native). This module compares them: C0, C1, C2 (the three levels) and S1, S2 (the quantized-minus-full-precision shifts), with
the Pearson correlation, the through-origin slope and R2, the sign agreement (all items and above the noise threshold) and crossed-bootstrap intervals. It is the scalar NLL gradient against a scalar NLL target;
the vector-valued first-order logit effect (the logit JVP) was never computed and is not tested here.

    python phase1_transfer_gradient_check.py --main <analysis_main> --external <analysis_external> --out <dir> [--states INT16 INT6 INT4]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .tables import sha256_file as _sha, write_csv

import phase1_transfer_analysis as ta
from . import cross_corpus as cc

COMPARISONS = {"C0": ("g_pred_fp", "dloss_FP_FP"), "C1": ("g_pred_q_fixed", "dloss_Q_Q"), "C2": ("g_pred_q_native", "dloss_Q_Q_native"),
               "S1": (("g_pred_q_fixed", "g_pred_fp"), ("dloss_Q_Q", "dloss_FP_FP")), "S2": (("g_pred_q_native", "g_pred_fp"), ("dloss_Q_Q_native", "dloss_FP_FP"))}
METRICS = ("r", "slope", "r2_origin", "sign_agree", "sign_agree_above_noise")
CHUNK = 100
CORPORA = ("flores_main", "wikipedia_external")


def _value(r, spec):
    """A scalar of a row, or the difference of two (first minus second); None if any is missing or not finite."""
    names = (spec,) if isinstance(spec, str) else spec; vals = [r.get(n) for n in names]
    if any(v is None or not np.isfinite(v) for v in vals):
        return None
    return vals[0] if len(vals) == 1 else vals[0] - vals[1]


def comparison_arrays(rows, kinds):
    """The arrays of the five comparisons for the items of the given direction kinds, sorted by (feature, sentence, kind): prediction `p`, actual change `a`, and per item the `feature`, `cluster`, FP edit norm `edit`
    and eligibility; an item with a missing or non-finite value in either quantity is left out of that comparison and counted in `n_missing`."""
    sel = sorted((r for r in rows.values() if r["kind"] in kinds), key=lambda r: (r["feature"], r["sentence"], r["kind"])); out = {}
    for name, (pspec, aspec) in COMPARISONS.items():
        keep = [(r, _value(r, pspec), _value(r, aspec)) for r in sel]; ok = [(r, p, a) for r, p, a in keep if p is not None and a is not None]
        out[name] = {"p": np.array([p for _, p, _ in ok], dtype=float), "a": np.array([a for _, _, a in ok], dtype=float), "feature": np.array([r["feature"] for r, _, _ in ok], dtype=np.int64),
                     "cluster": np.array([r["cluster"] for r, _, _ in ok], dtype=np.int64), "edit": np.array([r["edit_norm_fp"] for r, _, _ in ok], dtype=float), "eligible": np.array([bool(r["eligible"]) for r, _, _ in ok], dtype=bool),
                     "n_missing": len(sel) - len(ok)}
    return out


def noise_threshold(rows_int16):
    """The 95th percentile of |dloss_Q_Q - dloss_FP_FP| over the real items of the INT16 table (the rounding reference of the shift)."""
    v = [abs(r["dloss_Q_Q"] - r["dloss_FP_FP"]) for r in rows_int16.values() if r["kind"] == "real" and r.get("dloss_Q_Q") is not None and r.get("dloss_FP_FP") is not None]
    return float(np.percentile(v, 95))


def weighted_metrics(p, a, W, noise=None):
    """The metrics of the actual change `a` on the prediction `p` with the weights W ([B, n], one row per replicate; ones for the point): Pearson correlation `r`, through-origin slope `slope` = sum(w p a) / sum(w p^2),
    through-origin `r2_origin` = 1 - sum(w (a - slope p)^2) / sum(w a^2), `sign_agree` = weighted share of the items with p and a both non-zero whose signs agree; with `noise` also `sign_agree_above_noise` (the same among the
    items with |a| > noise) and `n_above_noise` (their number). Each metric is an array of length B."""
    p = np.asarray(p, dtype=float); a = np.asarray(a, dtype=float); W = np.atleast_2d(np.asarray(W, dtype=float)); sw = W.sum(1)
    mp = (W * p).sum(1) / sw; ma = (W * a).sum(1) / sw; dp = p[None, :] - mp[:, None]; da = a[None, :] - ma[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (W * dp * da).sum(1) / np.sqrt((W * dp * dp).sum(1) * (W * da * da).sum(1)); beta = (W * p * a).sum(1) / (W * p * p).sum(1)
        r2 = 1.0 - (W * (a[None, :] - beta[:, None] * p[None, :]) ** 2).sum(1) / (W * a * a).sum(1)
        valid = (p != 0) & (a != 0); agree = (np.sign(p) == np.sign(a)) & valid; sign = (W * agree).sum(1) / (W * valid).sum(1)
    out = {"r": r, "slope": beta, "r2_origin": r2, "sign_agree": sign}
    if noise is not None:
        m = valid & (np.abs(a) > noise)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["sign_agree_above_noise"] = (W * (agree & m)).sum(1) / (W * m).sum(1)
        out["n_above_noise"] = int(m.sum())
    return out


def replicates(p, a, fi, ci, mult_features, mult_clusters, noise, chunk=CHUNK):
    """The metrics in every bootstrap replicate (weight of an item = multiplicity of its feature times that of its cluster), computed `chunk` replicates at a time to bound the memory."""
    parts = []
    for s in range(0, len(mult_features), chunk):
        W = mult_features[s:s + chunk][:, fi] * mult_clusters[s:s + chunk][:, ci]; m = weighted_metrics(p, a, W, noise); m.pop("n_above_noise", None); parts.append(m)
    return {k: np.concatenate([m[k] for m in parts]) for k in parts[0]}


def metrics_ci(p, a, feature, cluster, noise, n_boot=ta.BOOTSTRAP_REPLICATES, seed=ta.BOOTSTRAP_SEED, level=ta.BOOTSTRAP_LEVEL):
    """The point (unweighted) and the percentile interval (crossed bootstrap over features and clusters) of every metric, with the counts. -> {metric: {"point", "low", "high"}, "n", "n_features", "n_clusters", "n_above_noise"}."""
    p = np.asarray(p, dtype=float); a = np.asarray(a, dtype=float); _, fi = np.unique(feature, return_inverse=True); _, ci = np.unique(cluster, return_inverse=True)
    nf, nc = int(fi.max()) + 1, int(ci.max()) + 1; mf, mc = ta.bootstrap_multiplicities(nf, nc, n_boot, seed); point = weighted_metrics(p, a, np.ones((1, len(p))), noise); reps = replicates(p, a, fi, ci, mf, mc, noise)
    lo, hi = (100 - level) / 2, 100 - (100 - level) / 2
    out = {k: {"point": float(point[k][0]), "low": float(np.nanpercentile(reps[k], lo)), "high": float(np.nanpercentile(reps[k], hi))} for k in METRICS}
    return {**out, "n": int(len(p)), "n_features": nf, "n_clusters": nc, "n_above_noise": point["n_above_noise"]}


def _groups(rows):
    """The item groups of a table: real_all, real_eligible, controls_all, and the terciles of the FP edit norm of the real items (boundaries: the terciles of the real items of the table)."""
    real = comparison_arrays(rows, ("real",)); ctrl = comparison_arrays(rows, ta.CONTROL_KINDS); edit = np.array([r["edit_norm_fp"] for r in rows.values() if r["kind"] == "real"], dtype=float); q1, q2 = np.quantile(edit, [1 / 3, 2 / 3])
    return real, ctrl, (q1, q2)


def run(main_dir, external_dir, states, out_dir, n_boot=ta.BOOTSTRAP_REPLICATES, seed=ta.BOOTSTRAP_SEED):
    """Per corpus and state the five comparisons for the groups real_all, real_eligible, controls_all and the three terciles of the real FP edit norm, with the noise threshold of the corpus (INT16 reference). Writes
    `gradient_vs_nll.csv` (one row per corpus, state, comparison and group) and `gradient_vs_nll.json` (the same plus the thresholds, whether C0 is identical across the state tables, and the sha256 of the six inputs)."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True); table, thresholds, c0_same, inputs = [], {}, {}, {}
    for corpus, directory in zip(CORPORA, (main_dir, external_dir)):
        tabs = {s: cc.rows_from_csv(Path(directory) / f"per_context_{s}.csv") for s in states}; inputs.update({f"{corpus}/per_context_{s}.csv": _sha(Path(directory) / f"per_context_{s}.csv") for s in states})
        ref = tabs["INT16"] if "INT16" in tabs else tabs[states[0]]; t = noise_threshold(ref); thresholds[corpus] = t
        first = tabs[states[0]]; c0_same[corpus] = all(all(tabs[s].get(k, {}).get(f) == v[f] for s in states for f in ("dloss_FP_FP", "g_pred_fp")) for k, v in first.items())
        for s in states:
            real, ctrl, (q1, q2) = _groups(tabs[s])
            for cname in COMPARISONS:
                r_, c_ = real[cname], ctrl[cname]; tercile = np.where(r_["edit"] <= q1, 1, np.where(r_["edit"] <= q2, 2, 3))
                sets = {"real_all": np.ones(len(r_["p"]), dtype=bool), "real_eligible": r_["eligible"], "real_edit_tercile_1": tercile == 1, "real_edit_tercile_2": tercile == 2, "real_edit_tercile_3": tercile == 3}
                for g, m in sets.items():
                    res = metrics_ci(r_["p"][m], r_["a"][m], r_["feature"][m], r_["cluster"][m], t, n_boot, seed) if m.sum() >= 3 else None; table.append(_row(corpus, s, cname, g, res, r_["n_missing"], t))
                table.append(_row(corpus, s, cname, "controls_all", metrics_ci(c_["p"], c_["a"], c_["feature"], c_["cluster"], t, n_boot, seed), c_["n_missing"], t))
    write_csv(out_dir / "gradient_vs_nll.csv", table)
    (out_dir / "gradient_vs_nll.json").write_text(json.dumps({"noise_threshold": thresholds, "c0_identical_across_states": c0_same, "inputs_sha256": inputs, "n_boot": n_boot, "seed": seed, "rows": table}, indent=1, default=float))
    return table


def _row(corpus, state, comparison, group, res, n_missing, t):
    row = {"corpus": corpus, "state": state, "comparison": comparison, "group": group, "noise_threshold": t, "n_missing": n_missing}
    if res is None:
        return {**row, "n": 0}
    row.update({"n": res["n"], "n_features": res["n_features"], "n_clusters": res["n_clusters"], "n_above_noise": res["n_above_noise"]})
    for k in METRICS:
        row.update({k: res[k]["point"], f"{k}_low": res[k]["low"], f"{k}_high": res[k]["high"]})
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--main", required=True); ap.add_argument("--external", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--states", nargs="*", default=["INT16", "INT6", "INT4"]); a = ap.parse_args()
    for row in run(a.main, a.external, a.states, a.out):
        if row["group"] == "real_all":
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k in ("corpus", "state", "comparison", "n", "r", "slope", "r2_origin", "sign_agree", "sign_agree_above_noise")})


if __name__ == "__main__":
    main()
