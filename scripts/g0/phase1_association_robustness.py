"""D-035 Task 6 robustness views of the association-retention result (descriptive, exploratory; plan Task 6 and the design in reports/survival_method_reassessment.md section 7).
  1. threshold sensitivity: the established and lost counts of every condition for several support rules (minimum positive tokens at the peak), on the aggregate profile of a stream.
  2. fragility: among the features established in full precision, how the loss of the association relates to baseline properties of the feature (target firing rate, target share, margin to the
     strongest other language, mean activation at the original language, maximum activation at the original language): loss fraction by quantile bin of the covariate and a rank correlation
     between the covariate and the loss indicator. No model is fitted and no hypothesis test is run; the covariates are related to each other (a feature with a high target share has a large margin),
     so a single column is not a causal statement about its role."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import phase1_association_analysis as an
import phase1_association_collect as ac
import phase1_association_metrics as am

KEYS = ("n_selective", "fp_established", "retained_of_established", "lost_of_established", "loss_fraction", "conditional_fraction")


def quantile_bins(x, mask, n_bins):
    """Bin ids (0 .. k-1, k <= n_bins) of the finite values of x inside mask, by quantile (equal counts up to ties); -1 elsewhere. Repeated quantile edges collapse bins."""
    x = np.asarray(x, dtype=np.float64); valid = np.asarray(mask, dtype=bool) & np.isfinite(x)
    out = np.full(x.shape, -1, dtype=np.int64)
    if not valid.any():
        return out
    edges = np.unique(np.quantile(x[valid], np.linspace(0, 1, n_bins + 1)[1:-1]))
    ids = np.digitize(x[valid], edges)
    _, inv = np.unique(ids, return_inverse=True)
    out[valid] = inv
    return out


def binned_loss(x, established, lost, n_bins=10):
    """[{bin, n, lost, loss_fraction, lo, hi}] over the established features with a finite covariate."""
    x = np.asarray(x, dtype=np.float64); est, lost = np.asarray(established, dtype=bool), np.asarray(lost, dtype=bool)
    b = quantile_bins(x, est, n_bins); rows = []
    for k in range(int(b.max()) + 1 if (b >= 0).any() else 0):
        m = b == k
        rows.append({"bin": k, "n": int(m.sum()), "lost": int((m & lost).sum()), "loss_fraction": float((m & lost).sum() / m.sum()), "lo": float(x[m].min()), "hi": float(x[m].max())})
    return rows


def rank_correlation(x, lost, established):
    """Spearman correlation between a covariate and the loss indicator among established features with a finite covariate; NaN when nobody (or everybody) is lost."""
    x = np.asarray(x, dtype=np.float64); ok = np.asarray(established, dtype=bool) & np.isfinite(x); y = np.asarray(lost, dtype=bool)[ok]
    if y.sum() == 0 or y.sum() == y.size or y.size < 3 or np.all(x[ok] == x[ok][0]):
        return float("nan")
    return float(spearmanr(x[ok], y.astype(float)).statistic)


def threshold_sensitivity(side, coh, thresholds=(20, 50, 100), conditions=None, min_panel_rate=am.MIN_PANEL_RATE):
    """{threshold: {condition: {fp_established, retained_of_established, lost_of_established, loss_fraction, conditional_fraction, n_selective}}} on the sidecar's profile."""
    out = {}
    for thr in thresholds:
        pe = an.point_estimate(side, coh, min_positives=thr, min_panel_rate=min_panel_rate, conditions=conditions)
        out[int(thr)] = {c: {k: v[k] for k in KEYS} for c, v in pe.items()}
    return out


def fragility(run_dir, side, coh, condition, stream="eq", min_positives=am.MIN_POSITIVES, min_panel_rate=am.MIN_PANEL_RATE, n_bins=10):
    """Loss of the association in `condition` against baseline (full-precision) properties of the features established in full precision."""
    ones = np.ones((1, 1)); fp = [a[0] for a in side.totals("FP", ones)]; q = [a[0] for a in side.totals(condition, ones)]
    fp_st = am.association_status(fp[0], fp[2], coh.peak, min_positives, min_panel_rate); q_st = am.association_status(q[0], q[2], coh.peak, min_positives, min_panel_rate)
    est = fp_st["retained"] & coh.selective; lost = est & ~q_st["retained"]
    rates = fp[0] / fp[2].astype(np.float64)[:, None]; means = fp[1] / fp[2].astype(np.float64)[:, None]; met = am.association_metrics(rates, means, coh.peak)
    amax = ac.read_profiles(run_dir, stream)["conditions"]["FP"]["activation_max"].numpy()
    f = np.arange(coh.peak.shape[0])
    cov = {"target_rate": met["target_rate"], "target_share": met["share_rate"], "margin": met["margin_rate"], "target_mean": met["target_mean"], "max_activation": amax[coh.peak, side.feature_ids]}
    return {"condition": condition, "n_established": int(est.sum()), "n_lost": int(lost.sum()),
            "covariates": {k: {"spearman": rank_correlation(v, lost, est), "bins": binned_loss(v, est, lost, n_bins)} for k, v in cov.items()}}


def render_markdown(out):
    L = ["# Robustness views of the association-retention result (D-035; descriptive, exploratory)", "",
         f"Generated by `scripts/g0/phase1_association_robustness.py` on the aggregate profile of stream `{out['stream']}`. The numbers are produced from the files; the prose states no results. No model is fitted and no hypothesis test is run; the covariates are related to each other, so a column is not a causal statement.", "",
         "## Sensitivity to the support rule (minimum positive tokens at the peak)", "", "| minimum positives | condition | established in FP | lost | loss fraction of the cohort | retained among established |", "|---|---|---|---|---|---|"]
    for thr, per in out["thresholds"].items():
        for c, v in per.items():
            L.append(f"| {thr} | {c} | {v['fp_established']} | {v['lost_of_established']} | {v['loss_fraction']:.4f} | {v['conditional_fraction']:.3f} |")
    L += ["", f"## Loss of the association against baseline properties (minimum positives {out['min_positives']})", "",
          "Established features only; bins are quantiles of the covariate in full precision. `rho` is the Spearman correlation between the covariate and the loss indicator (negative: lower values lose more).", ""]
    for c, fr in out["fragility"].items():
        L += [f"### {c}: {fr['n_lost']} of {fr['n_established']} established features lost", "", "| covariate | rho | loss fraction by bin (lowest to highest quantile bin) |", "|---|---|---|"]
        for name, v in fr["covariates"].items():
            rho = "n/a" if v["spearman"] != v["spearman"] else f"{v['spearman']:.3f}"
            L.append(f"| {name} | {rho} | " + " ; ".join(f"{b['loss_fraction']:.3f}" for b in v["bins"]) + " |")
        L.append("")
    return "\n".join(L) + "\n"


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="D-035 robustness views")
    ap.add_argument("--run", required=True); ap.add_argument("--baseline", required=True); ap.add_argument("--out", required=True); ap.add_argument("--stream", default="eq", choices=["eq", "all"])
    ap.add_argument("--thresholds", default="20,50,100"); ap.add_argument("--conditions", default=None, help="comma-separated; default all conditions except FP")
    ap.add_argument("--bins", type=int, default=10); ap.add_argument("--min-positives", type=int, default=am.MIN_POSITIVES); ap.add_argument("--allow-subset", action="store_true")
    a = ap.parse_args(argv)
    side = an.load_aggregate_sidecar(a.run, a.stream); coh = an.load_cohort(a.baseline, side, a.allow_subset)
    conds = a.conditions.split(",") if a.conditions else [c for c in side.conditions if c != "FP"]
    out = {"stream": a.stream, "min_positives": a.min_positives, "thresholds": threshold_sensitivity(side, coh, [int(x) for x in a.thresholds.split(",")], conds),
           "fragility": {c: fragility(a.run, side, coh, c, a.stream, a.min_positives, n_bins=a.bins) for c in conds if c != "INT0"}}
    d = Path(a.out); d.mkdir(parents=True, exist_ok=True)
    (d / "robustness.json").write_text(json.dumps(an._clean(out), indent=1, sort_keys=True, allow_nan=False) + "\n"); (d / "robustness.md").write_text(render_markdown(out))
    print("wrote", d)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv[1:]))
