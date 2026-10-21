"""D-035 Task 6 cross-checks of the association-retention status (design: reports/survival_method_reassessment.md section 7, "review agreement and disagreements between firing-rate
profiles, activation-mean profiles, original-language correlation and controls"). Two questions, both on the AGGREGATE profile of one stream and the frozen cohort:
  1. view agreement: among the features established in full precision (firing-rate view), how often does the activation-mean view (the language with the highest mean activation, zeros
     included, is the original language and is the unique mean peak, with the same support rule) agree with the firing-rate view that a condition retains the association?
  2. correlation: among the same features, how does the status relate to the historical Pearson correlation between full-precision and quantized activations at the original language
     (survival.pt of the original run; the measure of the earlier survival analysis)?
Nothing here changes the headline analysis; it reports agreement and disagreement."""
import json
from pathlib import Path

import numpy as np
import torch

import phase1_association_analysis as an
import phase1_association_metrics as am


def _views(p, peak, min_positives, min_panel_rate):
    counts, sums, ntok = np.asarray(p["counts"]), np.asarray(p["sums"], dtype=np.float64), np.asarray(p["ntok"])
    st = am.association_status(counts, ntok, peak, min_positives, min_panel_rate)
    rates = counts / ntok.astype(np.float64)[:, None]; means = sums / ntok.astype(np.float64)[:, None]
    met = am.association_metrics(rates, means, peak)
    mean_unique_original = met["unique_peak_mean"] & met["peak_is_original_mean"]
    return st, mean_unique_original


def view_agreement(fp, q, peak, mask, min_positives=am.MIN_POSITIVES, min_panel_rate=am.MIN_PANEL_RATE):
    """2 x 2 of 'retained by the firing-rate view' against 'retained by the activation-mean view' among the features of `mask` that are established in full precision (rate view).
    Mean-retained = supported in the condition (same support rule as the rate view) and the original language is the unique mean peak. `fp_mean_agrees` counts established features
    whose full-precision mean peak is also the original language, i.e. where the two views agree at the baseline."""
    peak, mask = np.asarray(peak), np.asarray(mask, dtype=bool)
    fp_st, fp_mean = _views(fp, peak, min_positives, min_panel_rate); q_st, q_mean = _views(q, peak, min_positives, min_panel_rate)
    est = fp_st["retained"] & mask
    rate_ret = est & q_st["retained"]
    mean_ret = est & ~q_st["inactive"] & ~q_st["low_support"] & q_mean
    return {"n_established": int(est.sum()), "rate_retained": int(rate_ret.sum()), "mean_retained": int(mean_ret.sum()), "both": int((rate_ret & mean_ret).sum()),
            "rate_only": int((rate_ret & ~mean_ret).sum()), "mean_only": int((~rate_ret & mean_ret).sum()), "neither": int((est & ~rate_ret & ~mean_ret).sum()), "fp_mean_agrees": int((est & fp_mean).sum())}


def correlation_split(retained, established, r, threshold=0.9):
    """Among established features: retained and lost by the correlation at the original language being above the threshold (strictly) or not; NaN correlations are counted apart."""
    ret, est, r = np.asarray(retained, dtype=bool), np.asarray(established, dtype=bool), np.asarray(r, dtype=np.float64)
    ok = est & ~np.isnan(r)
    hi = r > threshold
    med = lambda m: float(np.median(r[m])) if m.any() else float("nan")
    return {"retained_high": int((ok & ret & hi).sum()), "retained_low": int((ok & ret & ~hi).sum()), "lost_high": int((ok & ~ret & hi).sum()), "lost_low": int((ok & ~ret & ~hi).sum()),
            "median_r_retained": med(ok & ret), "median_r_lost": med(ok & ~ret), "n_nan": int((est & np.isnan(r)).sum()), "threshold": threshold}


def crosscheck(run_dir, baseline_dir, stream="eq", min_positives=am.MIN_POSITIVES, min_panel_rate=am.MIN_PANEL_RATE, r_threshold=0.9, allow_subset=False):
    side = an.load_aggregate_sidecar(run_dir, stream); coh = an.load_cohort(baseline_dir, side, allow_subset)
    surv = torch.load(Path(baseline_dir) / "survival.pt", map_location="cpu", weights_only=True)
    ones = np.ones((1, 1)); cnt, sm, nt = (a[0] for a in side.totals("FP", ones)); fp = {"counts": cnt, "sums": sm, "ntok": nt}
    fp_st = am.association_status(cnt, nt, coh.peak, min_positives, min_panel_rate); est = fp_st["retained"] & coh.selective
    out = {"stream": stream, "n_selective": int(coh.selective.sum()), "n_established": int(est.sum()), "min_positives": min_positives, "conditions": {}}
    for cond in [c for c in side.conditions if c != "FP"]:
        c2, s2, n2 = (a[0] for a in side.totals(cond, ones)); q = {"counts": c2, "sums": s2, "ntok": n2}
        st = am.association_status(c2, n2, coh.peak, min_positives, min_panel_rate)
        entry = {"views": view_agreement(fp, q, coh.peak, coh.selective, min_positives, min_panel_rate)}
        bits = int(cond[3:])
        if bits in surv["corr"]:
            r = surv["corr"][bits].numpy()[coh.peak_base, side.feature_ids]                          # correlation of every feature at its ORIGINAL language (baseline index)
            entry["correlation"] = correlation_split(st["retained"] & est, est, r, r_threshold)
        out["conditions"][cond] = entry
    return out


def render_markdown(out):
    L = ["# Cross-checks of the association status (D-035)", "",
         f"Generated by `scripts/g0/phase1_association_crosscheck.py` on the aggregate profile of stream `{out['stream']}`. Features: the {out['n_established']} of {out['n_selective']} original selective features established in full precision (firing-rate view; support at least {out['min_positives']} positive tokens). The numbers are produced from the files; the prose states no results.", "",
         "## Firing-rate view against activation-mean view", "",
         "Retained by the firing-rate view means the original language is the unique, supported firing-rate peak; retained by the mean view means it is the unique peak of the mean activation (zeros included) under the same support rule. `Mean view agrees in FP` counts established features whose full-precision mean peak is also the original language.", "",
         "| condition | established | retained, firing rate | retained, mean | both | rate only | mean only | neither | mean view agrees in FP |", "|---|---|---|---|---|---|---|---|---|"]
    for cond, e in out["conditions"].items():
        v = e["views"]
        L.append(f"| {cond} | {v['n_established']} | {v['rate_retained']} | {v['mean_retained']} | {v['both']} | {v['rate_only']} | {v['mean_only']} | {v['neither']} | {v['fp_mean_agrees']} |")
    L += ["", "## Status against the historical correlation at the original language", "",
          "Pearson correlation between full-precision and quantized activations of the feature in its original language (the quantity of the earlier survival analysis), above the threshold or not, for established features retained or lost by the firing-rate status.", "",
          "| condition | retained, r above | retained, r not above | lost, r above | lost, r not above | median r retained | median r lost | undefined r |", "|---|---|---|---|---|---|---|---|"]
    for cond, e in out["conditions"].items():
        if "correlation" in e:
            c = e["correlation"]; f = lambda x: "n/a" if x != x else f"{x:.3f}"
            L.append(f"| {cond} | {c['retained_high']} | {c['retained_low']} | {c['lost_high']} | {c['lost_low']} | {f(c['median_r_retained'])} | {f(c['median_r_lost'])} | {c['n_nan']} |")
    return "\n".join(L) + "\n"


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="D-035 cross-checks of the association status")
    ap.add_argument("--run", required=True); ap.add_argument("--baseline", required=True); ap.add_argument("--out", required=True); ap.add_argument("--stream", default="eq", choices=["eq", "all"])
    ap.add_argument("--min-positives", type=int, default=am.MIN_POSITIVES); ap.add_argument("--r-threshold", type=float, default=0.9); ap.add_argument("--allow-subset", action="store_true")
    a = ap.parse_args(argv)
    out = crosscheck(a.run, a.baseline, a.stream, a.min_positives, am.MIN_PANEL_RATE, a.r_threshold, a.allow_subset)
    d = Path(a.out); d.mkdir(parents=True, exist_ok=True)
    (d / "crosscheck.json").write_text(json.dumps(an._clean(out), indent=1, sort_keys=True, allow_nan=False) + "\n"); (d / "crosscheck.md").write_text(render_markdown(out))
    print("wrote", d)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv[1:]))
