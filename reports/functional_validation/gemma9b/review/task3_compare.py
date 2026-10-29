"""9B review task 3 (D-125): retention statistics of the imported original-RTN Gemma-2-9B study next to the matching Gemma-2-2B original-RTN study, with 95 % intervals.

Input: the two committed `curves.csv` (column `net` = net harm; E(state) = net harm at ratio 0, the zero-ablation; per-language retention = E(state) / E(FP)).
Intervals: percentile bootstrap over the 27 languages (5,000 resamples, seed 0; the SAME resample indices for 9B and 2B, so the difference is paired by language).
Output: `task3_compare.json` next to this file. CPU, numpy only, about one second. Written 2026-10-29 by Claude (D-125 cron); post hoc, not part of any freeze.

  python3 task3_compare.py
"""
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FV = HERE.parent.parent                                    # reports/functional_validation
CURVES = {"9B": FV / "gemma9b/results/analysis/curves.csv", "2B": FV / "gain_study/results/curves.csv"}
STATES = ["FP", "INT8", "INT7", "INT6", "INT5", "INT4"]
N_BOOT, SEED = 5000, 0


def load(path):
    return {(r["state"], r["language"], round(float(r["ratio"]), 4)): float(r["net"]) for r in csv.DictReader(open(path))}


def main():
    data = {m: load(p) for m, p in CURVES.items()}
    langs = sorted({k[1] for k in data["9B"]})
    assert langs == sorted({k[1] for k in data["2B"]}) and len(langs) == 27
    idx = np.random.default_rng(SEED).integers(0, len(langs), size=(N_BOOT, len(langs)))

    def effect(m, s, ratio=0.0):
        return np.array([data[m][(s, l, ratio)] for l in langs])

    def ci(x):
        return [float(v) for v in np.percentile(x, [2.5, 97.5])]

    def stat(point, boot):
        return {"estimate": float(point), "ci95": ci(boot)}

    out = {"method": f"language bootstrap, {N_BOOT} resamples, seed {SEED}, same indices for 9B and 2B", "n_languages": len(langs)}
    for m in ("9B", "2B"):
        fp = effect(m, "FP")
        res = {"FP_E_mean": stat(fp.mean(), fp[idx].mean(1)), "FP_E_median": stat(np.median(fp), np.median(fp[idx], 1)), "FP_E_positive_languages": int((fp > 0).sum())}
        for s in STATES[1:]:
            es = effect(m, s)
            r = es / fp
            res[s] = {"macro_mean_retention": stat(r.mean(), r[idx].mean(1)), "median_retention": stat(np.median(r), np.median(r[idx], 1)),
                      "ratio_of_mean_effects": stat(es.mean() / fp.mean(), es[idx].mean(1) / fp[idx].mean(1)),
                      "languages_above_1": int((r > 1).sum()), "languages_above_1.3": int((r > 1.3).sum()),
                      "macro_mean_without_the_largest_language": float(np.sort(r)[:-1].mean())}
        r3 = [0.3536, 0.1768, 0.0884, 0.0]                  # multipliers of the group, mildest to zero-ablation
        vals = np.array([[data[m][("FP", l, x)] for x in r3] for l in langs])
        res["FP_dose_order"] = {"languages_with_harm_strictly_increasing_as_the_multiplier_falls": int(sum(all(v[i] < v[i + 1] for i in range(3)) for v in vals)),
                                "languages_with_the_freeze_P3a_inequality_E(0.354)>E(0.177)>E(0.088)": int(sum(v[0] > v[1] > v[2] for v in vals))}
        a4, a0 = (np.array([data[m][(s, l, 0.0884)] for l in langs]) for s in ("INT4", "FP"))
        q = a4 / a0
        res["INT4_over_FP_at_multiplier_0.0884"] = {"median": stat(np.median(q), np.median(q[idx], 1)), "mean": stat(q.mean(), q[idx].mean(1)), "languages_above_1": int((q > 1).sum())}
        order = np.argsort(fp)
        res["smallest_FP_effects"] = [{"language": langs[i], "E_FP": round(float(fp[i]), 4), "INT4_retention": round(float(effect(m, "INT4")[i] / fp[i]), 2)} for i in order[:4]]
        r4 = effect(m, "INT4") / fp
        res["largest_INT4_retention"] = [{"language": langs[i], "retention": round(float(r4[i]), 2)} for i in np.argsort(-r4)[:3]]
        out[m] = res
    for s in ("INT4", "INT5", "INT6"):
        a, b = effect("9B", s) / effect("9B", "FP"), effect("2B", s) / effect("2B", "FP")
        rank = lambda x: np.argsort(np.argsort(x))
        out[f"difference_9B_minus_2B_{s}"] = {"macro_mean_retention": stat((a - b).mean(), (a - b)[idx].mean(1)), "median_retention": stat(np.median(a) - np.median(b), np.median(a[idx], 1) - np.median(b[idx], 1)),
                                              "spearman_of_the_per_language_retention_9B_vs_2B": float(np.corrcoef(rank(a), rank(b))[0, 1])}
    (HERE / "task3_compare.json").write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", HERE / "task3_compare.json")


if __name__ == "__main__":
    main()
