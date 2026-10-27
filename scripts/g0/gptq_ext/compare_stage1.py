"""Stage-1 comparison of the GPTQ extension (freeze endpoints E1 and E2; plan Task 9): the Phase 1 runs of three arms (legacy RTN = the old `p1-mask` run, matched RTN, GPTQ) compared per language and bit width.
Inputs are the `survival.pt` files that `phase1_run.py` writes (the equal-budget stream `eq/` is the pre-registered one); survival and the median correlation come from the project's own `phase1_run.survival_summary`
(active = FP firing rate above 0.001 in that language, survived r > 0.9), the loss is the per-language mean next-token cross-entropy of the state minus that of FP on the same tokens (nats per token). Every contrast is
arm A minus arm B and PAIRED BY LANGUAGE (27 pairs); the interval is a 95 % percentile bootstrap of the mean over languages; all contrasts are exploratory (D-101): the three pre-set tests T1 to T3 come from the unchanged
`phase1_analysis.py` run on each arm separately. The bridge check requires that the FP numbers and the legacy rounding floors (INT16, INT12, INT0) are identical across arms (the FP path is unchanged and was bit-for-bit
reproducible, A-017): a difference is reported, never merged silently.

  python3 scripts/g0/gptq_ext/compare_stage1.py --legacy <dir> --matched <dir> --gptq <dir> --roster reports/g0/phase1_languages.json --out <dir>"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phase1_run as pr  # noqa: E402

FLOOR_BITS = (16, 12, 0)
METRICS = ("nll_delta_diff", "survived_pct_diff", "median_corr_diff")
CONTRASTS = (("matched RTN minus legacy RTN", "matched RTN", "legacy RTN"), ("GPTQ minus matched RTN", "GPTQ", "matched RTN"), ("GPTQ minus legacy RTN", "GPTQ", "legacy RTN"))


def load_roster(path):
    return {r["code"]: r for r in json.loads(Path(path).read_text())["languages"]}


def load_arm(directory):
    s = torch.load(Path(directory) / "survival.pt", weights_only=False)
    langs, bits = list(s["languages"]), list(s["bits"])
    arm = {"dir": str(directory), "languages": langs, "bits": bits, "n": [int(x) for x in s["n"]], "fp_firing_rate": s["firing_rate"], "floor_corr": {b: s["corr"][b] for b in bits if b in FLOOR_BITS},
           "nll_abs": {}, "nll_delta": {}, "survived_pct": {}, "median_corr": {}, "n_active": {}, "fp_loss": {}}
    for li, lang in enumerate(langs):
        fp = float(s["loss_sum"]["FP"][li]) / float(s["loss_n"]["FP"][li])
        arm["fp_loss"][lang] = fp; arm["nll_abs"][lang], arm["nll_delta"][lang], arm["survived_pct"][lang], arm["median_corr"][lang] = {}, {}, {}, {}
        for b in bits:
            loss = float(s["loss_sum"][b][li]) / float(s["loss_n"][b][li])
            summ = pr.survival_summary(s["corr"][b][li].double(), s["firing_rate"][li].double(), 0.001)
            arm["nll_abs"][lang][b], arm["nll_delta"][lang][b] = loss, loss - fp
            arm["survived_pct"][lang][b], arm["median_corr"][lang][b] = summ["survived_>0.9_pct"], summ["median_corr"]
            arm["n_active"][lang] = summ["n_active_features"]
    return arm


def paired_table(a, b, roster, bits):
    """One row per language and bit width: the difference arm a minus arm b of the loss increase, the survival percentage and the median correlation."""
    if a["languages"] != b["languages"]:
        raise ValueError("the two arms have different languages or language order")
    missing = [x for x in bits if x not in a["bits"] or x not in b["bits"]]
    if missing:
        raise ValueError(f"bit widths {missing} are missing in one of the arms")
    rows = []
    for lang in a["languages"]:
        for x in bits:
            rows.append({"language": lang, "tier": roster[lang]["tier"], "bits": x,
                         "nll_delta_a": a["nll_delta"][lang][x], "nll_delta_b": b["nll_delta"][lang][x], "nll_delta_diff": a["nll_delta"][lang][x] - b["nll_delta"][lang][x],
                         "survived_pct_a": a["survived_pct"][lang][x], "survived_pct_b": b["survived_pct"][lang][x], "survived_pct_diff": a["survived_pct"][lang][x] - b["survived_pct"][lang][x],
                         "median_corr_a": a["median_corr"][lang][x], "median_corr_b": b["median_corr"][lang][x], "median_corr_diff": a["median_corr"][lang][x] - b["median_corr"][lang][x]})
    return rows


def summarize(rows, metric, n_boot=2000, seed=0):
    """Mean, median and 95 % percentile bootstrap interval of the mean over the languages of `rows` (one bit width), and how many languages have a negative / positive difference."""
    v = np.array([r[metric] for r in rows], dtype=float)
    rng = np.random.default_rng(seed)
    means = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)])
    return {"n_languages": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v)), "ci_low": float(np.quantile(means, 0.025)), "ci_high": float(np.quantile(means, 0.975)),
            "n_negative": int((v < 0).sum()), "n_positive": int((v > 0).sum())}


def summarize_by_tier(rows, metric):
    out = {}
    for tier in sorted({r["tier"] for r in rows}):
        v = np.array([r[metric] for r in rows if r["tier"] == tier], dtype=float)
        out[tier] = {"n_languages": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v))}
    return out


def bridge(legacy, other):
    """FP loss, FP firing rates, token counts and the legacy-rounding floors must be identical across arms."""
    fp = max(abs(legacy["fp_loss"][l] - other["fp_loss"][l]) for l in legacy["languages"])
    rate = float((legacy["fp_firing_rate"] - other["fp_firing_rate"]).abs().max())
    floors = [b for b in FLOOR_BITS if b in legacy["floor_corr"] and b in other["floor_corr"] and not torch.equal(legacy["floor_corr"][b], other["floor_corr"][b])]
    same_n = legacy["n"] == other["n"]
    return {"ok": fp == 0.0 and rate == 0.0 and not floors and same_n, "fp_loss_max_abs_diff": fp, "fp_firing_rate_max_abs_diff": rate, "floor_corr_differs": floors, "token_counts_equal": same_n}


def write_report(out, arms, roster, bits, n_boot=2000, seed=0):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    per_language, summary, bridges = [], [], {}
    md = ["# Stage 1 comparison: ordinary loss (E1) and survival (E2), arm against arm", "",
          "Exploratory (D-101). Every row is arm A minus arm B, paired by language (27 pairs); the interval is a 95 % percentile bootstrap of the mean over languages. Loss: per-language mean next-token cross-entropy of the state minus that of FP (nats per token), so a NEGATIVE difference means A is closer to FP. Survival: share of the language's FP-active features whose token correlation with FP is above 0.9 (percentage points), so a POSITIVE difference means A keeps more features. The three pre-set tests T1 to T3 are in the per-arm analysis files, not here.", ""]
    for name, other in (("matched RTN", arms["matched RTN"]), ("GPTQ", arms["GPTQ"])):
        bridges[name] = bridge(arms["legacy RTN"], other)
    md += ["## Bridge (FP numbers and legacy-rounding floors must equal the legacy run's)", "", "| arm | ok | max abs FP loss difference | max abs FP firing-rate difference | floors that differ | token counts equal |", "|---|---|---|---|---|---|"]
    for name, r in bridges.items():
        md.append(f"| {name} | {'yes' if r['ok'] else '**NO**'} | {r['fp_loss_max_abs_diff']:.3g} | {r['fp_firing_rate_max_abs_diff']:.3g} | {r['floor_corr_differs'] or 'none'} | {r['token_counts_equal']} |")
    for title, na, nb in CONTRASTS:
        rows = paired_table(arms[na], arms[nb], roster, bits)
        per_language += [{"contrast": title, **r} for r in rows]
        md += ["", f"## {title}", "", "| bits | metric | mean | median | 95 % interval of the mean | languages with A below B / above B |", "|---|---|---|---|---|---|"]
        for x in bits:
            sub = [r for r in rows if r["bits"] == x]
            for m in METRICS:
                s = summarize(sub, m, n_boot, seed)
                summary.append({"contrast": title, "bits": x, "metric": m, **s})
                unit = "nats/token" if m.startswith("nll") else ("pp" if m.startswith("survived") else "r")
                md.append(f"| INT{x} | {m.replace('_diff', '')} ({unit}) | {s['mean']:+.4f} | {s['median']:+.4f} | [{s['ci_low']:+.4f}, {s['ci_high']:+.4f}] | {s['n_negative']} / {s['n_positive']} |")
    (out / "stage1_comparison.md").write_text("\n".join(md) + "\n")
    for name, rows in (("stage1_paired_per_language.csv", per_language), ("stage1_summary.csv", summary)):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / "bridge.json").write_text(json.dumps(bridges, indent=1) + "\n")
    return bridges


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stage-1 comparison of the GPTQ extension.")
    ap.add_argument("--legacy", required=True); ap.add_argument("--matched", required=True); ap.add_argument("--gptq", required=True); ap.add_argument("--roster", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--bits", type=int, nargs="+", default=[8, 7, 6, 5, 4]); ap.add_argument("--n-boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    arms = {"legacy RTN": load_arm(a.legacy), "matched RTN": load_arm(a.matched), "GPTQ": load_arm(a.gptq)}
    b = write_report(a.out, arms, load_roster(a.roster), a.bits, a.n_boot, a.seed)
    print("bridge:", {k: v["ok"] for k, v in b.items()}, "wrote", a.out)


if __name__ == "__main__":
    main()
