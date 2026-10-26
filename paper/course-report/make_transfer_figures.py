"""Render the intervention-transfer figures of the report from the committed result tables (reports/intervention_transfer/results/main), without rerunning analyses.

    python make_transfer_figures.py [results directory]        (default: the main results; the external results directory gives the external-panel versions with the suffix -ext)
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
plt.rcParams.update({
    "font.family": "DejaVu Serif", "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False, "savefig.pad_inches": 0.03, "axes.linewidth": 0.6,
})
BLUE, ORANGE, GREEN, GREY = "#17629a", "#b95726", "#26835a", "#6b6b6b"


def read(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def save(fig, name):
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / (name + ".pdf"), bbox_inches="tight"); fig.savefig(HERE / "figures" / (name + ".png"), dpi=200, bbox_inches="tight"); plt.close(fig)


def get(rows, **kw):
    hit = [r for r in rows if all(r[k] == v for k, v in kw.items())]
    return hit[0] if hit else None


def terms_figure(res, suffix=""):
    comp, grp, av = read(res / "component_summary.csv"), read(res / "group_per_language.csv"), read(res / "averaged_fingerprints.csv")
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.15), gridspec_kw={"width_ratios": [1.45, 1.0, 0.8], "wspace": 0.42})
    ax = axs[0]; kinds = ["matched_swapped", "random1", "random2", "random3"]; x = 0
    ticks, labels = [], []
    for si, state in enumerate(("INT6", "INT4")):
        for ti, (m, lab) in enumerate((("C_rel", "C"), ("W_rel", "W"), ("I_rel", "I"))):
            r = get(comp, state=state, kind="real", split="all", metric=m)
            ax.errorbar([x], [float(r["point"])], yerr=[[float(r["point"]) - float(r["low"])], [float(r["high"]) - float(r["point"])]], fmt="o", color=BLUE, ms=4, capsize=1.5, lw=0.9, zorder=3, label="real direction" if (si, ti) == (0, 0) else None)
            ys = [float(get(comp, state=state, kind=k, split="all", metric=m)["point"]) for k in kinds]
            ax.scatter([x + 0.17 * (j - 1.5) for j in range(4)], ys, s=9, facecolor="none", edgecolor=GREY, lw=0.7, zorder=2, label="four control directions" if (si, ti) == (0, 0) else None)
            ticks.append(x); labels.append(lab); x += 1
        x += 0.7
    ax.set_xticks(ticks, labels); ax.set_ylabel("median norm / (FP norm + floor)"); ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    for cx, name in ((1, "INT6"), (4.7, "INT4")):
        ax.text(cx, -0.19, name, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7.5)
    ax.legend(frameon=False, loc="upper left", handletextpad=0.2, borderaxespad=0.1); ax.set_title("(a) feature level, eligible items", fontsize=7.5, loc="left")
    ax = axs[1]; lo = 3e-4; vals = []
    for state, col in (("INT6", ORANGE), ("INT4", GREEN)):
        L = [r for r in grp if r["state"] == state]; w = np.maximum(np.abs([float(r["W"]) for r in L]), lo); c = np.maximum(np.abs([float(r["C"]) for r in L]), lo); vals += list(w) + list(c)
        ax.scatter(w, c, s=9, color=col, alpha=0.85, lw=0, label=state)
    hi = max(vals) * 1.4; ax.plot([lo, hi], [lo, hi], color=GREY, lw=0.7, ls="--"); ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("|W| (weights), net harm"); ax.set_ylabel("|C| (context), net harm"); ax.legend(frameon=False, loc="lower right", handletextpad=0.1, markerscale=1.2); ax.set_title("(b) group level, 27 languages", fontsize=7.5, loc="left")
    ax = axs[2]
    for i, state in enumerate(("INT16", "INT6", "INT4")):
        r = get(av, state=state, kind="real", split="all", metric="cos_gap"); p, l, h = float(r["point"]), float(r["low"]), float(r["high"])
        ax.errorbar([i], [p], yerr=[[p - l], [h - p]], fmt="o", color=GREY if state == "INT16" else BLUE, ms=4, capsize=1.5, lw=0.9)
    ax.axhline(0, color="black", lw=0.5); ax.set_xticks([0, 1, 2], ["INT16", "INT6", "INT4"]); ax.set_xlim(-0.5, 2.5); ax.set_ylabel("cosine: averaged minus contexts"); ax.set_title("(c) H4", fontsize=7.5, loc="left")
    save(fig, "transfer-terms" + suffix)


def predict_figure(res, suffix=""):
    pv = read(res / "predictive_validation.csv"); fig, ax = plt.subplots(figsize=(3.35, 1.85), layout="constrained"); y = 0; ticks, labels = [], []
    for state in ("INT16", "INT6", "INT4"):
        for variant, col, lab in (("eligible", BLUE, "eligible items"), ("all_items", GREY, "all items")):
            r = [x for x in pv if x["state"] == state and x["variant"] == variant and x["model"] == "A+B"]
            if not r:
                continue
            r = r[0]; p, l, h = float(r["improvement_diagnostic"]), float(r["improvement_low"]), float(r["improvement_high"])
            ax.errorbar([p], [y], xerr=[[p - l], [h - p]], fmt="o", color=col, ms=3.5, capsize=1.5, lw=0.9, label=lab if state == "INT16" else None); y += 1
        y += 0.5; ticks.append(y - 1.75); labels.append(state)
    ax.axvline(0, color="black", lw=0.5); ax.axvline(0.10, color=ORANGE, lw=0.8, ls="--"); ax.set_yticks(ticks, labels); ax.invert_yaxis(); ax.set_xlabel("held-out error reduction, A+B vs strongest baseline")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handletextpad=0.2, columnspacing=1.0, borderaxespad=0.1); save(fig, "transfer-predict" + suffix)


def main():
    res = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "reports/intervention_transfer/results/main"
    suffix = "-ext" if "external" in res.name else ""
    terms_figure(res, suffix); predict_figure(res, suffix)


if __name__ == "__main__":
    main()
