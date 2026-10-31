"""Render the report and slide figures from committed results, without rerunning analyses.

Usage: python make_figures.py            light figures for the report  -> figures/
       python make_figures.py --dark     dark slide-* figures (one idea each) -> ../slides/figures/
(matplotlib is needed; e.g. `uv run --no-project --with matplotlib python make_figures.py`)

All figures use the study's own quantizer (legacy RTN); other quantizers are a robustness check
reported in the appendix table, not plotted.
  fig-harm          FP excess harm per language; per-language effect ratio E_b/E_FP from INT8 to INT4
  fig-dose          graded suppression: median excess harm vs activation multiplier per precision
  fig-dissociation  activation survival vs effect identity; own vs swapped direction; INT4 repair by restoring FP
  fig-terms         context / later-weight / interaction terms of a fixed edit (feature and group level)
  fig-tiers         survival by resource tier across bit widths; selected features per language (appendix)
"""
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
import csv
import json
import statistics
import sys

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPORTS = ROOT / "reports"
FV = REPORTS / "functional_validation"
GX = REPORTS / "gptq_extension/results"
EVAL = {"Legacy RTN": FV / "results/eval"}
RESCUE = {"Legacy RTN": FV / "results/rescue"}

# Blue for language groups, violet for "the feature", yellow for INT4 accents; light and dark steps.
THEMES = {
    "light": dict(out=HERE / "figures", bg="#ffffff", ink="#0b0b0b", muted="#52514e", grid="#d9d8d4",
                  quant=["#2a78d6", "#eb6834", "#1baf7a"], feature="#6250d6", accent="#eda100", bar="#2a78d6"),
    "dark": dict(out=HERE.parent / "slides/figures", bg="#16181d", ink="#f0efec", muted="#c3c2b7", grid="#3a3b3f",
                 quant=["#3987e5", "#d95926", "#199e70"], feature="#9085e9", accent="#c98500", bar="#3987e5"),
}
T = {}


def apply_theme(name):
    T.update(THEMES[name])
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.labelsize": 9, "axes.titlesize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": T["bg"], "axes.facecolor": T["bg"], "savefig.facecolor": T["bg"],
        "axes.edgecolor": T["muted"], "axes.labelcolor": T["ink"], "text.color": T["ink"],
        "xtick.color": T["muted"], "ytick.color": T["muted"], "legend.labelcolor": T["ink"],
        "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlecolor": T["ink"],
        "savefig.pad_inches": 0.02,
    })


def rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def save(fig, name):
    T["out"].mkdir(parents=True, exist_ok=True)
    fig.savefig(T["out"] / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(T["out"] / (name + ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=T["grid"], linewidth=0.5)
    ax.set_axisbelow(True)


def fmt3(v):
    return str(Decimal(repr(v)).quantize(Decimal("0.001"), ROUND_HALF_UP))


def harm_figure(figsize=(6.4, 2.55), name="fig-harm"):
    langs = json.loads((REPORTS / "g0/phase1_languages.json").read_text())["languages"]
    tier = {l["code"]: l["tier"] for l in langs}
    fp = {r["language"]: r for r in rows(EVAL["Legacy RTN"] / "effects_by_language.csv") if r["precision"] == "FP"}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=figsize, width_ratios=[2.1, 1], layout="constrained")

    # (a) FP excess harm, grouped by tier, sorted within tier; log scale (0.01 to 7.6 nats/token)
    x, ticks, names = 0, [], []
    for t in ["high", "mid", "low"]:
        codes = sorted((c for c in fp if tier[c] == t), key=lambda c: -float(fp[c]["effect"]))
        start = x
        for c in codes:
            e, lo, hi = (float(fp[c][k]) for k in ("effect", "lo", "hi"))
            ax.bar(x, e, width=0.72, color=T["bar"])
            ax.plot([x, x], [lo, hi], color=T["ink"], linewidth=0.6)
            ticks.append(x); names.append(c.split("_")[0]); x += 1
        ax.text((start + x - 1) / 2, 14, t.capitalize(), ha="center", va="bottom", fontsize=8.5, color=T["muted"])
        x += 0.8
    ax.set_yscale("log"); ax.set_ylim(0.005, 25)
    ax.set_xticks(ticks, names, rotation=90, fontsize=8)
    ax.tick_params(axis="x", length=0, pad=1)
    ax.set_xlim(-0.7, x - 0.8 + 0.1)
    ax.set_ylabel("Excess harm at FP\n(nats/token, log scale)")
    ax.set_title("(a) Excess harm at FP")
    grid(ax)

    # (b) per-language effect ratio E_b/E_FP across bit widths; median marked; log scale
    ret = rows(EVAL["Legacy RTN"] / "retention.csv")
    states = ["INT8", "INT7", "INT6", "INT5", "INT4"]
    for i, state in enumerate(states):
        r = [float(v["ratio"]) for v in ret if v["precision"] == state and not v["language"].isupper()]
        color = T["accent"] if state == "INT4" else T["bar"]
        bx.scatter([i + ((k % 7) - 3) * 0.06 for k in range(len(r))], r, s=8, color=color, alpha=0.65, linewidths=0)
        m = statistics.median(r)
        bx.plot([i - 0.3, i + 0.3], [m, m], color=T["ink"], linewidth=1.4)
    bx.axhline(1, color=T["muted"], linewidth=0.7, linestyle="--")
    bx.set_yscale("log"); bx.set_ylim(0.5, 8)
    bx.set_yticks([0.5, 1, 2, 4, 8], ["0.5", "1", "2", "4", "8"])
    bx.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    bx.set_xticks(range(5), ["8", "7", "6", "5", "4"]); bx.set_xlabel("Weight precision (bits)")
    bx.set_ylabel("Harm relative to FP (log)")
    bx.set_title("(b) Change with precision")
    bx.text(0.03, 0.97, "bar = median\n1 = same as FP", transform=bx.transAxes, va="top", fontsize=7.5, color=T["muted"])
    grid(bx)
    save(fig, name)


ORDER = ["INT16", "INT8", "INT6", "INT5", "INT4"]
BITS = ["16", "8", "6", "5", "4"]


def panel_identity(ax, title="(a) Activation vs. function"):
    # activation survival versus effect identity (F5 sample, legacy RTN)
    f5 = {r["state"]: r for r in rows(FV / "results/f5/f5_tables.csv") if r["group"] == "all"}
    surv = [100 * float(f5[s]["R1_duan_share"]) for s in ORDER]
    ident = [100 * float(f5[s]["retrieval_top1"]) for s in ORDER]
    ax.plot(range(5), ident, "o-", color=T["feature"], linewidth=2, markersize=4.5, label="Identified from effect")
    ax.plot(range(5), surv, "s-", color=T["accent"], linewidth=2, markersize=4.5, label="Activation survives ($R_1>0.9$)")
    ax.annotate(f"{ident[-1]:.1f}%", (4, ident[-1]), xytext=(0, -11), textcoords="offset points", ha="center", fontsize=8.5)
    ax.annotate(f"{surv[-1]:.1f}%", (4, surv[-1]), xytext=(-6, 0), textcoords="offset points", ha="right", va="center", fontsize=8.5)
    ax.set_xticks(range(5), BITS); ax.set_xlabel("Weight precision (bits)")
    ax.set_ylim(0, 140); ax.set_yticks(range(0, 101, 20)); ax.set_ylabel("Features (%)")
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper left", handlelength=1.6, borderaxespad=0, labelspacing=0.3)
    grid(ax)


def panel_swap(ax, title="(b) Drift is generic"):
    # effect cosine of the real feature versus a swapped other direction (median, IQR)
    lad = {r["state"]: r for r in rows(FV / "figures/f5/ladder_control_table.csv")}
    for stem, name, color, off, mk in [("features", "Feature", T["feature"], -0.07, "o"), ("control", "Swapped", T["muted"], 0.07, "D")]:
        med = [float(lad[s]["median_" + stem]) for s in ORDER]
        lo = [m - float(lad[s]["q25_" + stem]) for s, m in zip(ORDER, med)]
        hi = [float(lad[s]["q75_" + stem]) - m for s, m in zip(ORDER, med)]
        ax.errorbar([i + off for i in range(5)], med, yerr=[lo, hi], fmt=mk + "-", color=color,
                    markersize=4, linewidth=1.5, capsize=2, elinewidth=0.8, label=name)
    ax.set_xticks(range(5), BITS); ax.set_xlabel("Weight precision (bits)")
    ax.set_ylim(0.72, 1.01); ax.set_ylabel("Effect cosine with FP")
    ax.set_title(title)
    ax.legend(frameon=False, loc="lower left", borderaxespad=0.1, handlelength=1.4)
    grid(ax)


def panel_repair(ax, title="(c) Repair by restoring FP"):
    # share of the INT4 loss gap recovered by restoring FP activations
    parts = [("group", "Language group"), ("control", "Control group"), ("all", "All SAE features"), ("full", "Full residual")]
    macro = json.loads((RESCUE["Legacy RTN"] / "analysis.json").read_text())["rescue"]["INT4"]["macro_fraction"]
    for i, (key, _) in enumerate(parts):
        v = 100 * macro[key]["point"]
        ax.barh(i, v, height=0.6, color=T["feature"] if key == "group" else T["muted"])
        ax.text(v + 1.5, i, f"{v:.1f}%", va="center", fontsize=8.5, color=T["ink"])
    ax.set_yticks(range(4), [p[1] for p in parts]); ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 50); ax.set_xticks([0, 20, 40]); ax.set_xlabel("INT4 loss gap recovered (%)")
    ax.set_title(title)
    grid(ax, "x")


def dissociation_figure():
    fig, axs = plt.subplots(1, 3, figsize=(6.4, 2.5), layout="constrained")
    for ax, panel in zip(axs, (panel_identity, panel_swap, panel_repair)):
        panel(ax)
    save(fig, "fig-dissociation")


def dose_figure(figsize=(3.05, 1.9), name="fig-dose"):
    # Study G, legacy RTN: median over 27 languages of net (group minus control) harm per activation multiplier
    data = rows(FV / "gain_study/results/curves.csv")
    mults = [1.0, 0.3535533905932738, 0.1767766952966369, 0.08838834764831845, 0.0]
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    for state, color, style, mk in [("FP", T["ink"], "-", "o"), ("INT6", T["quant"][0], "-", "s"),
                                    ("INT5", T["muted"], "--", "^"), ("INT4", T["accent"], "-", "D")]:
        med = [statistics.median(float(r["net"]) for r in data if r["state"] == state and abs(float(r["ratio"]) - m) < 1e-9) for m in mults]
        ax.plot(range(5), med, style, marker=mk, color=color, linewidth=1.6, markersize=4, label=state)
    ax.set_xticks(range(5), ["1", "0.35", "0.18", "0.09", "0"])
    ax.set_xlabel("Activation multiplier (0 = removed)"); ax.set_ylabel("Median excess harm\n(nats/token)")
    ax.legend(frameon=False, loc="upper left", ncol=2, columnspacing=1.0, borderaxespad=0.1)
    grid(ax)
    save(fig, name)


def panel_tiers(ax, title="(a) Survival by resource tier"):
    # legacy RTN: mean active-feature survival per resource tier (9 languages each), Phase 1 equal-budget stream
    data = [r for r in rows(GX / "stage1/stage1_paired_per_language.csv") if r["contrast"] == "matched RTN minus legacy RTN"]
    bits = ["8", "7", "6", "5", "4"]
    for t, color, mk in (("high", T["quant"][0], "o"), ("mid", T["muted"], "s"), ("low", T["accent"], "D")):
        v = [statistics.mean(float(r["survived_pct_b"]) for r in data if r["bits"] == b and r["tier"] == t) for b in bits]
        ax.plot(range(5), v, "-", marker=mk, color=color, linewidth=1.6, markersize=4, label=t.capitalize() + " resource")
    ax.set_xticks(range(5), bits); ax.set_xlabel("Weight precision (bits)")
    ax.set_ylim(0, 105); ax.set_ylabel("Features surviving (%)")
    ax.set_title(title)
    ax.legend(frameon=False, loc="lower left", borderaxespad=0.1)
    grid(ax)


def feature_counts():
    langs = json.loads((REPORTS / "g0/phase1_languages.json").read_text())["languages"]
    sizes = json.loads((FV / "groups_manifest.json").read_text())["sizes"]
    return {l["code"]: (l["tier"], l["script"], sizes[l["code"]]["all"]) for l in langs}


def spearman(x, y):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y); mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return num / (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5


def count_summary():
    # descriptive, post hoc: does the number of selected features follow resource tier, script or survival?
    c = feature_counts(); codes = sorted(c)
    n = [c[k][2] for k in codes]
    tier = [{"high": 3, "mid": 2, "low": 1}[c[k][0]] for k in codes]
    stage1 = [r for r in rows(GX / "stage1/stage1_paired_per_language.csv") if r["contrast"] == "matched RTN minus legacy RTN"]
    surv = {b: {r["language"]: float(r["survived_pct_b"]) for r in stage1 if r["bits"] == b} for b in ("6", "4")}
    scripts = [c[k][1] for k in codes]
    own = [x for x, s in zip(n, scripts) if scripts.count(s) == 1]
    shared = [x for k, x, s in zip(codes, n, scripts) if scripts.count(s) > 1 and k != "eng_Latn"]
    print("feature counts: rho(tier)=%.2f rho(INT6 survival)=%.2f rho(INT4 survival)=%.2f; median own-script %.1f (n=%d), shared-script excl. English %.1f (n=%d)" % (
        spearman(n, tier), spearman(n, [surv["6"][k] for k in codes]), spearman(n, [surv["4"][k] for k in codes]),
        statistics.median(own), len(own), statistics.median(shared), len(shared)))


def panel_counts(ax, title="(b) Selected features per language"):
    c = feature_counts()
    x, ticks, names = 0, [], []
    for t in ["high", "mid", "low"]:
        codes = sorted((k for k in c if c[k][0] == t), key=lambda k: -c[k][2])
        start = x
        for k in codes:
            ax.bar(x, c[k][2], width=0.72, color=T["bar"])
            ticks.append(x); names.append(k.split("_")[0]); x += 1
        ax.text((start + x - 1) / 2, 2600, t.capitalize(), ha="center", va="bottom", fontsize=8.5, color=T["muted"])
        x += 0.8
    ax.set_yscale("log"); ax.set_ylim(50, 5000)
    ax.set_yticks([100, 300, 1000], ["100", "300", "1000"])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks(ticks, names, rotation=90, fontsize=8)
    ax.tick_params(axis="x", length=0, pad=1)
    ax.set_xlim(-0.7, x - 0.8 + 0.1)
    ax.set_ylabel("Selected features (log)")
    ax.set_title(title)
    grid(ax)


def tier_figure():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.4, 2.25), width_ratios=[1, 1.55], layout="constrained")
    panel_tiers(ax)
    panel_counts(bx)
    save(fig, "fig-tiers")
    count_summary()


def terms_figure():
    res = REPORTS / "intervention_transfer/results/main"
    comp, grp = rows(res / "component_summary.csv"), rows(res / "group_per_language.csv")

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.4, 2.1), width_ratios=[1.35, 1], layout="constrained")
    panel_terms_features(ax, comp)
    panel_terms_groups(bx, grp)
    save(fig, "fig-terms")


def panel_terms_features(ax, comp):
    def get(**kw):
        return next(r for r in comp if all(r[k] == v for k, v in kw.items()))

    # (a) feature level: median relative size of each term, real direction vs four control directions
    kinds = ["matched_swapped", "random1", "random2", "random3"]
    x, ticks, labels = 0, [], []
    for si, state in enumerate(("INT6", "INT4")):
        for ti, (m, lab) in enumerate((("C_rel", "C"), ("W_rel", "W"), ("I_rel", "I"))):
            r = get(state=state, kind="real", split="all", metric=m)
            p, lo, hi = float(r["point"]), float(r["low"]), float(r["high"])
            ax.errorbar([x], [p], yerr=[[p - lo], [hi - p]], fmt="o", color=T["feature"], markersize=5, capsize=2,
                        linewidth=1, zorder=3, label="Feature direction" if (si, ti) == (0, 0) else None)
            ys = [float(get(state=state, kind=k, split="all", metric=m)["point"]) for k in kinds]
            ax.scatter([x + 0.15 * (j - 1.5) for j in range(4)], ys, s=12, facecolor="none", edgecolor=T["muted"],
                       linewidths=0.8, zorder=2, label="Control directions" if (si, ti) == (0, 0) else None)
            ticks.append(x); labels.append(lab); x += 1
        x += 0.6
    ax.set_xticks(ticks, labels, fontsize=8.5)
    ax.tick_params(axis="x", length=0)
    for cx_, name in ((1, "INT6"), (4.6, "INT4")):
        ax.text(cx_, -0.2, name, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8.5, fontweight="bold")
    ax.set_ylim(0, 0.8); ax.set_ylabel("Median size relative to\nFP effect")
    ax.set_title("(a) Single features: output change (C, W, I)")
    ax.legend(frameon=False, loc="upper left", borderaxespad=0.1)
    grid(ax)


def panel_terms_groups(bx, grp, title="(b) Language groups: loss"):
    # group level: |context term| vs |weight term| of the language-group loss, one dot per language
    floor = 3e-4
    for state, color in (("INT6", T["muted"]), ("INT4", T["accent"])):
        L = [r for r in grp if r["state"] == state]
        bx.scatter([max(abs(float(r["W"])), floor) for r in L], [max(abs(float(r["C"])), floor) for r in L],
                   s=14, color=color, linewidths=0, alpha=0.9, label=state)
    bx.plot([floor, 3], [floor, 3], color=T["muted"], linewidth=0.7, linestyle="--")
    bx.set_xscale("log"); bx.set_yscale("log"); bx.set_xlim(floor, 3); bx.set_ylim(floor, 3)
    bx.set_xlabel("Later-weight term (nats/token)"); bx.set_ylabel("Context term (nats/token)")
    bx.set_title(title)
    bx.text(0.04, 0.96, "above line:\ncontext larger", transform=bx.transAxes, va="top", fontsize=7.5, color=T["muted"])
    bx.legend(frameon=False, loc="lower right", borderaxespad=0.1, handletextpad=0.1)


def slide_panels():
    # one idea per slide: single panels of the multi-panel report figures
    for name, panel in (("slide-identity", panel_identity), ("slide-swap", panel_swap), ("slide-repair", panel_repair)):
        fig, ax = plt.subplots(figsize=(3.6, 2.3), layout="constrained")
        panel(ax, title="")
        save(fig, name)
    fig, ax = plt.subplots(figsize=(3.3, 2.3), layout="constrained")
    panel_terms_groups(ax, rows(REPORTS / "intervention_transfer/results/main/group_per_language.csv"), title="")
    save(fig, "slide-terms")
    harm_figure((5.6, 2.5), "slide-harm")
    dose_figure((3.4, 2.2), "slide-dose")
    for name, panel, size in (("slide-tiers", panel_tiers, (3.4, 2.2)), ("slide-counts", panel_counts, (4.4, 2.3))):
        fig, ax = plt.subplots(figsize=size, layout="constrained")
        panel(ax, title="")
        save(fig, name)


def main():
    dark = "--dark" in sys.argv
    apply_theme("dark" if dark else "light")
    if dark:
        slide_panels()
        return
    for draw in (harm_figure, dose_figure, dissociation_figure, terms_figure, tier_figure):
        draw()


if __name__ == "__main__":
    main()
