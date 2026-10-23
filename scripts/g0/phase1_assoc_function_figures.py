"""Figure of the A3 analysis (plan Task 8, `phase1_assoc_function_link.py`): on the SAME features, how well does the full-precision margin separate the loss of the D-035 association and how well does it separate a change
of function (the F5 write-side similarity W1 below the numerical floor)? One dumbbell per precision: the within-language AUROC of minus the margin for each, 95 % intervals over languages, a reference at 0.5 (no information);
a row without a function-event contrast says so and has no marker. Same conventions as `phase1_func_figures.py` (light and dark themes from the validated palette tokens, direct labels, the status of the data in the
subtitle, undefined values named and never drawn as 0.5). Table twin: `margin_vs_function.csv`.

    ~/.venvs/anlp-torch/bin/python scripts/g0/phase1_assoc_function_figures.py --analysis reports/association_retention/results/margin-function/analysis.json --out <dir> --note "exploratory"  [--cohort without_excluded]"""
import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

import phase1_func_figures as base

PRECISIONS = ("INT8", "INT6", "INT5", "INT4")


def _defined(d):
    return d is not None and d.get("mean") is not None and d["mean"] == d["mean"]


def fig_margin_auroc(res, theme, note, precisions=PRECISIONS):
    """Rows (the first precision on the top) for the precisions of `precisions` present in `res` (an absent one is named in the subtitle); markers for the two AUROCs where defined, a gray link, intervals over languages."""
    bp = res["by_precision"]; rows = [p for p in precisions if p in bp]; dropped = [p for p in precisions if p not in bp]
    if not rows:
        raise ValueError("no precision to plot")
    n = len(rows); y = {p: n - 1 - i for i, p in enumerate(rows)}; fig, ax, t = base._new_fig(theme, 7.6, 4.4); c_a, c_f = t["series"][0], t["series"][1]; lows, highs = [0.5], [0.5]
    ax.axvline(0.5, color=t["muted"], linestyle="--", linewidth=1.2, zorder=1, gid="ref")
    spec = (("auroc_association", "assoc", c_a, "o"), ("auroc_function", "function", c_f, "s")); pts = {}
    for key, tag, col, mk in spec:
        ps = [p for p in rows if _defined(bp[p][key])]; pts[tag] = ps
        if not ps:
            continue
        xs = [float(bp[p][key]["mean"]) for p in ps]; ys = [y[p] for p in ps]; lo = [float(bp[p][key]["lo"]) for p in ps]; hi = [float(bp[p][key]["hi"]) for p in ps]; lows += lo; highs += hi
        ax.hlines(ys, lo, hi, color=col, linewidth=1.8, zorder=2, gid=f"ci:{tag}"); ax.plot(xs, ys, linestyle="none", marker=mk, markersize=7, color=col, markeredgecolor=t["surface"], zorder=4, gid=tag)
    both = [p for p in rows if p in pts["assoc"] and p in pts["function"]]
    if both:
        ax.hlines([y[p] for p in both], [min(bp[p]["auroc_association"]["mean"], bp[p]["auroc_function"]["mean"]) for p in both], [max(bp[p]["auroc_association"]["mean"], bp[p]["auroc_function"]["mean"]) for p in both], color=t["muted"], linewidth=1.0, zorder=1, gid="link")
    xmin, xmax = min(lows) - 0.03, min(max(highs) + 0.03, 1.05); ax.set_xlim(xmin, xmax)
    for p in rows:
        if p not in pts["function"] and p in pts["assoc"]:
            lo_p, hi_p = float(bp[p]["auroc_association"]["lo"]), float(bp[p]["auroc_association"]["hi"]); room_right, room_left = xmax - (hi_p + 0.02), (lo_p - 0.02) - xmin           # the text is about a third of the axis wide: on the side with more room
            if room_right >= room_left:
                ax.text(hi_p + 0.02, y[p], "no function-event contrast", ha="left", va="center", color=t["ink2"], fontsize=8.5, gid=f"undefined:{p}")
            else:
                ax.text(lo_p - 0.02, y[p], "no function-event contrast", ha="right", va="center", color=t["ink2"], fontsize=8.5, gid=f"undefined:{p}")
    top = rows[0]
    for key, tag, text in (("auroc_association", "assoc", "association loss"), ("auroc_function", "function", "function event")):
        if _defined(bp[top][key]):
            ax.text(float(bp[top][key]["mean"]), y[top] + 0.28, text, ha="center", va="bottom", color=t["ink2"], fontsize=8.5, gid=f"label:{tag}")
    ax.set_yticks(range(n)); ax.set_yticklabels(list(reversed(rows))); ax.set_ylim(-0.6, n - 0.2 + 0.4); ax.grid(False, axis="y")
    base._axis_labels(ax, t, xlabel="within-language AUROC of minus the full-precision margin (0.5 = no information)")
    ex = res.get("excluded") or []; undefined = f"; not drawn: {', '.join(dropped)} (not in the results)" if dropped else ""
    extra = base._title(ax, t, "Does the margin that predicts the lost association predict a change of function?",
                        f"within-language AUROC of minus the full-precision margin for the loss of the D-035 association and for the function event (W1 below the numerical floor), on the same features: n = {res['n_joined']:,}" + (f", without {', '.join(ex)}" if ex else "") + f"; 95 % intervals over languages{undefined} · {note}")
    handles = [Line2D([0], [0], color=c_a, marker="o", markersize=7, linewidth=1.8, markeredgecolor=t["surface"]), Line2D([0], [0], color=c_f, marker="s", markersize=7, linewidth=1.8, markeredgecolor=t["surface"]), Line2D([0], [0], color=t["muted"], linestyle="--", linewidth=1.2)]
    base._legend_below(fig, t, handles, ["AUROC for the loss of the D-035 association", "AUROC for the function event (W1 below the floor)", "0.5 = no information"], 1.0, 0.25); base._margins(fig, left_in=0.9, bottom_in=1.3, extra=extra)
    return fig


def write_margin_auroc_figure(res, out_dir, stem, note, themes=("light", "dark"), formats=("svg", "png"), **kw):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True); paths = []
    for theme in themes:
        paths += base.save_figure(fig_margin_auroc(res, theme, note, **kw), f"{stem}_{theme}", out, formats)
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description="Figure of the A3 analysis: AUROC of the margin for the association loss and for the function event.")
    ap.add_argument("--analysis", required=True, help="analysis.json of phase1_assoc_function_link.py"); ap.add_argument("--out", required=True); ap.add_argument("--note", required=True)
    ap.add_argument("--cohort", choices=["all", "without_excluded"], default="all"); ap.add_argument("--stem", default="fig_margin_auroc"); ap.add_argument("--themes", nargs="*", default=["light", "dark"]); ap.add_argument("--formats", nargs="*", default=["svg", "png"])
    a = ap.parse_args(argv); res = json.loads(Path(a.analysis).read_text())[a.cohort]; paths = write_margin_auroc_figure(res, a.out, a.stem, a.note, tuple(a.themes), tuple(a.formats))
    print("wrote", len(paths), "figure files to", a.out, "| cohort", a.cohort)


if __name__ == "__main__":
    main()
