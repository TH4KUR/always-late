"""Figure of the A2 analysis (plan Task 8, `phase1_assoc_margin.py`): the lost fraction of the D-035 association among the features established in full precision, by within-language quintile of the full-precision
margin, one line per precision with Wilson intervals and the rounding-only floor as a reference; the within-language AUROC of the margin in the legend. Same conventions as `phase1_func_figures.py` (light and dark
themes from the validated palette tokens, a one-hue ordinal ramp, direct labels, the status of the data in the subtitle, undefined values named and never drawn as zero). Table twin: `loss_by_margin_quintile.csv`.

    ~/.venvs/anlp-torch/bin/python scripts/g0/phase1_assoc_margin_figures.py --analysis reports/association_retention/results/within-language/analysis.json --out <dir> --note "exploratory"  [--cohort without_excluded]"""
import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

import phase1_func_figures as base

PRECISIONS = ("INT6", "INT5", "INT4")
FLOOR = "INT16"


def _f(x):
    return float("nan") if x is None else float(x)


def _series(res, p):
    bins = res["by_precision"][p]["bins_m0"]
    return (np.array([b["bin"] + 1 for b in bins], dtype=float), np.array([_f(b["fraction"]) for b in bins]), np.array([_f(b["lo"]) for b in bins]), np.array([_f(b["hi"]) for b in bins]))


def fig_loss_by_margin(res, theme, note, precisions=PRECISIONS, floor=FLOOR):
    """One line per precision of `precisions` (a precision absent from `res` is not drawn and is named in the subtitle) over the five within-language margin quintiles, the floor precision as a dashed gray reference."""
    bp = res["by_precision"]; drawn = [p for p in precisions if p in bp]; dropped = [p for p in precisions if p not in bp]
    if not drawn:
        raise ValueError("no precision to plot")
    fig, ax, t = base._new_fig(theme, 7.6, 4.8); cols = dict(zip(precisions, t["ordinal"] if len(precisions) == 3 else t["ordinal4"][:len(precisions)])); tops = []
    if floor in bp:
        x, y, lo, hi = _series(res, floor); tops.append(np.nanmax(hi))
        ax.plot(x, y, color=t["muted"], linestyle="--", linewidth=1.4, zorder=2, gid="floor"); ax.text(0.85, y[0], floor, ha="right", va="center", color=t["ink2"], fontsize=8.5, gid=f"label:{floor}")
    for p in drawn:
        x, y, lo, hi = _series(res, p); tops.append(np.nanmax(hi)); ax.vlines(x, lo, hi, color=cols[p], linewidth=1.4, zorder=2, gid=f"ci:{p}")
        ax.plot(x, y, color=cols[p], marker="o", markersize=5.5, linewidth=2, markeredgecolor=t["surface"], zorder=4, gid=f"line:{p}"); ax.text(0.85, y[0], p, ha="right", va="center", color=t["ink2"], fontsize=8.5, gid=f"label:{p}")
    ax.set_xticks([1, 2, 3, 4, 5]); ax.set_xlim(0.2, 5.4); ax.set_ylim(0.0, float(max(tops)) * 1.08)
    base._axis_labels(ax, t, xlabel="quintile of the full-precision margin within language (1 = most borderline, 5 = best separated)", ylabel="lost fraction of the established features")
    ex = res.get("excluded") or []; undefined = f"; not drawn: {', '.join(dropped)} (not in the results)" if dropped else ""
    extra = base._title(ax, t, "Is the lost association a matter of how borderline the feature already was?",
                        f"lost fraction of the features established in full precision by quintile of the full-precision margin within each language (1 = the most borderline fifth); Wilson 95 % intervals; n = {res['n_established']:,}" + (f", without {', '.join(ex)}" if ex else "") + f"{undefined} · {note}")
    handles = [Line2D([0], [0], color=cols[p], marker="o", markersize=5.5, linewidth=2, markeredgecolor=t["surface"]) for p in drawn]; labels = []
    for p in drawn:
        a = bp[p]["auroc"]["m0"]["mean"]; labels.append(f"{p}, AUROC {a:.2f}" if a is not None and a == a else f"{p}, AUROC n/a")
    if floor in bp:
        handles.append(Line2D([0], [0], color=t["muted"], linestyle="--", linewidth=1.4)); labels.append(f"{floor}, rounding only (the floor)")
    base._legend_below(fig, t, handles, labels, 1.0, 0.25); base._margins(fig, left_in=1.0, bottom_in=1.3, extra=extra)
    return fig


def write_margin_figure(res, out_dir, stem, note, themes=("light", "dark"), formats=("svg", "png"), **kw):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True); paths = []
    for theme in themes:
        paths += base.save_figure(fig_loss_by_margin(res, theme, note, **kw), f"{stem}_{theme}", out, formats)
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description="Figure of the A2 analysis: lost association by margin quintile.")
    ap.add_argument("--analysis", required=True, help="analysis.json of phase1_assoc_margin.py"); ap.add_argument("--out", required=True); ap.add_argument("--note", required=True)
    ap.add_argument("--cohort", choices=["all", "without_excluded"], default="all"); ap.add_argument("--stem", default="fig_loss_by_margin"); ap.add_argument("--themes", nargs="*", default=["light", "dark"]); ap.add_argument("--formats", nargs="*", default=["svg", "png"])
    a = ap.parse_args(argv); res = json.loads(Path(a.analysis).read_text())[a.cohort]; paths = write_margin_figure(res, a.out, a.stem, a.note, tuple(a.themes), tuple(a.formats))
    print("wrote", len(paths), "figure files to", a.out, "| cohort", a.cohort)


if __name__ == "__main__":
    main()
