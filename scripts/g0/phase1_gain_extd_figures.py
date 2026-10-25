"""Figure of Study G extension D (the optimal gain against the quantization shrinkage; design section 12). Reads alpha.csv and analysis.json of phase1_gain_extd.py and writes the numbers of the analysis into the
subtitle; it recomputes nothing. Left: per language the optimal gain at INT4 (grid optimum: the ratio with the lowest group harm; smooth optimum: the vertex of a quadratic fit) against 1 / g (the inverse of the
shrinkage of the group's activation, baseline G1), with the line alpha = 1 / g. Right: the grid optimum on the even against the odd sentences of the same data (the split half check of the estimator), with the
line of equality. Points that share a value are spread by a small deterministic offset on the logarithmic axis. Same themes and palette as phase1_gain_figures.py.

    ~/.venvs/anlp-torch/bin/python scripts/g0/phase1_gain_extd_figures.py --alpha <results>/alpha.csv --analysis <results>/analysis.json --out <dir> --note "exploratory" """
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

import phase1_func_figures as base
import phase1_gain_figures as gf

JITTER = 0.02                                       # half width of the spread, natural log units
GRID_TICKS = (1.0, 1.1, 1.2, 1.4, 1.7)


def load_alpha(path):
    """Rows of alpha.csv with the numbers converted."""
    with open(path) as fh:
        return [{k: (v if k == "language" else float(v)) for k, v in r.items()} for r in csv.DictReader(fh)]


def jitter(values, w=JITTER):
    """A deterministic offset (log units) per value: members of a group of equal values are spread evenly over [-w, w] in the order they appear, a value alone stays."""
    groups = {}
    for i, v in enumerate(values):
        groups.setdefault(v, []).append(i)
    out = [0.0] * len(values)
    for idx in groups.values():
        if len(idx) > 1:
            for i, o in zip(idx, np.linspace(-w, w, len(idx))):
                out[i] = float(o)
    return out


def summary_text(a):
    """The numbers of the analysis for the subtitle (no hyphens, so that wrapping never splits them)."""
    return (f"rank correlation of the optimum with 1 / g {a['rd1']['rho']:.2f} (one sided permutation p = {a['rd1']['p']:.2f}); split half reliability of the estimator {a['rd4']['sb']:.2f}; "
            f"verdict of the rule: {a['verdict']}")


def _log_axis(ax, t, lo, hi, xticks=None):
    ax.set_yscale("log"); ax.set_ylim(lo, hi)
    ax.set_yticks(list(GRID_TICKS)); ax.set_yticklabels([f"{v:g}" for v in GRID_TICKS]); ax.yaxis.set_minor_locator(NullLocator())
    if xticks is not None:
        ax.set_xscale("log"); ax.set_xticks(xticks); ax.set_xticklabels([f"{v:g}" for v in xticks]); ax.xaxis.set_minor_locator(NullLocator())


def fig_alpha(rows, analysis, theme, note):
    n = int(analysis["n_languages"])
    if len(rows) != n:
        raise ValueError(f"the analysis has {n} languages, the table {len(rows)}")
    fig, t = gf._fig(theme, 10.4, 5.9)
    w, h = fig.get_size_inches()
    tau = np.array([r["tau"] for r in rows]); a = [r["alpha"] for r in rows]; sm = np.array([r["alpha_smooth"] for r in rows]); ev = [r["alpha_even"] for r in rows]; od = [r["alpha_odd"] for r in rows]
    lo, hi = 0.95, 1.85
    ax = fig.add_axes([0.95 / w, 1.2 / h, 4.0 / w, 3.0 / h], gid="panel:optimum"); gf._style(ax, t)
    lim = (min(tau.min(), 1.0) * 0.98, max(tau.max(), 1.7) * 1.02)
    ax.plot(lim, lim, color=t["muted"], linewidth=1.0, linestyle="--", zorder=1, gid="identity")
    ax.scatter(tau, np.array(a) * np.exp(jitter(a)), s=34, color=t["series"][0], edgecolor=t["surface"], linewidth=0.8, alpha=0.9, zorder=3, gid="grid")
    ax.scatter(tau, sm, s=34, marker="D", facecolor="none", edgecolor=t["ink2"], linewidth=1.1, zorder=2, gid="smooth")
    _log_axis(ax, t, lo, hi, xticks=[1.1, 1.2, 1.3]); ax.set_xlim(tau.min() * 0.97, tau.max() * 1.03)
    ax.set_title("optimal gain against 1 / g (dashed: alpha = 1 / g)", loc="left", fontsize=9, color=t["ink2"], pad=4)
    ax.set_xlabel("1 / g: inverse of the INT4 shrinkage of the group's activation", color=t["ink2"], fontsize=8.5); ax.set_ylabel("optimal gain alpha", color=t["ink2"], fontsize=8.5)
    bx = fig.add_axes([5.7 / w, 1.2 / h, 4.0 / w, 3.0 / h], gid="panel:halves"); gf._style(bx, t)
    bx.plot((lo, hi), (lo, hi), color=t["muted"], linewidth=1.0, linestyle="--", zorder=1, gid="identity")
    bx.scatter(np.array(ev) * np.exp(jitter(ev)), np.array(od) * np.exp(jitter(od)), s=34, color=t["series"][0], edgecolor=t["surface"], linewidth=0.8, alpha=0.9, zorder=3, gid="halves")
    _log_axis(bx, t, lo, hi, xticks=list(GRID_TICKS)); bx.set_xlim(lo, hi)
    bx.set_title("grid optimum, even against odd sentences (dashed: equal)", loc="left", fontsize=9, color=t["ink2"], pad=4)
    bx.set_xlabel("alpha on the even sentences", color=t["ink2"], fontsize=8.5); bx.set_ylabel("alpha on the odd sentences", color=t["ink2"], fontsize=8.5)
    handles = [Line2D([0], [0], color=t["series"][0], marker="o", linestyle="none", markersize=6), Line2D([0], [0], color=t["ink2"], marker="D", markerfacecolor="none", linestyle="none", markersize=6)]
    leg = fig.legend(handles, ["grid optimum (ratio with the lowest harm)", "smooth optimum (quadratic fit)"], loc="lower center", ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    for tx in leg.get_texts():
        tx.set_color(t["ink2"])
    gf._head(fig, t, "Is the optimal gain of a language set by how far its group shrank at INT4?",
             f"one point per language ({n} languages); the optimal gain is the ratio among 1, 1.1, 1.2, 1.4, 1.7 with the lowest loss of the group scaled by it (grid) or the vertex of a quadratic fit to the four ratios (smooth), "
             f"on {analysis["n_sentences"]} held out sentences per language; points with the same value are spread vertically by a small fixed amount; {summary_text(analysis)}; {note}", 0.9)
    return fig


def write_figures(rows, analysis, out_dir, note, themes=("light", "dark"), formats=("svg", "png")):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True); paths = []
    for theme in themes:
        paths += base.save_figure(fig_alpha(rows, analysis, theme, note), f"fig_gd1_alpha_{theme}", out, formats)
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description="Study G extension D figure from alpha.csv and analysis.json.")
    ap.add_argument("--alpha", required=True); ap.add_argument("--analysis", required=True); ap.add_argument("--out", required=True); ap.add_argument("--note", required=True)
    ap.add_argument("--themes", nargs="*", default=["light", "dark"]); ap.add_argument("--formats", nargs="*", default=["svg", "png"])
    a = ap.parse_args(argv)
    paths = write_figures(load_alpha(a.alpha), json.loads(Path(a.analysis).read_text()), a.out, a.note, tuple(a.themes), tuple(a.formats))
    print("wrote", len(paths), "figure files to", a.out)


if __name__ == "__main__":
    main()
