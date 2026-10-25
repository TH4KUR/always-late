"""Figure of Study G extension B (state-defined groups; design section 10). Reads compare.csv of phase1_gain_extb.py, never the item files. One panel per precision other than FP (at FP the two groups are the same
set): the median over the languages of the net harm (group minus matched controls, nats per token) of the group selected in that precision and of the FP reference group, at every ratio; same axes, themes and
palette as phase1_gain_figures.py.

    ~/.venvs/anlp-torch/bin/python scripts/g0/phase1_gain_extb_figures.py --compare <results>/compare.csv --out <dir> --note "exploratory" """
import argparse
import csv
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

import phase1_func_figures as base
import phase1_gain_figures as gf


def load_compare(path):
    """Rows of compare.csv with the numbers converted."""
    with open(path) as fh:
        return [{k: (v if k == "state" else float(v)) for k, v in r.items()} for r in csv.DictReader(fh)]


def panel_states(rows):
    """The states to draw: those of the table other than FP, in ladder order."""
    present = {r["state"] for r in rows}
    out = [s for s in gf.LADDER if s in present and s != "FP"]
    if not out:
        raise ValueError("the table has no state other than FP")
    return out


def series(rows, state):
    """{"ratios": ascending (zero-ablation first), "state_defined": [...], "reference": [...]}: the medians over the languages of the net harm of the two groups at `state`."""
    rs = sorted((r for r in rows if r["state"] == state), key=lambda r: r["ratio"])
    return {"ratios": [r["ratio"] for r in rs], "state_defined": [r["median_net_state_defined"] for r in rs], "reference": [r["median_net_reference"] for r in rs]}


def fig_state_defined(rows, theme, note):
    states = panel_states(rows)
    data = {s: series(rows, s) for s in states}
    ratios = sorted({a for d in data.values() for a in d["ratios"]})
    for s, d in data.items():
        if d["ratios"] != ratios:
            raise ValueError(f"state {s} lacks ratios {sorted(set(ratios) - set(d['ratios']))}")
    fig, t = gf._fig(theme, 10.4 if len(states) > 1 else 8.8, 6.1)
    w, h = fig.get_size_inches()
    x = gf.x_positions(ratios)
    vals = [v for d in data.values() for v in d["state_defined"] + d["reference"]]
    lo, hi = min(min(vals), 0.0), max(max(vals), 0.0); pad = 0.05 * (hi - lo); ylim = (lo - pad, hi + pad)
    for k, s in enumerate(states):
        ax = fig.add_axes([(0.95 + k * 4.75) / w, 1.2 / h, 4.0 / w, 3.2 / h], gid=f"panel:{s}")
        gf._style(ax, t)
        ax.axhline(0.0, color=t["muted"], linewidth=0.9, zorder=1)
        xs = [x[a] for a in ratios]
        ax.plot(xs, data[s]["reference"], color=t["muted"], linewidth=1.8, linestyle="--", marker="o", markersize=3.4, zorder=2, gid="reference")
        ax.plot(xs, data[s]["state_defined"], color=t["series"][0], linewidth=1.8, marker="o", markersize=3.6, zorder=3, gid="state_defined")
        ax.set_yscale("symlog", linthresh=0.02); ax.set_ylim(*ylim); ax.set_xlim(min(xs) - 0.4, max(xs) + 0.6)
        gf._ratio_ticks(ax, x, ratios)
        ax.set_title(s, loc="left", fontsize=9, color=t["ink2"], pad=4)
        ax.set_xlabel("ratio applied to the group's activations (log scale, 0 = zero-ablation)", color=t["ink2"], fontsize=8.5)
        if k == 0:
            ax.set_ylabel("median over languages, net harm, nats per token", color=t["ink2"], fontsize=8.5)
    handles = [Line2D([0], [0], color=t["series"][0], linewidth=2.2, marker="o", markersize=4), Line2D([0], [0], color=t["muted"], linewidth=2.2, linestyle="--", marker="o", markersize=4)]
    leg = fig.legend(handles, ["state-defined group (selected in that precision)", "FP reference group (selected at full precision)"], loc="lower center", ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    for tx in leg.get_texts():
        tx.set_color(t["ink2"])
    gf._head(fig, t, "Does selecting the group in the quantized model change its dose-response?",
             f"median over the languages of the net harm (group minus matched controls, nats per token) of the 64 strongest own features of each language selected in the given precision, against the same selection made at full precision "
             f"and run at that precision; same selection text, same evaluation sentences; vertical scale linear up to 0.02 and logarithmic beyond; {note}", 0.9)
    return fig


def write_figures(rows, out_dir, note, themes=("light", "dark"), formats=("svg", "png")):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True); paths = []
    for theme in themes:
        paths += base.save_figure(fig_state_defined(rows, theme, note), f"fig_gb1_state_defined_{theme}", out, formats)
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description="Study G extension B figure from compare.csv.")
    ap.add_argument("--compare", required=True); ap.add_argument("--out", required=True); ap.add_argument("--note", required=True)
    ap.add_argument("--themes", nargs="*", default=["light", "dark"]); ap.add_argument("--formats", nargs="*", default=["svg", "png"])
    a = ap.parse_args(argv)
    paths = write_figures(load_compare(a.compare), a.out, a.note, tuple(a.themes), tuple(a.formats))
    print("wrote", len(paths), "figure files to", a.out)


if __name__ == "__main__":
    main()
