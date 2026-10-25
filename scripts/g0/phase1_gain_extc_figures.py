"""Figure of Study G extension C (the repair near ratio 1 with a larger sample; design section 11). Reads cells.csv of phase1_gain_extc.py, never the item files. Left: the median over the languages of the net
harm (group minus matched controls, nats per token) at each ratio, one line per precision with the 97.5 % interval of the median (bootstrap over the languages, seeded, the same function as the rules); below
0 = turning the group up lowers the loss. Right: the number of languages with a negative net harm, with the line at half of the languages. Same themes and palette as phase1_gain_figures.py.

    ~/.venvs/anlp-torch/bin/python scripts/g0/phase1_gain_extc_figures.py --cells <results>/cells.csv --out <dir> --note "exploratory" """
import argparse
import csv
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

import phase1_func_figures as base
import phase1_gain_extc as ec
import phase1_gain_figures as gf


def load_cells(path):
    """Rows of cells.csv with the numbers converted."""
    with open(path) as fh:
        return [{k: (v if k in ("state", "language") else float(v)) for k, v in r.items()} for r in csv.DictReader(fh)]


def net_by_state_ratio(rows):
    """{state: {ratio: [net harm per language]}} in ladder order of the states and ascending ratios, the languages in the order they first appear; a state or ratio without every language is an error."""
    langs = list(dict.fromkeys(r["language"] for r in rows))
    out = {s: {} for s in gf._states(rows)}
    for r in rows:
        out[r["state"]].setdefault(r["ratio"], {})[r["language"]] = r["net"]
    res = {}
    for s, per in out.items():
        res[s] = {}
        for a in sorted(per):
            missing = [l for l in langs if l not in per[a]]
            if missing:
                raise ValueError(f"state {s}, ratio {a:g}: no net harm for {missing}")
            res[s][a] = [per[a][l] for l in langs]
    return res


def band(values, n_boot=5000, seed=0):
    """Median over the languages and its 97.5 % interval (the function of the rules of the extension)."""
    lo, hi = ec.median_interval(values, n_boot, seed)
    return {"median": float(np.median(values)), "lo": lo, "hi": hi}


def n_negative(values):
    return sum(1 for v in values if v < 0)


def fig_repair_c(rows, theme, note):
    v = net_by_state_ratio(rows); states = list(v)
    ratios = sorted(next(iter(v.values())))
    for s in states:
        if sorted(v[s]) != ratios:
            raise ValueError(f"state {s} lacks ratios {sorted(set(ratios) - set(v[s]))}")
    n = len(next(iter(v[states[0]].values())))
    fig, t = gf._fig(theme, 10.4, 6.1)
    w, h = fig.get_size_inches()
    cols = base._precision_colors(t, len(states))
    xs = list(range(len(ratios)))
    bands = {s: [band(v[s][a]) for a in ratios] for s in states}
    lo = min(0.0, min(b["lo"] for s in states for b in bands[s])); hi = max(0.0, max(b["hi"] for s in states for b in bands[s])); pad = 0.06 * (hi - lo) or 0.001
    ax = fig.add_axes([0.95 / w, 1.2 / h, 4.0 / w, 3.2 / h], gid="panel:median"); gf._style(ax, t)
    ax.axhline(0.0, color=t["muted"], linewidth=0.9, zorder=1)
    for s, c in zip(states, cols):
        poly = ax.fill_between(xs, [b["lo"] for b in bands[s]], [b["hi"] for b in bands[s]], color=c, alpha=0.18, linewidth=0, zorder=2)
        poly.set_gid(f"band:{s}")
        ax.plot(xs, [b["median"] for b in bands[s]], color=c, linewidth=1.8, marker="o", markersize=3.8, zorder=3, gid=f"state:{s}")
    ax.set_ylim(lo - pad, hi + pad); ax.set_xlim(-0.3, len(xs) - 0.7)
    ax.set_title("median net harm over the languages", loc="left", fontsize=9, color=t["ink2"], pad=4)
    ax.set_ylabel("nats per token (band: 97.5 % interval of the median)", color=t["ink2"], fontsize=8.5)
    bx = fig.add_axes([5.7 / w, 1.2 / h, 4.0 / w, 3.2 / h], gid="panel:negative"); gf._style(bx, t)
    bx.plot([-0.3, len(xs) - 0.7], [n / 2.0] * 2, color=t["muted"], linewidth=0.9, linestyle="--", zorder=1, gid="half")
    for s, c in zip(states, cols):
        bx.plot(xs, [n_negative(v[s][a]) for a in ratios], color=c, linewidth=1.8, marker="o", markersize=3.8, zorder=3, gid=f"state:{s}")
    bx.set_ylim(-0.03 * n, 1.03 * n); bx.set_xlim(-0.3, len(xs) - 0.7); bx.yaxis.set_major_locator(MaxNLocator(integer=True))
    bx.set_title("languages with a negative net harm (dashed: half)", loc="left", fontsize=9, color=t["ink2"], pad=4)
    bx.set_ylabel("number of languages", color=t["ink2"], fontsize=8.5)
    for a_ in (ax, bx):
        a_.set_xticks(xs); a_.set_xticklabels([f"{a:g}" for a in ratios]); a_.set_xlabel("ratio applied to the group's activations", color=t["ink2"], fontsize=8.5)
    handles = [Line2D([0], [0], color=c, linewidth=2.2, marker="o", markersize=4) for c in cols]
    leg = fig.legend(handles, states, loc="lower center", ncol=len(states), frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    for tx in leg.get_texts():
        tx.set_color(t["ink2"])
    gf._head(fig, t, "Does turning the language-selective group up slightly lower the loss?",
             f"net harm = loss change of the group scaled by the ratio minus the mean change of the matched controls scaled the same way (nats per token, on each language's own held-out text, {n} languages); "
             f"left: median over the languages with its 97.5 % interval (below 0 = the loss falls); right: how many languages have a negative net harm; {note}", 0.9)
    return fig


def write_figures(rows, out_dir, note, themes=("light", "dark"), formats=("svg", "png")):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True); paths = []
    for theme in themes:
        paths += base.save_figure(fig_repair_c(rows, theme, note), f"fig_gc1_repair_{theme}", out, formats)
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description="Study G extension C figure from cells.csv.")
    ap.add_argument("--cells", required=True); ap.add_argument("--out", required=True); ap.add_argument("--note", required=True)
    ap.add_argument("--themes", nargs="*", default=["light", "dark"]); ap.add_argument("--formats", nargs="*", default=["svg", "png"])
    a = ap.parse_args(argv)
    paths = write_figures(load_cells(a.cells), a.out, a.note, tuple(a.themes), tuple(a.formats))
    print("wrote", len(paths), "figure files to", a.out)


if __name__ == "__main__":
    main()
