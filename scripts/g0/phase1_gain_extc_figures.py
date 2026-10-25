"""Figure of Study G extension C (the repair near ratio 1 with a larger sample; design section 11). Reads bands.csv of phase1_gain_extc.py (the median net harm over the languages, its 97.5 % interval and the
number of negative languages per state and ratio, computed there with the language order, draws and seed of the rules: the percentile of a bootstrap of 27 values moves with the order of the values, so this module
never recomputes the interval from cells.csv). Left: the median at each ratio, one line per precision with the interval as a band; below 0 = turning the group up lowers the loss. Right: the number of languages with a
negative net harm, with the line at half of the languages. Same themes and palette as phase1_gain_figures.py.

    ~/.venvs/anlp-torch/bin/python scripts/g0/phase1_gain_extc_figures.py --bands <results>/bands.csv --out <dir> --note "exploratory" """
import argparse
import csv
from pathlib import Path

from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, NullLocator

import phase1_func_figures as base
import phase1_gain_figures as gf


LINTHRESH = 0.002                                   # nats per token: the primary ratios 1.1 and 1.2 move the median by about a thousandth, the larger ratios by ten times more


SYM_TICKS = sorted({s * m for s in (-1, 1) for m in (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)} | {0.0})        # plain decimals on the symmetric log axis, never closer than a fifth of an inch


def load_bands(path):
    """Rows of bands.csv with the numbers converted."""
    with open(path) as fh:
        return [{k: (v if k == "state" else float(v)) for k, v in r.items()} for r in csv.DictReader(fh)]


def series(rows):
    """{state: {ratio: row}} in ladder order of the states and ascending ratios; a state that lacks a ratio of another state, or a row with another number of languages, is an error."""
    if not rows:
        raise ValueError("no rows")
    out = {s: {} for s in gf._states(rows)}
    for r in rows:
        out[r["state"]][r["ratio"]] = r
    ratios = sorted({r["ratio"] for r in rows}); n = {r["n_languages"] for r in rows}
    if len(n) != 1:
        raise ValueError(f"the rows differ in the number of languages: {sorted(n)}")
    for s, per in out.items():
        if sorted(per) != ratios:
            raise ValueError(f"state {s} lacks ratios {sorted(set(ratios) - set(per))}")
    return {s: {a: per[a] for a in ratios} for s, per in out.items()}


def fig_repair_c(rows, theme, note):
    v = series(rows); states = list(v)
    ratios = list(v[states[0]])
    n = int(next(iter(v[states[0]].values()))["n_languages"])
    fig, t = gf._fig(theme, 10.4, 6.1)
    w, h = fig.get_size_inches()
    cols = base._precision_colors(t, len(states))
    xs = list(range(len(ratios)))
    lo = min(0.0, min(r["lo"] for s in states for r in v[s].values())); hi = max(0.0, max(r["hi"] for s in states for r in v[s].values())); pad = 0.06 * (hi - lo) or 0.001
    ax = fig.add_axes([0.95 / w, 1.2 / h, 4.0 / w, 3.2 / h], gid="panel:median"); gf._style(ax, t)
    ax.axhline(0.0, color=t["muted"], linewidth=0.9, zorder=1)
    for s, c in zip(states, cols):
        poly = ax.fill_between(xs, [v[s][a]["lo"] for a in ratios], [v[s][a]["hi"] for a in ratios], color=c, alpha=0.18, linewidth=0, zorder=2)
        poly.set_gid(f"band:{s}")
        ax.plot(xs, [v[s][a]["median"] for a in ratios], color=c, linewidth=1.8, marker="o", markersize=3.8, zorder=3, gid=f"state:{s}")
    ax.set_yscale("symlog", linthresh=LINTHRESH); ax.set_ylim(lo - pad, hi + pad); ax.set_xlim(-0.3, len(xs) - 0.7)
    ticks = [y for y in SYM_TICKS if lo - pad <= y <= hi + pad]
    ax.set_yticks(ticks); ax.set_yticklabels([f"{y:g}" for y in ticks]); ax.yaxis.set_minor_locator(NullLocator())
    ax.set_title("median net harm over the languages", loc="left", fontsize=9, color=t["ink2"], pad=4)
    ax.set_ylabel("nats per token (band: 97.5 % interval of the median)", color=t["ink2"], fontsize=8.5)
    bx = fig.add_axes([5.7 / w, 1.2 / h, 4.0 / w, 3.2 / h], gid="panel:negative"); gf._style(bx, t)
    bx.plot([-0.3, len(xs) - 0.7], [n / 2.0] * 2, color=t["muted"], linewidth=0.9, linestyle="--", zorder=1, gid="half")
    for s, c in zip(states, cols):
        bx.plot(xs, [v[s][a]["n_negative"] for a in ratios], color=c, linewidth=1.8, marker="o", markersize=3.8, zorder=3, gid=f"state:{s}")
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
             f"left: median over the languages with its 97.5 % interval (below 0 = the loss falls; vertical scale linear up to 0.002 and logarithmic beyond); right: how many languages have a negative net harm; {note}", 0.9)
    return fig


def write_figures(rows, out_dir, note, themes=("light", "dark"), formats=("svg", "png")):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True); paths = []
    for theme in themes:
        paths += base.save_figure(fig_repair_c(rows, theme, note), f"fig_gc1_repair_{theme}", out, formats)
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description="Study G extension C figure from bands.csv.")
    ap.add_argument("--bands", required=True); ap.add_argument("--out", required=True); ap.add_argument("--note", required=True)
    ap.add_argument("--themes", nargs="*", default=["light", "dark"]); ap.add_argument("--formats", nargs="*", default=["svg", "png"])
    a = ap.parse_args(argv)
    paths = write_figures(load_bands(a.bands), a.out, a.note, tuple(a.themes), tuple(a.formats))
    print("wrote", len(paths), "figure files to", a.out)


if __name__ == "__main__":
    main()
