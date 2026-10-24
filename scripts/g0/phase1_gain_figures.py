"""Figures of Study G (graded gain; design section 7). Reads the OUTPUTS of phase1_gain_analysis.py (curves.csv, alpha_star.csv) and the G1 baseline.csv; never the item files. Static figures with the matplotlib
object API, light and dark themes and the palette of phase1_func_figures.py (dataviz skill: one accent hue against gray, an ordinal ramp for the precision states, direct labels, never two y axes in one panel, the
status of the data in the subtitle, an undefined cell is a dash and never a zero). The ratio axis is log2 with zero-ablation placed to the left of the axis; the vertical scale of the language panels is the fraction
of each language's own full-precision zero-ablation net harm, so that the shapes of the 27 languages can be compared.

    ~/.venvs/anlp-torch/bin/python scripts/g0/phase1_gain_figures.py --curves <results>/curves.csv --out <dir> --note "exploratory" [--alpha-star <results>/alpha_star.csv --baseline <baseline>/baseline.csv]"""
import argparse
import csv
import math
import textwrap
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.lines import Line2D

import phase1_func_figures as base

LADDER = ("FP", "INT8", "INT7", "INT6", "INT5", "INT4")
ZERO_GAP = 1.5


def load_curves(path):
    """Rows of curves.csv with the numbers converted."""
    with open(path) as fh:
        return [{k: (v if k in ("state", "language") else float(v)) for k, v in r.items()} for r in csv.DictReader(fh)]


def x_positions(ratios):
    """x of every ratio on the log2 axis; zero-ablation (ratio 0) sits ZERO_GAP to the left of the smallest positive ratio."""
    pos = [a for a in ratios if a > 0]
    x = {a: math.log2(a) for a in pos}
    if any(a == 0 for a in ratios):
        x[0.0] = min(x.values()) - ZERO_GAP
    return x


def _states(rows):
    present = {r["state"] for r in rows}
    return [s for s in LADDER if s in present]


def _languages(rows):
    return list(dict.fromkeys(r["language"] for r in rows))


def _cell(rows):
    return {(r["state"], r["language"], r["ratio"]): r for r in rows}


def macro_by_state(rows):
    """{state: {"ratios": [0, ..., ratios] ascending, "group": [...], "control": [...], "net": [...]}}: medians over the languages, in ladder order."""
    out = {}
    for s in _states(rows):
        rs = sorted({r["ratio"] for r in rows if r["state"] == s})
        cols = {k: [float(np.median([r[key] for r in rows if r["state"] == s and r["ratio"] == a])) for a in rs] for k, key in (("group", "group_harm"), ("control", "control_harm"), ("net", "net"))}
        out[s] = {"ratios": rs, **cols}
    return out


def _z_fp(rows):
    cell = _cell(rows)
    return {l: cell[("FP", l, 0.0)]["net"] for l in _languages(rows)}


def normalised_net(rows, state):
    """{language: {ratio: net harm / the language's full-precision zero-ablation net harm}} in `state`."""
    z = _z_fp(rows)
    out = {}
    for r in rows:
        if r["state"] == state:
            out.setdefault(r["language"], {})[r["ratio"]] = r["net"] / z[r["language"]] if z[r["language"]] else float("nan")
    return out


def heat_matrix(rows, ratio):
    """(languages sorted by the FP value, descending; states; matrix [language, state] of net harm at `ratio` over the FP zero-ablation net harm); NaN for a cell that is not in the rows."""
    if not any(r["ratio"] == ratio for r in rows):
        raise ValueError(f"ratio {ratio} is not in the curves")
    states, cell, z = _states(rows), _cell(rows), _z_fp(rows)
    langs = _languages(rows)
    mat = np.full((len(langs), len(states)), np.nan)
    for i, l in enumerate(langs):
        for j, s in enumerate(states):
            c = cell.get((s, l, ratio))
            if c is not None and z[l]:
                mat[i, j] = c["net"] / z[l]
    order = sorted(range(len(langs)), key=lambda i: (-(mat[i, 0] if mat[i, 0] == mat[i, 0] else -np.inf), langs[i]))
    return [langs[i] for i in order], states, mat[order]


# ------------------------------------------------------------------------------------------------------------------------------ layout helpers
def _fig(theme, w, h):
    t = base.theme_tokens(theme)
    fig = Figure(figsize=(w, h), facecolor=t["surface"])
    FigureCanvasAgg(fig)
    return fig, t


def _style(ax, t):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"]); ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=t["muted"], labelcolor=t["ink2"], length=3, width=1.0, labelsize=7.5)
    ax.grid(True, color=t["grid"], linewidth=0.6); ax.set_axisbelow(True)


def _head(fig, t, title, subtitle, top_in):
    """Title and wrapped subtitle at the top left of the figure; returns nothing (the caller leaves `top_in` inches)."""
    w, h = fig.get_size_inches()
    fig.text(0.6 / w, 1 - 0.28 / h, title, ha="left", va="top", fontsize=11, color=t["ink"], gid="title")
    sub = textwrap.fill(subtitle, width=max(40, int((w - 1.2) * 72 / 4.9)))
    fig.text(0.6 / w, 1 - 0.55 / h, sub, ha="left", va="top", fontsize=8.5, color=t["muted"], linespacing=1.3, gid="subtitle")


RATIO_TICKS = (0.0, 0.25, 1.0, 4.0)


def _ratio_ticks(ax, x, ratios):
    ticks = [a for a in RATIO_TICKS if a in x] or sorted(x)[:1]
    ax.set_xticks([x[a] for a in ticks]); ax.set_xticklabels(["0" if a == 0 else ("1/4" if a == 0.25 else f"{a:g}") for a in ticks])


# ------------------------------------------------------------------------------------------------------------------------------ figure 1: one panel per language (FP)
def fig_language_curves(rows, theme, note):
    """27 small panels (FP): the net harm of the group scaled by each ratio, as a fraction of the language's zero-ablation net harm (1 = removing the group), with the zero line and the 1 line."""
    langs = _languages(rows); norm = normalised_net(rows, "FP"); z = _z_fp(rows)
    ratios = sorted(next(iter(norm.values())))
    x = x_positions(ratios)
    ncol = 6; nrow = math.ceil(len(langs) / ncol)
    fig, t = _fig(theme, 11.6, 1.55 * nrow + 1.5)
    vals = [v for l in langs for v in norm[l].values() if v == v]
    lo, hi = min(min(vals), 0.0), max(max(vals), 1.0); pad = 0.06 * (hi - lo); ylim = (lo - pad, hi + pad)
    w, h = fig.get_size_inches()
    left, right, bottom, top = 0.55, 0.2, 0.6, 1.1
    cw, ch = (w - left - right) / ncol, (h - bottom - top) / nrow
    for i, l in enumerate(langs):
        r_, c_ = divmod(i, ncol)
        ax = fig.add_axes([(left + c_ * cw + 0.28) / w, (bottom + (nrow - 1 - r_) * ch + 0.22) / h, (cw - 0.36) / w, (ch - 0.5) / h], gid=f"panel:{l}")
        _style(ax, t)
        ax.axhline(0.0, color=t["muted"], linewidth=0.8, zorder=1); ax.axhline(1.0, color=t["muted"], linewidth=0.8, linestyle="--", zorder=1)
        xs = [x[a] for a in ratios]
        ax.plot(xs, [norm[l][a] for a in ratios], color=t["series"][0], linewidth=1.4, marker="o", markersize=2.6, zorder=3, gid="net")
        ax.set_ylim(*ylim); ax.set_xlim(min(xs) - 0.4, max(xs) + 0.4)
        ax.set_title(f"{l}  (z {z[l]:.3f})", loc="left", fontsize=7.5, color=t["ink2"], pad=2)
        _ratio_ticks(ax, x, ratios)
        if c_ != 0:
            ax.set_yticklabels([])
        if r_ != nrow - 1 and i + ncol < len(langs):
            ax.set_xticklabels([])
    _head(fig, t, "Net harm of scaling a language's selective features, full precision",
          f"each panel: net harm (group minus matched controls) at each ratio as a fraction of the language's own zero-ablation net harm z (in nats per token, in the panel title); x: ratio on a log scale, 0 = zero-ablation; dashed line: 1 = removing the group; {note}", top)
    fig.text(0.5, 0.2 / h, "ratio applied to the group's activations (log scale)", ha="center", va="bottom", fontsize=9, color=t["ink2"])
    return fig


# ------------------------------------------------------------------------------------------------------------------------------ figure 2: macro curves by state
def fig_macro(rows, theme, note):
    """Two panels with the median over the languages by state: group harm and net harm (nats per token), against the ratio; one line per state, coarser states darker."""
    m = macro_by_state(rows); states = list(m)
    fig, t = _fig(theme, 10.4, 5.9)
    cols = base._precision_colors(t, len(states))
    w, h = fig.get_size_inches()
    ratios = m[states[0]]["ratios"]; x = x_positions(ratios)
    panels = []
    for k, (key, title, gid) in enumerate((("group", "group scaled", "panel:group"), ("net", "group minus matched controls", "panel:net"))):
        ax = fig.add_axes([(0.95 + k * 4.75) / w, 1.15 / h, 4.0 / w, 3.2 / h], gid=gid)
        _style(ax, t)
        ax.axhline(0.0, color=t["muted"], linewidth=0.9, zorder=1)
        for s, c in zip(states, cols):
            ys = m[s][key]
            ax.plot([x[a] for a in m[s]["ratios"]], ys, color=c, linewidth=1.8, marker="o", markersize=3.6, zorder=3, gid=f"state:{s}")
        ax.set_xlim(min(x.values()) - 0.4, max(x.values()) + 0.6)
        ax.set_xticks([x[a] for a in ratios if a in (0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)])
        ax.set_xticklabels(["0" if a == 0 else f"{a:g}" for a in ratios if a in (0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)])
        ax.set_title(title, loc="left", fontsize=9, color=t["ink2"], pad=4)
        ax.set_xlabel("ratio applied to the group's activations (log scale, 0 = zero-ablation)", color=t["ink2"], fontsize=8.5)
        ax.set_ylabel("median over languages, nats per token", color=t["ink2"], fontsize=8.5)
        panels.append(ax)
    handles = [Line2D([0], [0], color=c, linewidth=2.2, marker="o", markersize=4) for c in cols]
    leg = fig.legend(handles, states, loc="lower center", ncol=len(states), frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    for tx in leg.get_texts():
        tx.set_color(t["ink2"])
    _head(fig, t, "Dose-response of the language-selective group across precisions",
          f"median over the languages of the harm in nats per token (NLL after the edit minus before, same precision, on the language's own text); left: the group scaled, right: minus the same ratio on matched controls; {note}", 0.9)
    return fig


# ------------------------------------------------------------------------------------------------------------------------------ figure 3: heatmaps language x state
def _text_colour(t, rgb):
    lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return "#0b0b0b" if lum > 0.55 else "#ffffff"


def fig_heatmap(rows, theme, note, down=None, up=None):
    """Language x state at the deepest chosen down ratio and the largest chosen up ratio (the smallest and largest ratio of the rows outside the fine ones unless given): net harm over the language's FP zero-ablation net harm."""
    ratios = sorted({r["ratio"] for r in rows if r["ratio"] > 0 and r["ratio"] not in (1.0, 1.1, 1.2)})
    down = down if down is not None else min(ratios); up = up if up is not None else max(ratios)
    langs0, states, _ = heat_matrix(rows, down)
    fig, t = _fig(theme, 10.4, 0.23 * len(langs0) + 2.35)
    w, h = fig.get_size_inches()
    cmap = LinearSegmentedColormap.from_list("seq", t["ramp"])
    for k, (ratio, gid, title) in enumerate(((down, "panel:down", f"ratio {down:g} (turned down)"), (up, "panel:up", f"ratio {up:g} (turned up)"))):
        langs, st, mat = heat_matrix(rows, ratio)
        langs = langs0; _, _, mat = heat_matrix(rows, ratio); idx = [heat_matrix(rows, ratio)[0].index(l) for l in langs0]; mat = mat[idx]
        ax = fig.add_axes([(1.35 + k * 4.55) / w, 0.55 / h, 3.7 / w, (h - 1.85) / h], gid=gid)
        ax.set_facecolor(t["surface"])
        vmax = float(np.nanmax(mat)) if np.isfinite(mat).any() else 1.0
        ax.imshow(np.nan_to_num(mat, nan=0.0), cmap=cmap, vmin=0.0, vmax=max(vmax, 1e-9), aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                col = _text_colour(t, to_rgb(cmap(0.0 if v != v else min(max(v / max(vmax, 1e-9), 0), 1))))
                ax.text(j, i, "-" if v != v else f"{v:.2f}", ha="center", va="center", fontsize=6.5, color=col if v == v else t["ink2"], gid="cell")
        ax.set_xticks(range(len(st))); ax.set_xticklabels(st, fontsize=8, color=t["ink2"])
        ax.set_yticks(range(len(langs0))); ax.set_yticklabels(langs0 if k == 0 else [], fontsize=7, color=t["ink2"])
        ax.set_title(title, loc="left", fontsize=9, color=t["ink2"], pad=4)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
    _head(fig, t, "Net harm by language and precision, as a fraction of the full-precision zero-ablation net harm",
          f"cell = net harm (group minus matched controls) at the stated ratio divided by the language's own FP zero-ablation net harm; rows sorted by the FP value of the left panel; a dash is a cell that was not measured; {note}", 0.9)
    return fig


# ------------------------------------------------------------------------------------------------------------------------------ figure 4: repair near 1 and the optimal gain
def fig_repair(rows, alpha_star, baseline_g_ratio, theme, note, fine=(1.1, 1.2)):
    """Left: median group harm and median matched-control harm at the fine up ratios by state. Right: the optimal gain alpha*(language, INT4) against 1 / the group's activation ratio at INT4 (baseline G1)."""
    states = _states(rows); cell = _cell(rows); langs = _languages(rows)
    fig, t = _fig(theme, 10.4, 5.9)
    w, h = fig.get_size_inches()
    ax = fig.add_axes([0.95 / w, 1.15 / h, 4.4 / w, 2.9 / h], gid="panel:fine"); _style(ax, t)
    ax.axhline(0.0, color=t["muted"], linewidth=0.9, zorder=1)
    xs = range(len(states))
    styles = {fine[0]: "o", fine[1]: "s"}
    for a in fine:
        g = [float(np.median([cell[(s, l, a)]["group_harm"] for l in langs])) for s in states]
        c = [float(np.median([cell[(s, l, a)]["control_harm"] for l in langs])) for s in states]
        ax.plot(xs, g, color=t["series"][0], marker=styles[a], markersize=4.5, linewidth=1.6, linestyle="-" if a == fine[0] else "--", zorder=3, gid=f"group:{a:g}")
        ax.plot(xs, c, color=t["muted"], marker=styles[a], markersize=4.5, linewidth=1.6, linestyle="-" if a == fine[0] else "--", zorder=2, gid=f"control:{a:g}")
    ax.set_xticks(list(xs)); ax.set_xticklabels(states); ax.set_ylabel("median over languages, nats per token", color=t["ink2"], fontsize=8.5)
    ax.set_title("harm of scaling by 1.1 and 1.2", loc="left", fontsize=9, color=t["ink2"], pad=4)
    handles = [Line2D([0], [0], color=t["series"][0], linewidth=1.8), Line2D([0], [0], color=t["muted"], linewidth=1.8), Line2D([0], [0], color=t["ink2"], marker="o", linewidth=1.4, markersize=4), Line2D([0], [0], color=t["ink2"], marker="s", linestyle="--", linewidth=1.4, markersize=4)]
    leg = fig.legend(handles, ["group", "matched controls", f"x{fine[0]:g}", f"x{fine[1]:g}"], loc="lower left", ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.8 / w, 0.0))
    for tx in leg.get_texts():
        tx.set_color(t["ink2"])
    bx = fig.add_axes([6.0 / w, 1.15 / h, 4.0 / w, 2.9 / h], gid="panel:alpha"); _style(bx, t)
    if baseline_g_ratio is None or "INT4" not in alpha_star:
        bx.text(0.5, 0.5, "no baseline file given:\nalpha* against 1 / g not drawn", ha="center", va="center", transform=bx.transAxes, fontsize=9, color=t["ink2"])
        bx.set_xticks([]); bx.set_yticks([])
    else:
        xv = [1.0 / baseline_g_ratio[l] for l in langs]; yv = [alpha_star["INT4"][l] for l in langs]
        bx.scatter(xv, yv, s=26, color=t["series"][0], edgecolor=t["surface"], linewidth=0.8, zorder=3)
        lim = [min(xv + yv) - 0.05, max(xv + yv) + 0.05]
        bx.plot(lim, lim, color=t["muted"], linewidth=0.9, linestyle="--", zorder=1); bx.set_xlim(*lim); bx.set_ylim(*lim)
        bx.set_xlabel("1 / (group activation at INT4 relative to FP), baseline G1", color=t["ink2"], fontsize=8.5); bx.set_ylabel("optimal gain alpha* at INT4", color=t["ink2"], fontsize=8.5)
    bx.set_title("optimal gain against the activation shrinkage", loc="left", fontsize=9, color=t["ink2"], pad=4)
    _head(fig, t, "Does turning the group up repair the quantized model?",
          f"left: median harm at the two fine up ratios, group and matched controls (below 0 = the loss falls); right: per language, the ratio with the lowest INT4 loss against the inverse of the group's activation change in the baseline (dashed: equal); {note}", 0.9)
    return fig


def write_figures(rows, out_dir, note, themes=("light", "dark"), formats=("svg", "png"), alpha_star=None, baseline=None):
    """Figures 1 to 3 (and figure 4 when `alpha_star` {state: {language: ratio}} is given; `baseline` {language: g_ratio} may be None); returns the file paths."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True); paths = []
    for theme in themes:
        for stem, fig in (("fig_g1_language_curves", fig_language_curves(rows, theme, note)), ("fig_g2_macro", fig_macro(rows, theme, note)), ("fig_g3_heatmap", fig_heatmap(rows, theme, note))):
            paths += base.save_figure(fig, f"{stem}_{theme}", out, formats)
        if alpha_star is not None:
            paths += base.save_figure(fig_repair(rows, alpha_star, baseline, theme, note), f"fig_g4_repair_{theme}", out, formats)
    return paths


def load_alpha_star(path):
    with open(path) as fh:
        out = {}
        for r in csv.DictReader(fh):
            out.setdefault(r["state"], {})[r["language"]] = float(r["alpha_star"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Study G figures from curves.csv (and alpha_star.csv, baseline.csv).")
    ap.add_argument("--curves", required=True); ap.add_argument("--out", required=True); ap.add_argument("--note", required=True)
    ap.add_argument("--alpha-star"); ap.add_argument("--baseline"); ap.add_argument("--themes", nargs="*", default=["light", "dark"]); ap.add_argument("--formats", nargs="*", default=["svg", "png"])
    a = ap.parse_args(argv)
    rows = load_curves(a.curves)
    star = load_alpha_star(a.alpha_star) if a.alpha_star else None
    baseline = None
    if a.baseline:
        import phase1_gain_analysis as ga
        baseline = ga.load_baseline_g_ratio(a.baseline)
    paths = write_figures(rows, a.out, a.note, tuple(a.themes), tuple(a.formats), star, baseline)
    print("wrote", len(paths), "figure files to", a.out)


if __name__ == "__main__":
    main()
