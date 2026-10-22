"""Figures and tables for the functional-validation study (D-039/D-040, plan Task 8). Reads the OUTPUTS of phase1_func_analysis.py (analysis.json and the CSVs next to it) and the D-035 language
summary; never the item files. Static figures with the matplotlib object API (no pyplot state), light and dark themes from the validated palette tokens (dataviz skill: reference palette, slots 1 to 3 pass
the all-pairs gates in both modes; the ordinal blue ramp passes --ordinal), SVG with live text and PNG. Design rules fixed in the plan (Task 8, figure design): one accent hue (slot 1) against gray for the
reference, never two y axes, undefined values are reported and never drawn as zero, values outside the axis are drawn on its edge and counted, every figure states its split in the title.

    python3 scripts/g0/phase1_func_figures.py --results reports/functional_validation/results/pilot --out reports/functional_validation/figures/pilot --note "pilot, exploratory"      (torch venv: matplotlib)
"""
import argparse
import csv
import json
import textwrap
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import NullLocator

PRECISIONS = ["INT16", "INT12", "INT8", "INT7", "INT6", "INT5", "INT4"]               # numerical floor to coarsest; INT0 (the identity control) is a table row only
_RAMP_LIGHT = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]    # blue steps 100 to 700 (palette.md)
_THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
              "series": ["#2a78d6", "#eb6834", "#1baf7a"], "ordinal": ["#86b6ef", "#2a78d6", "#104281"], "ordinal4": ["#0d366b", "#1c5cab", "#3987e5", "#86b6ef"], "ramp": _RAMP_LIGHT},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
             "series": ["#3987e5", "#d95926", "#199e70"], "ordinal": ["#184f95", "#3987e5", "#9ec5f4"], "ordinal4": ["#b7d3f6", "#6da7ec", "#2a78d6", "#184f95"], "ramp": _RAMP_LIGHT[::-1]},      # dark: low values recede into the surface, high values are the light end
}
Y_FLOOR = 0.25                                                                         # lower edge of the log retention axis
RESCUE_PATCHES = [("control", "matched\ncontrols"), ("group", "language group\n({K})"), ("all", "all SAE\nfeatures"), ("error", "SAE error\nterm"), ("full", "whole\nresidual")]   # from the least to the most that is replaced
RESCUE_PARTS = [("control", "matched controls"), ("group", "language group"), ("group_minus_control", "group minus controls"), ("all", "all SAE features"), ("error", "SAE error term"), ("full", "whole residual")]
RESCUE_Y_LIMITS = (-0.5, 2.0)                                                          # the data decide the axis inside these limits; an estimate beyond them is drawn on the edge and counted


def theme_tokens(mode):
    if mode not in _THEMES:
        raise ValueError(f"unknown theme {mode!r}; choose from {sorted(_THEMES)}")
    return _THEMES[mode]


def _new_fig(theme, width, height):
    t = theme_tokens(theme)
    fig = Figure(figsize=(width, height), facecolor=t["surface"])
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"]); ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=t["muted"], labelcolor=t["ink2"], length=3, width=1.0, labelsize=8.5)
    ax.grid(True, color=t["grid"], linewidth=0.8, linestyle="-"); ax.set_axisbelow(True)
    return fig, ax, t


def _legend(ax, t, handles, labels, loc):
    leg = ax.legend(handles, labels, loc=loc, frameon=False, fontsize=8.5, handlelength=1.8)
    for text in leg.get_texts():
        text.set_color(t["ink2"])
    return leg


def _title(ax, t, title, subtitle):
    """Title and a muted subtitle (what is plotted, the status of the data). The subtitle is wrapped to the figure width, so a long status note is never clipped; returns the extra top margin
    in inches that the wrapped lines need (pass it to _margins as `extra`)."""
    w_in = ax.figure.get_size_inches()[0]
    sub = textwrap.fill(subtitle, width=max(40, int((w_in - 1.3) * 72 / 4.9)))
    extra = sub.count("\n")
    ax.set_title(title, loc="left", fontsize=10.5, color=t["ink"], pad=22 + 11.5 * extra)
    ax.text(0.0, 1.012, sub, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5, color=t["muted"], linespacing=1.3, gid="subtitle")
    return extra * 0.16


def _margins(fig, top_in=0.75, bottom_in=0.85, left_in=0.95, right_in=0.25, extra=0.0):
    w, h = fig.get_size_inches()
    fig.subplots_adjust(left=left_in / w, right=1 - right_in / w, top=1 - (top_in + extra) / h, bottom=bottom_in / h)


def _axis_labels(ax, t, xlabel=None, ylabel=None):
    if xlabel is not None:
        ax.set_xlabel(xlabel, color=t["ink2"], fontsize=9)
    if ylabel is not None:
        ax.set_ylabel(ylabel, color=t["ink2"], fontsize=9)


def fig_effect_per_language(rows, theme, note, K=None, unit="nats per token"):
    """Full-precision effect per language: E (harm of zero-ablating the language's group beyond its activity-matched controls, nats per token, with its 95 % interval) against the harm of the controls
    themselves. rows: dicts with language, effect, lo, hi, harm_control (effects_by_language.csv, FP rows). Sorted by E, the largest on top; symlog axis (E spans orders of magnitude and an
    interval can cross zero)."""
    if not rows:
        raise ValueError("no rows to plot")
    rows = sorted(rows, key=lambda r: r["effect"]); n = len(rows); y = np.arange(n)
    fig, ax, t = _new_fig(theme, 7.6, 0.27 * n + 2.0)
    eff = np.array([r["effect"] for r in rows]); lo = np.array([r["lo"] for r in rows]); hi = np.array([r["hi"] for r in rows]); ctrl = np.array([r["harm_control"] for r in rows])
    ax.hlines(y, lo, hi, color=t["series"][0], linewidth=2.0, zorder=2, gid="interval")
    ax.vlines(np.concatenate([lo, hi]), np.concatenate([y - 0.14, y - 0.14]), np.concatenate([y + 0.14, y + 0.14]), color=t["series"][0], linewidth=1.6, zorder=2, gid="interval_cap")
    ax.scatter(ctrl, y, s=34, marker="D", color=t["muted"], edgecolors=t["surface"], linewidths=1.2, zorder=3, gid="control")
    ax.scatter(eff, y, s=44, marker="o", color=t["series"][0], edgecolors=t["surface"], linewidths=1.5, zorder=4, gid="effect")
    ax.set_xscale("symlog", linthresh=0.01, linscale=0.6)
    left = min(0.0, float(lo.min()) * 1.3); right = float(hi.max()) * 1.6
    ax.set_xlim(left, right)
    ticks = [v for v in (0, 0.01, 0.1, 1, 10) if left <= v <= right]
    ax.set_xticks(ticks); ax.set_xticklabels([f"{v:g}" for v in ticks])
    ax.set_yticks(y); ax.set_yticklabels([r["language"] for r in rows]); ax.set_ylim(-0.8, n - 0.2)
    ax.grid(False, axis="y")
    for k in (0, n - 1):                                                               # direct labels on the extremes only
        ax.text(max(eff[k], hi[k]) * 1.25 + (0.002 if hi[k] <= 0.01 else 0), y[k], f"{eff[k]:.3g}", va="center", fontsize=8.5, color=t["ink2"], gid="label")
    _axis_labels(ax, t, xlabel=f"harm beyond matched controls, {unit} (symmetric-log axis: linear below 0.01)")
    extra = _title(ax, t, "Removing a language's feature group: harm beyond matched controls", f"full precision, {n} languages{', group ' + K if K else ''} \u00b7 {note}")
    handles = [Line2D([0], [0], color=t["series"][0], marker="o", markersize=6, linewidth=2, markeredgecolor=t["surface"]), Line2D([0], [0], color=t["muted"], marker="D", markersize=5, linewidth=0)]
    _legend(ax, t, handles, ["group effect E (95 % interval)", "matched-control harm"], "lower right")
    _margins(fig, left_in=1.0, bottom_in=0.8, extra=extra)
    return fig


def _jitter(k):
    return ((k * 0.6180339887) % 1.0 - 0.5) * 0.36                                      # deterministic, within +-0.18 of the precision's position


def fig_retention(retention, bands, theme, note):
    """Retention by precision: per language the ratio E(b) / E(FP) as gray context, the macro mean of the ratios (primary, D-041) and the median ratio with their intervals, the preserved band
    as a wash, log axis. retention: analysis.json['retention'] ({precision: {macro, pooled, languages}}). Undefined languages are not plotted and are counted; a ratio at or below the axis floor is
    drawn on the floor and counted."""
    precs = [p for p in PRECISIONS if p in retention]
    if not precs:
        raise ValueError("no precision to plot")
    fig, ax, t = _new_fig(theme, 7.6, 5.4)
    langs = sorted({l for p in precs for l in retention[p]["languages"]}); order = {l: k for k, l in enumerate(langs)}
    ok_x, ok_y, floor_x = [], [], []
    undefined = set()
    for i, p in enumerate(precs):
        for l, d in retention[p]["languages"].items():
            if d.get("ratio") is None:
                undefined.add(l); continue
            x = i + _jitter(order[l])
            if d["ratio"] <= Y_FLOOR:
                floor_x.append(x)
            else:
                ok_x.append(x); ok_y.append(d["ratio"])
    top = max([retention[p]["macro"]["hi"] for p in precs if retention[p]["macro"].get("hi")] + [retention[p]["macro"].get("median_hi") or 0 for p in precs] + ok_y + [1.6 / 1.25]) * 1.25
    ax.axhspan(bands[0], bands[1], color=t["series"][0], alpha=0.10, linewidth=0, zorder=1, gid="band")
    ax.axhline(1.0, color=t["axis"], linewidth=1.0, zorder=1, gid="one")
    ax.scatter(ok_x, ok_y, s=14, color=t["muted"], alpha=0.65, linewidths=0, zorder=2, gid="language")
    ax.scatter(floor_x, [Y_FLOOR] * len(floor_x), s=22, marker="v", color=t["muted"], linewidths=0, zorder=2, clip_on=False, gid="language_floor")
    xs = np.arange(len(precs))
    mac = [retention[p]["macro"] for p in precs]; mx = xs - 0.1
    med = [retention[p]["macro"] for p in precs]; dx = xs + 0.1
    ax.plot(mx, [m["ratio"] for m in mac], color=t["series"][0], linewidth=1.6, zorder=3, gid="macro_line")
    ax.vlines(mx, [max(m["lo"], Y_FLOOR) for m in mac], [m["hi"] for m in mac], color=t["series"][0], linewidth=2.0, zorder=3, gid="macro_interval")
    for ends in ([max(m["lo"], Y_FLOOR) for m in mac], [m["hi"] for m in mac]):
        ax.hlines(ends, mx - 0.05, mx + 0.05, color=t["series"][0], linewidth=1.6, zorder=3, gid="macro_cap")
    ax.scatter(mx, [m["ratio"] for m in mac], s=56, marker="o", color=t["series"][0], edgecolors=t["surface"], linewidths=1.5, zorder=4, gid="macro")
    ax.vlines(dx, [max(m["median_lo"], Y_FLOOR) for m in med], [m["median_hi"] for m in med], color=t["series"][1], linewidth=2.0, zorder=3, gid="median_interval")
    for ends in ([max(m["median_lo"], Y_FLOOR) for m in med], [m["median_hi"] for m in med]):
        ax.hlines(ends, dx - 0.05, dx + 0.05, color=t["series"][1], linewidth=1.6, zorder=3, gid="median_cap")
    ax.scatter(dx, [m["median_ratio"] for m in med], s=50, marker="D", color=t["series"][1], edgecolors=t["surface"], linewidths=1.5, zorder=4, gid="median")
    ax.set_yscale("log"); ax.set_ylim(Y_FLOOR, top); ax.set_xlim(-0.6, len(precs) - 0.4)
    yt = [v for v in (0.25, 0.5, 0.75, 1, 1.33, 2, 3, 5, 10) if Y_FLOOR <= v <= top]
    ax.set_yticks(yt); ax.set_yticklabels([f"{v:g}" for v in yt]); ax.yaxis.set_minor_locator(NullLocator()); ax.set_xticks(xs); ax.set_xticklabels(precs)
    ax.grid(False, axis="x")
    _axis_labels(ax, t, xlabel="weight precision (round-to-nearest, coarser to the right)", ylabel="E at the precision / E at full precision (log axis)")
    extra = _title(ax, t, "Effect of the language feature groups, relative to full precision", f"ratio of the group's effect E at each precision to E at full precision \u00b7 {note}")
    handles = [Line2D([0], [0], color=t["muted"], marker="o", markersize=4, linewidth=0, alpha=0.7), Line2D([0], [0], color=t["series"][0], marker="o", markersize=6, linewidth=2, markeredgecolor=t["surface"]),
               Line2D([0], [0], color=t["series"][1], marker="D", markersize=5, linewidth=2, markeredgecolor=t["surface"]), Patch(facecolor=t["series"][0], alpha=0.10, linewidth=0)]
    labels = ["language ratio E(b) / E(FP)", "macro mean of ratios (primary), 95 % interval", "median ratio, 95 % interval", f"preserved band [{bands[0]:.2f}, {bands[1]:.2f}]"]
    _legend(ax, t, handles, labels, "upper left")
    notes = []
    if undefined:
        notes.append(f"{len(undefined)} language{'s' if len(undefined) != 1 else ''} undefined (no clearly positive full-precision effect), not plotted")
    if floor_x:
        notes.append(f"{len(floor_x)} value{'s' if len(floor_x) != 1 else ''} below the axis floor ({Y_FLOOR:g}) drawn on it")
    if notes:
        fig.text(0.012, 0.012, "; ".join(notes) + ".", fontsize=8, color=t["muted"], va="bottom", ha="left")
    _margins(fig, bottom_in=0.95, extra=extra)
    return fig



def _ladder(present):
    return [p for p in ["FP"] + PRECISIONS if p in present]


SIZE_FAMILIES = (("E_K16", "K16", "size_K16"), ("E_K64", "K64", "size_K64"), ("E_all", "all features", "size_all"))


def fig_size_curves(family_rows, theme, note):
    """Median over languages of E (harm beyond matched controls) by group size (K16, K64, all features) and precision, from families.csv. The size is an ordered quantity, so the three lines wear the
    ordinal one-hue ramp; direct end labels; the axis starts at zero."""
    by = {}
    for r in family_rows:
        if r["family"] in {f for f, _, _ in SIZE_FAMILIES}:
            by.setdefault(r["family"], {}).setdefault(r["precision"], []).append(r["harm"])
    precs = _ladder({p for d in by.values() for p in d})
    if not by or not precs:
        raise ValueError("no E_<size> rows to plot")
    n_lang = len({r["language"] for r in family_rows if r["family"] in by})
    fig, ax, t = _new_fig(theme, 7.0, 4.8)
    top = 0.0
    for k, (fam, label, gid) in enumerate(SIZE_FAMILIES):
        if fam not in by:
            continue
        pts = [(i, float(np.median(by[fam][p]))) for i, p in enumerate(precs) if p in by[fam]]
        x, y = zip(*pts); top = max(top, max(y))
        ax.plot(x, y, color=t["ordinal"][k], linewidth=2.0, marker="o", markersize=6, markeredgecolor=t["surface"], markeredgewidth=1.5, zorder=3, gid=gid)
        ax.text(x[-1] + 0.14, y[-1], label, va="center", ha="left", fontsize=8.5, color=t["ink2"], gid="end_label")
    ax.set_xlim(-0.4, len(precs) - 1 + 1.15); ax.set_ylim(0.0, top * 1.15)
    ax.set_xticks(np.arange(len(precs))); ax.set_xticklabels(precs); ax.grid(False, axis="x")
    _axis_labels(ax, t, xlabel="weight precision (round-to-nearest, coarser to the right)", ylabel="median harm beyond matched controls, nats per token")
    extra = _title(ax, t, "Effect of removing the language group, by group size and precision", f"median over {n_lang} languages of E for 16, 64 and all selective features \u00b7 {note}")
    _margins(fig, bottom_in=0.85, extra=extra)
    return fig


def fig_assoc_vs_function(assoc, retention, theme, note):
    """Association retention (D-035: share of the features established as language-selective in full precision that still are) next to the functional macro retention of the group effect E (D-041),
    both indexed to full precision = 1 on ONE axis. Association is a descriptive count (no interval drawn); function carries its bootstrap interval."""
    common = [p for p in PRECISIONS if p in assoc and p in retention]
    if not common:
        raise ValueError("no precision present in both the association and the function results")
    fig, ax, t = _new_fig(theme, 7.0, 4.8)
    xs = np.arange(len(common) + 1)
    m = [retention[p]["macro"] for p in common]
    func = [1.0] + [d["ratio"] for d in m]; lo = [1.0] + [d["lo"] for d in m]; hi = [1.0] + [d["hi"] for d in m]
    asc = [1.0] + [assoc[p]["macro"] for p in common]
    ax.axhline(1.0, color=t["axis"], linewidth=1.0, zorder=1, gid="one")
    ax.vlines(xs, lo, hi, color=t["series"][0], linewidth=2.0, zorder=2, gid="function_interval")
    ax.plot(xs, func, color=t["series"][0], linewidth=2.0, marker="o", markersize=6, markeredgecolor=t["surface"], markeredgewidth=1.5, zorder=3, gid="function")
    ax.plot(xs, asc, color=t["series"][1], linewidth=2.0, marker="s", markersize=5.5, markeredgecolor=t["surface"], markeredgewidth=1.5, zorder=3, gid="association")
    ax.text(xs[-1] + 0.12, func[-1], "function", va="center", fontsize=8.5, color=t["ink2"], gid="end_label")
    ax.text(xs[-1] + 0.12, asc[-1], "association", va="center", fontsize=8.5, color=t["ink2"], gid="end_label")
    ax.set_yscale("log"); bottom = min(asc) * 0.85; top = max(hi) * 1.5; ax.set_ylim(bottom, top); ax.set_xlim(-0.4, xs[-1] + 1.1)                  # headroom: the two-line legend sits in the upper left, above the highest interval
    yt = [v for v in (0.5, 0.7, 1, 1.5, 2, 3) if bottom <= v <= top]
    ax.set_yticks(yt); ax.set_yticklabels([f"{v:g}" for v in yt]); ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xticks(xs); ax.set_xticklabels(["FP"] + common); ax.grid(False, axis="x")
    _axis_labels(ax, t, xlabel="weight precision (round-to-nearest, coarser to the right)", ylabel="relative to full precision (= 1, log axis)")
    extra = _title(ax, t, "Association retention and functional retention across precision", f"association (D-035, among features established in full precision) and function (group effect E, D-041) \u00b7 {note}")
    handles = [Line2D([0], [0], color=t["series"][0], marker="o", markersize=6, linewidth=2, markeredgecolor=t["surface"]), Line2D([0], [0], color=t["series"][1], marker="s", markersize=5.5, linewidth=2, markeredgecolor=t["surface"])]
    _legend(ax, t, handles, ["function: macro retention of E (95 % interval)", "association: share of established features retained, macro over languages"], "upper left")
    _margins(fig, bottom_in=0.85, extra=extra)
    return fig


def fig_fidelity(fidelity_rows, theme, note):
    """SAE fidelity (fraction of variance unexplained at layer 12) per language and precision as a sequential heatmap; languages sorted by the full-precision value; cells that were not measured are
    left blank and counted."""
    if not fidelity_rows:
        raise ValueError("no fidelity rows")
    precs = _ladder({r["precision"] for r in fidelity_rows})
    fp = {r["language"]: r["fvu"] for r in fidelity_rows if r["precision"] == "FP"}
    langs = sorted({r["language"] for r in fidelity_rows}, key=lambda l: (l not in fp, fp.get(l, 0.0), l))
    arr = np.full((len(langs), len(precs)), np.nan)
    for r in fidelity_rows:
        if r["precision"] in precs:
            arr[langs.index(r["language"]), precs.index(r["precision"])] = r["fvu"]
    missing = int(np.isnan(arr).sum())
    fig, ax, t = _new_fig(theme, 6.4, 0.22 * len(langs) + 2.4)
    cmap = LinearSegmentedColormap.from_list("fidelity", t["ramp"]); cmap.set_bad(t["surface"])
    im = ax.imshow(np.ma.masked_invalid(arr), aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_xticks(np.arange(len(precs))); ax.set_xticklabels(precs); ax.set_yticks(np.arange(len(langs))); ax.set_yticklabels(langs)
    ax.grid(False)
    ax.set_xticks(np.arange(-0.5, len(precs), 1), minor=True); ax.set_yticks(np.arange(-0.5, len(langs), 1), minor=True)
    ax.grid(which="minor", color=t["surface"], linewidth=1.6); ax.tick_params(which="minor", length=0)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("FVU (fraction of variance unexplained)", color=t["ink2"], fontsize=9); cb.outline.set_visible(False)
    cb.ax.tick_params(colors=t["muted"], labelcolor=t["ink2"], labelsize=8.5)
    _axis_labels(ax, t, xlabel="weight precision (round-to-nearest, coarser to the right)")
    extra = _title(ax, t, "SAE fidelity at layer 12 across precision and language", f"fraction of variance unexplained (lower is better), languages sorted by the full-precision value \u00b7 {note}")
    if missing:
        fig.text(0.012, 0.012, f"{missing} cell{'s' if missing != 1 else ''} not measured (left blank).", fontsize=8, color=t["muted"], va="bottom", ha="left")
    _margins(fig, left_in=1.0, right_in=0.7, bottom_in=0.85, extra=extra)
    return fig


RANK_BINS = ["1-16", "17-64", "65-256", "257+"]


def fig_rank_bins(rows, theme, note):
    """Association retention (D-035: share of the features established in full precision that are still language-selective) by the rank of the feature within its language (dev mass; A1), pooled over
    languages, with Wilson 95 % intervals as a wash; full precision = 1 by definition. The four ordered bins wear the validated four-step ordinal ramp, the top bin the strongest step. rows:
    rank_table.csv (condition, bin, fraction, lo, hi); an empty bin is a gap, never a zero."""
    if not rows:
        raise ValueError("no rank-table rows")
    present = {r["condition"] for r in rows}
    precs = ["FP"] + [p for p in PRECISIONS if p in present]
    if len(precs) < 2:
        raise ValueError("no precision of the ladder in the rank table")
    fig, ax, t = _new_fig(theme, 7.0, 4.8)
    xs = np.arange(len(precs)); lows = []
    for k, b in enumerate(RANK_BINS):
        y, lo, hi = [1.0], [1.0], [1.0]
        for p in precs[1:]:
            r = next((x for x in rows if x["condition"] == p and x["bin"] == b), None)
            ok = r is not None and r["fraction"] is not None
            y.append(r["fraction"] if ok else np.nan); lo.append(r["lo"] if ok else np.nan); hi.append(r["hi"] if ok else np.nan)
        lows += [v for v in lo if v == v]
        ax.fill_between(xs, lo, hi, color=t["ordinal4"][k], alpha=0.12, linewidth=0, zorder=2, gid=f"band_{b}")
        ax.plot(xs, y, color=t["ordinal4"][k], linewidth=2.0, marker="o", markersize=5.5, markeredgecolor=t["surface"], markeredgewidth=1.5, zorder=3, gid=f"rank_{b}")
        last = max((i for i, v in enumerate(y) if v == v), default=0)
        ax.text(xs[last] + 0.12, y[last], f"ranks {b}", va="center", ha="left", fontsize=8.5, color=t["ink2"], gid="end_label")
    bottom, top = min(lows) - 0.04, 1.02
    ax.set_ylim(bottom, top); ax.set_xlim(-0.4, xs[-1] + 1.25)
    yt = [v for v in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0) if bottom <= v <= top]
    ax.set_yticks(yt); ax.set_yticklabels([f"{v:.1f}" for v in yt]); ax.set_xticks(xs); ax.set_xticklabels(precs); ax.grid(False, axis="x")
    _axis_labels(ax, t, xlabel="weight precision (round-to-nearest, coarser to the right)", ylabel="fraction still language-selective")
    extra = _title(ax, t, "Association retention by rank of the feature within its language", f"among the features established in full precision, ranked by dev mass; pooled over languages, Wilson 95 % intervals \u00b7 {note}")
    handles = [Line2D([0], [0], color=t["ordinal4"][k], marker="o", markersize=5.5, linewidth=2, markeredgecolor=t["surface"]) for k in range(len(RANK_BINS))]
    _legend(ax, t, handles, [f"ranks {b}" for b in RANK_BINS], "lower left")
    _margins(fig, bottom_in=0.85, extra=extra)
    return fig


def _precision_colors(t, n):
    """n steps of the ordinal ramp for the precisions of the ladder, finer to coarser, the coarsest strongest (the 4-step ramp is stored strongest first, hence reversed)."""
    if n == 1:
        return [t["ordinal"][1]]
    if n == 2:
        return [t["ordinal"][1], t["ordinal"][2]]
    if n == 3:
        return list(t["ordinal"])
    if n == 4:
        return t["ordinal4"][::-1]
    return [t["ramp"][i] for i in np.linspace(1, len(t["ramp"]) - 1, n).round().astype(int)]


def _rescue_cells(rescue, precs):
    """{(precision, patch): macro_fraction dict or None} for the five patches of the figure; None = undefined (no language with a clear gap, or the patch was not run)."""
    return {(p, part): ((rescue[p].get("macro_fraction") or {}).get(part) or None) for p in precs for part, _ in RESCUE_PATCHES}


def fig_rescue(rescue, K, theme, note):
    """Plan Task 8, figure 4: the share of the quantization-induced loss (NLL of the quantized run minus NLL of the full-precision run, on the language's own text) that is recovered when full-precision
    activations are patched into the quantized model at layer 12. x = what is patched (matched controls, the language group, all SAE features, the SAE error term, the whole residual: from the least to the
    most that is replaced), dodged by precision (ordinal ramp, coarser stronger); macro mean over the languages with a clearly positive gap with its 95 % interval. The patches overlap (the group is part of all
    features) and the effects are not additive, so nothing is stacked. rescue: the `rescue` block of analysis.json. Undefined cells are gaps and are counted, never zeros; estimates beyond the axis limits are
    drawn on the edge and counted."""
    precs = [p for p in PRECISIONS if p in rescue]
    if not precs:
        raise ValueError("no precision of the ladder in the rescue block")
    cells = _rescue_cells(rescue, precs); defined = [m for m in cells.values() if m is not None]
    bottom = max(RESCUE_Y_LIMITS[0], min(-0.1, min((m["lo"] for m in defined), default=0.0) - 0.05)); top = min(RESCUE_Y_LIMITS[1], max(1.1, max((m["hi"] for m in defined), default=1.0) + 0.05))
    n_edge = sum(1 for m in defined if any(not bottom <= m[k] <= top for k in ("point", "lo", "hi"))); n_undef = len(cells) - len(defined)
    fig, ax, t = _new_fig(theme, 7.6, 4.9)
    n = len(precs); w = min(0.24, 0.8 / n); colors = _precision_colors(t, n); handles, labels = [], []
    for j, p in enumerate(precs):
        off = (j - (n - 1) / 2) * w; xs, ys, los, his, counts = [], [], [], [], []
        for i, (part, _) in enumerate(RESCUE_PATCHES):
            m = cells[(p, part)]
            if m is None:
                continue
            xs.append(i + off); ys.append(min(max(m["point"], bottom), top)); los.append(min(max(m["lo"], bottom), top)); his.append(min(max(m["hi"], bottom), top)); counts.append(m.get("n_languages"))
        if xs:
            ax.vlines(xs, los, his, color=colors[j], linewidth=2.0, zorder=2, gid=f"interval_{p}")
            ax.hlines(los + his, np.array(xs + xs) - 0.04, np.array(xs + xs) + 0.04, color=colors[j], linewidth=1.6, zorder=2, gid=f"interval_cap_{p}")
            ax.scatter(xs, ys, s=44, color=colors[j], edgecolors=t["surface"], linewidths=1.5, zorder=4, gid=f"rescue_{p}")
        c = [v for v in counts if v is not None]
        labels.append(f"{p} (n = {min(c)})" if c and min(c) == max(c) else f"{p} (n = {min(c)} to {max(c)})" if c else f"{p} (n = 0)")
        handles.append(Line2D([0], [0], color=colors[j], marker="o", markersize=6, linewidth=2, markeredgecolor=t["surface"]))
    ax.axhline(0, color=t["axis"], linewidth=1.0, zorder=1, gid="reference_0"); ax.axhline(1, color=t["muted"], linewidth=1.2, zorder=1, gid="reference_1")
    xr = len(RESCUE_PATCHES) - 0.5
    ax.set_xlim(-0.6, xr + 1.4); ax.set_ylim(bottom, top)
    for v, text in ((0, "nothing recovered"), (1, "whole gap recovered")):
        ax.text(xr + 0.05, v, text, ha="left", va="center", fontsize=8.5, color=t["ink2"], gid="reference_label")
    ax.set_xticks(range(len(RESCUE_PATCHES))); ax.set_xticklabels([lab.format(K=K) for _, lab in RESCUE_PATCHES]); ax.grid(False, axis="x")
    yt = [v for v in np.arange(-0.5, 2.01, 0.25) if bottom - 1e-9 <= v <= top + 1e-9]
    ax.set_yticks(yt); ax.set_yticklabels([f"{v:g}" for v in yt])
    _axis_labels(ax, t, xlabel="what is patched from the full-precision model", ylabel="share of the quantization loss recovered")
    sub = (f"layer 12, language group = {K}; share of the NLL gap to full precision on the language's own text, macro mean over the languages with a clearly positive gap, 95 % interval; "
           "the patches overlap and are not additive")
    if n_undef:
        sub += f"; {n_undef} cell{'s' if n_undef != 1 else ''} without a defined fraction (not drawn)"
    if n_edge:
        sub += f"; {n_edge} estimate{'s' if n_edge != 1 else ''} beyond the axis (drawn on its edge)"
    extra = _title(ax, t, "Share of the quantization loss recovered by patching full-precision activations", f"{sub} \u00b7 {note}")
    _legend(ax, t, handles, labels, "upper left")
    _margins(fig, bottom_in=0.95, extra=extra)
    return fig


def rescue_table(rescue):
    """Table twin of figure 4: per precision of the ladder and per part (the five patches and the difference group minus controls) the languages with a defined fraction, the languages with rescue
    configurations, and the macro mean with its interval; undefined cells are None."""
    rows = []
    for p in [p for p in PRECISIONS if p in rescue]:
        mf = rescue[p].get("macro_fraction") or {}; n_total = len(rescue[p].get("languages") or {})
        for part, name in RESCUE_PARTS:
            m = mf.get(part) or None
            rows.append({"precision": p, "part": part, "patch": name, "n_languages": m.get("n_languages") if m else None, "n_total": n_total,
                         "fraction": m["point"] if m else None, "lo": m["lo"] if m else None, "hi": m["hi"] if m else None})
    return rows


def fig_f4(res, theme, note):
    """Plan Task 8, figure 9 (F4, exploratory): per language the harm of zero-ablating the features whose language association was LOST at precision b minus the harm of zero-ablating activity-matched features that
    RETAINED it (nats per token on the language's own text), at b (accent, with its 95 % interval) and, for the same subsets, at full precision (gray, thinner), joined by a connector so that a difference that pre-exists
    the quantization is visible; rows sorted by the contrast at b with the largest on top, the macro mean over the languages above a gap row. res: f4_results.json of phase1_func_f4_analysis.py. The axis is symmetric-log
    (contrasts can be small and of either sign); the two notes in the gap row describe the axis and claim nothing."""
    per, langs = res["per_language"], list(res["languages"])
    if not langs:
        raise ValueError("no languages in the F4 result")
    b = res["condition"]; n = len(langs); order = sorted(langs, key=lambda c: per[c]["E_b"]["point"]); y = np.arange(n); ym = n + 1
    col = lambda key, f: np.array([per[c][key][f] for c in order])
    eb, eb_lo, eb_hi, ef, ef_lo, ef_hi = col("E_b", "point"), col("E_b", "lo"), col("E_b", "hi"), col("E_fp", "point"), col("E_fp", "lo"), col("E_fp", "hi")
    mb, mf = res["macro"]["E_b"], res["macro"]["E_fp"]
    fig, ax, t = _new_fig(theme, 7.6, 0.27 * (n + 2) + 2.4)
    ax.hlines(y, ef, eb, color=t["axis"], linewidth=1.0, zorder=1, gid="connector")
    for gid, color, lw, pos, lo, hi in (("fp", t["muted"], 1.4, y, ef_lo, ef_hi), ("native", t["series"][0], 2.0, y, eb_lo, eb_hi)):
        ax.hlines(pos, lo, hi, color=color, linewidth=lw, zorder=2, gid=f"interval_{gid}")
        ax.vlines(np.concatenate([lo, hi]), np.concatenate([pos - 0.14, pos - 0.14]), np.concatenate([pos + 0.14, pos + 0.14]), color=color, linewidth=lw * 0.8, zorder=2, gid=f"interval_cap_{gid}")
    ax.scatter(ef, y, s=30, marker="D", color=t["muted"], edgecolors=t["surface"], linewidths=1.2, zorder=3, gid="f4_fp")
    ax.scatter(eb, y, s=44, marker="o", color=t["series"][0], edgecolors=t["surface"], linewidths=1.5, zorder=4, gid="f4_native")
    for gid, m, color, marker, size in (("fp", mf, t["muted"], "D", 46), ("native", mb, t["series"][0], "o", 70)):
        ax.hlines([ym], [m["lo"]], [m["hi"]], color=color, linewidth=2.6 if gid == "native" else 1.8, zorder=2, gid=f"interval_macro_{gid}")
        ax.scatter([m["point"]], [ym], s=size, marker=marker, color=color, edgecolors=t["ink"] if gid == "native" else t["surface"], linewidths=1.2, zorder=4, gid=f"f4_macro_{gid}")
    ax.axvline(0, color=t["muted"], linewidth=1.2, zorder=1, gid="reference_zero")
    every = np.concatenate([eb_lo, ef_lo, [mb["lo"], mf["lo"]]]); top = np.concatenate([eb_hi, ef_hi, [mb["hi"], mf["hi"]]])
    left = float(every.min()) * 2.6 if every.min() < 0 else -0.003; right = float(top.max()) * 2.6 if top.max() > 0 else 0.003      # room for the value labels beyond the intervals
    ax.set_xscale("symlog", linthresh=0.01, linscale=0.6); ax.set_xlim(left, right)
    ticks = [v for v in (-10, -1, -0.1, -0.01, 0, 0.01, 0.1, 1, 10) if left <= v <= right]
    ax.set_xticks(ticks); ax.set_xticklabels([f"{v:g}" for v in ticks])
    ax.set_yticks(list(y) + [n, ym]); ax.set_yticklabels(order + ["", f"macro mean (n = {n})"]); ax.set_ylim(-0.8, ym + 0.8); ax.grid(False, axis="y")
    ax.yaxis.get_major_ticks()[n].tick1line.set_visible(False)                                      # the gap row has no tick
    lab = ax.get_yticklabels()[-1]; lab.set_fontweight("bold"); lab.set_color(t["ink"])
    for k, ha in ((0, "right"), (n - 1, "left")):                                                   # direct labels on the extremes only
        x = (eb_lo[k] * 1.12) if ha == "right" else (eb_hi[k] * 1.12 + (0.002 if eb_hi[k] <= 0.01 else 0))
        ax.text(x, y[k], f"{eb[k]:+.3g}", va="center", ha=ha, fontsize=8.5, color=t["ink2"], gid="label")
    for x, ha, text in ((0.01, "left", "\u2190 lost features matter less"), (0.99, "right", "lost features matter more \u2192")):
        ax.text(x, n, text, transform=ax.get_yaxis_transform(), ha=ha, va="center", fontsize=8.5, color=t["muted"], gid="axis_note")
    _axis_labels(ax, t, xlabel="harm of lost minus retained features, nats per token (symlog axis, linear below 0.01)")
    extra = _title(ax, t, "Ablating features by their association status, matched on native activity",
                   f"{b}, {n} languages; harm of zero-ablating the features whose association was lost minus harm of activity-matched features that kept it (nats per token, own text); "
                   f"negative = the lost features matter less; 95 % intervals \u00b7 {note}")
    handles = [Line2D([0], [0], color=t["series"][0], marker="o", markersize=6, linewidth=2, markeredgecolor=t["surface"]), Line2D([0], [0], color=t["muted"], marker="D", markersize=5, linewidth=1.4, markeredgecolor=t["surface"])]
    _legend(ax, t, handles, [f"contrast at {b} (native)", "same subsets at full precision"], "lower right")
    _margins(fig, left_in=1.55, bottom_in=0.95, extra=extra)
    return fig


_F4_PARTS = (("E_b", "E_b"), ("E_fp", "E_fp"), ("delta", "delta"))


def f4_table(res):
    """Table twin of figure 9: a row per language (the contrast at b, at full precision and their paired difference with the 95 % intervals, the harms, the signs) and the macro mean; missing values are None."""
    rows = []
    for c in res["languages"]:
        p = res["per_language"][c]; r = {"language": c, "n_languages": None}
        for key, name in _F4_PARTS:
            r[name] = p[key]["point"]; r[f"{name}_lo"] = p[key]["lo"]; r[f"{name}_hi"] = p[key]["hi"]
        r.update({k: p[k] for k in ("harm_lost_b", "harm_retained_b", "harm_lost_fp", "harm_retained_fp", "n_controls", "sign_b", "sign_fp")})
        rows.append(r)
    m = {"language": "macro mean", "n_languages": len(res["languages"])}
    for key, name in _F4_PARTS:
        m[name] = res["macro"][key]["point"]; m[f"{name}_lo"] = res["macro"][key]["lo"]; m[f"{name}_hi"] = res["macro"][key]["hi"]
    m.update({k: None for k in ("harm_lost_b", "harm_retained_b", "harm_lost_fp", "harm_retained_fp", "n_controls", "sign_b", "sign_fp")})
    return rows + [m]


def _coverage_rows(cal):
    """The rows of the chosen format of a calibration record (phase1_func_mc_calibration.py): language, items, accuracy, Wilson interval, covered. The record's own `covered` flags and lists must agree with the
    rule (lower bound above chance + margin); a languages without items (n = 0) has no accuracy and is never covered."""
    fmt, thr = cal["format"], cal["chance"] + cal["margin"]
    rows = []
    for code, row in cal["languages"].items():
        c = row[fmt]; has = c["n"] > 0
        by_rule = bool(has and c["lo"] > thr)
        if bool(row["covered"]) != by_rule:
            raise ValueError(f"{code}: the record says covered={row['covered']} but the rule (lower bound above {thr:g}) says {by_rule}")
        rows.append({"language": code, "items": int(c["n"]), "accuracy": c["acc"] if has else None, "lo": c["lo"] if has else None, "hi": c["hi"] if has else None, "covered": by_rule})
    if sorted(cal["covered"]) != sorted(r["language"] for r in rows if r["covered"]):
        raise ValueError("the covered list of the record differs from the covered flags of its languages")
    return rows


def coverage_table(cal):
    return _coverage_rows(cal)


def fig_coverage(cal, theme, note):
    """Belebele coverage by language (plan Task 8, figure 7): accuracy of the chosen format on the calibration items with its Wilson 95 % interval, sorted with the highest on top, chance and the coverage
    threshold (chance + margin, from the record) as reference lines; covered languages in the accent (circle), the others in gray (square). Languages without items in the chosen format are listed in the
    subtitle and not drawn."""
    rows = _coverage_rows(cal)
    drawn = sorted((r for r in rows if r["items"] > 0), key=lambda r: r["accuracy"])
    if not drawn:
        raise ValueError("no language has items in the chosen format")
    missing = len(rows) - len(drawn); thr = cal["chance"] + cal["margin"]; n = len(drawn)
    fig, ax, t = _new_fig(theme, 7.4, 0.3 * n + 2.3)
    y = np.arange(n)
    for covered, gid, color, marker, size in ((True, "covered", t["series"][0], "o", 44), (False, "uncovered", t["muted"], "s", 36)):
        sel = [(r, k) for k, r in enumerate(drawn) if r["covered"] == covered]
        if not sel:
            continue
        ks = np.array([k for _, k in sel]); lo = np.array([r["lo"] for r, _ in sel]); hi = np.array([r["hi"] for r, _ in sel]); acc = np.array([r["accuracy"] for r, _ in sel])
        ax.hlines(ks, lo, hi, color=color, linewidth=2.0, zorder=2, gid=f"interval_{gid}")
        ax.vlines(np.concatenate([lo, hi]), np.concatenate([ks - 0.14, ks - 0.14]), np.concatenate([ks + 0.14, ks + 0.14]), color=color, linewidth=1.6, zorder=2, gid=f"interval_cap_{gid}")
        ax.scatter(acc, ks, s=size, marker=marker, color=color, edgecolors=t["surface"], linewidths=1.5, zorder=4, gid=f"coverage_{gid}")
    ax.axvline(cal["chance"], color=t["axis"], linewidth=1.0, zorder=1, gid="chance")
    ax.axvline(thr, color=t["muted"], linewidth=1.2, zorder=1, gid="threshold")
    ax.set_xlim(0.0, 1.0); ax.set_ylim(-0.8, n - 0.2); ax.set_yticks(y); ax.set_yticklabels([r["language"] for r in drawn]); ax.grid(False, axis="y")
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"])
    ax.text(cal["chance"] - 0.008, n - 0.45, f"chance {cal['chance']:g}", ha="right", va="center", fontsize=8.5, color=t["ink2"], gid="reference_label")
    ax.text(thr + 0.008, n - 0.45, f"threshold {thr:.2f}", ha="left", va="center", fontsize=8.5, color=t["ink2"], gid="reference_label")
    for k in (0, n - 1):
        ax.text(min(drawn[k]["hi"] + 0.012, 0.97), y[k], f"{drawn[k]['accuracy']:.2f}", va="center", fontsize=8.5, color=t["ink2"], gid="label")
    _axis_labels(ax, t, xlabel="accuracy on the calibration items, full precision (95 % Wilson interval)")
    n_cov = sum(1 for r in drawn if r["covered"])
    sub = f"{cal['format']} format; {n_cov} of {n} languages covered (lower bound above {thr:.2f})"
    if missing:
        sub += f"; {missing} language{'s' if missing != 1 else ''} without items in this format (not drawn)"
    extra = _title(ax, t, "Which languages the Belebele task can speak for", f"{sub} \u00b7 {note}")
    handles = [Line2D([0], [0], color=t["series"][0], marker="o", markersize=6, linewidth=2, markeredgecolor=t["surface"]), Line2D([0], [0], color=t["muted"], marker="s", markersize=5.5, linewidth=2, markeredgecolor=t["surface"])]
    _legend(ax, t, handles, ["covered", "not covered"], "lower right")
    _margins(fig, left_in=0.95, bottom_in=0.85, extra=extra)
    return fig


def load_rank_table(path):
    return _read_csv(path, {"n_established", "n_retained", "fraction", "lo", "hi", "n_languages"})


def _read_csv(path, floats):
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out.append({k: (None if v == "" else float(v)) if k in floats else v for k, v in row.items()})
    return out


def load_results(results_dir):
    d = Path(results_dir)
    return {"analysis": json.loads((d / "analysis.json").read_text()),
            "effects": _read_csv(d / "effects_by_language.csv", {"effect", "lo", "hi", "harm_group", "harm_control", "norm_ratio_group", "norm_ratio_control"}),
            "families": _read_csv(d / "families.csv", {"harm", "lo", "hi"}) if (d / "families.csv").exists() else []}                    # absent for T-MC


def load_association(path):
    """D-035 association retention per condition from language_summary.csv (group `selective`): macro = mean over languages of retained / FP-established, pooled = sum of retained over sum of FP-established;
    languages without an established feature are skipped."""
    per = {}
    for r in _read_csv(path, {"fp_established", "retained"}):
        if r["group"] != "selective" or not r["fp_established"] or r["language"].isupper():              # POOLED and MACRO_LANGUAGE_MEAN rows (all capitals) are aggregates, not languages
            continue
        per.setdefault(r["condition"], []).append((r["retained"], r["fp_established"]))
    return {c: {"macro": float(np.mean([a / b for a, b in v])), "pooled": sum(a for a, _ in v) / sum(b for _, b in v), "n_languages": len(v)} for c, v in per.items()}


def retention_table(analysis, assoc):
    rows = []
    for p in _ladder(set(analysis["macro"]) | set(analysis["retention"])):
        m = analysis["macro"].get(p, {}); r = analysis["retention"].get(p)
        mac = r["macro"] if r else {}; pooled = r["pooled"] if r else {}
        classes = [d["class"] for d in r["languages"].values()] if r else []
        row = {"precision": p, "mean_E": m.get("point"), "mean_E_lo": m.get("lo"), "mean_E_hi": m.get("hi"), "median_E": m.get("median"),
               "macro_ratio": mac.get("ratio"), "macro_lo": mac.get("lo"), "macro_hi": mac.get("hi"), "class": mac.get("class"),
               "median_ratio": mac.get("median_ratio"), "median_lo": mac.get("median_lo"), "median_hi": mac.get("median_hi"), "pooled_ratio": pooled.get("ratio"),
               "assoc_macro": assoc.get(p, {}).get("macro"), "assoc_pooled": assoc.get(p, {}).get("pooled")}
        for c in ("preserved", "attenuated", "amplified", "inconclusive", "undefined"):
            row[f"n_{c}"] = sum(1 for x in classes if x == c) if r else None
        rows.append(row)
    return rows


def effects_table(effect_rows, analysis, manifest):
    precs = [p for p in PRECISIONS if p in analysis["retention"]]
    rows = []
    for e in effect_rows:
        s = manifest["sizes"].get(e["language"], {})
        row = {"language": e["language"], "size_K16": s.get("K16"), "size_K64": s.get("K64"), "size_all": s.get("all"),
               "effect": e["effect"], "lo": e["lo"], "hi": e["hi"], "harm_control": e["harm_control"]}
        for p in precs:
            row[f"ratio_{p}"] = analysis["retention"][p]["languages"].get(e["language"], {}).get("ratio")
        rows.append(row)
    return rows


def config_table(manifest):
    rows = []
    for l, s in manifest["sizes"].items():
        c = manifest.get("controls", {}).get(l, {})
        rows.append({"language": l, **{f"size_{k}": s.get(k) for k in ("K16", "K64", "all")}, **{f"draws_{k}": (len(c[k]) if k in c else None) for k in ("K16", "K64", "all")}})
    return rows


def to_markdown(rows, columns):
    """columns: [(key, header, format)]; None is written as '-' (undefined or not measured), never as zero."""
    lines = ["| " + " | ".join(h for _, h, _ in columns) + " |", "|" + "---|" * len(columns)]
    for r in rows:
        lines.append("| " + " | ".join("-" if r.get(k) is None else f.format(r[k]) for k, _, f in columns) + " |")
    return "\n".join(lines)


def write_csv(rows, columns, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow([k for k, _, _ in columns])
        for r in rows:
            w.writerow([r.get(k) for k, _, _ in columns])                                         # csv writes None as an empty field


def _retention_columns():
    return [("precision", "precision", "{}"), ("mean_E", "mean E", "{:.3f}"), ("median_E", "median E", "{:.3f}"), ("macro_ratio", "macro retention", "{:.2f}"), ("macro_lo", "lo", "{:.2f}"), ("macro_hi", "hi", "{:.2f}"),
            ("class", "class", "{}"), ("median_ratio", "median ratio", "{:.2f}"), ("median_lo", "lo", "{:.2f}"), ("median_hi", "hi", "{:.2f}"), ("pooled_ratio", "pooled ratio", "{:.2f}"),
            ("n_preserved", "preserved", "{}"), ("n_attenuated", "attenuated", "{}"), ("n_amplified", "amplified", "{}"), ("n_inconclusive", "inconclusive", "{}"), ("n_undefined", "undefined", "{}"),
            ("assoc_macro", "association (D-035, macro)", "{:.3f}"), ("assoc_pooled", "association (pooled)", "{:.3f}")]


def _effects_columns(precs):
    return [("language", "language", "{}"), ("size_K16", "K16", "{}"), ("size_K64", "K64", "{}"), ("size_all", "all", "{}"), ("effect", "E", "{:.3f}"), ("lo", "lo", "{:.3f}"), ("hi", "hi", "{:.3f}"),
            ("harm_control", "control harm", "{:.3f}")] + [(f"ratio_{p}", f"E ratio {p}", "{:.2f}") for p in precs]


_RESCUE_COLUMNS = [("precision", "precision", "{}"), ("part", "part", "{}"), ("patch", "patch", "{}"), ("n_languages", "languages with a defined fraction", "{}"), ("n_total", "languages with rescue configurations", "{}"),
                   ("fraction", "share of the gap recovered (macro mean)", "{:.3f}"), ("lo", "lo", "{:.3f}"), ("hi", "hi", "{:.3f}")]
_F4_COLUMNS = [("language", "language", "{}"), ("n_languages", "languages", "{}"), ("E_b", "contrast at the precision", "{:+.4f}"), ("E_b_lo", "lo", "{:+.4f}"), ("E_b_hi", "hi", "{:+.4f}"),
               ("E_fp", "contrast at full precision", "{:+.4f}"), ("E_fp_lo", "lo", "{:+.4f}"), ("E_fp_hi", "hi", "{:+.4f}"), ("delta", "difference", "{:+.4f}"), ("delta_lo", "lo", "{:+.4f}"), ("delta_hi", "hi", "{:+.4f}"),
               ("harm_lost_b", "harm lost at the precision", "{:.4f}"), ("harm_retained_b", "harm retained at the precision", "{:.4f}"), ("harm_lost_fp", "harm lost, full precision", "{:.4f}"),
               ("harm_retained_fp", "harm retained, full precision", "{:.4f}"), ("n_controls", "retained draws", "{}"), ("sign_b", "sign at the precision", "{}"), ("sign_fp", "sign at full precision", "{}")]
_COVERAGE_COLUMNS = [("language", "language", "{}"), ("items", "items", "{}"), ("accuracy", "accuracy", "{:.3f}"), ("lo", "lo", "{:.3f}"), ("hi", "hi", "{:.3f}"), ("covered", "covered", "{}")]
_CONFIG_COLUMNS = [("language", "language", "{}"), ("size_K16", "K16", "{}"), ("size_K64", "K64", "{}"), ("size_all", "all", "{}"), ("draws_K16", "controls K16", "{}"), ("draws_K64", "controls K64", "{}"), ("draws_all", "controls all", "{}")]


def save_figure(fig, stem, out_dir, formats):
    paths = []
    with matplotlib.rc_context({"svg.fonttype": "none", "svg.hashsalt": "phase1-func-figures"}):
        for ext in formats:
            path = Path(out_dir) / f"{stem}.{ext}"
            fig.savefig(path, format=ext, facecolor=fig.get_facecolor(), dpi=200, metadata={"Date": None} if ext == "svg" else {"Software": None})
            paths.append(str(path))
    return paths


def main(argv=None):
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Figures and tables of the functional-validation study from the analysis outputs.")
    ap.add_argument("--results", required=True, help="directory with analysis.json, effects_by_language.csv, families.csv (phase1_func_analysis.py output)")
    ap.add_argument("--out", required=True); ap.add_argument("--note", required=True, help="status of the data in every title, e.g. 'pilot, exploratory'")
    ap.add_argument("--assoc", default=str(root / "reports/association_retention/results/eq/language_summary.csv")); ap.add_argument("--manifest", default=str(root / "reports/functional_validation/groups_manifest.json"))
    ap.add_argument("--calibration", default=str(root / "reports/functional_validation/belebele_calibration.json"), help="Belebele calibration record (figure 7 and its table); skipped when absent")
    ap.add_argument("--calibration-note", default="calibration split (dev-origin passages), full precision: a pipeline check and the basis of the coverage rule, not a result", help="status of the data of figure 7")
    ap.add_argument("--unit", default="nats per token", help="unit of the effect (T-MC: 'nats per item')")
    ap.add_argument("--rank-note", default="exploratory, descriptive (D-035 statuses measured on devtest)", help="status of the data of figure 8, which is not the data of the other figures")
    ap.add_argument("--rank-table", default=str(root / "reports/functional_validation/results/assoc_rank/rank_table.csv"), help="A1 rank_table.csv (figure 8); skipped when absent")
    ap.add_argument("--f4", nargs="*", default=[], help="directories with f4_results.json (phase1_func_f4_analysis.py output; one per precision): figure 9 and table_f4_<precision> for each")
    ap.add_argument("--f4-note", default="exploratory (the D-035 statuses were measured on devtest), evaluation split, frozen settings", help="status of the data of figure 9, which is not the data of the other figures")
    ap.add_argument("--themes", nargs="*", default=["light", "dark"], choices=sorted(_THEMES)); ap.add_argument("--formats", nargs="*", default=["svg", "png"], choices=["svg", "png"])
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res = load_results(a.results); an = res["analysis"]; K = an.get("K")
    fp_rows = [r for r in res["effects"] if r["precision"] == "FP"]
    assoc = load_association(a.assoc) if Path(a.assoc).exists() else {}
    rank_rows = load_rank_table(a.rank_table) if Path(a.rank_table).exists() else []
    cal = json.loads(Path(a.calibration).read_text()) if Path(a.calibration).exists() else None
    manifest = json.loads(Path(a.manifest).read_text())
    written = []
    for theme in a.themes:
        figs = [("fig1_effect_per_language", lambda: fig_effect_per_language(fp_rows, theme, a.note, K=K, unit=a.unit)), ("fig2_retention", lambda: fig_retention(an["retention"], an["bands"], theme, a.note)),
                ("fig3_group_size", lambda: fig_size_curves(res["families"], theme, a.note)), ("fig4_rescue", lambda: fig_rescue(an["rescue"], K, theme, a.note)), ("fig5_association_vs_function", lambda: fig_assoc_vs_function(assoc, an["retention"], theme, a.note)),
                ("fig6_sae_fidelity", lambda: fig_fidelity(an["fidelity"], theme, a.note)), ("fig7_belebele_coverage", lambda: fig_coverage(cal, theme, a.calibration_note)), ("fig8_association_by_rank", lambda: fig_rank_bins(rank_rows, theme, a.rank_note))]
        for stem, make in figs:
            has = {"fig5_association_vs_function": bool(assoc), "fig3_group_size": any(r["family"].startswith("E_") for r in res["families"]), "fig4_rescue": bool(an.get("rescue")), "fig6_sae_fidelity": bool(an.get("fidelity")), "fig7_belebele_coverage": cal is not None, "fig8_association_by_rank": bool(rank_rows)}
            if not has.get(stem, True):                                                                          # the inputs of this figure do not exist for this analysis (for example T-MC has no SAE fidelity)
                continue
            written += save_figure(make(), f"{stem}_{theme}", out, a.formats)
    f4s = [json.loads((Path(d) / "f4_results.json").read_text()) for d in a.f4]
    for r in f4s:
        for theme in a.themes:
            written += save_figure(fig_f4(r, theme, a.f4_note), f"fig9_f4_{r['condition']}_{theme}", out, a.formats)
    ret, eff, conf = retention_table(an, assoc), effects_table(fp_rows, an, manifest), config_table(manifest)
    tables = [("table_retention", ret, _retention_columns()), ("table_effects", eff, _effects_columns([p for p in PRECISIONS if p in an["retention"]])), ("table_configurations", conf, _CONFIG_COLUMNS)]
    tables += [(f"table_f4_{r['condition']}", f4_table(r), _F4_COLUMNS) for r in f4s]
    if an.get("rescue"):
        tables.append(("table_rescue", rescue_table(an["rescue"]), _RESCUE_COLUMNS))
    if cal is not None:
        tables.append(("table_coverage", coverage_table(cal), _COVERAGE_COLUMNS))
    for name, rows, cols in tables:
        write_csv(rows, cols, out / f"{name}.csv"); (out / f"{name}.md").write_text(to_markdown(rows, cols) + "\n")
        written += [str(out / f"{name}.csv"), str(out / f"{name}.md")]
    print("wrote", len(written), "files to", out)
    return written


if __name__ == "__main__":
    main()
