"""Figures of the F5 study (plan Task 14; design docs/superpowers/specs/2026-10-23-feature-function-equivalence-design.md): does a feature do the same thing in a quantized model, and is the survival metric therefore
sensible? Reads the files of the runner (per-feature arrays) and the output of `phase1_func_f5_analysis.py` (tables, floor, statements). Same conventions as `phase1_func_figures.py` (matplotlib object API, light
and dark themes from the validated palette tokens, one accent hue against gray, undefined values reported and never drawn as zero, every title states the status of the data), whose helpers it reuses.

    ~/.venvs/anlp-torch/bin/python scripts/g0/phase1_func_f5_figures.py --run <job dir> [--run <job dir 2>] --out <dir> --note "exploratory, smoke sample"

Figures: 1 the distribution of W1 (cosine of the effect fingerprints against full precision) by state with the numerical floor and the matched negative control; 2 read side against write side per feature with
the baseline survival cutoff and the floor, coloured by the D-035 status; 3 the three statements S1 to S3 with their rules and verdicts; 4 retrieval of a feature's own identity; 5 medians of the read side and the
write side by state; 6 the paired difference of W1 between retained and lost features by language."""
import argparse
import json
import textwrap
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import phase1_func_f5_analysis as an
import phase1_func_figures as base

STATE_ORDER = ["FP_rebatched", "INT16", "INT8", "INT6", "INT5", "INT4"]
REFERENCE_STATES = ("FP_rebatched", "INT16")                                              # the two noise-only states that define the numerical floor: drawn as references, never joined to the precision ladder
LABELS = {"FP_rebatched": "FP repeat\n(same weights)", "INT16": "INT16\n(weight rounding)", "INT8": "INT8", "INT6": "INT6", "INT5": "INT5", "INT4": "INT4"}
PLAIN = {"FP_rebatched": "FP repeat", "INT16": "INT16", "INT8": "INT8", "INT6": "INT6", "INT5": "INT5", "INT4": "INT4"}


def _f(x):
    return float("nan") if x is None else float(x)


def figure_arrays(stage_a, states, sample, T):
    """Per state (not FP) the arrays of the tested features that the figures need: W1 (`cos`), the read-side correlation of the same features (`corr`), the matched negative-control value (`control`) and, for the
    quantized states, whether the D-035 status is retained (`retained`)."""
    out = {}
    for s in STATE_ORDER:
        if s not in states:
            continue
        w = states[s]["write"]; nc = an.negative_control(w["S"], T["lang_idx"], T["rate_fp"])
        out[s] = {"cos": np.asarray(w["cos"], dtype=np.float64), "corr": np.asarray(states[s]["read"]["corr"], dtype=np.float64)[T["rows"]], "control": nc["values"]}
        if s in T["status"]:
            out[s]["retained"] = T["status"][s] == "retained"
    return out


def _style(ax, t):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"]); ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=t["muted"], labelcolor=t["ink2"], length=3, width=1.0, labelsize=8.5)
    ax.grid(True, color=t["grid"], linewidth=0.8, linestyle="-"); ax.set_axisbelow(True)


def _legend_row(fig, t, handles, labels, ncol=2, y_in=0.1):
    w_in, h_in = fig.get_size_inches()
    leg = fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False, fontsize=8.5, handlelength=1.8, bbox_to_anchor=(0.5, y_in / h_in))
    for text in leg.get_texts():
        text.set_color(t["ink2"])


def fig_w1_distribution(arrays, floor, theme, note):
    """One histogram row per state: W1 of the tested features, its median, the numerical floor (dashed) and the 5th to 95th percentile of the matched negative control (gray bar under the row)."""
    order = [s for s in STATE_ORDER if s in arrays]; n = len(order)
    if not n:
        raise ValueError("no state to plot")
    fig, ax, t = base._new_fig(theme, 8.2, 0.78 * n + 2.7)
    allv = np.concatenate([np.concatenate([arrays[s]["cos"], arrays[s]["control"]]) for s in order]); allv = allv[~np.isnan(allv)]; lo = min(-0.2, float(np.floor(allv.min() * 20) / 20)) if allv.size else -0.2
    edges = np.arange(lo, 1.0 + 1e-9, 0.02)
    for i, s in enumerate(order):
        y0 = n - 1 - i; v = arrays[s]["cos"]; d = v[~np.isnan(v)]; h, _ = np.histogram(np.clip(d, edges[0], edges[-1]), bins=edges); hn = h / h.max() * 0.82 if h.max() > 0 else h * 0.0
        ax.stairs(hn + y0, edges, baseline=y0, fill=True, color=t["series"][0], alpha=0.8, linewidth=0, gid=f"hist:{s}")
        if d.size:
            ax.plot([np.median(d)] * 2, [y0, y0 + 0.82], color=t["ink"], linewidth=1.6, solid_capstyle="butt", gid="median")
        c = arrays[s]["control"]; c = c[~np.isnan(c)]
        if c.size:
            ax.hlines(y0 - 0.07, np.percentile(c, 5), np.percentile(c, 95), color=t["muted"], linewidth=4.0, gid="control_band")
        else:
            ax.hlines(y0 - 0.07, 0, 0, gid="control_band")
    ax.axvline(floor["tau"], color=t["ink2"], linestyle="--", linewidth=1.2, gid="floor", zorder=3)
    ax.set_yticks([n - 1 - i + 0.4 for i in range(n)])
    ax.set_yticklabels([f"{LABELS[s]}\nn={int((~np.isnan(arrays[s]['cos'])).sum())}" + (f", {int(np.isnan(arrays[s]['cos']).sum())} undefined" if np.isnan(arrays[s]["cos"]).any() else "") for s in order])
    ax.set_ylim(-0.35, n + 0.05); ax.set_xlim(edges[0], 1.02); ax.grid(False, axis="y")
    base._axis_labels(ax, t, xlabel="cosine similarity of the single-feature effect fingerprint with full precision (1 = the same effect)")
    extra = base._title(ax, t, "Does switching a feature off still do the same thing?", f"W1 per tested feature and state; numerical floor {floor['tau']:.3f} (from {PLAIN.get(floor['source'], floor['source'])}) · {note}")
    handles = [Patch(facecolor=t["series"][0], alpha=0.8, linewidth=0), Line2D([0], [0], color=t["ink"], linewidth=1.6), Line2D([0], [0], color=t["ink2"], linestyle="--", linewidth=1.2), Line2D([0], [0], color=t["muted"], linewidth=4)]
    _legend_row(fig, t, handles, ["W1 of the tested features", "median", "numerical floor", "matched different-feature control, 5th to 95th percentile"], ncol=2, y_in=0.1)
    base._margins(fig, left_in=1.45, bottom_in=1.35, extra=extra)
    return fig


def fig_read_vs_write(arrays, tau, theme, note, states=("INT6", "INT5", "INT4")):
    """Read side against write side, one panel per quantized state: x = token-level correlation of the activations with full precision (the baseline survival basis, cutoff 0.9), y = W1; coloured by the
    D-035 status (retained or lost). The floor is the horizontal line."""
    t = base.theme_tokens(theme); panels = [s for s in states if s in arrays and "retained" in arrays[s]]
    if not panels:
        raise ValueError("no quantized state with a D-035 status to plot")
    n = len(panels); fig = Figure(figsize=(max(8.2, 3.7 * n + 1.3), 4.9), facecolor=t["surface"]); FigureCanvasAgg(fig); axes = fig.subplots(1, n, sharey=True, squeeze=False)[0]
    for ax, s in zip(axes, panels):
        _style(ax, t); a = arrays[s]; ret = a["retained"]
        ax.scatter(a["corr"][ret], a["cos"][ret], s=7, color=t["series"][0], alpha=0.4, linewidths=0, gid=f"retained:{s}")
        ax.scatter(a["corr"][~ret], a["cos"][~ret], s=7, color=t["series"][1], alpha=0.5, linewidths=0, gid=f"lost:{s}")
        ax.axvline(an.DUAN_CUTOFF, color=t["ink2"], linestyle=":", linewidth=1.2, gid="duan_cutoff"); ax.axhline(tau, color=t["ink2"], linestyle="--", linewidth=1.2, gid="floor")
        nund = int((np.isnan(a["cos"]) | np.isnan(a["corr"])).sum()); ax.set_title(PLAIN[s] + (f" ({nund} undefined, not drawn)" if nund else ""), loc="left", fontsize=9.5, color=t["ink"])
        ax.set_xlim(min(-0.05, float(np.nanmin(a["corr"])) - 0.02), 1.02); ax.set_xlabel("token correlation with full precision (read side)", color=t["ink2"], fontsize=8.5)
    ylo = min(-0.2, min(float(np.nanmin(arrays[s]["cos"])) for s in panels) - 0.02); axes[0].set_ylim(ylo, 1.03); axes[0].set_ylabel("W1: cosine of the effect fingerprints (write side)", color=t["ink2"], fontsize=9)
    w_in, h_in = fig.get_size_inches(); sub = textwrap.fill(f"one dot per tested feature; dotted: survival cutoff 0.9; dashed: numerical floor {tau:.3f} · {note}", width=max(40, int((w_in - 1.1) * 72 / 4.9))); extra = 0.16 * sub.count("\n")
    fig.subplots_adjust(left=0.85 / w_in, right=1 - 0.25 / w_in, top=1 - (1.1 + extra) / h_in, bottom=1.25 / h_in, wspace=0.08)
    fig.text(0.85 / w_in, 1 - 0.32 / h_in, "Do the features that survive by the activation cutoff still do the same thing?", fontsize=10.5, color=t["ink"], ha="left", va="center")
    fig.text(0.85 / w_in, 1 - 0.46 / h_in, sub, fontsize=8.5, color=t["muted"], ha="left", va="top", linespacing=1.3)
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=t["series"][0], markeredgecolor=t["surface"], markersize=6), Line2D([0], [0], marker="o", color="none", markerfacecolor=t["series"][1], markeredgecolor=t["surface"], markersize=6)]
    _legend_row(fig, t, handles, ["retained by the D-035 language-association status", "lost by the D-035 status"], ncol=2, y_in=0.1)
    return fig


STATEMENT_ROWS = [("S1", "S1  INT6: retained by D-035\nshare with W1 at or above floor", "share"), ("S2", "S2  INT4: F4 pairs, AUROC of status\nfor W1 at or above floor", "auroc"), ("S3", "S3  INT5: survive by cutoff 0.9\nshare with W1 at or above floor", "share")]


def fig_statements(validity, theme, note):
    """The three statements of the rule proposed in advance (Q23): value with its 95 % interval (languages resampled), the rule (diamond; S2 also the bound 0.5 for the interval) and the verdict as text."""
    fig, ax, t = base._new_fig(theme, 9.4, 3.9); st = validity["statements"]; n = len(STATEMENT_ROWS)
    for i, (k, label, key) in enumerate(STATEMENT_ROWS):
        y = n - 1 - i; s = st[k]; v, lo, hi = _f(s[key]), _f(s.get("lo")), _f(s.get("hi"))
        ax.scatter([0.8 if key == "share" else 0.6], [y], marker="D", s=46, color=t["muted"], zorder=3, gid=f"rule:{k}")
        if key == "auroc":
            ax.vlines(0.5, y - 0.22, y + 0.22, color=t["muted"], linewidth=1.5, zorder=3, gid=f"rule_lower:{k}")
        if not np.isnan(v) and s["verdict"] != "undefined":
            if not (np.isnan(lo) or np.isnan(hi)):
                ax.hlines(y, lo, hi, color=t["series"][0], linewidth=2.2, zorder=4, gid=f"interval:{k}")
            ax.scatter([v], [y], s=62, color=t["series"][0], edgecolors=t["surface"], linewidths=1.5, zorder=5, gid=f"point:{k}")
            ax.text(v, y + 0.27, f"{v:.2f}" + ("" if np.isnan(lo) else f" [{lo:.2f}, {hi:.2f}]"), ha="center", va="bottom", fontsize=8.5, color=t["ink2"])
        why = "" if s["verdict"] != "undefined" else (f"\n{s['n_pos']} events, {s['n_neg']} without" if key == "auroc" else f"\n{s['n_defined']} features defined")
        ax.text(1.03, y, s["verdict"] + why, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=9.5, color=t["ink"], fontweight="bold")
    ax.set_yticks(range(n)); ax.set_yticklabels([r[1] for r in STATEMENT_ROWS][::-1]); ax.set_ylim(-0.6, n - 0.2); ax.set_xlim(0, 1.0); ax.grid(False, axis="y")
    base._axis_labels(ax, t, xlabel="share of features (S1, S3) or AUROC (S2); diamond = the rule (S2: also the lower bound 0.5)")
    sub = f"overall: {validity['overall']}. Exploratory; holds, if at all, for this model, layer, quantizer and sample · {note}"
    extra = base._title(ax, t, "Is the survival metric sensible? Three statements fixed in advance", sub)
    if validity.get("note"):
        ax.text(0.0, -0.2, validity["note"], transform=ax.transAxes, ha="left", va="top", fontsize=8, color=t["muted"], wrap=True)
    handles = [Line2D([0], [0], color=t["series"][0], marker="o", markersize=6, linewidth=2, markeredgecolor=t["surface"]), Line2D([0], [0], color=t["muted"], marker="D", markersize=5, linewidth=0)]
    base._legend_below(fig, t, handles, ["value (95 % interval, languages resampled)", "the rule"], 2.1, 1.4)
    base._margins(fig, left_in=2.3, right_in=2.3, bottom_in=1.45, extra=extra)
    return fig


def _order(tables, exclude=()):
    return [s for s in STATE_ORDER if s in tables and s not in exclude]


def fig_retrieval(tables, theme, note):
    """Is the feature still itself? Top-1 retrieval of a feature's own identity among all tested features by the cosine of the effect fingerprints (overall, and among the features of its own language, with the
    chance level of that comparison)."""
    order = _order(tables)
    if not order:
        raise ValueError("no state to plot")
    fig, ax, t = base._new_fig(theme, 7.2, 4.6); x = np.arange(len(order)); g = lambda s, *p: _f(_dig(tables[s]["all"].get("retrieval", {}), *p))
    q = [i for i, s in enumerate(order) if s not in REFERENCE_STATES]; rep = [i for i, s in enumerate(order) if s in REFERENCE_STATES]            # the noise-only states are references, not steps of the ladder: no line to them
    for key, lab, color, mark, gid in (("top1", ("top1",), t["series"][0], "o", "top1"), ("within", ("within", "top1"), t["series"][1], "s", "within_top1")):
        ax.plot(x[q], [g(order[i], *lab) for i in q], color=color, marker=mark, markersize=6, linewidth=2, markeredgecolor=t["surface"], gid=gid)
        if rep:
            ax.plot(x[rep], [g(order[i], *lab) for i in rep], color=color, marker=mark, markersize=6, linestyle="none", markeredgecolor=t["surface"], gid=gid + "_repeat")
    ax.plot(x, [g(s, "within", "chance") for s in order], color=t["muted"], linestyle="--", linewidth=1.3, gid="chance")
    ax.set_xticks(x); ax.set_xticklabels([PLAIN[s] for s in order]); ax.set_ylim(0, 1.03); ax.set_xlim(-0.4, len(order) - 0.6)
    base._axis_labels(ax, t, ylabel="share of features retrieved as themselves (top 1)")
    extra = base._title(ax, t, "Is a quantized feature still the same feature?", f"the feature whose effect fingerprint in the quantized model is closest to its own full-precision fingerprint · {note}")
    handles = [Line2D([0], [0], color=t["series"][0], marker="o", markersize=6, linewidth=2), Line2D([0], [0], color=t["series"][1], marker="s", markersize=5.5, linewidth=2), Line2D([0], [0], color=t["muted"], linestyle="--", linewidth=1.3)]
    base._legend_below(fig, t, handles, ["among all tested features", "among the features of its own language", "chance level of the within-language comparison"], 1.0, 0.25)
    base._margins(fig, left_in=1.0, bottom_in=1.45, extra=extra)
    return fig


def _dig(d, *keys):
    for k in keys:
        d = d.get(k, {}) if isinstance(d, dict) else {}
    return d if not isinstance(d, dict) else None


def fig_median_ladder(tables, floor, theme, note):
    """Medians by state of the read side (token correlation of the activations, the survival basis) and of the write side (W1), with 95 % intervals (languages resampled) and the numerical floor."""
    order = _order(tables)
    if not order:
        raise ValueError("no state to plot")
    fig, ax, t = base._new_fig(theme, 7.6, 4.8); xr = [i for i, s in enumerate(order) if s not in REFERENCE_STATES]; qs = [order[i] for i in xr]
    r1 = [_f(tables[s]["all"]["R1"]["median"]) for s in qs]; r1lo = [_f(tables[s]["all"]["R1"]["median_lo"]) for s in qs]; r1hi = [_f(tables[s]["all"]["R1"]["median_hi"]) for s in qs]
    w1 = [_f(tables[s]["all"]["W1"]["median"]) for s in order]; w1lo = [_f(tables[s]["all"]["W1"]["median_lo"]) for s in order]; w1hi = [_f(tables[s]["all"]["W1"]["median_hi"]) for s in order]
    xs_r = np.array(xr, dtype=float) - 0.05; xs_w = np.arange(len(order), dtype=float) + 0.05
    ax.vlines(xs_r, r1lo, r1hi, color=t["series"][0], linewidth=1.8, gid="R1_ci"); ax.vlines(xs_w, w1lo, w1hi, color=t["series"][1], linewidth=1.8, gid="W1_ci")
    ax.plot(xs_r, r1, color=t["series"][0], marker="o", markersize=6, linewidth=2, markeredgecolor=t["surface"], gid="R1")
    iq = [i for i, s in enumerate(order) if s not in REFERENCE_STATES]; ir = [i for i, s in enumerate(order) if s in REFERENCE_STATES]                  # the noise-only states are references, not steps of the ladder: no line to them
    ax.plot(xs_w[iq], [w1[i] for i in iq], color=t["series"][1], marker="s", markersize=5.5, linewidth=2, markeredgecolor=t["surface"], gid="W1")
    if ir:
        ax.plot(xs_w[ir], [w1[i] for i in ir], color=t["series"][1], marker="s", markersize=5.5, linestyle="none", markeredgecolor=t["surface"], gid="W1_repeat")
    if ir:
        ax.plot(np.array(ir, dtype=float) - 0.05, [_f(tables[order[i]]["all"]["R1"]["median"]) for i in ir], color=t["series"][0], marker="o", markersize=6, linestyle="none", markeredgecolor=t["surface"], gid="R1_repeat")
    ax.axhline(floor["tau"], color=t["ink2"], linestyle="--", linewidth=1.2, gid="floor")
    ax.set_xticks(range(len(order))); ax.set_xticklabels([PLAIN[s] for s in order]); ax.set_xlim(-0.4, len(order) - 0.6)
    lo = min([v for v in w1lo + r1lo + w1 + r1 + [floor["tau"]] if not np.isnan(v)] + [1.0]); ax.set_ylim(max(0.0, lo - 0.1), 1.03)
    base._axis_labels(ax, t, ylabel="median over the features")
    extra = base._title(ax, t, "Does the activation (survival) basis track the function?", f"median read-side correlation and median write-side W1 by state; dashed: numerical floor {floor['tau']:.3f} · {note}")
    handles = [Line2D([0], [0], color=t["series"][0], marker="o", markersize=6, linewidth=2), Line2D([0], [0], color=t["series"][1], marker="s", markersize=5.5, linewidth=2), Line2D([0], [0], color=t["ink2"], linestyle="--", linewidth=1.2)]
    base._legend_below(fig, t, handles, ["read side: token correlation of the activations", "write side: W1, cosine of the effect fingerprints", "numerical floor"], 1.0, 0.25)
    base._margins(fig, left_in=1.0, bottom_in=1.45, extra=extra)
    return fig


def fig_paired(validity, theme, note):
    """H5c per language: median W1 of the F4 retained features minus median W1 of the F4 lost features (activity-matched), one dot per language and precision; zero = no difference."""
    p = {k: v for k, v in validity["paired"].items() if v["by_language"]}
    if not p:
        raise ValueError("no paired difference to plot")
    key = "INT4" if "INT4" in p else sorted(p)[0]; langs = sorted({l for v in p.values() for l in v["by_language"]}, key=lambda l: p[key]["by_language"].get(l, -9.0)); n = len(langs)
    fig, ax, t = base._new_fig(theme, 7.6, 0.25 * n + 2.7); order = [s for s in ("INT6", "INT5", "INT4") if s in p]; cols = dict(zip(order, base.theme_tokens(theme)["ordinal"] if len(order) == 3 else base.theme_tokens(theme)["ordinal4"][:len(order)]))
    for s in order:
        ys = [i for i, l in enumerate(langs) if l in p[s]["by_language"]]; xs = [p[s]["by_language"][langs[i]] for i in ys]
        ax.scatter(xs, ys, s=40, color=cols[s], edgecolors=t["surface"], linewidths=1.2, zorder=4, gid=f"paired:{s}")
    ax.axvline(0, color=t["ink2"], linewidth=1.2, gid="zero"); ax.set_yticks(range(n)); ax.set_yticklabels(langs); ax.set_ylim(-0.8, n - 0.2); ax.grid(False, axis="y")
    base._axis_labels(ax, t, xlabel="median W1 of retained minus median W1 of lost features\n(right of zero: the retained features keep more of their effect)")
    extra = base._title(ax, t, "Do retained features keep their function better than lost ones?", f"F4 pairs (activity-matched), one dot per language · {note}")
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=cols[s], markeredgecolor=t["surface"], markersize=7) for s in order]
    labels = [f"{s}: median {_f(p[s]['median']):+.3f}, {p[s]['sign']}" for s in order]
    base._legend_below(fig, t, handles, labels, 1.0, 0.25)
    base._margins(fig, left_in=1.1, bottom_in=1.5, extra=extra)
    return fig


def _nanify(x):
    if isinstance(x, dict):
        return {k: _nanify(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_nanify(v) for v in x]
    return float("nan") if x is None else x


def write_figures(stage_a, states, sample, T, res, out_dir, note, themes=("light", "dark"), formats=("svg", "png")):
    """All six figures in the requested themes and formats; a figure whose inputs do not exist (for example no quantized state with a D-035 status) is skipped and listed in f5_figures_skipped.txt."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True); arrays = figure_arrays(stage_a, states, sample, T); res = _nanify(res); paths, skipped = [], []
    makers = [("f5_fig1_w1_distribution", lambda th: fig_w1_distribution(arrays, res["floor"], th, note)), ("f5_fig2_read_vs_write", lambda th: fig_read_vs_write(arrays, res["floor"]["tau"], th, note)),
              ("f5_fig3_statements", lambda th: fig_statements(res["validity"], th, note)), ("f5_fig4_retrieval", lambda th: fig_retrieval(res["tables"], th, note)),
              ("f5_fig5_median_ladder", lambda th: fig_median_ladder(res["tables"], res["floor"], th, note)), ("f5_fig6_paired", lambda th: fig_paired(res["validity"], th, note))]
    for stem, make in makers:
        for theme in themes:
            try:
                fig = make(theme)
            except ValueError as e:
                if f"{stem}: {e}" not in skipped:
                    skipped.append(f"{stem}: {e}")
                continue
            paths += base.save_figure(fig, f"{stem}_{theme}", out, formats)
    sk = out / "f5_figures_skipped.txt"
    if skipped:
        sk.write_text("\n".join(skipped) + "\n")
    elif sk.exists():
        sk.unlink()
    return paths


def main(argv=None):
    ROOT = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Figures of the F5 study from the files of the runner and the analysis output.")
    ap.add_argument("--run", action="append", required=True); ap.add_argument("--sample", default=str(ROOT / "reports/functional_validation/f5/f5_sample.json"))
    ap.add_argument("--results", help="f5_results.json of phase1_func_f5_analysis.py (computed from the runs when absent)"); ap.add_argument("--out", required=True)
    ap.add_argument("--note", required=True, help="status of the data in every title, e.g. 'exploratory, smoke sample'"); ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--themes", nargs="*", default=["light", "dark"]); ap.add_argument("--formats", nargs="*", default=["svg", "png"])
    a = ap.parse_args(argv)
    sa, states, meta = an.load_runs(a.run); sample = json.loads(Path(a.sample).read_text()); T = an.assemble(sa, sample)
    res = json.loads(Path(a.results).read_text()) if a.results else an.analyse_f5(sa, states, sample, meta=meta, n_boot=a.n_boot)
    paths = write_figures(sa, states, sample, T, res, a.out, a.note, tuple(a.themes), tuple(a.formats))
    print("wrote", len(paths), "files to", a.out)


if __name__ == "__main__":
    main()
