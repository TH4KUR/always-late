"""The four figures of the intervention-transfer study (D-088; plan Task 6), drawn from the tables of phase1_transfer_report.py; nothing is recomputed here except the ratio of the two mean activations of an item
and the absolute values of C and W of a language. Same themes, fonts and layout helpers as the other figure modules (phase1_func_figures, phase1_gain_figures).

  fig_tr1_activation_effect     does the change of an activation tell the change of the intervention's effect? (eligible items of the real direction: activation ratio against the normalized transfer error)
  fig_tr2_crossed_contrasts     the relative norms of the context, weight and interaction terms with their intervals for the real and the control directions, and the group-level |C| against |W| per language (A3)
  fig_tr3_averaged_vs_context   per feature, the cosine of the averaged fingerprint against the mean cosine of its contexts (H4)
  fig_tr4_predictive            the held-out error of the models of the diagnostic, the null model, and the reduction against the strongest baseline with its interval and the 10 percent criterion

    ~/.venvs/anlp-torch/bin/python scripts/g0/phase1_transfer_figures.py --results <report dir> --out <figure dir> --note "exploratory" """
import argparse
import csv
import re
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

import phase1_func_figures as base
import phase1_gain_figures as gf

STATES = ("INT6", "INT4")
KINDS = ("real", "matched_swapped", "random1", "random2", "random3")
KIND_LABEL = {"real": "real direction", "matched_swapped": "matched swapped", "random1": "random 1", "random2": "random 2", "random3": "random 3"}
MODELS = ("A", "A+G", "A+B", "A+B+G")
MODEL_LABEL = {"A": "A  activation baseline", "A+G": "A+G  plus gradient", "A+B": "A+B  plus generic sensitivity (the diagnostic)", "A+B+G": "A+B+G  all three"}
CRITERION = 0.10


def _convert(v):
    if v == "":
        return None
    if v in ("True", "False"):
        return v == "True"
    if re.fullmatch(r"-?\d+", v):
        return int(v)
    if re.fullmatch(r"-?\d+;-?\d+(\.\d+)?|-?[\d.eE+-]+;-?[\d.eE+-]+", v):
        a, b = v.split(";"); return (float(a), float(b))
    try:
        return float(v)
    except ValueError:
        return v


def read_csv(path):
    """Rows of a report table with the values converted: integers, floats, booleans, 'low;high' pairs as tuples, blanks as None, anything else as text."""
    with open(path, newline="") as fh:
        return [{k: _convert(v) for k, v in r.items()} for r in csv.DictReader(fh)]


def load_tables(directory):
    """The tables of a report directory: per_context and averaged (by state), components, group, a3, predictive. A missing table is an error that names the file."""
    d = Path(directory)
    need = lambda name: read_csv(d / name) if (d / name).exists() else (_ for _ in ()).throw(FileNotFoundError(f"{name} is missing from {d}"))
    out = {"components": need("component_summary.csv"), "group": need("group_per_language.csv"), "a3": need("a3.csv"), "predictive": need("predictive_validation.csv"), "per_context": {}, "averaged": {}}
    for key, pat in (("per_context", "per_context_*.csv"), ("averaged", "averaged_features_*.csv")):
        for p in sorted(d.glob(pat)):
            out[key][p.stem.split("_")[-1]] = read_csv(p)
        if not out[key]:
            raise FileNotFoundError(f"no {pat} in {d}")
    return out


def _states(tables, key):
    return [s for s in STATES if s in tables[key]]


def _label(ax, t, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=9, color=t["ink2"], pad=4); ax.set_xlabel(xlabel, color=t["ink2"], fontsize=8.5); ax.set_ylabel(ylabel, color=t["ink2"], fontsize=8.5)


def _legend(fig, t, handles, labels, ncol):
    leg = fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    for tx in leg.get_texts():
        tx.set_color(t["ink2"])


def fig_activation_effect(tables, theme, note):
    states = _states(tables, "per_context"); fig, t = gf._fig(theme, 10.4, 5.9); w, h = fig.get_size_inches(); n_total = {}
    for i, s in enumerate(states):
        rows = [r for r in tables["per_context"][s] if r["kind"] == "real" and r["eligible"]]; n_total[s] = (len(rows), len({r["feature"] for r in rows}))
        x = np.array([r["amp_q_mean"] / r["amp_fp_mean"] for r in rows], dtype=float); y = np.array([r["normalized_error"] for r in rows], dtype=float)
        ax = fig.add_axes([(0.95 + 4.75 * i) / w, 1.2 / h, 4.0 / w, 3.0 / h], gid=f"panel:{s}"); gf._style(ax, t)
        ax.axvline(1.0, color=t["muted"], linewidth=1.0, linestyle="--", zorder=1, gid="unchanged")
        ax.scatter(x, y, s=22, color=t["series"][0], edgecolor=t["surface"], linewidth=0.5, alpha=0.85, zorder=3, gid="items")
        _label(ax, t, f"{s}: {len(rows)} eligible items of {n_total[s][1]} features", "mean activation in the quantized state / in full precision (dashed: unchanged)", "normalized transfer error")
    gf._head(fig, t, "Does the change of an activation tell how much the intervention's effect changed?",
             "one point per feature and context of the real decoder direction whose full-precision effect is above five times the numerical floor; horizontal: the mean activation of the feature at the edited positions "
             f"in the quantized state over its full-precision value; vertical: the error of the effect, the norm of the difference between the quantized and the full-precision effect over the full-precision norm plus the floor; {note}", 0.9)
    return fig


def _component_panel(fig, t, w, h, tables, state, x0):
    ax = fig.add_axes([x0 / w, 1.2 / h, 2.9 / w, 3.0 / h], gid=f"panel:components:{state}"); gf._style(ax, t)
    for ki, k in enumerate(KINDS):
        recs = [next((r for r in tables["components"] if r["state"] == state and r["kind"] == k and r["metric"] == m and r["split"] == "all"), None) for m in ("C_rel", "W_rel", "I_rel")]
        if any(r is None or r["point"] is None for r in recs):
            continue
        xs = np.arange(3) + (ki - 2) * 0.12; pts = np.array([r["point"] for r in recs], dtype=float); lo = np.array([r["low"] for r in recs], dtype=float); hi = np.array([r["high"] for r in recs], dtype=float)
        col = t["series"][0] if k == "real" else t["ink2"]
        v = ax.vlines(xs, lo, hi, color=col, linewidth=1.2, alpha=0.9, zorder=2); v.set_gid(f"ci:{k}")
        ax.scatter(xs, pts, s=34 if k == "real" else 20, color=col if k == "real" else "none", edgecolor=col, linewidth=1.1, zorder=3, gid=f"est:{k}")
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["context C", "weights W", "interaction I"]); ax.set_xlim(-0.5, 2.5)
    _label(ax, t, f"{state}: relative norm of each term", "term of the decomposition", "median norm / (FP norm + floor)")
    return ax


def fig_crossed_contrasts(tables, theme, note):
    states = [s for s in STATES if any(r["state"] == s for r in tables["components"])]; fig, t = gf._fig(theme, 11.4, 5.9); w, h = fig.get_size_inches()
    for i, s in enumerate(states):
        _component_panel(fig, t, w, h, tables, s, 0.95 + 3.5 * i)
    g = fig.add_axes([(0.95 + 3.5 * len(states)) / w, 1.2 / h, 2.9 / w, 3.0 / h], gid="panel:group"); gf._style(g, t); allv = []
    for si, s in enumerate(STATES):
        rows = [r for r in tables["group"] if r["state"] == s]
        if not rows:
            continue
        x = np.array([abs(r["W"]) for r in rows], dtype=float); y = np.array([abs(r["C"]) for r in rows], dtype=float); allv += list(x) + list(y)
        g.scatter(x, y, s=24, color=t["series"][1 + si], edgecolor=t["surface"], linewidth=0.6, alpha=0.9, zorder=3, gid=f"group:{s}")
    hi = max(allv) * 1.08 if allv else 1.0; g.plot((0, hi), (0, hi), color=t["muted"], linewidth=1.0, linestyle="--", zorder=1, gid="identity"); g.set_xlim(0, hi); g.set_ylim(0, hi)
    _label(g, t, "group level: one point per language", "|W|  weight term of the net harm", "|C|  context term (dashed: |C| = |W|)")
    handles = [Line2D([0], [0], color=t["series"][0], marker="o", linestyle="none", markersize=6), Line2D([0], [0], color=t["ink2"], marker="o", markerfacecolor="none", linestyle="none", markersize=5),
               Line2D([0], [0], color=t["series"][1], marker="o", linestyle="none", markersize=5), Line2D([0], [0], color=t["series"][2], marker="o", linestyle="none", markersize=5)]
    _legend(fig, t, handles, ["real direction", "four control directions (hollow, bars: crossed bootstrap interval)", "group level INT6", "group level INT4"], 4)
    gf._head(fig, t, "Which term carries the change of an intervention's effect: the context, the weights, or their interaction?",
             "left: the median norm of the context term C (activations change, weights stay), the weight term W (weights change, activations stay) and the interaction I over the full-precision effect norm plus the floor, for the real "
             f"direction and the four control directions of the same features; right: the same decomposition of the net harm (group minus matched controls) of each of the languages, absolute values; {note}", 0.9)
    return fig


def fig_averaged_vs_context(tables, theme, note):
    states = _states(tables, "averaged"); fig, t = gf._fig(theme, 10.4, 5.9); w, h = fig.get_size_inches()
    for i, s in enumerate(states):
        rows = [r for r in tables["averaged"][s] if r["pooled_cos"] is not None and r["item_cos"] is not None]; real = [r for r in rows if r["kind"] == "real"]; ctrl = [r for r in rows if r["kind"] != "real"]
        ax = fig.add_axes([(0.95 + 4.75 * i) / w, 1.2 / h, 4.0 / w, 3.0 / h], gid=f"panel:{s}"); gf._style(ax, t); vals = [r[k] for r in rows for k in ("pooled_cos", "item_cos")]
        lo, hi = min([-0.05] + vals) - 0.02, max([1.0] + vals) + 0.02; ax.plot((lo, hi), (lo, hi), color=t["muted"], linewidth=1.0, linestyle="--", zorder=1, gid="identity"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ticks = [v for v in np.arange(-1.0, 1.01, 0.25) if lo <= v <= hi]; ax.set_xticks(ticks); ax.set_yticks(ticks)                    # explicit ticks inside the limits (an automatic tick beyond them has a label outside the canvas)
        ax.scatter([r["item_cos"] for r in ctrl], [r["pooled_cos"] for r in ctrl], s=16, facecolor="none", edgecolor=t["series"][1], linewidth=0.8, alpha=0.8, zorder=2, gid="controls")
        ax.scatter([r["item_cos"] for r in real], [r["pooled_cos"] for r in real], s=26, color=t["series"][0], edgecolor=t["surface"], linewidth=0.6, alpha=0.9, zorder=3, gid="real")
        _label(ax, t, f"{s}: {len(real)} features (real direction), {len(ctrl)} control points", "mean cosine of the single contexts (full precision against quantized effect)", "cosine of the averaged fingerprint (dashed: equal)")
    handles = [Line2D([0], [0], color=t["series"][0], marker="o", linestyle="none", markersize=6), Line2D([0], [0], color=t["series"][1], marker="o", markerfacecolor="none", linestyle="none", markersize=6)]
    _legend(fig, t, handles, ["real decoder direction", "matched swapped and random directions"], 2)
    gf._head(fig, t, "Does the averaged fingerprint of a feature hide how its single contexts transfer?",
             "one point per feature and direction on the same feature-context pairs: horizontal, the row-weighted mean of the cosines between the full-precision and the quantized effect of its contexts; vertical, the cosine between "
             f"the full-precision and the quantized AVERAGED effect of the same contexts; points above the line: the average looks better than its parts; features whose average effect is exactly zero are not drawn; {note}", 0.9)
    return fig


def fig_predictive(tables, theme, note):
    fig, t = gf._fig(theme, 10.6, 5.9); w, h = fig.get_size_inches(); rows = [r for r in tables["predictive"] if r.get("variant") == "eligible"]
    for i, s in enumerate(STATES):
        ax = fig.add_axes([(0.95 + 3.2 * i) / w, 1.2 / h, 2.9 / w, 3.0 / h], gid=f"panel:mae:{s}"); gf._style(ax, t); recs = [r for r in rows if r["state"] == s]
        fitted = [r for r in recs if r["status"] == "fitted" and r.get("model") in MODELS]
        if not fitted:
            n = recs[0]["n_design"] if recs else 0; ax.text(0.5, 0.5, f"too few eligible items to fit a held-out model\n({n} usable items)", transform=ax.transAxes, ha="center", va="center", fontsize=9, color=t["ink2"], gid="note")
            ax.set_xticks([]); ax.set_yticks([]); _label(ax, t, f"{s}: held-out error", "", ""); continue
        ys = [MODELS.index(r["model"]) for r in fitted]; xs = [r["heldout_mae"] for r in fitted]; null = fitted[0]["null_mae"]
        ax.axvline(null, color=t["muted"], linewidth=1.0, linestyle="--", zorder=1, gid="null")
        ax.scatter(xs, ys, s=44, color=[t["series"][0] if r["model"] == "A+B" else t["ink2"] for r in fitted], edgecolor=t["surface"], linewidth=0.8, zorder=3, gid="models")
        ax.set_yticks(range(len(MODELS))); ax.set_yticklabels(MODELS); ax.set_ylim(len(MODELS) - 0.5, -0.5); ax.set_xlim(0, max(xs + [null]) * 1.12)
        _label(ax, t, f"{s}: held-out error ({fitted[0]['n_heldout']} items)", "held-out mean absolute error (dashed: null)", "")
    imp = fig.add_axes([(0.95 + 3.2 * len(STATES)) / w, 1.2 / h, 2.9 / w, 3.0 / h], gid="panel:improvement"); gf._style(imp, t); ys, labels = [], []
    for i, s in enumerate(STATES):
        r = next((x for x in rows if x["state"] == s and x.get("model") == "A+B" and x.get("improvement_diagnostic") is not None), None)
        if r is None:
            continue
        ys.append(i); labels.append(s); c = imp.hlines(i, r["improvement_low"], r["improvement_high"], color=t["series"][0], linewidth=2.0, zorder=2); c.set_gid("reduction-ci")
        imp.scatter([r["improvement_diagnostic"]], [i], s=50, color=t["series"][0], edgecolor=t["surface"], linewidth=0.8, zorder=3, gid="reduction")
    for i, s in enumerate(STATES):
        if s not in labels:
            imp.text(0.5, i, "not fitted: too few items", transform=imp.get_yaxis_transform(), ha="center", va="center", fontsize=8.5, color=t["muted"], gid=f"note:{s}")
    imp.axvline(0.0, color=t["muted"], linewidth=1.0, zorder=1, gid="zero"); imp.axvline(CRITERION, color=t["series"][1], linewidth=1.0, linestyle="--", zorder=1, gid="criterion")
    imp.set_yticks(range(len(STATES))); imp.set_yticklabels(STATES); imp.set_ylim(len(STATES) - 0.5, -0.5)
    _label(imp, t, "A+B against the strongest baseline", "relative reduction of the held-out error", "")
    handles = [Line2D([0], [0], color=t["series"][0], marker="o", linestyle="none", markersize=6), Line2D([0], [0], color=t["ink2"], marker="o", linestyle="none", markersize=6), Line2D([0], [0], color=t["series"][1], linestyle="--")]
    _legend(fig, t, handles, ["A+B, the diagnostic", "the other models", "the 10 percent criterion"], 3)
    base_names = sorted({r["strongest_baseline"] for r in rows if r.get("strongest_baseline")})
    gf._head(fig, t, "Does the sensitivity to other directions predict an intervention's transfer error better than activations and gradients?",
             "models fitted on the calibration features and scored on the held-out features (eligible items of the real direction): A the activation baseline, G the gradient baseline, B the generic sensitivity measured with the other four "
             f"directions of the same context; the strongest baseline is chosen on the calibration half ({', '.join(base_names) or 'none fitted'}); right: its relative error reduction with the crossed bootstrap interval; {note}", 0.9)
    return fig


def write_figures(tables, out_dir, note, themes=("light", "dark"), formats=("svg", "png")):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True); paths = []
    for stem, make in (("fig_tr1_activation_effect", fig_activation_effect), ("fig_tr2_crossed_contrasts", fig_crossed_contrasts), ("fig_tr3_averaged_vs_context", fig_averaged_vs_context), ("fig_tr4_predictive", fig_predictive)):
        for theme in themes:
            paths += base.save_figure(make(tables, theme, note), f"{stem}_{theme}", out, formats)
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description="Figures of the intervention-transfer study from the tables of phase1_transfer_report.py.")
    ap.add_argument("--results", required=True); ap.add_argument("--out", required=True); ap.add_argument("--note", required=True)
    ap.add_argument("--themes", nargs="*", default=["light", "dark"]); ap.add_argument("--formats", nargs="*", default=["svg", "png"])
    a = ap.parse_args(argv)
    paths = write_figures(load_tables(a.results), a.out, a.note, tuple(a.themes), tuple(a.formats))
    print("wrote", len(paths), "figure files to", a.out)


if __name__ == "__main__":
    main()
