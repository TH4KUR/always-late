"""Study G, task G3 (design section 3 and amendment A1): the ratios of the gain study are chosen from a calibration on the PILOT split by a rule fixed before the calibration runs.

Grid: down `2^(-k/2)`, k = 1..8; up the reciprocals; plus two fine up ratios 1.1 and 1.2 that are run but never chosen by the rule (they serve the optimal-gain analysis near 1).
Rule (spec section 3): z = FP median over languages of the net harm of zero-ablation (group minus mean of the zeroed controls, nats per token on the language's own text). For each target
fraction phi in (0.10, 0.30, 0.60), in this order, the down-grid ratio whose FP median net harm (group scaled minus mean of the same ratio applied to the matched controls) is nearest to phi * z;
a value already chosen is skipped (the next nearest is taken), equally near values go to the milder (larger) ratio. If the median net harm never reaches 0.10 z on the down grid the three down
ratios are 0.25, 0.125, 0.0625 (branch `flat`). The three up ratios are the reciprocals of the chosen down ratios. Reads the item files of `phase1_func_run` through `phase1_func_analysis.Run`."""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import phase1_func_analysis as fa

TARGETS = (0.10, 0.30, 0.60)
FALLBACK = (0.25, 0.125, 0.0625)
FINE_UP = (1.1, 1.2)


def grid():
    """(down, up): the eight down ratios 2^(-k/2), largest first, and their reciprocals."""
    down = [2.0 ** (-k / 2.0) for k in range(1, 9)]
    return down, [1.0 / d for d in down]


def calibration_ratios():
    """The 18 ratios of a calibration run: the down grid, the up grid and the two fine up ratios."""
    down, up = grid()
    return down + up + list(FINE_UP)


def group_name(alpha, label, code):
    return f"scale{alpha:g}_{label}@{code}"


def control_name(alpha, draw, label, code):
    return f"cscale{alpha:g}.{draw}_{label}@{code}"


def zero_name(label, code):
    return f"zero_{label}@{code}"


def zeroed_control_name(draw, label, code):
    return f"ctrl{draw}_{label}@{code}"


def select_ratios(net_by_ratio, z, targets=TARGETS, fallback=FALLBACK):
    """The rule above. net_by_ratio: {down-grid ratio: FP median net harm}; z: FP median net harm of zero-ablation. Returns {"down": shallow to deep, "up": reciprocals, "branch", "z"}."""
    if not (z > 0):
        raise ValueError(f"the zero-ablation net harm z must be positive, got {z}")
    down_grid = grid()[0]
    missing = [g for g in down_grid if g not in net_by_ratio]
    if missing:
        raise ValueError(f"net harm missing for grid ratios {missing}")
    if max(net_by_ratio[g] for g in down_grid) < min(targets) * z:
        down, branch = list(fallback), "flat"
    else:
        chosen = []
        for phi in targets:
            order = sorted(down_grid, key=lambda g: (abs(net_by_ratio[g] - phi * z), -g))
            chosen.append(next(g for g in order if g not in chosen))
        down, branch = sorted(chosen, reverse=True), "targets"
    return {"down": down, "up": [1.0 / d for d in down], "branch": branch, "z": float(z)}


def calibration_rows(run, languages, ratios, label="K64", n_controls=2):
    """One row per ratio (and one for zero-ablation, ratio 0, when the run has it for every language): medians over `languages` of the group harm, the mean control harm and the net harm in nats
    per token (pooled over the language's tokens), the number of languages with a positive net harm and the net harm as a fraction of z."""
    def per_language(group, controls):
        g, c = [], []
        for code in languages:
            for n in [group(code)] + [x(code) for x in controls]:
                if not run.has(code, n):
                    raise KeyError(f"{run.label}: {code} lacks {n}")
            dg, nt = run.delta(code, group(code))
            dc = np.mean([run.delta(code, x(code))[0] for x in controls], axis=0)
            g.append(fa.pooled(dg, nt)); c.append(fa.pooled(dc, nt))
        return np.array(g), np.array(c)

    def row(ratio, g, c):
        net = g - c
        return {"state": run.label, "ratio": float(ratio), "group_median": float(np.median(g)), "control_median": float(np.median(c)), "net_median": float(np.median(net)),
                "languages_net_positive": int((net > 0).sum()), "n_languages": len(languages)}

    rows = []
    for a in ratios:
        ctl = [(lambda code, r=r, a=a: control_name(a, r, label, code)) for r in range(n_controls)]
        rows.append(row(a, *per_language(lambda code, a=a: group_name(a, label, code), ctl)))
    if all(run.has(code, zero_name(label, code)) for code in languages):
        zc = [(lambda code, r=r: zeroed_control_name(r, label, code)) for r in range(n_controls)]
        rows.append(row(0.0, *per_language(lambda code: zero_name(label, code), zc)))
    z = [r for r in rows if r["ratio"] == 0.0]
    for r in rows:
        r["net_over_z"] = r["net_median"] / z[0]["net_median"] if z and z[0]["net_median"] != 0 else float("nan")
    return sorted(rows, key=lambda r: r["ratio"])


def zero_net_median(rows):
    z = [r for r in rows if r["ratio"] == 0.0]
    if not z:
        raise KeyError("no zero-ablation row")
    return z[0]["net_median"]


def _fmt(v, d=4):
    return "-" if v != v else f"{v:.{d}f}"


def report_text(rows_by_state, sel, meta):
    states = list(rows_by_state)
    out = ["# Study G, G3: calibration of the ratios on the pilot split", "",
           f"Pilot split ({meta.get('n_sentences', 'pilot')} sentences per language), K64 groups and {meta.get('n_controls', '?')} matched-control draws scaled with the same ratios; medians over the languages in nats per token. "
           "The rule (design section 3) was fixed before the runs: z = FP median net harm of zero-ablation; the three down ratios are the grid values whose FP net harm is nearest to 0.10 z, 0.30 z and 0.60 z; the up ratios are their reciprocals; "
           "the two fine up ratios (1.1, 1.2) are never chosen by the rule.", "",
           f"**Result: down ratios {', '.join(_fmt(d) for d in sel['down'])}; up ratios {', '.join(_fmt(u) for u in sel['up'])} (branch `{sel['branch']}`; z = {_fmt(sel['z'])}).**", ""]
    for s in states:
        out += [f"## {s}", "", "| ratio | group harm | control harm | net harm | net / z (FP) | languages with net > 0 |", "|---|---|---|---|---|---|"]
        for r in rows_by_state[s]:
            out.append(f"| {_fmt(r['ratio'])} | {_fmt(r['group_median'])} | {_fmt(r['control_median'])} | {_fmt(r['net_median'])} | {_fmt(r['net_over_z'], 3)} | {r['languages_net_positive']} of {r['n_languages']} |")
        out.append("")
    return "\n".join(out)


def write_outputs(out_dir, rows_by_state, sel, meta):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    all_rows = [r for s in rows_by_state for r in rows_by_state[s]]
    with open(out / "calibration.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0])); w.writeheader(); w.writerows(all_rows)
    (out / "ratios.json").write_text(json.dumps({"down": sel["down"], "up": sel["up"], "fine_up": list(FINE_UP), "branch": sel["branch"], "targets": list(TARGETS), "meta": meta}, indent=1) + "\n")
    (out / "calibration.md").write_text(report_text(rows_by_state, sel, meta))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Study G, G3: apply the pre-registered rule to the pilot calibration runs (files func_lm_pilot_<state>.pt).")
    ap.add_argument("--runs", nargs="+", required=True, help="directories holding func_lm_pilot_<state>.pt"); ap.add_argument("--out", required=True)
    ap.add_argument("--n-controls", type=int, default=2); ap.add_argument("--label", default="K64")
    a = ap.parse_args(argv)
    files = sorted(f for d in a.runs for f in Path(d).glob("func_lm_pilot_*.pt"))
    runs = {}
    for f in files:
        run = fa.Run.from_file(f)
        runs[run.label] = (run, f)
    if "FP" not in runs:
        raise ValueError("the full-precision calibration run is required (the rule reads its net harms)")
    states = ["FP"] + sorted(s for s in runs if s != "FP")
    languages = list(runs["FP"][0].languages)
    ratios = calibration_ratios()
    rows = {s: calibration_rows(runs[s][0], languages, ratios, a.label, a.n_controls) for s in states}
    z = zero_net_median(rows["FP"])
    down = grid()[0]
    net = {g: [r for r in rows["FP"] if r["ratio"] == g][0]["net_median"] for g in down}
    sel = select_ratios(net, z)
    meta = {"states": states, "files": {s: {"name": runs[s][1].name, "sha256": _sha256(runs[s][1])} for s in states}, "z": z, "languages": languages, "n_controls": a.n_controls,
            "label": a.label, "n_sentences": len(runs["FP"][0].ids)}
    write_outputs(a.out, rows, sel, meta)
    print(f"wrote {a.out}: down {sel['down']} up {sel['up']} branch {sel['branch']} z {z:.4f}")


if __name__ == "__main__":
    main()
