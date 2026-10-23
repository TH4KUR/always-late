"""Sensitivity comparison of two F5 runs on the same features (design section 11.3, plan Task 15a): is the write-side similarity W1 of a feature (the cosine of its effect fingerprint in a quantized state with the one in full
precision) stable when the number of context sentences changes (M = 8 or 32 against the main run's 16)? Aligns the tested features of the two runs by feature id, and compares W1 state by state: the medians, the median and
mean absolute difference, the rank correlation, the share at or above each run's own numerical floor (the q-quantile of W1 at the floor state, default INT16, within the compared features) and the share of features whose
above-floor classification flips. CPU only, numpy; reads the files of the runner through the analysis module's loader. Exploratory.

    ~/.venvs/anlp-torch/bin/python scripts/g0/phase1_func_f5_sensitivity.py --main-run <A> --main-run <B> --sens-run <M8 dir> --label "M = 8 against M = 16" --states INT16 INT4 --out <dir>"""
import argparse
import json
from pathlib import Path

import numpy as np

import phase1_func_f5_analysis as an
import phase1_func_f5_metrics as met


def align(stage_a, stage_b):
    """Indices (into the write rows of each run) of the features tested in both runs, in ascending feature id."""
    fa, fb = list(stage_a["write_features"]), list(stage_b["write_features"]); common = sorted(set(fa) & set(fb))
    if not common:
        raise ValueError("the two runs have no tested feature in common")
    pa, pb = {f: i for i, f in enumerate(fa)}, {f: i for i, f in enumerate(fb)}
    return np.array([pa[f] for f in common]), np.array([pb[f] for f in common])


def compare_state(a, b, tau_a, tau_b):
    """Two measurements a (reference) and b of W1 for the same features: the summary described in the module docstring; undefined values (NaN in either) are left out and counted."""
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64); d = ~(np.isnan(a) | np.isnan(b)); x, y = a[d], b[d]
    return {"n": int(d.sum()), "n_undefined": int((~d).sum()), "median_a": float(np.median(x)) if x.size else float("nan"), "median_b": float(np.median(y)) if y.size else float("nan"),
            "median_diff": float(np.median(y - x)) if x.size else float("nan"), "mean_abs_diff": float(np.mean(np.abs(y - x))) if x.size else float("nan"), "spearman": met.spearman(x, y)["rho"],
            "above_a": float((x >= tau_a).mean()) if x.size else float("nan"), "above_b": float((y >= tau_b).mean()) if x.size else float("nan"),
            "flip_share": float(((x >= tau_a) != (y >= tau_b)).mean()) if x.size else float("nan")}


def pair_arrays(a, b, states, q=0.05, floor_state="INT16"):
    """The aligned per-feature W1 arrays of two runs for every state of `states` that both have ({"a": reference, "b": variant}, ascending feature id), the common count, the floor of each run (the q-quantile of W1 at
    `floor_state` over the common features) and the states missing in one run."""
    ia, ib = align(a["stage"], b["stage"]); cos = lambda r, s, i: np.asarray(r["states"][s]["write"]["cos"], dtype=np.float64)[i]
    if floor_state not in a["states"] or floor_state not in b["states"]:
        raise ValueError(f"the floor state {floor_state} is missing in one of the runs")
    taus = [met.floor_threshold(cos(r, floor_state, i), q) for r, i in ((a, ia), (b, ib))]; out, missing = {}, []
    for s in states:
        if s not in a["states"] or s not in b["states"]:
            missing.append(s); continue
        out[s] = {"a": cos(a, s, ia), "b": cos(b, s, ib)}
    return {"n_common": int(len(ia)), "q": q, "floor_state": floor_state, "floor": {"tau_a": taus[0], "tau_b": taus[1]}, "states": out, "missing": missing}


def compare_runs(a, b, states, q=0.05, floor_state="INT16"):
    """a, b: {"stage": stage_a, "states": states} of two runs (a = the reference). Compares every state of `states` that both runs have; the floor of each run is the q-quantile of W1 at `floor_state` over the common features."""
    pa = pair_arrays(a, b, states, q, floor_state); t = pa["floor"]
    return {"n_common": pa["n_common"], "q": q, "floor_state": floor_state, "floor": t, "states": {s: compare_state(v["a"], v["b"], t["tau_a"], t["tau_b"]) for s, v in pa["states"].items()}, "missing": pa["missing"]}


def report_text(cmp, label):
    L = [f"# F5 sensitivity: {label}", "",
         f"This comparison is exploratory. The tested features common to the two runs: {cmp['n_common']}. W1 is the cosine of the effect fingerprint in the quantized state with the one in full precision. Run a is the reference, run b the variant. "
         f"The numerical floor of each run is the {int(cmp['q'] * 100)}th percentile of W1 at {cmp['floor_state']} over the common features: {cmp['floor']['tau_a']:.4f} (a) and {cmp['floor']['tau_b']:.4f} (b).", "",
         "| state | features | median W1 a | median W1 b | median difference (b - a) | mean abs difference | Spearman | at or above floor a | at or above floor b | classification flips |", "|---|---|---|---|---|---|---|---|---|---|"]
    for s, c in cmp["states"].items():
        L.append(f"| {s} | {c['n']} | {c['median_a']:.4f} | {c['median_b']:.4f} | {c['median_diff']:+.4f} | {c['mean_abs_diff']:.4f} | {c['spearman']:.3f} | {c['above_a']:.3f} | {c['above_b']:.3f} | {c['flip_share']:.3f} |")
    if cmp["missing"]:
        L += ["", f"Not compared (missing in one run): {', '.join(cmp['missing'])}."]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compare W1 of the same features between a reference F5 run and a sensitivity run.")
    ap.add_argument("--main-run", action="append", required=True); ap.add_argument("--sens-run", action="append", required=True); ap.add_argument("--label", required=True)
    ap.add_argument("--states", nargs="*", default=["INT16", "INT4"]); ap.add_argument("--floor-state", default="INT16"); ap.add_argument("--q", type=float, default=0.05); ap.add_argument("--out", required=True)
    ap.add_argument("--figures", action="store_true", help="also draw the paired figure (W1 of the reference run against the variant run, per state)"); ap.add_argument("--label-a", default="reference run"); ap.add_argument("--label-b", default="variant run")
    ap.add_argument("--figure-note", default="exploratory"); ap.add_argument("--figure-title", default="Is W1 the same in the two runs?")
    a = ap.parse_args(argv)
    load = lambda dirs: dict(zip(("stage", "states", "meta"), an.load_runs(dirs)))
    ra, rb = load(a.main_run), load(a.sens_run); cmp = compare_runs(ra, rb, a.states, a.q, a.floor_state)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "sensitivity.json").write_text(json.dumps(an.to_jsonable(cmp), indent=1, allow_nan=False) + "\n"); (out / "sensitivity.md").write_text(report_text(cmp, a.label))
    if a.figures:
        import phase1_func_f5_figures as ff
        ff.write_pair_figure(pair_arrays(ra, rb, a.states, a.q, a.floor_state), out, "pair", a.figure_note, a.figure_title, a.label_a, a.label_b)
    print("wrote", out, "|", a.label, "|", {s: (round(c["spearman"], 3), round(c["median_diff"], 4), round(c["flip_share"], 3)) for s, c in cmp["states"].items()})


if __name__ == "__main__":
    main()
