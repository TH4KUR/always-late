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


LADDER_STATES = ["INT16", "INT8", "INT6", "INT5", "INT4"]


def ladder_arrays(main, controls, states=LADDER_STATES, q=0.05, floor_state="INT16"):
    """The swap control along the precision ladder (plan Task 15f): the W1 of the features (run `main`, merged jobs allowed) and of the swapped directions (the control runs, which may split the states between them, for
    example the high precisions in one job and INT4 in another) on the features that EVERY control run tested, per state of `states`: {"n_common", "q", "floor_state", "floor": {"tau_a", "tau_b"}, "states": {s: {"a": features,
    "b": control}}, "missing"}. A state is taken from the first control run that has it; the floors are those of `pair_arrays` for the first control run (the features' floor on the common features and the control's own
    INT16 floor). Control runs that do not test the same features are an error (one set of features for the whole ladder)."""
    if not controls:
        raise ValueError("no control run")
    pairs = [pair_arrays(main, c, states, q, floor_state) for c in controls]; rows = [align(main["stage"], c["stage"])[0] for c in controls]
    for r in rows[1:]:
        if len(r) != len(rows[0]) or not np.array_equal(r, rows[0]):
            raise ValueError("the control runs do not test the same features")
    out, missing = {}, []
    for s in states:
        src = next((p for p in pairs if s in p["states"]), None)
        if src is None:
            missing.append(s); continue
        out[s] = src["states"][s]
    return {"n_common": pairs[0]["n_common"], "q": q, "floor_state": floor_state, "floor": pairs[0]["floor"], "states": out, "missing": missing}


def ladder_stats(la):
    """Per state of `ladder_arrays`: the number of features with both values defined (`n`) and the number left out (`n_undefined`), median and quartiles of W1 for the features (`_a`) and the swapped directions (`_b`), the paired median
    difference and the difference of the medians (control minus feature), the shortfall ratio (1 - control median) / (1 - feature median) (NaN when the feature median is 1) and the shares at or above each run's own floor. A state without
    a defined pair has n = 0 and NaN values."""
    nan = float("nan"); t = la["floor"]; out = {}
    for s, v in la["states"].items():
        a, b = np.asarray(v["a"], dtype=np.float64), np.asarray(v["b"], dtype=np.float64); d = ~(np.isnan(a) | np.isnan(b)); x, y = a[d], b[d]; r = {"n": int(d.sum()), "n_undefined": int((~d).sum())}
        if x.size == 0:
            r.update({k: nan for k in ("median_a", "q25_a", "q75_a", "median_b", "q25_b", "q75_b", "median_diff", "diff_of_medians", "shortfall_ratio", "above_a", "above_b")}); out[s] = r; continue
        r.update({"median_a": float(np.median(x)), "q25_a": float(np.percentile(x, 25)), "q75_a": float(np.percentile(x, 75)), "median_b": float(np.median(y)), "q25_b": float(np.percentile(y, 25)), "q75_b": float(np.percentile(y, 75)),
                  "median_diff": float(np.median(y - x)), "above_a": float((x >= t["tau_a"]).mean()), "above_b": float((y >= t["tau_b"]).mean())})
        r["diff_of_medians"] = r["median_b"] - r["median_a"]; r["shortfall_ratio"] = (1 - r["median_b"]) / (1 - r["median_a"]) if r["median_a"] < 1 else nan; out[s] = r
    return out


LADDER_COLUMNS = [("state", None), ("n", "n"), ("n_undefined", "n_undefined"), ("median_features", "median_a"), ("q25_features", "q25_a"), ("q75_features", "q75_a"), ("median_control", "median_b"), ("q25_control", "q25_b"), ("q75_control", "q75_b"),
                  ("diff_of_medians", "diff_of_medians"), ("median_paired_diff", "median_diff"), ("shortfall_ratio", "shortfall_ratio"), ("above_floor_features", "above_a"), ("above_floor_control", "above_b")]


def ladder_csv_rows(la):
    """One dict per state (the columns of LADDER_COLUMNS) for the table twin of the ladder figure."""
    return [{name: (s if key is None else v[key]) for name, key in LADDER_COLUMNS} for s, v in ladder_stats(la).items()]


def ladder_report_text(la, label):
    st = ladder_stats(la); f = la["floor"]
    L = [f"# F5 swap control along the precision ladder: {label}", "",
         f"This comparison is exploratory. The features tested in the feature run and in every control run: {la['n_common']}. W1 is the cosine of the effect fingerprint in the quantized state with the one in full precision; the control ablates "
         f"the decoder direction of a matched different feature at the same positions. Each run's numerical floor is the {int(la['q'] * 100)}th percentile of W1 at {la['floor_state']} over these features: {f['tau_a']:.4f} (features) and {f['tau_b']:.4f} (control). "
         "The shortfall ratio is the control's shortfall below 1 (1 minus the median W1) over the features'.", "",
         "| state | features | median features | quartiles features | median control | quartiles control | median difference (control - features) | paired median difference | shortfall ratio | at or above floor features | at or above floor control |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s, c in st.items():
        L.append(f"| {s} | {c['n']} | {c['median_a']:.4f} | {c['q25_a']:.4f} to {c['q75_a']:.4f} | {c['median_b']:.4f} | {c['q25_b']:.4f} to {c['q75_b']:.4f} | {c['diff_of_medians']:+.4f} | {c['median_diff']:+.4f} | {c['shortfall_ratio']:.2f} | {c['above_a']:.3f} | {c['above_b']:.3f} |")
    if la["missing"]:
        L += ["", f"Not compared (no control run has them): {', '.join(la['missing'])}."]
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
