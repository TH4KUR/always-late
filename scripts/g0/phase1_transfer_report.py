"""Report driver of the intervention-transfer study (D-088; ledger reports/intervention_transfer/STATUS.md; plan Task 3). Joins the analysis functions of phase1_transfer_analysis.py to the chunk directories of the
full runs: per quantized state the coverage, the component table, the specificity contrast, H4 (averaged fingerprints), the predictive diagnostic; per state of the group panel the net C/W/I decomposition by language and
the A3 summary; and a claim status by RULES WRITTEN BEFORE ANY FULL-RUN OUTPUT EXISTED (the CLAIM_RULES below, recorded in the ledger on 2026-10-26).

CLAIM RULES (all on the Gemma-2-2B runs, INT6 and INT4 judged separately; INT16 is the rounding reference and never a result):
  A3  : supported at a state iff the lower end of the interval of the median over languages of |C| - |W| (group level, net harm, sentence bootstrap) is above zero; reversed iff the upper end is below zero;
        otherwise inconclusive. The overall status needs the same verdict at INT6 and at INT4, else 'mixed'.
  H1  : supported at a state iff, for the real direction over the eligible items, the lower ends of the intervals of BOTH the median share of the error left after the best rescaling of the FP effect and the median
        share left after the native amplitude exceed 0.5; not supported iff the upper end of either is below 0.5; otherwise inconclusive.
  H2  : descriptive (no frozen threshold): the medians of the relative norms of C, W and I with intervals, and 'reproducible' = the calibration and held-out intervals overlap for all three.
  H3  : supported at a state iff the interval of the relative held-out MAE reduction of the diagnostic over baseline A alone is above zero; reversed iff below zero; otherwise inconclusive.
  H4  : supported at a state iff the interval of the median cosine gap (averaged fingerprint minus the row-weighted context mean, real direction, all features) excludes zero; inconclusive otherwise.
  UTILITY (project criterion): met at a state iff the diagnostic reduces the held-out MAE by at least 10 % relative to the strongest baseline with a paired interval above zero; overall needs INT6 and INT4.
  Every status carries its population (the number of eligible items and features) and the fraction of items that were eligible; a result on a few hundred items is reported as such."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

import phase1_transfer_analysis as ta

REFERENCE_STATE = "INT16"
JUDGED_STATES = ("INT6", "INT4")


def _records(coverage):
    return [{"kind": k, "split": s, **v} for (k, s), v in sorted(coverage.items())]


def jsonable(x):
    """Plain Python for JSON: numpy scalars and arrays become numbers and lists, tuples lists, and the per-item absolute errors of the diagnostic are dropped (they are the inputs of the bootstrap, not results)."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items() if k != "abs_error"}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.floating,)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, float) and np.isnan(x):
        return None
    return x


def feature_level(dirs, core, split, cluster_mod, n_boot=ta.BOOTSTRAP_REPLICATES, seed=ta.BOOTSTRAP_SEED, floor_state=REFERENCE_STATE):
    """The feature-level results of every state in `dirs` ({state: chunk directory}); the floor of every direction kind is fixed from the calibration items of `floor_state` (INT16) and applied to all states. The
    items of the states must be the same set of (feature, sentence, direction) keys, else an error."""
    if floor_state not in dirs:
        raise ValueError(f"the floor state {floor_state} is not among the states {sorted(dirs)}")
    loaded = {s: ta.load_chunks(d, "transfer", s) for s, d in dirs.items()}; ref_state = floor_state; ref_keys = set(loaded[ref_state][0])
    for s, (items, _, _) in loaded.items():
        if set(items) != ref_keys:
            raise ValueError(f"state {s} has {len(set(items) ^ ref_keys)} item keys that {ref_state} does not share: the states must hold the same feature-context-direction items")
    raw = {s: ta.item_table(loaded[s][0], core, split, cluster_mod) for s in dirs}; floors = ta.floors_by_kind(raw[floor_state]); out = {"floors": floors, "floor_state": floor_state, "states": {}}
    for s in dirs:
        rows = ta.apply_floor(raw[s], floors); res = {"reference": s == floor_state, "n_items": len(rows), "n_skipped": len(loaded[s][2]), "rows": rows, "coverage": _records(ta.coverage(rows)),
                                                       "components": ta.component_summary(rows, n_boot, seed), "specificity": ta.specificity_summary(rows, n_boot, seed)}
        res["averaged_features"] = ta.averaged_fingerprints(rows, loaded[s][1]); res["averaged"] = ta.averaged_summary(res["averaged_features"], n_boot, seed); res["predictive"] = {}
        for variant, eligible in (("eligible", True), ("all_items", False)):
            d = ta.design(rows, eligible_only=eligible); res["predictive"][variant] = {"n_design": int(len(d["y"])), "n_dropped": d["n_dropped"], "dropped_reasons": d["dropped_reasons"], "result": ta.diagnostic(d, n_boot, seed)}
        out["states"][s] = res
    return out


def group_level(dirs, languages, n_sentences, n_controls, n_boot=ta.BOOTSTRAP_REPLICATES, seed=ta.BOOTSTRAP_SEED):
    """The group-level results of every state in `dirs`: per language the pooled harm of the group, of the mean control and the net harm in the five cells, its decomposition (C, W, I, total, native), and the A3
    summary with its sentence bootstrap."""
    out = {"languages": list(languages), "states": {}}
    for s, d in dirs.items():
        items, _, skipped = ta.load_chunks(d, "group", s); A = ta.group_arrays(items, languages, n_sentences, n_controls); pool = ta.pooled_harms(A); dec = ta.decompose_net(pool["net"]); table = []
        for li, lang in enumerate(languages):
            rec = {"language": lang, "n_sentences": int(pool["n_sentences"][li]), "n_sentences_total": int(n_sentences)}
            for name, arr in (("group", pool["group"]), ("control", pool["control"]), ("net", pool["net"])):
                rec.update({f"{name}_{c}": float(arr[li, ci]) for ci, c in enumerate(("FP_FP", "Q_FP", "FP_Q", "Q_Q", "Q_Q_native"))})
            rec.update({k: float(v[li]) for k, v in dec.items()}); table.append(rec)
        out["states"][s] = {"reference": s == REFERENCE_STATE, "n_items": len(items), "n_skipped": len(skipped), "per_language": table, "a3": ta.bootstrap_a3(A, n_boot, seed)}
    return out


def _pick(records, **kw):
    hit = [r for r in records if all(r[k] == v for k, v in kw.items())]
    return hit[0] if len(hit) == 1 else None


def _verdict(low, high, threshold=0.0):
    if low is None or high is None or np.isnan(low) or np.isnan(high):
        return "inconclusive"
    return "supported" if low > threshold else ("reversed" if high < threshold else "inconclusive")


def claim_status(feat, group):
    """The status of each pre-declared claim by the CLAIM_RULES of the module docstring. feat, group: the results of `feature_level` and `group_level` (either may be None: the claims that need it are then
    'not evaluated')."""
    out = {"rules": "CLAIM_RULES of phase1_transfer_report.py (fixed 2026-10-26 before any full-run output)", "population": "Gemma-2-2B, layer-12 Gemma Scope 16k SAE, RTN per-output-channel; INT16 is the rounding reference", "claims": {}}
    claims = out["claims"]; combine = lambda per: (next(iter(set(per.values()))) if len(set(per.values())) == 1 and per else "mixed")
    if group is not None:
        per = {}
        for s in JUDGED_STATES:
            st = group["states"].get(s); per[s] = "not evaluated" if st is None else _verdict(st["a3"]["median_difference"][0], st["a3"]["median_difference"][1])
        claims["A3"] = {"per_state": per, "overall": combine(per) if "not evaluated" not in per.values() else "not evaluated"}
    if feat is not None:
        h1, h3, h4, util, h2, pop = {}, {}, {}, {}, {}, {}
        for s in JUDGED_STATES:
            st = feat["states"].get(s)
            if st is None:
                for d in (h1, h3, h4, util, h2):
                    d[s] = "not evaluated"
                continue
            comp = st["components"]; sr = _pick(comp, kind="real", split="all", metric="scalar_residual_share"); ne = _pick(comp, kind="real", split="all", metric="native_error_share")
            if sr is None or ne is None or sr["n"] == 0 or ne["n"] == 0:
                h1[s] = "inconclusive"
            elif sr["low"] > 0.5 and ne["low"] > 0.5:
                h1[s] = "supported"
            elif sr["high"] < 0.5 or ne["high"] < 0.5:
                h1[s] = "not supported"
            else:
                h1[s] = "inconclusive"
            pred = st["predictive"]["eligible"]["result"]
            if pred["status"] != "fitted":
                h3[s] = "inconclusive"; util[s] = "not met (insufficient items)"
            else:
                h3[s] = _verdict(pred["improvement_over_A"]["low"], pred["improvement_over_A"]["high"]); util[s] = "met" if pred["criterion_met"] else "not met"
            av = _pick(st["averaged"], kind="real", split="all", metric="cos_gap"); h4[s] = "inconclusive" if av is None or av["n"] == 0 else ("supported" if av["low"] > 0 or av["high"] < 0 else "inconclusive")
            cov = [r for r in st["coverage"] if r["kind"] == "real"]; n_all = sum(r["items"] for r in cov); n_el = sum(r["eligible"] for r in cov)
            pop[s] = {"real_items": n_all, "eligible_real_items": n_el, "eligible_fraction": (n_el / n_all) if n_all else None}
            rel = {}
            for m in ("C_rel", "W_rel", "I_rel"):
                c, h = _pick(comp, kind="real", split="calibration", metric=m), _pick(comp, kind="real", split="heldout", metric=m)
                rel[m] = None if c is None or h is None or c["n"] == 0 or h["n"] == 0 else bool(c["low"] <= h["high"] and h["low"] <= c["high"])
            h2[s] = {"reproducible": None if None in rel.values() else all(rel.values()), "overlap_by_component": rel}
        claims["H1"] = {"per_state": h1, "overall": combine(h1) if "not evaluated" not in h1.values() else "not evaluated"}; claims["H2"] = {"per_state": h2, "status": "descriptive"}
        claims["H3"] = {"per_state": h3, "overall": combine(h3) if "not evaluated" not in h3.values() else "not evaluated"}; claims["H4"] = {"per_state": h4, "overall": combine(h4) if "not evaluated" not in h4.values() else "not evaluated"}
        claims["UTILITY"] = {"per_state": util, "overall": "met" if all(v == "met" for v in util.values()) else "not met"}; claims["POPULATION"] = pop
    return out


def write_csv(path, records):
    """Records (dicts) to CSV with the union of the keys in first-seen order; tuples are written as 'low;high'."""
    keys = []
    for r in records:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in records:
            w.writerow({k: (";".join(str(x) for x in v) if isinstance(v, (tuple, list)) else v) for k, v in r.items()})


def write_outputs(out_dir, feat, group, claims):
    """Writes the tables (CSV), the full results (JSON) and the claim status; the per-item rows go to per_context_<state>.csv."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    if feat is not None:
        write_csv(out / "coverage.csv", [{"state": s, **r} for s, st in feat["states"].items() for r in st["coverage"]])
        write_csv(out / "component_summary.csv", [{"state": s, **r} for s, st in feat["states"].items() for r in st["components"]])
        write_csv(out / "specificity.csv", [{"state": s, **r} for s, st in feat["states"].items() for r in st["specificity"]])
        write_csv(out / "averaged_fingerprints.csv", [{"state": s, **r} for s, st in feat["states"].items() for r in st["averaged"]])
        pv = []
        for s, st in feat["states"].items():
            for variant, p in st["predictive"].items():
                r = p["result"]; base = {"state": s, "variant": variant, "n_design": p["n_design"], "n_dropped": p["n_dropped"], "status": r["status"], "n_calibration": r["n_calibration"], "n_heldout": r["n_heldout"]}
                if r["status"] != "fitted":
                    pv.append(base); continue
                for name, m in r["models"].items():
                    pv.append({**base, "model": name, "lambda": m["lambda"], "cv_mae": m["cv_mae"], "heldout_mae": m["heldout_mae"], "null_mae": r["null_mae"], "strongest_baseline": r["strongest_baseline"],
                               "improvement_diagnostic": r["improvement"]["point"] if name == r["diagnostic"] else None, "improvement_low": r["improvement"]["low"] if name == r["diagnostic"] else None,
                               "improvement_high": r["improvement"]["high"] if name == r["diagnostic"] else None, "criterion_met": r["criterion_met"] if name == r["diagnostic"] else None})
        write_csv(out / "predictive_validation.csv", pv)
        for s, st in feat["states"].items():
            write_csv(out / f"per_context_{s}.csv", [{k: v for k, v in r.items()} for _, r in sorted(st["rows"].items())])
            write_csv(out / f"averaged_features_{s}.csv", [r for _, r in sorted(st["averaged_features"].items())])
    if group is not None:
        write_csv(out / "group_per_language.csv", [{"state": s, **r} for s, st in group["states"].items() for r in st["per_language"]])
        write_csv(out / "a3.csv", [{"state": s, **{k: v for k, v in st["a3"]["point"].items()}, **{f"{k}_ci": st["a3"][k] for k in ("median_difference", "median_abs_c", "median_abs_w", "c_gt_w")}, "share_holds": st["a3"]["share_holds"]} for s, st in group["states"].items()])
    (out / "claim_status.json").write_text(json.dumps(jsonable(claims), indent=1, sort_keys=True) + "\n")
    summary = {"feature_level": None if feat is None else {"floors": feat["floors"], "states": {s: {k: v for k, v in st.items() if k not in ("rows", "averaged_features")} for s, st in feat["states"].items()}},
               "group_level": None if group is None else group}
    (out / "analysis_summary.json").write_text(json.dumps(jsonable(summary), indent=1, sort_keys=True) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--freeze", required=True); ap.add_argument("--core", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--feature-dir", action="append", default=[], metavar="STATE=DIR"); ap.add_argument("--group-dir", action="append", default=[], metavar="STATE=DIR")
    ap.add_argument("--n-boot", type=int, default=ta.BOOTSTRAP_REPLICATES); ap.add_argument("--seed", type=int, default=ta.BOOTSTRAP_SEED)
    ap.add_argument("--cluster-mod", type=int, help="items with the same sentence index modulo this number share a bootstrap cluster (default: the dev sentences per language of the freeze, 997: the parallel FLORES sentence); the external panel uses a value above the number of passages")
    ap.add_argument("--group-languages", nargs="+", help="the languages of the group-level directories in the order of the run (default: the languages of the core; the external panel: its six)"); ap.add_argument("--group-sentences", type=int, help="sentences per language of the group-level run (default: the freeze's 300)")
    a = ap.parse_args(argv)
    freeze = json.loads(Path(a.freeze).read_text()); core = json.loads(Path(a.core).read_text()); spec = freeze["spec"]; parse = lambda xs: {x.split("=", 1)[0]: x.split("=", 1)[1] for x in xs}
    feat = feature_level(parse(a.feature_dir), core, freeze["split"], a.cluster_mod or spec["feature_subset"]["dev_per_language"], a.n_boot, a.seed) if a.feature_dir else None
    group = group_level(parse(a.group_dir), a.group_languages or core["languages"], a.group_sentences or spec["group_panel"]["evaluation_sentences"], spec["group_panel"]["control_draws"], a.n_boot, a.seed) if a.group_dir else None
    write_outputs(a.out, feat, group, claim_status(feat, group))
    print(json.dumps(jsonable(claim_status(feat, group)["claims"]), indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
