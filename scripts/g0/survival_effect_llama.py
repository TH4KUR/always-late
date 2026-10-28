"""Llama-3.2-1B F5 check of the survival-and-intervention plan (Stage 5; addendum reports/survival_effect_rescue/llama/llama_f5_addendum_2026-10-28.md). The statistical functions are those of
`survival_effect.py` UNCHANGED (the Gemma Stage 3 analysis, D-113); this module adds only what is specific to Llama: the table of a Llama F5 run (baselines from the sample file's native activity), the instrument
gate with Llama's OWN floor, and the labels of the side-by-side with Gemma.

  python3 scripts/g0/survival_effect_llama.py --run <job dir> --sample reports/survival_effect_rescue/llama/llama_f5_sample.json --out <dir>"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import survival_effect as se  # noqa: E402

met = se.met
ROOT = Path(__file__).resolve().parents[2]
ADEQUATE_MEDIAN = 0.7          # the F5 design rule (phase1_func_f5_analysis.ADEQUATE_MEDIAN)
MIN_RELIABLE = 300


def feature_table(sa, states, sample):
    """The same table as survival_effect.feature_table, for a Llama run: baselines (n_active from stage A, the native firing rate and mean activation from the sample file, |dnll| and n_pos from the FP state),
    the INT16 reference and, for INT6 and INT4, R1, jaccard, scale deviation and W1."""
    rows = np.asarray(sa["write_rows"]).astype(int); ids = np.asarray(sa["feat_ids"]).astype(int)[rows]; fl = np.asarray(sa["feat_lang"]).astype(int)[rows]; feats = sample["features"]
    for k in ids.tolist():
        if str(k) not in feats:
            raise ValueError(f"feature {k} of the run is not in the sample")
    fpw = states["FP"]["write"]
    base = np.column_stack([np.asarray(sa["n_active"], dtype=np.float64)[rows], [feats[str(k)]["native"]["FP"][0] for k in ids.tolist()], [feats[str(k)]["native"]["FP"][1] for k in ids.tolist()],
                            np.abs(np.asarray(fpw["dnll"], dtype=np.float64)), np.asarray(fpw["n_pos"], dtype=np.float64)])
    T = {"ids": ids, "lang": fl, "base": base, "dnll_fp": np.asarray(fpw["dnll"], dtype=np.float64), "cos16": np.asarray(states["INT16"]["write"]["cos"], dtype=np.float64), "dnll16": np.asarray(states["INT16"]["write"]["dnll"], dtype=np.float64)}
    for s in se.STATES:
        r, w = states[s]["read"], states[s]["write"]
        T[s] = {"r1": np.asarray(r["corr"], dtype=np.float64)[rows], "jaccard": np.asarray(r["jaccard"], dtype=np.float64)[rows], "scale_dev": np.abs(np.asarray(r["scale"], dtype=np.float64)[rows] - 1.0), "w1": np.asarray(w["cos"], dtype=np.float64)}
    return T


def own_floor(cos16, q=0.05):
    """The 5th percentile of the INT16 cosine of THIS system (Gemma's 0.9489 does not transfer)."""
    return float(met.floor_threshold(cos16, q if q < 1 else q / 100.0))


def instrument_gate(cos16, min_reliable=MIN_RELIABLE, adequate_median=ADEQUATE_MEDIAN):
    """Gate 5 of the addendum: the instrument is adequate if the median FP-versus-INT16 cosine is at least 0.7 and at least `min_reliable` features reach this system's floor."""
    c = np.asarray(cos16, dtype=np.float64); floor = own_floor(c); med = float(np.nanmedian(c)); n_rel = int((c >= floor).sum()); why = []
    if not med >= adequate_median:
        why.append(f"median INT16 cosine {med:.3f} is below {adequate_median}")
    if n_rel < min_reliable:
        why.append(f"only {n_rel} reliable features (fewer than {min_reliable})")
    return {"ok": not why, "why": why, "floor": floor, "median_cos16": med, "n_reliable": n_rel, "n_features": int(len(c))}


def association_label(lo, hi):
    """'repeats' if the partial Spearman interval of the Llama cell lies above 0, 'reverses' if below 0, else 'cannot be resolved' (undefined intervals included)."""
    if lo == lo and hi == hi:
        if lo > 0:
            return "repeats"
        if hi < 0:
            return "reverses"
    return "cannot be resolved"


def value_label(cell_met):
    return "repeats" if cell_met else "does not repeat"


def analyse(T, n_boot=se.N_BOOT):
    """All populations and both cells of a Llama run; the floor and the noise threshold are the system's own (INT16 reference only)."""
    gate = instrument_gate(T["cos16"])
    out = {"gate": gate, "cells": {}}
    if not gate["ok"]:
        out["overall_statement"] = "instrument inadequate: no relation reported"; return out
    tau = gate["floor"]; eps = se.reference_noise(T["dnll_fp"], T["dnll16"]); rel = se.reliable_mask(T["cos16"], tau)
    pops = {"primary": rel, "noisy_stratum": ~rel, "S_all": np.ones(len(rel), dtype=bool), "S_loss": se.loss_mask(T["dnll_fp"], eps)}
    out.update({"tau": tau, "eps_dnll": eps, "n_features": int(len(rel)), "population_counts": {k: int(v.sum()) for k, v in pops.items()}, "n_boot": n_boot, "seed": se.SEED})
    for s in se.STATES:
        Xs = se.design(T["base"], T[s]["r1"], T[s]["jaccard"], T[s]["scale_dev"])
        for name, mask in pops.items():
            out["cells"][f"{s}|{name}"] = se.analyse_cell(Xs, T[s]["w1"], T["lang"], mask, tau=tau, n_boot=n_boot)
    prim = [out["cells"][f"{s}|primary"]["cell_met"] for s in se.STATES]
    out["cells_met_primary"] = int(sum(prim)); out["overall_statement"] = se.overall_statement(prim)
    return out


def load_run(run_dir):
    import phase1_func_f5_analysis as fa
    sa, states, meta = fa.load_runs([str(run_dir)])
    return sa, states, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--run", required=True); ap.add_argument("--sample", default=str(ROOT / "reports/survival_effect_rescue/llama/llama_f5_sample.json"))
    ap.add_argument("--out", required=True); ap.add_argument("--n-boot", type=int, default=se.N_BOOT); a = ap.parse_args(argv)
    sa, states, meta = load_run(a.run); T = feature_table(sa, states, json.loads(Path(a.sample).read_text())); res = analyse(T, a.n_boot)
    import phase1_func_f5_analysis as fa
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); (out / "llama_survival_effect_results.json").write_text(json.dumps(fa.to_jsonable(res), indent=1, allow_nan=False))
    print("gate:", res["gate"]); print("overall:", res["overall_statement"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
