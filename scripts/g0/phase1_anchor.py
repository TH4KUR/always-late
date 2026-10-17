"""S5 anchor: compare the summary of our Phase 1 runner (wikitext corpus, regime none, their 3a_test budget, batch size 1) with the vendored smoke run of their own
script (reports/g0/baseline_smoke_vendored/phase3_summary_final.csv). The tolerances are fixed here BEFORE the run (PLAN.md, S5): they allow for GPU / bf16
numerics, not for a different method. Conditions are matched by name; the bits label of the FP row is ignored (ours is 32 by wandb schema v1, theirs 16).

  python3 scripts/g0/phase1_anchor.py OURS_summary.csv [REF_summary.csv]     exit 0 only if every field of every common condition is inside its tolerance"""
import csv, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REF_DEFAULT = ROOT / "reports/g0/baseline_smoke_vendored/phase3_summary_final.csv"
TOL = {"perplexity": ("rel", 0.005), "n_active_features": ("abs", 10), "survived_>0.9_pct": ("abs", 0.5), "degraded_0.5_0.9_pct": ("abs", 0.5), "damaged_<0.5_pct": ("abs", 0.5),
       "median_corr": ("abs", 0.005), "mean_corr": ("abs", 0.005), "n_tokens": ("abs", 0)}


def read(path):
    with open(path, newline="") as f:
        return {r["condition"]: {k: (float(v) if k not in ("condition", "method") else v) for k, v in r.items()} for r in csv.DictReader(f)}


def compare_summary(ours_csv, ref_csv=REF_DEFAULT):
    """{"failures": [{condition, field, ours, ref, detail}], "compared_conditions": [...]}; a condition of the reference that ours lacks is a failure."""
    ours, ref = read(ours_csv), read(ref_csv)
    common = [c for c in ref if c in ours]; fails = []
    missing = [c for c in ref if c not in ours]
    if missing or not common:
        fails.append({"condition": None, "field": "conditions", "ours": sorted(ours), "ref": sorted(ref), "detail": f"missing in ours: {missing}"})
    for c in common:
        for field, (kind, tol) in TOL.items():
            o, r = ours[c][field], ref[c][field]
            d = abs(o - r) / abs(r) if kind == "rel" else abs(o - r)
            if d > tol + 1e-12:
                fails.append({"condition": c, "field": field, "ours": o, "ref": r, "detail": f"{kind} difference {d:.6g} > {tol}"})
    return {"failures": fails, "compared_conditions": common}


if __name__ == "__main__":
    r = compare_summary(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else REF_DEFAULT)
    print(json.dumps(r, indent=1)); sys.exit(1 if r["failures"] else 0)
