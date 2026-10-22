"""F4 subgroups (plan Task 7c, design written 2026-10-22 before any code or number): for each language and precision b, the selective features that were established in full precision are split by
their D-035 status into LOST at b (`status != retained`) and RETAINED at b, and the lost features are matched to retained features of equal NATIVE activity at b (the activity in the quantized
model, from the same status file). Zero-ablating a feature that never fires is a no-op, so the comparison is only meaningful at equal native activity; a lost feature that is inactive at b has no
partner and is excluded and counted.

Matching: x1 = log10(target_rate at b), x2 = log10(target_mean at b), distances standardised by the pooled SD of the language's established features at b, greedy nearest neighbour without
replacement (lost features in decreasing x1), a Euclidean caliper (default 0.25 SD). The lost subset is capped (default 64) by a seeded hash of the feature id, never by an outcome. R retained draws
(default 3) are disjoint from each other and from the lost features and have exactly the size of the lost subset; a draw that cannot match every lost feature is dropped; a language with fewer than
`min_pairs` matched lost features is excluded at that precision. The manifest has the SAME schema as groups_manifest.json (the lost subset as the group, the retained draws as the K64 controls) so
that the unchanged, frozen runner and analysis compute E = harm(lost) - mean harm(retained matched), the F4 contrast, with `--manifest <file> --languages <included> --preset floor --precisions FP <b>`.
Hypothesis written before looking (H-F4, two-sided, exploratory): E_b differs from 0; "association predicts function" predicts E_b < 0."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def read_status(path):
    """Rows of the selective group of feature_status.csv: language, feature_id (int), condition, fp_status, status, rate and mean (native activity of the feature on its language at the condition)."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["group"] == "selective":
                rows.append({"language": r["original_language"], "feature_id": int(r["feature_id"]), "condition": r["condition"], "fp_status": r["fp_status"], "status": r["status"],
                             "rate": float(r["target_rate"]), "mean": float(r["target_mean"])})
    return rows


def _x(row):
    """(log10 rate, log10 mean) of the native activity, None for a feature that is inactive (rate or mean not positive)."""
    if row["rate"] <= 0 or row["mean"] <= 0:
        return None
    return (float(np.log10(row["rate"])), float(np.log10(row["mean"])))


def match(lost, pool, sd, caliper):
    """Greedy nearest-neighbour matching without replacement. lost: [(feature id, (x1, x2))] in processing order; pool: {feature id: (x1, x2)}; sd: (sd1, sd2). Returns {lost id: pool id} for the
    lost features that have an unused partner within the caliper (Euclidean distance of the standardised coordinates); ties go to the smaller feature id."""
    ids = sorted(pool)
    out = {}
    if not ids:
        return out
    pts = np.array([pool[i] for i in ids], dtype=float)
    avail = np.ones(len(ids), dtype=bool)
    for fid, (a, b) in lost:
        if not avail.any():
            break
        d = np.sqrt(((pts[:, 0] - a) / sd[0]) ** 2 + ((pts[:, 1] - b) / sd[1]) ** 2)
        d[~avail] = np.inf
        j = int(np.argmin(d))
        if d[j] <= caliper:
            out[fid] = ids[j]; avail[j] = False
    return out


def _hash_key(seed, condition, lang, fid):
    return hashlib.sha256(f"{seed}:{condition}:{lang}:{fid}".encode()).hexdigest()


def build_condition(rows, condition, languages, cap=64, R=3, min_pairs=8, caliper=0.25, seed=0):
    """{language: record} for one precision (see the module docstring). Record keys: lost (ids of the selected lost subset), draws (R' <= R lists of retained ids, each as long as `lost`),
    n_lost_total, n_inactive_lost, n_matched, n_selected, n_retained_pool, excluded (None or the reason), smd_x1 and smd_x2 (standardised mean differences lost minus retained, averaged over the draws)."""
    out = {}
    for lang in languages:
        est = [r for r in rows if r["language"] == lang and r["condition"] == condition and r["fp_status"] == "retained"]
        lost_x = {r["feature_id"]: _x(r) for r in est if r["status"] != "retained"}
        ret_x = {r["feature_id"]: _x(r) for r in est if r["status"] == "retained" and _x(r) is not None}
        n_inactive = sum(1 for v in lost_x.values() if v is None)
        allx = np.array([v for v in lost_x.values() if v is not None] + list(ret_x.values()), dtype=float).reshape(-1, 2)
        sd = tuple(float(s) if s > 0 else 1.0 for s in (allx.std(axis=0) if len(allx) else (1.0, 1.0)))
        candidates = sorted(((fid, x) for fid, x in lost_x.items() if x is not None), key=lambda t: (-t[1][0], t[0]))
        m0 = match(candidates, ret_x, sd, caliper)
        matched = [(fid, x) for fid, x in candidates if fid in m0]
        rec = {"lost": [], "draws": [], "n_lost_total": len(lost_x), "n_inactive_lost": n_inactive, "n_matched": len(matched), "n_selected": 0, "n_retained_pool": len(ret_x),
               "excluded": None, "smd_x1": None, "smd_x2": None}
        if len(matched) < min_pairs:
            rec["excluded"] = f"fewer than {min_pairs} matched pairs ({len(matched)} matched of {len(candidates)} lost features with native activity at {condition})"
            out[lang] = rec; continue
        sel = matched
        if len(matched) > cap:
            keep = {fid for fid, _ in sorted(matched, key=lambda t: _hash_key(seed, condition, lang, t[0]))[:cap]}
            sel = [t for t in matched if t[0] in keep]
        draws = [[m0[fid] for fid, _ in sel]]
        used = set(draws[0])
        for _ in range(1, R):
            mr = match(sel, {k: v for k, v in ret_x.items() if k not in used}, sd, caliper)
            if len(mr) < len(sel):
                break
            d = [mr[fid] for fid, _ in sel]; draws.append(d); used |= set(d)
        lx = np.array([x for _, x in sel], dtype=float)
        diffs = np.array([(lx.mean(axis=0) - np.array([ret_x[i] for i in d], dtype=float).mean(axis=0)) / np.array(sd) for d in draws])
        rec.update({"lost": [int(fid) for fid, _ in sel], "draws": [[int(i) for i in d] for d in draws], "n_selected": len(sel),
                    "smd_x1": float(diffs[:, 0].mean()), "smd_x2": float(diffs[:, 1].mean())})
        out[lang] = rec
    return out


def build_manifest(rows, roster, condition, source_sha, cap=64, R=3, min_pairs=8, caliper=0.25, seed=0):
    """The F4 manifest of one precision in the schema of groups_manifest.json (languages in roster order; an excluded language has empty lists and size 0) plus an `f4` block."""
    rec = build_condition(rows, condition, roster, cap, R, min_pairs, caliper, seed)
    groups, controls, sizes = {}, {}, {}
    for l in roster:
        n = rec[l]["n_selected"]
        groups[l] = {"all": rec[l]["lost"]}; controls[l] = {"K64": rec[l]["draws"]}; sizes[l] = {"K16": min(16, n), "K64": n, "all": n}
    keys = ("n_lost_total", "n_inactive_lost", "n_matched", "n_selected", "n_retained_pool", "smd_x1", "smd_x2")
    return {"schema": 1, "kind": "f4-subgroups", "source": {"sha256": source_sha}, "languages": list(roster), "Ks": [64], "Rs": [R], "seed": seed, "groups": groups, "controls": controls, "sizes": sizes,
            "f4": {"condition": condition, "params": {"cap": cap, "R": R, "min_pairs": min_pairs, "caliper": caliper, "seed": seed},
                   "included": [l for l in roster if not rec[l]["excluded"]], "excluded": {l: rec[l]["excluded"] for l in roster if rec[l]["excluded"]},
                   "records": {l: {**{k: rec[l][k] for k in keys}, "n_draws": len(rec[l]["draws"])} for l in roster}}}


def report_text(manifests):
    L = ["# F4 subgroups: lost against retained features at equal native activity (matching report)", "",
         "**Exploratory and descriptive.** Subsets of the selective features that were established in full precision, split by their D-035 status at a precision (lost, retained) and matched on the native "
         "activity at that precision (log rate and log mean activation of the feature on its own language in the quantized model). The functional comparison is run by the unchanged runner and analysis "
         "(the lost subset as the group, the matched retained draws as the controls): E = harm(lost) - harm(retained matched).", "",
         "**Hypothesis written before any number was computed (H-F4, two-sided):** E_b differs from 0 at precision b; \"association predicts function\" predicts E_b < 0 (lost features matter less); a null or a positive "
         "E_b is reported as it is. The same subsets are also run in full precision as a reference (does a difference pre-exist the quantization?).", ""]
    for b, m in manifests.items():
        f = m["f4"]; p = f["params"]; rec = f["records"]
        used = [r for l, r in rec.items() if l in f["included"]]
        smd = [abs(r[k]) for r in used for k in ("smd_x1", "smd_x2")]
        L += [f"## {b}", "",
              f"Caliper {p['caliper']} SD, cap {p['cap']} lost features per language, {p['R']} retained draws, at least {p['min_pairs']} matched pairs, seed {p['seed']}. Languages included: {len(f['included'])} of {len(m['languages'])}; "
              f"lost features used: {sum(r['n_selected'] for r in used)} (of {sum(r['n_lost_total'] for r in rec.values())} established features lost at {b}, {sum(r['n_inactive_lost'] for r in rec.values())} of them inactive at {b} and therefore not matchable). "
              + (f"Balance (standardised mean difference of the native activity, lost minus retained, per language): mean |SMD| {np.mean(smd):.3f}, maximum {np.max(smd):.3f}." if smd else "No language is included."), "",
              "| language | lost at b | inactive at b | matched | selected | draws | SMD log rate | SMD log mean |", "|---|---|---|---|---|---|---|---|"]
        for l in m["languages"]:
            r = rec[l]; fmt = lambda v: "-" if v is None else f"{v:+.3f}"
            L.append(f"| {l} | {r['n_lost_total']} | {r['n_inactive_lost']} | {r['n_matched']} | {r['n_selected']} | {r['n_draws']} | {fmt(r['smd_x1'])} | {fmt(r['smd_x2'])} |")
        if f["excluded"]:
            L += ["", "Excluded: " + "; ".join(f"{l} ({why})" for l, why in f["excluded"].items())]
        L.append("")
    return "\n".join(L) + "\n"


def main(argv=None):
    import argparse
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="F4 subgroup manifests (lost against retained features at equal native activity).")
    ap.add_argument("--status", required=True, help="feature_status.csv of the D-035 analysis (stream eq)")
    ap.add_argument("--groups-manifest", default=str(root / "reports/functional_validation/groups_manifest.json"), help="only the roster order is read")
    ap.add_argument("--out-dir", default=str(root / "reports/functional_validation/f4")); ap.add_argument("--conditions", nargs="*", default=["INT6", "INT5", "INT4"])
    ap.add_argument("--cap", type=int, default=64); ap.add_argument("--R", type=int, default=3); ap.add_argument("--min-pairs", type=int, default=8)
    ap.add_argument("--caliper", type=float, default=0.25); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rows = read_status(a.status); roster = json.loads(Path(a.groups_manifest).read_text())["languages"]
    sha = hashlib.sha256(Path(a.status).read_bytes()).hexdigest()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for b in a.conditions:
        m = build_manifest(rows, roster, b, sha, a.cap, a.R, a.min_pairs, a.caliper, a.seed)
        (out / f"f4_manifest_{b}.json").write_text(json.dumps(m, separators=(",", ":")) + "\n"); manifests[b] = m
    (out / "f4_matching.md").write_text(report_text(manifests))
    print("wrote", out, {b: len(m["f4"]["included"]) for b, m in manifests.items()}, "languages included")


if __name__ == "__main__":
    main()
