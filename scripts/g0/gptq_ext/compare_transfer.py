"""Stage-2 comparison of the GPTQ extension (freeze endpoint E6, plan Task 10). The unchanged `phase1_transfer_report.py` (claim rules of amendment A14, crossed bootstrap) is run ONCE per arm on that arm's INT6 and INT4
runs (the INT16 floor runs are the legacy rounding, identical in all arms, and are reused); this tool puts the three outputs side by side and writes nothing into them: (1) the claim verdicts (UTILITY, H1, H3, A3, H4, per
state), (2) the feature-level medians with their own crossed-bootstrap intervals, the difference of the points and whether two arms' intervals overlap, (3) the group-level quantities per language (net group effect in FP,
C, W, I, total, native total, native minus fixed), paired by language, arm A minus arm B, with a bootstrap over the languages of the data (27 in the main study, 6 in the external panel; `--label` names the panel in the heading). All contrasts are exploratory (D-101); the verdicts are the pre-set rules' own.

  python3 scripts/g0/gptq_ext/compare_transfer.py --legacy <analysis dir> --matched <analysis dir> --gptq <analysis dir> --roster reports/g0/phase1_languages.json --out <dir>"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gptq_ext import compare_stage1 as cs  # noqa: E402

GROUP_COLUMNS = ("net_FP_FP", "C", "W", "I", "total", "native_total", "native_minus_fixed")
CONTRASTS = (("GPTQ minus matched RTN", "GPTQ", "matched RTN"), ("matched RTN minus legacy RTN", "matched RTN", "legacy RTN"), ("GPTQ minus legacy RTN", "GPTQ", "legacy RTN"))
ARMS = ("legacy RTN", "matched RTN", "GPTQ")
load_roster = cs.load_roster


def load_arm(directory):
    d = Path(directory)
    claims = json.loads((d / "claim_status.json").read_text())["claims"]
    feature = {}
    with open(d / "component_summary.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["kind"] == "real" and r["split"] == "all" and r["state"] != "INT16":
                key = (r["state"], r["metric"])
                if key in feature:
                    raise ValueError(f"{d}: more than one row for {key}")
                feature[key] = {"point": float(r["point"]), "low": float(r["low"]), "high": float(r["high"]), "n": int(r["n"]), "stat": r["stat"]}
    group = {}
    with open(d / "group_per_language.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["state"] != "INT16":
                group[(r["state"], r["language"])] = {c: float(r[c]) for c in GROUP_COLUMNS}
    return {"dir": str(d), "claims": claims, "feature": feature, "group": group}


def claims_side_by_side(arms):
    names = sorted({k for a in arms.values() for k in a["claims"] if k != "POPULATION"})
    rows = []
    for claim in names:
        scopes = {}
        for arm, a in arms.items():
            c = a["claims"].get(claim, {})
            if isinstance(c.get("overall"), str):
                scopes.setdefault("overall", {})[arm] = c["overall"]
            for st, v in (c.get("per_state") or {}).items():
                if isinstance(v, str):
                    scopes.setdefault(st, {})[arm] = v
            if isinstance(c.get("status"), str):
                scopes.setdefault("status", {})[arm] = c["status"]
        for scope in sorted(scopes, key=lambda s: (s != "overall", s)):
            rows.append({"claim": claim, "scope": scope, **{arm: scopes[scope].get(arm, "") for arm in ARMS if arm in arms}})
    return rows


def feature_side_by_side(arms):
    keys = sorted(arms["legacy RTN"]["feature"])
    rows = []
    for key in keys:
        for arm, a in arms.items():
            if key not in a["feature"]:
                raise ValueError(f"{arm} has no feature-level row for {key}")
        r = {"state": key[0], "metric": key[1]}
        for arm in ARMS:
            f = arms[arm]["feature"][key]
            r[arm], r[f"{arm} low"], r[f"{arm} high"] = f["point"], f["low"], f["high"]
        r["n_equal"] = len({arms[arm]["feature"][key]["n"] for arm in ARMS}) == 1
        for title, na, nb in CONTRASTS:
            r[title] = arms[na]["feature"][key]["point"] - arms[nb]["feature"][key]["point"]
            a, b = arms[na]["feature"][key], arms[nb]["feature"][key]
            r[f"intervals_overlap {na} vs {nb}"] = bool(a["low"] <= b["high"] and b["low"] <= a["high"])
        rows.append(r)
    return rows


def group_paired(a, b, roster):
    if set(a["group"]) != set(b["group"]):
        raise ValueError("the two arms do not cover the same states and languages at the group level")
    rows = []
    for (state, lang) in sorted(a["group"], key=lambda k: (k[0], list(roster).index(k[1]))):
        for col in GROUP_COLUMNS:
            va, vb = a["group"][(state, lang)][col], b["group"][(state, lang)][col]
            rows.append({"state": state, "language": lang, "tier": roster[lang]["tier"], "column": col, "a": va, "b": vb, "diff": va - vb})
    return rows


def group_summary(rows, n_boot=2000, seed=0):
    out = []
    for state in sorted({r["state"] for r in rows}):
        for col in GROUP_COLUMNS:
            sub = [r for r in rows if r["state"] == state and r["column"] == col]
            out.append({"state": state, "column": col, **cs.summarize(sub, "diff", n_boot, seed)})
    return out


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def write_report(out, arms, roster, n_boot=2000, seed=0, label=""):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    claims, feat = claims_side_by_side(arms), feature_side_by_side(arms)
    _write_csv(out / "transfer_claims.csv", claims); _write_csv(out / "transfer_feature_level.csv", feat)
    paired_all, summary_all = [], []
    n_lang = len({lang for (_, lang) in next(iter(arms.values()))["group"]})
    md = ["# Stage 2 comparison: the intervention-transfer study (E6), arm against arm" + (f": {label}" if label else ""), "",
          "Exploratory (D-101). The verdicts below are the pre-set rules' own (amendment A14), computed by the unchanged report on each arm; the contrasts are descriptive. Feature level: medians over the same eligible real items with the report's own crossed-bootstrap intervals (not paired contrasts). Group level: per-language quantities, arm A minus arm B, paired by language, 95 % percentile bootstrap of the mean over the " + str(n_lang) + " languages.", "",
          "## Claim verdicts", "", "| claim | scope | " + " | ".join(a for a in ARMS if a in arms) + " |", "|---|---|" + "---|" * len([a for a in ARMS if a in arms])]
    for r in claims:
        md.append(f"| {r['claim']} | {r['scope']} | " + " | ".join(str(r.get(a, "")) for a in ARMS if a in arms) + " |")
    md += ["", "## Feature level: medians (real items, all splits)", "", "| state | metric | legacy RTN | matched RTN | GPTQ | GPTQ minus matched RTN | intervals overlap (GPTQ vs matched) | same item count |", "|---|---|---|---|---|---|---|---|"]
    for r in feat:
        md.append(f"| {r['state']} | {r['metric']} | {r['legacy RTN']:.4f} | {r['matched RTN']:.4f} | {r['GPTQ']:.4f} | {r['GPTQ minus matched RTN']:+.4f} | {'yes' if r['intervals_overlap GPTQ vs matched RTN'] else 'no'} | {'yes' if r['n_equal'] else '**NO**'} |")
    for title, na, nb in CONTRASTS:
        rows = group_paired(arms[na], arms[nb], roster); paired_all += [{"contrast": title, **r} for r in rows]
        summ = group_summary(rows, n_boot, seed); summary_all += [{"contrast": title, **s} for s in summ]
        md += ["", f"## Group level: {title}", "", "| state | quantity | mean | median | 95 % interval of the mean | languages below / above 0 |", "|---|---|---|---|---|---|"]
        for s in summ:
            md.append(f"| {s['state']} | {s['column']} | {s['mean']:+.4f} | {s['median']:+.4f} | [{s['ci_low']:+.4f}, {s['ci_high']:+.4f}] | {s['n_negative']} / {s['n_positive']} |")
    (out / "transfer_comparison.md").write_text("\n".join(md) + "\n")
    _write_csv(out / "transfer_group_paired.csv", paired_all); _write_csv(out / "transfer_group_summary.csv", summary_all)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stage-2 comparison of the GPTQ extension.")
    ap.add_argument("--legacy", required=True); ap.add_argument("--matched", required=True); ap.add_argument("--gptq", required=True); ap.add_argument("--roster", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default="", help="added to the heading, e.g. 'external panel, six languages'")
    a = ap.parse_args(argv)
    write_report(a.out, {"legacy RTN": load_arm(a.legacy), "matched RTN": load_arm(a.matched), "GPTQ": load_arm(a.gptq)}, load_roster(a.roster), a.n_boot, a.seed, a.label)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
