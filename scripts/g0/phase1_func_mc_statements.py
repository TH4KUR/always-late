"""The confirmatory statements of the Belebele evaluation (T-MC), evaluated as frozen (plan Task 6c, specified 2026-10-22 before the code). Pure json, no torch.

The freeze file (evaluation_freeze_mc.json) fixes six statements: S0 (the macro effect E over the covered languages at full precision is greater than 0: one-sided, decided by the lower bound of the 95 % interval) and
S1 to S5 (the macro retention of E at INT8, INT7, INT6, INT5, INT4 is classified preserved, attenuated, amplified or inconclusive against the bands, two-sided, 99 % intervals). This tool reads the analysis of
phase1_func_mc_analysis.py, REFUSES it unless its provenance equals the freeze (task, format, K, draws, seed, bands, covered languages and their order, retention level, item counts per language and precision; the note of a
calibration-split run), and writes the outcomes with their numbers. It adds no statement, changes no wording and interprets nothing.

    python3 scripts/g0/phase1_func_mc_statements.py --freeze reports/functional_validation/evaluation_freeze_mc.json --analysis <results/mc_eval/analysis.json> --out <dir>
"""
import argparse
import csv
import json
from pathlib import Path

LADDER_NOTE = ("`undefined`: no covered language has a clearly positive full-precision effect, so the macro retention does not exist (rule D-041); `no data`: the precision is not in the analysis "
               "(its job has not been analysed); `not met` and every class are reported as they are.")


class ProvenanceError(ValueError):
    """The analysis is not the confirmatory analysis of the freeze."""


def check_provenance(freeze, analysis):
    """Every way in which the analysis differs from the freeze, as a list of messages that name the field (empty when it is the confirmatory analysis)."""
    fa = freeze["analysis"]; bad = []
    if freeze.get("part") != "T-MC":
        bad.append(f"part: the freeze is of part {freeze.get('part')!r}, not T-MC")
    if analysis.get("task") != "T-MC":
        bad.append(f"task: the analysis is of {analysis.get('task')!r}, not T-MC")
    for key, fkey in (("format", "format"), ("K", "primary_K"), ("n_boot", "n_boot"), ("seed", "seed")):
        if analysis.get(key) != fa[fkey]:
            bad.append(f"{key}: the analysis has {analysis.get(key)!r}, the freeze {fa[fkey]!r}")
    if [float(x) for x in analysis.get("bands", [])] != [float(x) for x in fa["bands"]]:
        bad.append(f"bands: the analysis has {analysis.get('bands')!r}, the freeze {fa['bands']!r}")
    if list(analysis.get("covered", [])) != list(fa["covered"]):
        bad.append(f"covered: the covered languages (or their order) differ: analysis {analysis.get('covered')!r}, freeze {fa['covered']!r}")
    levels = {s["interval_level"] for s in fa["statements"] if s["precision"] != "FP"}
    if len(levels) != 1 or analysis.get("retention_level") not in levels:
        bad.append(f"retention_level: the analysis has {analysis.get('retention_level')!r}, the retention statements of the freeze use {sorted(levels)!r}")
    if "calibration" in (analysis.get("note") or "").lower():
        bad.append("note: the analysis says it is a calibration-split run (dev-origin passages, a pipeline check), not the held-out evaluation")
    precs = list(analysis.get("macro", {}))
    if "FP" not in precs:
        bad.append("FP: the analysis has no full-precision run (statement S0 and every ratio need it)")
    for p in precs:
        if p not in freeze["precisions"]:
            bad.append(f"{p}: precision is not in the freeze")
    rows = analysis.get("accuracy", [])
    wrong = {}
    for r in rows:
        want = freeze["items"].get(r["language"], {}).get("n_items")
        if want is None or r["n_items"] != want:
            wrong.setdefault(r["precision"], []).append((r["language"], r["n_items"], want))
    for p, items in wrong.items():                                                                          # one line per precision, with the first examples
        ex = "; ".join(f"{lang} has {n!r} against {w!r}" for lang, n, w in items[:3])
        bad.append(f"n_items: {p}: {len(items)} language(s) differ from the freeze ({ex}{'; ...' if len(items) > 3 else ''})")
    have = {(r["precision"], r["language"]) for r in rows}
    for p in precs:
        for lang in fa["covered"]:
            if (p, lang) not in have:
                bad.append(f"items: {p}/{lang} has no items in the analysis (a covered language without a row)")
    for p in precs:
        if p != "FP" and p not in analysis.get("retention", {}):
            bad.append(f"retention: {p} has an effect but no retention block")
    return bad


def evaluate(freeze, analysis):
    """The outcomes of the frozen statements. Raises ProvenanceError, with every problem listed, when the analysis is not the confirmatory one."""
    bad = check_provenance(freeze, analysis)
    if bad:
        raise ProvenanceError("the analysis is not the confirmatory analysis of the freeze:\n- " + "\n- ".join(bad))
    n_cov = len(freeze["analysis"]["covered"]); out = []
    for s in freeze["analysis"]["statements"]:
        p = s["precision"]; r = {"id": s["id"], "precision": p, "claim": s["claim"], "sidedness": s["sidedness"], "interval_level": s["interval_level"], "decision": s["decision"]}
        if p == "FP":
            m = analysis["macro"]["FP"]
            r.update(outcome="met" if m["lo"] > 0 else "not met", n_languages=n_cov, point=m["point"], lo=m["lo"], hi=m["hi"],
                     detail=f"macro E {m['point']:+.4f} nats per item, {100 * s['interval_level']:g} % interval [{m['lo']:+.4f}, {m['hi']:+.4f}], languages with E > 0: {m['n_positive']} of {n_cov}")
        elif p not in analysis["macro"]:
            r.update(outcome="no data", n_languages=None, point=None, lo=None, hi=None, detail="the precision is not in the analysis")
        else:
            m = analysis["retention"][p]["macro"]
            if m["class"] == "undefined":
                r.update(outcome="undefined", n_languages=0, point=None, lo=None, hi=None, detail="no covered language has a clearly positive full-precision effect (rule D-041)")
            else:
                r.update(outcome=m["class"], n_languages=m["n_languages"], point=m["ratio"], lo=m["lo"], hi=m["hi"],
                         detail=f"macro retention {m['ratio']:.2f} over {m['n_languages']} languages with a clearly positive full-precision effect, {100 * s['interval_level']:g} % interval [{m['lo']:.2f}, {m['hi']:.2f}]")
        out.append(r)
    order = list(freeze["precisions"])
    context = [{"precision": p, "point": analysis["macro"][p]["point"], "lo": analysis["macro"][p]["lo"], "hi": analysis["macro"][p]["hi"], "median": analysis["macro"][p].get("median"),
                "n_positive": analysis["macro"][p].get("n_positive"), "n_covered": n_cov} for p in order if p in analysis["macro"]]
    missing = [p for p in order if p != "FP" and p not in analysis["macro"]]
    return {"freeze_commit": freeze.get("commit"), "n_covered": n_cov, "statements": out, "context": context, "complete": not missing, "missing_precisions": missing}


def report_text(res):
    L = ["# T-MC confirmatory statements, evaluated as frozen", "",
         f"Freeze: `evaluation_freeze_mc.json` (code commit {res['freeze_commit']}). **These statements are the ones in the freeze file; their wording, sidedness, levels and decision rules are not changed here. "
         "Everything else (per-language effects, F3, F4, subgroups, doses, the K = all run, other formats) is exploratory and labelled.** The analysis passed the provenance check against the freeze (task, format, K, draws, seed, "
         "bands, covered languages, retention level, item counts per language and precision).", ""]
    if not res["complete"]:
        L += ["**INCOMPLETE: the analysis has no data for " + ", ".join(res["missing_precisions"]) + ".** Statements for those precisions are `no data` and are not evaluated.", ""]
    L += ["## Statements", "", "| id | precision | claim | level | outcome | detail |", "|---|---|---|---|---|---|"]
    for s in res["statements"]:
        L.append(f"| {s['id']} | {s['precision']} | {s['claim']} ({s['sidedness']}) | {100 * s['interval_level']:g} % | {s['outcome']} | {s['detail']} |")
    L += ["", LADDER_NOTE, "", "## Context: macro effect E by precision (mean over the covered languages, nats per item)", "",
          "| precision | macro E | 95 % interval | languages with E > 0 | covered |", "|---|---|---|---|---|"]
    for c in res["context"]:
        L.append(f"| {c['precision']} | {c['point']:+.4f} | [{c['lo']:+.4f}, {c['hi']:+.4f}] | {c['n_positive']} | {c['n_covered']} |")
    return "\n".join(L) + "\n"


_COLUMNS = ["id", "precision", "claim", "sidedness", "interval_level", "outcome", "n_languages", "point", "lo", "hi", "detail"]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate the frozen T-MC statements S0 to S5 from the analysis of phase1_func_mc_analysis.py.")
    ap.add_argument("--freeze", required=True); ap.add_argument("--analysis", required=True, help="analysis.json of the held-out evaluation (all six precisions, --retention-level 0.99)"); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    freeze = json.loads(Path(a.freeze).read_text()); analysis = json.loads(Path(a.analysis).read_text())
    try:
        res = evaluate(freeze, analysis)
    except ProvenanceError as e:
        raise SystemExit(f"refused: {e}")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "statements.json").write_text(json.dumps(res, indent=1) + "\n"); (out / "statements.md").write_text(report_text(res))
    with open(out / "statements.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(_COLUMNS)
        for s in res["statements"]:
            w.writerow([s[k] for k in _COLUMNS])
    print("wrote", out, [(s["id"], s["outcome"]) for s in res["statements"]])


if __name__ == "__main__":
    main()
