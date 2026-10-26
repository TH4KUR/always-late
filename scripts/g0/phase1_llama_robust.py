"""POST HOC robustness analyses of the Llama-3.2-1B consistency check (plan docs/superpowers/plans/2026-10-26-llama-consistency-check.md, section 8; D-087). CPU only.
Written because the control draws of the Llama SAE turned out heavy-tailed (one of three draws can cost the loss 3.7 nats). Every table is labelled post hoc and none of it can change the verdict of
`phase1_llama_compare.py`, which stays the pre-registered function. Per language and state, from the item files of the frozen runner (zero-ablation of the 64-feature group and of the control draws):
  R1  E with the MEDIAN of the control draws (the plan's E uses their mean);
  R2  whether the group harm exceeds EVERY single control draw (strictly), and how many languages do;
  R3  outlier draws: a draw whose harm is above five times the median of the OTHER draws (the median floored at 0.01 nats per token, so that noise around zero is not flagged; the floor is part of the
      operationalisation and was fixed with the tests, before the code saw the Llama files), and E with the outlier draws left out (the mean of the remaining draws).

    ~/.venvs/anlp-torch/bin/python scripts/g0/phase1_llama_robust.py --runs <job dirs> --manifest <groups_manifest.json> --out <dir>"""
import csv
import json
import re
from pathlib import Path

import numpy as np

import phase1_func_analysis as fa
import phase1_gain_analysis as ga
import phase1_llama_compare as lc

OUTLIER_FACTOR = 5.0
OUTLIER_FLOOR = 0.01


def draw_harms(run, code, label="K64"):
    """(pooled harm of the zero-ablation of the group, [pooled harm of every control draw in the order of the draw index]) on the language's own text."""
    name = f"zero_{label}@{code}"
    ctrl = sorted((n for n in run.configs_on(code) if re.fullmatch(rf"ctrl\d+_{re.escape(label)}@{re.escape(code)}", n)), key=lambda n: int(re.match(r"ctrl(\d+)_", n).group(1)))
    if not run.has(code, name) or not ctrl:
        raise KeyError(f"{run.label}: {code} lacks {name} or its control draws")
    dG, nt = run.delta(code, name)
    return fa.pooled(dG, nt), [fa.pooled(run.delta(code, n)[0], nt) for n in ctrl]


def outlier_flags(draws, factor=OUTLIER_FACTOR, floor=OUTLIER_FLOOR):
    """True for a draw whose harm is above `factor` times the median of the other draws (that median floored at `floor`)."""
    d = [float(x) for x in draws]
    return [d[k] > factor * max(float(np.median(d[:k] + d[k + 1:])), floor) for k in range(len(d))]


def language_rows(run, languages, label="K64"):
    rows = []
    for c in languages:
        g, d = draw_harms(run, c, label)
        flags = outlier_flags(d); keep = [x for x, f in zip(d, flags) if not f] or d
        rows.append({"language": c, "group_harm": g, "draws": d, "n_draws": len(d), "E_mean": g - float(np.mean(d)), "E_median": g - float(np.median(d)), "exceeds_all": bool(g > max(d)),
                     "outlier_draws": [k for k, f in enumerate(flags) if f], "E_without_outliers": g - float(np.mean(keep))})
    return rows


def evaluate(runs, tested):
    tested = list(tested)
    if not tested:
        raise ValueError("no tested language")
    out = {}
    for state, run in runs.items():
        rows = language_rows(run, tested)
        s = {"n": len(rows), "E_mean_positive": int(sum(r["E_mean"] > 0 for r in rows)), "E_median_positive": int(sum(r["E_median"] > 0 for r in rows)),
             "exceeds_all_count": int(sum(r["exceeds_all"] for r in rows)), "E_without_outliers_positive": int(sum(r["E_without_outliers"] > 0 for r in rows)),
             "languages_with_an_outlier_draw": int(sum(bool(r["outlier_draws"]) for r in rows)), "E_mean_median": float(np.median([r["E_mean"] for r in rows])),
             "E_median_median": float(np.median([r["E_median"] for r in rows])), "E_without_outliers_median": float(np.median([r["E_without_outliers"] for r in rows]))}
        out[state] = {"rows": rows, "summary": s}
    return out


def retention_variants(runs, tested, n_boot=2000, seed=0, label="K64"):
    """R6 (post hoc): per state other than FP the frozen macro retention (the MEAN of the per-language ratios E(state) / E(FP) over the languages whose FP effect is credibly above zero, as in `phase1_llama_compare`),
    its MEDIAN with the bootstrap interval of the frozen function, the languages that enter, every ratio, and the language with the largest ratio. The mean is dominated by one language when some FP effect is tiny."""
    tested = list(tested)
    if not tested:
        raise ValueError("no tested language")
    if "FP" not in runs:
        raise ValueError("the FP run is needed")
    D = fa.draws(len(runs["FP"].ids[tested[0]]), n_boot, seed)
    eff = {s: {c: fa.functional_effect(r, c, label, D) for c in tested} for s, r in runs.items()}
    out = {}
    for s in runs:
        if s == "FP":
            continue
        m = fa.retention_macro(eff[s], eff["FP"], tested)
        per = {c: eff[s][c]["point"] / eff["FP"][c]["point"] for c in m["languages"]}
        top = max(per, key=lambda c: abs(per[c])) if per else None
        out[s] = {"mean_ratio": m["ratio"], "mean_lo": m["lo"], "mean_hi": m["hi"], "median_ratio": m.get("median_ratio"), "median_lo": m.get("median_lo"), "median_hi": m.get("median_hi"),
                  "n_languages": m["n_languages"], "languages": list(m["languages"]), "per_language": per, "largest_ratio_language": top, "largest_ratio": per[top] if top else None}
    return out


def retention_lines(ret, n):
    out = ["", "## R6: macro retention, the mean and the median of the per-language ratios (post hoc)", "",
           "The pre-registered P2 uses the mean of the ratios E(state) / E(FP) over the languages whose full-precision effect is credibly above zero (the frozen estimator, amendment D-041). On this SAE several full-precision effects are tiny, "
           "so one ratio can dominate the mean; the median of the ratios and the language with the largest ratio are shown next to it. The verdict is not recomputed from this table.", "",
           f"| state | languages entering (of {n}) | mean of the ratios (pre-registered) | median of the ratios [95 % interval] | largest ratio (language) |", "|---|---|---|---|---|"]
    for s, r in ret.items():
        if r["n_languages"] == 0:
            out.append(f"| {s} | 0 | undefined | undefined | - |"); continue
        out.append(f"| {s} | {r['n_languages']} | {r['mean_ratio']:+.3f} | {r['median_ratio']:+.3f} [{r['median_lo']:+.3f}, {r['median_hi']:+.3f}] | {r['largest_ratio']:+.2f} ({r['largest_ratio_language']}) |")
    return out


def report_text(res, meta, retention=None):
    n = next(iter(res.values()))["summary"]["n"]
    out = ["# Llama-3.2-1B consistency check: post hoc robustness of the effect E to the control draws", "",
           f"**Post hoc** (plan section 8, written after the first look at the full-precision state and before this module was run): these tables cannot change the verdict of the pre-registered comparison. {n} tested languages. "
           f"E = zero-ablation harm of the 64-feature group minus a summary of the control draws (nats per token, own text). R1: the median of the control draws instead of their mean; R2: the group harm exceeds every single "
           f"control draw; R3: outlier draws (harm above {OUTLIER_FACTOR:g} times the median of the other draws, that median floored at {OUTLIER_FLOOR:g}) left out.", "",
           "| state | E > 0 (mean of draws, pre-registered) | E > 0 (median of draws, R1) | group harm exceeds every draw (R2) | languages with an outlier draw (R3) | E > 0 without outlier draws (R3) | median E (mean / median / without outliers) |",
           "|---|---|---|---|---|---|---|"]
    for s, d in res.items():
        x = d["summary"]
        out.append(f"| {s} | {x['E_mean_positive']} of {n} | {x['E_median_positive']} of {n} | {x['exceeds_all_count']} of {n} | {x['languages_with_an_outlier_draw']} | {x['E_without_outliers_positive']} of {n} | "
                   f"{x['E_mean_median']:+.4f} / {x['E_median_median']:+.4f} / {x['E_without_outliers_median']:+.4f} |")
    for s, d in res.items():
        out += ["", f"## {s}: per language", "", "| language | group harm | control draws | E (mean) | E (median) | exceeds every draw | outlier draws | E without outliers |", "|---|---|---|---|---|---|---|---|"]
        for r in d["rows"]:
            out.append(f"| {r['language']} | {r['group_harm']:.4f} | {', '.join(f'{x:.3f}' for x in r['draws'])} | {r['E_mean']:+.4f} | {r['E_median']:+.4f} | {'yes' if r['exceeds_all'] else 'no'} | "
                       f"{', '.join(map(str, r['outlier_draws'])) or '-'} | {r['E_without_outliers']:+.4f} |")
    if retention:
        out += retention_lines(retention, n)
    out += ["", "## Limits", "",
            "- Post hoc; three control draws per language in the runs (five were drawn in the manifest): a median of three draws is itself noisy; the outlier rule and its floor are conventions, not a test.",
            "- A language whose negative E disappears with the median or without the outlier draw has a group effect that the mean of three draws could not show; a language that stays negative has no group effect in this SAE at this resolution.", ""]
    return "\n".join(out)


def write_outputs(out_dir, res, tested, untested, meta, retention=None):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    with open(out / "languages.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["state", "language", "group_harm", "draws", "E_mean", "E_median", "exceeds_all", "outlier_draws", "E_without_outliers"])
        for s, d in res.items():
            for r in d["rows"]:
                w.writerow([s, r["language"], r["group_harm"], ";".join(f"{x:.6f}" for x in r["draws"]), r["E_mean"], r["E_median"], r["exceeds_all"], ";".join(map(str, r["outlier_draws"])), r["E_without_outliers"]])
    (out / "robust.json").write_text(json.dumps({"states": list(res), "tested": tested, "untested": untested, "summary": {s: d["summary"] for s, d in res.items()}, "rows": {s: d["rows"] for s, d in res.items()},
                                                 "outlier_factor": OUTLIER_FACTOR, "outlier_floor": OUTLIER_FLOOR, "post_hoc": True, "retention": retention, "meta": meta}, indent=1, default=float) + "\n")
    (out / "report.md").write_text(report_text(res, meta, retention))


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Llama-3.2-1B consistency check: post hoc robustness of E to the control draws.")
    ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True); ap.add_argument("--split", default="eval")
    a = ap.parse_args(argv)
    runs = {}
    for d in map(Path, a.runs):
        for f in sorted(d.glob(f"func_lm_{a.split}_*.pt")):
            r = fa.Run.from_file(f); runs[r.label] = r
    if not runs:
        raise FileNotFoundError(f"no func_lm_{a.split}_<state>.pt in {a.runs}")
    runs = {s: runs[s] for s in ga.LADDER if s in runs}
    man = json.loads(Path(a.manifest).read_text())
    tested = lc.tested_languages(man); untested = [l for l in man["languages"] if l not in tested]
    res = evaluate(runs, tested)
    ret = retention_variants(runs, tested) if "FP" in runs else None
    write_outputs(a.out, res, tested, untested, {"runs": [str(x) for x in a.runs], "manifest_sha256": lc._sha256(a.manifest)}, ret)
    print(f"wrote {a.out}: " + "; ".join(f"{s}: E>0 mean {d['summary']['E_mean_positive']}, median {d['summary']['E_median_positive']}, exceeds all {d['summary']['exceeds_all_count']} of {d['summary']['n']}" for s, d in res.items()))


if __name__ == "__main__":
    main()
