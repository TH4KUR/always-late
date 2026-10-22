"""F4 analysis (plan Task 7c, step 2a; design written 2026-10-22 before the code). CPU only, numpy; reuses the effect and bootstrap functions of phase1_func_analysis.py.

The F4 jobs run the unchanged runner with an F4 manifest (phase1_func_f4_groups.py): the group slot holds the features whose language association was LOST at precision b, the control slots hold
activity-matched features that RETAINED it. The loss analysis would describe that wrongly (its report says that the selective group matters more than controls, and it computes retention ratios), so
this module reports the F4 quantities: per language and as a macro mean, E_b = harm(lost) - harm(retained matched) at the native precision b, E_FP = the same contrast for the same subsets in the full-
precision model, and Delta = E_b - E_FP with the same sentence-cluster draws for both (does the lost subset matter less only at b, or did it matter less already?). Signs are decided by the 95 %
percentile interval. No retention ratio, no class, no band."""
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

import phase1_func_analysis as fa

HYPOTHESIS = ("H-F4 (written into the plan before any functional number, two-sided, exploratory because the D-035 statuses were measured on devtest): E_b differs from 0 at precision b; "
              "\"association predicts function\" predicts E_b < 0 (the lost features matter less than activity-matched retained ones); a null result or a positive E_b is reported as it is. E_FP, the same "
              "contrast for the same subsets in the full-precision model, shows whether a difference pre-exists the quantization.")


def _sign(lo, hi):
    return "negative" if hi < 0 else "positive" if lo > 0 else "inconclusive"


def _robust(eff, langs):
    """Step 2c (added after the first F4 result showed one language dominating the macro mean): the median over languages with a 95 % interval (per draw the median of the languages' draw values) and the mean without
    the language with the largest ABSOLUTE point value (named), from the same paired draws. Returns the keys that are added to a macro entry."""
    pts = np.array([eff[c]["point"] for c in langs]); S = np.stack([eff[c]["samples"] for c in langs])
    med = fa._row(np.median(pts), np.median(S, axis=0)); top = int(np.argmax(np.abs(pts))); keep = [k for k in range(len(langs)) if k != top]
    wo = {"language": langs[top], **fa._row(pts[keep].mean(), S[keep].mean(axis=0))} if keep else None
    return {"median_lo": med["lo"], "median_hi": med["hi"], "without_top": wo}


def analyse_f4(runs, out_dir, K="K64", n_boot=2000, seed=0):
    """runs: {"FP": Run, "<precision>": Run}; writes f4_results.json, f4_per_language.csv and report.md into out_dir and returns the result dictionary."""
    if "FP" not in runs:
        raise ValueError("the full-precision run is needed: it shows whether the contrast pre-exists the quantization")
    others = [k for k in runs if k != "FP"]
    if len(others) != 1:
        raise ValueError(f"exactly one precision besides full precision is needed, got {others}")
    b = others[0]; fp, rb = runs["FP"], runs[b]
    if list(fp.languages) != list(rb.languages):
        raise ValueError(f"the languages of the two runs differ: {sorted(set(fp.languages) ^ set(rb.languages))}")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    langs = list(fp.languages)
    D = fa.draws(len(next(iter(fp.ids.values()))), n_boot, seed)                                           # one draw matrix for both precisions: Delta is paired
    e_fp = {c: fa.functional_effect(fp, c, K, D) for c in langs}
    e_b = {c: fa.functional_effect(rb, c, K, D) for c in langs}
    delta = {c: {"point": e_b[c]["point"] - e_fp[c]["point"], "samples": e_b[c]["samples"] - e_fp[c]["samples"]} for c in langs}
    per = {}
    for c in langs:
        rb_row, rf_row, rd = fa._row(e_b[c]["point"], e_b[c]["samples"]), fa._row(e_fp[c]["point"], e_fp[c]["samples"]), fa._row(delta[c]["point"], delta[c]["samples"])
        per[c] = {"E_b": rb_row, "E_fp": rf_row, "delta": rd, "harm_lost_b": e_b[c]["harm_group"], "harm_retained_b": e_b[c]["harm_control"], "harm_lost_fp": e_fp[c]["harm_group"],
                  "harm_retained_fp": e_fp[c]["harm_control"], "n_controls": e_b[c]["n_controls"], "sign_b": _sign(rb_row["lo"], rb_row["hi"]), "sign_fp": _sign(rf_row["lo"], rf_row["hi"])}
    macro = {}
    for name, eff in (("E_b", e_b), ("E_fp", e_fp), ("delta", delta)):
        m = fa.macro(eff, langs)
        macro[name] = {**fa._row(m["point"], m["samples"]), "median": float(np.median([eff[c]["point"] for c in langs])), **_robust(eff, langs)}
    counts = {name: {s: sum(1 for c in langs if per[c][key] == s) for s in ("negative", "positive", "inconclusive")} for name, key in (("E_b", "sign_b"), ("E_fp", "sign_fp"))}
    res = {"condition": b, "K": K, "n_boot": n_boot, "seed": seed, "languages": langs, "per_language": per, "macro": macro, "counts": counts}
    (out / "f4_results.json").write_text(json.dumps(res, indent=1) + "\n")
    cols = ["language", "E_b", "E_b_lo", "E_b_hi", "E_fp", "E_fp_lo", "E_fp_hi", "delta", "delta_lo", "delta_hi", "harm_lost_b", "harm_retained_b", "harm_lost_fp", "harm_retained_fp", "sign_b", "sign_fp"]
    with open(out / "f4_per_language.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for c in langs:
            p = per[c]
            w.writerow([c, p["E_b"]["point"], p["E_b"]["lo"], p["E_b"]["hi"], p["E_fp"]["point"], p["E_fp"]["lo"], p["E_fp"]["hi"], p["delta"]["point"], p["delta"]["lo"], p["delta"]["hi"],
                        p["harm_lost_b"], p["harm_retained_b"], p["harm_lost_fp"], p["harm_retained_fp"], p["sign_b"], p["sign_fp"]])
    (out / "report.md").write_text(report_text(res))
    return res


def report_text(res):
    b, B = res["condition"], res["n_boot"]; n = len(res["languages"])
    L = [f"# F4: lost against retained features at equal native activity, {b}", "",
         "**Exploratory.** Contrast = harm of zero-ablating the features whose language association was LOST at the precision minus the harm of zero-ablating activity-matched features that kept it "
         "(lost minus retained, nats per token on the language's own text, same sentences, same precision). **Negative = the lost features matter less.** "
         f"{B} paired cluster-bootstrap draws over sentence ids (the same draws for both precisions; percentile 95 % intervals, descriptive; the observed value is the estimate). The matching and its balance "
         "are in `f4_matching.md` (the subsets come from the F4 manifest of the precision).", "", HYPOTHESIS, "",
         f"## Macro mean over {n} languages", "", "| quantity | mean | 95 % interval | median | languages negative / positive / inconclusive |", "|---|---|---|---|---|"]
    names = {"E_b": f"E at {b} (native)", "E_fp": "E at full precision (same subsets)", "delta": f"Delta = E at {b} minus E at full precision"}
    for k in ("E_b", "E_fp", "delta"):
        m = res["macro"][k]; c = res["counts"].get(k)
        L.append(f"| {names[k]} | {m['point']:+.4f} | [{m['lo']:+.4f}, {m['hi']:+.4f}] | {m['median']:+.4f} | " + (f"{c['negative']} / {c['positive']} / {c['inconclusive']} |" if c else "- |"))
    L += ["", "**Robust summaries, added after the first F4 result (INT4) because one language dominated the macro mean (post hoc; applied unchanged to every condition).** Median over the languages and the mean without the "
           "language with the largest absolute value, from the same paired draws:", ""]
    for k in ("E_b", "E_fp", "delta"):
        m = res["macro"][k]; w = m.get("without_top")
        L.append(f"- {names[k]}: median {m['median']:+.4f} [{m['median_lo']:+.4f}, {m['median_hi']:+.4f}]" + (f"; mean without {w['language']} {w['point']:+.4f} [{w['lo']:+.4f}, {w['hi']:+.4f}]" if w else ""))
    L += ["", "The intervals describe sentence sampling only: the retained subsets are single or double draws (the number per language is in `f4_matching.md`), so the draw-to-draw variation of a retained subset, which can be "
          "large when it contains one influential feature, is not in them.", "", "## By language", "", f"| language | E at {b} | 95 % interval | E at FP | 95 % interval | Delta | 95 % interval | harm lost at {b} | harm retained at {b} |", "|---|---|---|---|---|---|---|---|---|"]
    for c in res["languages"]:
        p = res["per_language"][c]
        L.append(f"| {c} | {p['E_b']['point']:+.4f} | [{p['E_b']['lo']:+.4f}, {p['E_b']['hi']:+.4f}] | {p['E_fp']['point']:+.4f} | [{p['E_fp']['lo']:+.4f}, {p['E_fp']['hi']:+.4f}] | "
                 f"{p['delta']['point']:+.4f} | [{p['delta']['lo']:+.4f}, {p['delta']['hi']:+.4f}] | {p['harm_lost_b']:.4f} | {p['harm_retained_b']:.4f} |")
    return "\n".join(L) + "\n"


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Analyse an F4 job: func_lm_<split>_FP.pt and func_lm_<split>_<condition>.pt of one F4 manifest.")
    ap.add_argument("--runs", required=True); ap.add_argument("--split", default="eval"); ap.add_argument("--condition", required=True, help="the precision of the F4 manifest, for example INT4")
    ap.add_argument("--out", required=True); ap.add_argument("--K", default="K64"); ap.add_argument("--n-boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    runs = {}
    for lab in ("FP", a.condition):
        p = Path(a.runs) / f"func_lm_{a.split}_{lab}.pt"
        if not p.exists():
            raise SystemExit(f"missing {p}")
        runs[lab] = fa.Run.from_file(p)
    analyse_f4(runs, a.out, a.K, a.n_boot, a.seed)
    print("wrote", a.out, a.condition)


if __name__ == "__main__":
    main()
