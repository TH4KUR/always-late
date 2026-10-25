"""Task 17 part (b) of the functional-validation plan (design: docs/superpowers/plans/2026-10-21-functional-validation.md, last bullets of Task 17): the prompt cross-check of the Belebele pipeline. CPU only.

Input: the two result directories of `run_promptcheck.sbatch` (`frozen/` and `harness/`, each with `func_mc_cal_letter_FP.pt` and `meta_mc_cal.json`) and the stored calibration record (`belebele_calibration.json`).
Per language the accuracy of the frozen letter prompt and of the lm-evaluation-harness prompt on the same items (arg max of the stored `none` scores against the answers), the paired difference (harness minus frozen)
with a 95 % interval from a bootstrap over items (2,000 draws, seed 0), the exact McNemar p on the discordant pairs and the agreement of the two predictions. Over the languages: the mean difference with a bootstrap
over languages (5,000 draws, seed 0). Rule (fixed before the run, margin 0.05): 'equivalent' when the interval lies within [-0.05, +0.05], 'harness prompt better' when its lower end is above +0.05, 'ours better'
when its upper end is below -0.05, otherwise 'inconclusive'; for the primary language (English) and for the mean over the languages. Before anything is read the frozen variant must reproduce the stored
calibration accuracies: exactly (same GPU model), at most one item off in a language is flagged for a hardware explanation (D-079), more is an anomaly and stops the analysis.

    ~/.venvs/anlp-torch/bin/python scripts/g0/phase1_func_promptcheck_analysis.py --frozen-dir <dir> --harness-dir <dir> --calibration-json <belebele_calibration.json> --out <dir>"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats

import phase1_func_mc_analysis as ma

MARGIN = 0.05


def mcnemar_exact(b, c):
    """Exact two-sided McNemar p: b = frozen right and harness wrong, c = frozen wrong and harness right (binomial test of the smaller count among the b + c discordant pairs)."""
    n = b + c
    return 1.0 if n == 0 else float(stats.binomtest(min(b, c), n, 0.5).pvalue)


def paired_bootstrap(cs, ch, n_boot=2000, seed=0):
    """Accuracy difference (harness minus frozen) with a 95 % percentile interval from a seeded bootstrap over items; cs, ch: boolean correctness of the same items."""
    cs, ch = np.asarray(cs, dtype=float), np.asarray(ch, dtype=float)
    if cs.shape != ch.shape:
        raise ValueError(f"the correctness arrays differ in length: {cs.shape} and {ch.shape}")
    idx = np.random.default_rng(seed).integers(0, len(cs), size=(n_boot, len(cs)))
    d = ch[idx].mean(1) - cs[idx].mean(1)
    return {"diff": float(ch.mean() - cs.mean()), "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5))}


def label(lo, hi, margin=MARGIN):
    if lo >= -margin and hi <= margin:
        return "equivalent"
    if lo > margin:
        return "harness prompt better"
    if hi < -margin:
        return "ours better"
    return "inconclusive"


def reproduction(run_s, stored, languages):
    """Frozen variant against the stored calibration record: per language the accuracy, the stored one, the number of items off (accuracy difference times the item count) and the item counts;
    status 'exact', 'within one item' (flag for hardware) or 'anomaly' (another item count, or more than one item off)."""
    rows, status = [], "exact"
    for c in languages:
        st = stored["languages"][c]["letter"]
        acc = run_s.accuracy(c, "none"); n = len(run_s.keys[c])
        off = int(round(abs(acc - st["acc"]) * st["n"]))
        rows.append({"language": c, "acc": acc, "acc_stored": st["acc"], "n": n, "n_stored": st["n"], "items_off": off})
        if n != st["n"] or off > 1:
            status = "anomaly"
        elif off == 1 and status == "exact":
            status = "within one item"
    return {"status": status, "rows": rows}


def evaluate(run_s, run_h, stored, languages, primary="eng_Latn", margin=MARGIN, n_boot=2000, n_boot_languages=5000, seed=0):
    langs = list(languages)
    if primary not in langs:
        raise ValueError(f"the primary language {primary} is not among the languages {langs}")
    for c in langs:
        if list(map(tuple, run_s.keys[c])) != list(map(tuple, run_h.keys[c])) or list(run_s.answers[c]) != list(run_h.answers[c]):
            raise ValueError(f"{c}: the two variants were not run on the same items")
    rep = reproduction(run_s, stored, langs)
    if rep["status"] == "anomaly":
        bad = [r["language"] for r in rep["rows"] if r["n"] != r["n_stored"] or r["items_off"] > 1]
        raise ValueError(f"the frozen variant does not reproduce the stored calibration accuracies in {bad}: explain this before any reading")
    rows = []
    for c in langs:
        ps, ph = run_s.scores[(c, "none")].argmax(1), run_h.scores[(c, "none")].argmax(1); ans = np.asarray(run_s.answers[c])
        cs, ch = ps == ans, ph == ans
        b, cc = int((cs & ~ch).sum()), int((~cs & ch).sum())
        bs = paired_bootstrap(cs, ch, n_boot, seed)
        rows.append({"language": c, "n": len(ans), "acc_frozen": float(cs.mean()), "acc_harness": float(ch.mean()), **bs, "b": b, "c": cc, "p_mcnemar": mcnemar_exact(b, cc), "agreement": float((ps == ph).mean()),
                     "label": label(bs["lo"], bs["hi"], margin)})
    d = np.array([r["diff"] for r in rows]); idx = np.random.default_rng(seed).integers(0, len(d), size=(n_boot_languages, len(d))); m = d[idx].mean(1)
    macro = {"diff": float(d.mean()), "lo": float(np.percentile(m, 2.5)), "hi": float(np.percentile(m, 97.5)), "n_languages": len(d)}
    macro["label"] = label(macro["lo"], macro["hi"], margin)
    return {"rows": rows, "primary": next(r for r in rows if r["language"] == primary), "macro": macro, "reproduction": rep, "margin": margin, "n_boot": n_boot, "n_boot_languages": n_boot_languages, "seed": seed}


def report_text(res, meta):
    p, m, rep = res["primary"], res["macro"], res["reproduction"]
    n = len(res["rows"])
    out = ["# Task 17 part (b): prompt cross-check of the Belebele pipeline", "",
           "Exploratory; changes no frozen statement. Full precision, no edit, calibration items of each language (250 per language), the letter format scored on the next token among ' A' to ' D'. "
           "**Frozen** = the prompt of the frozen task module ('Passage:', 'Question:', 'A.' to 'D.'); **harness** = the lm-evaluation-harness Belebele prompt ('P:', 'Q:', 'A:' to 'D:', source S08 of `reports/research_sources.md`); "
           f"the same items, the same model and scoring. Rule (fixed before the run): margin {res['margin']:g}; 'equivalent' when the 95 % interval of the accuracy difference (harness minus frozen) lies within the margin, "
           "'harness prompt better' when its lower end is above the margin, 'ours better' when its upper end is below minus the margin, otherwise 'inconclusive'.", "",
           "## Verdict", "", "| quantity | harness minus frozen accuracy | 95 % interval | label |", "|---|---|---|---|",
           f"| primary: {p['language']} ({p['n']} items) | {p['diff']:+.4f} ({p['acc_harness']:.3f} against {p['acc_frozen']:.3f}) | [{p['lo']:+.4f}, {p['hi']:+.4f}] (bootstrap over items, {res['n_boot']} draws) | **{p['label']}** |",
           f"| mean over {m['n_languages']} languages | {m['diff']:+.4f} | [{m['lo']:+.4f}, {m['hi']:+.4f}] (bootstrap over languages, {res['n_boot_languages']} draws) | **{m['label']}** |", "",
           f"## Reproduction of the stored calibration (checked before the reading): **{rep['status']}**", "",
           "The frozen variant is run again by the same pipeline and must give the accuracies of the stored calibration record (`belebele_calibration.json`, letter format, full precision): exact on the same GPU model; "
           "one item off in a language is flagged for a hardware explanation (D-079); more is an anomaly and stops the analysis.", "",
           f"Items off per language: {', '.join(f'{r['language']} {r['items_off']}' for r in rep['rows'])}.", "",
           "## Per language", "", "| language | items | accuracy frozen | accuracy harness | difference | 95 % interval | frozen right, harness wrong | frozen wrong, harness right | McNemar p | agreement | label |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        out.append(f"| {r['language']} | {r['n']} | {r['acc_frozen']:.3f} | {r['acc_harness']:.3f} | {r['diff']:+.3f} | [{r['lo']:+.3f}, {r['hi']:+.3f}] | {r['b']} | {r['c']} | {r['p_mcnemar']:.3g} | {r['agreement']:.3f} | {r['label']} |")
    out += ["", "## Reading", "",
            "- equivalent: our prompt is not an unusual choice, and the absolute accuracies of the frozen Belebele statement do not depend on it at this margin.",
            "- harness prompt better: the frozen format is prompt-sensitive. The frozen choice stays (a change would be a D-entry for the user), the null of the group edits (D-050, D-062) is stated for the frozen prompt only, and a rerun of the K = all edit with the harness prompt becomes a question for the user (default: do not run it).",
            "- ours better: reported, nothing else. inconclusive: reported with the intervals.", "",
            "## Limits", "",
            f"- Exploratory; one model (Gemma-2-2B, full precision), {n} languages, 250 calibration items each; the interval of a single language is wide (the difference of two accuracies of 250 items).",
            "- The harness prompt is the lm-evaluation-harness one (mutable upstream source, fetched 2026-10-25), not the prompt of the original Belebele paper; the loader's whitespace normalisation is applied to both variants.",
            "- A label 'equivalent' says that the accuracy level does not move by more than the margin; it does not say that the two prompts give the same answers item by item (see the agreement column).", ""]
    return "\n".join(out)


def write_outputs(out_dir, res, meta):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    with open(out / "languages.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res["rows"][0])); w.writeheader(); w.writerows(res["rows"])
    (out / "analysis.json").write_text(json.dumps({"primary": res["primary"], "macro": res["macro"], "reproduction": res["reproduction"], "margin": res["margin"], "n_boot": res["n_boot"], "n_boot_languages": res["n_boot_languages"],
                                                   "seed": res["seed"], "languages": [r["language"] for r in res["rows"]], "meta": meta}, indent=1) + "\n")
    (out / "report.md").write_text(report_text(res, meta))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_variant(d):
    """(McRun, info) of one result directory; refuses a job that is not complete, skipped a state for time or failed a gate."""
    d = Path(d)
    meta = json.loads((d / "meta_mc_cal.json").read_text())
    if not meta.get("complete") or meta.get("skipped_for_time"):
        raise ValueError(f"{d}: the job is not complete or skipped a state for time ({meta.get('skipped_for_time')})")
    failed = [k for k, v in (meta.get("gates") or {}).items() if not v]
    if failed:
        raise ValueError(f"{d}: gates failed: {failed}")
    f = d / "func_mc_cal_letter_FP.pt"
    return ma.McRun.from_file(f), {"file": f.name, "dir": str(d), "sha256": _sha256(f), "meta_sha256": _sha256(d / "meta_mc_cal.json")}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Task 17 (b): frozen against harness Belebele prompt (rule: margin 0.05).")
    ap.add_argument("--frozen-dir", required=True); ap.add_argument("--harness-dir", required=True); ap.add_argument("--calibration-json", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--primary", default="eng_Latn"); ap.add_argument("--margin", type=float, default=MARGIN); ap.add_argument("--n-boot", type=int, default=2000); ap.add_argument("--n-boot-languages", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    run_s, info_s = load_variant(a.frozen_dir); run_h, info_h = load_variant(a.harness_dir)
    if list(run_s.languages) != list(run_h.languages):
        raise ValueError("the two variants cover different languages")
    stored = json.loads(Path(a.calibration_json).read_text())
    res = evaluate(run_s, run_h, stored, run_s.languages, a.primary, a.margin, a.n_boot, a.n_boot_languages, a.seed)
    write_outputs(a.out, res, {"frozen": info_s, "harness": info_h, "calibration_sha256": _sha256(a.calibration_json)})
    print(f"wrote {a.out}: reproduction {res['reproduction']['status']}; {a.primary}: {res['primary']['diff']:+.4f} [{res['primary']['lo']:+.4f}, {res['primary']['hi']:+.4f}] {res['primary']['label']}; "
          f"mean over {res['macro']['n_languages']} languages {res['macro']['diff']:+.4f} [{res['macro']['lo']:+.4f}, {res['macro']['hi']:+.4f}] {res['macro']['label']}")


if __name__ == "__main__":
    main()
