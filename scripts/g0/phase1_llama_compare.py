"""Llama-3.2-1B consistency check against the Gemma findings (plan docs/superpowers/plans/2026-10-26-llama-consistency-check.md, sections 3, 6 and 7; D-087). CPU only.

Input: the job directories of the Llama T-LM runs (Study G main preset, states FP, INT8, INT6, INT5, INT4), the groups manifest, the committed Study G decision of the ratios and the committed Gemma outputs.
Only the frozen code computes numbers: E = `fa.functional_effect` (zero-ablation of the group minus the mean of the zeroed controls), macro retention = `fa.retention_macro`, the dose rule = `ga.h_g1`.
Tested languages: those whose group has at least 16 members; counts are proportional (`ga.required`). Predictions (fixed in the plan before any Llama number):
  P1  median E over the tested languages > 0 with its 95 % bootstrap interval (over languages) above 0, and E > 0 in at least required(24) languages;
  P2a macro retention within [0.85, 1.15] at INT8 and INT6;
  P2b macro retention at INT4 at least 0.30 above INT6 and the largest of the ladder;
  P3a h_g1 (graded in the down ratio, in at least required(24) languages and in the median) at FP and at INT4;
  P3b the net harm at the deepest down ratio is larger at INT4 than at FP in at least required(20) languages and the median INT4/FP ratio (languages with a positive FP value) is above 1.3 (reported, never part of the verdict).
Verdict: consistent = P1 and P2b and P3a; partly consistent = P1 and P3a and not P2b; limited = P1 and not P3a; not consistent = not P1.

    ~/.venvs/anlp-torch/bin/python scripts/g0/phase1_llama_compare.py --runs <job dirs> --manifest <groups_manifest.json> --ratios-json <ratios.json> --gemma-root reports/functional_validation --out <dir>"""
import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import phase1_func_analysis as fa
import phase1_gain_analysis as ga

MIN_GROUP = 16
P2A_BAND = (0.85, 1.15)
P2B_RISE = 0.30
P3B_RATIO = 1.3


def tested_languages(manifest, min_group=MIN_GROUP):
    """The languages of the manifest whose group (`groups[l].all`) has at least `min_group` members, in manifest order."""
    return [l for l in manifest["languages"] if len(manifest["groups"][l]["all"]) >= min_group]


def median_ci(values, n_boot=5000, seed=0, level=0.95):
    """Percentile interval of the median over the values (languages), seeded bootstrap."""
    v = np.asarray(values, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(v), size=(n_boot, len(v)))
    m = np.median(v[idx], axis=1); a = (1 - level) / 2 * 100
    return float(np.percentile(m, a)), float(np.percentile(m, 100 - a))


def evaluate(gains, tested, n_boot=2000, n_boot_languages=5000, seed=0):
    tested = list(tested)
    if not tested:
        raise ValueError("no tested language")
    missing = [l for l in tested if l not in gains.languages]
    if missing:
        raise ValueError(f"languages {missing} are not in the runs")
    for s in ("FP", "INT4"):
        if s not in gains.states:
            raise ValueError(f"the {s} run is needed")
    n = len(tested); label = gains.label
    D = fa.draws(gains.n_sentences, n_boot, seed)
    eff = {s: {c: fa.functional_effect(gains.runs[s], c, label, D) for c in tested} for s in gains.states}
    e = [eff["FP"][c]["point"] for c in tested]
    lo, hi = median_ci(e, n_boot_languages, seed); req24 = ga.required(n, 24)
    p1 = {"count_positive": int(sum(x > 0 for x in e)), "required": req24, "median": float(np.median(e)), "lo": lo, "hi": hi}
    p1["met"] = bool(p1["count_positive"] >= req24 and lo > 0)
    ret = {s: fa.retention_macro(eff[s], eff["FP"], tested)["ratio"] for s in gains.states if s != "FP"}
    ok = lambda v: v is not None
    p2a = {"band": list(P2A_BAND), "values": {s: ret.get(s) for s in ("INT8", "INT6")}}
    p2a["met"] = bool(all(ok(v) and P2A_BAND[0] <= v <= P2A_BAND[1] for v in p2a["values"].values()))
    r4, r6 = ret.get("INT4"), ret.get("INT6"); known = [v for v in ret.values() if ok(v)]
    p2b = {"int4": r4, "int6": r6, "rise_required": P2B_RISE, "met": bool(ok(r4) and ok(r6) and r4 >= r6 + P2B_RISE and r4 >= max(known))}
    table = ga.build_table(gains, n_boot=n_boot, seed=seed)
    g2 = copy.copy(gains); g2.languages = tested
    h = {s: ga.h_g1(table, g2, s) for s in ("FP", "INT4")}
    p3a = {s: {"agree": h[s]["agree"], "n": h[s]["n"], "median_net": h[s]["median_net"], "median_ordered": h[s]["median_ordered"], "met": h[s]["met"]} for s in h}
    p3a["met"] = bool(h["FP"]["met"] and h["INT4"]["met"])
    a = gains.down[-1]
    net = {s: {c: table.get(s, c, a)["net"] for c in tested} for s in ("FP", "INT4")}
    larger = int(sum(net["INT4"][c] > net["FP"][c] for c in tested))
    rat = [net["INT4"][c] / net["FP"][c] for c in tested if net["FP"][c] > 0]
    req20 = ga.required(n, 20); med = float(np.median(rat)) if rat else None
    p3b = {"ratio": a, "count_larger": larger, "required": req20, "median_ratio": med, "median_net": {s: float(np.median(list(net[s].values()))) for s in net}, "met": bool(larger >= req20 and med is not None and med > P3B_RATIO)}
    med = lambda xs: float(np.median(xs)) if xs else None
    diag = {s: {"harm_group_median": med([eff[s][c]["harm_group"] for c in tested]), "harm_control_median": med([eff[s][c]["harm_control"] for c in tested]),
                "norm_ratio_group_median": med([eff[s][c]["norm_ratio_group"] for c in tested if eff[s][c]["norm_ratio_group"] is not None]),
                "norm_ratio_control_median": med([eff[s][c]["norm_ratio_control"] for c in tested if eff[s][c]["norm_ratio_control"] is not None])} for s in ("FP", "INT4")}
    res = {"P1": p1, "diagnostics": diag, "P2": {"retention": ret}, "P2a": p2a, "P2b": p2b, "P3a": p3a, "P3b": p3b, "n_tested": n, "tested": tested, "untested": [c for c in gains.languages if c not in tested], "states": list(gains.states),
           "n_sentences": gains.n_sentences, "n_boot": n_boot, "n_boot_languages": n_boot_languages, "seed": seed}
    res["verdict"] = verdict(res)
    return res


def verdict(p):
    if not p["P1"]["met"]:
        return "not consistent"
    if not p["P3a"]["met"]:
        return "limited"
    return "consistent" if p["P2b"]["met"] else "partly consistent"


def gemma_reference(root):
    """The Gemma numbers of the committed outputs: median E at FP (`results/eval/effects_by_language.csv`), macro retention (`results/eval/retention.csv`), median net harm at the deepest down ratio at FP and
    INT4 and the median INT4/FP ratio (`gain_study/results/curves.csv`)."""
    root = Path(root)
    with open(root / "results/eval/effects_by_language.csv") as fh:
        e_fp = [float(r["effect"]) for r in csv.DictReader(fh) if r["precision"] == "FP"]
    with open(root / "results/eval/retention.csv") as fh:
        ret = {r["precision"]: float(r["ratio"]) for r in csv.DictReader(fh) if r["language"] == "MACRO_MEAN_OF_RATIOS" and r["precision"] in ("INT8", "INT6", "INT5", "INT4")}
    with open(root / "gain_study/results/curves.csv") as fh:
        rows = [r for r in csv.DictReader(fh)]
    a = min(float(r["ratio"]) for r in rows if float(r["ratio"]) > 0)
    net = {s: {r["language"]: float(r["net"]) for r in rows if r["state"] == s and abs(float(r["ratio"]) - a) < 1e-12} for s in ("FP", "INT4")}
    rat = [net["INT4"][c] / net["FP"][c] for c in net["FP"] if net["FP"][c] > 0 and c in net["INT4"]]
    return {"median_E_FP": float(np.median(e_fp)), "retention": ret, "deepest_net_median": {s: float(np.median(list(v.values()))) for s, v in net.items()}, "ratio_int4_over_fp_median": float(np.median(rat))}


def _f(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def report_text(res, ref, meta):
    p1, p2a, p2b, p3a, p3b = res["P1"], res["P2a"], res["P2b"], res["P3a"], res["P3b"]
    ret = res["P2"]["retention"]; n = res["n_tested"]
    yes = lambda b: "met" if b else "not met"
    out = ["# Llama-3.2-1B against the Gemma findings: a minimum consistency check", "",
           f"Exploratory; the predictions and the verdict function were fixed in the plan (`docs/superpowers/plans/2026-10-26-llama-consistency-check.md`, sections 3, 6 and 7) before any Llama number existed. {n} languages tested "
           f"(group of at least 16 members; untested: {', '.join(res['untested']) or 'none'}), {res['n_sentences']} held-out sentences per language, three matched-control draws; SAE `{meta.get('sae', 'fineweb42')}`. "
           "All numbers come from the frozen Gemma code (functional effect, macro retention, the Study G dose rule) applied to the Llama item files.", "",
           f"## Verdict: **{res['verdict']}**", "",
           "Setting differences that travel with every result: depth 12 of 16 (Gemma 12 of 26), width 2048, a TopK SAE (k = 48, 14,336 features) against a JumpReLU one, another SAE family and training data, another tokenizer, "
           "another model size. Not tested here: the INT4 repair at ratios 1.1 and 1.2, the Belebele task, single-feature fingerprints (F5), state-defined groups, the optimal-gain analysis.", "",
           "| prediction | rule (fixed beforehand) | Llama | Gemma | result |", "|---|---|---|---|---|",
           f"| P1 existence at FP | median E above 0 with its 95 % interval above 0 and E > 0 in at least {p1['required']} of {n} | median E {_f(p1['median'])} [{_f(p1['lo'])}, {_f(p1['hi'])}]; {p1['count_positive']} of {n} positive | median E {_f(ref['median_E_FP'])}; 27 of 27 | **{yes(p1['met'])}** |",
           f"| P2a flat through INT6 | macro retention within [0.85, 1.15] at INT8 and INT6 | INT8 {_f(ret.get('INT8'))}, INT6 {_f(ret.get('INT6'))} | INT8 {_f(ref['retention'].get('INT8'))}, INT6 {_f(ref['retention'].get('INT6'))} | {yes(p2a['met'])} (reported) |",
           f"| P2b the deep end changes | macro retention at INT4 at least 0.30 above INT6 and the largest | INT5 {_f(ret.get('INT5'))}, INT4 {_f(ret.get('INT4'))} | INT5 {_f(ref['retention'].get('INT5'))}, INT4 {_f(ref['retention'].get('INT4'))} | **{yes(p2b['met'])}** |",
           f"| P3a graded in the down ratio | the frozen H-G1 (ordered in at least {ga.required(n, 24)} of {n} and in the median) at FP and at INT4 | FP {p3a['FP']['agree']} of {p3a['FP']['n']}, INT4 {p3a['INT4']['agree']} of {p3a['INT4']['n']} | Study G: H-G1 met at every state | **{yes(p3a['met'])}** |",
           f"| P3b the deepest turn-down costs more at INT4 | larger at INT4 in at least {p3b['required']} of {n} and a median INT4/FP ratio above 1.3 | {p3b['count_larger']} of {n}; median ratio {_f(p3b['median_ratio'])}; median net harm FP {_f(p3b['median_net']['FP'])}, INT4 {_f(p3b['median_net']['INT4'])} | median net FP {_f(ref['deepest_net_median']['FP'])}, INT4 {_f(ref['deepest_net_median']['INT4'])}; ratio {_f(ref['ratio_int4_over_fp_median'])} | {yes(p3b['met'])} (reported) |", "",
           "## Diagnostics (not predictions)", "",
           "Medians over the tested languages of the group harm, the matched-control harm (mean of the three draws) and the perturbation norm ratio of the zeroed set (group and controls; the matching should make the two norm ratios similar). "
           "A control harm above the group harm together with similar norm ratios is a result about the model, not about the matching; clearly different norm ratios would point at the matching.", "",
           "| state | group harm | control harm | perturbation norm, group | perturbation norm, controls |", "|---|---|---|---|---|"] + [
           f"| {s} | {_f(d['harm_group_median'], 4)} | {_f(d['harm_control_median'], 4)} | {_f(d['norm_ratio_group_median'], 4)} | {_f(d['norm_ratio_control_median'], 4)} |" for s, d in res["diagnostics"].items()] + ["",
           "## Reading", "",
           "- consistent = P1 and P2b and P3a: the qualitative picture of Gemma (a language-specific, graded dependence, flat through the middle of the ladder, larger at INT4) is found on a second model with a second SAE family. It is not a replication of the numbers and does not turn an exploratory Gemma claim into a confirmatory one.",
           "- partly consistent = P1 and P3a but not P2b: the dependence exists and is graded, but the quantization pattern differs; report the difference, do not explain it away. limited = P1 but not P3a: a language-specific dependence that is not graded. not consistent = P1 fails.",
           "- P2a and P3b are reported and never change the verdict.", "",
           "## Limits", "",
           "- Exploratory (devtest sentences, selection on dev); one small model, one layer, one SAE (a sensitivity SAE is optional); the Gemma numbers are those of the committed outputs; groups with fewer than 64 members are the whole selective set; languages with fewer than 16 selective features are not tested.", ""]
    return "\n".join(out)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Llama-3.2-1B consistency check against the Gemma findings (P1 to P3b and the verdict).")
    ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--manifest", required=True); ap.add_argument("--ratios-json", required=True); ap.add_argument("--gemma-root", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000); ap.add_argument("--n-boot-languages", type=int, default=5000); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--sae", default="fineweb42")
    a = ap.parse_args(argv)
    gains, info = ga.load_runs(a.runs, a.ratios_json)
    man = json.loads(Path(a.manifest).read_text())
    if list(man["languages"]) != list(gains.languages):
        raise ValueError("the manifest and the runs cover different languages")
    res = evaluate(gains, tested_languages(man), a.n_boot, a.n_boot_languages, a.seed)
    ref = gemma_reference(a.gemma_root)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps({**res, "gemma_reference": ref, "meta": {"runs": info, "manifest_sha256": _sha256(a.manifest), "sae": a.sae}}, indent=1, default=float) + "\n")
    (out / "report.md").write_text(report_text(res, ref, {"sae": a.sae}))
    print(f"wrote {a.out}: verdict {res['verdict']} (P1 {res['P1']['met']}, P2a {res['P2a']['met']}, P2b {res['P2b']['met']}, P3a {res['P3a']['met']}, P3b {res['P3b']['met']})")


if __name__ == "__main__":
    main()
