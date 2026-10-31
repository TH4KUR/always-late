"""Analysis of the frozen predictions of the Gemma-2-9B three-arm confirmation (spec docs/superpowers/specs/2026-10-28-gemma9b-three-arm-confirmation-design.md section 5, amendment A1; plan Tasks 6 and 7). CPU only.

Tier 1 (Study G across the legacy-RTN, matched-RTN and GPTQ arms): every number comes from the frozen code, `phase1_llama_compare.evaluate` (P1 to P3b of the legacy freeze: E, macro retention, the dose rule H-G1) and
`phase1_gain_analysis` (net harm per language, H-G5). New predictions, thresholds from the 2B arms (D-107, D-109), tested languages = group of at least 16 members, req(x) = ga.required(n, x):
  Q0  in the matched-RTN arm and in the GPTQ arm, P1 holds at FP and the dose rule (H-G1) holds at INT4;
  Q1  r(arm) = median over the tested languages with a positive FP value of the net harm at the deepest down ratio, INT4 over FP. Evaluable only if the legacy arm meets P3b (its median ratio above 1.3 in at least
      req(20) languages). Holds if r(GPTQ) <= r(matched RTN), r(GPTQ) <= r(legacy) - 0.3 and the per-language ratio is below the legacy one in at least req(20) languages;
  Q2  the macro retention at INT6 lies in [0.85, 1.15] in all three arms;
  Q3  (reported, never part of the verdict) H-G5b per arm and the number of languages whose best factor at INT4 is above 1.
Verdict: consistent = Q0 and Q1 and Q2; partly consistent = Q0 and Q2 and not Q1 (failed or not evaluable); different = not Q0 or not Q2.
Arms need the states FP, INT6 and INT4; a missing arm, state or tested language raises a ValueError that names it."""
import numpy as np

import phase1_gain_analysis as ga
import phase1_llama_compare as lc

ARMS = ("legacy", "rtn-matched", "gptq")
NEW_ARMS = ("rtn-matched", "gptq")
REQUIRED_STATES = ("FP", "INT6", "INT4")
Q1_DROP = 0.3
Q2_BAND = lc.P2A_BAND


def _check_arms(arms, tested):
    missing = [a for a in ARMS if a not in arms]
    if missing:
        raise ValueError(f"missing arm(s) {missing}: Study G needs {list(ARMS)}")
    for name in ARMS:
        g = arms[name]
        absent = [s for s in REQUIRED_STATES if s not in g.states]
        if absent:
            raise ValueError(f"arm {name}: the state(s) {absent} are missing (needed: {list(REQUIRED_STATES)})")
        gone = [c for c in tested if c not in g.languages]
        if gone:
            raise ValueError(f"arm {name}: the tested language(s) {gone} are not in its runs")
    ref = arms[ARMS[0]].runs["FP"].ids
    for name in ARMS[1:]:
        if any(list(arms[name].runs["FP"].ids[c]) != list(ref[c]) for c in tested):
            raise ValueError(f"arm {name}: the sentence ids of the full-precision run differ from those of arm {ARMS[0]}")


def _restrict_p2a(res, gains):
    """An arm without INT8 (matched RTN and GPTQ run INT6 and INT4 only): P2a is judged at INT6 alone and says so."""
    if "INT8" in gains.states:
        return res["P2a"]
    v = res["P2"]["retention"].get("INT6")
    return {"band": list(lc.P2A_BAND), "values": {"INT6": v}, "restricted_to": ["INT6"], "met": bool(v is not None and lc.P2A_BAND[0] <= v <= lc.P2A_BAND[1])}


def _ratios(table, gains, tested):
    """{language: net harm at the deepest down ratio, INT4 over FP} for the tested languages with a positive FP value."""
    a = gains.down[-1]
    out = {}
    for c in tested:
        fp = table.get("FP", c, a)["net"]
        if fp > 0:
            out[c] = table.get("INT4", c, a)["net"] / fp
    return out


def q1(per_arm, ratios, tested):
    """per_arm: {arm: lc.evaluate result}; ratios: {arm: {language: ratio}}. -> {"evaluable", "legacy_p3b_met", "r", "required_drop", "count_lower", "required", "met"}."""
    n = len(tested)
    required = ga.required(n, 20)
    r = {a: per_arm[a]["P3b"]["median_ratio"] for a in ARMS}
    evaluable = bool(per_arm["legacy"]["P3b"]["met"])
    lower = sum(1 for c, v in ratios["gptq"].items() if c in ratios["legacy"] and v < ratios["legacy"][c])
    ok = evaluable and all(v is not None for v in r.values()) and r["gptq"] <= r["rtn-matched"] and r["gptq"] <= r["legacy"] - Q1_DROP and lower >= required
    return {"evaluable": evaluable, "legacy_p3b_met": bool(per_arm["legacy"]["P3b"]["met"]), "r": r, "required_drop": Q1_DROP, "count_lower": int(lower), "required": required, "met": bool(ok)}


def verdict_tier1(q):
    if not (q["Q0"]["met"] and q["Q2"]["met"]):
        return "different"
    return "consistent" if q["Q1"]["met"] else "partly consistent"


def evaluate_arms(arms, tested, n_boot=2000, n_boot_languages=5000, seed=0):
    """arms: {"legacy", "rtn-matched", "gptq": ga.GainRuns}; tested: the tested languages. -> {"arms", "Q0", "Q1", "Q2", "Q3", "verdict_tier1", "tested", "n_tested", "n_boot", "n_boot_languages", "seed"}."""
    tested = list(tested)
    _check_arms(arms, tested)
    per_arm, tables, h5 = {}, {}, {}
    for name in ARMS:
        g = arms[name]
        res = lc.evaluate(g, tested, n_boot, n_boot_languages, seed)
        res["P2a"] = _restrict_p2a(res, g)
        per_arm[name] = res
        tables[name] = ga.build_table(g, n_boot=n_boot, seed=seed)
        h5[name] = ga.h_g5(tables[name], g, None, seed, n_boot)
    ratios = {name: _ratios(tables[name], arms[name], tested) for name in ARMS}
    q = {"Q0": {"per_arm": {a: {"P1": bool(per_arm[a]["P1"]["met"]), "P3a_INT4": bool(per_arm[a]["P3a"]["INT4"]["met"])} for a in NEW_ARMS}}}
    q["Q0"]["met"] = all(v["P1"] and v["P3a_INT4"] for v in q["Q0"]["per_arm"].values())
    q["Q1"] = q1(per_arm, ratios, tested)
    q["Q2"] = {"band": list(Q2_BAND), "retention_int6": {a: per_arm[a]["P2"]["retention"].get("INT6") for a in ARMS}}
    q["Q2"]["met"] = all(v is not None and Q2_BAND[0] <= v <= Q2_BAND[1] for v in q["Q2"]["retention_int6"].values())
    q["Q3"] = {a: {"H-G5b": {"count": h5[a]["b"]["count"], "n": h5[a]["b"]["n"], "met": h5[a]["b"]["met"]}, "H-G5a_met": h5[a]["a"]["met"], "best_factor_above_one": int(h5[a]["alpha_up_count"]["INT4"])} for a in ARMS}
    return {"arms": per_arm, **q, "verdict_tier1": verdict_tier1(q), "tested": tested, "n_tested": len(tested), "n_boot": n_boot, "n_boot_languages": n_boot_languages, "seed": seed,
            "ratios_int4_over_fp": ratios}


# ------------------------------------------------------------------------------------------------------------------------------ tier 2: Phase 1 survival and loss (plan Task 7)
import argparse
import csv
import json
from pathlib import Path

from gptq_ext import compare_stage1 as cs

ROOT = Path(__file__).resolve().parents[2]
S_BITS = (6, 4)
GAP_BITS = 6
TIER_PAIR = ("high", "low")
P_LEVEL = 0.05
PATHS_2B = {"legacy": "reports/functional_validation/gain_study/results/curves.csv", "rtn-matched": "reports/gptq_extension/results/gain/rtnm/curves.csv", "gptq": "reports/gptq_extension/results/gain/gptq/curves.csv"}
PAIRED_2B = "reports/gptq_extension/results/stage1/stage1_paired_per_language.csv"
ROSTER_2B = "reports/g0/phase1_languages.json"
SPEC_REL = "docs/superpowers/specs/2026-10-28-gemma9b-three-arm-confirmation-design.md"
"""Tier 2 (spec section 5), arms as `compare_stage1.load_arm` returns them (survived_pct and nll_delta per language and bit width; equal-budget stream):
  S1  mean survival over languages ordered GPTQ > matched RTN > legacy RTN at INT6 and INT4, GPTQ above legacy in at least req(24) languages at each;
  S2  mean loss increase ordered GPTQ < matched RTN < legacy RTN at INT6 and INT4, GPTQ below legacy in at least req(24) languages at each;
  S3  high-tier minus low-tier mean survival at INT6 smaller under GPTQ than under legacy RTN (mid-tier languages do not enter);
  S4  (reported) the verdict of the original three-test family per arm: the Holm-corrected p of T1 to T3 from `phase1_analysis.py`, and whether the arms agree.
Verdict: consistent = S1 and S2 and S3; partly consistent = S1 and S2 and not S3; different = not S1 or not S2."""


def _check_stage1(arms, roster, bits):
    missing = [a for a in ARMS if a not in arms]
    if missing:
        raise ValueError(f"missing arm(s) {missing}: Phase 1 needs {list(ARMS)}")
    ref = arms[ARMS[0]]
    for name in ARMS:
        a = arms[name]
        absent = [b for b in bits if b not in a["bits"]]
        if absent:
            raise ValueError(f"arm {name}: bit width(s) {absent} are missing (it has {sorted(a['bits'])})")
        if set(a["languages"]) != set(ref["languages"]):
            raise ValueError(f"arm {name}: its languages differ from those of arm {ARMS[0]}")
    if set(ref["languages"]) != set(roster):
        raise ValueError(f"the arms cover {len(set(ref['languages']))} languages but the roster has {len(roster)}: Phase 1 must cover the whole roster (a smoke-sized or partial run cannot be judged against the predictions)")


def evaluate_stage1(arms, roster, bits=S_BITS):
    """arms: {"legacy", "rtn-matched", "gptq": load_arm-like dicts}; roster: {language: {"tier": ...}}. -> {"S1", "S2", "S3", "verdict_tier2", "n_languages"}."""
    _check_stage1(arms, roster, bits)
    langs = list(arms[ARMS[0]]["languages"]); n = len(langs); req = ga.required(n, 24)
    out = {}
    for key, field, better in (("S1", "survived_pct", lambda g, l: g > l), ("S2", "nll_delta", lambda g, l: g < l)):
        per_bit = {}
        for b in bits:
            mean = {a: float(np.mean([arms[a][field][l][b] for l in langs])) for a in ARMS}
            ordered = (mean["gptq"] > mean["rtn-matched"] > mean["legacy"]) if key == "S1" else (mean["gptq"] < mean["rtn-matched"] < mean["legacy"])
            count = sum(1 for l in langs if better(arms["gptq"][field][l][b], arms["legacy"][field][l][b]))
            per_bit[str(b)] = {"mean": mean, "ordered": bool(ordered), "count_gptq_better_than_legacy": int(count)}
        out[key] = {"bits": per_bit, "required": req, "n": n, "met": all(v["ordered"] and v["count_gptq_better_than_legacy"] >= req for v in per_bit.values())}
    tiers = {t: [l for l in langs if roster[l]["tier"] == t] for t in TIER_PAIR}
    empty = [t for t, v in tiers.items() if not v]
    if empty:
        raise ValueError(f"S3 needs languages of the tiers {list(TIER_PAIR)}; none in {empty}")
    gap = {a: float(np.mean([arms[a]["survived_pct"][l][GAP_BITS] for l in tiers[TIER_PAIR[0]]]) - np.mean([arms[a]["survived_pct"][l][GAP_BITS] for l in tiers[TIER_PAIR[1]]])) for a in ARMS}
    out["S3"] = {"bits": GAP_BITS, "gap": gap, "n_per_tier": {t: len(v) for t, v in tiers.items()}, "met": bool(gap["gptq"] < gap["legacy"])}
    out["n_languages"] = n
    out["verdict_tier2"] = verdict_tier2(out)
    return out


def verdict_tier2(s):
    if not (s["S1"]["met"] and s["S2"]["met"]):
        return "different"
    return "consistent" if s["S3"]["met"] else "partly consistent"


def s4(analyses):
    """analyses: {arm: the `analysis.json` of `phase1_analysis.py` for that arm, or None}. -> {"words": {arm: "null" | "below 0.05: <tests>" | "not provided"}, "holm": {arm: {test: p}}, "agree": bool or None (fewer than two arms provided)}."""
    words, holm = {}, {}
    for a in ARMS:
        an = analyses.get(a)
        if an is None:
            words[a] = "not provided"; continue
        holm[a] = dict(an["holm"]); below = [k for k, p in an["holm"].items() if p < P_LEVEL]
        words[a] = f"below {P_LEVEL}: {', '.join(below)}" if below else "null"
    given = [words[a] for a in ARMS if words[a] != "not provided"]
    return {"words": words, "holm": holm, "agree": (len(set(given)) == 1) if len(given) >= 2 else None}


# ------------------------------------------------------------------------------------------------------------------------------ the committed 2B outputs, read with the same statistics
def _curve_ratios(path):
    """{language: net harm at the deepest down ratio, INT4 over FP} and the ratio, from a committed `curves.csv` (the statistic of `phase1_llama_compare.gemma_reference`, per language)."""
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    a = min(float(r["ratio"]) for r in rows if float(r["ratio"]) > 0)
    net = {s: {r["language"]: float(r["net"]) for r in rows if r["state"] == s and abs(float(r["ratio"]) - a) < 1e-12} for s in ("FP", "INT4")}
    return {c: net["INT4"][c] / net["FP"][c] for c in net["FP"] if net["FP"][c] > 0 and c in net["INT4"]}, a


def arms_from_paired_csv(path):
    """The three 2B arms of Phase 1 (survival percentage and loss increase per language and bit width) rebuilt from the committed paired table of the GPTQ extension's stage 1: legacy = side b and matched RTN = side a of the first contrast, GPTQ = side a of the second."""
    arms = {a: {"languages": [], "bits": set(), "survived_pct": {}, "nll_delta": {}} for a in ARMS}

    def put(arm, r, side):
        lang, b = r["language"], int(r["bits"]); x = arms[arm]
        if lang not in x["languages"]:
            x["languages"].append(lang)
        x["bits"].add(b)
        x["survived_pct"].setdefault(lang, {})[b] = float(r[f"survived_pct_{side}"]); x["nll_delta"].setdefault(lang, {})[b] = float(r[f"nll_delta_{side}"])
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["contrast"] == "matched RTN minus legacy RTN":
                put("rtn-matched", r, "a"); put("legacy", r, "b")
            elif r["contrast"] == "GPTQ minus matched RTN":
                put("gptq", r, "a")
    for x in arms.values():
        x["bits"] = sorted(x["bits"])
    return arms


def reference_2b(repo_root=ROOT):
    """The 2B numbers behind the thresholds, from the committed outputs: tier 1 (r per arm, the per-language ratios, how many languages GPTQ is below legacy) and tier 2 (S1 to S3 on the rebuilt arms)."""
    root = Path(repo_root)
    ratios, deepest = {}, None
    for arm, rel in PATHS_2B.items():
        ratios[arm], deepest = _curve_ratios(root / rel)
    r = {a: float(np.median(list(v.values()))) for a, v in ratios.items()}
    tier1 = {"r": r, "ratios": ratios, "deepest_down_ratio": deepest, "n": len(ratios["legacy"]), "count_gptq_below_legacy": sum(1 for c, v in ratios["gptq"].items() if v < ratios["legacy"][c])}
    tier2 = evaluate_stage1(arms_from_paired_csv(root / PAIRED_2B), cs.load_roster(root / ROSTER_2B))
    return {"tier1": tier1, "tier2": tier2}


# ------------------------------------------------------------------------------------------------------------------------------ report and command line
def _f(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def _yn(b):
    return "met" if b else "not met"


def report_text(res):
    """The report of both tiers (a tier that was not run says so and gives no verdict)."""
    t1, t2, ref, s4r = res["tier1"], res["tier2"], res["reference_2b"], res["s4"]
    out = ["# Gemma-2-9B against the project's findings: three quantizer arms (confirmation of the updated project)", "",
           f"The predictions, thresholds and verdict functions were frozen before any 9B number existed (`reports/functional_validation/gemma9b/amendment_A1_three_arm.json`, text in `{SPEC_REL}` section 5). "
           "All numbers come from the frozen Study G and Phase 1 code. Reference values are the committed Gemma-2-2B outputs (the same statistics).", "",
           f"## Tier 1, Study G (loss caused by removing a language's features): **{res['verdict_tier1']}**", ""]
    if t1 is None:
        out += ["Tier 1: not run.", ""]
    else:
        q0, q1_, q2, q3, r = t1["Q0"], t1["Q1"], t1["Q2"], t1["Q3"], ref["tier1"]
        p3b = t1["arms"]["legacy"]["P3b"]
        legacy_p3b = f"legacy arm meets P3b: {'yes' if q1_['evaluable'] else 'no'} (median ratio {_f(p3b['median_ratio'])}, larger at INT4 in {p3b['count_larger']} of {t1['n_tested']}, needs {p3b['required']} and a ratio above 1.3)"
        out += [f"{t1['n_tested']} languages tested (group of at least 16 members), {t1['arms']['legacy']['n_sentences']} held-out sentences per language, three matched-control draws.", "",
                "| prediction | rule (fixed beforehand) | Gemma-2-9B | Gemma-2-2B | result |", "|---|---|---|---|---|",
                f"| Q0 existence and grading under the better quantizers | P1 at FP and H-G1 at INT4 in the matched-RTN and GPTQ arms | "
                + "; ".join(f"{a}: P1 {_yn(v['P1'])}, H-G1 INT4 {_yn(v['P3a_INT4'])}" for a, v in q0["per_arm"].items()) + " | met in both arms | **" + _yn(q0["met"]) + "** |",
                f"| Q1 the INT4 amplification is smaller with GPTQ | r(GPTQ) <= r(matched), r(GPTQ) <= r(legacy) - {q1_['required_drop']}, GPTQ below legacy in at least {q1_['required']} languages; evaluable only if the legacy arm meets P3b | "
                f"r legacy {_f(q1_['r']['legacy'])}, matched {_f(q1_['r']['rtn-matched'])}, GPTQ {_f(q1_['r']['gptq'])}; GPTQ below legacy in {q1_['count_lower']} languages; {legacy_p3b} | "
                f"r {_f(r['r']['legacy'])} / {_f(r['r']['rtn-matched'])} / {_f(r['r']['gptq'])}; {r['count_gptq_below_legacy']} of {r['n']} | " + (f"**{_yn(q1_['met'])}**" if q1_["evaluable"] else "**not evaluable**") + " |",
                f"| Q2 6 bits are the same in every arm | macro retention at INT6 within [{q2['band'][0]}, {q2['band'][1]}] in all three arms | " + ", ".join(f"{a} {_f(v)}" for a, v in q2["retention_int6"].items()) + " | flat in all arms | **" + _yn(q2["met"]) + "** |",
                "| Q3 repair hint (reported) | H-G5b per arm; languages whose best factor at INT4 is above 1 | " + "; ".join(f"{a}: H-G5b {_yn(v['H-G5b']['met'])} ({v['H-G5b']['count']} of {v['H-G5b']['n']}), best factor above 1 in {v['best_factor_above_one']}" for a, v in q3.items())
                + " | H-G5b not met under matched RTN and GPTQ; best factor above 1 in 21, 13, 7 of 27 languages (legacy, matched, GPTQ) | reported |", "",
                "Legacy-freeze predictions of the legacy arm (P1 to P3b, verdict function of 2026-10-26): " + ", ".join(f"{k} {_yn(t1['arms']['legacy'][k]['met'])}" for k in ("P1", "P2a", "P2b", "P3b")) + f", P3a {_yn(t1['arms']['legacy']['P3a']['met'])}; label **{t1['arms']['legacy']['verdict']}**.", ""]
    out += [f"## Tier 2, Phase 1 survival and loss: **{res['verdict_tier2']}**", ""]
    if t2 is None:
        out += ["Tier 2: not run.", ""]
    else:
        r2 = ref["tier2"]
        def means(s, key, b, nd):
            return " / ".join(_f(s[key]["bits"][str(b)]["mean"][a], nd) for a in ARMS)
        out += [f"{t2['n_languages']} languages, equal-budget stream. Means over languages in the order legacy RTN / matched RTN / GPTQ.", "", "| prediction | rule (fixed beforehand) | Gemma-2-9B | Gemma-2-2B | result |", "|---|---|---|---|---|",
                f"| S1 survival ordering | GPTQ > matched > legacy at INT6 and INT4, GPTQ above legacy in at least {t2['S1']['required']} languages | INT6 {means(t2, 'S1', 6, 1)} %, INT4 {means(t2, 'S1', 4, 1)} %; GPTQ above legacy in "
                f"{t2['S1']['bits']['6']['count_gptq_better_than_legacy']} / {t2['S1']['bits']['4']['count_gptq_better_than_legacy']} languages | INT6 {means(r2, 'S1', 6, 1)} %, INT4 {means(r2, 'S1', 4, 1)} % | **{_yn(t2['S1']['met'])}** |",
                f"| S2 loss ordering | GPTQ < matched < legacy at INT6 and INT4, GPTQ below legacy in at least {t2['S2']['required']} languages | INT6 {means(t2, 'S2', 6, 4)}, INT4 {means(t2, 'S2', 4, 3)} nats/token | "
                f"INT6 {means(r2, 'S2', 6, 4)}, INT4 {means(r2, 'S2', 4, 3)} | **{_yn(t2['S2']['met'])}** |",
                f"| S3 the tier gap at 6 bits | high minus low mean survival at INT6 smaller under GPTQ than under legacy | gap " + " / ".join(_f(t2['S3']['gap'][a], 1) for a in ARMS) + " points | gap " + " / ".join(_f(r2['S3']['gap'][a], 1) for a in ARMS)
                + f" points | **{_yn(t2['S3']['met'])}** |",
                "| S4 the original tests T1 to T3 (reported) | the Holm-corrected p per arm | " + "; ".join(f"{a}: {w}" for a, w in s4r["words"].items()) + f"; arms agree: {s4r['agree']} | legacy null, matched RTN below 0.05 (T1, T2), GPTQ null | reported |", ""]
    out += ["## Reading and limits", "",
            "- Both verdicts are separate and are not merged. consistent / partly consistent / different are defined in the amendment; a result that is not consistent is a valid result.",
            "- This is a scale replication within one architecture (Gemma 2) and one SAE family (Gemma Scope, layer 20 of 42, width 16k): one model, one layer, one dictionary, weights-only rounding. It says nothing about other architectures, layers or SAEs.",
            "- Contrasts are per arm; no multiplicity rule covers three arms. The Gemma-2-2B legacy numbers are the committed ones; the 2B matched-RTN and GPTQ arms are exploratory (D-101 to D-110).",
            "- Hardware: one RTX PRO 6000 (Blackwell) with a different torch build from the 2B runs. The artifact hashes of the 9B weight sets are recorded on this GPU type; equality on other hardware is not claimed. The 2B bridge (bit-identical full-precision numbers) does not exist for 9B.",
            "- GPTQ saw Wikipedia text in the 27 languages and RTN saw none: a GPTQ-versus-RTN gap mixes error compensation and calibration; matched RTN separates the grid and granularity change only.", ""]
    return "\n".join(out)


def _jsonable(x):
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(f"not serializable: {type(x)}")


def _load_study_g(legacy_dirs, rtnm_dirs, gptq_dirs, ratios_json, manifest):
    """-> ({arm: GainRuns}, {arm: info}, tested). The two new arms run INT6 and INT4 only: their full-precision run is the legacy arm's (the directory that holds `func_lm_eval_FP.pt`)."""
    fp_dirs = [d for d in legacy_dirs if list(Path(d).glob("func_lm_eval_FP.pt"))]
    if len(fp_dirs) != 1:
        raise ValueError(f"exactly one legacy directory must hold func_lm_eval_FP.pt, found {len(fp_dirs)}")
    man = json.loads(Path(manifest).read_text())
    arms, info = {}, {}
    for name, dirs in (("legacy", list(legacy_dirs)), ("rtn-matched", fp_dirs + list(rtnm_dirs)), ("gptq", fp_dirs + list(gptq_dirs))):
        arms[name], info[name] = ga.load_runs(dirs, ratios_json)
        if list(man["languages"]) != list(arms[name].languages):
            raise ValueError(f"the manifest and the {name} runs cover different languages")
    return arms, info, lc.tested_languages(man)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Gemma-2-9B three-arm confirmation: the frozen predictions of tier 1 (Study G: Q0 to Q3) and tier 2 (Phase 1: S1 to S4) and both verdicts.")
    for arm in ("legacy", "rtnm", "gptq"):
        ap.add_argument(f"--{arm}-g", nargs="+", help=f"tier 1: job directories of the {arm} arm (func_lm_eval_<state>.pt + meta_lm_eval.json); the legacy ones include the FP run")
        ap.add_argument(f"--{arm}-p1", help=f"tier 2: the {arm} arm's Phase 1 stream directory (the `eq/` directory holding survival.pt)")
        ap.add_argument(f"--analysis-{arm}", help=f"tier 2, optional: `analysis.json` of phase1_analysis.py for the {arm} arm (S4)")
    ap.add_argument("--manifest"); ap.add_argument("--ratios-json"); ap.add_argument("--roster"); ap.add_argument("--repo-root", default=str(ROOT)); ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000); ap.add_argument("--n-boot-languages", type=int, default=5000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    g_dirs, p_dirs = [a.legacy_g, a.rtnm_g, a.gptq_g], [a.legacy_p1, a.rtnm_p1, a.gptq_p1]
    if any(g_dirs) and not all(g_dirs):
        ap.error("tier 1 needs --legacy-g, --rtnm-g and --gptq-g together")
    if any(p_dirs) and not all(p_dirs):
        ap.error("tier 2 needs --legacy-p1, --rtnm-p1 and --gptq-p1 together")
    if not all(g_dirs) and not all(p_dirs):
        ap.error("give tier 1 (--*-g) and/or tier 2 (--*-p1)")
    if all(g_dirs) and not (a.manifest and a.ratios_json):
        ap.error("tier 1 needs --manifest and --ratios-json")
    if all(p_dirs) and not a.roster:
        ap.error("tier 2 needs --roster")
    res = {"tier1": None, "tier2": None, "verdict_tier1": "not run", "verdict_tier2": "not run", "meta": {}}
    if all(g_dirs):
        arms, info, tested = _load_study_g(a.legacy_g, a.rtnm_g, a.gptq_g, a.ratios_json, a.manifest)
        res["tier1"] = evaluate_arms(arms, tested, a.n_boot, a.n_boot_languages, a.seed); res["verdict_tier1"] = res["tier1"]["verdict_tier1"]; res["meta"]["study_g_runs"] = info
    analyses = {}
    if all(p_dirs):
        loaded = {"legacy": cs.load_arm(a.legacy_p1), "rtn-matched": cs.load_arm(a.rtnm_p1), "gptq": cs.load_arm(a.gptq_p1)}
        res["tier2"] = evaluate_stage1(loaded, cs.load_roster(a.roster)); res["verdict_tier2"] = res["tier2"]["verdict_tier2"]; res["meta"]["phase1_dirs"] = {k: v["dir"] for k, v in loaded.items()}
        analyses = {arm: (json.loads(Path(p).read_text()) if p else None) for arm, p in (("legacy", a.analysis_legacy), ("rtn-matched", a.analysis_rtnm), ("gptq", a.analysis_gptq))}
    res["s4"] = s4(analyses)
    res["reference_2b"] = reference_2b(a.repo_root)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(res, indent=1, default=_jsonable) + "\n")
    (out / "report.md").write_text(report_text(res))
    print(f"wrote {a.out}: tier 1 {res['verdict_tier1']}, tier 2 {res['verdict_tier2']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
