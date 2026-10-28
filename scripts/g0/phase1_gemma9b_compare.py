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
