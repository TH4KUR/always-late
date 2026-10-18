"""N8: does survival depend on the language's resource tier, at the LANGUAGE level? (decision D-022; pre-specified in handoff 004 section 4 when only the header of the per-language
files had been read). Source: proposal RQ2 ("does survival rate ... depend on the resource level of the language?") and the RQ2 row of paper/main.tex (per-language outcome, 10,000-draw
permutation test with script family as covariate, one-sided); the unchanged Freedman-Lane test of phase1_analysis; Srivastava (2026, arXiv:2607.24276) character-fertility tax multiplier
(phase1_behavior.fertility_multiplier, N1). Outcome per language and width: the fraction of the language's ACTIVE features (full-precision firing rate in that language above 0.001) that
survive (r > 0.9) -- all active features, not only selective ones, unweighted across languages; secondary outcome the language's median r.
Statistic: slope on the ordinal tier (low 0, mid 1, high 2), ONE-SIDED (higher tier, higher survival: the proposal's positive branch), Freedman-Lane permutation with script-family dummies;
Holm over the two confirmatory widths INT7 and INT6; INT5 descriptive; High-minus-Low and the Joshi-class slope as sensitivities; the same test with the fertility multiplier as a further
nuisance covariate. Reading rule (fixed beforehand): survival depends on tier if the INT6 slope is positive with Holm < 0.05; if so, with fertility added a p >= 0.05 (strict) means the
effect is explained by tokenization cost, otherwise it is beyond fertility; if the first condition fails nothing is claimed and no equivalence is asserted."""
import csv
import json
from pathlib import Path

import numpy as np

import phase1_analysis as an
import phase1_behavior as bh

ALPHA = 0.05
NO = "no language-level tier dependence detected (no equivalence claimed)"
EXPLAINED = "depends on tier; explained by tokenization cost (fertility)"
BEYOND = "depends on tier; beyond fertility"


def language_table(csv_path, codes, widths=(7, 6, 5)):
    """Per-language survival fraction (percent / 100), median r and active-feature count at each width, in the order of `codes`; a missing language or width, or a duplicated row, is an error."""
    seen = {}
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        key = (r["language"], int(r["bits"]))
        if key in seen:
            raise ValueError(f"duplicate row for {key}")
        seen[key] = r
    out = {"survival": {}, "median_corr": {}, "n_active": {}}
    for b in widths:
        missing = [c for c in codes if (c, b) not in seen]
        if missing:
            raise ValueError(f"width {b}: no row for {missing[:3]}")
        out["survival"][b] = np.array([float(seen[(c, b)]["survived_>0.9_pct"]) / 100 for c in codes])
        out["median_corr"][b] = np.array([float(seen[(c, b)]["median_corr"]) for c in codes])
        out["n_active"][b] = np.array([int(seen[(c, b)]["n_active_features"]) for c in codes])
    return out


def reading_n8(slope_int6, holm_int6, p_fertility_int6, alpha=ALPHA):
    if not (slope_int6 > 0 and holm_int6 < alpha):
        return NO
    return EXPLAINED if p_fertility_int6 >= alpha else BEYOND


def run_language_level(per_language_csv, meta_json, stats_json, roster_path, out_dir, widths=(7, 6, 5), n_perm=10000, seed=0, split="devtest"):
    if 7 not in widths or 6 not in widths:
        raise ValueError("the confirmatory widths INT7 and INT6 are required")
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    roster = json.loads(Path(roster_path).read_text())["languages"]; codes = [l["code"] for l in roster]
    tier = np.array([an.tier_score(l["tier"]) for l in roster]); fam = sorted({l["script_family"] for l in roster}); script = np.array([fam.index(l["script_family"]) for l in roster])
    joshi = np.array([l["joshi_class"] for l in roster], dtype=float)
    meta = json.loads(Path(meta_json).read_text()); tokens = {r["code"]: r["tokens"] for r in meta["corpus_report"][split]}
    stats_all = json.loads(Path(stats_json).read_text()); stats = {c: stats_all[c][split] for c in codes}
    fert = bh.fertility_multiplier(tokens, stats, codes, unit="char", ref="eng_Latn"); tab = language_table(per_language_csv, codes, widths)
    W = {}
    for b in widths:
        y = tab["survival"][b]; base = an.freedman_lane_tier_test(y, tier, script, n_perm, seed); wf = an.freedman_lane_tier_test(y, tier, script, n_perm, seed, covariates=fert)
        jo = an.freedman_lane_tier_test(y, joshi, script, n_perm, seed); sec = an.freedman_lane_tier_test(tab["median_corr"][b], tier, script, n_perm, seed)
        W[str(b)] = {"slope": base["slope"], "t": base["t"], "p": base["p"], "high_minus_low": base["high_minus_low"], "slope_fertility": wf["slope"], "p_fertility": wf["p"],
                     "joshi_slope": jo["slope"], "joshi_p": jo["p"], "mean_by_tier": {str(k): float(y[tier == k].mean()) for k in (0, 1, 2)},
                     "secondary_median_r": {"slope": sec["slope"], "p": sec["p"], "slope_fertility": an.freedman_lane_tier_test(tab["median_corr"][b], tier, script, n_perm, seed, covariates=fert)["slope"]}}
    adj = an.holm([W["7"]["p"], W["6"]["p"]]); W["7"]["holm_p"], W["6"]["holm_p"] = float(adj[0]), float(adj[1])
    res = {"decision": "D-022", "stream": meta.get("stream"), "regime": meta.get("regime"), "n_languages": len(codes), "n_perm": n_perm, "seed": seed, "languages": codes, "fertility": [float(x) for x in fert],
           "widths": W, "reading": reading_n8(W["6"]["slope"], W["6"]["holm_p"], W["6"]["p_fertility"])}
    (out_dir / "language_level.json").write_text(json.dumps(res, indent=1) + "\n"); (out_dir / "language_level.md").write_text(render_md(res)); return res


def render_md(res):
    L = [f"# Language-level survival by resource tier, regime `{res['regime']}`, stream `{res['stream']}` (decision D-022)", "",
         "This analysis was pre-specified (handoff 004, written when only the header of the per-language files had been read). Question: does the survival of ALL active features of a language depend on the language's resource tier "
         "(proposal RQ2; paper RQ2 row), and is it explained by character fertility (Srivastava 2026, arXiv:2607.24276: tax multiplier = tokens per character relative to English, FLORES-200 parallel text)? "
         f"Outcome per language: the fraction of its active features with r > 0.9 (unweighted over {res['n_languages']} languages). Statistic: slope on the ordinal tier (low 0, mid 1, high 2), one-sided "
         f"(higher tier, higher survival), Freedman-Lane permutation ({res['n_perm']} draws, seed {res['seed']}) with script-family dummies as nuisance; Holm over the two confirmatory widths INT7 and INT6; "
         "INT5 descriptive; the same test with the fertility multiplier added as a nuisance covariate. **Reading rule fixed beforehand:** survival depends on tier if the INT6 slope is positive with Holm < 0.05; "
         "if so, a fertility-adjusted p >= 0.05 (strict) means the effect is explained by tokenization cost, otherwise it is beyond fertility; if the first condition fails nothing is claimed and no equivalence is asserted.", "",
         "| width | mean survival low / mid / high tier | slope (p, Holm) | High - Low | slope with fertility (p) | Joshi-class slope (p) | median-r slope (p) |", "|---|---|---|---|---|---|---|"]
    for b, w in res["widths"].items():
        m = w["mean_by_tier"]; holm = f", Holm {w['holm_p']:.3g}" if "holm_p" in w else " (descriptive)"
        L.append(f"| INT{b} | {m['0'] * 100:.1f} / {m['1'] * 100:.1f} / {m['2'] * 100:.1f} % | {w['slope']:+.4f} (p {w['p']:.3g}{holm}) | {w['high_minus_low'] * 100:+.1f} pp | {w['slope_fertility']:+.4f} (p {w['p_fertility']:.3g}) | "
                 f"{w['joshi_slope']:+.4f} (p {w['joshi_p']:.3g}) | {w['secondary_median_r']['slope']:+.4f} (p {w['secondary_median_r']['p']:.3g}) |")
    L += ["", f"**Reading (by the rule fixed beforehand): {res['reading']}.**"]
    return "\n".join(L) + "\n"


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=run_language_level.__doc__)
    ap.add_argument("--per-language", required=True); ap.add_argument("--meta", required=True); ap.add_argument("--stats", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--roster", default=str(Path(__file__).resolve().parents[2] / "reports/g0/phase1_languages.json")); ap.add_argument("--n-perm", type=int, default=10000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv); r = run_language_level(a.per_language, a.meta, a.stats, a.roster, a.out, n_perm=a.n_perm, seed=a.seed)
    for b, w in r["widths"].items():
        print(f"INT{b}: slope {w['slope']:+.4f} p {w['p']:.3g}" + (f" Holm {w['holm_p']:.3g}" if "holm_p" in w else "") + f"; with fertility {w['slope_fertility']:+.4f} p {w['p_fertility']:.3g}")
    print("reading:", r["reading"]); print("wrote", Path(a.out) / "language_level.md")


if __name__ == "__main__":
    main()
