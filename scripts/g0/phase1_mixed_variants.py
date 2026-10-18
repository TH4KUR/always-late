"""N7: reconcile A-034 (c) (decision D-021; pre-specified in handoff 003 section 4 before it was run). The exploratory mixed-effects logistic model of phase1_analysis
(section 7 of the proposal: logit Pr(survived) ~ bits + firing-rate stratum + S~ + tier + S~ x tier + random intercept per feature) has a POSITIVE S~ coefficient in the primary stream
(+1.19), while the firing-rate-matched contrast is negative at high S thresholds (N2), the magnitude-matched contrast is not negative (N6) and the threshold-free within-stratum
Spearman(S, survival) is about zero (N6a). The SAME model (`mixed_logit`, same estimator, same data, same seed) is refitted in variants: (i) unchanged, which must reproduce analysis.json
exactly; (ii) without the tier terms; (iii) with log10 of the maximum activation at the peak language (N6's m) added; (iv) the full formula restricted to the matched sets of N2 (firing-rate
strata) and of N6 (firing-rate x 10 magnitude bins) at one threshold. No new estimator.
Reading rule (fixed beforehand): if the S~ coefficient is <= 0 or not significant in variant (ii) or (iii), the conflict is explained by that term (report which); if it stays significantly
positive in all variants it is UNRESOLVED and no further variants are tried. Significance is strict (p exactly 0.05 is not significant). A pattern outside the rule is labelled as such."""
import json
from pathlib import Path

import numpy as np
import torch

import phase1_analysis as an
import phase1_magnitude as mg
import phase1_threshold as th

NO_TIER = "bits_c + C(stratum) + S_rank"
LOG_M = an.MIXED_FORMULA + " + log10_m"
TERM_NAMES = {"no_tier_all": "the tier terms", "log_m_all": "activation magnitude"}
ALPHA = 0.05


def matched_features(selp, tau, seed=0, strata=None):
    """Union of the selective features at S > tau and their matched controls (the run's own matched_sample), with the firing-rate strata of the run or the given replacement."""
    alt = th.alternative_selection(selp if strata is None else {**selp, "strata": strata}, tau, seed)
    return np.array(sorted(set(alt["selective"].nonzero().squeeze(1).tolist()) | set(alt["controls"].tolist())))


def reading_n7(v):
    positive = lambda k: v[k]["S_rank"]["coef"] > 0 and v[k]["S_rank"]["p"] < ALPHA
    terms = [k for k in ("no_tier_all", "log_m_all") if not positive(k)]
    if terms:
        return {"label": "explained by " + " and ".join(TERM_NAMES[k] for k in terms), "terms": terms}
    if all(positive(k) for k in ("full_matched_N2", "full_matched_N6")):
        return {"label": "unresolved", "terms": []}
    return {"label": "unresolved in the full sample; matched sets differ (outside the pre-specified rule)", "terms": []}


def _summary(out, formula):
    m = out["mixed"]; i = m["names"].index("S_rank")
    return {**m, "gee": out["gee"], "n_features": out["n_features"], "n_obs": out["n_obs"], "formula": formula, "S_rank": {k: m[k][i] for k in ("coef", "se", "p")}}


def run_variants(run_dir, roster_path, out_dir, tau=5.0, seed=0, analysis_json=None, gee=True):
    run_dir, out_dir = Path(run_dir), Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    surv = torch.load(run_dir / "survival.pt"); selp = torch.load(run_dir / "selectivity.pt"); meta = json.loads((run_dir / "meta.json").read_text())
    roster = {l["code"]: l for l in json.loads(Path(roster_path).read_text())["languages"]}; langs = list(surv["languages"]); lm = [roster[c] for c in langs]
    corr = {b: surv["corr"][b].numpy() for b in surv["bits"]}; fire = surv["firing_rate"].numpy(); bits = [b for b in (7, 6) if b in corr]       # exactly as run_analysis builds its mixed model
    sel = {k: selp[k].numpy() for k in ("selective", "supported", "peak", "S_rank", "strata", "controls")}; peak = sel["peak"]; maxa = surv["max_activation"].numpy().astype(float)
    table = lambda feats: an.feature_table(corr, fire, sel, lm, bits, features=feats)
    def log_m(st):
        m = maxa[peak[st["feature"]], st["feature"]]; return np.where(m > 0, np.log10(np.where(m > 0, m, 1.0)), np.nan)
    sup = np.flatnonzero(sel["supported"]); st_all = table(sup); V = {}
    V["full_all"] = _summary(an.mixed_logit(st_all, bits, gee=gee), an.MIXED_FORMULA)
    V["no_tier_all"] = _summary(an.mixed_logit(st_all, bits, gee=gee, formula=NO_TIER), NO_TIER)
    V["log_m_all"] = _summary(an.mixed_logit(st_all, bits, gee=gee, formula=LOG_M, extra={"log10_m": log_m(st_all)}), LOG_M)
    ids2 = matched_features(selp, tau, seed); V["full_matched_N2"] = _summary(an.mixed_logit(table(ids2), bits, gee=gee), an.MIXED_FORMULA)
    ev_full = sel["supported"].astype(bool) & (fire[peak, np.arange(len(peak))] > 0.001); m_full = maxa[peak, np.arange(len(peak))]
    ids6 = matched_features(selp, tau, seed, strata=mg.magnitude_strata(selp["strata"], selp["supported"], ev_full, m_full, 10)); V["full_matched_N6"] = _summary(an.mixed_logit(table(ids6), bits, gee=gee), an.MIXED_FORMULA)
    repro = diff = None
    if analysis_json is not None:
        saved = json.loads(Path(analysis_json).read_text())["exploratory"]["mixed_logit"]["mixed"]; same_names = saved["names"] == V["full_all"]["names"]
        diff = float(np.max(np.abs(np.array(saved["coef"], float) - np.array(V["full_all"]["coef"], float)))) if same_names else float("inf")
        repro = bool(same_names and [float(x) for x in saved["coef"]] == V["full_all"]["coef"])
    res = {"decision": "D-021", "regime": meta.get("regime"), "stream": meta.get("stream"), "tau": float(tau), "seed": seed, "variants": V, "reading": reading_n7(V), "reproduces_analysis_json": repro, "max_abs_coef_diff": diff}
    (out_dir / "mixed_variants.json").write_text(json.dumps(res, indent=1, default=float) + "\n"); (out_dir / "mixed_variants.md").write_text(render_md(res)); return res


def render_md(res):
    L = [f"# Mixed-model variants for the S~ coefficient, regime `{res['regime']}`, stream `{res['stream']}` (decision D-021; A-034 (c))", "",
         "The exploratory model of the analysis (proposal section 7) has a positive S~ coefficient while the matched contrasts are not positive (N2, N6) and the threshold-free within-stratum correlation is about zero. "
         "The SAME model and estimator are refitted on the same data: (i) unchanged, (ii) without the tier terms, (iii) with log10 maximum activation at the peak language added, (iv) the full formula restricted to the matched sets of N2 "
         f"(firing-rate strata) and N6 (firing-rate x 10 magnitude bins) at S > {res['tau']:g}. **Pre-specified reading rule:** if the S~ coefficient is <= 0 or not significant in variant (ii) or (iii), the conflict is explained by that term; "
         "if it stays significantly positive in all variants it is unresolved and no further variants are tried. Significance is strict (p exactly 0.05 is not significant).", "",
         f"Variant (i) reproduces the saved `analysis.json` coefficients exactly: **{res['reproduces_analysis_json']}** (largest absolute difference {res['max_abs_coef_diff']}). The quadrature-ML coefficients are bit-identical only at a fixed BLAS "
         "thread count (the saved analysis used 4; another thread count moves them by about 1e-10, A-036).", "",
         "| variant | formula | features | S~ (SE), p | tier | S~:tier | log10 m | GEE S~ (p) | sigma | converged |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in res["variants"].items():
        g = dict(zip(v["names"], v["coef"])); gee = dict(zip(v["gee"]["names"], zip(v["gee"]["coef"], v["gee"]["p"]))) if v["gee"] else {}
        fmt = lambda n: f"{g[n]:+.3f}" if n in g else ""
        L.append(f"| {k} | `{v['formula']}` | {v['n_features']} | {v['S_rank']['coef']:+.3f} ({v['S_rank']['se']:.3f}), p {v['S_rank']['p']:.3g} | {fmt('tier')} | {fmt('S_rank:tier')} | {fmt('log10_m')} | "
                 + (f"{gee['S_rank'][0]:+.3f} ({gee['S_rank'][1]:.2g})" if "S_rank" in gee else "") + f" | {v['sigma']:.2f} | {v['converged']} |")
    L += ["", f"**Reading (by the rule fixed beforehand): {res['reading']['label']}.**"]
    return "\n".join(L) + "\n"


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=run_variants.__doc__)
    ap.add_argument("--run", required=True); ap.add_argument("--out", required=True); ap.add_argument("--analysis-json", default=None)
    ap.add_argument("--roster", default=str(Path(__file__).resolve().parents[2] / "reports/g0/phase1_languages.json")); ap.add_argument("--tau", type=float, default=5.0); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv); r = run_variants(a.run, a.roster, a.out, a.tau, a.seed, a.analysis_json)
    for k, v in r["variants"].items():
        print(f"{k}: S~ {v['S_rank']['coef']:+.3f} (p {v['S_rank']['p']:.3g}), {v['n_features']} features, converged {v['converged']}")
    print("reproduces analysis.json:", r["reproduces_analysis_json"], "| reading:", r["reading"]["label"]); print("wrote", Path(a.out) / "mixed_variants.md")


if __name__ == "__main__":
    main()
