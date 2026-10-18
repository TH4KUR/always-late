"""N1: the behavioural link of the team's evaluation-plan table (paper/main.tex, row "Behavioral Link"; decision D-018): per-language selective-feature damage fraction
against the per-language next-token loss increase, Spearman rank correlation and multivariable OLS with overall feature damage and token fertility as covariates.
Definitions and their sources (nothing here is a replication of Duan et al.; it is a labelled extension that reads the files phase1_run.save_outputs wrote):
- survival and damage thresholds t_s = 0.9, t_d = 0.5, active = full-precision firing rate > 0.001: Duan et al. 2026 (their script 03, strict inequalities) and the proposal.
- damage fraction = 1 - survival (r not above 0.9), the proposal's own usage in the G0 criterion (b) "damaged fraction 18.74% = 1 - 81.26%"; r < 0.5 (their 'damaged' class) as sensitivity.
- selective features are read at their peak language (D-013); overall damage = the same fraction over every feature active in the language.
- loss increase = mean next-token cross-entropy of the quantized model minus that of the full-precision model, same tokens (phase1_run's exact float64 sums).
- token fertility = Srivastava (2026, arXiv:2607.24276, section 3; the source the proposal cites for fertility): headline 'tax multiplier' = a language's fertility divided by
  English's under the same tokenizer on FLORES-200 parallel text; their fallback for scripts without word spaces is character fertility (tokens per character), which is the
  only unit defined for all 27 languages here (five scripts have no spaces), so it is the default; word fertility is available for space-delimited languages.
- Spearman rho with a one-sided test in the direction of the table's 'positive' branch (more selective damage, more loss increase); OLS on the raw predictors."""
import json
from pathlib import Path

import numpy as np

UNITS = ("char", "word")


class UndefinedLink(ValueError):
    """The rank correlation or the regression cannot be computed for this cell (constant input or too few languages); a data error such as a NaN correlation stays a plain ValueError."""


def damage_fractions(corr, fire, selective, peak, cut=0.9, threshold=0.001, kind="survival", damaged_cut=0.5):
    """Per language: damage fraction of the selective features whose peak language it is, and of every active feature. Evaluable = full-precision firing rate in that
    language above the threshold (their activity rule); damage = r not above `cut` (kind 'survival': 1 - survival) or r below `damaged_cut` (kind 'damaged')."""
    if kind not in ("survival", "damaged"):
        raise ValueError(f"unknown kind {kind!r}")
    corr, fire, selective, peak = np.asarray(corr, float), np.asarray(fire, float), np.asarray(selective, bool), np.asarray(peak)
    n_lang = corr.shape[0]; out = {k: np.full(n_lang, np.nan) for k in ("selective", "overall")}; out["n_selective"] = np.zeros(n_lang, int); out["n_active"] = np.zeros(n_lang, int)
    for l in range(n_lang):
        active = fire[l] > threshold
        for key, m in (("overall", active), ("selective", active & selective & (peak == l))):
            r = corr[l, m]
            if np.isnan(r).any():
                raise ValueError(f"NaN correlation among the {int(m.sum())} evaluated features of language {l}")
            if key == "selective":
                out["n_selective"][l] = int(m.sum())
            else:
                out["n_active"][l] = int(m.sum())
            if m.any():
                out[key][l] = float((~(r > cut)).mean()) if kind == "survival" else float((r < damaged_cut).mean())
    return out


def delta_loss(loss_sum, loss_n, bits):
    """Per-language mean next-token loss of the quantized model minus that of the full-precision model, on the same tokens (the token counts must be identical)."""
    fp_s, fp_n, q_s, q_n = (np.asarray(x, float) for x in (loss_sum["FP"], loss_n["FP"], loss_sum[bits], loss_n[bits]))
    if not np.array_equal(fp_n, q_n):
        raise ValueError("full-precision and quantized losses must be summed over the same tokens")
    return q_s / q_n - fp_s / fp_n


def fertility_multiplier(tokens, stats, codes, unit="char", ref="eng_Latn"):
    """Srivastava's 'tax multiplier': (tokens per unit of text in the language) / (the same for English) on the parallel FLORES sentences. tokens {code: token count},
    stats {code: {sentences, chars, words}} of the same split; the sentence counts must agree because the text is parallel."""
    if unit not in UNITS:
        raise ValueError(f"unit must be one of {UNITS}")
    key = "chars" if unit == "char" else "words"
    n_sent = {stats[c]["sentences"] for c in list(codes) + [ref]}
    if len(n_sent) != 1:
        raise ValueError(f"parallel text must have equal sentence counts, got {sorted(n_sent)}")
    fert = lambda c: tokens[c] / stats[c][key]
    return np.array([fert(c) / fert(ref) for c in codes])


def _ols(y, cols, names):
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X = sm.add_constant(np.column_stack(cols), prepend=True); fit = sm.OLS(y, X).fit(); labels = ["intercept"] + list(names); ci = fit.conf_int()
    return {"coef": dict(zip(labels, map(float, fit.params))), "se": dict(zip(labels, map(float, fit.bse))), "p": dict(zip(labels, map(float, fit.pvalues))),
            "ci95": {k: [float(ci[i, 0]), float(ci[i, 1])] for i, k in enumerate(labels)}, "r2": float(fit.rsquared), "n": int(len(y)), "df_resid": int(fit.df_resid),
            "vif": {k: float(variance_inflation_factor(X, i + 1)) for i, k in enumerate(names)}}


def behavioural_link(delta, dmg_sel, dmg_all, fert, names=None, min_languages=8):
    """Spearman rho (one-sided 'greater' and two-sided) of selective damage against the loss increase, and OLS delta ~ selective + overall damage + fertility.
    Languages with an undefined value are dropped and listed."""
    delta, dmg_sel, dmg_all, fert = (np.asarray(x, float) for x in (delta, dmg_sel, dmg_all, fert))
    names = list(names) if names is not None else [str(i) for i in range(len(delta))]
    ok = ~(np.isnan(delta) | np.isnan(dmg_sel) | np.isnan(dmg_all) | np.isnan(fert))
    if int(ok.sum()) < min_languages:
        raise UndefinedLink(f"only {int(ok.sum())} languages with all four values (need {min_languages})")
    if np.ptp(dmg_sel[ok]) == 0 or np.ptp(delta[ok]) == 0:
        raise UndefinedLink("constant selective damage or loss increase: rank correlation undefined")
    from scipy.stats import spearmanr
    two, one = spearmanr(dmg_sel[ok], delta[ok]), spearmanr(dmg_sel[ok], delta[ok], alternative="greater")
    return {"n_languages": int(ok.sum()), "dropped": [n for n, k in zip(names, ok) if not k],
            "spearman": {"rho": float(two.statistic), "p_two_sided": float(two.pvalue), "p_one_sided": float(one.pvalue)},
            "ols": _ols(delta[ok], [dmg_sel[ok], dmg_all[ok], fert[ok]], ("selective_damage", "overall_damage", "fertility"))}


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, (np.floating, float)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def run_link(run_dir, stats_path, out_dir, widths=(7, 6, 5), threshold=0.001, cut=0.9, damaged_cut=0.5, split="devtest"):
    """Behavioural link for one stream directory (survival.pt, selectivity.pt, meta.json) and the FLORES text statistics file ({code: {split: {sentences, chars, words}}}).
    Fertility is character fertility relative to English: the only unit defined for all languages (five panel scripts have no word spaces; Srivastava's own limitation note)."""
    import torch
    run_dir, out_dir = Path(run_dir), Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    surv, selp = torch.load(run_dir / "survival.pt"), torch.load(run_dir / "selectivity.pt"); meta = json.loads((run_dir / "meta.json").read_text())
    stats_all = json.loads(Path(stats_path).read_text()); langs = list(surv["languages"])
    tokens = {r["code"]: r["tokens"] for r in meta["corpus_report"][split]}; stats = {c: stats_all[c][split] for c in langs}
    if "eng_Latn" not in langs:
        raise ValueError("English is the reference of the fertility multiplier (Srivastava 2026) and must be in the panel")
    fert = fertility_multiplier(tokens, stats, langs, unit="char", ref="eng_Latn")
    res = {"stream": meta.get("stream"), "regime": meta.get("regime"), "split": split, "unit": "character", "languages": langs, "fertility": fert, "widths": {}, "decision": "D-018"}
    sel, peak, fire = selp["selective"].numpy(), selp["peak"].numpy(), surv["firing_rate"].numpy()
    for w in widths:
        delta = delta_loss(surv["loss_sum"], surv["loss_n"], w); entry = {"delta_loss": delta}
        for kind in ("survival", "damaged"):
            dm = damage_fractions(surv["corr"][w].numpy(), fire, sel, peak, cut=cut, threshold=threshold, kind=kind, damaged_cut=damaged_cut)
            try:
                entry[kind] = behavioural_link(delta, dm["selective"], dm["overall"], fert, names=langs)
            except UndefinedLink as e:
                entry[kind] = {"undefined": str(e)}
            entry[kind]["per_language"] = {k: dm[k] for k in dm}
        res["widths"][str(w)] = entry
    res = _clean(res); (out_dir / "behavior.json").write_text(json.dumps(res, indent=1) + "\n"); (out_dir / "behavior.md").write_text(render_md(res)); return res


def render_md(res):
    L = [f"# Behavioural link, stream `{res['stream']}` (regime `{res['regime']}`; decision D-018)", "",
         "Per-language selective-feature damage fraction against the next-token loss increase, as in the evaluation-plan table of `paper/main.tex` (row 'Behavioral Link'). "
         f"Damage fraction = 1 - survival (r not above 0.9, the proposal's own usage in G0 (b)); 'damaged' = r below 0.5 as sensitivity (Duan et al.). Token fertility = Srivastava (2026, arXiv:2607.24276) "
         f"tax multiplier relative to English on parallel FLORES-200 {res['split']} text, in character fertility (tokens per character; their fallback for scripts without word spaces, five of our scripts have none). "
         "Spearman rho with a one-sided test in the table's 'positive' direction (more selective damage, more loss increase); OLS on raw predictors; each language is one observation.", ""]
    for w, e in res["widths"].items():
        L += [f"## INT{w}", "", "| damage definition | n languages | Spearman rho | p (one-sided) | p (two-sided) | OLS slope of selective damage | SE | p | VIF (selective / overall / fertility) |", "|---|---|---|---|---|---|---|---|---|"]
        for kind, label in (("survival", "1 - survival (r <= 0.9)"), ("damaged", "damaged (r < 0.5)")):
            k = e[kind]
            if "undefined" in k:
                L.append(f"| {label} | undefined ({k['undefined']}) | | | | | | | |"); continue
            o = k["ols"]; v = o["vif"]
            L.append(f"| {label} | {k['n_languages']} | {k['spearman']['rho']:+.3f} | {k['spearman']['p_one_sided']:.3g} | {k['spearman']['p_two_sided']:.3g} | {o['coef']['selective_damage']:+.4g} | "
                     f"{o['se']['selective_damage']:.3g} | {o['p']['selective_damage']:.3g} | {v['selective_damage']:.1f} / {v['overall_damage']:.1f} / {v['fertility']:.1f} |")
        L.append("")
    return "\n".join(L) + "\n"


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=run_link.__doc__)
    ap.add_argument("--run", required=True); ap.add_argument("--stats", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--widths", type=int, nargs="*", default=[7, 6, 5])
    a = ap.parse_args(argv); r = run_link(a.run, a.stats, a.out, widths=tuple(a.widths))
    for w, e in r["widths"].items():
        cell = lambda k: f"{k} undefined" if "undefined" in e[k] else f"{k} rho {e[k]['spearman']['rho']:+.3f} p1 {e[k]['spearman']['p_one_sided']:.3g}"
        print(f"INT{w}: " + "; ".join(cell(k) for k in ("survival", "damaged")))
    print("wrote", Path(a.out) / "behavior.md")


if __name__ == "__main__":
    main()
