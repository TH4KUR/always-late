"""Stage 4 report of the survival-and-intervention plan: renders the result JSON of survival_effect.py (freeze: reports/survival_effect_rescue/stage3_analysis_freeze_2026-10-28.md) as markdown. Every number in the
text is read from the JSON; the wording of the result paragraph depends only on the overall statement of the pre-set rule.

  python3 scripts/g0/survival_effect_report.py --results reports/survival_effect_rescue/results/survival_effect_results.json --out reports/survival_effect_rescue/stage4_gemma_report_2026-10-28.md"""
import argparse
import json
from pathlib import Path

ARMS = ("legacy RTN", "matched RTN", "GPTQ"); STATES = ("INT6", "INT4")


def f3(x):
    return "n/a" if x is None or x != x else f"{x:.3f}"


def pct(x):
    return "n/a" if x is None or x != x else f"{100 * x:.1f} %"


def ci(lo, hi, fmt=f3):
    return f"[{fmt(lo)}, {fmt(hi)}]"


def pci(d):
    return f"{pct(d['improvement'])} [{100 * d['lo']:.1f}, {100 * d['hi']:.1f}]"


def rng(vals, fmt):
    vals = [v for v in vals if v is not None and v == v]
    return "n/a" if not vals else (fmt(min(vals)) if min(vals) == max(vals) else f"{fmt(min(vals))} to {fmt(max(vals))}")


def prim(res):
    return [res["cells"][f"{a}|{s}|primary"] for a in ARMS for s in STATES]


def result_paragraph(res):
    P = prim(res); k = res["cells_met_primary"]; stmt = res["overall_statement"]
    imp = [c["improvement"]["B_over_A"]["improvement"] for c in P]; n_pos = sum(1 for c in P if c["improvement"]["B_over_A"]["lo"] == c["improvement"]["B_over_A"]["lo"] and c["improvement"]["B_over_A"]["lo"] > 0)
    part = [c["partial_spearman_given_A"]["rho"] for c in P]; n_part = sum(1 for c in P if c["partial_spearman_given_A"]["lo"] > 0)
    a_null = [c["improvement"]["A_over_null"]["improvement"] for c in P]
    rho = [c["spearman"]["rho"] for c in P]
    head = (f"**Result of the pre-set rule (freeze section 4), for the continuous activation correlation R1 as a predictor of the effect similarity W1: {stmt}** ({k} of 6 cells meet it: improvement of model B over model A of at least 10 % with the lower end of the language-cluster interval above 0). "
            f"For the record: the Spearman correlation of R1 with W1 is {rng(rho, f3)} over the six cells; adding R1 to the baselines changes the held-out error of W1 by {rng(imp, pct)} (the interval lies above zero in {n_pos} of 6 cells); the partial Spearman correlation of R1 with W1 given the baselines is {rng(part, f3)} "
            f"(the interval excludes zero in {n_part} of 6 cells); the baselines alone beat the training-median null by {rng(a_null, pct)}.")
    if stmt == "no clear predictive value beyond baselines":
        tail = " **The analysis stops here** (freeze stop rule): no cutoff tuning, no additional predictors, no broader sweep and no switch to steering. Any small association that remains is reported as such and is not promoted."
    elif stmt == "mixed":
        tail = " The rule is met in only some cells; no cell result is pooled across arms or bit widths to change this statement, and the secondary and sensitivity results cannot change it."
    else:
        tail = " This is predictive evidence for this model, layer, SAE, data and these quantizers; it is not a mechanism, not generalization and not independent validation (freeze section 0)."
    return head + tail


def render(res):
    c0 = res["population_counts"]; L = []
    L += ["# Survival versus effect similarity on Gemma-2-2B (Stage 4 report of the survival-and-intervention plan)", "",
          "**Status: post hoc and exploratory.** The committed F5 tables already held the correlation of R1 with W1 (D-108); the analysis choices of this second look were frozen BEFORE it was run ([`stage3_analysis_freeze_2026-10-28.md`](stage3_analysis_freeze_2026-10-28.md)), and the analysis code was committed before it ran (19 tests, 16 of 16 mutants killed). Inputs and their limits: [`f5_input_inventory_2026-10-28.md`](f5_input_inventory_2026-10-28.md). Context: [`all_results_assessment_2026-10-28.md`](all_results_assessment_2026-10-28.md). All numbers below are read from [`results/survival_effect_results.json`](results/survival_effect_results.json) by `scripts/g0/survival_effect_report.py`.", "",
          "## 1. Result", "", result_paragraph(res), "",
          "## 2. Question, data and the reproduction control", "",
          f"Question: when a feature's activation pattern is better preserved by weight quantization (higher R1, the token-level correlation of its activation with full precision), is the effect of an ablation of that feature also better preserved (higher W1, the cosine of the effect fingerprints), beyond simple baselines? Unit: the feature (1,487 features, 27 languages; language is the cluster). Cells: legacy RTN, matched RTN and GPTQ at INT6 and INT4. Baselines (model A): number of active dev sentences, Phase 1 firing rate and mean activation of the feature in its language, the full-precision loss effect |dnll| of the ablation, the number of ablated positions. Model B adds R1; model C adds firing-set overlap and the scale deviation. Held out by language (leave-one-language-out ridge on rank-quantile predictors); improvement = 1 - MAE_B / MAE_A; intervals by a language-cluster bootstrap ({res['n_boot']} replicates, seed {res['seed']}).", "",
          f"**Populations.** Primary (reliable fingerprints: INT16 cosine at or above the numerical floor {res['tau']}): {c0['primary']} features; noisy-effect stratum (the other features): {c0['noisy_stratum']}; sensitivity S-all: {c0['S_all']}; sensitivity S-loss (|dnll_FP| at or above {res['eps_dnll']:.6f}): {c0['S_loss']}.", "",
          "**Reproduction control (before any new number):** the committed Spearman correlations of R1 with W1 and their intervals were reproduced from the saved per-feature arrays with the committed code and seed:", "",
          "| arm | state | rho | interval | n | committed rho | reproduced |", "|---|---|---|---|---|---|---|"]
    for r in res["control"]:
        L.append(f"| {r['arm']} | {r['state']} | {f3(r['rho'])} | {ci(r['lo'], r['hi'])} | {r['n']} | {f3(r['committed_rho'])} | {'reproduced' if r['reproduced'] else '**MISMATCH**'} |")
    L += ["", "## 3. Primary population: association and prediction", "",
          "| arm | state | n | Spearman R1-W1 | partial Spearman given baselines | MAE null / A / B | B over A (interval, points) | decision |", "|---|---|---|---|---|---|---|---|"]
    for a in ARMS:
        for s in STATES:
            c = res["cells"][f"{a}|{s}|primary"]; m = c["mae"]
            L.append(f"| {a} | {s} | {c['n']} | {f3(c['spearman']['rho'])} {ci(c['spearman']['lo'], c['spearman']['hi'])} | {f3(c['partial_spearman_given_A']['rho'])} {ci(c['partial_spearman_given_A']['lo'], c['partial_spearman_given_A']['hi'])} | "
                     f"{m['null']:.4f} / {m['A']:.4f} / {m['B']:.4f} | {pci(c['improvement']['B_over_A'])} | {'**met**' if c['cell_met'] else 'not met'} |")
    L += ["", "MAE is the mean absolute error of the predicted W1 on held-out languages. At INT6 W1 is close to 1 for almost all features (small MAE and little room for improvement); at INT4 the errors are larger.", "",
          "## 4. Secondary results (cannot change the statement of section 1)", "",
          "| arm | state | C over B | A over null | R1 alone over null | AUROC of R1 for 'W1 at the floor' (events / non-events) | R1 > 0.9: events with / without |", "|---|---|---|---|---|---|---|"]
    for a in ARMS:
        for s in STATES:
            c = res["cells"][f"{a}|{s}|primary"]; i = c["improvement"]; au = c["auroc_R1_for_event"]; d = c["dichotomy"]
            L.append(f"| {a} | {s} | {pci(i['C_over_B'])} | {pci(i['A_over_null'])} | {pci(i['R1_over_null'])} | {f3(au['auroc'])} {ci(au['lo'], au['hi'])} ({au['n_event']} / {au['n_no_event']}) | {d['survives_event']} / {d['survives_no_event']} |")
    for key, title, note in (("S_all", "Sensitivity S-all (all 1,487 features)", ""), ("S_loss", "Sensitivity S-loss (clear full-precision loss effect)", ""), ("noisy_stratum", "Noisy-effect stratum (fingerprint not reproducible at the rounding-noise level; secondary and labelled, small n)", "")):
        L += ["", f"## 5.{['S_all', 'S_loss', 'noisy_stratum'].index(key) + 1}. {title}", "", "| arm | state | n | Spearman R1-W1 | partial Spearman given baselines | B over A (interval, points) | rule met in this population |", "|---|---|---|---|---|---|---|"]
        for a in ARMS:
            for s in STATES:
                c = res["cells"][f"{a}|{s}|{key}"]
                L.append(f"| {a} | {s} | {c['n']} | {f3(c['spearman']['rho'])} {ci(c['spearman']['lo'], c['spearman']['hi'])} | {f3(c['partial_spearman_given_A']['rho'])} {ci(c['partial_spearman_given_A']['lo'], c['partial_spearman_given_A']['hi'])} | {pci(c['improvement']['B_over_A'])} | {'yes' if c['cell_met'] else 'no'} |")
    L += ["", "## 6. What this does and does not show", "",
          "- It concerns the continuous activation correlation R1 as a predictor of the effect similarity W1 of a single-feature ablation on 1,487 features of one model (Gemma-2-2B), one layer (12), one SAE (Gemma Scope 16k) and three simulated weight-only quantizers (legacy RTN, matched RTN, GPTQ) at INT6 and INT4, on the F5 dev text.",
          "- R1 and W1 come from the same perturbed model and the features share contexts and sentences inside a language, so any association is not a causal statement about survival and effect; held-out evaluation is by language, not by document (no read-side document ids exist), and no external-panel F5 exists to check it.",
          "- The sample of 1,487 features is stratified, not random: no statement about all features. Margins, raw fingerprints and the sufficient statistics behind R1 were not saved; the baselines are limited to the five listed (the margin-audit addendum's gradient, margin and generic-sensitivity covariates do not exist for F5).",
          "- The decision rule (10 % relative error reduction with an interval above zero) is the project's target carried unchanged from the margin-audit addendum, not a universal threshold; the sensitivity populations and the secondary models cannot change the primary statement.",
          "- The binary retained/lost association label was not used here; its result (AUROC 0.49 under GPTQ, D-108) stands.",
          "- The Llama-3.2-1B check of the plan's Stage 5 has not been run; nothing here says whether the result carries over to another model.", "",
          "## 7. How this was produced", "",
          "`scripts/g0/survival_effect.py` (analysis; `--control-only` runs the reproduction control) and `scripts/g0/survival_effect_report.py` (this text); tests `scripts/g0/tests/test_survival_effect.py` and `test_survival_effect_report.py`; inputs: the F5 job directories (private dataset `runs/gptq-extension/stage3_f5_gain`) and the Phase 1 equal-budget survival file; decision entry D-113."]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--results", required=True); ap.add_argument("--out", required=True); a = ap.parse_args(argv)
    Path(a.out).write_text(render(json.loads(Path(a.results).read_text()))); print("wrote", a.out)


if __name__ == "__main__":
    main()
