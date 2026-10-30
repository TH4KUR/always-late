# Does activation survival predict whether an intervention keeps its effect? (plain-language summary and file map)

**Reporting update, 30 October 2026:** The positive association is a result: activation correlation adds about 3–4% predictive improvement beyond the specified baselines in Gemma, with intervals above zero. Missing the pre-set 10% practical target does not mean no predictive information. This result complements the later measurement study; gain and perturbation metrics do not add resolved predictive value there. The repeated analyses overlap in data, so their agreement is convergent evidence rather than independent replications. [Current synthesis](../FINAL_RESEARCH_QUESTIONS.md), [research paper](../../paper/research-paper/main.tex), [current status](../CURRENT_STATUS.md).

Written 2026-10-28 22:57 IST by Claude. Authorized by the user (D-110); plan: [`../../docs/superpowers/plans/2026-10-28-survival-and-intervention-publication-rescue.md`](../../docs/superpowers/plans/2026-10-28-survival-and-intervention-publication-rescue.md). Everything here is **post hoc and exploratory**; the analysis choices were frozen before each analysis was run and the code was committed before it ran.

**The question.** A feature's "survival" after rounding the weights is how well its activation pattern still matches the full-precision model (R1, a correlation). The effect similarity (W1) is how much the effect of switching that feature off still matches. Does a higher R1 predict a higher W1, beyond simple facts about the feature (how often it fires, how strongly, how big its effect is)?

**The answer, in short.** Only weakly. On Gemma-2-2B (1,487 features, three ways of rounding, 6 and 4 bits) R1 is moderately correlated with W1 (rank correlation 0.30 to 0.52) and part of that survives the simple baselines (0.24 to 0.40), but adding R1 improves the held-out prediction of W1 by only 3 to 4 %, against the 10 % target fixed in advance, in all six cells (D-113): by the pre-set rule, "no clear predictive value beyond baselines". On Llama-3.2-1B (a different model and dictionary, 643 features) the statement is the same: the association beyond the baselines repeats at 4 bits (partial 0.25, as on Gemma), cannot be resolved at 6 bits, and the 10 % target is not met (D-114). The simple baselines themselves predict W1 hardly better than the median. This agrees with the earlier finding that the yes/no "retained or lost" label does not predict function (D-108) and with the all-results review (D-112): survival numbers are a weak guide to whether an intervention keeps its effect.

**What this does not show.** Not a mechanism (R1 and W1 come from the same perturbed model); not generalization beyond these two model-and-dictionary setups; not independent validation (held out by language, not by document; no external-panel F5 exists); no population claim from two systems.

| file | what |
|---|---|
| [`all_results_assessment_2026-10-28.md`](all_results_assessment_2026-10-28.md) | Stage 1: the review of all results and the decision gate (D-112) |
| [`f5_input_inventory_2026-10-28.md`](f5_input_inventory_2026-10-28.md) | Stage 0: what the saved F5 files hold |
| [`stage3_analysis_freeze_2026-10-28.md`](stage3_analysis_freeze_2026-10-28.md) | Stage 3: the frozen analysis choices (committed before the analysis) |
| [`stage4_gemma_report_2026-10-28.md`](stage4_gemma_report_2026-10-28.md), [`results/`](results/survival_effect_results.json) | Stage 4: the Gemma result, generated from the result file (D-113) |
| [`llama/llama_f5_addendum_2026-10-28.md`](llama/llama_f5_addendum_2026-10-28.md), [`llama/llama_f5_sample.json`](llama/llama_f5_sample.json) | Stage 5: the frozen Llama setting and sample (committed before any Llama outcome) |
| [`llama/llama_vs_gemma_report_2026-10-28.md`](llama/llama_vs_gemma_report_2026-10-28.md), [`llama/results/`](llama/results/llama_survival_effect_results.json) | Stage 5: the Llama result next to Gemma (D-114) |

Code: `scripts/g0/survival_effect.py`, `survival_effect_report.py`, `survival_effect_llama.py`, `survival_effect_llama_report.py`, `phase1_llama_f5_sample.py`, the `f5` task of `phase1_llama_run.py`; tests in `scripts/g0/tests/` and `tests_torch/`. Decisions D-110 to D-114 in [`../decisions_and_anomalies.md`](../decisions_and_anomalies.md).
