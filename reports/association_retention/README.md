# Association-retention study (D-035): index

Question: do the SAE features selected as language-selective before quantization keep their language association across all 27 languages after RTN weight quantization? Exploratory, mask regime, Gemma-2-2B, layer-12 Gemma Scope 16k SAE, FLORES-200.

| file | what it is |
|---|---|
| [`interpretation.md`](interpretation.md) | the reading of the results, with limits (start here) |
| [`next_steps.md`](next_steps.md) | follow-up options A to F for the review; nothing started |
| [`results/eq/association_retention.md`](results/eq/association_retention.md) | generated report, equal-budget stream, 1,000 paired bootstrap draws (also `analysis.json`, `language_summary.csv`) |
| [`results/all/association_retention.md`](results/all/association_retention.md) | generated report, all-token stream, aggregate profile (no intervals) |
| [`results/crosscheck-eq/crosscheck.md`](results/crosscheck-eq/crosscheck.md) | firing-rate view against activation-mean view; status against the historical correlation |
| [`results/robustness-eq/robustness.md`](results/robustness-eq/robustness.md) | support-rule sensitivity; loss against baseline covariates |
| [`artifact_inventory.md`](artifact_inventory.md), `frozen_cohort_manifest.json` | what the old saved files could support; frozen cohort and provenance |
| [`plan_verification.md`](plan_verification.md) | verification of the plan before implementation (code and result statements, prior art, register corrections) |
| [`STATUS.md`](STATUS.md), [`cron_prompt.md`](cron_prompt.md) | live status log and the prompt of the 50-minute job |

Plan and design: [`docs/superpowers/plans/2026-10-21-language-association-retention.md`](../../docs/superpowers/plans/2026-10-21-language-association-retention.md), [`reports/survival_method_reassessment.md` section 7](../survival_method_reassessment.md#7-selected-next-study-language-association-retention-d-035). Sources: [`reports/research_sources.md`](../research_sources.md). Decisions: D-035 to D-038 in [`reports/decisions_and_anomalies.md`](../decisions_and_anomalies.md).

Code (`scripts/g0/`, each with tests in `scripts/g0/tests/` or `tests_torch/`): `phase1_association_inventory.py`, `phase1_association_metrics.py`, `phase1_association_collect.py` (the sink behind `--association-profiles` of `phase1_run.py`), `phase1_association_analysis.py`, `phase1_association_crosscheck.py`, `phase1_association_robustness.py`; plain checks `check_association_headline_independent.py` and `check_association_retained_share.py`; launchers `run_phase1.sbatch` and `run_association_analysis.sbatch`.

Large artifacts (not in git): the 2.76 GB sidecar and the full per-feature tables are on Ada under `/home2/karma/duan-verify/out/phase1/association-full`; their Hugging Face upload is in progress (see `STATUS.md`).
