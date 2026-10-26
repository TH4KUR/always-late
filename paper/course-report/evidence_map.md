# Evidence map for the course-report draft

All paths below are relative to the repository root. The report introduces no
new model runs or hypothesis tests. Figures only summarize committed CSVs.
Rounded manuscript numbers must retain the source's population, estimand and
analysis status. Sources are the current reports, not the retired pipeline.

| Manuscript block | Source | Required qualification |
| --- | --- | --- |
| Fixed setting, groups, checks and freezes | `docs/superpowers/specs/2026-10-21-functional-validation-design.md`; `reports/functional_validation/evaluation_freeze_lm.json`; `evaluation_freeze_mc.json`; `groups_manifest.json` | Use implemented/frozen settings where early design estimates differ. FP is bfloat16 model inference, not an ideal exact-real baseline. |
| Selectivity discovery rule | `scripts/g0/phase1_run.py:166`; `scripts/g0/phase1_lib.py:180`; `reports/phase1_results.md` | Null threshold and discovery stability are distinct from devtest association establishment. |
| Language roster and descriptive tiers | `reports/g0/phase1_languages.json` | Nine languages per tier; taxonomy does not identify training-data causality. |
| BOS replication and pipeline retirement | `reports/audit_our_vs_duan.md`; `reports/progress_log.md` (16 October post-mortem) | Same-window BOS diagnostic changes regime and potentially active population; no use of retired outputs as current results. |
| Original tests and magnitude/tier checks | `reports/phase1_summary.md`; `reports/phase1_results.md` | No verified external OSF registration; stricter thresholds are post hoc, adjustment is correlational. |
| 8,086 selected, 6,663 established; precision retention | `reports/association_retention/interpretation.md`; `results/eq/analysis.json` | Retention among established differs from retention in the fixed original cohort and from pooled all-active correlation survival. |
| Margin prediction and function comparison | `reports/association_retention/results/within-language/report.md`; `results/margin-function/report.md` | Same-data descriptive prediction and stratified samples; margin is not a causal explanation. |
| Group effect and retention ratios | `reports/functional_validation/results/eval/report.md`; `retention.csv`; `figures/eval/table_effects.csv` | 300 sentences per language, mean of per-language ratios; 2.14 is not the ratio of macro effects (1.33). |
| Control mismatches | `reports/functional_validation/groups_matching.md`; `scripts/g0/phase1_func_groups.py` | Sampling among the three nearest unused candidates; five main K64 draws; pool overlap and Yoruba/Lao/Thai mismatches remain. |
| Rescue fractions | `reports/functional_validation/figures/rescue/table_rescue.csv` | INT6 uses 12 qualifying languages; nonadditive patches are not an exact causal partition. |
| Lost versus retained matched effects | `reports/functional_validation/results/f4/INT4/report.md`; `INT5/report.md`; `INT6/report.md` | Small observed contrasts do not prove universal equivalence. |
| Frozen Belebele statements | `reports/functional_validation/results/mc_eval/statements.md`; `accuracy.csv` | Only confirmatory family; S0 not met, retention statements undefined. Primary endpoint is four-way correct-option log probability, not accuracy alone. |
| Whole-group and positive-control results | `reports/functional_validation/results/mc_ksens/report.md`; `results/mc_posctrl/summary.md` | Exploratory; random sets affect the task, but do not establish own-group irrelevance. |
| Prompt check | `reports/functional_validation/promptcheck/report.md`; `languages.csv` | 250 calibration items; distinct from 385 held-out evaluation items; one alternative prompt. |
| Individual-effect table and threshold | `reports/functional_validation/results/f5/report.md`; `f5_tables.csv`; `scripts/g0/phase1_func_f5_metrics.py` | 1,487 sampled features; numerical floor 0.9489 is not semantic truth. |
| Direction-control figure | `reports/functional_validation/figures/f5/ladder_control_table.csv` | Same 404 features; quartile bars are not confidence intervals; generic drift does not prove exact equivalence. |
| Fixed-amplitude/context sensitivities | `reports/functional_validation/results/f5_sensitivity/` | Sensitivities are exploratory; no new causal mechanism is inferred. |
| Gain suppression figure and hypothesis failures | `reports/functional_validation/gain_study/results/curves.csv`; `report.md`; `calibration/calibration.md` | Plot medians of stored per-language net effects; main strict precision rule failed. |
| Fine-gain figure | `reports/functional_validation/gain_study/extension_c/bands.csv`; `report.md`; `analysis.json` | Use saved interval endpoints exactly; primary ratios 1.1/1.2, 97.5% intervals, only 1.1 rule met. |
| State-defined and optimum/shrinkage checks | `reports/functional_validation/gain_study/state_defined/results/report.md`; `extension_d/report.md` | Exploratory, conditional findings; no compensation mechanism and no feature-identity claim. |
| Second-model check (Llama-3.2-1B): design, predictions, verdict function | `docs/superpowers/plans/2026-10-26-llama-consistency-check.md` sections 3, 6, 7; `reports/decisions_and_anomalies.md` D-087 | Written before any Llama number; predictions P1 to P3b and the verdict function are copied in the report's Table 2 and Section 8. |
| Second-model check: results table, retention, diagnostics | `reports/functional_validation/llama/report.md`; `analysis.json`; `reports/decisions_and_anomalies.md` D-090 | Exploratory; model, SAE architecture and recipe, depth and tokenizer all differ from Gemma; the verdict is a statement about this model and SAE, not about other models. |
| Second-model check: control draws, outlier draws, median of the retention ratios | `reports/functional_validation/llama_robust/report.md`; `robust.json`; `languages.csv` | Post hoc; labelled so; cannot change the pre-registered verdict. The perturbation norms at FP and INT4 are in `llama/analysis.json` (`diagnostics`). |
| Appendix rules table (H-G1 to H-G6, H-Gc1 to H-Gc3, H-Gb1 to H-Gb4b) | `reports/functional_validation/gain_study/results/report.md`; `extension_c/report.md`; `state_defined/results/report.md` | Rules written in the design documents before the data; outcomes copied from the committed verdict tables. |
| Appendix result inventory | `reports/functional_validation/RESULTS_SUMMARY.md` claims 1 to 17 | Same numbers as the sections above; status words as in the summary. |
| Software, hardware and cost statement | `HANDOFF.md` section 5; `reports/functional_validation/STATUS.md` (job runtimes) | Versions of the Ada environment; per-state GPU hours are rounded. |
| Reproduction counts and artifacts | `reports/phase1_reproduction.md`; `reports/functional_validation/REPRODUCE.md`; `reports/handoffs/039-2026-10-26-task17-prompt-check-done-all-queues-empty.md` | Regeneration of analysis outputs is not independent repetition of inference; newer inputs not all archived to HF. |

## Citation provenance

The six Anthology BibTeX records were downloaded directly from the cited
Anthology pages on 26 October 2026. Duan (2606.03002), Borobia et al.
(2603.25325), and Hoang et al. (2607.24645) were checked against their primary
arXiv pages in this chat and are identified as preprints. They are linked using
their genuine arXiv-issued DOI, rather than being mislabelled as proceedings
papers. The manuscript does not make a first-ever or exhaustive novelty claim.

Added 2026-10-26 (second revision): Gemma 2 (arXiv 2408.00118), JumpReLU SAEs (2407.14435), Cunningham et al. (2309.08600), Gao et al. TopK SAEs (2406.04093), the NLLB Team paper for FLORES-200 (2207.04672) and The Llama 3 Herd of Models (2407.21783) were checked against their arXiv abstract pages on 26 October 2026 (titles, author order, years); long author lists are truncated with `and others`, which the official bibliography style supports.
