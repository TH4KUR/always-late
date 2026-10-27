# Documentation map, and the trail that is never deleted

Written 2026-10-23 23:48 IST (Claude), after the user's instruction of the same day: **"ensure that you dont remove progress tracking (so that we know at the end of the project about all the directions and things we explored and thought of)".**

## The rule

**Progress tracking is never removed, shortened or overwritten**: the decision records, the dated logs, the handoffs, the plans and designs with their predictions, the question files with their options and answers, the search logs, the audits and the archive. Cleaning up documentation means only: (a) moving a stale block VERBATIM into `reports/archive/` with a pointer; (b) adding a pointer or an 'Update <date>' line; (c) removing byte-identical generated copies (checked by hash) or text that exists unchanged elsewhere. A direction that was dropped, not selected or judged redundant stays recorded, with its reason. The rule is also written into [`functional_validation/cron_prompt.md`](functional_validation/cron_prompt.md), [`scheduled_routine.md`](scheduled_routine.md) and [`../HANDOFF.md`](../HANDOFF.md).

## 1. Start here (shortened 2026-10-27; the previous 22-row table is kept verbatim in [`archive/doc_cleanup_2026-10-27/reports__INDEX.md.txt`](archive/doc_cleanup_2026-10-27/reports__INDEX.md.txt))

| file | what it is |
|---|---|
| [`PROJECT_RESULTS.md`](PROJECT_RESULTS.md) | **All results from the start in plain words:** why they are not conclusive, what the null results tell us, the second-model and downstream-task checks, per-language tables, guide to every data file |
| [`CURRENT_STATUS.md`](CURRENT_STATUS.md) | **Current state page:** evidence state, the HOLD, open choice Q29, deliverables |
| [`../README.md`](../README.md), [`../HANDOFF.md`](../HANDOFF.md) | orientation; rules, environment and rerun notes for whoever continues |
| [`../docs/CODEBASE.md`](../docs/CODEBASE.md) | code ownership map |
| [`functional_validation/RESULTS_SUMMARY.md`](functional_validation/RESULTS_SUMMARY.md), [`FIGURE_INDEX.md`](functional_validation/FIGURE_INDEX.md), [`REPRODUCE.md`](functional_validation/REPRODUCE.md) | the functional study's claim map (evidence and limit per claim), every finished figure, one command that reruns the analyses |
| [`functional_validation/gain_study/README.md`](functional_validation/gain_study/README.md) | Study G (the user's major study, 2026-10-24), full record |
| [`intervention_transfer/results/`](intervention_transfer/results/main/README.md) | main, external and recovery results of the transfer study (D-094 to D-099) |
| [`handoffs/`](handoffs/) | numbered handoffs; newest at cleanup: 045 (scientific), 046 (modular cleanup), 047 (this documentation cleanup); 040 = the Gemma-2-9B hand-over to another team |
| [`novelty_assessment_2026-10-26.md`](novelty_assessment_2026-10-26.md), [`publication_reassessment_2026-10-26/README.md`](publication_reassessment_2026-10-26/README.md) | dated novelty verdict; Codex's post-D-094 reassessment (historical) |
| [`wandb_schema_v2.md`](wandb_schema_v2.md) | wandb schema v2 and the post-hoc upload of the finished results (D-091) |
| [`../DECISIONS_FOR_REVIEW.md`](../DECISIONS_FOR_REVIEW.md) | the questions for the user (Q1 to Q29) with options, defaults and answers |
| [`../paper/course-report/README.md`](../paper/course-report/README.md) | course report (31 October) |

## 2. The trail (progress tracking; keep forever)

| kind | where |
|---|---|
| decisions and anomalies (append-only; D-, A-, V- ids) | [`decisions_and_anomalies.md`](decisions_and_anomalies.md) |
| questions with the options considered, the defaults and the answers | [`../DECISIONS_FOR_REVIEW.md`](../DECISIONS_FOR_REVIEW.md) |
| dated logs | [`progress_log.md`](progress_log.md); the Log of [`functional_validation/STATUS.md`](functional_validation/STATUS.md) (from 2026-10-23); [`archive/status_log_2026-10-21_to_22.md`](archive/status_log_2026-10-21_to_22.md); [`association_retention/STATUS.md`](association_retention/STATUS.md) |
| handoffs, one per step (001 onward, with dated addenda) | [`handoffs/`](handoffs/) |
| progress and plan pages | [`../PROGRESS.md`](../PROGRESS.md), [`../PLAN.md`](../PLAN.md) (pointers since 2026-10-26; full texts in [`archive/root_pages_2026-10-26/`](archive/root_pages_2026-10-26/)) |
| designs and plans written before the data, with their hypotheses and predictions | [`../docs/superpowers/specs/`](../docs/superpowers/specs/), [`../docs/superpowers/plans/`](../docs/superpowers/plans/) |
| options considered, including those not selected | [`association_retention/next_steps.md`](association_retention/next_steps.md) (options A to F of Q16), [`survival_method_reassessment.md`](survival_method_reassessment.md), [`association_retention/plan_verification.md`](association_retention/plan_verification.md) |
| literature and novelty checks with their search logs | [`research_sources.md`](research_sources.md), [`functional_validation/prior_art.md`](functional_validation/prior_art.md) |
| audits and reviews | [`audit_our_vs_duan.md`](audit_our_vs_duan.md), [`duan_codebase_audit.md`](duan_codebase_audit.md), [`../issues.md`](../issues.md), [`legacy_pipeline/`](legacy_pipeline/) |
| text moved out of living pages, verbatim | [`archive/`](archive/) |
| how the work was run | [`scheduled_routine.md`](scheduled_routine.md), [`functional_validation/cron_prompt.md`](functional_validation/cron_prompt.md), [`association_retention/cron_prompt.md`](association_retention/cron_prompt.md) (ended 2026-10-21); the first and last are pointers since 2026-10-26, full texts in [`archive/root_pages_2026-10-26/`](archive/root_pages_2026-10-26/) |

## 3. Directions explored or considered (a map; if it disagrees with the trail, the trail wins)

| direction | what became of it | where it is recorded |
|---|---|---|
| Reproduce Duan et al. (2026), Phase 3, with their own code on Ada | reproduced within 0.15 pp; the disagreement with our first pipeline was a regime difference (BOS), not a bug; our first pipeline was discarded | D-009 (closure D-NOISE); [`audit_our_vs_duan.md`](audit_our_vs_duan.md), [`duan_codebase_audit.md`](duan_codebase_audit.md), [`../issues.md`](../issues.md) |
| Which regime: BOS kept, BOS masked, no BOS | `mask` only; `none` retired from ongoing work | Q1, D-028 |
| Phase 1: language-selective features (S1 to S8, N1 to N13) | the pre-registered selective-versus-control tests are null after correction in the primary regime; a resource-tier effect on survival; magnitude accounts for the selectivity penalty | D-016, D-026; [`phase1_summary.md`](phase1_summary.md), [`phase1_results.md`](phase1_results.md) |
| Magnitude as confounder or mediator; which analysis is the confirmatory RQ2 test; a GPU diagnostic of why low-resource languages have a lower activation scale | both matching comparisons reported as associations; the GPU diagnostic was not run (its job remained to be designed) | Q13 to Q15, D-032 to D-034 |
| Faithful no-BOS regime with their window shape; optional replication diagnostics of their paper | closed without execution | Q5, Q9 ([`../PROGRESS.md`](../PROGRESS.md)) |
| Reassess how survival is detected; include INT5 and INT4 | reassessment authorised and carried out as the studies below | Q3, Q4, D-030, D-031; [`survival_method_reassessment.md`](survival_method_reassessment.md) |
| Does a feature keep its original language association across all 27 languages | done 2026-10-21 | D-035, D-036; [`association_retention/interpretation.md`](association_retention/interpretation.md) |
| Follow-ups after that study (Q16): A closer analysis of existing results, B SAE fidelity, C functional interventions across precision, D random-weight-noise controls, E all-token intervals, F figures and tables | A, B, C and F selected; D and E not selected for now | Q16, D-039; [`association_retention/next_steps.md`](association_retention/next_steps.md) |
| A: closer analyses of the association loss | A1 rank, A2 margin (the loss is mostly a margin effect), A3 margin against function (it hardly predicts function) | D-066, D-067; [`association_retention/results/`](association_retention/results/) |
| C: functional validation F1 to F4, the rescue F3 | group effect on next-token loss, retention by precision, where the damage sits, association status against function | D-040 onward (D-041, D-048 to D-051, D-058); [`functional_validation/RESULTS_SUMMARY.md`](functional_validation/RESULTS_SUMMARY.md) |
| Belebele as the only confirmatory family, with a K = all sensitivity | frozen before scoring; the dependence is not found on the task | D-044, D-047, D-050, D-062 |
| F5: does a feature do the same thing after quantization; its controls | through INT8 within noise; the loss of similarity is generic (swapped direction, fixed amplitudes, number of contexts); a mean-ablation control was judged redundant | D-053 to D-064; design [`../docs/superpowers/specs/2026-10-23-feature-function-equivalence-design.md`](../docs/superpowers/specs/2026-10-23-feature-function-equivalence-design.md) |
| Does the model support all 27 languages; are the results meaningful for those it does not | the loss-task conclusions hold on the 18 languages with measurable ability and on the 9 without; statements are made per support status; a positive control of the Belebele edit path shows that it registers effects in proportion to the set size (plan Task 17, D-069) | D-068, D-069 (2026-10-24); [`functional_validation/belebele_calibration.md`](functional_validation/belebele_calibration.md) |
| Study G (user, 2026-10-24): scale each language's selective features down (3 ratios) and up (3 ratios), per language, at FP and INT8 to INT4: is the effect monotone in the ratio, monotone across precisions, consistent across languages | complete, including extensions B/C/D; see final status and result tables | [`functional_validation/gain_study/README.md`](functional_validation/gain_study/README.md) |
| Prior-art and novelty checks, the scoop risk | rechecked after D-094 on 26 October; broad claims already covered; remaining candidate unestablished; [current verdict](novelty_assessment_2026-10-26.md) | D-040, D-065, Q18; [`functional_validation/prior_art.md`](functional_validation/prior_art.md) |
| The two parallel branches of a teammate; a second model | historical branch/access alternatives retained in the decision trail; Llama access was resolved and its exploratory check is complete (D-090); 9B handled separately (D-092) | D-052, Q22, Q25, Q26 |
| Llama-3.2-1B consistency check (user, 2026-10-26) | designed and run (D-087); interim FP misses P1 (21 of 27); **relabelled exploratory cross-architecture check (D-089)** because model, SAE type (TopK), SAE recipe and depth are confounded; **finished 2026-10-26 08:38: pre-registered verdict 'not consistent' (D-090; results `functional_validation/llama/` and `llama_robust/`, RESULTS_SUMMARY claim 17)** | D-087, D-089, D-090; [`../docs/superpowers/plans/2026-10-26-llama-consistency-check.md`](../docs/superpowers/plans/2026-10-26-llama-consistency-check.md) |
| Gemma-2-9B confirmation (user, 2026-10-26): same architecture, same Gemma Scope SAE, matched relative depth | replaces Llama as the second model; run by another team on a separate cluster, branch `gemma9b-confirmation`, merged later | D-089; [`handoffs/040-2026-10-26-gemma9b-confirmation-external-cluster.md`](handoffs/040-2026-10-26-gemma9b-confirmation-external-cluster.md) |
| Intervention transfer under quantization (Codex, 2026-10-26): crossed prefix FP/Q x suffix FP/Q decomposition of a fixed feature intervention, held-out diagnostic, branches A and B | main/external matrices and CPU recovery complete (D-094–D-099); original utility main not met / external met; precision audit is separate (D-100); novelty not established; [current status](CURRENT_STATUS.md) | D-088, Q28; [`intervention_transfer/verification_2026-10-26.md`](intervention_transfer/verification_2026-10-26.md) Codex's [memo](publication_direction_2026-10-26.md), [novelty register](publication_strategy_2026-10-26/future_work_and_novelty.md), [`../RESEARCH_DIRECTION.md`](../RESEARCH_DIRECTION.md), [design](../docs/superpowers/specs/2026-10-26-intervention-transfer-research-design.md), [plan](../docs/superpowers/plans/2026-10-26-intervention-transfer-main-and-contingencies.md) |
| Left open or not started | extensions beyond Phase 1 (Q8); upload of item-level files to the private dataset (Q21); the whole 4,382-feature F5 sample (Q24, default no); the rule S1 to S3 (Q23); activation quantization and other models are not studied | [`../DECISIONS_FOR_REVIEW.md`](../DECISIONS_FOR_REVIEW.md) |

## 4. Evidence (generated; reproducible)

[`phase1/`](phase1/) (Phase 1 analyses, `mask` and `none`, `eq` and `all`; reproduced in [`phase1_reproduction.md`](phase1_reproduction.md)), [`g0/`](g0/) (the baseline, the R-runs and the smoke runs of the Duan code), [`association_retention/results/`](association_retention/results/), [`functional_validation/results/`](functional_validation/results/) and [`functional_validation/figures/`](functional_validation/figures/) (every functional analysis re-runs with [`functional_validation/REPRODUCE.md`](functional_validation/REPRODUCE.md)); sizes and checksums of the files that live only on the cluster: [`functional_validation/artifact_checksums.md`](functional_validation/artifact_checksums.md).

## 5. Reference

[`wandb_schema.md`](wandb_schema.md), [`artifacts.md`](artifacts.md), [`g0/README.md`](g0/README.md), [`g0/hf_dataset_card.md`](g0/hf_dataset_card.md), [`handoffs/TEMPLATE.md`](handoffs/TEMPLATE.md).

## 6. What was moved or removed, and how to restore

- **27 October documentation cleanup (afternoon, requested by the user):** cron jobs stopped; `README.md`, `HANDOFF.md`, `RESEARCH_DIRECTION.md`, `CURRENT_STATUS.md` and section 1 of this index shortened; their full pre-edit text is in [`archive/doc_cleanup_2026-10-27/`](archive/doc_cleanup_2026-10-27/README.md) with checksums; new single results document [`PROJECT_RESULTS.md`](PROJECT_RESULTS.md). No progress record was touched. Handoff 047.

- **27 October modular cleanup:** living summaries now route to [CURRENT_STATUS.md](CURRENT_STATUS.md); code ownership is mapped separately. Full pre-edit bytes of 12 pages are [archived](archive/modular_cleanup_2026-10-27/README.md), with checksums. Historical ledger/log bodies, research protocols, decisions, results and prior archives are retained. Five completed analysis implementations moved into a package, with all original entry points preserved.

- **26 October post-D-094 novelty cleanup:** replaced stale current-state and priority summaries; marked design-stage novelty records as historical. Full pre-edit bytes, including previous uncommitted updates, are in [the cleanup archive](archive/novelty_cleanup_2026-10-26/README.md), with SHA256 manifest. Dated logs, frozen inputs, original plans/specs and result files are preserved.

All of it also stays in git history (`git show <commit>:<path>`).

- **Moved verbatim to `archive/` on 2026-10-23** (living pages keep a pointer): the Log entries of 2026-10-21 and 2026-10-22 of `functional_validation/STATUS.md`; the stale State block of the same file; the stale state sections of `HANDOFF.md`; the stale 'NEXT RUN'S GOAL' block of `PROGRESS.md` and the chat-only paragraphs and steps of `scheduled_routine.md`; the superseded banners of README, PLAN and PROGRESS (2026-10-22).
- **Moved verbatim to `archive/` on 2026-10-25** (the living page keeps a pointer): the update of 2026-10-24 08:04 and the done items 1 to 5 of the NEXT STEP block of `functional_validation/STATUS.md` -> [`archive/status_next_step_items_2026-10-24.md`](archive/status_next_step_items_2026-10-24.md) (8 lines, 12.9 KB; checked: every moved line is in the archive file, only two relative links were rebased, nothing else was removed).
- **Removed on 2026-10-23: 24 generated files (1.7 MB)**, byte-identical (sha256) to the same-named files of `functional_validation/figures/eval/`, which stay: in `figures/pilot/` and `figures/mc_eval/` the figures 7 (Belebele coverage) and 8 (association by rank) in light and dark, SVG and PNG, and the tables `table_coverage` and `table_configurations` (CSV and markdown). Last commit that holds them: `7147f0e`.
- **Moved verbatim to `archive/` on 2026-10-26** (documentation cleanup requested by the user; the living files keep a short current version or a pointer; relative links rebased): the previous `README.md`, `HANDOFF.md`, `PLAN.md`, `PROGRESS.md`, `scheduled_routine.md` and `association_retention/cron_prompt.md` -> [`archive/root_pages_2026-10-26/`](archive/root_pages_2026-10-26/); the stale State block, the stacked NEXT STEP updates of 2026-10-25/26 and the Log entries of 2026-10-23 to 2026-10-25 of `functional_validation/STATUS.md` -> [`archive/status_2026-10-26_cleanup.md`](archive/status_2026-10-26_cleanup.md). Update 2026-10-26 08:40: the NEXT STEP block of 08:30 (replaced when the Llama check finished) -> [`archive/status_next_step_2026-10-26_0830.md`](archive/status_next_step_2026-10-26_0830.md).
- **Not removed, on purpose:** every handoff, log, plan, design, decision, question, audit, source register and result. Nothing in section 2 was touched except by pointers and 'Update' lines.
