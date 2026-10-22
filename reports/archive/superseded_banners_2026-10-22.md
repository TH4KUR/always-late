# Superseded banners (2026-10-22)

Paragraphs removed from the front-page documents `README.md`, `PLAN.md` and `PROGRESS.md` because they described a state that no longer holds (the D-035 study as the "selected next study", the first functional-study job as "queued", an old "next work" list) or were verbatim copies of the D-035 result paragraph that stays in `README.md`. They are copied VERBATIM, each inside a fenced block (so that the text is exact and its old relative links cannot rot), with the file and the position they came from and the reason; nothing else in those files was changed apart from the short current-state block that replaced them. Earlier versions of the three files are in the git history before the commit that introduced this archive (plan Task 9, step 2).

## README.md, paragraph 1 of the file (counting from the title)

*Reason:* the D-035 study is finished; the current study is the functional validation.

```text
**Selected next study, 2026-10-21 (D-035):** check whether the original features retain their language association across all 27 languages. [Implementation plan](docs/superpowers/plans/2026-10-21-language-association-retention.md) and [measurement design](reports/survival_method_reassessment.md#7-selected-next-study-language-association-retention-d-035) specify fixed cohorts, full pre/post response profiles, peak/specificity changes, controls and paired uncertainty across mask INT8/7/6/5/4. The plan was [verified](reports/association_retention/plan_verification.md) on 2026-10-21 (D-036: our code and result statements correct, prior art checked, no novelty claimed); implementation is under way ([status](reports/association_retention/STATUS.md)), nothing measured yet and no GPU job queued; wider semantic/causal claims remain outside this step. This supersedes older notices that no specific next measurement question has been selected.
```

## PLAN.md, paragraph 1 of the file (counting from the title)

*Reason:* replaced by the current-state block (this paragraph said only that the functional study follows its plan).

```text
**Current work, 2026-10-21 (D-040):** the functional-validation study (Q16/D-039) follows [its plan](docs/superpowers/plans/2026-10-21-functional-validation.md); live state in [`reports/functional_validation/STATUS.md`](reports/functional_validation/STATUS.md).
```

## PLAN.md, paragraph 2 of the file (counting from the title)

*Reason:* the direction it announced is now the functional-validation design and plan.

```text
**Next work selected, 2026-10-21 (Q16, D-039):** analyse existing findings more closely; assess SAE fidelity; test whether original language-associated features make language-targeted contributions to multilingual tasks using activation patching/complementary disruptive interventions across FP and INT8/7/6/5/4; prepare figures/tables. [Direction, comparison rules and source-grounded controls](reports/association_retention/next_steps.md#selected-functional-study-direction-d-039). C is selected, not merely conditional on spare time. Detailed task/language coverage, feature groups, doses, held-out evaluation and pilot cost precede implementation. The completed D-035 study is preserved; no new job or schedule is started by recording this answer. This supersedes older “no follow-up selected” notices below.
```

## PLAN.md, paragraph 5 of the file (counting from the title)

*Reason:* the D-035 study is finished.

```text
**Selected next study (D-035, 2026-10-21):** test whether the original features retain their language association across all 27 languages under mask INT8/7/6/5/4. The user requested a documented plan: [implementation tasks, tests and run gates](docs/superpowers/plans/2026-10-21-language-association-retention.md), [measurement design and interpretation](reports/survival_method_reassessment.md#7-selected-next-study-language-association-retention-d-035). Freeze the original cohort/labels, compare full firing/mean profiles, retain inactive/unstable cases, and estimate paired sentence-level uncertainty. Start with the artifact inventory; collect missing sidecars only if needed. **Verified 2026-10-21 (D-036):** [the plan verification](reports/association_retention/plan_verification.md) rechecked the plan against our code and saved results (correct), searched prior art (the study was not found; not a novelty certificate) and added technical amendments, now in the plan's "Verification amendments" section (smoke masks differ from the full run, storage gate, GPU-memory margin, dev ties, FP baseline ceiling 6,663 of 8,086). Implementation is under way through [the status file](reports/association_retention/STATUS.md); new measurements and GPU jobs are pending their gates. Wider SAE-quality, functional and Q15 experiments remain separate.
```

## PLAN.md, paragraph 6 of the file (counting from the title)

*Reason:* verbatim copy of the paragraph that stays in README.md.

```text
**Result of the D-035 study, 2026-10-21:** the run (Slurm 31640) and its analyses (31807, 31808; reproduced byte for byte by 31919, 31920) are finished. Of the 8,086 original selective features, 6,663 (82.4 %) reproduce their dev language peak on devtest in full precision; among those, 98.1 / 97.5 / 94.7 / 89.1 / 70.8 % keep their original language as the unique, supported peak at INT8 / INT7 / INT6 / INT5 / INT4 (numerical floor INT16/INT12: 99.2 %). Features are lost by the peak moving to another language, almost never by going silent; features that keep their peak show no visible change of target share or margin; the correlation criterion of the earlier survival analysis disagrees with association retention in both directions. Exploratory on already examined data; nothing is shown about function or meaning. [Interpretation and limits](reports/association_retention/interpretation.md), [generated report](reports/association_retention/results/eq/association_retention.md), [status](reports/association_retention/STATUS.md).
```

## PROGRESS.md, paragraph 1 of the file (counting from the title)

*Reason:* stale: it announced the first smoke job as queued.

```text
**Functional-validation study, 2026-10-21 (D-040):** design, plan, prior-art check and the first code (task layer, intervention library, frozen groups and controls, T-LM engine) are committed; the first GPU job (smoke, Slurm 33771) is queued; a 40-minute recurring job continues the work. [Status](reports/functional_validation/STATUS.md), [design](docs/superpowers/specs/2026-10-21-functional-validation-design.md), [plan](docs/superpowers/plans/2026-10-21-functional-validation.md), [handoff 023](reports/handoffs/023-2026-10-21-functional-study-design-and-code.md). Q17 asks you to confirm the design defaults (not blocking).
```

## PROGRESS.md, paragraph 2 of the file (counting from the title)

*Reason:* the selection it recorded is carried by D-039, the design and the status file.

```text
**Q16 applied, 2026-10-21 (D-039):** selected closer existing-data analyses, SAE fidelity, cross-precision multilingual functional interventions, and figures/tables (A/B/C/F). [Selected direction and controls](reports/association_retention/next_steps.md#selected-functional-study-direction-d-039); [handoff 022](reports/handoffs/022-2026-10-21-q16-functional-validation-selected.md). D-035 is complete; the next work is the bounded follow-up design/pilot, not a restart of its old collection plan. Functional comparisons span FP and INT8/7/6/5/4 and must distinguish language association, task contribution and output-language steering. No new implementation/job or scheduling was performed in this documentation step. Earlier notices below describing active D-035 implementation/cadence are historical; the schedule remains ended.
```

## PROGRESS.md, paragraph 3 of the file (counting from the title)

*Reason:* the D-035 study is finished.

```text
**Selected next study, 2026-10-21 (D-035):** the user chose the 27-language association-retention check and requested its plan. [Plan and implementation gates](docs/superpowers/plans/2026-10-21-language-association-retention.md); [measurement design](reports/survival_method_reassessment.md#7-selected-next-study-language-association-retention-d-035). Documentation is prepared; **the plan was verified on 2026-10-21** (code and result statements correct, prior art checked, register corrected: [verification](reports/association_retention/plan_verification.md), D-036). Implementation starts from [Claude's status file](reports/association_retention/STATUS.md), run by a 20-minute recurring job; no GPU job has been queued and none may be before the status file's gates pass. Latest handoffs: [014](reports/handoffs/014-2026-10-21-language-association-plan.md) (Codex), [015](reports/handoffs/015-2026-10-21-plan-verified-cron-set.md) (Claude). This selects bounded M0/M1 work, not all Q13-Q15 follow-ups.
```

## PROGRESS.md, paragraph 4 of the file (counting from the title)

*Reason:* verbatim copy of the paragraph that stays in README.md.

```text
**Result of the D-035 study, 2026-10-21:** the run (Slurm 31640) and its analyses (31807, 31808; reproduced byte for byte by 31919, 31920) are finished. Of the 8,086 original selective features, 6,663 (82.4 %) reproduce their dev language peak on devtest in full precision; among those, 98.1 / 97.5 / 94.7 / 89.1 / 70.8 % keep their original language as the unique, supported peak at INT8 / INT7 / INT6 / INT5 / INT4 (numerical floor INT16/INT12: 99.2 %). Features are lost by the peak moving to another language, almost never by going silent; features that keep their peak show no visible change of target share or margin; the correlation criterion of the earlier survival analysis disagrees with association retention in both directions. Exploratory on already examined data; nothing is shown about function or meaning. [Interpretation and limits](reports/association_retention/interpretation.md), [generated report](reports/association_retention/results/eq/association_retention.md), [status](reports/association_retention/STATUS.md).
```

## HANDOFF.md, the banner line (superseded 2026-10-22 15:45 by the pointer to handoff 027)

*Reason:* the line described the state of 2026-10-22 02:58 (pilot jobs ending, evaluation not started, cron cancelled) and a 07:17 cron update; the state is in STATUS.md and handoff 027. Verbatim:

```text
> **Latest handoff: [026 -- loss-task evaluation ladder done, calibration running, cron active](reports/handoffs/026-2026-10-22-eval-ladder-done-calibration-running.md).** Earlier: [025](reports/handoffs/025-2026-10-22-chat-end-functional-study-midrun.md) (the work loop, history). Pilot ladder of the cross-precision functional study (INT16 = INT12 = FP, INT8 and INT7 preserved, INT4 amplified), pilot jobs 33788/33789 ending at about 02:58, evaluation not started, cron cancelled, the work loop is Appendix A. **Update 2026-10-22 07:17: the 40-minute cron job was recreated at the user's request** (entries and recipe: [`reports/functional_validation/cron_prompt.md`](reports/functional_validation/cron_prompt.md); current state: [`reports/functional_validation/STATUS.md`](reports/functional_validation/STATUS.md)). Earlier: [022](reports/handoffs/022-2026-10-21-q16-functional-validation-selected.md) (Q16 selected), 023 and 024 (design, code, pilot).
```

## reports/scheduled_routine.md, paragraphs 1 to 7 after the title (superseded 2026-10-22 19:12 by the current-state block)

*Reason:* seven stacked update paragraphs of 2026-10-21 (the Q16 selection, the D-035 "selected next study", the Q13 to Q15 answers, the "no periodic job / 20-minute recurring job" statements, the Q3/Q4 update, the research policy, the scope update) described states that have been superseded (the D-035 study is complete, the functional study runs under a 40-minute job); the operative sentences (scope, research policy, selected direction) are restated in the new block. Verbatim, in the original order:

### Paragraph 1

```text
**Q16 update, 2026-10-21 (D-039):** A/B/C/F are selected, including multilingual functional interventions across FP and INT8/7/6/5/4. Use [the selected direction](association_retention/next_steps.md#selected-functional-study-direction-d-039) and [handoff 022](handoffs/022-2026-10-21-q16-functional-validation-selected.md). Detailed experiment design/source checks and a measured pilot precede large new runs. This records the user's research choice; it does not restart the ended schedule or execute the old D-035 NEXT STEP. D-035 is complete.
```

### Paragraph 2

```text
**Selected next study, 2026-10-21 (D-035):** use [the language-association plan](../docs/superpowers/plans/2026-10-21-language-association-retention.md) and [design](survival_method_reassessment.md#7-selected-next-study-language-association-retention-d-035). The user selected this question and requested planning; no implementation or job was executed. Next technical work begins with M0 and frozen inputs/cohorts, then the optional collector and smoke gates. Full collection is needed only if artifacts are insufficient. Broader Q13-Q15 work remains separate. This supersedes older statements below that no concrete measurement question/plan has been selected; it does not launch the historical jobs or require numerical reruns for this documentation-only step.
```

### Paragraph 3

```text
**Q13-Q15 update, 2026-10-21 (D-032/D-033/D-034):** their agreed answers are applied. Use [the source register and saved-result synthesis](research_sources.md) alongside the method reassessment. Q13 adopts both associational contrasts and a narrower magnitude question; Q14 keeps overall and extra-selective disparities distinct without rerunning completed tests; Q15 adopts an expanded confidence/diffuse-activity investigation against representation-scale/SAE alternatives. Do not launch the old N12 job or treat its 1-2 h estimate as the revised design. First inventory artifacts and specify the prospective measurement plan for mask INT8/7/6/5/4. Other unanswered review items remain separate.
```

### Paragraph 4

```text
**Ended 2026-10-21 19:20 (user): the scheduled job was deleted; no periodic job runs; see [handoff 021](handoffs/021-2026-10-21-chat-end-study-complete-awaiting-review.md).** **Recurring-job update, 2026-10-21 (user, D-036):** the user asked for a recurring job every 20 minutes to continue the selected D-035 study ("setup a recurring job for every 20 minutes to continue the work"). This supersedes the "no cron" statements below for that work. A session-only cron job (`7,27,47 * * * *`, auto-expires after 7 days; prompt copied in [cron_prompt.md](association_retention/cron_prompt.md)) does one bounded step per firing from Claude's [status file](association_retention/STATUS.md), writes a handoff (numbers after 015; Codex's 010-014 stay) when a task or sub-task completes, queues no GPU job before the status file's gates pass, and edits the shared documents it needs to keep current (the user ended communication with Codex on 2026-10-21, D-037). The user also instructed (same day) that a plan be verified for factual correctness, prior art and exaggeration before implementation ([done for D-035](association_retention/plan_verification.md)) and allowed Claude to commit and push Codex's work after checking its alignment.
```

### Paragraph 5

```text
**Q3/Q4 update, 2026-10-21 (D-030/D-031):** reassess survival detection; include INT5/INT4 in the study ladder. Use [the accepted-source and candidate-check record](survival_method_reassessment.md). The next method step is artifact/schema feasibility before selecting new measurements. Existing correlation outputs are not proof of retained language identity or causal function. No particular replacement metric or new GPU experiment is selected by these decisions.
```

### Paragraph 6

```text
**Research policy update, 2026-10-21 (Q2, D-029):** publishability is the main objective. Questions, proposal and direction may change for documented reasons grounded in current results and analysis. Before committing to a new direction, check the specific contribution against the closest primary literature and record what is new and how it will be tested. Preserve prior outcomes and distinguish exploration motivated by seen results from prospective confirmation. Historical OSF freeze status is unverified, not a blocker to revisions; do not infer pre-registration from the proposal. This supersedes older fixed-proposal requirements, while the mask-only/no-Duan-comparison scope (D-028) remains in force. Q3/Q4 and Q13-Q15 reporting/investigation directions are recorded; exact replacement measurements and a new run remain to be designed.
```

### Paragraph 7

```text
**Scope update, 2026-10-21 (D-028):** the study now uses `mask` only. Do not launch or extend `none` work, including sensitivity analyses, or run further Duan numerical comparisons. Historical commands and cached reproduction tooling remain available for provenance; their existence does not put retired work back in scope. Q5 and Q9 are closed without execution. Q15's expanded question is adopted under D-034; its exact run remains to be designed. Other unanswered runs remain separate and must be mask-only. This overrides older two-regime and Duan-baseline requirements below.
```

## HANDOFF.md, the banner line of 2026-10-22 (superseded 2026-10-23 by the pointer to handoff 028)

*Reason:* the line named handoff 027 as the latest and described the state of 2026-10-22 15:45; the state is in STATUS.md and handoff 028. Verbatim:

```text
> **Latest handoff: [027 -- loss-task evaluation ladder complete (INT8 to INT4), F4 INT4 queued, Belebele pilot running](reports/handoffs/027-2026-10-22-eval-ladder-complete-f4-queued.md).** Earlier: [026](reports/handoffs/026-2026-10-22-eval-ladder-done-calibration-running.md), [025](reports/handoffs/025-2026-10-22-chat-end-functional-study-midrun.md) (the work loop, history). The 40-minute cron job is active (entries and recipe: [`reports/functional_validation/cron_prompt.md`](reports/functional_validation/cron_prompt.md); current state: [`reports/functional_validation/STATUS.md`](reports/functional_validation/STATUS.md)). Earlier: [022](reports/handoffs/022-2026-10-21-q16-functional-validation-selected.md) (Q16 selected), 023 and 024 (design, code, pilot).
```
