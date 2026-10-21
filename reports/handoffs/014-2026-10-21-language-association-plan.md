# Handoff 014: 27-language association-retention plan (2026-10-21)

## User instruction and status

The user selected “Does the feature retain its original language association across all 27 languages?” and requested a plan added to the documentation. Recorded as D-035. Planning is complete; implementation and new measurements have not begun. The user selected this question, not every detailed cutoff/bootstrap default or the wider semantic/functional/SAE-training alternatives.

## Canonical design and implementation plan

- [Measurement design, section 7](../survival_method_reassessment.md#7-selected-next-study-language-association-retention-d-035): quantities, original cohort, baseline reproducibility, status labels, denominators, uncertainty and interpretation.
- [Task-by-task implementation plan](../../docs/superpowers/plans/2026-10-21-language-association-retention.md): exact file ownership/responsibilities, collector interfaces/sidecar schema, test cases, smoke/full-run gates and completion criteria. Checkboxes are future work.
- [Source register](../research_sources.md): existing primary sources, publication status, scope limits and search record. No new literature search or novelty certification was performed in this planning step.

Freeze original mask/eq selective IDs, dev peak labels and saved controls. On matched devtest token positions, collect each condition's own full 27-language firing/mean profiles under the frozen layer-12 16k SAE. Cover INT8/7/6/5/4 plus FP, INT0 and INT16/12. Report peak switches, ties, inactivity, support, target share/margin and full-profile change; a retained peak does not establish unchanged specificity. Keep baseline-unstable cases in the original denominator and show their status explicitly. All-token sensitivity uses the same cohort. No retagging/rematching or reopened Q6 decision.

Uncertainty requires new saved paired sentence counts/sums if unavailable: bootstrap aligned devtest sentence IDs jointly across languages and conditions, preserving the selected masks and cohort. The proposed default is 1,000 draws/seed 0 with pointwise descriptive intervals. Both FLORES splits have already been examined, so this remains exploratory. No causal or semantic-function claim follows from these profiles.

## Next technical step

Begin Task 1's bounded local/cache/known-Ada artifact inventory and frozen provenance manifest. Established schema finding: saved correlations cover all languages, but quantized response profiles and sentence summaries are missing from inspected legacy outputs; per-rung firing/magnitude arrays refer to FP/reference activations. Do not relabel them or claim every possible cache was searched. Only collect new sidecars if the inventory cannot satisfy the contract. The optional collector must preserve historical outputs/default behavior and pass planted broadening/switch/inactivity tests and a real-model smoke before full collection.

Estimate 1-2 calendar days when compute is available, with 2-4 days allowed for revisions; re-estimate after smoke using measured I/O, sparse density and bootstrap CPU cost. Prior mask run: 3 h 37 min, without these sentence sidecars. No new GPU submission, monitoring, training, numerical result, commit or push occurred here.

## Scope, ownership and other decisions

Mask only; no additional none work or Duan numerical comparison; preserve historical data/vendor attribution and paper prose. Q13-Q15 decisions from handoff 013 stay in force, but their broader magnitude/confidence/reconstruction investigations are separate. Q6/Q7/Q8/Q11/Q12 remain unanswered and Q10's earlier follow-through is separate. No new corpus, probe, intervention or independently trained SAE is selected.

Codex retains ownership of the shared plan/review/progress/reassessment documents. Read new shared messages before future edits; Claude's acknowledgment of the Q13-Q15 message remains pending. This planning step assigned no work to Claude and created no new communication message or watcher. Preserve all existing replies and unrelated uncommitted changes.

## Verification

Documentation-only checks: local plan/design/handoff links and anchors, D-035 uniqueness/table placement, task/interface consistency, unchanged code/numerical files and `git diff --check`. No model tests or historical numerical reproduction are needed for this documentation-only step. Future code/result changes must run the plan's relevant suites, mask-only reproduction and summary provenance checks.
