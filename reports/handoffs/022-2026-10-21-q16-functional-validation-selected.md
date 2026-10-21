# Handoff 022: Q16 selected; multilingual functional validation across precision (2026-10-21)

## User answer and applied decision

The user selected A, B, C and F, and explicitly proposed testing whether language-associated SAE features help multilingual tasks through activation patching or other mechanistic interventions, then comparing effects across unquantized Gemma-2-2B and INT8/7/6/5/4. The user requested that Q16 be answered accordingly. Recorded D-039 and marked Q16 APPLIED for the decision/documentation; implementation is pending.

In plain language: closer analysis of existing results; check SAE fidelity; test language-targeted functional contributions across precision; prepare figures/tables. Functional study C is selected, no longer merely “if time allows.” Random model-weight-noise study D and additional all-token intervals E are unselected for now. This does not remove sham/matched-feature controls needed inside the functional study.

## Canonical record and next work

Read [Q16](../../DECISIONS_FOR_REVIEW.md), [selected functional-study direction](../association_retention/next_steps.md#selected-functional-study-direction-d-039), D-039 in [the register](../decisions_and_anomalies.md), and method references M06/M07 plus existing P02/P08 in [research_sources.md](../research_sources.md).

The next step is a bounded, source-grounded experiment design with explicit task/language coverage, feature/group selection, clean/corrupt prompt pairs and token alignment, intervention doses, controls, metrics, uncertainty and measured pilot cost. Start with the already-available data and a small FP task/intervention validity pilot; compare the same defined experiment across all six precisions. Existing-data/fidelity analyses retain the 27-language panel; the expensive task study's language coverage must be stated and justified rather than silently implying all 27 were evaluated.

Use fixed original feature identities for primary cross-precision curves. Compare each precision's intervention against its own appropriate unpatched baseline on matched task items, then compare those effects with FP. Native within-model clean-donor patching and FP-donor rescue of a quantized recipient answer different questions and must be reported separately. Show target-language versus other-language and matched-control effects, task correctness separately from output-language compliance, and calibrated dose-response/actual perturbation norms. Preserve the recipient reconstruction-error component when editing SAE contributions. Do not assume identical numeric boosts mean equally strong interventions across precisions.

Language association, language-specific task contribution and semantic meaning are distinct. Generic disruption/gain alone does not establish language-specific function; a single null ablation does not disprove a redundant or task-dependent role. Establish measurable functional effects in FP before interpreting their change; retain null/negative results and check baseline headroom/floors. A claim of “similar” effects needs predefined practical tolerances, not merely nonsignificance. Separate calibration from held-out task evaluation; existing FLORES analyses stay exploratory. The exact cross-precision contribution is not novelty-certified; language-feature ablation/steering already exists in P02.

## D-035 state and other decisions

D-035 remains complete: run 31640, analysis/rerun pairs 31807/31808 and 31919/31920; historical mask reproduction 47/47 identical, as recorded in handoff 021. Its outcomes are association measurements, not evidence of preserved function. Its shifted percentile intervals require scrutiny before inferential reuse. Large sentence shards remain on Ada; compact HF upload was verified and the dataset stays private for the user's later publication. These are previously recorded states, not new job/network checks in this documentation step.

Remaining unanswered numbered questions: Q6, Q7, Q8, Q11 and Q12. Q13-Q15 decisions remain in force; selecting B/C does not automatically solve their full confidence/causal questions. The own-statistics licence, sentence-shard backup and historical OSF fact remain unresolved. Mask-only/no new Duan numerical comparisons; preserve old outcomes, vendor files and team-written paper prose.

## Actions and verification in this step

Updated review/register, follow-up options/direction, source register, plan/progress pointers, routine, reassessment and this handoff. M06's official ICLR 2024 abstract/venue were checked; M07's primary preprint sections 2.5-4 were read. No exhaustive new novelty search, code, new task measurements, GPU job, restarted automation, inter-agent message, commit or push. No tests/models were rerun for documentation-only edits. Verify local links/anchors, Q16 status/D-039 table continuity, scope agreement and `git diff --check` before reporting completion. Check the current working tree before any future edits; communication remains retired.
