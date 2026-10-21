# Association-retention study: status file (D-035)

Owner: Claude. Plan: [`docs/superpowers/plans/2026-10-21-language-association-retention.md`](../../docs/superpowers/plans/2026-10-21-language-association-retention.md) (Tasks 1-6 and its "Verification amendments"); design: [`reports/survival_method_reassessment.md` section 7](../survival_method_reassessment.md#7-selected-next-study-language-association-retention-d-035); plan check: [`plan_verification.md`](plan_verification.md); sources: [`reports/research_sources.md`](../research_sources.md).

Updated: 2026-10-21 (after removing the communication folder, D-037).

## State

- Plan verified (D-036): statements about our code and results correct; study not found in prior work (not a novelty claim); register corrected and P20-P29 added.
- Communication with Codex ended on the user's instruction (D-037); the `communication/` folder is removed (recoverable from commit `58c3fff`; its still-useful points are in the appendix of `plan_verification.md`). Shared documents may be edited by Claude as needed; check `git status` before every commit in case Codex edits on her own, and stage explicit paths.
- Hugging Face dataset `karma-skz/always-late-artifacts`: private by the user's decision; the user will make it public later; Claude only needs access (verified from Ada, 277 files, 2026-10-21).
- Implementation: see the Log. Ada: `squeue` empty; `/home2` 7.9 GB free; no job queued.

## NEXT STEP (the recurring job does exactly one step per firing, then rewrites this block)

1. Read this file, `git status --short` and `git log --oneline -5`.
2. Continue the plan from the Log's last line. Order: grounding of the statistical choices (register entries, checked against the primary pages) -> Task 1 (inventory and frozen manifest) -> Task 3 (metrics, pure NumPy) -> Task 2 (collector, fake-model tests) -> Task 4 (paired bootstrap analysis) -> Task 5 (smoke, gates, GPU) -> Task 6. Tests first and watched to fail, minimal code, mutation check of key behaviours, evidence before claims.

## Gates before any GPU job (all must be recorded here as passed)

- Task 2 tests and the existing `test_phase1_run.py` / `test_phase1_budget.py` pass, collection-off outputs unchanged.
- Measured bytes per condition for sentence shards (smoke extrapolation) times nine conditions times the number of streams is below the free space on Ada (7.9 GB at 2026-10-21) with a stated margin.
- Peak GPU memory with the sink is below the card size (previous run 9.6 of 11 GB).

## Questions for the user (do not block on them)

- Licence of our own statistics in the dataset card (currently `license: other`, none chosen).
- Q6, Q7, Q8, Q11, Q12 remain unanswered in `DECISIONS_FOR_REVIEW.md`; not blocking.

## Log

- 2026-10-21: verification report, HF card commit `43026509` (private; scan clean), Codex's work committed (`e05db89`) and aligned (`58c3fff`).
- 2026-10-21 (later): communication folder removed; carried-over points in the appendix of `plan_verification.md`; D-037.
