# Association-retention study: Claude's status file (D-035)

Owner: Claude (Codex owns `PROGRESS.md`, `HANDOFF.md`, `PLAN.md`, the review file, the register, the progress log and the reassessment record; see `communication/README.md`). Plan: [`docs/superpowers/plans/2026-10-21-language-association-retention.md`](../../docs/superpowers/plans/2026-10-21-language-association-retention.md); design: [`reports/survival_method_reassessment.md` section 7](../survival_method_reassessment.md#7-selected-next-study-language-association-retention-d-035); plan check: [`plan_verification.md`](plan_verification.md).

Updated: 2026-10-21 (session start of the recurring job).

## State

- Plan verified (2026-10-21): statements about our code and results are correct; study not found in prior work (not a novelty claim); register corrections sent to Codex; technical corrections listed in `plan_verification.md` section 3.
- Codex's documentation is committed (`e05db89`) and amended for alignment (register P-notes and P20-P29, plan amendments, "no cron" notices, D-036); message `communication/messages/2026-10-21-claude-to-codex-commit-and-amendments.md`; she has not acknowledged yet. User authorisation: "you are allowed to commit and push codex's work if it is needed. just verify that it still aligns with our current work, or update codex's work accordingly". Updating her documents is allowed when needed for alignment; still announce in `communication/` and re-read a file before patching it.
- Implementation: **not started.** No code, no test, no GPU job, no commit for the study yet.
- Hugging Face dataset: card rewritten and uploaded (private); the step making it public was refused by the tool permission layer; **do not retry**; the user must allow it or run it (needs the user's decision; see "Questions for the user").
- Ada: `squeue` empty at 2026-10-21; `/home2` has 7.9 GB free; no job of mine queued.

## NEXT STEP (the recurring job does exactly one step per firing, then updates this block)

1. Read `communication/messages/` and `communication/replies/` for files newer than `replies/2026-10-21-claude-to-codex-q13-q15-ack-and-verification.md`.
2. **Task 1, step A (tests first):** write `scripts/g0/tests/test_phase1_association_inventory.py` with the fixtures the plan lists (FP-only response statistics, all-language correlations without quantized profiles, complete new profiles, mismatched language order); run it and show it fail; then implement `scripts/g0/phase1_association_inventory.py` minimally. Source: plan Task 1.
3. After Task 1 (inventory report, frozen manifest from the saved files): Task 3 (`phase1_association_metrics.py`, pure functions, torch-free NumPy tests), then Task 2 (collector, fake-model tests), then Task 4. Task 5 (any GPU) needs the gates below.

## Gates before any GPU job (all must be recorded here as passed)

- Task 2 tests and the existing `test_phase1_run.py` / `test_phase1_budget.py` pass, collection-off outputs unchanged.
- Measured bytes per condition for sentence shards (smoke extrapolation) times nine conditions is below the free space on Ada (7.9 GB at 2026-10-21) with a stated margin.
- Peak GPU memory with the sink is below the card size (previous run 9.6 of 11 GB).
- Codex has been told in `communication/`; the user has not withdrawn the instruction to continue.

## Questions for the user (do not block on them)

- Make `karma-skz/always-late-artifacts` public (Q10 answer is yes; the action was refused by the permission layer; the card is ready; scan clean).
- Licence of our own statistics in the dataset (card says `license: other`, none chosen).

## Log

- 2026-10-21 (later): committed Codex's work (`e05db89`), applied alignment amendments (see the message above), checks: links resolve, summary `--check` up to date, 161 stdlib tests pass; handoff 016.
- 2026-10-21: verification report, reply to Codex, HF card commit `43026509` (private), HF pre-publication scan clean (no tokens, keys, emails or addresses; only `/home2/karma/duan-verify/...` paths).
