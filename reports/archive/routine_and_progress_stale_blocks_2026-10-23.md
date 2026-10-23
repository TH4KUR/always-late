# Stale blocks of reports/scheduled_routine.md and PROGRESS.md (archived 2026-10-23 23:12 IST, verbatim)

The routine page was written on 2026-10-18 for a chat-only workflow and patched since; PROGRESS.md carried a 'NEXT RUN'S GOAL' block of 2026-10-19/21 that told the next step to 'say so and stop' because 'everything left needs the user'. Both contradict the present practice (the recurring 40-minute job of 2026-10-21, state in `reports/functional_validation/STATUS.md`). The texts are moved unchanged except for the relative links of the PROGRESS block, which were rebased for this directory. They are also in git history (last commit that holds them in place: `c7473b5`).

## reports/scheduled_routine.md: the paragraph '(Historical; the cron part is superseded ...)'

(Historical; the cron part is superseded by the recurring-job update at the top.) Defined 2026-10-18 on the user's instruction, **changed the same day**: it does NOT run as a cron job and NOT in a new chat. The user said: "do it in this chat only; after each one create a handoff and use that as the only source of the previous chat; do that instead of autocompacting." This file is the single source of truth for what one step does.

## reports/scheduled_routine.md: the paragraph '(Until 2026-10-21 there was no scheduled job ...)'

(Until 2026-10-21 there was no scheduled job; see the recurring-job update at the top.) There was no scheduled job (the earlier ones `34efad57` and `2b110f38` are deleted; an in-chat `/loop` or cron is not used because the user asked for chat-only work).

## reports/scheduled_routine.md: rule 5

5. **Progress always states the next run's goal.** At the end of every invocation rewrite the block "NEXT RUN'S GOAL" at the top of `PROGRESS.md` (what the next invocation must do, in order, with the source for each step and what "done" looks like).

## reports/scheduled_routine.md: step 1 of 'Each step'

1. Read the newest handoff (only source of the previous chat); `git pull`; then read `PROGRESS.md` "NEXT RUN'S GOAL", `DECISIONS_FOR_REVIEW.md`, the last entry of `reports/progress_log.md` as the handoff directs.

## reports/scheduled_routine.md: step 4 of 'Each step'

4. **Otherwise do the next step** of "NEXT RUN'S GOAL" (CPU-side research and analysis; the Ada environment is on the login node and in Slurm jobs only).

## reports/scheduled_routine.md: step 6 of 'Each step'

6. Rewrite "NEXT RUN'S GOAL"; add a dated entry to `reports/progress_log.md`; write the next handoff, commit and push it.

## reports/scheduled_routine.md: the paragraph 'How a step ends'

**How a step ends.** Commit and push the work, then write the next handoff (`reports/handoffs/NNN-<date>-<slug>.md`, template in `reports/handoffs/TEMPLATE.md`), commit and push it, and tell the user in a few lines what was done, what the next step is, and that the handoff is written. The handoff must be complete enough that a fresh chat with no memory can continue from it and nothing else. After the handoff the chat may be cleared or compacted by the user; never depend on automatic compaction.

## PROGRESS.md: the section "NEXT RUN'S GOAL"

## NEXT RUN'S GOAL (rewritten at the end of every scheduled invocation; the routine is `reports/scheduled_routine.md`, run in its own chat every 2 hours)

**Updated 2026-10-21 (D-035):** the bounded 27-language association study is now selected and planned. Read [handoff 014](../handoffs/014-2026-10-21-language-association-plan.md), the linked design and implementation plan; implementation begins with an artifact inventory and frozen input/cohort manifest. Specific jobs await missing-artifact confirmation and smoke gates. No new measurement has run. **Progress and the next concrete step are kept in `reports/association_retention/STATUS.md` (its NEXT STEP block), rewritten by the 20-minute recurring job; the Mode statement below ("no cron, no second chat") was superseded on 2026-10-21 by the user's request for that job (D-036).** Broader Q13-Q15 questions, remaining review items and novelty claims are separate. Q1/Q2 remain in force. The state below is historical where it conflicts with this update.

State at 2026-10-19 ~17:00 IST: Phase 1 (S1-S8) and N1 to N13 are done and **reproduced end to end** (`reports/phase1_reproduction.md`: 89 of 89 committed result files identical when regenerated from the cached run files at 4 BLAS threads). The one-page evidence summary is `reports/phase1_summary.md` (generated; status labels). **No analysis is unblocked: everything left needs the user.** HF verified (274 files; nothing large added), wandb 19 runs conforming, no Ada job running. **Mode (user): everything runs in this one chat, step by step, a handoff after every step (`reports/handoffs/`, newest = only source of the previous chat); no cron, no second chat.** Newest handoff: `reports/handoffs/009-2026-10-19-n13-reproduction-done.md`.

What a next step does, in this order:
1. Read the newest handoff; read `DECISIONS_FOR_REVIEW.md` for answers.
2. **If any question is answered:** apply it as the handoff's section 4 says (D-entry, test-first change, mark `APPLIED`, regenerate `reports/phase1_summary.md`, handoff).
3. **If none is answered: say so and stop.** Do not invent analyses (the CPU analyses available without new data are exhausted and reproduced); low-cost read-only checks (git, Ada `squeue`, HF `verify`, `CronList`) are fine.
4. Rewrite this block; update the progress log; commit and push; write the next handoff.
