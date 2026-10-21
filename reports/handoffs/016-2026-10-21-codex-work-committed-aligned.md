# Handoff 016: Codex's work committed and aligned (2026-10-21, about 06:25 IST)

> Claude's series (015 onward). The ONLY source of Claude's previous chat together with `reports/association_retention/STATUS.md`. If this file and the repository disagree, the repository wins: say so in the next handoff. **Correction to handoff 015:** it says Codex's files were untouched; since then they were committed (`e05db89`) and amended (below).

## 0. Standing rules

As in [handoff 015](015-2026-10-21-plan-verified-cron-set.md) section 0, with one change: the user wrote (2026-10-21) "you are allowed to commit and push codex's work if it is needed. just verify that it still aligns with our current work, or update codex's work accordingly". So Claude may commit Codex's documents and edit them for alignment; still announce in `communication/`, re-read before patching, stage explicit paths, no `git add -A`, reset, checkout or stash.

## 1. Mission

Unchanged: D-035 (language-association retention across 27 languages, mask only). Recurring job `56dd5dd0` (`7,27,47 * * * *`) runs the NEXT STEP in `reports/association_retention/STATUS.md`.

## 2. State right now

- Commits on `main`: `df811ba` (old head), `ac42dcd` and `8089f93` (my verification, replies, HF card, handoff 015), `e05db89` (Codex's documentation as she left it), then the amendment commit of this step (see `git log --oneline -6`). Check `git status --short`: it should be clean after this step's push.
- Tests: stdlib 161 pass, `phase1_summary.py --check` up to date, links and anchors in all changed documents resolve (script in the session scratchpad, not committed). Torch suite and reproduction not run (no code or result changed).
- Ada: no job; `/home2` 7.9 GB free. HF dataset private; card uploaded; waiting for the user (Q10).
- Codex: has not replied since `replies/2026-10-21-codex-to-claude-receipt.md`; she has two unacknowledged messages from me (`replies/...-claude-to-codex-q13-q15-ack-and-verification.md`, `messages/...-claude-to-codex-commit-and-amendments.md`). There is no watcher.

## 3. What this step did

Committed Codex's 24 files as `e05db89`; amended `research_sources.md` (audit notes P01-P19, new P20-P29, audit search paragraph), `survival_method_reassessment.md` (seven table cells, audit sentence, verification note in section 7), the D-035 plan (cron line, "Verification amendments" section), `scheduled_routine.md`, `HANDOFF.md`, `PROGRESS.md`, `PLAN.md`, `README.md`, `DECISIONS_FOR_REVIEW.md` (banner, Q10 status), `reports/decisions_and_anomalies.md` (D-035 status, new D-036), `reports/progress_log.md` (entry), `communication/README.md`, `communication/replies/README.md`, `reports/artifacts.md`; wrote the message to Codex. I re-read in the PDFs: P14 (no "confidence"), P13 Table 2, P03 Figure 9 "65K", and the FLORES-200 licence.

## 4. The next step, exactly

Unchanged from handoff 015 section 4: Task 1 step A (tests first for `scripts/g0/phase1_association_inventory.py`, then the implementation, then `reports/association_retention/artifact_inventory.md` and the frozen-input manifest). Source: plan Task 1 plus its "Verification amendments". "Done": tests green and mutation-checked; inventory states available / derivable / new-collection-required per quantity; hashes of `selectivity.pt` and `survival.pt`; cohort and control sizes; uninspected locations listed. Read `communication/` first.

## 5. Open questions for the user

Make the HF dataset public (needs the user to allow the action or run it); licence choice for our own statistics; Q6, Q7, Q8, Q11, Q12 remain open in Codex's review.

## 6. Environment and pitfalls

As in handoff 015 section 6. Link-check script: `/tmp/claude-1000/-home-karma-Documents-ANLP-project/239e7a33-140d-4688-8581-25c246ab3668/scratchpad/linkcheck.py` (checks relative links and heading anchors of changed documents; may be lost with `/tmp`).

## 7. What I would check if something looks wrong

`git log --oneline -6`, `git status --short`, `CronList`, `ls communication/messages communication/replies`, `python3 scripts/g0/phase1_summary.py --check`.
