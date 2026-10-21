# Handoff 020: D-035 results interpreted and documented; HF upload and historical reproduction running (2026-10-21, about 13:30 IST)

> Claude's series. Read this and `reports/association_retention/STATUS.md` (NEXT STEP and Log are the live state; the index of the study is `reports/association_retention/README.md`). Standing rules as in [handoff 017](017-2026-10-21-tasks-1-3-2-done-smoke-queued.md) section 0 and [018](018-2026-10-21-full-run-submitted.md): no communication with Codex, the HF dataset stays private until the user publishes it, mask only, at most 2 Ada jobs, no waiting or polling (one check per firing), never redeploy while a job runs, explicit-path commits ending with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`. Job cadence: every 50 minutes, a self-rescheduling one-shot cron chain (`reports/association_retention/cron_prompt.md`).

## State

- Git `main` pushed (last: review item Q16). Tests: stdlib 180, torch 400.
- Study finished: full run Slurm 31640 (validated; reproduces the original full-precision profile exactly), analyses 31807 and 31808 reproduced byte for byte by 31919 and 31920, cross-checks (`results/crosscheck-eq`), robustness views (`results/robustness-eq`), independent re-derivation of the headline counts (identical), interpretation with limits (`interpretation.md`), follow-up options (`next_steps.md`), review item Q16. Documentation updated (README, PROGRESS, PLAN, review banner, register D-035, reassessment outcome, plan checkboxes, progress log).
- Running: HF upload of `runs/phase1/association-full` on Ada (`hf_upload_retry4.sh`, log `/home2/karma/duan-verify/hf_upload4.log`, 539 files, 3.1 GB; the link from Ada is slow and lossy, restarts resume by deduplication); local historical reproduction `reproduce_phase1.py --streams mask/eq mask/all` into the session scratchpad (`.../scratchpad/repro`).

## Next step

STATUS.md NEXT STEP: check the upload (`UPLOAD OK`, then `hf_artifacts.py verify` must print ok true, then update `reports/artifacts.md`); check the reproduction verdict (REPRODUCED for both streams; then `phase1_summary.py --check` and the plan's Task 5 last checkbox); if the upload keeps stalling use the fallback in the STATUS log (compact files only, shards stay on Ada). Then wait for the user's review (Q16).

## Open questions for the user

Q16 (which follow-up, options A to F in `next_steps.md`); licence of our own statistics (`license: other`); making the dataset public; Q6, Q7, Q8, Q11, Q12.

## Pitfalls

Ada to HF throughput is poor; `scp` from Ada about 0.26 MB/s; `pkill -f` with a pattern that occurs in the ssh command line kills the session; `/home2` has 5.7 GB free; `u22-cpu` needs account `devalab`.
