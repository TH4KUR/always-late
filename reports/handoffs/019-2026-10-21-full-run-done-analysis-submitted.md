# Handoff 019: D-035 full association run finished and validated, analysis jobs submitted (2026-10-21, about 10:40 IST)

> Claude's series. Read this and `reports/association_retention/STATUS.md` (NEXT STEP and Log are the live state). Standing rules as in [handoff 017](017-2026-10-21-tasks-1-3-2-done-smoke-queued.md) section 0 and [018](018-2026-10-21-full-run-submitted.md): no communication with Codex, the HF dataset stays private until the user publishes it, mask only, at most 2 Ada jobs, no waiting or polling, never redeploy while a job runs, explicit-path commits ending with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`. The job cadence is every 50 minutes as a self-rescheduling one-shot cron chain (`reports/association_retention/cron_prompt.md`).

## State

- Git `main` at `dc7c3e1` (plus the status commit of this step). Tests: stdlib 180, torch 377.
- Full run Slurm `31640` COMPLETED (3 h 32 min, exit 0, peak GPU 9.645 GB). Sidecar `/home2/karma/duan-verify/out/phase1/association-full` (2.76 GB, complete, valid). It reproduces the original `p1-mask` full-precision profile and legacy results exactly (details in the STATUS.md log).
- Analysis jobs submitted 10:35 IST on Ada (CPU-only, partition `u22`, deployed commit `dc7c3e1`): `31807` (equal-budget stream, 1,000 draws, seed 0) and `31808` (all-token stream, aggregate profile only).

## Next step

STATUS.md NEXT STEP: check the two jobs once per firing; when finished copy the small outputs, confirm byte reproducibility with a rerun, push the sidecar and analyses to the HF dataset, then Task 6 (interpretation and documentation).

## Open questions for the user

Licence of our own statistics in the dataset card (`license: other`); Q6, Q7, Q8, Q11, Q12 remain open (not blocking). The dataset is private until the user publishes it.

## Pitfalls

`u22-cpu` needs account `devalab`; `/share1` is not mounted on compute nodes; `scp` from Ada is about 0.26 MB/s (analyse on Ada); `/home2` has 5.7 GB free.
