# Handoff 018: D-035 full association run submitted (2026-10-21, about 07:10 IST)

> Claude's series. Read this and `reports/association_retention/STATUS.md` (NEXT STEP and Log are the live state). If this file and the repository disagree, the repository wins. Standing rules: as in [handoff 017](017-2026-10-21-tasks-1-3-2-done-smoke-queued.md) section 0 (no communication with Codex, the HF dataset stays private until the user publishes it, mask only, at most 2 Ada jobs, no waiting or polling, explicit-path commits, `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`).

## State

- Git `main` at `df9577a` (pushed). Tasks 1 to 4 are implemented, test first and mutation-checked (inventory 14, metrics 21, collector 34 plus hook 5 plus CLI 4, analysis 33 mutants killed; equivalent mutants noted in the log). Suites: stdlib 180, torch 375.
- Smoke (Slurm 31613 collection on, 31614 off, 4 min 26 s each): historical outputs byte-identical with and without collection; peak GPU memory 9.355 against 9.353 GB; the sidecar passes `validate_run`; FP counts equal the legacy reference-side counts exactly; INT0 equals FP; the analysis ran end to end on the real output (`--allow-subset`). Gate B is met (STATUS.md).
- **Full run: Slurm `31640` `assoc-full`, submitted 07:00 IST**, deployed commit `1655450`, equal-budget sentence shards with float32 sums, output `/home2/karma/duan-verify/out/phase1/association-full` (about 3.6 GB), log `/home2/karma/duan-verify/logs/assoc-full-31640.out`. Start unknown; expected 2.5 to 4 hours of run time.
- Cron `5e64064d` (`7,27,47 * * * *`) does one step per firing from STATUS.md; recreate it from `reports/association_retention/cron_prompt.md` if the session restarted.

## Decisions

D-037 (communication folder removed) and D-038 (implementation rulings: equal-budget sentence shards with float32 sums, percentile intervals plus diagnostics and the paired loss fraction, analysis on an Ada CPU node because copying home runs at 0.26 MB/s) are in `reports/decisions_and_anomalies.md`.

## Next step

STATUS.md NEXT STEP, items 1 to 3: check the job once per firing; when finished validate on Ada, compare the new FP profile with the original run, redeploy, submit `run_association_analysis.sbatch association-full --stream eq --n-boot 1000 --seed 0` (CPU partition, `--test-only` first), copy the four small outputs into `reports/association_retention/results/`, push the sidecar to the HF dataset, then Task 6 (interpretation, limits, documentation).

## Open questions for the user

Licence of our own statistics in the dataset card (`license: other`); Q6, Q7, Q8, Q11, Q12 remain open in `DECISIONS_FOR_REVIEW.md` (not blocking).

## Pitfalls

`/share1` is not mounted on compute nodes; `scp` from Ada runs at about 0.26 MB/s (analyse on Ada, copy only small files); a file read while `scp` is still copying fails its hash check (wait for completion); `sbatch --test-only` start estimates are pessimistic (the smoke started at once); the Ada environment has scipy 1.15.3, numpy 2.2.6 and torch 2.14.1; scratch scripts live in `/tmp/claude-1000/-home-karma-Documents-ANLP-project/239e7a33-140d-4688-8581-25c246ab3668/scratchpad/` and may be lost.
