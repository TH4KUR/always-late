# The 2-hourly routine (run it in its own Claude Code chat)

Defined 2026-10-18 on the user's instruction. This file is the single source of truth for what each scheduled invocation does; the cron prompt only points here.

**Start it** (in a NEW Claude Code session whose working directory is this repository; cron jobs are session-only, fire only while the session is idle and expire after 7 days):

```
/loop 2h Follow reports/scheduled_routine.md exactly, from the section "Each invocation".
```

**Stop it:** tell that session "stop the scheduled job" (it runs `CronDelete`; `CronList` shows the id) or exit the session. The interactive session that built Phase 1 has **no** scheduled job (the old ones `34efad57` and `2b110f38` are deleted).

## The rules (from the user, 2026-10-18)

1. **Replicate the source the way the source did it.** Where our goal is to replicate a method (Duan et al. 2026, Gemma Scope, Deng et al. 2025, Joshi et al., FLORES-200), use their code and setup unchanged. Every method choice cites a paper section, equation or repository line; if the source is silent, take the most literal reading, write the gap into `DECISIONS_FOR_REVIEW.md` and say "default used". Anything that deviates is a labelled diagnostic with a D-entry and never replaces the replication, so that our implementation cannot be the cause of a misleading result. No change of an estimand, threshold or exclusion of an analysis that has already produced a result (D-017).
2. **Do not monitor jobs.** After submitting a job, record its Slurm id, the command, the commit and the expected finish time in `HANDOFF.md` and `PROGRESS.md`, and move on. Check once per invocation with read-only commands (`squeue`, `sacct`); never poll or wait.
3. **When a job has finished:** fetch the small files (`scripts/g0/fetch_phase1.sh`), verify them (sizes against Ada, test suites at HEAD), analyse with the existing code, record the results (`reports/`), push the large outputs to Hugging Face (`scripts/g0/hf_artifacts.py upload`, extend `reports/g0/hf_artifact_spec.json`, then `verify` must print `"ok": true`; update `reports/artifacts.md`), rebuild the runs under the fixed wandb schema (`scripts/g0/phase1_wandb_manifest.py add`, `wandb_rebuild.py rebuild --only ...`, then `verify` must print `"ok": true`), and update every document that mentions the state (`README.md`, `PROGRESS.md`, `HANDOFF.md`, `PLAN.md`, `reports/progress_log.md`, `reports/decisions_and_anomalies.md`).
4. **Decisions go to the user.** Any important decision you would have asked about is appended to `DECISIONS_FOR_REVIEW.md` (question, context with source, options, the default you used, the item's effect) and you continue with the default. The user answers there; apply answered items first (see below).
5. **Progress always states the next run's goal.** At the end of every invocation rewrite the block "NEXT RUN'S GOAL" at the top of `PROGRESS.md` (what the next invocation must do, in order, with the source for each step and what "done" looks like).
6. **Test first, mutation-checked, evidence before claims** (superpowers:test-driven-development, superpowers:verification-before-completion): write the failing test, watch it fail, minimal code, then check key behaviours by mutation (`python3 -B`, clear `__pycache__`, `scripts/g0/mutcheck.py`). Say "verified" only with fresh command output.
7. **Never:** edit `paper/` (the team writes the prose), edit `third_party/` or the token files, put a secret in a file or a commit (run the secret scan before every commit), queue more than 2 Ada jobs at a time, queue a job that is not listed in `PLAN.md` under "Next runs" with its source, delete anything on wandb or Hugging Face except a run you created and verified as replaced (D-008).

## Each invocation

1. `git pull`; read `PROGRESS.md` "NEXT RUN'S GOAL", `DECISIONS_FOR_REVIEW.md`, the last entry of `reports/progress_log.md`.
2. **Answers first:** for every item in `DECISIONS_FOR_REVIEW.md` with text after `ANSWER:` and `STATUS: open`, act on it (D-entry in `reports/decisions_and_anomalies.md`, code or analysis change test-first, update the plan), then set `STATUS: APPLIED <date>`. If an answer contradicts an earlier D-entry, supersede it explicitly; never delete history.
3. **Jobs:** `ssh karma@ada-gw1 'squeue -u karma; sacct -u karma -S <yesterday> -X --format=JobID,JobName%14,State,Elapsed,ExitCode'`. Finished? Do rule 3 of the rules above. Failed? Read the log tail (filter progress bars), record an A-entry with the cause, fix test-first, resubmit only if it is listed in `PLAN.md`.
4. **Otherwise do the next step** of "NEXT RUN'S GOAL" (CPU-side research and analysis; the Ada environment is on the login node and in Slurm jobs only).
5. Commit and push after every green step (message ends with the Claude co-author line; run the secret scan first).
6. Rewrite "NEXT RUN'S GOAL"; add a dated entry to `reports/progress_log.md`. If nothing was due, one line saying so.

## Environment notes (verified 2026-10-18)

- Local environments are rebuilt when `/tmp` is wiped: `python3 -m venv <scratch>/torchvenv && pip install --index-url https://download.pytorch.org/whl/cpu torch && pip install numpy scipy pandas statsmodels pytest` (suite: 147 torch tests, `python -B -m unittest discover -s scripts/g0/tests_torch`; 123 stdlib tests in `scripts/g0/tests`); wandb needs its own venv (`pip install wandb`; the key is in `~/.netrc`).
- Ada: `ssh karma@ada-gw1`, project `/home2/karma/duan-verify`, env `env/`, outputs `out/`, Hugging Face token at `hf/token` (never printed or copied), deploy with `bash scripts/g0/deploy_phase1.sh`, submit with `sbatch [-A nlp] run_phase1.sbatch <name> <flores|wikitext> <mask|none> [args]`. Never `cd` inside a backgrounded `&&` chain; never `cat` an upload log (progress-bar spam: filter with `tr "\r" "\n" | sed "s/\x1b\[[0-9;]*[A-Za-z]//g" | grep`).
- Hugging Face dataset (private): https://huggingface.co/datasets/karma-skz/always-late-artifacts; wandb project `anlp-project`, entity `karma-skz-`.
