# Ledger: survival measurement-validity study (D-117)

Plan: [`../../docs/superpowers/plans/2026-10-29-survival-measurement-validity-study.md`](../../docs/superpowers/plans/2026-10-29-survival-measurement-validity-study.md). Executed from 2026-10-29 by the 40-minute cron job of session project-6d (user's direction, see the 2026-10-29 block in [`../gptq_extension/cron_prompt.md`](../gptq_extension/cron_prompt.md)). Add only; never shorten.

## Task table

| # | task | state |
|---|---|---|
| 1 | Plan, D-117, Q33 | done (4e91c96) |
| 2 | Stage 1 freeze, code, tests, run, results (D-118) | done (99f15d6, 2cabb88, cf1f86e) |
| 3 | Stage 2 design note (D-entry): sidecar inputs and outputs, states, streams, gates, disk and runtime estimate, how weights are rebuilt per job | **done 2026-10-29 05:29 IST** (D-119, [`stage2_design.md`](stage2_design.md)) |
| 4 | Stage 2 sidecar code + tests (test first, mutation-checked) + local CPU/tiny check | open |
| 5 | Read-only review of the sidecar by session project-fe (send it the commit id by SendMessage; it replies by message) | open |
| 6 | Deploy snapshot, gate job (INT0 == FP; corr from sums == committed survival.pt), smoke, timing/memory | open |
| 7 | Full Stage 2 jobs (queue all at once after the gates pass), validate, fetch small outputs, raw to the private HF dataset | open |
| 8 | Stage 3 freeze (H4 bound against the INT16 tier slope, not zero), reviewed by project-fe, committed BEFORE reading Stage 2 outputs | open |
| 9 | Stage 3 analysis (generated report, D-entry) | open |
| 10 | Stage 4 write-up: plain-language report, PROJECT_RESULTS and CURRENT_STATUS update lines, handoff | open |

## NEXT STEP

**NEXT STEP (rewritten 2026-10-29 05:29 IST): task 4, the sidecar code, test first.** Write `tests_torch/test_survival_sidecar.py` first (toy dictionary of 3 features and 2 languages, known a, b, pre, theta: each of the 21 sums, the eq mask, a language without a selected token, flips and both exceed counts, r rebuilt from the sums equals `LangPearson`, the INT0 identity, several batches equal one batch, the sink inside `run_survival` on the existing toy harness returns identical results with and without it, saved files load and validate), watch them fail, then `scripts/g0/survival_sidecar.py` (`SurvivalSink`, the entry point `main`, the gates G1 to G5 of the design, a validator like `gptq_ext/validate_job.py`), mutation-check with `scripts/g0/mutcheck.py`, run on the toy on CPU. Check first in the installed sae_lens (`~/.venvs/anlp-torch`) whether `sae.hook_sae_acts_pre` exists; the Ada environment has sae_lens 6.54.5. Then task 5: send the commit id to session project-fe (SendMessage) for the read-only review and continue with other CPU work in later firings (sbatch, deploy path, validator) while waiting; no Ada action before the review and before the user's confirmation noted in the design (section 7). Done = the tests pass, the mutants are killed, the toy run reproduces `LangPearson`, committed.

(Superseded 05:29, kept as history: the NEXT STEP below.)

Task 3: write the Stage 2 design note as the next free D-entry (D-119 if still free; re-read the file first) before any code. Facts from the GPTQ work: /home2/karma has about 2.3 GB free; GPTQ and matched-RTN weight sets are rebuilt in every job (about 19 and 15 min) and checked against reports/gptq_extension/artifact_hashes.json; use scripts/g0/gptq_ext/run_with_weights.py and scripts/g0/run_gptq.sbatch; add new paths to scripts/g0/deploy_phase1.sh PATHS; FP-only arrays are identical across arms; JumpReLU thresholds in reports/phase1/sae_thresholds.csv. Done = the D-entry states every streamed quantity, the states, the gates and the disk/runtime budget.

## Log

- 2026-10-29 (session project-fe, Claude): ledger created; the user directed that the cron job of session project-fe's peer (project-6d) carries out Stages 2 to 4. Ownership of D-117 work and of the shared documents passes to that cron job (see cron_prompt.md block of 2026-10-29).
- 2026-10-29 05:29 IST (cron session project-6d, Claude): read the cron block and this ledger; task 3 done: Stage 2 design note (D-119, `stage2_design.md`, written before any code): one new sidecar entry point on the unchanged `run_survival(profile_sink=...)` and the weight wrapper, 21 float64 statistics per (condition, language, feature) for the equal-budget stream, conditions FP, INT16 (legacy job only), INT6, INT4, INT0 gate in three arm jobs, gates G1 to G5, about 6 GPU hours and 525 MB. The Ada jobs rest on the user's Q33 (a) answer as recorded and relayed by project-fe; I have not received those words from the user directly, so the first Ada submission waits for the user's confirmation in my report (design section 7), after the code, the tests and project-fe's review. NEXT: task 4.
