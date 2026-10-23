# Stale sections of HANDOFF.md (archived 2026-10-23 18:32 IST, verbatim)

HANDOFF.md was written on 2026-10-16 and patched since. Its reference material (the rules, the runbook, the environment facts, the do-not list, the file map, the baseline result, the wandb schema, the Phase 1 code) is still current and stays there. The blocks below described the STATE of that day (pending diagnostics, open decisions, a routine) and contradicted the present state of the project; they are moved here unchanged (only relative links of the banner paragraphs were rebased for this directory). The current state is in [`reports/functional_validation/STATUS.md`](../functional_validation/STATUS.md) and the newest file of [`reports/handoffs/`](../handoffs/). The text is also in git history (last commit holding it in HANDOFF.md: `f4f1d8d`).

## Banner paragraphs (the two handoff series of 2026-10-21 and the Codex 014 note), formerly lines 4 to 7 under the title

> **Two handoff series since 2026-10-21 (D-036):** Codex's 010-014 (decision review, planning) and Claude's [015 -- plan verified, recurring job set](../handoffs/015-2026-10-21-plan-verified-cron-set.md) and [016 -- Codex's work committed and aligned](../handoffs/016-2026-10-21-codex-work-committed-aligned.md) (implementation of the D-035 study; Claude's progress is in [its status file](../association_retention/STATUS.md); a 20-minute recurring job runs it). The newest file in `reports/handoffs/` is still the one to read first; if both series continue, read the highest number and the status file. The plan was [verified](../association_retention/plan_verification.md) before implementation started; no GPU job was queued when this note was written.
>
> **Current Codex handoff: [014 -- language-association plan](../handoffs/014-2026-10-21-language-association-plan.md) (2026-10-21).** The user selected checking original language association across all 27 languages and requested a documented plan (D-035). [Implementation tasks and gates](../../docs/superpowers/plans/2026-10-21-language-association-retention.md); [measurement design](../survival_method_reassessment.md#7-selected-next-study-language-association-retention-d-035). Planning is recorded; implementation/new measurements are pending. Q13-Q15 decisions remain in force but their broader work is separate. Mask-only, no new Duan numerical comparison, full INT8/7/6/5/4 ladder and honest exploratory status remain required. This supersedes older directions below.

## Section 0, the paragraph 'Mission now' (2026-10-16)

**Mission now.** (1) Finish the two pending diagnostics (section 5). (2) Get the team's decisions (section 9), above all which regime (BOS or no BOS) the study uses. (3) If the team chooses, rebuild the language-selective analysis as a small, labelled diff on top of their script, verified against the baseline before any new claim.

## Section 2, State of play (2026-10-16, night)

## 2. State of play (2026-10-16, night)

| Item | Status |
|---|---|
| Review of the old code and results, `issues.md` | done (kept as evidence of the discarded work) |
| Their repo vendored and pruned (`third_party/duan2026/`, content tree hash `a84e43da...adc`), 54 tests | done |
| Baseline: their `3b_full` (500k tokens, INT8..INT4, pruning off), job 23472 | **done; reproduces their table** (section 8) |
| Audit of why the old pipeline disagreed: `reports/audit_our_vs_duan.md` | **done**: R2 (their code + masked BOS) and R5 (8 cells, old HF pipeline on their tokens) |
| Old codebase | **discarded** (tag `legacy-pipeline-2026-10-16`); Phase-1 results kept in `reports/legacy_pipeline/` as hypotheses |
| Diagnostics: R1 (their code, no BOS, INT0/16/12) and R5c (`O_mask`) | **both done 2026-10-17**: R1 gives Gate 1 `noise_dominated` in their regime (audit 6.7); R5c passes Q4 for the mask regime (6.5) |
| Closure protocol (`duan_codebase_audit.md` section 7) | **CLOSED 2026-10-17: D-NOISE** (Gate 0 pass, Gate 1 noise-dominated in their no-BOS regime by a 0.001 margin, passes with BOS masked; decision D-009, audit 6.7). Terminal: no further verification of Duan is planned |
| Duan (2026) paper read in full (V-3) | **done 2026-10-17**: BOS never mentioned, context-reset explanation contradicted (A-014), q/k/v scale pooling contradicts Eq. 2 (A-015); audit 6.6 |
| Study design on top of their code | **not started**; waits for the team's regime decision |


## Section 5, Diagnostics on their code and pending jobs (R1 and R5c, both completed 2026-10-17)

## 5. Diagnostics on THEIR code (modified; always labelled) and pending jobs

Tooling: `scripts/g0/make_duan_runners_v3.py` (reads the vendored *pristine* scripts; BOS arms `none`/`prepend_keep`/`prepend_mask`, INT0 restore-only control, ladder, snapshots, resume, wandb heartbeats), `scripts/g0/run_g0.sbatch <arm> <none|prepend_keep|prepend_mask> [bits] [tokens] [mode]` (2 x 2080 Ti), `scripts/g0/g0_wandb.py` (logger and `compute_gate_metrics`), `scripts/g0/r5_compare.py` (rules of the audit as code; reads summary CSVs only). Deployed copies on Ada: `/home2/karma/duan-verify/duan03_runner_v3.py` (hash prefix `2c34fecaee1d2e32`), `run_g0.sbatch`, and the legacy harness copy `r5code/` (used only by R5c).

| Job | What | Ada output | Log |
|---|---|---|---|
| **24451 `R1-nobos-floor`** (`research`, 2 GPUs) | their code, **no BOS**, ladder INT0/16/12: Gate 1 in THEIR regime. **COMPLETED 2026-10-17 (3 h 06 min, exit 0); fetched to `reports/g0/R1_nobos_floor/`; judged (audit 6.7); wandb rebuilt** | `/home2/karma/duan-verify/out/R1/` | `logs/R1-nobos-floor-24451.out` |
| **24452 `R5c-nlp`** (`nlp`, 1 GPU; **COMPLETED 2026-10-17, exit 0; fetched to `reports/g0/r5/R5c/`; judged; wandb rebuilt**) | `O_mask`: the legacy HF pipeline with BOS masked, INT8/7/6 (completes Q4 for the mask regime); runs the deployed legacy harness, unaffected by the repo cleanup | `/home2/karma/duan-verify/r5/R5c/O_mask/` | `logs/R5c-nlp-24452.out` |

Already done and fetched: R2 (`reports/g0/R2_bos_mask/`), R5a/R5b (`reports/g0/r5/`), smoke (`reports/g0/r5_smoke/`).


## Section 9, Open decisions for the user / team (answered since: Q1, Q2, D-028, D-029)

## 9. Open decisions for the user / team

1. **Which regime does the study use?** Their no-BOS windows (perplexity 457, INT7 improves perplexity: a degraded regime) or BOS (perplexity 16, survival >= 97% at INT6)? This decides how the paper treats Duan's numbers and whether INT7/INT6 can still be confirmatory (in the BOS regime they are at ceiling; INT5/INT4 have the headroom and were never measured with BOS in a verified setup).
2. Rebuild the language-selective analysis (FLORES panel, selectivity, tiers) as a diff on their code, or stop at the audit result ("survival depends on BOS handling by 18-46 pp at INT7/INT6") and write that up? Deadline for the final submission: Oct 31.
3. Who writes the rewritten results/discussion (must be the team)?
4. Optional: R3 (their TL code, BOS kept), a small-sample cell for the 2,048-token demo residual, INT5/INT4 with BOS.


## Section 10, the checklist for R1 and R5c (completed 2026-10-17)

## 10. (COMPLETED 2026-10-17) check R1 and R5c, fetch, judge, finish the audit: both done, closure D-NOISE; kept for reference

1. `ssh karma@ada-gw1 'squeue -u karma; sacct -j 24451,24452 -o JobID,JobName,State,Elapsed,ExitCode'`; read the log tails (`tail -c 3000 ... | tr '\r' '\n'`). If a job died: fix test-first or resubmit with the same out-name (R5c skips finished cells; R1 has resume), never touch the user's other jobs.
2. Fetch small files only, excludes before includes, then **verify by listing**: R1 `out/R1/` -> `reports/g0/R1_nobos_floor/` (summaries, `weight_perturbation_*.json`, `run_meta.json`; not `*_per_feature.csv`, `snap_*`, `*.pt`); R5c `r5/R5c/` -> `reports/g0/r5/R5c/` (summary, `cell_done.json`, `selfcheck.json`, `code.sha256`).
3. **Gate 1 in their regime** (audit section 7.2): rows = R1's FP/INT0/INT16/INT12 conditions plus the vendored baseline's INT8/7/6 (identical code and tokens; bit-identical smokes), perturbation sizes from R2's `weight_perturbation_INT*.json` (same weights and quantizer), `g0_wandb.compute_gate_metrics`; pass iff median r at INT16 and INT12 >= 0.99 and slope in [0.5, 1.5]. Gate 0: R1's INT0 must be identical to FP.
4. **Q4, mask regime**: `python3 scripts/g0/r5_compare.py --r5 reports/g0/r5 --baseline reports/g0/baseline_full/phase3_gemma_outputs_3b_full_500k_L12/phase3_summary_final.csv --published third_party/duan2026/pristine/results/tables/phase3_summary_final.csv --r2 reports/g0/R2_bos_mask/phase3_summary_final.csv --out reports/g0/r5/r5_report.md` (it now also compares `O_mask` with `T_mask`); the per-feature Spearman at INT6 of Q4 needs the per-feature CSVs on Ada (compute there with the Ada env or fetch the two files to the scratch directory, not into the repo).
5. Assign a disposition from audit section 7.3 only if its inputs exist (R3, their TL keep arm, is still missing: say so), fill audit sections 6.3-6.4, add a progress-log entry, update `PLAN.md`/`PROGRESS.md`/this file, run the tests, push.
6. Report to the user: Gate 1 verdict in their regime, Q4 for the mask regime, what is still open. No paper prose.


## Section 11, The routine (historical text of 2026-10-18)

## 11. The routine (rewritten 2026-10-18 twice; latest user instruction)

**Updated 2026-10-21 (D-036):** the user has since asked for a recurring job every 20 minutes for the D-035 study, so there is now a session-only cron job (see `reports/scheduled_routine.md`, top). Historical text follows: there is **no scheduled job and no second chat**: the user said "do it in this chat only; after each one create a handoff and use that as the only source of the previous chat; do that instead of autocompacting". Work proceeds one step at a time in the main chat; after each step the next numbered handoff is written in `reports/handoffs/` and pushed. The step definition, rules and environment notes are in `reports/scheduled_routine.md`; open questions for the user live in `DECISIONS_FOR_REVIEW.md`; `PROGRESS.md` starts with "NEXT RUN'S GOAL".

History: hourly cron jobs (`34efad57`, `2b110f38`) were session-only and are deleted; one of them had silently disappeared after a session restart (nothing fired between about 01:30 and 05:40 IST on 2026-10-18; the S6 jobs finished unseen at 03:58 / 04:03).


## Section 13, the last paragraph ('Earlier status line': S6 running)

(Earlier status line, kept for the record:)  S1-S5 and S7 done (the anchor reproduces their smoke run bit-identically); **S6 running: Slurm 25745 `p1-mask`, 25746 `p1-none`** (commit 618496f; outputs `/home2/karma/duan-verify/out/phase1/<name>/{eq,all}/`, 4-6 h each; `eq` = the pre-registered equal-token-budget stream = primary, `all` = all tokens = sensitivity, D-015). Then `phase1_analysis.run_analysis` on each directory, HF upload, wandb rebuild, S8 `reports/phase1_results.md`. Anchor command: `run_phase1.sbatch anchor wikitext none --budget 20000 --bits 8 6 --batch-size 1 --loss tl`; compare with `python3 scripts/g0/phase1_anchor.py <summary.csv>`.
