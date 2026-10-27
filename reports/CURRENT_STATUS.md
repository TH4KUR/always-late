# Current project status

Updated 2026-10-27 14:34 IST (documentation cleanup requested by the user). This is a snapshot from the repository, not a live cluster check. The previous text of this page is kept verbatim in [`archive/doc_cleanup_2026-10-27/reports__CURRENT_STATUS.md.txt`](archive/doc_cleanup_2026-10-27/reports__CURRENT_STATUS.md.txt).

**All results, in plain words, with per-language tables and a guide to the data: [`PROJECT_RESULTS.md`](PROJECT_RESULTS.md).**

## Evidence state

| work | state | where |
|---|---|---|
| Phase 1 (three pre-set tests), association retention, F1–F5, Belebele (only confirmatory family), Study G and extensions B/C/D | complete; Phase 1 and Belebele null | [`PROJECT_RESULTS.md`](PROJECT_RESULTS.md) §4.1–4.8, [claim map](functional_validation/RESULTS_SUMMARY.md) |
| Llama-3.2-1B second-model check | complete; pre-set verdict **not consistent** (D-090). Gemma-2-9B belongs to another team: no results here | §4.9 |
| Intervention transfer, main (27 languages) and external (Wikipedia, six languages) | analysed once under the pre-set rules: UTILITY **not met** (main, D-094), **met** (external, D-095) | §4.10, [main](intervention_transfer/results/main/README.md), [external](intervention_transfer/results/external/README.md) |
| Recovery analyses R1–R4 (locked cross-corpus prediction, gradient scalars, common support, native versus fixed) | complete, all post hoc (D-096 to D-099) | §4.11, [results](intervention_transfer/results/recovery/README.md) |
| Paired arithmetic-precision audit (R5, D-100) | **built and tested locally, never deployed or run** | §4.12, [freeze](intervention_transfer/precision_audit_freeze.json), [plan](../docs/superpowers/plans/2026-10-27-precision-audit-task2-implementation.md) |
| **GPTQ extension (D-101, started 2026-10-27 17:34 IST at the user's request): the same experiments with weights chosen by GPTQ and by a matched RTN, everything else unchanged** | **in progress; no model run yet** | [ledger](gptq_extension/STATUS.md), [plan](../docs/superpowers/plans/2026-10-27-gptq-matched-comparison.md), [audit](quantization_gptq_audit_2026-10-27.md) |

## Next action

**HOLD (the user, 2026-10-27 09:50 IST): stop at this checkpoint and finalise.** Ada is idle; nothing is queued. Do not deploy, submit Ada jobs or start new experiments or analyses until the user says so. **Update 2026-10-27 (the user): the recurring 40-minute cron jobs were stopped; none exists now.** **Update 2026-10-27 17:34 IST (the user): the HOLD is lifted for ONE purpose, the GPTQ extension ([D-101](decisions_and_anomalies.md), [ledger](gptq_extension/STATUS.md), open question Q30); a new 40-minute cron job runs it ([prompt](gptq_extension/cron_prompt.md)). Nothing else may start (Q29 stays open).** **Update 2026-10-27 18:32 IST: tasks 1 to 7 done (calibration pool built and frozen, calibrator, wrapper, freeze); the first GPU job of the extension, gate job A (Slurm 42848, bits 4), is submitted; see the [ledger](gptq_extension/STATUS.md).** **Update 2026-10-27 19:52 IST: gate jobs A and B passed (artifact hashes equal on two nodes); stage 1 (both arms, whole ladder) and the INT4/INT6 transfer matrix are queued: 10 jobs, Slurm 42951 to 42960, 8 running.**

Open questions for the user: Q29 (run the precision audit or stop and keep the time for the course deliverables) and the older ones listed at the top of [`DECISIONS_FOR_REVIEW.md`](../DECISIONS_FOR_REVIEW.md). If the audit is approved: deploy a committed snapshot, validate the bounded instrument job first (real GPU, all gates, dtype checks, runtime and memory), only then the frozen 128-feature / 512-pair run, analysed by revision R5 ([handoff 045](handoffs/045-2026-10-27-external-panel-analysed-report-deck-final-recovery-plan-next.md)).

## Deliverables (due **31 October 2026**)

- [Course report](../paper/course-report/README.md): revision 3, 11 pages, official checker **All Clear** (manifest refreshed 2026-10-27 04:37 IST).
- [Presentation](../paper/presentation/): 18 pages, final-mode build. Final archive: `bash scripts/make_final_archive.sh <outdir> paper/presentation/AlwaysLate-Final-Presentation.pdf` (trial build 31.5 MB, not yet the final one).
- Still for the team: author order and final wording, and the course AI-use disclosure.
- Novelty: [dated assessment](novelty_assessment_2026-10-26.md); an A-tier contribution is **not** established. Research direction in one line: which definitions of "an SAE intervention is preserved" stay reliable under weight quantization, new documents and measured arithmetic precision (the old page is archived; the [recovery plan](../docs/superpowers/plans/2026-10-26-publication-recovery-after-results.md) stays).

## Maintenance and history

[Codebase map](../docs/CODEBASE.md), [reports index](INDEX.md), [27 October modular-cleanup archive](archive/modular_cleanup_2026-10-27/README.md), [27 October documentation-cleanup archive](archive/doc_cleanup_2026-10-27/README.md). No experiment history was deleted.
