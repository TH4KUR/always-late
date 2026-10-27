# Current project status

Updated during the 27 October 2026 documentation cleanup. Execution snapshot:
handoff 045 through its 09:15 IST update, commit `e17f926`; this is not a live
cluster check. Execution owners should update this page after each completed step.
Keep detailed history in dated handoffs, decisions and logs rather than duplicating
this summary across entry pages.

## Completed evidence

| Work | State and source |
|---|---|
| Phase 1, association retention, F1–F5, Belebele, Study G and extensions | Complete; [functional claim map](functional_validation/RESULTS_SUMMARY.md), [Study G](functional_validation/gain_study/README.md) |
| Llama check | Complete; exploratory verdict **not consistent** (D-090), [report](functional_validation/llama/report.md) |
| Main transfer matrix | All six runs validated and analysed; original **UTILITY not met** (D-094), [results](intervention_transfer/results/main/README.md) |
| Six-language Wikipedia transfer matrix | All six runs validated and analysed; original **UTILITY met** (D-095), [results](intervention_transfer/results/external/README.md) |
| Locked FLORES-to-Wikipedia prediction | Complete (D-096); primary INT4 reduction 19.9% [7.8, 31.3], INT6 15.8% [-2.7, 28.7] against the stronger calibration-selected null, INT16 28.2% [21.3, 35.1] |
| Gradient, common-support and native/fixed checks | Complete (D-097–D-099); all post hoc, [recovery results and limits](intervention_transfer/results/recovery/README.md) |
| Paired arithmetic audit implementation | Built and locally tested (D-100); deployment/instrument validation next in the recorded snapshot. No empirical precision-audit outcome is recorded here. |

The locked cross-corpus result supports transfer to independent documents. The
INT6 interval includes zero against the constant baseline. Improvement at INT16 is
at least as large as at INT4; quantization specificity remains unestablished.
The method uses four control interventions per test context and is not demonstrated
to be cheaper than measuring the target directly.

On their signed-loss target, stored gradient scalars predict signs well above the
reference threshold; this corrects the earlier overbroad “gradients fail” reading.
Common-support comparisons show modest direction-specific error on small retained
populations, not selective destruction of language features. Native/fixed edit
differences are consistent at INT4, but do not identify a causal percentage due
to amplitude. Preserve all main and external verdicts next to these later analyses.

## Next action

1. Follow the latest update of [handoff 045](handoffs/045-2026-10-27-external-panel-analysed-report-deck-final-recovery-plan-next.md)
   and the [precision-audit implementation plan](../docs/superpowers/plans/2026-10-27-precision-audit-task2-implementation.md):
   deploy a committed snapshot and validate the bounded instrument job before a
   full run. Read the log for an actual GPU, all gates, dtype checks, checkpoints,
   runtime and memory. Do not treat passing fake/local tests as Ada integration proof.
2. Only if the instrument gates and recorded cost/memory bounds pass, execute the
   frozen 128-feature/512-pair audit and analyse it by revision R5. Preserve its
   same-stored-weight controls, shared prefix cache and batching reference.
3. Use its result to decide whether the publication candidate earns further work.
   No automatic layer/model/task expansion. Update the report/deck only from
   validated outcomes with their scope and status.
4. Complete the course archive after the final report/deck revision and required
   team review; course deadline **31 October 2026**.

The old hard 32-GPU-hour ceiling was relaxed by the user (A8). The arithmetic audit
retains its separately recorded allowance of at most eight additional GPU-hours.
The study's recorded elapsed target is 28 October 13:53 IST; report any overrun
explicitly. The separate Gemma-2-9B branch remains outside this execution scope.

## Deliverables

- [Course report](../paper/course-report/README.md): revision 3, 11 pages;
  official checker recorded **All Clear**, validation manifest refreshed at
  27 October 04:37 IST and matched the source/PDF at cleanup.
- [Presentation](../paper/presentation/): 18-page final-mode build recorded;
  external/recovery results included. A final submission archive remains pending
  in the latest handoff. Team authorship/order and final wording still need review.
- [Novelty search](novelty_assessment_2026-10-26.md): dated literature assessment;
  new empirical evidence does not establish priority. The [research direction](../RESEARCH_DIRECTION.md)
  describes the remaining candidate. **An A-tier contribution is not yet established.**

## Maintenance and history

[Codebase map](../docs/CODEBASE.md) describes module ownership and compatibility
entry points. [Reports index](INDEX.md) maps all results, plans, decisions,
source registers and archives. Older status pages are historical ledgers/runbooks.
The [27 October archive](archive/modular_cleanup_2026-10-27/README.md) preserves
their full pre-cleanup text and file hashes; no experiment history was deleted.
