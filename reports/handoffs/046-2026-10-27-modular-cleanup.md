# Handoff 046: modular code and documentation cleanup, 27 October 2026

User request: another documentation/code cleanup, modular structure, preserve important files.
This was maintenance; no new experiment, job, scientific threshold or report prose was changed.

## Current entry points

- [CURRENT_STATUS.md](../CURRENT_STATUS.md) is the one current-state page. Update it
  after each completed execution/analysis step; use the older STATUS files as ledgers.
- [Codebase map](../../docs/CODEBASE.md) maps ownership and the module boundary.
- [Research direction](../../RESEARCH_DIRECTION.md) separates interpretation from job status.
- [Handoff 045](045-2026-10-27-external-panel-analysed-report-deck-final-recovery-plan-next.md)
  retains the scientific updates and exact precision-audit execution instructions.

The research snapshot remains handoff 045 through 09:15 IST, `e17f926`: main,
external and CPU recovery complete; precision implementation built, deployment/instrument
validation next. This maintenance step did not query or modify Slurm state.

## Code change

Local main commit `c72a090`: five completed follow-up implementations now live under
`scripts/g0/transfer_followup/`, with shared typed CSV parsing, ordered CSV writing and
SHA256 in `tables.py`. All five original files remain as compatibility modules and
CLI entry points, including private function access and shared module attribute state.
Frozen primary analysis/runners and all precision modules are unchanged by this cleanup.

Existing deployment archives the entire `scripts/g0` directory, so no deployment
path change was needed. A smoke test of that committed archive ran all five old
commands with `--help` from a different working directory. Source mutation checks
for future changes should target the implementations, not the compatibility files.
Historical mutation records retain their original commit/path references.

## Documentation and preservation

The README, handoff/runbook, PLAN/PROGRESS pointers and both older status pages now
route to the current page. The cron instruction source was updated only to route
current work there, without changing scheduling or job authorization. Result/novelty
pages keep their dated scientific text with pointers to later evidence.

Full pre-edit bytes of 12 pages, including earlier uncommitted updates, are in
[the 27 October archive](../archive/modular_cleanup_2026-10-27/README.md). Its manifest
records source paths, byte counts and SHA256 plus hashes of 176 protected files.
The earlier 26 October archive remains untouched. Original plans/specs, freezes,
locked fits, main/external/recovery results, report/presentation files and complete
historical ledger tails were verified unchanged. No important file was deleted.

## Verification

- Existing transfer suite: 476 tests before extraction; 484 tests after extraction
  and again after integration, including eight new compatibility/I/O contract tests.
- Existing stdlib suite: 390 tests pass.
- All five real-data follow-up analyses rerun with their original 2,000 bootstrap
  replicates: all 13 JSON/CSV files byte-identical before/after.
- All five legacy command help outputs unchanged.
- Snapshot checksums, protected-file hashes, original main-freeze input hashes,
  documentation links and whitespace checked. The staged whitespace check disables
  only blank-at-EOF warnings to preserve original historical/snapshot bytes. See the
  [code verification record](../../docs/maintenance/2026-10-27-code-verification.json)
  and [maintenance plan](../../docs/maintenance/2026-10-27-cleanup-plan.md).

Tests emitted the same existing resource/numerical warnings on the baseline; no
new failure was introduced. No dependency was installed and nothing was pushed.

## Next work

Resume the precision audit from the latest scientific handoff and current status.
Do not rerun the completed external matrix or CPU recovery analyses as new research.
Use the existing protocol/gates and report only validated outcomes. Finish the course
archive after final report/deck review. Preserve negative results and claim limits.
