# 9B result integration: Claude cron queue (D-125)

Queued 2026-10-29 16:32 IST by Codex. Direct user request: "move the 9b results to main, add to claude's cronjob".

## NEXT STEP

**Pending: review the imported original-RTN 9B outputs on CPU, then integrate supported findings with main.** Detailed tables and metadata are already in `results/`; provenance is in `import_provenance.json`. Do one bounded step per existing 40-minute firing. No new GPU job or paid Jarvis session is part of this queue.

## Tasks

1. **Source and instrument audit:** compare the execution freeze, metadata, all six result states and the original adapter at commits `d27a6a0`/`521f3f3`. Verify model/SAE/hook, sentence and feature-group identity, quantizer, gates and sample completeness. The uploaded `meta_lm_eval.json` lists only INT4 although the analysis has six states; audit per-state raw-file headers before claiming that metadata validates the entire ladder.
2. **CPU reproduction:** fetch only needed raw F5-independent Study G item outputs from private `karma-skz/always-late-artifacts`, prefix `gemma9b/runs/full/`, pinned artifact revision in `import_provenance.json`, into a directory outside Git. Use the existing analysis with the original settings; compare the regenerated tables and reported numbers, record hashes and every mismatch. Leave weights and large files out of Git.
3. **Claim audit and synthesis:** verify the source verdict's P1 confidence interval, retention denominator/aggregation, sample/item counts, P3a dose-order inequality, and P3b definition against raw outputs and the freeze. The detailed report gives increasing harm as the multiplier falls (0.3536 to 0.0884); the verdict/freeze ordering needs checking, not silently relabeling. Compare matching original-RTN 2B statistics, including uncertainty; separately state the missing 9B GPTQ and matched-RTN arms. Amplification does not identify the causal source of quantization damage.
4. **Record:** a reviewed 9B summary with uncertainty and verification status; next free D-entry and numbered handoff; dated additions to `reports/PROJECT_RESULTS.md` and `reports/CURRENT_STATUS.md`. Keep source files and past results unchanged; any correction is an addendum. Missing artifacts yield an explicit unresolved status, not "no 9B results" or a new GPU rerun.

## Constraints

The user has 3 Jarvis GPU-hours remaining plus Ada. This task authorizes analysis of existing results only. Do not start Jarvis, submit new Ada jobs, rerun completed RTN work, restart completed D-117 analyses, or change frozen criteria. Work on main; the surviving implementation branch is `gemma9b-confirmation`. The deleted adapter and v2 branch names are historical.

## Log

- 2026-10-29 16:32 IST: source report and execution freeze copied byte for byte; eight small artifact outputs downloaded at pinned revision; provenance saved; review queued in the existing shared cron prompt. Scientific reproduction has not yet been performed.
