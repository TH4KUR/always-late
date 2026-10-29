# Handoff 056: 9B existing results on main; cron review queued

Written 2026-10-29 16:32 IST. User: "move the 9b results to main, add to claude's cronjob" (D-125).

The single implementation branch is `gemma9b-confirmation`; the redundant adapter and v2 branches were consolidated and deleted. Main now contains [the source report and detailed tables](../functional_validation/gemma9b/README.md), the completed RTN execution freeze and import hashes. GPTQ/matched-RTN 9B results remain missing. The report is copied, not independently reproduced.

The existing Claude 40-minute jobs already read the first fenced block of [their shared prompt](../gptq_extension/cron_prompt.md); the new first block routes them to [the review ledger](../functional_validation/gemma9b/REVIEW_STATUS.md). No new schedule or session was created. D-117 remains complete. Next bounded step: CPU audit of run metadata and source claims, then reproduce from the existing raw artifacts. No new GPU jobs or paid-machine startup under this request.

Known points to verify: the shared uploaded metadata lists only INT4, though the analysis reports six states; the source verdict's P1 interval, P3a inequality, retention denominator and item counts need raw checks. Preserve the original report and record corrections separately. The user has only 3 Jarvis GPU-hours left plus Ada. No paper/slides edits were requested here.
