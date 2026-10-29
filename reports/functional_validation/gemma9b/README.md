# Gemma-2-9B results on main

Imported 2026-10-29 at the user's request (D-125). **These are completed original-RTN results. GPTQ and matched-RTN 9B results are still missing.** Runtime implementation remains on the single `gemma9b-confirmation` branch.

## Files and status

- [Pushed verdict](verdict.md): source commit `0c3d3c7`, preserved from consolidation commit `34c6d11`. It reports that all five earlier confirmation predictions passed; independent reproduction is pending.
- [Execution freeze](legacy_rtn_run_freeze_2026-10-29.json): the completed run's original freeze, byte for byte. It does not freeze the pending GPTQ runs.
- [Detailed analysis](results/analysis/report.md), [analysis JSON](results/analysis/analysis.json), [curves](results/analysis/curves.csv), [reliability](results/analysis/reliability.csv), [best gain factors](results/analysis/alpha_star.csv), [run metadata](results/meta_lm_eval.json), [feature groups](results/manifest.json), [tagging metadata](results/eq/meta.json). These eight small outputs were copied from the private artifact store at a pinned revision.
- [Import provenance](import_provenance.json): source commit, artifact revision and SHA-256 for every copied file. No weights or raw tensor files were added to Git.
- [Review queue](REVIEW_STATUS.md): the existing Claude cron's next task.

## Reported observations, awaiting the raw-output review

The source reports positive full-precision language-group ablation effects in 27/27 languages; relative ablation harm 1.003 at INT8, 1.006 at INT6 and 2.479 at INT4; ordered down-scaling harm in 27/27 languages; no reliable repair from boosting the group (8/27 languages meet the group-specific repair criterion, against a threshold of 14). These ratios describe the harm caused by removing features, not feature survival rates or model accuracy.

Amplified ablation harm does not establish that the original quantization damage is concentrated in language features. A clean replication of all current main-branch findings is not established by this original-RTN-only report. Missing uncertainty and incomplete metadata must be checked before promoting claims.

## Review notes 2026-10-29

The Study G analysis was reproduced byte-identically from the raw per-state files (task 2 of the review, [`REVIEW_STATUS.md`](REVIEW_STATUS.md)). Open problems of meaning, not arithmetic: [anomaly A-043](../../decisions_and_anomalies.md) (the frozen P3a ordering as written holds in 0 of 27 languages while the reversed ordering holds in 27 of 27; P3b was relabelled; the INT4 retention 2.479 is a mean, the median is 1.586). The adapter `phase1_gemma9b_run.py` named in the freeze exists only on branch `gemma9b-confirmation`, not on main. A GPTQ-only recovery run is prepared on that branch (code and tests, amendment A2, plan `RECOVERY_PLAN.md`) and has NOT been run.

**2026-10-29 18:33 IST:** the review is complete: the reviewed statement with intervals and the 2B comparison is [`REVIEWED_SUMMARY.md`](REVIEWED_SUMMARY.md) (decision D-127); the scripts and outputs are in [`review/`](review/). The files above stay unchanged.
