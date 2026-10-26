# Main results of the intervention-transfer study (Gemma-2-2B; written 2026-10-26 21:21 IST)

Produced ONCE by `scripts/g0/phase1_transfer_report.py` (code at commit `e99a457`, claim rules of amendment A14, 2,000 crossed-bootstrap replicates, seed 20261026) on the six validated main runs, after all six had been validated and before any outcome statistic was read:

```
OMP_NUM_THREADS=4 python scripts/g0/phase1_transfer_report.py --freeze reports/intervention_transfer/freeze.json --core reports/intervention_transfer/core_sample.json \
  --out <transfer_runs/analysis_main> --feature-dir INT16=<transfer-feature-int16> --feature-dir INT6=<transfer-feature-int6> --feature-dir INT4=<transfer-feature-int4> \
  --group-dir INT16=<transfer-group-int16> --group-dir INT6=<transfer-group-int6> --group-dir INT4=<transfer-group-int4r> --survival <phase1-p1-mask-all/survival.pt>
```

Inputs: the output directories of Slurm jobs 40399, 40400, 40401 (feature level INT4, INT6, INT16), 40403, 40514, 40404 (group level INT6, INT4, INT16), their sha256 in [`../../artifact_checksums.md`](../../artifact_checksums.md); the Phase 1 survival file (mask regime) in the same file. Files here: `claim_status.json` (status of every pre-declared claim with its scope), `analysis_summary.json` (every number), `coverage.csv`, `component_summary.csv`, `specificity.csv`, `averaged_fingerprints.csv`, `averaged_features_<state>.csv`, `predictive_validation.csv`, `survival_validation.csv`, `a3.csv`, `group_per_language.csv`, `figures/`. The per-item tables `per_context_<state>.csv` (26 MB each) stay outside git in `/home/karma/Documents/ANLP/project/transfer_runs/analysis_main/` (sha256 of `per_context_INT4.csv`: 9fd618f2697fab2674026725d48f7e0aa9e72a9c54526aeffdc3ea6d48b5370f). The reading of the numbers, with its limits, is in the ledger [`../../STATUS.md`](../../STATUS.md) (newest Log entry) and decision D-094.
