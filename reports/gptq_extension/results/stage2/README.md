# Stage 2 of the GPTQ extension: the intervention-transfer study with GPTQ and matched RTN (written 2026-10-28 03:54 IST)

The UNCHANGED `scripts/g0/phase1_transfer_report.py` (claim rules of amendment A14, 2,000 crossed-bootstrap replicates, seed 20261026) was run ONCE per arm on the validated jobs of that arm, before any outcome statistic was read:

```
OMP_NUM_THREADS=2 python scripts/g0/phase1_transfer_report.py --freeze reports/intervention_transfer/freeze.json --core reports/intervention_transfer/core_sample.json --out <transfer_runs/analysis_ARM> \
  --feature-dir INT16=<transfer-feature-int16> --feature-dir INT6=<ARM-transfer-feature-int6> --feature-dir INT4=<ARM-transfer-feature-int4> \
  --group-dir INT16=<transfer-group-int16> --group-dir INT6=<ARM-transfer-group-int6> --group-dir INT4=<ARM-transfer-group-int4> --survival <ARM-p1-mask/all/survival.pt>
```

`ARM` = `gptq` (jobs 42953, 42955, 42957, 42959; stage-1 survival of job 42951) and `rtnm` = matched RTN (jobs 42954, 42956, 42958, 42960; survival of job 42952). The INT16 floor directories are the old study's (the legacy rounding, identical in every arm). The legacy arm is the committed [`../../../intervention_transfer/results/main`](../../../intervention_transfer/results/main/README.md). **Built-in reproducibility check:** the INT16 rows of both new analyses equal the committed old rows exactly (component summary 315 rows, group table 27, specificity 9, coverage 10).

Files: `analysis_gptq/` and `analysis_rtnm/` (every output of the report except the three large `per_context_*.csv` files, which stay in `transfer_runs/`), and the comparison written by `scripts/g0/gptq_ext/compare_transfer.py`: `transfer_comparison.md` (read this), `transfer_claims.csv`, `transfer_feature_level.csv`, `transfer_group_paired.csv`, `transfer_group_summary.csv`. Decision: D-105. All contrasts are exploratory (D-101).
