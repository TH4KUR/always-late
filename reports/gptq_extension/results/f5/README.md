# F5 (per-feature sameness of function) under the three quantizers (D-108, exploratory)

Outputs of the UNCHANGED `scripts/g0/phase1_func_f5_analysis.py`, one directory per new arm (`gptq/`, `rtnm/`: `f5_results.json`, `f5_tables.csv`, `report.md`, `statements.md`). The legacy arm is the committed [`../../../functional_validation/results/f5/`](../../../functional_validation/results/f5/). Jobs 44072 (GPTQ) and 44073 (matched RTN): states FP, INT6, INT4 on the old 1,487-feature stratified subset (`--per-cell 12 --subset-seed 0`). Raw job files (`f5_stage_a.pt`, `f5_state_*.pt`, about 14 MB per job) stay outside git in `/home/karma/Documents/ANLP/project/transfer_runs/func_inputs/jobs/`.

Command per arm (run A = the old `f5-a` files FP and INT16, so the numerical floor is the legacy rounding in every arm; run B = the new job's files):
```
OMP_NUM_THREADS=4 ~/.venvs/anlp-torch/bin/python -B scripts/g0/phase1_func_f5_analysis.py --run <func_inputs/legacy_f5_a> --run <func_inputs/jobs/ARM-f5-int64> --out <func_inputs/analysis_ARM_f5>
```
Control (same command on the old files, states INT16, INT6, INT4): all 10 rows of its `f5_tables.csv` equal rows of the committed table. Validation of the jobs: `scripts/g0/gptq_ext/validate_func_job.py --task f5 --states FP 6 4 --reference-meta <legacy_f5_b/meta_f5.json>`. The statuses lost / retained and the F4 pairs behind S1 and S2 are the OLD ones; INT5 was not run (S3 undefined). Reading and limits: D-108 in [`../../../decisions_and_anomalies.md`](../../../decisions_and_anomalies.md); plain-language version: section 4.6 of [`../../RESULTS.md`](../../RESULTS.md).
