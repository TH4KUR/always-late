# Stage 1 of the GPTQ extension: Phase 1 survival and loss with GPTQ and matched RTN (written 2026-10-28 04:34 IST)

Jobs 42951 (GPTQ) and 42952 (matched RTN): the old `p1-mask` run (`phase1_run.py --corpus flores --regime mask`, whole ladder 8 7 6 5 4 16 12 0) through `gptq_ext/run_with_weights.py`. Files:
- `stage1_comparison.md` (read this), `stage1_paired_per_language.csv`, `stage1_summary.csv`, `bridge.json`: `scripts/g0/gptq_ext/compare_stage1.py` on the three arms, paired by language (D-104).
- `analysis_gptq/`, `analysis_rtnm/`: the UNCHANGED `scripts/g0/phase1_analysis.py --run <arm>/eq --out <dir>` (defaults: 10,000 permutations, 2,000 bootstrap draws, seed 0, `OMP_NUM_THREADS=4`), the pre-set confirmatory family T1 to T3 (D-014) per arm (D-106). **Control:** the same command on the OLD `p1-mask/eq` run reproduces the committed `reports/phase1/mask/eq/analysis.json` and `analysis.md` BYTE FOR BYTE. The legacy arm is that committed result.
All contrasts and re-runs are exploratory (D-101).
