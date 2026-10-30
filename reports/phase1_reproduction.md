# Clean-room reproduction of the Phase 1 result files (decision D-026)

Verdict: **REPRODUCED** (89 of 89 compared files identical).

Every committed result file under `reports/phase1/` was regenerated from the cached run files (the Ada outputs fetched with `fetch_phase1.sh`) by the committed scripts at commit `0a71b35`, in a fresh directory outside the repository, with `OMP_NUM_THREADS=4` and `MKL_NUM_THREADS=4` (A-036: other thread counts move the mixed-model coefficients at about 1e-10), on 2026-10-19, Python/numpy/scipy/torch 3.13.13 2.5.4 1.18.1 2.14.1+cpu. **Comparison rule (fixed before the run):** a regenerated file is reproduced if it equals the committed one byte for byte; JSON files are compared after dropping path-valued `run` keys, numbers exactly; the committed `per_language.csv`, `meta.json` and `summary.csv` are Ada outputs and are compared with the cached fetched copies instead of being regenerated. The generated summary is checked with `phase1_summary.py --check`.

| stream | compared | identical |
|---|---|---|
| all | 1 | 1 |
| mask/all | 23 | 23 |
| mask/eq | 23 | 23 |
| none/all | 21 | 21 |
| none/eq | 21 | 21 |

## Current reporting pointer, 30 October 2026

[Current synthesis](FINAL_RESEARCH_QUESTIONS.md), [research paper](../paper/research-paper/main.tex), [current status](CURRENT_STATUS.md). This page retains its technical or dated historical role; the final synthesis uses the completed evidence through 29 October, with 9B GPTQ pending. The reporting revision does not change experiments, frozen rules, numerical outputs or reproduction code.
