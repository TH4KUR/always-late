# GPTQ extension: raw job outputs (private dataset, D-012; written 2026-10-28)

Raw outputs of the jobs of the GPTQ extension (D-101, `reports/gptq_extension/`): the same experiments as the main study with weights rounded in two other ways, **matched RTN** (`rtnm`) and **GPTQ** (`gptq`), for the three-arm comparison with the legacy RTN results. Small result tables, analyses and the plain-language results are in the git repository (`reports/gptq_extension/RESULTS.md`, `results/`); this folder holds what is too large for git. **No model or quantized weights, no text of the evaluation corpora and no secrets are stored here** (numbers, indices and metadata only).

| folder | what | jobs |
|---|---|---|
| `stage1/{gptq,rtnm}-p1-mask/` | Phase 1 (survival, loss) on the 27 languages, bits 4 to 8 and the legacy-rounding controls; `survival.pt`, `selectivity.pt`, per-language tables, `weights_manifest.json` (artifact hashes) | Slurm 42951 to 42952 |
| `stage2/{gptq,rtnm}-transfer-{feature,group}-int{4,6}/` | intervention-transfer study, main matrix (chunk files, metadata, gate records) | 42953 to 42960 |
| `analysis/transfer-main-{gptq,rtnm}/` | the unchanged transfer report per arm (`per_context_*.csv` item tables are the large files) | local analysis of the above |
| `stage3_f5_gain/` | functional study jobs: loss task and rescue (E4, E5), Belebele (E7), F5 (per-feature sameness of function), Study G (graded gain), per arm | 43862 to 43869, 43893 to 43896, 44072 to 44077 |
| `external/` | external Wikipedia panel (six languages), group level and (second upload) feature level | 44209 to 44216 |

Every job rebuilt its quantized weights from the full-precision model and refused to run unless the artifact sha256 equalled the frozen value (`weights_manifest.json` in each folder; the recorded hashes are in `reports/gptq_extension/artifact_hashes.json`). Full-precision and INT16 reference files of the old study are NOT repeated here (they are identical in every arm and live in the main dataset). `MANIFEST.sha256` of the first uploads is untouched; this folder has its own manifest files (`MANIFEST*.sha256` next to this README). Upstream: GPTQ is the authors' implementation (IST-DASLab/gptq, pinned commit `2d65066e`, Apache-2.0); Gemma-2-2B and Gemma Scope weights are not stored here.
