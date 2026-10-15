# Duan (2026) upstream code, vendored for the Gemma Phase-3 replication

**Source.** *Perplexity Can Miss SAE Feature Damage Under Quantization*, arXiv:2606.03002. Artifact: `https://anonymous.4open.science/r/sae-feature-survival-quantization-0141/` (zip via `…/api/repo/sae-feature-survival-quantization-0141/zip`). MIT license (`pristine/LICENSE`, kept). Downloaded 2026-10-15 and again 2026-10-16: both extractions are byte-identical (36 files).
**Identity.** The zip's own hash is *not* stable (the server stamps files with the download time). Identity is the **content tree hash** of the 36 files: `a84e43dac17e5a9c6b1404e4b502c97d2b184b0168e860035432e4c261e44adc` (sha256 over the sorted lines `"<file sha256>  <path>\n"`), recorded with every file hash in `MANIFEST.sha256`.

## What was removed (not needed for this study; not touched otherwise)
Pythia scripts `00_smoke_test*.py`, `01_*`, `02_*`, `04_*`; `05_phase5_feature_class_predictor.py`; `src/` (metrics helper, unused by script 03); all Pythia/pruning-overlap result tables and figures; `.gitignore`, `environment.yml`, `requirements.txt`, `MANIFEST.json`. Their hashes stay in `MANIFEST.sha256` marked `removed`, so a fresh download can be checked against the full upstream tree.

## What was retained, byte-identical (`pristine/`)
`LICENSE`, `README.md`; **`scripts/03_gemma_phase3_streaming_sweep.py`** (the study); `scripts/06_gemma_sliding_window_ppl_check.py` (their sliding-window perplexity check, kept pristine for later; it does not parse as shipped and is not edited yet); their published Gemma results to compare against: `results/tables/phase3_summary_final.csv`, `results/tables/gemma_chunked_vs_sliding_ppl_w2048_s512_FULL.csv`, `results/per_feature/phase5b_gemma_rtn_int6_vs_pruning_per_feature_overlap.csv`.

## The file we run: `scripts/03_gemma_phase3_streaming_sweep.py` = pristine + exactly these edits
Nothing else differs, over the whole file including top-level statements (`patches/03_working_vs_pristine.diff`; enforced by `scripts/g0/tests/test_vendor.py`, which rebuilds the working copy from the pristine file plus the whitelist in `scripts/g0/verify_vendor.py` and requires byte equality, and carries a committed mutation suite).

| # | Edit | Why | Touches their logic? |
|---|---|---|---|
| tail | delete everything from `# %% Cell 22` on | those cells call `corrected_ppl_sweep`, `compute_ppl_shifted_manual`, `restore_work_from_ref`, which are never defined in the file; they run only after every result file is written (and re-define `target_tl_weight` without `W_gate`) | no: removed unused/unrunnable diagnostics |
| 1 | `RUN_PRUNING = True` -> `False` in their `3b_full` branch | their own flag; their pruning stage (matched-perplexity pruning) is not needed for this study and has never run in our setup | no: their switch |
| 2 | add `WORK_DEVICE = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE` | two bf16 models (6.4 GB each in TransformerLens) do not fit one 11 GB card; on a single large GPU this equals the shipped behaviour | device reference only |
| 3 | `device == "cuda"` -> `str(device).startswith("cuda")` in `load_gemma_tl_model` | their dtype test would silently give float32 for `"cuda:1"`; unchanged result for `"cuda"` | device-string comparison only |
| 4 | `load_gemma_tl_model([MODEL_NAME_USED])` -> `..., device=WORK_DEVICE)` | place `model_work` on the second card | device reference only |
| 5 | `model_work.to(DEVICE)` -> `model_work.to(WORK_DEVICE)` in `restore_model_work` | otherwise the restore moves it back to card 0 | device reference only |

**No edit for `display()`:** their notebook-only `display(...)` works when the script is run with IPython (`ipython script.py`), so the file is not touched for it. Mode `3a_test` (smoke) is produced at run time by replacing the single line `RUN_MODE = "3b_full"` in a *temporary copy*; the baseline job runs the vendored file byte-for-byte (the launcher verifies `working.sha256` first).

## Not verified
We never ran their unmodified script: it needs a >= 16 GB GPU (two bf16 models) and no such GPU is reachable. Equivalence of edits 2-5 to the shipped single-device behaviour is argued from the code and from bit-identical logits across the two cards, not demonstrated on a large GPU.
