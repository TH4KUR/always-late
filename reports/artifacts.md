# Large artifacts (Hugging Face) and where everything lives

**Hugging Face dataset (private):** https://huggingface.co/datasets/karma-skz/always-late-artifacts  
Owner `karma-skz`; contents and provenance are in the dataset card (`reports/g0/hf_dataset_card.md`, uploaded as its `README.md`); `MANIFEST.sha256` there lists the SHA-256 of every file. Download (needs a token with read access): `hf download karma-skz/always-late-artifacts --repo-type dataset --local-dir artifacts`.

**Upload status:** started 2026-10-17 from Ada (225 files, 297 MB, one commit via `scripts/g0/hf_artifacts.py`); remote size verification is recorded below when done. **STATUS: UPLOADING / NOT YET VERIFIED** (the link from Ada is slow, about 70 kB/s).

## What is where

| artifact | git repo (small) | Hugging Face dataset (large) | other copies |
|---|---|---|---|
| Duan baseline, their code (job 23472) | `reports/g0/baseline_full/` (summaries, per-feature CSVs, report) | `baseline/full/` | Ada `/home2/karma/duan-verify/baseline/full` |
| smokes (23355, 23419, 23463) | `reports/g0/r0_smoke`, `baseline_smoke`, `baseline_smoke_vendored` | `smoke/*` | Ada |
| R1 (24451): their code, no BOS, INT0/16/12 | `reports/g0/R1_nobos_floor/` | `runs/R1-nobos-floor/` (per-feature CSVs, bootstrap snapshots, token windows, plots) | Ada `out/R1` |
| R2 (23643): their code, BOS masked | `reports/g0/R2_bos_mask/` | `runs/R2-bos-mask/` | Ada `out/R2` |
| R5a / R5b / R5c (23649, 23660, 24452): legacy HF harness cells | `reports/g0/r5/` (summaries, selfchecks, hashes, `r5_report.md`) | `runs/R5a`, `runs/R5b`, `runs/R5c` (per-feature CSVs; R5a also holds the 1953 x 256 token windows) | Ada `r5/` |
| wandb runs (15, fixed schema) | `reports/wandb_schema.md`, `reports/g0/wandb_manifest.json`, `reports/g0/wandb_backup_pre_schema.json` | per-feature CSVs are also in each wandb artifact | https://wandb.ai/karma-skz-/anlp-project |
| Phase 1 outputs (to come) | summaries and manifests | uploaded at job end by `hf_artifacts.py` | Ada |

Not stored anywhere because they are not ours: Gemma-2-2B and Gemma Scope weights (downloaded by each job to node-local scratch), WikiText-2 and FLORES-200 (public, re-downloadable; FLORES archive sha256 `b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6`).

Policy (decision D-012): large outputs go to the dataset repo, git keeps summaries, manifests and links. New launchers upload at job end with `scripts/g0/hf_artifacts.py upload ...`. The token stays on Ada (`HF_TOKEN_PATH`) and is never copied.
