---
pretty_name: always-late artifacts (SAE feature survival under RTN quantization, Gemma-2-2B layer 12)
license: other
tags:
- sparse-autoencoders
- quantization
- gemma-2
- interpretability
---

# always-late: large run artifacts

Large artifacts of the ANLP project "Does compression forget language?" (team Always Late, IIIT-H). Code, documentation and small summaries live in the GitHub repository `TH4KUR/always-late` (see `reports/artifacts.md` there for the index and `reports/audit_our_vs_duan.md` for what the runs mean). This dataset holds what is too large for git: per-feature tables, bootstrap snapshots and token windows.

**Contents are derived statistics only** (per-feature correlations, firing rates and activation summaries of the Gemma Scope layer-12 16k SAE, perplexities, token ids of WikiText-2 windows). No model weights, no SAE weights and no generated text. Gemma and Gemma Scope remain under their own terms; WikiText-2 is CC BY-SA. The repository is private; check licences before making anything public.

| folder | what | Slurm job | code |
|---|---|---|---|
| `baseline/full` | Duan et al. (2026) Phase-3 script, vendored, 500k WikiText-2 tokens (256-token windows, **no BOS**), INT8..INT4 | 23472 | `third_party/duan2026` (working.sha256 `7f1e638a0508c3ca`) |
| `smoke/R0-v3-runner`, `smoke/generator-runner`, `smoke/vendored` | 20k-token smokes (INT8, INT6) | 23355, 23419, 23463 | diagnostic generator / removed faithful runner / vendored script |
| `runs/R1-nobos-floor` | their code, **no BOS**, INT0 / INT16 / INT12 (noise floor) | 24451 | `scripts/g0/make_duan_runners_v3.py`, runner hash `2c34fecaee1d2e32` |
| `runs/R2-bos-mask` | their code with **BOS prepended and masked**, INT0 / 16 / 12 / 8 / 7 / 6 | 23643 | same generator |
| `runs/R5a`, `runs/R5b`, `runs/R5c` | the discarded HF harness (git tag `legacy-pipeline-2026-10-16`) on their tokens, one factor flipped per cell: R5a = T, O, T_sdpa, T_bf16sae; R5b = T_rowquant, T_mask, T_keep, O_keep; R5c = O_mask | 23649, 23660, 24452 | `scripts/g0/r5_run.py` (tag), hashes in each `code.sha256` |

File types: `*_per_feature.csv` (feature_id, condition, layer, corr, firing_rate, mean_activation, max_activation, active, survived, damaged: Duan's schema), `*_summary.csv` / `phase3_summary_final.csv`, `weight_perturbation_INT*.json`, `snap_*.pt` (block-bootstrap snapshots written by the diagnostic generator, `snap_every`), `tokens_2d.pt` (token windows: `runs/R5a/tokens_2d.pt` is 1953 x 256 int64 without BOS, SHA-256 `23b44fad30303c6a...`; `runs/R2-bos-mask/tokens_2d.pt` has a BOS column; `runs/R2-bos-mask/tokens_2d.pt[:, 1:]` equals the R5 windows), `selfcheck.json`, `code.sha256`, `gpu_mem.log`, plots. `MANIFEST.sha256` lists the SHA-256 of every file.

Download (private repo, needs a token with read access): `hf download karma-skz/always-late-artifacts --repo-type dataset --local-dir artifacts`.
