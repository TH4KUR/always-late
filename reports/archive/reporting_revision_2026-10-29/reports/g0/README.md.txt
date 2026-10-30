# reports/g0: artifacts of the Duan (2026) baseline / diagnostics work

| File / dir | What | Provenance |
|---|---|---|
| `r0_smoke/` | summary CSV, run metadata and weight-perturbation JSONs of the first Ada smoke run (v3 runner, `3a_test`, 20k tokens, job 23355) | produced on Ada 2026-10-16; **diagnostic runner**, not the baseline |
| `baseline_smoke/` | `3a_test` smoke with the earlier (now removed) generator-based runner, job 23419, bit-identical to `r0_smoke/` (neutrality check). The vendored file's own smoke (job 23463, bit-identical) is in `baseline_smoke_vendored/` | produced on Ada 2026-10-16 |
| `bench_fit_result_2x2080ti_20261015.json` | fit/speed benchmark on 2 x RTX 2080 Ti (plain-HF stand-in on the ungated mirror) | `scripts/g0/bench_fit.py` |
| `ada_env_freeze_20261015.txt` | `pip freeze` of the Ada environment (torch 2.14.1+cu126, transformer-lens 3.9.0, sae-lens 6.54.5, transformers 5.19.0) | Ada |
| `duan03_modifications_v3.diff`, `duan06_modifications_v3.diff` | unified diffs of the **diagnostic** (v3) runners against Duan's originals | `scripts/g0/make_duan_runners_v3.py` |

Not stored here: model weights, `.pt` tokens/snapshots, wandb run folders (see `.gitignore`). The large files are in the private Hugging Face dataset https://huggingface.co/datasets/karma-skz/always-late-artifacts (index: `reports/artifacts.md`). Runs are tracked in wandb project `anlp-project` (entity `karma-skz-`). The baseline run will add `baseline_full/` (their output CSVs, `baseline_report.md/.json`, `gpu_mem.log`).

| `R2_bos_mask/` | R2 (job 23643): their TransformerLens code with a BOS token prepended and masked; ladder INT0/16/12/8/7/6, 500k tokens; summaries, weight-perturbation JSONs, run metadata (snapshots, tokens and per-feature files stay on Ada) | Ada 2026-10-16 |
| `r5/` | R5a (23649) and R5b (23660): the old HF pipeline on their tokens with one factor flipped per cell (T, O, T_sdpa, T_bf16sae, T_rowquant, T_mask, T_keep, O_keep), selfchecks, code hashes, `r5_report.md` (made by `scripts/g0/r5_compare.py`). The harness that produced it (`r5_run.py`, `r5_lib.py`) was removed from the tree; it is at the git tag `legacy-pipeline-2026-10-16`. R5c (O_mask) and R1 (their code, no BOS, INT16/INT12) are pending: Slurm 24452 and 24451 | Ada 2026-10-16 |
| `r5_smoke/` | smoke job 23648 (R5 code on Ada, 40 windows, cells T and O at INT8): `selfcheck.json`, summaries, `code.sha256` (hashes of the deployed code), key log lines. Not comparable to the 500k-token tables | Ada 2026-10-16 |
| `audit` (see `../audit_our_vs_duan.md`) | why our pipeline disagrees with theirs; R2 and R5 outputs will be added as `R2_bos_mask/` and `r5/` after the jobs finish (Slurm 23643 / 23648, 23649, 23652; HANDOFF section 11) | Ada, pending |
| `baseline_full/` | **the baseline**: their vendored script, `3b_full` (500k tokens, INT8..INT4, pruning off), job 23472, 4 h 08 min: their per-condition CSVs and per-feature files, `baseline_report.md/.json` (Q1 passed), `gpu_mem.log`; the wandb helper dir is git-ignored; the on-cluster `wandb_run_id.txt` was removed because that in-job upload failed and never created a run (the results are in wandb run `86f26788`, uploaded from the laptop) | Ada 2026-10-16 |
