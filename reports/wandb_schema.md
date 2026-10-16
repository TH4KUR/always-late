# wandb schema (v1): one vocabulary for every run

Why: the runs table showed 158 distinct summary keys over 15 runs, 82 of them in a single run, because the condition was part of the metric name (`INT7/median_corr`, `baseline/median_corr`, ...). Now the same metric has the same name in every run, and the condition is data (`bits`) or a fixed key suffix.

Code: `scripts/g0/g0_wandb.py` (schema constants, `condition_row`, `run_summary`, `normalize_config`, `G0Logger`), `scripts/g0/wandb_rebuild.py` (backup, rebuild from result files, verify, guarded delete), `reports/g0/wandb_manifest.json` (what was rebuilt), `reports/g0/wandb_backup_pre_schema.json` (metadata of the 15 runs that existed before). Tests: `scripts/g0/tests/test_g0_wandb.py`, `test_wandb_schema.py`; live check: `scripts/g0/selftest_wandb.py`.

## Config (every run has exactly these keys; unknown values are `n/a`)

| key | meaning |
|---|---|
| `experiment` | group of runs: `baseline`, `R1`, `R2`, `R5a`, `smoke`, ... |
| `cell` | variant inside the experiment (`full`, `T`, `O_keep`, `bos-mask`, ...); `n/a` if none |
| `stack` | `TL` (their TransformerLens script) or `HF` (the discarded HF harness) |
| `bos_regime` | `none`, `mask` (BOS prepended, position 0 excluded) or `keep` |
| `attn_impl` | `tl`, `eager` or `sdpa` |
| `sae_dtype` | `fp32` or `bf16` |
| `quantizer` | `duan` (their q/k/v grouping) or `ours` (per output row) |
| `n_tokens` | token budget of the run |
| `layer` | SAE layer |
| `model` | model id |
| `slurm_job` | Slurm job id, `n/a` if none |
| `source` | `live` (logged by the run) or `rebuilt-from-files` |
| `code_ref` | hash or tag of the code that produced the run |
| `schema_version` | integer, currently 1 |
| `config_extra` | everything that is not a schema key, as one JSON string (e.g. library versions, `replaces_wandb_run`) |

## Per-condition metrics (history; one step per condition; x-axis `bits`; hidden from the runs table)

| name | meaning |
|---|---|
| `bits` | x-axis of all metrics: INT bit-width; the unquantized bf16 baseline is `32` (16 would collide with INT16); INT0 is the restore-only control |
| `n_tokens` | positions entering the statistics |
| `n_active_features` | features with FP firing rate > 0.001 |
| `mean_corr` | mean Pearson r over active features |
| `median_corr` | median Pearson r |
| `survival_pct` | % of active features with r > 0.9 |
| `degraded_pct` | % with 0.5 < r <= 0.9 |
| `damaged_pct` | % with r < 0.5 |
| `perplexity` | perplexity (their loss arithmetic) |
| `loss` | mean loss |
| `ppl_delta_pct` | perplexity change vs the baseline, % |
| `weight_rel_l2` | relative L2 weight change of the rung (when measured) |

Table `conditions` (every run): columns `condition` + the metrics above, one row per condition. Progress and memory heartbeats use `sys/windows_done`, `sys/windows_total`, `sys/elapsed_s`, `sys/mem_<device>_gib` and are hidden from the runs table.

## Run summary = the runs-table columns (fixed list, 29 keys)

- `fp_perplexity`, `fp_n_active_features`: the unquantized anchors.
- For each rung in [16, 12, 8, 7, 6, 5, 4]: `survival_pct_intN`, `median_corr_intN`, `ppl_delta_pct_intN` (blank if the run did not measure that rung).
- Closure-protocol gates: `gate0_int0_identity`, `gate1_median_r_int16`, `gate1_median_r_int12`, `gate1_slope`, `gate1_verdict` (only where the ladder has INT0/INT16/INT12).
- `q1_pass`: baseline replication verdict (baseline run only).
- Tables allowed in a summary: `conditions`, `comparison_vs_published`, `per_feature_int6`.

## State (2026-10-17)

13 finished runs were rebuilt (R5c/O_mask was added on 2026-10-17, making 14) from the result files in this repo (manifest: `reports/g0/wandb_manifest.json`, with old and new ids) and verified before the 13 old runs were deleted: 146 summary values compared equal, every old artifact file present in the new run except `tokens_2d.pt` (stays on Ada) and `phase3_summary_partial.csv` (in the repo) where accepted. Runs: `baseline/full`, `smoke/R0-v3-runner`, `smoke/generator-runner`, `smoke/vendored`, `R2/bos-mask`, `R5a/{T,O,T_sdpa,T_bf16sae}`, `R5b/{T_rowquant,T_mask,T_keep,O_keep}`. Across them the summaries use 31 distinct keys (29 schema keys plus 2 table names); before there were 158. Still on the old logger until it finishes and is rebuilt: the live run of R1 (`339751bd`).

## Conventions

- Run name = `experiment/cell` (or just `experiment`), e.g. `baseline/full`, `R2/bos-mask`, `R5b/T_mask`; tags carry the experiment family.
- A rebuilt run has `source = rebuilt-from-files` and `config_extra.replaces_wandb_run` = the id of the run it replaced; creation times are the rebuild time, not the original run time (wandb cannot backdate).
- Adding a metric or rung means changing the constants in `g0_wandb.py`, bumping `SCHEMA_VERSION`, updating the tests and rebuilding; ad-hoc keys are not allowed (`wandb_rebuild.py verify` fails on them).
