# GPTQ extension: ledger and execution log (D-101)

Started 2026-10-27 17:34 IST at the user's request ("run tests again with GPTQ quantization and try to keep other things same as before so that we can draw a fair comparison"; cron every 40 minutes). Plan: [`docs/superpowers/plans/2026-10-27-gptq-matched-comparison.md`](../../docs/superpowers/plans/2026-10-27-gptq-matched-comparison.md). Source audit: [`reports/quantization_gptq_audit_2026-10-27.md`](../quantization_gptq_audit_2026-10-27.md). Decision: D-101. Open question: Q30 (disk and scope). The user's HOLD of 2026-10-27 09:50 is lifted for this extension only (nothing else may start).

**Rules that matter most here:** keep everything except the quantizer as in the old experiments; three arms (legacy RTN = existing results, matched RTN, GPTQ); INT16, INT12 and INT0 stay the legacy rounding and restore controls; no result selects a setting; the stage order of the plan is fixed; an artifact is rebuilt in every job and refused unless its sha256 equals the frozen value; mask regime only; at most two Ada jobs running; Gemma-2-9B is another team's; never touch the user's `plan02-train` jobs; Ada host key: `ssh -o HostKeyAlias=ada-gw1 karma@10.4.24.30` (never accept a changed key).

## NEXT STEP

(Rewritten by every firing; the first line says which task of the plan comes next and what "done" looks like.)

**Task 1 of the plan:** vendor the authors' pinned GPTQ core into `scripts/g0/gptq_ext/upstream/` (hash-checked) with the two tests of the plan; done = tests pass, mutation check shown, commit pushed.

## Task table (the plan's task numbers)

| task | what | state |
|---|---|---|
| 0 | governance: D-101, Q30, plan, ledger, cron prompt, cron, handoff 048 | done 2026-10-27 |
| 1 | vendor the pinned GPTQ core + tests | not started |
| 2 | local CPU TransformerLens environment + canonical adapter | not started |
| 3 | grids and matched RTN | not started |
| 4 | calibration pool (Wikipedia, 27 languages) and its freeze | not started |
| 5 | sequential calibrator | not started |
| 6 | artifact identity and weight-state shim in the five entry points | not started |
| 7 | extension freeze (`freeze.json`) | not started |
| 8 | deploy, instrument jobs, gates, cross-node hash check | not started |
| 9 | stage 1: ordinary loss and survival (E1, E2) | not started |
| 10 | stage 2: transfer study (E6) | not started |
| 11 | stage 3: functional panels (E3, E4, E5, E7, F4, F5, Study G) | not started |
| 12 | stage 4: external panel | not started |
| 13 | comparison report, documentation, uploads | not started |
| 14 | close | not started |

## Ada jobs of this extension

| job | Slurm id | state | submitted | elapsed | GPU hours | note |
|---|---|---|---|---|---|---|

## Log (newest last; times from `date`)

- 2026-10-27 17:34 IST (Claude, the user's request): **Task 0 done.** Read the audit and the code; wrote the plan and D-101 and Q30; created this ledger and the cron prompt. Facts found: the five entry points call the injected `env["quantize_rtn_per_output_channel_tl"](model, bits=...)`, so one shim covers them (`phase1_run.py:312`, `phase1_func_run.py:484,619`, `phase1_func_f5_run.py:386`, `phase1_transfer_run.py:477`, `phase1_transfer_precision_main.py:175`); the local environment has torch but no TransformerLens (Task 2 adds a CPU environment); Ada: queue empty, `/home2/karma` 50 GB quota with 2.7 GB free, compute nodes have node-local `/scratch` and no `/share1`, the login node has internet and the `datasets` library; the six-language Wikipedia panel (`wikimedia/wikipedia` 20231101, revision b04c8d1c) is the text source to reuse for calibration.
- 2026-10-27 17:35 IST (Claude): **CRON CREATED**: three recurring entries `bf1ae581` (`1 0-22/2 * * *`), `ad8593e3` (`41 0-22/2 * * *`) and `8db9afdb` (`21 1-23/2 * * *`), pointer prompt of `cron_prompt.md`; printed next runs 18:31, 19:11 and 19:51, so the firings are 18:31, 19:11, 19:51, 20:31, ... exactly 40 minutes apart (the scheduler adds a fixed delay of 30 minutes to each listed time; 3 candidates created and deleted: `5e0e859d`, `39143488`, `0197a0fc`). They expire after 7 days (about 2026-11-03) and live only in this session. Plan, D-101, Q30, ledger, cron prompt and handoff 048 are committed and pushed (`1bb362c`).
