# HANDOFF: building on Duan (2026)'s OWN code

Written 2026-10-16, rewritten the same night after the audit and the decision to **discard our previous codebase**. For an agent with no memory of the previous sessions. Read this file fully, then the "POST-MORTEM of our previous codebase" entry at the end of `reports/progress_log.md`, `reports/audit_our_vs_duan.md` (what was found), `third_party/duan2026/UPSTREAM.md` (exactly what was removed and edited) and `reports/duan_codebase_audit.md` (Ada facts, closure protocol). Items marked **UNVERIFIED** were not verified.

## 0. Mission and the rules that matter

**Where we are.** Duan et al.'s own Phase-3 script is vendored and reproduces their published Gemma table on Ada (within 0.15 pp). The audit showed that our earlier pipeline disagreed with theirs because of a **regime difference** (their windows have no BOS token; ours had one), not a bug. Our old pipeline (`src/`, Phase-1 scripts) was **discarded**: recoverable only from the git tag `legacy-pipeline-2026-10-16`. Everything new is built on their code.

**Mission now.** (1) Finish the two pending diagnostics (section 5). (2) Get the team's decisions (section 9), above all which regime (BOS or no BOS) the study uses. (3) If the team chooses, rebuild the language-selective analysis as a small, labelled diff on top of their script, verified against the baseline before any new claim.

**The rule.** We run **their code, in their repo layout**. `third_party/duan2026/` is their repository downloaded fresh, pruned to what this study needs, with the working copy of the one script we run differing from their file by a **short explicit whitelist** and nothing else:

| What | Detail |
|---|---|
| Removed (not needed) | Pythia scripts, the feature-class predictor, `src/`, Pythia/pruning-overlap results and figures, env files. Hashes of every removed file stay in `MANIFEST.sha256`. |
| Retained byte-identical (`pristine/`) | `LICENSE`, `README.md`, `scripts/03_gemma_phase3_streaming_sweep.py` (the study), `scripts/06_gemma_sliding_window_ppl_check.py`, and their published Gemma results (`phase3_summary_final.csv`, `gemma_chunked_vs_sliding_ppl_w2048_s512_FULL.csv`, the INT6 per-feature file). |
| Working copy = pristine script + | (tail) delete their cells 22+, which call three functions never defined in the file and run after all results are written; (1) `RUN_PRUNING = False`; (2) `WORK_DEVICE = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE`; (3) `device == "cuda"` -> `str(device).startswith("cuda")` in their dtype test; (4) `load_gemma_tl_model(..., device=WORK_DEVICE)`; (5) `model_work.to(WORK_DEVICE)`. Edits 2-5 are device references forced by 11 GB GPUs. |
| No edit for `display()` | their notebook-only `display(...)` works under `ipython script.py`. |

**Do not edit the working copy by hand, and do not add edits.** The whitelist lives in `scripts/g0/verify_vendor.py` (`EDITS`, `TRUNCATE_AT`). `python3 -m unittest discover -s scripts/g0/tests -v` must print `OK` before any run (54 tests: exact reconstruction of the working copy, a committed mutation suite, report/wandb tooling). Anything beyond the baseline (BOS arms, extra bit-widths, new study code) is a **diagnostic / modified** experiment, must be labelled so and never called a replication. New study code must be a diff against their pristine script (the pattern in `scripts/g0/make_duan_runners_v3.py`), with tests, and must reproduce the baseline when its new switches are off.

**Honest limit (UNVERIFIED).** We never ran their *unmodified* script (needs >= 16 GB; none reachable). That edits 2-5 do not change results is argued from the code and from bit-identical logits across the two cards, not demonstrated on a large GPU.

## 1. Why this exists (short)

The team's own pipeline gave INT7/INT6 survival of 98%/94% where Duan reports 81%/51%. After running their code (reproduces) and a factor-by-factor audit (`reports/audit_our_vs_duan.md`): with one BOS token their own pipeline gives 99.97% / 97.5% (their TransformerLens code and an HF reimplementation agree to 0.1 pp); perplexity is 457 without BOS and 16 with it. Their quantizer's shared q/k/v scales cost ~8 / 3.5 pp but only without BOS; HF's sdpa attention path and a bf16 SAE make no measurable difference. The old code, its flaws and the verdict on whether its direction was worth it are in the post-mortem entry. The paper and README of that period contradicted the team's own data (`issues.md` A1-A6); the paper must be rewritten **by the team** (course policy forbids AI-generated report text; do not write paper prose).

## 2. State of play (2026-10-16, night)

| Item | Status |
|---|---|
| Review of the old code and results, `issues.md` | done (kept as evidence of the discarded work) |
| Their repo vendored and pruned (`third_party/duan2026/`, content tree hash `a84e43da...adc`), 54 tests | done |
| Baseline: their `3b_full` (500k tokens, INT8..INT4, pruning off), job 23472 | **done; reproduces their table** (section 8) |
| Audit of why the old pipeline disagreed: `reports/audit_our_vs_duan.md` | **done**: R2 (their code + masked BOS) and R5 (8 cells, old HF pipeline on their tokens) |
| Old codebase | **discarded** (tag `legacy-pipeline-2026-10-16`); Phase-1 results kept in `reports/legacy_pipeline/` as hypotheses |
| Pending diagnostics: R1 (their code, no BOS, INT0/16/12) and R5c (`O_mask`) | **queued/running** (section 5) |
| Closure protocol (`duan_codebase_audit.md` section 7) | **not yet assignable**: needs Gate 1 in the no-BOS regime (R1) and Q4 in the mask/keep regimes (R5c, R3 missing); see audit section 6.4 |
| Study design on top of their code | **not started**; waits for the team's regime decision |

## 3. Runbook for the baseline (done once; keep for re-runs)

Commands from the repo root. Ada: `ssh karma@ada-gw1` (key auth works from the user's laptop; `ada.iiit.ac.in` is closed).

1. **Pre-flight (local).** `python3 -m unittest discover -s scripts/g0/tests -v` -> `OK`; `python3 scripts/g0/verify_vendor.py --verify` -> `pristine problems: none | working copy == pristine + whitelist: True`. To re-verify against upstream: download `https://anonymous.4open.science/api/repo/sae-feature-survival-quantization-0141/zip`, extract with a path-safe extractor into an empty directory, run `python3 scripts/g0/verify_vendor.py --build <that dir>`, then `git status third_party/duan2026` must show no change.
2. **Deploy.** `V=third_party/duan2026; scp $V/scripts/03_gemma_phase3_streaming_sweep.py karma@ada-gw1:/home2/karma/duan-verify/vendor/scripts/; scp $V/working.sha256 karma@ada-gw1:/home2/karma/duan-verify/vendor/; scp scripts/g0/run_baseline.sbatch scripts/g0/baseline_report.py scripts/g0/g0_wandb.py karma@ada-gw1:/home2/karma/duan-verify/`. The launcher refuses to run if the deployed script's sha256 does not match `working.sha256`.
3. **Pre-flight (Ada).** `sinfo -N -p u22 -t idle -h -o "%N %f" | grep -c 2080ti` (>= 1), `df -h /home2/karma` (>= 12 GB free), `squeue -u karma` (the user's own `plan02-train` job is not yours: never touch it), outbound network (`curl -sI https://huggingface.co`).
4. **Submit.** `ssh karma@ada-gw1 'cd /home2/karma/duan-verify && sbatch -J base-full run_baseline.sbatch full full'` (~4.1 h compute + 10-20 min model download). No resume.
5. **Watch / collect / judge.** `squeue -u karma`; logs in `/home2/karma/duan-verify/logs/`; results in `/home2/karma/duan-verify/baseline/full/`; judge with `scripts/g0/baseline_report.py` (pre-committed bands; Q1 passes iff INT7 survival within 5 pp of 81.26%, INT6 within 10 pp of 51.30%, active features within 10% of 7,144, perplexity signs hold).

## 4. Environment facts that cost us time (all verified)

- **Cluster.** Slurm. Accounts `nlp` (QOS normal, no per-user cap) and `research` (QOS medium: max 4 GPUs, **40 CPUs**, 125 GB per user, 1 node per job). Only partition `u22` is open. GPUs reachable are 11 GB-class only: GTX 1080 Ti (gnode001-042) and RTX 2080 Ti (gnode044-092, `--constraint=2080ti`). Turing has no native bf16: PyTorch emulates it (correct, ~2x slower).
- **CPU cap trap.** `-c 8` with `--mem=40G` is inflated by Slurm to 14 CPUs (and 18 for the 2-GPU jobs), so several jobs on `research` hit `QOSMaxCpuPerUserLimit`. `-A nlp` starts immediately. Check `scontrol show job <id> | grep NumCPUs`.
- **Filesystems.** `/home2/karma` (NFS, ~19 GB free, **shared with the user's other jobs**): keep the model off it. **`/share1` is NOT mounted on compute nodes** (3,000-file quota). Node-local `/scratch` holds the 10.5 GB gated model per job (deleted at job end). Never write into shared areas you do not own.
- **Software.** `/home2/karma/duan-verify/env`: Python 3.10, torch 2.14.1+cu126, transformer-lens 3.9.0, sae-lens 6.54.5, transformers 5.19.0, wandb 0.30.0, ipython 8.39.0 (freeze in `reports/g0/`). Their `requirements.txt` was unpinned; these versions are part of the result.
- **Hugging Face.** The gated `google/gemma-2-2b` needs a token already stored on Ada (`/home2/karma/duan-verify/hf/token`; jobs use `HF_TOKEN_PATH`). **Never copy or print a token.** **Do not set `HF_HUB_DISABLE_XET=1`** (15x slower, drops connections).
- **wandb.** Key in `~/.netrc` (account `karma-skz`, entity `karma-skz-`), project `anlp-project`. Never ask for or handle a key.
- **Measured costs (TransformerLens, 2 x 2080 Ti).** 40-45 min per condition at 500k tokens, peak GPU 8.2 GiB per card, host RSS 21.8 GB.
- **Fetching results.** rsync filters: **excludes must come before includes** (first match wins); a transient DNS failure on the laptop can leave an empty directory while the loop still echoes "fetched": verify fetches by listing the files.
- **Tests.** Python reuses stale `.pyc` for same-size edits within one second: run mutation checks with `python -B` after clearing `__pycache__`.

## 5. Diagnostics on THEIR code (modified; always labelled) and pending jobs

Tooling: `scripts/g0/make_duan_runners_v3.py` (reads the vendored *pristine* scripts; BOS arms `none`/`prepend_keep`/`prepend_mask`, INT0 restore-only control, ladder, snapshots, resume, wandb heartbeats), `scripts/g0/run_g0.sbatch <arm> <none|prepend_keep|prepend_mask> [bits] [tokens] [mode]` (2 x 2080 Ti), `scripts/g0/g0_wandb.py` (logger and `compute_gate_metrics`), `scripts/g0/r5_compare.py` (rules of the audit as code; reads summary CSVs only). Deployed copies on Ada: `/home2/karma/duan-verify/duan03_runner_v3.py` (hash prefix `2c34fecaee1d2e32`), `run_g0.sbatch`, and the legacy harness copy `r5code/` (used only by R5c).

| Job | What | Ada output | Log |
|---|---|---|---|
| **24451 `R1-nobos-floor`** (`research`, 2 GPUs, ~3 h) | their code, **no BOS**, ladder INT0/16/12, 500k tokens: Gate 1 (noise floor) in THEIR regime | `/home2/karma/duan-verify/out/R1/` | `logs/R1-nobos-floor-24451.out` |
| **24452 `R5c-nlp`** (`nlp`, 1 GPU, ~45 min) | `O_mask`: the legacy HF pipeline with BOS masked, INT8/7/6 (completes Q4 for the mask regime); runs the deployed legacy harness, unaffected by the repo cleanup | `/home2/karma/duan-verify/r5/R5c/O_mask/` | `logs/R5c-nlp-24452.out` |

Already done and fetched: R2 (`reports/g0/R2_bos_mask/`), R5a/R5b (`reports/g0/r5/`), smoke (`reports/g0/r5_smoke/`).

## 6. Do-not list

- Do not edit `third_party/duan2026/` by hand, add edits, change the baseline constants, or change the comparison bands. Do not describe diagnostics as replication.
- Do not resurrect the legacy pipeline into the working tree. If something from it is needed, check out the tag and port the *idea* as a tested diff on their code.
- Do not edit `paper/` prose or write report text.
- Do not touch the user's `plan02-train` jobs, other users' files, `~/.netrc`, `~/.env`, `~/.ssh`, HF tokens. Do not fill `/home2`. No long jobs on the login node.
- Do not commit data, weights, `.pt` files, per-feature CSV dumps or secrets. Commit small CSV/JSON/MD results under `reports/g0/`.

## 7. File map

| Path | What |
|---|---|
| `HANDOFF.md`, `PLAN.md`, `PROGRESS.md`, `reports/progress_log.md` | handoff, plan, status, dated evidence log (ends with the post-mortem of the old codebase) |
| `issues.md` | review of the discarded pipeline and paper |
| `reports/audit_our_vs_duan.md` | why the old pipeline disagreed with theirs: 12 differences, experiments, pre-committed rules, results (sections 6.1-6.4) |
| `reports/duan_codebase_audit.md` | audit of Duan's repo, verified Ada facts, closure protocol (section 7) |
| `reports/wandb_schema.md`, `scripts/g0/g0_wandb.py`, `wandb_rebuild.py`, `reports/g0/wandb_manifest.json`, `wandb_backup_pre_schema.json` | the fixed wandb schema, its code and tests, what was rebuilt, and the backup of the old runs' metadata |
| `reports/g0/` | `baseline_full/`, `baseline_smoke*/`, `R2_bos_mask/`, `r5/` (+ `r5_report.md`), `r5_smoke/`, benchmark, env freeze, runner diffs |
| `reports/legacy_pipeline/` | Phase-1 results of the discarded pipeline: hypotheses, not findings |
| `third_party/duan2026/` | **their code**: `UPSTREAM.md`, `MANIFEST.sha256`, `pristine/`, `scripts/` (working copy), `patches/`, `working.sha256` |
| `scripts/g0/` | `verify_vendor.py`, `run_baseline.sbatch`, `baseline_report.py`, `make_duan_runners_v3.py`, `run_g0.sbatch`, `unit_test_runner.py`, `g0_wandb.py`, `selftest_wandb.py`, `bench_fit.py`, `r5_compare.py`, `tests/` |

## 8. Baseline result (job 23472, completed 2026-10-16)

**Q1 passed on all four pre-committed bands.** Outputs: `reports/g0/baseline_full/`. wandb run `https://wandb.ai/karma-skz-/anlp-project/runs/86f26788`.

| condition | ppl (ours / published) | survived % (r>0.9) | median r | active features |
|---|---|---|---|---|
| FP16 baseline | 458.4 / 458.5 | 100 / 100 | 1.0000 / 1.0000 | 7,134 / 7,144 |
| RTN INT8 | 469.9 / 470.6 | 99.12 / 99.09 | 0.9707 / 0.9707 | same |
| RTN INT7 | 432.7 / 432.6 | 81.19 / 81.26 | 0.9499 / 0.9499 | same |
| RTN INT6 | 476.8 / 476.8 | 51.15 / 51.30 | 0.9027 / 0.9027 | same |
| RTN INT5 | 539.7 / 539.8 | 27.57 / 27.72 | 0.7775 / 0.7780 | same |
| RTN INT4 | 670.4 / 669.8 | 12.39 / 12.40 | 0.4504 / 0.4512 | same |

Largest survival gap 0.15 pp; per-feature Spearman at INT6 over 7,121 commonly active features 0.99965. Job 23472 on gnode057: 4 h 08 min, exit 0, peak GPU 8,183 / 8,009 MiB. Not demonstrated: that our four device edits equal their unmodified single-device run.

## 9. Open decisions for the user / team

1. **Which regime does the study use?** Their no-BOS windows (perplexity 457, INT7 improves perplexity: a degraded regime) or BOS (perplexity 16, survival >= 97% at INT6)? This decides how the paper treats Duan's numbers and whether INT7/INT6 can still be confirmatory (in the BOS regime they are at ceiling; INT5/INT4 have the headroom and were never measured with BOS in a verified setup).
2. Rebuild the language-selective analysis (FLORES panel, selectivity, tiers) as a diff on their code, or stop at the audit result ("survival depends on BOS handling by 18-46 pp at INT7/INT6") and write that up? Deadline for the final submission: Oct 31.
3. Who writes the rewritten results/discussion (must be the team)?
4. Optional: R3 (their TL code, BOS kept), a small-sample cell for the 2,048-token demo residual, INT5/INT4 with BOS.

## 10. NEXT PROMPT: check R1 and R5c, fetch, judge, finish the audit

1. `ssh karma@ada-gw1 'squeue -u karma; sacct -j 24451,24452 -o JobID,JobName,State,Elapsed,ExitCode'`; read the log tails (`tail -c 3000 ... | tr '\r' '\n'`). If a job died: fix test-first or resubmit with the same out-name (R5c skips finished cells; R1 has resume), never touch the user's other jobs.
2. Fetch small files only, excludes before includes, then **verify by listing**: R1 `out/R1/` -> `reports/g0/R1_nobos_floor/` (summaries, `weight_perturbation_*.json`, `run_meta.json`; not `*_per_feature.csv`, `snap_*`, `*.pt`); R5c `r5/R5c/` -> `reports/g0/r5/R5c/` (summary, `cell_done.json`, `selfcheck.json`, `code.sha256`).
3. **Gate 1 in their regime** (audit section 7.2): rows = R1's FP/INT0/INT16/INT12 conditions plus the vendored baseline's INT8/7/6 (identical code and tokens; bit-identical smokes), perturbation sizes from R2's `weight_perturbation_INT*.json` (same weights and quantizer), `g0_wandb.compute_gate_metrics`; pass iff median r at INT16 and INT12 >= 0.99 and slope in [0.5, 1.5]. Gate 0: R1's INT0 must be identical to FP.
4. **Q4, mask regime**: `python3 scripts/g0/r5_compare.py --r5 reports/g0/r5 --baseline reports/g0/baseline_full/phase3_gemma_outputs_3b_full_500k_L12/phase3_summary_final.csv --published third_party/duan2026/pristine/results/tables/phase3_summary_final.csv --r2 reports/g0/R2_bos_mask/phase3_summary_final.csv --out reports/g0/r5/r5_report.md` (it now also compares `O_mask` with `T_mask`); the per-feature Spearman at INT6 of Q4 needs the per-feature CSVs on Ada (compute there with the Ada env or fetch the two files to the scratch directory, not into the repo).
5. Assign a disposition from audit section 7.3 only if its inputs exist (R3, their TL keep arm, is still missing: say so), fill audit sections 6.3-6.4, add a progress-log entry, update `PLAN.md`/`PROGRESS.md`/this file, run the tests, push.
6. Report to the user: Gate 1 verdict in their regime, Q4 for the mask regime, what is still open. No paper prose.

## 11. Scheduled hourly check (set up 2026-10-17 at the user's request)

A recurring job (id `5a2e98c0`, cron `17 * * * *`, local time; it replaced the first version `9bc324b1` when the wandb schema was approved) runs inside the Claude Code session: it checks R1 24451 and R5c 24452 on Ada, fetches and judges results, works through the source-verification queue in `reports/decisions_and_anomalies.md`, updates the docs and the plan, and commits and pushes when something changed. It does **not** queue new Ada jobs, edit `paper/` or change the vendored code. Its only wandb duty: when R1 and R5c finish it adds them to the manifest, rebuilds them under the fixed schema, verifies, deletes their two old live runs, and deploys the schema-aware logger to Ada (section 12).

Limits: it is **session-only** (it dies when Claude Code exits, nothing is stored on disk), it **auto-expires after 7 days** (2026-10-24), it only fires while the session is idle, and the harness delays each run by up to 30 minutes after the nominal minute.

**To stop it:** tell Claude "stop the scheduled job" (it runs `CronDelete 5a2e98c0`; `CronList` shows what is scheduled), or exit the Claude Code session. A job run already in progress finishes its current step.

## 12. wandb: one fixed schema (since 2026-10-17)

Every run, live or rebuilt, uses the vocabulary in `reports/wandb_schema.md` (15 config keys, 12 per-condition metrics with the condition as data in `bits`, 29 fixed summary keys, 3 fixed table names). The old per-condition-prefixed names (`INT7/median_corr`, `baseline/...`) no longer exist. Rules: never log ad-hoc keys (`python3 scripts/g0/wandb_rebuild.py verify` fails on them); to change the vocabulary change the constants in `g0_wandb.py`, bump `SCHEMA_VERSION`, update the tests and rebuild; run the live check `selftest_wandb.py` (creates and deletes one run) after any logger change. wandb access needs the throwaway venv `/tmp/claude-1000/wbvenv/bin/python` (system python has no wandb; recreate it with `python3 -m venv` and `pip install wandb` if it is gone); never print or handle keys. **Status 2026-10-17: the 13 finished runs are rebuilt under the schema and their 13 old runs deleted (verified first; see `reports/wandb_schema.md`).** The schema-aware `g0_wandb.py`, `run_g0.sbatch`, `run_baseline.sbatch` and `baseline_report.py` are deployed to Ada only after R1 and R5c have finished (R1 and R5c still run the old logger; their live wandb runs `339751bd` and `968e9ddb` are rebuilt and replaced when they finish).
