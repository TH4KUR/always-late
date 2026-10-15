# HANDOFF: establish the baseline by replicating Duan (2026) with THEIR code

Written 2026-10-16 for an agent with no memory of the previous sessions. Read this file fully, then `issues.md` (why we are doing this), `PLAN.md` (current plan), `reports/duan_codebase_audit.md` (evidence, Ada facts, closure protocol). Everything here was verified in the sessions it describes; items marked **UNVERIFIED** were not.

## 0. Mission and the one rule that matters

**Mission.** Establish *baseline results*: run Duan et al.'s own Phase-3 pipeline (Gemma-2-2B, Gemma Scope layer-12 16k SAE, RTN INT8..INT4 on WikiText-2, perplexity + per-feature Pearson survival, plus their matched-perplexity pruning row) on Ada, and compare with their published table (`duan-sae-feature-survival/results/tables/phase3_summary_final.csv`). This tells us whether their numbers are reproducible and gives us a reference for judging our own pipeline.

**The rule.** The baseline is a *replication of their code*, not a re-implementation and not an experiment. **Do not modify their code or its computation.** Exactly three kinds of edit are allowed, because without them their script cannot run on our hardware, and `scripts/g0/make_faithful_runner.py` is the only tool that may apply them:
1. **[FIX]** make the notebook export runnable: `display()` is notebook-only; cells 22+ call three functions that are never defined in the file and run only after every result file is written.
2. **[PLACE]** put the second bf16 model on a second 11 GB card (no GPU larger than 11 GB is reachable), plus the dtype-test fix their code needs for that (`device == "cuda"` would silently load float32 on `"cuda:1"`).
3. **[MODE]** choose one of *their two* shipped modes (`3a_test`, `3b_full`) by editing their own documented switch.

`python3 -m unittest discover -s scripts/g0/tests -v` must print `OK` before every run. `tests/test_faithful.py` proves that every computation, quantization, pruning, tokenization and aggregation function in the faithful runner is AST-identical to their cells 1-21, that only `load_gemma_tl_model`, `restore_model_work` and `add_summary` differ, that no experimental feature leaked in, and that the whole diff is under 40 changed lines. A mutation check confirmed it fails when one character of their math is changed. **Never edit a generated runner by hand; regenerate it.**

Anything else (BOS arms, extra bit-width rungs, snapshots, resume, our own pipeline on their tokens) is a **diagnostic experiment** (the "v3" runner, `make_duan_runners_v3.py`). It may be run only *after* the baseline exists, must be labelled "modified/diagnostic", and must never be called a replication.

## 1. Why this exists (2 minutes)

The team's own pipeline (HF transformers, `src/`) gives INT7/INT6 survival of 98%/94% where Duan reports 81%/51%. The Gate-G0 replication was never run; it was an independent re-implementation. Their released data show a degraded regime (no BOS token, baseline WikiText-2 perplexity 458 chunked vs 46 in their own BOS sliding-window run; non-monotone perplexity vs bits). Before trusting either side we run *their* code. Results from that run decide what we may claim (`reports/duan_codebase_audit.md` section 7). Also read `issues.md`: the paper and README contradict the team's own data and must be rewritten **by the team** (course policy forbids AI-generated report text; do not write paper prose).

## 2. State of play (2026-10-16)

| Item | Status |
|---|---|
| Review of code and results, `issues.md` | done |
| Duan's repo downloaded to `../duan-sae-feature-survival/` (zip sha256 `447f25e5b4cfc258e591ef12b059e8c9b541e0b2d71901b9af0c260fcc262478`); released tables verified offline | done |
| Ada probed; env built; benchmark; independent re-verification by a second reviewer | done |
| wandb logging (`scripts/g0/g0_wandb.py`), built test-first, live-tested against the real service | done |
| R0 smoke (v3 runner, 3a_test): TransformerLens works on 2 x RTX 2080 Ti, peak 8.2 of 11 GiB per card, 40 min per condition at 500k tokens | done (`reports/g0/r0_smoke/`) |
| Faithful runner + tests (this handoff's centrepiece) | done: 41 tests pass |
| Faithful smoke on Ada (`3a_test`), job 23419 | **done, passed; bit-identical to R0** (section 8, `reports/g0/baseline_smoke/`) |
| **Baseline: faithful `3b_full` (500k tokens, INT8..INT4 + pruning)** | **NOT RUN. This is your job.** Its pruning stage has never run in our setup (section 3, step 4) |
| Diagnostics R1-R3 (BOS arms), R4 (their script 06), R5 (our pipeline on their tokens) | not run; R5 script not written |

Early signal (R0, v3 runner in its default arm, 20k tokens, test split; **not** the baseline): INT6 survived 49.8% (published 51.3%), INT6 median r 0.8994 (0.9027), INT8 median r 0.9713 (0.9707), baseline perplexity 414 (458).

## 3. Runbook for the baseline

All commands run from the repo root unless stated. Ada access: `ssh karma@ada-gw1` (key auth works non-interactively from the user's laptop; `ada.iiit.ac.in` is closed).

1. **Pre-flight (local).**
   - `python3 -m unittest discover -s scripts/g0/tests -v` -> `OK`. If `../duan-sae-feature-survival/` is missing the faithfulness tests are *skipped*, which is not good enough: re-download it (below) and rerun.
   - Re-download if needed: `curl -L -o duan.zip https://anonymous.4open.science/api/repo/sae-feature-survival-quantization-0141/zip`, check the sha256 above, extract with a path-safe extractor into `../duan-sae-feature-survival/` (treat as untrusted: read it, never run it directly).
2. **Regenerate the faithful runners and deploy (local).**
   ```
   D=../duan-sae-feature-survival
   for m in 3a_test 3b_full; do python3 scripts/g0/make_faithful_runner.py $D/scripts/03_gemma_phase3_streaming_sweep.py /tmp/faithful_$m.py --mode $m; done
   scp /tmp/faithful_3a_test.py /tmp/faithful_3b_full.py scripts/g0/baseline_report.py scripts/g0/g0_wandb.py scripts/g0/run_baseline.sbatch karma@ada-gw1:/home2/karma/duan-verify/
   ```
3. **Pre-flight (Ada).** `sinfo -N -p u22 -t idle -h -o "%N %f" | grep -c 2080ti` (need >= 1; there were 8-11), `df -h /home2/karma` (need >= 12 GB free; it was 19 GB), `squeue -u karma` (the user's own `plan02-train` job is not yours: never touch it).
4. **Submit the baseline** (their shipped `3b_full`: 500k tokens, INT8/7/6/5/4, matched-perplexity pruning):
   `ssh karma@ada-gw1 'cd /home2/karma/duan-verify && sbatch -J base-full run_baseline.sbatch full 3b_full'`
   Expected: about 5.5 h (40 min per condition x 7 + pruning search), after ~18 min of model download to node-local scratch. The faithful runner has **no resume** (adding it would modify their loop). If the job dies, rerun the whole job.
   **Known untested path.** The smoke (`3a_test`) exercised only their RTN conditions. `3b_full` additionally runs their **matched-perplexity pruning stage** (their Cell 18: 6-step sparsity search, then the pruned condition) **after** all RTN conditions. That stage has never been executed in our setup (two cards, TransformerLens 3.9.0, restore after in-place pruning). Their RTN results are written to disk condition by condition before it starts, so a failure there loses only the pruning row: keep the RTN outputs (`RTN_INT*_summary.csv`, `*_per_feature.csv`, `phase3_summary_partial.csv`), report the pruning failure to the user, and do not patch their pruning code. Q1 does not depend on the pruning row.
5. **Watch.** `squeue -u karma`, `tail -c 2000 /home2/karma/duan-verify/logs/base-full-<jobid>.out | tr '\r' '\n'`, GPU memory in `/home2/karma/duan-verify/baseline/full/gpu_mem.log` (expect ~8.2 GiB per card). wandb is *not* logged from inside their code; the report step at the end of the job uploads to project `anlp-project` (entity `karma-skz-`).
6. **Collect.** `scp -r karma@ada-gw1:/home2/karma/duan-verify/baseline/full/phase3_gemma_outputs_3b_full_500k_L12 reports/g0/baseline_full/` plus `baseline_report.md/.json` and `gpu_mem.log`, the slurm log, and `run_meta`-equivalent facts (`pip freeze` is in `reports/g0/ada_env_freeze_20261015.txt`; note torch/TL/sae-lens versions in the report).
7. **Judge with the pre-committed bands** (no moving goalposts): `python3 scripts/g0/baseline_report.py --run-dir reports/g0/baseline_full/phase3_gemma_outputs_3b_full_500k_L12 --published ../duan-sae-feature-survival/results/tables/phase3_summary_final.csv --per-feature <run>/RTN_INT6_per_feature.csv --published-per-feature ../duan-sae-feature-survival/results/per_feature/phase5b_gemma_rtn_int6_vs_pruning_per_feature_overlap.csv`. **Q1 passes** iff INT7 survival is within 5 pp of 81.26%, INT6 within 10 pp of 51.30%, active features within 10% of 7,144, and perplexity signs hold (INT7 improves, INT6 worsens). Also record the per-feature Spearman at INT6 (should be near 1) and every INT5/INT4/pruning delta.
8. **Record and decide.**
   - Q1 passes -> baseline established. Write the numbers into `reports/progress_log.md`, `PROGRESS.md`, `PLAN.md` (status table), and tell the user. Then, only if the user agrees, start the diagnostics (section 5).
   - Q1 fails -> do **not** touch their code. Rerun once to see whether the result is deterministic; record versions; compare per-feature agreement; report non-reproducibility to the user (`D-NOT-RUNNABLE` / non-reproduced branch in audit section 7.3). Do not tune anything to make it pass.

## 4. Environment facts that cost us time (all verified)

- **Cluster.** Slurm. Accounts `nlp` (QOS normal) and `research` (QOS medium: max 4 GPUs, 40 CPUs, 125 GB per user, 1 node per job). Only partition `u22` is open to us. GPUs reachable are 11 GB-class only: GTX 1080 Ti (gnode001-042) and RTX 2080 Ti (gnode044-092, select with `--constraint=2080ti`). The 3090 nodes (`plafnet2`, `rrc`) and 3080 Ti nodes (`ihub`) are closed to our accounts. Turing has no native bf16: PyTorch emulates it (bf16 matmul ~2x slower than fp32, results correct, logits bit-identical across cards).
- **Filesystems.** `/home2/karma` (NFS, visible everywhere, ~19 GB free, **shared with the user's other jobs**: an earlier retry nearly filled it to 97%; keep the model off it). **`/share1` is NOT mounted on compute nodes** and has a 3,000-file quota (jobs writing there die in 2 s). Node-local `/scratch` (1.2 TB) is where the 10.5 GB gated model goes, per job, deleted at job end. `/share7` and `/data3` are mounted on compute nodes but we have no directories there: do not write into shared areas you do not own.
- **Software.** `/home2/karma/duan-verify/env`: Python 3.10, torch 2.14.1+cu126 (the cu126 wheel is needed because 2080 Ti drivers range 570-595), transformer-lens 3.9.0, sae-lens 6.54.5, transformers 5.19.0, wandb 0.30.0. The team's own paper ran on transformers 5.17.0 / torch 2.13.0; this difference matters for any claim about *our* pipeline (audit section 7.4 F5), not for the baseline.
- **Hugging Face.** The gated `google/gemma-2-2b` needs a token: `hf auth login` was done on Ada (token at `/home2/karma/duan-verify/hf/token`; jobs use `HF_TOKEN_PATH`). **Never copy a token anywhere, never print it.** **Do not set `HF_HUB_DISABLE_XET=1`**: plain HTTP was ~15x slower and kept dropping connections. Model download inside a job takes ~18 min at 5-30 MiB/s depending on the node.
- **wandb.** Key already in `~/.netrc` on Ada (account `karma-skz`, entity `karma-skz-`); the user created project `anlp-project`. To switch account the user runs `wandb login --relogin` themselves. Never ask for or handle a key.
- **Measured costs (TransformerLens, 2 x 2080 Ti).** Perplexity pass 3.3 it/s, streaming pass 1.08 it/s (their float64 CPU accumulation is part of that), peak GPU 8.2 GiB per card, host RSS peak 21.8 GB, 40 min per condition at 500k tokens.
- **Queue.** Wait was ~0 s for all jobs so far. `sbatch --test-only` predictions ("starts tomorrow") were wrong; ignore them.

## 5. After the baseline: diagnostics (modified code, clearly labelled)

Only after the baseline exists and the user agrees. Full design and closure rules: `reports/duan_codebase_audit.md` sections 6-7; run tooling: `scripts/g0/make_duan_runners_v3.py` (v3 runner: BOS arms, INT0 restore-only control, ladder 0/16/12/8/7/6, snapshots for block-bootstrap CIs, resume, wandb heartbeats), `scripts/g0/run_g0.sbatch`. R1 no BOS, R2 BOS prepended with position 0 masked, R3 BOS prepended and kept (our pipeline's regime), each ~4.8 h (about 15 h sequential, ~5 h in parallel on three nodes with `-A nlp`, untested). R4 = their script 06 (does not parse as shipped; the generator fixes it). R5 = our pipeline on their exact `tokens_2d.pt` (script not written yet; must import `src/` with a recorded hash). The closure protocol maps every outcome to a terminal disposition so Duan is never re-verified afterwards.

## 6. Do-not list

- Do not modify their code, the faithful runner by hand, the baseline mode constants, or the comparison bands. Do not describe diagnostics as replication.
- Do not edit `paper/` prose or write report text. Do not change `src/` or `scripts/` outside `scripts/g0/` without asking the user (the user earlier asked for documentation only, then explicitly authorised the `scripts/g0` tooling and the wandb logging).
- Do not touch the user's `plan02-train` jobs, other users' files, `~/.netrc`, `~/.env`, `~/.ssh`, HF tokens. Do not fill `/home2`. Do not run long jobs on the login node (short pip/hf downloads only).
- Do not commit data, model weights, `.pt` snapshots, tokens or secrets (`.gitignore` covers `data/`, `*.pt`, `*.npz`, `wandb/`). Commit results as small CSV/JSON/MD under `reports/g0/`.

## 7. File map

| Path | What |
|---|---|
| `HANDOFF.md` | this file |
| `PLAN.md`, `PROGRESS.md`, `reports/progress_log.md` | plan, status, dated evidence log |
| `issues.md` | 40+ issues from the review (A paper vs data, B instrument, C method, D implementation, 8 external resources) |
| `reports/duan_codebase_audit.md` | audit of Duan's repo, verified Ada facts, closure protocol, reviewer reconciliation |
| `scripts/g0/make_faithful_runner.py` | **baseline** generator (only [FIX]/[PLACE]/[MODE]) |
| `scripts/g0/run_baseline.sbatch` | **baseline** launcher + automatic comparison and wandb upload |
| `scripts/g0/baseline_report.py` | comparison with their published table, pre-committed bands, per-feature Spearman |
| `scripts/g0/make_duan_runners_v3.py`, `run_g0.sbatch`, `unit_test_runner.py` | diagnostic runner (modified; not a replication) |
| `scripts/g0/g0_wandb.py`, `selftest_wandb.py`, `bench_fit.py` | wandb helper and its live self-test; Ada fit/speed benchmark |
| `scripts/g0/tests/` | 41 stdlib tests (faithfulness, comparison, logger, generator) |
| `reports/g0/` | R0 smoke results, benchmark JSON, env freeze, diffs of generated runners vs their originals |
| `../duan-sae-feature-survival/` | their repo (outside git; re-download with the hash above) |

## 8. Faithful smoke result (done, job 23419, 20.5 min incl. model download)

The faithful runner (their code only) was run in their `3a_test` mode (20k tokens, WikiText-2 test split, INT8 + INT6) on 2 x RTX 2080 Ti and **completed with exit 0**; the automatic report step and the wandb upload also worked (`https://wandb.ai/karma-skz-/anlp-project/runs/8f6f4d3a`, run `baseline-smoke-23419`).

**Neutrality check.** Compared with R0 (the v3 runner's default arm, job 23355, same mode and data, a different node): **bit-identical**. Maximum absolute difference 0.0 over every summary column and over the per-feature correlations of all 16,384 features for the baseline, INT8 and INT6 conditions; active flags identical. So (a) the instrumentation in the v3 runner did not change their computation, and (b) results are deterministic across jobs and nodes of this GPU class. Evidence: `reports/g0/baseline_smoke/neutrality_check.json`.

| 3a_test, 20k tokens | baseline ppl | INT8 ppl delta / survived / median r | INT6 ppl delta / survived / median r | active |
|---|---|---|---|---|
| faithful replication (this) | 414.07 | +2.06% / 96.15% / 0.9713 | +3.26% / 49.82% / 0.8994 | 6,702 |
| Duan published (3b_full, 500k tokens, train split) | 458.53 | +2.62% / 99.09% / 0.9707 | +3.99% / 51.30% / 0.9027 | 7,144 |

Different token budgets and splits, so differences are expected; the regime (baseline perplexity in the hundreds, INT6 survival near 50%) reproduces. Per-feature agreement with their released INT6 file over 6,271 commonly active features: Spearman(r) 0.926, Spearman(firing rate) 0.918. **This is a smoke test: Q1 is NOT evaluated** (INT7 not run, 20k tokens). Q1 needs the 500k-token baseline below.

## 9. Open questions for the user

1. Go-ahead for the ~5.5 h baseline job (the user said "establish baseline results"; launching was not yet explicitly approved).
2. After the baseline: run the diagnostics R1-R3 (~15 GPU-hours)? Parallel arms need `-A nlp` (untested).
3. Who writes the paper's rewritten results/discussion (must be the team)?
