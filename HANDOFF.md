# HANDOFF: establish the baseline by running Duan (2026)'s OWN code

Written 2026-10-16 (revised the same day) for an agent with no memory of the previous sessions. Read this file fully, then `issues.md` (why we are doing this), `PLAN.md`, `third_party/duan2026/UPSTREAM.md` (exactly what was removed and edited) and `reports/duan_codebase_audit.md` (evidence, Ada facts, closure protocol). Items marked **UNVERIFIED** were not verified.

## 0. Mission and the one rule that matters

**Mission.** Establish *baseline results*: run Duan et al.'s own Phase-3 script (Gemma-2-2B, Gemma Scope layer-12 16k SAE, RTN INT8..INT4 on WikiText-2, perplexity + per-feature Pearson survival) on Ada and compare with their published table. This tells us whether their numbers are reproducible and gives a reference for judging our own pipeline.

**The rule.** We run **their code, in their repo layout**, not a re-implementation and not a generated rewrite. `third_party/duan2026/` is their repository downloaded fresh, pruned to what this study needs, with the working copy of the one script we run differing from their file by a **short explicit whitelist** and nothing else:

| What | Detail |
|---|---|
| Removed (not needed) | Pythia scripts, the feature-class predictor, `src/`, Pythia/pruning-overlap results and figures, env files. Hashes of every removed file stay in `MANIFEST.sha256`. |
| Retained byte-identical (`pristine/`) | `LICENSE`, `README.md`, `scripts/03_gemma_phase3_streaming_sweep.py` (the study), `scripts/06_gemma_sliding_window_ppl_check.py` (kept for later), and their published Gemma results (`phase3_summary_final.csv`, `gemma_chunked_vs_sliding_ppl_w2048_s512_FULL.csv`, the INT6 per-feature file). |
| Working copy = pristine script + | (tail) delete their cells 22+, which call three functions never defined in the file and run after all results are written; (1) `RUN_PRUNING = False` in their `3b_full` branch (their own flag; pruning is not needed here); (2) `WORK_DEVICE = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE`; (3) `device == "cuda"` -> `str(device).startswith("cuda")` in their dtype test (otherwise `"cuda:1"` silently gives float32); (4) `load_gemma_tl_model(..., device=WORK_DEVICE)`; (5) `model_work.to(WORK_DEVICE)`. Edits 2-5 are device references forced by 11 GB GPUs (two bf16 models do not fit one card). |
| No edit for `display()` | their notebook-only `display(...)` works when run with `ipython script.py` (IPython is installed in the Ada env). |

**Do not edit the working copy by hand, and do not add edits.** The whitelist lives in `scripts/g0/verify_vendor.py` (`EDITS`, `TRUNCATE_AT`). `python3 -m unittest discover -s scripts/g0/tests -v` must print `OK` before any run: `tests/test_vendor.py` requires the working copy to equal *their pristine file plus exactly that whitelist, byte for byte over the whole file including top-level statements*, requires every retained file to match the recorded upstream hash and every removed file to be absent, and carries a **committed mutation suite** (13 mutations of their math, thresholds, constants, config, structure and an appended statement; each must be rejected). If you need a different edit, stop and ask the user. Anything beyond the baseline (BOS arms, extra bit-widths, our pipeline on their tokens) is a diagnostic experiment (section 5), must be labelled "modified/diagnostic" and must never be called a replication.

**Honest limit (UNVERIFIED).** We never ran their *unmodified* script: it needs a >= 16 GB GPU and none is reachable. That edits 2-5 do not change results relative to a single-device run is argued from the code and from bit-identical logits across the two cards, not demonstrated on a large GPU.

## 1. Why this exists

The team's own pipeline (HF transformers, `src/`) gives INT7/INT6 survival of 98%/94% where Duan reports 81%/51%. The Gate-G0 replication was never run; it was an independent re-implementation. Their released data show a degraded regime (no BOS token, baseline WikiText-2 perplexity 458 chunked vs 46 in their own BOS sliding-window run; non-monotone perplexity vs bits). Before trusting either side we run *their* code. Read `issues.md`: the paper and README contradicted the team's own data and must be rewritten **by the team** (course policy forbids AI-generated report text; do not write paper prose).

## 2. State of play (2026-10-16)

| Item | Status |
|---|---|
| Review of code and results, `issues.md` | done |
| Their repo downloaded twice (byte-identical both times), vendored and pruned in `third_party/duan2026/`, content tree hash `a84e43da...adc` (the zip hash is NOT stable: the server stamps download time) | done |
| Ada probed; env built (+ IPython); benchmark; independent re-verification by a second reviewer | done |
| wandb logging (`scripts/g0/g0_wandb.py`), live-tested; post-hoc upload from `baseline_report.py` | done |
| Vendoring tests (exact reconstruction + committed mutation suite) | done: 43 tests in the suite pass |
| Earlier generator-based "faithful runner" | **removed**: it rewrote their file by string substitution and its test compared only 13 function bodies, not top-level statements |
| Smoke of the vendored file on Ada (job 23463, their `3a_test` mode) | **done, passed; bit-identical to the two earlier smokes** (section 8) |
| **Baseline: their `3b_full` config (500k tokens, INT8..INT4), pruning off** | **RUNNING as Slurm job 23472** (gnode057, started 2026-10-16). Next agent: check it (section 3 steps 5-8), do NOT resubmit unless it failed |
| Diagnostics R1-R3 (BOS arms), R4 (their script 06), R5 (our pipeline on their tokens) | not run; R5 script not written |

## 3. Runbook for the baseline

Commands from the repo root. Ada: `ssh karma@ada-gw1` (key auth works from the user's laptop; `ada.iiit.ac.in` is closed).

1. **Pre-flight (local).** `python3 -m unittest discover -s scripts/g0/tests -v` -> `OK`. `python3 scripts/g0/verify_vendor.py --verify` -> `pristine problems: none | working copy == pristine + whitelist: True`. To re-verify against upstream yourself: download `https://anonymous.4open.science/api/repo/sae-feature-survival-quantization-0141/zip`, extract with a path-safe extractor into an empty directory, run `python3 scripts/g0/verify_vendor.py --build <that dir>`, then `git status third_party/duan2026` must show no change (the build is deterministic).
2. **Deploy.**
   ```
   V=third_party/duan2026
   scp $V/scripts/03_gemma_phase3_streaming_sweep.py karma@ada-gw1:/home2/karma/duan-verify/vendor/scripts/
   scp $V/working.sha256 karma@ada-gw1:/home2/karma/duan-verify/vendor/
   scp scripts/g0/run_baseline.sbatch scripts/g0/baseline_report.py scripts/g0/g0_wandb.py karma@ada-gw1:/home2/karma/duan-verify/
   ```
   The launcher refuses to run if the deployed script's sha256 does not match `working.sha256` and logs the hash of the file it runs.
3. **Pre-flight (Ada).** `sinfo -N -p u22 -t idle -h -o "%N %f" | grep -c 2080ti` (>= 1; there were 7-11), `df -h /home2/karma` (>= 12 GB free; it was 19 GB), `squeue -u karma` (the user's own `plan02-train` job is not yours: never touch it).
4. **Submit.** `ssh karma@ada-gw1 'cd /home2/karma/duan-verify && sbatch -J base-full run_baseline.sbatch full full'`. Their shipped `3b_full` (500k tokens, INT8/7/6/5/4), pruning off: 6 conditions x ~40 min = **~4.1 h compute**, plus 10-20 min of model download to node-local scratch. There is no resume (adding it would modify their loop); if the job dies, rerun it whole.
5. **Watch.** `squeue -u karma`, `tail -c 2000 /home2/karma/duan-verify/logs/base-full-<jobid>.out | tr '\r' '\n'`, GPU memory in `/home2/karma/duan-verify/baseline/full/gpu_mem.log` (expect ~8.2 GiB per card). Nothing is logged to wandb from inside their code: the report step at the end of the job uploads to project `anlp-project` (entity `karma-skz-`).
6. **Collect.** `scp -r karma@ada-gw1:/home2/karma/duan-verify/baseline/full/phase3_gemma_outputs_3b_full_500k_L12 reports/g0/baseline_full/`; also `gpu_mem.log` and the slurm log; note the library versions (`reports/g0/ada_env_freeze_20261015.txt`, plus `pip show ipython`).
7. **Judge with the pre-committed bands** (no moving goalposts): `python3 scripts/g0/baseline_report.py --run-dir reports/g0/baseline_full/phase3_gemma_outputs_3b_full_500k_L12 --published third_party/duan2026/pristine/results/tables/phase3_summary_final.csv --per-feature <run>/RTN_INT6_per_feature.csv --published-per-feature third_party/duan2026/pristine/results/per_feature/phase5b_gemma_rtn_int6_vs_pruning_per_feature_overlap.csv`. **Q1 passes** iff INT7 survival is within 5 pp of 81.26%, INT6 within 10 pp of 51.30%, active features within 10% of 7,144, and perplexity signs hold (INT7 improves, INT6 worsens). Also record the per-feature Spearman at INT6 and every INT5/INT4 delta. (Their published table also has a pruning row; it is intentionally not replicated here.)
8. **Record and decide.** Q1 passes -> baseline established: write the numbers into `reports/progress_log.md`, `PROGRESS.md`, `PLAN.md`, tell the user, then (only if they agree) start the diagnostics (section 5). Q1 fails -> do **not** touch their code: rerun once to check determinism (results were bit-identical across two jobs/nodes in the smoke tests), record versions, report non-reproducibility.

## 4. Environment facts that cost us time (all verified)

- **Cluster.** Slurm. Accounts `nlp` (QOS normal) and `research` (QOS medium: max 4 GPUs, 40 CPUs, 125 GB per user, 1 node per job). Only partition `u22` is open to us. GPUs reachable are 11 GB-class only: GTX 1080 Ti (gnode001-042) and RTX 2080 Ti (gnode044-092, `--constraint=2080ti`). The 3090 nodes (`plafnet2`, `rrc`) and 3080 Ti nodes (`ihub`) are closed to our accounts. Turing has no native bf16: PyTorch emulates it (correct results, ~2x slower matmuls).
- **Filesystems.** `/home2/karma` (NFS, ~19 GB free, **shared with the user's other jobs**; an earlier retry nearly filled it to 97%: keep the model off it). **`/share1` is NOT mounted on compute nodes** and has a 3,000-file quota. Node-local `/scratch` (1.2 TB) holds the 10.5 GB gated model per job, deleted at job end. `/share7` and `/data3` are mounted on compute nodes but we have no directories there: never write into shared areas you do not own.
- **Software.** `/home2/karma/duan-verify/env`: Python 3.10, torch 2.14.1+cu126, transformer-lens 3.9.0, sae-lens 6.54.5, transformers 5.19.0, wandb 0.30.0, ipython 8.39.0. Their `requirements.txt` was unpinned; these versions are part of the result.
- **Hugging Face.** The gated `google/gemma-2-2b` needs a token: `hf auth login` was done on Ada (`/home2/karma/duan-verify/hf/token`; jobs use `HF_TOKEN_PATH`). **Never copy or print a token.** **Do not set `HF_HUB_DISABLE_XET=1`** (plain HTTP was ~15x slower and kept dropping connections).
- **wandb.** Key already in `~/.netrc` on Ada (account `karma-skz`, entity `karma-skz-`); project `anlp-project`. To change account the user runs `wandb login --relogin` themselves. Never ask for or handle a key.
- **Measured costs (TransformerLens, 2 x 2080 Ti).** Perplexity pass 3.3 it/s, streaming pass 1.08 it/s, peak GPU 8.2 GiB per card, host RSS peak 21.8 GB, **40 min per condition at 500k tokens**.
- **Queue.** Wait was ~0 s for every job so far. `sbatch --test-only` predictions ("starts tomorrow") were wrong; ignore them.

## 5. After the baseline: diagnostics (modified code, clearly labelled)

Only after the baseline exists and the user agrees. Design and closure rules: `reports/duan_codebase_audit.md` sections 6-7. Tooling: `scripts/g0/make_duan_runners_v3.py` (reads the vendored *pristine* scripts; BOS arms, INT0 restore-only control, ladder, snapshots for block-bootstrap CIs, resume, wandb heartbeats) and `scripts/g0/run_g0.sbatch`. R1 no BOS, R2 BOS prepended with position 0 masked, R3 BOS prepended and kept (our pipeline's regime), each ~4.8 h; R4 = their script 06 (does not parse as shipped; the generator fixes it); R5 = our pipeline on their exact tokens (script not written). The closure protocol maps every outcome to a terminal disposition so Duan is never re-verified afterwards.

## 6. Do-not list

- Do not edit `third_party/duan2026/` by hand, add edits, change the baseline constants, or change the comparison bands. Do not describe diagnostics as replication.
- Do not edit `paper/` prose or write report text. Do not change `src/` or scripts outside `scripts/g0/` without asking the user.
- Do not touch the user's `plan02-train` jobs, other users' files, `~/.netrc`, `~/.env`, `~/.ssh`, HF tokens. Do not fill `/home2`. No long jobs on the login node.
- Do not commit data, weights, `.pt` files or secrets. Commit results as small CSV/JSON/MD under `reports/g0/`.

## 7. File map

| Path | What |
|---|---|
| `HANDOFF.md`, `PLAN.md`, `PROGRESS.md`, `reports/progress_log.md` | handoff, plan, status, dated evidence log |
| `issues.md` | the review (A paper vs data, B instrument, C method, D implementation, external-resources audit) |
| `reports/duan_codebase_audit.md` | audit of Duan's repo, verified Ada facts, closure protocol, reviewer reconciliation |
| `third_party/duan2026/` | **their code**: `UPSTREAM.md`, `MANIFEST.sha256`, `pristine/`, `scripts/` (working copy), `patches/`, `working.sha256` |
| `scripts/g0/verify_vendor.py` | the edit whitelist + verification/build CLI |
| `scripts/g0/run_baseline.sbatch`, `baseline_report.py` | baseline launcher and the comparison with their published table (post-hoc wandb) |
| `scripts/g0/make_duan_runners_v3.py`, `run_g0.sbatch`, `unit_test_runner.py` | diagnostic runner (modified; not a replication) |
| `scripts/g0/g0_wandb.py`, `selftest_wandb.py`, `bench_fit.py` | wandb helper and its live self-test; Ada fit/speed benchmark |
| `scripts/g0/tests/` | stdlib tests (vendoring + mutation suite, comparison, logger, generator) |
| `reports/g0/` | smoke results, benchmark JSON, env freeze, diffs of diagnostic runners |

## 8. Smoke of the vendored file (done, job 23463, 14.5 min incl. model download)

Their vendored script (their file + the 5 whitelisted edits), run **under IPython** in their `3a_test` mode (a temporary copy that differs from the vendored file by the single `RUN_MODE` line; the launcher printed that diff), 20k tokens, 2 x RTX 2080 Ti: **completed with exit 0, no traceback**, `display()` worked unedited, the report step and wandb upload worked (`https://wandb.ai/karma-skz-/anlp-project/runs/f722c81a`).

**Bit-identical to both earlier smoke runs** (the generator-based runner, job 23419, and the instrumented v3 runner R0, job 23355): max absolute difference 0.0 over every summary column and over the per-feature correlations of all 16,384 features for the baseline, INT8 and INT6 conditions; active flags identical. Evidence: `reports/g0/baseline_smoke_vendored/neutrality_check.json`.

| 3a_test, 20k tokens | baseline ppl | INT8 ppl delta / survived / median r | INT6 ppl delta / survived / median r | active |
|---|---|---|---|---|
| vendored script (this) | 414.07 | +2.06% / 96.15% / 0.9713 | +3.26% / 49.82% / 0.8994 | 6,702 |
| Duan published (3b_full, 500k tokens, train split) | 458.53 | +2.62% / 99.09% / 0.9707 | +3.99% / 51.30% / 0.9027 | 7,144 |

Different token budgets and splits, so differences are expected; the regime reproduces. **Q1 is NOT evaluated here** (INT7 not run, 20k tokens). Peak GPU memory 8,205 / 7,993 MiB per card.

## 9. Open questions for the user

1. After the baseline: run the diagnostics R1-R3 (~15 GPU-hours)? Parallel arms need `-A nlp` (untested).
2. Who writes the paper's rewritten results/discussion (must be the team)?
