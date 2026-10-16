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
| **Baseline: their `3b_full` config (500k tokens, INT8..INT4), pruning off** | **DONE, Q1 PASSED** (job 23472; section 10). Results in `reports/g0/baseline_full/`. Do not rerun. |
| **Audit of the disagreement** (`reports/audit_our_vs_duan.md`: 12 listed differences, pre-committed reading rules) | **written; experiments running** (section 11) |
| R2 (their code + BOS prepended and masked; ladder INT0/16/12/8/7/6), Slurm 23643 | **running** (submitted 2026-10-16 12:04 IST, ~4.8 h) |
| R5 (our HF pipeline on their tokens, one factor flipped per cell): `r5_run.py`, `r5_lib.py`, `run_r5.sbatch`; 31 CPU tests (+ mutation checks) pass locally and in the Ada env | smoke 23648 -> R5a 23649, R5b 23652 (`afterok`) queued (section 11) |
| R1/R3 (BOS arms of the v3 generator), R4 (their script 06) | not run; R3 is superseded by `T_keep`/`O_keep` |

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
8. **Record and decide.** (Done for job 23472: Q1 passed.) Q1 passes -> baseline established: write the numbers into `reports/progress_log.md`, `PROGRESS.md`, `PLAN.md`, tell the user, then (only if they agree) start the diagnostics (section 5). Q1 fails -> do **not** touch their code: rerun once to check determinism (results were bit-identical across two jobs/nodes in the smoke tests), record versions, report non-reproducibility.

## 4. Environment facts that cost us time (all verified)

- **Cluster.** Slurm. Accounts `nlp` (QOS normal) and `research` (QOS medium: max 4 GPUs, 40 CPUs, 125 GB per user, 1 node per job). Only partition `u22` is open to us. GPUs reachable are 11 GB-class only: GTX 1080 Ti (gnode001-042) and RTX 2080 Ti (gnode044-092, `--constraint=2080ti`). The 3090 nodes (`plafnet2`, `rrc`) and 3080 Ti nodes (`ihub`) are closed to our accounts. Turing has no native bf16: PyTorch emulates it (correct results, ~2x slower matmuls).
- **Filesystems.** `/home2/karma` (NFS, ~19 GB free, **shared with the user's other jobs**; an earlier retry nearly filled it to 97%: keep the model off it). **`/share1` is NOT mounted on compute nodes** and has a 3,000-file quota. Node-local `/scratch` (1.2 TB) holds the 10.5 GB gated model per job, deleted at job end. `/share7` and `/data3` are mounted on compute nodes but we have no directories there: never write into shared areas you do not own.
- **Software.** `/home2/karma/duan-verify/env`: Python 3.10, torch 2.14.1+cu126, transformer-lens 3.9.0, sae-lens 6.54.5, transformers 5.19.0, wandb 0.30.0, ipython 8.39.0. Their `requirements.txt` was unpinned; these versions are part of the result.
- **Hugging Face.** The gated `google/gemma-2-2b` needs a token: `hf auth login` was done on Ada (`/home2/karma/duan-verify/hf/token`; jobs use `HF_TOKEN_PATH`). **Never copy or print a token.** **Do not set `HF_HUB_DISABLE_XET=1`** (plain HTTP was ~15x slower and kept dropping connections).
- **wandb.** Key already in `~/.netrc` on Ada (account `karma-skz`, entity `karma-skz-`); project `anlp-project`. To change account the user runs `wandb login --relogin` themselves. Never ask for or handle a key.
- **Measured costs (TransformerLens, 2 x 2080 Ti).** Perplexity pass 3.3 it/s, streaming pass 1.08 it/s, peak GPU 8.2 GiB per card, host RSS peak 21.8 GB, **40 min per condition at 500k tokens**.
- **Queue.** Wait was ~0 s for every job so far. `sbatch --test-only` predictions ("starts tomorrow") were wrong; ignore them.

## 5. After the baseline: diagnostics (modified code, clearly labelled)

Only after the baseline exists and the user agrees. Design and closure rules: `reports/duan_codebase_audit.md` sections 6-7. Tooling: `scripts/g0/make_duan_runners_v3.py` (reads the vendored *pristine* scripts; BOS arms, INT0 restore-only control, ladder, snapshots for block-bootstrap CIs, resume, wandb heartbeats) and `scripts/g0/run_g0.sbatch`. R1 no BOS, R2 BOS prepended with position 0 masked, R3 BOS prepended and kept (our pipeline's regime), each ~4.8 h; R4 = their script 06 (does not parse as shipped; the generator fixes it); R5 = our pipeline on their exact tokens (`scripts/g0/r5_run.py`, section 11; its design and the list of differences it isolates are in `reports/audit_our_vs_duan.md`). The closure protocol maps every outcome to a terminal disposition so Duan is never re-verified afterwards.

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
| `scripts/g0/r5_run.py`, `r5_lib.py`, `run_r5.sbatch`, `tests_torch/` | R5: OUR pipeline on THEIR tokens with factor toggles (attention impl, SAE dtype, quantizer grouping, BOS regime); torch tests (`/tmp/claude-1000/torchvenv/bin/python -m unittest discover -s scripts/g0/tests_torch -v`, 31 tests, CPU) |
| `reports/audit_our_vs_duan.md` | why our pipeline disagrees with theirs: difference table, experiments, pre-committed reading rules, results (section 6 there, filled in after the jobs finish) |
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

## 9. Baseline result (job 23472, completed 2026-10-16)

**Q1 passed on all four pre-committed bands** (INT7 survival within 5 pp, INT6 within 10 pp, active features within 10%, perplexity signs): the replication is within ~0.15 pp of their published table at every rung. Outputs: `reports/g0/baseline_full/` (their per-condition CSVs and per-feature files, `baseline_report.md/.json`, `gpu_mem.log`).

| condition | ppl (ours / published) | ppl delta % | survived % (r>0.9) | median r | active features |
|---|---|---|---|---|---|
| FP16 baseline | 458.4 / 458.5 | 0 / 0 | 100 / 100 | 1.0000 / 1.0000 | 7,134 / 7,144 |
| RTN INT8 | 469.9 / 470.6 | +2.52 / +2.62 | 99.12 / 99.09 | 0.9707 / 0.9707 | same |
| RTN INT7 | 432.7 / 432.6 | **-5.61 / -5.65** | 81.19 / 81.26 | 0.9499 / 0.9499 | same |
| RTN INT6 | 476.8 / 476.8 | +4.02 / +3.99 | 51.15 / 51.30 | 0.9027 / 0.9027 | same |
| RTN INT5 | 539.7 / 539.8 | +17.73 / +17.72 | 27.57 / 27.72 | 0.7775 / 0.7780 | same |
| RTN INT4 | 670.4 / 669.8 | +46.25 / +46.08 | 12.39 / 12.40 | 0.4504 / 0.4512 | same |

Largest survival gap over all five rungs **0.15 pp**; largest median-r gap **0.0008**; per-feature Spearman at INT6 over 7,121 commonly active features: **r 0.99965, firing rate 0.99995**. Job 23472 on gnode057: 4 h 08 min, exit 0, no errors, peak GPU 8,183 / 8,009 MiB of 11,264 per card. Vendored script sha256 prefix `7f1e638a0508c3ca` (verified by the launcher before running). wandb: `https://wandb.ai/karma-skz-/anlp-project/runs/86f26788`.

**What it means.** (1) Their published Gemma Phase-3 numbers are reproducible with their own code on different hardware (our Turing GPUs emulate bf16; theirs did not), including the odd parts: baseline perplexity of 458 and **INT7 improving perplexity by 5.6%**. So their degraded regime is real in their pipeline, not a typo or a one-off. (2) The per-feature correlations agree across hardware at 0.9996, so the INT6 survival values are a deterministic function of the weight perturbation, **not** hardware-noise realizations; this argues against the idea that their numbers are chaotic rounding noise (inference; the INT16/INT12 noise-floor control of the closure protocol has not been run). (3) It does **not** yet explain why our own pipeline gives 98%/94% at INT7/INT6: that is the diagnostics layer (BOS regime, TransformerLens vs HF, SAE dtype), which has not been run. (4) Not demonstrated: that our four device edits equal their unmodified single-device run (no GPU of >= 16 GB is reachable); the agreement above is strong indirect evidence.

**Operational note.** On 2026-10-16 ~09:24 Ada lost *all* outbound internet (W&B, Hugging Face and PyPI time out from the login node; they answered from the laptop). The job's own post-hoc wandb upload therefore failed (non-fatal, by design) and the results were uploaded from the laptop. Check outbound access before launching any job that downloads the 10.5 GB model.

## 10. Open questions for the user

1. ~~Run the diagnostics?~~ Authorised on 2026-10-16 ("audit why our implementation was so off compared to theirs"; R2 and R5 started). Still open: R1/R3 (superseded by the R5 BOS cells), their script 06, INT5/INT4 in R5, a cell for the corpus/population difference (D5/D6) if `O_keep` does not land near our original 98%/94%.
2. Who writes the paper's rewritten results/discussion (must be the team)?

## 11. Audit jobs in flight and what to do on the next prompt (written 2026-10-16, around midday IST)

**NEXT PROMPT: check R2/R5 status and fetch results.** The user asked for the jobs to be queued and the response ended; the next message is the cue to look.

| Job | What | Ada output | Log |
|---|---|---|---|
| 23643 `R2-bos-mask` | their TL script, BOS prepended to identical 256-token windows, position 0 masked; ladder INT0, 16, 12, 8, 7, 6; 2 GPUs, ~4.8 h | `/home2/karma/duan-verify/out/R2/` | `logs/R2-bos-mask-23643.out` |
| 23648 `R5smoke` | 40 windows, cells `T,O`, INT8, wandb off, selfcheck on 4 windows (gate for the two jobs below) | `/home2/karma/duan-verify/r5/R5smoke/` | `logs/R5smoke-23648.out` |
| 23649 `R5a` | cells `T,O,T_sdpa,T_bf16sae`, INT8/7/6, 1 GPU (est. 3 h) | `r5/R5a/<cell>/` | `logs/R5a-23649.out` |
| 23652 `R5b` | cells `T_rowquant,T_mask,T_keep,O_keep`, INT8/7/6, 1 GPU (est. 3 h) | `r5/R5b/<cell>/` | `logs/R5b-23652.out` |

(`/home2/karma/duan-verify/` prefix on every path. R5a/R5b start only if the smoke exits 0 (`--dependency=afterok`); if the smoke fails they stay `DependencyNeverSatisfied` and must be cancelled (`scancel 23649 23652`), the bug fixed test-first, the code redeployed (`rsync` `r5_run.py r5_lib.py g0_wandb.py` and `src/*.py` to `r5code/`, `run_r5.sbatch` to the parent), and resubmitted. A finished cell is skipped on resubmission with the same out-name.)

Steps for the next prompt, in order:

1. `ssh karma@ada-gw1 'squeue -u karma; sacct -j 23643,23648,23649,23652 -o JobID,JobName,State,Elapsed,ExitCode'`; read the log tails (`tail -c 3000 ... | tr '\r' '\n'`). Never touch the user's other jobs.
2. **Smoke first.** Read `r5/R5smoke/selfcheck.json`: `int0_identity.identical` must be true; record `sdpa_vs_eager` (the size of the missing soft-cap effect), `ours_fp32_vs_sae_lens` (should be Jaccard ~1, max abs diff ~0), `ours_bf16_vs_fp32`, and the token SHA-256. Check the first cell's printed numbers are sane (FP perplexity ~450, not NaN).
3. When R5a/R5b/R2 finish, fetch the results (small files only: summaries, selfcheck, a few per-feature files; leave `*.pt`, snapshots and the full per-feature set on Ada): `rsync -a --exclude '*.pt' --exclude 'snap_*' karma@ada-gw1:/home2/karma/duan-verify/r5/R5a/ reports/g0/r5/R5a/` (same for R5b, R5smoke; R2 to `reports/g0/R2_bos_mask/`).
4. Verify the R2/R5 token windows agree on Ada (`torch.equal(R2_tokens[:, 1:], R5_tokens)` on the two `tokens_2d.pt` files, for the `R2` and `R5a` directories); the R5 selfcheck also records the SHA-256.
5. Judge with the rules in `reports/audit_our_vs_duan.md` section 5 (anchor `T` vs `reports/g0/baseline_full`, single-factor deltas with the 5 pp / 1 pp thresholds, `O` vs `T`, `O_keep` vs our original 98.28 % / 94.31 %, R2 vs `T_mask`, R2's own 10 pp rule). Fill section 6 of the audit with the numbers; do not move the thresholds.
6. Upload to wandb from the fetched files if the live logging failed (project `anlp-project`; Ada has outbound access again since ~12:00; if not, use the throwaway venv on the laptop and never handle the key). Update `PROGRESS.md`, `PLAN.md`, `README.md`, `reports/progress_log.md`, `issues.md` B1 with what was found, then commit and push to `main`.
7. Report to the user: which differences are material, which are not, and what is still unexplained. No paper prose.
