# Audit of the Duan (2026) codebase: compute needs, what we need from it, how to run it

Date: 2026-10-15 (updated late evening after probing Ada, building their stack there, a benchmark on 2 × RTX 2080 Ti, and an independent re-verification by a second reviewer). Status: **their code has not been executed yet.** The remaining blocker is one manual step: `hf auth login` on Ada for the gated `google/gemma-2-2b` (§4.5). Numbers marked *measured* come from `reports/g0/bench_fit_result_2x2080ti_20261015.json`; the rest are derived from the code's control flow.

Why this exists: our G0 "replication" was an independent re-implementation (see `issues.md` B1 and PLAN.md). Running their code directly is the cleaner test of two things: (a) are their reported numbers reproducible at all, and (b) is the gap to our pipeline caused by our implementation or by theirs.

## 1. Source and integrity

| | |
|---|---|
| Paper | Duan (2026), *Perplexity Can Miss SAE Feature Damage Under Quantization*, arXiv:2606.03002 |
| Artifact | `https://anonymous.4open.science/r/sae-feature-survival-quantization-0141/` (zip via `…/api/repo/sae-feature-survival-quantization-0141/zip`) |
| Downloaded | 2026-10-15; zip sha256 `447f25e5b4cfc258e591ef12b059e8c9b541e0b2d71901b9af0c260fcc262478` |
| Local copy | `/home/karma/Documents/ANLP/project/duan-sae-feature-survival/`. It is **outside** our git repo on purpose, so it can't be committed by accident. |
| Contents | 36 files, 5.3 MB: 8 auto-exported scripts, 11 result tables, 2 per-feature CSVs, 7 figures, `metrics.py`, env files |
| License | MIT (code); model, SAE and data keep their own licenses |
| Treated as | untrusted third-party code. It has only been read, never executed, and should run in its own virtualenv |

Things that don't match their README:
- The README references a `notebooks/` directory, but only `scripts/` (auto-exports of the notebooks) was released.
- `requirements.txt` and `environment.yml` are **unpinned** (`transformer-lens`, `sae-lens`, `torch`… with no versions). A fresh install may behave differently from what they ran.

## 2. What each script does, and whether we need it

| Script | What it does | Model / SAE | Need it? |
|---|---|---|---|
| `03_gemma_phase3_streaming_sweep.py` | **The G0 pipeline**: Gemma-2-2B, Gemma Scope layer-12 SAE, WikiText-2, RTN INT8…INT4, perplexity + per-feature Pearson survival; optional matched-perplexity pruning | Gemma-2-2B (TransformerLens), `gemma-scope-2b-pt-res-canonical` | **YES.** Cells 1–21 only; skip the pruning cell for now |
| `06_gemma_sliding_window_ppl_check.py` | HF sliding-window perplexity (window 2048, stride 512, with BOS) for FP / INT8 / INT7 / INT6. **No SAE code.** | `google/gemma-2-2b` via HF | **YES.** Cheap, and it measures the baseline we are missing |
| `05_phase5_feature_class_predictor.py` | Logistic regression of survival on rarity, activation, concentration; AUC | reads per-feature CSVs | Not needed. Its key finding is already checkable offline from their released per-feature file (§5) |
| `03` cell 18 (matched-perplexity pruning) + Phase 5b overlap | Quantization-vs-pruning overlap (P3 in our proposal) | same | **Later**, only if we pursue the pruning extension |
| `00_smoke_test*.py`, `01_…`, `02_…`, `04_…` | Pythia-70M smoke tests, sweeps, stability ablations | Pythia-70M | **No.** Not our model, not our claims |

## 3. Compute audit

### 3.1 `03_gemma_phase3_streaming_sweep.py` (cells 1–21)

How the script is built (from the code):
- It loads **two** full models: `model_ref` and `model_work`, both bf16 on the GPU (Cell 7), and a **CPU copy** of the work model's state (Cell 11).
- The SAE runs in fp32 (`acts.float()`), batch size 1, 256-token windows, TransformerLens `run_with_cache`.
- Per batch it moves a `[256, 16384]` float64 feature block to CPU and updates five running sums.

| Resource | Requirement | Basis |
|---|---|---|
| **GPU memory** | **Does not fit on one 11 GB card; fits on two.** *Measured* with a plain-HF stand-in on 2 × RTX 2080 Ti (10.57 GiB usable each): 5.44 GiB on the card holding model + SAE, 5.15 GiB on the other. TransformerLens makes the model bigger: it **un-ties the unembedding** (`unembed.W_U` is a separate parameter, +≈1.2 GB) and builds a transient `W_U.T.contiguous()` (≈1.2 GB) on every forward (read in TL 3.9.0 source). Expect ≈ 7.5–8.4 GiB peak on the card with `model_ref`+SAE, ~2 GiB headroom. R0 will measure it. | HF measurement + TL source |
| Host RAM | ~6 GB resident (state copy) plus **transient ~10 GB per model load** while TransformerLens converts the HF checkpoint, so plan for **≥ 16 GB** | 5.23 GB state clone + load transients |
| Disk | **~11 GB** for the HF download (official gated repo is fp32, 10.5 GB), SAE ~0.3 GB, WikiText ~6 MB. On Ada this must live on **`/home2`** (19 GB free) because `/share1` is not mounted on compute nodes (§4.4) | `progress_log.md` fp32 size; Ada probes |
| Gating | TransformerLens name `gemma-2-2b` resolves (as I understand its alias table; not verified) to the **gated** `google/gemma-2-2b`, so an HF token with an accepted license is needed (`DO_HF_LOGIN=True`) | Cell 4 / Cell 6 |
| bf16 | Turing has no native bf16; PyTorch emulates it. *Measured* on the 2080 Ti: bf16 matmul 1.90 ms vs fp32 1.00 ms vs fp16 0.36 ms (256×2304×9216). It works and is correct (logits bit-identical across the two cards), but is ~2× slower than fp32 | benchmark |

Cost of the real TransformerLens pipeline on the Ada 2080 Ti pair, **measured in R0** (job 23355): perplexity pass 3.3 it/s (0.30 s per 256-token window); streaming pass 1.08 it/s (0.93 s per window: two forwards, two SAE encodes, float64 accumulation on the CPU); both cards peak at **8.2 GiB of 11.0 GiB**; host RSS peak 21.8 GB (48 GB requested); model download ~18 min per job onto node-local scratch. One condition = 1,953 windows × (0.30 + 0.93) s ≈ **40 min**.

| Configuration | Windows | Conditions | Time |
|---|---|---|---|
| `3a_test` as shipped (20k tokens; baseline + INT8 + INT6) | 78 | 3 | **measured: 6 min compute** (+ ~18 min first-time model download) |
| **One G0 arm: 500k tokens; baseline + INT0, 16, 12, 8, 7, 6** | 1,953 | 7 | **≈ 4.7–4.8 h** (7 × 40 min + model restores) |
| Three arms (no-BOS, BOS-masked, BOS-kept) | 1,953 | 21 | **≈ 14.5 h sequential; ≈ 5 h in parallel on three nodes** (needs `-A nlp`, untested) |

The earlier projection (2.4 h per arm) came from a plain-HF stand-in and was 2× too optimistic: TransformerLens is ~2.3× slower per streaming step. A faster GPU would help less than expected, because about 0.3 s of each streaming step is CPU-side float64 accumulation inside their code.

Known breakages in the shipped script (it is a notebook export):
- `display(...)` (lines ~537, 694, 759) exists only in Jupyter. Replace with `print`.
- Cells 24–28 call functions that are **not defined anywhere in the file** (`corrected_ppl_sweep`, `compute_ppl_shifted_manual`, `restore_work_from_ref`). Stop after cell 21.
- Configuration is by editing constants (`RUN_MODE`, `TOKEN_BUDGET`, `BITS_TO_TEST`, `RUN_PRUNING`). That is a config change, not a code change, but any edit beyond constants makes it "modified", and we must say so.
- The script reloads the dataset from the HF hub (WikiText-2 train for budgets above 20k), so it needs network or a pre-warmed cache.

### 3.2 `06_gemma_sliding_window_ppl_check.py`

| Resource | Requirement | Basis |
|---|---|---|
| GPU memory | one HF model at a time: 5.2 GB weights plus logits `[2048, 256000]` (≈ 1 GB bf16, ≈ 2 GB fp32 for the loss) and attention, so roughly **≥ 12–16 GB**. They note "reduce WINDOW_SIZE to 1024 if OOM" | their comments, my arithmetic |
| Passes | 4 conditions (FP, INT8, INT7, INT6) × ~565 windows of 2,048 tokens ≈ 2,260 forwards, each ≈ 10.7 TFLOP; the model is **re-loaded 4 times** | their code |
| Time | similar order to the G0 core run *(estimate)*; a 50k-token smoke mode exists (`RUN_MODE="smoke_test"`) | |
| Local notes | writes to `/content/...` (Colab convention). Change `OUT_DIR` | line ~70 |

### 3.3 Dependency risk

- `transformer-lens` and `sae-lens` are unpinned. The script has compatibility code for `sae_lens` ≥ 6 (`SAE.from_pretrained` returning a single object) and falls back to `blocks.12.hook_resid_post` if the SAE config lacks `hook_name`.
- Their stack may conflict with our pinned one (`transformers==5.17.0`, `torch==2.13.0`). **Use a separate virtualenv.** Record the exact versions installed, since they are part of the result.

## 4. Where to run it: verified on Ada (2026-10-15, read-only probes plus 2-minute GPU jobs)

### 4.1 Access
`ada.iiit.ac.in` now refuses SSH and tells users to use `ada-gw1` / `ada-gw2`. `ssh karma@ada-gw1` works non-interactively with the local key (campus network). Slurm accounts: `nlp` (QOS normal, no limits shown) and `research` (QOS medium: 4 days max wall, **4 GPUs, 40 CPUs, 125 GB per user**, 1 node per job, 20 jobs). Another job of yours (`plan02-train`, 1 GPU) shares those caps, so only one 2-GPU job fits at a time under `research`. Use `-A nlp` if parallel arms are needed.

### 4.2 What GPUs we can actually use

| Partition | GPUs | Open to our accounts? |
|---|---|---|
| `u22` (77 nodes) | gnode001–042: **GTX 1080 Ti** 11 GB (cc 6.1, probed gnode001); gnode044–092: **RTX 2080 Ti** 11 GB (cc 7.5; probed gnode046/048/057/069/074). Slurm tag `--constraint=2080ti` selects most 2080 Ti nodes (a few, e.g. gnode074, are untagged). 3–4 GPUs and 125 GB RAM per node | **Yes** |
| `plafnet2` (gnode113–116, 118) and `rrc` (gnode117): **RTX 3090 24 GB** | | **No**: `AllowAccounts=plafnet2` / `rrc` |
| `ihub` (gnode094–107): RTX 3080 Ti 12 GB | | **No**: AllowAccounts excludes `nlp`/`research` |
| `u22-cpu` | none | No |

**No GPU larger than 11 GB is reachable.** An unmodified run of Duan's script (two models on one device) is therefore impossible on Ada under our accounts. The 24 GB 3090 nodes would run it verbatim, but a lab with `plafnet2`/`rrc` access would have to lend them (optional cross-check, not required).

### 4.3 Decision: two RTX 2080 Ti per job, with a placement-only edit
`--constraint=2080ti --gres=gpu:2`: `model_ref` + SAE on `cuda:0`, `model_work` on `cuda:1`. The edit is numerics-preserving (same dtype and ops; *measured*: identical logits across the two cards). The gnode001–042 GTX 1080 Ti nodes are **not** used (cc 6.1, no tensor cores, and recent PyTorch wheels may no longer include sm_61; untested).

### 4.4 Environment facts that bite
- **`/share1` is not mounted on compute nodes** (jobs writing there die in 2 s with no output). Use `/home2/karma/...` (NFS, visible everywhere) and node-local `/scratch`.
- `/share1` also has a **3,000-file quota** (a conda env there fails with EDQUOT). `/home2/karma` has 19 GB free after the 6 GB env.
- Compute nodes have **outbound internet** (huggingface.co, pypi.org, repo.anaconda.com all HTTP 200).
- Stack built at `/home2/karma/duan-verify/env`: Python 3.10, **torch 2.14.1+cu126** (the cu126 wheel works on the 2080 Ti nodes; their drivers vary from 570 to 595, so the default newest wheel is not guaranteed), **transformer-lens 3.9.0, sae-lens 6.54.5, transformers 5.19.0**, datasets 5.1.0. Freeze in `reports/g0/ada_env_freeze_20261015.txt`. Our own paper ran on transformers 5.17.0 / torch 2.13.0, so these differ (§7 gate F5).
- Default HF attention backend in this stack is `sdpa` (*measured*), relevant to our pipeline and to script 06.

### 4.5 What is verified and what is not
| Item | Status |
|---|---|
| Ada access, partitions, GPU types, QOS caps, internet, quotas, mounts | **verified** (re-checked independently by a second reviewer, with corrections folded in) |
| torch + CUDA + bf16 on 2080 Ti; two models fit on two cards; cross-card numerics identical; SAE loads through the installed `sae_lens` (`hook_name` = `blocks.12.hook_resid_post`, fp32); WikiText-2 train has 2,433,794 tokens (≥ 500k, 1,953 windows) | **verified** by `bench_fit.py` |
| Patched runners compile; patched metric function is **identical** to the original on stub data in the default arm; snapshots rebuild totals and per-feature correlations exactly; BOS arms give 256 / 256 / 257 positions; wandb hooks fire (heartbeat every 100 windows, one `log_condition` per summary) | **verified** (`scripts/g0/unit_test_runner.py` on Ada) |
| wandb logging helper (`scripts/g0/g0_wandb.py`): 17 unit tests with a fake wandb (disabled mode never touches wandb, init failure and logging errors never propagate, resume id, artifact excludes snapshots, gate metrics) plus 4 generator regression tests; **live self-test against the real service** (`scripts/g0/selftest_wandb.py`: run logged to `karma-skz-/anlp-project`, metrics and gate verdict read back through the API, run deleted) | **verified** (built test-first: each suite was run red before the code existed) |
| **TransformerLens loading the gated `google/gemma-2-2b` and running on `cuda:1`; TL memory and speed; script 03 end to end (3a_test)** | **verified by R0** (job 23355): asserts pass, peak 8.2 GiB per card, 3.3 it/s / 1.08 it/s, outputs + wandb run complete. The first live heartbeat inside a long arm is checked after launching R1 |
| Script 06 end to end | not run (same gating); it also did not parse as shipped (§6) |

**Update:** `hf auth login` is done (token present on Ada; gated repo returns HTTP 206 from Ada). **Download lesson:** the first R0 attempt stalled at 4 of 10.5 GB after 35 minutes because I had forced `HF_HUB_DISABLE_XET=1` (plain HTTP, ~1.7 MB/s, repeated dropped connections). With xet enabled the same files download at ~29 MiB/s. The model is now pre-downloaded on the login node (`scripts/g0/predownload.sh`, xet chunk cache on login-node `/tmp`, removed afterwards) so jobs never wait on the network.

## 5. Verification of the released results we depend on (done offline, no GPU)

Scope: only the parts that matter to our open questions (Gemma-2-2B, layer 12, RTN INT8–INT4, perplexity). Pythia, pruning and the feature-class predictor were not checked.

| Check | Result |
|---|---|
| Recompute the INT6 summary from `results/per_feature/phase5b_gemma_rtn_int6_vs_pruning_per_feature_overlap.csv` | **Matches exactly**: 7,144 features, mean r 0.8911, median 0.9027, survived 51.302%, degraded 48.698%, damaged 0.000% |
| Token count | 499,968 = 1,953 × 256 ✓ |
| Quantized parameter count (`2,024,275,968`) | = 26 × (q + k + v + o + gate + up + down) for Gemma-2-2B. **Same 182 modules we quantize** |
| `exp(loss)` vs reported perplexity (chunked 6.128 → 458.5; sliding 3.837 → 46.4) | consistent ✓ |
| Sliding-window table's feature columns | **identical to the chunked run** (81.257 / 51.302 / 99.090). There is no separate feature measurement under sliding-window or BOS |
| Baseline perplexity: chunked 458.5 vs sliding (BOS, 2048) 46.4 | ~10× gap. The 46.4 is itself unexplained (`issues.md` B1) |
| Perplexity vs bits (INT8 / INT7 / INT6) | chunked +2.6% / −5.6% / +4.0%; sliding +0.8% / −6.5% / −3.5%. **Non-monotone in both** |
| What predicts survival (INT6 per-feature data) | Spearman(r, max activation) **0.78**; Spearman(r, firing rate) **−0.085**. Within every rate quintile, above-median-activation features survive 66–91% vs 21–27% below the median. This directly affects our matched-control design (`issues.md` C6) |

**Verdict:** their released tables are internally consistent and reproduce from their own per-feature data, but they describe a degraded pipeline, so "reproducing their numbers" proves reproducibility, not correctness.

## 6. Two layers: the faithful baseline and the labelled diagnostics

**Layer 1, the baseline replication (their code, vendored).** `third_party/duan2026/` holds their repository downloaded fresh (twice, byte-identical; content tree hash `a84e43dac17e5a9c6b1404e4b502c97d2b184b0168e860035432e4c261e44adc`), pruned to the Gemma Phase-3 study. Retained byte-identical in `pristine/`: `LICENSE`, `README.md`, scripts 03 and 06, their published Gemma tables and the INT6 per-feature file. Removed: Pythia scripts, the feature-class predictor, `src/`, unrelated results, env files (hashes kept in `MANIFEST.sha256`). The file we run, `scripts/03_gemma_phase3_streaming_sweep.py`, equals their file plus a whitelist: delete cells 22+ (they call three functions never defined in the file and run after every result file is written; they also re-define `target_tl_weight` without `W_gate`, which therefore never affects the shipped results); `RUN_PRUNING = False` in their `3b_full` branch (their own flag); and four device references (`WORK_DEVICE`, the dtype-test string comparison that would silently give float32 for `"cuda:1"`, loading `model_work` on it, restoring it there). No edit for `display()`: it works under IPython. `scripts/g0/tests/test_vendor.py` rebuilds the working copy from the pristine file plus the whitelist in `scripts/g0/verify_vendor.py` and requires byte equality over the whole file (top-level statements included), checks every retained file against the recorded upstream hashes, and carries a committed mutation suite (13 mutations of their math, thresholds, constants, config, structure; all rejected). Results are compared with their published table by `scripts/g0/baseline_report.py` and uploaded to wandb *after* the run, from their own output files. Launcher: `scripts/g0/run_baseline.sbatch` (refuses to run unless the deployed file matches `working.sha256`). Configuration: their shipped `3b_full` with pruning off. We never ran their unmodified script (no >= 16 GB GPU reachable), so equivalence of the device edits to the single-device behaviour is argued, not demonstrated. An earlier generator-based "faithful runner" was removed: its test compared only 13 function bodies and its mutation check was a one-off.

**Layer 2, diagnostics (modified code, always labelled).** `scripts/g0/make_duan_runners_v3.py` (the "v3" runner) adds experimental arms and instrumentation. Never call its results a replication.

Script 03, v3 runner (cells 1-21 only):
- **[FIX]** `display()`→`print` (4×). Cells 22+ dropped (they call three functions never defined in the file).
- **[PLACE]** `WORK_DEVICE` (string `"cuda:1"`; a `torch.device` would drop the index inside TransformerLens). **Trap:** the shipped dtype test is `device == "cuda"`, so `"cuda:1"` would silently load `model_work` in **float32**; patched to `startswith("cuda")`. `restore_model_work()` is also patched, otherwise it would move the model back to `cuda:0` and run out of memory.
- **[ARM]** `G0_RUN_MODE=g0_core` (no pruning; configurable tokens and bit ladder); `G0_BOS` in `none` (shipped) / `prepend_keep` / `prepend_mask`. The BOS arms prepend BOS to the **identical** 256-token windows (rows become 257 long) and `prepend_mask` drops exactly position 0 from the feature statistics, so arms differ **only** by BOS. (An earlier draft sliced 255-token windows, which shifted the text; the reviewer caught this and it is fixed.) Bit ladder default `0,16,12,8,7,6`.
- **[ARM]** `INT0` = restore-only control: no quantization, so r must be exactly 1.0 and perplexity identical. This tests that [PLACE] is numerics-preserving and that restore works.
- **[META]** wandb logging (project `anlp-project`; heartbeat, per-condition metrics, gate metrics, artifact); dtype/device asserts after load and after every restore; `run_meta.json`; the exact `tokens_2d.pt` saved; relative L2 weight change logged per width; per-condition resume; cumulative-sum snapshots every 100 windows (block bootstrap, no extra GPU cost).

Script 06: **does not parse as shipped** (`from google.XXXX-13 import files`, an anonymization artifact at line 380, verified with `ast.parse`); it writes to `/content`; and `display()` runs before `to_csv`, so a notebook-only `NameError` would discard a multi-hour run. The v3 generator drops the Colab packaging cell, uses `print`, takes mode/bits/output/attention backend from environment variables, and uses `dtype=` instead of the deprecated `torch_dtype=`.

Diagnostic runs (R0-R5, outputs under `/home2/karma/duan-verify/out/<arm>`, copied to `reports/g0/`):
1. **R0** smoke: shipped `3a_test` through the v3 runner (done, see `reports/g0/r0_smoke/`).
2. **R1** arm A, as shipped: `g0_core`, `G0_BOS=none`, 500k tokens, `0,16,12,8,7,6`.
3. **R2** arm B: `prepend_mask`, same ladder.
4. **R3** arm C (**mandatory**, it is the regime of our own pipeline, which keeps BOS in the statistics): `prepend_keep`, same ladder.
5. **R4** script 06: smoke, then full, default attention; plus one FP-only run with `G0_ATTN=eager` (`G0_BITS06=none`). Explains their 46.4 baseline perplexity.
6. **R5** our pipeline on **their exact tokens** (`tokens_2d.pt` from R1), in three regimes (none / mask / keep) × {bf16 SAE as ours, fp32 SAE as theirs} × {our per-row grouping, their Q/K/V grouping across heads}. A new script on our side that must import `src/` from this repo with a recorded hash, and must run with the same ladder.

**Not needed now:** Pythia scripts, the feature-class predictor, INT5/INT4 in the diagnostics (the baseline includes them because their shipped `3b_full` does).

## 7. Closure protocol (decided before running; every outcome is terminal)

**Principle.** The reference for validating *our* pipeline is **their code as executed on Ada (R1–R3)**, not their published numbers. That makes the verdict independent of whether their paper's numbers reproduce on different hardware. Their published numbers only decide one informational row (Q1).

### 7.1 Why INT12/INT10 are not the controls
The quantizer does its arithmetic in bf16 and stores bf16 weights. A numpy bf16 emulation of the exact recipe shows INT16 already changes 12% of the stored weights (relative L2 change 1.7e-3, the bf16 re-rounding floor), and INT12 is indistinguishable from INT16 (2.3e-3). So INT16/INT12 are **small-perturbation controls at the bf16 floor**, not "lossless". The test is therefore a **scaling test**: in a signal-limited regime (1 − median r) is proportional to the actual weight perturbation (our own data: ratio roughly constant from INT8 to INT4). The runner logs the real relative L2 weight change per width.

### 7.2 Gates and measured quantities
| Name | Definition | Pass condition |
|---|---|---|
| **Gate 0: run integrity** | INT0 rung gives r = 1 for every active feature and perplexity identical to the baseline; asserts pass; **FP anchors vs the released data**: baseline perplexity within ±3% of 458.5; active features within ±10% of 7,144; Spearman of per-feature firing rate vs the released per-feature CSV ≥ 0.99 | all pass, else the run is not the same experiment: fix and rerun (max 2 reruns); not a scientific outcome |
| **Gate 1: noise floor** (R1) | (a) median r at INT16 and INT12 ≥ 0.99; (b) slope of log(1 − median r) vs log(relative L2 weight change) over INT12…INT6 within [0.5, 1.5] | both → signal-limited; otherwise **noise/chaos-dominated** |
| Q1 reproduction of published numbers (**taken from the Layer-1 faithful baseline**, `baseline_report.py`) | baseline INT7 survival within max(5 pp, block-bootstrap 95% half-width) of 81.26%; INT6 within max(10 pp, half-width) of 51.30%; Spearman(r at INT6, released `corr_rtn`) ≥ 0.9; perplexity signs (INT7 improves, INT6 worsens) | informational only |
| Q3 BOS effect | INT7 survival in R2 (and R3) minus R1; material if ≥ 10 pp and larger than the bootstrap half-widths | informational only |
| **Q4 equivalence of our pipeline**, per regime X ∈ {none, mask, keep} | R5(X) vs R1/R2/R3: \|Δ survival\| ≤ 5 pp at INT7 and INT6, **and** Spearman of per-feature r at INT6 ≥ 0.95, **and** FP-level agreement (Spearman of per-feature firing rate and max activation ≥ 0.99) | gate for the dispositions below |

Thresholds are judgement calls, not derived; the block-bootstrap intervals (from the snapshots) are reported next to every number so a reader can see how much slack they hide.

### 7.3 Dispositions (exhaustive; each ends the question)
| Gate 0 | Gate 1 | Q4 | Disposition | What we may say |
|---|---|---|---|---|
| fail | n/a | n/a | **Invalid run** | nothing; fix and rerun (≤ 2 reruns) |
| pass | noise/chaos-dominated | any | **D-NOISE** | Their pipeline's feature statistics are noise-dominated even for near-lossless perturbations, so its survival numbers are not a usable reference. Cite Duan as prior work only. Our validity rests on §7.4 |
| pass | signal-limited | all three regimes match | **D-EQUIVALENT** | Our implementation reproduces their code in every regime. Differences from their published numbers are due to their input regime (if Q1 yes and Q3 material) or to environment (if Q1 no) |
| pass | signal-limited | match only with their Q/K/V grouping and/or fp32 SAE toggled | **D-EQUIV-UP-TO-TOGGLE** | Equivalent up to the named toggle; we keep ours and document it |
| pass | signal-limited | matches in some regimes only | **D-PARTIAL** | Equivalence stated only for the regimes that match; claims restricted accordingly |
| pass | signal-limited | no match anywhere in the grid | **D-UNRESOLVED** | We make no claim of equivalence and say so; our results stand on §7.4 alone. No further hunting beyond the pre-listed grid |
| R1 cannot run after ≤ 2 attempts (record versions and errors) | n/a | n/a | **D-NOT-RUNNABLE** | Duan's numbers are cited as unverified, with that caveat; do not retry |

After any disposition: record it in `reports/progress_log.md`; archive `reports/g0/` (outputs, `run_meta.json`, snapshots, tokens) and the Duan zip hash; **no further verification of Duan's work is planned**. This protocol is terminal for the question as defined (is our instrument valid relative to their code and how do the numbers relate). It cannot stop a reviewer from asking a different question later, such as other layers, models or bit-widths.

### 7.4 Our own pipeline's validity (independent of Duan; required before any paper claim)
F1: own WikiText-2 perplexity with BOS under the default (`sdpa`) and `eager` attention backends. F2: HF-vs-TL FP feature agreement on identical tokens (part of Q4). F3: mean L0 ≈ 82 and explained variance on text. F4: FP-vs-FP noise floor under a different batch layout plus the INT16/INT12 scaling test on our pipeline. F5: log versions; Ada's stack (transformers 5.19.0 / torch 2.14.1) differs from the paper's (5.17.0 / 2.13.0), so repeat F4 under the pinned versions or document the difference. F6: FLORES content validity (`issues.md` C1) is a separate problem outside G0.

## 7.5 Baseline result (Layer 1, job 23472, 2026-10-16)

**Q1 passed on all four bands.**

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

## 8. Open items
- **Layer 1 baseline** (their `3b_full`, pruning off, ~4.1 h): runbook in `HANDOFF.md` section 3.
- Layer 2 (R1-R3, R4, R5): after the baseline, with the user's OK. R5 (our pipeline on their tokens) and the section 7.4 checks need a new script on our side; not written yet.
- Optional: ask a lab with `plafnet2`/`rrc` access, or the 96 GB RTX PRO 6000 machine's owner, for a 24 GB cross-check of the unmodified script.
- Commit the plan (this document plus `PLAN.md`) before launching, so the decision rules are fixed before the data exist (done together with the handoff push).

## 9. Independent re-verification (second reviewer) and what changed
A separate reviewer re-probed Ada and attacked the plan. I checked its main claims myself before adopting them.

| Reviewer claim | My check | Outcome |
|---|---|---|
| TL un-ties the unembedding, so a TL model is ≈ 6.4 GB, not 5.23 GB | Read `gemma.py:131` and `unembed.py`: separate `W_U`, transient `W_U.T.contiguous()` | **Confirmed**; memory estimate corrected (§3.1) |
| Script 06 does not parse | `ast.parse` → SyntaxError at line 380 | **Confirmed** |
| `restore_model_work` would move the model back to `cuda:0`; `cuda:1` would load fp32 | Read the script | **Confirmed**; both patched, plus asserts |
| TL moves input tokens to the model device, but `move_to_and_update_config` drops the device index for a `torch.device` | Read `HookedTransformer.py:421-422` and `devices.py:172` | **Confirmed**; we pass strings only |
| My BOS arm slid the text windows | Re-read my edit: `tokens[:n*255]` | **Confirmed** (my error); fixed: prepend to identical 256-token windows |
| INT12 is not near-lossless | My own numpy emulation had already shown it | **Confirmed**; ladder is now 0, 16, 12, 8, 7, 6 with a scaling test |
| Q4 needs three regimes, a Q/K/V-grouping arm, an SAE-dtype arm; Q2 must gate Q4; FP-level anchors; per-feature comparison | Design review | **Adopted** (§6–§7) |
| `hf auth login` should use `HF_HOME=/share1/...` | Contradicted by my probe: `/share1` is **not mounted on compute nodes** | **Rejected**; `/home2` is used |
| "gnode001–042 = 1080 Ti is unverified" | I probed gnode001 (GTX 1080 Ti) | Confirmed for gnode001; untagged nodes are mixed (gnode074 is a 2080 Ti) |
| "Defaults reproduce shipped behaviour" was not literally true | Default `RUN_MODE` is now the smoke mode and the output directory name changed | **Confirmed**; wording corrected in §6 |
