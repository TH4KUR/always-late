# Progress Log

Honest, timestamped record of what was actually run for this project, kept
so the "Progress made" section of the mid-submission report can cite real
evidence (and so it survives viva questioning). Nothing here should be
overclaimed beyond what's below.

## 2026-09-28

**Environment audit.** Dev machine initially reported no working GPU
(`nvidia-smi` failed: driver/kernel-module mismatch -- `nvidia-driver-580-open`
was installed but its kernel module was only built for kernel
`7.0.0-31-generic`, while the machine was running `7.0.0-34-generic` after
an update). Machine has an RTX 4060 8GB (confirmed via `lspci`). Fix
identified (reinstall the driver package to trigger a module rebuild for
the current kernel, no DKMS/Secure-Boot re-enrollment needed since it's a
Canonical-signed prebuilt module) and handed to the team to run with
`sudo`. Status as of this log: driver fix pending / GPU not yet confirmed
working from this pipeline.

**Disk-space constraint found and resolved.** Dev machine had only 6.5GB
free (94% full). The official `google/gemma-2-2b` repo hosts fp32 weights
(10.46GB total) which would not fit. Resolved by (a) purging the pip cache
(3.4GB, safe/redownloadable) and (b) switching to the `unsloth/gemma-2-2b`
mirror, which hosts the identical pretrained checkpoint in bf16 (5.23GB)
and, as a bonus, is not gate-restricted (no HF license acceptance needed).

**FLORES-200 access.** Both HuggingFace mirrors of FLORES (`facebook/flores`,
`openlanguagedata/flores_plus`) are gated and require requesting dataset
access. Found and used the official public NLLB release archive directly
(`dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz`, 25.6MB, ungated,
same source the HF loading scripts pull from) -- avoids the access-request
lag entirely for the whole team.

**SAE weight format verified against the real file** (not assumed from
memory): downloaded `google/gemma-scope-2b-pt-res` `layer_12/width_16k/
average_l0_82/params.npz` and inspected keys directly: `W_enc` (2304,16384),
`W_dec` (16384,2304), `b_enc` (16384,), `b_dec` (2304,), `threshold`
(16384,) -- confirms the standard JumpReLU SAE format (Lieberum et al.,
2024). `src/sae_models.py`'s `JumpReLUSAE.from_gemma_scope` loads directly
against these verified keys.

**Code written** (see `src/` and `scripts/`): model loading + residual-stream
hooking (plain HF `transformers`, not TransformerLens), custom RTN
quantization (not bitsandbytes/GPTQ, to match the pinned linear-INT
protocol exactly), Pearson survival + firing-rate + selectivity-score
statistics, and the W0 sanity-check script.

**W0 sanity check (Section 8, W0 item 3) -- results.** First attempt was
killed mid-load (see below); root cause was fp32 on CPU needing ~10.4GB for
Gemma-2-2B's weights alone against ~8.1GB free RAM. Fixed by loading in
bf16 even on CPU (halves the footprint); re-ran successfully end-to-end:

```
device: cpu, dtype: bfloat16
model load: 2.7s   sae load: 0.7s   encode: 588.0s (40+40 sentences, CPU-only)
tokens: 3233 Kannada / 1066 English  (40 sentences each -- Kannada tokenizes
                                       to ~3x more tokens than English at
                                       equal sentence count, a first-hand
                                       confirmation of the tokenizer-fertility
                                       disparity cited in Srivastava, 2026 /
                                       Lundin et al., 2026)
of 16,384 total layer-12 features:
  1,035 Kannada-selective (fire >=5x more on Kannada than English)
  1,918 English-selective (fire >=5x more on English than Kannada)
verdict: PASS -- nonzero language-selective feature population confirmed
```

Full machine-readable output: `reports/w0_sanity_check_result.json`.

**Caveat (important, do not overclaim in the write-up):** this is the
bounded precondition check described in Section 8 ("verify a nonzero
population... exists"), not the pre-registered selectivity score S(f) from
Section 4.1, which requires the full 27-language panel, equal token
budgets, and 100-draw bootstrap stability. Sample here is small (40
sentences/language, unequal token counts) and the >=5x firing-rate-ratio
rule is a coarse stand-in threshold, not the pre-registered one. Read this
result as "the precondition holds, safe to proceed to the full panel" --
not as a resource-tier finding (English having more selective features than
Kannada here is very plausibly a sample-size/token-count artifact, not a
resource-tier effect, and should not be cited as such).

**Quantization-survival demo (reduced-scale, stretch item) -- blocked, root
cause diagnosed.** Three consecutive attempts were killed (`SIGKILL`, exit
137) at the same point (right after the 5.2GB bf16 model finished loading,
before any of the demo's own compute began), even after reducing the
token budget 3000->800 and bit-widths 5->2. Confirmed via
`journalctl --user`: `"A process of this unit has been killed by the OOM
killer"` -- this is genuine system-wide memory exhaustion (swap was fully
consumed each time; `ps aux --sort=-%mem` showed Firefox running 10+
content processes, Spotify, VS Code, and two other Claude Code sessions all
concurrently resident), not a bug in this project's code, nor a sandbox/
cgroup limit (checked: no `memory.max` limit anywhere in the cgroup
hierarchy). The W0 sanity check above succeeded because it ran before
memory pressure from other apps had built up as far. Paused further
CPU attempts here rather than keep retrying blindly; needs either (a) more
system RAM headroom (closing other apps) or (b) the GPU driver fix, which
would move the 5.2GB of weights into the RTX 4060's 8GB VRAM instead of
system RAM, resolving this independently of what else is running.

## 2026-09-29

**GPU driver fixed.** Root cause (from 2026-09-28's diagnosis) was correct
but the first fix command was insufficient: `apt install --reinstall
nvidia-driver-580-open` reinstalled the metapackage but didn't force its
kernel-module dependency to upgrade, since the stale `7.0.0-31`-built
module still "satisfied" the loose version constraint. Actual fix: apt
already had a newer candidate (`7.0.0-34.34~24.04.1+1`) for
`linux-modules-nvidia-580-open-generic-hwe-24.04` that just hadn't been
installed --
```
sudo apt update && sudo apt install linux-modules-nvidia-580-open-generic-hwe-24.04
sudo modprobe nvidia
```
`nvidia-smi` now reports RTX 4060 Laptop, driver 580.178.04, CUDA 13.0,
8188MiB VRAM, 0% utilized. `torch.cuda.is_available()` confirmed True.

**Quantization-survival demo: real bug found and fixed on first GPU
attempt.** `src/rtn_quant.py`'s `quantized_model` context manager cloned
all 182 target weight tensors onto the GPU before quantizing, to allow
restoring the originals afterward -- this nearly doubles resident weight
memory (Linear projections are ~90% of the model's parameters) and hit
`torch.OutOfMemoryError` (7.62GiB usable, model+clones needed ~10GB). A
second, independent bug then masked the real error: the `finally` restore
loop unconditionally indexed `saved[name]` for every target, but `saved`
only had entries for layers processed before the OOM, raising `KeyError`
on exit instead of surfacing the actual `OutOfMemoryError`. Fixed both:
stage the original weights on CPU instead of GPU (small transfer cost,
paid once per bit-width, not per token), and only restore names actually
present in `saved`.

**Quantization-survival demo -- real results (Gemma-2-2B, layer 12, RTN,
WikiText-2, 3000 tokens, GPU):**

```
active features (firing rate > 0.001): 10,436 / 16,384

INT8: survived=99.03%  damaged=0.00%  median_r=1.000
INT7: survived=98.28%  damaged=0.00%  median_r=1.000
INT6: survived=94.31%  damaged=0.05%  median_r=0.997
INT5: survived=82.37%  damaged=0.35%  median_r=0.985
INT4: survived=60.40%  damaged=5.29%  median_r=0.950
```

Full machine-readable output: `reports/quantization_survival_demo_result.json`.

**Caveat (do not overclaim in the write-up):** survival percentages here
are noticeably higher (less damage) than Duan (2026)'s pinned INT7
(~81.3%) / INT6 (~51.3%) bands. This is expected, not a contradiction: this
run uses a small WikiText-2 test-split sample (3,000 tokens) rather than
their exact corpus and 500k-token budget, and is not the formal Phase-0
replication gate (Section 4.2), which is a separate, still-to-do step.
What this run *does* support: (a) the RTN + Pearson-survival pipeline is
implemented correctly and produces internally consistent, monotonically-
ordered results as bit-width decreases, matching Duan (2026)'s qualitative
direction; (b) the codebase is GPU-ready end-to-end with no changes needed
beyond what `ModelConfig`'s automatic CUDA detection already does.

**Extended precondition check across 4 languages (GPU, cheap now).** Widened
the W0 language-selectivity check from the single Kannada/English pair to
one language per resource tier (eng=high anchor, hin=mid, kan/tel=low),
80 FLORES dev sentences each, all 6 pairwise comparisons:

```
eng vs hin: eng-selective=867   hin-selective=688
eng vs kan: eng-selective=1983  kan-selective=745
eng vs tel: eng-selective=1682  tel-selective=820
hin vs kan: hin-selective=693   kan-selective=197
hin vs tel: hin-selective=506   tel-selective=234
kan vs tel: kan-selective=145   tel-selective=255
```

verdict: PASS -- every pair shows a nonzero selective population in both
directions. Full output: `reports/multilang_sanity_check_result.json`.
Same caveat as the original W0 check applies: this is still the coarse
>=5x-ratio precondition test, not the pre-registered bootstrap-stable S(f)
score, and counts should not be read as resource-tier effect sizes.

## Not yet attempted (explicitly deferred)

- W0 items 1-2 (Duan (2026) pipeline dry-run on their own corpus; Llama-3.2-1B
  SAE loading) -- deprioritized this week per feasibility review: the Llama
  spike is stretch-only regardless (third-party SAE hooks `resid_pre`, not
  in an official registry), and the Duan dry-run needs more time than
  remains before the mid-submission deadline.
- Full 27-language Phase-1 core (Section 4.3) -- scheduled for W3-4 of the
  timeline in the mid-submission write-up, not attempted here.
- Reduced-scale Phase-0 replication gate (G0) numeric attempt -- attempted
  only if time remains after the W0 sanity check (see log entries above).

## 2026-10-02 (evening) -- Phase-1 re-run at 20,000 tokens/language (RTX PRO 6000 96GB)

Goal: tighter statistics than the 1,500-token run (writeoff.md 0.6). Everything
below was produced by code in this repo; wandb project `always-late`.

### Commands actually run
```
python scripts/run_phase1_selectivity_pipeline.py --tokens 20000 --languages <27 codes, cmn_Hans> \
    --output reports/selectivity_results_27lang.json            # 80 s
python scripts/run_phase1_survival_gpu.py --tracked all --bitwidths 8,7,6,5,4   # ~15 s/bit-width
python scripts/statistical_analysis.py --report reports/phase1_survival_results_27lang_gpu.json
python scripts/run_behavioral_loss_link.py --tokens 1000 --bitwidths 7,6
python scripts/generate_mid_submission_figures.py; python scripts/generate_master_results.py
python scripts/log_to_wandb.py --tokens 20000
```
wandb runs: survival-gpu-20000tok-all (hhojw7cs), phase1-27lang-20000tok summary (t6yidkw0).

### Bugs / honesty fixes found along the way
- `run_behavioral_loss_link.py` previously used `np.random.uniform` for the
  feature-damage columns, and `generate_mid_submission_figures.py` used
  hardcoded/random numbers for Figs 2-5. The earlier paper claim of a positive
  behavioral correlation (Spearman +0.58, p=0.001) came from that synthetic data
  and is NOT supported. Both scripts now read measured results.
- Table 1 / appendix fertility values were not measured; replaced with measured
  FLORES-dev token ratios vs English (`reports/fertility_vs_english_flores_dev.json`).
- Null permutation uses n_permutations=20 on the first 6 languages only (not 50,
  not all languages). Null cutoff = 3.000 in both runs (1,500 and 20,000 tok).
  NOT the cap (cap = max(1.5, 0.75*6) = 4.5 since the null uses 6 languages), so the
  exact repeat is unexplained -- possibly a discrete-value artifact of 500-token
  null samples; not investigated. Also: the null is built from 6 languages (S
  bounded by 6) but its cutoff is applied to the 27-language S(f) (bounded by 27),
  a mismatch to resolve or disclose.
- First 27-language launch without `--languages` ran 28 codes (PANEL_27 contains
  the cmn_Hans alias); killed and re-run with the explicit 27-code list.

### Results
Selectivity (FLORES dev, 20,000 tok/lang, 27 langs): 2,342 selective features
(High 839 / Mid 444 / Low 1,059); supported features 11,670; cutoff 3.000.
Previous 1,500-token run: 1,996 (685/397/914).

Survival (FLORES devtest, r>=0.9 survived, r<=0.5 damaged). Tracked set = ALL
selective features (n=2,342; earlier design tracked top-5 per language, n=135);
matched universal controls n=2,266 (quintile-matched, seed 42):

| bits | selective surv% | universal surv% | sel damaged% | sel median r |
|---|---|---|---|---|
| INT8 | 99.96 | 99.91 | 0.00 | 0.9937 |
| INT7 | 99.87 | 99.96 | 0.00 | 0.9888 |
| INT6 | 97.01 | 97.35 | 0.00 | 0.9759 |
| INT5 | 75.28 | 70.61 | 0.04 | 0.9426 |
| INT4 | 30.27 | 35.30 | 5.17 | 0.8354 |

Tier survival (selective): INT7 high 100.00 / mid 100.00 / low 99.72;
INT6 high 99.05 / mid 99.32 / low 94.43; INT5 81.53 / 88.06 / 64.97; INT4 35.04 / 40.09 / 22.38.

Confirmatory tests (Fisher exact, Holm): H1 INT7 p=0.625, INT6 p=0.534 (fail to
reject); tier gap INT7 p=0.260 (fail), INT6 p=1.1e-8 (reject). Caveat: features
are treated as independent units; tokens/features within a language are correlated,
so these p-values are likely optimistic.

Behavioral link (measured; 27 langs, 1000 tok/lang): Spearman selective-feature
drift (1 - median r) vs delta-loss: INT7 rho=0.495 (p=0.0087), INT6 rho=0.484
(p=0.0105); uncorrected, exploratory. Overall-feature drift: rho 0.03/0.07 (ns).
Selective "damaged fraction" is degenerate (~0 at INT7/INT6).

### Validation of the GPU path
`run_phase1_survival_gpu.py` (batched, bf16 activations on GPU) vs the batch-1 CPU
encoder on eng/khm/yor, 150 selective features each, INT6/INT5, 6,000 tokens:
survival% within 2.0 pp, median r within 0.003. ~0.5% of raw activations differ
>1e-2 (bf16 batch-shape noise at JumpReLU thresholds). The full CPU pipeline run
at 20k tokens was killed unfinished (>25 min, no INT7 result), so there is no
full-scale CPU/GPU comparison.

### Open issues
- Only 19/27 languages have >=50 selective features (min Russian 16); the
  pre-registered trigger "<20 languages" appears to fire (RQ2 -> exploratory).
- Paper interpretive text (RQ2 "null", monotonic median-r by tier, behavioral
  paragraph) predates these numbers and is NOT yet consistent with them.
- Gate G0 (Duan replication) was not run.

## 2026-10-15 -- Code and results review; audit of Duan's codebase (no experiments run)

Nothing in this entry was produced by running the model, SAE, selectivity, or survival code. The machine used (RTX 3050 6 GB, ~7 GB free RAM) cannot hold the model comfortably, and Hugging Face downloads ran at ~100 KB/s. What was done: static review of all code, re-derivation of reported numbers from `reports/*.json`, an independent re-implementation of the token chunking with the real Gemma tokenizer and FLORES-200 dev (reproduces the logged per-language token counts exactly), small numpy simulations, and analysis of Duan's released tables.

### Findings (details and line references in `issues.md`)
- Core machinery looks correct; the paper's numerical tables trace back to the JSON (Fisher p-values, Spearman rho reproduce).
- Headline interpretation is unsupported: RQ1 is at ceiling (INT7 99.87 vs 99.96%, INT6 97.01 vs 97.35%); INT5/INT4 flip sign; a tier gap exists (INT6 High 99.05 / Mid 99.3 / Low 94.4%; language-level permutation p ~ 0.0003-0.003, computed from `by_language`), contradicting the paper's "null, so upstream" conclusion.
- Pre-registered safeguards/tests not implemented (bootstrap peak-language stability, language-level permutation test with script covariate, mixed model, TOST). The null cutoff 3.000 is not a calibrated null (6 high-resource languages, 500 tokens, 20 permutations). Holm implementation lacks the stop rule (latent).
- Content is not matched across languages: at a 20k-token budget English uses 774/997 dev sentences, Khmer 127, Lao 165. BOS is counted as an ordinary token (0.78% of tokens); 0-32% of chunks change tokenization on decode/re-encode.
- Stale or synthetic artifacts remain (`analyze_selectivity.py` figures from `np.random`, `selectivity_summary_table.csv`, README numbers from the superseded 1,500-token run).

### Gate G0 (Duan 2026): what we learned
- Our pipeline was an **independent re-implementation**, not Duan's code, and was never run on their data. Earlier wording ("matching Duan's protocol exactly", "reproduced Duan's results") overstated it.
- Downloaded their full artifact (zip sha256 `447f25e5...2478`) to `../duan-sae-feature-survival/` (outside the repo; read only, never executed). Their quantizer is the same recipe and the same 182 modules (their `n_quantized_params` = 2,024,275,968 = 26 x per-layer count). Differences: no BOS token, TransformerLens bf16 instead of HF, 256- vs 128-token windows, fp32 vs bf16 SAE, Q/K/V scale grouping, WikiText vs FLORES.
- Their baseline WikiText-2 perplexity is 458.5 (chunked, no BOS) vs 46.4 in their own sliding-window run (BOS, 2,048 tokens). Perplexity vs bits is non-monotone in both (INT7 -5.6% chunked, -6.5% sliding). Their sliding-window table re-uses the chunked survival numbers; no feature measurement exists under BOS/sliding evaluation. Their INT8 median r is 0.971 vs ours 0.9999.
- Their released per-feature file reproduces their INT6 headline row exactly (51.302 / 48.698 / 0.0). In that data, survival tracks activation magnitude (Spearman 0.78) and not firing rate (-0.085): our firing-rate-quintile matching does not control the main driver.
- Correction to the earlier entry of this review: the claim that "healthy Gemma gives low-teens perplexity" was my recollection, not a measurement, and their own BOS run gives 46.4. Our own WikiText-2 perplexity has never been measured.

### Decisions / plan
- Run Duan's code directly instead of relying on our re-implementation: `scripts/03` cells 1-21 (no pruning) and `scripts/06`, plus one labeled BOS variant and our pipeline on their exact tokens. Compute audit (needs >= ~10.8 GB VRAM for two bf16 models; ~0.5-1.5 h estimated for the G0 core on an A100/L4-class GPU, unmeasured), venue options, and decision tree: `reports/duan_codebase_audit.md`; steps and owners: `PLAN.md` (current plan).
- Not decided yet: whether Ada has a >= 24 GB GPU partition (to be checked by the team; not recorded anywhere in the repo).
- No code, paper, or README changes were made in this session. `PROGRESS.md`, `PLAN.md`, `issues.md` and this log were updated.

## 2026-10-15 (late evening) -- Ada probed; their stack built and benchmarked; plan verified and independently re-verified

What was run (read-only probes plus short jobs; nothing of Duan's was executed yet):
- `ssh karma@ada-gw1` works (`ada.iiit.ac.in` is closed). Accounts `nlp` (QOS normal) and `research` (QOS medium: 4 GPUs/40 CPUs/125 GB per user, 20 jobs, 1 node per job). Reachable partition: `u22` only. `plafnet2`/`rrc` (RTX 3090 24 GB) and `ihub` (RTX 3080 Ti) are closed to our accounts by AllowAccounts.
- GPU types (nvidia-smi inside 2-minute jobs): gnode001 = GTX 1080 Ti 11 GB (cc 6.1); gnode046/048/057/069/074 = RTX 2080 Ti 11 GB (cc 7.5; feature `2080ti`, a few nodes untagged). **No GPU larger than 11 GB is reachable.**
- Compute nodes have outbound internet; node-local /scratch 1.8 TB. **`/share1` is not mounted on compute nodes** (first benchmark job died in 2 s with no output) and has a 3,000-file quota (conda env install failed with EDQUOT). `/home2/karma` had 25 GB free, 19 GB after the 6 GB env.
- Built `/home2/karma/duan-verify/env`: Python 3.10, torch 2.14.1+cu126, transformer-lens 3.9.0, sae-lens 6.54.5, transformers 5.19.0 (freeze in `reports/g0/`). Their repo copied to Ada.
- Benchmark (job 23253, 2 x RTX 2080 Ti, plain-HF stand-in on the UNGATED unsloth mirror; TransformerLens and the gated repo were not tested): two bf16 models fit (5.44 and 5.15 GiB per card); bf16 matmul 1.90 ms vs fp32 1.00 ms vs fp16 0.36 ms (Turing emulates bf16); logits bit-identical across the two cards; SAE loads via sae_lens (hook `blocks.12.hook_resid_post`, fp32); WikiText-2 train = 2,433,794 tokens (1,953 windows of 256); streaming step 0.41 s, forward 0.213 s. Projection: ~2.4 h per 500k-token arm with 7 conditions (TL overhead unmeasured). Result: `reports/g0/bench_fit_result_2x2080ti_20261015.json`.
- Generated runners for their scripts 03 and 06 (`scripts/g0/make_duan_runners_v3.py`; every edit asserted; diffs in `reports/g0/`). Found by reading: the shipped dtype test `device == "cuda"` would silently load the work model in fp32 on `cuda:1`; script 06 does not parse (anonymization artifact); cells 22+ of script 03 call undefined functions. Stub unit test on Ada: patched metric function identical to the original; snapshots rebuild totals and per-feature r exactly; BOS arms give 256/256/257 positions.
- Independent re-verification by a second reviewer (subagent) re-probed Ada and attacked the plan; I checked its main claims myself (TL un-ties the unembedding, script 06 parse error, restore moving the model to cuda:0, device-index trap in TL, my BOS arm sliding the text windows) and adopted or rejected each (audit doc §9). It corrected my INT12 "near-lossless" assumption (also shown by my numpy emulation: INT16 already changes 12% of bf16 weights), my BOS-arm bug, and caught that the plan lacked a full outcome matrix.

Decisions:
- Venue: Ada `u22`, 2 x RTX 2080 Ti per job, `model_ref`+SAE on cuda:0 and `model_work` on cuda:1 (placement-only edit).
- Closure protocol fixed before running (audit doc §7): gates (run integrity incl. an INT0 restore-only control and FP anchors vs released data; noise floor with a scaling test), then Q1-Q4 map to terminal dispositions. The reference for validating our pipeline is their code as run on Ada, not their published numbers.
- No code in `src/` or the paper was changed; `scripts/g0/` and `reports/g0/` are new.

Blocked: TransformerLens loading the gated `google/gemma-2-2b` was NOT tested because no HF token exists on Ada; the local HF account has access (HTTP 206 with token, 401 without). Needs `hf auth login` on Ada by the user. No token was copied. Next: R0 smoke, then R1-R3, R4, R5.

## 2026-10-16 -- wandb logging for the G0 runs; R0 smoke passed on Ada

- wandb: the G0 runners (their scripts 03/06) had no logging. Added `scripts/g0/g0_wandb.py` (logger + closure-protocol gate metrics + a never-raising GPU-memory helper) and wired it into the generator. Built test-first (each suite run red before its code existed): 18 helper tests with a fake wandb, 5 generator regression tests (25 total, stdlib only), an extended stub test on Ada (patched streaming loop identical to the original; heartbeats and `add_summary` hooks fire), and a live self-test against the real service (run logged to `karma-skz-/anlp-project`, metrics/summary/artifact read back via the API, run deleted). Key: the account behind `~/.netrc` on Ada (`karma-skz`) already worked; no key was handled by the assistant. Project `anlp-project` (created by the user) lives under entity `karma-skz-`.
- Download lesson: first R0 attempt stalled at 4 of 10.5 GB after 35 min because `HF_HUB_DISABLE_XET=1` (my choice) forced plain HTTP (~1.7 MB/s, dropped connections); cancelled. A retry pre-download on the login node then filled `/home2` to 97% (stale `.incomplete` partials plus new ones; the user's training job saw no write errors) and was stopped and cleaned within minutes. Now each job fetches the model to node-local `/scratch` (xet on, retries, deleted at job end); `/share7` and `/data3` are mounted on compute nodes but we have no directories there, so they were not used.
- R0 (job 23355, 2 x RTX 2080 Ti, 3a_test, 20k tokens, test split): completed, exit 0, 24.5 min wall (about 18 of them downloading). TransformerLens loads both models, dtype/device asserts pass; peak GPU memory 8,205 / 8,239 MiB of 11,264; host RSS 21.8 GB; perplexity 3.3 it/s; streaming 1.08 it/s. Results (their pipeline, unmodified except placement): FP16 baseline ppl 414.1; INT8 ppl +2.06%, survived 96.15%, median r 0.9713; INT6 ppl +3.26%, survived 49.82%, median r 0.8994, active features 6,702. Published (500k tokens): ppl 458.5; INT8 +2.62%, 99.09%, 0.9707; INT6 +3.99%, 51.30%, 0.9027, 7,144 active. Relative L2 weight change: INT8 1.09%, INT6 4.28%. This is a smoke at 20k tokens, not the closure-protocol Q1 (which needs 500k), but it shows their regime (baseline perplexity in the hundreds, INT6 survival near 50%) reproduces on our hardware.
- Compute verdict: memory fits with ~3 GiB headroom; time is ~40 min per condition, ~4.8 h per 7-condition arm (the earlier 2.4 h projection from a plain-HF stand-in was too optimistic). Not "barely manageable", but heavy: ~15 GPU-hours for R1-R3. Queue wait was ~0 s for all 6 jobs so far (8-11 idle 2080 Ti nodes); `sbatch --test-only` predictions (start tomorrow evening) did not match reality and were ignored.
- Not yet launched: R1-R3 (needs the user's go-ahead), R4, R5. wandb heartbeats did not fire in R0 (78 windows < 100); the first live heartbeat is to be checked after R1 starts.

## 2026-10-16 (faithful baseline tooling; faithful smoke; handoff) -- "just replicate their code"

Requirement from the user: the baseline must be a replication of Duan's code, not a modification. The v3 runner (BOS arms, INT0, snapshots, resume, wandb hooks) is a diagnostic tool and was therefore separated from the baseline.
- Built (test-first, each suite run red before its code existed) `scripts/g0/make_faithful_runner.py`: only [FIX] runnable export (display->print, drop cells 22+), [PLACE] second-card placement + the dtype-test fix it needs, [MODE] one of their two shipped modes. `tests/test_faithful.py` proves every computation function (streaming metrics, perplexity, hooks, SAE encode, RTN quantizer, target selection, magnitude pruning, pruning search, token building, state clone, outputs) is AST-identical to their cells 1-21, that only `load_gemma_tl_model`, `restore_model_work` and `add_summary` differ, that no experimental marker leaked, and that the diff is < 40 lines. A first run of the test failed because the original script defines `target_tl_weight` twice (Cell 12 with W_gate runs; Cell 22 without W_gate is in the dropped cells and runs after all results are written); the test now compares against the executed cells 1-21. A mutation check (0.9 -> 0.91; qmax) confirmed the AST test detects changes to their math.
- `scripts/g0/baseline_report.py` (+8 tests): compares a replicated Phase-3 summary with their published table using the pre-committed bands (INT7 within 5 pp, INT6 within 10 pp, active features within 10%, perplexity signs), per-feature Spearman at INT6 against their released file; optional post-hoc wandb upload (nothing is logged from inside their code). `scripts/g0/run_baseline.sbatch` runs the faithful runner then the report. Suite: 41 stdlib tests, all passing.
- **Faithful smoke on Ada** (job 23419, `3a_test`, 2 x RTX 2080 Ti, 20.5 min incl. model download): completed, exit 0, report + wandb upload worked (`karma-skz-/anlp-project/runs/8f6f4d3a`). **Bit-identical to R0** (v3 runner, job 23355, other node): max abs difference 0.0 over all summary columns and over the per-feature correlations of all 16,384 features for the baseline, INT8 and INT6 conditions. So the v3 instrumentation does not change their computation and the results are deterministic across nodes of this GPU class. Results (20k tokens, test split; not Q1): baseline ppl 414.07; INT8 +2.06% / survived 96.15% / median r 0.9713; INT6 +3.26% / 49.82% / 0.8994; 6,702 active features. Per-feature agreement with their released INT6 file over 6,271 common active features: Spearman(r) 0.926, Spearman(firing rate) 0.918.
- Documentation updated for the handoff: new `HANDOFF.md` (mission, the no-modification rule, runbook, environment facts, do-not list, file map), `README.md` (status banner; superseded claims replaced by the measured numbers; baseline quick start), `PLAN.md` (Layer 1 faithful baseline, Layer 2 labelled diagnostics), `PROGRESS.md`, `reports/duan_codebase_audit.md` (section 6 rewritten as two layers; Q1 taken from the baseline), `reports/g0/README.md`, `issues.md` pointer, and a SUPERSEDED banner on `writeoff.md`. `paper/` was deliberately not touched (course policy: report prose must be written by the team; `issues.md` A1-A6 lists what is stale).
- Not run: the 500k-token baseline (`3b_full`, ~5.5 h), diagnostics R1-R5. Baseline needs the user's go-ahead.
- Caveat recorded in HANDOFF.md: the faithful smoke exercised only their RTN conditions; their matched-perplexity pruning stage (last stage of `3b_full`) has never run in our two-card TransformerLens setup. RTN results are persisted before it starts, so a failure there loses only the pruning row.

## 2026-10-16 (later) -- baseline restarted on their VENDORED code; generator-based runner removed

The user pointed out (after a side-chat self-review) that the "faithful runner" was not their code: it was a generator that rewrote their file, its test compared only 13 function bodies (not top-level statements), I had never run their unmodified script, and the mutation check was a one-off, not committed. The user asked for the original repo, pruned of what this study does not need, with only reference-style edits. The running baseline job (23439, 1 h 22 min in) was cancelled.
- Re-downloaded their repo: byte-identical to the first download (36 files; content tree hash a84e43dac17e5a9c6b1404e4b502c97d2b184b0168e860035432e4c261e44adc). The zip's own hash differs between downloads (the server stamps download time), so identity is now the content tree hash; the handoff's earlier zip hash was wrong as an identifier.
- Vendored into `third_party/duan2026/` (MIT license kept): retained byte-identical in `pristine/` the LICENSE, README, scripts 03 and 06, and their published Gemma tables and INT6 per-feature file; removed Pythia scripts, the feature-class predictor, `src/`, unrelated results/figures and env files (hashes kept in `MANIFEST.sha256`). The one script we run = their file + a whitelist (`scripts/g0/verify_vendor.py`): delete cells 22+ (call undefined functions, run after all results are written), `RUN_PRUNING = False` (their own flag; also removes the never-run pruning stage), and four device references (WORK_DEVICE, the `device == "cuda"` dtype test that would give float32 for "cuda:1", loading and restoring `model_work` on it). `display()` needs no edit: it works under IPython (installed in the Ada env; exit codes behave).
- Built test-first: `tests/test_vendor.py` was run red (module and files missing), then green. It rebuilds the working copy from the pristine file plus the whitelist and requires byte equality over the WHOLE file (top-level statements included), checks retained files against the recorded upstream hashes and removed files absent, checks the exact file set, and carries a committed mutation suite (13 mutations incl. top-level constants, bit list, SAE id, layer, a deleted statement, an appended statement, re-enabled pruning flag) that must all be rejected. Removed `make_faithful_runner.py` and `test_faithful.py`; repointed `test_generator.py` at the vendored pristine files. Suite: 43 tests, all passing.
- `run_baseline.sbatch` rewritten: verifies `working.sha256` on Ada before running, runs their script under `ipython`, smoke mode uses a temporary copy differing by the single `RUN_MODE` line (printed in the log).
- Smoke of the vendored file (job 23463, 14.5 min): exit 0, no traceback; **bit-identical** to both earlier smokes (max abs diff 0.0, summary and 3 x 16,384 per-feature correlations). wandb `karma-skz-/anlp-project/runs/f722c81a`.
- Still not demonstrated: that the four device edits equal the single-device behaviour of their unmodified script on a large GPU (none reachable); argued from the code and from bit-identical logits across the cards.
- Docs updated: HANDOFF.md (rewritten around the vendored approach), README, PLAN, PROGRESS, audit doc section 6, issues pointer, reports/g0/README, `third_party/duan2026/UPSTREAM.md` (new). Entries above that mention the generator-based "faithful runner" describe a superseded approach.
- Baseline queued: Slurm job **23472** (`base-full`, `run_baseline.sbatch full full`, gnode057), submitted after the vendored smoke passed and the full verification (43 tests, `verify_vendor.py --verify`, retained files re-checked against a fresh download, deployed hash == vendored hash). The partial outputs of the cancelled job 23439 were moved to `baseline/cancelled_23439` on Ada.

## 2026-10-16 (afternoon) -- BASELINE RESULT: their code reproduces their published table (job 23472)

Status check found job 23472 COMPLETED (exit 0, 4 h 08 min 42 s, gnode057, no errors; the launcher verified the vendored script hash `7f1e638a0508c3ca` before running). Results fetched to `reports/g0/baseline_full/` and judged locally with the pre-committed bands, then independently recomputed from the raw per-feature CSVs (not via `baseline_report.py`): **Q1 passed on all four bands.**
- Replicated vs published (500k tokens, WikiText-2 train): baseline ppl 458.4 / 458.5; INT8 +2.52% / +2.62%, survived 99.12 / 99.09, median r 0.9707 / 0.9707; INT7 -5.61% / -5.65%, 81.19 / 81.26, 0.9499 / 0.9499; INT6 +4.02% / +3.99%, 51.15 / 51.30, 0.9027 / 0.9027; INT5 +17.73% / +17.72%, 27.57 / 27.72, 0.7775 / 0.7780; INT4 +46.25% / +46.08%, 12.39 / 12.40, 0.4504 / 0.4512; active features 7,134 / 7,144. Largest survival gap 0.152 pp, largest median-r gap 0.00082. Per-feature Spearman at INT6 over 7,121 common active features: r 0.99965, firing rate 0.99995. Peak GPU 8,183 / 8,009 MiB per card; streaming 1.09 it/s, perplexity 3.27 it/s.
- Interpretation (see HANDOFF section 9 / audit section 7.5): their regime (baseline perplexity 458, INT7 improving perplexity) is real in their pipeline and reproducible on different hardware; per-feature r agrees across hardware at 0.9996, i.e. deterministic in the weight perturbation, not rounding-noise realizations (inference; the INT16/INT12 control is not run). It does not explain why our pipeline gives 98%/94%: diagnostics (Layer 2) not run. Not demonstrated: equivalence of the four device edits to a single-device run of their unmodified script (no >= 16 GB GPU reachable).
- Operational: at ~09:24 Ada lost all outbound internet (W&B, Hugging Face, PyPI time out from the login node; reachable from the laptop). The job's post-hoc wandb upload failed non-fatally; the same report was uploaded from the laptop with a throwaway venv and a temp copy of the outputs: wandb run `baseline-full-23472` (`runs/86f26788`, finished, q1_pass true, 22 artifact files). The upload tag `duan03-faithful` (leftover from the removed generator approach) was renamed to `duan03-vendored` in `baseline_report.py` for future uploads.

## 2026-10-16 (midday) -- audit of why our pipeline disagrees with theirs: static findings, tooling, jobs queued

Request from the user: "we need to audit why our implementation was so off compared to theirs" (ours INT7/INT6 survival 98%/94% vs their 81%/51%, which their own code reproduces on Ada).
- Wrote `reports/audit_our_vs_duan.md`: the 12 differences with how each is known, expected direction and which experiment isolates it; the cells; the reading rules fixed before any result (anchor `T` vs the vendored baseline within 3 pp / 0.01 / 1%; single-factor deltas: >= 5 pp material, 1-5 pp minor; `O_keep` vs our original 98.28/94.31; R2 vs `T_mask`; R2's own 10 pp rule).
- Static findings from the installed library code on Ada (transformers 5.19.0, transformer-lens 3.9.0): HF `integrations/sdpa_attention.py` has no `softcap` (it falls into `**kwargs`), `modeling_gemma2.py:274` passes `softcap=attn_logit_softcapping` (50.0), and only the `eager` path applies it (`modeling_gemma2.py:184-198`); our `src/` loads the model with the default attention implementation (sdpa), so it skips the cap, while TransformerLens applies it (`components/abstract_attention.py:585`, config `attn_scores_soft_cap: 50.0`, `output_logits_soft_cap: 30.0`). The size of the effect on layer-12 residuals/features is not measured yet (the R5 selfcheck does it). Our original runs pinned transformers 5.17.0, not re-checked.
- **Correction**: `issues.md` B1 had said TransformerLens computes RMSNorm in bf16. That was wrong (TL upcasts to fp32 like HF). The sentence was replaced with the corrected statement.
- TL's loss is `log_softmax` on bf16 logits and a bf16 mean (`utilities/lm_utils.py:32`), so their perplexity carries bf16 rounding; R5 records both that and an fp32 loss.
- Built test-first (each suite red before its code): `scripts/g0/r5_lib.py` (windows by their rule, BOS regimes, streaming float64 Pearson, their summary bins, sparse cache of FP features, their quantizer grouping, tested bit-identical against THEIR quantizer extracted verbatim from the pristine script for all 7 module kinds at INT8/7/6/4: ours equals theirs except q/k/v), `scripts/g0/r5_run.py` (cell runner: our HF pipeline on their tokens; selfcheck stage: INT0 restore identity (aborts on failure), sdpa-vs-eager size, our fp32 JumpReLU vs the installed `sae_lens` SAE, bf16-vs-fp32 SAE, token SHA-256), `run_r5.sbatch`, `scripts/g0/r5_compare.py` (the reading rules as code). 31 torch tests (CPU, tiny random Gemma-2; also pass inside the Ada env) + 10 stdlib tests; mutation checks (swap attention, SAE dtype, BOS masking, quantizer, cached-window alignment, FP-pass features, the O_keep spec; thresholds in `r5_compare`) are all caught. Lesson recorded: an earlier mutation run on `r5_compare` was invalid because Python reused a stale `.pyc` (same-size edit within one second); the checks were rerun with `python -B` after deleting `__pycache__`.
- Jobs on Ada (all `research`/`u22`): **R2 23643** (their code + BOS prepended and masked; ladder INT0/16/12/8/7/6; submitted 12:04 IST); **R5smoke 23648** (40 windows, cells T,O, INT8; wandb off) gating **R5a 23649** (T, O, T_sdpa, T_bf16sae) and **R5b 23652** (T_rowquant, T_mask, T_keep, O_keep), INT8/7/6, via `--dependency=afterok`. Code deployed to `/home2/karma/duan-verify/r5code/` (hashes recorded in each job's `code.sha256`).
- **Next prompt: check R2/R5 status and fetch results** (HANDOFF section 11).

## 2026-10-16 (12:40-12:55 IST) -- R5 smoke passed; R5a/R5b running; CPU-cap lesson

- **R5smoke (23648) COMPLETED, exit 0, 10 min 57 s**, so R5a (23649, gnode053, account `research`) started automatically. Smoke selfcheck (4 windows each; `reports/g0/r5_smoke/selfcheck.json`, audit section 6.1): quantize/restore bit-identical; our fp32 JumpReLU identical to `sae_lens` (Jaccard 1.0, max diff 0.0); bf16 vs fp32 SAE Jaccard 0.993, mean relative difference 0.25%; sdpa vs eager moves layer-12 resid_post by 1.98% relative L2 (combined effect of soft-capping and kernel numerics). Smoke cells at 10,240 tokens, INT8 only: `T` 93.73% survival / median r 0.9721 / ppl +3.01%; `O` 95.89% / 0.9760 / +2.53% (not comparable to 500k-token tables).
- **R5b was stuck** (job 23652, `QOSMaxCpuPerUserLimit`): `-c 8` + `--mem=40G` is inflated to 14 CPUs per R5 job, R2 holds 18, and QOS `medium` caps the user at 40 CPUs. Resubmitted as **23660** under `-A nlp` (QOS `normal`, no per-user cap; the parallel-arm route noted as untested earlier), which started at once on gnode058; the `research` copy was cancelled before it ran. R5b writes to `r5/R5b/` as planned.
- R2 (23643): wandb run `168a7bd4` (`g0-R2-23643`) heartbeats verified live (100/1953 windows, per-GPU memory), resolving the open item from R0.
- Test-suite state at this commit boundary: 53 stdlib tests, 31 torch tests, vendor verification clean.
- R5a selfcheck (16 windows, `reports/g0/r5/R5a_selfcheck.json`) reproduces the smoke: restore bit-identical; fp32 SAE == `sae_lens`; bf16 SAE Jaccard 0.9928; sdpa vs eager 1.92% relative L2 at layer 12. R5a windows (1953, 256) = 499,968 tokens. R2's `tokens_2d.pt[:, 1:]` equals R5a's tokens exactly (checked on Ada), so R2 and R5 share identical windows.

## 2026-10-16 (15:00-15:20 IST) -- R5 results (partial): their regime reproduced in HF; BOS and q/k/v grouping are the material factors

- R5a (23649: T, O, T_sdpa, T_bf16sae) completed; R5b (23660) completed T_rowquant, T_mask, T_keep and is running O_keep; R2 (23643) is at INT12 streaming (its control rungs: INT0 +0.00%, INT16 +0.01%, INT12 +0.02% perplexity). Fetched summaries to `reports/g0/r5/` (per-feature CSVs kept on Ada; a first fetch pulled 38 MB of them into the repo directory because of an rsync filter-order mistake (first match wins); they were moved out before committing). One fetch also hit a transient DNS failure on the laptop (`Could not resolve hostname ada-gw1`) and left `T_mask` empty; it was re-fetched and verified by listing the directory, not by the loop's "fetched" echo.
- Judged with `scripts/g0/r5_compare.py` (rules fixed in the audit before the data): anchor `T` PASS; material: BOS (+18.5 / +46 pp at INT7 / INT6) and q/k/v grouping (+7.7 / +3.6); none: sdpa and bf16 SAE; additive (sum +7.81 / +3.51 vs joint +8.16 / +3.51). Table and readings: `reports/audit_our_vs_duan.md` section 6.2.
- Pending: `O_keep` (loop closure against our 98.28 / 94.31), R2 (TL + masked BOS, cross-stack check with `T_mask`).

## 2026-10-16 (~15:40 IST) -- R5 finished; implementation differences matter only without BOS

- R5a (2 h 28 min) and R5b (2 h 47 min) completed, exit 0. `O_keep` (our pipeline, BOS kept, their tokens): INT8/7/6 survival 99.93 / 99.84 / 99.19, median r 0.9974 / 0.9952 / 0.9897, perplexity +0.08 / +0.19 / +1.33%. Versus `T_keep`: -0.04 pp (INT7), +0.83 pp (INT6): class none, so the quantizer-grouping effect seen without BOS is absent with BOS. Versus our original 2,048-token demo (98.28 / 94.31): +1.56 / +4.88 pp (sample size / corpus, not isolated).
- Final R5 table, readings and the 7-point interpretation are in `reports/audit_our_vs_duan.md` section 6.2; generated report `reports/g0/r5/r5_report.md`. R5b's selfcheck equals R5a's (same token SHA-256 `23b44fad...`); the full-job code hashes equal the smoke's.
- R2 (23643) control rungs INT0 / INT16 / INT12 perplexity +0.00 / +0.01 / +0.02%, INT8 +0.18%; its survival rungs INT8/7/6 are still to come (~17:15 IST).
- Fetching lesson: 3 fetch mistakes in this session (rsync include-before-exclude order, "fetched" echo printed after a failed rsync, a transient DNS failure) were all caught by listing the directories; verify fetches by listing files, not by the loop's own output.
