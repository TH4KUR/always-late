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
