# Does Compression Forget Language?

Language-Specific Feature Survival Under Quantization and Pruning in Multilingual LLMs.

ANLP (Monsoon 2026), Team AlwaysLate.

Full research design: [`anlp_revised.txt`](anlp_revised.txt) (locked proposal, Sections 1-8).
Team plan / task split / timeline: [`PLAN.md`](PLAN.md).
Progress at a glance: [`PROGRESS.md`](PROGRESS.md) (full technical log: [`reports/progress_log.md`](reports/progress_log.md)).

## What this is

We test whether language-specific SAE (sparse autoencoder) features in
multilingual LLMs survive quantization/pruning at different rates than
universal features, and whether survival tracks a language's resource
level, using Gemma-2-2B + the pinned Gemma Scope residual-stream SAE
(layer 12, width 16k) as the primary target.

## Repo layout

```
src/
  sae_models.py   JumpReLU SAE (Gemma Scope) and TopK SAE (EleutherAI/seonglae) loaders.
                  No dependency on sae_lens / EleutherAI's `sae` package -- both pull in
                  volatile extra dependencies; each architecture is ~40 lines of plain torch.
  pipeline.py     HF AutoModelForCausalLM + forward-hook residual capture + SAE encode.
                  Deliberately avoids TransformerLens (Gemma-2 RMSNorm/soft-cap handling
                  there has been a recurring source of numerical mismatches across versions).
  rtn_quant.py    Custom per-channel round-to-nearest fake-quantization (not bitsandbytes/
                  GPTQ -- neither matches the pinned protocol's linear INT quantization).
  survival.py     Firing-rate / Pearson-survival / selectivity-score statistics,
                  matching Duan (2026) and Deng et al. (2025)'s definitions exactly.
scripts/
  run_w0_sanity_check.py   W0 feasibility spike: verifies language-selective SAE
                            features exist at all (Kannada vs. English, FLORES-200 dev).
data/               FLORES-200 (public NLLB release) + HF model/SAE cache. Gitignored.
reports/            Logs and result JSON/CSV from actual runs (small artifacts only).
```

## Setup

```
pip install -r requirements.txt
python scripts/run_w0_sanity_check.py
```

Model weights: we use the **`unsloth/gemma-2-2b`** mirror rather than
`google/gemma-2-2b` directly -- it is bf16 (5.2GB vs. 10.5GB fp32 for the
same weights) and, unlike the official repo, is not gate-restricted, so no
HuggingFace license acceptance/token is needed to run this code. It is the
same pretrained checkpoint, just stored in a smaller dtype.

FLORES-200 data is fetched directly from the official public NLLB release
(`dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz`, ~25MB, all 200+
languages) rather than through HuggingFace's `facebook/flores` mirror,
which is gated and would otherwise require requesting dataset access.

## Compute notes

Developed and smoke-tested without a working GPU on the primary dev
machine (driver/kernel-module mismatch); see `reports/progress_log.md` for
exact CPU-fallback numbers. The full 27-language Phase-1 core run
(Section 4.3 of the proposal) needs a GPU and is designed to run
unmodified once one is available (`ModelConfig` auto-selects CUDA if
`torch.cuda.is_available()`).
