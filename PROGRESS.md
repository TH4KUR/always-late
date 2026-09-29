# Progress at a Glance

Last updated: 2026-09-29 (GPU fixed). Full technical log with real numbers/timestamps: `reports/progress_log.md` — this file is just the skimmable summary; that one is the evidence.

## Resolved

**GPU driver — fixed.** RTX 4060 Laptop (8GB VRAM) now working (`nvidia-smi` clean, `torch.cuda.is_available()` True). The first fix attempt (`--reinstall` the driver metapackage) wasn't enough since it didn't force the kernel-module dependency to upgrade; the real fix was installing the already-available newer candidate for `linux-modules-nvidia-580-open-generic-hwe-24.04` directly. Full diagnosis in `reports/progress_log.md`.

## Done

- [x] Private repo live: https://github.com/TH4KUR/always-late
- [x] SAE loaders for JumpReLU (Gemma Scope) and TopK (EleutherAI/seonglae, stretch) — no `sae_lens`/`transformer_lens` dependency, verified against real downloaded weight files (`src/sae_models.py`)
- [x] Model + residual-stream hooking pipeline, plain HF `transformers` (`src/pipeline.py`) — verified against real Gemma-2-2B module names
- [x] Custom RTN quantization matching the pinned protocol (`src/rtn_quant.py`) — verified target modules (182 = 26 layers x 7 projections, no layernorm/embedding leakage)
- [x] Survival/firing-rate/selectivity-score statistics (`src/survival.py`)
- [x] **W0 sanity check: PASSED.** Real run on Gemma-2-2B + Gemma Scope (layer 12), FLORES-200 Kannada vs. English: 1,035 Kannada-selective and 1,918 English-selective features found among 16,384 total — confirms language-selective structure exists before compression even enters the picture. (`reports/w0_sanity_check_result.json`)
- [x] Solved along the way: disk space (used the `unsloth/gemma-2-2b` bf16 mirror instead of the gated fp32 official repo — 5.2GB vs 10.5GB, and not gated), FLORES-200 access (pulled directly from the public NLLB archive instead of HF's gated mirror), exact dependency version pinning.

- [x] **Quantization-survival demo (INT8-INT4 sweep): DONE on GPU.** Real results, Gemma-2-2B layer 12, RTN, WikiText-2 (3,000 tokens): INT8 survived=99.03%, INT7=98.28%, INT6=94.31%, INT5=82.37%, INT4=60.40% (damage rising from 0% to 5.29%). Monotonic trend matches Duan (2026)'s qualitative direction; exact numbers differ from their pinned bands since this uses a small sample, not their full corpus/budget (see caveat in `reports/progress_log.md`). Along the way, found and fixed a real memory-doubling bug in the RTN quantize/restore context manager (cloned weights were landing on GPU instead of CPU).

## In progress

- [ ] Full 27-language Phase-1 core panel — not started; GPU now available so realistic to attempt if time allows, but still explicitly staged/optional for this submission (see PLAN.md).

## Not started (team work, no code dependency — start anytime)

- [ ] Literature review expansion
- [ ] Refined problem statement
- [ ] "Concrete timeline for the rest of the project" section
- [ ] "Progress made" write-up section (source material for this is ready — see above)
- [ ] Report assembly / ACL formatting / page-limit pass

## Deliberately not attempted this week (per the proposal's own staging, see PLAN.md)

- Duan (2026)'s exact-corpus replication gate (G0)
- Llama-3.2-1B cross-model extension
- Pruning contrast (magnitude/Wanda)
- Phase-2 causal ablation check
