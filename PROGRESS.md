# Progress at a Glance

Last updated: 2026-09-29. Full technical log with real numbers/timestamps: `reports/progress_log.md` — this file is just the skimmable summary; that one is the evidence.

## Blocked right now

**GPU driver.** Dev machine has an RTX 4060 8GB, but its kernel module isn't loaded for the currently-running kernel (a routine kernel update left it behind — nothing malicious, just needs the matching module package installed). Fix identified and in progress:
```
sudo apt update
sudo apt install linux-modules-nvidia-580-open-generic-hwe-24.04
sudo modprobe nvidia
nvidia-smi
```
Until this lands, heavier experiments run CPU-only, which is slow and has already hit real system-memory contention (see below).

## Done

- [x] Private repo live: https://github.com/TH4KUR/always-late
- [x] SAE loaders for JumpReLU (Gemma Scope) and TopK (EleutherAI/seonglae, stretch) — no `sae_lens`/`transformer_lens` dependency, verified against real downloaded weight files (`src/sae_models.py`)
- [x] Model + residual-stream hooking pipeline, plain HF `transformers` (`src/pipeline.py`) — verified against real Gemma-2-2B module names
- [x] Custom RTN quantization matching the pinned protocol (`src/rtn_quant.py`) — verified target modules (182 = 26 layers x 7 projections, no layernorm/embedding leakage)
- [x] Survival/firing-rate/selectivity-score statistics (`src/survival.py`)
- [x] **W0 sanity check: PASSED.** Real run on Gemma-2-2B + Gemma Scope (layer 12), FLORES-200 Kannada vs. English: 1,035 Kannada-selective and 1,918 English-selective features found among 16,384 total — confirms language-selective structure exists before compression even enters the picture. (`reports/w0_sanity_check_result.json`)
- [x] Solved along the way: disk space (used the `unsloth/gemma-2-2b` bf16 mirror instead of the gated fp32 official repo — 5.2GB vs 10.5GB, and not gated), FLORES-200 access (pulled directly from the public NLLB archive instead of HF's gated mirror), exact dependency version pinning.

## Blocked / in progress

- [ ] **Quantization-survival demo (INT8-INT4 sweep).** Killed 3x by the kernel OOM killer — root-caused to system-wide memory contention (other apps + the 5.2GB model both in system RAM at once, swap fully exhausted), not a code bug. Needs the GPU fix (moves weights to VRAM) or freed RAM to complete.
- [ ] Full 27-language Phase-1 core panel — not started, needs GPU for realistic runtime.

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
