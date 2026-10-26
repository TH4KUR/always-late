# Gemma-2-9B confirmation test (replaces Llama-3.2-1B as the second model), plan

Date: 2026-10-26 08:15 IST. Author: Claude, at the user's request of the same day: "find a suitable model for an alternate confirmation test that we can replace the current llama model with, such that the model also has a prefitted SAE like gemma2-2b and uses the same architecture. i want the SAE to be same as the one for gemma so that any results we get are at least reliable." Decision record: D-089.

## 1. Why Llama is replaced

The Llama-3.2-1B check (D-087) changes four things at once: model family, SAE architecture (TopK k=48 vs JumpReLU), SAE training recipe and data, and relative depth (block 12 of 16 vs 12 of 26). A disagreement with Gemma-2-2B cannot be attributed to any one of them, so it is uninterpretable. The interim symptom: 91 % of the Llama SAE's supported features pass our selectivity filter, so the filter does not mean the same thing there. The running Llama job is NOT cancelled. It finishes, and its comparison is written up as an exploratory cross-architecture check, not as the replication.

## 2. Choice: Gemma-2-9B with Gemma Scope (verified 2026-10-26 08:10 on Ada)

| Requirement | Gemma-2-2B (main) | Gemma-2-9B (confirmation) |
|---|---|---|
| Architecture | Gemma 2 | Gemma 2 (same code path in TransformerLens; 42 layers, d_model 3584) |
| SAE | Gemma Scope, JumpReLU, `gemma-scope-2b-pt-res-canonical`, `layer_12/width_16k/canonical` | Gemma Scope, JumpReLU, `gemma-scope-9b-pt-res-canonical` (repo `google/gemma-scope-9b-pt-res`), **`layer_20/width_16k/canonical`** (in the installed sae_lens registry on Ada) |
| Relative depth | block 12 of 26: (12+1)/26 = 0.500; 12/25 = 0.480 | block 20 of 42: (20+1)/42 = 0.500; 20/41 = 0.488 (layer 19 would give 0.476 / 0.463) |
| Access | granted | `google/gemma-2-9b` config.json downloads with our token on Ada (42 layers, hidden 3584) |

Rejected: Gemma 3 with Gemma Scope 2 (same SAE method, but a different architecture generation: local/global attention pattern, QK-norm); `gemma-2-2b-it` (the same pretrained weights, not an independent model); Gemma-2-27B (only a few Gemma Scope layers, too large for the cluster). The differences that remain go with every result: model scale (2.6B vs 9.2B) and the dictionary-to-width ratio (16k over 2304 = 7.1 vs 16k over 3584 = 4.6). Everything else is held fixed: tokenizer, SAE architecture, training recipe and data, canonical L0 selection, relative depth, quantizer, corpus, languages, sentences, controls, preset.

## 3. Scope: the user's sample-size floor (D-075) applies in full

27 languages x 6 states (FP, INT8, INT7, INT6, INT5, INT4; the Llama check had dropped INT7) x 300 sentences x 3 matched-control draws, Study G MAIN preset, tagging on dev with the equal budget, the same manifest rules (K64, and K16 only if the control pool forces it, which must be reported). Predictions: P1 to P3b and the verdict wording of the Llama plan (`2026-10-26-llama-consistency-check.md` sections 3 and 7), copied unchanged into this study's freeze file BEFORE any 9B number exists. The `req(n)` thresholds and Gemma reference values stay as they are. Its own robustness module R1 to R5 is reused unchanged.

## 4. Compute (to be measured by the smoke job, not trusted)

9.24B parameters in bf16 = 18.5 GB > one 11 GB RTX 2080 Ti. Run with TransformerLens `n_devices=2` (or 3 if the smoke run runs out of memory; the 256k vocabulary makes logits the peak). The QOS `medium` allows 4 GPUs per user and 4 days of wall time. Estimate: Gemma-2-2B about 3 GPU hours per state. 9B is about 3.5x the compute per token, so about 10 to 11 h of wall time per state per job, and about 130 GPU hours for 6 states (two 2-GPU jobs in parallel: about 32 h elapsed; one 3-GPU job: about 63 h). CPU RAM for the FP weight copy used to restore after RTN: request at least 80G. Every state is its own resumable job (D-077: all queued at once). The cron entries expire about 2026-10-29: the cron must recreate them if they lapse (rule 7 of cron_prompt.md).

## 5. Tasks (tests first, as in the Llama build)

- [ ] C1 Adapter: generalise the Llama namespace builder (`phase1_llama_run.py`) into a model-config switch or add `phase1_gemma9b_run.py` with model `google/gemma-2-9b`, SAE release `gemma-scope-9b-pt-res-canonical`, id `layer_20/width_16k/canonical`, `LAYER = 20`, `n_devices`. The Gemma SAE loads through the existing `load_sae_compat` (no new loader). The frozen runners stay unedited. Tests with fakes. Check that the hook is `blocks.20.hook_resid_post` and that the split forward equals the full forward on multiple devices.
- [ ] C2 Smoke (<= 30 min, 3 languages, FP and INT4): GPU line, memory peak, throughput, the sham exact, quantizer coverage of all Gemma-2 weight tensors (same list as 2B). Decides 2 vs 3 GPUs. It is an instrument gate, not a look at the predictions.
- [ ] C3 Freeze file `reports/functional_validation/gemma9b/freeze.json`: model, SAE, layer, sentence ids, preset, predictions P1-P3b, `req(n)`, and the sha256 of the predictions text, committed before C4.
- [ ] C4 Tagging, then the manifest (CPU), then 6 state jobs, all queued at once; `--exclude=gnode068,gnode069`.
- [ ] C5 Validation per job (sacct, GPU line, no CUDA warning, gates, counts, checksums), then the comparison (`phase1_llama_compare.py`, generalised by name only) and robustness (`phase1_llama_robust.py`), D-entry, README, RESULTS_SUMMARY, handoff.
- [ ] C6 Second model of the intervention-transfer study (D-088): Gemma-2-9B replaces Llama-3.2-1B there too (an amendment recorded in that study's own STATUS before its freeze).

## 6. What a result means

Agreement: the Gemma-2-2B picture holds at another scale within the same architecture and the same SAE family. That is a scale replication, not an architecture-general claim. Disagreement: attributable to scale or to the width ratio, not to the SAE method. That is a real finding, reported as such.
