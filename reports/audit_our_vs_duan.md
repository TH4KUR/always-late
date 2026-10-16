# Audit: why our pipeline disagrees with Duan et al. (2026)

Status: **written 2026-10-16; R2 and all R5 cells finished the same day (section 6).** Sections 1-3 are established (code reading, unit tests, or the finished baseline job). Section 5 is the reading guide fixed before the experiments. Section 6 holds the results, section 6.4 the status of the closure protocol.

## 1. The disagreement

| | ours (`src/`, HF transformers) | theirs (published table, and our vendored re-run of their code) |
|---|---|---|
| INT8 survival (r > 0.9, active features) | 99.03 % | 99.09 % |
| INT7 survival | 98.28 % | 81.26 % |
| INT6 survival | 94.31 % | 51.30 % |
| INT5 / INT4 | 82.37 % / 60.40 % | 27.72 % / 12.40 % |
| INT8 median r | 0.9999 | 0.971 |
| baseline WikiText-2 perplexity | not measured | 458.5 (their sliding-window + BOS run: 46.4) |

Our numbers are from the 2,048-token WikiText-2 demo (all active features). Their numbers are reproduced by `third_party/duan2026/` on Ada (job 23472, within 0.15 pp at every rung), so **the disagreement is between two pipelines, not a one-off on their side.**

## 2. What is already established

- Their regime is degraded and reproducible: no BOS token, 256-token contiguous windows of WikiText-2 train, baseline perplexity 458, non-monotone perplexity vs bit width (`issues.md` B1, `reports/g0/baseline_full/`).
- Our quantizer equals theirs bit-for-bit for `o_proj`, `gate_proj`, `up_proj`, `down_proj` at INT8/7/6/4 and differs **only** for `q/k/v_proj` (they share one scale per head-dim index across heads, we use one per output row). Evidence: `scripts/g0/tests_torch/test_r5_lib.py`, which runs their own quantizer (extracted verbatim from the pristine script) on a TransformerLens-layout copy.
- Both stacks upcast RMSNorm to fp32. (`issues.md` B1 said TL computes RMSNorm in bf16; that was wrong and is corrected there.)

## 3. Differences between the pipelines

"Verified" says how we know the difference exists; "direction" is the expected sign on our survival numbers, not a result.

| # | Difference | Ours | Theirs | Verified by | Expected direction | Isolated by |
|---|---|---|---|---|---|---|
| D1 | BOS token | present at position 0 of every text (HF tokenizer default, `src/pipeline.py:encode_texts`) | none: `encode(..., add_special_tokens=False)`, tensors are fed to TL directly | their code; baseline job | BOS present -> higher survival | R2 (their stack, BOS masked), `T_mask`, `T_keep`, `O_keep` |
| D2 | attention-logit soft-capping (Gemma-2, cap 50) | **skipped**: `src/` uses HF's default `sdpa` path, and in transformers 5.19 `integrations/sdpa_attention.py` has no `softcap` (it falls into `**kwargs`) while `modeling_gemma2.py:274` passes it; only the `eager` path applies it (`modeling_gemma2.py:184-198`). Our original runs pinned transformers 5.17.0, not re-checked | applied (TL `components/abstract_attention.py:585`, config `attn_scores_soft_cap: 50.0`) | reading installed library code on Ada (static) | unknown; matters where attention logits exceed ~50 (sink tokens, massive activations) | `T` vs `T_sdpa`; size measured by the selfcheck stage (`sdpa_vs_eager`) |
| D3 | SAE precision | `load_sae` casts the SAE to bf16, so pre-activations and the JumpReLU threshold comparison are bf16 (`src/pipeline.py:55-57`) | fp32 SAE (sae_lens), `acts.float()` | code | bf16 flips near-threshold features; sign unclear | `T` vs `T_bf16sae`; selfcheck reports bf16-vs-fp32 active-set overlap |
| D4 | quantizer grouping and arithmetic | per output row for every module, bf16 arithmetic, `clamp(min=1e-8)` | per output row except q/k/v (scale shared across heads), `where(scale == 0, 1, scale)` | unit tests (bit-exact on 4 of 7 module kinds, differs on q/k/v) | theirs is coarser for q/k/v -> lower survival | `T` vs `T_rowquant` |
| D5 | corpus and size | demo: WikiText-2 **test**, 2,048 tokens. Phase 1: FLORES, 128-token decode/re-encode chunks | WikiText-2 **train**, 499,968 tokens, 256-token windows | code | small samples change which features pass the 0.001 firing cut | `O_keep` (our whole pipeline, with BOS, on their tokens) vs our demo numbers |
| D6 | survival population | demo: all active features; Phase 1 headline: language-selective subset | all features with FP firing rate > 0.001 on their tokens | code | selective features are higher-activation, so likely more robust | not isolated by R5 (R5 uses their population on purpose); noted for the paper |
| D7 | weights file | `unsloth/gemma-2-2b` (bf16 mirror) | `google/gemma-2-2b` (fp32) cast to bf16 by TL | not byte-compared | expected none | R5 uses the Google files; **open, low prior** |
| D8 | batch layout | one text per forward, variable length | batch 1 x 256 | code | none expected | R5 uses batch 1 x 256 in every cell |
| D9 | perplexity arithmetic | (not measured) | TL `log_softmax` on bf16 logits, bf16 mean | TL source | changes perplexity, not features | R5 records both TL-style and fp32 perplexity |
| D10 | remaining forward-stack differences (TL vs HF: RoPE, GQA, einsum order, anything not listed) | HF | TL (`from_pretrained_no_processing`) | not enumerated | unknown | the residual `T` vs vendored baseline |
| D11 | noise floor | bf16 weights re-rounded by the quantizer: INT16 already changes ~12 % of stored weights (rel. L2 1.7e-3) | same bf16 storage | numpy emulation (`reports/duan_codebase_audit.md` section 7) | sets the floor under every rung | R2 ladder includes INT16 and INT12; selfcheck INT0 identity |
| D12 | survival-threshold operators | Phase-1 scripts: survived `r >= 0.9`, damaged `r <= 0.5` (`scripts/run_phase1_survival_gpu.py:187`); the firing cut is `>` in both | survived `r > 0.9`, damaged `r < 0.5` | code | negligible for continuous r (a tie at exactly 0.9 is improbable) | not isolated |

## 4. Experiments

All run on Ada, 1 x RTX 2080 Ti per R5 job (R2 uses 2). Our R5 code is `scripts/g0/r5_run.py` + `r5_lib.py` (CPU tests: `scripts/g0/tests_torch`, 31 tests, run in the Ada environment too; the committed mutation checks confirm that swapping attention, SAE dtype, BOS masking, quantizer, cached-window alignment or the FP-pass features is caught).

**R2** (job 23643, their TransformerLens code modified only to prepend BOS to the identical windows and mask position 0 from the statistics; ladder INT0, 16, 12, 8, 7, 6). Answers: how much of the gap is BOS, in their own stack. Output `out/R2/` on Ada.

**R5** (our HF pipeline on their tokens; one bf16 model quantized in place; batch 1 x 256; FP features cached sparsely and compared per window; feature statistics and perplexity share one forward pass; their Pearson/summary arithmetic). Cells flip one factor from the anchor `T`:

| cell | attention | SAE | quantizer | BOS | role |
|---|---|---|---|---|---|
| `T` | eager | fp32 | theirs | none | "their pipeline inside HF"; should land on the vendored baseline if D2-D4 are the only differences |
| `T_sdpa` | **sdpa** | fp32 | theirs | none | D2 |
| `T_bf16sae` | eager | **bf16** | theirs | none | D3 |
| `T_rowquant` | eager | fp32 | **ours** | none | D4 |
| `T_mask` / `T_keep` | eager | fp32 | theirs | **masked / kept** | D1 in HF |
| `O` | sdpa | bf16 | ours | none | our pipeline as is, no BOS |
| `O_keep` | sdpa | bf16 | ours | **kept** | our pipeline as is, with BOS, on their 500k tokens |

Jobs: smoke 23648 (40 windows, cells `T,O`, INT8, wandb off; **completed, exit 0, 10 min 57 s**) -> `R5a` 23649 (`T,O,T_sdpa,T_bf16sae`, account `research`) and `R5b` **23660** (`T_rowquant,T_mask,T_keep,O_keep`, account `nlp`), ladder INT8/7/6. The first R5b submission (23652, `research`) was cancelled: R2 (18 CPUs) + R5a (14) + R5b (14) exceeded QOS `medium`'s 40-CPU per-user cap (`-c 8` with `--mem=40G` is inflated to 14 CPUs), so it sat in `QOSMaxCpuPerUserLimit`; `-A nlp` (QOS `normal`, no per-user cap) started immediately. The first stage of every job is the **selfcheck**: INT0 identity (quantize then restore leaves the FP logits bit-identical; the job aborts if not), sdpa-vs-eager size on layer-12 residuals/logits/loss, our fp32 JumpReLU vs the installed `sae_lens` SAE on real activations, bf16-vs-fp32 SAE agreement, and a SHA-256 of the token windows (R2's `tokens_2d.pt[:, 1:]` must equal R5's `tokens_2d.pt`).

Not covered: D6 (population), D7 (weights mirror), the INT5/INT4 rungs, and any claim about which pipeline is *correct*.

## 5. How the results will be read (fixed before seeing them)

1. **Anchor.** `T` against `reports/g0/baseline_full` at INT7 and INT6: survival within 3 pp, median r within 0.01, perplexity within 1 %. If this holds, D2-D4 plus "no BOS" account for the HF-vs-TL difference and D10 is negligible. If not, there is an unlisted forward-stack difference and single-factor attributions below are only relative to `T`.
2. **Single factors.** Delta = survival(cell) - survival(`T`) at INT7 and INT6. |delta| >= 5 pp at either rung is *material*, 1-5 pp *minor*, < 1 pp *none* (identical re-runs of their code differ by <= 0.15 pp).
3. **Our pipeline on their tokens.** `O` vs `T`: the total implementation effect without BOS. `O_keep` vs `T_keep`: the same with BOS. Compare the sum of single-factor deltas with the `O` - `T` difference; a large mismatch means the factors interact.
4. **Closing the loop with our original numbers.** `O_keep` vs 98.28 % / 94.31 % (D5): if close, corpus and sample size do not matter and the original gap is D1 + implementation factors; if not, corpus/size/population is a factor and needs its own cell.
5. **Cross-stack consistency.** R2 (TL + masked BOS) vs `T_mask` (HF + masked BOS) at INT7 and INT6 within 3 pp; otherwise the TL/HF difference is larger than the listed factors.
6. **R2's own rule.** The BOS effect in their stack is material if INT7 survival differs by >= 10 pp from 81.19 % and the difference exceeds the feature-bootstrap half-widths.

Interpretation limits: survival percentages are counts over ~7,100 features, so differences under ~1 pp are not meaningful even before systematic effects; matching their numbers shows we reproduce a regime, not that the regime is the right instrument.

## 6. Results (R2, R5, R5c and R1 complete)

wandb (project `anlp-project`, entity `karma-skz-`; all runs finished): `g0-R2-23643` (`168a7bd4`), `R5a-23649-{T, O, T_sdpa, T_bf16sae}`, `R5b-23660-{T_rowquant, T_mask, T_keep, O_keep}`; each run logged live heartbeats and per-condition metrics and uploaded the cell's files as an artifact.

### 6.1 Selfcheck of the smoke job (23648; small samples: 4 windows = 1,024 positions for each check, `reports/g0/r5_smoke/selfcheck.json`)

- **Restore integrity**: quantize-then-restore leaves the FP logits bit-identical (max abs difference 0.0). The cell results do not depend on leftover state.
- **Our SAE code vs `sae_lens`** (the SAE their script uses): active set identical (94,326 of 94,326 entries, Jaccard 1.0), maximum value difference 0.0. Our JumpReLU encoder is not a source of the gap; any SAE effect is the dtype (D3).
- **bf16 vs fp32 SAE on the same activations**: Jaccard 0.9930 (about 0.7% of active entries differ), mean relative difference 0.25% on entries active in both, maximum absolute difference 0.43.
- **sdpa vs eager (D2)**: layer-12 resid_post differs by 1.98% in relative L2 (max abs 2.0); logits differ by up to 2.4; mean loss changes by -0.006 (6.764 -> 6.759). This is the combined difference between the two attention paths (soft-capping plus kernel numerics), not the soft-cap alone. Whether it matters for survival is what `T` vs `T_sdpa` measures.
- Token windows: SHA-256 `f7340069...54163f2` for the 40-window smoke subset (the full jobs record their own).
- Smoke cells (40 windows = 10,240 tokens, INT8 only; **not comparable to the 500k-token tables**): `T` INT8 survival 93.73%, median r 0.9721, perplexity +3.01% (their full-run INT8 median r is 0.9707); `O` INT8 survival 95.89%, median r 0.9760, +2.53%. The survival levels at this sample size are depressed (their own vendored smoke at 20k tokens had INT8 survival 96.15%, against 99.09% at 500k).

### 6.1b Selfcheck of the full R5a job (23649; 16 windows = 4,096 positions, `reports/g0/r5/R5a/selfcheck.json`; R5b's selfcheck is identical, same token SHA-256 and numbers)

Same picture with four times the sample: quantize/restore bit-identical; our fp32 SAE identical to `sae_lens` (391,237 of 391,237 active entries, max diff 0.0); bf16 vs fp32 SAE Jaccard 0.9928, mean relative difference 0.25%, max abs 0.49; sdpa vs eager layer-12 resid_post 1.92% relative L2 (max abs 2.25), logits up to 2.55, mean loss 6.713 (eager) vs 6.718 (sdpa). R5a's windows are (1953, 256) = 499,968 tokens (the same count as their published run); SHA-256 `23b44fad30303c6a...`. **Verified on Ada: R2's `tokens_2d.pt[:, 1:]` equals R5a's `tokens_2d.pt` exactly** (R2's first column is BOS id 2 everywhere), so R2 and R5 see identical text windows.

### 6.2 R5 cells (complete; R5c = `O_mask` added in 6.5)

Generated by `scripts/g0/r5_compare.py` from `reports/g0/r5/` (summaries only; the per-feature CSVs stay on Ada under `/home2/karma/duan-verify/r5/`). Full table: `reports/g0/r5/r5_report.md`. R5a (23649) and R5b (23660) completed with exit 0; both ran the identical deployed code (`code.sha256`, equal to the smoke's). 500k tokens (1,953 x 256), INT8/7/6; survival = % of active features with r > 0.9.

| cell | INT8 | INT7 | INT6 | INT7 median r | FP ppl | INT7 ppl % | active |
|---|---|---|---|---|---|---|---|
| published (Duan) | 99.09 | 81.26 | 51.30 | 0.9499 | 458.5 | -5.65 | 7,144 |
| vendored baseline (job 23472) | 99.12 | 81.19 | 51.15 | 0.9499 | 458.4 | -5.61 | 7,134 |
| `T` (anchor: their pipeline inside HF) | 99.17 | 81.41 | 51.30 | 0.9503 | 456.9 | -5.74 | 7,133 |
| `T_sdpa` | 99.27 | 81.44 | 51.22 | 0.9502 | 458.3 | -5.72 | 7,154 |
| `T_bf16sae` | 99.21 | 81.46 | 51.31 | 0.9502 | 456.9 | -5.74 | 7,113 |
| `T_rowquant` | 99.76 | 89.15 | 54.87 | 0.9578 | 456.9 | -2.05 | 7,133 |
| `O` (our pipeline, no BOS) | 99.75 | 89.57 | 54.80 | 0.9580 | 458.3 | -2.07 | 7,131 |
| `T_mask` (BOS masked) | 100.00 | 99.96 | 97.47 | 0.9875 | 16.11 | +0.20 | 9,983 |
| `T_keep` (BOS kept) | 99.90 | 99.88 | 98.36 | 0.9946 | 16.11 | +0.20 | 12,011 |
| `O_keep` (our pipeline, BOS kept) | 99.93 | 99.84 | 99.19 | 0.9952 | 16.11 | +0.19 | 12,012 |
| `O_mask` (our pipeline, BOS masked; job 24452, section 6.5) | 100.00 | 99.99 | 98.79 | 0.9889 | 16.11 | +0.19 | 9,973 |

Readings under the rules of section 5 (not moved after seeing the data):
1. **Anchor passes**: `T` is within 0.22 pp (INT7) and 0.15 pp (INT6) of the vendored baseline, median r within 0.0006, FP perplexity within 0.4%. Their regime, including perplexity improving at INT7 and the 458 baseline perplexity, is reproduced inside our HF stack with the factors set their way, so the single-factor deltas against `T` are meaningful.
2. **Material** (>= 5 pp at INT7 or INT6): BOS present in the windows (`T_mask` +18.55 / +46.17 pp, `T_keep` +18.47 / +47.06 pp) and the quantizer grouping (`T_rowquant` +7.74 / +3.57 pp). **None** (< 1 pp at every rung): sdpa attention without soft-capping (`T_sdpa` +0.03 / -0.08) and the bf16 SAE (`T_bf16sae` +0.05 / +0.02).
3. **Additivity**: the three single-factor deltas sum to +7.81 pp (INT7) and +3.51 pp (INT6), against `O` minus `T` of +8.16 and +3.51 pp, so the implementation factors do not interact.
4. **BOS regime**: perplexity is 16.1 with BOS against 457 without, in both stacks (R2 TL, see 6.3); in the BOS regime perplexity rises monotonically with fewer bits (+0.11, +0.20, +1.00%) instead of improving at INT7. Masking the BOS position from the statistics (`T_mask`) versus keeping it (`T_keep`) changes survival by < 1 pp (-0.08 INT7, +0.89 INT6) but changes the active-feature count (9,983 vs 12,011).
5. **What the data say about the original gap** (ours 98.28 / 94.31, theirs 81.26 / 51.30): the BOS token alone takes their pipeline to 99.9 / 97-98, more than the whole gap; the quantizer grouping accounts for a further ~8 / ~3.5 pp inside the no-BOS regime; attention path and SAE dtype account for nothing measurable.
6. **Implementation differences vanish with BOS**: `O_keep` (our whole pipeline: sdpa, bf16 SAE, per-row quantizer, BOS kept) vs `T_keep` (theirs, BOS kept) differs by -0.04 pp (INT7) and +0.83 pp (INT6), median r 0.9952 vs 0.9946 and 0.9897 vs 0.9882: below 1 pp, class *none*. The q/k/v-grouping effect (+7.7 / +3.6 pp without BOS) is absent when BOS is present, so the BOS and the quantizer grouping interact. With BOS, the implementation factors tested do not matter at this experiment's resolution.
7. **Loop closure**: `O_keep` vs our original 2,048-token demo (98.28 / 94.31) is +1.56 pp (INT7) and +4.88 pp (INT6). Our pipeline on their 500k tokens with BOS gives higher survival than the demo. The likely residual is the demo's small sample (2,048 tokens of WikiText-test) and corpus (D5), which this design does not isolate; the 10,240-token smoke already showed depressed survival (`T` INT8 93.73% vs 99.17% at 500k). The audit's rule had no numeric threshold for "close"; the INT6 difference is just under the 5 pp line used for single factors, so it is a minor-to-moderate sample-size effect, not a verdict.

Not shown by these data: which regime is the right instrument. They show that the published 81% / 51% are a property of feeding windows with no BOS (perplexity 457); with BOS, every configuration tested leaves at least 97% of features above r = 0.9 at INT6.

### 6.3 R2: their TransformerLens code with BOS prepended and masked (job 23643, 5 h 11 min, exit 0; `reports/g0/R2_bos_mask/`)

| rung | active | survival % | degraded % | median r | mean r | perplexity | relative L2 weight change |
|---|---|---|---|---|---|---|---|
| FP baseline | 9,972 | 100.00 | 0.00 | 1.0000 | 1.0000 | 16.106 | |
| INT0 (restore only) | 9,972 | 100.00 | 0.00 | 1.0000 | 1.0000 | 16.106 | 0 |
| INT16 | 9,972 | 100.00 | 0.00 | 0.9969 | 0.9961 | 16.107 | 0.00133 |
| INT12 | 9,972 | 100.00 | 0.00 | 0.9968 | 0.9961 | 16.108 | 0.00140 |
| INT8 | 9,972 | 100.00 | 0.00 | 0.9932 | 0.9916 | 16.134 | 0.01088 |
| INT7 | 9,972 | 99.97 | 0.03 | 0.9874 | 0.9844 | 16.141 | 0.02123 |
| INT6 | 9,972 | 97.52 | 2.48 | 0.9730 | 0.9669 | 16.272 | 0.04277 |

- **Cross-stack check passes**: R2 (TL) vs `T_mask` (HF), survival +0.01 pp at INT7 and +0.06 pp at INT6 (limit 3 pp); median r 0.9874 / 0.9730 vs 0.9875 / 0.9731; perplexity 16.106 / 16.134 / 16.141 / 16.272 vs 16.110 / 16.128 / 16.142 / 16.271. The two stacks agree in the BOS regime to a tenth of a percentage point.
- **R2's own BOS rule (Q3)**: INT7 survival moves +18.78 pp from the no-BOS baseline (81.19 -> 99.97); material at the 10 pp threshold.
- **Gate 0 (run integrity)**: INT0 gives r = 1 for every active feature and perplexity identical to the baseline.
- **Noise floor in the BOS regime**: median r is already 0.9969 at INT16 and 0.9968 at INT12 (the bf16 re-rounding floor, weight change 1.3e-3), so INT8's 0.9932 is only 0.004 below the floor. The scaling test (slope of log(1 - median r) against log relative L2 weight change, `g0_wandb.compute_gate_metrics`) gives **0.592, inside [0.5, 1.5]**: with BOS the decorrelation follows the weight perturbation (signal-limited). This is Gate 1 evaluated on R2, **not** on the arm the protocol names (R1, no BOS).

### 6.4 Where the closure protocol (audit `duan_codebase_audit.md` section 7) stands (**final: D-NOISE, see 6.7**)

The protocol fixed its dispositions before the experiments and cannot be assigned yet, honestly:
- **Gate 0**: passes (R2 INT0; baseline anchors within bands).
- **Gate 1** is defined on R1 (their pipeline, no BOS, ladder incl. INT16/INT12). R1 was not run; the vendored baseline has no INT16/INT12 rungs. R2 passes Gate 1 but R2 is the BOS regime. Whether their no-BOS regime is signal-limited or noise-amplifying is **not measured**.
- **Q4** needs our pipeline vs their code in all three regimes (none, mask, keep). Available: none: `T` vs the baseline passes (0.22 / 0.15 pp), but our own pipeline `O` differs from the baseline by +8.4 / +3.7 pp and matches only with their q/k/v grouping toggled; mask: `T_mask` vs R2 passes, but our pipeline in the mask regime (`O_mask`) was not run; keep: `O_keep` vs `T_keep` (+/- 0.83 pp) but their TL keep arm (R3) was not run.
- So the closest literal reading is **D-EQUIV-UP-TO-TOGGLE for the no-BOS regime** (quantizer grouping), and **incomplete** for mask and keep. Minimal runs that would complete it: (a) Gate 1 in their regime: `T` at INT16/INT12 in HF (about 45 min, 1 GPU) or R1 ladder 0/16/12 in their TL code (about 2 h, 2 GPUs); (b) `O_mask` (about 45 min); (c) R3 in TL (about 5 h, 2 GPUs) or accept `T_keep` as its HF stand-in, which departs from the protocol text. Whether to run them is a decision for the user/team.

**Update 2026-10-17 (R5c finished; see 6.5):** Q4 for the **mask** regime now passes. Still open for a protocol disposition: Gate 1 in the no-BOS regime (R1, job 24451, running) and Q4 for the keep regime (needs R3). Reading of the available Q4 evidence per regime: *none* (their regime): our pipeline `O` differs from the vendored baseline by +8.4 / +3.7 pp and matches only with their q/k/v grouping toggled (`T`); *mask*: matches (6.5); *keep*: `O_keep` vs `T_keep` +/-0.8 pp, but their TransformerLens keep arm (R3) was not run.

### 6.5 R5c: our pipeline with BOS masked (`O_mask`, job 24452, 59 min, exit 0, `reports/g0/r5/R5c/`) against R2 and `T_mask`

| rung | `O_mask` surv % / median r / ppl | `T_mask` (HF, their settings) | R2 (their TL code) | active |
|---|---|---|---|---|
| FP | 100 / 1.0000 / 16.109 | 100 / 1.0000 / 16.110 | 100 / 1.0000 / 16.106 | 9,973 / 9,983 / 9,972 |
| INT8 | 100.00 / 0.9939 / 16.121 | 100.00 / 0.9932 / 16.128 | 100.00 / 0.9932 / 16.134 | |
| INT7 | 99.99 / 0.9889 / 16.139 | 99.96 / 0.9875 / 16.142 | 99.97 / 0.9874 / 16.141 | |
| INT6 | 98.79 / 0.9766 / 16.323 | 97.47 / 0.9731 / 16.271 | 97.52 / 0.9730 / 16.272 | |

Q4 conditions for the mask regime, `O_mask` vs R2 (audit `duan_codebase_audit.md` section 7.2): |delta survival| <= 5 pp at INT7 and INT6: **+0.02 / +1.26 pp, pass**; Spearman of per-feature r at INT6 over the 9,964 common active features >= 0.95: **0.9953, pass**; FP-level Spearman of per-feature firing rate and max activation >= 0.99: **0.99994 / 0.99986, pass**. Same-direction, small gap at INT6 (A-013). Selfcheck identical to R5a/R5b (token SHA-256 `23b44fad...`, restore bit-identical). Code: legacy harness `r5_run.py` 2a1c8dc5 (recorded in `r5/R5c/code.sha256`).

### 6.6 Primary-source check of Duan et al. (2026) (V-3, read 2026-10-17: arXiv:2606.03002v2, 29 pages, text-searched)

- The paper's setup matches the code we vendored: Gemma Scope layer-12 16,384-latent SAE; active if FP16 firing rate > 0.001; survival > 0.9, damage < 0.5; WikiText-2-raw non-empty lines concatenated; chunked blocks of 256 tokens, 500k tokens; sliding-window check W=2048, S=512 on a HuggingFace bf16 Gemma (182 tensors quantized); A100 hardware.
- **BOS is never mentioned** (no match for BOS, `<bos>`, special token, prepend, attention sink, add_special in the full text). The paper explains the high chunked perplexity by context resets (section 3.5, Appendix D). Our measurement (A-014): the same 256-token chunks give perplexity 16.1 with one BOS and 457 without, so context resets are not the cause.
- **Quantizer**: Eq. 2 and Appendix L state one RTN scale per output channel for every targeted tensor; their TransformerLens code pools the q/k/v scale across heads (A-015), worth +7.7 / +3.6 pp (INT7 / INT6) on survival without BOS.
- Table 3 and Appendix D disclose that the sliding-window rows reuse the chunked survival (A-007); the abstract's phrasing can be read as implying survival under sliding windows.
- Consistent with our reproduction: Table H.7 (INT8..INT4 survival 99.09 / 81.26 / 51.30 / 27.72 / 12.40, 7,144 active), Table 3 (chunked perplexity 458.53, sliding 46.39).

### 6.7 R1: their code, no BOS, ladder INT0 / INT16 / INT12 (job 24451, 3 h 06 min, exit 0, `reports/g0/R1_nobos_floor/`): Gate 1 in their regime

| rung | active | survival % | median r | 1 - median r | perplexity | relative L2 weight change | BOS-masked 1 - median r (R2) |
|---|---|---|---|---|---|---|---|
| FP baseline | 7,134 | 100.00 | 1.0000 | 0 | 458.401 | | 0 |
| INT0 (restore only) | 7,134 | 100.00 | 1.0000 | 0 | 458.401 | 0 | 0 |
| INT16 | 7,134 | 100.00 | 0.98897 | 0.01103 | 458.665 | 0.001334 | 0.00315 |
| INT12 | 7,134 | 100.00 | 0.98895 | 0.01105 | 458.496 | 0.001397 | 0.00317 |
| INT8 (baseline_full) | 7,134 | 99.12 | 0.9707 | 0.02932 | | 0.010881 | 0.00677 |
| INT7 (baseline_full) | | 81.19 | 0.9499 | 0.05007 | | 0.021235 | 0.01258 |
| INT6 (baseline_full) | | 51.15 | 0.9027 | 0.09727 | | 0.042769 | 0.02700 |

- **Gate 0 passes**: INT0 gives r = 1 for every active feature and a perplexity identical to the baseline; R1's FP perplexity equals the vendored baseline's in every printed digit (A-017).
- **Gate 1 (protocol section 7.2): (a) median r at INT16 and INT12 >= 0.99: 0.98897 and 0.98895, fails by 0.001; (b) slope of log(1 - median r) against log(relative L2) over INT12..INT6: 0.613, inside [0.5, 1.5], passes.** Computed with `g0_wandb.compute_gate_metrics` from R1's rungs, the baseline's INT8/7/6 (identical code and tokens, bit-identical smokes) and the weight-perturbation JSONs (R1's and R2's are identical). Verdict: **`noise_dominated`**. In the BOS-masked regime the same function gives `signal_limited` (R2: 0.99685 / 0.99683, slope 0.592).
- At the same weight perturbation the no-BOS regime decorrelates 3.5x (INT16, INT12) to 4.3x (INT8) more than the BOS-masked one. The bf16 floor (INT16) is 38% / 22% / 11% of the no-BOS 1 - r at INT8 / 7 / 6; survival (r > 0.9) at INT16 and INT12 is 100%.
- **Mean L0 per token** (sum of FP firing rates, A-018): 95.06 without BOS, 82.4-82.6 with BOS masked, against the SAE's nominal 82.

**Disposition under the pre-committed table (section 7.3 of `duan_codebase_audit.md`): Gate 0 pass, Gate 1 noise-dominated, Q4 any -> D-NOISE (decision D-009).** The protocol's wording of what may be said: their pipeline's feature statistics are noise-dominated even for near-lossless perturbations, so its survival numbers are not a usable reference; cite Duan as prior work only; our validity rests on section 7.4. Recorded qualifiers that do not change the disposition: the margin is 0.001 and the thresholds were judgement calls; the verdict holds for their regime (no BOS) and the same gate passes with BOS masked; 100% survival at INT16 / INT12 means the INT7 / INT6 survival percentages are not artifacts of the r > 0.9 cut at the floor. The protocol is terminal: no further verification of Duan's work is planned (their script 06 stays optional, D-005).

**Section 7.4 checklist (validity of the instrument we build on, independent of Duan), status 2026-10-17:**

| item | status |
|---|---|
| F1 own WikiText-2 perplexity with BOS under default (sdpa) and eager attention | **done**: HF eager (`T_mask`) 16.110, HF sdpa (`O_mask`) 16.109 (499,968 tokens); TL (R2) 16.106 |
| F2 HF-vs-TL feature agreement on identical tokens | **done** for the mask regime: `O_mask` vs R2 Spearman of firing rate 0.99994, max activation 0.99986, r at INT6 0.9953 over 9,964 common active features |
| F3 mean L0 near 82 and explained variance on text | **L0 done**: 82.39 (R2, `O_mask`); explained variance **not computed** (needs the SAE reconstructions) |
| F4 FP-vs-FP noise floor under another batch layout, INT16/INT12 scaling test | scaling test **done** in TL for both regimes (R2 slope 0.592, R1 0.613); the different-batch-layout null and the scaling test for the HF stack **not run** |
| F5 log versions and repeat under the paper's pinned versions or document | **documented** (Ada: transformers 5.19.0, torch 2.14.1; paper pinned 5.17.0 / 2.13.0, not repeated) |
| F6 FLORES content validity | out of scope here (`issues.md` C1) |
