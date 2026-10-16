# Audit: why our pipeline disagrees with Duan et al. (2026)

Status: **written 2026-10-16, experiments queued, no result below is from R2/R5 yet.** Sections 1-3 are established (code reading, unit tests, or the finished baseline job). Section 5 is a pre-committed reading guide for the experiments. Section 6 is filled in when the jobs finish.

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

Jobs: smoke 23648 (40 windows, cells `T,O`, INT8, wandb off) -> `R5a` 23649 (`T,O,T_sdpa,T_bf16sae`) and `R5b` 23652 (`T_rowquant,T_mask,T_keep,O_keep`), both `afterok` on the smoke, ladder INT8/7/6. The first stage of every job is the **selfcheck**: INT0 identity (quantize then restore leaves the FP logits bit-identical; the job aborts if not), sdpa-vs-eager size on layer-12 residuals/logits/loss, our fp32 JumpReLU vs the installed `sae_lens` SAE on real activations, bf16-vs-fp32 SAE agreement, and a SHA-256 of the token windows (R2's `tokens_2d.pt[:, 1:]` must equal R5's `tokens_2d.pt`).

Not covered: D6 (population), D7 (weights mirror), the INT5/INT4 rungs, and any claim about which pipeline is *correct*.

## 5. How the results will be read (fixed before seeing them)

1. **Anchor.** `T` against `reports/g0/baseline_full` at INT7 and INT6: survival within 3 pp, median r within 0.01, perplexity within 1 %. If this holds, D2-D4 plus "no BOS" account for the HF-vs-TL difference and D10 is negligible. If not, there is an unlisted forward-stack difference and single-factor attributions below are only relative to `T`.
2. **Single factors.** Delta = survival(cell) - survival(`T`) at INT7 and INT6. |delta| >= 5 pp at either rung is *material*, 1-5 pp *minor*, < 1 pp *none* (identical re-runs of their code differ by <= 0.15 pp).
3. **Our pipeline on their tokens.** `O` vs `T`: the total implementation effect without BOS. `O_keep` vs `T_keep`: the same with BOS. Compare the sum of single-factor deltas with the `O` - `T` difference; a large mismatch means the factors interact.
4. **Closing the loop with our original numbers.** `O_keep` vs 98.28 % / 94.31 % (D5): if close, corpus and sample size do not matter and the original gap is D1 + implementation factors; if not, corpus/size/population is a factor and needs its own cell.
5. **Cross-stack consistency.** R2 (TL + masked BOS) vs `T_mask` (HF + masked BOS) at INT7 and INT6 within 3 pp; otherwise the TL/HF difference is larger than the listed factors.
6. **R2's own rule.** The BOS effect in their stack is material if INT7 survival differs by >= 10 pp from 81.19 % and the difference exceeds the feature-bootstrap half-widths.

Interpretation limits: survival percentages are counts over ~7,100 features, so differences under ~1 pp are not meaningful even before systematic effects; matching their numbers shows we reproduce a regime, not that the regime is the right instrument.

## 6. Results

*(to be filled in when R2 and R5 finish; see `HANDOFF.md` "Next prompt")*
