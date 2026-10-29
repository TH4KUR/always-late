# Gemma-2-9B Functional Validation Verdict (D-089)

Study: confirmation of the Gemma-2-2B functional-validation results on Gemma-2-9B.

Model: `google/gemma-2-9b`, SAE: Gemma Scope JumpReLU `gemma-scope-9b-pt-res-canonical` layer 20 (width 16k, canonical), hook `blocks.20.hook_resid_post`, d_model 3584, d_sae 16384.

300 eval sentences per language, 27 languages, 6 precision states (FP, INT8, INT7, INT6, INT5, INT4), 3 matched-control draws, K64 group, Study G ratios from D-071 (not recalibrated for 9B).

## Predictions

| prediction | criterion | result | met? |
|---|---|---|---|
| P1 | At least 24/27 languages show positive zero-ablation E(K64) at FP, 95% CI above zero | 27/27 positive | **met** |
| P2a | Macro retention at INT8 and INT6 within [0.85, 1.15] | INT8 = 1.003, INT6 = 1.006 | **met** |
| P2b | Macro retention at INT4 at least 0.30 above INT6, INT4 has the largest retention | INT4 = 2.479, INT6 = 1.006, diff = 1.473; INT4 is largest | **met** |
| P3a | Dose-response ordering at FP for at least 20/27 languages | 27/27 strictly ordered | **met** |
| P3b | At least 20/27 languages with median selectivity ratio above 1.3 | 27/27 above 1.3 | **met** |

## Verdict

**Consistent** (all predictions met).

The Gemma-2-9B results replicate the Gemma-2-2B findings:

1. Language-selective SAE features produce a clear, measurable zero-ablation effect across all 27 languages (P1).
2. The effect is stable under moderate quantization — INT8 and INT6 retention is essentially 1.0 (P2a).
3. Aggressive quantization (INT4) amplifies the effect to 2.48x retention (P2b), consistent with the 2B observation that quantization damage concentrates on language-selective circuits.
4. The dose-response relationship is monotonic: milder ablation produces less harm, with perfect 27/27 agreement (P3a).
5. Features are strongly language-selective, with all 27 languages exceeding the 1.3 selectivity threshold (P3b).

## Study G Hypotheses (detailed)

| hypothesis | verdict |
|---|---|
| H-G1 dose, down | **met** (27/27 strictly ordered, median ordered) |
| H-G2 down is specific | **met** (27/27 lower bound above 0) |
| H-G3 dose, up | **met** (27 increasing, 23 net positive) |
| H-G4 precision, down | **met** (25/27 tau > 0, median not falling) |
| H-G5a generic repair | **not met** (control harm intervals include 0) |
| H-G5b group-specific repair | **not met** (8/27, below 14 threshold) |
| H-G6 consistency | **met** (27/27 in all states, rank correlation 0.904) |

H-G5 (repair by turning up features at INT4) is not met, consistent with the 2B findings: the fine-up ratios (1.1, 1.2) do not reliably undo quantization damage. The effect is too small and the controls show similar patterns, leaving no net benefit specific to the language group.

## Retention ladder

| state | macro retention (E / E_FP) |
|---|---|
| FP | 1.000 |
| INT8 | 1.003 |
| INT7 | 1.047 |
| INT6 | 1.006 |
| INT5 | 1.205 |
| INT4 | 2.479 |

## Run details

- GPU: NVIDIA RTX PRO 6000 (96 GB), peak 23.6 GB
- Throughput: ~10,310 tokens/sec, ~29 min per state
- 999 configurations per state, 1130 items per state
- All gates passed: split_equals_full, scale_one_identity, donor_self_identity, sham_identity
- Selective features: 5,218 of 16,384 (31.8%)
- Freeze commit: `d27a6a0` (before any numbers)
- Artifacts: `karma-skz/always-late-artifacts` dataset repo, folder `gemma9b/runs/full`
- wandb entity: `karma-skz-`

## Files

- `freeze.json` — pre-registered predictions and evaluation parameters
- `verdict.md` — this file
- Analysis outputs on HF Hub: `gemma9b/runs/full/analysis/` (report.md, curves.csv, reliability.csv, alpha_star.csv, analysis.json)
