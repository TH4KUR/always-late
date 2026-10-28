# Llama-3.2-1B F5 check, side by side with Gemma-2-2B (Stage 5 of the survival-and-intervention plan)

**Status: post hoc and exploratory.** Frozen before any Llama outcome: [`llama_f5_addendum_2026-10-28.md`](llama_f5_addendum_2026-10-28.md); Gemma analysis it repeats: [`../stage4_gemma_report_2026-10-28.md`](../stage4_gemma_report_2026-10-28.md) (D-113). All numbers are read from the result files by `scripts/g0/survival_effect_llama_report.py`. Model, SAE family, SAE training data, tokenizer, size and layer depth all differ from Gemma, so a difference cannot be attributed to any one of them; no feature identities are compared across the two SAEs; with two systems no population claim is made and overlapping intervals are not equivalence.

## 1. Instrument gate (Llama's own floor)

Median W1 between full precision and the INT16 repeat: 1.000 (the F5 design rule needs at least 0.7); this system's floor (5th percentile of the INT16 cosine): 0.997; reliable features: 610 of 643 (at least 300 needed). Gate: **passed**.

## 2. Result

**Pre-set rule (identical to Gemma: improvement of model B over model A of at least 10 % with the lower end of the language-cluster interval above 0), Llama: no clear predictive value beyond baselines** (0 of 2 cells).

Populations (fixed from the INT16 reference only): primary 610 features (INT16 cosine at least 0.997), noisy-effect stratum 33, S-all 643, S-loss 443.

## 3. Side by side (legacy RTN on both systems, same states, same rule)

| system | state | reliable n | floor | Spearman R1-W1 | partial Spearman given baselines | association | B over A (interval, points) | value |
|---|---|---|---|---|---|---|---|---|
| Gemma-2-2B | INT6 | 1412 | 0.949 | 0.472 [0.423, 0.513] | 0.359 [0.295, 0.412] | repeats | 4.4 % [2.6, 6.2] | not met |
| Llama-3.2-1B | INT6 | 610 | 0.997 | -0.007 [-0.119, 0.094] | 0.037 [-0.030, 0.112] | cannot be resolved | -0.7 % [-2.0, 0.4] | not met |
| Gemma-2-2B | INT4 | 1412 | 0.949 | 0.336 [0.280, 0.384] | 0.265 [0.199, 0.320] | repeats | 3.7 % [2.3, 5.2] | not met |
| Llama-3.2-1B | INT4 | 610 | 0.997 | 0.316 [0.238, 0.393] | 0.252 [0.184, 0.322] | repeats | 4.2 % [1.6, 6.7] | not met |

Labels (defined before the run): the association **repeats** if the partial Spearman interval lies above 0, **cannot be resolved** if it contains 0, **reverses** if it lies below 0; predictive value repeats only if the cell meets the 10 % rule (no Gemma cell did). Each system's estimates are relative to its own noise floor.

## 4. Llama cells in detail

| state | n | MAE null / A / B | C over B | A over null |
|---|---|---|---|---|
| INT6 | 610 | 0.0079 / 0.0081 / 0.0082 | -0.4 % [-1.0, 0.1] | -3.1 % [-13.9, 4.6] |
| INT4 | 610 | 0.1574 / 0.1631 / 0.1562 | 0.7 % [-1.4, 2.4] | -3.6 % [-7.4, -0.0] |

## 5. Sensitivity populations (cannot change the statement)

| population | state | n | Spearman R1-W1 | partial Spearman | B over A |
|---|---|---|---|---|---|
| S_all | INT6 | 643 | 0.093 [-0.024, 0.201] | 0.085 [0.013, 0.168] | -0.5 % [-2.5, 1.4] |
| S_all | INT4 | 643 | 0.366 [0.296, 0.438] | 0.290 [0.231, 0.351] | 4.6 % [2.0, 7.1] |
| S_loss | INT6 | 443 | -0.169 [-0.294, -0.051] | -0.023 [-0.108, 0.075] | -0.7 % [-1.5, 0.1] |
| S_loss | INT4 | 443 | 0.237 [0.161, 0.313] | 0.203 [0.134, 0.284] | 1.5 % [0.7, 2.4] |
| noisy_stratum | INT6 | 33 | 0.431 [0.054, 0.681] | 0.378 [-0.005, 0.713] | -4.5 % [-18.0, 3.6] |
| noisy_stratum | INT4 | 33 | 0.491 [0.262, 0.758] | 0.454 [0.108, 0.695] | 21.1 % [12.7, 31.5] |

## 6. Limits

- Two fixed systems: the result concerns these two model-plus-SAE setups only; Llama's sample (643 features, two roles) is smaller and differently composed than Gemma's (1,487 features, more roles).
- R1 and W1 come from the same perturbed model; held out by language, not by document; the evaluation sentences were examined in earlier studies (post hoc); the same shared RTN method on both systems, no GPTQ on Llama.
- Llama's earlier group-level check was not consistent with Gemma (D-090); nothing here explains why the systems differ.
