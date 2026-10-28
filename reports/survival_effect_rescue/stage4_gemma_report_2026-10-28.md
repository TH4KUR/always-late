# Survival versus effect similarity on Gemma-2-2B (Stage 4 report of the survival-and-intervention plan)

**Status: post hoc and exploratory.** The committed F5 tables already held the correlation of R1 with W1 (D-108); the analysis choices of this second look were frozen BEFORE it was run ([`stage3_analysis_freeze_2026-10-28.md`](stage3_analysis_freeze_2026-10-28.md)), and the analysis code was committed before it ran (19 tests, 16 of 16 mutants killed). Inputs and their limits: [`f5_input_inventory_2026-10-28.md`](f5_input_inventory_2026-10-28.md). Context: [`all_results_assessment_2026-10-28.md`](all_results_assessment_2026-10-28.md). All numbers below are read from [`results/survival_effect_results.json`](results/survival_effect_results.json) by `scripts/g0/survival_effect_report.py`.

## 1. Result

**Result of the pre-set rule (freeze section 4), for the continuous activation correlation R1 as a predictor of the effect similarity W1: no clear predictive value beyond baselines** (0 of 6 cells meet it: improvement of model B over model A of at least 10 % with the lower end of the language-cluster interval above 0). For the record: the Spearman correlation of R1 with W1 is 0.299 to 0.524 over the six cells; adding R1 to the baselines changes the held-out error of W1 by 3.1 % to 4.4 % (the interval lies above zero in 6 of 6 cells); the partial Spearman correlation of R1 with W1 given the baselines is 0.239 to 0.399 (the interval excludes zero in 6 of 6 cells); the baselines alone beat the training-median null by 0.2 % to 3.2 %. **The analysis stops here** (freeze stop rule): no cutoff tuning, no additional predictors, no broader sweep and no switch to steering. Any small association that remains is reported as such and is not promoted.

## 2. Question, data and the reproduction control

Question: when a feature's activation pattern is better preserved by weight quantization (higher R1, the token-level correlation of its activation with full precision), is the effect of an ablation of that feature also better preserved (higher W1, the cosine of the effect fingerprints), beyond simple baselines? Unit: the feature (1,487 features, 27 languages; language is the cluster). Cells: legacy RTN, matched RTN and GPTQ at INT6 and INT4. Baselines (model A): number of active dev sentences, Phase 1 firing rate and mean activation of the feature in its language, the full-precision loss effect |dnll| of the ablation, the number of ablated positions. Model B adds R1; model C adds firing-set overlap and the scale deviation. Held out by language (leave-one-language-out ridge on rank-quantile predictors); improvement = 1 - MAE_B / MAE_A; intervals by a language-cluster bootstrap (2000 replicates, seed 20261028).

**Populations.** Primary (reliable fingerprints: INT16 cosine at or above the numerical floor 0.94888916015625): 1412 features; noisy-effect stratum (the other features): 75; sensitivity S-all: 1487; sensitivity S-loss (|dnll_FP| at or above 0.001465): 842.

**Reproduction control (before any new number):** the committed Spearman correlations of R1 with W1 and their intervals were reproduced from the saved per-feature arrays with the committed code and seed:

| arm | state | rho | interval | n | committed rho | reproduced |
|---|---|---|---|---|---|---|
| legacy RTN | INT6 | 0.533 | [0.476, 0.573] | 1487 | 0.533 | reproduced |
| legacy RTN | INT4 | 0.413 | [0.346, 0.464] | 1487 | 0.413 | reproduced |
| matched RTN | INT6 | 0.520 | [0.465, 0.566] | 1487 | 0.520 | reproduced |
| matched RTN | INT4 | 0.378 | [0.319, 0.426] | 1487 | 0.378 | reproduced |
| GPTQ | INT6 | 0.575 | [0.527, 0.615] | 1487 | 0.575 | reproduced |
| GPTQ | INT4 | 0.381 | [0.320, 0.431] | 1487 | 0.381 | reproduced |

## 3. Primary population: association and prediction

| arm | state | n | Spearman R1-W1 | partial Spearman given baselines | MAE null / A / B | B over A (interval, points) | decision |
|---|---|---|---|---|---|---|---|
| legacy RTN | INT6 | 1412 | 0.472 [0.423, 0.513] | 0.359 [0.295, 0.412] | 0.0105 / 0.0104 / 0.0100 | 4.4 % [2.6, 6.2] | not met |
| legacy RTN | INT4 | 1412 | 0.336 [0.280, 0.384] | 0.265 [0.199, 0.320] | 0.0562 / 0.0545 / 0.0524 | 3.7 % [2.3, 5.2] | not met |
| matched RTN | INT6 | 1412 | 0.460 [0.408, 0.506] | 0.339 [0.273, 0.398] | 0.0097 / 0.0097 / 0.0093 | 3.8 % [1.6, 5.8] | not met |
| matched RTN | INT4 | 1412 | 0.299 [0.245, 0.345] | 0.239 [0.177, 0.290] | 0.0445 / 0.0437 / 0.0424 | 3.1 % [1.8, 4.4] | not met |
| GPTQ | INT6 | 1412 | 0.524 [0.478, 0.565] | 0.399 [0.341, 0.448] | 0.0085 / 0.0085 / 0.0081 | 4.1 % [1.9, 6.2] | not met |
| GPTQ | INT4 | 1412 | 0.306 [0.246, 0.357] | 0.251 [0.186, 0.305] | 0.0299 / 0.0297 / 0.0287 | 3.5 % [2.1, 5.0] | not met |

MAE is the mean absolute error of the predicted W1 on held-out languages. At INT6 W1 is close to 1 for almost all features (small MAE and little room for improvement); at INT4 the errors are larger.

## 4. Secondary results (cannot change the statement of section 1)

| arm | state | C over B | A over null | R1 alone over null | AUROC of R1 for 'W1 at the floor' (events / non-events) | R1 > 0.9: events with / without |
|---|---|---|---|---|---|---|
| legacy RTN | INT6 | -0.5 % [-1.1, 0.1] | 0.6 % [-7.9, 7.1] | 0.9 % [-5.4, 6.1] | 0.762 [0.707, 0.819] (1334 / 78) | 1250 / 54 |
| legacy RTN | INT4 | 0.0 % [-0.8, 0.8] | 3.2 % [-2.0, 7.1] | 3.2 % [-0.5, 6.2] | 0.787 [0.407, 0.996] (5 / 1407) | 3 / 193 |
| matched RTN | INT6 | -0.4 % [-0.7, -0.1] | 0.8 % [-8.3, 7.5] | -0.6 % [-8.1, 5.5] | 0.774 [0.713, 0.837] (1347 / 65) | 1295 / 55 |
| matched RTN | INT4 | -0.2 % [-1.1, 0.6] | 1.8 % [-3.2, 5.6] | 0.9 % [-2.7, 3.9] | 0.704 [0.521, 0.877] (10 / 1402) | 5 / 286 |
| GPTQ | INT6 | -0.3 % [-0.9, 0.3] | 0.2 % [-9.9, 8.4] | 0.1 % [-8.3, 7.3] | 0.815 [0.759, 0.867] (1373 / 39) | 1368 / 39 |
| GPTQ | INT4 | 0.4 % [-0.4, 1.1] | 0.7 % [-5.9, 6.3] | 0.3 % [-4.1, 4.4] | 0.594 [0.556, 0.630] (174 / 1238) | 84 / 423 |

## 5.1. Sensitivity S-all (all 1,487 features)

| arm | state | n | Spearman R1-W1 | partial Spearman given baselines | B over A (interval, points) | rule met in this population |
|---|---|---|---|---|---|---|
| legacy RTN | INT6 | 1487 | 0.533 [0.480, 0.575] | 0.394 [0.325, 0.448] | 6.6 % [4.6, 8.7] | no |
| legacy RTN | INT4 | 1487 | 0.413 [0.350, 0.466] | 0.310 [0.238, 0.366] | 4.6 % [2.8, 6.5] | no |
| matched RTN | INT6 | 1487 | 0.520 [0.465, 0.566] | 0.373 [0.305, 0.431] | 6.5 % [4.5, 8.5] | no |
| matched RTN | INT4 | 1487 | 0.378 [0.320, 0.426] | 0.282 [0.217, 0.334] | 3.5 % [1.9, 5.0] | no |
| GPTQ | INT6 | 1487 | 0.575 [0.526, 0.616] | 0.426 [0.366, 0.475] | 7.2 % [4.9, 9.5] | no |
| GPTQ | INT4 | 1487 | 0.381 [0.317, 0.433] | 0.288 [0.220, 0.346] | 4.4 % [2.8, 5.9] | no |

## 5.2. Sensitivity S-loss (clear full-precision loss effect)

| arm | state | n | Spearman R1-W1 | partial Spearman given baselines | B over A (interval, points) | rule met in this population |
|---|---|---|---|---|---|---|
| legacy RTN | INT6 | 842 | 0.332 [0.255, 0.401] | 0.270 [0.180, 0.346] | 2.7 % [0.9, 4.3] | no |
| legacy RTN | INT4 | 842 | 0.146 [0.071, 0.211] | 0.154 [0.072, 0.224] | 0.5 % [-0.6, 1.6] | no |
| matched RTN | INT6 | 842 | 0.327 [0.255, 0.391] | 0.262 [0.181, 0.334] | 2.2 % [0.3, 4.2] | no |
| matched RTN | INT4 | 842 | 0.118 [0.048, 0.183] | 0.154 [0.073, 0.224] | 2.2 % [0.7, 3.7] | no |
| GPTQ | INT6 | 842 | 0.438 [0.362, 0.505] | 0.348 [0.263, 0.420] | 2.9 % [0.2, 5.3] | no |
| GPTQ | INT4 | 842 | 0.204 [0.134, 0.266] | 0.225 [0.149, 0.293] | 2.8 % [1.3, 4.4] | no |

## 5.3. Noisy-effect stratum (fingerprint not reproducible at the rounding-noise level; secondary and labelled, small n)

| arm | state | n | Spearman R1-W1 | partial Spearman given baselines | B over A (interval, points) | rule met in this population |
|---|---|---|---|---|---|---|
| legacy RTN | INT6 | 75 | 0.400 [0.212, 0.571] | 0.316 [0.028, 0.564] | 5.1 % [-2.1, 12.9] | no |
| legacy RTN | INT4 | 75 | 0.447 [0.262, 0.641] | 0.412 [0.147, 0.644] | 15.6 % [8.5, 24.1] | yes |
| matched RTN | INT6 | 75 | 0.393 [0.217, 0.545] | 0.250 [0.047, 0.441] | 8.1 % [1.1, 15.4] | no |
| matched RTN | INT4 | 75 | 0.309 [0.120, 0.527] | 0.308 [0.015, 0.554] | 6.9 % [2.0, 11.6] | no |
| GPTQ | INT6 | 75 | 0.415 [0.252, 0.572] | 0.358 [0.175, 0.507] | 9.7 % [0.1, 19.7] | no |
| GPTQ | INT4 | 75 | 0.404 [0.231, 0.607] | 0.348 [0.095, 0.563] | 15.0 % [7.1, 23.8] | yes |

## 6. What this does and does not show

- It concerns the continuous activation correlation R1 as a predictor of the effect similarity W1 of a single-feature ablation on 1,487 features of one model (Gemma-2-2B), one layer (12), one SAE (Gemma Scope 16k) and three simulated weight-only quantizers (legacy RTN, matched RTN, GPTQ) at INT6 and INT4, on the F5 dev text.
- R1 and W1 come from the same perturbed model and the features share contexts and sentences inside a language, so any association is not a causal statement about survival and effect; held-out evaluation is by language, not by document (no read-side document ids exist), and no external-panel F5 exists to check it.
- The sample of 1,487 features is stratified, not random: no statement about all features. Margins, raw fingerprints and the sufficient statistics behind R1 were not saved; the baselines are limited to the five listed (the margin-audit addendum's gradient, margin and generic-sensitivity covariates do not exist for F5).
- The decision rule (10 % relative error reduction with an interval above zero) is the project's target carried unchanged from the margin-audit addendum, not a universal threshold; the sensitivity populations and the secondary models cannot change the primary statement.
- The binary retained/lost association label was not used here; its result (AUROC 0.49 under GPTQ, D-108) stands.
- The Llama-3.2-1B check of the plan's Stage 5 has not been run; nothing here says whether the result carries over to another model.

## 7. How this was produced

`scripts/g0/survival_effect.py` (analysis; `--control-only` runs the reproduction control) and `scripts/g0/survival_effect_report.py` (this text); tests `scripts/g0/tests/test_survival_effect.py` and `test_survival_effect_report.py`; inputs: the F5 job directories (private dataset `runs/gptq-extension/stage3_f5_gain`) and the Phase 1 equal-budget survival file; decision entry D-113.
