# Locked FLORES-to-Wikipedia scoring (revision R1, task 1a of the publication recovery plan; POST HOC; written 2026-10-27 02:49 IST)

Protocol fixed and hashed in [`../../revision_R1.json`](../../revision_R1.json) (commit `ed80bca`) BEFORE this run; plan: [`docs/superpowers/plans/2026-10-26-publication-recovery-after-results.md`](../../../../docs/superpowers/plans/2026-10-26-publication-recovery-after-results.md) Task 1 (Codex's plan; the locked models are hers: `reports/publication_reassessment_2026-10-26/locked_flores_models.json`, exported after the main outcomes with no external outcome used). Code: `scripts/g0/phase1_transfer_recovery.py` (26 tests, 26 mutants killed; 328 transfer tests green). Command:

```
python scripts/g0/phase1_transfer_recovery.py --locked reports/publication_reassessment_2026-10-26/locked_flores_models.json --main <transfer_runs/analysis_main> \
  --external <transfer_runs/analysis_external> --committed reports/intervention_transfer/results/main/predictive_validation.csv --out <transfer_runs/analysis_external/recovery>
```

Files: `cross_corpus_locked.csv` (one row per state and protocol), `cross_corpus_locked.json` (all numbers, the hashes of the locked file and of the six per-item inputs), `verification.json`.

## Verification before scoring (all passed, with differences of exactly 0.0)

(V1) a refit on the main calibration items by the frozen protocol equals the exported fit (12 models: strengths equal, mean, scale, coefficients, intercepts equal to the last digit); (V2) the locked models give the committed held-out MAE of every model and state on the main held-out items (12 values); (V3) `error_norm / (fp_norm + FLORES floor of the kind)` and the five-floor eligibility reproduce the normalized error and the eligibility of all 32,320 main items per state. The secondary rows below also reproduce the A11 E3 numbers of [`../external/README.md`](../external/README.md) exactly (INT4 20.63 %, INT6 19.65 %, INT16 26.69 %), an independent confirmation that the exported fits equal the refit used there.

## Result (frozen fits, no refit; external held-out-feature items of the real direction; reduction of the held-out MAE of A+B)

PRIMARY protocol (FLORES floors for the target, the covariates and the eligibility; 1,368 of 6,406 external items eligible, 621 held-out items, 99 features, 459 passages):

| state | held-out MAE: A / A+G / A+B / A+B+G / null | main-calibration CV: A / A+G / null | old comparator (locked baseline) | reduction | new comparator (A, A+G, null by main CV) | reduction | against A (H3 quantity) |
|---|---|---|---|---|---|---|---|
| INT16 (rounding reference) | 0.0213 / 0.0211 / 0.0151 / 0.0154 / 0.0298 | 0.01495 / 0.01484 / 0.02291 | A+G | 28.2 % [21.3, 35.1] | A+G | 28.2 % [21.3, 35.1] | 29.0 % [22.0, 35.6] |
| INT6 | 0.0380 / 0.0380 / 0.0309 / 0.0303 / 0.0367 | 0.02916 / 0.02937 / **0.02820** | A | 18.6 % [4.9, 29.5] | **null** | **15.8 % [-2.7, 28.7]** | 18.6 % [4.9, 29.5] |
| INT4 | 0.1375 / 0.1360 / 0.1088 / 0.1069 / 0.1397 | 0.11137 / 0.11057 / 0.11474 | A+G | 19.9 % [7.8, 31.3] | A+G | 19.9 % [7.8, 31.3] | 20.9 % [8.8, 32.0] |

SECONDARY protocol (the external floors of D-095 for the target and the eligibility; 1,444 eligible items, 659 held-out, 100 features, 483 passages): INT16 26.7 % [19.5, 34.0] (A+G); INT6 19.7 % [7.5, 29.5] against the old comparator A and **17.5 % [1.3, 28.9] against the null**; INT4 20.6 % [9.3, 31.2] (A+G).

## Reading

- The sensitivity to other directions, fitted only on FLORES calibration features, reduces the external held-out error with an interval above zero in every state and under both floors, except one cell: at INT6 with the FLORES floors the null (the calibration median) is a stronger comparator than both activation baselines (main-calibration CV 0.0282 against 0.0292 and 0.0294; on the external items 0.0367 against 0.0380), and the reduction against it is 15.8 % [-2.7, 28.7], an interval that includes zero. Against the old comparator (the locked baseline A) the reduction at INT6 is 18.6 % [4.9, 29.5]. At INT4 the null is not competitive (0.1397 against 0.1360 for A+G) and the reduction is 19.9 % [7.8, 31.3].
- The reduction in the rounding-only INT16 state is as large (28.2 %): the predictor transfers across corpora and languages but is still not shown to be quantization-specific; a quantization-specific claim would need incremental gain over measurement-sensitivity covariates and an outcome outside the original overlapping context set (plan, promotion decision), which this table does not provide.
- The external floors change the numbers little (INT4 20.6 % against 19.9 %): the cross-corpus result does not hinge on the normalization of the target.

## Limits

Post hoc (selected after D-094) and written after the external panel had been analysed under the original rules (D-095) and the A11 E3 row had been read; no choice here depends on those numbers (revision_R1.json, `honesty_note`). Feature-held-out only for the external items (the same feature split as the main study; the external contexts are other documents, which is the point of the test). Six languages and one corpus family; `held-out` items share features with nothing in the calibration half. Pairwise-eligible real items still differ in FP effect scale from their controls (plan: common-support analysis, not done here). The prediction needs four extra control interventions per feature and context, so it is not shown to be a cheaper audit than measuring the target. The original UTILITY verdicts (main: not met; external: met, D-095) are unchanged and are not re-evaluated here.

## Update 2026-10-27 03:18 IST: the stored gradient scalars against the signed NLL changes (revision R2, task 1b; POST HOC)

Protocol: [`../../revision_R2_gradient_scalars.json`](../../revision_R2_gradient_scalars.json) (commit `a9184d5`, written before the code and the run). Code: `scripts/g0/phase1_transfer_gradient_check.py` (17 tests, 23 mutants killed; 345 transfer tests green). Command: `python scripts/g0/phase1_transfer_gradient_check.py --main <analysis_main> --external <analysis_external> --out <dir>`. Files: `gradient_vs_nll.csv` (corpus, state, comparison, group; 180 rows), `gradient_vs_nll.json` (also the noise thresholds, the check that C0 is identical across the state tables (it is, in both corpora), and the sha256 of the six inputs).

**What is compared.** The A9 baseline stores per item the first-order loss prediction of the edit (minus the dot product of the autograd gradient of the mean NLL and the edit, summed over the edited rows) in the FP model (g_pred_fp, C0), in the quantized model for the fixed FP edit (C1) and for the native edit (C2); S1 and S2 are the quantized-minus-FP shifts of these predictions against the shifts of the actual signed NLL changes of the cells. This is the **scalar NLL gradient against a scalar NLL target**; the logit-vector JVP was never computed. Noise threshold of the actual change: the 95th percentile of the INT16 shift (0.0062 nats FLORES, 0.0034 Wikipedia). Intervals: crossed bootstrap, 2,000 replicates. Real direction, all 6,464 (FLORES) and 6,406 (Wikipedia) items unless stated.

| | FLORES (27 languages) | Wikipedia (6 languages) |
|---|---|---|
| C0, FP model, FP edit: Pearson r | 0.63 [0.46, 0.80] | 0.53 [0.38, 0.69] |
| C0 slope of actual on predicted | 1.29 [1.05, 1.51] | 0.95 [0.77, 1.22] |
| C0 sign agreement above noise | 92 % [89, 95] | 90 % [86, 93] |
| C1 at INT6 / INT4: r | 0.62 / 0.59 | 0.54 / 0.59 |
| S2 (native edit shift), INT16 reference: r | 0.09 [0.03, 0.16] | 0.03 [-0.03, 0.08] |
| S2, INT6: r | 0.20 [-0.29, 0.62] | 0.38 [0.23, 0.49] |
| S2, INT4: r (slope, sign agreement above noise) | 0.58 [0.45, 0.70] (0.85, 87 %) | 0.66 [0.53, 0.76] (0.91, 86 %) |
| S1 (fixed edit shift), INT4: r (slope) | 0.51 [0.35, 0.67] (0.76) | 0.67 [0.56, 0.76] (1.03) |

By size of the intervention (terciles of the FP edit norm, C0): FLORES r 0.92 (slope 0.93), 0.88 (1.23), 0.57 (1.38); Wikipedia r 0.84 (1.01), 0.84 (1.12), 0.46 (0.89): the first-order approximation is accurate for the small edits and degrades for the largest ones, as expected for a full zero-ablation, which is not locally linear; the numbers are reported at the actual edit size only. Eligible real items (2,203 / 1,444) give the same picture (C0 r 0.62 / 0.45; S2 at INT4 r 0.52 / 0.52). The four control directions pooled are predicted better than the real direction (C0 r 0.89 / 0.79; S2 at INT4 0.82 / 0.67), consistent with their smaller, more linear effects.

**Reading.**
- On its own target the gradient baseline is not a failed predictor: it reproduces the sign of the signed NLL change in about nine of ten items above the noise level and its linear fit is moderate (r about 0.5 to 0.6, driven by the large edits). The earlier result that gradient attribution adds nothing to the prediction of the effect-VECTOR error (D-094) is therefore not evidence that gradients fail at the scalar loss; the vector-valued first-order effect remains untested.
- The shift of the prediction between the FP and the quantized model tracks the actual shift of the loss change at INT4 (r 0.5 to 0.7, slope about 0.8 to 1.0, sign agreement above noise 86 to 88 %), is weak and unreliable at INT6 (on FLORES the interval of r spans [-0.29, 0.62]) and shows no relation in the INT16 rounding reference (r 0.0 to 0.1), which is the noise reference of the shift.
- No threshold declares the baseline good or bad (protocol). Pearson correlations are sensitive to the few large edits (heavy tails); no rank correlation was in the protocol and none is reported. The result does not make gradients a cheaper audit than measuring the target.

## Update 2026-10-27 03:57 IST: common-support analysis of the real direction against its controls (revision R3; POST HOC)

Protocol: [`../../revision_R3_common_support.json`](../../revision_R3_common_support.json) (commit `b44af5f`, written before the code and the run). Code: `scripts/g0/phase1_transfer_support.py` (19 tests, 28 mutants killed; 364 transfer tests green). Command: `python scripts/g0/phase1_transfer_support.py --main <analysis_main> --external <analysis_external> --out <dir>`. Files: `support_coverage.csv`, `support_contrast.csv`, `support.json` (also the check that the sets are identical in the three state tables: they are, in both corpora, and the sha256 of the six inputs).

**Design.** For each control kind separately, the pairs of the real item and the control item of the same feature and context; S0 all pairs, S1 both items eligible, S2 S1 with the FP effect-norm ratio rho = fp_norm(real) / fp_norm(control) in the predefined band [0.8, 1.25]. Coverage first; a contrast only with at least 100 pairs and 20 features. No stored effect vector is rescaled.

**Coverage (matched swapped direction; the random directions are in the CSV).** FLORES: 6,464 pairs; **986 jointly eligible (144 features; the median rho is 2.02, so the real direction has about twice the FP effect of its swapped control)**; **144 inside the band (43 features, 113 source clusters; 14.6 % of S1)**; the real items with all four controls eligible: 323. Wikipedia: 6,406; 566 (113 features; median rho 1.89); **107 in the band (50 features, 102 passages; 18.9 %)**; all four: 149. The FP edit norms and FP residual norms are equal for real and control items (median ratio 1.00: the controls are norm-matched and share the context), so the FP effect norm is the quantity that differs. The three random directions have only 4 to 12 pairs in the band (S1 of 222 to 470, with median rho about 3.5): **no contrast is reported for them (insufficient support)**.

**Contrast, real minus matched swapped, inside the band (S2; medians over pairs, crossed-bootstrap intervals):**

| | FLORES (144 pairs) | Wikipedia (107 pairs) |
|---|---|---|
| normalized-error gap, INT16 (reference) / INT6 / INT4 | -0.004 [-0.009, 0.000] / +0.006 [-0.001, +0.013] / **+0.043 [+0.006, +0.087]** | -0.006 [-0.012, -0.001] / +0.005 [-0.005, +0.012] / +0.036 [-0.008, +0.076] |
| ratio of the error norms (real / swapped), INT16 / INT6 / INT4 | x1.005 [0.98, 1.04] / **x1.12 [1.06, 1.17]** / **x1.14 [1.07, 1.25]** | x1.02 [0.98, 1.05] / **x1.12 [1.06, 1.18]** / **x1.18 [1.05, 1.24]** |
| share of pairs with the larger error for the real direction, INT16 / INT6 / INT4 | 0.33 [0.17, 0.50] / 0.65 [0.48, 0.80] / 0.68 [0.51, 0.82] | 0.30 [0.15, 0.46] / 0.63 [0.44, 0.80] / 0.68 [0.48, 0.86] |
| cosine gap at INT4 (INT6; INT16) | **-0.028 [-0.046, -0.001]** (-0.001; 0.000) | -0.019 [-0.043, +0.010] (-0.001; +0.001) |

Without the band (S1, both eligible, the real direction has about twice the FP effect) the error-norm ratio is about x2.1 at INT4 (log2 +1.10 [0.89, 1.29] on FLORES), which is the scale difference, and the cosine gap is about zero.

**Reading.**
- At matched FP effect size the real direction's quantization error is only about 12 % (INT6) to 14 to 18 % (INT4) larger in norm than that of a norm-matched swapped decoder direction, with an interval above the rounding reference (x1.00 to x1.02 at INT16), and it is larger in about two thirds of the pairs at INT6 and INT4 against about one third at INT16. This is a modest direction-specific component on top of the dominant generic one, in both corpora. The normalized-error gap is separated from zero only at INT4 on FLORES.
- The cosine gap changes sign relative to the original unmatched table (D-094: +0.065 at INT4, the real direction less rotated): at matched support the real direction is, if anything, slightly MORE rotated at INT4 (-0.028 on FLORES). The earlier 'less rotated' therefore reflected the larger FP effect of the real direction, as the plan suspected. The original specificity statement ('close to its controls') stays qualified: it mixed scale and specificity; it is not retracted, it is refined.
- This is not proof of selectivity or of equivalence. The in-band pairs are a small selected part of all pairs (144 of 6,464 and 107 of 6,406) chosen on a FP quantity; the control is another feature's decoder row, matched in norm but not in role; one control kind only; no claim about damage to language selectivity follows. The INT16 reference shows no direction-specific error under rounding.

## Update 2026-10-27 04:36 IST: native versus fixed edit, paired per language (revision R4; POST HOC)

Protocol: [`../../revision_R4_native_fixed.json`](../../revision_R4_native_fixed.json) (commit `85b3a46`, written before the code and the run). Code: `scripts/g0/phase1_transfer_native_fixed.py` (15 tests, 22 mutants killed; 379 transfer tests green). Command: `python scripts/g0/phase1_transfer_native_fixed.py --main-dirs INT16=<dir> INT6=<dir> INT4=<dir> --external-dirs INT16=<dir> INT6=<dir> INT4=<dir> --core reports/intervention_transfer/core_sample.json --external-languages eng_Latn spa_Latn rus_Cyrl zho_Hans hin_Deva yor_Latn --out <dir>`. Files: `native_fixed_per_language.csv` (the five cells, C, W, I, F, N, R, D and intervals per language), `native_fixed_summary.csv`, `native_fixed.json` (the largest identity error is 1.1e-16).

**Quantities** (nats per token, net harm = group minus the mean of the three control draws, pooled by target rows; per language and state): W = FQ - FF (weights only), F = QQ - FF (fixed edit), N = QN - FF (native edit), **R = QN - FQ = N - W (the native remainder)**, **D = QN - QQ = N - F (native minus fixed)**. Medians over languages of these PAIRED per-language values, 95 % sentence-bootstrap intervals (the scheme and seed of the A3 bootstrap, the same resampled sentences for every language and cell).

| | FLORES, 27 languages | Wikipedia, 6 languages |
|---|---|---|
| INT4: median R (native minus weights-only) | **+0.0138 [+0.0093, +0.0233]**; R > 0 in 21 languages, range -0.78 to +0.23 | +0.0092 [-0.0089, +0.0138]; R > 0 in 4, range -0.51 to +0.09 |
| INT4: median D (native minus fixed) | **-0.351 [-0.358, -0.293]; negative in all 27** | **-0.485 [-0.520, -0.452]; negative in all 6** |
| INT4: median W / N / F | 0.096 / 0.134 / 0.492 | 0.073 / 0.093 / 0.284 |
| INT4: median \|R\| - \|W\|; languages with \|R\| > \|W\| | -0.057 [-0.077, -0.050]; 5 of 27 | -0.055 [-0.069, -0.049]; 1 of 6 |
| INT6: median R; median D | +0.0016 [+0.0004, +0.0031]; -0.0009 [-0.0013, -0.0003] (D < 0 in 15) | -0.0009 [-0.0024, -0.0001]; -0.0038 [-0.0046, -0.0030] (D < 0 in 3) |
| INT16 (rounding reference): median R; median D | -0.0000 [-0.0001, +0.0002]; +0.0002 [+0.0001, +0.0005] | -0.0001 [-0.0005, +0.0001]; +0.0001 [-0.0000, +0.0003] |

**Reading.**
- At INT4 the native harm is smaller than the fixed-edit harm in every language of both corpora (median paired difference -0.35 and -0.49 nats per token), and in the median language it exceeds the weights-only effect by a small positive remainder (+0.014 [0.009, 0.023] on FLORES; the paired median agrees with the value Codex derived from the committed table). So the context term of the fixed edit (D-094: |C| about three times |W|) mostly belongs to the fixed edit and is largely absent for the native one, as the cell medians suggested; this is now shown with paired intervals, not with differences of medians.
- The statement 'the native harm is that of the weights' is true of the median language only: the native remainder R ranges from -0.78 to +0.23 across the 27 FLORES languages at INT4 (lao_Laoo -0.67, fra_Latn -0.21, amh_Ethi +0.23), the median of |R| / |N| is 0.29, and |R| exceeds |W| in 5 languages. Where R is negative the native amplitude shift REDUCES the weights-only harm; where it is positive it adds to it.
- Nothing here says that amplitude 'causes' a share of the harm: D is a paired difference of two edits (the fixed one subtracts a full-precision-sized contribution, the native one the quantized state's own), R contains the interaction of the amplitude change with the weights, and the median of |D| / |F| (0.61 on FLORES) is a descriptive ratio of paired values, not a causal share.
- At INT6 the differences are tiny (|R|, |D| about 0.001 to 0.004) and mixed in sign between the corpora (R +0.0016 on FLORES, -0.0009 on Wikipedia); at INT16 they are numerical noise (all medians within 0.0005), the reference for these quantities.
- Limits: the intervals cover sentence sampling only (the groups and the control draws are fixed); six languages on Wikipedia; one model, layer and quantizer; the median over languages treats the languages as exchangeable although they differ strongly in full-precision harm (zho_Hans FP net harm 7.6).
