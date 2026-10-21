# Language association retention under weight quantization: reading of the D-035 results

Written 2026-10-21 from the generated reports in [`results/eq`](results/eq/association_retention.md) (equal-budget stream, 1,000 paired bootstrap draws, seed 0) and [`results/all`](results/all/association_retention.md) (all-token stream, aggregate profile, no intervals). All numbers below are copied from those files; this text adds reading and limits. Design: [`reports/survival_method_reassessment.md` section 7](../survival_method_reassessment.md#7-selected-next-study-language-association-retention-d-035); plan, checks and the run: [plan](../../docs/superpowers/plans/2026-10-21-language-association-retention.md), [verification](plan_verification.md), [status](STATUS.md). This is an exploratory measurement on already examined data, not a confirmatory test and not paper prose.

## What was measured

The 8,086 SAE features that the original Phase 1 run selected as language-selective on the FLORES dev split (each with a dev peak language) were evaluated on FLORES devtest with the same frozen layer-12 Gemma Scope 16k SAE, on the full-precision model and on RTN-quantized models, using each condition's own activations (the old results saved only correlations and full-precision-side statistics). A feature counts as **established** if, in full precision on devtest, its dev peak language is the unique peak with at least 50 positive tokens (6,663 of 8,086, 82.4 %), and as **retained** in a condition if its original language is again the unique, supported peak. **Lost** means established in full precision and not retained.

## Results (equal-budget stream, fixed denominator of 8,086)

| condition | lost (count, share of 8,086) | retained among established | median total variation of the language profile |
|---|---|---|---|
| INT8 | 124 (1.5 %) | 98.1 % | 0.010 |
| INT7 | 168 (2.1 %) | 97.5 % | 0.015 |
| INT6 | 356 (4.4 %) | 94.7 % | 0.024 |
| INT5 | 726 (9.0 %) | 89.1 % | 0.042 |
| INT4 | 1,945 (24.1 %) | 70.8 % | 0.094 |
| INT16 / INT12 (numerical floor) | 56 (0.7 %) | 99.2 % | 0.006 |
| INT0 (restore only) | 0 | 100 % | 0 |

- **How features are lost.** Almost never by falling silent: no established feature becomes inactive at any width. At INT6 278 of the 356 lost features have a different unique peak language, 33 a tied peak and 45 too little support; at INT4 the counts are 1,443, 57 and 445. Losing the association is mostly the peak moving to another language.
- **Retained features do not visibly weaken.** The median change of target share is about zero at every width (largest magnitude 0.002) and the median change of margin about zero. Among features that keep their peak, specificity as measured by share and margin barely moves; the profile distance grows with bit-width loss but its median stays small (0.094 at INT4).
- **Numerical floor.** INT16 and INT12 (near-lossless quantization) already lose 56 features (0.7 %); INT8 loses about twice that, INT6 about six times. Part of what is called loss at the high widths is the instrument's noise floor (small activations crossing the JumpReLU threshold, ties near the peak), not a language effect.
- **The two streams agree.** The all-token aggregate gives 6,841 established (84.6 %) and 103 / 157 / 358 / 667 / 1,860 lost at INT8 / INT7 / INT6 / INT5 / INT4, floor 57 to 60, so the picture does not depend on the equal-budget subsampling.

## By original language

- The lost fraction among established features is similar across resource tiers at the widths where little is lost and differs at INT4. Median over the nine languages of each tier (descriptive; no test was run): at INT6 4.1 % (high), 4.5 % (mid), 6.4 % (low); at INT4 26.2 %, 30.1 %, 36.9 %; the INT16 floor is 0.6 to 1.1 %. At INT4 the extremes are hau_Latn (49 %), kan_Knda (46 %) and lao_Laoo (43 %) against eng_Latn (12 %), por_Latn (21 %) and fra_Latn (22 %).
- At INT4 every one of the 27 languages has a loss interval entirely above the INT16 loss interval; at INT6 ten of 27 do. For the other languages at INT6 a language-level difference from the floor is not separable with these sentence samples.
- English holds 1,719 of the 8,086 selective features, so pooled fractions are English-weighted (INT6: 0.780 retained of all selective pooled against 0.747 as the mean over languages). Read the per-language rows for statements about languages.
- The retention fraction by language in the table of the report has the FP establishment rate in it (compare the INT0 column); the loss fraction among established features is the comparison between languages.

## Selective features against matched controls

The 2,860 matched controls (supported, non-selective, same firing-rate strata) are far less often established (1,163, 41 %), and those that are lose their peak label more often: retained among established at INT6 84.8 % against 94.7 % for selective features, at INT4 52.3 % against 70.8 %. Controls have weaker language peaks by construction and their label is only their argmax, so this does not mean that selectivity is protective; it shows that peak retention is easier for features with a strong language peak, which is the property the selection enriches for.

## Comparison with the earlier survival measure

The earlier measure (Pearson correlation at the peak language above 0.9, all active features pooled) gave 94.9 %, 59.0 % and 13.75 % at INT6, INT5 and INT4 (`reports/phase1/mask/eq/summary.csv`). Association retention among established selective features is 94.7 %, 89.1 % and 70.8 % at the same widths. The populations differ (all active features against the selective cohort) and the quantities differ, so this is not a like-for-like comparison, but it supports the point of D-030: activation-pattern correlation falls much faster than the peak-language association does, so the earlier survival percentages should not be read as loss of language identity.

## Limits

- **Exploratory, already examined data.** Both FLORES splits had been analysed; the cohort was selected on dev and evaluated on devtest. No claim of confirmation; no new hypothesis-test family was run.
- **Intervals are conservative upward for loss and off-centre.** In the report the observed loss fraction lies outside its percentile interval at INT6 and INT16 (observed 4.4 % and 0.7 %, bootstrap means 5.0 % and 1.1 %): resampling sentences adds noise that makes more features look lost, so the intervals describe sampling variation and are shifted upward relative to the observed values (diagnostics are in `analysis.json`). The observed values, not the interval midpoints, are the estimates. Intervals do not cover other corpora, other SAEs or other thresholds.
- **Threshold dependent definitions.** Established, retained and low support use at least 50 positive tokens at the peak and a panel-mean rate of at least 1e-5, the tagging defaults, as operational safeguards; no sensitivity analysis over these thresholds was run.
- **What is not shown.** That a retained feature keeps its meaning or causal effect, that a lost feature has lost its function, that the SAE remains a good measurement basis under quantization (a language peak that moves may reflect the frozen dictionary rather than the model), or anything about other models, quantizers (RTN only) or languages.
- **Baseline ceiling.** Only 82.4 % of the original selective features reproduce their dev association on devtest in full precision, so the headline retained fraction cannot exceed that; both fractions are reported.

## Checks still pending

1. Reproducibility: the reruns `31919` and `31920` must give byte-identical analysis files (plan Task 4); Ada was unreachable at 12:19 IST.
2. Agreement between the firing-rate view (used for status) and the activation-mean view (target mean, nu in `feature_status.csv`) and the historical correlation at the original language (column `historical_corr`): needs `feature_status.csv`, which is on Ada (16 MB per stream) and not yet copied.
3. Optional: sensitivity of the lost fraction to the 50-positives support rule.
