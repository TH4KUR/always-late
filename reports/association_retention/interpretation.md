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
- **Retained features barely change their share and margin at the higher widths; at INT4 they spread.** The median change of target share over all established features is about zero at every width (largest magnitude 0.002). Restricted to the features that are retained in both (`scripts/g0/check_association_retained_share.py`, `feature_status.csv`): the 10th to 90th percentile of the share change is -0.003 to +0.003 at INT8, -0.010 to +0.009 at INT6, -0.020 to +0.015 at INT5 and -0.031 to +0.043 at INT4, with 0 %, 1.5 %, 4.8 % and 13.3 % of retained features changing their share by more than 0.05; the median margin change is about zero at every width. At INT4 the median share of the retained features is slightly higher than in full precision (0.119 to 0.127), which is what survivorship selection produces (the features whose peak weakened are the ones that were lost). So the statement that retention hides no weakening holds up to INT5 and is qualified at INT4. The profile distance grows with bit-width loss but its median stays small (0.094 at INT4).
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

## Agreement of the firing-rate status with the activation-mean view and with the historical correlation

From [`results/crosscheck-eq/crosscheck.md`](results/crosscheck-eq/crosscheck.md) (aggregate profile of the equal-budget stream, the 6,663 established features; produced by `scripts/g0/phase1_association_crosscheck.py`).

- **Firing-rate peak against mean-activation peak.** In full precision the original language is also the unique mean-activation peak for 5,832 of the 6,663 established features (87.5 %); for the other 831 the two views already disagree at the baseline (the status uses firing rates, Deng et al.'s nu uses mean activations, as the register records). Among the 5,832 features on which the views agree at the baseline, the mean view retains 99.6 % at INT8, 96.9 % at INT6 and 74.6 % at INT4, against 98.1 %, 94.7 % and 70.8 % for the firing-rate view on all established features. The two views therefore give the same retention curve; they disagree on individual features (mean only 48 to 197, rate only mostly the 831 baseline disagreements), not on the picture.
- **Status against the correlation criterion of the earlier analysis.** The correlation at the original language and the association status are different quantities and each fails to predict the other. At INT8, 97.6 % of the lost features still have r above 0.9 (their peak moved while their activation pattern stayed almost identical); at INT6 75.8 % of the lost (270 of 356) still have r above 0.9; 7.3 % of the retained (458 of 6,307) have r not above 0.9. At INT5 and INT4 the direction reverses: 36 % (2,136 of 5,937) and 79 % (3,733 of 4,718) of the retained features have r not above 0.9, i.e. they keep their language peak while their activation pattern is no longer highly correlated, and only 6.3 % of the 1,945 INT4-lost features have r above 0.9. Median r is 0.967 (retained) against 0.946 (lost) at INT6 and 0.792 against 0.642 at INT4. Reading the earlier correlation survival as retention of language identity is therefore wrong in both directions, which is the point D-030 made without data.

## Robustness: the support rule and what predicts loss

From [`results/robustness-eq`](results/robustness-eq/robustness.md) (aggregate profile of the equal-budget stream; descriptive, no model fitted, no test; produced by `scripts/g0/phase1_association_robustness.py`).

- **Support rule.** Requiring at least 20, 50 or 100 positive tokens at the peak changes the number established in full precision (6,755 / 6,663 / 5,810) but not the picture: the loss as a share of the cohort at INT6 is 4.1 / 4.4 / 4.1 % and at INT4 22.0 / 24.1 / 23.0 %; the retained share among established at INT4 is 73.6 / 70.8 / 68.0 %; the numerical floor stays 0.6 to 0.7 %.
- **Loss concentrates in features with a small baseline margin.** The margin is the firing rate in the original language minus that of the strongest other language, in full precision. At INT6 the lowest margin decile loses 28.8 % and the top five deciles lose 2 % or less (rank correlation of margin with loss -0.28); at INT8 the lowest decile loses 15.8 % and most other deciles 0 to 1 %; the INT16/INT12 floor shows the same shape (7.4 to 7.8 % in the lowest decile, about 0 elsewhere). So at the higher widths what is lost is mostly a near tie that small numerical changes flip, not a language-wide change; target share, target rate and mean activation relate to loss more weakly (rank correlations about -0.06 to -0.16 up to INT5).
- **Activation scale matters more at INT4.** The maximum activation of the feature at its original language is almost unrelated to loss at INT8 to INT5 (rank correlation -0.02 to -0.06) and clearly related at INT4 (-0.21; lowest decile 55.8 % lost, highest 17.1 %), as are target rate (-0.22) and mean activation (-0.27); the margin is still related (-0.29) but even its top decile loses 7.2 % at INT4. This is consistent with the earlier finding that per-language activation scale carries the survival differences at the lowest widths; it is a bin-wise association among correlated covariates, not a causal statement (Q13 asks for a defined causal question before that).

## Does the frozen SAE permit a clear association conclusion?

Partly. The instrument behaves consistently where it was tested: the new full-precision evaluation reproduces the original run exactly, the restore-only control equals full precision exactly, the near-lossless INT16/INT12 controls lose only 0.7 % of the cohort, and two independent views of the peak (firing rate and mean activation) give the same retention curve. What the measurement cannot separate is a change in the model's language information from a change in how a dictionary trained on the full-precision model decomposes the quantized activations: a peak that moves to another language is a statement about the 16,384 frozen coordinates. This study did not measure SAE reconstruction error or fidelity by language and width under quantization (candidates M2 and Q15 in the reassessment, not selected), did not use a second dictionary (M4) and did not test function (M3). If any of these is taken up, the lost features of this study (`feature_status.csv`, status `different_peak` and `low_support`) are the natural cohort; none is started automatically.

## Reproducibility

The analysis was run twice from the same sidecar with the same seed (Slurm `31807` and `31808`, then `31919` and `31920`): all four files of each stream (`analysis.json`, `feature_status.csv`, `language_summary.csv`, `association_retention.md`) are byte-identical (sha256 equal, 2026-10-21). The full-precision evaluation of the new run equals the original Phase 1 run exactly (see `STATUS.md`). The headline counts were also re-derived with a separate, plain implementation that imports none of the analysis modules (`scripts/g0/check_association_headline_independent.py`, run on the raw equal-budget profile): 6,663 established and 124 / 168 / 356 / 726 / 1,945 lost at INT8 / INT7 / INT6 / INT5 / INT4, 56 at INT16 and INT12, 0 at INT0, identical to the report.

## Remaining

The all-token stream has no sentence-level intervals (sentence shards were saved for the equal-budget stream only, D-038).
