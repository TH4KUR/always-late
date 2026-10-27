# Stage 2 comparison: the intervention-transfer study (E6), arm against arm

Exploratory (D-101). The verdicts below are the pre-set rules' own (amendment A14), computed by the unchanged report on each arm; the contrasts are descriptive. Feature level: medians over the same eligible real items with the report's own crossed-bootstrap intervals (not paired contrasts). Group level: per-language quantities, arm A minus arm B, paired by language, 95 % percentile bootstrap of the mean over the 27 languages.

## Claim verdicts

| claim | scope | legacy RTN | matched RTN | GPTQ |
|---|---|---|---|---|
| A3 | overall | mixed | inconclusive | mixed |
| A3 | INT4 | supported | inconclusive | supported |
| A3 | INT6 | inconclusive | inconclusive | inconclusive |
| H1 | overall | supported | supported | supported |
| H1 | INT4 | supported | supported | supported |
| H1 | INT6 | supported | supported | supported |
| H2 | status | descriptive | descriptive | descriptive |
| H3 | overall | mixed | mixed | mixed |
| H3 | INT4 | supported | supported | supported |
| H3 | INT6 | inconclusive | inconclusive | reversed |
| H4 | overall | supported | supported | mixed |
| H4 | INT4 | supported | supported | supported |
| H4 | INT6 | supported | supported | inconclusive |
| UTILITY | overall | not met | not met | not met |
| UTILITY | INT4 | met | met | met |
| UTILITY | INT6 | not met | not met | not met |

## Feature level: medians (real items, all splits)

| state | metric | legacy RTN | matched RTN | GPTQ | GPTQ minus matched RTN | intervals overlap (GPTQ vs matched) | same item count |
|---|---|---|---|---|---|---|---|
| INT4 | C_rel | 0.3587 | 0.2914 | 0.2264 | -0.0650 | no | yes |
| INT4 | I_rel | 0.3146 | 0.2347 | 0.1726 | -0.0621 | no | yes |
| INT4 | W_rel | 0.5597 | 0.4945 | 0.4181 | -0.0764 | no | yes |
| INT4 | c_gt_w | 0.1039 | 0.0685 | 0.0513 | -0.0172 | yes | yes |
| INT4 | cosine | 0.7683 | 0.8224 | 0.8774 | +0.0550 | no | yes |
| INT4 | dloss_FP_FP | 0.0022 | 0.0022 | 0.0022 | +0.0000 | yes | yes |
| INT4 | dloss_Q_Q | 0.0048 | 0.0047 | 0.0033 | -0.0015 | yes | yes |
| INT4 | dloss_Q_Q_native | 0.0043 | 0.0038 | 0.0030 | -0.0008 | yes | yes |
| INT4 | dloss_error | 0.0023 | 0.0017 | 0.0006 | -0.0011 | yes | yes |
| INT4 | error_over_floor | 7.6935 | 6.7126 | 5.5357 | -1.1768 | yes | yes |
| INT4 | interaction_share | 0.3403 | 0.2952 | 0.2647 | -0.0305 | no | yes |
| INT4 | loss_base_FP | 3.4697 | 3.4697 | 3.4697 | +0.0000 | yes | yes |
| INT4 | loss_base_Q | 3.8182 | 3.6954 | 3.5870 | -0.1084 | yes | yes |
| INT4 | native_error_share | 0.9703 | 0.9776 | 0.9874 | +0.0098 | yes | yes |
| INT4 | native_normalized_error | 0.6354 | 0.5542 | 0.4643 | -0.0900 | no | yes |
| INT4 | native_scalar_normalized_error | 0.5797 | 0.5207 | 0.4445 | -0.0762 | no | yes |
| INT4 | norm_ratio | 1.0945 | 1.0770 | 1.0706 | -0.0064 | yes | yes |
| INT4 | normalized_error | 0.6641 | 0.5715 | 0.4756 | -0.0959 | no | yes |
| INT4 | scalar_normalized_error | 0.6302 | 0.5516 | 0.4639 | -0.0877 | no | yes |
| INT4 | scalar_residual_share | 0.9744 | 0.9800 | 0.9892 | +0.0092 | no | yes |
| INT4 | total_rel | 0.6641 | 0.5715 | 0.4756 | -0.0959 | no | yes |
| INT6 | C_rel | 0.0999 | 0.0940 | 0.0835 | -0.0105 | no | yes |
| INT6 | I_rel | 0.0860 | 0.0852 | 0.0834 | -0.0018 | yes | yes |
| INT6 | W_rel | 0.1455 | 0.1374 | 0.1183 | -0.0190 | no | yes |
| INT6 | c_gt_w | 0.0599 | 0.0608 | 0.0676 | +0.0068 | yes | yes |
| INT6 | cosine | 0.9840 | 0.9863 | 0.9898 | +0.0035 | no | yes |
| INT6 | dloss_FP_FP | 0.0022 | 0.0022 | 0.0022 | +0.0000 | yes | yes |
| INT6 | dloss_Q_Q | 0.0024 | 0.0023 | 0.0026 | +0.0003 | yes | yes |
| INT6 | dloss_Q_Q_native | 0.0024 | 0.0023 | 0.0025 | +0.0002 | yes | yes |
| INT6 | dloss_error | 0.0002 | -0.0000 | -0.0000 | -0.0000 | yes | yes |
| INT6 | error_over_floor | 1.8786 | 1.7482 | 1.5193 | -0.2289 | yes | yes |
| INT6 | interaction_share | 0.3633 | 0.3814 | 0.4217 | +0.0403 | yes | yes |
| INT6 | loss_base_FP | 3.4697 | 3.4697 | 3.4697 | +0.0000 | yes | yes |
| INT6 | loss_base_Q | 3.4991 | 3.4884 | 3.4878 | -0.0006 | yes | yes |
| INT6 | native_error_share | 1.0017 | 0.9998 | 1.0008 | +0.0011 | yes | yes |
| INT6 | native_normalized_error | 0.1653 | 0.1526 | 0.1321 | -0.0205 | no | yes |
| INT6 | native_scalar_normalized_error | 0.1617 | 0.1498 | 0.1299 | -0.0200 | no | yes |
| INT6 | norm_ratio | 1.0009 | 1.0034 | 1.0028 | -0.0006 | yes | yes |
| INT6 | normalized_error | 0.1647 | 0.1529 | 0.1318 | -0.0211 | no | yes |
| INT6 | scalar_normalized_error | 0.1622 | 0.1508 | 0.1300 | -0.0208 | no | yes |
| INT6 | scalar_residual_share | 0.9919 | 0.9931 | 0.9944 | +0.0013 | yes | yes |
| INT6 | total_rel | 0.1647 | 0.1529 | 0.1318 | -0.0211 | no | yes |

## Group level: GPTQ minus matched RTN

| state | quantity | mean | median | 95 % interval of the mean | languages below / above 0 |
|---|---|---|---|---|---|
| INT4 | net_FP_FP | +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0 / 0 |
| INT4 | C | -0.1761 | -0.0501 | [-0.2761, -0.0920] | 25 / 2 |
| INT4 | W | -0.1365 | -0.0725 | [-0.2335, -0.0591] | 25 / 2 |
| INT4 | I | -0.0319 | -0.0053 | [-0.0693, +0.0007] | 18 / 9 |
| INT4 | total | -0.3445 | -0.1509 | [-0.5205, -0.2010] | 27 / 0 |
| INT4 | native_total | -0.1198 | -0.0421 | [-0.3054, -0.0066] | 23 / 4 |
| INT4 | native_minus_fixed | +0.2246 | +0.1251 | [+0.1260, +0.3263] | 3 / 24 |
| INT6 | net_FP_FP | +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0 / 0 |
| INT6 | C | -0.0190 | -0.0035 | [-0.0397, -0.0034] | 20 / 7 |
| INT6 | W | -0.0089 | -0.0018 | [-0.0229, +0.0028] | 17 / 10 |
| INT6 | I | -0.0002 | +0.0001 | [-0.0015, +0.0009] | 11 / 16 |
| INT6 | total | -0.0281 | -0.0086 | [-0.0542, -0.0069] | 19 / 8 |
| INT6 | native_total | -0.0318 | -0.0088 | [-0.0567, -0.0114] | 19 / 8 |
| INT6 | native_minus_fixed | -0.0037 | -0.0001 | [-0.0186, +0.0107] | 14 / 13 |

## Group level: matched RTN minus legacy RTN

| state | quantity | mean | median | 95 % interval of the mean | languages below / above 0 |
|---|---|---|---|---|---|
| INT4 | net_FP_FP | +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0 / 0 |
| INT4 | C | -0.0627 | -0.0368 | [-0.1444, +0.0174] | 22 / 5 |
| INT4 | W | -0.0412 | -0.0094 | [-0.1558, +0.0738] | 17 / 10 |
| INT4 | I | -0.0123 | -0.0015 | [-0.0747, +0.0520] | 14 / 13 |
| INT4 | total | -0.1162 | -0.0556 | [-0.2397, +0.0133] | 22 / 5 |
| INT4 | native_total | -0.0339 | -0.0385 | [-0.1916, +0.1726] | 21 / 6 |
| INT4 | native_minus_fixed | +0.0823 | +0.0210 | [+0.0092, +0.1849] | 5 / 22 |
| INT6 | net_FP_FP | +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0 / 0 |
| INT6 | C | +0.0175 | +0.0020 | [+0.0009, +0.0368] | 9 / 18 |
| INT6 | W | -0.0123 | -0.0007 | [-0.0262, -0.0013] | 16 / 11 |
| INT6 | I | -0.0005 | -0.0002 | [-0.0024, +0.0011] | 16 / 11 |
| INT6 | total | +0.0047 | +0.0006 | [-0.0195, +0.0269] | 10 / 17 |
| INT6 | native_total | -0.0127 | +0.0003 | [-0.0451, +0.0067] | 13 / 14 |
| INT6 | native_minus_fixed | -0.0174 | -0.0014 | [-0.0368, -0.0018] | 19 / 8 |

## Group level: GPTQ minus legacy RTN

| state | quantity | mean | median | 95 % interval of the mean | languages below / above 0 |
|---|---|---|---|---|---|
| INT4 | net_FP_FP | +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0 / 0 |
| INT4 | C | -0.2388 | -0.1329 | [-0.3511, -0.1371] | 25 / 2 |
| INT4 | W | -0.1777 | -0.0688 | [-0.2896, -0.0817] | 25 / 2 |
| INT4 | I | -0.0442 | -0.0494 | [-0.1097, +0.0303] | 19 / 8 |
| INT4 | total | -0.4607 | -0.4196 | [-0.6237, -0.3087] | 26 / 1 |
| INT4 | native_total | -0.1537 | -0.0953 | [-0.2601, -0.0704] | 25 / 2 |
| INT4 | native_minus_fixed | +0.3070 | +0.2157 | [+0.1976, +0.4228] | 0 / 27 |
| INT6 | net_FP_FP | +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0 / 0 |
| INT6 | C | -0.0015 | -0.0002 | [-0.0281, +0.0200] | 15 / 12 |
| INT6 | W | -0.0212 | -0.0040 | [-0.0415, -0.0030] | 18 / 9 |
| INT6 | I | -0.0008 | -0.0002 | [-0.0020, +0.0004] | 15 / 12 |
| INT6 | total | -0.0234 | -0.0055 | [-0.0611, +0.0019] | 15 / 12 |
| INT6 | native_total | -0.0445 | -0.0066 | [-0.0974, -0.0088] | 17 / 10 |
| INT6 | native_minus_fixed | -0.0211 | -0.0001 | [-0.0419, -0.0043] | 14 / 13 |
