# Stage 2 comparison: the intervention-transfer study (E6), arm against arm: external panel, six languages

Exploratory (D-101). The verdicts below are the pre-set rules' own (amendment A14), computed by the unchanged report on each arm; the contrasts are descriptive. Feature level: medians over the same eligible real items with the report's own crossed-bootstrap intervals (not paired contrasts). Group level: per-language quantities, arm A minus arm B, paired by language, 95 % percentile bootstrap of the mean over the 6 languages.

## Claim verdicts

| claim | scope | legacy RTN | matched RTN | GPTQ |
|---|---|---|---|---|
| A3 | overall | mixed | mixed | inconclusive |
| A3 | INT4 | supported | inconclusive | inconclusive |
| A3 | INT6 | reversed | reversed | inconclusive |
| H1 | overall | supported | supported | supported |
| H1 | INT4 | supported | supported | supported |
| H1 | INT6 | supported | supported | supported |
| H2 | status | descriptive | descriptive | descriptive |
| H3 | overall | supported | supported | mixed |
| H3 | INT4 | supported | supported | supported |
| H3 | INT6 | supported | supported | inconclusive |
| H4 | overall | mixed | mixed | mixed |
| H4 | INT4 | supported | supported | supported |
| H4 | INT6 | inconclusive | inconclusive | inconclusive |
| UTILITY | overall | met | met | not met |
| UTILITY | INT4 | met | met | met |
| UTILITY | INT6 | met | met | not met |

## Feature level: medians (real items, all splits)

| state | metric | legacy RTN | matched RTN | GPTQ | GPTQ minus matched RTN | intervals overlap (GPTQ vs matched) | same item count |
|---|---|---|---|---|---|---|---|
| INT4 | C_rel | 0.3516 | 0.2934 | 0.2181 | -0.0752 | no | yes |
| INT4 | I_rel | 0.3403 | 0.2574 | 0.1857 | -0.0717 | no | yes |
| INT4 | W_rel | 0.5792 | 0.5120 | 0.4244 | -0.0876 | no | yes |
| INT4 | c_gt_w | 0.0720 | 0.0568 | 0.0457 | -0.0111 | yes | yes |
| INT4 | cosine | 0.7541 | 0.8083 | 0.8758 | +0.0675 | no | yes |
| INT4 | dloss_FP_FP | 0.0009 | 0.0009 | 0.0009 | +0.0000 | yes | yes |
| INT4 | dloss_Q_Q | 0.0015 | 0.0013 | 0.0010 | -0.0003 | yes | yes |
| INT4 | dloss_Q_Q_native | 0.0012 | 0.0012 | 0.0009 | -0.0003 | yes | yes |
| INT4 | dloss_error | 0.0006 | 0.0005 | 0.0000 | -0.0004 | yes | yes |
| INT4 | error_over_floor | 7.0657 | 5.9926 | 4.9092 | -1.0834 | yes | yes |
| INT4 | interaction_share | 0.3628 | 0.3190 | 0.2810 | -0.0380 | no | yes |
| INT4 | loss_base_FP | 2.4381 | 2.4381 | 2.4381 | +0.0000 | yes | yes |
| INT4 | loss_base_Q | 2.8165 | 2.6918 | 2.5644 | -0.1274 | yes | yes |
| INT4 | native_error_share | 0.9777 | 0.9817 | 0.9915 | +0.0099 | no | yes |
| INT4 | native_normalized_error | 0.6558 | 0.5726 | 0.4699 | -0.1027 | no | yes |
| INT4 | native_scalar_normalized_error | 0.6036 | 0.5340 | 0.4529 | -0.0811 | no | yes |
| INT4 | norm_ratio | 1.0884 | 1.0741 | 1.0757 | +0.0017 | yes | yes |
| INT4 | normalized_error | 0.6789 | 0.5892 | 0.4758 | -0.1133 | no | yes |
| INT4 | scalar_normalized_error | 0.6344 | 0.5609 | 0.4623 | -0.0986 | no | yes |
| INT4 | scalar_residual_share | 0.9684 | 0.9767 | 0.9889 | +0.0123 | no | yes |
| INT4 | total_rel | 0.6789 | 0.5892 | 0.4758 | -0.1133 | no | yes |
| INT6 | C_rel | 0.1066 | 0.1008 | 0.0892 | -0.0116 | yes | yes |
| INT6 | I_rel | 0.0993 | 0.0974 | 0.0957 | -0.0017 | yes | yes |
| INT6 | W_rel | 0.1562 | 0.1465 | 0.1254 | -0.0211 | no | yes |
| INT6 | c_gt_w | 0.0658 | 0.0512 | 0.0582 | +0.0069 | yes | yes |
| INT6 | cosine | 0.9821 | 0.9843 | 0.9887 | +0.0044 | no | yes |
| INT6 | dloss_FP_FP | 0.0009 | 0.0009 | 0.0009 | +0.0000 | yes | yes |
| INT6 | dloss_Q_Q | 0.0008 | 0.0008 | 0.0008 | -0.0000 | yes | yes |
| INT6 | dloss_Q_Q_native | 0.0008 | 0.0009 | 0.0008 | -0.0001 | yes | yes |
| INT6 | dloss_error | -0.0000 | 0.0001 | -0.0000 | -0.0001 | yes | yes |
| INT6 | error_over_floor | 1.7734 | 1.6807 | 1.4437 | -0.2370 | no | yes |
| INT6 | interaction_share | 0.3860 | 0.4134 | 0.4610 | +0.0476 | no | yes |
| INT6 | loss_base_FP | 2.4381 | 2.4381 | 2.4381 | +0.0000 | yes | yes |
| INT6 | loss_base_Q | 2.4549 | 2.4524 | 2.4411 | -0.0113 | yes | yes |
| INT6 | native_error_share | 1.0020 | 1.0018 | 1.0029 | +0.0011 | yes | yes |
| INT6 | native_normalized_error | 0.1742 | 0.1636 | 0.1386 | -0.0250 | no | yes |
| INT6 | native_scalar_normalized_error | 0.1706 | 0.1602 | 0.1353 | -0.0249 | no | yes |
| INT6 | norm_ratio | 1.0053 | 1.0090 | 1.0043 | -0.0047 | yes | yes |
| INT6 | normalized_error | 0.1742 | 0.1624 | 0.1372 | -0.0252 | no | yes |
| INT6 | scalar_normalized_error | 0.1712 | 0.1603 | 0.1353 | -0.0250 | no | yes |
| INT6 | scalar_residual_share | 0.9916 | 0.9939 | 0.9941 | +0.0002 | yes | yes |
| INT6 | total_rel | 0.1742 | 0.1624 | 0.1372 | -0.0252 | no | yes |

## Group level: GPTQ minus matched RTN

| state | quantity | mean | median | 95 % interval of the mean | languages below / above 0 |
|---|---|---|---|---|---|
| INT4 | net_FP_FP | +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0 / 0 |
| INT4 | C | -0.1610 | -0.0514 | [-0.2979, -0.0295] | 6 / 0 |
| INT4 | W | -0.2996 | -0.1203 | [-0.7267, -0.0483] | 5 / 1 |
| INT4 | I | -0.0499 | +0.0031 | [-0.1682, +0.0301] | 3 / 3 |
| INT4 | total | -0.5106 | -0.1872 | [-1.0541, -0.0872] | 6 / 0 |
| INT4 | native_total | -0.3513 | -0.0693 | [-0.8793, -0.0328] | 6 / 0 |
| INT4 | native_minus_fixed | +0.1592 | +0.0821 | [+0.0347, +0.3526] | 1 / 5 |
| INT6 | net_FP_FP | +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0 / 0 |
| INT6 | C | -0.0168 | -0.0021 | [-0.0618, +0.0115] | 4 / 2 |
| INT6 | W | -0.0116 | -0.0125 | [-0.0177, -0.0052] | 5 / 1 |
| INT6 | I | -0.0005 | -0.0004 | [-0.0011, +0.0000] | 4 / 2 |
| INT6 | total | -0.0289 | -0.0073 | [-0.0783, +0.0005] | 4 / 2 |
| INT6 | native_total | -0.0415 | -0.0119 | [-0.1050, -0.0066] | 5 / 1 |
| INT6 | native_minus_fixed | -0.0126 | -0.0089 | [-0.0297, +0.0004] | 4 / 2 |

## Group level: matched RTN minus legacy RTN

| state | quantity | mean | median | 95 % interval of the mean | languages below / above 0 |
|---|---|---|---|---|---|
| INT4 | net_FP_FP | +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0 / 0 |
| INT4 | C | -0.1189 | -0.0527 | [-0.2418, -0.0294] | 5 / 1 |
| INT4 | W | +0.0747 | -0.0049 | [-0.0686, +0.2845] | 4 / 2 |
| INT4 | I | +0.0224 | +0.0232 | [-0.0227, +0.0703] | 2 / 4 |
| INT4 | total | -0.0218 | -0.0362 | [-0.2563, +0.2320] | 4 / 2 |
| INT4 | native_total | +0.1667 | +0.0069 | [-0.1200, +0.6098] | 2 / 4 |
| INT4 | native_minus_fixed | +0.1885 | +0.0822 | [+0.0480, +0.4016] | 0 / 6 |
| INT6 | net_FP_FP | +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0 / 0 |
| INT6 | C | -0.0042 | +0.0016 | [-0.0275, +0.0116] | 2 / 4 |
| INT6 | W | -0.0160 | -0.0035 | [-0.0435, -0.0006] | 3 / 3 |
| INT6 | I | +0.0009 | -0.0002 | [-0.0010, +0.0035] | 3 / 3 |
| INT6 | total | -0.0193 | +0.0006 | [-0.0451, +0.0050] | 2 / 4 |
| INT6 | native_total | -0.0280 | +0.0001 | [-0.0872, +0.0044] | 3 / 3 |
| INT6 | native_minus_fixed | -0.0087 | -0.0005 | [-0.0513, +0.0256] | 3 / 3 |

## Group level: GPTQ minus legacy RTN

| state | quantity | mean | median | 95 % interval of the mean | languages below / above 0 |
|---|---|---|---|---|---|
| INT4 | net_FP_FP | +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0 / 0 |
| INT4 | C | -0.2799 | -0.2721 | [-0.4501, -0.1134] | 6 / 0 |
| INT4 | W | -0.2250 | -0.0786 | [-0.4658, -0.0456] | 6 / 0 |
| INT4 | I | -0.0274 | -0.0233 | [-0.1433, +0.0828] | 3 / 3 |
| INT4 | total | -0.5323 | -0.4808 | [-0.8712, -0.2207] | 6 / 0 |
| INT4 | native_total | -0.1846 | -0.1264 | [-0.3193, -0.0622] | 6 / 0 |
| INT4 | native_minus_fixed | +0.3477 | +0.2715 | [+0.1382, +0.5704] | 0 / 6 |
| INT6 | net_FP_FP | +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0 / 0 |
| INT6 | C | -0.0210 | -0.0038 | [-0.0613, +0.0084] | 4 / 2 |
| INT6 | W | -0.0276 | -0.0161 | [-0.0599, -0.0064] | 5 / 1 |
| INT6 | I | +0.0004 | -0.0004 | [-0.0019, +0.0033] | 3 / 3 |
| INT6 | total | -0.0482 | -0.0147 | [-0.1199, -0.0016] | 4 / 2 |
| INT6 | native_total | -0.0695 | -0.0122 | [-0.1905, -0.0056] | 5 / 1 |
| INT6 | native_minus_fixed | -0.0214 | +0.0007 | [-0.0779, +0.0150] | 3 / 3 |
