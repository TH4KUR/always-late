# A2: is the loss of language association predicted by how borderline the feature was? (D-035 per-feature statuses, within language)

**Exploratory and descriptive; nothing here is causal.** The statuses are the D-035 statuses of the `eq` stream (measured on devtest; features selected and ranked on dev). Established = `retained` in full precision; lost at a precision = not `retained` there. The predictors are read from the full-precision row of the same feature: m0 = the peak language's rate minus the runner-up's (`margin`), r0 = m0 over the target rate, s0 = the target share, a0 = the activity scale (`target_mean`). Input: `feature_status_eq.csv`, sha256 `3ab979852f3007987bfd364d334d9199e0cb2c3064d57cc932e287e70aec6a71`.

**Limit that travels with the numbers:** m0 was measured on the same text as the statuses, so a borderline feature's m0 is itself noisy and regression to the mean inflates the association of m0 with loss; the rates at INT16 and INT12 (rounding noise only) are the floor to compare with.

## Verdicts of the hypotheses written before any number (plan A2)

- **H-A2a** (loss decreases with the margin; AUROC of -m0 at INT6 at least 0.70; lowest quintile loses more than the highest at INT6, INT5, INT4): **met** (AUROC at INT6 0.897).
- **H-A2b** (at INT4 the loss exceeds the floor in every m0 quintile): **exceeds the floor in every quintile** (top quintile: 0.073 at INT4 against 0.000 at INT16).
- **H-A2c** (given m0, scale and share have odds ratios closer to 1): **met**.

### All languages

Established features: 6663 in 27 languages.

Lost fraction by within-language quintile of the full-precision margin m0 (quintile 1 = the most borderline fifth of each language; pooled counts, Wilson 95 % intervals):

| precision | all lost | q1 (smallest m0) | q2 | q3 | q4 | q5 (largest m0) |
|---|---|---|---|---|---|---|
| INT16 | 0.008 | 0.038 [0.029, 0.050] (51/1343) | 0.003 [0.001, 0.008] (4/1334) | 0.001 [0.000, 0.004] (1/1334) | 0.000 [0.000, 0.003] (0/1334) | 0.000 [0.000, 0.003] (0/1318) |
| INT12 | 0.008 | 0.039 [0.030, 0.050] (52/1343) | 0.003 [0.001, 0.008] (4/1334) | 0.000 [0.000, 0.003] (0/1334) | 0.000 [0.000, 0.003] (0/1334) | 0.000 [0.000, 0.003] (0/1318) |
| INT8 | 0.019 | 0.088 [0.074, 0.104] (118/1343) | 0.004 [0.002, 0.009] (5/1334) | 0.001 [0.000, 0.004] (1/1334) | 0.000 [0.000, 0.003] (0/1334) | 0.000 [0.000, 0.003] (0/1318) |
| INT7 | 0.025 | 0.106 [0.091, 0.124] (143/1343) | 0.015 [0.010, 0.023] (20/1334) | 0.003 [0.001, 0.008] (4/1334) | 0.000 [0.000, 0.003] (0/1334) | 0.001 [0.000, 0.004] (1/1318) |
| INT6 | 0.053 | 0.207 [0.186, 0.229] (278/1343) | 0.043 [0.034, 0.056] (58/1334) | 0.011 [0.007, 0.018] (15/1334) | 0.003 [0.001, 0.008] (4/1334) | 0.001 [0.000, 0.004] (1/1318) |
| INT5 | 0.109 | 0.330 [0.305, 0.355] (443/1343) | 0.121 [0.104, 0.139] (161/1334) | 0.054 [0.043, 0.067] (72/1334) | 0.030 [0.022, 0.041] (40/1334) | 0.008 [0.004, 0.014] (10/1318) |
| INT4 | 0.292 | 0.535 [0.508, 0.561] (718/1343) | 0.408 [0.382, 0.434] (544/1334) | 0.265 [0.242, 0.289] (353/1334) | 0.175 [0.156, 0.197] (234/1334) | 0.073 [0.060, 0.088] (96/1318) |

Within-language AUROC for loss (the score is minus the predictor, so 0.5 = no information and values above 0.5 mean that a smaller value goes with loss; mean over the languages with at least 20 features and 3 lost and kept ones, percentile interval over languages, 2000 draws):

| precision | -m0 | -r0 | -s0 (share) | -a0 (scale) | languages |
|---|---|---|---|---|---|
| INT16 | 0.933 [0.885, 0.973] | 0.841 [0.712, 0.941] | 0.678 [0.552, 0.802] | 0.704 [0.602, 0.794] | 9 |
| INT12 | 0.942 [0.897, 0.985] | 0.882 [0.786, 0.967] | 0.674 [0.616, 0.734] | 0.702 [0.589, 0.810] | 6 |
| INT8 | 0.960 [0.946, 0.973] | 0.906 [0.863, 0.946] | 0.727 [0.682, 0.771] | 0.663 [0.622, 0.699] | 17 |
| INT7 | 0.911 [0.893, 0.928] | 0.869 [0.824, 0.909] | 0.670 [0.626, 0.717] | 0.637 [0.590, 0.683] | 23 |
| INT6 | 0.897 [0.880, 0.912] | 0.830 [0.801, 0.858] | 0.695 [0.666, 0.725] | 0.701 [0.662, 0.742] | 25 |
| INT5 | 0.813 [0.787, 0.838] | 0.771 [0.742, 0.797] | 0.660 [0.627, 0.695] | 0.651 [0.620, 0.680] | 27 |
| INT4 | 0.752 [0.736, 0.767] | 0.671 [0.651, 0.693] | 0.643 [0.619, 0.667] | 0.690 [0.669, 0.712] | 27 |

Logistic regression of loss on z(log m0), z(log a0) and z(s0) with a fixed effect per language (odds ratio per standard deviation, cluster-robust 95 % interval by language; languages without variation in the outcome are dropped):

| precision | m0 | scale a0 | share s0 | features | languages used |
|---|---|---|---|---|---|
| INT16 | 0.089 [0.058, 0.134] | 0.862 [0.494, 1.504] | 1.819 [1.295, 2.555] | 5814 | 20 |
| INT12 | 0.057 [0.032, 0.100] | 1.376 [0.784, 2.416] | 1.917 [1.495, 2.459] | 5729 | 21 |
| INT8 | 0.077 [0.054, 0.109] | 1.443 [1.118, 1.861] | 1.017 [0.528, 1.958] | 6602 | 26 |
| INT7 | 0.126 [0.099, 0.161] | 1.697 [1.281, 2.249] | 1.171 [0.814, 1.684] | 6624 | 26 |
| INT6 | 0.149 [0.126, 0.176] | 1.093 [0.890, 1.342] | 1.045 [0.784, 1.393] | 6663 | 27 |
| INT5 | 0.208 [0.177, 0.245] | 1.413 [1.202, 1.662] | 1.067 [0.922, 1.235] | 6663 | 27 |
| INT4 | 0.420 [0.364, 0.484] | 0.764 [0.630, 0.926] | 0.923 [0.836, 1.019] | 6663 | 27 |

### Cohort without eng_Latn

Established features: 5216 in 26 languages (without eng_Latn).

Lost fraction by within-language quintile of the full-precision margin m0 (quintile 1 = the most borderline fifth of each language; pooled counts, Wilson 95 % intervals):

| precision | all lost | q1 (smallest m0) | q2 | q3 | q4 | q5 (largest m0) |
|---|---|---|---|---|---|---|
| INT16 | 0.009 | 0.040 [0.030, 0.053] (42/1053) | 0.003 [0.001, 0.008] (3/1045) | 0.001 [0.000, 0.005] (1/1044) | 0.000 [0.000, 0.004] (0/1045) | 0.000 [0.000, 0.004] (0/1029) |
| INT12 | 0.008 | 0.036 [0.026, 0.049] (38/1053) | 0.003 [0.001, 0.008] (3/1045) | 0.000 [0.000, 0.004] (0/1044) | 0.000 [0.000, 0.004] (0/1045) | 0.000 [0.000, 0.004] (0/1029) |
| INT8 | 0.019 | 0.087 [0.072, 0.106] (92/1053) | 0.004 [0.001, 0.010] (4/1045) | 0.001 [0.000, 0.005] (1/1044) | 0.000 [0.000, 0.004] (0/1045) | 0.000 [0.000, 0.004] (0/1029) |
| INT7 | 0.029 | 0.121 [0.102, 0.142] (127/1053) | 0.019 [0.012, 0.029] (20/1045) | 0.004 [0.001, 0.010] (4/1044) | 0.000 [0.000, 0.004] (0/1045) | 0.001 [0.000, 0.005] (1/1029) |
| INT6 | 0.057 | 0.219 [0.195, 0.245] (231/1053) | 0.050 [0.038, 0.065] (52/1045) | 0.011 [0.007, 0.020] (12/1044) | 0.004 [0.001, 0.010] (4/1045) | 0.000 [0.000, 0.004] (0/1029) |
| INT5 | 0.111 | 0.323 [0.295, 0.352] (340/1053) | 0.127 [0.108, 0.149] (133/1045) | 0.060 [0.047, 0.076] (63/1044) | 0.033 [0.024, 0.046] (35/1045) | 0.007 [0.003, 0.014] (7/1029) |
| INT4 | 0.339 | 0.610 [0.580, 0.639] (642/1053) | 0.478 [0.448, 0.509] (500/1045) | 0.306 [0.278, 0.334] (319/1044) | 0.212 [0.189, 0.238] (222/1045) | 0.084 [0.068, 0.102] (86/1029) |

Within-language AUROC for loss (the score is minus the predictor, so 0.5 = no information and values above 0.5 mean that a smaller value goes with loss; mean over the languages with at least 20 features and 3 lost and kept ones, percentile interval over languages, 2000 draws):

| precision | -m0 | -r0 | -s0 (share) | -a0 (scale) | languages |
|---|---|---|---|---|---|
| INT16 | 0.930 [0.877, 0.976] | 0.837 [0.694, 0.949] | 0.677 [0.535, 0.812] | 0.697 [0.586, 0.799] | 8 |
| INT12 | 0.937 [0.883, 0.990] | 0.874 [0.750, 0.977] | 0.658 [0.600, 0.716] | 0.714 [0.588, 0.840] | 5 |
| INT8 | 0.960 [0.945, 0.973] | 0.904 [0.857, 0.943] | 0.728 [0.679, 0.774] | 0.663 [0.623, 0.704] | 16 |
| INT7 | 0.908 [0.891, 0.926] | 0.864 [0.822, 0.903] | 0.662 [0.617, 0.705] | 0.638 [0.589, 0.686] | 22 |
| INT6 | 0.897 [0.880, 0.914] | 0.828 [0.799, 0.855] | 0.694 [0.662, 0.727] | 0.702 [0.663, 0.741] | 24 |
| INT5 | 0.811 [0.784, 0.835] | 0.768 [0.738, 0.795] | 0.658 [0.621, 0.696] | 0.653 [0.622, 0.682] | 26 |
| INT4 | 0.753 [0.737, 0.769] | 0.675 [0.655, 0.696] | 0.648 [0.624, 0.671] | 0.688 [0.665, 0.710] | 26 |

Logistic regression of loss on z(log m0), z(log a0) and z(s0) with a fixed effect per language (odds ratio per standard deviation, cluster-robust 95 % interval by language; languages without variation in the outcome are dropped):

| precision | m0 | scale a0 | share s0 | features | languages used |
|---|---|---|---|---|---|
| INT16 | 0.078 [0.045, 0.137] | 0.906 [0.473, 1.733] | 1.936 [1.336, 2.806] | 4367 | 19 |
| INT12 | 0.059 [0.029, 0.120] | 1.328 [0.671, 2.627] | 1.913 [1.458, 2.510] | 4282 | 20 |
| INT8 | 0.077 [0.052, 0.114] | 1.470 [1.085, 1.993] | 0.982 [0.476, 2.023] | 5155 | 25 |
| INT7 | 0.125 [0.098, 0.161] | 1.715 [1.257, 2.339] | 1.185 [0.811, 1.732] | 5177 | 25 |
| INT6 | 0.144 [0.118, 0.177] | 1.096 [0.858, 1.400] | 1.042 [0.763, 1.422] | 5216 | 26 |
| INT5 | 0.212 [0.181, 0.247] | 1.399 [1.159, 1.690] | 1.075 [0.918, 1.259] | 5216 | 26 |
| INT4 | 0.385 [0.334, 0.443] | 0.823 [0.727, 0.930] | 0.918 [0.815, 1.034] | 5216 | 26 |

