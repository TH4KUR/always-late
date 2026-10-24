# Study G: graded gain of language-selective features

Exploratory (the FLORES devtest sentences were examined in earlier studies, D-039); the rules below were fixed in the design before any number (`docs/superpowers/specs/2026-10-24-graded-gain-study-design.md` sections 5 and 9). Harm = NLL after the edit minus NLL before it in the same state (nats per token on the language's own text); net harm = group minus the mean of the matched controls scaled by the same ratio; intervals are paired bootstraps over sentences.

Down ratios 0.3536, 0.1768, 0.0884 (plus zero-ablation); up ratios 2.8284, 5.6569, 11.3137; fine up ratios 1.10, 1.20; 3 control draws; states FP, INT6, INT5, INT4; 27 languages.

## Verdicts

| hypothesis | rule (counts out of the languages) | result | verdict |
|---|---|---|---|
| H-G1 dose, down (FP) | net harm strictly ordered over the three down ratios and zero-ablation, in at least 24, and the median ordered | 27 of 27; median net 0.0223, 0.0902, 0.1687, 0.2624 | **met** |
| H-G2 down is specific (FP) | lower bound of net harm above 0 at ratio 0.0884, in at least 24 | 26 of 27 | **met** |
| H-G3 dose, up (FP) | group harm increasing over the up ratios in at least 24 and net harm above 0 at the largest in at least 20 | 27 and 24 of 27 | **met** |
| H-G4 precision, down | tau > 0 at the deepest down ratio in at least 24, median not falling by more than one standard error | 27 of 27; median ok False | **not met** |
| H-G5a generic repair (INT4) | median control harm below 0 at 1.1 or 1.2, interval below 0 | x1.1: median control harm 0.0000 [-0.0005, 0.0002]; x1.2: median control harm 0.0002 [-0.0004, 0.0008] | **not met** |
| H-G5b group-specific repair (INT4) | upper bound of net harm below 0 at the better fine ratio in at least 14 | 12 of 27 | **not met** |
| H-G5 alpha* against 1 / g_ratio (INT4) | Spearman above 0.3 | -0.034 | **not met** |
| H-G6 consistency | net harm positive at ratio 0.0884 in at least 24 in every state and Spearman FP against INT4 above 0.5 | FP 27, INT6 27, INT5 27, INT4 27; rank correlation 0.936 | **met** |

H-G4, detail: median group harm at the deepest down ratio by state: FP 0.1939, INT6 0.1861, INT5 0.1940, INT4 0.3807. The pre-registered clause allows a fall of the median of at most one bootstrap error (the larger of the two states'); result False. **Labelled post hoc sensitivity, not part of the verdict:** with a tolerance of two bootstrap errors the clause gives True (the pre-registered rule is not changed).

## Median over the languages by state and ratio (nats per token)

### FP

| ratio | group harm | control harm | net harm | languages with net > 0 | total change vs unedited FP |
|---|---|---|---|---|---|
| 0.0000 | 0.2862 | 0.0235 | 0.2624 | 27 of 27 | 0.2862 |
| 0.0884 | 0.1939 | 0.0193 | 0.1687 | 27 of 27 | 0.1939 |
| 0.1768 | 0.1051 | 0.0120 | 0.0902 | 26 of 27 | 0.1051 |
| 0.3536 | 0.0304 | 0.0065 | 0.0223 | 25 of 27 | 0.0304 |
| 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0 of 27 | 0.0000 |
| 1.1000 | 0.0005 | 0.0002 | 0.0007 | 20 of 27 | 0.0005 |
| 1.2000 | 0.0022 | 0.0004 | 0.0020 | 21 of 27 | 0.0022 |
| 2.8284 | 0.0840 | 0.0336 | 0.0533 | 25 of 27 | 0.0840 |
| 5.6569 | 0.5935 | 0.2353 | 0.3328 | 25 of 27 | 0.5935 |
| 11.3137 | 2.3537 | 1.2605 | 1.2128 | 25 of 27 | 2.3537 |

### INT6

| ratio | group harm | control harm | net harm | languages with net > 0 | total change vs unedited FP |
|---|---|---|---|---|---|
| 0.0000 | 0.2953 | 0.0234 | 0.2709 | 27 of 27 | 0.3182 |
| 0.0884 | 0.1861 | 0.0183 | 0.1658 | 27 of 27 | 0.2165 |
| 0.1768 | 0.1071 | 0.0141 | 0.0917 | 26 of 27 | 0.1300 |
| 0.3536 | 0.0331 | 0.0064 | 0.0243 | 25 of 27 | 0.0568 |
| 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0 of 27 | 0.0233 |
| 1.1000 | 0.0008 | 0.0001 | 0.0008 | 18 of 27 | 0.0252 |
| 1.2000 | 0.0017 | 0.0002 | 0.0014 | 21 of 27 | 0.0277 |
| 2.8284 | 0.0861 | 0.0346 | 0.0545 | 25 of 27 | 0.1096 |
| 5.6569 | 0.5750 | 0.2354 | 0.3413 | 25 of 27 | 0.5916 |
| 11.3137 | 2.3384 | 1.3184 | 1.1868 | 24 of 27 | 2.3835 |

### INT5

| ratio | group harm | control harm | net harm | languages with net > 0 | total change vs unedited FP |
|---|---|---|---|---|---|
| 0.0000 | 0.3273 | 0.0265 | 0.2808 | 27 of 27 | 0.4521 |
| 0.0884 | 0.1940 | 0.0218 | 0.1795 | 27 of 27 | 0.3186 |
| 0.1768 | 0.1195 | 0.0158 | 0.1021 | 26 of 27 | 0.1970 |
| 0.3536 | 0.0442 | 0.0075 | 0.0286 | 25 of 27 | 0.1187 |
| 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0 of 27 | 0.0778 |
| 1.1000 | 0.0005 | -0.0001 | 0.0006 | 21 of 27 | 0.0802 |
| 1.2000 | 0.0017 | 0.0004 | 0.0017 | 22 of 27 | 0.0839 |
| 2.8284 | 0.0938 | 0.0379 | 0.0578 | 25 of 27 | 0.1848 |
| 5.6569 | 0.6160 | 0.2676 | 0.3809 | 25 of 27 | 0.6801 |
| 11.3137 | 2.4247 | 1.3249 | 1.0695 | 25 of 27 | 2.5550 |

### INT4

| ratio | group harm | control harm | net harm | languages with net > 0 | total change vs unedited FP |
|---|---|---|---|---|---|
| 0.0000 | 0.4930 | 0.0272 | 0.4316 | 27 of 27 | 1.1252 |
| 0.0884 | 0.3807 | 0.0225 | 0.3233 | 27 of 27 | 0.9301 |
| 0.1768 | 0.2661 | 0.0186 | 0.2429 | 27 of 27 | 0.7096 |
| 0.3536 | 0.1091 | 0.0117 | 0.0771 | 27 of 27 | 0.6294 |
| 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0 of 27 | 0.5172 |
| 1.1000 | -0.0014 | 0.0000 | -0.0010 | 6 of 27 | 0.5156 |
| 1.2000 | -0.0017 | 0.0002 | -0.0013 | 9 of 27 | 0.5155 |
| 2.8284 | 0.0785 | 0.0534 | 0.0403 | 22 of 27 | 0.6215 |
| 5.6569 | 0.5870 | 0.3270 | 0.2645 | 24 of 27 | 1.0745 |
| 11.3137 | 2.3011 | 1.4685 | 0.8095 | 23 of 27 | 2.8958 |

## By language: net harm at ratio 0.0884 and 11.3137

| language | FP down | INT6 down | INT5 down | INT4 down | FP up | INT6 up | INT5 up | INT4 up |
|---|---|---|---|---|---|---|---|---|
| eng_Latn | 0.1067 | 0.1021 | 0.1137 | 0.1127 | -0.2195 | -0.2168 | -0.2177 | -0.3830 |
| deu_Latn | 0.4222 | 0.4454 | 0.4402 | 0.4379 | 1.5292 | 1.4659 | 1.4089 | 1.3122 |
| spa_Latn | 0.2160 | 0.2281 | 0.2524 | 0.3205 | 0.7209 | 0.7565 | 0.7216 | 0.4360 |
| fra_Latn | 0.3067 | 0.3435 | 0.3063 | 0.5860 | 0.8444 | 0.8454 | 0.8231 | 0.5988 |
| por_Latn | 0.5633 | 0.6416 | 0.7653 | 0.6187 | 0.8934 | 0.8976 | 0.9278 | 0.9401 |
| rus_Cyrl | 0.0320 | 0.0373 | 0.0369 | 0.1434 | 1.2348 | 1.2319 | 1.0695 | 0.5253 |
| zho_Hans | 5.4597 | 5.8784 | 5.3285 | 6.4423 | 1.6129 | 1.6048 | 1.3359 | 1.2388 |
| jpn_Jpan | 0.1569 | 0.1603 | 0.1713 | 0.2561 | 0.9992 | 1.0015 | 0.8812 | 0.4929 |
| kor_Hang | 0.1926 | 0.1754 | 0.2206 | 0.7076 | 0.8366 | 0.8148 | 0.7413 | 0.4031 |
| hin_Deva | 0.0516 | 0.0706 | 0.0479 | 0.1979 | 1.5405 | 1.4356 | 1.2921 | 1.0692 |
| ben_Beng | 0.0076 | 0.0072 | 0.0094 | 0.0311 | 0.6351 | 0.6136 | 0.6957 | 0.4508 |
| tam_Taml | 0.0217 | 0.0230 | 0.0313 | 0.0634 | 2.0507 | 1.9919 | 2.1672 | 1.6236 |
| tur_Latn | 0.1260 | 0.1228 | 0.1371 | 0.3983 | 0.5040 | 0.4660 | 0.3831 | -0.4464 |
| vie_Latn | 0.0007 | 0.0002 | 0.0025 | 0.0513 | 0.7718 | 0.7929 | 0.6955 | 0.5191 |
| ind_Latn | 0.0182 | 0.0170 | 0.0274 | 0.0223 | 0.2211 | 0.2113 | 0.2288 | 0.3323 |
| tha_Thai | 0.0311 | 0.0354 | 0.0429 | 0.1282 | 0.0694 | -0.0061 | -0.0724 | -0.4021 |
| pol_Latn | 0.1793 | 0.1658 | 0.2331 | 0.3865 | 1.3897 | 1.3632 | 1.2183 | 1.2418 |
| ukr_Cyrl | 0.2045 | 0.2436 | 0.1795 | 0.6416 | 2.5303 | 2.4182 | 2.1036 | 1.6081 |
| kan_Knda | 0.0261 | 0.0262 | 0.0261 | 0.0374 | 1.2128 | 1.2773 | 1.3754 | 1.9266 |
| tel_Telu | 0.0199 | 0.0266 | 0.0216 | 0.1299 | 1.2267 | 1.1868 | 1.3628 | 0.8095 |
| yor_Latn | 1.1457 | 1.1839 | 1.4640 | 1.7138 | -0.0152 | -0.2763 | 0.0983 | -0.2677 |
| sun_Latn | 0.6771 | 0.5972 | 0.7530 | 0.8869 | 0.3112 | 0.3196 | 0.3194 | 0.1801 |
| amh_Ethi | 0.7678 | 0.8377 | 0.8276 | 1.7646 | 6.1792 | 6.2704 | 5.3773 | 4.1466 |
| ibo_Latn | 0.2782 | 0.2729 | 0.2953 | 0.3737 | 1.7387 | 1.7365 | 1.8042 | 1.2312 |
| hau_Latn | 0.1289 | 0.1344 | 0.1462 | 0.1737 | 1.3263 | 1.2506 | 1.2719 | 1.2759 |
| khm_Khmr | 0.1687 | 0.1746 | 0.2134 | 0.3233 | 5.4296 | 5.2514 | 4.8172 | 2.6741 |
| lao_Laoo | 1.1689 | 1.1950 | 1.9964 | 1.9968 | 7.1586 | 7.0190 | 6.6028 | 4.4897 |

## Optimal gain alpha* (the ratio with the lowest loss; 1 = leave the group alone)

| state | languages with alpha* above 1 | languages with alpha* below 1 | median alpha* |
|---|---|---|---|
| FP | 7 of 27 | 2 of 27 | 1.000 |
| INT6 | 7 of 27 | 2 of 27 | 1.000 |
| INT5 | 7 of 27 | 1 of 27 | 1.000 |
| INT4 | 21 of 27 | 0 of 27 | 1.200 |

## Reliability of the sample

The sample is 300 held-out sentences per language, 3 matched-control draws per language and 27 languages in 4 states; it was not reduced to save time (user instruction 2026-10-24). The intervals of the decision rules come from the bootstrap over sentences and are conditional on the control draws; this table adds the second source of uncertainty (the variation between control draws). Per state and ratio (medians over the languages): |net harm|, the standard error from the sentences, the standard error from the control draws, and the number of languages whose |net harm| exceeds twice the sentence error alone / twice both errors combined.

| state | ratio | median abs net harm | SE sentences | SE control draws | languages clear (sentences) | languages clear (both) |
|---|---|---|---|---|---|---|
| FP | 0 | 0.2624 | 0.0115 | 0.0024 | 27 of 27 | 27 of 27 |
| FP | 0.0884 | 0.1687 | 0.0086 | 0.0020 | 26 of 27 | 26 of 27 |
| FP | 0.177 | 0.0902 | 0.0065 | 0.0017 | 26 of 27 | 26 of 27 |
| FP | 0.354 | 0.0223 | 0.0037 | 0.0012 | 24 of 27 | 23 of 27 |
| FP | 1.1 | 0.0011 | 0.0005 | 0.0002 | 16 of 27 | 14 of 27 |
| FP | 1.2 | 0.0024 | 0.0008 | 0.0003 | 21 of 27 | 21 of 27 |
| FP | 2.83 | 0.0533 | 0.0048 | 0.0041 | 27 of 27 | 25 of 27 |
| FP | 5.66 | 0.3328 | 0.0166 | 0.0358 | 26 of 27 | 22 of 27 |
| FP | 11.3 | 1.2128 | 0.0416 | 0.1613 | 25 of 27 | 23 of 27 |
| INT6 | 0 | 0.2709 | 0.0115 | 0.0027 | 27 of 27 | 27 of 27 |
| INT6 | 0.0884 | 0.1658 | 0.0088 | 0.0022 | 26 of 27 | 26 of 27 |
| INT6 | 0.177 | 0.0917 | 0.0067 | 0.0018 | 26 of 27 | 26 of 27 |
| INT6 | 0.354 | 0.0243 | 0.0038 | 0.0011 | 22 of 27 | 22 of 27 |
| INT6 | 1.1 | 0.0009 | 0.0005 | 0.0002 | 14 of 27 | 14 of 27 |
| INT6 | 1.2 | 0.0025 | 0.0008 | 0.0003 | 20 of 27 | 19 of 27 |
| INT6 | 2.83 | 0.0545 | 0.0048 | 0.0047 | 27 of 27 | 24 of 27 |
| INT6 | 5.66 | 0.3413 | 0.0166 | 0.0364 | 26 of 27 | 22 of 27 |
| INT6 | 11.3 | 1.1868 | 0.0409 | 0.1635 | 26 of 27 | 24 of 27 |
| INT5 | 0 | 0.2808 | 0.0118 | 0.0027 | 27 of 27 | 27 of 27 |
| INT5 | 0.0884 | 0.1795 | 0.0090 | 0.0023 | 26 of 27 | 26 of 27 |
| INT5 | 0.177 | 0.1021 | 0.0067 | 0.0020 | 26 of 27 | 26 of 27 |
| INT5 | 0.354 | 0.0286 | 0.0042 | 0.0010 | 24 of 27 | 22 of 27 |
| INT5 | 1.1 | 0.0011 | 0.0005 | 0.0002 | 14 of 27 | 14 of 27 |
| INT5 | 1.2 | 0.0019 | 0.0008 | 0.0003 | 16 of 27 | 13 of 27 |
| INT5 | 2.83 | 0.0578 | 0.0052 | 0.0055 | 27 of 27 | 25 of 27 |
| INT5 | 5.66 | 0.3809 | 0.0164 | 0.0392 | 26 of 27 | 23 of 27 |
| INT5 | 11.3 | 1.0695 | 0.0444 | 0.1546 | 26 of 27 | 24 of 27 |
| INT4 | 0 | 0.4316 | 0.0159 | 0.0069 | 27 of 27 | 27 of 27 |
| INT4 | 0.0884 | 0.3233 | 0.0113 | 0.0056 | 27 of 27 | 27 of 27 |
| INT4 | 0.177 | 0.2429 | 0.0087 | 0.0043 | 27 of 27 | 27 of 27 |
| INT4 | 0.354 | 0.0771 | 0.0052 | 0.0024 | 26 of 27 | 26 of 27 |
| INT4 | 1.1 | 0.0015 | 0.0006 | 0.0004 | 16 of 27 | 14 of 27 |
| INT4 | 1.2 | 0.0029 | 0.0011 | 0.0006 | 15 of 27 | 15 of 27 |
| INT4 | 2.83 | 0.0584 | 0.0069 | 0.0092 | 25 of 27 | 21 of 27 |
| INT4 | 5.66 | 0.2645 | 0.0183 | 0.0473 | 27 of 27 | 22 of 27 |
| INT4 | 11.3 | 0.8095 | 0.0436 | 0.1640 | 27 of 27 | 22 of 27 |

## Limits

- Exploratory, already examined data; one model, one layer, one SAE, RTN weights only; the loss on the language's own text, not task accuracy.
- The group is the frozen full-precision K64 group (small for most languages); the controls are activity-matched but not identical in function; alpha* is read on a coarse grid.
- Counts are the decision rules; the cell tables are descriptive and uncorrected for multiple comparisons.
