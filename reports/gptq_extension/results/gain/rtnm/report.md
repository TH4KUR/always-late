# Study G: graded gain of language-selective features

Exploratory (the FLORES devtest sentences were examined in earlier studies, D-039); the rules below were fixed in the design before any number (`docs/superpowers/specs/2026-10-24-graded-gain-study-design.md` sections 5 and 9). Harm = NLL after the edit minus NLL before it in the same state (nats per token on the language's own text); net harm = group minus the mean of the matched controls scaled by the same ratio; intervals are paired bootstraps over sentences.

Down ratios 0.3536, 0.1768, 0.0884 (plus zero-ablation); up ratios 2.8284, 5.6569, 11.3137; fine up ratios 1.10, 1.20; 3 control draws; states FP, INT6, INT4; 27 languages.

## Verdicts

| hypothesis | rule (counts out of the languages) | result | verdict |
|---|---|---|---|
| H-G1 dose, down (FP) | net harm strictly ordered over the three down ratios and zero-ablation, in at least 24, and the median ordered | 27 of 27; median net 0.0223, 0.0902, 0.1687, 0.2624 | **met** |
| H-G2 down is specific (FP) | lower bound of net harm above 0 at ratio 0.0884, in at least 24 | 26 of 27 | **met** |
| H-G3 dose, up (FP) | group harm increasing over the up ratios in at least 24 and net harm above 0 at the largest in at least 20 | 27 and 24 of 27 | **met** |
| H-G4 precision, down | tau > 0 at the deepest down ratio in at least 24, median not falling by more than one standard error | 25 of 27; median ok False | **not met** |
| H-G5a generic repair (INT4) | median control harm below 0 at 1.1 or 1.2, interval below 0 | x1.1: median control harm 0.0002 [-0.0003, 0.0004]; x1.2: median control harm 0.0006 [-0.0000, 0.0008] | **not met** |
| H-G5b group-specific repair (INT4) | upper bound of net harm below 0 at the better fine ratio in at least 14 | 10 of 27 | **not met** |
| H-G5 alpha* against 1 / g_ratio (INT4) | Spearman above 0.3 | -0.036 | **not met** |
| H-G6 consistency | net harm positive at ratio 0.0884 in at least 24 in every state and Spearman FP against INT4 above 0.5 | FP 27, INT6 27, INT4 27; rank correlation 0.972 | **met** |

H-G4, detail: median group harm at the deepest down ratio by state: FP 0.1939, INT6 0.1815, INT4 0.3196. The pre-registered clause allows a fall of the median of at most one bootstrap error (the larger of the two states'); result False. **Labelled post hoc sensitivity, not part of the verdict:** with a tolerance of two bootstrap errors the clause gives True (the pre-registered rule is not changed).

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
| 0.0000 | 0.2797 | 0.0251 | 0.2546 | 27 of 27 | 0.2997 |
| 0.0884 | 0.1815 | 0.0199 | 0.1696 | 27 of 27 | 0.2033 |
| 0.1768 | 0.1046 | 0.0154 | 0.0966 | 26 of 27 | 0.1237 |
| 0.3536 | 0.0325 | 0.0062 | 0.0239 | 25 of 27 | 0.0526 |
| 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0 of 27 | 0.0168 |
| 1.1000 | 0.0006 | -0.0001 | 0.0004 | 18 of 27 | 0.0187 |
| 1.2000 | 0.0022 | 0.0002 | 0.0013 | 22 of 27 | 0.0206 |
| 2.8284 | 0.0874 | 0.0360 | 0.0517 | 25 of 27 | 0.1052 |
| 5.6569 | 0.5834 | 0.2379 | 0.3423 | 25 of 27 | 0.6017 |
| 11.3137 | 2.3197 | 1.2767 | 1.1670 | 24 of 27 | 2.3562 |

### INT4

| ratio | group harm | control harm | net harm | languages with net > 0 | total change vs unedited FP |
|---|---|---|---|---|---|
| 0.0000 | 0.4526 | 0.0298 | 0.3946 | 27 of 27 | 0.8449 |
| 0.0884 | 0.3196 | 0.0244 | 0.2814 | 27 of 27 | 0.6518 |
| 0.1768 | 0.2285 | 0.0189 | 0.1963 | 27 of 27 | 0.5149 |
| 0.3536 | 0.0739 | 0.0107 | 0.0647 | 27 of 27 | 0.4195 |
| 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0 of 27 | 0.3570 |
| 1.1000 | 0.0003 | 0.0002 | -0.0000 | 13 of 27 | 0.3598 |
| 1.2000 | 0.0013 | 0.0006 | 0.0009 | 15 of 27 | 0.3594 |
| 2.8284 | 0.0907 | 0.0510 | 0.0552 | 23 of 27 | 0.4517 |
| 5.6569 | 0.6418 | 0.3139 | 0.3114 | 25 of 27 | 0.9713 |
| 11.3137 | 2.4881 | 1.4212 | 0.8538 | 23 of 27 | 2.8336 |

## By language: net harm at ratio 0.0884 and 11.3137

| language | FP down | INT6 down | INT4 down | FP up | INT6 up | INT4 up |
|---|---|---|---|---|---|---|
| eng_Latn | 0.1067 | 0.1050 | 0.1187 | -0.2195 | -0.2135 | -0.2530 |
| deu_Latn | 0.4222 | 0.4636 | 0.4904 | 1.5292 | 1.4510 | 1.4664 |
| spa_Latn | 0.2160 | 0.2358 | 0.5616 | 0.7209 | 0.7584 | 0.5555 |
| fra_Latn | 0.3067 | 0.3410 | 0.3931 | 0.8444 | 0.8438 | 0.5823 |
| por_Latn | 0.5633 | 0.6488 | 0.4046 | 0.8934 | 0.8795 | 0.8531 |
| rus_Cyrl | 0.0320 | 0.0400 | 0.1046 | 1.2348 | 1.1956 | 0.8538 |
| zho_Hans | 5.4597 | 5.5732 | 8.5469 | 1.6129 | 1.5172 | 1.7635 |
| jpn_Jpan | 0.1569 | 0.1519 | 0.3313 | 0.9992 | 0.9446 | 0.7372 |
| kor_Hang | 0.1926 | 0.1696 | 0.3471 | 0.8366 | 0.8212 | 0.5729 |
| hin_Deva | 0.0516 | 0.0680 | 0.1307 | 1.5405 | 1.4369 | 1.7067 |
| ben_Beng | 0.0076 | 0.0071 | 0.0219 | 0.6351 | 0.6084 | 0.7125 |
| tam_Taml | 0.0217 | 0.0233 | 0.0496 | 2.0507 | 2.2032 | 1.7481 |
| tur_Latn | 0.1260 | 0.1305 | 0.2814 | 0.5040 | 0.5612 | -0.0829 |
| vie_Latn | 0.0007 | 0.0050 | 0.0305 | 0.7718 | 0.7992 | 0.5483 |
| ind_Latn | 0.0182 | 0.0176 | 0.0360 | 0.2211 | 0.2178 | 0.3111 |
| tha_Thai | 0.0311 | 0.0377 | 0.1067 | 0.0694 | -0.0103 | -0.0530 |
| pol_Latn | 0.1793 | 0.1934 | 0.2247 | 1.3897 | 1.4065 | 1.1944 |
| ukr_Cyrl | 0.2045 | 0.2233 | 0.4735 | 2.5303 | 2.3900 | 1.7992 |
| kan_Knda | 0.0261 | 0.0249 | 0.0328 | 1.2128 | 1.1670 | 1.8600 |
| tel_Telu | 0.0199 | 0.0247 | 0.0972 | 1.2267 | 1.2359 | 1.3341 |
| yor_Latn | 1.1457 | 1.1700 | 1.3895 | -0.0152 | -0.2169 | -0.1313 |
| sun_Latn | 0.6771 | 0.6148 | 0.5659 | 0.3112 | 0.2943 | 0.3390 |
| amh_Ethi | 0.7678 | 0.7977 | 1.0205 | 6.1792 | 6.4224 | 6.1484 |
| ibo_Latn | 0.2782 | 0.2725 | 0.3323 | 1.7387 | 1.7325 | 1.3715 |
| hau_Latn | 0.1289 | 0.1360 | 0.1617 | 1.3263 | 1.2455 | 1.2137 |
| khm_Khmr | 0.1687 | 0.1724 | 0.2380 | 5.4296 | 5.0945 | 5.3130 |
| lao_Laoo | 1.1689 | 1.2423 | 1.8408 | 7.1586 | 6.9560 | 6.0220 |

## Optimal gain alpha* (the ratio with the lowest loss; 1 = leave the group alone)

| state | languages with alpha* above 1 | languages with alpha* below 1 | median alpha* |
|---|---|---|---|
| FP | 7 of 27 | 2 of 27 | 1.000 |
| INT6 | 9 of 27 | 1 of 27 | 1.000 |
| INT4 | 13 of 27 | 0 of 27 | 1.000 |

## Reliability of the sample

The sample is 300 held-out sentences per language, 3 matched-control draws per language and 27 languages in 3 states; it was not reduced to save time (user instruction 2026-10-24). The intervals of the decision rules come from the bootstrap over sentences and are conditional on the control draws; this table adds the second source of uncertainty (the variation between control draws). Per state and ratio (medians over the languages): |net harm|, the standard error from the sentences, the standard error from the control draws, and the number of languages whose |net harm| exceeds twice the sentence error alone / twice both errors combined.

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
| INT6 | 0 | 0.2546 | 0.0116 | 0.0024 | 27 of 27 | 27 of 27 |
| INT6 | 0.0884 | 0.1696 | 0.0089 | 0.0021 | 26 of 27 | 26 of 27 |
| INT6 | 0.177 | 0.0966 | 0.0067 | 0.0017 | 26 of 27 | 26 of 27 |
| INT6 | 0.354 | 0.0239 | 0.0038 | 0.0012 | 23 of 27 | 23 of 27 |
| INT6 | 1.1 | 0.0011 | 0.0005 | 0.0002 | 14 of 27 | 13 of 27 |
| INT6 | 1.2 | 0.0021 | 0.0009 | 0.0003 | 17 of 27 | 16 of 27 |
| INT6 | 2.83 | 0.0517 | 0.0048 | 0.0042 | 27 of 27 | 24 of 27 |
| INT6 | 5.66 | 0.3423 | 0.0165 | 0.0372 | 26 of 27 | 22 of 27 |
| INT6 | 11.3 | 1.1670 | 0.0406 | 0.1694 | 26 of 27 | 24 of 27 |
| INT4 | 0 | 0.3946 | 0.0149 | 0.0037 | 27 of 27 | 27 of 27 |
| INT4 | 0.0884 | 0.2814 | 0.0117 | 0.0028 | 27 of 27 | 27 of 27 |
| INT4 | 0.177 | 0.1963 | 0.0092 | 0.0028 | 27 of 27 | 27 of 27 |
| INT4 | 0.354 | 0.0647 | 0.0048 | 0.0023 | 26 of 27 | 25 of 27 |
| INT4 | 1.1 | 0.0014 | 0.0006 | 0.0004 | 19 of 27 | 18 of 27 |
| INT4 | 1.2 | 0.0020 | 0.0009 | 0.0005 | 17 of 27 | 16 of 27 |
| INT4 | 2.83 | 0.0552 | 0.0063 | 0.0077 | 27 of 27 | 24 of 27 |
| INT4 | 5.66 | 0.3114 | 0.0178 | 0.0421 | 27 of 27 | 24 of 27 |
| INT4 | 11.3 | 0.8538 | 0.0443 | 0.1811 | 25 of 27 | 23 of 27 |

## Limits

- Exploratory, already examined data; one model, one layer, one SAE, RTN weights only; the loss on the language's own text, not task accuracy.
- The group is the frozen full-precision K64 group (small for most languages); the controls are activity-matched but not identical in function; alpha* is read on a coarse grid.
- Counts are the decision rules; the cell tables are descriptive and uncorrected for multiple comparisons.
