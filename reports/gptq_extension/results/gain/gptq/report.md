# Study G: graded gain of language-selective features

Exploratory (the FLORES devtest sentences were examined in earlier studies, D-039); the rules below were fixed in the design before any number (`docs/superpowers/specs/2026-10-24-graded-gain-study-design.md` sections 5 and 9). Harm = NLL after the edit minus NLL before it in the same state (nats per token on the language's own text); net harm = group minus the mean of the matched controls scaled by the same ratio; intervals are paired bootstraps over sentences.

Down ratios 0.3536, 0.1768, 0.0884 (plus zero-ablation); up ratios 2.8284, 5.6569, 11.3137; fine up ratios 1.10, 1.20; 3 control draws; states FP, INT6, INT4; 27 languages.

## Verdicts

| hypothesis | rule (counts out of the languages) | result | verdict |
|---|---|---|---|
| H-G1 dose, down (FP) | net harm strictly ordered over the three down ratios and zero-ablation, in at least 24, and the median ordered | 27 of 27; median net 0.0223, 0.0902, 0.1687, 0.2624 | **met** |
| H-G2 down is specific (FP) | lower bound of net harm above 0 at ratio 0.0884, in at least 24 | 26 of 27 | **met** |
| H-G3 dose, up (FP) | group harm increasing over the up ratios in at least 24 and net harm above 0 at the largest in at least 20 | 27 and 24 of 27 | **met** |
| H-G4 precision, down | tau > 0 at the deepest down ratio in at least 24, median not falling by more than one standard error | 24 of 27; median ok True | **met** |
| H-G5a generic repair (INT4) | median control harm below 0 at 1.1 or 1.2, interval below 0 | x1.1: median control harm 0.0001 [-0.0003, 0.0003]; x1.2: median control harm 0.0002 [-0.0001, 0.0008] | **not met** |
| H-G5b group-specific repair (INT4) | upper bound of net harm below 0 at the better fine ratio in at least 14 | 4 of 27 | **not met** |
| H-G5 alpha* against 1 / g_ratio (INT4) | Spearman above 0.3 | -0.268 | **not met** |
| H-G6 consistency | net harm positive at ratio 0.0884 in at least 24 in every state and Spearman FP against INT4 above 0.5 | FP 27, INT6 27, INT4 27; rank correlation 0.985 | **met** |

H-G4, detail: median group harm at the deepest down ratio by state: FP 0.1939, INT6 0.1897, INT4 0.2171. The pre-registered clause allows a fall of the median of at most one bootstrap error (the larger of the two states'); result True. **Labelled post hoc sensitivity, not part of the verdict:** with a tolerance of two bootstrap errors the clause gives True (the pre-registered rule is not changed).

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
| 0.0000 | 0.2878 | 0.0243 | 0.2635 | 27 of 27 | 0.2942 |
| 0.0884 | 0.1897 | 0.0200 | 0.1659 | 27 of 27 | 0.1948 |
| 0.1768 | 0.1021 | 0.0134 | 0.0904 | 26 of 27 | 0.1087 |
| 0.3536 | 0.0322 | 0.0060 | 0.0224 | 25 of 27 | 0.0374 |
| 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0 of 27 | 0.0062 |
| 1.1000 | 0.0006 | 0.0001 | 0.0006 | 19 of 27 | 0.0065 |
| 1.2000 | 0.0021 | 0.0004 | 0.0017 | 23 of 27 | 0.0076 |
| 2.8284 | 0.0855 | 0.0353 | 0.0523 | 25 of 27 | 0.0918 |
| 5.6569 | 0.5901 | 0.2409 | 0.3403 | 25 of 27 | 0.5967 |
| 11.3137 | 2.3422 | 1.2819 | 1.1954 | 24 of 27 | 2.3517 |

### INT4

| ratio | group harm | control harm | net harm | languages with net > 0 | total change vs unedited FP |
|---|---|---|---|---|---|
| 0.0000 | 0.3613 | 0.0268 | 0.3345 | 27 of 27 | 0.5005 |
| 0.0884 | 0.2171 | 0.0218 | 0.1867 | 27 of 27 | 0.3855 |
| 0.1768 | 0.1507 | 0.0168 | 0.1235 | 27 of 27 | 0.2911 |
| 0.3536 | 0.0485 | 0.0080 | 0.0373 | 26 of 27 | 0.1823 |
| 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0 of 27 | 0.1258 |
| 1.1000 | 0.0006 | 0.0001 | 0.0008 | 20 of 27 | 0.1285 |
| 1.2000 | 0.0017 | 0.0002 | 0.0019 | 20 of 27 | 0.1312 |
| 2.8284 | 0.1047 | 0.0435 | 0.0669 | 25 of 27 | 0.2526 |
| 5.6569 | 0.6442 | 0.2758 | 0.4052 | 26 of 27 | 0.7607 |
| 11.3137 | 2.5873 | 1.3583 | 1.1456 | 26 of 27 | 2.6950 |

## By language: net harm at ratio 0.0884 and 11.3137

| language | FP down | INT6 down | INT4 down | FP up | INT6 up | INT4 up |
|---|---|---|---|---|---|---|
| eng_Latn | 0.1067 | 0.1051 | 0.0987 | -0.2195 | -0.2322 | -0.2179 |
| deu_Latn | 0.4222 | 0.3915 | 0.4806 | 1.5292 | 1.4645 | 1.5651 |
| spa_Latn | 0.2160 | 0.1965 | 0.3058 | 0.7209 | 0.7249 | 0.6743 |
| fra_Latn | 0.3067 | 0.3064 | 0.3330 | 0.8444 | 0.8435 | 0.8659 |
| por_Latn | 0.5633 | 0.5450 | 0.5548 | 0.8934 | 0.9029 | 0.9930 |
| rus_Cyrl | 0.0320 | 0.0298 | 0.0576 | 1.2348 | 1.1954 | 1.1456 |
| zho_Hans | 5.4597 | 5.3067 | 6.2081 | 1.6129 | 1.5285 | 1.5801 |
| jpn_Jpan | 0.1569 | 0.1586 | 0.2071 | 0.9992 | 0.9653 | 0.9806 |
| kor_Hang | 0.1926 | 0.1858 | 0.3926 | 0.8366 | 0.8483 | 0.8313 |
| hin_Deva | 0.0516 | 0.0514 | 0.0887 | 1.5405 | 1.5542 | 1.6259 |
| ben_Beng | 0.0076 | 0.0079 | 0.0115 | 0.6351 | 0.6202 | 0.6663 |
| tam_Taml | 0.0217 | 0.0233 | 0.0274 | 2.0507 | 2.1488 | 2.4090 |
| tur_Latn | 0.1260 | 0.1234 | 0.1756 | 0.5040 | 0.4729 | 0.2758 |
| vie_Latn | 0.0007 | 0.0009 | 0.0203 | 0.7718 | 0.7941 | 0.7552 |
| ind_Latn | 0.0182 | 0.0154 | 0.0272 | 0.2211 | 0.2143 | 0.2713 |
| tha_Thai | 0.0311 | 0.0310 | 0.0522 | 0.0694 | -0.0213 | 0.0409 |
| pol_Latn | 0.1793 | 0.1784 | 0.1867 | 1.3897 | 1.3976 | 1.3681 |
| ukr_Cyrl | 0.2045 | 0.2075 | 0.3501 | 2.5303 | 2.4633 | 2.3586 |
| kan_Knda | 0.0261 | 0.0263 | 0.0275 | 1.2128 | 1.2205 | 1.5046 |
| tel_Telu | 0.0199 | 0.0207 | 0.0389 | 1.2267 | 1.2674 | 1.3119 |
| yor_Latn | 1.1457 | 1.1410 | 1.3033 | -0.0152 | -0.0272 | 0.0757 |
| sun_Latn | 0.6771 | 0.6814 | 0.6083 | 0.3112 | 0.2863 | 0.3720 |
| amh_Ethi | 0.7678 | 0.7697 | 0.9085 | 6.1792 | 6.3135 | 5.9725 |
| ibo_Latn | 0.2782 | 0.2769 | 0.2863 | 1.7387 | 1.7862 | 2.0982 |
| hau_Latn | 0.1289 | 0.1302 | 0.1389 | 1.3263 | 1.3294 | 1.4084 |
| khm_Khmr | 0.1687 | 0.1659 | 0.1812 | 5.4296 | 5.4739 | 5.6353 |
| lao_Laoo | 1.1689 | 1.1574 | 1.4487 | 7.1586 | 7.1433 | 7.3343 |

## Optimal gain alpha* (the ratio with the lowest loss; 1 = leave the group alone)

| state | languages with alpha* above 1 | languages with alpha* below 1 | median alpha* |
|---|---|---|---|
| FP | 7 of 27 | 2 of 27 | 1.000 |
| INT6 | 7 of 27 | 2 of 27 | 1.000 |
| INT4 | 7 of 27 | 0 of 27 | 1.000 |

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
| INT6 | 0 | 0.2635 | 0.0115 | 0.0024 | 27 of 27 | 27 of 27 |
| INT6 | 0.0884 | 0.1659 | 0.0086 | 0.0021 | 26 of 27 | 26 of 27 |
| INT6 | 0.177 | 0.0904 | 0.0067 | 0.0015 | 26 of 27 | 26 of 27 |
| INT6 | 0.354 | 0.0224 | 0.0038 | 0.0010 | 24 of 27 | 23 of 27 |
| INT6 | 1.1 | 0.0014 | 0.0005 | 0.0002 | 17 of 27 | 15 of 27 |
| INT6 | 1.2 | 0.0025 | 0.0008 | 0.0002 | 19 of 27 | 19 of 27 |
| INT6 | 2.83 | 0.0523 | 0.0048 | 0.0044 | 27 of 27 | 25 of 27 |
| INT6 | 5.66 | 0.3403 | 0.0168 | 0.0376 | 26 of 27 | 22 of 27 |
| INT6 | 11.3 | 1.1954 | 0.0417 | 0.1592 | 25 of 27 | 23 of 27 |
| INT4 | 0 | 0.3345 | 0.0119 | 0.0027 | 27 of 27 | 27 of 27 |
| INT4 | 0.0884 | 0.1867 | 0.0100 | 0.0022 | 27 of 27 | 27 of 27 |
| INT4 | 0.177 | 0.1235 | 0.0075 | 0.0019 | 27 of 27 | 26 of 27 |
| INT4 | 0.354 | 0.0373 | 0.0045 | 0.0011 | 23 of 27 | 23 of 27 |
| INT4 | 1.1 | 0.0012 | 0.0006 | 0.0002 | 15 of 27 | 13 of 27 |
| INT4 | 1.2 | 0.0026 | 0.0009 | 0.0003 | 20 of 27 | 19 of 27 |
| INT4 | 2.83 | 0.0669 | 0.0055 | 0.0056 | 25 of 27 | 23 of 27 |
| INT4 | 5.66 | 0.4052 | 0.0168 | 0.0427 | 26 of 27 | 22 of 27 |
| INT4 | 11.3 | 1.1456 | 0.0453 | 0.1575 | 25 of 27 | 23 of 27 |

## Limits

- Exploratory, already examined data; one model, one layer, one SAE, RTN weights only; the loss on the language's own text, not task accuracy.
- The group is the frozen full-precision K64 group (small for most languages); the controls are activity-matched but not identical in function; alpha* is read on a coarse grid.
- Counts are the decision rules; the cell tables are descriptive and uncorrected for multiple comparisons.
