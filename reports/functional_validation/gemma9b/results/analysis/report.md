# Study G: graded gain of language-selective features

Exploratory (the FLORES devtest sentences were examined in earlier studies, D-039); the rules below were fixed in the design before any number (`docs/superpowers/specs/2026-10-24-graded-gain-study-design.md` sections 5 and 9). Harm = NLL after the edit minus NLL before it in the same state (nats per token on the language's own text); net harm = group minus the mean of the matched controls scaled by the same ratio; intervals are paired bootstraps over sentences.

Down ratios 0.3536, 0.1768, 0.0884 (plus zero-ablation); up ratios 2.8284, 5.6569, 11.3137; fine up ratios 1.10, 1.20; 3 control draws; states FP, INT8, INT7, INT6, INT5, INT4; 27 languages.

## Verdicts

| hypothesis | rule (counts out of the languages) | result | verdict |
|---|---|---|---|
| H-G1 dose, down (FP) | net harm strictly ordered over the three down ratios and zero-ablation, in at least 24, and the median ordered | 27 of 27; median net 0.0167, 0.0982, 0.1634, 0.2696 | **met** |
| H-G2 down is specific (FP) | lower bound of net harm above 0 at ratio 0.0884, in at least 24 | 27 of 27 | **met** |
| H-G3 dose, up (FP) | group harm increasing over the up ratios in at least 24 and net harm above 0 at the largest in at least 20 | 27 and 23 of 27 | **met** |
| H-G4 precision, down | tau > 0 at the deepest down ratio in at least 24, median not falling by more than one standard error | 25 of 27; median ok True | **met** |
| H-G5a generic repair (INT4) | median control harm below 0 at 1.1 or 1.2, interval below 0 | x1.1: median control harm 0.0000 [-0.0002, 0.0003]; x1.2: median control harm 0.0003 [-0.0000, 0.0006] | **not met** |
| H-G5b group-specific repair (INT4) | upper bound of net harm below 0 at the better fine ratio in at least 14 | 8 of 27 | **not met** |
| H-G5 alpha* against 1 / g_ratio (INT4) | Spearman above 0.3 | - | not testable |
| H-G6 consistency | net harm positive at ratio 0.0884 in at least 24 in every state and Spearman FP against INT4 above 0.5 | FP 27, INT8 27, INT7 27, INT6 27, INT5 27, INT4 27; rank correlation 0.904 | **met** |

H-G4, detail: median group harm at the deepest down ratio by state: FP 0.1796, INT8 0.1819, INT7 0.1924, INT6 0.1874, INT5 0.2023, INT4 0.2960. The pre-registered clause allows a fall of the median of at most one bootstrap error (the larger of the two states'); result True. **Labelled post hoc sensitivity, not part of the verdict:** with a tolerance of two bootstrap errors the clause gives True (the pre-registered rule is not changed).

## Median over the languages by state and ratio (nats per token)

### FP

| ratio | group harm | control harm | net harm | languages with net > 0 | total change vs unedited FP |
|---|---|---|---|---|---|
| 0.0000 | 0.3037 | 0.0080 | 0.2696 | 27 of 27 | 0.3037 |
| 0.0884 | 0.1796 | 0.0062 | 0.1634 | 27 of 27 | 0.1796 |
| 0.1768 | 0.1241 | 0.0047 | 0.0982 | 25 of 27 | 0.1241 |
| 0.3536 | 0.0197 | 0.0020 | 0.0167 | 17 of 27 | 0.0197 |
| 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0 of 27 | 0.0000 |
| 1.1000 | 0.0010 | 0.0001 | 0.0009 | 23 of 27 | 0.0010 |
| 1.2000 | 0.0022 | 0.0003 | 0.0019 | 24 of 27 | 0.0022 |
| 2.8284 | 0.0478 | 0.0136 | 0.0371 | 26 of 27 | 0.0478 |
| 5.6569 | 0.3711 | 0.1102 | 0.2862 | 25 of 27 | 0.3711 |
| 11.3137 | 1.6561 | 0.8040 | 0.9379 | 23 of 27 | 1.6561 |

### INT8

| ratio | group harm | control harm | net harm | languages with net > 0 | total change vs unedited FP |
|---|---|---|---|---|---|
| 0.0000 | 0.3048 | 0.0081 | 0.2753 | 27 of 27 | 0.3070 |
| 0.0884 | 0.1819 | 0.0064 | 0.1656 | 27 of 27 | 0.1827 |
| 0.1768 | 0.1258 | 0.0047 | 0.0997 | 25 of 27 | 0.1266 |
| 0.3536 | 0.0198 | 0.0025 | 0.0171 | 17 of 27 | 0.0199 |
| 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0 of 27 | 0.0008 |
| 1.1000 | 0.0011 | 0.0001 | 0.0009 | 24 of 27 | 0.0019 |
| 1.2000 | 0.0023 | 0.0004 | 0.0022 | 23 of 27 | 0.0033 |
| 2.8284 | 0.0483 | 0.0132 | 0.0371 | 26 of 27 | 0.0479 |
| 5.6569 | 0.3713 | 0.1102 | 0.2849 | 25 of 27 | 0.3718 |
| 11.3137 | 1.6615 | 0.8057 | 0.9338 | 23 of 27 | 1.6623 |

### INT7

| ratio | group harm | control harm | net harm | languages with net > 0 | total change vs unedited FP |
|---|---|---|---|---|---|
| 0.0000 | 0.2965 | 0.0087 | 0.2769 | 27 of 27 | 0.3013 |
| 0.0884 | 0.1924 | 0.0068 | 0.1680 | 27 of 27 | 0.1972 |
| 0.1768 | 0.1225 | 0.0047 | 0.0981 | 25 of 27 | 0.1284 |
| 0.3536 | 0.0196 | 0.0022 | 0.0169 | 17 of 27 | 0.0235 |
| 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0 of 27 | 0.0049 |
| 1.1000 | 0.0012 | 0.0003 | 0.0012 | 23 of 27 | 0.0059 |
| 1.2000 | 0.0026 | 0.0005 | 0.0023 | 24 of 27 | 0.0075 |
| 2.8284 | 0.0477 | 0.0135 | 0.0367 | 26 of 27 | 0.0519 |
| 5.6569 | 0.3808 | 0.1112 | 0.2871 | 25 of 27 | 0.3851 |
| 11.3137 | 1.6818 | 0.8047 | 0.9427 | 23 of 27 | 1.6861 |

### INT6

| ratio | group harm | control harm | net harm | languages with net > 0 | total change vs unedited FP |
|---|---|---|---|---|---|
| 0.0000 | 0.3045 | 0.0089 | 0.2928 | 27 of 27 | 0.3130 |
| 0.0884 | 0.1874 | 0.0073 | 0.1639 | 27 of 27 | 0.2029 |
| 0.1768 | 0.1253 | 0.0055 | 0.0993 | 25 of 27 | 0.1337 |
| 0.3536 | 0.0197 | 0.0028 | 0.0158 | 17 of 27 | 0.0377 |
| 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0 of 27 | 0.0146 |
| 1.1000 | 0.0011 | 0.0001 | 0.0010 | 22 of 27 | 0.0161 |
| 1.2000 | 0.0026 | 0.0002 | 0.0020 | 24 of 27 | 0.0167 |
| 2.8284 | 0.0477 | 0.0128 | 0.0359 | 26 of 27 | 0.0624 |
| 5.6569 | 0.3713 | 0.1131 | 0.2867 | 25 of 27 | 0.3832 |
| 11.3137 | 1.6096 | 0.8268 | 0.9008 | 23 of 27 | 1.6233 |

### INT5

| ratio | group harm | control harm | net harm | languages with net > 0 | total change vs unedited FP |
|---|---|---|---|---|---|
| 0.0000 | 0.3133 | 0.0092 | 0.3012 | 27 of 27 | 0.4214 |
| 0.0884 | 0.2023 | 0.0072 | 0.1735 | 27 of 27 | 0.2655 |
| 0.1768 | 0.1231 | 0.0053 | 0.1094 | 24 of 27 | 0.1754 |
| 0.3536 | 0.0205 | 0.0024 | 0.0201 | 19 of 27 | 0.0837 |
| 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0 of 27 | 0.0668 |
| 1.1000 | 0.0014 | 0.0002 | 0.0013 | 21 of 27 | 0.0683 |
| 1.2000 | 0.0029 | 0.0003 | 0.0024 | 23 of 27 | 0.0701 |
| 2.8284 | 0.0534 | 0.0145 | 0.0416 | 26 of 27 | 0.1234 |
| 5.6569 | 0.3975 | 0.1105 | 0.3052 | 25 of 27 | 0.5044 |
| 11.3137 | 1.6638 | 0.8305 | 0.9984 | 23 of 27 | 1.7263 |

### INT4

| ratio | group harm | control harm | net harm | languages with net > 0 | total change vs unedited FP |
|---|---|---|---|---|---|
| 0.0000 | 0.4821 | 0.0143 | 0.4764 | 27 of 27 | 0.9704 |
| 0.0884 | 0.2960 | 0.0115 | 0.2651 | 27 of 27 | 0.7706 |
| 0.1768 | 0.2160 | 0.0088 | 0.2009 | 27 of 27 | 0.6161 |
| 0.3536 | 0.0576 | 0.0043 | 0.0568 | 27 of 27 | 0.4531 |
| 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0 of 27 | 0.3866 |
| 1.1000 | -0.0001 | 0.0000 | -0.0005 | 10 of 27 | 0.3862 |
| 1.2000 | 0.0005 | 0.0003 | -0.0004 | 12 of 27 | 0.3868 |
| 2.8284 | 0.0595 | 0.0204 | 0.0384 | 24 of 27 | 0.4389 |
| 5.6569 | 0.4551 | 0.1502 | 0.3251 | 25 of 27 | 0.8078 |
| 11.3137 | 1.9354 | 0.9537 | 0.9912 | 25 of 27 | 2.2396 |

## By language: net harm at ratio 0.0884 and 11.3137

| language | FP down | INT8 down | INT7 down | INT6 down | INT5 down | INT4 down | FP up | INT8 up | INT7 up | INT6 up | INT5 up | INT4 up |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| eng_Latn | 0.1634 | 0.1636 | 0.1539 | 0.1639 | 0.1735 | 0.1971 | 0.0918 | 0.0869 | 0.0928 | 0.0776 | 0.0529 | 0.0618 |
| deu_Latn | 0.1560 | 0.1573 | 0.1606 | 0.1602 | 0.1886 | 0.2107 | 1.0519 | 1.0568 | 1.0718 | 1.0259 | 1.1275 | 1.4991 |
| spa_Latn | 0.0548 | 0.0562 | 0.0575 | 0.0586 | 0.0566 | 0.0908 | -0.6066 | -0.6131 | -0.5938 | -0.7401 | -0.8392 | 0.4752 |
| fra_Latn | 0.2840 | 0.2877 | 0.2840 | 0.2702 | 0.4462 | 0.3768 | 1.2409 | 1.2475 | 1.2676 | 1.2147 | 1.2769 | 1.3753 |
| por_Latn | 0.1795 | 0.1736 | 0.1988 | 0.1572 | 0.2294 | 0.1426 | 0.9480 | 0.9599 | 0.9736 | 0.9008 | 0.9984 | 1.2900 |
| rus_Cyrl | 0.0181 | 0.0160 | 0.0212 | 0.0122 | 0.0368 | 0.1196 | -0.0693 | -0.0568 | -0.0573 | -0.1154 | -0.0959 | 0.2257 |
| zho_Hans | 4.6422 | 4.6364 | 4.5839 | 4.8646 | 5.2365 | 5.5964 | 0.8898 | 0.8855 | 0.8538 | 0.8978 | 0.8713 | 1.3478 |
| jpn_Jpan | 0.6846 | 0.6743 | 0.6859 | 0.7023 | 0.6912 | 0.8174 | 1.1673 | 1.1638 | 1.1929 | 1.1744 | 1.1712 | 1.2294 |
| kor_Hang | 0.0191 | 0.0170 | 0.0264 | 0.0160 | 0.0153 | 0.2149 | 0.8109 | 0.8202 | 0.8253 | 0.7973 | 0.8023 | 0.7263 |
| hin_Deva | 0.0209 | 0.0204 | 0.0229 | 0.0261 | 0.0157 | 0.0614 | 0.4949 | 0.4973 | 0.5015 | 0.4673 | 0.5507 | 0.3852 |
| ben_Beng | 0.0202 | 0.0210 | 0.0196 | 0.0259 | 0.0357 | 0.1729 | 1.6304 | 1.6489 | 1.6462 | 1.5926 | 1.7934 | 1.4744 |
| tam_Taml | 1.1769 | 1.2326 | 1.2991 | 1.1963 | 0.8724 | 2.4037 | 1.1168 | 1.1234 | 1.1248 | 1.1135 | 1.1228 | 1.0946 |
| tur_Latn | 0.2760 | 0.2781 | 0.2651 | 0.3057 | 0.3539 | 0.3556 | 1.7911 | 1.7737 | 1.8268 | 1.7878 | 1.7541 | 1.8217 |
| vie_Latn | 0.0122 | 0.0128 | 0.0149 | 0.0103 | 0.0168 | 0.0801 | 1.1454 | 1.1484 | 1.1698 | 1.0643 | 1.0890 | 1.3959 |
| ind_Latn | 0.0152 | 0.0145 | 0.0140 | 0.0124 | 0.0188 | 0.0394 | 0.3188 | 0.3153 | 0.3204 | 0.3300 | 0.3347 | 0.3588 |
| tha_Thai | 0.3843 | 0.3871 | 0.4061 | 0.3824 | 0.4600 | 0.9526 | 1.3368 | 1.3506 | 1.3354 | 1.3093 | 1.3222 | 0.9912 |
| pol_Latn | 0.2023 | 0.1901 | 0.1950 | 0.2186 | 0.3166 | 1.0254 | 1.8437 | 1.8522 | 1.8452 | 1.8569 | 1.9484 | 2.1178 |
| ukr_Cyrl | 0.0042 | 0.0048 | 0.0048 | 0.0042 | 0.0056 | 0.0248 | 0.7374 | 0.7451 | 0.7346 | 0.7692 | 0.7530 | 0.6250 |
| kan_Knda | 0.0036 | 0.0029 | 0.0047 | 0.0031 | 0.0090 | 0.0606 | 0.3883 | 0.3973 | 0.3911 | 0.3859 | 0.3971 | 0.5402 |
| tel_Telu | 0.1629 | 0.1656 | 0.1778 | 0.1642 | 0.1660 | 0.4107 | 0.8391 | 0.8377 | 0.8324 | 0.8238 | 0.8642 | 0.7923 |
| yor_Latn | 2.2217 | 2.2455 | 2.1942 | 2.2065 | 2.2976 | 3.2698 | -2.8124 | -2.8711 | -2.7538 | -2.7790 | -2.8064 | -3.0265 |
| sun_Latn | 1.0123 | 1.0042 | 1.0012 | 0.9223 | 1.0410 | 1.3960 | -0.4795 | -0.5099 | -0.4763 | -0.5112 | -0.4635 | -0.4093 |
| amh_Ethi | 0.5649 | 0.5658 | 0.5569 | 0.5893 | 0.6263 | 0.7922 | 0.5686 | 0.5719 | 0.5726 | 0.5683 | 0.6793 | 0.8105 |
| ibo_Latn | 0.1569 | 0.1587 | 0.1566 | 0.1591 | 0.1507 | 0.2651 | 1.0863 | 1.0751 | 1.0894 | 1.0787 | 1.1341 | 1.3457 |
| hau_Latn | 0.1731 | 0.1746 | 0.1777 | 0.1818 | 0.1878 | 0.2700 | 0.9379 | 0.9338 | 0.9427 | 0.9510 | 1.0143 | 1.1057 |
| khm_Khmr | 0.1449 | 0.1471 | 0.1456 | 0.1500 | 0.1640 | 0.2445 | 1.9535 | 1.9565 | 1.9572 | 1.9033 | 2.2798 | 1.4807 |
| lao_Laoo | 0.1695 | 0.1722 | 0.1680 | 0.1992 | 0.1627 | 0.8581 | 2.9598 | 2.9332 | 3.0472 | 2.8655 | 3.3641 | 0.9467 |

## Optimal gain alpha* (the ratio with the lowest loss; 1 = leave the group alone)

| state | languages with alpha* above 1 | languages with alpha* below 1 | median alpha* |
|---|---|---|---|
| FP | 4 of 27 | 8 of 27 | 1.000 |
| INT8 | 4 of 27 | 8 of 27 | 1.000 |
| INT7 | 3 of 27 | 8 of 27 | 1.000 |
| INT6 | 4 of 27 | 9 of 27 | 1.000 |
| INT5 | 6 of 27 | 8 of 27 | 1.000 |
| INT4 | 14 of 27 | 0 of 27 | 1.100 |

## Reliability of the sample

The sample is 300 held-out sentences per language, 3 matched-control draws per language and 27 languages in 6 states; it was not reduced to save time (user instruction 2026-10-24). The intervals of the decision rules come from the bootstrap over sentences and are conditional on the control draws; this table adds the second source of uncertainty (the variation between control draws). Per state and ratio (medians over the languages): |net harm|, the standard error from the sentences, the standard error from the control draws, and the number of languages whose |net harm| exceeds twice the sentence error alone / twice both errors combined.

| state | ratio | median abs net harm | SE sentences | SE control draws | languages clear (sentences) | languages clear (both) |
|---|---|---|---|---|---|---|
| FP | 0 | 0.2696 | 0.0178 | 0.0026 | 27 of 27 | 27 of 27 |
| FP | 0.0884 | 0.1634 | 0.0097 | 0.0021 | 27 of 27 | 26 of 27 |
| FP | 0.177 | 0.0982 | 0.0057 | 0.0016 | 22 of 27 | 21 of 27 |
| FP | 0.354 | 0.0179 | 0.0029 | 0.0008 | 20 of 27 | 20 of 27 |
| FP | 1.1 | 0.0012 | 0.0004 | 0.0002 | 19 of 27 | 17 of 27 |
| FP | 1.2 | 0.0024 | 0.0006 | 0.0002 | 24 of 27 | 24 of 27 |
| FP | 2.83 | 0.0388 | 0.0035 | 0.0025 | 27 of 27 | 27 of 27 |
| FP | 5.66 | 0.2862 | 0.0130 | 0.0156 | 27 of 27 | 25 of 27 |
| FP | 11.3 | 0.9480 | 0.0358 | 0.1512 | 27 of 27 | 22 of 27 |
| INT8 | 0 | 0.2753 | 0.0179 | 0.0028 | 27 of 27 | 27 of 27 |
| INT8 | 0.0884 | 0.1656 | 0.0097 | 0.0024 | 26 of 27 | 26 of 27 |
| INT8 | 0.177 | 0.0997 | 0.0056 | 0.0017 | 22 of 27 | 22 of 27 |
| INT8 | 0.354 | 0.0172 | 0.0029 | 0.0008 | 20 of 27 | 20 of 27 |
| INT8 | 1.1 | 0.0012 | 0.0004 | 0.0002 | 20 of 27 | 19 of 27 |
| INT8 | 1.2 | 0.0024 | 0.0006 | 0.0002 | 24 of 27 | 23 of 27 |
| INT8 | 2.83 | 0.0389 | 0.0035 | 0.0027 | 27 of 27 | 27 of 27 |
| INT8 | 5.66 | 0.2849 | 0.0130 | 0.0158 | 27 of 27 | 25 of 27 |
| INT8 | 11.3 | 0.9599 | 0.0361 | 0.1525 | 26 of 27 | 21 of 27 |
| INT7 | 0 | 0.2769 | 0.0179 | 0.0026 | 27 of 27 | 27 of 27 |
| INT7 | 0.0884 | 0.1680 | 0.0096 | 0.0021 | 27 of 27 | 26 of 27 |
| INT7 | 0.177 | 0.0981 | 0.0058 | 0.0016 | 22 of 27 | 21 of 27 |
| INT7 | 0.354 | 0.0171 | 0.0031 | 0.0008 | 20 of 27 | 20 of 27 |
| INT7 | 1.1 | 0.0014 | 0.0004 | 0.0001 | 20 of 27 | 18 of 27 |
| INT7 | 1.2 | 0.0026 | 0.0006 | 0.0002 | 23 of 27 | 22 of 27 |
| INT7 | 2.83 | 0.0371 | 0.0034 | 0.0024 | 27 of 27 | 27 of 27 |
| INT7 | 5.66 | 0.2871 | 0.0130 | 0.0162 | 27 of 27 | 25 of 27 |
| INT7 | 11.3 | 0.9736 | 0.0363 | 0.1540 | 26 of 27 | 22 of 27 |
| INT6 | 0 | 0.2928 | 0.0187 | 0.0025 | 27 of 27 | 27 of 27 |
| INT6 | 0.0884 | 0.1639 | 0.0101 | 0.0020 | 26 of 27 | 26 of 27 |
| INT6 | 0.177 | 0.0993 | 0.0059 | 0.0015 | 23 of 27 | 22 of 27 |
| INT6 | 0.354 | 0.0185 | 0.0029 | 0.0008 | 20 of 27 | 19 of 27 |
| INT6 | 1.1 | 0.0013 | 0.0004 | 0.0002 | 21 of 27 | 19 of 27 |
| INT6 | 1.2 | 0.0028 | 0.0006 | 0.0002 | 23 of 27 | 23 of 27 |
| INT6 | 2.83 | 0.0392 | 0.0035 | 0.0027 | 27 of 27 | 27 of 27 |
| INT6 | 5.66 | 0.2867 | 0.0134 | 0.0155 | 27 of 27 | 25 of 27 |
| INT6 | 11.3 | 0.9510 | 0.0352 | 0.1541 | 27 of 27 | 21 of 27 |
| INT5 | 0 | 0.3012 | 0.0220 | 0.0027 | 27 of 27 | 27 of 27 |
| INT5 | 0.0884 | 0.1735 | 0.0091 | 0.0022 | 27 of 27 | 27 of 27 |
| INT5 | 0.177 | 0.1094 | 0.0058 | 0.0017 | 23 of 27 | 23 of 27 |
| INT5 | 0.354 | 0.0201 | 0.0027 | 0.0009 | 20 of 27 | 19 of 27 |
| INT5 | 1.1 | 0.0017 | 0.0004 | 0.0002 | 21 of 27 | 21 of 27 |
| INT5 | 1.2 | 0.0030 | 0.0006 | 0.0003 | 24 of 27 | 24 of 27 |
| INT5 | 2.83 | 0.0420 | 0.0034 | 0.0030 | 27 of 27 | 27 of 27 |
| INT5 | 5.66 | 0.3052 | 0.0126 | 0.0131 | 26 of 27 | 25 of 27 |
| INT5 | 11.3 | 1.0143 | 0.0371 | 0.1429 | 26 of 27 | 22 of 27 |
| INT4 | 0 | 0.4764 | 0.0297 | 0.0032 | 27 of 27 | 27 of 27 |
| INT4 | 0.0884 | 0.2651 | 0.0155 | 0.0028 | 27 of 27 | 27 of 27 |
| INT4 | 0.177 | 0.2009 | 0.0101 | 0.0025 | 27 of 27 | 27 of 27 |
| INT4 | 0.354 | 0.0568 | 0.0048 | 0.0010 | 22 of 27 | 22 of 27 |
| INT4 | 1.1 | 0.0012 | 0.0005 | 0.0002 | 12 of 27 | 11 of 27 |
| INT4 | 1.2 | 0.0020 | 0.0008 | 0.0004 | 11 of 27 | 9 of 27 |
| INT4 | 2.83 | 0.0431 | 0.0044 | 0.0047 | 26 of 27 | 23 of 27 |
| INT4 | 5.66 | 0.3251 | 0.0154 | 0.0252 | 27 of 27 | 25 of 27 |
| INT4 | 11.3 | 1.0946 | 0.0391 | 0.1479 | 26 of 27 | 23 of 27 |

## Limits

- Exploratory, already examined data; one model, one layer, one SAE, RTN weights only; the loss on the language's own text, not task accuracy.
- The group is the frozen full-precision K64 group (small for most languages); the controls are activity-matched but not identical in function; alpha* is read on a coarse grid.
- Counts are the decision rules; the cell tables are descriptive and uncorrected for multiple comparisons.
