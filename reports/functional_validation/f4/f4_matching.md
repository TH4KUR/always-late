# F4 subgroups: lost against retained features at equal native activity (matching report)

**Exploratory and descriptive.** Subsets of the selective features that were established in full precision, split by their D-035 status at a precision (lost, retained) and matched on the native activity at that precision (log rate and log mean activation of the feature on its own language in the quantized model). The functional comparison is run by the unchanged runner and analysis (the lost subset as the group, the matched retained draws as the controls): E = harm(lost) - harm(retained matched).

**Hypothesis written before any number was computed (H-F4, two-sided):** E_b differs from 0 at precision b; "association predicts function" predicts E_b < 0 (lost features matter less); a null or a positive E_b is reported as it is. The same subsets are also run in full precision as a reference (does a difference pre-exist the quantization?).

## INT6

Caliper 0.25 SD, cap 64 lost features per language, 3 retained draws, at least 8 matched pairs, seed 0. Languages included: 18 of 27; lost features used: 288 (of 356 established features lost at INT6, 1 of them inactive at INT6 and therefore not matchable). Balance (standardised mean difference of the native activity, lost minus retained, per language): mean |SMD| 0.014, maximum 0.045.

| language | lost at b | inactive at b | matched | selected | draws | SMD log rate | SMD log mean |
|---|---|---|---|---|---|---|---|
| eng_Latn | 57 | 0 | 54 | 54 | 1 | -0.002 | +0.000 |
| deu_Latn | 12 | 0 | 11 | 11 | 1 | +0.005 | -0.011 |
| spa_Latn | 11 | 0 | 11 | 11 | 2 | +0.016 | -0.005 |
| fra_Latn | 9 | 0 | 7 | 0 | 0 | - | - |
| por_Latn | 1 | 0 | 1 | 0 | 0 | - | - |
| rus_Cyrl | 1 | 0 | 1 | 0 | 0 | - | - |
| zho_Hans | 15 | 0 | 15 | 15 | 2 | -0.005 | -0.017 |
| jpn_Jpan | 29 | 0 | 29 | 29 | 3 | +0.000 | -0.005 |
| kor_Hang | 7 | 0 | 7 | 0 | 0 | - | - |
| hin_Deva | 5 | 0 | 5 | 0 | 0 | - | - |
| ben_Beng | 8 | 0 | 6 | 0 | 0 | - | - |
| tam_Taml | 8 | 0 | 8 | 8 | 1 | -0.013 | +0.037 |
| tur_Latn | 11 | 0 | 11 | 11 | 2 | -0.022 | -0.032 |
| vie_Latn | 9 | 0 | 9 | 9 | 2 | -0.016 | -0.021 |
| ind_Latn | 4 | 0 | 4 | 0 | 0 | - | - |
| tha_Thai | 9 | 0 | 8 | 8 | 3 | -0.034 | +0.003 |
| pol_Latn | 4 | 0 | 4 | 0 | 0 | - | - |
| ukr_Cyrl | 9 | 0 | 8 | 8 | 1 | +0.006 | -0.025 |
| kan_Knda | 9 | 0 | 9 | 9 | 2 | +0.003 | -0.004 |
| tel_Telu | 4 | 0 | 4 | 0 | 0 | - | - |
| yor_Latn | 22 | 0 | 19 | 19 | 1 | -0.024 | +0.001 |
| sun_Latn | 8 | 0 | 8 | 8 | 2 | -0.045 | -0.006 |
| amh_Ethi | 14 | 1 | 9 | 9 | 2 | -0.002 | +0.002 |
| ibo_Latn | 12 | 0 | 10 | 10 | 1 | -0.032 | -0.019 |
| hau_Latn | 21 | 0 | 19 | 19 | 1 | -0.015 | +0.011 |
| khm_Khmr | 26 | 0 | 20 | 20 | 1 | -0.034 | -0.004 |
| lao_Laoo | 31 | 0 | 30 | 30 | 1 | -0.008 | -0.006 |

Excluded: fra_Latn (fewer than 8 matched pairs (7 matched of 9 lost features with native activity at INT6)); por_Latn (fewer than 8 matched pairs (1 matched of 1 lost features with native activity at INT6)); rus_Cyrl (fewer than 8 matched pairs (1 matched of 1 lost features with native activity at INT6)); kor_Hang (fewer than 8 matched pairs (7 matched of 7 lost features with native activity at INT6)); hin_Deva (fewer than 8 matched pairs (5 matched of 5 lost features with native activity at INT6)); ben_Beng (fewer than 8 matched pairs (6 matched of 8 lost features with native activity at INT6)); ind_Latn (fewer than 8 matched pairs (4 matched of 4 lost features with native activity at INT6)); pol_Latn (fewer than 8 matched pairs (4 matched of 4 lost features with native activity at INT6)); tel_Telu (fewer than 8 matched pairs (4 matched of 4 lost features with native activity at INT6))

## INT5

Caliper 0.25 SD, cap 64 lost features per language, 3 retained draws, at least 8 matched pairs, seed 0. Languages included: 24 of 27; lost features used: 554 (of 726 established features lost at INT5, 1 of them inactive at INT5 and therefore not matchable). Balance (standardised mean difference of the native activity, lost minus retained, per language): mean |SMD| 0.010, maximum 0.033.

| language | lost at b | inactive at b | matched | selected | draws | SMD log rate | SMD log mean |
|---|---|---|---|---|---|---|---|
| eng_Latn | 148 | 0 | 142 | 64 | 1 | -0.002 | +0.004 |
| deu_Latn | 19 | 0 | 18 | 18 | 2 | +0.011 | +0.003 |
| spa_Latn | 16 | 0 | 13 | 13 | 2 | -0.005 | +0.013 |
| fra_Latn | 11 | 0 | 9 | 9 | 1 | +0.014 | -0.007 |
| por_Latn | 4 | 0 | 3 | 0 | 0 | - | - |
| rus_Cyrl | 10 | 0 | 8 | 8 | 1 | +0.012 | -0.020 |
| zho_Hans | 41 | 0 | 40 | 40 | 1 | -0.011 | -0.010 |
| jpn_Jpan | 41 | 0 | 40 | 40 | 1 | -0.012 | -0.012 |
| kor_Hang | 21 | 0 | 17 | 17 | 1 | +0.004 | +0.010 |
| hin_Deva | 20 | 0 | 18 | 18 | 1 | +0.004 | +0.020 |
| ben_Beng | 11 | 0 | 10 | 10 | 1 | -0.028 | -0.033 |
| tam_Taml | 21 | 0 | 19 | 19 | 1 | -0.000 | -0.018 |
| tur_Latn | 30 | 0 | 26 | 26 | 2 | -0.007 | +0.010 |
| vie_Latn | 28 | 0 | 25 | 25 | 2 | +0.002 | -0.003 |
| ind_Latn | 5 | 0 | 3 | 0 | 0 | - | - |
| tha_Thai | 31 | 0 | 28 | 28 | 2 | -0.019 | +0.010 |
| pol_Latn | 8 | 0 | 8 | 8 | 3 | -0.014 | -0.021 |
| ukr_Cyrl | 6 | 0 | 6 | 0 | 0 | - | - |
| kan_Knda | 20 | 0 | 16 | 16 | 1 | -0.003 | +0.013 |
| tel_Telu | 15 | 0 | 15 | 15 | 2 | -0.013 | +0.001 |
| yor_Latn | 40 | 0 | 36 | 36 | 1 | -0.009 | +0.016 |
| sun_Latn | 27 | 0 | 23 | 23 | 2 | -0.005 | +0.011 |
| amh_Ethi | 15 | 1 | 9 | 9 | 3 | -0.010 | +0.011 |
| ibo_Latn | 27 | 0 | 19 | 19 | 1 | +0.008 | -0.004 |
| hau_Latn | 18 | 0 | 16 | 16 | 1 | -0.004 | -0.012 |
| khm_Khmr | 42 | 0 | 34 | 34 | 1 | -0.002 | -0.007 |
| lao_Laoo | 51 | 0 | 43 | 43 | 1 | -0.012 | +0.001 |

Excluded: por_Latn (fewer than 8 matched pairs (3 matched of 4 lost features with native activity at INT5)); ind_Latn (fewer than 8 matched pairs (3 matched of 5 lost features with native activity at INT5)); ukr_Cyrl (fewer than 8 matched pairs (6 matched of 6 lost features with native activity at INT5))

## INT4

Caliper 0.25 SD, cap 64 lost features per language, 3 retained draws, at least 8 matched pairs, seed 0. Languages included: 26 of 27; lost features used: 1073 (of 1945 established features lost at INT4, 11 of them inactive at INT4 and therefore not matchable). Balance (standardised mean difference of the native activity, lost minus retained, per language): mean |SMD| 0.016, maximum 0.067.

| language | lost at b | inactive at b | matched | selected | draws | SMD log rate | SMD log mean |
|---|---|---|---|---|---|---|---|
| eng_Latn | 176 | 0 | 120 | 64 | 2 | -0.011 | -0.001 |
| deu_Latn | 43 | 0 | 32 | 32 | 1 | +0.006 | -0.008 |
| spa_Latn | 53 | 0 | 39 | 39 | 1 | +0.041 | -0.004 |
| fra_Latn | 28 | 0 | 22 | 22 | 1 | -0.015 | -0.001 |
| por_Latn | 8 | 0 | 5 | 0 | 0 | - | - |
| rus_Cyrl | 15 | 0 | 9 | 9 | 1 | +0.067 | +0.028 |
| zho_Hans | 88 | 0 | 65 | 64 | 1 | +0.003 | -0.007 |
| jpn_Jpan | 146 | 0 | 110 | 64 | 1 | +0.016 | +0.010 |
| kor_Hang | 57 | 0 | 36 | 36 | 1 | +0.008 | +0.009 |
| hin_Deva | 37 | 0 | 29 | 29 | 1 | +0.023 | -0.001 |
| ben_Beng | 34 | 1 | 22 | 22 | 1 | +0.026 | +0.011 |
| tam_Taml | 58 | 0 | 44 | 44 | 1 | +0.011 | +0.004 |
| tur_Latn | 100 | 0 | 69 | 64 | 1 | +0.041 | +0.014 |
| vie_Latn | 62 | 0 | 44 | 44 | 1 | +0.016 | -0.000 |
| ind_Latn | 25 | 0 | 18 | 18 | 1 | +0.060 | -0.004 |
| tha_Thai | 64 | 0 | 49 | 49 | 1 | +0.002 | -0.008 |
| pol_Latn | 29 | 0 | 22 | 22 | 1 | +0.028 | +0.022 |
| ukr_Cyrl | 16 | 0 | 10 | 10 | 1 | +0.020 | -0.029 |
| kan_Knda | 63 | 0 | 38 | 38 | 1 | +0.015 | +0.001 |
| tel_Telu | 36 | 0 | 24 | 24 | 1 | +0.028 | +0.009 |
| yor_Latn | 148 | 0 | 96 | 64 | 1 | +0.003 | +0.012 |
| sun_Latn | 61 | 0 | 40 | 40 | 1 | +0.014 | +0.008 |
| amh_Ethi | 70 | 3 | 39 | 39 | 1 | +0.010 | -0.005 |
| ibo_Latn | 92 | 0 | 45 | 45 | 1 | -0.003 | -0.009 |
| hau_Latn | 105 | 0 | 63 | 63 | 1 | +0.057 | +0.013 |
| khm_Khmr | 149 | 2 | 74 | 64 | 1 | +0.021 | +0.027 |
| lao_Laoo | 182 | 5 | 85 | 64 | 1 | +0.012 | +0.009 |

Excluded: por_Latn (fewer than 8 matched pairs (5 matched of 8 lost features with native activity at INT4))

