# Study G, G1: baseline of language selectivity in every precision state

Descriptive. Input: the D-035 profiles (FLORES devtest, equal-budget stream; sha256 `4ed15f779ca8d6d3892758322bda5f7b73a6be2c7fd52f135ca11397a9e12c83`), the Phase 1 dev selectivity (sha256 `2f0422c2c4c4fdeae9024b8afa9c0b68a2baca926f0df5a16aac838e5a0db3f6`), the frozen groups manifest; group size K = 64. The profiles are devtest: they describe each state and do not select any group that is evaluated on devtest.

## Median over the languages, by state

| state | features selective for the language (state view) | of its FP-dev cohort still selective | features new to the language | group peak retained | group activation on the language, ratio to FP | the language's whole activation mass, ratio to FP | matched-control pool, ratio to FP | group ratio over whole-language ratio | share of the language's mass carried by the group | by the matched-control pool |
|---|---|---|---|---|---|---|---|---|---|---|
| FP | 258.0 | 0.797 | 80.0 | 0.922 | 1.000 | 1.000 | 1.000 | 1.000 | 0.048 | 0.414 |
| INT8 | 260.0 | 0.796 | 79.0 | 0.922 | 1.000 | 0.997 | 0.998 | 1.002 | 0.048 | 0.414 |
| INT7 | 259.0 | 0.787 | 81.0 | 0.906 | 0.998 | 1.000 | 1.001 | 0.997 | 0.048 | 0.414 |
| INT6 | 263.0 | 0.777 | 84.0 | 0.922 | 0.995 | 0.994 | 0.992 | 1.000 | 0.047 | 0.412 |
| INT5 | 283.0 | 0.722 | 101.0 | 0.875 | 0.990 | 1.015 | 1.010 | 0.975 | 0.046 | 0.413 |
| INT4 | 250.0 | 0.561 | 120.0 | 0.766 | 0.852 | 0.843 | 0.853 | 1.004 | 0.051 | 0.419 |

## By language: FP and INT4

| language | cohort size (FP dev) | cohort still selective FP | INT4 | group peak retained FP | INT4 | group activation ratio INT4 | group mass share FP | INT4 |
|---|---|---|---|---|---|---|---|---|
| eng_Latn | 1719 | 0.822 | 0.797 | 0.953 | 0.938 | 0.773 | 0.067 | 0.059 |
| deu_Latn | 234 | 0.697 | 0.560 | 0.922 | 0.828 | 0.852 | 0.051 | 0.051 |
| spa_Latn | 214 | 0.631 | 0.444 | 0.781 | 0.672 | 0.858 | 0.043 | 0.043 |
| fra_Latn | 169 | 0.751 | 0.627 | 0.828 | 0.781 | 0.824 | 0.042 | 0.041 |
| por_Latn | 73 | 0.521 | 0.521 | 0.547 | 0.562 | 0.922 | 0.026 | 0.028 |
| rus_Cyrl | 83 | 0.518 | 0.434 | 0.625 | 0.531 | 0.779 | 0.029 | 0.027 |
| zho_Hans | 465 | 0.759 | 0.628 | 0.969 | 0.938 | 0.872 | 0.074 | 0.075 |
| jpn_Jpan | 514 | 0.761 | 0.531 | 0.969 | 0.797 | 0.832 | 0.064 | 0.063 |
| kor_Hang | 216 | 0.801 | 0.551 | 0.906 | 0.625 | 0.884 | 0.036 | 0.038 |
| hin_Deva | 148 | 0.797 | 0.615 | 0.922 | 0.781 | 0.875 | 0.056 | 0.059 |
| ben_Beng | 108 | 0.787 | 0.546 | 0.922 | 0.609 | 0.831 | 0.018 | 0.018 |
| tam_Taml | 209 | 0.837 | 0.598 | 0.906 | 0.703 | 0.847 | 0.030 | 0.030 |
| tur_Latn | 317 | 0.726 | 0.498 | 0.953 | 0.875 | 0.805 | 0.057 | 0.054 |
| vie_Latn | 298 | 0.752 | 0.611 | 0.938 | 0.859 | 0.824 | 0.034 | 0.033 |
| ind_Latn | 139 | 0.683 | 0.561 | 0.844 | 0.656 | 0.860 | 0.018 | 0.019 |
| tha_Thai | 329 | 0.809 | 0.641 | 0.969 | 0.781 | 0.839 | 0.040 | 0.040 |
| pol_Latn | 137 | 0.584 | 0.482 | 0.812 | 0.656 | 0.880 | 0.042 | 0.043 |
| ukr_Cyrl | 84 | 0.702 | 0.571 | 0.750 | 0.688 | 0.839 | 0.037 | 0.036 |
| kan_Knda | 160 | 0.844 | 0.487 | 0.953 | 0.656 | 0.857 | 0.036 | 0.037 |
| tel_Telu | 135 | 0.867 | 0.652 | 0.938 | 0.766 | 0.934 | 0.048 | 0.053 |
| yor_Latn | 453 | 0.932 | 0.620 | 0.969 | 0.828 | 0.888 | 0.233 | 0.234 |
| sun_Latn | 250 | 0.864 | 0.636 | 0.922 | 0.859 | 0.925 | 0.087 | 0.094 |
| amh_Ethi | 189 | 0.905 | 0.561 | 0.969 | 0.719 | 0.912 | 0.117 | 0.127 |
| ibo_Latn | 296 | 0.855 | 0.568 | 0.906 | 0.812 | 0.849 | 0.092 | 0.090 |
| hau_Latn | 239 | 0.858 | 0.460 | 0.953 | 0.719 | 0.799 | 0.070 | 0.067 |
| khm_Khmr | 444 | 0.901 | 0.597 | 0.891 | 0.875 | 0.874 | 0.099 | 0.105 |
| lao_Laoo | 464 | 0.905 | 0.550 | 0.938 | 0.766 | 0.743 | 0.169 | 0.153 |

## H-G7 (decision rule of the design)

Verdict: **not met**. Medians over languages non-increasing along FP > INT8 > INT7 > INT6 > INT5 > INT4 and, in at least 24 languages, the value in INT4 strictly below FP, for the share of the language's activation mass carried by the group (median ordered: False, languages declining: 11) and for the retained fraction of the group's peak (median ordered: False, languages declining: 26). Reported, not part of the rule: the ratio of the group's activation to FP (median ordered: True, languages declining: 27).

## Limits

- Devtest profiles of the D-035 cohort (equal-budget stream, 27,036 content tokens per language); no confidence intervals: the rows are descriptive.
- The state view applies the support rule and the dev null cutoff of the cohort to each state's own rates; the dev bootstrap stability of the peak is not recomputed, so the state view is looser than the cohort definition.
- `group peak retained` and `cohort still selective` include the baseline loss of the cohort on devtest (only 82 % of the dev cohort reproduces its association on devtest in FP, D-035): compare states with FP, not with 1.
