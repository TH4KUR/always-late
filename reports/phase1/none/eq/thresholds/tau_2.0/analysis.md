# Phase 1 analysis, regime `none` (decision D-014)

10000 permutations, 2000 bootstrap draws, seed 0.

## Confirmatory family (Holm over three tests)

| test | estimate | 95% CI / detail | p | Holm p |
|---|---|---|---|---|
| INT7 selective minus matched-universal survival | +0.1086 | [+0.0704, +0.1479] (selective 0.7214, control 0.6746, n 4947+1054) | 9.999e-05 | 0.0003 |
| INT6 selective minus matched-universal survival | +0.0964 | [+0.0591, +0.1326] (selective 0.4261, control 0.3501, n 4947+1054) | 9.999e-05 | 0.0003 |
| language-level tier slope (median-r gap, INT7+INT6, one-sided) | +0.0082 | languages used 27 (need 20: ok); High minus Low 0.0343204571141137 | 0.1823 | 0.1823 |

TOST (90% bootstrap interval vs bound): slope -0.0021..+0.0190, equivalent within +-0.02: True; High minus Low +0.0174..+0.0488, equivalent: False.

### Per-language gaps (median r of selective minus matched controls, mean of INT7 and INT6)

| language | tier | Joshi | selective features | gap (median r) | gap (survival rate) | used |
|---|---|---|---|---|---|---|
| eng_Latn | high | 5 | 791 | -0.0154 | +0.0761 | yes |
| deu_Latn | high | 5 | 144 | +0.0483 | +0.5162 | yes |
| spa_Latn | high | 5 | 108 | +0.0766 | +0.5749 | yes |
| fra_Latn | high | 5 | 138 | +0.0637 | +0.0857 | yes |
| por_Latn | high | 4 | 100 | +0.0468 | +0.1157 | yes |
| rus_Cyrl | high | 4 | 105 | +0.0206 | +0.0106 | yes |
| zho_Hans | high | 5 | 195 | +0.0423 | +0.5013 | yes |
| jpn_Jpan | high | 5 | 204 | +0.0426 | +0.1157 | yes |
| kor_Hang | high | 4 | 148 | +0.0828 | +0.5394 | yes |
| hin_Deva | mid | 4 | 60 | +0.0395 | +0.5653 | yes |
| ben_Beng | mid | 3 | 87 | +0.0339 | +0.0761 | yes |
| tam_Taml | mid | 3 | 172 | +0.0097 | +0.0953 | yes |
| tur_Latn | mid | 4 | 77 | +0.0188 | +0.4441 | yes |
| vie_Latn | mid | 4 | 147 | +0.0635 | +0.4441 | yes |
| ind_Latn | mid | 3 | 100 | +0.0696 | +0.5598 | yes |
| tha_Thai | mid | 3 | 124 | +0.0449 | +0.5592 | yes |
| pol_Latn | mid | 4 | 73 | +0.0146 | -0.0192 | yes |
| ukr_Cyrl | mid | 3 | 99 | +0.0430 | +0.1157 | yes |
| kan_Knda | low | 1 | 130 | +0.0558 | +0.5663 | yes |
| tel_Telu | low | 1 | 97 | +0.0243 | +0.5354 | yes |
| yor_Latn | low | 2 | 287 | +0.0300 | +0.0761 | yes |
| sun_Latn | low | 1 | 153 | +0.0605 | +0.4456 | yes |
| amh_Ethi | low | 2 | 200 | -0.0504 | -0.4796 | yes |
| ibo_Latn | low | 1 | 277 | -0.0054 | -0.0396 | yes |
| hau_Latn | low | 2 | 146 | +0.0187 | +0.0448 | yes |
| khm_Khmr | low | 1 | 361 | -0.0063 | +0.0761 | yes |
| lao_Laoo | low | 2 | 424 | -0.0278 | -0.0396 | yes |

## Exploratory analyses (not part of the confirmatory family; p-values are uncorrected)

| width | selective minus control | 95% CI | p (uncorrected) |
|---|---|---|---|
| INT5 | +0.0381 | [+0.0068, +0.0703] | 0.024 |
| INT4 | +0.0089 | [-0.0130, +0.0296] | 0.4203 |

Mixed-effects logistic model (random intercept per feature, quadrature ML; 6001 features, sigma 5.839, converged True):

| term | estimate | SE | p |
|---|---|---|---|
| Intercept | -3.9776 | 0.4801 | 1.187e-16 |
| C(stratum)[T.1] | +1.2676 | 0.3052 | 3.285e-05 |
| C(stratum)[T.2] | +1.5958 | 0.3204 | 6.344e-07 |
| C(stratum)[T.3] | +1.1105 | 0.3234 | 0.0005961 |
| C(stratum)[T.4] | +4.9876 | 0.4569 | 9.542e-28 |
| bits_c | +5.0357 | 0.2914 | 6.49e-67 |
| S_rank | +5.5357 | 0.6003 | 2.937e-20 |
| tier | -0.2121 | 0.2072 | 0.3061 |
| S_rank:tier | +1.5188 | 0.3530 | 1.687e-05 |

P1 (selective features fire less): mean panel firing rate selective 0.00804, other supported 0.06216, Mann-Whitney p 0.

## Instrument check: survival of the same features at the noise-floor widths INT16, INT12 and restore-only INT0 (all should be at the ceiling)

| width | selective | control |
|---|---|---|
| INT16 | 0.9986 | 0.9991 |
| INT12 | 0.9984 | 1.0000 |
| INT0 | 1.0000 | 1.0000 |
