# Phase 1 analysis, regime `none` (decision D-014)

10000 permutations, 2000 bootstrap draws, seed 0.

## Confirmatory family (Holm over three tests)

| test | estimate | 95% CI / detail | p | Holm p |
|---|---|---|---|---|
| INT7 selective minus matched-universal survival | +0.1186 | [+0.0816, +0.1548] (selective 0.7121, control 0.6528, n 5582+1132) | 9.999e-05 | 0.0003 |
| INT6 selective minus matched-universal survival | +0.0745 | [+0.0390, +0.1093] (selective 0.4271, control 0.3613, n 5582+1132) | 9.999e-05 | 0.0003 |
| language-level tier slope (median-r gap, INT7+INT6, one-sided) | +0.0094 | languages used 27 (need 20: ok); High minus Low 0.02743094828393724 | 0.2062 | 0.2062 |

TOST (90% bootstrap interval vs bound): slope -0.0041..+0.0187, equivalent within +-0.02: True; High minus Low +0.0076..+0.0414, equivalent: False.

### Per-language gaps (median r of selective minus matched controls, mean of INT7 and INT6)

| language | tier | Joshi | selective features | gap (median r) | gap (survival rate) | used |
|---|---|---|---|---|---|---|
| eng_Latn | high | 5 | 798 | -0.0174 | +0.0425 | yes |
| deu_Latn | high | 5 | 157 | +0.0456 | +0.5615 | yes |
| spa_Latn | high | 5 | 121 | +0.0632 | +0.5615 | yes |
| fra_Latn | high | 5 | 157 | +0.0570 | +0.0888 | yes |
| por_Latn | high | 4 | 111 | +0.0681 | +0.0984 | yes |
| rus_Cyrl | high | 4 | 113 | +0.0243 | +0.0000 | yes |
| zho_Hans | high | 5 | 217 | +0.0374 | +0.5481 | yes |
| jpn_Jpan | high | 5 | 239 | -0.0053 | +0.0000 | yes |
| kor_Hang | high | 4 | 186 | +0.0567 | +0.5615 | yes |
| hin_Deva | mid | 4 | 70 | +0.0691 | +0.5432 | yes |
| ben_Beng | mid | 3 | 116 | +0.0352 | +0.0984 | yes |
| tam_Taml | mid | 3 | 194 | +0.0208 | +0.5475 | yes |
| tur_Latn | mid | 4 | 92 | +0.0205 | +0.0000 | yes |
| vie_Latn | mid | 4 | 179 | +0.0557 | +0.4587 | yes |
| ind_Latn | mid | 3 | 105 | +0.0957 | +0.5868 | yes |
| tha_Thai | mid | 3 | 153 | +0.0760 | +0.5761 | yes |
| pol_Latn | mid | 4 | 76 | +0.0924 | +0.4865 | yes |
| ukr_Cyrl | mid | 3 | 124 | +0.0718 | +0.5074 | yes |
| kan_Knda | low | 1 | 154 | -0.0089 | +0.0699 | yes |
| tel_Telu | low | 1 | 114 | +0.0572 | +0.5761 | yes |
| yor_Latn | low | 2 | 337 | +0.0286 | +0.0000 | yes |
| sun_Latn | low | 1 | 172 | +0.0506 | +0.0279 | yes |
| amh_Ethi | low | 2 | 214 | -0.0355 | -0.4873 | yes |
| ibo_Latn | low | 1 | 339 | +0.0089 | +0.0000 | yes |
| hau_Latn | low | 2 | 164 | +0.0193 | +0.0392 | yes |
| khm_Khmr | low | 1 | 404 | -0.0039 | +0.0565 | yes |
| lao_Laoo | low | 2 | 476 | -0.0338 | -0.0336 | yes |

## Exploratory analyses (not part of the confirmatory family; p-values are uncorrected)

| width | selective minus control | 95% CI | p (uncorrected) |
|---|---|---|---|
| INT5 | +0.0504 | [+0.0228, +0.0763] | 0.0008999 |
| INT4 | +0.0328 | [+0.0171, +0.0469] | 0.0015 |

Mixed-effects logistic model (random intercept per feature, quadrature ML; 6714 features, sigma 5.993, converged True):

| term | estimate | SE | p |
|---|---|---|---|
| Intercept | -3.9455 | 0.4591 | 8.394e-18 |
| C(stratum)[T.1] | +0.6397 | 0.3037 | 0.03514 |
| C(stratum)[T.2] | +1.5331 | 0.3127 | 9.457e-07 |
| C(stratum)[T.3] | +1.0384 | 0.3228 | 0.001295 |
| C(stratum)[T.4] | +4.5558 | 0.4256 | 9.618e-27 |
| bits_c | +4.8876 | 0.2602 | 9.8e-79 |
| S_rank | +5.3928 | 0.5699 | 2.997e-21 |
| tier | +0.0393 | 0.2019 | 0.8458 |
| S_rank:tier | +1.4867 | 0.3503 | 2.192e-05 |

P1 (selective features fire less): mean panel firing rate selective 0.00689, other supported 0.05552, Mann-Whitney p 0.

## Instrument check: survival of the same features at the noise-floor widths INT16, INT12 and restore-only INT0 (all should be at the ceiling)

| width | selective | control |
|---|---|---|
| INT16 | 0.9984 | 0.9982 |
| INT12 | 0.9982 | 0.9991 |
| INT0 | 1.0000 | 1.0000 |
