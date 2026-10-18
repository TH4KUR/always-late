# Phase 1 analysis, regime `none` (decision D-014)

10000 permutations, 2000 bootstrap draws, seed 0.

## Confirmatory family (Holm over three tests)

| test | estimate | 95% CI / detail | p | Holm p |
|---|---|---|---|---|
| INT7 selective minus matched-universal survival | +0.1730 | [+0.1384, +0.2087] (selective 0.7661, control 0.6079, n 3326+2063) | 9.999e-05 | 0.0003 |
| INT6 selective minus matched-universal survival | +0.1497 | [+0.1172, +0.1833] (selective 0.4765, control 0.3296, n 3326+2063) | 9.999e-05 | 0.0003 |
| language-level tier slope (median-r gap, INT7+INT6, one-sided) | +0.0025 | languages used 27 (need 20: ok); High minus Low 0.02648432221677568 | 0.4225 | 0.4225 |

TOST (90% bootstrap interval vs bound): slope -0.0054..+0.0115, equivalent within +-0.02: True; High minus Low +0.0122..+0.0378, equivalent: False.

### Per-language gaps (median r of selective minus matched controls, mean of INT7 and INT6)

| language | tier | Joshi | selective features | gap (median r) | gap (survival rate) | used |
|---|---|---|---|---|---|---|
| eng_Latn | high | 5 | 647 | +0.0177 | +0.0858 | yes |
| deu_Latn | high | 5 | 96 | +0.1032 | +0.5369 | yes |
| spa_Latn | high | 5 | 74 | +0.0986 | +0.5028 | yes |
| fra_Latn | high | 5 | 92 | +0.1060 | +0.9752 | yes |
| por_Latn | high | 4 | 66 | +0.1315 | +0.9752 | yes |
| rus_Cyrl | high | 4 | 63 | +0.0187 | +0.0695 | yes |
| zho_Hans | high | 5 | 108 | +0.0255 | +0.0118 | yes |
| jpn_Jpan | high | 5 | 122 | +0.0628 | +0.4008 | yes |
| kor_Hang | high | 4 | 94 | +0.0690 | +0.4702 | yes |
| hin_Deva | mid | 4 | 25 | +0.0511 | +0.5619 | yes |
| ben_Beng | mid | 3 | 49 | +0.0793 | +0.5561 | yes |
| tam_Taml | mid | 3 | 101 | -0.0005 | +0.0319 | yes |
| tur_Latn | mid | 4 | 43 | +0.0420 | +0.4318 | yes |
| vie_Latn | mid | 4 | 85 | +0.0781 | +0.4318 | yes |
| ind_Latn | mid | 3 | 50 | +0.0839 | +0.4318 | yes |
| tha_Thai | mid | 3 | 77 | +0.0396 | +0.4568 | yes |
| pol_Latn | mid | 4 | 32 | -0.0090 | -0.4718 | yes |
| ukr_Cyrl | mid | 3 | 49 | +0.1110 | +0.9559 | yes |
| kan_Knda | low | 1 | 93 | +0.0640 | +0.5561 | yes |
| tel_Telu | low | 1 | 59 | +0.0095 | +0.4510 | yes |
| yor_Latn | low | 2 | 214 | +0.0458 | +0.0858 | yes |
| sun_Latn | low | 1 | 63 | +0.1084 | +0.9752 | yes |
| amh_Ethi | low | 2 | 147 | -0.0124 | -0.0326 | yes |
| ibo_Latn | low | 1 | 192 | +0.0480 | +0.1184 | yes |
| hau_Latn | low | 2 | 85 | +0.0987 | +0.5059 | yes |
| khm_Khmr | low | 1 | 268 | +0.0294 | +0.1184 | yes |
| lao_Laoo | low | 2 | 332 | +0.0032 | +0.0858 | yes |

## Exploratory analyses (not part of the confirmatory family; p-values are uncorrected)

| width | selective minus control | 95% CI | p (uncorrected) |
|---|---|---|---|
| INT5 | +0.0728 | [+0.0456, +0.1010] | 9.999e-05 |
| INT4 | +0.0302 | [+0.0114, +0.0468] | 0.0046 |

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

P1 (selective features fire less): mean panel firing rate selective 0.00477, other supported 0.03343, Mann-Whitney p 0.

## Instrument check: survival of the same features at the noise-floor widths INT16, INT12 and restore-only INT0 (all should be at the ceiling)

| width | selective | control |
|---|---|---|
| INT16 | 0.9982 | 0.9990 |
| INT12 | 0.9982 | 0.9990 |
| INT0 | 1.0000 | 1.0000 |
