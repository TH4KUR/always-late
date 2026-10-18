# Phase 1 analysis, regime `none` (decision D-014)

10000 permutations, 2000 bootstrap draws, seed 0.

## Confirmatory family (Holm over three tests)

| test | estimate | 95% CI / detail | p | Holm p |
|---|---|---|---|---|
| INT7 selective minus matched-universal survival | +0.1920 | [+0.1592, +0.2247] (selective 0.8172, control 0.6250, n 1329+1240) | 9.999e-05 | 0.0003 |
| INT6 selective minus matched-universal survival | +0.1746 | [+0.1394, +0.2088] (selective 0.5365, control 0.3605, n 1329+1240) | 9.999e-05 | 0.0003 |
| language-level tier slope (median-r gap, INT7+INT6, one-sided) | -0.0006 | languages used 26 (need 20: ok); High minus Low 0.0053296900457806 | 0.518 | 0.518 |

TOST (90% bootstrap interval vs bound): slope -0.0158..+0.0085, equivalent within +-0.02: True; High minus Low -0.0143..+0.0203, equivalent: False.

### Per-language gaps (median r of selective minus matched controls, mean of INT7 and INT6)

| language | tier | Joshi | selective features | gap (median r) | gap (survival rate) | used |
|---|---|---|---|---|---|---|
| eng_Latn | high | 5 | 329 | +0.0326 | +0.0321 | yes |
| deu_Latn | high | 5 | 35 | +0.0718 | +0.4889 | yes |
| spa_Latn | high | 5 | 26 | +0.0841 | +0.5001 | yes |
| fra_Latn | high | 5 | 33 | +0.1245 | +0.5294 | yes |
| por_Latn | high | 4 | 29 | +0.1825 | +0.9675 | yes |
| rus_Cyrl | high | 4 | 17 | +0.1192 | +0.3795 | yes |
| zho_Hans | high | 5 | 28 | +0.0113 | +0.0048 | yes |
| jpn_Jpan | high | 5 | 44 | +0.0573 | +0.5157 | yes |
| kor_Hang | high | 4 | 26 | +0.0028 | +0.0000 | yes |
| hin_Deva | mid | 4 | 10 | +0.0353 | +0.4956 | yes |
| ben_Beng | mid | 3 | 16 | +0.1093 | +0.5027 | yes |
| tam_Taml | mid | 3 | 18 | +0.0316 | +0.5001 | yes |
| tur_Latn | mid | 4 | 16 | +0.0137 | +0.0017 | yes |
| vie_Latn | mid | 4 | 20 | +0.0457 | +0.0084 | yes |
| ind_Latn | mid | 3 | 19 | +0.0150 | +0.0084 | yes |
| tha_Thai | mid | 3 | 26 | +0.0380 | +0.0111 | yes |
| pol_Latn | mid | 4 | 8 | +0.0474 | +0.4659 | no |
| ukr_Cyrl | mid | 3 | 12 | +0.1256 | +0.9736 | yes |
| kan_Knda | low | 1 | 28 | +0.0860 | +0.5625 | yes |
| tel_Telu | low | 1 | 20 | +0.0393 | +0.4979 | yes |
| yor_Latn | low | 2 | 98 | +0.0650 | +0.0954 | yes |
| sun_Latn | low | 1 | 19 | +0.1308 | +0.5888 | yes |
| amh_Ethi | low | 2 | 47 | +0.0503 | +0.0661 | yes |
| ibo_Latn | low | 1 | 73 | +0.1170 | +0.9625 | yes |
| hau_Latn | low | 2 | 24 | +0.0892 | +0.5611 | yes |
| khm_Khmr | low | 1 | 138 | +0.0460 | +0.0954 | yes |
| lao_Laoo | low | 2 | 170 | +0.0145 | +0.0661 | yes |

## Exploratory analyses (not part of the confirmatory family; p-values are uncorrected)

| width | selective minus control | 95% CI | p (uncorrected) |
|---|---|---|---|
| INT5 | +0.0923 | [+0.0622, +0.1228] | 9.999e-05 |
| INT4 | +0.0594 | [+0.0374, +0.0806] | 9.999e-05 |

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

P1 (selective features fire less): mean panel firing rate selective 0.00285, other supported 0.02173, Mann-Whitney p 0.

## Instrument check: survival of the same features at the noise-floor widths INT16, INT12 and restore-only INT0 (all should be at the ceiling)

| width | selective | control |
|---|---|---|
| INT16 | 0.9970 | 0.9984 |
| INT12 | 0.9977 | 0.9976 |
| INT0 | 1.0000 | 1.0000 |
