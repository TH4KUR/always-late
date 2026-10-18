# Phase 1 analysis, regime `none` (decision D-014)

10000 permutations, 2000 bootstrap draws, seed 0.

## Confirmatory family (Holm over three tests)

| test | estimate | 95% CI / detail | p | Holm p |
|---|---|---|---|---|
| INT7 selective minus matched-universal survival | +0.1938 | [+0.1632, +0.2263] (selective 0.7572, control 0.5859, n 3761+2369) | 9.999e-05 | 0.0003 |
| INT6 selective minus matched-universal survival | +0.1576 | [+0.1269, +0.1864] (selective 0.4791, control 0.3212, n 3761+2369) | 9.999e-05 | 0.0003 |
| language-level tier slope (median-r gap, INT7+INT6, one-sided) | +0.0017 | languages used 27 (need 20: ok); High minus Low 0.01813491185506185 | 0.4393 | 0.4393 |

TOST (90% bootstrap interval vs bound): slope -0.0057..+0.0115, equivalent within +-0.02: True; High minus Low +0.0058..+0.0318, equivalent: False.

### Per-language gaps (median r of selective minus matched controls, mean of INT7 and INT6)

| language | tier | Joshi | selective features | gap (median r) | gap (survival rate) | used |
|---|---|---|---|---|---|---|
| eng_Latn | high | 5 | 650 | +0.0164 | +0.0733 | yes |
| deu_Latn | high | 5 | 98 | +0.0933 | +0.5524 | yes |
| spa_Latn | high | 5 | 86 | +0.1017 | +0.5524 | yes |
| fra_Latn | high | 5 | 105 | +0.0963 | +0.7757 | yes |
| por_Latn | high | 4 | 75 | +0.1305 | +0.9472 | yes |
| rus_Cyrl | high | 4 | 65 | +0.0329 | +0.4681 | yes |
| zho_Hans | high | 5 | 115 | +0.0233 | +0.0113 | yes |
| jpn_Jpan | high | 5 | 141 | +0.0560 | +0.4332 | yes |
| kor_Hang | high | 4 | 123 | +0.0434 | +0.4811 | yes |
| hin_Deva | mid | 4 | 33 | +0.1104 | +0.5544 | yes |
| ben_Beng | mid | 3 | 65 | +0.0729 | +0.4941 | yes |
| tam_Taml | mid | 3 | 122 | +0.0305 | +0.4608 | yes |
| tur_Latn | mid | 4 | 49 | +0.0244 | +0.0092 | yes |
| vie_Latn | mid | 4 | 100 | +0.0654 | +0.4608 | yes |
| ind_Latn | mid | 3 | 56 | +0.0995 | +0.4618 | yes |
| tha_Thai | mid | 3 | 96 | +0.0393 | +0.4628 | yes |
| pol_Latn | mid | 4 | 34 | +0.0305 | +0.2678 | yes |
| ukr_Cyrl | mid | 3 | 67 | +0.1152 | +0.9548 | yes |
| kan_Knda | low | 1 | 100 | +0.0303 | +0.4881 | yes |
| tel_Telu | low | 1 | 78 | +0.0312 | +0.4791 | yes |
| yor_Latn | low | 2 | 247 | +0.0658 | +0.2707 | yes |
| sun_Latn | low | 1 | 82 | +0.0923 | +0.9456 | yes |
| amh_Ethi | low | 2 | 154 | +0.0092 | -0.0259 | yes |
| ibo_Latn | low | 1 | 246 | +0.0551 | +0.4941 | yes |
| hau_Latn | low | 2 | 99 | +0.1026 | +0.9456 | yes |
| khm_Khmr | low | 1 | 298 | +0.0333 | +0.0992 | yes |
| lao_Laoo | low | 2 | 377 | +0.0109 | +0.0000 | yes |

## Exploratory analyses (not part of the confirmatory family; p-values are uncorrected)

| width | selective minus control | 95% CI | p (uncorrected) |
|---|---|---|---|
| INT5 | +0.0908 | [+0.0668, +0.1132] | 9.999e-05 |
| INT4 | +0.0449 | [+0.0313, +0.0571] | 9.999e-05 |

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

P1 (selective features fire less): mean panel firing rate selective 0.00396, other supported 0.02963, Mann-Whitney p 0.

## Instrument check: survival of the same features at the noise-floor widths INT16, INT12 and restore-only INT0 (all should be at the ceiling)

| width | selective | control |
|---|---|---|
| INT16 | 0.9981 | 0.9987 |
| INT12 | 0.9981 | 0.9983 |
| INT0 | 1.0000 | 1.0000 |
