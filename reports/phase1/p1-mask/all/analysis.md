# Phase 1 analysis, regime `mask` (decision D-014)

10000 permutations, 2000 bootstrap draws, seed 0.

## Confirmatory family (Holm over three tests)

| test | estimate | 95% CI / detail | p | Holm p |
|---|---|---|---|---|
| INT7 selective minus matched-universal survival | -0.0039 | [-0.0074, +0.0002] (selective 0.9921, control 0.9976, n 8579+2890) | 0.165 | 0.33 |
| INT6 selective minus matched-universal survival | -0.0193 | [-0.0332, -0.0046] (selective 0.8968, control 0.9381, n 8579+2890) | 0.0183 | 0.05489 |
| language-level tier slope (median-r gap, INT7+INT6, one-sided) | -0.0035 | languages used 27 (need 20: ok); High minus Low -0.0045019255744086365 | 0.9295 | 0.9295 |

TOST (90% bootstrap interval vs bound): slope -0.0066..-0.0013, equivalent within +-0.02: True; High minus Low -0.0099..-0.0013, equivalent: True.

### Per-language gaps (median r of selective minus matched controls, mean of INT7 and INT6)

| language | tier | Joshi | selective features | gap (median r) | gap (survival rate) | used |
|---|---|---|---|---|---|---|
| eng_Latn | high | 5 | 1742 | -0.0052 | +0.0045 | yes |
| deu_Latn | high | 5 | 228 | +0.0122 | +0.0000 | yes |
| spa_Latn | high | 5 | 203 | +0.0049 | +0.0000 | yes |
| fra_Latn | high | 5 | 180 | +0.0090 | +0.0009 | yes |
| por_Latn | high | 4 | 78 | +0.0098 | +0.0049 | yes |
| rus_Cyrl | high | 4 | 82 | +0.0082 | +0.0000 | yes |
| zho_Hans | high | 5 | 472 | +0.0030 | +0.0000 | yes |
| jpn_Jpan | high | 5 | 533 | -0.0011 | +0.0049 | yes |
| kor_Hang | high | 4 | 232 | +0.0151 | +0.0136 | yes |
| hin_Deva | mid | 4 | 144 | +0.0096 | +0.0045 | yes |
| ben_Beng | mid | 3 | 112 | +0.0123 | +0.0009 | yes |
| tam_Taml | mid | 3 | 218 | +0.0083 | +0.0185 | yes |
| tur_Latn | mid | 4 | 333 | -0.0033 | +0.0000 | yes |
| vie_Latn | mid | 4 | 316 | +0.0087 | +0.0053 | yes |
| ind_Latn | mid | 3 | 140 | +0.0157 | +0.0049 | yes |
| tha_Thai | mid | 3 | 358 | +0.0026 | +0.0181 | yes |
| pol_Latn | mid | 4 | 141 | -0.0004 | +0.0000 | yes |
| ukr_Cyrl | mid | 3 | 80 | +0.0117 | +0.0121 | yes |
| kan_Knda | low | 1 | 178 | +0.0088 | +0.0100 | yes |
| tel_Telu | low | 1 | 156 | +0.0131 | +0.0049 | yes |
| yor_Latn | low | 2 | 506 | +0.0242 | -0.0085 | yes |
| sun_Latn | low | 1 | 280 | +0.0178 | +0.0089 | yes |
| amh_Ethi | low | 2 | 208 | +0.0034 | +0.0089 | yes |
| ibo_Latn | low | 1 | 347 | +0.0142 | +0.0142 | yes |
| hau_Latn | low | 2 | 260 | -0.0038 | +0.0004 | yes |
| khm_Khmr | low | 1 | 503 | +0.0086 | +0.0047 | yes |
| lao_Laoo | low | 2 | 549 | +0.0101 | +0.0004 | yes |

## Exploratory analyses (not part of the confirmatory family; p-values are uncorrected)

| width | selective minus control | 95% CI | p (uncorrected) |
|---|---|---|---|
| INT5 | -0.0130 | [-0.0344, +0.0086] | 0.2702 |
| INT4 | +0.0195 | [+0.0033, +0.0351] | 0.0226 |

Mixed-effects logistic model (random intercept per feature, quadrature ML; 11469 features, sigma 2.747, converged True):

| term | estimate | SE | p |
|---|---|---|---|
| Intercept | +2.1847 | 0.3469 | 3.02e-10 |
| C(stratum)[T.1] | +2.2155 | 0.2560 | 5.017e-18 |
| C(stratum)[T.2] | +2.9896 | 0.3242 | 2.895e-20 |
| C(stratum)[T.3] | +3.3633 | 0.3529 | 1.56e-21 |
| C(stratum)[T.4] | +4.5310 | 0.4333 | 1.375e-25 |
| bits_c | +4.8898 | 0.4292 | 4.595e-30 |
| S_rank | +1.0307 | 0.3255 | 0.001542 |
| tier | +1.4394 | 0.2004 | 6.877e-13 |
| S_rank:tier | +1.3109 | 0.3044 | 1.662e-05 |

P1 (selective features fire less): mean panel firing rate selective 0.00607, other supported 0.00866, Mann-Whitney p 0.

## Instrument check: survival of the same features at the noise-floor widths INT16, INT12 and restore-only INT0 (all should be at the ceiling)

| width | selective | control |
|---|---|---|
| INT16 | 0.9993 | 1.0000 |
| INT12 | 0.9995 | 1.0000 |
| INT0 | 1.0000 | 1.0000 |
