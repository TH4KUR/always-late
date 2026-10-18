# Phase 1 analysis, regime `mask` (decision D-014)

10000 permutations, 2000 bootstrap draws, seed 0.

## Confirmatory family (Holm over three tests)

| test | estimate | 95% CI / detail | p | Holm p |
|---|---|---|---|---|
| INT7 selective minus matched-universal survival | -0.0028 | [-0.0063, +0.0013] (selective 0.9935, control 0.9982, n 8062+2843) | 0.2571 | 0.5141 |
| INT6 selective minus matched-universal survival | -0.0144 | [-0.0265, -0.0013] (selective 0.9145, control 0.9420, n 8062+2843) | 0.0427 | 0.1281 |
| language-level tier slope (median-r gap, INT7+INT6, one-sided) | -0.0048 | languages used 27 (need 20: ok); High minus Low -0.008215147587988112 | 0.9902 | 0.9902 |

TOST (90% bootstrap interval vs bound): slope -0.0075..-0.0016, equivalent within +-0.02: True; High minus Low -0.0129..-0.0031, equivalent: True.

### Per-language gaps (median r of selective minus matched controls, mean of INT7 and INT6)

| language | tier | Joshi | selective features | gap (median r) | gap (survival rate) | used |
|---|---|---|---|---|---|---|
| eng_Latn | high | 5 | 1711 | -0.0051 | +0.0000 | yes |
| deu_Latn | high | 5 | 231 | +0.0083 | +0.0000 | yes |
| spa_Latn | high | 5 | 214 | +0.0024 | +0.0000 | yes |
| fra_Latn | high | 5 | 169 | +0.0109 | +0.0000 | yes |
| por_Latn | high | 4 | 72 | +0.0076 | +0.0015 | yes |
| rus_Cyrl | high | 4 | 82 | +0.0033 | +0.0000 | yes |
| zho_Hans | high | 5 | 461 | +0.0028 | +0.0030 | yes |
| jpn_Jpan | high | 5 | 512 | -0.0015 | +0.0030 | yes |
| kor_Hang | high | 4 | 215 | +0.0183 | +0.0104 | yes |
| hin_Deva | mid | 4 | 146 | +0.0095 | +0.0030 | yes |
| ben_Beng | mid | 3 | 108 | +0.0098 | +0.0134 | yes |
| tam_Taml | mid | 3 | 209 | +0.0028 | +0.0134 | yes |
| tur_Latn | mid | 4 | 317 | -0.0023 | +0.0000 | yes |
| vie_Latn | mid | 4 | 297 | +0.0086 | +0.0039 | yes |
| ind_Latn | mid | 3 | 139 | +0.0099 | +0.0030 | yes |
| tha_Thai | mid | 3 | 329 | +0.0036 | +0.0104 | yes |
| pol_Latn | mid | 4 | 137 | -0.0009 | +0.0000 | yes |
| ukr_Cyrl | mid | 3 | 84 | +0.0063 | +0.0134 | yes |
| kan_Knda | low | 1 | 160 | +0.0099 | +0.0039 | yes |
| tel_Telu | low | 1 | 135 | +0.0154 | +0.0000 | yes |
| yor_Latn | low | 2 | 453 | +0.0180 | +0.0000 | yes |
| sun_Latn | low | 1 | 250 | +0.0173 | +0.0085 | yes |
| amh_Ethi | low | 2 | 188 | +0.0196 | +0.0085 | yes |
| ibo_Latn | low | 1 | 296 | +0.0191 | +0.0085 | yes |
| hau_Latn | low | 2 | 239 | +0.0011 | +0.0000 | yes |
| khm_Khmr | low | 1 | 444 | +0.0099 | +0.0030 | yes |
| lao_Laoo | low | 2 | 464 | +0.0106 | +0.0000 | yes |

## Exploratory analyses (not part of the confirmatory family; p-values are uncorrected)

| width | selective minus control | 95% CI | p (uncorrected) |
|---|---|---|---|
| INT5 | -0.0226 | [-0.0435, -0.0009] | 0.0451 |
| INT4 | +0.0042 | [-0.0129, +0.0203] | 0.6292 |

Mixed-effects logistic model (random intercept per feature, quadrature ML; 10905 features, sigma 2.703, converged True):

| term | estimate | SE | p |
|---|---|---|---|
| Intercept | +2.5904 | 0.3755 | 5.242e-12 |
| C(stratum)[T.1] | +1.7783 | 0.2317 | 1.647e-14 |
| C(stratum)[T.2] | +2.4893 | 0.2907 | 1.112e-17 |
| C(stratum)[T.3] | +2.7170 | 0.3075 | 9.861e-19 |
| C(stratum)[T.4] | +3.8949 | 0.3861 | 6.297e-24 |
| bits_c | +4.5690 | 0.3976 | 1.464e-30 |
| S_rank | +1.1940 | 0.3393 | 0.0004337 |
| tier | +1.4547 | 0.2034 | 8.653e-13 |
| S_rank:tier | +0.9431 | 0.2932 | 0.001298 |

P1 (selective features fire less): mean panel firing rate selective 0.00659, other supported 0.00898, Mann-Whitney p 0.

## Instrument check: survival of the same features at the noise-floor widths INT16, INT12 and restore-only INT0 (all should be at the ceiling)

| width | selective | control |
|---|---|---|
| INT16 | 0.9994 | 1.0000 |
| INT12 | 0.9996 | 1.0000 |
| INT0 | 1.0000 | 1.0000 |
