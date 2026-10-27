# Phase 1 analysis, regime `mask` (decision D-014)

10000 permutations, 2000 bootstrap draws, seed 0.

## Confirmatory family (Holm over three tests)

| test | estimate | 95% CI / detail | p | Holm p |
|---|---|---|---|---|
| INT7 selective minus matched-universal survival | -0.0014 | [-0.0022, -0.0006] (selective 0.9986, control 1.0000, n 8062+2843) | 0.38 | 1 |
| INT6 selective minus matched-universal survival | +0.0016 | [-0.0040, +0.0076] (selective 0.9901, control 0.9923, n 8062+2843) | 0.5741 | 1 |
| language-level tier slope (median-r gap, INT7+INT6, one-sided) | -0.0025 | languages used 27 (need 20: ok); High minus Low -0.0033741527133517796 | 0.9788 | 1 |

TOST (90% bootstrap interval vs bound): slope -0.0042..-0.0009, equivalent within +-0.02: True; High minus Low -0.0060..-0.0009, equivalent: True.

### Per-language gaps (median r of selective minus matched controls, mean of INT7 and INT6)

| language | tier | Joshi | selective features | gap (median r) | gap (survival rate) | used |
|---|---|---|---|---|---|---|
| eng_Latn | high | 5 | 1711 | -0.0036 | +0.0000 | yes |
| deu_Latn | high | 5 | 231 | +0.0049 | +0.0000 | yes |
| spa_Latn | high | 5 | 214 | +0.0012 | +0.0000 | yes |
| fra_Latn | high | 5 | 169 | +0.0063 | +0.0000 | yes |
| por_Latn | high | 4 | 72 | +0.0034 | +0.0000 | yes |
| rus_Cyrl | high | 4 | 82 | +0.0028 | +0.0000 | yes |
| zho_Hans | high | 5 | 461 | +0.0018 | +0.0000 | yes |
| jpn_Jpan | high | 5 | 512 | -0.0015 | +0.0000 | yes |
| kor_Hang | high | 4 | 215 | +0.0085 | +0.0023 | yes |
| hin_Deva | mid | 4 | 146 | +0.0032 | +0.0000 | yes |
| ben_Beng | mid | 3 | 108 | +0.0068 | +0.0023 | yes |
| tam_Taml | mid | 3 | 209 | +0.0014 | +0.0028 | yes |
| tur_Latn | mid | 4 | 317 | -0.0021 | +0.0005 | yes |
| vie_Latn | mid | 4 | 297 | +0.0051 | +0.0023 | yes |
| ind_Latn | mid | 3 | 139 | +0.0063 | +0.0005 | yes |
| tha_Thai | mid | 3 | 329 | +0.0018 | +0.0023 | yes |
| pol_Latn | mid | 4 | 137 | +0.0008 | +0.0005 | yes |
| ukr_Cyrl | mid | 3 | 84 | +0.0046 | +0.0023 | yes |
| kan_Knda | low | 1 | 160 | +0.0045 | +0.0000 | yes |
| tel_Telu | low | 1 | 135 | +0.0052 | +0.0005 | yes |
| yor_Latn | low | 2 | 453 | +0.0123 | +0.0000 | yes |
| sun_Latn | low | 1 | 250 | +0.0119 | +0.0018 | yes |
| amh_Ethi | low | 2 | 188 | +0.0086 | +0.0005 | yes |
| ibo_Latn | low | 1 | 296 | +0.0078 | +0.0000 | yes |
| hau_Latn | low | 2 | 239 | -0.0005 | +0.0000 | yes |
| khm_Khmr | low | 1 | 444 | +0.0017 | +0.0000 | yes |
| lao_Laoo | low | 2 | 464 | +0.0025 | +0.0000 | yes |

## Exploratory analyses (not part of the confirmatory family; p-values are uncorrected)

| width | selective minus control | 95% CI | p (uncorrected) |
|---|---|---|---|
| INT5 | -0.0062 | [-0.0229, +0.0110] | 0.4723 |
| INT4 | +0.0209 | [-0.0003, +0.0429] | 0.06689 |

Mixed-effects logistic model (random intercept per feature, quadrature ML; 10905 features, sigma 3.336, converged True):

| term | estimate | SE | p |
|---|---|---|---|
| Intercept | +6.1207 | 1.2043 | 3.731e-07 |
| C(stratum)[T.1] | +2.2798 | 0.5363 | 2.125e-05 |
| C(stratum)[T.2] | +3.8914 | 0.8106 | 1.581e-06 |
| C(stratum)[T.3] | +3.7079 | 0.7753 | 1.729e-06 |
| C(stratum)[T.4] | +6.8109 | 1.5016 | 5.741e-06 |
| bits_c | +3.3392 | 0.5754 | 6.489e-09 |
| S_rank | +1.8048 | 0.8993 | 0.04475 |
| tier | +0.9191 | 0.4693 | 0.05016 |
| S_rank:tier | +1.2021 | 0.7524 | 0.1101 |

P1 (selective features fire less): mean panel firing rate selective 0.00659, other supported 0.00898, Mann-Whitney p 0.

## Instrument check: survival of the same features at the noise-floor widths INT16, INT12 and restore-only INT0 (all should be at the ceiling)

| width | selective | control |
|---|---|---|
| INT16 | 0.9994 | 1.0000 |
| INT12 | 0.9996 | 1.0000 |
| INT0 | 1.0000 | 1.0000 |
