# Phase 1 analysis, regime `mask` (decision D-014)

10000 permutations, 2000 bootstrap draws, seed 0.

## Confirmatory family (Holm over three tests)

| test | estimate | 95% CI / detail | p | Holm p |
|---|---|---|---|---|
| INT7 selective minus matched-universal survival | -0.0027 | [-0.0065, +0.0017] (selective 0.9932, control 0.9983, n 7446+3459) | 0.2685 | 0.5369 |
| INT6 selective minus matched-universal survival | -0.0102 | [-0.0232, +0.0024] (selective 0.9132, control 0.9399, n 7446+3459) | 0.1575 | 0.4725 |
| language-level tier slope (median-r gap, INT7+INT6, one-sided) | -0.0041 | languages used 27 (need 20: ok); High minus Low -0.005335031284226311 | 0.9881 | 0.9881 |

TOST (90% bootstrap interval vs bound): slope -0.0067..-0.0015, equivalent within +-0.02: True; High minus Low -0.0107..-0.0021, equivalent: True.

### Per-language gaps (median r of selective minus matched controls, mean of INT7 and INT6)

| language | tier | Joshi | selective features | gap (median r) | gap (survival rate) | used |
|---|---|---|---|---|---|---|
| eng_Latn | high | 5 | 1549 | -0.0046 | +0.0000 | yes |
| deu_Latn | high | 5 | 213 | +0.0092 | +0.0000 | yes |
| spa_Latn | high | 5 | 194 | +0.0025 | +0.0000 | yes |
| fra_Latn | high | 5 | 161 | +0.0112 | +0.0009 | yes |
| por_Latn | high | 4 | 63 | +0.0084 | +0.0000 | yes |
| rus_Cyrl | high | 4 | 69 | +0.0082 | +0.0000 | yes |
| zho_Hans | high | 5 | 414 | +0.0045 | +0.0039 | yes |
| jpn_Jpan | high | 5 | 453 | +0.0004 | +0.0030 | yes |
| kor_Hang | high | 4 | 204 | +0.0178 | +0.0134 | yes |
| hin_Deva | mid | 4 | 138 | +0.0121 | +0.0030 | yes |
| ben_Beng | mid | 3 | 99 | +0.0126 | +0.0104 | yes |
| tam_Taml | mid | 3 | 197 | +0.0019 | +0.0134 | yes |
| tur_Latn | mid | 4 | 259 | +0.0011 | +0.0000 | yes |
| vie_Latn | mid | 4 | 263 | +0.0115 | +0.0134 | yes |
| ind_Latn | mid | 3 | 120 | +0.0104 | +0.0030 | yes |
| tha_Thai | mid | 3 | 318 | +0.0040 | +0.0104 | yes |
| pol_Latn | mid | 4 | 115 | +0.0015 | +0.0000 | yes |
| ukr_Cyrl | mid | 3 | 74 | +0.0072 | +0.0134 | yes |
| kan_Knda | low | 1 | 153 | +0.0122 | +0.0039 | yes |
| tel_Telu | low | 1 | 130 | +0.0156 | +0.0005 | yes |
| yor_Latn | low | 2 | 444 | +0.0123 | -0.0042 | yes |
| sun_Latn | low | 1 | 237 | +0.0133 | +0.0085 | yes |
| amh_Ethi | low | 2 | 184 | +0.0018 | +0.0085 | yes |
| ibo_Latn | low | 1 | 285 | +0.0207 | +0.0114 | yes |
| hau_Latn | low | 2 | 223 | +0.0035 | +0.0087 | yes |
| khm_Khmr | low | 1 | 433 | +0.0125 | +0.0030 | yes |
| lao_Laoo | low | 2 | 454 | +0.0136 | +0.0000 | yes |

## Exploratory analyses (not part of the confirmatory family; p-values are uncorrected)

| width | selective minus control | 95% CI | p (uncorrected) |
|---|---|---|---|
| INT5 | -0.0067 | [-0.0284, +0.0146] | 0.5646 |
| INT4 | +0.0135 | [-0.0028, +0.0300] | 0.1143 |

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

P1 (selective features fire less): mean panel firing rate selective 0.00648, other supported 0.00879, Mann-Whitney p 0.

## Instrument check: survival of the same features at the noise-floor widths INT16, INT12 and restore-only INT0 (all should be at the ceiling)

| width | selective | control |
|---|---|---|
| INT16 | 0.9993 | 1.0000 |
| INT12 | 0.9996 | 1.0000 |
| INT0 | 1.0000 | 1.0000 |
