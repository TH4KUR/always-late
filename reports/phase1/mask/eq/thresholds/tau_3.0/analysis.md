# Phase 1 analysis, regime `mask` (decision D-014)

10000 permutations, 2000 bootstrap draws, seed 0.

## Confirmatory family (Holm over three tests)

| test | estimate | 95% CI / detail | p | Holm p |
|---|---|---|---|---|
| INT7 selective minus matched-universal survival | -0.0042 | [-0.0088, +0.0007] (selective 0.9895, control 0.9958, n 3917+3114) | 0.1295 | 0.259 |
| INT6 selective minus matched-universal survival | -0.0214 | [-0.0352, -0.0074] (selective 0.8892, control 0.9226, n 3917+3114) | 0.005499 | 0.0165 |
| language-level tier slope (median-r gap, INT7+INT6, one-sided) | +0.0020 | languages used 27 (need 20: ok); High minus Low 0.00013375778992970842 | 0.1437 | 0.259 |

TOST (90% bootstrap interval vs bound): slope -0.0014..+0.0035, equivalent within +-0.02: True; High minus Low -0.0046..+0.0033, equivalent: True.

### Per-language gaps (median r of selective minus matched controls, mean of INT7 and INT6)

| language | tier | Joshi | selective features | gap (median r) | gap (survival rate) | used |
|---|---|---|---|---|---|---|
| eng_Latn | high | 5 | 437 | +0.0029 | +0.0067 | yes |
| deu_Latn | high | 5 | 101 | +0.0171 | -0.0063 | yes |
| spa_Latn | high | 5 | 68 | +0.0094 | -0.0004 | yes |
| fra_Latn | high | 5 | 96 | +0.0116 | +0.0000 | yes |
| por_Latn | high | 4 | 29 | +0.0145 | +0.0000 | yes |
| rus_Cyrl | high | 4 | 22 | +0.0116 | -0.0067 | yes |
| zho_Hans | high | 5 | 193 | +0.0077 | +0.0000 | yes |
| jpn_Jpan | high | 5 | 164 | +0.0078 | +0.0067 | yes |
| kor_Hang | high | 4 | 111 | +0.0076 | +0.0000 | yes |
| hin_Deva | mid | 4 | 72 | +0.0132 | +0.0063 | yes |
| ben_Beng | mid | 3 | 58 | +0.0131 | +0.0063 | yes |
| tam_Taml | mid | 3 | 128 | +0.0071 | +0.0067 | yes |
| tur_Latn | mid | 4 | 78 | +0.0108 | +0.0049 | yes |
| vie_Latn | mid | 4 | 135 | +0.0119 | +0.0063 | yes |
| ind_Latn | mid | 3 | 53 | +0.0111 | +0.0067 | yes |
| tha_Thai | mid | 3 | 191 | +0.0105 | +0.0000 | yes |
| pol_Latn | mid | 4 | 36 | +0.0151 | +0.0000 | yes |
| ukr_Cyrl | mid | 3 | 36 | +0.0111 | +0.0000 | yes |
| kan_Knda | low | 1 | 99 | +0.0144 | +0.0000 | yes |
| tel_Telu | low | 1 | 86 | +0.0161 | +0.0067 | yes |
| yor_Latn | low | 2 | 368 | -0.0028 | -0.0004 | yes |
| sun_Latn | low | 1 | 143 | +0.0042 | +0.0192 | yes |
| amh_Ethi | low | 2 | 137 | +0.0163 | +0.0197 | yes |
| ibo_Latn | low | 1 | 215 | +0.0188 | +0.0214 | yes |
| hau_Latn | low | 2 | 130 | -0.0015 | +0.0201 | yes |
| khm_Khmr | low | 1 | 352 | +0.0128 | +0.0000 | yes |
| lao_Laoo | low | 2 | 379 | +0.0106 | +0.0197 | yes |

## Exploratory analyses (not part of the confirmatory family; p-values are uncorrected)

| width | selective minus control | 95% CI | p (uncorrected) |
|---|---|---|---|
| INT5 | +0.0262 | [+0.0027, +0.0492] | 0.0315 |
| INT4 | +0.0248 | [+0.0078, +0.0418] | 0.009999 |

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

P1 (selective features fire less): mean panel firing rate selective 0.00579, other supported 0.00801, Mann-Whitney p 0.

## Instrument check: survival of the same features at the noise-floor widths INT16, INT12 and restore-only INT0 (all should be at the ceiling)

| width | selective | control |
|---|---|---|
| INT16 | 0.9987 | 1.0000 |
| INT12 | 0.9992 | 1.0000 |
| INT0 | 1.0000 | 1.0000 |
