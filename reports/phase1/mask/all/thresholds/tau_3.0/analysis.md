# Phase 1 analysis, regime `mask` (decision D-014)

10000 permutations, 2000 bootstrap draws, seed 0.

## Confirmatory family (Holm over three tests)

| test | estimate | 95% CI / detail | p | Holm p |
|---|---|---|---|---|
| INT7 selective minus matched-universal survival | -0.0075 | [-0.0124, -0.0021] (selective 0.9865, control 0.9963, n 4281+3246) | 0.0235 | 0.047 |
| INT6 selective minus matched-universal survival | -0.0404 | [-0.0564, -0.0248] (selective 0.8570, control 0.9217, n 4281+3246) | 9.999e-05 | 0.0003 |
| language-level tier slope (median-r gap, INT7+INT6, one-sided) | +0.0024 | languages used 27 (need 20: ok); High minus Low 0.004849495159255134 | 0.101 | 0.101 |

TOST (90% bootstrap interval vs bound): slope -0.0001..+0.0045, equivalent within +-0.02: True; High minus Low +0.0009..+0.0081, equivalent: True.

### Per-language gaps (median r of selective minus matched controls, mean of INT7 and INT6)

| language | tier | Joshi | selective features | gap (median r) | gap (survival rate) | used |
|---|---|---|---|---|---|---|
| eng_Latn | high | 5 | 434 | +0.0043 | +0.0138 | yes |
| deu_Latn | high | 5 | 104 | +0.0188 | -0.0138 | yes |
| spa_Latn | high | 5 | 64 | +0.0097 | +0.0000 | yes |
| fra_Latn | high | 5 | 106 | +0.0112 | +0.0000 | yes |
| por_Latn | high | 4 | 31 | +0.0150 | +0.0138 | yes |
| rus_Cyrl | high | 4 | 21 | +0.0146 | +0.0000 | yes |
| zho_Hans | high | 5 | 193 | +0.0097 | +0.0131 | yes |
| jpn_Jpan | high | 5 | 169 | +0.0087 | +0.0138 | yes |
| kor_Hang | high | 4 | 111 | +0.0125 | +0.0000 | yes |
| hin_Deva | mid | 4 | 76 | +0.0099 | +0.0065 | yes |
| ben_Beng | mid | 3 | 62 | +0.0180 | +0.0131 | yes |
| tam_Taml | mid | 3 | 137 | +0.0121 | +0.0138 | yes |
| tur_Latn | mid | 4 | 78 | +0.0119 | +0.0025 | yes |
| vie_Latn | mid | 4 | 144 | +0.0129 | +0.0131 | yes |
| ind_Latn | mid | 3 | 56 | +0.0142 | +0.0069 | yes |
| tha_Thai | mid | 3 | 205 | +0.0031 | +0.0000 | yes |
| pol_Latn | mid | 4 | 36 | +0.0158 | -0.0013 | yes |
| ukr_Cyrl | mid | 3 | 40 | +0.0089 | +0.0000 | yes |
| kan_Knda | low | 1 | 114 | +0.0099 | +0.0150 | yes |
| tel_Telu | low | 1 | 100 | +0.0099 | +0.0143 | yes |
| yor_Latn | low | 2 | 417 | +0.0075 | +0.0007 | yes |
| sun_Latn | low | 1 | 165 | -0.0013 | +0.0157 | yes |
| amh_Ethi | low | 2 | 152 | -0.0015 | +0.0170 | yes |
| ibo_Latn | low | 1 | 262 | +0.0182 | +0.0091 | yes |
| hau_Latn | low | 2 | 144 | -0.0016 | +0.0091 | yes |
| khm_Khmr | low | 1 | 411 | +0.0092 | +0.0176 | yes |
| lao_Laoo | low | 2 | 449 | +0.0104 | +0.0007 | yes |

## Exploratory analyses (not part of the confirmatory family; p-values are uncorrected)

| width | selective minus control | 95% CI | p (uncorrected) |
|---|---|---|---|
| INT5 | +0.0170 | [-0.0052, +0.0394] | 0.164 |
| INT4 | +0.0332 | [+0.0152, +0.0514] | 9.999e-05 |

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

P1 (selective features fire less): mean panel firing rate selective 0.00501, other supported 0.00778, Mann-Whitney p 0.

## Instrument check: survival of the same features at the noise-floor widths INT16, INT12 and restore-only INT0 (all should be at the ceiling)

| width | selective | control |
|---|---|---|
| INT16 | 0.9986 | 1.0000 |
| INT12 | 0.9991 | 1.0000 |
| INT0 | 1.0000 | 1.0000 |
