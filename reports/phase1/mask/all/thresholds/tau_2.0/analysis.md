# Phase 1 analysis, regime `mask` (decision D-014)

10000 permutations, 2000 bootstrap draws, seed 0.

## Confirmatory family (Holm over three tests)

| test | estimate | 95% CI / detail | p | Holm p |
|---|---|---|---|---|
| INT7 selective minus matched-universal survival | -0.0037 | [-0.0073, +0.0004] (selective 0.9917, control 0.9974, n 7956+3513) | 0.2267 | 0.4534 |
| INT6 selective minus matched-universal survival | -0.0189 | [-0.0328, -0.0041] (selective 0.8935, control 0.9382, n 7956+3513) | 0.0273 | 0.08189 |
| language-level tier slope (median-r gap, INT7+INT6, one-sided) | -0.0034 | languages used 27 (need 20: ok); High minus Low -0.004350240031878154 | 0.9658 | 0.9658 |

TOST (90% bootstrap interval vs bound): slope -0.0059..-0.0008, equivalent within +-0.02: True; High minus Low -0.0087..-0.0007, equivalent: True.

### Per-language gaps (median r of selective minus matched controls, mean of INT7 and INT6)

| language | tier | Joshi | selective features | gap (median r) | gap (survival rate) | used |
|---|---|---|---|---|---|---|
| eng_Latn | high | 5 | 1574 | -0.0045 | +0.0049 | yes |
| deu_Latn | high | 5 | 212 | +0.0133 | +0.0022 | yes |
| spa_Latn | high | 5 | 183 | +0.0037 | +0.0000 | yes |
| fra_Latn | high | 5 | 174 | +0.0098 | +0.0009 | yes |
| por_Latn | high | 4 | 67 | +0.0122 | +0.0049 | yes |
| rus_Cyrl | high | 4 | 72 | +0.0109 | +0.0000 | yes |
| zho_Hans | high | 5 | 422 | +0.0041 | +0.0000 | yes |
| jpn_Jpan | high | 5 | 476 | +0.0006 | +0.0049 | yes |
| kor_Hang | high | 4 | 219 | +0.0158 | +0.0185 | yes |
| hin_Deva | mid | 4 | 135 | +0.0114 | +0.0045 | yes |
| ben_Beng | mid | 3 | 104 | +0.0131 | +0.0117 | yes |
| tam_Taml | mid | 3 | 206 | +0.0099 | +0.0185 | yes |
| tur_Latn | mid | 4 | 281 | -0.0004 | +0.0004 | yes |
| vie_Latn | mid | 4 | 278 | +0.0126 | +0.0185 | yes |
| ind_Latn | mid | 3 | 127 | +0.0151 | +0.0049 | yes |
| tha_Thai | mid | 3 | 339 | +0.0040 | +0.0185 | yes |
| pol_Latn | mid | 4 | 121 | +0.0054 | +0.0000 | yes |
| ukr_Cyrl | mid | 3 | 71 | +0.0115 | +0.0185 | yes |
| kan_Knda | low | 1 | 170 | +0.0100 | +0.0057 | yes |
| tel_Telu | low | 1 | 149 | +0.0125 | +0.0049 | yes |
| yor_Latn | low | 2 | 498 | +0.0231 | +0.0004 | yes |
| sun_Latn | low | 1 | 266 | +0.0091 | +0.0089 | yes |
| amh_Ethi | low | 2 | 205 | +0.0064 | +0.0089 | yes |
| ibo_Latn | low | 1 | 335 | +0.0187 | +0.0142 | yes |
| hau_Latn | low | 2 | 239 | +0.0058 | +0.0004 | yes |
| khm_Khmr | low | 1 | 492 | +0.0104 | +0.0098 | yes |
| lao_Laoo | low | 2 | 541 | +0.0090 | +0.0004 | yes |

## Exploratory analyses (not part of the confirmatory family; p-values are uncorrected)

| width | selective minus control | 95% CI | p (uncorrected) |
|---|---|---|---|
| INT5 | -0.0046 | [-0.0259, +0.0174] | 0.6942 |
| INT4 | +0.0229 | [+0.0065, +0.0385] | 0.007199 |

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

P1 (selective features fire less): mean panel firing rate selective 0.00594, other supported 0.00853, Mann-Whitney p 0.

## Instrument check: survival of the same features at the noise-floor widths INT16, INT12 and restore-only INT0 (all should be at the ceiling)

| width | selective | control |
|---|---|---|
| INT16 | 0.9992 | 1.0000 |
| INT12 | 0.9995 | 1.0000 |
| INT0 | 1.0000 | 1.0000 |
