# Phase 1 analysis, regime `none` (decision D-014)

10000 permutations, 2000 bootstrap draws, seed 0.

## Confirmatory family (Holm over three tests)

| test | estimate | 95% CI / detail | p | Holm p |
|---|---|---|---|---|
| INT7 selective minus matched-universal survival | +0.0874 | [+0.0500, +0.1208] (selective 0.7070, control 0.6717, n 5791+923) | 9.999e-05 | 0.0003 |
| INT6 selective minus matched-universal survival | +0.0408 | [+0.0045, +0.0753] (selective 0.4196, control 0.3933, n 5791+923) | 0.0368 | 0.07359 |
| language-level tier slope (median-r gap, INT7+INT6, one-sided) | +0.0080 | languages used 27 (need 20: ok); High minus Low 0.02559556398126814 | 0.2364 | 0.2364 |

TOST (90% bootstrap interval vs bound): slope -0.0071..+0.0182, equivalent within +-0.02: True; High minus Low +0.0026..+0.0391, equivalent: False.

### Per-language gaps (median r of selective minus matched controls, mean of INT7 and INT6)

| language | tier | Joshi | selective features | gap (median r) | gap (survival rate) | used |
|---|---|---|---|---|---|---|
| eng_Latn | high | 5 | 805 | -0.0152 | +0.0425 | yes |
| deu_Latn | high | 5 | 168 | +0.0234 | +0.0888 | yes |
| spa_Latn | high | 5 | 124 | +0.0551 | +0.5475 | yes |
| fra_Latn | high | 5 | 159 | +0.0391 | +0.0704 | yes |
| por_Latn | high | 4 | 115 | +0.0463 | +0.1040 | yes |
| rus_Cyrl | high | 4 | 121 | +0.0176 | +0.0000 | yes |
| zho_Hans | high | 5 | 231 | +0.0280 | +0.0000 | yes |
| jpn_Jpan | high | 5 | 248 | -0.0182 | -0.4682 | yes |
| kor_Hang | high | 4 | 192 | +0.0434 | +0.5615 | yes |
| hin_Deva | mid | 4 | 70 | +0.0691 | +0.5432 | yes |
| ben_Beng | mid | 3 | 123 | -0.0051 | +0.0984 | yes |
| tam_Taml | mid | 3 | 202 | +0.0082 | +0.0882 | yes |
| tur_Latn | mid | 4 | 94 | +0.0063 | +0.0000 | yes |
| vie_Latn | mid | 4 | 186 | +0.0298 | +0.4279 | yes |
| ind_Latn | mid | 3 | 107 | +0.0871 | +0.5868 | yes |
| tha_Thai | mid | 3 | 161 | +0.0671 | +0.5761 | yes |
| pol_Latn | mid | 4 | 82 | +0.0645 | +0.1167 | yes |
| ukr_Cyrl | mid | 3 | 129 | +0.0577 | +0.1097 | yes |
| kan_Knda | low | 1 | 158 | -0.0117 | +0.0699 | yes |
| tel_Telu | low | 1 | 117 | +0.0418 | +0.5811 | yes |
| yor_Latn | low | 2 | 348 | +0.0320 | +0.0000 | yes |
| sun_Latn | low | 1 | 188 | +0.0330 | +0.0279 | yes |
| amh_Ethi | low | 2 | 222 | -0.0422 | -0.4873 | yes |
| ibo_Latn | low | 1 | 352 | +0.0071 | +0.0000 | yes |
| hau_Latn | low | 2 | 174 | +0.0039 | +0.0392 | yes |
| khm_Khmr | low | 1 | 422 | -0.0247 | -0.0279 | yes |
| lao_Laoo | low | 2 | 493 | -0.0499 | -0.4870 | yes |

## Exploratory analyses (not part of the confirmatory family; p-values are uncorrected)

| width | selective minus control | 95% CI | p (uncorrected) |
|---|---|---|---|
| INT5 | +0.0409 | [+0.0114, +0.0668] | 0.008299 |
| INT4 | +0.0321 | [+0.0162, +0.0477] | 0.0018 |

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

P1 (selective features fire less): mean panel firing rate selective 0.00794, other supported 0.05946, Mann-Whitney p 2.099e-31.

## Instrument check: survival of the same features at the noise-floor widths INT16, INT12 and restore-only INT0 (all should be at the ceiling)

| width | selective | control |
|---|---|---|
| INT16 | 0.9984 | 0.9978 |
| INT12 | 0.9983 | 0.9989 |
| INT0 | 1.0000 | 1.0000 |
