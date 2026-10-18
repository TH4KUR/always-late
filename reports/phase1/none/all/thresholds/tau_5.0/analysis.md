# Phase 1 analysis, regime `none` (decision D-014)

10000 permutations, 2000 bootstrap draws, seed 0.

## Confirmatory family (Holm over three tests)

| test | estimate | 95% CI / detail | p | Holm p |
|---|---|---|---|---|
| INT7 selective minus matched-universal survival | +0.1408 | [+0.1098, +0.1720] (selective 0.8015, control 0.6632, n 1491+1336) | 9.999e-05 | 0.0003 |
| INT6 selective minus matched-universal survival | +0.1272 | [+0.0932, +0.1598] (selective 0.5292, control 0.4019, n 1491+1336) | 9.999e-05 | 0.0003 |
| language-level tier slope (median-r gap, INT7+INT6, one-sided) | -0.0171 | languages used 26 (need 20: ok); High minus Low -0.006611448195245527 | 0.9615 | 0.9615 |

TOST (90% bootstrap interval vs bound): slope -0.0265..-0.0018, equivalent within +-0.02: False; High minus Low -0.0203..+0.0147, equivalent: False.

### Per-language gaps (median r of selective minus matched controls, mean of INT7 and INT6)

| language | tier | Joshi | selective features | gap (median r) | gap (survival rate) | used |
|---|---|---|---|---|---|---|
| eng_Latn | high | 5 | 329 | +0.0248 | +0.0199 | yes |
| deu_Latn | high | 5 | 38 | +0.0320 | +0.0365 | yes |
| spa_Latn | high | 5 | 30 | +0.0480 | -0.0039 | yes |
| fra_Latn | high | 5 | 35 | +0.0795 | +0.4823 | yes |
| por_Latn | high | 4 | 27 | +0.0735 | +0.5779 | yes |
| rus_Cyrl | high | 4 | 17 | +0.1127 | +0.4760 | yes |
| zho_Hans | high | 5 | 26 | +0.0350 | +0.0244 | yes |
| jpn_Jpan | high | 5 | 44 | +0.0373 | +0.2543 | yes |
| kor_Hang | high | 4 | 31 | +0.0070 | +0.0225 | yes |
| hin_Deva | mid | 4 | 14 | +0.0620 | +0.4962 | yes |
| ben_Beng | mid | 3 | 20 | +0.0269 | +0.4959 | yes |
| tam_Taml | mid | 3 | 29 | +0.0238 | +0.0317 | yes |
| tur_Latn | mid | 4 | 18 | -0.0110 | -0.0045 | yes |
| vie_Latn | mid | 4 | 23 | +0.0438 | +0.0064 | yes |
| ind_Latn | mid | 3 | 19 | +0.1036 | +0.5199 | yes |
| tha_Thai | mid | 3 | 30 | +0.0204 | +0.0045 | yes |
| pol_Latn | mid | 4 | 7 | +0.0651 | +0.4667 | no |
| ukr_Cyrl | mid | 3 | 24 | +0.1005 | +0.5011 | yes |
| kan_Knda | low | 1 | 36 | +0.0386 | +0.4977 | yes |
| tel_Telu | low | 1 | 27 | +0.0278 | +0.0045 | yes |
| yor_Latn | low | 2 | 109 | +0.0694 | +0.0817 | yes |
| sun_Latn | low | 1 | 22 | +0.1221 | +0.5902 | yes |
| amh_Ethi | low | 2 | 56 | +0.0020 | +0.0000 | yes |
| ibo_Latn | low | 1 | 102 | +0.0686 | +0.5284 | yes |
| hau_Latn | low | 2 | 24 | +0.1064 | +0.5779 | yes |
| khm_Khmr | low | 1 | 157 | +0.0504 | +0.4935 | yes |
| lao_Laoo | low | 2 | 197 | +0.0240 | +0.0787 | yes |

## Exploratory analyses (not part of the confirmatory family; p-values are uncorrected)

| width | selective minus control | 95% CI | p (uncorrected) |
|---|---|---|---|
| INT5 | +0.0967 | [+0.0680, +0.1263] | 9.999e-05 |
| INT4 | +0.0580 | [+0.0371, +0.0777] | 9.999e-05 |

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

P1 (selective features fire less): mean panel firing rate selective 0.00239, other supported 0.01887, Mann-Whitney p 0.

## Instrument check: survival of the same features at the noise-floor widths INT16, INT12 and restore-only INT0 (all should be at the ceiling)

| width | selective | control |
|---|---|---|
| INT16 | 0.9966 | 0.9970 |
| INT12 | 0.9973 | 0.9978 |
| INT0 | 1.0000 | 1.0000 |
