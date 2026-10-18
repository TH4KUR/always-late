# Phase 1 analysis, regime `mask` (decision D-014)

10000 permutations, 2000 bootstrap draws, seed 0.

## Confirmatory family (Holm over three tests)

| test | estimate | 95% CI / detail | p | Holm p |
|---|---|---|---|---|
| INT7 selective minus matched-universal survival | -0.0174 | [-0.0252, -0.0096] (selective 0.9744, control 0.9917, n 1913+1930) | 9.999e-05 | 0.0003 |
| INT6 selective minus matched-universal survival | -0.0824 | [-0.1027, -0.0619] (selective 0.7909, control 0.8725, n 1913+1930) | 9.999e-05 | 0.0003 |
| language-level tier slope (median-r gap, INT7+INT6, one-sided) | +0.0028 | languages used 25 (need 20: ok); High minus Low 0.0013628962139288579 | 0.2136 | 0.2136 |

TOST (90% bootstrap interval vs bound): slope -0.0004..+0.0063, equivalent within +-0.02: True; High minus Low -0.0034..+0.0062, equivalent: True.

### Per-language gaps (median r of selective minus matched controls, mean of INT7 and INT6)

| language | tier | Joshi | selective features | gap (median r) | gap (survival rate) | used |
|---|---|---|---|---|---|---|
| eng_Latn | high | 5 | 99 | +0.0086 | +0.1036 | yes |
| deu_Latn | high | 5 | 35 | +0.0196 | -0.0298 | yes |
| spa_Latn | high | 5 | 15 | +0.0078 | +0.0000 | yes |
| fra_Latn | high | 5 | 49 | +0.0133 | +0.0071 | yes |
| por_Latn | high | 4 | 12 | +0.0058 | +0.0518 | yes |
| rus_Cyrl | high | 4 | 3 | +0.0085 | -0.1291 | no |
| zho_Hans | high | 5 | 70 | +0.0060 | -0.1036 | yes |
| jpn_Jpan | high | 5 | 45 | +0.0135 | +0.0000 | yes |
| kor_Hang | high | 4 | 39 | +0.0067 | +0.0000 | yes |
| hin_Deva | mid | 4 | 27 | +0.0018 | +0.0000 | yes |
| ben_Beng | mid | 3 | 31 | +0.0188 | -0.0071 | yes |
| tam_Taml | mid | 3 | 58 | +0.0137 | -0.1036 | yes |
| tur_Latn | mid | 4 | 16 | +0.0024 | -0.0071 | yes |
| vie_Latn | mid | 4 | 51 | +0.0062 | -0.0143 | yes |
| ind_Latn | mid | 3 | 19 | +0.0083 | +0.0000 | yes |
| tha_Thai | mid | 3 | 58 | +0.0013 | -0.1036 | yes |
| pol_Latn | mid | 4 | 5 | +0.0073 | -0.0398 | no |
| ukr_Cyrl | mid | 3 | 10 | +0.0007 | -0.1036 | yes |
| kan_Knda | low | 1 | 52 | +0.0167 | +0.1177 | yes |
| tel_Telu | low | 1 | 41 | +0.0087 | -0.1036 | yes |
| yor_Latn | low | 2 | 229 | +0.0171 | -0.0047 | yes |
| sun_Latn | low | 1 | 81 | -0.0137 | -0.0071 | yes |
| amh_Ethi | low | 2 | 85 | +0.0110 | -0.0095 | yes |
| ibo_Latn | low | 1 | 154 | +0.0221 | +0.0071 | yes |
| hau_Latn | low | 2 | 65 | -0.0149 | +0.0042 | yes |
| khm_Khmr | low | 1 | 259 | +0.0104 | +0.0005 | yes |
| lao_Laoo | low | 2 | 305 | +0.0218 | +0.3722 | yes |

## Exploratory analyses (not part of the confirmatory family; p-values are uncorrected)

| width | selective minus control | 95% CI | p (uncorrected) |
|---|---|---|---|
| INT5 | -0.0428 | [-0.0713, -0.0152] | 0.007699 |
| INT4 | +0.0067 | [-0.0149, +0.0281] | 0.568 |

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

P1 (selective features fire less): mean panel firing rate selective 0.00336, other supported 0.00744, Mann-Whitney p 0.

## Instrument check: survival of the same features at the noise-floor widths INT16, INT12 and restore-only INT0 (all should be at the ceiling)

| width | selective | control |
|---|---|---|
| INT16 | 0.9969 | 1.0000 |
| INT12 | 0.9979 | 1.0000 |
| INT0 | 1.0000 | 1.0000 |
