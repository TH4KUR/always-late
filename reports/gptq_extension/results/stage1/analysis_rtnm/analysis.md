# Phase 1 analysis, regime `mask` (decision D-014)

10000 permutations, 2000 bootstrap draws, seed 0.

## Confirmatory family (Holm over three tests)

| test | estimate | 95% CI / detail | p | Holm p |
|---|---|---|---|---|
| INT7 selective minus matched-universal survival | -0.0048 | [-0.0068, -0.0023] (selective 0.9944, control 0.9996, n 8062+2843) | 0.0211 | 0.0422 |
| INT6 selective minus matched-universal survival | -0.0153 | [-0.0259, -0.0042] (selective 0.9405, control 0.9676, n 8062+2843) | 0.0128 | 0.0384 |
| language-level tier slope (median-r gap, INT7+INT6, one-sided) | -0.0035 | languages used 27 (need 20: ok); High minus Low -0.0053544458415773185 | 0.9838 | 0.9838 |

TOST (90% bootstrap interval vs bound): slope -0.0067..-0.0012, equivalent within +-0.02: True; High minus Low -0.0097..-0.0014, equivalent: True.

### Per-language gaps (median r of selective minus matched controls, mean of INT7 and INT6)

| language | tier | Joshi | selective features | gap (median r) | gap (survival rate) | used |
|---|---|---|---|---|---|---|
| eng_Latn | high | 5 | 1711 | -0.0049 | +0.0000 | yes |
| deu_Latn | high | 5 | 231 | +0.0065 | +0.0000 | yes |
| spa_Latn | high | 5 | 214 | +0.0026 | +0.0000 | yes |
| fra_Latn | high | 5 | 169 | +0.0093 | +0.0000 | yes |
| por_Latn | high | 4 | 72 | +0.0058 | +0.0000 | yes |
| rus_Cyrl | high | 4 | 82 | +0.0037 | +0.0000 | yes |
| zho_Hans | high | 5 | 461 | +0.0023 | +0.0053 | yes |
| jpn_Jpan | high | 5 | 512 | -0.0005 | +0.0053 | yes |
| kor_Hang | high | 4 | 215 | +0.0135 | +0.0072 | yes |
| hin_Deva | mid | 4 | 146 | +0.0066 | +0.0053 | yes |
| ben_Beng | mid | 3 | 108 | +0.0076 | +0.0072 | yes |
| tam_Taml | mid | 3 | 209 | +0.0018 | +0.0124 | yes |
| tur_Latn | mid | 4 | 317 | -0.0014 | +0.0000 | yes |
| vie_Latn | mid | 4 | 297 | +0.0078 | +0.0124 | yes |
| ind_Latn | mid | 3 | 139 | +0.0077 | +0.0053 | yes |
| tha_Thai | mid | 3 | 329 | +0.0023 | +0.0072 | yes |
| pol_Latn | mid | 4 | 137 | -0.0007 | +0.0000 | yes |
| ukr_Cyrl | mid | 3 | 84 | +0.0072 | +0.0124 | yes |
| kan_Knda | low | 1 | 160 | +0.0110 | +0.0053 | yes |
| tel_Telu | low | 1 | 135 | +0.0102 | +0.0000 | yes |
| yor_Latn | low | 2 | 453 | +0.0158 | +0.0066 | yes |
| sun_Latn | low | 1 | 250 | +0.0145 | +0.0073 | yes |
| amh_Ethi | low | 2 | 188 | +0.0155 | +0.0073 | yes |
| ibo_Latn | low | 1 | 296 | +0.0131 | +0.0086 | yes |
| hau_Latn | low | 2 | 239 | -0.0001 | +0.0073 | yes |
| khm_Khmr | low | 1 | 444 | +0.0052 | +0.0046 | yes |
| lao_Laoo | low | 2 | 464 | +0.0013 | +0.0066 | yes |

## Exploratory analyses (not part of the confirmatory family; p-values are uncorrected)

| width | selective minus control | 95% CI | p (uncorrected) |
|---|---|---|---|
| INT5 | -0.0223 | [-0.0419, -0.0019] | 0.0376 |
| INT4 | +0.0064 | [-0.0126, +0.0255] | 0.5268 |

Mixed-effects logistic model (random intercept per feature, quadrature ML; 10905 features, sigma 1.957, converged True):

| term | estimate | SE | p |
|---|---|---|---|
| Intercept | +2.4657 | 0.3405 | 4.418e-13 |
| C(stratum)[T.1] | +1.5909 | 0.1960 | 4.808e-16 |
| C(stratum)[T.2] | +2.1596 | 0.2404 | 2.622e-19 |
| C(stratum)[T.3] | +2.6209 | 0.2670 | 9.639e-23 |
| C(stratum)[T.4] | +3.9845 | 0.3417 | 1.989e-31 |
| bits_c | +3.5088 | 0.2622 | 7.455e-41 |
| S_rank | +0.8108 | 0.3172 | 0.01059 |
| tier | +1.2800 | 0.1911 | 2.104e-11 |
| S_rank:tier | +1.1323 | 0.3136 | 0.0003056 |

P1 (selective features fire less): mean panel firing rate selective 0.00659, other supported 0.00898, Mann-Whitney p 0.

## Instrument check: survival of the same features at the noise-floor widths INT16, INT12 and restore-only INT0 (all should be at the ceiling)

| width | selective | control |
|---|---|---|
| INT16 | 0.9994 | 1.0000 |
| INT12 | 0.9996 | 1.0000 |
| INT0 | 1.0000 | 1.0000 |
