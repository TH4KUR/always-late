# Phase 1 analysis, regime `mask` (decision D-014)

10000 permutations, 2000 bootstrap draws, seed 0.

## Confirmatory family (Holm over three tests)

| test | estimate | 95% CI / detail | p | Holm p |
|---|---|---|---|---|
| INT7 selective minus matched-universal survival | -0.0017 | [-0.0039, +0.0007] (selective 0.9967, control 0.9987, n 5175+3931) | 0.2902 | 0.5803 |
| INT6 selective minus matched-universal survival | -0.0146 | [-0.0220, -0.0071] (selective 0.9631, control 0.9837, n 5175+3931) | 0.003 | 0.008999 |
| language-level tier slope (median-r gap, INT7+INT6, one-sided) | -0.0001 | languages used 27 (need 20: ok); High minus Low -0.0005185902118682861 | 0.5338 | 0.5803 |

TOST (90% bootstrap interval vs bound): slope -0.0014..+0.0013, equivalent within +-0.02: True; High minus Low -0.0029..+0.0016, equivalent: True.

### Per-language gaps (median r of selective minus matched controls, mean of INT7 and INT6)

| language | tier | Joshi | selective features | gap (median r) | gap (survival rate) | used |
|---|---|---|---|---|---|---|
| eng_Latn | high | 5 | 1246 | -0.0029 | +0.0000 | yes |
| deu_Latn | high | 5 | 112 | +0.0034 | +0.0000 | yes |
| spa_Latn | high | 5 | 137 | +0.0015 | +0.0000 | yes |
| fra_Latn | high | 5 | 118 | +0.0047 | +0.0075 | yes |
| por_Latn | high | 4 | 59 | +0.0038 | +0.0075 | yes |
| rus_Cyrl | high | 4 | 70 | +0.0016 | +0.0000 | yes |
| zho_Hans | high | 5 | 333 | +0.0014 | +0.0075 | yes |
| jpn_Jpan | high | 5 | 394 | +0.0001 | +0.0009 | yes |
| kor_Hang | high | 4 | 106 | +0.0070 | +0.0075 | yes |
| hin_Deva | mid | 4 | 81 | -0.0030 | +0.0005 | yes |
| ben_Beng | mid | 3 | 60 | +0.0034 | +0.0075 | yes |
| tam_Taml | mid | 3 | 87 | +0.0024 | +0.0075 | yes |
| tur_Latn | mid | 4 | 139 | +0.0033 | +0.0000 | yes |
| vie_Latn | mid | 4 | 190 | +0.0033 | +0.0075 | yes |
| ind_Latn | mid | 3 | 105 | +0.0042 | +0.0000 | yes |
| tha_Thai | mid | 3 | 208 | +0.0060 | +0.0075 | yes |
| pol_Latn | mid | 4 | 76 | +0.0009 | +0.0000 | yes |
| ukr_Cyrl | mid | 3 | 57 | +0.0033 | +0.0075 | yes |
| kan_Knda | low | 1 | 75 | -0.0036 | +0.0075 | yes |
| tel_Telu | low | 1 | 65 | +0.0040 | +0.0093 | yes |
| yor_Latn | low | 2 | 369 | +0.0070 | +0.0025 | yes |
| sun_Latn | low | 1 | 181 | -0.0047 | +0.0003 | yes |
| amh_Ethi | low | 2 | 154 | +0.0036 | +0.0003 | yes |
| ibo_Latn | low | 1 | 149 | +0.0026 | +0.0000 | yes |
| hau_Latn | low | 2 | 142 | +0.0049 | +0.0000 | yes |
| khm_Khmr | low | 1 | 236 | +0.0067 | +0.0025 | yes |
| lao_Laoo | low | 2 | 226 | +0.0049 | +0.0020 | yes |

## Exploratory analyses (not part of the confirmatory family; p-values are uncorrected)

| width | selective minus control | 95% CI | p (uncorrected) |
|---|---|---|---|
| INT5 | -0.0439 | [-0.0610, -0.0258] | 9.999e-05 |
| INT4 | -0.0361 | [-0.0568, -0.0159] | 0.0008999 |

Mixed-effects logistic model (random intercept per feature, quadrature ML; 9704 features, sigma 2.760, converged True):

| term | estimate | SE | p |
|---|---|---|---|
| Intercept | +4.2088 | 0.6629 | 2.159e-10 |
| C(stratum)[T.1] | +1.7196 | 0.3294 | 1.782e-07 |
| C(stratum)[T.2] | +2.7480 | 0.4423 | 5.184e-10 |
| C(stratum)[T.3] | +4.0001 | 0.5831 | 6.859e-12 |
| C(stratum)[T.4] | +3.7332 | 0.5213 | 8.013e-13 |
| bits_c | +3.8052 | 0.4427 | 8.284e-18 |
| S_rank | +0.5156 | 0.5213 | 0.3227 |
| tier | +1.7596 | 0.3391 | 2.11e-07 |
| S_rank:tier | +0.5477 | 0.4755 | 0.2494 |

P1 (selective features fire less): mean panel firing rate selective 0.00517, other supported 0.00675, Mann-Whitney p 0.

## Instrument check: survival of the same features at the noise-floor widths INT16, INT12 and restore-only INT0 (all should be at the ceiling)

| width | selective | control |
|---|---|---|
| INT16 | 0.9996 | 0.9997 |
| INT12 | 0.9996 | 1.0000 |
| INT0 | 1.0000 | 1.0000 |
