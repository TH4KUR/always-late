# Phase 1 analysis, regime `mask` (decision D-014)

10000 permutations, 2000 bootstrap draws, seed 0.

## Confirmatory family (Holm over three tests)

| test | estimate | 95% CI / detail | p | Holm p |
|---|---|---|---|---|
| INT7 selective minus matched-universal survival | -0.0004 | [-0.0012, +0.0004] (selective 0.9994, control 0.9997, n 5175+3931) | 0.7934 | 0.9698 |
| INT6 selective minus matched-universal survival | -0.0022 | [-0.0051, +0.0011] (selective 0.9940, control 0.9967, n 5175+3931) | 0.3233 | 0.9698 |
| language-level tier slope (median-r gap, INT7+INT6, one-sided) | +0.0002 | languages used 27 (need 20: ok); High minus Low 1.0662608676486425e-05 | 0.3917 | 0.9698 |

TOST (90% bootstrap interval vs bound): slope -0.0007..+0.0013, equivalent within +-0.02: True; High minus Low -0.0015..+0.0017, equivalent: True.

### Per-language gaps (median r of selective minus matched controls, mean of INT7 and INT6)

| language | tier | Joshi | selective features | gap (median r) | gap (survival rate) | used |
|---|---|---|---|---|---|---|
| eng_Latn | high | 5 | 1246 | -0.0022 | +0.0001 | yes |
| deu_Latn | high | 5 | 112 | +0.0027 | +0.0000 | yes |
| spa_Latn | high | 5 | 137 | +0.0013 | +0.0000 | yes |
| fra_Latn | high | 5 | 118 | +0.0036 | +0.0003 | yes |
| por_Latn | high | 4 | 59 | +0.0012 | +0.0003 | yes |
| rus_Cyrl | high | 4 | 70 | +0.0007 | +0.0000 | yes |
| zho_Hans | high | 5 | 333 | +0.0009 | +0.0003 | yes |
| jpn_Jpan | high | 5 | 394 | +0.0003 | +0.0006 | yes |
| kor_Hang | high | 4 | 106 | +0.0034 | +0.0003 | yes |
| hin_Deva | mid | 4 | 81 | -0.0015 | +0.0003 | yes |
| ben_Beng | mid | 3 | 60 | +0.0039 | +0.0003 | yes |
| tam_Taml | mid | 3 | 87 | +0.0024 | +0.0003 | yes |
| tur_Latn | mid | 4 | 139 | +0.0025 | +0.0000 | yes |
| vie_Latn | mid | 4 | 190 | +0.0019 | +0.0003 | yes |
| ind_Latn | mid | 3 | 105 | +0.0030 | +0.0000 | yes |
| tha_Thai | mid | 3 | 208 | +0.0041 | +0.0003 | yes |
| pol_Latn | mid | 4 | 76 | +0.0008 | +0.0000 | yes |
| ukr_Cyrl | mid | 3 | 57 | +0.0026 | +0.0003 | yes |
| kan_Knda | low | 1 | 75 | -0.0025 | +0.0003 | yes |
| tel_Telu | low | 1 | 65 | +0.0034 | +0.0009 | yes |
| yor_Latn | low | 2 | 369 | +0.0015 | +0.0015 | yes |
| sun_Latn | low | 1 | 181 | -0.0022 | +0.0000 | yes |
| amh_Ethi | low | 2 | 154 | +0.0018 | +0.0013 | yes |
| ibo_Latn | low | 1 | 149 | +0.0022 | +0.0000 | yes |
| hau_Latn | low | 2 | 142 | +0.0033 | +0.0000 | yes |
| khm_Khmr | low | 1 | 236 | +0.0019 | +0.0015 | yes |
| lao_Laoo | low | 2 | 226 | +0.0025 | +0.0013 | yes |

## Exploratory analyses (not part of the confirmatory family; p-values are uncorrected)

| width | selective minus control | 95% CI | p (uncorrected) |
|---|---|---|---|
| INT4 | -0.0261 | [-0.0490, -0.0043] | 0.0235 |

Mixed-effects logistic model (random intercept per feature, quadrature ML; 9704 features, sigma 1295.675, converged False):

| term | estimate | SE | p |
|---|---|---|---|
| Intercept | +2550.6164 | 505.6706 | 4.558e-07 |
| C(stratum)[T.1] | +541.8858 | 680.2260 | 0.4257 |
| C(stratum)[T.2] | +550.9059 | 695.3494 | 0.4282 |
| C(stratum)[T.3] | +1470.3623 | 800.4953 | 0.06624 |
| C(stratum)[T.4] | +1835.1616 | 290.3876 | 2.621e-10 |
| bits_c | +1410.9239 | 728.9644 | 0.05293 |
| S_rank | +589.7042 | 741.5471 | 0.4265 |
| tier | +713.0613 | 156.1939 | 4.989e-06 |
| S_rank:tier | -122.3365 | 212.7169 | 0.5652 |

P1 (selective features fire less): mean panel firing rate selective 0.00517, other supported 0.00675, Mann-Whitney p 0.

## Instrument check: survival of the same features at the noise-floor widths INT16, INT12 and restore-only INT0 (all should be at the ceiling)

| width | selective | control |
|---|---|---|
| INT16 | 0.9996 | 0.9997 |
| INT12 | 0.9996 | 1.0000 |
| INT0 | 1.0000 | 1.0000 |
