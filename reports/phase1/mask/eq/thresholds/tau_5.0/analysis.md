# Phase 1 analysis, regime `mask` (decision D-014)

10000 permutations, 2000 bootstrap draws, seed 0.

## Confirmatory family (Holm over three tests)

| test | estimate | 95% CI / detail | p | Holm p |
|---|---|---|---|---|
| INT7 selective minus matched-universal survival | -0.0093 | [-0.0174, -0.0017] (selective 0.9815, control 0.9909, n 1676+1651) | 0.0189 | 0.0378 |
| INT6 selective minus matched-universal survival | -0.0716 | [-0.0914, -0.0520] (selective 0.8413, control 0.9134, n 1676+1651) | 9.999e-05 | 0.0003 |
| language-level tier slope (median-r gap, INT7+INT6, one-sided) | +0.0033 | languages used 24 (need 20: ok); High minus Low 0.002695801771349377 | 0.1285 | 0.1285 |

TOST (90% bootstrap interval vs bound): slope -0.0008..+0.0060, equivalent within +-0.02: True; High minus Low -0.0038..+0.0067, equivalent: True.

### Per-language gaps (median r of selective minus matched controls, mean of INT7 and INT6)

| language | tier | Joshi | selective features | gap (median r) | gap (survival rate) | used |
|---|---|---|---|---|---|---|
| eng_Latn | high | 5 | 105 | +0.0052 | +0.0742 | yes |
| deu_Latn | high | 5 | 37 | +0.0138 | +0.0000 | yes |
| spa_Latn | high | 5 | 16 | +0.0004 | +0.0000 | yes |
| fra_Latn | high | 5 | 42 | +0.0090 | -0.0062 | yes |
| por_Latn | high | 4 | 11 | +0.0056 | +0.0000 | yes |
| rus_Cyrl | high | 4 | 3 | +0.0069 | -0.0374 | no |
| zho_Hans | high | 5 | 69 | +0.0106 | +0.0679 | yes |
| jpn_Jpan | high | 5 | 43 | +0.0130 | +0.0000 | yes |
| kor_Hang | high | 4 | 35 | +0.0058 | +0.0000 | yes |
| hin_Deva | mid | 4 | 23 | +0.0047 | +0.0000 | yes |
| ben_Beng | mid | 3 | 29 | +0.0208 | +0.0000 | yes |
| tam_Taml | mid | 3 | 52 | +0.0123 | +0.0000 | yes |
| tur_Latn | mid | 4 | 13 | -0.0013 | +0.0810 | yes |
| vie_Latn | mid | 4 | 46 | +0.0084 | +0.0000 | yes |
| ind_Latn | mid | 3 | 18 | +0.0108 | +0.0000 | yes |
| tha_Thai | mid | 3 | 60 | +0.0057 | -0.0742 | yes |
| pol_Latn | mid | 4 | 5 | +0.0070 | -0.0249 | no |
| ukr_Cyrl | mid | 3 | 8 | +0.0072 | -0.0371 | no |
| kan_Knda | low | 1 | 44 | +0.0124 | +0.0742 | yes |
| tel_Telu | low | 1 | 35 | +0.0163 | +0.0068 | yes |
| yor_Latn | low | 2 | 191 | +0.0153 | +0.0050 | yes |
| sun_Latn | low | 1 | 67 | -0.0043 | +0.0058 | yes |
| amh_Ethi | low | 2 | 75 | +0.0091 | +0.0173 | yes |
| ibo_Latn | low | 1 | 115 | +0.0040 | +0.0000 | yes |
| hau_Latn | low | 2 | 51 | -0.0209 | +0.0133 | yes |
| khm_Khmr | low | 1 | 227 | +0.0105 | +0.0000 | yes |
| lao_Laoo | low | 2 | 256 | +0.0050 | -0.0002 | yes |

## Exploratory analyses (not part of the confirmatory family; p-values are uncorrected)

| width | selective minus control | 95% CI | p (uncorrected) |
|---|---|---|---|
| INT5 | -0.0433 | [-0.0743, -0.0123] | 0.008699 |
| INT4 | -0.0019 | [-0.0259, +0.0220] | 0.8908 |

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

P1 (selective features fire less): mean panel firing rate selective 0.00417, other supported 0.00777, Mann-Whitney p 0.

## Instrument check: survival of the same features at the noise-floor widths INT16, INT12 and restore-only INT0 (all should be at the ceiling)

| width | selective | control |
|---|---|---|
| INT16 | 0.9970 | 1.0000 |
| INT12 | 0.9988 | 0.9994 |
| INT0 | 1.0000 | 1.0000 |
