# Phase 1 analysis, regime `mask` (decision D-014)

10000 permutations, 2000 bootstrap draws, seed 0.

## Confirmatory family (Holm over three tests)

| test | estimate | 95% CI / detail | p | Holm p |
|---|---|---|---|---|
| INT7 selective minus matched-universal survival | -0.0009 | [-0.0028, +0.0010] (selective 0.9979, control 0.9992, n 5175+3931) | 0.4811 | 0.7907 |
| INT6 selective minus matched-universal survival | -0.0132 | [-0.0198, -0.0063] (selective 0.9722, control 0.9893, n 5175+3931) | 0.0017 | 0.005099 |
| language-level tier slope (median-r gap, INT7+INT6, one-sided) | +0.0003 | languages used 27 (need 20: ok); High minus Low 0.00033623311254713287 | 0.3954 | 0.7907 |

TOST (90% bootstrap interval vs bound): slope -0.0012..+0.0016, equivalent within +-0.02: True; High minus Low -0.0022..+0.0024, equivalent: True.

### Per-language gaps (median r of selective minus matched controls, mean of INT7 and INT6)

| language | tier | Joshi | selective features | gap (median r) | gap (survival rate) | used |
|---|---|---|---|---|---|---|
| eng_Latn | high | 5 | 1246 | -0.0024 | +0.0000 | yes |
| deu_Latn | high | 5 | 112 | +0.0022 | +0.0000 | yes |
| spa_Latn | high | 5 | 137 | +0.0007 | +0.0000 | yes |
| fra_Latn | high | 5 | 118 | +0.0050 | +0.0053 | yes |
| por_Latn | high | 4 | 59 | +0.0042 | +0.0053 | yes |
| rus_Cyrl | high | 4 | 70 | +0.0002 | +0.0000 | yes |
| zho_Hans | high | 5 | 333 | +0.0013 | +0.0053 | yes |
| jpn_Jpan | high | 5 | 394 | +0.0003 | +0.0011 | yes |
| kor_Hang | high | 4 | 106 | +0.0062 | +0.0053 | yes |
| hin_Deva | mid | 4 | 81 | -0.0036 | +0.0053 | yes |
| ben_Beng | mid | 3 | 60 | +0.0033 | +0.0053 | yes |
| tam_Taml | mid | 3 | 87 | +0.0019 | +0.0053 | yes |
| tur_Latn | mid | 4 | 139 | +0.0032 | +0.0000 | yes |
| vie_Latn | mid | 4 | 190 | +0.0033 | +0.0053 | yes |
| ind_Latn | mid | 3 | 105 | +0.0040 | +0.0000 | yes |
| tha_Thai | mid | 3 | 208 | +0.0061 | +0.0053 | yes |
| pol_Latn | mid | 4 | 76 | -0.0003 | +0.0000 | yes |
| ukr_Cyrl | mid | 3 | 57 | +0.0025 | +0.0053 | yes |
| kan_Knda | low | 1 | 75 | -0.0049 | +0.0053 | yes |
| tel_Telu | low | 1 | 65 | +0.0048 | +0.0074 | yes |
| yor_Latn | low | 2 | 369 | +0.0030 | +0.0034 | yes |
| sun_Latn | low | 1 | 181 | -0.0049 | +0.0000 | yes |
| amh_Ethi | low | 2 | 154 | +0.0042 | +0.0024 | yes |
| ibo_Latn | low | 1 | 149 | +0.0029 | +0.0000 | yes |
| hau_Latn | low | 2 | 142 | +0.0068 | +0.0000 | yes |
| khm_Khmr | low | 1 | 236 | +0.0017 | +0.0029 | yes |
| lao_Laoo | low | 2 | 226 | +0.0010 | +0.0021 | yes |

## Exploratory analyses (not part of the confirmatory family; p-values are uncorrected)

| width | selective minus control | 95% CI | p (uncorrected) |
|---|---|---|---|
| INT4 | -0.0617 | [-0.0835, -0.0401] | 9.999e-05 |

Mixed-effects logistic model (random intercept per feature, quadrature ML; 9704 features, sigma 3.515, converged True):

| term | estimate | SE | p |
|---|---|---|---|
| Intercept | +5.7468 | 1.0843 | 1.158e-07 |
| C(stratum)[T.1] | +2.0128 | 0.4820 | 2.967e-05 |
| C(stratum)[T.2] | +3.1007 | 0.6485 | 1.742e-06 |
| C(stratum)[T.3] | +4.0913 | 0.8049 | 3.715e-07 |
| C(stratum)[T.4] | +4.8405 | 0.8973 | 6.866e-08 |
| bits_c | +4.4647 | 0.7039 | 2.26e-10 |
| S_rank | +0.5631 | 0.7027 | 0.423 |
| tier | +2.5006 | 0.5626 | 8.794e-06 |
| S_rank:tier | -0.0769 | 0.6658 | 0.9081 |

P1 (selective features fire less): mean panel firing rate selective 0.00517, other supported 0.00675, Mann-Whitney p 0.

## Instrument check: survival of the same features at the noise-floor widths INT16, INT12 and restore-only INT0 (all should be at the ceiling)

| width | selective | control |
|---|---|---|
| INT16 | 0.9996 | 0.9997 |
| INT12 | 0.9996 | 1.0000 |
| INT0 | 1.0000 | 1.0000 |
