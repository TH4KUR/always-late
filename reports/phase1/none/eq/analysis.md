# Phase 1 analysis, regime `none` (decision D-014)

10000 permutations, 2000 bootstrap draws, seed 0.

## Confirmatory family (Holm over three tests)

| test | estimate | 95% CI / detail | p | Holm p |
|---|---|---|---|---|
| INT7 selective minus matched-universal survival | +0.0916 | [+0.0518, +0.1323] (selective 0.7179, control 0.6857, n 5126+875) | 9.999e-05 | 0.0003 |
| INT6 selective minus matched-universal survival | +0.0771 | [+0.0395, +0.1137] (selective 0.4200, control 0.3703, n 5126+875) | 0.0005 | 0.0009999 |
| language-level tier slope (median-r gap, INT7+INT6, one-sided) | +0.0026 | languages used 27 (need 20: ok); High minus Low 0.021218859487109713 | 0.378 | 0.378 |

TOST (90% bootstrap interval vs bound): slope -0.0089..+0.0150, equivalent within +-0.02: True; High minus Low +0.0044..+0.0400, equivalent: False.

### Per-language gaps (median r of selective minus matched controls, mean of INT7 and INT6)

| language | tier | Joshi | selective features | gap (median r) | gap (survival rate) | used |
|---|---|---|---|---|---|---|
| eng_Latn | high | 5 | 799 | -0.0126 | +0.0000 | yes |
| deu_Latn | high | 5 | 156 | +0.0114 | +0.0761 | yes |
| spa_Latn | high | 5 | 112 | +0.0640 | +0.3453 | yes |
| fra_Latn | high | 5 | 143 | +0.0489 | +0.0959 | yes |
| por_Latn | high | 4 | 102 | +0.0402 | +0.1263 | yes |
| rus_Cyrl | high | 4 | 109 | +0.0172 | +0.0000 | yes |
| zho_Hans | high | 5 | 206 | +0.0414 | +0.5394 | yes |
| jpn_Jpan | high | 5 | 213 | +0.0314 | +0.1157 | yes |
| kor_Hang | high | 4 | 153 | +0.0540 | +0.5202 | yes |
| hin_Deva | mid | 4 | 63 | +0.0396 | +0.5749 | yes |
| ben_Beng | mid | 3 | 91 | +0.0141 | +0.1263 | yes |
| tam_Taml | mid | 3 | 182 | -0.0016 | +0.0953 | yes |
| tur_Latn | mid | 4 | 80 | +0.0167 | +0.4345 | yes |
| vie_Latn | mid | 4 | 154 | +0.0345 | +0.4441 | yes |
| ind_Latn | mid | 3 | 104 | +0.0468 | +0.5400 | yes |
| tha_Thai | mid | 3 | 134 | +0.0346 | +0.5394 | yes |
| pol_Latn | mid | 4 | 79 | -0.0130 | +0.0192 | yes |
| ukr_Cyrl | mid | 3 | 104 | +0.0247 | +0.0000 | yes |
| kan_Knda | low | 1 | 133 | +0.0570 | +0.5610 | yes |
| tel_Telu | low | 1 | 98 | +0.0225 | +0.5354 | yes |
| yor_Latn | low | 2 | 297 | +0.0286 | +0.0761 | yes |
| sun_Latn | low | 1 | 162 | +0.0562 | +0.0501 | yes |
| amh_Ethi | low | 2 | 208 | -0.0556 | -0.4796 | yes |
| ibo_Latn | low | 1 | 284 | +0.0149 | +0.0198 | yes |
| hau_Latn | low | 2 | 152 | -0.0046 | +0.0448 | yes |
| khm_Khmr | low | 1 | 372 | +0.0167 | +0.0761 | yes |
| lao_Laoo | low | 2 | 436 | -0.0308 | -0.0396 | yes |

## Exploratory analyses (not part of the confirmatory family; p-values are uncorrected)

| width | selective minus control | 95% CI | p (uncorrected) |
|---|---|---|---|
| INT5 | +0.0311 | [+0.0007, +0.0614] | 0.06689 |
| INT4 | +0.0064 | [-0.0165, +0.0278] | 0.5782 |

Mixed-effects logistic model (random intercept per feature, quadrature ML; 6001 features, sigma 5.839, converged True):

| term | estimate | SE | p |
|---|---|---|---|
| Intercept | -3.9776 | 0.4801 | 1.187e-16 |
| C(stratum)[T.1] | +1.2676 | 0.3052 | 3.285e-05 |
| C(stratum)[T.2] | +1.5958 | 0.3204 | 6.344e-07 |
| C(stratum)[T.3] | +1.1105 | 0.3234 | 0.0005961 |
| C(stratum)[T.4] | +4.9876 | 0.4569 | 9.542e-28 |
| bits_c | +5.0357 | 0.2914 | 6.49e-67 |
| S_rank | +5.5357 | 0.6003 | 2.937e-20 |
| tier | -0.2121 | 0.2072 | 0.3061 |
| S_rank:tier | +1.5188 | 0.3530 | 1.687e-05 |

P1 (selective features fire less): mean panel firing rate selective 0.00914, other supported 0.06677, Mann-Whitney p 1.825e-41.

## Instrument check: survival of the same features at the noise-floor widths INT16, INT12 and restore-only INT0 (all should be at the ceiling)

| width | selective | control |
|---|---|---|
| INT16 | 0.9986 | 0.9989 |
| INT12 | 0.9984 | 1.0000 |
| INT0 | 1.0000 | 1.0000 |
