> **N11 (D-025): in this report every activation is divided by its feature's learned JumpReLU threshold (the margin ratio; thresholds in `reports/phase1/sae_thresholds.csv`); read 'maximum activation' as 'margin ratio' and 'magnitude' as 'margin' throughout.**

# Mixed-model variants for the S~ coefficient, regime `mask`, stream `all` (decision D-021; A-034 (c))

The exploratory model of the analysis (proposal section 7) has a positive S~ coefficient while the matched contrasts are not positive (N2, N6) and the threshold-free within-stratum correlation is about zero. The SAME model and estimator are refitted on the same data: (i) unchanged, (ii) without the tier terms, (iii) with log10 maximum activation at the peak language added, (iv) the full formula restricted to the matched sets of N2 (firing-rate strata) and N6 (firing-rate x 10 magnitude bins) at S > 5. **Pre-specified reading rule:** if the S~ coefficient is <= 0 or not significant in variant (ii) or (iii), the conflict is explained by that term; if it stays significantly positive in all variants it is unresolved and no further variants are tried. Significance is strict (p exactly 0.05 is not significant).

Variant (i) reproduces the saved `analysis.json` coefficients exactly: **None** (largest absolute difference None). The quadrature-ML coefficients are bit-identical only at a fixed BLAS thread count (the saved analysis used 4; another thread count moves them by about 1e-10, A-036).

| variant | formula | features | S~ (SE), p | tier | S~:tier | log10 m | GEE S~ (p) | sigma | converged |
|---|---|---|---|---|---|---|---|---|---|
| full_all | `bits_c + C(stratum) + S_rank + tier + S_rank:tier` | 11469 | +1.031 (0.325), p 0.00154 | +1.439 | +1.311 |  | +0.709 (9.5e-05) | 2.75 | True |
| no_tier_all | `bits_c + C(stratum) + S_rank` | 11469 | -1.042 (0.296), p 0.000424 |  |  |  | -0.645 (1.1e-05) | 3.55 | True |
| log_m_all | `bits_c + C(stratum) + S_rank + tier + S_rank:tier + log10_m` | 11469 | +3.385 (0.400), p 2.76e-17 | +1.434 | +0.090 | +14.585 | +2.358 (4.3e-28) | 2.18 | True |
| full_matched_N2 | `bits_c + C(stratum) + S_rank + tier + S_rank:tier` | 3843 | +0.139 (0.532), p 0.793 | +0.871 | +2.290 |  | +0.138 (0.68) | 2.55 | True |
| full_matched_N6 | `bits_c + C(stratum) + S_rank + tier + S_rank:tier` | 3471 | +2.197 (0.558), p 8.34e-05 | +1.208 | +1.680 |  | +1.265 (2.8e-05) | 2.59 | True |

**Reading (by the rule fixed beforehand): explained by the tier terms.**
