> **N11 (D-025): in this report every activation is divided by its feature's learned JumpReLU threshold (the margin ratio; thresholds in `reports/phase1/sae_thresholds.csv`); read 'maximum activation' as 'margin ratio' and 'magnitude' as 'margin' throughout.**

# Mixed-model variants for the S~ coefficient, regime `none`, stream `eq` (decision D-021; A-034 (c))

The exploratory model of the analysis (proposal section 7) has a positive S~ coefficient while the matched contrasts are not positive (N2, N6) and the threshold-free within-stratum correlation is about zero. The SAME model and estimator are refitted on the same data: (i) unchanged, (ii) without the tier terms, (iii) with log10 maximum activation at the peak language added, (iv) the full formula restricted to the matched sets of N2 (firing-rate strata) and N6 (firing-rate x 10 magnitude bins) at S > 5. **Pre-specified reading rule:** if the S~ coefficient is <= 0 or not significant in variant (ii) or (iii), the conflict is explained by that term; if it stays significantly positive in all variants it is unresolved and no further variants are tried. Significance is strict (p exactly 0.05 is not significant).

Variant (i) reproduces the saved `analysis.json` coefficients exactly: **None** (largest absolute difference None). The quadrature-ML coefficients are bit-identical only at a fixed BLAS thread count (the saved analysis used 4; another thread count moves them by about 1e-10, A-036).

| variant | formula | features | S~ (SE), p | tier | S~:tier | log10 m | GEE S~ (p) | sigma | converged |
|---|---|---|---|---|---|---|---|---|---|
| full_all | `bits_c + C(stratum) + S_rank + tier + S_rank:tier` | 6001 | +5.536 (0.600), p 2.94e-20 | -0.212 | +1.519 |  | +1.414 (6.8e-27) | 5.84 | True |
| no_tier_all | `bits_c + C(stratum) + S_rank` | 6001 | +6.619 (0.509), p 1.16e-38 |  |  |  | +1.782 (1.1e-69) | 5.95 | True |
| log_m_all | `bits_c + C(stratum) + S_rank + tier + S_rank:tier + log10_m` | 6001 | +3.933 (0.417), p 4.35e-21 | -0.531 | +0.116 | +24.588 | +1.861 (1e-28) | 3.33 | True |
| full_matched_N2 | `bits_c + C(stratum) + S_rank + tier + S_rank:tier` | 2569 | +3.246 (0.962), p 0.000743 | -0.299 | +1.106 |  | +1.938 (7.6e-19) | 98.05 | True |
| full_matched_N6 | `bits_c + C(stratum) + S_rank + tier + S_rank:tier` | 2508 | +3.003 (0.820), p 0.000252 | -0.252 | +1.428 |  | +0.858 (0.00031) | 5.15 | True |

**Reading (by the rule fixed beforehand): unresolved.**
