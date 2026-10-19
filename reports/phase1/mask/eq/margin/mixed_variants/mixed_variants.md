> **N11 (D-025): in this report every activation is divided by its feature's learned JumpReLU threshold (the margin ratio; thresholds in `reports/phase1/sae_thresholds.csv`); read 'maximum activation' as 'margin ratio' and 'magnitude' as 'margin' throughout.**

# Mixed-model variants for the S~ coefficient, regime `mask`, stream `eq` (decision D-021; A-034 (c))

The exploratory model of the analysis (proposal section 7) has a positive S~ coefficient while the matched contrasts are not positive (N2, N6) and the threshold-free within-stratum correlation is about zero. The SAME model and estimator are refitted on the same data: (i) unchanged, (ii) without the tier terms, (iii) with log10 maximum activation at the peak language added, (iv) the full formula restricted to the matched sets of N2 (firing-rate strata) and N6 (firing-rate x 10 magnitude bins) at S > 5. **Pre-specified reading rule:** if the S~ coefficient is <= 0 or not significant in variant (ii) or (iii), the conflict is explained by that term; if it stays significantly positive in all variants it is unresolved and no further variants are tried. Significance is strict (p exactly 0.05 is not significant).

Variant (i) reproduces the saved `analysis.json` coefficients exactly: **None** (largest absolute difference None). The quadrature-ML coefficients are bit-identical only at a fixed BLAS thread count (the saved analysis used 4; another thread count moves them by about 1e-10, A-036).

| variant | formula | features | S~ (SE), p | tier | S~:tier | log10 m | GEE S~ (p) | sigma | converged |
|---|---|---|---|---|---|---|---|---|---|
| full_all | `bits_c + C(stratum) + S_rank + tier + S_rank:tier` | 10905 | +1.194 (0.339), p 0.000434 | +1.455 | +0.943 |  | +0.819 (2e-05) | 2.70 | True |
| no_tier_all | `bits_c + C(stratum) + S_rank` | 10905 | -0.812 (0.285), p 0.00438 |  |  |  | -0.529 (0.00055) | 3.39 | True |
| log_m_all | `bits_c + C(stratum) + S_rank + tier + S_rank:tier + log10_m` | 10905 | +3.002 (0.390), p 1.48e-14 | +0.879 | +0.022 | +16.454 | +2.127 (1.1e-21) | 2.03 | True |
| full_matched_N2 | `bits_c + C(stratum) + S_rank + tier + S_rank:tier` | 3327 | +0.271 (0.618), p 0.661 | +1.081 | +1.694 |  | +0.192 (0.63) | 2.48 | True |
| full_matched_N6 | `bits_c + C(stratum) + S_rank + tier + S_rank:tier` | 3160 | +3.111 (0.637), p 1.02e-06 | +1.401 | +1.438 |  | +1.680 (8.8e-09) | 2.79 | True |

**Reading (by the rule fixed beforehand): explained by the tier terms.**
