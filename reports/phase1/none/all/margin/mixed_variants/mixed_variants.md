> **N11 (D-025): in this report every activation is divided by its feature's learned JumpReLU threshold (the margin ratio; thresholds in `reports/phase1/sae_thresholds.csv`); read 'maximum activation' as 'margin ratio' and 'magnitude' as 'margin' throughout.**

# Mixed-model variants for the S~ coefficient, regime `none`, stream `all` (decision D-021; A-034 (c))

The exploratory model of the analysis (proposal section 7) has a positive S~ coefficient while the matched contrasts are not positive (N2, N6) and the threshold-free within-stratum correlation is about zero. The SAME model and estimator are refitted on the same data: (i) unchanged, (ii) without the tier terms, (iii) with log10 maximum activation at the peak language added, (iv) the full formula restricted to the matched sets of N2 (firing-rate strata) and N6 (firing-rate x 10 magnitude bins) at S > 5. **Pre-specified reading rule:** if the S~ coefficient is <= 0 or not significant in variant (ii) or (iii), the conflict is explained by that term; if it stays significantly positive in all variants it is unresolved and no further variants are tried. Significance is strict (p exactly 0.05 is not significant).

Variant (i) reproduces the saved `analysis.json` coefficients exactly: **None** (largest absolute difference None). The quadrature-ML coefficients are bit-identical only at a fixed BLAS thread count (the saved analysis used 4; another thread count moves them by about 1e-10, A-036).

| variant | formula | features | S~ (SE), p | tier | S~:tier | log10 m | GEE S~ (p) | sigma | converged |
|---|---|---|---|---|---|---|---|---|---|
| full_all | `bits_c + C(stratum) + S_rank + tier + S_rank:tier` | 6714 | +5.393 (0.570), p 3e-21 | +0.039 | +1.487 |  | +1.329 (5.3e-28) | 5.99 | True |
| no_tier_all | `bits_c + C(stratum) + S_rank` | 6714 | +6.454 (0.477), p 1.2e-41 |  |  |  | +1.671 (3.7e-72) | 6.17 | True |
| log_m_all | `bits_c + C(stratum) + S_rank + tier + S_rank:tier + log10_m` | 6714 | +3.909 (0.401), p 1.78e-22 | -0.025 | +0.074 | +24.825 | +1.672 (5e-28) | 3.55 | True |
| full_matched_N2 | `bits_c + C(stratum) + S_rank + tier + S_rank:tier` | 2827 | +3.227 (0.962), p 0.000798 | +0.636 | +0.144 |  | +1.779 (2.5e-20) | 91.57 | True |
| full_matched_N6 | `bits_c + C(stratum) + S_rank + tier + S_rank:tier` | 2794 | +4.071 (0.780), p 1.78e-07 | +0.747 | +0.706 |  | +1.138 (1.9e-08) | 5.45 | True |

**Reading (by the rule fixed beforehand): unresolved.**
