# Activation-magnitude check of the selectivity dose-response (decision D-020)

Question: is the N2 gap (selective features survive less than firing-rate-matched controls, growing with the S threshold) a selectivity effect or an activation-magnitude effect? Source: `issues.md` C6 / `PLAN.md` P1-5 (Duan et al.'s released per-feature data: survival follows maximum activation, Spearman 0.78, not firing rate, -0.085); the proposal matches controls on firing-rate quintile only. Magnitude m = maximum activation at the feature's peak language (full-precision stream); survival = r > 0.9 there. No new estimator: (a) threshold-free within-stratum dose-response, (b) descriptive magnitude comparison, (c) the unchanged D-014 / D-016 analysis with controls matched on firing-rate stratum x magnitude bin (3 bins = terciles, pre-specified; 5 and 10 added before any real number because coarse bins leave residual confounding when survival follows magnitude steeply). Thresholds are the fixed S thresholds of N2 (labelled deviation, D-019).

**Pre-specified reading rule** (INT6 contrast at the threshold; written before the numbers): not significant (Holm-adjusted p >= 0.05) or within +-1 pp -> *explained by magnitude*; significant (Holm < 0.05) and positive -> *reversed*; significant (Holm < 0.05), more negative than -1 pp and retaining at least half of the rate-only gap -> *supported* (selectivity reading); significant but retaining less than half of the rate-only gap -> *partly explained*. Headline reading = the finest bin resolution that keeps at least half of the selective features (common support); all resolutions are shown.

## (a) Threshold-free dose-response (S against survival inside firing-rate strata)

| width | features | rho(S, survival) | p | rho(S, r) | rho(m, survival) | rho(S, m) |
|---|---|---|---|---|---|---|
| INT7 | 6714 | +0.194 | 0.0001 | +0.216 | +0.598 | +0.118 |
| INT6 | 6714 | +0.164 | 0.0001 | +0.215 | +0.649 | +0.118 |
| INT5 | 6714 | +0.140 | 0.0001 | +0.217 | +0.533 | +0.118 |

Stratum-standardised survival by decile of S (INT6):

| decile of S | features | mean S | raw survival | standardised survival |
|---|---|---|---|---|
| 1 | 672 | 1.73 | 0.345 | 0.283 |
| 2 | 671 | 2.18 | 0.288 | 0.261 |
| 3 | 672 | 2.47 | 0.336 | 0.323 |
| 4 | 671 | 2.78 | 0.359 | 0.354 |
| 5 | 671 | 3.11 | 0.377 | 0.387 |
| 6 | 672 | 3.50 | 0.469 | 0.476 |
| 7 | 671 | 4.00 | 0.462 | 0.464 |
| 8 | 672 | 4.76 | 0.440 | 0.454 |
| 9 | 671 | 6.11 | 0.508 | 0.556 |
| 10 | 671 | 11.18 | 0.575 | 0.646 |

## (b) Magnitude of selective features against their controls

| threshold | selective | controls | median max activation (selective / control) | standardised difference of log10 m (p) |
|---|---|---|---|---|
| null | 5791 | 923 | 8.05 / 8.75 | +0.001 (0.905) |
| 3.0 | 3761 | 2369 | 8.22 / 7.47 | +0.050 (0.0001) |
| 5.0 | 1491 | 1336 | 8.21 / 7.52 | +0.024 (0.002) |

## (c) INT6 contrast, firing-rate matching only against firing-rate x magnitude matching

| threshold | matching | INT6 selective - control (p, Holm) | selective kept | reading |
|---|---|---|---|---|
| null | firing rate only (N2) | +4.08 pp (p 0.0368, Holm 0.0736) | | |
| null | rate x 3 magnitude bins | +2.94 pp (Holm 0.129) | 100 % | explained by magnitude |
| null | rate x 5 magnitude bins | +3.59 pp (Holm 0.034) | 100 % | reversed |
| null | rate x 10 magnitude bins | +3.18 pp (Holm 0.079) | 100 % | explained by magnitude |
| **null** | **headline (10 bins)** | | 100 % | **explained by magnitude** (resolutions disagree)  |
| 3.0 | firing rate only (N2) | +15.76 pp (p 0.0001, Holm 0.0003) | | |
| 3.0 | rate x 3 magnitude bins | +6.69 pp (Holm 0.0003) | 100 % | reversed |
| 3.0 | rate x 5 magnitude bins | +6.83 pp (Holm 0.0003) | 100 % | reversed |
| 3.0 | rate x 10 magnitude bins | +5.74 pp (Holm 0.0004) | 100 % | reversed |
| **3.0** | **headline (10 bins)** | | 100 % | **reversed** (all resolutions agree)  |
| 5.0 | firing rate only (N2) | +12.72 pp (p 0.0001, Holm 0.0003) | | |
| 5.0 | rate x 3 magnitude bins | +6.64 pp (Holm 0.0003) | 100 % | reversed |
| 5.0 | rate x 5 magnitude bins | +7.01 pp (Holm 0.0003) | 100 % | reversed |
| 5.0 | rate x 10 magnitude bins | +5.84 pp (Holm 0.0004) | 100 % | reversed |
| **5.0** | **headline (10 bins)** | | 100 % | **reversed** (all resolutions agree)  |
