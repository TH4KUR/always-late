# Activation-magnitude check of the selectivity dose-response (decision D-020)

Question: is the N2 gap (selective features survive less than firing-rate-matched controls, growing with the S threshold) a selectivity effect or an activation-magnitude effect? Source: `issues.md` C6 / `PLAN.md` P1-5 (Duan et al.'s released per-feature data: survival follows maximum activation, Spearman 0.78, not firing rate, -0.085); the proposal matches controls on firing-rate quintile only. Magnitude m = maximum activation at the feature's peak language (full-precision stream); survival = r > 0.9 there. No new estimator: (a) threshold-free within-stratum dose-response, (b) descriptive magnitude comparison, (c) the unchanged D-014 / D-016 analysis with controls matched on firing-rate stratum x magnitude bin (3 bins = terciles, pre-specified; 5 and 10 added before any real number because coarse bins leave residual confounding when survival follows magnitude steeply). Thresholds are the fixed S thresholds of N2 (labelled deviation, D-019).

**Pre-specified reading rule** (INT6 contrast at the threshold; written before the numbers): not significant (Holm-adjusted p >= 0.05) or within +-1 pp -> *explained by magnitude*; significant (Holm < 0.05) and positive -> *reversed*; significant (Holm < 0.05), more negative than -1 pp and retaining at least half of the rate-only gap -> *supported* (selectivity reading); significant but retaining less than half of the rate-only gap -> *partly explained*. Headline reading = the finest bin resolution that keeps at least half of the selective features (common support); all resolutions are shown.

## (a) Threshold-free dose-response (S against survival inside firing-rate strata)

| width | features | rho(S, survival) | p | rho(S, r) | rho(m, survival) | rho(S, m) |
|---|---|---|---|---|---|---|
| INT7 | 6001 | +0.204 | 0.0001 | +0.225 | +0.580 | +0.113 |
| INT6 | 6001 | +0.173 | 0.0001 | +0.226 | +0.650 | +0.113 |
| INT5 | 6001 | +0.136 | 0.0001 | +0.234 | +0.551 | +0.113 |

Stratum-standardised survival by decile of S (INT6):

| decile of S | features | mean S | raw survival | standardised survival |
|---|---|---|---|---|
| 1 | 601 | 1.71 | 0.356 | 0.327 |
| 2 | 600 | 2.16 | 0.273 | 0.280 |
| 3 | 600 | 2.46 | 0.328 | 0.291 |
| 4 | 600 | 2.76 | 0.362 | 0.364 |
| 5 | 600 | 3.09 | 0.353 | 0.353 |
| 6 | 600 | 3.49 | 0.448 | 0.457 |
| 7 | 600 | 3.98 | 0.457 | 0.459 |
| 8 | 600 | 4.75 | 0.458 | 0.477 |
| 9 | 600 | 6.10 | 0.517 | 0.562 |
| 10 | 600 | 11.17 | 0.575 | 0.640 |

## (b) Magnitude of selective features against their controls

| threshold | selective | controls | median max activation (selective / control) | standardised difference of log10 m (p) |
|---|---|---|---|---|
| null | 5126 | 875 | 7.76 / 8.57 | -0.001 (0.876) |
| 3.0 | 3326 | 2063 | 7.92 / 7.31 | +0.032 (0.0001) |
| 5.0 | 1329 | 1240 | 8.04 / 6.98 | +0.039 (0.0001) |

## (c) INT6 contrast, firing-rate matching only against firing-rate x magnitude matching

| threshold | matching | INT6 selective - control (p, Holm) | selective kept | reading |
|---|---|---|---|---|
| null | firing rate only (N2) | +7.71 pp (p 0.0005, Holm 0.001) | | |
| null | rate x 3 magnitude bins | +7.53 pp (Holm 0.0003) | 100 % | reversed |
| null | rate x 5 magnitude bins | +7.45 pp (Holm 0.0003) | 100 % | reversed |
| null | rate x 10 magnitude bins | +7.53 pp (Holm 0.0003) | 100 % | reversed |
| **null** | **headline (10 bins)** | | 100 % | **reversed** (all resolutions agree)  |
| 3.0 | firing rate only (N2) | +14.97 pp (p 0.0001, Holm 0.0003) | | |
| 3.0 | rate x 3 magnitude bins | +9.15 pp (Holm 0.0003) | 100 % | reversed |
| 3.0 | rate x 5 magnitude bins | +9.80 pp (Holm 0.0003) | 100 % | reversed |
| 3.0 | rate x 10 magnitude bins | +9.42 pp (Holm 0.0003) | 100 % | reversed |
| **3.0** | **headline (10 bins)** | | 100 % | **reversed** (all resolutions agree)  |
| 5.0 | firing rate only (N2) | +17.46 pp (p 0.0001, Holm 0.0003) | | |
| 5.0 | rate x 3 magnitude bins | +9.10 pp (Holm 0.0003) | 100 % | reversed |
| 5.0 | rate x 5 magnitude bins | +9.20 pp (Holm 0.0003) | 100 % | reversed |
| 5.0 | rate x 10 magnitude bins | +9.29 pp (Holm 0.0003) | 100 % | reversed |
| **5.0** | **headline (10 bins)** | | 100 % | **reversed** (all resolutions agree)  |
