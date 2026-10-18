# Activation-magnitude check of the selectivity dose-response (decision D-020)

Question: is the N2 gap (selective features survive less than firing-rate-matched controls, growing with the S threshold) a selectivity effect or an activation-magnitude effect? Source: `issues.md` C6 / `PLAN.md` P1-5 (Duan et al.'s released per-feature data: survival follows maximum activation, Spearman 0.78, not firing rate, -0.085); the proposal matches controls on firing-rate quintile only. Magnitude m = maximum activation at the feature's peak language (full-precision stream); survival = r > 0.9 there. No new estimator: (a) threshold-free within-stratum dose-response, (b) descriptive magnitude comparison, (c) the unchanged D-014 / D-016 analysis with controls matched on firing-rate stratum x magnitude bin (3 bins = terciles, pre-specified; 5 and 10 added before any real number because coarse bins leave residual confounding when survival follows magnitude steeply). Thresholds are the fixed S thresholds of N2 (labelled deviation, D-019).

**Pre-specified reading rule** (INT6 contrast at the threshold; written before the numbers): not significant (Holm-adjusted p >= 0.05) or within +-1 pp -> *explained by magnitude*; significant (Holm < 0.05) and positive -> *reversed*; significant (Holm < 0.05), more negative than -1 pp and retaining at least half of the rate-only gap -> *supported* (selectivity reading); significant but retaining less than half of the rate-only gap -> *partly explained*. Headline reading = the finest bin resolution that keeps at least half of the selective features (common support); all resolutions are shown.

## (a) Threshold-free dose-response (S against survival inside firing-rate strata)

| width | features | rho(S, survival) | p | rho(S, r) | rho(m, survival) | rho(S, m) |
|---|---|---|---|---|---|---|
| INT7 | 11469 | -0.003 | 0.793 | +0.016 | +0.098 | -0.172 |
| INT6 | 11469 | -0.032 | 0.0006 | +0.014 | +0.344 | -0.172 |
| INT5 | 11469 | +0.013 | 0.164 | +0.027 | +0.523 | -0.172 |

Stratum-standardised survival by decile of S (INT6):

| decile of S | features | mean S | raw survival | standardised survival |
|---|---|---|---|---|
| 1 | 1147 | 1.62 | 0.949 | 0.906 |
| 2 | 1147 | 1.87 | 0.941 | 0.933 |
| 3 | 1147 | 2.05 | 0.938 | 0.905 |
| 4 | 1147 | 2.24 | 0.938 | 0.919 |
| 5 | 1147 | 2.46 | 0.949 | 0.942 |
| 6 | 1147 | 2.75 | 0.921 | 0.919 |
| 7 | 1147 | 3.19 | 0.924 | 0.924 |
| 8 | 1147 | 3.93 | 0.905 | 0.923 |
| 9 | 1147 | 5.52 | 0.871 | 0.905 |
| 10 | 1146 | 12.52 | 0.737 | 0.879 |

## (b) Magnitude of selective features against their controls

| threshold | selective | controls | median max activation (selective / control) | standardised difference of log10 m (p) |
|---|---|---|---|---|
| null | 8579 | 2890 | 14.2 / 16.6 | -0.045 (0.0001) |
| 3.0 | 4281 | 3246 | 12.4 / 15.7 | -0.066 (0.0001) |
| 5.0 | 1913 | 1930 | 9.67 / 13.5 | -0.113 (0.0001) |

## (c) INT6 contrast, firing-rate matching only against firing-rate x magnitude matching

| threshold | matching | INT6 selective - control (p, Holm) | selective kept | reading |
|---|---|---|---|---|
| null | firing rate only (N2) | -1.93 pp (p 0.0183, Holm 0.0549) | | |
| null | rate x 3 magnitude bins | +1.10 pp (Holm 0.58) | 100 % | explained by magnitude |
| null | rate x 5 magnitude bins | +1.80 pp (Holm 0.151) | 100 % | explained by magnitude |
| null | rate x 10 magnitude bins | +2.09 pp (Holm 0.0915) | 100 % | explained by magnitude |
| **null** | **headline (10 bins)** | | 100 % | **explained by magnitude** (all resolutions agree)  |
| 3.0 | firing rate only (N2) | -4.04 pp (p 0.0001, Holm 0.0003) | | |
| 3.0 | rate x 3 magnitude bins | +1.90 pp (Holm 0.155) | 100 % | explained by magnitude |
| 3.0 | rate x 5 magnitude bins | +2.58 pp (Holm 0.0432) | 100 % | reversed |
| 3.0 | rate x 10 magnitude bins | +3.01 pp (Holm 0.0201) | 100 % | reversed |
| **3.0** | **headline (10 bins)** | | 100 % | **reversed** (resolutions disagree)  |
| 5.0 | firing rate only (N2) | -8.24 pp (p 0.0001, Holm 0.0003) | | |
| 5.0 | rate x 3 magnitude bins | +3.45 pp (Holm 0.0291) | 100 % | reversed |
| 5.0 | rate x 5 magnitude bins | +5.06 pp (Holm 0.0018) | 100 % | reversed |
| 5.0 | rate x 10 magnitude bins | +5.59 pp (Holm 0.0012) | 100 % | reversed |
| **5.0** | **headline (10 bins)** | | 100 % | **reversed** (all resolutions agree)  |
