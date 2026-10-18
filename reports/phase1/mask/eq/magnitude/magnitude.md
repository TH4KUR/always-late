# Activation-magnitude check of the selectivity dose-response (decision D-020)

Question: is the N2 gap (selective features survive less than firing-rate-matched controls, growing with the S threshold) a selectivity effect or an activation-magnitude effect? Source: `issues.md` C6 / `PLAN.md` P1-5 (Duan et al.'s released per-feature data: survival follows maximum activation, Spearman 0.78, not firing rate, -0.085); the proposal matches controls on firing-rate quintile only. Magnitude m = maximum activation at the feature's peak language (full-precision stream); survival = r > 0.9 there. No new estimator: (a) threshold-free within-stratum dose-response, (b) descriptive magnitude comparison, (c) the unchanged D-014 / D-016 analysis with controls matched on firing-rate stratum x magnitude bin (3 bins = terciles, pre-specified; 5 and 10 added before any real number because coarse bins leave residual confounding when survival follows magnitude steeply). Thresholds are the fixed S thresholds of N2 (labelled deviation, D-019).

**Pre-specified reading rule** (INT6 contrast at the threshold; written before the numbers): not significant (Holm-adjusted p >= 0.05) or within +-1 pp -> *explained by magnitude*; significant (Holm < 0.05) and positive -> *reversed*; significant (Holm < 0.05), more negative than -1 pp and retaining at least half of the rate-only gap -> *supported* (selectivity reading); significant but retaining less than half of the rate-only gap -> *partly explained*. Headline reading = the finest bin resolution that keeps at least half of the selective features (common support); all resolutions are shown.

## (a) Threshold-free dose-response (S against survival inside firing-rate strata)

| width | features | rho(S, survival) | p | rho(S, r) | rho(m, survival) | rho(S, m) |
|---|---|---|---|---|---|---|
| INT7 | 10905 | -0.017 | 0.12 | +0.019 | +0.092 | -0.181 |
| INT6 | 10905 | -0.028 | 0.0038 | +0.021 | +0.354 | -0.181 |
| INT5 | 10905 | +0.019 | 0.0465 | +0.033 | +0.535 | -0.181 |

Stratum-standardised survival by decile of S (INT6):

| decile of S | features | mean S | raw survival | standardised survival |
|---|---|---|---|---|
| 1 | 1091 | 1.62 | 0.945 | 0.948 |
| 2 | 1090 | 1.86 | 0.947 | 0.930 |
| 3 | 1091 | 2.04 | 0.941 | 0.930 |
| 4 | 1090 | 2.22 | 0.942 | 0.927 |
| 5 | 1091 | 2.44 | 0.945 | 0.941 |
| 6 | 1090 | 2.72 | 0.929 | 0.930 |
| 7 | 1091 | 3.12 | 0.933 | 0.933 |
| 8 | 1090 | 3.79 | 0.927 | 0.936 |
| 9 | 1091 | 5.21 | 0.902 | 0.920 |
| 10 | 1090 | 11.59 | 0.806 | 0.902 |

## (b) Magnitude of selective features against their controls

| threshold | selective | controls | median max activation (selective / control) | standardised difference of log10 m (p) |
|---|---|---|---|---|
| null | 8062 | 2843 | 13.9 / 16 | -0.048 (0.0001) |
| 3.0 | 3917 | 3114 | 12.1 / 14.8 | -0.065 (0.0001) |
| 5.0 | 1676 | 1651 | 9.77 / 14.2 | -0.119 (0.0001) |

## (c) INT6 contrast, firing-rate matching only against firing-rate x magnitude matching

| threshold | matching | INT6 selective - control (p, Holm) | selective kept | reading |
|---|---|---|---|---|
| null | firing rate only (N2) | -1.44 pp (p 0.0427, Holm 0.128) | | |
| null | rate x 3 magnitude bins | +1.35 pp (Holm 0.242) | 100 % | explained by magnitude |
| null | rate x 5 magnitude bins | +2.69 pp (Holm 0.0039) | 100 % | reversed |
| null | rate x 10 magnitude bins | +3.12 pp (Holm 0.0003) | 100 % | reversed |
| **null** | **headline (10 bins)** | | 100 % | **reversed** (resolutions disagree)  |
| 3.0 | firing rate only (N2) | -2.14 pp (p 0.0055, Holm 0.0165) | | |
| 3.0 | rate x 3 magnitude bins | +1.79 pp (Holm 0.111) | 100 % | explained by magnitude |
| 3.0 | rate x 5 magnitude bins | +3.90 pp (Holm 0.0003) | 100 % | reversed |
| 3.0 | rate x 10 magnitude bins | +4.26 pp (Holm 0.0003) | 100 % | reversed |
| **3.0** | **headline (10 bins)** | | 100 % | **reversed** (resolutions disagree)  |
| 5.0 | firing rate only (N2) | -7.16 pp (p 0.0001, Holm 0.0003) | | |
| 5.0 | rate x 3 magnitude bins | +0.80 pp (Holm 1) | 100 % | explained by magnitude |
| 5.0 | rate x 5 magnitude bins | +4.13 pp (Holm 0.0048) | 100 % | reversed |
| 5.0 | rate x 10 magnitude bins | +4.58 pp (Holm 0.0012) | 100 % | reversed |
| **5.0** | **headline (10 bins)** | | 100 % | **reversed** (resolutions disagree)  |
