# Fixed-threshold sensitivity of the selective split, regime `none`, stream `eq` (decision D-019)

**Labelled deviation from the proposal (section 4.1):** the proposal sets the binary threshold at percentiles of the shuffled-label null of S; the runs saved only the 95th-percentile cutoff (1.8284), not the null samples, so fixed S thresholds are reported instead (DECISIONS_FOR_REVIEW.md Q6). The rule is the run's own: selective = supported, peak language stable in at least 80 of the bootstrap draws, and S > tau (strict); controls = firing-rate-stratum-matched universal features drawn by the same `matched_sample`; the estimators are the unchanged D-014 / D-016 analysis (10000 permutations, 2000 bootstrap draws, seed 0). The `null` row is the run's own cutoff and must equal the original analysis.

| tau | selective | controls | strata short | T1 INT7 selective - control (95% CI), p, Holm | T2 INT6 selective - control (95% CI), p, Holm | T3 tier slope, p, Holm |
|---|---|---|---|---|---|---|
| null (1.828) | 5129 | 877 | 5 | +9.16 pp [+5.18, +13.23], p 0.0001, Holm 0.0003 | +7.71 pp [+3.95, +11.37], p 0.0005, Holm 0.001 | +0.0026, p 0.378, Holm 0.378 |
| 2.0 (2) | 4950 | 1056 | 5 | +10.86 pp [+7.04, +14.79], p 0.0001, Holm 0.0003 | +9.64 pp [+5.91, +13.26], p 0.0001, Holm 0.0003 | +0.0082, p 0.182, Holm 0.182 |
| 3.0 (3) | 3327 | 2067 | 3 | +17.30 pp [+13.84, +20.87], p 0.0001, Holm 0.0003 | +14.97 pp [+11.72, +18.33], p 0.0001, Holm 0.0003 | +0.0025, p 0.422, Holm 0.422 |
| 5.0 (5) | 1329 | 1245 | 1 | +19.20 pp [+15.92, +22.47], p 0.0001, Holm 0.0003 | +17.46 pp [+13.94, +20.88], p 0.0001, Holm 0.0003 | -0.0006, p 0.518, Holm 0.518 |
