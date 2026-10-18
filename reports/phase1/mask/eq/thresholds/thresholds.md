# Fixed-threshold sensitivity of the selective split, regime `mask`, stream `eq` (decision D-019)

**Labelled deviation from the proposal (section 4.1):** the proposal sets the binary threshold at percentiles of the shuffled-label null of S; the runs saved only the 95th-percentile cutoff (1.8329), not the null samples, so fixed S thresholds are reported instead (DECISIONS_FOR_REVIEW.md Q6). The rule is the run's own: selective = supported, peak language stable in at least 80 of the bootstrap draws, and S > tau (strict); controls = firing-rate-stratum-matched universal features drawn by the same `matched_sample`; the estimators are the unchanged D-014 / D-016 analysis (10000 permutations, 2000 bootstrap draws, seed 0). The `null` row is the run's own cutoff and must equal the original analysis.

| tau | selective | controls | strata short | T1 INT7 selective - control (95% CI), p, Holm | T2 INT6 selective - control (95% CI), p, Holm | T3 tier slope, p, Holm |
|---|---|---|---|---|---|---|
| null (1.833) | 8086 | 2860 | 5 | -0.28 pp [-0.63, +0.13], p 0.257, Holm 0.514 | -1.44 pp [-2.65, -0.13], p 0.0427, Holm 0.128 | -0.0048, p 0.99, Holm 0.99 |
| 2.0 (2) | 7470 | 3476 | 5 | -0.27 pp [-0.65, +0.17], p 0.268, Holm 0.537 | -1.02 pp [-2.32, +0.24], p 0.157, Holm 0.472 | -0.0041, p 0.988, Holm 0.988 |
| 3.0 (3) | 3922 | 3148 | 1 | -0.42 pp [-0.88, +0.07], p 0.129, Holm 0.259 | -2.14 pp [-3.52, -0.74], p 0.0055, Holm 0.0165 | +0.0020, p 0.144, Holm 0.259 |
| 5.0 (5) | 1677 | 1677 | 0 | -0.93 pp [-1.74, -0.17], p 0.0189, Holm 0.0378 | -7.16 pp [-9.14, -5.20], p 0.0001, Holm 0.0003 | +0.0033, p 0.128, Holm 0.128 |
