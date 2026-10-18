# Fixed-threshold sensitivity of the selective split, regime `mask`, stream `all` (decision D-019)

**Labelled deviation from the proposal (section 4.1):** the proposal sets the binary threshold at percentiles of the shuffled-label null of S; the runs saved only the 95th-percentile cutoff (1.8379), not the null samples, so fixed S thresholds are reported instead (DECISIONS_FOR_REVIEW.md Q6). The rule is the run's own: selective = supported, peak language stable in at least 80 of the bootstrap draws, and S > tau (strict); controls = firing-rate-stratum-matched universal features drawn by the same `matched_sample`; the estimators are the unchanged D-014 / D-016 analysis (10000 permutations, 2000 bootstrap draws, seed 0). The `null` row is the run's own cutoff and must equal the original analysis.

| tau | selective | controls | strata short | T1 INT7 selective - control (95% CI), p, Holm | T2 INT6 selective - control (95% CI), p, Holm | T3 tier slope, p, Holm |
|---|---|---|---|---|---|---|
| null (1.838) | 8845 | 2948 | 5 | -0.39 pp [-0.74, +0.02], p 0.165, Holm 0.33 | -1.93 pp [-3.32, -0.46], p 0.0183, Holm 0.0549 | -0.0035, p 0.93, Holm 0.93 |
| 2.0 (2) | 8221 | 3572 | 5 | -0.37 pp [-0.73, +0.04], p 0.227, Holm 0.453 | -1.89 pp [-3.28, -0.41], p 0.0273, Holm 0.0819 | -0.0034, p 0.966, Holm 0.966 |
| 3.0 (3) | 4504 | 3343 | 1 | -0.75 pp [-1.24, -0.21], p 0.0235, Holm 0.047 | -4.04 pp [-5.64, -2.48], p 0.0001, Holm 0.0003 | +0.0024, p 0.101, Holm 0.101 |
| 5.0 (5) | 2076 | 2076 | 0 | -1.74 pp [-2.52, -0.96], p 0.0001, Holm 0.0003 | -8.24 pp [-10.27, -6.19], p 0.0001, Holm 0.0003 | +0.0028, p 0.214, Holm 0.214 |
