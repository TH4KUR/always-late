# Fixed-threshold sensitivity of the selective split, regime `none`, stream `all` (decision D-019)

**Labelled deviation from the proposal (section 4.1):** the proposal sets the binary threshold at percentiles of the shuffled-label null of S; the runs saved only the 95th-percentile cutoff (1.8194), not the null samples, so fixed S thresholds are reported instead (DECISIONS_FOR_REVIEW.md Q6). The rule is the run's own: selective = supported, peak language stable in at least 80 of the bootstrap draws, and S > tau (strict); controls = firing-rate-stratum-matched universal features drawn by the same `matched_sample`; the estimators are the unchanged D-014 / D-016 analysis (10000 permutations, 2000 bootstrap draws, seed 0). The `null` row is the run's own cutoff and must equal the original analysis.

| tau | selective | controls | strata short | T1 INT7 selective - control (95% CI), p, Holm | T2 INT6 selective - control (95% CI), p, Holm | T3 tier slope, p, Holm |
|---|---|---|---|---|---|---|
| null (1.819) | 6037 | 973 | 5 | +8.74 pp [+5.00, +12.08], p 0.0001, Holm 0.0003 | +4.08 pp [+0.45, +7.53], p 0.0368, Holm 0.0736 | +0.0080, p 0.236, Holm 0.236 |
| 2.0 (2) | 5827 | 1183 | 5 | +11.86 pp [+8.16, +15.48], p 0.0001, Holm 0.0003 | +7.45 pp [+3.90, +10.93], p 0.0001, Holm 0.0003 | +0.0094, p 0.206, Holm 0.206 |
| 3.0 (3) | 3968 | 2458 | 3 | +19.38 pp [+16.32, +22.63], p 0.0001, Holm 0.0003 | +15.76 pp [+12.69, +18.64], p 0.0001, Holm 0.0003 | +0.0017, p 0.439, Holm 0.439 |
| 5.0 (5) | 1605 | 1511 | 1 | +14.08 pp [+10.98, +17.20], p 0.0001, Holm 0.0003 | +12.72 pp [+9.32, +15.98], p 0.0001, Holm 0.0003 | -0.0171, p 0.962, Holm 0.962 |
