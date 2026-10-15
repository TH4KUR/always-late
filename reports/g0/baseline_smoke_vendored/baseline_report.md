| condition | ppl (rep / pub) | ppl delta % (rep / pub) | survived % (rep / pub) | d pp | median r (rep / pub) | active (rep / pub) |
|---|---|---|---|---|---|---|
| FP16 baseline | 414.1 / 458.5 | 0.00 / 0.00 | 100.00 / 100.00 | 0.00 | 1.0000 / 1.0000 | 6702 / 7144 |
| RTN INT8 | 422.6 / 470.6 | 2.06 / 2.62 | 96.15 / 99.09 | -2.94 | 0.9713 / 0.9707 | 6702 / 7144 |
| RTN INT6 | 427.6 / 476.8 | 3.26 / 3.99 | 49.82 / 51.30 | -1.48 | 0.8994 / 0.9027 | 6702 / 7144 |

Bands (pre-committed): int7_survival_within_5pp=n/a, int6_survival_within_10pp=True, active_features_within_10pct=True, ppl_signs_int7_improves_int6_worsens=n/a
**Q1 (published numbers reproduced): NOT EVALUABLE (missing conditions)**
Per-feature agreement at INT6 over 6271 common active features: Spearman(r)=0.9264677496909866, Spearman(firing rate)=0.9175498262371911
