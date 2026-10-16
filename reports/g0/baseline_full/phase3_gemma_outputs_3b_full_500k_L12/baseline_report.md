| condition | ppl (rep / pub) | ppl delta % (rep / pub) | survived % (rep / pub) | d pp | median r (rep / pub) | active (rep / pub) |
|---|---|---|---|---|---|---|
| FP16 baseline | 458.4 / 458.5 | 0.00 / 0.00 | 100.00 / 100.00 | 0.00 | 1.0000 / 1.0000 | 7134 / 7144 |
| RTN INT8 | 469.9 / 470.6 | 2.52 / 2.62 | 99.12 / 99.09 | 0.03 | 0.9707 / 0.9707 | 7134 / 7144 |
| RTN INT7 | 432.7 / 432.6 | -5.61 / -5.65 | 81.19 / 81.26 | -0.07 | 0.9499 / 0.9499 | 7134 / 7144 |
| RTN INT6 | 476.8 / 476.8 | 4.02 / 3.99 | 51.15 / 51.30 | -0.15 | 0.9027 / 0.9027 | 7134 / 7144 |
| RTN INT5 | 539.7 / 539.8 | 17.73 / 17.72 | 27.57 / 27.72 | -0.14 | 0.7775 / 0.7780 | 7134 / 7144 |
| RTN INT4 | 670.4 / 669.8 | 46.25 / 46.08 | 12.39 / 12.40 | -0.01 | 0.4504 / 0.4512 | 7134 / 7144 |

Bands (pre-committed): int7_survival_within_5pp=True, int6_survival_within_10pp=True, active_features_within_10pct=True, ppl_signs_int7_improves_int6_worsens=True
**Q1 (published numbers reproduced): True**
Per-feature agreement at INT6 over 7121 common active features: Spearman(r)=0.9996451864656191, Spearman(firing rate)=0.9999528521444792
