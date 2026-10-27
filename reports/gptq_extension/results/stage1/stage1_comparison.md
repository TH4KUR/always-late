# Stage 1 comparison: ordinary loss (E1) and survival (E2), arm against arm

Exploratory (D-101). Every row is arm A minus arm B, paired by language (27 pairs); the interval is a 95 % percentile bootstrap of the mean over languages. Loss: per-language mean next-token cross-entropy of the state minus that of FP (nats per token), so a NEGATIVE difference means A is closer to FP. Survival: share of the language's FP-active features whose token correlation with FP is above 0.9 (percentage points), so a POSITIVE difference means A keeps more features. The three pre-set tests T1 to T3 are in the per-arm analysis files, not here.

## Bridge (FP numbers and legacy-rounding floors must equal the legacy run's)

| arm | ok | max abs FP loss difference | max abs FP firing-rate difference | floors that differ | token counts equal |
|---|---|---|---|---|---|
| matched RTN | yes | 0 | 0 | none | True |
| GPTQ | yes | 0 | 0 | none | True |

## matched RTN minus legacy RTN

| bits | metric | mean | median | 95 % interval of the mean | languages with A below B / above B |
|---|---|---|---|---|---|
| INT8 | nll_delta (nats/token) | +0.0006 | +0.0004 | [+0.0002, +0.0011] | 8 / 19 |
| INT8 | survived_pct (pp) | +0.0447 | +0.0104 | [+0.0132, +0.0834] | 3 / 14 |
| INT8 | median_corr (r) | +0.0012 | +0.0010 | [+0.0010, +0.0014] | 0 / 27 |
| INT7 | nll_delta (nats/token) | -0.0004 | -0.0002 | [-0.0011, +0.0003] | 15 / 12 |
| INT7 | survived_pct (pp) | +0.4397 | +0.1312 | [+0.2510, +0.6514] | 0 / 27 |
| INT7 | median_corr (r) | +0.0023 | +0.0019 | [+0.0019, +0.0027] | 0 / 27 |
| INT6 | nll_delta (nats/token) | -0.0057 | -0.0048 | [-0.0078, -0.0038] | 26 / 1 |
| INT6 | survived_pct (pp) | +4.7267 | +3.8439 | [+3.5730, +5.9734] | 0 / 27 |
| INT6 | median_corr (r) | +0.0071 | +0.0060 | [+0.0058, +0.0086] | 0 / 27 |
| INT5 | nll_delta (nats/token) | -0.0188 | -0.0148 | [-0.0239, -0.0144] | 27 / 0 |
| INT5 | survived_pct (pp) | +7.9029 | +7.9253 | [+7.4443, +8.3514] | 0 / 27 |
| INT5 | median_corr (r) | +0.0166 | +0.0145 | [+0.0142, +0.0192] | 0 / 27 |
| INT4 | nll_delta (nats/token) | -0.2056 | -0.1670 | [-0.2681, -0.1543] | 27 / 0 |
| INT4 | survived_pct (pp) | +5.9675 | +6.6490 | [+4.9759, +6.9528] | 0 / 27 |
| INT4 | median_corr (r) | +0.0779 | +0.0725 | [+0.0686, +0.0874] | 0 / 27 |

## GPTQ minus matched RTN

| bits | metric | mean | median | 95 % interval of the mean | languages with A below B / above B |
|---|---|---|---|---|---|
| INT8 | nll_delta (nats/token) | +0.0001 | +0.0005 | [-0.0008, +0.0008] | 11 / 16 |
| INT8 | survived_pct (pp) | +0.0534 | +0.0107 | [+0.0214, +0.0943] | 2 / 16 |
| INT8 | median_corr (r) | +0.0023 | +0.0016 | [+0.0018, +0.0028] | 0 / 27 |
| INT7 | nll_delta (nats/token) | -0.0022 | -0.0020 | [-0.0030, -0.0015] | 23 / 4 |
| INT7 | survived_pct (pp) | +0.7800 | +0.1457 | [+0.3719, +1.2670] | 0 / 27 |
| INT7 | median_corr (r) | +0.0055 | +0.0040 | [+0.0043, +0.0069] | 0 / 27 |
| INT6 | nll_delta (nats/token) | -0.0139 | -0.0120 | [-0.0168, -0.0114] | 27 / 0 |
| INT6 | survived_pct (pp) | +7.7551 | +3.9893 | [+5.1234, +10.6845] | 0 / 27 |
| INT6 | median_corr (r) | +0.0130 | +0.0095 | [+0.0101, +0.0163] | 0 / 27 |
| INT5 | nll_delta (nats/token) | -0.0450 | -0.0358 | [-0.0588, -0.0330] | 26 / 1 |
| INT5 | survived_pct (pp) | +19.0222 | +18.1878 | [+16.3850, +21.7769] | 0 / 27 |
| INT5 | median_corr (r) | +0.0315 | +0.0237 | [+0.0248, +0.0387] | 0 / 27 |
| INT4 | nll_delta (nats/token) | -0.2552 | -0.1994 | [-0.3270, -0.1933] | 27 / 0 |
| INT4 | survived_pct (pp) | +11.8594 | +12.0951 | [+11.1789, +12.5007] | 0 / 27 |
| INT4 | median_corr (r) | +0.0908 | +0.0679 | [+0.0720, +0.1113] | 0 / 27 |

## GPTQ minus legacy RTN

| bits | metric | mean | median | 95 % interval of the mean | languages with A below B / above B |
|---|---|---|---|---|---|
| INT8 | nll_delta (nats/token) | +0.0006 | +0.0007 | [-0.0000, +0.0013] | 10 / 17 |
| INT8 | survived_pct (pp) | +0.0981 | +0.0208 | [+0.0397, +0.1687] | 2 / 19 |
| INT8 | median_corr (r) | +0.0034 | +0.0026 | [+0.0028, +0.0042] | 0 / 27 |
| INT7 | nll_delta (nats/token) | -0.0026 | -0.0024 | [-0.0035, -0.0017] | 25 / 2 |
| INT7 | survived_pct (pp) | +1.2197 | +0.2405 | [+0.6244, +1.9113] | 0 / 27 |
| INT7 | median_corr (r) | +0.0078 | +0.0059 | [+0.0062, +0.0095] | 0 / 27 |
| INT6 | nll_delta (nats/token) | -0.0196 | -0.0179 | [-0.0229, -0.0167] | 27 / 0 |
| INT6 | survived_pct (pp) | +12.4818 | +8.2726 | [+8.7239, +16.6854] | 0 / 27 |
| INT6 | median_corr (r) | +0.0201 | +0.0153 | [+0.0158, +0.0247] | 0 / 27 |
| INT5 | nll_delta (nats/token) | -0.0637 | -0.0487 | [-0.0805, -0.0489] | 27 / 0 |
| INT5 | survived_pct (pp) | +26.9251 | +26.8775 | [+24.3087, +29.6548] | 0 / 27 |
| INT5 | median_corr (r) | +0.0481 | +0.0385 | [+0.0389, +0.0579] | 0 / 27 |
| INT4 | nll_delta (nats/token) | -0.4608 | -0.3660 | [-0.5945, -0.3483] | 27 / 0 |
| INT4 | survived_pct (pp) | +17.8270 | +18.6887 | [+16.2944, +19.3817] | 0 / 27 |
| INT4 | median_corr (r) | +0.1687 | +0.1448 | [+0.1410, +0.1985] | 0 / 27 |
