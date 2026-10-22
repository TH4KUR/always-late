# Functional effects of language feature groups on Belebele (T-MC, letter format, group size K64)

**Calibration-split run (dev-origin passages, the text the features were selected on): a pipeline check, not a result.**

E is the harm (nats, the drop in the log-probability of the correct option among the four, after the edit minus before it, same precision, same items) of zero-ablating the language's selective group beyond the mean harm of its activity-matched controls; positive = the selective group matters more than equally active non-selective features. Intervals: 2000 paired cluster-bootstrap draws over passages (the same draws for every language and precision; percentile, descriptive; the observed value is the estimate). Macro retention is the mean over languages of the per-language ratio E at a precision over E in full precision (D-041), classified against the bands [0.75, 1.33] using 99 % intervals (the effect intervals are 95 % intervals); undefined when the full-precision 95 % interval does not lie above zero.

Covered languages: 18 (eng_Latn, deu_Latn, spa_Latn, fra_Latn, por_Latn, rus_Cyrl, zho_Hans, jpn_Jpan, kor_Hang, hin_Deva, ben_Beng, tur_Latn, vie_Latn, ind_Latn, tha_Thai, pol_Latn, ukr_Cyrl, kan_Knda). Passages: 109.

## Macro effect and retention

| precision | mean E | 95 % interval | median E | languages with E > 0 | retention (mean of per-language ratios) | 99 % interval | class (languages) |
|---|---|---|---|---|---|---|---|
| FP | 0.0003 | [-0.0057, 0.0066] | -0.0000 | 9 of 18 | - | - | reference |
| INT4 | 0.0035 | [-0.0031, 0.0094] | 0.0121 | 12 of 18 | 0.07 | [-2.87, 3.93] | inconclusive (1) |

## Sham check (the edit path with no intended change; must be exactly 0)

| precision | language | harm | interval |
|---|---|---|---|
| FP | eng_Latn | 0.00000 | [0.00000, 0.00000] |
| FP | deu_Latn | 0.00000 | [0.00000, 0.00000] |
| FP | spa_Latn | 0.00000 | [0.00000, 0.00000] |
| FP | fra_Latn | 0.00000 | [0.00000, 0.00000] |
| FP | por_Latn | 0.00000 | [0.00000, 0.00000] |
| FP | rus_Cyrl | 0.00000 | [0.00000, 0.00000] |
| FP | zho_Hans | 0.00000 | [0.00000, 0.00000] |
| FP | jpn_Jpan | 0.00000 | [0.00000, 0.00000] |
| FP | kor_Hang | 0.00000 | [0.00000, 0.00000] |
| FP | hin_Deva | 0.00000 | [0.00000, 0.00000] |
| FP | ben_Beng | 0.00000 | [0.00000, 0.00000] |
| FP | tur_Latn | 0.00000 | [0.00000, 0.00000] |
| FP | vie_Latn | 0.00000 | [0.00000, 0.00000] |
| FP | ind_Latn | 0.00000 | [0.00000, 0.00000] |
| FP | tha_Thai | 0.00000 | [0.00000, 0.00000] |
| FP | pol_Latn | 0.00000 | [0.00000, 0.00000] |
| FP | ukr_Cyrl | 0.00000 | [0.00000, 0.00000] |
| FP | kan_Knda | 0.00000 | [0.00000, 0.00000] |
| INT4 | eng_Latn | 0.00000 | [0.00000, 0.00000] |
| INT4 | deu_Latn | 0.00000 | [0.00000, 0.00000] |
| INT4 | spa_Latn | 0.00000 | [0.00000, 0.00000] |
| INT4 | fra_Latn | 0.00000 | [0.00000, 0.00000] |
| INT4 | por_Latn | 0.00000 | [0.00000, 0.00000] |
| INT4 | rus_Cyrl | 0.00000 | [0.00000, 0.00000] |
| INT4 | zho_Hans | 0.00000 | [0.00000, 0.00000] |
| INT4 | jpn_Jpan | 0.00000 | [0.00000, 0.00000] |
| INT4 | kor_Hang | 0.00000 | [0.00000, 0.00000] |
| INT4 | hin_Deva | 0.00000 | [0.00000, 0.00000] |
| INT4 | ben_Beng | 0.00000 | [0.00000, 0.00000] |
| INT4 | tur_Latn | 0.00000 | [0.00000, 0.00000] |
| INT4 | vie_Latn | 0.00000 | [0.00000, 0.00000] |
| INT4 | ind_Latn | 0.00000 | [0.00000, 0.00000] |
| INT4 | tha_Thai | 0.00000 | [0.00000, 0.00000] |
| INT4 | pol_Latn | 0.00000 | [0.00000, 0.00000] |
| INT4 | ukr_Cyrl | 0.00000 | [0.00000, 0.00000] |
| INT4 | kan_Knda | 0.00000 | [0.00000, 0.00000] |

## Effect by language

| precision | language | E | interval | group harm | control harm | items |
|---|---|---|---|---|---|---|
| FP | eng_Latn | 0.0075 | [-0.0066, 0.0229] | 0.0060 | -0.0015 | 120 |
| FP | deu_Latn | 0.0042 | [-0.0080, 0.0162] | 0.0074 | 0.0032 | 120 |
| FP | spa_Latn | 0.0006 | [-0.0114, 0.0126] | -0.0032 | -0.0038 | 120 |
| FP | fra_Latn | -0.0148 | [-0.0273, -0.0031] | -0.0076 | 0.0073 | 120 |
| FP | por_Latn | -0.0007 | [-0.0199, 0.0185] | 0.0024 | 0.0031 | 120 |
| FP | rus_Cyrl | -0.0064 | [-0.0209, 0.0073] | 0.0031 | 0.0095 | 120 |
| FP | zho_Hans | 0.0026 | [-0.0146, 0.0200] | 0.0086 | 0.0059 | 120 |
| FP | jpn_Jpan | -0.0048 | [-0.0235, 0.0133] | 0.0117 | 0.0166 | 120 |
| FP | kor_Hang | -0.0048 | [-0.0175, 0.0085] | -0.0061 | -0.0012 | 120 |
| FP | hin_Deva | 0.0054 | [-0.0204, 0.0327] | 0.0099 | 0.0045 | 120 |
| FP | ben_Beng | -0.0036 | [-0.0138, 0.0065] | 0.0014 | 0.0050 | 120 |
| FP | tur_Latn | 0.0143 | [-0.0020, 0.0302] | 0.0136 | -0.0007 | 120 |
| FP | vie_Latn | -0.0047 | [-0.0384, 0.0274] | -0.0112 | -0.0065 | 120 |
| FP | ind_Latn | 0.0049 | [-0.0116, 0.0208] | 0.0017 | -0.0032 | 120 |
| FP | tha_Thai | 0.0008 | [-0.0137, 0.0150] | 0.0022 | 0.0014 | 120 |
| FP | pol_Latn | 0.0186 | [0.0045, 0.0329] | 0.0009 | -0.0177 | 120 |
| FP | ukr_Cyrl | -0.0051 | [-0.0199, 0.0097] | -0.0027 | 0.0025 | 120 |
| FP | kan_Knda | -0.0081 | [-0.0187, 0.0030] | -0.0012 | 0.0069 | 120 |
| INT4 | eng_Latn | -0.0130 | [-0.0339, 0.0063] | -0.0137 | -0.0007 | 120 |
| INT4 | deu_Latn | -0.0257 | [-0.0436, -0.0081] | -0.0119 | 0.0138 | 120 |
| INT4 | spa_Latn | -0.0207 | [-0.0353, -0.0072] | -0.0154 | 0.0053 | 120 |
| INT4 | fra_Latn | 0.0109 | [-0.0075, 0.0290] | 0.0029 | -0.0080 | 120 |
| INT4 | por_Latn | -0.0465 | [-0.0703, -0.0238] | -0.0585 | -0.0120 | 120 |
| INT4 | rus_Cyrl | 0.0157 | [-0.0002, 0.0317] | 0.0099 | -0.0059 | 120 |
| INT4 | zho_Hans | -0.0646 | [-0.0912, -0.0396] | -0.0777 | -0.0131 | 120 |
| INT4 | jpn_Jpan | 0.0308 | [-0.0057, 0.0679] | -0.0061 | -0.0369 | 120 |
| INT4 | kor_Hang | 0.0446 | [0.0267, 0.0609] | 0.0339 | -0.0106 | 120 |
| INT4 | hin_Deva | 0.0299 | [0.0082, 0.0515] | 0.0224 | -0.0075 | 120 |
| INT4 | ben_Beng | 0.0013 | [-0.0096, 0.0123] | -0.0014 | -0.0027 | 120 |
| INT4 | tur_Latn | 0.0158 | [-0.0014, 0.0333] | 0.0109 | -0.0049 | 120 |
| INT4 | vie_Latn | -0.0085 | [-0.0473, 0.0265] | -0.0368 | -0.0283 | 120 |
| INT4 | ind_Latn | 0.0134 | [-0.0035, 0.0302] | -0.0188 | -0.0322 | 120 |
| INT4 | tha_Thai | 0.0439 | [0.0230, 0.0631] | 0.0387 | -0.0052 | 120 |
| INT4 | pol_Latn | 0.0013 | [-0.0133, 0.0160] | -0.0143 | -0.0157 | 120 |
| INT4 | ukr_Cyrl | 0.0183 | [0.0016, 0.0348] | 0.0072 | -0.0111 | 120 |
| INT4 | kan_Knda | 0.0166 | [0.0050, 0.0286] | 0.0090 | -0.0076 | 120 |

## Accuracy (derived from the same scores; secondary)

| precision | language | items | none | zero-ablation | control mean | sham |
|---|---|---|---|---|---|---|
| FP | eng_Latn | 120 | 0.583 | 0.567 | 0.581 | 0.583 |
| FP | deu_Latn | 120 | 0.567 | 0.567 | 0.561 | 0.567 |
| FP | spa_Latn | 120 | 0.525 | 0.525 | 0.528 | 0.525 |
| FP | fra_Latn | 120 | 0.517 | 0.508 | 0.497 | 0.517 |
| FP | por_Latn | 120 | 0.567 | 0.533 | 0.544 | 0.567 |
| FP | rus_Cyrl | 120 | 0.467 | 0.475 | 0.453 | 0.467 |
| FP | zho_Hans | 120 | 0.475 | 0.483 | 0.478 | 0.475 |
| FP | jpn_Jpan | 120 | 0.400 | 0.375 | 0.406 | 0.400 |
| FP | kor_Hang | 120 | 0.450 | 0.467 | 0.456 | 0.450 |
| FP | hin_Deva | 120 | 0.350 | 0.350 | 0.358 | 0.350 |
| FP | ben_Beng | 120 | 0.367 | 0.367 | 0.378 | 0.367 |
| FP | tur_Latn | 120 | 0.417 | 0.408 | 0.411 | 0.417 |
| FP | vie_Latn | 120 | 0.458 | 0.458 | 0.489 | 0.458 |
| FP | ind_Latn | 120 | 0.442 | 0.450 | 0.461 | 0.442 |
| FP | tha_Thai | 120 | 0.467 | 0.467 | 0.456 | 0.467 |
| FP | pol_Latn | 120 | 0.417 | 0.408 | 0.422 | 0.417 |
| FP | ukr_Cyrl | 120 | 0.425 | 0.450 | 0.436 | 0.425 |
| FP | kan_Knda | 120 | 0.358 | 0.358 | 0.344 | 0.358 |
| INT4 | eng_Latn | 120 | 0.342 | 0.367 | 0.314 | 0.342 |
| INT4 | deu_Latn | 120 | 0.250 | 0.233 | 0.236 | 0.250 |
| INT4 | spa_Latn | 120 | 0.258 | 0.267 | 0.253 | 0.258 |
| INT4 | fra_Latn | 120 | 0.267 | 0.267 | 0.267 | 0.267 |
| INT4 | por_Latn | 120 | 0.300 | 0.317 | 0.300 | 0.300 |
| INT4 | rus_Cyrl | 120 | 0.250 | 0.242 | 0.256 | 0.250 |
| INT4 | zho_Hans | 120 | 0.308 | 0.275 | 0.283 | 0.308 |
| INT4 | jpn_Jpan | 120 | 0.258 | 0.258 | 0.267 | 0.258 |
| INT4 | kor_Hang | 120 | 0.267 | 0.267 | 0.267 | 0.267 |
| INT4 | hin_Deva | 120 | 0.250 | 0.250 | 0.267 | 0.250 |
| INT4 | ben_Beng | 120 | 0.258 | 0.258 | 0.256 | 0.258 |
| INT4 | tur_Latn | 120 | 0.267 | 0.258 | 0.281 | 0.267 |
| INT4 | vie_Latn | 120 | 0.283 | 0.267 | 0.267 | 0.283 |
| INT4 | ind_Latn | 120 | 0.242 | 0.250 | 0.242 | 0.242 |
| INT4 | tha_Thai | 120 | 0.250 | 0.250 | 0.256 | 0.250 |
| INT4 | pol_Latn | 120 | 0.283 | 0.267 | 0.283 | 0.283 |
| INT4 | ukr_Cyrl | 120 | 0.242 | 0.242 | 0.242 | 0.242 |
| INT4 | kan_Knda | 120 | 0.250 | 0.250 | 0.250 | 0.250 |
