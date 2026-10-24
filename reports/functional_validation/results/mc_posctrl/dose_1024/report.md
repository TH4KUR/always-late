# Functional effects of language feature groups on Belebele (T-MC, letter format, group size K64)

**EXPLORATORY positive control of the Belebele edit path (plan Task 17): in this run K64 stands for a language-independent random set of 1,024 features of the pool of the non-selective supported features, and the controls are random sets of the same size from the selective features of the other languages; evaluation split, frozen items; full precision only; the retention and statements of the frozen family do not apply**

E is the harm (nats, the drop in the log-probability of the correct option among the four, after the edit minus before it, same precision, same items) of zero-ablating the language's selective group beyond the mean harm of its activity-matched controls; positive = the selective group matters more than equally active non-selective features. Intervals: 2000 paired cluster-bootstrap draws over passages (the same draws for every language and precision; percentile, descriptive; the observed value is the estimate). Macro retention is the mean over languages of the per-language ratio E at a precision over E in full precision (D-041), classified against the bands [0.75, 1.33] using 99 % intervals (the effect intervals are 95 % intervals); undefined when the full-precision 95 % interval does not lie above zero.

Covered languages: 18 (eng_Latn, deu_Latn, spa_Latn, fra_Latn, por_Latn, rus_Cyrl, zho_Hans, jpn_Jpan, kor_Hang, hin_Deva, ben_Beng, tur_Latn, vie_Latn, ind_Latn, tha_Thai, pol_Latn, ukr_Cyrl, kan_Knda). Passages: 219.

## Macro effect and retention

| precision | mean E | 95 % interval | median E | languages with E > 0 | retention (mean of per-language ratios) | 99 % interval | class (languages) |
|---|---|---|---|---|---|---|---|
| FP | -0.0056 | [-0.0154, 0.0042] | -0.0011 | 8 of 18 | - | - | reference |

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

## Effect by language

| precision | language | E | interval | group harm | control harm | items |
|---|---|---|---|---|---|---|
| FP | eng_Latn | 0.0058 | [-0.0156, 0.0269] | 0.0555 | 0.0497 | 385 |
| FP | deu_Latn | 0.0181 | [0.0068, 0.0300] | 0.0418 | 0.0238 | 385 |
| FP | spa_Latn | -0.0011 | [-0.0179, 0.0158] | 0.0324 | 0.0335 | 385 |
| FP | fra_Latn | 0.0173 | [0.0019, 0.0335] | 0.0304 | 0.0131 | 385 |
| FP | por_Latn | -0.0072 | [-0.0219, 0.0083] | 0.0214 | 0.0286 | 385 |
| FP | rus_Cyrl | 0.0150 | [-0.0006, 0.0313] | 0.0422 | 0.0272 | 385 |
| FP | zho_Hans | -0.0232 | [-0.0418, -0.0034] | 0.0336 | 0.0568 | 385 |
| FP | jpn_Jpan | -0.0350 | [-0.0562, -0.0135] | 0.0203 | 0.0552 | 385 |
| FP | kor_Hang | 0.0120 | [0.0013, 0.0229] | 0.0310 | 0.0190 | 385 |
| FP | hin_Deva | -0.0301 | [-0.0515, -0.0082] | 0.0032 | 0.0333 | 385 |
| FP | ben_Beng | -0.0270 | [-0.0411, -0.0125] | 0.0045 | 0.0315 | 385 |
| FP | tur_Latn | -0.0012 | [-0.0161, 0.0146] | 0.0162 | 0.0174 | 385 |
| FP | vie_Latn | 0.0070 | [-0.0056, 0.0205] | 0.0082 | 0.0011 | 385 |
| FP | ind_Latn | -0.0058 | [-0.0207, 0.0093] | 0.0216 | 0.0274 | 385 |
| FP | tha_Thai | -0.0043 | [-0.0165, 0.0074] | 0.0064 | 0.0107 | 385 |
| FP | pol_Latn | 0.0035 | [-0.0178, 0.0256] | 0.0269 | 0.0234 | 385 |
| FP | ukr_Cyrl | 0.0033 | [-0.0104, 0.0170] | 0.0151 | 0.0118 | 385 |
| FP | kan_Knda | -0.0488 | [-0.0767, -0.0220] | -0.0069 | 0.0419 | 385 |

## Accuracy (derived from the same scores; secondary)

| precision | language | items | none | zero-ablation | control mean | sham |
|---|---|---|---|---|---|---|
| FP | eng_Latn | 385 | 0.548 | 0.525 | 0.508 | 0.548 |
| FP | deu_Latn | 385 | 0.468 | 0.421 | 0.425 | 0.468 |
| FP | spa_Latn | 385 | 0.475 | 0.462 | 0.425 | 0.475 |
| FP | fra_Latn | 385 | 0.462 | 0.444 | 0.442 | 0.462 |
| FP | por_Latn | 385 | 0.416 | 0.416 | 0.409 | 0.416 |
| FP | rus_Cyrl | 385 | 0.460 | 0.452 | 0.436 | 0.460 |
| FP | zho_Hans | 385 | 0.452 | 0.421 | 0.419 | 0.452 |
| FP | jpn_Jpan | 385 | 0.397 | 0.403 | 0.380 | 0.397 |
| FP | kor_Hang | 385 | 0.436 | 0.429 | 0.442 | 0.436 |
| FP | hin_Deva | 385 | 0.371 | 0.366 | 0.373 | 0.371 |
| FP | ben_Beng | 385 | 0.379 | 0.371 | 0.357 | 0.379 |
| FP | tur_Latn | 385 | 0.403 | 0.366 | 0.395 | 0.403 |
| FP | vie_Latn | 385 | 0.387 | 0.382 | 0.385 | 0.387 |
| FP | ind_Latn | 385 | 0.439 | 0.421 | 0.410 | 0.439 |
| FP | tha_Thai | 385 | 0.387 | 0.377 | 0.363 | 0.387 |
| FP | pol_Latn | 385 | 0.423 | 0.418 | 0.418 | 0.423 |
| FP | ukr_Cyrl | 385 | 0.421 | 0.431 | 0.412 | 0.421 |
| FP | kan_Knda | 385 | 0.338 | 0.343 | 0.314 | 0.338 |
