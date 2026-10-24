# Functional effects of language feature groups on Belebele (T-MC, letter format, group size K64)

**EXPLORATORY positive control of the Belebele edit path (plan Task 17): in this run K64 stands for a language-independent random set of 256 features of the pool of the non-selective supported features, and the controls are random sets of the same size from the selective features of the other languages; evaluation split, frozen items; full precision only; the retention and statements of the frozen family do not apply**

E is the harm (nats, the drop in the log-probability of the correct option among the four, after the edit minus before it, same precision, same items) of zero-ablating the language's selective group beyond the mean harm of its activity-matched controls; positive = the selective group matters more than equally active non-selective features. Intervals: 2000 paired cluster-bootstrap draws over passages (the same draws for every language and precision; percentile, descriptive; the observed value is the estimate). Macro retention is the mean over languages of the per-language ratio E at a precision over E in full precision (D-041), classified against the bands [0.75, 1.33] using 99 % intervals (the effect intervals are 95 % intervals); undefined when the full-precision 95 % interval does not lie above zero.

Covered languages: 18 (eng_Latn, deu_Latn, spa_Latn, fra_Latn, por_Latn, rus_Cyrl, zho_Hans, jpn_Jpan, kor_Hang, hin_Deva, ben_Beng, tur_Latn, vie_Latn, ind_Latn, tha_Thai, pol_Latn, ukr_Cyrl, kan_Knda). Passages: 219.

## Macro effect and retention

| precision | mean E | 95 % interval | median E | languages with E > 0 | retention (mean of per-language ratios) | 99 % interval | class (languages) |
|---|---|---|---|---|---|---|---|
| FP | -0.0032 | [-0.0060, -0.0003] | -0.0027 | 5 of 18 | - | - | reference |

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
| FP | eng_Latn | -0.0096 | [-0.0278, 0.0086] | 0.0109 | 0.0205 | 385 |
| FP | deu_Latn | 0.0025 | [-0.0042, 0.0094] | 0.0081 | 0.0057 | 385 |
| FP | spa_Latn | 0.0066 | [-0.0021, 0.0151] | 0.0075 | 0.0009 | 385 |
| FP | fra_Latn | 0.0071 | [0.0001, 0.0143] | 0.0102 | 0.0030 | 385 |
| FP | por_Latn | -0.0021 | [-0.0088, 0.0048] | 0.0061 | 0.0082 | 385 |
| FP | rus_Cyrl | 0.0063 | [-0.0013, 0.0136] | 0.0062 | -0.0001 | 385 |
| FP | zho_Hans | -0.0091 | [-0.0169, -0.0016] | 0.0091 | 0.0182 | 385 |
| FP | jpn_Jpan | -0.0053 | [-0.0140, 0.0029] | 0.0025 | 0.0078 | 385 |
| FP | kor_Hang | -0.0004 | [-0.0077, 0.0068] | 0.0081 | 0.0085 | 385 |
| FP | hin_Deva | -0.0160 | [-0.0269, -0.0049] | 0.0024 | 0.0185 | 385 |
| FP | ben_Beng | -0.0012 | [-0.0095, 0.0068] | 0.0004 | 0.0016 | 385 |
| FP | tur_Latn | -0.0063 | [-0.0174, 0.0042] | 0.0071 | 0.0134 | 385 |
| FP | vie_Latn | -0.0008 | [-0.0073, 0.0058] | -0.0008 | -0.0001 | 385 |
| FP | ind_Latn | -0.0148 | [-0.0233, -0.0060] | -0.0018 | 0.0130 | 385 |
| FP | tha_Thai | 0.0010 | [-0.0068, 0.0084] | 0.0025 | 0.0015 | 385 |
| FP | pol_Latn | -0.0084 | [-0.0203, 0.0033] | 0.0071 | 0.0155 | 385 |
| FP | ukr_Cyrl | -0.0032 | [-0.0098, 0.0032] | -0.0012 | 0.0020 | 385 |
| FP | kan_Knda | -0.0032 | [-0.0116, 0.0042] | -0.0002 | 0.0030 | 385 |

## Accuracy (derived from the same scores; secondary)

| precision | language | items | none | zero-ablation | control mean | sham |
|---|---|---|---|---|---|---|
| FP | eng_Latn | 385 | 0.548 | 0.553 | 0.537 | 0.548 |
| FP | deu_Latn | 385 | 0.468 | 0.442 | 0.453 | 0.468 |
| FP | spa_Latn | 385 | 0.475 | 0.475 | 0.463 | 0.475 |
| FP | fra_Latn | 385 | 0.462 | 0.439 | 0.455 | 0.462 |
| FP | por_Latn | 385 | 0.416 | 0.400 | 0.410 | 0.416 |
| FP | rus_Cyrl | 385 | 0.460 | 0.447 | 0.455 | 0.460 |
| FP | zho_Hans | 385 | 0.452 | 0.434 | 0.442 | 0.452 |
| FP | jpn_Jpan | 385 | 0.397 | 0.400 | 0.392 | 0.397 |
| FP | kor_Hang | 385 | 0.436 | 0.449 | 0.434 | 0.436 |
| FP | hin_Deva | 385 | 0.371 | 0.369 | 0.364 | 0.371 |
| FP | ben_Beng | 385 | 0.379 | 0.387 | 0.376 | 0.379 |
| FP | tur_Latn | 385 | 0.403 | 0.387 | 0.394 | 0.403 |
| FP | vie_Latn | 385 | 0.387 | 0.377 | 0.388 | 0.387 |
| FP | ind_Latn | 385 | 0.439 | 0.434 | 0.424 | 0.439 |
| FP | tha_Thai | 385 | 0.387 | 0.369 | 0.370 | 0.387 |
| FP | pol_Latn | 385 | 0.423 | 0.423 | 0.423 | 0.423 |
| FP | ukr_Cyrl | 385 | 0.421 | 0.408 | 0.414 | 0.421 |
| FP | kan_Knda | 385 | 0.338 | 0.332 | 0.344 | 0.338 |
