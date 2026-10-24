# Functional effects of language feature groups on Belebele (T-MC, letter format, group size K64)

**EXPLORATORY positive control of the Belebele edit path (plan Task 17): in this run K64 stands for a language-independent random set of the whole pool, 2,860 features of the non-selective supported features, and the controls are random sets of the same size from the selective features of the other languages; evaluation split, frozen items; full precision only; the retention and statements of the frozen family do not apply**

E is the harm (nats, the drop in the log-probability of the correct option among the four, after the edit minus before it, same precision, same items) of zero-ablating the language's selective group beyond the mean harm of its activity-matched controls; positive = the selective group matters more than equally active non-selective features. Intervals: 2000 paired cluster-bootstrap draws over passages (the same draws for every language and precision; percentile, descriptive; the observed value is the estimate). Macro retention is the mean over languages of the per-language ratio E at a precision over E in full precision (D-041), classified against the bands [0.75, 1.33] using 99 % intervals (the effect intervals are 95 % intervals); undefined when the full-precision 95 % interval does not lie above zero.

Covered languages: 18 (eng_Latn, deu_Latn, spa_Latn, fra_Latn, por_Latn, rus_Cyrl, zho_Hans, jpn_Jpan, kor_Hang, hin_Deva, ben_Beng, tur_Latn, vie_Latn, ind_Latn, tha_Thai, pol_Latn, ukr_Cyrl, kan_Knda). Passages: 219.

## Macro effect and retention

| precision | mean E | 95 % interval | median E | languages with E > 0 | retention (mean of per-language ratios) | 99 % interval | class (languages) |
|---|---|---|---|---|---|---|---|
| FP | -0.0410 | [-0.0649, -0.0156] | -0.0279 | 0 of 18 | - | - | reference |

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
| FP | eng_Latn | -0.0790 | [-0.1094, -0.0485] | 0.0616 | 0.1407 | 385 |
| FP | deu_Latn | -0.0253 | [-0.0633, 0.0111] | 0.0635 | 0.0888 | 385 |
| FP | spa_Latn | -0.0305 | [-0.0663, 0.0025] | 0.0436 | 0.0741 | 385 |
| FP | fra_Latn | -0.0945 | [-0.1455, -0.0450] | 0.0512 | 0.1457 | 385 |
| FP | por_Latn | -0.1123 | [-0.1438, -0.0809] | 0.0290 | 0.1413 | 385 |
| FP | rus_Cyrl | -0.0516 | [-0.0891, -0.0123] | 0.0694 | 0.1210 | 385 |
| FP | zho_Hans | -0.0156 | [-0.0604, 0.0280] | 0.0612 | 0.0768 | 385 |
| FP | jpn_Jpan | -0.0241 | [-0.0540, 0.0041] | 0.0439 | 0.0680 | 385 |
| FP | kor_Hang | -0.0334 | [-0.0559, -0.0104] | 0.0496 | 0.0830 | 385 |
| FP | hin_Deva | -0.0678 | [-0.1068, -0.0286] | 0.0402 | 0.1081 | 385 |
| FP | ben_Beng | -0.0254 | [-0.0642, 0.0137] | 0.0383 | 0.0637 | 385 |
| FP | tur_Latn | -0.0145 | [-0.0485, 0.0203] | 0.0396 | 0.0542 | 385 |
| FP | vie_Latn | -0.0117 | [-0.0416, 0.0195] | 0.0322 | 0.0439 | 385 |
| FP | ind_Latn | -0.0226 | [-0.0648, 0.0190] | 0.0512 | 0.0738 | 385 |
| FP | tha_Thai | -0.0224 | [-0.0517, 0.0049] | 0.0326 | 0.0550 | 385 |
| FP | pol_Latn | -0.0329 | [-0.0555, -0.0093] | 0.0387 | 0.0715 | 385 |
| FP | ukr_Cyrl | -0.0087 | [-0.0383, 0.0209] | 0.0397 | 0.0484 | 385 |
| FP | kan_Knda | -0.0655 | [-0.0995, -0.0289] | 0.0119 | 0.0773 | 385 |

## Accuracy (derived from the same scores; secondary)

| precision | language | items | none | zero-ablation | control mean | sham |
|---|---|---|---|---|---|---|
| FP | eng_Latn | 385 | 0.548 | 0.517 | 0.448 | 0.548 |
| FP | deu_Latn | 385 | 0.468 | 0.413 | 0.416 | 0.468 |
| FP | spa_Latn | 385 | 0.475 | 0.452 | 0.421 | 0.475 |
| FP | fra_Latn | 385 | 0.462 | 0.421 | 0.333 | 0.462 |
| FP | por_Latn | 385 | 0.416 | 0.413 | 0.366 | 0.416 |
| FP | rus_Cyrl | 385 | 0.460 | 0.436 | 0.345 | 0.460 |
| FP | zho_Hans | 385 | 0.452 | 0.397 | 0.388 | 0.452 |
| FP | jpn_Jpan | 385 | 0.397 | 0.405 | 0.372 | 0.397 |
| FP | kor_Hang | 385 | 0.436 | 0.434 | 0.361 | 0.436 |
| FP | hin_Deva | 385 | 0.371 | 0.345 | 0.332 | 0.371 |
| FP | ben_Beng | 385 | 0.379 | 0.345 | 0.352 | 0.379 |
| FP | tur_Latn | 385 | 0.403 | 0.382 | 0.359 | 0.403 |
| FP | vie_Latn | 385 | 0.387 | 0.371 | 0.345 | 0.387 |
| FP | ind_Latn | 385 | 0.439 | 0.395 | 0.372 | 0.439 |
| FP | tha_Thai | 385 | 0.387 | 0.371 | 0.327 | 0.387 |
| FP | pol_Latn | 385 | 0.423 | 0.390 | 0.377 | 0.423 |
| FP | ukr_Cyrl | 385 | 0.421 | 0.423 | 0.374 | 0.421 |
| FP | kan_Knda | 385 | 0.338 | 0.356 | 0.289 | 0.338 |
