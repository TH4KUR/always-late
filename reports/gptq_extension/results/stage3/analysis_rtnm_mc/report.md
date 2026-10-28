# Functional effects of language feature groups on Belebele (T-MC, letter format, group size K64)

**GPTQ extension subset (FP, INT6, INT4); the T-MC settings are the frozen ones.**

E is the harm (nats, the drop in the log-probability of the correct option among the four, after the edit minus before it, same precision, same items) of zero-ablating the language's selective group beyond the mean harm of its activity-matched controls; positive = the selective group matters more than equally active non-selective features. Intervals: 2000 paired cluster-bootstrap draws over passages (the same draws for every language and precision; percentile, descriptive; the observed value is the estimate). Macro retention is the mean over languages of the per-language ratio E at a precision over E in full precision (D-041), classified against the bands [0.75, 1.33] using 99 % intervals (the effect intervals are 95 % intervals); undefined when the full-precision 95 % interval does not lie above zero.

Covered languages: 18 (eng_Latn, deu_Latn, spa_Latn, fra_Latn, por_Latn, rus_Cyrl, zho_Hans, jpn_Jpan, kor_Hang, hin_Deva, ben_Beng, tur_Latn, vie_Latn, ind_Latn, tha_Thai, pol_Latn, ukr_Cyrl, kan_Knda). Passages: 219.

## Macro effect and retention

| precision | mean E | 95 % interval | median E | languages with E > 0 | retention (mean of per-language ratios) | 99 % interval | class (languages) |
|---|---|---|---|---|---|---|---|
| FP | -0.0001 | [-0.0035, 0.0031] | 0.0001 | 10 of 18 | - | - | reference |
| INT4 | -0.0009 | [-0.0038, 0.0019] | 0.0041 | 11 of 18 | - | - | undefined |
| INT6 | -0.0005 | [-0.0037, 0.0026] | -0.0005 | 8 of 18 | - | - | undefined |

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
| INT6 | eng_Latn | 0.00000 | [0.00000, 0.00000] |
| INT6 | deu_Latn | 0.00000 | [0.00000, 0.00000] |
| INT6 | spa_Latn | 0.00000 | [0.00000, 0.00000] |
| INT6 | fra_Latn | 0.00000 | [0.00000, 0.00000] |
| INT6 | por_Latn | 0.00000 | [0.00000, 0.00000] |
| INT6 | rus_Cyrl | 0.00000 | [0.00000, 0.00000] |
| INT6 | zho_Hans | 0.00000 | [0.00000, 0.00000] |
| INT6 | jpn_Jpan | 0.00000 | [0.00000, 0.00000] |
| INT6 | kor_Hang | 0.00000 | [0.00000, 0.00000] |
| INT6 | hin_Deva | 0.00000 | [0.00000, 0.00000] |
| INT6 | ben_Beng | 0.00000 | [0.00000, 0.00000] |
| INT6 | tur_Latn | 0.00000 | [0.00000, 0.00000] |
| INT6 | vie_Latn | 0.00000 | [0.00000, 0.00000] |
| INT6 | ind_Latn | 0.00000 | [0.00000, 0.00000] |
| INT6 | tha_Thai | 0.00000 | [0.00000, 0.00000] |
| INT6 | pol_Latn | 0.00000 | [0.00000, 0.00000] |
| INT6 | ukr_Cyrl | 0.00000 | [0.00000, 0.00000] |
| INT6 | kan_Knda | 0.00000 | [0.00000, 0.00000] |

## Effect by language

| precision | language | E | interval | group harm | control harm | items |
|---|---|---|---|---|---|---|
| FP | eng_Latn | 0.0001 | [-0.0111, 0.0098] | -0.0006 | -0.0007 | 385 |
| FP | deu_Latn | -0.0059 | [-0.0128, 0.0009] | 0.0022 | 0.0081 | 385 |
| FP | spa_Latn | 0.0002 | [-0.0062, 0.0071] | 0.0038 | 0.0036 | 385 |
| FP | fra_Latn | 0.0060 | [-0.0026, 0.0141] | 0.0040 | -0.0020 | 385 |
| FP | por_Latn | -0.0011 | [-0.0102, 0.0081] | -0.0039 | -0.0028 | 385 |
| FP | rus_Cyrl | -0.0040 | [-0.0116, 0.0034] | -0.0030 | 0.0010 | 385 |
| FP | zho_Hans | -0.0069 | [-0.0164, 0.0026] | -0.0020 | 0.0049 | 385 |
| FP | jpn_Jpan | 0.0040 | [-0.0065, 0.0149] | 0.0021 | -0.0019 | 385 |
| FP | kor_Hang | -0.0063 | [-0.0137, 0.0012] | -0.0051 | 0.0012 | 385 |
| FP | hin_Deva | 0.0026 | [-0.0104, 0.0167] | 0.0027 | 0.0001 | 385 |
| FP | ben_Beng | -0.0015 | [-0.0079, 0.0044] | -0.0006 | 0.0009 | 385 |
| FP | tur_Latn | 0.0010 | [-0.0076, 0.0099] | 0.0040 | 0.0030 | 385 |
| FP | vie_Latn | -0.0107 | [-0.0306, 0.0098] | -0.0132 | -0.0024 | 385 |
| FP | ind_Latn | 0.0061 | [-0.0032, 0.0147] | 0.0050 | -0.0010 | 385 |
| FP | tha_Thai | 0.0065 | [-0.0018, 0.0143] | 0.0050 | -0.0015 | 385 |
| FP | pol_Latn | 0.0055 | [-0.0015, 0.0119] | 0.0058 | 0.0002 | 385 |
| FP | ukr_Cyrl | -0.0023 | [-0.0102, 0.0052] | -0.0023 | -0.0000 | 385 |
| FP | kan_Knda | 0.0050 | [-0.0010, 0.0114] | 0.0084 | 0.0035 | 385 |
| INT4 | eng_Latn | -0.0067 | [-0.0188, 0.0057] | -0.0052 | 0.0015 | 385 |
| INT4 | deu_Latn | -0.0045 | [-0.0119, 0.0028] | -0.0017 | 0.0029 | 385 |
| INT4 | spa_Latn | -0.0356 | [-0.0447, -0.0269] | -0.0331 | 0.0025 | 385 |
| INT4 | fra_Latn | 0.0230 | [0.0106, 0.0358] | 0.0198 | -0.0033 | 385 |
| INT4 | por_Latn | -0.0449 | [-0.0586, -0.0320] | -0.0468 | -0.0019 | 385 |
| INT4 | rus_Cyrl | 0.0168 | [0.0083, 0.0251] | 0.0062 | -0.0106 | 385 |
| INT4 | zho_Hans | -0.0286 | [-0.0418, -0.0157] | -0.0361 | -0.0075 | 385 |
| INT4 | jpn_Jpan | 0.0099 | [-0.0012, 0.0211] | -0.0129 | -0.0228 | 385 |
| INT4 | kor_Hang | 0.0082 | [0.0010, 0.0166] | 0.0043 | -0.0039 | 385 |
| INT4 | hin_Deva | 0.0026 | [-0.0105, 0.0153] | -0.0014 | -0.0040 | 385 |
| INT4 | ben_Beng | 0.0166 | [0.0099, 0.0232] | 0.0061 | -0.0105 | 385 |
| INT4 | tur_Latn | -0.0127 | [-0.0202, -0.0052] | -0.0169 | -0.0041 | 385 |
| INT4 | vie_Latn | 0.0042 | [-0.0132, 0.0219] | -0.0056 | -0.0098 | 385 |
| INT4 | ind_Latn | 0.0069 | [0.0002, 0.0139] | -0.0009 | -0.0078 | 385 |
| INT4 | tha_Thai | 0.0202 | [0.0126, 0.0278] | 0.0139 | -0.0063 | 385 |
| INT4 | pol_Latn | -0.0029 | [-0.0098, 0.0046] | -0.0047 | -0.0018 | 385 |
| INT4 | ukr_Cyrl | 0.0068 | [-0.0007, 0.0144] | 0.0024 | -0.0044 | 385 |
| INT4 | kan_Knda | 0.0040 | [-0.0034, 0.0114] | 0.0012 | -0.0028 | 385 |
| INT6 | eng_Latn | -0.0049 | [-0.0152, 0.0043] | -0.0010 | 0.0039 | 385 |
| INT6 | deu_Latn | 0.0034 | [-0.0037, 0.0105] | 0.0050 | 0.0016 | 385 |
| INT6 | spa_Latn | 0.0041 | [-0.0030, 0.0110] | 0.0074 | 0.0033 | 385 |
| INT6 | fra_Latn | 0.0036 | [-0.0040, 0.0114] | -0.0019 | -0.0055 | 385 |
| INT6 | por_Latn | -0.0071 | [-0.0157, 0.0010] | -0.0079 | -0.0008 | 385 |
| INT6 | rus_Cyrl | -0.0056 | [-0.0130, 0.0022] | -0.0022 | 0.0033 | 385 |
| INT6 | zho_Hans | -0.0031 | [-0.0126, 0.0064] | 0.0016 | 0.0048 | 385 |
| INT6 | jpn_Jpan | -0.0048 | [-0.0154, 0.0065] | -0.0015 | 0.0033 | 385 |
| INT6 | kor_Hang | -0.0070 | [-0.0144, 0.0006] | -0.0020 | 0.0050 | 385 |
| INT6 | hin_Deva | -0.0009 | [-0.0134, 0.0117] | 0.0038 | 0.0047 | 385 |
| INT6 | ben_Beng | 0.0043 | [-0.0012, 0.0102] | 0.0034 | -0.0009 | 385 |
| INT6 | tur_Latn | 0.0059 | [-0.0025, 0.0145] | 0.0020 | -0.0038 | 385 |
| INT6 | vie_Latn | -0.0096 | [-0.0296, 0.0104] | -0.0096 | -0.0001 | 385 |
| INT6 | ind_Latn | -0.0000 | [-0.0080, 0.0073] | 0.0006 | 0.0006 | 385 |
| INT6 | tha_Thai | 0.0060 | [-0.0018, 0.0134] | 0.0067 | 0.0007 | 385 |
| INT6 | pol_Latn | 0.0085 | [0.0013, 0.0150] | 0.0087 | 0.0002 | 385 |
| INT6 | ukr_Cyrl | -0.0020 | [-0.0096, 0.0051] | -0.0024 | -0.0004 | 385 |
| INT6 | kan_Knda | 0.0010 | [-0.0048, 0.0072] | 0.0001 | -0.0009 | 385 |

## Accuracy (derived from the same scores; secondary)

| precision | language | items | none | zero-ablation | control mean | sham |
|---|---|---|---|---|---|---|
| FP | eng_Latn | 385 | 0.548 | 0.558 | 0.555 | 0.548 |
| FP | deu_Latn | 385 | 0.468 | 0.447 | 0.451 | 0.468 |
| FP | spa_Latn | 385 | 0.475 | 0.470 | 0.472 | 0.475 |
| FP | fra_Latn | 385 | 0.462 | 0.460 | 0.452 | 0.462 |
| FP | por_Latn | 385 | 0.416 | 0.423 | 0.424 | 0.416 |
| FP | rus_Cyrl | 385 | 0.460 | 0.460 | 0.461 | 0.460 |
| FP | zho_Hans | 385 | 0.452 | 0.475 | 0.442 | 0.452 |
| FP | jpn_Jpan | 385 | 0.397 | 0.395 | 0.417 | 0.397 |
| FP | kor_Hang | 385 | 0.436 | 0.455 | 0.458 | 0.436 |
| FP | hin_Deva | 385 | 0.371 | 0.369 | 0.371 | 0.371 |
| FP | ben_Beng | 385 | 0.379 | 0.387 | 0.375 | 0.379 |
| FP | tur_Latn | 385 | 0.403 | 0.392 | 0.398 | 0.403 |
| FP | vie_Latn | 385 | 0.387 | 0.392 | 0.386 | 0.387 |
| FP | ind_Latn | 385 | 0.439 | 0.442 | 0.438 | 0.439 |
| FP | tha_Thai | 385 | 0.387 | 0.369 | 0.376 | 0.387 |
| FP | pol_Latn | 385 | 0.423 | 0.421 | 0.423 | 0.423 |
| FP | ukr_Cyrl | 385 | 0.421 | 0.410 | 0.411 | 0.421 |
| FP | kan_Knda | 385 | 0.338 | 0.335 | 0.342 | 0.338 |
| INT4 | eng_Latn | 385 | 0.371 | 0.369 | 0.381 | 0.371 |
| INT4 | deu_Latn | 385 | 0.332 | 0.335 | 0.333 | 0.332 |
| INT4 | spa_Latn | 385 | 0.299 | 0.319 | 0.305 | 0.299 |
| INT4 | fra_Latn | 385 | 0.327 | 0.327 | 0.347 | 0.327 |
| INT4 | por_Latn | 385 | 0.270 | 0.294 | 0.275 | 0.270 |
| INT4 | rus_Cyrl | 385 | 0.319 | 0.301 | 0.320 | 0.319 |
| INT4 | zho_Hans | 385 | 0.304 | 0.332 | 0.313 | 0.304 |
| INT4 | jpn_Jpan | 385 | 0.299 | 0.304 | 0.309 | 0.299 |
| INT4 | kor_Hang | 385 | 0.294 | 0.283 | 0.283 | 0.294 |
| INT4 | hin_Deva | 385 | 0.262 | 0.257 | 0.265 | 0.262 |
| INT4 | ben_Beng | 385 | 0.236 | 0.236 | 0.236 | 0.236 |
| INT4 | tur_Latn | 385 | 0.291 | 0.299 | 0.296 | 0.291 |
| INT4 | vie_Latn | 385 | 0.278 | 0.288 | 0.284 | 0.278 |
| INT4 | ind_Latn | 385 | 0.314 | 0.314 | 0.319 | 0.314 |
| INT4 | tha_Thai | 385 | 0.249 | 0.252 | 0.252 | 0.249 |
| INT4 | pol_Latn | 385 | 0.296 | 0.309 | 0.298 | 0.296 |
| INT4 | ukr_Cyrl | 385 | 0.270 | 0.262 | 0.278 | 0.270 |
| INT4 | kan_Knda | 385 | 0.229 | 0.226 | 0.225 | 0.229 |
| INT6 | eng_Latn | 385 | 0.519 | 0.530 | 0.522 | 0.519 |
| INT6 | deu_Latn | 385 | 0.439 | 0.423 | 0.436 | 0.439 |
| INT6 | spa_Latn | 385 | 0.452 | 0.468 | 0.465 | 0.452 |
| INT6 | fra_Latn | 385 | 0.442 | 0.457 | 0.453 | 0.442 |
| INT6 | por_Latn | 385 | 0.429 | 0.426 | 0.431 | 0.429 |
| INT6 | rus_Cyrl | 385 | 0.449 | 0.439 | 0.455 | 0.449 |
| INT6 | zho_Hans | 385 | 0.439 | 0.462 | 0.439 | 0.439 |
| INT6 | jpn_Jpan | 385 | 0.434 | 0.423 | 0.423 | 0.434 |
| INT6 | kor_Hang | 385 | 0.434 | 0.439 | 0.442 | 0.434 |
| INT6 | hin_Deva | 385 | 0.377 | 0.374 | 0.376 | 0.377 |
| INT6 | ben_Beng | 385 | 0.387 | 0.371 | 0.379 | 0.387 |
| INT6 | tur_Latn | 385 | 0.400 | 0.395 | 0.398 | 0.400 |
| INT6 | vie_Latn | 385 | 0.392 | 0.387 | 0.391 | 0.392 |
| INT6 | ind_Latn | 385 | 0.431 | 0.447 | 0.436 | 0.431 |
| INT6 | tha_Thai | 385 | 0.366 | 0.366 | 0.370 | 0.366 |
| INT6 | pol_Latn | 385 | 0.439 | 0.436 | 0.443 | 0.439 |
| INT6 | ukr_Cyrl | 385 | 0.449 | 0.449 | 0.441 | 0.449 |
| INT6 | kan_Knda | 385 | 0.345 | 0.345 | 0.360 | 0.345 |
