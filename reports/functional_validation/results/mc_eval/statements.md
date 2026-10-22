# T-MC confirmatory statements, evaluated as frozen

Freeze: `evaluation_freeze_mc.json` (code commit ff9d230efdbaf46dd85e02034b42db706124b6e1). **These statements are the ones in the freeze file; their wording, sidedness, levels and decision rules are not changed here. Everything else (per-language effects, F3, F4, subgroups, doses, the K = all run, other formats) is exploratory and labelled.** The analysis passed the provenance check against the freeze (task, format, K, draws, seed, bands, covered languages, retention level, item counts per language and precision).

## Statements

| id | precision | claim | level | outcome | detail |
|---|---|---|---|---|---|
| S0 | FP | the macro effect E over the covered languages is greater than 0 (one-sided) | 95 % | not met | macro E -0.0001 nats per item, 95 % interval [-0.0035, +0.0031], languages with E > 0: 10 of 18 |
| S1 | INT8 | the macro retention of E (mean of per-language ratios E_l(b) / E_l(FP), D-041) is classified preserved, attenuated, amplified or inconclusive against the bands (two-sided) | 99 % | undefined | no covered language has a clearly positive full-precision effect (rule D-041) |
| S2 | INT7 | the macro retention of E (mean of per-language ratios E_l(b) / E_l(FP), D-041) is classified preserved, attenuated, amplified or inconclusive against the bands (two-sided) | 99 % | undefined | no covered language has a clearly positive full-precision effect (rule D-041) |
| S3 | INT6 | the macro retention of E (mean of per-language ratios E_l(b) / E_l(FP), D-041) is classified preserved, attenuated, amplified or inconclusive against the bands (two-sided) | 99 % | undefined | no covered language has a clearly positive full-precision effect (rule D-041) |
| S4 | INT5 | the macro retention of E (mean of per-language ratios E_l(b) / E_l(FP), D-041) is classified preserved, attenuated, amplified or inconclusive against the bands (two-sided) | 99 % | undefined | no covered language has a clearly positive full-precision effect (rule D-041) |
| S5 | INT4 | the macro retention of E (mean of per-language ratios E_l(b) / E_l(FP), D-041) is classified preserved, attenuated, amplified or inconclusive against the bands (two-sided) | 99 % | undefined | no covered language has a clearly positive full-precision effect (rule D-041) |

`undefined`: no covered language has a clearly positive full-precision effect, so the macro retention does not exist (rule D-041); `no data`: the precision is not in the analysis (its job has not been analysed); `not met` and every class are reported as they are.

## Context: macro effect E by precision (mean over the covered languages, nats per item)

| precision | macro E | 95 % interval | languages with E > 0 | covered |
|---|---|---|---|---|
| FP | -0.0001 | [-0.0035, +0.0031] | 10 | 18 |
| INT8 | +0.0001 | [-0.0032, +0.0033] | 8 | 18 |
| INT7 | -0.0007 | [-0.0038, +0.0024] | 9 | 18 |
| INT6 | -0.0012 | [-0.0045, +0.0019] | 8 | 18 |
| INT5 | -0.0066 | [-0.0102, -0.0031] | 6 | 18 |
| INT4 | +0.0042 | [+0.0003, +0.0081] | 10 | 18 |
