# F4: lost against retained features at equal native activity, INT5

**Exploratory.** Contrast = harm of zero-ablating the features whose language association was LOST at the precision minus the harm of zero-ablating activity-matched features that kept it (lost minus retained, nats per token on the language's own text, same sentences, same precision). **Negative = the lost features matter less.** 2000 paired cluster-bootstrap draws over sentence ids (the same draws for both precisions; percentile 95 % intervals, descriptive; the observed value is the estimate). The matching and its balance are in `f4_matching.md` (the subsets come from the F4 manifest of the precision).

H-F4 (written into the plan before any functional number, two-sided, exploratory because the D-035 statuses were measured on devtest): E_b differs from 0 at precision b; "association predicts function" predicts E_b < 0 (the lost features matter less than activity-matched retained ones); a null result or a positive E_b is reported as it is. E_FP, the same contrast for the same subsets in the full-precision model, shows whether a difference pre-exists the quantization.

## Macro mean over 24 languages

| quantity | mean | 95 % interval | median | languages negative / positive / inconclusive |
|---|---|---|---|---|
| E at INT5 (native) | -0.0002 | [-0.0008, +0.0004] | +0.0007 | 3 / 9 / 12 |
| E at full precision (same subsets) | +0.0013 | [+0.0008, +0.0018] | +0.0005 | 4 / 6 / 14 |
| Delta = E at INT5 minus E at full precision | -0.0015 | [-0.0019, -0.0011] | +0.0002 | - |

**Robust summaries, added after the first F4 result (INT4) because one language dominated the macro mean (post hoc; applied unchanged to every condition).** Median over the languages and the mean without the language with the largest absolute value, from the same paired draws:

- E at INT5 (native): median +0.0007 [+0.0001, +0.0011]; mean without lao_Laoo +0.0037 [+0.0032, +0.0042]
- E at full precision (same subsets): median +0.0005 [-0.0002, +0.0008]; mean without hin_Deva -0.0004 [-0.0007, +0.0000]
- Delta = E at INT5 minus E at full precision: median +0.0002 [-0.0002, +0.0008]; mean without lao_Laoo +0.0016 [+0.0013, +0.0019]

The intervals describe sentence sampling only: the retained subsets are single or double draws (the number per language is in `f4_matching.md`), so the draw-to-draw variation of a retained subset, which can be large when it contains one influential feature, is not in them.

## By language

| language | E at INT5 | 95 % interval | E at FP | 95 % interval | Delta | 95 % interval | harm lost at INT5 | harm retained at INT5 |
|---|---|---|---|---|---|---|---|---|
| eng_Latn | -0.0010 | [-0.0031, +0.0009] | -0.0005 | [-0.0026, +0.0014] | -0.0005 | [-0.0022, +0.0012] | 0.0004 | 0.0015 |
| deu_Latn | +0.0022 | [+0.0003, +0.0045] | +0.0008 | [-0.0008, +0.0028] | +0.0014 | [+0.0002, +0.0026] | 0.0011 | -0.0011 |
| spa_Latn | +0.0008 | [-0.0005, +0.0021] | +0.0010 | [-0.0003, +0.0023] | -0.0002 | [-0.0014, +0.0011] | 0.0012 | 0.0004 |
| fra_Latn | +0.0035 | [+0.0019, +0.0052] | +0.0025 | [+0.0009, +0.0042] | +0.0010 | [-0.0003, +0.0022] | 0.0032 | -0.0003 |
| rus_Cyrl | -0.0013 | [-0.0022, -0.0003] | -0.0007 | [-0.0017, +0.0003] | -0.0006 | [-0.0016, +0.0005] | -0.0004 | 0.0009 |
| zho_Hans | +0.0012 | [-0.0007, +0.0031] | +0.0023 | [+0.0005, +0.0042] | -0.0012 | [-0.0026, +0.0004] | 0.0005 | -0.0006 |
| jpn_Jpan | -0.0013 | [-0.0030, +0.0002] | -0.0007 | [-0.0023, +0.0009] | -0.0007 | [-0.0021, +0.0008] | -0.0005 | 0.0008 |
| kor_Hang | -0.0014 | [-0.0027, -0.0000] | -0.0019 | [-0.0031, -0.0006] | +0.0005 | [-0.0007, +0.0016] | 0.0002 | 0.0016 |
| hin_Deva | +0.0463 | [+0.0387, +0.0547] | +0.0402 | [+0.0334, +0.0474] | +0.0061 | [+0.0036, +0.0089] | 0.0396 | -0.0068 |
| ben_Beng | +0.0044 | [+0.0027, +0.0062] | +0.0029 | [+0.0014, +0.0046] | +0.0015 | [+0.0004, +0.0026] | 0.0056 | 0.0012 |
| tam_Taml | -0.0011 | [-0.0026, +0.0003] | -0.0016 | [-0.0029, -0.0002] | +0.0005 | [-0.0007, +0.0016] | 0.0017 | 0.0028 |
| tur_Latn | +0.0029 | [+0.0010, +0.0051] | +0.0015 | [-0.0004, +0.0034] | +0.0015 | [+0.0002, +0.0029] | 0.0024 | -0.0005 |
| vie_Latn | +0.0001 | [-0.0013, +0.0015] | +0.0008 | [-0.0006, +0.0022] | -0.0006 | [-0.0019, +0.0006] | -0.0005 | -0.0006 |
| tha_Thai | -0.0006 | [-0.0019, +0.0006] | -0.0005 | [-0.0018, +0.0008] | -0.0001 | [-0.0013, +0.0011] | 0.0009 | 0.0015 |
| pol_Latn | +0.0001 | [-0.0007, +0.0009] | +0.0010 | [+0.0003, +0.0017] | -0.0009 | [-0.0018, +0.0001] | 0.0004 | 0.0003 |
| kan_Knda | +0.0010 | [-0.0018, +0.0038] | -0.0008 | [-0.0034, +0.0016] | +0.0018 | [+0.0003, +0.0033] | 0.0103 | 0.0094 |
| tel_Telu | -0.0006 | [-0.0017, +0.0007] | -0.0003 | [-0.0013, +0.0008] | -0.0003 | [-0.0013, +0.0006] | 0.0003 | 0.0009 |
| yor_Latn | +0.0018 | [-0.0004, +0.0040] | -0.0006 | [-0.0027, +0.0013] | +0.0024 | [+0.0011, +0.0038] | 0.0017 | -0.0001 |
| sun_Latn | +0.0094 | [+0.0059, +0.0130] | +0.0029 | [-0.0005, +0.0063] | +0.0065 | [+0.0040, +0.0093] | 0.0232 | 0.0138 |
| amh_Ethi | +0.0006 | [+0.0001, +0.0012] | +0.0008 | [+0.0002, +0.0014] | -0.0001 | [-0.0009, +0.0006] | -0.0003 | -0.0010 |
| ibo_Latn | -0.0011 | [-0.0025, +0.0001] | +0.0002 | [-0.0009, +0.0013] | -0.0013 | [-0.0027, -0.0000] | 0.0007 | 0.0019 |
| hau_Latn | +0.0021 | [+0.0008, +0.0034] | +0.0007 | [-0.0006, +0.0021] | +0.0013 | [-0.0000, +0.0027] | 0.0010 | -0.0011 |
| khm_Khmr | +0.0166 | [+0.0136, +0.0197] | -0.0026 | [-0.0043, -0.0008] | +0.0192 | [+0.0169, +0.0216] | 0.0206 | 0.0040 |
| lao_Laoo | -0.0894 | [-0.0971, -0.0817] | -0.0157 | [-0.0198, -0.0114] | -0.0737 | [-0.0790, -0.0684] | 0.0008 | 0.0902 |
