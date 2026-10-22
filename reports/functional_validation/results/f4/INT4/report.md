# F4: lost against retained features at equal native activity, INT4

**Exploratory.** Contrast = harm of zero-ablating the features whose language association was LOST at the precision minus the harm of zero-ablating activity-matched features that kept it (lost minus retained, nats per token on the language's own text, same sentences, same precision). **Negative = the lost features matter less.** 2000 paired cluster-bootstrap draws over sentence ids (the same draws for both precisions; percentile 95 % intervals, descriptive; the observed value is the estimate). The matching and its balance are in `f4_matching.md` (the subsets come from the F4 manifest of the precision).

H-F4 (written into the plan before any functional number, two-sided, exploratory because the D-035 statuses were measured on devtest): E_b differs from 0 at precision b; "association predicts function" predicts E_b < 0 (the lost features matter less than activity-matched retained ones); a null result or a positive E_b is reported as it is. E_FP, the same contrast for the same subsets in the full-precision model, shows whether a difference pre-exists the quantization.

## Macro mean over 26 languages

| quantity | mean | 95 % interval | median | languages negative / positive / inconclusive |
|---|---|---|---|---|
| E at INT4 (native) | -0.0493 | [-0.0513, -0.0472] | +0.0006 | 6 / 7 / 13 |
| E at full precision (same subsets) | -0.0227 | [-0.0244, -0.0211] | +0.0005 | 3 / 6 / 17 |
| Delta = E at INT4 minus E at full precision | -0.0265 | [-0.0286, -0.0244] | -0.0005 | - |

**Robust summaries, added after the first F4 result (INT4) because one language dominated the macro mean (post hoc; applied unchanged to every condition).** Median over the languages and the mean without the language with the largest absolute value, from the same paired draws:

- E at INT4 (native): median +0.0006 [-0.0001, +0.0012]; mean without lao_Laoo +0.0032 [+0.0026, +0.0038]
- E at full precision (same subsets): median +0.0005 [-0.0001, +0.0012]; mean without lao_Laoo +0.0014 [+0.0010, +0.0019]
- Delta = E at INT4 minus E at full precision: median -0.0005 [-0.0012, +0.0001]; mean without lao_Laoo +0.0017 [+0.0012, +0.0023]

The intervals describe sentence sampling only: the retained subsets are single or double draws (the number per language is in `f4_matching.md`), so the draw-to-draw variation of a retained subset, which can be large when it contains one influential feature, is not in them.

## By language

| language | E at INT4 | 95 % interval | E at FP | 95 % interval | Delta | 95 % interval | harm lost at INT4 | harm retained at INT4 |
|---|---|---|---|---|---|---|---|---|
| eng_Latn | -0.0014 | [-0.0028, -0.0001] | +0.0001 | [-0.0013, +0.0016] | -0.0015 | [-0.0030, +0.0000] | 0.0001 | 0.0015 |
| deu_Latn | +0.0007 | [-0.0014, +0.0027] | -0.0001 | [-0.0020, +0.0018] | +0.0008 | [-0.0013, +0.0028] | 0.0013 | 0.0006 |
| spa_Latn | +0.0000 | [-0.0017, +0.0017] | +0.0014 | [-0.0004, +0.0031] | -0.0014 | [-0.0031, +0.0004] | 0.0012 | 0.0011 |
| fra_Latn | +0.0107 | [+0.0069, +0.0148] | +0.0065 | [+0.0037, +0.0096] | +0.0042 | [+0.0016, +0.0067] | 0.0124 | 0.0017 |
| rus_Cyrl | +0.0006 | [-0.0006, +0.0017] | +0.0000 | [-0.0011, +0.0011] | +0.0006 | [-0.0009, +0.0021] | 0.0010 | 0.0005 |
| zho_Hans | -0.0007 | [-0.0032, +0.0018] | +0.0021 | [-0.0002, +0.0043] | -0.0028 | [-0.0051, -0.0006] | 0.0028 | 0.0035 |
| jpn_Jpan | +0.0006 | [-0.0013, +0.0026] | +0.0018 | [-0.0004, +0.0041] | -0.0012 | [-0.0031, +0.0008] | 0.0009 | 0.0003 |
| kor_Hang | -0.0021 | [-0.0041, -0.0001] | -0.0014 | [-0.0035, +0.0006] | -0.0007 | [-0.0026, +0.0014] | 0.0020 | 0.0041 |
| hin_Deva | +0.0016 | [+0.0001, +0.0031] | +0.0021 | [+0.0008, +0.0034] | -0.0005 | [-0.0020, +0.0010] | 0.0024 | 0.0008 |
| ben_Beng | -0.0305 | [-0.0347, -0.0265] | -0.0013 | [-0.0035, +0.0011] | -0.0292 | [-0.0331, -0.0255] | 0.0074 | 0.0379 |
| tam_Taml | -0.0085 | [-0.0116, -0.0057] | +0.0015 | [-0.0008, +0.0039] | -0.0100 | [-0.0126, -0.0074] | 0.0133 | 0.0218 |
| tur_Latn | +0.0024 | [-0.0006, +0.0055] | +0.0016 | [-0.0012, +0.0044] | +0.0008 | [-0.0018, +0.0034] | 0.0046 | 0.0021 |
| vie_Latn | +0.0006 | [-0.0024, +0.0034] | +0.0004 | [-0.0027, +0.0034] | +0.0001 | [-0.0027, +0.0029] | 0.0060 | 0.0054 |
| ind_Latn | -0.0006 | [-0.0034, +0.0019] | -0.0013 | [-0.0039, +0.0010] | +0.0007 | [-0.0017, +0.0033] | -0.0004 | 0.0002 |
| tha_Thai | -0.0008 | [-0.0030, +0.0016] | +0.0006 | [-0.0014, +0.0026] | -0.0014 | [-0.0033, +0.0006] | 0.0047 | 0.0054 |
| pol_Latn | +0.0004 | [-0.0010, +0.0018] | +0.0006 | [-0.0008, +0.0020] | -0.0002 | [-0.0018, +0.0014] | 0.0010 | 0.0006 |
| ukr_Cyrl | +0.0008 | [-0.0009, +0.0027] | -0.0017 | [-0.0028, -0.0005] | +0.0025 | [+0.0007, +0.0043] | 0.0021 | 0.0013 |
| kan_Knda | +0.0036 | [+0.0010, +0.0062] | +0.0042 | [+0.0017, +0.0067] | -0.0006 | [-0.0027, +0.0013] | 0.0122 | 0.0086 |
| tel_Telu | +0.0023 | [+0.0005, +0.0040] | +0.0020 | [+0.0007, +0.0032] | +0.0003 | [-0.0014, +0.0021] | 0.0054 | 0.0031 |
| yor_Latn | +0.0054 | [+0.0015, +0.0093] | -0.0028 | [-0.0065, +0.0008] | +0.0082 | [+0.0047, +0.0117] | 0.0141 | 0.0087 |
| sun_Latn | +0.0190 | [+0.0128, +0.0249] | +0.0227 | [+0.0181, +0.0271] | -0.0037 | [-0.0086, +0.0011] | 0.0367 | 0.0177 |
| amh_Ethi | +0.0868 | [+0.0793, +0.0946] | -0.0006 | [-0.0023, +0.0011] | +0.0874 | [+0.0801, +0.0951] | 0.0934 | 0.0066 |
| ibo_Latn | +0.0008 | [-0.0012, +0.0028] | -0.0002 | [-0.0022, +0.0018] | +0.0010 | [-0.0014, +0.0034] | 0.0044 | 0.0035 |
| hau_Latn | -0.0145 | [-0.0182, -0.0108] | -0.0071 | [-0.0106, -0.0034] | -0.0074 | [-0.0113, -0.0036] | 0.0051 | 0.0196 |
| khm_Khmr | +0.0019 | [-0.0019, +0.0055] | +0.0048 | [+0.0026, +0.0069] | -0.0029 | [-0.0067, +0.0010] | 0.0177 | 0.0158 |
| lao_Laoo | -1.3600 | [-1.4123, -1.3076] | -0.6272 | [-0.6676, -0.5874] | -0.7329 | [-0.7856, -0.6815] | -0.0868 | 1.2732 |
