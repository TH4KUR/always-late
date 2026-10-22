# F4: lost against retained features at equal native activity, INT6

**Exploratory.** Contrast = harm of zero-ablating the features whose language association was LOST at the precision minus the harm of zero-ablating activity-matched features that kept it (lost minus retained, nats per token on the language's own text, same sentences, same precision). **Negative = the lost features matter less.** 2000 paired cluster-bootstrap draws over sentence ids (the same draws for both precisions; percentile 95 % intervals, descriptive; the observed value is the estimate). The matching and its balance are in `f4_matching.md` (the subsets come from the F4 manifest of the precision).

H-F4 (written into the plan before any functional number, two-sided, exploratory because the D-035 statuses were measured on devtest): E_b differs from 0 at precision b; "association predicts function" predicts E_b < 0 (the lost features matter less than activity-matched retained ones); a null result or a positive E_b is reported as it is. E_FP, the same contrast for the same subsets in the full-precision model, shows whether a difference pre-exists the quantization.

## Macro mean over 18 languages

| quantity | mean | 95 % interval | median | languages negative / positive / inconclusive |
|---|---|---|---|---|
| E at INT6 (native) | -0.0006 | [-0.0009, -0.0003] | +0.0002 | 2 / 3 / 13 |
| E at full precision (same subsets) | -0.0008 | [-0.0011, -0.0005] | +0.0003 | 2 / 4 / 12 |
| Delta = E at INT6 minus E at full precision | +0.0002 | [-0.0001, +0.0004] | -0.0001 | - |

**Robust summaries, added after the first F4 result (INT4) because one language dominated the macro mean (post hoc; applied unchanged to every condition).** Median over the languages and the mean without the language with the largest absolute value, from the same paired draws:

- E at INT6 (native): median +0.0002 [-0.0002, +0.0005]; mean without lao_Laoo +0.0005 [+0.0002, +0.0008]
- E at full precision (same subsets): median +0.0003 [-0.0001, +0.0006]; mean without lao_Laoo +0.0006 [+0.0004, +0.0009]
- Delta = E at INT6 minus E at full precision: median -0.0001 [-0.0003, +0.0003]; mean without lao_Laoo -0.0001 [-0.0004, +0.0001]

The intervals describe sentence sampling only: the retained subsets are single or double draws (the number per language is in `f4_matching.md`), so the draw-to-draw variation of a retained subset, which can be large when it contains one influential feature, is not in them.

## By language

| language | E at INT6 | 95 % interval | E at FP | 95 % interval | Delta | 95 % interval | harm lost at INT6 | harm retained at INT6 |
|---|---|---|---|---|---|---|---|---|
| eng_Latn | +0.0004 | [-0.0011, +0.0020] | +0.0007 | [-0.0009, +0.0025] | -0.0003 | [-0.0017, +0.0010] | 0.0001 | -0.0003 |
| deu_Latn | -0.0003 | [-0.0010, +0.0005] | +0.0006 | [-0.0002, +0.0013] | -0.0008 | [-0.0018, +0.0001] | -0.0001 | 0.0002 |
| spa_Latn | +0.0005 | [-0.0007, +0.0016] | +0.0016 | [+0.0006, +0.0027] | -0.0012 | [-0.0023, +0.0001] | 0.0006 | 0.0002 |
| zho_Hans | +0.0007 | [-0.0005, +0.0021] | +0.0011 | [-0.0003, +0.0025] | -0.0004 | [-0.0015, +0.0008] | 0.0001 | -0.0007 |
| jpn_Jpan | +0.0002 | [-0.0012, +0.0015] | +0.0004 | [-0.0010, +0.0017] | -0.0002 | [-0.0013, +0.0010] | -0.0007 | -0.0009 |
| tam_Taml | +0.0004 | [-0.0004, +0.0011] | +0.0004 | [-0.0004, +0.0012] | +0.0000 | [-0.0008, +0.0008] | 0.0009 | 0.0005 |
| tur_Latn | -0.0004 | [-0.0012, +0.0004] | -0.0002 | [-0.0010, +0.0006] | -0.0002 | [-0.0013, +0.0008] | -0.0002 | 0.0002 |
| vie_Latn | -0.0003 | [-0.0010, +0.0004] | -0.0001 | [-0.0008, +0.0005] | -0.0001 | [-0.0010, +0.0007] | -0.0006 | -0.0004 |
| tha_Thai | +0.0001 | [-0.0007, +0.0008] | -0.0004 | [-0.0011, +0.0004] | +0.0005 | [-0.0003, +0.0012] | 0.0005 | 0.0004 |
| ukr_Cyrl | -0.0012 | [-0.0025, +0.0002] | -0.0015 | [-0.0028, -0.0001] | +0.0003 | [-0.0007, +0.0013] | -0.0013 | -0.0001 |
| kan_Knda | +0.0054 | [+0.0037, +0.0070] | +0.0046 | [+0.0028, +0.0062] | +0.0008 | [+0.0000, +0.0016] | 0.0080 | 0.0026 |
| yor_Latn | -0.0011 | [-0.0022, -0.0000] | -0.0008 | [-0.0018, +0.0003] | -0.0003 | [-0.0014, +0.0008] | 0.0002 | 0.0013 |
| sun_Latn | +0.0002 | [-0.0007, +0.0011] | +0.0001 | [-0.0008, +0.0010] | +0.0001 | [-0.0010, +0.0012] | 0.0002 | -0.0001 |
| amh_Ethi | +0.0025 | [+0.0017, +0.0034] | +0.0024 | [+0.0016, +0.0033] | +0.0001 | [-0.0007, +0.0009] | 0.0036 | 0.0010 |
| ibo_Latn | -0.0002 | [-0.0010, +0.0006] | -0.0001 | [-0.0009, +0.0007] | -0.0000 | [-0.0010, +0.0010] | 0.0003 | 0.0005 |
| hau_Latn | +0.0004 | [-0.0008, +0.0017] | +0.0018 | [+0.0004, +0.0030] | -0.0013 | [-0.0027, +0.0002] | 0.0009 | 0.0005 |
| khm_Khmr | +0.0012 | [+0.0005, +0.0019] | +0.0003 | [-0.0004, +0.0011] | +0.0009 | [+0.0002, +0.0016] | 0.0017 | 0.0005 |
| lao_Laoo | -0.0199 | [-0.0233, -0.0165] | -0.0254 | [-0.0291, -0.0217] | +0.0055 | [+0.0041, +0.0068] | 0.0206 | 0.0405 |
