# Functional effects of language feature groups (group size K64)

**Exploratory on already examined data** (FLORES devtest sentences; the features were selected on dev). Next-token loss only; no claim about task accuracy or meaning. E is the harm (nats per token, NLL after the edit minus NLL before it, same precision, same sentences) of zero-ablating the language's selective group beyond the mean harm of its activity-matched controls on the language's own text; positive = the selective group matters more than equally active non-selective features. Intervals: 2000 paired cluster-bootstrap draws over sentence ids (percentile, descriptive; the observed value is the estimate). Macro retention is the mean over languages of the per-language ratio E at a precision over E in full precision (D-041; languages whose full-precision interval lies above zero), classified preserved / attenuated / amplified against the bands [0.75, 1.33], inconclusive otherwise, undefined when the full-precision interval does not lie above zero.

Languages in the macro average: 27 (eng_Latn, deu_Latn, spa_Latn, fra_Latn, por_Latn, rus_Cyrl, zho_Hans, jpn_Jpan, kor_Hang, hin_Deva, ben_Beng, tam_Taml, tur_Latn, vie_Latn, ind_Latn, tha_Thai, pol_Latn, ukr_Cyrl, kan_Knda, tel_Telu, yor_Latn, sun_Latn, amh_Ethi, ibo_Latn, hau_Latn, khm_Khmr, lao_Laoo). Positions (sentences per language): 150.

## Macro effect and retention

| precision | mean E | 95 % interval | median E | languages with E > 0 | retention (mean of per-language ratios) | interval | class (languages) | pooled ratio (dominated by large effects) |
|---|---|---|---|---|---|---|---|---|
| FP | 0.6927 | [0.6673, 0.7191] | 0.2656 | 27 of 27 | - | - | reference | - |

## Dose, size and replacement (median over languages of the harm in nats per token on the language's own text)

| family | FP |
|---|---|
| E_K16 | 0.2157 |
| E_K64 | 0.2656 |
| E_all | 0.3001 |
| ctrl_K16 | 0.0159 |
| ctrl_K64 | 0.0218 |
| ctrl_all | 0.0265 |
| mean_K64 | 0.2369 |
| res_K64 | 0.3001 |
| scale0.5_K64 | 0.0092 |
| scale2_K64 | 0.0303 |
| sham_K64 | 0.0000 |
| zero_K16 | 0.2494 |
| zero_K64 | 0.2973 |
| zero_all | 0.3470 |

Languages whose E (zero ablation beyond controls) has a 95 % lower bound above 0: FP: E_K16 25, E_K64 26, E_all 26


## Sham check (the edit path with no intended change; must be about 0 nats per token)

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
| FP | tam_Taml | 0.00000 | [0.00000, 0.00000] |
| FP | tur_Latn | 0.00000 | [0.00000, 0.00000] |
| FP | vie_Latn | 0.00000 | [0.00000, 0.00000] |
| FP | ind_Latn | 0.00000 | [0.00000, 0.00000] |
| FP | tha_Thai | 0.00000 | [0.00000, 0.00000] |
| FP | pol_Latn | 0.00000 | [0.00000, 0.00000] |
| FP | ukr_Cyrl | 0.00000 | [0.00000, 0.00000] |
| FP | kan_Knda | 0.00000 | [0.00000, 0.00000] |
| FP | tel_Telu | 0.00000 | [0.00000, 0.00000] |
| FP | yor_Latn | 0.00000 | [0.00000, 0.00000] |
| FP | sun_Latn | 0.00000 | [0.00000, 0.00000] |
| FP | amh_Ethi | 0.00000 | [0.00000, 0.00000] |
| FP | ibo_Latn | 0.00000 | [0.00000, 0.00000] |
| FP | hau_Latn | 0.00000 | [0.00000, 0.00000] |
| FP | khm_Khmr | 0.00000 | [0.00000, 0.00000] |
| FP | lao_Laoo | 0.00000 | [0.00000, 0.00000] |

## Effect by language

| precision | language | E | interval | group harm | control harm | removed norm group / control |
|---|---|---|---|---|---|---|
| FP | eng_Latn | 0.1920 | [0.1577, 0.2265] | 0.2377 | 0.0458 | 0.144 / 0.140 |
| FP | deu_Latn | 0.9696 | [0.8751, 1.0627] | 0.9802 | 0.0106 | 0.122 / 0.110 |
| FP | spa_Latn | 0.4938 | [0.3852, 0.6312] | 0.5156 | 0.0218 | 0.107 / 0.098 |
| FP | fra_Latn | 0.6250 | [0.5506, 0.7088] | 0.6445 | 0.0194 | 0.120 / 0.099 |
| FP | por_Latn | 0.9737 | [0.8409, 1.1086] | 0.9758 | 0.0021 | 0.075 / 0.067 |
| FP | rus_Cyrl | 0.0675 | [0.0524, 0.0838] | 0.0713 | 0.0038 | 0.079 / 0.081 |
| FP | zho_Hans | 7.3441 | [6.8644, 7.8182] | 7.3705 | 0.0264 | 0.178 / 0.162 |
| FP | jpn_Jpan | 0.2656 | [0.2318, 0.3011] | 0.2973 | 0.0317 | 0.141 / 0.140 |
| FP | kor_Hang | 0.3942 | [0.3237, 0.4619] | 0.4090 | 0.0147 | 0.097 / 0.095 |
| FP | hin_Deva | 0.1116 | [0.0875, 0.1375] | 0.1246 | 0.0130 | 0.128 / 0.129 |
| FP | ben_Beng | 0.0071 | [0.0025, 0.0117] | 0.0167 | 0.0096 | 0.055 / 0.050 |
| FP | tam_Taml | 0.0301 | [0.0236, 0.0367] | 0.0434 | 0.0133 | 0.077 / 0.072 |
| FP | tur_Latn | 0.2193 | [0.1886, 0.2541] | 0.2616 | 0.0423 | 0.129 / 0.127 |
| FP | vie_Latn | 0.0054 | [-0.0064, 0.0179] | 0.0175 | 0.0121 | 0.084 / 0.085 |
| FP | ind_Latn | 0.0187 | [0.0053, 0.0340] | 0.0287 | 0.0100 | 0.051 / 0.052 |
| FP | tha_Thai | 0.0388 | [0.0288, 0.0489] | 0.0537 | 0.0149 | 0.110 / 0.150 |
| FP | pol_Latn | 0.3167 | [0.2704, 0.3726] | 0.3446 | 0.0279 | 0.100 / 0.094 |
| FP | ukr_Cyrl | 0.4033 | [0.3568, 0.4526] | 0.4181 | 0.0149 | 0.108 / 0.091 |
| FP | kan_Knda | 0.0354 | [0.0280, 0.0427] | 0.0514 | 0.0160 | 0.090 / 0.086 |
| FP | tel_Telu | 0.0251 | [0.0170, 0.0343] | 0.0559 | 0.0308 | 0.119 / 0.131 |
| FP | yor_Latn | 1.5172 | [1.4544, 1.5794] | 1.5820 | 0.0648 | 0.343 / 0.277 |
| FP | sun_Latn | 0.9378 | [0.8665, 1.0052] | 1.0412 | 0.1033 | 0.169 / 0.172 |
| FP | amh_Ethi | 1.0284 | [0.9912, 1.0677] | 1.0607 | 0.0323 | 0.237 / 0.181 |
| FP | ibo_Latn | 0.3621 | [0.3321, 0.3921] | 0.4082 | 0.0461 | 0.184 / 0.168 |
| FP | hau_Latn | 0.1779 | [0.1548, 0.2012] | 0.2304 | 0.0525 | 0.139 / 0.120 |
| FP | khm_Khmr | 0.2207 | [0.1947, 0.2466] | 0.2546 | 0.0339 | 0.174 / 0.167 |
| FP | lao_Laoo | 1.9218 | [1.8178, 2.0269] | 1.9503 | 0.0286 | 0.254 / 0.194 |

## Off-target harm of the group's zero-ablation on other languages' text (manipulation check)

| precision | group | text | harm | interval |
|---|---|---|---|---|
| FP | eng_Latn | rus_Cyrl | 0.00481 | [0.00082, 0.00870] |
| FP | eng_Latn | zho_Hans | 0.00609 | [0.00057, 0.01170] |
| FP | eng_Latn | hin_Deva | 0.00093 | [-0.00142, 0.00334] |
| FP | deu_Latn | eng_Latn | 0.01156 | [0.00321, 0.02092] |
| FP | deu_Latn | rus_Cyrl | 0.01881 | [0.01327, 0.02537] |
| FP | deu_Latn | zho_Hans | 0.01049 | [0.00680, 0.01409] |
| FP | deu_Latn | hin_Deva | 0.00367 | [0.00165, 0.00586] |
| FP | spa_Latn | eng_Latn | 0.04696 | [0.03486, 0.05990] |
| FP | spa_Latn | rus_Cyrl | 0.00499 | [0.00137, 0.00893] |
| FP | spa_Latn | zho_Hans | 0.00373 | [0.00127, 0.00648] |
| FP | spa_Latn | hin_Deva | 0.00198 | [0.00032, 0.00380] |
| FP | fra_Latn | eng_Latn | 0.05409 | [0.03656, 0.07388] |
| FP | fra_Latn | rus_Cyrl | 0.01075 | [0.00554, 0.01616] |
| FP | fra_Latn | zho_Hans | 0.00664 | [0.00198, 0.01180] |
| FP | fra_Latn | hin_Deva | 0.00514 | [0.00268, 0.00781] |
| FP | por_Latn | eng_Latn | 0.04466 | [0.03287, 0.05718] |
| FP | por_Latn | rus_Cyrl | 0.02598 | [0.01943, 0.03358] |
| FP | por_Latn | zho_Hans | 0.00240 | [0.00001, 0.00494] |
| FP | por_Latn | hin_Deva | 0.01238 | [0.00836, 0.01681] |
| FP | rus_Cyrl | eng_Latn | 0.00591 | [0.00299, 0.00916] |
| FP | rus_Cyrl | zho_Hans | 0.00288 | [0.00017, 0.00546] |
| FP | rus_Cyrl | hin_Deva | 0.00251 | [0.00076, 0.00433] |
| FP | zho_Hans | eng_Latn | 0.04556 | [0.02366, 0.07281] |
| FP | zho_Hans | rus_Cyrl | 0.04174 | [0.02512, 0.05923] |
| FP | zho_Hans | hin_Deva | 0.02400 | [0.01336, 0.03736] |
| FP | jpn_Jpan | eng_Latn | 0.04457 | [0.03537, 0.05397] |
| FP | jpn_Jpan | rus_Cyrl | 0.10475 | [0.08882, 0.12094] |
| FP | jpn_Jpan | zho_Hans | 0.03653 | [0.02689, 0.04600] |
| FP | jpn_Jpan | hin_Deva | 0.02828 | [0.02287, 0.03363] |
| FP | kor_Hang | eng_Latn | 0.00401 | [0.00006, 0.00902] |
| FP | kor_Hang | rus_Cyrl | 0.00127 | [-0.00094, 0.00327] |
| FP | kor_Hang | zho_Hans | 0.01121 | [0.00512, 0.01777] |
| FP | kor_Hang | hin_Deva | 0.01269 | [0.00852, 0.01682] |
| FP | hin_Deva | eng_Latn | -0.00017 | [-0.00313, 0.00294] |
| FP | hin_Deva | rus_Cyrl | 0.02177 | [0.01410, 0.03116] |
| FP | hin_Deva | zho_Hans | 0.03547 | [0.02722, 0.04416] |
| FP | ben_Beng | eng_Latn | 0.00407 | [0.00081, 0.00785] |
| FP | ben_Beng | rus_Cyrl | 0.00529 | [0.00142, 0.00964] |
| FP | ben_Beng | zho_Hans | 0.00533 | [0.00158, 0.00956] |
| FP | ben_Beng | hin_Deva | 0.01000 | [0.00660, 0.01329] |
| FP | tam_Taml | eng_Latn | 0.00497 | [0.00127, 0.00994] |
| FP | tam_Taml | rus_Cyrl | 0.00637 | [0.00198, 0.01181] |
| FP | tam_Taml | zho_Hans | 0.00467 | [0.00139, 0.00828] |
| FP | tam_Taml | hin_Deva | 0.01539 | [0.01125, 0.01981] |
| FP | tur_Latn | eng_Latn | 0.00170 | [-0.00104, 0.00432] |
| FP | tur_Latn | rus_Cyrl | 0.00264 | [0.00047, 0.00476] |
| FP | tur_Latn | zho_Hans | 0.00357 | [0.00009, 0.00714] |
| FP | tur_Latn | hin_Deva | 0.02092 | [0.01647, 0.02545] |
| FP | vie_Latn | eng_Latn | 0.00399 | [-0.00216, 0.01015] |
| FP | vie_Latn | rus_Cyrl | 0.00604 | [0.00293, 0.00940] |
| FP | vie_Latn | zho_Hans | 0.00875 | [0.00489, 0.01278] |
| FP | vie_Latn | hin_Deva | 0.00205 | [-0.00022, 0.00465] |
| FP | ind_Latn | eng_Latn | 0.04181 | [0.02449, 0.06175] |
| FP | ind_Latn | rus_Cyrl | 0.02776 | [0.01384, 0.04674] |
| FP | ind_Latn | zho_Hans | 0.01845 | [0.00600, 0.03555] |
| FP | ind_Latn | hin_Deva | 0.01021 | [0.00502, 0.01671] |
| FP | tha_Thai | eng_Latn | 0.00628 | [0.00251, 0.01067] |
| FP | tha_Thai | rus_Cyrl | 0.00263 | [0.00034, 0.00513] |
| FP | tha_Thai | zho_Hans | 0.00363 | [0.00113, 0.00615] |
| FP | tha_Thai | hin_Deva | 0.00205 | [0.00042, 0.00372] |
| FP | pol_Latn | eng_Latn | 0.00305 | [0.00001, 0.00725] |
| FP | pol_Latn | rus_Cyrl | 0.01933 | [0.01394, 0.02450] |
| FP | pol_Latn | zho_Hans | 0.00680 | [0.00362, 0.00995] |
| FP | pol_Latn | hin_Deva | 0.00880 | [0.00543, 0.01237] |
| FP | ukr_Cyrl | eng_Latn | 0.00470 | [0.00054, 0.00945] |
| FP | ukr_Cyrl | rus_Cyrl | 0.40410 | [0.35879, 0.45165] |
| FP | ukr_Cyrl | zho_Hans | 0.00348 | [0.00042, 0.00664] |
| FP | ukr_Cyrl | hin_Deva | 0.01784 | [0.01319, 0.02256] |
| FP | kan_Knda | eng_Latn | 0.00240 | [0.00025, 0.00443] |
| FP | kan_Knda | rus_Cyrl | 0.00908 | [0.00377, 0.01467] |
| FP | kan_Knda | zho_Hans | 0.01157 | [0.00709, 0.01642] |
| FP | kan_Knda | hin_Deva | 0.01859 | [0.01298, 0.02505] |
| FP | tel_Telu | eng_Latn | 0.01237 | [0.00537, 0.02069] |
| FP | tel_Telu | rus_Cyrl | 0.12104 | [0.10226, 0.14048] |
| FP | tel_Telu | zho_Hans | 0.00917 | [0.00421, 0.01488] |
| FP | tel_Telu | hin_Deva | 0.05909 | [0.05001, 0.06848] |
| FP | yor_Latn | eng_Latn | 0.00651 | [0.00213, 0.01204] |
| FP | yor_Latn | rus_Cyrl | 0.04947 | [0.03697, 0.06324] |
| FP | yor_Latn | zho_Hans | 0.01792 | [0.01029, 0.02662] |
| FP | yor_Latn | hin_Deva | 0.05521 | [0.04622, 0.06448] |
| FP | sun_Latn | eng_Latn | 0.00318 | [0.00037, 0.00601] |
| FP | sun_Latn | rus_Cyrl | 0.00482 | [0.00105, 0.00853] |
| FP | sun_Latn | zho_Hans | 0.00294 | [-0.00090, 0.00692] |
| FP | sun_Latn | hin_Deva | 0.01238 | [0.00716, 0.01835] |
| FP | amh_Ethi | eng_Latn | 0.05518 | [0.03533, 0.07544] |
| FP | amh_Ethi | rus_Cyrl | 0.04912 | [0.03358, 0.06513] |
| FP | amh_Ethi | zho_Hans | 0.02615 | [0.01834, 0.03497] |
| FP | amh_Ethi | hin_Deva | 0.07237 | [0.05628, 0.09060] |
| FP | ibo_Latn | eng_Latn | 0.02303 | [0.01178, 0.03585] |
| FP | ibo_Latn | rus_Cyrl | 0.04173 | [0.03112, 0.05233] |
| FP | ibo_Latn | zho_Hans | 0.01111 | [0.00632, 0.01596] |
| FP | ibo_Latn | hin_Deva | 0.02236 | [0.01638, 0.02918] |
| FP | hau_Latn | eng_Latn | 0.01343 | [0.00584, 0.02223] |
| FP | hau_Latn | rus_Cyrl | 0.03220 | [0.02317, 0.04281] |
| FP | hau_Latn | zho_Hans | 0.02445 | [0.01586, 0.03396] |
| FP | hau_Latn | hin_Deva | 0.03605 | [0.02838, 0.04448] |
| FP | khm_Khmr | eng_Latn | 0.00561 | [0.00192, 0.01016] |
| FP | khm_Khmr | rus_Cyrl | 0.01617 | [0.01025, 0.02248] |
| FP | khm_Khmr | zho_Hans | 0.00819 | [0.00295, 0.01437] |
| FP | khm_Khmr | hin_Deva | 0.02879 | [0.02197, 0.03609] |
| FP | lao_Laoo | eng_Latn | 0.00597 | [0.00200, 0.01063] |
| FP | lao_Laoo | rus_Cyrl | 0.02791 | [0.02033, 0.03627] |
| FP | lao_Laoo | zho_Hans | 0.01250 | [0.00604, 0.02010] |
| FP | lao_Laoo | hin_Deva | 0.09127 | [0.07923, 0.10416] |
