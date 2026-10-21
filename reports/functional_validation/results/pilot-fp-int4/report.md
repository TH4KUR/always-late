# Functional effects of language feature groups (group size K64)

**Exploratory on already examined data** (FLORES devtest sentences; the features were selected on dev). Next-token loss only; no claim about task accuracy or meaning. E is the harm (nats per token, NLL after the edit minus NLL before it, same precision, same sentences) of zero-ablating the language's selective group beyond the mean harm of its activity-matched controls on the language's own text; positive = the selective group matters more than equally active non-selective features. Intervals: 2000 paired cluster-bootstrap draws over sentence ids (percentile, descriptive; the observed value is the estimate). Macro retention is the mean over languages of the per-language ratio E at a precision over E in full precision (D-041; languages whose full-precision interval lies above zero), classified preserved / attenuated / amplified against the bands [0.75, 1.33], inconclusive otherwise, undefined when the full-precision interval does not lie above zero.

Languages in the macro average: 27 (eng_Latn, deu_Latn, spa_Latn, fra_Latn, por_Latn, rus_Cyrl, zho_Hans, jpn_Jpan, kor_Hang, hin_Deva, ben_Beng, tam_Taml, tur_Latn, vie_Latn, ind_Latn, tha_Thai, pol_Latn, ukr_Cyrl, kan_Knda, tel_Telu, yor_Latn, sun_Latn, amh_Ethi, ibo_Latn, hau_Latn, khm_Khmr, lao_Laoo). Positions (sentences per language): 150.

## Macro effect and retention

| precision | mean E | 95 % interval | median E | languages with E > 0 | retention (mean of per-language ratios) | interval | class (languages) | median ratio [interval] | pooled ratio (dominated by large effects) |
|---|---|---|---|---|---|---|---|---|---|
| FP | 0.6927 | [0.6673, 0.7191] | 0.2656 | 27 of 27 | - | - | reference | - | - |
| INT4 | 0.9267 | [0.8998, 0.9551] | 0.4583 | 27 of 27 | 2.14 | [2.02, 2.55] | amplified (26) | 1.43 [1.40, 1.61] | 1.34 |

## Dose, size and replacement (median over languages of the harm in nats per token on the language's own text)

| family | FP | INT4 |
|---|---|---|
| E_K16 | 0.2157 | 0.3568 |
| E_K64 | 0.2656 | 0.4583 |
| E_all | 0.3001 | 0.5640 |
| ctrl_K16 | 0.0159 | 0.0224 |
| ctrl_K64 | 0.0218 | 0.0306 |
| ctrl_all | 0.0265 | 0.0418 |
| mean_K64 | 0.2369 | 0.4153 |
| res_K64 | 0.3001 | 0.4959 |
| scale0.5_K64 | 0.0092 | 0.0496 |
| scale2_K64 | 0.0303 | 0.0211 |
| sham_K64 | 0.0000 | 0.0000 |
| zero_K16 | 0.2494 | 0.3846 |
| zero_K64 | 0.2973 | 0.5182 |
| zero_all | 0.3470 | 0.6097 |

Languages whose E (zero ablation beyond controls) has a 95 % lower bound above 0: FP: E_K16 25, E_K64 26, E_all 26; INT4: E_K16 26, E_K64 27, E_all 27


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
| INT4 | tam_Taml | 0.00000 | [0.00000, 0.00000] |
| INT4 | tur_Latn | 0.00000 | [0.00000, 0.00000] |
| INT4 | vie_Latn | 0.00000 | [0.00000, 0.00000] |
| INT4 | ind_Latn | 0.00000 | [0.00000, 0.00000] |
| INT4 | tha_Thai | 0.00000 | [0.00000, 0.00000] |
| INT4 | pol_Latn | 0.00000 | [0.00000, 0.00000] |
| INT4 | ukr_Cyrl | 0.00000 | [0.00000, 0.00000] |
| INT4 | kan_Knda | 0.00000 | [0.00000, 0.00000] |
| INT4 | tel_Telu | 0.00000 | [0.00000, 0.00000] |
| INT4 | yor_Latn | 0.00000 | [0.00000, 0.00000] |
| INT4 | sun_Latn | 0.00000 | [0.00000, 0.00000] |
| INT4 | amh_Ethi | 0.00000 | [0.00000, 0.00000] |
| INT4 | ibo_Latn | 0.00000 | [0.00000, 0.00000] |
| INT4 | hau_Latn | 0.00000 | [0.00000, 0.00000] |
| INT4 | khm_Khmr | 0.00000 | [0.00000, 0.00000] |
| INT4 | lao_Laoo | 0.00000 | [0.00000, 0.00000] |

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
| INT4 | eng_Latn | 0.2043 | [0.1684, 0.2407] | 0.2618 | 0.0575 | 0.122 / 0.136 |
| INT4 | deu_Latn | 0.7430 | [0.6660, 0.8232] | 0.7617 | 0.0187 | 0.109 / 0.101 |
| INT4 | spa_Latn | 0.6833 | [0.5707, 0.8099] | 0.7140 | 0.0306 | 0.098 / 0.094 |
| INT4 | fra_Latn | 0.8911 | [0.8140, 0.9765] | 0.9210 | 0.0299 | 0.105 / 0.100 |
| INT4 | por_Latn | 0.8762 | [0.7691, 0.9897] | 0.8808 | 0.0046 | 0.071 / 0.058 |
| INT4 | rus_Cyrl | 0.2046 | [0.1710, 0.2412] | 0.2137 | 0.0090 | 0.065 / 0.077 |
| INT4 | zho_Hans | 8.0449 | [7.5982, 8.5110] | 8.0722 | 0.0273 | 0.167 / 0.153 |
| INT4 | jpn_Jpan | 0.3793 | [0.3370, 0.4241] | 0.4181 | 0.0388 | 0.126 / 0.137 |
| INT4 | kor_Hang | 0.8392 | [0.7116, 0.9733] | 0.8610 | 0.0218 | 0.088 / 0.089 |
| INT4 | hin_Deva | 0.2979 | [0.2575, 0.3408] | 0.3225 | 0.0246 | 0.118 / 0.122 |
| INT4 | ben_Beng | 0.0427 | [0.0354, 0.0501] | 0.0789 | 0.0362 | 0.049 / 0.041 |
| INT4 | tam_Taml | 0.0782 | [0.0694, 0.0872] | 0.1002 | 0.0219 | 0.068 / 0.065 |
| INT4 | tur_Latn | 0.5859 | [0.5261, 0.6517] | 0.6417 | 0.0558 | 0.108 / 0.130 |
| INT4 | vie_Latn | 0.0555 | [0.0396, 0.0726] | 0.0820 | 0.0265 | 0.072 / 0.079 |
| INT4 | ind_Latn | 0.0206 | [0.0059, 0.0365] | 0.0335 | 0.0129 | 0.046 / 0.045 |
| INT4 | tha_Thai | 0.1749 | [0.1563, 0.1939] | 0.1945 | 0.0196 | 0.095 / 0.139 |
| INT4 | pol_Latn | 0.5671 | [0.4998, 0.6392] | 0.6069 | 0.0398 | 0.094 / 0.087 |
| INT4 | ukr_Cyrl | 0.9692 | [0.8785, 1.0655] | 0.9957 | 0.0264 | 0.094 / 0.091 |
| INT4 | kan_Knda | 0.0391 | [0.0294, 0.0489] | 0.0726 | 0.0334 | 0.081 / 0.074 |
| INT4 | tel_Telu | 0.1762 | [0.1543, 0.2020] | 0.2129 | 0.0367 | 0.112 / 0.132 |
| INT4 | yor_Latn | 2.1761 | [2.0861, 2.2647] | 2.2940 | 0.1180 | 0.308 / 0.259 |
| INT4 | sun_Latn | 1.0970 | [1.0284, 1.1635] | 1.2093 | 0.1123 | 0.158 / 0.166 |
| INT4 | amh_Ethi | 2.3317 | [2.2544, 2.4121] | 2.3700 | 0.0383 | 0.220 / 0.169 |
| INT4 | ibo_Latn | 0.4583 | [0.4271, 0.4904] | 0.5182 | 0.0599 | 0.161 / 0.146 |
| INT4 | hau_Latn | 0.2160 | [0.1930, 0.2392] | 0.2965 | 0.0805 | 0.124 / 0.106 |
| INT4 | khm_Khmr | 0.4075 | [0.3770, 0.4371] | 0.4758 | 0.0684 | 0.156 / 0.151 |
| INT4 | lao_Laoo | 2.4621 | [2.3849, 2.5356] | 2.4623 | 0.0002 | 0.197 / 0.182 |

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
| INT4 | eng_Latn | rus_Cyrl | 0.00817 | [0.00491, 0.01167] |
| INT4 | eng_Latn | zho_Hans | 0.00357 | [-0.00078, 0.00770] |
| INT4 | eng_Latn | hin_Deva | 0.00322 | [0.00138, 0.00516] |
| INT4 | deu_Latn | eng_Latn | 0.01770 | [0.00728, 0.03023] |
| INT4 | deu_Latn | rus_Cyrl | 0.02690 | [0.02051, 0.03374] |
| INT4 | deu_Latn | zho_Hans | 0.01193 | [0.00810, 0.01593] |
| INT4 | deu_Latn | hin_Deva | 0.00832 | [0.00549, 0.01136] |
| INT4 | spa_Latn | eng_Latn | 0.06378 | [0.04741, 0.08131] |
| INT4 | spa_Latn | rus_Cyrl | 0.01443 | [0.00941, 0.01945] |
| INT4 | spa_Latn | zho_Hans | 0.00307 | [0.00045, 0.00568] |
| INT4 | spa_Latn | hin_Deva | 0.00283 | [0.00117, 0.00460] |
| INT4 | fra_Latn | eng_Latn | 0.05889 | [0.04100, 0.07799] |
| INT4 | fra_Latn | rus_Cyrl | 0.01369 | [0.00903, 0.01903] |
| INT4 | fra_Latn | zho_Hans | 0.00657 | [0.00228, 0.01191] |
| INT4 | fra_Latn | hin_Deva | 0.00823 | [0.00577, 0.01064] |
| INT4 | por_Latn | eng_Latn | 0.05275 | [0.04033, 0.06676] |
| INT4 | por_Latn | rus_Cyrl | 0.03705 | [0.02963, 0.04527] |
| INT4 | por_Latn | zho_Hans | 0.00407 | [0.00096, 0.00727] |
| INT4 | por_Latn | hin_Deva | 0.01813 | [0.01374, 0.02275] |
| INT4 | rus_Cyrl | eng_Latn | 0.00651 | [0.00288, 0.01033] |
| INT4 | rus_Cyrl | zho_Hans | 0.00306 | [-0.00009, 0.00618] |
| INT4 | rus_Cyrl | hin_Deva | 0.00646 | [0.00458, 0.00844] |
| INT4 | zho_Hans | eng_Latn | 0.04399 | [0.02205, 0.07115] |
| INT4 | zho_Hans | rus_Cyrl | 0.04319 | [0.02672, 0.06053] |
| INT4 | zho_Hans | hin_Deva | 0.02595 | [0.01685, 0.03720] |
| INT4 | jpn_Jpan | eng_Latn | 0.04942 | [0.03818, 0.06128] |
| INT4 | jpn_Jpan | rus_Cyrl | 0.10965 | [0.09399, 0.12588] |
| INT4 | jpn_Jpan | zho_Hans | 0.04052 | [0.02936, 0.05131] |
| INT4 | jpn_Jpan | hin_Deva | 0.04310 | [0.03577, 0.05051] |
| INT4 | kor_Hang | eng_Latn | 0.00579 | [0.00003, 0.01247] |
| INT4 | kor_Hang | rus_Cyrl | 0.00439 | [0.00147, 0.00726] |
| INT4 | kor_Hang | zho_Hans | 0.01724 | [0.01004, 0.02479] |
| INT4 | kor_Hang | hin_Deva | 0.02912 | [0.02429, 0.03387] |
| INT4 | hin_Deva | eng_Latn | -0.00017 | [-0.00399, 0.00390] |
| INT4 | hin_Deva | rus_Cyrl | 0.04226 | [0.02918, 0.05675] |
| INT4 | hin_Deva | zho_Hans | 0.03695 | [0.02635, 0.04838] |
| INT4 | ben_Beng | eng_Latn | 0.00415 | [0.00022, 0.00836] |
| INT4 | ben_Beng | rus_Cyrl | 0.01240 | [0.00730, 0.01800] |
| INT4 | ben_Beng | zho_Hans | 0.00558 | [0.00160, 0.00955] |
| INT4 | ben_Beng | hin_Deva | 0.01899 | [0.01448, 0.02378] |
| INT4 | tam_Taml | eng_Latn | 0.00780 | [0.00339, 0.01274] |
| INT4 | tam_Taml | rus_Cyrl | 0.01138 | [0.00591, 0.01737] |
| INT4 | tam_Taml | zho_Hans | 0.00335 | [0.00023, 0.00650] |
| INT4 | tam_Taml | hin_Deva | 0.03185 | [0.02613, 0.03793] |
| INT4 | tur_Latn | eng_Latn | 0.00162 | [-0.00068, 0.00394] |
| INT4 | tur_Latn | rus_Cyrl | 0.00515 | [0.00254, 0.00809] |
| INT4 | tur_Latn | zho_Hans | 0.00872 | [0.00377, 0.01562] |
| INT4 | tur_Latn | hin_Deva | 0.03874 | [0.03334, 0.04434] |
| INT4 | vie_Latn | eng_Latn | 0.01078 | [0.00392, 0.01855] |
| INT4 | vie_Latn | rus_Cyrl | 0.00954 | [0.00623, 0.01300] |
| INT4 | vie_Latn | zho_Hans | 0.01034 | [0.00675, 0.01419] |
| INT4 | vie_Latn | hin_Deva | 0.00345 | [0.00098, 0.00606] |
| INT4 | ind_Latn | eng_Latn | 0.04550 | [0.02825, 0.06466] |
| INT4 | ind_Latn | rus_Cyrl | 0.02936 | [0.01504, 0.04759] |
| INT4 | ind_Latn | zho_Hans | 0.01652 | [0.00489, 0.03209] |
| INT4 | ind_Latn | hin_Deva | 0.01038 | [0.00510, 0.01654] |
| INT4 | tha_Thai | eng_Latn | 0.00675 | [0.00193, 0.01244] |
| INT4 | tha_Thai | rus_Cyrl | 0.00478 | [0.00242, 0.00735] |
| INT4 | tha_Thai | zho_Hans | 0.00347 | [0.00047, 0.00634] |
| INT4 | tha_Thai | hin_Deva | 0.00219 | [0.00009, 0.00444] |
| INT4 | pol_Latn | eng_Latn | 0.00355 | [-0.00043, 0.00944] |
| INT4 | pol_Latn | rus_Cyrl | 0.02425 | [0.01855, 0.03003] |
| INT4 | pol_Latn | zho_Hans | 0.00820 | [0.00416, 0.01232] |
| INT4 | pol_Latn | hin_Deva | 0.01661 | [0.01188, 0.02158] |
| INT4 | ukr_Cyrl | eng_Latn | 0.01016 | [0.00495, 0.01581] |
| INT4 | ukr_Cyrl | rus_Cyrl | 0.83337 | [0.76080, 0.90479] |
| INT4 | ukr_Cyrl | zho_Hans | 0.00617 | [0.00169, 0.01155] |
| INT4 | ukr_Cyrl | hin_Deva | 0.02248 | [0.01722, 0.02788] |
| INT4 | kan_Knda | eng_Latn | 0.00522 | [0.00286, 0.00768] |
| INT4 | kan_Knda | rus_Cyrl | 0.01931 | [0.01321, 0.02561] |
| INT4 | kan_Knda | zho_Hans | 0.01124 | [0.00492, 0.01773] |
| INT4 | kan_Knda | hin_Deva | 0.03481 | [0.02771, 0.04240] |
| INT4 | tel_Telu | eng_Latn | 0.02045 | [0.01122, 0.03104] |
| INT4 | tel_Telu | rus_Cyrl | 0.22738 | [0.20336, 0.25225] |
| INT4 | tel_Telu | zho_Hans | 0.02001 | [0.01226, 0.02851] |
| INT4 | tel_Telu | hin_Deva | 0.13897 | [0.12491, 0.15277] |
| INT4 | yor_Latn | eng_Latn | 0.01026 | [0.00440, 0.01720] |
| INT4 | yor_Latn | rus_Cyrl | 0.08338 | [0.06664, 0.10057] |
| INT4 | yor_Latn | zho_Hans | 0.02384 | [0.01340, 0.03545] |
| INT4 | yor_Latn | hin_Deva | 0.07321 | [0.06071, 0.08564] |
| INT4 | sun_Latn | eng_Latn | 0.00500 | [0.00027, 0.00990] |
| INT4 | sun_Latn | rus_Cyrl | 0.01698 | [0.01176, 0.02228] |
| INT4 | sun_Latn | zho_Hans | 0.00464 | [-0.00004, 0.00950] |
| INT4 | sun_Latn | hin_Deva | 0.03036 | [0.02369, 0.03759] |
| INT4 | amh_Ethi | eng_Latn | 0.05648 | [0.03678, 0.07748] |
| INT4 | amh_Ethi | rus_Cyrl | 0.05567 | [0.04158, 0.06997] |
| INT4 | amh_Ethi | zho_Hans | 0.02532 | [0.01675, 0.03482] |
| INT4 | amh_Ethi | hin_Deva | 0.09352 | [0.07682, 0.11102] |
| INT4 | ibo_Latn | eng_Latn | 0.02720 | [0.01589, 0.03969] |
| INT4 | ibo_Latn | rus_Cyrl | 0.05933 | [0.04793, 0.07120] |
| INT4 | ibo_Latn | zho_Hans | 0.01480 | [0.00974, 0.02001] |
| INT4 | ibo_Latn | hin_Deva | 0.04061 | [0.03224, 0.04934] |
| INT4 | hau_Latn | eng_Latn | 0.01826 | [0.01115, 0.02569] |
| INT4 | hau_Latn | rus_Cyrl | 0.05195 | [0.04152, 0.06256] |
| INT4 | hau_Latn | zho_Hans | 0.02737 | [0.01838, 0.03774] |
| INT4 | hau_Latn | hin_Deva | 0.06475 | [0.05554, 0.07430] |
| INT4 | khm_Khmr | eng_Latn | 0.01454 | [0.00801, 0.02306] |
| INT4 | khm_Khmr | rus_Cyrl | 0.02413 | [0.01650, 0.03241] |
| INT4 | khm_Khmr | zho_Hans | 0.01171 | [0.00475, 0.01933] |
| INT4 | khm_Khmr | hin_Deva | 0.05635 | [0.04817, 0.06499] |
| INT4 | lao_Laoo | eng_Latn | 0.00896 | [0.00485, 0.01356] |
| INT4 | lao_Laoo | rus_Cyrl | 0.05534 | [0.04459, 0.06587] |
| INT4 | lao_Laoo | zho_Hans | 0.02554 | [0.01439, 0.03899] |
| INT4 | lao_Laoo | hin_Deva | 0.15754 | [0.14333, 0.17193] |
