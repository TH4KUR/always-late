# Functional effects of language feature groups (group size K64)

**Exploratory on already examined data** (FLORES devtest sentences; the features were selected on dev). Next-token loss only; no claim about task accuracy or meaning. E is the harm (nats per token, NLL after the edit minus NLL before it, same precision, same sentences) of zero-ablating the language's selective group beyond the mean harm of its activity-matched controls on the language's own text; positive = the selective group matters more than equally active non-selective features. Intervals: 2000 paired cluster-bootstrap draws over sentence ids (percentile, descriptive; the observed value is the estimate). Macro retention is the mean over languages of the per-language ratio E at a precision over E in full precision (D-041; languages whose full-precision interval lies above zero), classified preserved / attenuated / amplified against the bands [0.75, 1.33], inconclusive otherwise, undefined when the full-precision interval does not lie above zero.

Languages in the macro average: 27 (eng_Latn, deu_Latn, spa_Latn, fra_Latn, por_Latn, rus_Cyrl, zho_Hans, jpn_Jpan, kor_Hang, hin_Deva, ben_Beng, tam_Taml, tur_Latn, vie_Latn, ind_Latn, tha_Thai, pol_Latn, ukr_Cyrl, kan_Knda, tel_Telu, yor_Latn, sun_Latn, amh_Ethi, ibo_Latn, hau_Latn, khm_Khmr, lao_Laoo). Positions (sentences per language): 150.

## Macro effect and retention

| precision | mean E | 95 % interval | median E | languages with E > 0 | retention (mean of per-language ratios) | interval | class (languages) | median ratio [interval] | pooled ratio (dominated by large effects) |
|---|---|---|---|---|---|---|---|---|---|
| FP | 0.6927 | [0.6673, 0.7191] | 0.2656 | 27 of 27 | - | - | reference | - | - |
| INT16 | 0.6938 | [0.6685, 0.7203] | 0.2655 | 27 of 27 | 1.01 | [1.00, 1.02] | preserved (26) | 1.00 [1.00, 1.00] | 1.00 |
| INT4 | 0.9267 | [0.8998, 0.9551] | 0.4583 | 27 of 27 | 2.14 | [2.02, 2.55] | amplified (26) | 1.43 [1.40, 1.61] | 1.34 |
| INT8 | 0.7005 | [0.6746, 0.7275] | 0.2704 | 27 of 27 | 1.01 | [1.00, 1.03] | preserved (26) | 1.01 [1.00, 1.01] | 1.01 |

## Dose, size and replacement (median over languages of the harm in nats per token on the language's own text)

| family | FP | INT16 | INT4 | INT8 |
|---|---|---|---|---|
| E_K16 | 0.2157 | 0.2149 | 0.3568 | 0.2184 |
| E_K64 | 0.2656 | 0.2655 | 0.4583 | 0.2704 |
| E_all | 0.3001 | 0.2993 | 0.5640 | 0.2999 |
| ctrl_K16 | 0.0159 | 0.0162 | 0.0224 | 0.0158 |
| ctrl_K64 | 0.0218 | 0.0219 | 0.0306 | 0.0218 |
| ctrl_all | 0.0265 | 0.0263 | 0.0418 | 0.0263 |
| mean_K64 | 0.2369 | 0.2361 | 0.4153 | 0.2367 |
| res_K64 | 0.3001 | 0.3004 | 0.4959 | 0.3121 |
| scale0.5_K64 | 0.0092 | 0.0084 | 0.0496 | 0.0091 |
| scale2_K64 | 0.0303 | 0.0306 | 0.0211 | 0.0294 |
| sham_K64 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| zero_K16 | 0.2494 | 0.2486 | 0.3846 | 0.2521 |
| zero_K64 | 0.2973 | 0.2977 | 0.5182 | 0.3025 |
| zero_all | 0.3470 | 0.3484 | 0.6097 | 0.3496 |

Languages whose E (zero ablation beyond controls) has a 95 % lower bound above 0: FP: E_K16 25, E_K64 26, E_all 26; INT16: E_K16 25, E_K64 26, E_all 26; INT4: E_K16 26, E_K64 27, E_all 27; INT8: E_K16 25, E_K64 26, E_all 26


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
| INT16 | eng_Latn | 0.00000 | [0.00000, 0.00000] |
| INT16 | deu_Latn | 0.00000 | [0.00000, 0.00000] |
| INT16 | spa_Latn | 0.00000 | [0.00000, 0.00000] |
| INT16 | fra_Latn | 0.00000 | [0.00000, 0.00000] |
| INT16 | por_Latn | 0.00000 | [0.00000, 0.00000] |
| INT16 | rus_Cyrl | 0.00000 | [0.00000, 0.00000] |
| INT16 | zho_Hans | 0.00000 | [0.00000, 0.00000] |
| INT16 | jpn_Jpan | 0.00000 | [0.00000, 0.00000] |
| INT16 | kor_Hang | 0.00000 | [0.00000, 0.00000] |
| INT16 | hin_Deva | 0.00000 | [0.00000, 0.00000] |
| INT16 | ben_Beng | 0.00000 | [0.00000, 0.00000] |
| INT16 | tam_Taml | 0.00000 | [0.00000, 0.00000] |
| INT16 | tur_Latn | 0.00000 | [0.00000, 0.00000] |
| INT16 | vie_Latn | 0.00000 | [0.00000, 0.00000] |
| INT16 | ind_Latn | 0.00000 | [0.00000, 0.00000] |
| INT16 | tha_Thai | 0.00000 | [0.00000, 0.00000] |
| INT16 | pol_Latn | 0.00000 | [0.00000, 0.00000] |
| INT16 | ukr_Cyrl | 0.00000 | [0.00000, 0.00000] |
| INT16 | kan_Knda | 0.00000 | [0.00000, 0.00000] |
| INT16 | tel_Telu | 0.00000 | [0.00000, 0.00000] |
| INT16 | yor_Latn | 0.00000 | [0.00000, 0.00000] |
| INT16 | sun_Latn | 0.00000 | [0.00000, 0.00000] |
| INT16 | amh_Ethi | 0.00000 | [0.00000, 0.00000] |
| INT16 | ibo_Latn | 0.00000 | [0.00000, 0.00000] |
| INT16 | hau_Latn | 0.00000 | [0.00000, 0.00000] |
| INT16 | khm_Khmr | 0.00000 | [0.00000, 0.00000] |
| INT16 | lao_Laoo | 0.00000 | [0.00000, 0.00000] |
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
| INT8 | eng_Latn | 0.00000 | [0.00000, 0.00000] |
| INT8 | deu_Latn | 0.00000 | [0.00000, 0.00000] |
| INT8 | spa_Latn | 0.00000 | [0.00000, 0.00000] |
| INT8 | fra_Latn | 0.00000 | [0.00000, 0.00000] |
| INT8 | por_Latn | 0.00000 | [0.00000, 0.00000] |
| INT8 | rus_Cyrl | 0.00000 | [0.00000, 0.00000] |
| INT8 | zho_Hans | 0.00000 | [0.00000, 0.00000] |
| INT8 | jpn_Jpan | 0.00000 | [0.00000, 0.00000] |
| INT8 | kor_Hang | 0.00000 | [0.00000, 0.00000] |
| INT8 | hin_Deva | 0.00000 | [0.00000, 0.00000] |
| INT8 | ben_Beng | 0.00000 | [0.00000, 0.00000] |
| INT8 | tam_Taml | 0.00000 | [0.00000, 0.00000] |
| INT8 | tur_Latn | 0.00000 | [0.00000, 0.00000] |
| INT8 | vie_Latn | 0.00000 | [0.00000, 0.00000] |
| INT8 | ind_Latn | 0.00000 | [0.00000, 0.00000] |
| INT8 | tha_Thai | 0.00000 | [0.00000, 0.00000] |
| INT8 | pol_Latn | 0.00000 | [0.00000, 0.00000] |
| INT8 | ukr_Cyrl | 0.00000 | [0.00000, 0.00000] |
| INT8 | kan_Knda | 0.00000 | [0.00000, 0.00000] |
| INT8 | tel_Telu | 0.00000 | [0.00000, 0.00000] |
| INT8 | yor_Latn | 0.00000 | [0.00000, 0.00000] |
| INT8 | sun_Latn | 0.00000 | [0.00000, 0.00000] |
| INT8 | amh_Ethi | 0.00000 | [0.00000, 0.00000] |
| INT8 | ibo_Latn | 0.00000 | [0.00000, 0.00000] |
| INT8 | hau_Latn | 0.00000 | [0.00000, 0.00000] |
| INT8 | khm_Khmr | 0.00000 | [0.00000, 0.00000] |
| INT8 | lao_Laoo | 0.00000 | [0.00000, 0.00000] |

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
| INT16 | eng_Latn | 0.1934 | [0.1592, 0.2281] | 0.2392 | 0.0458 | 0.144 / 0.140 |
| INT16 | deu_Latn | 0.9743 | [0.8798, 1.0678] | 0.9858 | 0.0115 | 0.122 / 0.110 |
| INT16 | spa_Latn | 0.4965 | [0.3888, 0.6348] | 0.5184 | 0.0219 | 0.107 / 0.098 |
| INT16 | fra_Latn | 0.6243 | [0.5494, 0.7068] | 0.6432 | 0.0190 | 0.120 / 0.099 |
| INT16 | por_Latn | 0.9725 | [0.8409, 1.1073] | 0.9734 | 0.0009 | 0.075 / 0.067 |
| INT16 | rus_Cyrl | 0.0681 | [0.0531, 0.0842] | 0.0729 | 0.0048 | 0.079 / 0.081 |
| INT16 | zho_Hans | 7.3628 | [6.8869, 7.8379] | 7.3888 | 0.0260 | 0.178 / 0.162 |
| INT16 | jpn_Jpan | 0.2655 | [0.2319, 0.3014] | 0.2977 | 0.0321 | 0.141 / 0.140 |
| INT16 | kor_Hang | 0.3932 | [0.3233, 0.4614] | 0.4085 | 0.0153 | 0.097 / 0.095 |
| INT16 | hin_Deva | 0.1121 | [0.0879, 0.1384] | 0.1245 | 0.0124 | 0.128 / 0.129 |
| INT16 | ben_Beng | 0.0075 | [0.0028, 0.0121] | 0.0173 | 0.0098 | 0.055 / 0.050 |
| INT16 | tam_Taml | 0.0305 | [0.0238, 0.0370] | 0.0430 | 0.0125 | 0.077 / 0.072 |
| INT16 | tur_Latn | 0.2214 | [0.1913, 0.2553] | 0.2632 | 0.0418 | 0.128 / 0.127 |
| INT16 | vie_Latn | 0.0061 | [-0.0055, 0.0184] | 0.0183 | 0.0123 | 0.084 / 0.085 |
| INT16 | ind_Latn | 0.0195 | [0.0060, 0.0346] | 0.0287 | 0.0092 | 0.051 / 0.052 |
| INT16 | tha_Thai | 0.0386 | [0.0286, 0.0488] | 0.0535 | 0.0149 | 0.110 / 0.150 |
| INT16 | pol_Latn | 0.3169 | [0.2704, 0.3725] | 0.3451 | 0.0282 | 0.100 / 0.094 |
| INT16 | ukr_Cyrl | 0.4032 | [0.3570, 0.4524] | 0.4178 | 0.0147 | 0.108 / 0.091 |
| INT16 | kan_Knda | 0.0358 | [0.0283, 0.0432] | 0.0515 | 0.0157 | 0.090 / 0.086 |
| INT16 | tel_Telu | 0.0257 | [0.0178, 0.0348] | 0.0559 | 0.0302 | 0.119 / 0.131 |
| INT16 | yor_Latn | 1.5191 | [1.4567, 1.5813] | 1.5837 | 0.0646 | 0.343 / 0.277 |
| INT16 | sun_Latn | 0.9369 | [0.8654, 1.0046] | 1.0403 | 0.1034 | 0.169 / 0.172 |
| INT16 | amh_Ethi | 1.0306 | [0.9921, 1.0719] | 1.0627 | 0.0321 | 0.236 / 0.181 |
| INT16 | ibo_Latn | 0.3618 | [0.3315, 0.3915] | 0.4076 | 0.0458 | 0.184 / 0.168 |
| INT16 | hau_Latn | 0.1783 | [0.1549, 0.2015] | 0.2306 | 0.0523 | 0.139 / 0.120 |
| INT16 | khm_Khmr | 0.2210 | [0.1949, 0.2469] | 0.2548 | 0.0337 | 0.174 / 0.168 |
| INT16 | lao_Laoo | 1.9181 | [1.8134, 2.0250] | 1.9475 | 0.0294 | 0.254 / 0.194 |
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
| INT8 | eng_Latn | 0.1916 | [0.1574, 0.2260] | 0.2367 | 0.0452 | 0.143 / 0.140 |
| INT8 | deu_Latn | 0.9684 | [0.8731, 1.0613] | 0.9780 | 0.0097 | 0.123 / 0.110 |
| INT8 | spa_Latn | 0.5227 | [0.4121, 0.6628] | 0.5445 | 0.0218 | 0.107 / 0.098 |
| INT8 | fra_Latn | 0.6452 | [0.5649, 0.7322] | 0.6653 | 0.0201 | 0.121 / 0.099 |
| INT8 | por_Latn | 0.9294 | [0.7995, 1.0651] | 0.9307 | 0.0014 | 0.075 / 0.066 |
| INT8 | rus_Cyrl | 0.0721 | [0.0567, 0.0887] | 0.0762 | 0.0042 | 0.079 / 0.081 |
| INT8 | zho_Hans | 7.4842 | [7.0015, 7.9556] | 7.5095 | 0.0253 | 0.178 / 0.162 |
| INT8 | jpn_Jpan | 0.2704 | [0.2365, 0.3058] | 0.3025 | 0.0321 | 0.142 / 0.140 |
| INT8 | kor_Hang | 0.3536 | [0.2901, 0.4165] | 0.3701 | 0.0165 | 0.097 / 0.095 |
| INT8 | hin_Deva | 0.1083 | [0.0845, 0.1335] | 0.1211 | 0.0128 | 0.128 / 0.129 |
| INT8 | ben_Beng | 0.0083 | [0.0035, 0.0131] | 0.0173 | 0.0090 | 0.055 / 0.050 |
| INT8 | tam_Taml | 0.0309 | [0.0243, 0.0376] | 0.0441 | 0.0132 | 0.077 / 0.072 |
| INT8 | tur_Latn | 0.2212 | [0.1906, 0.2553] | 0.2631 | 0.0419 | 0.129 / 0.127 |
| INT8 | vie_Latn | 0.0065 | [-0.0055, 0.0188] | 0.0188 | 0.0123 | 0.084 / 0.085 |
| INT8 | ind_Latn | 0.0183 | [0.0049, 0.0335] | 0.0268 | 0.0085 | 0.051 / 0.052 |
| INT8 | tha_Thai | 0.0364 | [0.0265, 0.0467] | 0.0511 | 0.0147 | 0.110 / 0.150 |
| INT8 | pol_Latn | 0.3302 | [0.2808, 0.3877] | 0.3578 | 0.0275 | 0.101 / 0.094 |
| INT8 | ukr_Cyrl | 0.4223 | [0.3740, 0.4738] | 0.4367 | 0.0144 | 0.108 / 0.091 |
| INT8 | kan_Knda | 0.0366 | [0.0290, 0.0442] | 0.0528 | 0.0162 | 0.090 / 0.086 |
| INT8 | tel_Telu | 0.0254 | [0.0172, 0.0348] | 0.0565 | 0.0311 | 0.119 / 0.130 |
| INT8 | yor_Latn | 1.5275 | [1.4650, 1.5890] | 1.5901 | 0.0626 | 0.342 / 0.277 |
| INT8 | sun_Latn | 0.9245 | [0.8545, 0.9915] | 1.0268 | 0.1023 | 0.169 / 0.172 |
| INT8 | amh_Ethi | 1.0209 | [0.9826, 1.0613] | 1.0537 | 0.0327 | 0.236 / 0.180 |
| INT8 | ibo_Latn | 0.3622 | [0.3323, 0.3914] | 0.4087 | 0.0465 | 0.184 / 0.168 |
| INT8 | hau_Latn | 0.1786 | [0.1557, 0.2020] | 0.2330 | 0.0544 | 0.139 / 0.120 |
| INT8 | khm_Khmr | 0.2220 | [0.1954, 0.2483] | 0.2563 | 0.0343 | 0.173 / 0.167 |
| INT8 | lao_Laoo | 1.9956 | [1.8867, 2.1066] | 2.0261 | 0.0305 | 0.253 / 0.194 |

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
| INT16 | eng_Latn | rus_Cyrl | 0.00512 | [0.00125, 0.00897] |
| INT16 | eng_Latn | zho_Hans | 0.00674 | [0.00124, 0.01222] |
| INT16 | eng_Latn | hin_Deva | 0.00020 | [-0.00220, 0.00255] |
| INT16 | deu_Latn | eng_Latn | 0.01129 | [0.00312, 0.02029] |
| INT16 | deu_Latn | rus_Cyrl | 0.01869 | [0.01303, 0.02560] |
| INT16 | deu_Latn | zho_Hans | 0.01067 | [0.00662, 0.01451] |
| INT16 | deu_Latn | hin_Deva | 0.00354 | [0.00132, 0.00582] |
| INT16 | spa_Latn | eng_Latn | 0.04738 | [0.03537, 0.06043] |
| INT16 | spa_Latn | rus_Cyrl | 0.00598 | [0.00217, 0.01005] |
| INT16 | spa_Latn | zho_Hans | 0.00494 | [0.00239, 0.00763] |
| INT16 | spa_Latn | hin_Deva | 0.00167 | [-0.00001, 0.00360] |
| INT16 | fra_Latn | eng_Latn | 0.05540 | [0.03762, 0.07505] |
| INT16 | fra_Latn | rus_Cyrl | 0.01180 | [0.00681, 0.01722] |
| INT16 | fra_Latn | zho_Hans | 0.00803 | [0.00359, 0.01309] |
| INT16 | fra_Latn | hin_Deva | 0.00507 | [0.00271, 0.00758] |
| INT16 | por_Latn | eng_Latn | 0.04396 | [0.03214, 0.05688] |
| INT16 | por_Latn | rus_Cyrl | 0.02677 | [0.02002, 0.03399] |
| INT16 | por_Latn | zho_Hans | 0.00234 | [0.00000, 0.00470] |
| INT16 | por_Latn | hin_Deva | 0.01246 | [0.00810, 0.01702] |
| INT16 | rus_Cyrl | eng_Latn | 0.00600 | [0.00330, 0.00881] |
| INT16 | rus_Cyrl | zho_Hans | 0.00249 | [-0.00050, 0.00548] |
| INT16 | rus_Cyrl | hin_Deva | 0.00116 | [-0.00069, 0.00316] |
| INT16 | zho_Hans | eng_Latn | 0.04500 | [0.02312, 0.07154] |
| INT16 | zho_Hans | rus_Cyrl | 0.04198 | [0.02546, 0.05982] |
| INT16 | zho_Hans | hin_Deva | 0.02374 | [0.01327, 0.03711] |
| INT16 | jpn_Jpan | eng_Latn | 0.04472 | [0.03524, 0.05425] |
| INT16 | jpn_Jpan | rus_Cyrl | 0.10581 | [0.09005, 0.12217] |
| INT16 | jpn_Jpan | zho_Hans | 0.03618 | [0.02645, 0.04569] |
| INT16 | jpn_Jpan | hin_Deva | 0.02698 | [0.02185, 0.03231] |
| INT16 | kor_Hang | eng_Latn | 0.00494 | [0.00110, 0.00976] |
| INT16 | kor_Hang | rus_Cyrl | 0.00168 | [-0.00087, 0.00405] |
| INT16 | kor_Hang | zho_Hans | 0.01077 | [0.00488, 0.01709] |
| INT16 | kor_Hang | hin_Deva | 0.01293 | [0.00889, 0.01694] |
| INT16 | hin_Deva | eng_Latn | -0.00087 | [-0.00370, 0.00224] |
| INT16 | hin_Deva | rus_Cyrl | 0.02240 | [0.01466, 0.03197] |
| INT16 | hin_Deva | zho_Hans | 0.03539 | [0.02684, 0.04430] |
| INT16 | ben_Beng | eng_Latn | 0.00402 | [0.00076, 0.00779] |
| INT16 | ben_Beng | rus_Cyrl | 0.00557 | [0.00172, 0.01012] |
| INT16 | ben_Beng | zho_Hans | 0.00474 | [0.00120, 0.00846] |
| INT16 | ben_Beng | hin_Deva | 0.01011 | [0.00682, 0.01334] |
| INT16 | tam_Taml | eng_Latn | 0.00480 | [0.00119, 0.00960] |
| INT16 | tam_Taml | rus_Cyrl | 0.00826 | [0.00376, 0.01361] |
| INT16 | tam_Taml | zho_Hans | 0.00404 | [0.00076, 0.00739] |
| INT16 | tam_Taml | hin_Deva | 0.01550 | [0.01125, 0.01983] |
| INT16 | tur_Latn | eng_Latn | 0.00180 | [-0.00085, 0.00418] |
| INT16 | tur_Latn | rus_Cyrl | 0.00350 | [0.00136, 0.00564] |
| INT16 | tur_Latn | zho_Hans | 0.00341 | [0.00004, 0.00683] |
| INT16 | tur_Latn | hin_Deva | 0.02043 | [0.01603, 0.02489] |
| INT16 | vie_Latn | eng_Latn | 0.00295 | [-0.00289, 0.00888] |
| INT16 | vie_Latn | rus_Cyrl | 0.00562 | [0.00250, 0.00890] |
| INT16 | vie_Latn | zho_Hans | 0.00758 | [0.00400, 0.01138] |
| INT16 | vie_Latn | hin_Deva | 0.00196 | [-0.00039, 0.00448] |
| INT16 | ind_Latn | eng_Latn | 0.04268 | [0.02534, 0.06229] |
| INT16 | ind_Latn | rus_Cyrl | 0.02796 | [0.01391, 0.04704] |
| INT16 | ind_Latn | zho_Hans | 0.01846 | [0.00612, 0.03510] |
| INT16 | ind_Latn | hin_Deva | 0.00906 | [0.00371, 0.01586] |
| INT16 | tha_Thai | eng_Latn | 0.00661 | [0.00279, 0.01117] |
| INT16 | tha_Thai | rus_Cyrl | 0.00259 | [0.00036, 0.00509] |
| INT16 | tha_Thai | zho_Hans | 0.00335 | [0.00075, 0.00592] |
| INT16 | tha_Thai | hin_Deva | 0.00083 | [-0.00086, 0.00246] |
| INT16 | pol_Latn | eng_Latn | 0.00276 | [-0.00032, 0.00691] |
| INT16 | pol_Latn | rus_Cyrl | 0.01966 | [0.01436, 0.02513] |
| INT16 | pol_Latn | zho_Hans | 0.00755 | [0.00437, 0.01056] |
| INT16 | pol_Latn | hin_Deva | 0.00777 | [0.00451, 0.01110] |
| INT16 | ukr_Cyrl | eng_Latn | 0.00411 | [-0.00006, 0.00902] |
| INT16 | ukr_Cyrl | rus_Cyrl | 0.40535 | [0.36033, 0.45295] |
| INT16 | ukr_Cyrl | zho_Hans | 0.00460 | [0.00181, 0.00737] |
| INT16 | ukr_Cyrl | hin_Deva | 0.01776 | [0.01276, 0.02291] |
| INT16 | kan_Knda | eng_Latn | 0.00259 | [0.00060, 0.00449] |
| INT16 | kan_Knda | rus_Cyrl | 0.01044 | [0.00512, 0.01601] |
| INT16 | kan_Knda | zho_Hans | 0.01075 | [0.00611, 0.01546] |
| INT16 | kan_Knda | hin_Deva | 0.01843 | [0.01281, 0.02474] |
| INT16 | tel_Telu | eng_Latn | 0.01214 | [0.00487, 0.02040] |
| INT16 | tel_Telu | rus_Cyrl | 0.12164 | [0.10255, 0.14141] |
| INT16 | tel_Telu | zho_Hans | 0.00849 | [0.00360, 0.01440] |
| INT16 | tel_Telu | hin_Deva | 0.05868 | [0.04956, 0.06796] |
| INT16 | yor_Latn | eng_Latn | 0.00540 | [0.00083, 0.01105] |
| INT16 | yor_Latn | rus_Cyrl | 0.04994 | [0.03701, 0.06398] |
| INT16 | yor_Latn | zho_Hans | 0.01754 | [0.01017, 0.02592] |
| INT16 | yor_Latn | hin_Deva | 0.05527 | [0.04642, 0.06444] |
| INT16 | sun_Latn | eng_Latn | 0.00417 | [0.00139, 0.00692] |
| INT16 | sun_Latn | rus_Cyrl | 0.00554 | [0.00176, 0.00942] |
| INT16 | sun_Latn | zho_Hans | 0.00329 | [-0.00057, 0.00733] |
| INT16 | sun_Latn | hin_Deva | 0.01232 | [0.00692, 0.01866] |
| INT16 | amh_Ethi | eng_Latn | 0.05643 | [0.03709, 0.07664] |
| INT16 | amh_Ethi | rus_Cyrl | 0.04889 | [0.03296, 0.06476] |
| INT16 | amh_Ethi | zho_Hans | 0.02540 | [0.01770, 0.03387] |
| INT16 | amh_Ethi | hin_Deva | 0.07100 | [0.05509, 0.08899] |
| INT16 | ibo_Latn | eng_Latn | 0.02304 | [0.01187, 0.03606] |
| INT16 | ibo_Latn | rus_Cyrl | 0.04271 | [0.03216, 0.05314] |
| INT16 | ibo_Latn | zho_Hans | 0.01125 | [0.00657, 0.01593] |
| INT16 | ibo_Latn | hin_Deva | 0.02090 | [0.01482, 0.02780] |
| INT16 | hau_Latn | eng_Latn | 0.01340 | [0.00605, 0.02198] |
| INT16 | hau_Latn | rus_Cyrl | 0.03317 | [0.02426, 0.04346] |
| INT16 | hau_Latn | zho_Hans | 0.02476 | [0.01645, 0.03412] |
| INT16 | hau_Latn | hin_Deva | 0.03523 | [0.02773, 0.04316] |
| INT16 | khm_Khmr | eng_Latn | 0.00575 | [0.00195, 0.01039] |
| INT16 | khm_Khmr | rus_Cyrl | 0.01699 | [0.01141, 0.02300] |
| INT16 | khm_Khmr | zho_Hans | 0.00947 | [0.00412, 0.01575] |
| INT16 | khm_Khmr | hin_Deva | 0.02787 | [0.02104, 0.03552] |
| INT16 | lao_Laoo | eng_Latn | 0.00609 | [0.00225, 0.01080] |
| INT16 | lao_Laoo | rus_Cyrl | 0.02847 | [0.02105, 0.03657] |
| INT16 | lao_Laoo | zho_Hans | 0.01309 | [0.00670, 0.02057] |
| INT16 | lao_Laoo | hin_Deva | 0.09154 | [0.07926, 0.10450] |
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
| INT8 | eng_Latn | rus_Cyrl | 0.00516 | [0.00104, 0.00910] |
| INT8 | eng_Latn | zho_Hans | 0.00520 | [-0.00000, 0.01062] |
| INT8 | eng_Latn | hin_Deva | -0.00017 | [-0.00254, 0.00233] |
| INT8 | deu_Latn | eng_Latn | 0.01132 | [0.00317, 0.02033] |
| INT8 | deu_Latn | rus_Cyrl | 0.01944 | [0.01385, 0.02610] |
| INT8 | deu_Latn | zho_Hans | 0.00951 | [0.00541, 0.01339] |
| INT8 | deu_Latn | hin_Deva | 0.00352 | [0.00143, 0.00583] |
| INT8 | spa_Latn | eng_Latn | 0.04831 | [0.03578, 0.06211] |
| INT8 | spa_Latn | rus_Cyrl | 0.00529 | [0.00145, 0.00934] |
| INT8 | spa_Latn | zho_Hans | 0.00358 | [0.00090, 0.00629] |
| INT8 | spa_Latn | hin_Deva | 0.00101 | [-0.00070, 0.00294] |
| INT8 | fra_Latn | eng_Latn | 0.05452 | [0.03694, 0.07463] |
| INT8 | fra_Latn | rus_Cyrl | 0.01169 | [0.00660, 0.01709] |
| INT8 | fra_Latn | zho_Hans | 0.00592 | [0.00173, 0.01073] |
| INT8 | fra_Latn | hin_Deva | 0.00482 | [0.00247, 0.00728] |
| INT8 | por_Latn | eng_Latn | 0.04416 | [0.03236, 0.05760] |
| INT8 | por_Latn | rus_Cyrl | 0.02734 | [0.02066, 0.03480] |
| INT8 | por_Latn | zho_Hans | 0.00254 | [0.00016, 0.00536] |
| INT8 | por_Latn | hin_Deva | 0.01162 | [0.00748, 0.01626] |
| INT8 | rus_Cyrl | eng_Latn | 0.00539 | [0.00255, 0.00850] |
| INT8 | rus_Cyrl | zho_Hans | 0.00357 | [0.00065, 0.00642] |
| INT8 | rus_Cyrl | hin_Deva | 0.00126 | [-0.00051, 0.00313] |
| INT8 | zho_Hans | eng_Latn | 0.04533 | [0.02370, 0.07203] |
| INT8 | zho_Hans | rus_Cyrl | 0.04306 | [0.02631, 0.06056] |
| INT8 | zho_Hans | hin_Deva | 0.02381 | [0.01296, 0.03719] |
| INT8 | jpn_Jpan | eng_Latn | 0.04455 | [0.03475, 0.05482] |
| INT8 | jpn_Jpan | rus_Cyrl | 0.10659 | [0.09043, 0.12318] |
| INT8 | jpn_Jpan | zho_Hans | 0.03461 | [0.02487, 0.04465] |
| INT8 | jpn_Jpan | hin_Deva | 0.02677 | [0.02127, 0.03248] |
| INT8 | kor_Hang | eng_Latn | 0.00511 | [0.00107, 0.01005] |
| INT8 | kor_Hang | rus_Cyrl | 0.00079 | [-0.00137, 0.00274] |
| INT8 | kor_Hang | zho_Hans | 0.01078 | [0.00488, 0.01721] |
| INT8 | kor_Hang | hin_Deva | 0.01243 | [0.00839, 0.01655] |
| INT8 | hin_Deva | eng_Latn | -0.00100 | [-0.00387, 0.00222] |
| INT8 | hin_Deva | rus_Cyrl | 0.02363 | [0.01580, 0.03321] |
| INT8 | hin_Deva | zho_Hans | 0.03379 | [0.02564, 0.04248] |
| INT8 | ben_Beng | eng_Latn | 0.00341 | [0.00016, 0.00689] |
| INT8 | ben_Beng | rus_Cyrl | 0.00606 | [0.00233, 0.01048] |
| INT8 | ben_Beng | zho_Hans | 0.00499 | [0.00141, 0.00895] |
| INT8 | ben_Beng | hin_Deva | 0.01003 | [0.00657, 0.01351] |
| INT8 | tam_Taml | eng_Latn | 0.00379 | [-0.00004, 0.00886] |
| INT8 | tam_Taml | rus_Cyrl | 0.00762 | [0.00305, 0.01321] |
| INT8 | tam_Taml | zho_Hans | 0.00421 | [0.00075, 0.00764] |
| INT8 | tam_Taml | hin_Deva | 0.01525 | [0.01103, 0.01956] |
| INT8 | tur_Latn | eng_Latn | 0.00169 | [-0.00095, 0.00417] |
| INT8 | tur_Latn | rus_Cyrl | 0.00323 | [0.00104, 0.00551] |
| INT8 | tur_Latn | zho_Hans | 0.00304 | [-0.00054, 0.00667] |
| INT8 | tur_Latn | hin_Deva | 0.02000 | [0.01562, 0.02448] |
| INT8 | vie_Latn | eng_Latn | 0.00380 | [-0.00208, 0.00990] |
| INT8 | vie_Latn | rus_Cyrl | 0.00542 | [0.00228, 0.00883] |
| INT8 | vie_Latn | zho_Hans | 0.00771 | [0.00355, 0.01170] |
| INT8 | vie_Latn | hin_Deva | 0.00246 | [-0.00001, 0.00499] |
| INT8 | ind_Latn | eng_Latn | 0.04236 | [0.02556, 0.06224] |
| INT8 | ind_Latn | rus_Cyrl | 0.02754 | [0.01358, 0.04651] |
| INT8 | ind_Latn | zho_Hans | 0.01825 | [0.00533, 0.03596] |
| INT8 | ind_Latn | hin_Deva | 0.00982 | [0.00460, 0.01637] |
| INT8 | tha_Thai | eng_Latn | 0.00673 | [0.00288, 0.01114] |
| INT8 | tha_Thai | rus_Cyrl | 0.00319 | [0.00098, 0.00551] |
| INT8 | tha_Thai | zho_Hans | 0.00379 | [0.00148, 0.00617] |
| INT8 | tha_Thai | hin_Deva | 0.00117 | [-0.00048, 0.00293] |
| INT8 | pol_Latn | eng_Latn | 0.00311 | [-0.00005, 0.00724] |
| INT8 | pol_Latn | rus_Cyrl | 0.01934 | [0.01403, 0.02483] |
| INT8 | pol_Latn | zho_Hans | 0.00560 | [0.00229, 0.00900] |
| INT8 | pol_Latn | hin_Deva | 0.00757 | [0.00425, 0.01110] |
| INT8 | ukr_Cyrl | eng_Latn | 0.00487 | [0.00095, 0.00940] |
| INT8 | ukr_Cyrl | rus_Cyrl | 0.42410 | [0.37781, 0.47344] |
| INT8 | ukr_Cyrl | zho_Hans | 0.00340 | [0.00039, 0.00666] |
| INT8 | ukr_Cyrl | hin_Deva | 0.01743 | [0.01276, 0.02228] |
| INT8 | kan_Knda | eng_Latn | 0.00317 | [0.00126, 0.00514] |
| INT8 | kan_Knda | rus_Cyrl | 0.01014 | [0.00485, 0.01566] |
| INT8 | kan_Knda | zho_Hans | 0.01033 | [0.00568, 0.01510] |
| INT8 | kan_Knda | hin_Deva | 0.01847 | [0.01278, 0.02483] |
| INT8 | tel_Telu | eng_Latn | 0.01183 | [0.00473, 0.02018] |
| INT8 | tel_Telu | rus_Cyrl | 0.12058 | [0.10156, 0.13978] |
| INT8 | tel_Telu | zho_Hans | 0.00874 | [0.00377, 0.01451] |
| INT8 | tel_Telu | hin_Deva | 0.05891 | [0.04973, 0.06815] |
| INT8 | yor_Latn | eng_Latn | 0.00562 | [0.00101, 0.01129] |
| INT8 | yor_Latn | rus_Cyrl | 0.04959 | [0.03704, 0.06323] |
| INT8 | yor_Latn | zho_Hans | 0.01671 | [0.00883, 0.02526] |
| INT8 | yor_Latn | hin_Deva | 0.05481 | [0.04595, 0.06367] |
| INT8 | sun_Latn | eng_Latn | 0.00368 | [0.00120, 0.00634] |
| INT8 | sun_Latn | rus_Cyrl | 0.00557 | [0.00193, 0.00955] |
| INT8 | sun_Latn | zho_Hans | 0.00264 | [-0.00129, 0.00672] |
| INT8 | sun_Latn | hin_Deva | 0.01262 | [0.00736, 0.01868] |
| INT8 | amh_Ethi | eng_Latn | 0.05597 | [0.03607, 0.07649] |
| INT8 | amh_Ethi | rus_Cyrl | 0.04970 | [0.03371, 0.06575] |
| INT8 | amh_Ethi | zho_Hans | 0.02594 | [0.01770, 0.03501] |
| INT8 | amh_Ethi | hin_Deva | 0.07129 | [0.05531, 0.08944] |
| INT8 | ibo_Latn | eng_Latn | 0.02226 | [0.01102, 0.03495] |
| INT8 | ibo_Latn | rus_Cyrl | 0.04314 | [0.03246, 0.05397] |
| INT8 | ibo_Latn | zho_Hans | 0.01109 | [0.00652, 0.01584] |
| INT8 | ibo_Latn | hin_Deva | 0.02191 | [0.01563, 0.02906] |
| INT8 | hau_Latn | eng_Latn | 0.01421 | [0.00684, 0.02287] |
| INT8 | hau_Latn | rus_Cyrl | 0.03296 | [0.02411, 0.04343] |
| INT8 | hau_Latn | zho_Hans | 0.02427 | [0.01576, 0.03394] |
| INT8 | hau_Latn | hin_Deva | 0.03651 | [0.02888, 0.04465] |
| INT8 | khm_Khmr | eng_Latn | 0.00583 | [0.00204, 0.01028] |
| INT8 | khm_Khmr | rus_Cyrl | 0.01663 | [0.01092, 0.02256] |
| INT8 | khm_Khmr | zho_Hans | 0.00853 | [0.00311, 0.01486] |
| INT8 | khm_Khmr | hin_Deva | 0.02889 | [0.02189, 0.03665] |
| INT8 | lao_Laoo | eng_Latn | 0.00531 | [0.00114, 0.01029] |
| INT8 | lao_Laoo | rus_Cyrl | 0.02747 | [0.02016, 0.03555] |
| INT8 | lao_Laoo | zho_Hans | 0.01234 | [0.00597, 0.01967] |
| INT8 | lao_Laoo | hin_Deva | 0.09132 | [0.07928, 0.10420] |
