# Functional effects of language feature groups (group size K64)

**Exploratory on already examined data** (FLORES devtest sentences; the features were selected on dev). Next-token loss only; no claim about task accuracy or meaning. E is the harm (nats per token, NLL after the edit minus NLL before it, same precision, same sentences) of zero-ablating the language's selective group beyond the mean harm of its activity-matched controls on the language's own text; positive = the selective group matters more than equally active non-selective features. Intervals: 2000 paired cluster-bootstrap draws over sentence ids (percentile, descriptive; the observed value is the estimate). Macro retention is the mean over languages of the per-language ratio E at a precision over E in full precision (D-041; languages whose full-precision interval lies above zero), classified preserved / attenuated / amplified against the bands [0.75, 1.33], inconclusive otherwise, undefined when the full-precision interval does not lie above zero.

Languages in the macro average: 27 (eng_Latn, deu_Latn, spa_Latn, fra_Latn, por_Latn, rus_Cyrl, zho_Hans, jpn_Jpan, kor_Hang, hin_Deva, ben_Beng, tam_Taml, tur_Latn, vie_Latn, ind_Latn, tha_Thai, pol_Latn, ukr_Cyrl, kan_Knda, tel_Telu, yor_Latn, sun_Latn, amh_Ethi, ibo_Latn, hau_Latn, khm_Khmr, lao_Laoo). Positions (sentences per language): 300.

## Macro effect and retention

| precision | mean E | 95 % interval | median E | languages with E > 0 | retention (mean of per-language ratios) | interval | class (languages) | median ratio [interval] | pooled ratio (dominated by large effects) |
|---|---|---|---|---|---|---|---|---|---|
| FP | 0.7037 | [0.6844, 0.7225] | 0.2624 | 27 of 27 | - | - | reference | - | - |
| INT4 | 0.9001 | [0.8784, 0.9195] | 0.3946 | 27 of 27 | 1.75 | [1.68, 1.99] | amplified (27) | 1.36 [1.33, 1.39] | 1.28 |
| INT6 | 0.7226 | [0.7038, 0.7414] | 0.2546 | 27 of 27 | 1.06 | [1.05, 1.10] | preserved (27) | 1.04 [1.03, 1.06] | 1.03 |

## Dose, size and replacement (median over languages of the harm in nats per token on the language's own text)

| family | FP | INT4 | INT6 |
|---|---|---|---|
| E_K16 | 0.2077 | nan | nan |
| E_K64 | 0.2624 | 0.3946 | 0.2546 |
| E_all | 0.2987 | nan | nan |
| ctrl_K16 | 0.0144 | nan | nan |
| ctrl_K64 | 0.0235 | 0.0298 | 0.0251 |
| ctrl_all | 0.0258 | nan | nan |
| res_K64 | 0.2936 | nan | nan |
| sham_K64 | 0.0000 | 0.0000 | 0.0000 |
| zero_K16 | 0.2425 | nan | nan |
| zero_K64 | 0.2862 | 0.4526 | 0.2797 |
| zero_all | 0.3466 | nan | nan |

Languages whose E (zero ablation beyond controls) has a 95 % lower bound above 0: FP: E_K16 26, E_K64 27, E_all 27; INT4: E_K64 27; INT6: E_K64 27


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
| INT6 | tam_Taml | 0.00000 | [0.00000, 0.00000] |
| INT6 | tur_Latn | 0.00000 | [0.00000, 0.00000] |
| INT6 | vie_Latn | 0.00000 | [0.00000, 0.00000] |
| INT6 | ind_Latn | 0.00000 | [0.00000, 0.00000] |
| INT6 | tha_Thai | 0.00000 | [0.00000, 0.00000] |
| INT6 | pol_Latn | 0.00000 | [0.00000, 0.00000] |
| INT6 | ukr_Cyrl | 0.00000 | [0.00000, 0.00000] |
| INT6 | kan_Knda | 0.00000 | [0.00000, 0.00000] |
| INT6 | tel_Telu | 0.00000 | [0.00000, 0.00000] |
| INT6 | yor_Latn | 0.00000 | [0.00000, 0.00000] |
| INT6 | sun_Latn | 0.00000 | [0.00000, 0.00000] |
| INT6 | amh_Ethi | 0.00000 | [0.00000, 0.00000] |
| INT6 | ibo_Latn | 0.00000 | [0.00000, 0.00000] |
| INT6 | hau_Latn | 0.00000 | [0.00000, 0.00000] |
| INT6 | khm_Khmr | 0.00000 | [0.00000, 0.00000] |
| INT6 | lao_Laoo | 0.00000 | [0.00000, 0.00000] |

## Effect by language

| precision | language | E | interval | group harm | control harm | removed norm group / control |
|---|---|---|---|---|---|---|
| FP | eng_Latn | 0.1728 | [0.1510, 0.1965] | 0.2088 | 0.0360 | 0.144 / 0.143 |
| FP | deu_Latn | 1.0733 | [0.9969, 1.1606] | 1.0865 | 0.0132 | 0.123 / 0.111 |
| FP | spa_Latn | 0.4941 | [0.4287, 0.5600] | 0.5139 | 0.0198 | 0.107 / 0.100 |
| FP | fra_Latn | 0.5835 | [0.5381, 0.6306] | 0.6095 | 0.0259 | 0.117 / 0.101 |
| FP | por_Latn | 1.0238 | [0.9385, 1.1068] | 1.0309 | 0.0071 | 0.076 / 0.064 |
| FP | rus_Cyrl | 0.0731 | [0.0607, 0.0865] | 0.0801 | 0.0070 | 0.080 / 0.079 |
| FP | zho_Hans | 7.5751 | [7.2060, 7.9375] | 7.5987 | 0.0235 | 0.180 / 0.154 |
| FP | jpn_Jpan | 0.2624 | [0.2400, 0.2859] | 0.2862 | 0.0238 | 0.141 / 0.140 |
| FP | kor_Hang | 0.3511 | [0.3058, 0.4046] | 0.3630 | 0.0119 | 0.097 / 0.091 |
| FP | hin_Deva | 0.1042 | [0.0878, 0.1220] | 0.1206 | 0.0163 | 0.130 / 0.130 |
| FP | ben_Beng | 0.0098 | [0.0068, 0.0130] | 0.0157 | 0.0058 | 0.055 / 0.048 |
| FP | tam_Taml | 0.0281 | [0.0238, 0.0325] | 0.0408 | 0.0127 | 0.076 / 0.072 |
| FP | tur_Latn | 0.2077 | [0.1877, 0.2294] | 0.2435 | 0.0359 | 0.126 / 0.121 |
| FP | vie_Latn | 0.0118 | [0.0038, 0.0198] | 0.0229 | 0.0111 | 0.084 / 0.083 |
| FP | ind_Latn | 0.0266 | [0.0131, 0.0410] | 0.0364 | 0.0098 | 0.049 / 0.051 |
| FP | tha_Thai | 0.0462 | [0.0397, 0.0527] | 0.0582 | 0.0121 | 0.109 / 0.131 |
| FP | pol_Latn | 0.3599 | [0.3185, 0.4030] | 0.3882 | 0.0283 | 0.103 / 0.092 |
| FP | ukr_Cyrl | 0.4191 | [0.3819, 0.4532] | 0.4292 | 0.0102 | 0.109 / 0.088 |
| FP | kan_Knda | 0.0341 | [0.0283, 0.0404] | 0.0522 | 0.0180 | 0.090 / 0.083 |
| FP | tel_Telu | 0.0284 | [0.0228, 0.0339] | 0.0543 | 0.0259 | 0.119 / 0.123 |
| FP | yor_Latn | 1.5781 | [1.5200, 1.6366] | 1.6397 | 0.0616 | 0.344 / 0.287 |
| FP | sun_Latn | 0.8728 | [0.8223, 0.9239] | 0.9715 | 0.0987 | 0.167 / 0.176 |
| FP | amh_Ethi | 1.0202 | [0.9976, 1.0452] | 1.0472 | 0.0270 | 0.237 / 0.162 |
| FP | ibo_Latn | 0.3458 | [0.3223, 0.3679] | 0.3914 | 0.0456 | 0.184 / 0.161 |
| FP | hau_Latn | 0.1625 | [0.1454, 0.1792] | 0.2153 | 0.0529 | 0.139 / 0.118 |
| FP | khm_Khmr | 0.2196 | [0.1996, 0.2411] | 0.2514 | 0.0319 | 0.174 / 0.164 |
| FP | lao_Laoo | 1.9150 | [1.8374, 1.9943] | 1.9405 | 0.0255 | 0.254 / 0.189 |
| INT4 | eng_Latn | 0.1778 | [0.1542, 0.2030] | 0.2368 | 0.0590 | 0.135 / 0.139 |
| INT4 | deu_Latn | 0.9782 | [0.9067, 1.0514] | 0.9972 | 0.0190 | 0.115 / 0.103 |
| INT4 | spa_Latn | 1.0527 | [0.9208, 1.1813] | 1.0783 | 0.0255 | 0.098 / 0.099 |
| INT4 | fra_Latn | 0.6446 | [0.5944, 0.6954] | 0.6764 | 0.0317 | 0.106 / 0.102 |
| INT4 | por_Latn | 0.6792 | [0.6228, 0.7361] | 0.6885 | 0.0093 | 0.071 / 0.054 |
| INT4 | rus_Cyrl | 0.1793 | [0.1559, 0.2039] | 0.1902 | 0.0109 | 0.068 / 0.077 |
| INT4 | zho_Hans | 10.2648 | [9.8966, 10.6127] | 10.2890 | 0.0243 | 0.176 / 0.149 |
| INT4 | jpn_Jpan | 0.5191 | [0.4788, 0.5609] | 0.5508 | 0.0317 | 0.132 / 0.136 |
| INT4 | kor_Hang | 0.4840 | [0.4270, 0.5440] | 0.5005 | 0.0165 | 0.089 / 0.084 |
| INT4 | hin_Deva | 0.2171 | [0.1890, 0.2467] | 0.2469 | 0.0298 | 0.122 / 0.125 |
| INT4 | ben_Beng | 0.0269 | [0.0226, 0.0309] | 0.0387 | 0.0118 | 0.050 / 0.041 |
| INT4 | tam_Taml | 0.0592 | [0.0531, 0.0653] | 0.0770 | 0.0178 | 0.069 / 0.065 |
| INT4 | tur_Latn | 0.4055 | [0.3772, 0.4367] | 0.4526 | 0.0472 | 0.111 / 0.119 |
| INT4 | vie_Latn | 0.0461 | [0.0355, 0.0564] | 0.0634 | 0.0173 | 0.074 / 0.077 |
| INT4 | ind_Latn | 0.0421 | [0.0240, 0.0614] | 0.0572 | 0.0151 | 0.045 / 0.045 |
| INT4 | tha_Thai | 0.1480 | [0.1368, 0.1593] | 0.1639 | 0.0158 | 0.099 / 0.127 |
| INT4 | pol_Latn | 0.3506 | [0.3221, 0.3788] | 0.3899 | 0.0393 | 0.095 / 0.085 |
| INT4 | ukr_Cyrl | 0.7955 | [0.7465, 0.8429] | 0.8110 | 0.0155 | 0.095 / 0.090 |
| INT4 | kan_Knda | 0.0399 | [0.0329, 0.0470] | 0.0681 | 0.0282 | 0.084 / 0.076 |
| INT4 | tel_Telu | 0.1235 | [0.1139, 0.1334] | 0.1597 | 0.0362 | 0.108 / 0.119 |
| INT4 | yor_Latn | 1.8105 | [1.7551, 1.8660] | 1.9095 | 0.0990 | 0.315 / 0.276 |
| INT4 | sun_Latn | 0.7102 | [0.6650, 0.7540] | 0.8346 | 0.1244 | 0.157 / 0.174 |
| INT4 | amh_Ethi | 1.3797 | [1.3419, 1.4176] | 1.4257 | 0.0461 | 0.227 / 0.157 |
| INT4 | ibo_Latn | 0.3946 | [0.3715, 0.4181] | 0.4554 | 0.0607 | 0.165 / 0.151 |
| INT4 | hau_Latn | 0.1957 | [0.1768, 0.2139] | 0.2702 | 0.0746 | 0.131 / 0.108 |
| INT4 | khm_Khmr | 0.2960 | [0.2691, 0.3236] | 0.3596 | 0.0635 | 0.162 / 0.158 |
| INT4 | lao_Laoo | 2.2819 | [2.2479, 2.3152] | 2.3442 | 0.0623 | 0.214 / 0.188 |
| INT6 | eng_Latn | 0.1661 | [0.1444, 0.1897] | 0.2082 | 0.0421 | 0.144 / 0.141 |
| INT6 | deu_Latn | 1.1504 | [1.0682, 1.2444] | 1.1645 | 0.0141 | 0.122 / 0.110 |
| INT6 | spa_Latn | 0.5354 | [0.4663, 0.6065] | 0.5553 | 0.0199 | 0.107 / 0.098 |
| INT6 | fra_Latn | 0.6590 | [0.6075, 0.7142] | 0.6850 | 0.0260 | 0.116 / 0.100 |
| INT6 | por_Latn | 1.1286 | [1.0399, 1.2148] | 1.1365 | 0.0080 | 0.076 / 0.063 |
| INT6 | rus_Cyrl | 0.0826 | [0.0694, 0.0964] | 0.0888 | 0.0062 | 0.078 / 0.078 |
| INT6 | zho_Hans | 7.6548 | [7.2991, 8.0129] | 7.6803 | 0.0255 | 0.179 / 0.153 |
| INT6 | jpn_Jpan | 0.2546 | [0.2329, 0.2775] | 0.2797 | 0.0251 | 0.139 / 0.139 |
| INT6 | kor_Hang | 0.2973 | [0.2579, 0.3427] | 0.3102 | 0.0128 | 0.096 / 0.089 |
| INT6 | hin_Deva | 0.1361 | [0.1174, 0.1562] | 0.1526 | 0.0165 | 0.129 / 0.129 |
| INT6 | ben_Beng | 0.0091 | [0.0059, 0.0121] | 0.0151 | 0.0060 | 0.055 / 0.048 |
| INT6 | tam_Taml | 0.0293 | [0.0249, 0.0337] | 0.0421 | 0.0127 | 0.076 / 0.071 |
| INT6 | tur_Latn | 0.2136 | [0.1934, 0.2352] | 0.2500 | 0.0364 | 0.127 / 0.120 |
| INT6 | vie_Latn | 0.0177 | [0.0092, 0.0263] | 0.0284 | 0.0108 | 0.084 / 0.082 |
| INT6 | ind_Latn | 0.0274 | [0.0143, 0.0413] | 0.0376 | 0.0102 | 0.049 / 0.051 |
| INT6 | tha_Thai | 0.0539 | [0.0475, 0.0606] | 0.0666 | 0.0127 | 0.107 / 0.131 |
| INT6 | pol_Latn | 0.3833 | [0.3397, 0.4287] | 0.4140 | 0.0306 | 0.102 / 0.091 |
| INT6 | ukr_Cyrl | 0.4442 | [0.4070, 0.4797] | 0.4533 | 0.0091 | 0.108 / 0.087 |
| INT6 | kan_Knda | 0.0327 | [0.0270, 0.0388] | 0.0510 | 0.0183 | 0.089 / 0.083 |
| INT6 | tel_Telu | 0.0335 | [0.0276, 0.0396] | 0.0595 | 0.0259 | 0.117 / 0.122 |
| INT6 | yor_Latn | 1.6075 | [1.5499, 1.6660] | 1.6660 | 0.0584 | 0.345 / 0.288 |
| INT6 | sun_Latn | 0.8071 | [0.7566, 0.8558] | 0.9084 | 0.1013 | 0.166 / 0.174 |
| INT6 | amh_Ethi | 1.0918 | [1.0672, 1.1202] | 1.1240 | 0.0321 | 0.239 / 0.159 |
| INT6 | ibo_Latn | 0.3398 | [0.3174, 0.3618] | 0.3854 | 0.0456 | 0.181 / 0.160 |
| INT6 | hau_Latn | 0.1691 | [0.1511, 0.1858] | 0.2250 | 0.0559 | 0.137 / 0.118 |
| INT6 | khm_Khmr | 0.2250 | [0.2053, 0.2463] | 0.2570 | 0.0320 | 0.172 / 0.163 |
| INT6 | lao_Laoo | 1.9589 | [1.8871, 2.0334] | 1.9867 | 0.0278 | 0.250 / 0.188 |

## Rescue: share of the quantization-induced loss recovered by patching full-precision activations (same tokens) into the quantized model

Fractions are given only where the gap (quantized minus full-precision NLL per token) is clearly above 0.02 nats; raw recovered amounts are in `rescue.csv`. This is a different question from the effects above: it says where the damage sits at layer 12, not that the quantized model natively kept the function.

| precision | part | macro fraction | 95 % interval | languages |
|---|---|---|---|---|
| INT4 | group | 0.023 | [0.021, 0.024] | 27 |
| INT4 | control | 0.010 | [0.009, 0.011] | 27 |
| INT4 | group_minus_control | 0.013 | [0.011, 0.015] | 27 |
| INT4 | all | 0.137 | [0.127, 0.146] | 27 |
| INT4 | error | -0.024 | [-0.032, -0.017] | 27 |
| INT4 | full | 0.352 | [0.343, 0.361] | 27 |
| INT6 | group | -0.001 | [-0.018, 0.016] | 4 |
| INT6 | control | 0.000 | [-0.010, 0.010] | 4 |
| INT6 | group_minus_control | -0.001 | [-0.018, 0.016] | 4 |
| INT6 | all | -0.122 | [-0.167, -0.083] | 4 |
| INT6 | error | -0.257 | [-0.298, -0.221] | 4 |
| INT6 | full | 0.273 | [0.244, 0.300] | 4 |

## Off-target harm of the group's zero-ablation on other languages' text (manipulation check)

| precision | group | text | harm | interval |
|---|---|---|---|---|
| FP | eng_Latn | hin_Deva | 0.00080 | [-0.00110, 0.00253] |
| FP | deu_Latn | eng_Latn | 0.01344 | [0.00698, 0.02086] |
| FP | deu_Latn | hin_Deva | 0.00490 | [0.00323, 0.00653] |
| FP | spa_Latn | eng_Latn | 0.05005 | [0.04028, 0.06095] |
| FP | spa_Latn | hin_Deva | 0.00289 | [0.00139, 0.00433] |
| FP | fra_Latn | eng_Latn | 0.03820 | [0.02734, 0.04971] |
| FP | fra_Latn | hin_Deva | 0.00596 | [0.00384, 0.00792] |
| FP | por_Latn | eng_Latn | 0.04682 | [0.03737, 0.05643] |
| FP | por_Latn | hin_Deva | 0.01797 | [0.01484, 0.02119] |
| FP | rus_Cyrl | eng_Latn | 0.00579 | [0.00280, 0.00888] |
| FP | rus_Cyrl | hin_Deva | 0.00300 | [0.00147, 0.00450] |
| FP | zho_Hans | eng_Latn | 0.03569 | [0.02217, 0.05076] |
| FP | zho_Hans | hin_Deva | 0.01841 | [0.01260, 0.02495] |
| FP | jpn_Jpan | eng_Latn | 0.05022 | [0.04155, 0.05878] |
| FP | jpn_Jpan | hin_Deva | 0.02529 | [0.02126, 0.02955] |
| FP | kor_Hang | eng_Latn | 0.00402 | [0.00155, 0.00660] |
| FP | kor_Hang | hin_Deva | 0.01053 | [0.00768, 0.01362] |
| FP | hin_Deva | eng_Latn | 0.00348 | [0.00028, 0.00752] |
| FP | ben_Beng | eng_Latn | 0.00632 | [0.00274, 0.01033] |
| FP | ben_Beng | hin_Deva | 0.01280 | [0.00979, 0.01587] |
| FP | tam_Taml | eng_Latn | 0.00603 | [0.00224, 0.01054] |
| FP | tam_Taml | hin_Deva | 0.01819 | [0.01486, 0.02176] |
| FP | tur_Latn | eng_Latn | 0.00154 | [-0.00005, 0.00312] |
| FP | tur_Latn | hin_Deva | 0.01895 | [0.01567, 0.02243] |
| FP | vie_Latn | eng_Latn | 0.00662 | [0.00297, 0.01046] |
| FP | vie_Latn | hin_Deva | 0.00260 | [0.00126, 0.00396] |
| FP | ind_Latn | eng_Latn | 0.05193 | [0.03549, 0.06887] |
| FP | ind_Latn | hin_Deva | 0.01547 | [0.00952, 0.02201] |
| FP | tha_Thai | eng_Latn | 0.00381 | [0.00142, 0.00646] |
| FP | tha_Thai | hin_Deva | 0.00276 | [0.00167, 0.00387] |
| FP | pol_Latn | eng_Latn | 0.00057 | [-0.00202, 0.00376] |
| FP | pol_Latn | hin_Deva | 0.01059 | [0.00749, 0.01365] |
| FP | ukr_Cyrl | eng_Latn | 0.00514 | [0.00149, 0.00955] |
| FP | ukr_Cyrl | hin_Deva | 0.01605 | [0.01308, 0.01912] |
| FP | kan_Knda | eng_Latn | 0.00369 | [0.00133, 0.00629] |
| FP | kan_Knda | hin_Deva | 0.02196 | [0.01837, 0.02606] |
| FP | tel_Telu | eng_Latn | 0.01615 | [0.01035, 0.02275] |
| FP | tel_Telu | hin_Deva | 0.06332 | [0.05740, 0.07015] |
| FP | yor_Latn | eng_Latn | 0.01407 | [0.00715, 0.02288] |
| FP | yor_Latn | hin_Deva | 0.05684 | [0.04977, 0.06438] |
| FP | sun_Latn | eng_Latn | 0.00221 | [-0.00004, 0.00445] |
| FP | sun_Latn | hin_Deva | 0.01248 | [0.00908, 0.01603] |
| FP | amh_Ethi | eng_Latn | 0.05959 | [0.04508, 0.07553] |
| FP | amh_Ethi | hin_Deva | 0.07733 | [0.06522, 0.09017] |
| FP | ibo_Latn | eng_Latn | 0.02518 | [0.01655, 0.03494] |
| FP | ibo_Latn | hin_Deva | 0.02151 | [0.01715, 0.02613] |
| FP | hau_Latn | eng_Latn | 0.01337 | [0.00883, 0.01824] |
| FP | hau_Latn | hin_Deva | 0.03153 | [0.02694, 0.03643] |
| FP | khm_Khmr | eng_Latn | 0.00375 | [0.00083, 0.00718] |
| FP | khm_Khmr | hin_Deva | 0.04377 | [0.03874, 0.04895] |
| FP | lao_Laoo | eng_Latn | 0.00963 | [0.00461, 0.01663] |
| FP | lao_Laoo | hin_Deva | 0.09076 | [0.08215, 0.09893] |
