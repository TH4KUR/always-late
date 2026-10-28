# Functional effects of language feature groups (group size K64)

**Exploratory on already examined data** (FLORES devtest sentences; the features were selected on dev). Next-token loss only; no claim about task accuracy or meaning. E is the harm (nats per token, NLL after the edit minus NLL before it, same precision, same sentences) of zero-ablating the language's selective group beyond the mean harm of its activity-matched controls on the language's own text; positive = the selective group matters more than equally active non-selective features. Intervals: 2000 paired cluster-bootstrap draws over sentence ids (percentile, descriptive; the observed value is the estimate). Macro retention is the mean over languages of the per-language ratio E at a precision over E in full precision (D-041; languages whose full-precision interval lies above zero), classified preserved / attenuated / amplified against the bands [0.75, 1.33], inconclusive otherwise, undefined when the full-precision interval does not lie above zero.

Languages in the macro average: 27 (eng_Latn, deu_Latn, spa_Latn, fra_Latn, por_Latn, rus_Cyrl, zho_Hans, jpn_Jpan, kor_Hang, hin_Deva, ben_Beng, tam_Taml, tur_Latn, vie_Latn, ind_Latn, tha_Thai, pol_Latn, ukr_Cyrl, kan_Knda, tel_Telu, yor_Latn, sun_Latn, amh_Ethi, ibo_Latn, hau_Latn, khm_Khmr, lao_Laoo). Positions (sentences per language): 300.

## Macro effect and retention

| precision | mean E | 95 % interval | median E | languages with E > 0 | retention (mean of per-language ratios) | interval | class (languages) | median ratio [interval] | pooled ratio (dominated by large effects) |
|---|---|---|---|---|---|---|---|---|---|
| FP | 0.7037 | [0.6844, 0.7225] | 0.2624 | 27 of 27 | - | - | reference | - | - |
| INT4 | 0.7793 | [0.7597, 0.7986] | 0.3345 | 27 of 27 | 1.31 | [1.27, 1.50] | inconclusive (27) | 1.22 [1.14, 1.24] | 1.11 |
| INT6 | 0.6898 | [0.6710, 0.7083] | 0.2635 | 27 of 27 | 0.99 | [0.98, 1.00] | preserved (27) | 1.00 [0.99, 1.00] | 0.98 |

## Dose, size and replacement (median over languages of the harm in nats per token on the language's own text)

| family | FP | INT4 | INT6 |
|---|---|---|---|
| E_K16 | 0.2077 | nan | nan |
| E_K64 | 0.2624 | 0.3345 | 0.2635 |
| E_all | 0.2987 | nan | nan |
| ctrl_K16 | 0.0144 | nan | nan |
| ctrl_K64 | 0.0235 | 0.0268 | 0.0243 |
| ctrl_all | 0.0258 | nan | nan |
| res_K64 | 0.2936 | nan | nan |
| sham_K64 | 0.0000 | 0.0000 | 0.0000 |
| zero_K16 | 0.2425 | nan | nan |
| zero_K64 | 0.2862 | 0.3613 | 0.2878 |
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
| INT4 | eng_Latn | 0.1541 | [0.1333, 0.1767] | 0.2015 | 0.0473 | 0.137 / 0.143 |
| INT4 | deu_Latn | 1.0793 | [1.0054, 1.1606] | 1.0944 | 0.0151 | 0.121 / 0.109 |
| INT4 | spa_Latn | 0.6233 | [0.5368, 0.7104] | 0.6424 | 0.0191 | 0.104 / 0.100 |
| INT4 | fra_Latn | 0.5891 | [0.5393, 0.6408] | 0.6162 | 0.0271 | 0.114 / 0.102 |
| INT4 | por_Latn | 0.9423 | [0.8682, 1.0196] | 0.9513 | 0.0089 | 0.075 / 0.060 |
| INT4 | rus_Cyrl | 0.1125 | [0.0953, 0.1309] | 0.1194 | 0.0069 | 0.077 / 0.077 |
| INT4 | zho_Hans | 8.1377 | [7.7727, 8.4949] | 8.1654 | 0.0277 | 0.177 / 0.154 |
| INT4 | jpn_Jpan | 0.3345 | [0.3087, 0.3627] | 0.3613 | 0.0268 | 0.140 / 0.141 |
| INT4 | kor_Hang | 0.5984 | [0.5254, 0.6763] | 0.6130 | 0.0145 | 0.095 / 0.090 |
| INT4 | hin_Deva | 0.1634 | [0.1406, 0.1877] | 0.1818 | 0.0185 | 0.128 / 0.129 |
| INT4 | ben_Beng | 0.0140 | [0.0107, 0.0173] | 0.0226 | 0.0086 | 0.055 / 0.048 |
| INT4 | tam_Taml | 0.0344 | [0.0294, 0.0395] | 0.0490 | 0.0146 | 0.075 / 0.071 |
| INT4 | tur_Latn | 0.2769 | [0.2557, 0.3009] | 0.3194 | 0.0424 | 0.123 / 0.121 |
| INT4 | vie_Latn | 0.0407 | [0.0307, 0.0505] | 0.0548 | 0.0141 | 0.083 / 0.082 |
| INT4 | ind_Latn | 0.0340 | [0.0192, 0.0502] | 0.0461 | 0.0120 | 0.049 / 0.049 |
| INT4 | tha_Thai | 0.0757 | [0.0672, 0.0841] | 0.0905 | 0.0148 | 0.106 / 0.130 |
| INT4 | pol_Latn | 0.3388 | [0.3047, 0.3750] | 0.3734 | 0.0346 | 0.101 / 0.090 |
| INT4 | ukr_Cyrl | 0.6262 | [0.5863, 0.6664] | 0.6381 | 0.0119 | 0.105 / 0.088 |
| INT4 | kan_Knda | 0.0354 | [0.0292, 0.0422] | 0.0572 | 0.0218 | 0.089 / 0.082 |
| INT4 | tel_Telu | 0.0508 | [0.0431, 0.0587] | 0.0793 | 0.0285 | 0.115 / 0.122 |
| INT4 | yor_Latn | 1.7770 | [1.7169, 1.8367] | 1.8486 | 0.0716 | 0.341 / 0.284 |
| INT4 | sun_Latn | 0.7819 | [0.7344, 0.8306] | 0.8893 | 0.1074 | 0.166 / 0.176 |
| INT4 | amh_Ethi | 1.2523 | [1.2228, 1.2850] | 1.2839 | 0.0316 | 0.234 / 0.161 |
| INT4 | ibo_Latn | 0.3521 | [0.3292, 0.3754] | 0.4050 | 0.0528 | 0.182 / 0.162 |
| INT4 | hau_Latn | 0.1730 | [0.1537, 0.1911] | 0.2394 | 0.0664 | 0.138 / 0.116 |
| INT4 | khm_Khmr | 0.2390 | [0.2186, 0.2610] | 0.2833 | 0.0443 | 0.171 / 0.162 |
| INT4 | lao_Laoo | 2.2043 | [2.1249, 2.2854] | 2.2334 | 0.0291 | 0.250 / 0.187 |
| INT6 | eng_Latn | 0.1688 | [0.1472, 0.1925] | 0.2061 | 0.0373 | 0.143 / 0.143 |
| INT6 | deu_Latn | 1.0137 | [0.9393, 1.0989] | 1.0269 | 0.0132 | 0.122 / 0.111 |
| INT6 | spa_Latn | 0.4507 | [0.3916, 0.5109] | 0.4699 | 0.0192 | 0.106 / 0.100 |
| INT6 | fra_Latn | 0.5868 | [0.5426, 0.6345] | 0.6132 | 0.0264 | 0.117 / 0.101 |
| INT6 | por_Latn | 0.9885 | [0.9068, 1.0704] | 0.9955 | 0.0070 | 0.076 / 0.064 |
| INT6 | rus_Cyrl | 0.0684 | [0.0565, 0.0812] | 0.0745 | 0.0061 | 0.079 / 0.079 |
| INT6 | zho_Hans | 7.3971 | [7.0403, 7.7565] | 7.4213 | 0.0243 | 0.180 / 0.154 |
| INT6 | jpn_Jpan | 0.2635 | [0.2411, 0.2871] | 0.2878 | 0.0243 | 0.140 / 0.140 |
| INT6 | kor_Hang | 0.3411 | [0.2983, 0.3906] | 0.3545 | 0.0134 | 0.097 / 0.091 |
| INT6 | hin_Deva | 0.1049 | [0.0885, 0.1222] | 0.1206 | 0.0157 | 0.130 / 0.130 |
| INT6 | ben_Beng | 0.0108 | [0.0076, 0.0138] | 0.0174 | 0.0067 | 0.055 / 0.049 |
| INT6 | tam_Taml | 0.0295 | [0.0251, 0.0340] | 0.0422 | 0.0127 | 0.076 / 0.072 |
| INT6 | tur_Latn | 0.2034 | [0.1839, 0.2254] | 0.2391 | 0.0357 | 0.126 / 0.121 |
| INT6 | vie_Latn | 0.0123 | [0.0041, 0.0206] | 0.0237 | 0.0114 | 0.084 / 0.083 |
| INT6 | ind_Latn | 0.0248 | [0.0124, 0.0382] | 0.0352 | 0.0104 | 0.049 / 0.051 |
| INT6 | tha_Thai | 0.0466 | [0.0400, 0.0532] | 0.0590 | 0.0124 | 0.108 / 0.131 |
| INT6 | pol_Latn | 0.3521 | [0.3114, 0.3936] | 0.3804 | 0.0283 | 0.102 / 0.091 |
| INT6 | ukr_Cyrl | 0.4162 | [0.3796, 0.4514] | 0.4263 | 0.0101 | 0.109 / 0.088 |
| INT6 | kan_Knda | 0.0344 | [0.0286, 0.0407] | 0.0520 | 0.0176 | 0.090 / 0.084 |
| INT6 | tel_Telu | 0.0291 | [0.0235, 0.0348] | 0.0554 | 0.0263 | 0.119 / 0.123 |
| INT6 | yor_Latn | 1.5706 | [1.5123, 1.6281] | 1.6322 | 0.0616 | 0.344 / 0.288 |
| INT6 | sun_Latn | 0.8724 | [0.8212, 0.9239] | 0.9701 | 0.0977 | 0.167 / 0.176 |
| INT6 | amh_Ethi | 1.0292 | [1.0050, 1.0558] | 1.0563 | 0.0271 | 0.237 / 0.161 |
| INT6 | ibo_Latn | 0.3456 | [0.3226, 0.3672] | 0.3918 | 0.0462 | 0.184 / 0.160 |
| INT6 | hau_Latn | 0.1632 | [0.1453, 0.1798] | 0.2155 | 0.0523 | 0.139 / 0.119 |
| INT6 | khm_Khmr | 0.2168 | [0.1973, 0.2376] | 0.2469 | 0.0301 | 0.173 / 0.165 |
| INT6 | lao_Laoo | 1.8842 | [1.8060, 1.9644] | 1.9095 | 0.0253 | 0.254 / 0.188 |

## Rescue: share of the quantization-induced loss recovered by patching full-precision activations (same tokens) into the quantized model

Fractions are given only where the gap (quantized minus full-precision NLL per token) is clearly above 0.02 nats; raw recovered amounts are in `rescue.csv`. This is a different question from the effects above: it says where the damage sits at layer 12, not that the quantized model natively kept the function.

| precision | part | macro fraction | 95 % interval | languages |
|---|---|---|---|---|
| INT4 | group | 0.020 | [0.016, 0.023] | 27 |
| INT4 | control | 0.011 | [0.009, 0.013] | 27 |
| INT4 | group_minus_control | 0.009 | [0.005, 0.012] | 27 |
| INT4 | all | 0.108 | [0.094, 0.119] | 27 |
| INT4 | error | -0.044 | [-0.056, -0.033] | 27 |
| INT4 | full | 0.414 | [0.402, 0.424] | 27 |

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
