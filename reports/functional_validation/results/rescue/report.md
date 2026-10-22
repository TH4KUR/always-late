# Functional effects of language feature groups (group size K64)

**Exploratory on already examined data** (FLORES devtest sentences; the features were selected on dev). Next-token loss only; no claim about task accuracy or meaning. E is the harm (nats per token, NLL after the edit minus NLL before it, same precision, same sentences) of zero-ablating the language's selective group beyond the mean harm of its activity-matched controls on the language's own text; positive = the selective group matters more than equally active non-selective features. Intervals: 2000 paired cluster-bootstrap draws over sentence ids (percentile, descriptive; the observed value is the estimate). Macro retention is the mean over languages of the per-language ratio E at a precision over E in full precision (D-041; languages whose full-precision interval lies above zero), classified preserved / attenuated / amplified against the bands [0.75, 1.33], inconclusive otherwise, undefined when the full-precision interval does not lie above zero.

Languages in the macro average: 27 (eng_Latn, deu_Latn, spa_Latn, fra_Latn, por_Latn, rus_Cyrl, zho_Hans, jpn_Jpan, kor_Hang, hin_Deva, ben_Beng, tam_Taml, tur_Latn, vie_Latn, ind_Latn, tha_Thai, pol_Latn, ukr_Cyrl, kan_Knda, tel_Telu, yor_Latn, sun_Latn, amh_Ethi, ibo_Latn, hau_Latn, khm_Khmr, lao_Laoo). Positions (sentences per language): 300.

## Macro effect and retention

| precision | mean E | 95 % interval | median E | languages with E > 0 | retention (mean of per-language ratios) | interval | class (languages) | median ratio [interval] | pooled ratio (dominated by large effects) |
|---|---|---|---|---|---|---|---|---|---|
| FP | 0.7037 | [0.6844, 0.7225] | 0.2624 | 27 of 27 | - | - | reference | - | - |
| INT4 | 0.9346 | [0.9144, 0.9526] | 0.4316 | 27 of 27 | 2.14 | [2.04, 2.50] | amplified (27) | 1.54 [1.43, 1.62] | 1.33 |
| INT5 | 0.7642 | [0.7448, 0.7835] | 0.2808 | 27 of 27 | 1.13 | [1.11, 1.18] | preserved (27) | 1.09 [1.07, 1.12] | 1.09 |
| INT6 | 0.7356 | [0.7165, 0.7548] | 0.2709 | 27 of 27 | 1.04 | [1.02, 1.05] | preserved (27) | 1.03 [1.02, 1.04] | 1.05 |

## Dose, size and replacement (median over languages of the harm in nats per token on the language's own text)

| family | FP | INT4 | INT5 | INT6 |
|---|---|---|---|---|
| E_K16 | 0.2077 | nan | nan | nan |
| E_K64 | 0.2624 | 0.4316 | 0.2808 | 0.2709 |
| E_all | 0.2987 | nan | nan | nan |
| ctrl_K16 | 0.0144 | nan | nan | nan |
| ctrl_K64 | 0.0235 | 0.0272 | 0.0265 | 0.0234 |
| ctrl_all | 0.0258 | nan | nan | nan |
| res_K64 | 0.2936 | nan | nan | nan |
| sham_K64 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| zero_K16 | 0.2425 | nan | nan | nan |
| zero_K64 | 0.2862 | 0.4930 | 0.3273 | 0.2953 |
| zero_all | 0.3466 | nan | nan | nan |

Languages whose E (zero ablation beyond controls) has a 95 % lower bound above 0: FP: E_K16 26, E_K64 27, E_all 27; INT4: E_K64 27; INT5: E_K64 27; INT6: E_K64 27


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
| INT5 | eng_Latn | 0.00000 | [0.00000, 0.00000] |
| INT5 | deu_Latn | 0.00000 | [0.00000, 0.00000] |
| INT5 | spa_Latn | 0.00000 | [0.00000, 0.00000] |
| INT5 | fra_Latn | 0.00000 | [0.00000, 0.00000] |
| INT5 | por_Latn | 0.00000 | [0.00000, 0.00000] |
| INT5 | rus_Cyrl | 0.00000 | [0.00000, 0.00000] |
| INT5 | zho_Hans | 0.00000 | [0.00000, 0.00000] |
| INT5 | jpn_Jpan | 0.00000 | [0.00000, 0.00000] |
| INT5 | kor_Hang | 0.00000 | [0.00000, 0.00000] |
| INT5 | hin_Deva | 0.00000 | [0.00000, 0.00000] |
| INT5 | ben_Beng | 0.00000 | [0.00000, 0.00000] |
| INT5 | tam_Taml | 0.00000 | [0.00000, 0.00000] |
| INT5 | tur_Latn | 0.00000 | [0.00000, 0.00000] |
| INT5 | vie_Latn | 0.00000 | [0.00000, 0.00000] |
| INT5 | ind_Latn | 0.00000 | [0.00000, 0.00000] |
| INT5 | tha_Thai | 0.00000 | [0.00000, 0.00000] |
| INT5 | pol_Latn | 0.00000 | [0.00000, 0.00000] |
| INT5 | ukr_Cyrl | 0.00000 | [0.00000, 0.00000] |
| INT5 | kan_Knda | 0.00000 | [0.00000, 0.00000] |
| INT5 | tel_Telu | 0.00000 | [0.00000, 0.00000] |
| INT5 | yor_Latn | 0.00000 | [0.00000, 0.00000] |
| INT5 | sun_Latn | 0.00000 | [0.00000, 0.00000] |
| INT5 | amh_Ethi | 0.00000 | [0.00000, 0.00000] |
| INT5 | ibo_Latn | 0.00000 | [0.00000, 0.00000] |
| INT5 | hau_Latn | 0.00000 | [0.00000, 0.00000] |
| INT5 | khm_Khmr | 0.00000 | [0.00000, 0.00000] |
| INT5 | lao_Laoo | 0.00000 | [0.00000, 0.00000] |
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
| INT4 | eng_Latn | 0.1672 | [0.1429, 0.1925] | 0.2318 | 0.0647 | 0.122 / 0.137 |
| INT4 | deu_Latn | 0.7704 | [0.7126, 0.8307] | 0.7972 | 0.0268 | 0.110 / 0.101 |
| INT4 | spa_Latn | 0.6329 | [0.5607, 0.7034] | 0.6563 | 0.0233 | 0.097 / 0.101 |
| INT4 | fra_Latn | 0.8975 | [0.8303, 0.9683] | 0.9341 | 0.0366 | 0.101 / 0.104 |
| INT4 | por_Latn | 0.9725 | [0.8989, 1.0462] | 0.9816 | 0.0091 | 0.073 / 0.053 |
| INT4 | rus_Cyrl | 0.2132 | [0.1878, 0.2397] | 0.2247 | 0.0116 | 0.065 / 0.075 |
| INT4 | zho_Hans | 8.2183 | [7.8884, 8.5408] | 8.2499 | 0.0316 | 0.169 / 0.146 |
| INT4 | jpn_Jpan | 0.3731 | [0.3435, 0.4031] | 0.4120 | 0.0388 | 0.125 / 0.138 |
| INT4 | kor_Hang | 0.9249 | [0.8252, 1.0273] | 0.9447 | 0.0198 | 0.089 / 0.086 |
| INT4 | hin_Deva | 0.3025 | [0.2731, 0.3348] | 0.3278 | 0.0253 | 0.121 / 0.125 |
| INT4 | ben_Beng | 0.0383 | [0.0339, 0.0426] | 0.0625 | 0.0242 | 0.048 / 0.040 |
| INT4 | tam_Taml | 0.0739 | [0.0676, 0.0807] | 0.0944 | 0.0205 | 0.067 / 0.066 |
| INT4 | tur_Latn | 0.5716 | [0.5292, 0.6212] | 0.6231 | 0.0515 | 0.107 / 0.121 |
| INT4 | vie_Latn | 0.0673 | [0.0559, 0.0790] | 0.0907 | 0.0234 | 0.073 / 0.078 |
| INT4 | ind_Latn | 0.0279 | [0.0144, 0.0419] | 0.0420 | 0.0141 | 0.045 / 0.044 |
| INT4 | tha_Thai | 0.1725 | [0.1601, 0.1848] | 0.1873 | 0.0148 | 0.095 / 0.123 |
| INT4 | pol_Latn | 0.5695 | [0.5214, 0.6161] | 0.6080 | 0.0385 | 0.096 / 0.085 |
| INT4 | ukr_Cyrl | 0.9658 | [0.9012, 1.0325] | 0.9829 | 0.0171 | 0.095 / 0.089 |
| INT4 | kan_Knda | 0.0442 | [0.0369, 0.0513] | 0.0745 | 0.0304 | 0.081 / 0.073 |
| INT4 | tel_Telu | 0.1633 | [0.1517, 0.1758] | 0.1975 | 0.0342 | 0.112 / 0.122 |
| INT4 | yor_Latn | 2.1856 | [2.1159, 2.2540] | 2.3180 | 0.1324 | 0.310 / 0.266 |
| INT4 | sun_Latn | 1.0290 | [0.9757, 1.0851] | 1.1390 | 0.1100 | 0.157 / 0.169 |
| INT4 | amh_Ethi | 2.3602 | [2.3039, 2.4166] | 2.3874 | 0.0272 | 0.220 / 0.157 |
| INT4 | ibo_Latn | 0.4316 | [0.4106, 0.4537] | 0.4930 | 0.0614 | 0.160 / 0.143 |
| INT4 | hau_Latn | 0.2090 | [0.1920, 0.2262] | 0.2828 | 0.0738 | 0.125 / 0.104 |
| INT4 | khm_Khmr | 0.4155 | [0.3902, 0.4415] | 0.4843 | 0.0688 | 0.157 / 0.154 |
| INT4 | lao_Laoo | 2.4377 | [2.3907, 2.4864] | 2.4462 | 0.0085 | 0.196 / 0.177 |
| INT5 | eng_Latn | 0.1755 | [0.1538, 0.1988] | 0.2153 | 0.0397 | 0.143 / 0.142 |
| INT5 | deu_Latn | 0.9809 | [0.9049, 1.0631] | 0.9954 | 0.0145 | 0.121 / 0.107 |
| INT5 | spa_Latn | 0.5763 | [0.5060, 0.6503] | 0.5989 | 0.0226 | 0.107 / 0.098 |
| INT5 | fra_Latn | 0.5515 | [0.5092, 0.5947] | 0.5814 | 0.0300 | 0.115 / 0.101 |
| INT5 | por_Latn | 1.2476 | [1.1577, 1.3398] | 1.2546 | 0.0070 | 0.075 / 0.058 |
| INT5 | rus_Cyrl | 0.0743 | [0.0620, 0.0875] | 0.0828 | 0.0085 | 0.075 / 0.079 |
| INT5 | zho_Hans | 7.2771 | [6.9301, 7.6240] | 7.3036 | 0.0265 | 0.176 / 0.154 |
| INT5 | jpn_Jpan | 0.2804 | [0.2571, 0.3044] | 0.3081 | 0.0277 | 0.139 / 0.144 |
| INT5 | kor_Hang | 0.3638 | [0.3188, 0.4152] | 0.3797 | 0.0158 | 0.093 / 0.092 |
| INT5 | hin_Deva | 0.0952 | [0.0778, 0.1142] | 0.1126 | 0.0174 | 0.125 / 0.127 |
| INT5 | ben_Beng | 0.0124 | [0.0090, 0.0158] | 0.0221 | 0.0096 | 0.056 / 0.049 |
| INT5 | tam_Taml | 0.0384 | [0.0336, 0.0431] | 0.0519 | 0.0135 | 0.077 / 0.072 |
| INT5 | tur_Latn | 0.2268 | [0.2045, 0.2513] | 0.2662 | 0.0394 | 0.125 / 0.119 |
| INT5 | vie_Latn | 0.0149 | [0.0064, 0.0235] | 0.0280 | 0.0131 | 0.083 / 0.082 |
| INT5 | ind_Latn | 0.0368 | [0.0213, 0.0525] | 0.0479 | 0.0111 | 0.052 / 0.051 |
| INT5 | tha_Thai | 0.0622 | [0.0548, 0.0695] | 0.0755 | 0.0133 | 0.106 / 0.131 |
| INT5 | pol_Latn | 0.4142 | [0.3682, 0.4595] | 0.4452 | 0.0310 | 0.100 / 0.091 |
| INT5 | ukr_Cyrl | 0.3716 | [0.3410, 0.4024] | 0.3833 | 0.0118 | 0.105 / 0.087 |
| INT5 | kan_Knda | 0.0343 | [0.0281, 0.0409] | 0.0552 | 0.0209 | 0.088 / 0.083 |
| INT5 | tel_Telu | 0.0308 | [0.0242, 0.0373] | 0.0636 | 0.0328 | 0.117 / 0.123 |
| INT5 | yor_Latn | 2.0156 | [1.9495, 2.0836] | 2.0918 | 0.0762 | 0.345 / 0.279 |
| INT5 | sun_Latn | 0.9298 | [0.8764, 0.9825] | 1.0277 | 0.0978 | 0.169 / 0.175 |
| INT5 | amh_Ethi | 1.1354 | [1.1098, 1.1636] | 1.1886 | 0.0532 | 0.233 / 0.160 |
| INT5 | ibo_Latn | 0.3644 | [0.3420, 0.3866] | 0.4155 | 0.0511 | 0.186 / 0.157 |
| INT5 | hau_Latn | 0.1810 | [0.1608, 0.1994] | 0.2371 | 0.0561 | 0.139 / 0.116 |
| INT5 | khm_Khmr | 0.2808 | [0.2617, 0.3006] | 0.3273 | 0.0465 | 0.170 / 0.162 |
| INT5 | lao_Laoo | 2.8618 | [2.7712, 2.9532] | 2.9088 | 0.0471 | 0.252 / 0.185 |
| INT6 | eng_Latn | 0.1654 | [0.1436, 0.1891] | 0.2033 | 0.0379 | 0.144 / 0.140 |
| INT6 | deu_Latn | 1.1664 | [1.0803, 1.2617] | 1.1810 | 0.0145 | 0.121 / 0.110 |
| INT6 | spa_Latn | 0.5140 | [0.4462, 0.5812] | 0.5339 | 0.0200 | 0.107 / 0.098 |
| INT6 | fra_Latn | 0.6757 | [0.6188, 0.7350] | 0.7007 | 0.0250 | 0.116 / 0.100 |
| INT6 | por_Latn | 1.1210 | [1.0318, 1.2075] | 1.1289 | 0.0078 | 0.076 / 0.063 |
| INT6 | rus_Cyrl | 0.0792 | [0.0661, 0.0933] | 0.0857 | 0.0065 | 0.079 / 0.078 |
| INT6 | zho_Hans | 7.9948 | [7.6357, 8.3550] | 8.0190 | 0.0242 | 0.179 / 0.152 |
| INT6 | jpn_Jpan | 0.2709 | [0.2480, 0.2960] | 0.2953 | 0.0244 | 0.139 / 0.136 |
| INT6 | kor_Hang | 0.3097 | [0.2700, 0.3564] | 0.3225 | 0.0127 | 0.095 / 0.089 |
| INT6 | hin_Deva | 0.1418 | [0.1224, 0.1613] | 0.1575 | 0.0158 | 0.128 / 0.128 |
| INT6 | ben_Beng | 0.0092 | [0.0060, 0.0124] | 0.0168 | 0.0075 | 0.055 / 0.048 |
| INT6 | tam_Taml | 0.0285 | [0.0243, 0.0329] | 0.0416 | 0.0131 | 0.074 / 0.071 |
| INT6 | tur_Latn | 0.1989 | [0.1792, 0.2211] | 0.2362 | 0.0373 | 0.126 / 0.120 |
| INT6 | vie_Latn | 0.0098 | [0.0015, 0.0179] | 0.0209 | 0.0112 | 0.084 / 0.082 |
| INT6 | ind_Latn | 0.0264 | [0.0134, 0.0401] | 0.0371 | 0.0108 | 0.048 / 0.051 |
| INT6 | tha_Thai | 0.0514 | [0.0449, 0.0579] | 0.0654 | 0.0140 | 0.107 / 0.130 |
| INT6 | pol_Latn | 0.3303 | [0.2915, 0.3713] | 0.3612 | 0.0309 | 0.101 / 0.090 |
| INT6 | ukr_Cyrl | 0.4818 | [0.4428, 0.5202] | 0.4925 | 0.0107 | 0.107 / 0.087 |
| INT6 | kan_Knda | 0.0343 | [0.0283, 0.0405] | 0.0522 | 0.0179 | 0.090 / 0.083 |
| INT6 | tel_Telu | 0.0354 | [0.0294, 0.0416] | 0.0587 | 0.0234 | 0.117 / 0.121 |
| INT6 | yor_Latn | 1.6159 | [1.5571, 1.6757] | 1.6742 | 0.0583 | 0.342 / 0.285 |
| INT6 | sun_Latn | 0.7898 | [0.7432, 0.8376] | 0.8917 | 0.1019 | 0.165 / 0.172 |
| INT6 | amh_Ethi | 1.1694 | [1.1424, 1.1997] | 1.1971 | 0.0277 | 0.236 / 0.158 |
| INT6 | ibo_Latn | 0.3395 | [0.3170, 0.3610] | 0.3872 | 0.0476 | 0.180 / 0.157 |
| INT6 | hau_Latn | 0.1681 | [0.1504, 0.1854] | 0.2247 | 0.0566 | 0.136 / 0.117 |
| INT6 | khm_Khmr | 0.2263 | [0.2064, 0.2476] | 0.2550 | 0.0286 | 0.171 / 0.161 |
| INT6 | lao_Laoo | 1.9074 | [1.8382, 1.9788] | 1.9322 | 0.0248 | 0.249 / 0.183 |

## Rescue: share of the quantization-induced loss recovered by patching full-precision activations (same tokens) into the quantized model

Fractions are given only where the gap (quantized minus full-precision NLL per token) is clearly above 0.02 nats; raw recovered amounts are in `rescue.csv`. This is a different question from the effects above: it says where the damage sits at layer 12, not that the quantized model natively kept the function.

| precision | part | macro fraction | 95 % interval | languages |
|---|---|---|---|---|
| INT4 | group | 0.026 | [0.024, 0.028] | 27 |
| INT4 | control | 0.012 | [0.011, 0.013] | 27 |
| INT4 | group_minus_control | 0.014 | [0.012, 0.016] | 27 |
| INT4 | all | 0.174 | [0.166, 0.182] | 27 |
| INT4 | error | 0.011 | [0.005, 0.017] | 27 |
| INT4 | full | 0.393 | [0.385, 0.400] | 27 |
| INT5 | group | 0.001 | [-0.003, 0.004] | 27 |
| INT5 | control | 0.004 | [0.001, 0.007] | 27 |
| INT5 | group_minus_control | -0.003 | [-0.008, 0.001] | 27 |
| INT5 | all | -0.002 | [-0.025, 0.017] | 27 |
| INT5 | error | -0.062 | [-0.079, -0.046] | 27 |
| INT5 | full | 0.297 | [0.278, 0.315] | 27 |
| INT6 | group | -0.001 | [-0.011, 0.008] | 12 |
| INT6 | control | -0.003 | [-0.011, 0.004] | 12 |
| INT6 | group_minus_control | 0.002 | [-0.008, 0.012] | 12 |
| INT6 | all | -0.063 | [-0.105, -0.029] | 12 |
| INT6 | error | -0.164 | [-0.197, -0.134] | 12 |
| INT6 | full | 0.350 | [0.322, 0.376] | 12 |

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
