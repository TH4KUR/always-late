# Study G, extension D: does the optimal gain follow the quantization shrinkage?

Post hoc in origin (suggested by the result of extension C, D-082), exploratory, no new job; rules fixed before the numbers (spec section 12, D-083). Data: INT4, 562 reserve sentences per language, 27 languages; alpha = the ratio among 1, 1.1, 1.2, 1.4, 1.7 with the lowest group harm; target 1 / g_ratio (baseline G1, INT4; range 1.07 to 1.35).

## Verdict

| rule | statistic | result | verdict |
|---|---|---|---|
| R-D1 rank correlation of alpha with 1 / g_ratio | rho >= 0.30, one-sided permutation p < 0.05 (10000 permutations, seed 0), reliability >= 0.30 | rho = -0.170, p = 0.8036; split-half reliability 0.917 | **not supported** |
| R-D2 the same with the smooth estimator | descriptive | rho = -0.151, p = 0.7758 | |
| R-D3 placebo target 1 / controls_ratio | descriptive | rho = 0.186; difference (group target minus control target) -0.356 [-0.952, +0.284] (95 %) (5000 bootstrap draws over languages, 0 resamples left out) | |
| R-D4 split-half (even and odd sentence positions) | descriptive | correlation between halves 0.846; Spearman-Brown reliability 0.917; corrected R-D1 correlation -0.178 | |

## Per language

| language | 1 / g | 1 / controls | alpha (grid) | alpha (smooth) | alpha, even half | alpha, odd half |
|---|---|---|---|---|---|---|
| eng_Latn | 1.294 | 1.119 | 1 | 1.000 | 1.2 | 1 |
| deu_Latn | 1.174 | 1.158 | 1 | 1.007 | 1 | 1 |
| spa_Latn | 1.166 | 1.161 | 1.1 | 1.076 | 1.1 | 1.2 |
| fra_Latn | 1.214 | 1.148 | 1.2 | 1.193 | 1.2 | 1.1 |
| por_Latn | 1.084 | 1.148 | 1.4 | 1.265 | 1.2 | 1.4 |
| rus_Cyrl | 1.283 | 1.149 | 1.1 | 1.151 | 1.1 | 1.1 |
| zho_Hans | 1.147 | 1.153 | 1 | 1.040 | 1 | 1.1 |
| jpn_Jpan | 1.202 | 1.169 | 1 | 1.060 | 1 | 1.1 |
| kor_Hang | 1.131 | 1.175 | 1.4 | 1.400 | 1.4 | 1.4 |
| hin_Deva | 1.142 | 1.179 | 1.1 | 1.049 | 1.1 | 1.1 |
| ben_Beng | 1.203 | 1.199 | 1.4 | 1.466 | 1.4 | 1.4 |
| tam_Taml | 1.180 | 1.200 | 1.4 | 1.285 | 1.2 | 1.4 |
| tur_Latn | 1.242 | 1.173 | 1.7 | 1.568 | 1.7 | 1.7 |
| vie_Latn | 1.213 | 1.166 | 1 | 1.018 | 1 | 1 |
| ind_Latn | 1.162 | 1.172 | 1.1 | 1.124 | 1.1 | 1.2 |
| tha_Thai | 1.192 | 1.176 | 1.4 | 1.326 | 1.4 | 1.4 |
| pol_Latn | 1.137 | 1.151 | 1.2 | 1.173 | 1.2 | 1.2 |
| ukr_Cyrl | 1.191 | 1.150 | 1.2 | 1.157 | 1.1 | 1.2 |
| kan_Knda | 1.167 | 1.205 | 1.1 | 1.077 | 1.1 | 1.1 |
| tel_Telu | 1.070 | 1.202 | 1.4 | 1.327 | 1.4 | 1.4 |
| yor_Latn | 1.126 | 1.110 | 1.2 | 1.237 | 1.2 | 1.2 |
| sun_Latn | 1.081 | 1.208 | 1.7 | 1.700 | 1.7 | 1.7 |
| amh_Ethi | 1.096 | 1.199 | 1 | 1.030 | 1 | 1 |
| ibo_Latn | 1.178 | 1.161 | 1.4 | 1.351 | 1.4 | 1.4 |
| hau_Latn | 1.251 | 1.216 | 1.2 | 1.205 | 1.2 | 1.2 |
| khm_Khmr | 1.144 | 1.203 | 1 | 1.027 | 1 | 1 |
| lao_Laoo | 1.347 | 1.188 | 1.1 | 1.103 | 1.1 | 1.1 |

## Limits

- Post hoc: the question came from the extension C result; the rules were written before this analysis was run, but the data are the same data. A language-level correlation over 27 points; alpha lies on a grid of four ratios.
- 'uninformative' (reliability below 0.30) is not a refutation; 'not supported' with a reliable estimator is a failure to find the relation at this resolution, not proof of its absence (the spread of 1 / g is about 0.06 in standard deviation).
- g_ratio is measured on the selection-independent text of the baseline, not on the evaluation sentences; one model, layer, SAE and quantizer.
