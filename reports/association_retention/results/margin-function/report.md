# A3: does the function of a feature depend on the margin that predicts the loss of its association?

**Exploratory and descriptive; nothing here is causal.** Function = the F5 write-side similarity W1 (cosine of the effect fingerprint in the quantized state with the one in full precision); the function event is W1 below the numerical floor. Association = the D-035 status. Margin m0, scale a0 and share s0 are read from the full-precision row of the D-035 per-feature file `feature_status_eq.csv` (sha256 `3ab979852f3007987bfd364d334d9199e0cb2c3064d57cc932e287e70aec6a71`). **The F5 sample is stratified by role (retained and lost at each precision, at most 12 features per language cell): class shares are not the population's; AUROCs and correlations are not affected by the shares, a pooled loss fraction would be.** m0 was measured on devtest like the statuses.

## Verdicts of the hypotheses written before any number (plan A3)

- **H-A3a** (|Spearman(m0, W1)| below 0.15 at INT6, INT5, INT4): **met** (INT6 0.109, INT5 0.099, INT4 0.097).
- **H-A3b** (AUROC for the association loss at INT6 at least 0.70; AUROC for the function event at most 0.60 at INT6 and INT5): **met** (association INT6 0.864; function INT6 0.577, INT5 0.525).
- **H-A3c** (the two AUROCs differ by at least 0.15 at INT6 and INT5): **met** (differences INT6 0.287, INT5 0.241).

### All languages

Tested features 1487, joined with an established D-035 feature 1449, dropped 38; numerical floor tau = 0.9489 (5th percentile of W1 at INT16 over all tested features); languages: 27.

Within-language correlation and AUROCs (mean over the languages with at least 20 features, and 3 of each class for an AUROC; percentile interval over languages, 2000 draws). The AUROCs score -m0, so 0.5 = no information:

| precision | features | function events (W1 < tau) | Spearman(m0, W1) | AUROC of -m0, association lost | AUROC of -m0, function event |
|---|---|---|---|---|---|
| INT8 | 1449 | 88 (0.061) | 0.097 [0.046, 0.147] (26) | 0.947 [0.923, 0.970] (8) | 0.612 [0.560, 0.661] (13) |
| INT6 | 1449 | 151 (0.104) | 0.109 [0.056, 0.165] (26) | 0.864 [0.837, 0.887] (20) | 0.577 [0.527, 0.630] (20) |
| INT5 | 1449 | 483 (0.333) | 0.099 [0.041, 0.158] (26) | 0.766 [0.740, 0.790] (25) | 0.525 [0.489, 0.564] (26) |
| INT4 | 1449 | 1444 (0.997) | 0.097 [0.038, 0.154] (26) | 0.736 [0.705, 0.764] (26) | n/a [n/a, n/a] (0) |

OLS of W1 on z(log m0), z(log a0), z(s0) with a fixed effect per language (change of W1 per standard deviation; cluster-robust 95 % interval by language):

| precision | m0 | scale a0 | share s0 | features |
|---|---|---|---|---|
| INT8 | 0.0013 [-0.0003, 0.0029] | 0.0049 [0.0026, 0.0072] | -0.0038 [-0.0062, -0.0015] | 1449 |
| INT6 | 0.0009 [-0.0028, 0.0046] | 0.0063 [0.0025, 0.0102] | -0.0043 [-0.0078, -0.0009] | 1449 |
| INT5 | 0.0023 [-0.0027, 0.0073] | 0.0066 [0.0017, 0.0116] | -0.0070 [-0.0127, -0.0013] | 1449 |
| INT4 | 0.0095 [-0.0002, 0.0192] | 0.0120 [0.0029, 0.0212] | -0.0248 [-0.0381, -0.0116] | 1449 |

### Cohort without eng_Latn

Tested features 1487, joined with an established D-035 feature 1379 (without eng_Latn), dropped 38; numerical floor tau = 0.9489 (5th percentile of W1 at INT16 over all tested features); languages: 26.

Within-language correlation and AUROCs (mean over the languages with at least 20 features, and 3 of each class for an AUROC; percentile interval over languages, 2000 draws). The AUROCs score -m0, so 0.5 = no information:

| precision | features | function events (W1 < tau) | Spearman(m0, W1) | AUROC of -m0, association lost | AUROC of -m0, function event |
|---|---|---|---|---|---|
| INT8 | 1379 | 84 (0.061) | 0.093 [0.043, 0.142] (25) | 0.947 [0.923, 0.970] (8) | 0.601 [0.552, 0.652] (12) |
| INT6 | 1379 | 144 (0.104) | 0.108 [0.052, 0.163] (25) | 0.868 [0.842, 0.891] (19) | 0.573 [0.522, 0.629] (19) |
| INT5 | 1379 | 449 (0.326) | 0.099 [0.040, 0.158] (25) | 0.766 [0.740, 0.789] (24) | 0.527 [0.487, 0.566] (25) |
| INT4 | 1379 | 1374 (0.996) | 0.094 [0.036, 0.152] (25) | 0.735 [0.703, 0.766] (25) | n/a [n/a, n/a] (0) |

OLS of W1 on z(log m0), z(log a0), z(s0) with a fixed effect per language (change of W1 per standard deviation; cluster-robust 95 % interval by language):

| precision | m0 | scale a0 | share s0 | features |
|---|---|---|---|---|
| INT8 | 0.0013 [-0.0004, 0.0030] | 0.0049 [0.0025, 0.0073] | -0.0040 [-0.0065, -0.0015] | 1379 |
| INT6 | 0.0011 [-0.0027, 0.0049] | 0.0060 [0.0020, 0.0099] | -0.0047 [-0.0082, -0.0012] | 1379 |
| INT5 | 0.0024 [-0.0028, 0.0076] | 0.0064 [0.0013, 0.0116] | -0.0071 [-0.0130, -0.0011] | 1379 |
| INT4 | 0.0101 [0.0001, 0.0201] | 0.0110 [0.0016, 0.0205] | -0.0260 [-0.0395, -0.0125] | 1379 |

