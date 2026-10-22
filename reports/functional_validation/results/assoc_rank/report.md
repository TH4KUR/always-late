# Association retention by dev-mass rank within each language (A1)

**Exploratory and descriptive; nothing here is causal.** The statuses are the D-035 statuses (measured on devtest, features selected and ranked on dev). Established = the feature was classed `retained` in full precision (6,663 of 8,086 selective features); retained at a precision = still `retained` there. Pooled counts reproduce the D-035 headline (checked against its summary file).

**Hypothesis written before any number was computed (H-A1):** association is retained at least as well in the top 64 features of a language (the features that carry the function, ranked by dev mass) as in the rest, monotone in rank, i.e. marginal features lose their association first. A reversed or null result is reported as it is.

**Limitation that travels with the numbers:** the rank by dev mass is confounded with the activity level and the selectivity margin of a feature (high-mass features are less likely to fall below a support threshold in a quantized model), so this describes where association loss concentrates; it does not show that function protects association. The comparison at equal native activity is F4 (subgroup interventions, a later GPU job).

## Fraction retained among the features established in full precision, by rank bin (pooled over languages, Wilson 95 % intervals)

| precision | 1-16 | 17-64 | 65-256 | 257+ |
|---|---|---|---|---|
| INT16 | 0.995 [0.982, 0.999] (402/404) | 0.998 [0.994, 1.000] (1127/1129) | 0.990 [0.985, 0.993] (2902/2932) | 0.990 [0.985, 0.993] (2176/2198) |
| INT12 | 0.995 [0.982, 0.999] (402/404) | 0.996 [0.990, 0.998] (1124/1129) | 0.991 [0.987, 0.994] (2907/2932) | 0.989 [0.984, 0.993] (2174/2198) |
| INT8 | 0.993 [0.978, 0.997] (401/404) | 0.991 [0.984, 0.995] (1119/1129) | 0.981 [0.975, 0.985] (2876/2932) | 0.975 [0.968, 0.981] (2143/2198) |
| INT7 | 0.978 [0.958, 0.988] (395/404) | 0.981 [0.972, 0.988] (1108/1129) | 0.971 [0.964, 0.976] (2846/2932) | 0.976 [0.969, 0.982] (2146/2198) |
| INT6 | 0.980 [0.961, 0.990] (396/404) | 0.973 [0.962, 0.981] (1099/1129) | 0.938 [0.929, 0.946] (2751/2932) | 0.938 [0.927, 0.947] (2061/2198) |
| INT5 | 0.936 [0.907, 0.956] (378/404) | 0.931 [0.915, 0.944] (1051/1129) | 0.888 [0.877, 0.899] (2605/2932) | 0.866 [0.851, 0.879] (1903/2198) |
| INT4 | 0.854 [0.816, 0.885] (345/404) | 0.794 [0.769, 0.816] (896/1129) | 0.666 [0.649, 0.683] (1954/2932) | 0.693 [0.673, 0.712] (1523/2198) |

## Within-language contrast: fraction retained in the top 64 minus the rest

Languages with at least 3 established features in both parts; mean over languages with a percentile interval from 2000 bootstrap draws over languages (seed 0).

| precision | languages | mean difference | 95 % interval | top retains better | rest retains better |
|---|---|---|---|---|---|
| INT16 | 27 | +0.010 | [+0.006, +0.015] | 17 | 2 |
| INT12 | 27 | +0.012 | [+0.003, +0.024] | 16 | 5 |
| INT8 | 27 | +0.012 | [+0.003, +0.020] | 21 | 5 |
| INT7 | 27 | +0.017 | [+0.006, +0.029] | 19 | 7 |
| INT6 | 27 | +0.050 | [+0.033, +0.067] | 24 | 3 |
| INT5 | 27 | +0.069 | [+0.046, +0.092] | 23 | 4 |
| INT4 | 27 | +0.219 | [+0.180, +0.260] | 27 | 0 |
