# Stage 3 results (generated; D-117 survival measurement-validity study)

Rules: `reports/survival_measurement/analysis_freeze.json` (D-120, committed before the Stage 2 outputs existed). **Every threshold below (H3: 0.2 and 0.5; H4: the 0.20 bound; H5: the 10 % target) is a pre-set convention, not a claim: the numbers and their intervals are always shown next to the label (D-121).**

Committed comparison (G2(ii)): run.

## Gates: identity (r from g, sigma_a, sigma_e against r from the sums)

| cell | checked | excluded (degenerate) | largest difference |
|---|---|---|---|
| legacy|INT16 | 214499 | 7 | 5.11e-15 |
| legacy|INT6 | 214498 | 8 | 4.33e-15 |
| legacy|INT4 | 214267 | 239 | 4.55e-15 |
| matched|INT6 | 214502 | 4 | 5.44e-15 |
| matched|INT4 | 214391 | 115 | 4.11e-15 |
| gptq|INT6 | 214500 | 6 | 3.89e-15 |
| gptq|INT4 | 214499 | 7 | 3.66e-15 |

## FP side across jobs (D-123): cells whose n_a differs from the reference among the FP-active cells

| cell | cells with different n_a | largest difference | cells with different active status | max relative difference of Sa |
|---|---|---|---|---|
| legacy|INT16 | 0 | 0 | 0 | 0.00e+00 |
| legacy|INT6 | 0 | 0 | 0 | 0.00e+00 |
| legacy|INT4 | 0 | 0 | 0 | 0.00e+00 |
| matched|INT6 | 0 | 0 | 0 | 0.00e+00 |
| matched|INT4 | 0 | 0 | 0 | 0.00e+00 |
| gptq|INT6 | 0 | 0 | 0 | 0.00e+00 |
| gptq|INT4 | 0 | 0 | 0 | 0.00e+00 |

## Populations (FP-active cells; silent = no quantized activation; clipped lambda counted, never dropped)

| cell | active | silent | clipped low | clipped high |
|---|---|---|---|---|
| legacy|INT16 | 214506 | 0 | 0 | 30 |
| legacy|INT6 | 214506 | 3 | 3 | 5 |
| legacy|INT4 | 214506 | 239 | 577 | 0 |
| matched|INT6 | 214506 | 2 | 2 | 4 |
| matched|INT4 | 214506 | 114 | 211 | 1 |
| gptq|INT6 | 214506 | 1 | 1 | 6 |
| gptq|INT4 | 214506 | 5 | 8 | 2 |

## H3: does lambda follow the feature's scale, and nu not?

| cell | median Spearman(lambda, log sigma_a) | median Spearman(nu, log sigma_a) | rule (convention) |
|---|---|---|---|
| legacy|INT6 | -0.515 | 0.100 | met |
| legacy|INT4 | -0.507 | -0.004 | met |
| matched|INT6 | -0.509 | 0.105 | met |
| matched|INT4 | -0.524 | 0.003 | met |
| gptq|INT6 | -0.510 | 0.114 | met |
| gptq|INT4 | -0.534 | 0.073 | met |
| legacy|INT16 (reported) | -0.332 | 0.144 | - |

Overall: met in all six cells; failing cells none; not estimable none; missing none.

## H4: tier pattern in the encoder perturbation (outcome = minus the mean of log nu per language)

Primary s = z(width) - z(INT16) (quantization-specific part, standardized slopes); secondary z(width) against 0. Outcome words: present (95 % interval excludes 0 and |estimate| >= 0.20), absent (90 % interval inside +-0.20), otherwise inconclusive; inconclusive is an accepted result.

| cell | covariate | slope INT16 | slope width | primary s [95 %] | primary outcome | secondary z [95 %] | secondary outcome | dropped replicates |
|---|---|---|---|---|---|---|---|---|
| legacy|INT6 | none | 0.063 | 0.077 | -0.072 [-0.195, -0.004] | absent | 0.838 [0.472, 1.119] | present | 0 |
| legacy|INT6 | sigma_a | 0.038 | 0.038 | -0.139 [-0.402, 0.010] | inconclusive | 0.411 [0.035, 1.130] | present | 0 |
| legacy|INT4 | none | 0.063 | 0.084 | -0.098 [-0.230, -0.013] | inconclusive | 0.812 [0.443, 1.112] | present | 0 |
| legacy|INT4 | sigma_a | 0.038 | 0.055 | -0.015 [-0.299, 0.222] | inconclusive | 0.535 [0.065, 1.322] | present | 0 |
| matched|INT6 | none | 0.063 | 0.077 | -0.023 [-0.089, 0.016] | absent | 0.887 [0.582, 1.162] | present | 0 |
| matched|INT6 | sigma_a | 0.038 | 0.041 | -0.078 [-0.262, 0.038] | inconclusive | 0.471 [0.040, 1.204] | present | 0 |
| matched|INT4 | none | 0.063 | 0.085 | -0.060 [-0.184, 0.022] | absent | 0.850 [0.507, 1.150] | present | 0 |
| matched|INT4 | sigma_a | 0.038 | 0.044 | -0.112 [-0.380, 0.122] | inconclusive | 0.438 [-0.066, 1.206] | inconclusive | 0 |
| gptq|INT6 | none | 0.063 | 0.012 | -0.789 [-1.336, -0.346] | present | 0.121 [-0.593, 0.733] | inconclusive | 0 |
| gptq|INT6 | sigma_a | 0.038 | 0.075 | 0.233 [-0.137, 0.774] | inconclusive | 0.783 [0.124, 1.814] | present | 0 |
| gptq|INT4 | none | 0.063 | 0.002 | -0.895 [-1.390, -0.417] | present | 0.015 [-0.626, 0.651] | inconclusive | 0 |
| gptq|INT4 | sigma_a | 0.038 | 0.089 | 0.235 [-0.155, 0.751] | inconclusive | 0.785 [0.137, 1.776] | present | 0 |

## H5: do g and log nu add to model B for W1? (every improvement with its interval)

the 10 % relative improvement is the D-113 convention, not a claim: every improvement is reported with its interval and worded from the interval (D-121).

| cell | n | B over A | B over A on the full D-113 population (n) | B+ over B | B+ over A | g only over B | nu only over B | wording of B+ over B | meets the 10 % convention |
|---|---|---|---|---|---|---|---|---|---|
| legacy|INT6 | 1412 | 0.044 [0.026, 0.062] | 0.044 [0.026, 0.062] (n 1412) | -0.001 [-0.016, 0.014] | 0.044 [0.022, 0.065] | -0.003 [-0.007, 0.001] | 0.001 [-0.014, 0.015] | not distinguishable from zero (interval includes zero) | no |
| legacy|INT4 | 1412 | 0.037 [0.023, 0.052] | 0.037 [0.023, 0.052] (n 1412) | 0.018 [-0.003, 0.037] | 0.054 [0.034, 0.074] | 0.007 [-0.002, 0.016] | 0.019 [-0.001, 0.037] | not distinguishable from zero (interval includes zero) | no |
| matched|INT6 | 1412 | 0.038 [0.016, 0.058] | 0.038 [0.016, 0.058] (n 1412) | -0.004 [-0.016, 0.009] | 0.034 [0.009, 0.059] | -0.002 [-0.004, 0.001] | -0.003 [-0.015, 0.010] | not distinguishable from zero (interval includes zero) | no |
| matched|INT4 | 1412 | 0.031 [0.018, 0.044] | 0.031 [0.018, 0.044] (n 1412) | 0.010 [-0.005, 0.025] | 0.041 [0.026, 0.055] | 0.003 [-0.008, 0.014] | 0.011 [-0.004, 0.027] | not distinguishable from zero (interval includes zero) | no |
| gptq|INT6 | 1412 | 0.041 [0.019, 0.062] | 0.041 [0.019, 0.062] (n 1412) | -0.002 [-0.019, 0.012] | 0.038 [0.010, 0.065] | -0.001 [-0.001, -0.000] | -0.002 [-0.018, 0.013] | not distinguishable from zero (interval includes zero) | no |
| gptq|INT4 | 1412 | 0.035 [0.021, 0.050] | 0.035 [0.021, 0.050] (n 1412) | 0.007 [-0.015, 0.028] | 0.042 [0.013, 0.070] | 0.006 [-0.002, 0.015] | 0.008 [-0.013, 0.029] | not distinguishable from zero (interval includes zero) | no |

D-113 overall statement over the six cells: no clear predictive value beyond baselines.

## Descriptive tables

### Lambda decomposition between arms (arm minus legacy; lambda = log sigma_e^2 - 2 log|g| - 2 log sigma_a, exact)

| pair | cells | median delta lambda | median delta log sigma_e^2 | median -2 delta log abs g | identity error |
|---|---|---|---|---|---|
| matched_vs_legacy|INT6 | 214498 | -0.168 | -0.142 | -0.017 | 1.7e-10 |
| matched_vs_legacy|INT4 | 213805 | -0.441 | -0.185 | -0.204 | 9.2e-12 |
| gptq_vs_legacy|INT6 | 214498 | -0.515 | -0.456 | -0.036 | 3.3e-10 |
| gptq_vs_legacy|INT4 | 213926 | -0.982 | -0.274 | -0.622 | 2.0e-11 |

### log nu of an arm against legacy

| pair | cells | Spearman | SD-ratio slope | OLS slope | median difference |
|---|---|---|---|---|---|
| matched_vs_legacy|INT6 | 214506 | 0.810 | 0.951 | 0.788 | -0.141 |
| matched_vs_legacy|INT4 | 214506 | 0.831 | 0.979 | 0.835 | -0.223 |
| gptq_vs_legacy|INT6 | 214506 | 0.712 | 0.924 | 0.682 | -0.414 |
| gptq_vs_legacy|INT4 | 214506 | 0.659 | 0.865 | 0.598 | -0.608 |

### Per arm and width

| cell | median g | median log sigma_e^2 | median lambda | on-off median | off-on median | nu median [p25, p75] | silent share | predicted vs exact flips (Spearman) |
|---|---|---|---|---|---|---|---|---|
| legacy|INT16 | 0.997 | -6.808 | -4.779 | 0.0106 | 0.0000 | 0.018 [0.015, 0.021] | 0.0000 | 0.714 |
| legacy|INT6 | 0.958 | -4.685 | -2.447 | 0.0921 | 0.0004 | 0.130 [0.110, 0.157] | 0.0000 | 0.752 |
| legacy|INT4 | 0.594 | -3.352 | -0.025 | 0.5000 | 0.0009 | 0.706 [0.579, 0.856] | 0.0011 | 0.957 |
| matched|INT6 | 0.967 | -4.824 | -2.613 | 0.0769 | 0.0003 | 0.113 [0.096, 0.135] | 0.0000 | 0.709 |
| matched|INT4 | 0.672 | -3.534 | -0.461 | 0.4286 | 0.0007 | 0.562 [0.461, 0.683] | 0.0005 | 0.946 |
| gptq|INT6 | 0.976 | -5.143 | -2.956 | 0.0588 | 0.0003 | 0.086 [0.073, 0.101] | 0.0000 | 0.802 |
| gptq|INT4 | 0.823 | -3.580 | -0.996 | 0.2703 | 0.0010 | 0.378 [0.320, 0.444] | 0.0000 | 0.908 |

### Selectivity (secondary, descriptive): mean log nu of selective cells minus cells of globally non-selective features

Reading: a cell (language, feature) is selective iff the feature is selective and peaks in that language; cells of features selective for another language are in neither group.

| cell | languages | mean difference [95 %] |
|---|---|---|
| legacy|INT6 | 27 | -0.174 [-0.222, -0.129] |
| legacy|INT4 | 27 | -0.215 [-0.271, -0.160] |
| matched|INT6 | 27 | -0.149 [-0.194, -0.107] |
| matched|INT4 | 27 | -0.213 [-0.268, -0.161] |
| gptq|INT6 | 27 | -0.157 [-0.206, -0.112] |
| gptq|INT4 | 27 | -0.186 [-0.241, -0.135] |

## Limits

One model, one layer, one SAE, simulated weight-only quantizers, 27 languages (clusters). Single-language script families are absorbed by the script dummies, so the tier slope rests on about 23 languages. Post hoc relative to Stage 1 (same tokens and features), with the Stage 3 rules fixed before the new statistics existed. Nulls, negatives and inconclusive outcomes are results.
