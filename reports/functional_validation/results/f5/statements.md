# F5 statements: is the survival metric sensible? (exploratory)

Rule proposed before any number (design section 5, Q23); every number is reported whatever the verdict (`f5_tables.csv`, `f5_results.json`).

**Numerical floor.** W1 (cosine of the single-feature effect fingerprints) between full precision and the repeat measurement of the same weights (FP-rebatched): median n/a, 5th percentile n/a; between full precision and INT16 (weights rounded at the bf16 noise level): median 0.997, 5th percentile 0.949. The numerical floor used below is 0.9489 (the lower of the two, from INT16). INT8, a real quantization, is reported but not part of the floor: median 0.997. Instrument adequate: True.

| statement | value | 95 % interval (languages resampled) | defined | verdict |
|---|---|---|---|---|
| S1 INT6, retained by the D-035 status: share with W1 >= floor (rule: >= 0.80) | 0.902 | [0.874, 0.926] | 1240 | met |
| S2 INT4, F4 pairs: AUROC of the status for W1 >= floor (rule: >= 0.60, lower bound > 0.5) | 0.256 | [0.249, 0.264] | 2 event, 811 no event | undefined |
| S3 INT5, baseline cutoff (correlation >= 0.9): share with W1 >= floor (rule: >= 0.80) | 0.824 | [0.777, 0.863] | 833 | met |
| descriptive S2c (descriptive, no rule) INT4, F4 pairs: AUROC of W1 for the status retained (0.5 = no relation; below 0.5 the lost features keep more) | 0.506 | [0.469, 0.543] | 396 retained, 417 lost | none |

**Overall: undetermined.** Work dtype float32; sham gate exact True; logit gates ok True; run complete True.

If the three statements hold, they hold for this model, this layer, this quantizer and this sample, nothing more. If they do not hold, the survival label describes the input-side agreement of a coordinate system and no functional claim may be built on it. The study is exploratory: no confirmatory claim is made from it.
