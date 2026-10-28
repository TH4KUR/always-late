# F5 report: does a feature do the same thing after quantization? (exploratory)

## 1. Instrument and numerical floor

1487 sample features, 1487 tested on the write side, 0 too rare to test (read side only). Work dtype float32; sham ablation exact: True; logit gates ok: True; run complete: True. Stratified subset: a per-cell cap of 12 features per language and role (seed 0).

**Numerical floor.** W1 (cosine of the single-feature effect fingerprints) between full precision and the repeat measurement of the same weights (FP-rebatched): median n/a, 5th percentile n/a; between full precision and INT16 (weights rounded at the bf16 noise level): median 0.997, 5th percentile 0.949. The numerical floor used below is 0.9489 (the lower of the two, from INT16). INT8, a real quantization, is reported but not part of the floor: median n/a. Instrument adequate: True.

## 2. Main table (all tested features, by state)

R1 token correlation of the activations with full precision (the survival basis); R2 firing-set Jaccard; R3 scale ratio; W1 cosine of the single-feature effect fingerprints; the share at or above the numerical floor and above the 95th percentile of the matched different-feature control; 'does nothing' = the feature has no effect in the quantized model; top-1 = retrieval of the feature's own identity (within language, chance in brackets). Intervals: 95 %, languages resampled.

| state | tested | R1 median | R1 >= 0.9 | R2 | R3 | W1 median | W1 >= floor | W1 >= control | does nothing | top-1 | within top-1 (chance) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| INT16 | 1487 | 0.995 [0.995, 0.996] | 0.999 | 0.973 | 1.000 | 0.997 [0.997, 0.998] | 0.950 [0.933, 0.966] | 1.000 | 0 (0.000) | 1.000 | 1.000 (0.018) |
| INT4 | 1487 | 0.868 [0.859, 0.878] | 0.341 | 0.597 | 0.983 | 0.923 [0.919, 0.928] | 0.117 [0.099, 0.136] | 1.000 | 0 (0.000) | 1.000 | 1.000 (0.018) |
| INT6 | 1487 | 0.976 [0.974, 0.978] | 0.991 | 0.883 | 1.000 | 0.992 [0.991, 0.994] | 0.924 [0.902, 0.946] | 1.000 | 0 (0.000) | 1.000 | 1.000 (0.018) |

## 3. Features whose association was lost and retained (F4 subsets), by precision

| group | tested | W1 median | W1 >= floor | R1 median | R1 >= 0.9 |
| --- | --- | --- | --- | --- | --- |
| lost@INT6 | 189 | 0.992 [0.989, 0.994] | 0.884 [0.839, 0.926] | 0.970 | 0.989 |
| retained@INT6 | 208 | 0.990 [0.988, 0.992] | 0.918 [0.885, 0.950] | 0.967 | 0.981 |
| lost@INT4 | 417 | 0.920 [0.913, 0.927] | 0.101 [0.063, 0.144] | 0.866 | 0.338 |
| retained@INT4 | 396 | 0.923 [0.918, 0.929] | 0.093 [0.067, 0.124] | 0.861 | 0.306 |

## 4. Statements: is the survival metric sensible?

| statement | value | 95 % interval (languages resampled) | defined | verdict |
|---|---|---|---|---|
| S1 INT6, retained by the D-035 status: share with W1 >= floor (rule: >= 0.80) | 0.930 | [0.908, 0.952] | 1240 | met |
| S2 INT4, F4 pairs: AUROC of the status for W1 >= floor (rule: >= 0.60, lower bound > 0.5) | 0.490 | [0.431, 0.560] | 79 event, 734 no event | not met |
| S3 INT5, baseline cutoff (correlation >= 0.9): share with W1 >= floor (rule: >= 0.80) | n/a |  | 0 | undefined |
| descriptive S2c (descriptive, no rule) INT4, F4 pairs: AUROC of W1 for the status retained (0.5 = no relation; below 0.5 the lost features keep more) | 0.513 | [0.477, 0.548] | 396 retained, 417 lost | none |

**Overall: not supported.** Work dtype float32; sham gate exact True; logit gates ok True; run complete True.

If the three statements hold, they hold for this model, this layer, this quantizer and this sample, nothing more. If they do not hold, the survival label describes the input-side agreement of a coordinate system and no functional claim may be built on it. The study is exploratory: no confirmatory claim is made from it.

## 5. Descriptive hypotheses (H5a to H5c)

- H5a (the median token correlation is at least 0.9 at INT8 and INT6 and lower at INT4 than at INT6): None; medians INT6 0.976, INT4 0.868.
- H5b (the median W1 is below the median token correlation at every quantized precision): False; median W1 INT6 0.992, INT4 0.923; share below the floor INT6 0.076, INT4 0.883.
- H5c (retained features keep more of their effect than lost ones, paired by language): signs INT6 inconclusive, INT4 inconclusive; medians INT6 -0.001, INT4 0.002.

## 6. Limits

- This study is exploratory: no confirmatory claim is made from it. The rule S1 to S3 was fixed before the first number (Q23); the sample, the contexts and the analysis are one design among several.
- One model (Gemma-2-2B), one layer (the layer-12 residual), one SAE (Gemma Scope, 16,384 features), one quantizer (round-to-nearest per output channel). The ablation is a single-feature zero ablation: a test of the necessity of the feature's term, not of its sufficiency.
- The effect fingerprint is the immediate next-token effect at the ablated positions plus what flows through later positions of the same sentence; the contexts are the dev sentences in which the feature is most active in full precision.
- Features that are active in fewer than 4 candidate dev sentences cannot be tested on the write side; they are counted per language and have read-side values only.
- The D-035 statuses were measured on devtest; the read side uses the 300 frozen evaluation sentences of every language (the same devtest sentences of the frozen loss task).
- The survival verdict is a verdict on a metric, not on the features.
- The sample is a stratified subset of the F5 sample (a per-cell cap of 12 per language and role, seed 0), chosen for cost before any result.

## 7. Figures

- none
