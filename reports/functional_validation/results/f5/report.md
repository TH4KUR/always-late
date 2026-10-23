# F5 report: does a feature do the same thing after quantization? (exploratory)

## 1. Instrument and numerical floor

1487 sample features, 1487 tested on the write side, 0 too rare to test (read side only). Work dtype float32; sham ablation exact: True; logit gates ok: True; run complete: True. Stratified subset: a per-cell cap of 12 features per language and role (seed 0).

**Numerical floor.** W1 (cosine of the single-feature effect fingerprints) between full precision and the repeat measurement of the same weights (FP-rebatched): median n/a, 5th percentile n/a; between full precision and INT16 (weights rounded at the bf16 noise level): median 0.997, 5th percentile 0.949. The numerical floor used below is 0.9489 (the lower of the two, from INT16). INT8, a real quantization, is reported but not part of the floor: median 0.997. Instrument adequate: True.

## 2. Main table (all tested features, by state)

R1 token correlation of the activations with full precision (the survival basis); R2 firing-set Jaccard; R3 scale ratio; W1 cosine of the single-feature effect fingerprints; the share at or above the numerical floor and above the 95th percentile of the matched different-feature control; 'does nothing' = the feature has no effect in the quantized model; top-1 = retrieval of the feature's own identity (within language, chance in brackets). Intervals: 95 %, languages resampled.

| state | tested | R1 median | R1 >= 0.9 | R2 | R3 | W1 median | W1 >= floor | W1 >= control | does nothing | top-1 | within top-1 (chance) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| INT16 | 1487 | 0.995 [0.995, 0.996] | 0.999 | 0.973 | 1.000 | 0.997 [0.997, 0.998] | 0.950 [0.933, 0.966] | 1.000 | 0 (0.000) | 1.000 | 1.000 (0.018) |
| INT8 | 1487 | 0.990 [0.989, 0.991] | 0.999 | 0.945 | 0.999 | 0.997 [0.996, 0.997] | 0.940 [0.923, 0.957] | 1.000 | 0 (0.000) | 1.000 | 1.000 (0.018) |
| INT4 | 1487 | 0.738 [0.714, 0.764] | 0.132 | 0.395 | 0.922 | 0.839 [0.829, 0.848] | 0.003 [0.001, 0.006] | 0.999 | 1 (0.001) | 0.997 | 0.999 (0.018) |
| INT5 | 1487 | 0.909 [0.902, 0.916] | 0.560 | 0.674 | 1.007 | 0.961 [0.958, 0.964] | 0.665 [0.617, 0.716] | 1.000 | 0 (0.000) | 1.000 | 1.000 (0.018) |
| INT6 | 1487 | 0.961 [0.957, 0.964] | 0.898 | 0.823 | 0.994 | 0.988 [0.986, 0.989] | 0.897 [0.871, 0.922] | 1.000 | 0 (0.000) | 1.000 | 1.000 (0.018) |

## 3. Features whose association was lost and retained (F4 subsets), by precision

| group | tested | W1 median | W1 >= floor | R1 median | R1 >= 0.9 |
| --- | --- | --- | --- | --- | --- |
| lost@INT6 | 189 | 0.986 [0.982, 0.990] | 0.868 [0.822, 0.913] | 0.948 | 0.836 |
| retained@INT6 | 208 | 0.985 [0.982, 0.987] | 0.894 [0.857, 0.929] | 0.947 | 0.837 |
| lost@INT5 | 307 | 0.962 [0.958, 0.964] | 0.713 [0.646, 0.780] | 0.896 | 0.479 |
| retained@INT5 | 317 | 0.960 [0.957, 0.965] | 0.666 [0.588, 0.745] | 0.904 | 0.527 |
| lost@INT4 | 417 | 0.838 [0.824, 0.849] | 0.005 [0.000, 0.012] | 0.731 | 0.118 |
| retained@INT4 | 396 | 0.841 [0.829, 0.852] | 0.000 [0.000, 0.000] | 0.735 | 0.119 |

## 4. Statements: is the survival metric sensible?

| statement | value | 95 % interval (languages resampled) | defined | verdict |
|---|---|---|---|---|
| S1 INT6, retained by the D-035 status: share with W1 >= floor (rule: >= 0.80) | 0.902 | [0.874, 0.926] | 1240 | met |
| S2 INT4, F4 pairs: AUROC of the status for W1 >= floor (rule: >= 0.60, lower bound > 0.5) | 0.256 | [0.249, 0.264] | 2 event, 811 no event | undefined |
| S3 INT5, baseline cutoff (correlation >= 0.9): share with W1 >= floor (rule: >= 0.80) | 0.824 | [0.777, 0.863] | 833 | met |
| descriptive S2c (descriptive, no rule) INT4, F4 pairs: AUROC of W1 for the status retained (0.5 = no relation; below 0.5 the lost features keep more) | 0.506 | [0.469, 0.543] | 396 retained, 417 lost | none |

**Overall: undetermined.** Work dtype float32; sham gate exact True; logit gates ok True; run complete True.

If the three statements hold, they hold for this model, this layer, this quantizer and this sample, nothing more. If they do not hold, the survival label describes the input-side agreement of a coordinate system and no functional claim may be built on it. The study is exploratory: no confirmatory claim is made from it.

## 5. Descriptive hypotheses (H5a to H5c)

- H5a (the median token correlation is at least 0.9 at INT8 and INT6 and lower at INT4 than at INT6): True; medians INT8 0.990, INT6 0.961, INT5 0.909, INT4 0.738.
- H5b (the median W1 is below the median token correlation at every quantized precision): False; median W1 INT8 0.997, INT6 0.988, INT5 0.961, INT4 0.839; share below the floor INT6 0.103, INT5 0.335, INT4 0.997.
- H5c (retained features keep more of their effect than lost ones, paired by language): signs INT6 inconclusive, INT5 inconclusive, INT4 inconclusive; medians INT6 -0.001, INT5 -0.001, INT4 0.003.

## 6. Limits

- This study is exploratory: no confirmatory claim is made from it. The rule S1 to S3 was fixed before the first number (Q23); the sample, the contexts and the analysis are one design among several.
- One model (Gemma-2-2B), one layer (the layer-12 residual), one SAE (Gemma Scope, 16,384 features), one quantizer (round-to-nearest per output channel). The ablation is a single-feature zero ablation: a test of the necessity of the feature's term, not of its sufficiency.
- The effect fingerprint is the immediate next-token effect at the ablated positions plus what flows through later positions of the same sentence; the contexts are the dev sentences in which the feature is most active in full precision.
- Features that are active in fewer than 4 candidate dev sentences cannot be tested on the write side; they are counted per language and have read-side values only.
- The D-035 statuses were measured on devtest; the read side uses the 300 frozen evaluation sentences of every language (the same devtest sentences of the frozen loss task).
- The survival verdict is a verdict on a metric, not on the features.
- The sample is a stratified subset of the F5 sample (a per-cell cap of 12 per language and role, seed 0), chosen for cost before any result.

## 7. Figures

- ![../../figures/f5/f5_fig1_w1_distribution_light.png](../../figures/f5/f5_fig1_w1_distribution_light.png)
- ![../../figures/f5/f5_fig2_read_vs_write_light.png](../../figures/f5/f5_fig2_read_vs_write_light.png)
- ![../../figures/f5/f5_fig3_statements_light.png](../../figures/f5/f5_fig3_statements_light.png)
- ![../../figures/f5/f5_fig4_retrieval_light.png](../../figures/f5/f5_fig4_retrieval_light.png)
- ![../../figures/f5/f5_fig5_median_ladder_light.png](../../figures/f5/f5_fig5_median_ladder_light.png)
- ![../../figures/f5/f5_fig6_paired_light.png](../../figures/f5/f5_fig6_paired_light.png)
