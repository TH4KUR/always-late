# F5 sensitivity: direction-swap control against the features, both with full-precision amplitudes

This comparison is exploratory. The tested features common to the two runs: 404. W1 is the cosine of the effect fingerprint in the quantized state with the one in full precision. Run a is the reference, run b the variant. The numerical floor of each run is the 5th percentile of W1 at INT16 over the common features: 0.9465 (a) and 0.9371 (b).

| state | features | median W1 a | median W1 b | median difference (b - a) | mean abs difference | Spearman | at or above floor a | at or above floor b | classification flips |
|---|---|---|---|---|---|---|---|---|---|
| INT16 | 404 | 0.9974 | 0.9942 | -0.0021 | 0.0133 | 0.638 | 0.948 | 0.948 | 0.074 |
| INT4 | 404 | 0.8462 | 0.8378 | +0.0004 | 0.0659 | 0.290 | 0.000 | 0.000 | 0.000 |
