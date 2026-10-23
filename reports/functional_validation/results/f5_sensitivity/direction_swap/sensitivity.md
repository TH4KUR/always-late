# F5 sensitivity: direction-swap control against feature (same 404 features, M = 16)

This comparison is exploratory. The tested features common to the two runs: 404. W1 is the cosine of the effect fingerprint in the quantized state with the one in full precision. Run a is the reference, run b the variant. The numerical floor of each run is the 5th percentile of W1 at INT16 over the common features: 0.9469 (a) and 0.9381 (b).

| state | features | median W1 a | median W1 b | median difference (b - a) | mean abs difference | Spearman | at or above floor a | at or above floor b | classification flips |
|---|---|---|---|---|---|---|---|---|---|
| INT16 | 404 | 0.9973 | 0.9942 | -0.0021 | 0.0135 | 0.633 | 0.948 | 0.948 | 0.069 |
| INT4 | 404 | 0.8416 | 0.8282 | -0.0071 | 0.0691 | 0.402 | 0.000 | 0.000 | 0.000 |
