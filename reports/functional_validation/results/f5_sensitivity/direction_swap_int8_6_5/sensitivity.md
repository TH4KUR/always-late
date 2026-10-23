# F5 sensitivity: direction-swap control against the features at INT16, INT8, INT6 and INT5 (ordinary amplitudes)

This comparison is exploratory. The tested features common to the two runs: 404. W1 is the cosine of the effect fingerprint in the quantized state with the one in full precision. Run a is the reference, run b the variant. The numerical floor of each run is the 5th percentile of W1 at INT16 over the common features: 0.9469 (a) and 0.9381 (b).

| state | features | median W1 a | median W1 b | median difference (b - a) | mean abs difference | Spearman | at or above floor a | at or above floor b | classification flips |
|---|---|---|---|---|---|---|---|---|---|
| INT16 | 404 | 0.9973 | 0.9942 | -0.0021 | 0.0135 | 0.633 | 0.948 | 0.948 | 0.069 |
| INT8 | 404 | 0.9964 | 0.9926 | -0.0024 | 0.0144 | 0.617 | 0.938 | 0.941 | 0.087 |
| INT6 | 404 | 0.9884 | 0.9826 | -0.0027 | 0.0195 | 0.497 | 0.921 | 0.896 | 0.119 |
| INT5 | 404 | 0.9609 | 0.9538 | -0.0038 | 0.0297 | 0.415 | 0.700 | 0.658 | 0.304 |
