# F5 swap control along the precision ladder: swapped directions against features

This comparison is exploratory. The features tested in the feature run and in every control run: 404. W1 is the cosine of the effect fingerprint in the quantized state with the one in full precision; the control ablates the decoder direction of a matched different feature at the same positions. Each run's numerical floor is the 5th percentile of W1 at INT16 over these features: 0.9469 (features) and 0.9381 (control). The shortfall ratio is the control's shortfall below 1 (1 minus the median W1) over the features'.

| state | features | median features | quartiles features | median control | quartiles control | median difference (control - features) | paired median difference | shortfall ratio | at or above floor features | at or above floor control |
|---|---|---|---|---|---|---|---|---|---|---|
| INT16 | 404 | 0.9973 | 0.9894 to 0.9994 | 0.9942 | 0.9807 to 0.9979 | -0.0031 | -0.0021 | 2.18 | 0.948 | 0.948 |
| INT8 | 404 | 0.9964 | 0.9878 to 0.9988 | 0.9926 | 0.9780 to 0.9972 | -0.0038 | -0.0024 | 2.06 | 0.938 | 0.941 |
| INT6 | 404 | 0.9884 | 0.9776 to 0.9921 | 0.9826 | 0.9659 to 0.9907 | -0.0058 | -0.0027 | 1.50 | 0.921 | 0.896 |
| INT5 | 404 | 0.9609 | 0.9431 to 0.9707 | 0.9538 | 0.9265 to 0.9717 | -0.0071 | -0.0038 | 1.18 | 0.700 | 0.658 |
| INT4 | 404 | 0.8416 | 0.7879 to 0.8724 | 0.8282 | 0.7594 to 0.8705 | -0.0134 | -0.0071 | 1.08 | 0.000 | 0.000 |
