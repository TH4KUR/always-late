# Study G, extension C: the repair near ratio 1 with a larger sample

Exploratory in status (devtest sentences, one model, one layer, RTN only); the rules below were fixed before the runs (spec section 11). Sample: 562 sentences per language (the reserve sentences no earlier run used), 5 matched-control draws, 27 languages, states FP, INT6, INT5, INT4; ratios 1.1, 1.2, 1.4, 1.7; the primary ratios are 1.1, 1.2 (intervals at 97.5 %, Holm). Units of the rules: the languages; net harm = group minus the mean of the control draws (nats per token); a negative net harm means that turning the group up lowers the loss.

## Verdicts

| hypothesis | rule | result | verdict |
|---|---|---|---|
| H-Gc1 repair at INT4, ratio 1.1 | interval of the median net harm below 0 and at least 19 of 27 negative | median -0.0013 [-0.0020, -0.0003]; 20 of 27 negative (sign test p = 0.0096) | **met** |
| H-Gc1 repair at INT4, ratio 1.2 | interval of the median net harm below 0 and at least 19 of 27 negative | median -0.0017 [-0.0031, 0.0003]; 18 of 27 negative (sign test p = 0.0610) | **not met** |
| H-Gc1 (either primary ratio) | | | **met** |
| H-Gc2 INT4 minus FP, ratio 1.1 | interval of the paired difference below 0 | median difference -0.0024 [-0.0042, -0.0012] | **met** |
| H-Gc2 INT4 minus FP, ratio 1.2 | interval of the paired difference below 0 | median difference -0.0041 [-0.0071, -0.0025] | **met** |
| H-Gc2 (every primary ratio) | | | **met** |
| H-Gc3 matched controls at INT4, ratio 1.1 | group-specific only if the control interval is not entirely below 0 | median control harm -0.0002 [-0.0004, 0.0000] | |
| H-Gc3 matched controls at INT4, ratio 1.2 | group-specific only if the control interval is not entirely below 0 | median control harm 0.0001 [-0.0004, 0.0008] | |
| H-Gc3 (group-specific) | H-Gc1 met and controls not falling too | | **met** |

## H-Gc4: medians by state and ratio (nats per token)

| state | ratio | median net | median group harm | median control harm | languages with net < 0 | languages with alpha* here | lowest median group harm |
|---|---|---|---|---|---|---|---|
| FP | 1.1 | 0.0006 | 0.0005 | 0.0000 | 7 of 27 | 6 |  |
| FP | 1.2 | 0.0014 | 0.0014 | 0.0002 | 4 of 27 | 1 |  |
| FP | 1.4 | 0.0044 | 0.0066 | 0.0011 | 3 of 27 | 0 |  |
| FP | 1.7 | 0.0135 | 0.0164 | 0.0047 | 2 of 27 | 1 |  |
| INT6 | 1.1 | 0.0004 | 0.0005 | 0.0000 | 9 of 27 | 6 |  |
| INT6 | 1.2 | 0.0013 | 0.0013 | 0.0001 | 7 of 27 | 2 |  |
| INT6 | 1.4 | 0.0044 | 0.0061 | 0.0012 | 3 of 27 | 0 |  |
| INT6 | 1.7 | 0.0138 | 0.0162 | 0.0047 | 2 of 27 | 1 |  |
| INT5 | 1.1 | 0.0005 | 0.0004 | -0.0001 | 8 of 27 | 6 |  |
| INT5 | 1.2 | 0.0013 | 0.0015 | 0.0000 | 5 of 27 | 2 |  |
| INT5 | 1.4 | 0.0049 | 0.0071 | 0.0012 | 3 of 27 | 1 |  |
| INT5 | 1.7 | 0.0133 | 0.0184 | 0.0050 | 2 of 27 | 0 |  |
| INT4 | 1.1 | -0.0013 | -0.0017 | -0.0002 | 20 of 27 | 6 |  |
| INT4 | 1.2 | -0.0017 | -0.0017 | 0.0001 | 18 of 27 | 5 | yes |
| INT4 | 1.4 | -0.0007 | 0.0003 | 0.0018 | 15 of 27 | 7 |  |
| INT4 | 1.7 | 0.0046 | 0.0093 | 0.0062 | 8 of 27 | 2 |  |

## Reliability of the sample

562 sentences per language, 5 control draws: standard errors of the net harm from the sentences and from the control draws (medians over languages) and the number of languages whose |net harm| exceeds twice the sentence error alone / both errors combined.

| state | ratio | median abs net harm | SE sentences | SE control draws | languages clear (sentences) | languages clear (both) |
|---|---|---|---|---|---|---|
| FP | 1.1 | 0.0009 | 0.0004 | 0.0001 | 17 of 27 | 17 of 27 |
| FP | 1.2 | 0.0019 | 0.0006 | 0.0002 | 21 of 27 | 20 of 27 |
| FP | 1.4 | 0.0053 | 0.0010 | 0.0004 | 24 of 27 | 23 of 27 |
| FP | 1.7 | 0.0158 | 0.0016 | 0.0009 | 26 of 27 | 25 of 27 |
| INT6 | 1.1 | 0.0006 | 0.0004 | 0.0001 | 14 of 27 | 13 of 27 |
| INT6 | 1.2 | 0.0017 | 0.0006 | 0.0002 | 20 of 27 | 18 of 27 |
| INT6 | 1.4 | 0.0046 | 0.0010 | 0.0005 | 24 of 27 | 23 of 27 |
| INT6 | 1.7 | 0.0149 | 0.0017 | 0.0010 | 26 of 27 | 25 of 27 |
| INT5 | 1.1 | 0.0010 | 0.0004 | 0.0001 | 17 of 27 | 16 of 27 |
| INT5 | 1.2 | 0.0019 | 0.0006 | 0.0002 | 18 of 27 | 18 of 27 |
| INT5 | 1.4 | 0.0056 | 0.0011 | 0.0004 | 23 of 27 | 23 of 27 |
| INT5 | 1.7 | 0.0137 | 0.0017 | 0.0010 | 25 of 27 | 25 of 27 |
| INT4 | 1.1 | 0.0017 | 0.0005 | 0.0003 | 18 of 27 | 17 of 27 |
| INT4 | 1.2 | 0.0025 | 0.0008 | 0.0005 | 23 of 27 | 19 of 27 |
| INT4 | 1.4 | 0.0054 | 0.0014 | 0.0010 | 19 of 27 | 18 of 27 |
| INT4 | 1.7 | 0.0127 | 0.0022 | 0.0019 | 24 of 27 | 20 of 27 |

## Limits

- Exploratory (devtest sentences); one model, layer, SAE and quantizer; the effect, if present, is small (a thousandth of a nat per token at ratio 1.1): state it with its size.
- The interval condition of H-Gc1 holds whenever at least 20 of 27 language values are negative and fails with 18 or fewer; with exactly 19 it is borderline and decided by the seeded bootstrap (seed 0, 5000 draws).
- The motivating data are the main-study data (D-074); the rules and ratios were fixed before the data of this extension existed.
