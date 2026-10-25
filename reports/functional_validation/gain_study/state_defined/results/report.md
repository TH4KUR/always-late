# Study G, extension B: groups selected in each precision

Exploratory in status (devtest sentences; selection on the 712 sentences outside the 300 evaluation sentences, `reserve` text of the D-035 sidecar; evaluation on the 300 held-out sentences); the rules were fixed before the runs (spec section 10 and its note). 'State-defined' = the 64 strongest own features of the language in that precision; 'FP reference' = the groups selected the same way at FP, run at every precision (the same selection text); 3 matched-control draws; 27 languages; 300 sentences per language.

## Verdicts

| hypothesis | rule | result | verdict |
|---|---|---|---|
| H-Gb1 the contributing features move with precision | Jaccard(FP reference, INT4) < Jaccard(FP reference, INT6) in at least 24 of 27 | 27 of 27; median 0.488 (INT4) against 0.855 (INT6) | **met** |
| H-Gb2 state-defined minus FP reference, net harm at ratio 0.0884, INT4 | positive in at least 18 and the median positive | 11 of 27; median difference -0.0098 (11 with the lower bound above 0) | **not met** |
| H-Gb2 state-defined minus FP reference, net harm at ratio 0.0884, INT6 | positive in at least 18 and the median positive | 9 of 27; median difference -0.0021 (8 with the lower bound above 0) | **not met** |
| H-Gb3 dose-response of the state-defined group, FP | H-G1 and H-G3 as in the main study | H-G1 True, H-G3 True | **met** |
| H-Gb3 dose-response of the state-defined group, INT6 | H-G1 and H-G3 as in the main study | H-G1 True, H-G3 True | **met** |
| H-Gb3 dose-response of the state-defined group, INT4 | H-G1 and H-G3 as in the main study | H-G1 True, H-G3 True | **met** |
| H-Gb3 (all states) | | | **met** |
| H-Gb4a generic repair at INT4, state-defined groups | rule H-G5a | x1.1: median control harm 0.0004 [0.0000, 0.0006]; x1.2: median control harm 0.0014 [0.0004, 0.0020] | **not met** |
| H-Gb4b group-specific repair at INT4, state-defined groups | rule H-G5b | 16 of 27 | **met** |

## Median net harm by state and ratio: state-defined group against the FP reference group (nats per token)

| state | ratio | state-defined | FP reference | median difference | languages where the state-defined group is larger |
|---|---|---|---|---|---|
| FP | 0 | 0.2638 | 0.2638 | +0.0000 | 0 of 27 |
| FP | 0.0884 | 0.1592 | 0.1592 | +0.0000 | 0 of 27 |
| FP | 0.177 | 0.0909 | 0.0909 | +0.0000 | 0 of 27 |
| FP | 0.354 | 0.0257 | 0.0257 | +0.0000 | 0 of 27 |
| FP | 1.1 | 0.0006 | 0.0006 | +0.0000 | 0 of 27 |
| FP | 1.2 | 0.0018 | 0.0018 | +0.0000 | 0 of 27 |
| FP | 2.83 | 0.0602 | 0.0602 | +0.0000 | 0 of 27 |
| FP | 5.66 | 0.3794 | 0.3794 | +0.0000 | 0 of 27 |
| FP | 11.3 | 1.1233 | 1.1233 | +0.0000 | 0 of 27 |
| INT6 | 0 | 0.2518 | 0.2519 | -0.0040 | 9 of 27 |
| INT6 | 0.0884 | 0.1611 | 0.1601 | -0.0021 | 9 of 27 |
| INT6 | 0.177 | 0.0907 | 0.0935 | -0.0020 | 9 of 27 |
| INT6 | 0.354 | 0.0255 | 0.0275 | -0.0010 | 11 of 27 |
| INT6 | 1.1 | 0.0006 | 0.0004 | +0.0000 | 15 of 27 |
| INT6 | 1.2 | 0.0011 | 0.0011 | +0.0001 | 16 of 27 |
| INT6 | 2.83 | 0.0622 | 0.0614 | -0.0008 | 12 of 27 |
| INT6 | 5.66 | 0.3421 | 0.3599 | -0.0071 | 12 of 27 |
| INT6 | 11.3 | 1.1649 | 1.1213 | +0.0078 | 14 of 27 |
| INT4 | 0 | 0.4521 | 0.4767 | -0.0118 | 11 of 27 |
| INT4 | 0.0884 | 0.3907 | 0.4116 | -0.0098 | 11 of 27 |
| INT4 | 0.177 | 0.2316 | 0.2517 | -0.0068 | 11 of 27 |
| INT4 | 0.354 | 0.1039 | 0.0893 | -0.0010 | 12 of 27 |
| INT4 | 1.1 | -0.0020 | -0.0011 | -0.0001 | 11 of 27 |
| INT4 | 1.2 | -0.0030 | -0.0030 | -0.0003 | 13 of 27 |
| INT4 | 2.83 | 0.0336 | 0.0318 | +0.0050 | 14 of 27 |
| INT4 | 5.66 | 0.2551 | 0.2628 | -0.0076 | 12 of 27 |
| INT4 | 11.3 | 0.8581 | 0.8056 | -0.0793 | 10 of 27 |

## Limits

- Exploratory (devtest sentences); one model, layer, SAE and quantizer; the universe of selectable features is the 10,946 supported on FP dev; the state view has no bootstrap stability; selection and evaluation texts are disjoint devtest sentences, not independent samples of a population.
- The comparison with the main study (groups selected on dev at FP) mixes selection state and selection text, which is why the FP reference run exists.
