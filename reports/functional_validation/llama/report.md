# Llama-3.2-1B against the Gemma findings: a minimum consistency check

Exploratory; the predictions and the verdict function were fixed in the plan (`docs/superpowers/plans/2026-10-26-llama-consistency-check.md`, sections 3, 6 and 7) before any Llama number existed. 27 languages tested (group of at least 16 members; untested: none), 300 held-out sentences per language, three matched-control draws; SAE `fineweb42`. All numbers come from the frozen Gemma code (functional effect, macro retention, the Study G dose rule) applied to the Llama item files.

## Verdict: **not consistent**

Setting differences that travel with every result: depth 12 of 16 (Gemma 12 of 26), width 2048, a TopK SAE (k = 48, 14,336 features) against a JumpReLU one, another SAE family and training data, another tokenizer, another model size. Not tested here: the INT4 repair at ratios 1.1 and 1.2, the Belebele task, single-feature fingerprints (F5), state-defined groups, the optimal-gain analysis.

| prediction | rule (fixed beforehand) | Llama | Gemma | result |
|---|---|---|---|---|
| P1 existence at FP | median E above 0 with its 95 % interval above 0 and E > 0 in at least 24 of 27 | median E 0.048 [0.018, 0.173]; 21 of 27 positive | median E 0.262; 27 of 27 | **not met** |
| P2a flat through INT6 | macro retention within [0.85, 1.15] at INT8 and INT6 | INT8 1.010, INT6 1.024 | INT8 1.006, INT6 1.038 | met (reported) |
| P2b the deep end changes | macro retention at INT4 at least 0.30 above INT6 and the largest | INT5 0.139, INT4 -3.571 | INT5 1.129, INT4 2.136 | **not met** |
| P3a graded in the down ratio | the frozen H-G1 (ordered in at least 24 of 27 and in the median) at FP and at INT4 | FP 20 of 27, INT4 15 of 27 | Study G: H-G1 met at every state | **not met** |
| P3b the deepest turn-down costs more at INT4 | larger at INT4 in at least 20 of 27 and a median INT4/FP ratio above 1.3 | 14 of 27; median ratio 1.168; median net harm FP 0.035, INT4 0.052 | median net FP 0.169, INT4 0.323; ratio 1.911 | not met (reported) |

## Diagnostics (not predictions)

Medians over the tested languages of the group harm, the matched-control harm (mean of the three draws) and the perturbation norm ratio of the zeroed set (group and controls; the matching should make the two norm ratios similar). A control harm above the group harm together with similar norm ratios is a result about the model, not about the matching; clearly different norm ratios would point at the matching.

| state | group harm | control harm | perturbation norm, group | perturbation norm, controls |
|---|---|---|---|---|
| FP | 0.1329 | 0.0850 | 0.1789 | 0.1710 |
| INT4 | 0.2454 | 0.2668 | 0.1207 | 0.1898 |

## Reading

- consistent = P1 and P2b and P3a: the qualitative picture of Gemma (a language-specific, graded dependence, flat through the middle of the ladder, larger at INT4) is found on a second model with a second SAE family. It is not a replication of the numbers and does not turn an exploratory Gemma claim into a confirmatory one.
- partly consistent = P1 and P3a but not P2b: the dependence exists and is graded, but the quantization pattern differs; report the difference, do not explain it away. limited = P1 but not P3a: a language-specific dependence that is not graded. not consistent = P1 fails.
- P2a and P3b are reported and never change the verdict.

## Limits

- Exploratory (devtest sentences, selection on dev); one small model, one layer, one SAE (a sensitivity SAE is optional); the Gemma numbers are those of the committed outputs; groups with fewer than 64 members are the whole selective set; languages with fewer than 16 selective features are not tested.
