# Positive control of the Belebele edit path (plan Task 17)

**Exploratory; a positive control, not a result about the language groups.** In each run the label K64 stands for a language-independent random set of the non-selective supported features of the stated size (full pool 2,860; 1,024); the controls are random sets of the same size from the selective features of the other languages (random sets of the same size from the selective features of the other languages). Full precision, the 18 covered languages, all evaluation items, letter format, the frozen analysis chain. The item-level numbers were read before the summary was specified (see the plan), so the verdicts below are applied to them, not predictions.

| dose | accuracy without edit | zero-edit | controls | sham | zero - none [95 % interval over languages] | controls - none | zero - controls | languages with a fall (zero) |
|---|---|---|---|---|---|---|---|---|
| 1024 | 0.4257 | 0.4137 | 0.4063 | 0.4257 | -0.0120 [-0.0188, -0.0058] | -0.0194 [-0.0266, -0.0124] | +0.0075 | 14 of 18 |
| all | 0.4257 | 0.4038 | 0.3656 | 0.4257 | -0.0219 [-0.0309, -0.0124] | -0.0601 [-0.0739, -0.0484] | +0.0381 | 15 of 18 |

Control effect (controls - none) for English as the target language against the other languages (the prompt scaffolding of every item is English: 'Passage:', 'Question:', 'Answer:'; the controls of a non-English language contain the English-selective features, those of English do not):

| dose | English | mean of the other languages | other languages |
|---|---|---|---|
| 1024 | -0.0398 | -0.0182 | 17 |
| all | -0.0996 | -0.0578 | 17 |

## Verdicts of the predictions of the plan

- **P1** (the whole pool lowers the macro accuracy by at least 0.05): **not met** (fall -0.0219, reading: intermediate).
- **P2** (the fall of the zero-edit is monotone in the dose): **met** (1024 -0.0120, all -0.0219).
- **P3** (the sham equals no edit in every language): **met**.
- **P4** (at the whole pool the zero-edit lowers the accuracy more than the controls): **not met** (zero - controls +0.0381).

**Largest fall of any edit set in the series:** controls at dose all, -0.0601 in macro accuracy. This, and not P1 alone, answers whether the edit path can register an effect on Belebele at all.

