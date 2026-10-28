# What changes when we quantize with GPTQ instead of RTN? (plain-language results)

Written 2026-10-28 (IST), at the user's request of 2026-10-27: "run tests again with GPTQ quantization and try to keep other things same as before so that we can draw a fair comparison." Written for readers who did not follow the work. Short words on purpose. The full record is in [`STATUS.md`](STATUS.md) (ledger and log), the decisions D-101 to D-107 in [`../decisions_and_anomalies.md`](../decisions_and_anomalies.md), and the plan in [`../../docs/superpowers/plans/2026-10-27-gptq-matched-comparison.md`](../../docs/superpowers/plans/2026-10-27-gptq-matched-comparison.md). **Status: stages 1 to 3 are done and checked; Study G, F5 and the external panel are still running (section 7).** Everything here is **exploratory**: it asks "does the old finding survive another quantizer?", it does not test a new claim.

## 1. The short answer

- **Yes, a lot depends on the quantizer, but mostly the size of the effect, not the direction.** With the better quantizer (GPTQ) the 4-bit model is much closer to the original: it keeps more of the model's inner features, loses less next-word accuracy, and the damage of the "transfer" test shrinks a lot.
- **The main conclusions of the project do not flip.** The language-specific features still matter for their language (removing them hurts in all 27 languages). The damage is still not concentrated in those features (they carry only 2 to 3 % of it). Removing a language's features still does not change Belebele accuracy (a reading test) by more than half a point, and this now also holds at 4 bits where the model is clearly above chance. The headline verdict of the transfer study (the prediction criterion is met at 4 bits, not at 6 bits, so not met overall) stays the same.
- **One finding was fragile:** the old null result of Phase 1 (do language features survive rounding worse than ordinary features?) changes its verdict word under one of the two new quantizers (matched RTN) and disappears under the other (GPTQ). The effect is tiny in every case (at most 1.5 points out of about 95) and this is **not** a new confirmed result (section 4.2).
- **Part of the legacy RTN's weakness was not RTN itself but how we had implemented it** (it shared one scale across attention heads and used a slightly different rounding grid). About a third of the gap between legacy RTN and GPTQ at 4 bits is already closed by simply doing RTN the standard way ("matched RTN").

## 2. What we did, in simple words

We compare **three ways of rounding the model's weights to few bits** and run the same experiments on each:

| name | what it is |
|---|---|
| **legacy RTN** | the project's original rounding: every weight is rounded to the nearest allowed value; one scale per attention coordinate shared across heads. All earlier results of the project are this. We did not rerun it; its old files are the reference. |
| **matched RTN** | the same idea (nearest value) done the standard way: one scale per output channel, the same rounding grid as GPTQ, no calibration. It exists to separate two effects: "a better implementation of rounding" and "GPTQ's cleverness". |
| **GPTQ** | rounds weight by weight and corrects the remaining weights to cancel the error, using some example text (calibration text). We used 8,192 word pieces (tokens) of Wikipedia text in each of the 27 languages (221,184 in all), chosen so that none of it appears in the test sentences. |

**What we kept the same:** the model (Gemma-2-2B), the feature dictionary (Gemma Scope, layer 12), the test sentences, the language groups, the matched controls, the edit positions, the batching and the arithmetic, and the old analysis code. The extension only swaps how the weights are rounded. The old code files were not edited at all (some are locked by hash in the project's pre-set freezes); a small wrapper swaps the rounding from outside.

**Bit widths:** INT4 and INT6 for everything; the whole ladder INT8 to INT4 for the first stage (Phase 1), because the project's pre-set tests are defined at 7 and 6 bits.

## 3. How we checked the comparison is fair

- **The full-precision part is bit-for-bit the old one.** In every stage, the numbers that depend only on the unquantized model (loss, feature firing rates, token counts, the full-precision effects, and the INT16, INT12 and INT0 controls) are identical to the old runs, to the last digit, in both new arms. (Phase 1: exact; transfer study: 259,000 items, 0 differences; Belebele full-precision accuracy 0.4257 in all arms.)
- **The weight sets are the same on every machine.** Each job rebuilds its quantized weights and the fingerprint (a hash) of every weight set was equal in all jobs that built the same one, on more than ten different cluster machines (4-bit: 8 jobs per method; 6-bit: 6 jobs).
- **The old analysis code reproduces the old results.** Every time we ran an old analysis on old files first (a control), the output equalled the committed old output exactly (Phase 1 tests byte for byte; loss-task rows 162 of 162; Belebele rows 54, 54 and 40 of 54, 54 and 40).
- **Calibration text is disjoint from the tests.** No FLORES sentence is in the calibration text, and no article of the Wikipedia test panel is.

## 4. Results

### 4.1 Features and loss on the same sentences (Phase 1, all 27 languages)

"Survival" = of the features that fire in a language, how many still behave the same (correlation above 0.9) after rounding. "Loss increase" = how much worse the next-word guess gets, in nats per token (smaller is better). Averages over the 27 languages; source `results/stage1/` (D-104).

| bits | survival: legacy / matched / GPTQ | loss increase: legacy / matched / GPTQ |
|---|---|---|
| INT8 | about 100 % in all | no practical difference (GPTQ minus legacy +0.0006) |
| INT7 | no practical difference (GPTQ over legacy +1.2 points) | GPTQ minus legacy -0.0026 |
| INT6 | 85.0 % / 89.7 % / **97.5 %** | 0.0254 / 0.0197 / **0.0058** |
| INT5 | GPTQ +26.9 points over legacy | GPTQ -0.064 |
| INT4 | 12.7 % / 18.7 % / **30.5 %** | 0.590 / 0.385 / **0.130** |

GPTQ is better than legacy RTN in **27 of 27 languages** at INT4 and INT6, on both measures. The lowest language at INT6 keeps 53 % under legacy RTN and 92.7 % under GPTQ. By resource tier at INT6 the gain is biggest for the low-resource languages (+25 points) and small for high-resource ones (+4): **most of the "low-resource languages suffer more" pattern at 6 bits belongs to the legacy quantizer**. At INT4 the order flips (+22 / +19 / +13 for high / mid / low) and 4-bit survival is low for everyone (GPTQ: 30 %).

### 4.2 The project's original pre-set tests (Phase 1, T1 to T3) — a fragile null

The original question: do language-specific features survive rounding **worse** than matched ordinary features? (Difference of survival, selective minus control; p-values corrected over the three tests.) Source `results/stage1/analysis_*` (D-106). Old code reproduced exactly (control).

| test | legacy RTN | matched RTN | GPTQ |
|---|---|---|---|
| at 7 bits | -0.28 points, corrected p 0.51 | **-0.48 points, corrected p 0.042** | -0.14 points, corrected p 1 |
| at 6 bits | -1.44 points, corrected p 0.13 | **-1.53 points, corrected p 0.038** | +0.16 points, corrected p 1 |
| tier slope | corrected p 0.99 | 0.98 | 1 |

Under legacy RTN the family is null, under matched RTN two tests cross the 0.05 line, under GPTQ nothing is left. The direction (selective features survive a little less) is the same for the two RTN versions and the size at 6 bits is almost equal; the change of verdict comes mostly from the 7-bit test. **Why this is not a new result:** the tests were written for the legacy quantizer; running them on two more quantizers is two more families with no correction rule (a factor of 2 would take the matched-RTN values to 0.084 and 0.077, above 0.05); the effect is at most 1.5 points near the ceiling. The honest summary: the old null depends on the quantizer in a delicate way and the effect is tiny whatever the quantizer.

### 4.3 The "transfer" study (does an edit made in the full model still work in the rounded model?)

Same pre-set rules, run once per arm (D-105, `results/stage2/`). **The headline verdict is identical in all three arms:** the project's criterion (a small model that predicts the transfer error beats the best simple baseline by at least 10 %) is "met" at 4 bits and "not met" at 6 bits, so "not met" overall. What changes is the size of the error:

| | legacy RTN | matched RTN | GPTQ |
|---|---|---|---|
| relative transfer error, 4 bits | 0.664 | 0.572 | **0.476** |
| relative transfer error, 6 bits | 0.165 | 0.153 | **0.132** |
| error over the rounding noise floor, 6 bits | 1.88 | 1.75 | **1.52** |

The recipe of the error keeps its order in every arm (weights part largest, context part next, interaction smallest). With GPTQ at 6 bits the error is close to the noise floor, which is why two secondary claims (H3, H4) weaken at 6 bits: at that point the test cannot tell the damage from rounding noise. Across 27 languages (paired), the 4-bit total harm of the group edit is lower under GPTQ than under legacy RTN by 0.46 [0.31, 0.62], in 26 of 27 languages.

### 4.4 What the language-feature groups do (loss task) and where the damage sits (rescue)

(D-107, `results/stage3/`.) Removing a language's top-64 features hurts that language's text in **all 27 languages in every arm** (full precision included). At 4 bits the harm is **bigger than in full precision** ("amplified"), but less so with a better quantizer:

| 4 bits | legacy RTN | matched RTN | GPTQ |
|---|---|---|---|
| mean harm (nats/token) | 0.935 | 0.900 | 0.779 |
| median ratio to full precision | 1.54 | 1.36 | 1.22 |
| word the rule gives | amplified | amplified | inconclusive |

At 6 bits the harm is "preserved" (ratio about 1.0) in all arms. **Where the damage sits:** if we copy the full-precision activations back into the rounded model at layer 12, the language group recovers only **2.6 % / 2.3 % / 2.0 %** of the lost accuracy (legacy / matched / GPTQ), all SAE features 17 % / 14 % / 11 %, the whole layer-12 residual 39 % / 35 % / 41 %. So the damage is not concentrated in the language-specific features, whatever the quantizer.

### 4.5 A reading test (Belebele, 18 languages)

(D-107.) Accuracy of the model **without any edit** (chance is 0.25): full precision 0.426 in all arms; 6 bits 0.423 / 0.424 / 0.423; **4 bits 0.236 / 0.291 / 0.319** (legacy / matched / GPTQ). The old 4-bit model was at chance, so "removing the group changes nothing" could not be read at 4 bits. With GPTQ the 4-bit model answers well above chance and **removing a language's feature group still changes accuracy by less than half a point** (effect within ±0.5 points in every arm and state). The old null is not an artifact of a broken 4-bit model.

## 5. What this tells us, finding by finding

| old finding | under GPTQ |
|---|---|
| Rounding hurts low-resource languages' features more (6 bits) | **mostly a feature of the legacy quantizer**: the tier gap at 6 bits nearly closes; at 4 bits all languages suffer and the order reverses |
| Language-specific features survive no worse than ordinary ones (Phase 1 null) | **fragile** (section 4.2): verdict depends on the quantizer, effect tiny |
| Removing the language group hurts that language | **holds** in 27 of 27 languages |
| That harm is amplified at 4 bits | **holds, smaller** (median 1.22 instead of 1.54) |
| The damage is not concentrated in the language features | **holds** (2 to 3 %) |
| Transfer test: verdict "not met" | **holds**; the error itself shrinks by about 28 % at 4 bits |
| Belebele: no effect of removing the group | **holds**, and now also at an above-chance 4-bit model |

## 6. What we cannot say

- **We cannot say GPTQ is better "because of error compensation".** GPTQ saw example text from the same 27 languages and RTN saw none. Matched RTN is there to separate the grid and granularity change; the remaining gap mixes compensation and calibration. A GPTQ run with no calibration text is not possible; a different or smaller calibration text would be a new experiment.
- **One model, one layer, one feature dictionary, weights-only rounding.** Nothing here covers other models (the Gemma-2-9B check belongs to another team), other layers, or activation quantization.
- **All contrasts are exploratory.** No multiplicity rule covers three arms. The three-arm intervals in sections 4.3 and 4.4 are each arm's own, not paired contrasts (the stage 1 and group-level transfer contrasts are paired by language).
- **Not repeated:** F4 (its feature sets come from the legacy quantizer's own association losses) and the association retention study E3 (a 2.76 GB file per arm does not fit the cluster disk, 2.4 GB free). The Belebele pre-set statements S1 to S5 need all six bit widths and were not rerun (only 6 and 4 bits were).

## 7. Still running or not yet read (updated when they finish)

- **F5** (does a single feature do the same thing after rounding?): jobs 44072, 44073.
- **Study G** (graded gain, the user's major study; INT4 and INT6): jobs 44074 to 44077.
- **External panel** (Wikipedia text in six languages, transfer study at group and feature level): jobs 44209 to 44216. **Caution:** the GPTQ calibration text is Wikipedia too (different articles), so under GPTQ the external panel is in the calibration domain; an advantage there may partly be domain match.

## 8. Where everything is

- Results files: `results/stage1/` (survival, loss, Phase 1 tests), `results/stage2/` (transfer study), `results/stage3/` (loss task, rescue, Belebele); each has a README with the exact commands. Raw job outputs stay outside git (`/home/karma/Documents/ANLP/project/transfer_runs/`), small files are in the repository.
- Tools: `scripts/g0/gptq_ext/` (calibration, weight sets, wrapper, comparison, validation, bridge), tests `scripts/g0/tests_torch/test_gptq_ext_*.py` (all mutation-checked), freeze `freeze.json` and amendment `amendment_A1_stage1_ladder.json`, artifact hashes `artifact_hashes.json`.
- Jobs and times: the job table in [`STATUS.md`](STATUS.md) (Slurm 42848 to 44216).
