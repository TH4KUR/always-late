# Gemma-2-9B original-RTN study: reviewed summary (D-125 review, tasks 1 to 4)

Written 2026-10-29 18:32 IST by Claude (the 40-minute cron job, D-125; the user asked Codex to move the 9B results to main and into this job). It sits NEXT TO the imported source files and does not replace them: [`verdict.md`](verdict.md), [`legacy_rtn_run_freeze_2026-10-29.json`](legacy_rtn_run_freeze_2026-10-29.json) and [`results/`](results/) are unchanged, byte for byte ([`import_provenance.json`](import_provenance.json)). The step-by-step log of the review is [`REVIEW_STATUS.md`](REVIEW_STATUS.md); the scripts and outputs behind the numbers are in [`review/`](review/).

Scope: the **legacy round-to-nearest (RTN) arm only**. The 9B GPTQ arm and the 9B matched-RTN arm were **not run**. Nothing below says anything about them.

## 1. What the study is

Gemma-2-9B, Gemma Scope layer-20 SAE (16k features), 27 FLORES languages, 300 held-out sentences per language, six precision states (FP, INT8, INT7, INT6, INT5, INT4), the same Study G recipe as the 2B study: scale the features that are selective for a language to a multiplier (0 = remove, 0.354, 0.177, 0.088 down; 2.83, 5.66, 11.31 up; 1.1, 1.2 fine), 3 matched-control draws, K64 group. Net harm = loss increase from editing the group minus the same for matched control features. "E" below is the net harm at multiplier 0 (zero-ablation); **retention** = E(state) / E(FP), per language.

## 2. Verification status

| item | status |
|---|---|
| The ten imported files equal the recorded hashes | **verified** (task 1) |
| The frozen code is the same code as the 2B studies; same sentences and corpus | **verified** (task 1) |
| Six per-state raw files at the pinned dataset revision equal the hashes in `analysis.json`; headers (state label, split, 27 languages, 300 ids = the freeze ids, preset, 1,130 items) | **verified** (task 2) |
| The unchanged analysis code on the raw files gives `curves.csv`, `reliability.csv`, `alpha_star.csv`, `report.md` byte-identically | **verified, byte-identical** (task 2) |
| P1 interval, retention aggregation, dose-order inequality, P3b definition | **recomputed** (task 3, `review/`) |
| That the raw files were produced by the model, weights and hook the report names, on the machine it names | **not independently verifiable**: only header consistency and the byte-identical analysis are checked; the GPU run was done by someone else and was not repeated |
| The 9B GPTQ and matched-RTN arms | **missing** (a GPTQ-only recovery run is built on the branch `gemma9b-confirmation`, not run) |

## 3. Statistics with intervals

Intervals below are 95 % percentile bootstrap intervals over the 27 languages (5,000 resamples, seed 0, the same resample for 9B and 2B, so differences are paired by language). They describe the spread between languages, not the sentence noise inside a language (the per-language sentence intervals are in `curves.csv`). Every number is from the committed curves; the 2B numbers use the committed 2B original-RTN `gain_study/results/curves.csv` with the same method.

**P1, effect of removing the language group at FP.** 27 of 27 languages positive; in 27 of 27 the lower end of the per-language interval is above 0 (smallest 0.0035, Ukrainian). Mean E 0.794 [0.359, 1.394], median 0.270 [0.197, 0.608]. 2B: mean 0.704 [0.308, 1.310], median 0.262 [0.104, 0.494]. The imported verdict did not report the interval of the mean that the freeze asks for; it is above zero, so P1 is met in both of its parts.

**Retention ladder** (three ways to aggregate; a language with a tiny FP effect makes the macro mean jump):

| state | 9B macro mean | 9B median | 9B ratio of mean effects | 2B macro mean | 2B median | 2B ratio of mean effects |
|---|---|---|---|---|---|---|
| INT8 | 1.003 [0.987, 1.019] | 1.005 [0.999, 1.016] | 1.002 [0.991, 1.014] | 1.006 [0.991, 1.019] | 1.007 [0.990, 1.028] | 1.010 [0.992, 1.020] |
| INT7 | 1.047 [1.017, 1.082] | 1.026 [0.993, 1.046] | 1.010 [0.996, 1.040] | 1.034 [1.006, 1.065] | 1.021 [1.005, 1.037] | 1.012 [0.979, 1.035] |
| INT6 | 1.006 [0.957, 1.050] | 1.019 [0.978, 1.045] | 1.017 [0.984, 1.037] | 1.038 [0.997, 1.083] | 1.031 [0.989, 1.084] | 1.045 [1.009, 1.072] |
| INT5 | 1.205 [1.086, 1.336] | 1.092 [1.006, 1.311] | 1.102 [1.021, 1.247] | 1.129 [1.071, 1.188] | 1.092 [1.036, 1.219] | 1.086 [0.993, 1.254] |
| INT4 | **2.479 [1.781, 3.368]** | **1.586 [1.238, 2.264]** | **1.474 [1.263, 1.937]** | 2.136 [1.684, 2.656] | 1.538 [1.281, 2.629] | 1.328 [1.169, 1.705] |

- P2a (INT8 and INT6 inside [0.85, 1.15]) and P2b (INT4 at least 0.30 above INT6 and the largest) hold under all three aggregations in both models.
- INT4 above 1 in 26 of 27 languages (2B: 24 of 27); above 1.3 in 18 of 27 (2B: 16 of 27).
- The 9B headline 2.479 is a mean: without Kannada (9.93 on an FP effect of 0.010) it is 2.19. The languages with the smallest FP effects are Ukrainian 0.0075, Kannada 0.010, Indonesian 0.021 and Korean 0.047. The size of the INT4 amplification is therefore **about 1.5 (median, ratio of means) to 2.5 (macro mean)**, not one number.
- 9B against 2B (paired by language): INT4 macro mean +0.343 [-0.488, 1.302]; INT4 median +0.048 [-0.777, 0.838]; INT5 +0.075 [-0.068, 0.232]; INT6 -0.032 [-0.094, 0.025]. **All intervals include zero.** The ladder has the same shape in both models (flat to INT6, a small rise at INT5, about 2x at INT4 by the macro mean); the data do not show that 9B differs from 2B in size.
- The per-language ordering is only weakly shared between the models: Spearman of the per-language INT4 retention 0.45; at INT5 -0.17 and at INT6 -0.04 (below INT4 the per-language ratios are mostly noise around 1).

**P3a (dose-response at FP).** The freeze's inequality E(0.354) > E(0.177) > E(0.088) holds in **0 of 27** languages (2B: 0 of 27). Harm that increases as the multiplier falls (0.354 < 0.177 < 0.088 < zero-ablation) holds in **27 of 27** (2B: 27 of 27). The verdict and Study G's H-G1 use the second reading; the frozen sentence as written has the direction of the inequality the wrong way round. P3a is met only in that reversed reading, and the slip is recorded (A-043), not repaired.

**P3b ("at least 20 of 27 languages have a median selectivity ratio above 1.3").** The freeze does not define the per-language ratio. With the only selectivity ratio in the code, S(f) = max over languages of the firing rate divided by the mean over languages, the median S over each language's selective peak features is above 1.3 in 27 of 27 languages (smallest 2.350). **This cannot fail:** a feature is "selective" only if S exceeds the shuffled-null cutoff 1.9625 (smallest S among the 5,218 selective features 1.9628), and every language has at least 57 selective features; even the median over all supported peak features is at least 1.82. P3b is met and carries no information. In the meaning the Llama plan gave P3b (INT4 harm over FP harm at multiplier 0.088) the 9B numbers are: median 1.689 [1.402, 2.94], mean 3.614 [2.389, 5.125], above 1 in 26 of 27 languages (2B: median 1.911 [1.434, 3.137], 27 of 27).

**Study G hypotheses** of the imported report (reproduced byte-identically): H-G1, H-G2, H-G3, H-G4 and H-G6 met; H-G5a (generic repair) and H-G5b (group-specific repair, 8 of 27 against a threshold of 14) **not met**. The turning-up repair at INT4 therefore does not work in 9B, as in 2B.

## 4. What the data support, and what they do not

**Supported (legacy RTN, 9B, language level):** removing a language's selective SAE features harms that language in all 27 languages at full precision; the removal effect is unchanged through INT6 and larger at INT4 (about 1.5 to 2.5 times, 26 of 27 languages above 1); harm rises monotonically as the multiplier falls; turning features up does not undo the quantization damage. This is the same picture as in 2B at the same scale, with intervals that overlap.

**Not supported by these data:**
1. **That the quantization damage "concentrates on language-selective circuits".** A larger removal effect at INT4 is measured on a model whose baseline has also changed; it does not say where the damage came from. The verdict's sentence is an interpretation; the consolidation note at the top of `verdict.md` already says so.
2. **That the result is independent of the quantizer.** In 2B the picture changed with the quantizer (D-103 to D-110); for 9B only the legacy RTN arm exists.
3. **That the "consistent" verdict means five clean predictions.** Three (P1, P2a, P2b) are informative and met with intervals; P3a is met only after reading the inequality the other way; P3b cannot fail. The word "consistent" is fair for the first three and for the dose-response; it should not be quoted as five independent confirmations.
4. **That 9B is better, worse or different in size from 2B.** No paired difference separates them.

## 5. Open items

- The 9B GPTQ and matched-RTN arms: missing; the GPTQ-only recovery (about 630 INR on Jarvis, budget 3 GPU-hours) is built and tested on the branch `gemma9b-confirmation`, **not run and not pushed**, awaiting the user's decision.
- Amendment A2 of that branch pins the legacy raw files by hash, so any later GPTQ result can be compared with exactly the files reviewed here.
- Q29 and the other open questions are unchanged.
