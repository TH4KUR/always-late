# All-results assessment (Stage 1 of the survival-and-intervention plan)

Written 2026-10-28 20:32 IST by Claude, after the user's confirmation of the plan (D-110) and after the last job of the GPTQ extension was validated and analysed (D-111). It follows [the plan](../../docs/superpowers/plans/2026-10-28-survival-and-intervention-publication-rescue.md), Stage 1: one evidence-based review of ALL current results (full precision, legacy RTN, matched RTN, GPTQ; main study and external panel) before any research plan is changed. **No new analysis was run for this document.** Every number is quoted from a committed decision (D-xxx in [`decisions_and_anomalies.md`](../decisions_and_anomalies.md)) or result file, with its source. Nothing in the original results or frozen plans is revised: where a verdict word differs between arms, both are reported. Plain-language versions: [`PROJECT_RESULTS.md`](../PROJECT_RESULTS.md) (legacy) and [`gptq_extension/RESULTS.md`](../gptq_extension/RESULTS.md) (three arms).

## 0. Result of the review in five lines

1. **The original selective-destruction story is not supported under any of the three quantizers.** What looks like "language features die first" is mostly "weakly firing features die first"; the pre-set Phase 1 tests are null or at most 1.5 points of survival wide, and their verdict words flip between arms at the threshold (D-106), which is why they are not a result to build on in either direction.
2. **What survives every arm is a set of defined-intervention effects in Gemma-2-2B (layer 12, one SAE):** removing the 64 language-selective features hurts the language's next-word loss in 27 of 27 languages; at 4 bits that harm is amplified, and the amplification shrinks as the quantizer improves (median ratio 1.54 / 1.36 / 1.22); the group's full-precision patch recovers only 2.0 to 2.6 % of the INT4 loss gap (a recovery share, not a causal share); the transfer error of an edit is mostly generic and its size, not its structure, depends on the quantizer; the reading task shows no group effect in any arm (within ±0.5 points).
3. **The survival measures are weak stand-ins for functional continuity:** the binary association label does not predict function (AUROC 0.49 under GPTQ; 0.51 earlier); the continuous activation correlation R1 is moderately related to the effect similarity W1 (Spearman 0.53 / 0.52 / 0.58 at INT6 and 0.41 / 0.38 / 0.38 at INT4, legacy / matched / GPTQ), and that relation has never been tested against simple baselines.
4. **The external panel agrees with the main study on structure and ordering** (legacy > matched > GPTQ in transfer error; same verdict pattern for UTILITY under GPTQ: INT4 met, INT6 not met) with two cautions: only six languages, and the GPTQ calibration text is Wikipedia like the panel.
5. **Decision gate: PROCEED, NARROWED** (section 8): the saved F5 files can answer a bounded form of the plan's question (does R1 add predictive value for W1 beyond simple baselines, per quantizer and state, held-out by language), so Stages 2 to 4 go ahead on existing Gemma data under the plan's validity gates; Stage 5 (Llama) stays conditional on a hard time gate. The project's defensible output is a careful, mostly negative and limits-focused result; the plan itself says it does not promise a publishable result and this review does not change that.

## 1. Evidence base and its status

| family | arms | status under the original rules | where |
|---|---|---|---|
| Phase 1: survival and the three tests T1 to T3 (INT7, INT6, tier slope) | legacy RTN (original), matched RTN, GPTQ | **confirmatory for the legacy arm only** (D-014, pre-set, Holm over three); the two new arms are exploratory (two more families, no multiplicity rule, D-106) | D-014, D-104, D-106; `reports/phase1*`, `gptq_extension/results/stage1` |
| Belebele reading task (S0 to S5) | legacy, matched, GPTQ at INT6 and INT4 | **confirmatory for the legacy arm** (frozen); S1 to S5 undefined because S0 is not met; new arms exploratory | D-047, D-050, D-107 |
| Loss task F1/F2 (group K64 zero-ablation), rescue F3, F4, F5, Study G | legacy; matched and GPTQ at INT6, INT4 (F5 and Study G also) | exploratory: rules written before the data, but the test sentences had been looked at (D-039) | D-041, D-058, D-059 to D-064, D-070 to D-085, D-107 to D-109 |
| Intervention-transfer study (main matrix, 27 languages, FLORES) | legacy, matched, GPTQ (INT6, INT4; INT16 floors shared) | pre-set rules (A14) but exploratory | D-088, D-094, D-105 |
| External panel (Wikipedia, six languages) | legacy, matched, GPTQ | same rules, fresh text; exploratory | D-095, D-111 |
| Recovery analyses R1 to R4, Llama-3.2-1B check | legacy only | post hoc / pre-registered-but-not-consistent | D-096 to D-099, D-087, D-090 |

Fairness of the three-arm comparison (D-101 to D-103): same sentences, features, groups, edits, analysis code; weights rebuilt per job and hash-checked; full-precision quantities bit-identical to the old runs in every job (bridges exact); INT16/INT12/INT0 stay the legacy rounding. The arms differ in more than one thing: legacy to matched RTN = grid and granularity; matched RTN to GPTQ = error compensation **and** calibration text (221,184 Wikipedia tokens in the 27 languages; RTN saw none).

## 2. The original tests, and what happened in each arm

| test (source) | legacy RTN | matched RTN | GPTQ | status | reading |
|---|---|---|---|---|---|
| **T1** INT7 selective minus control survival (D-014, D-106) | -0.0028, Holm p 0.514 | -0.0048, **Holm 0.042** | -0.0014, Holm 1 | legacy: confirmatory, not rejected; new arms exploratory | tiny; the matched-RTN crossing is near the threshold; with a factor 2 for two extra families it would be 0.084 |
| **T2** INT6 same | -0.0144, Holm 0.128 | -0.0153, **Holm 0.038** | +0.0016, Holm 1 | same | -1.4 and -1.5 pp in legacy and matched; gone under GPTQ because survival is near the ceiling (0.990 against 0.992) |
| **T3** tier slope (INT7+INT6) | -0.0048, Holm 0.990 | -0.0035, 0.984 | -0.0025, 1 | confirmatory (legacy); equivalent within ±0.02 in all arms | no tier effect on the selective-minus-control gap in any arm |
| Phase 1 survival by tier (descriptive) | INT6 low 69.4 %, mid 90.6 %, high 95.1 % | not repeated by tier here | GPTQ minus legacy INT6 +4.3 / +8.0 / +25.2 pp (high / mid / low) | exploratory | **the INT6 tier gap of the old result is largely a property of the legacy quantizer** (D-104) |
| **Belebele S0** (group removal beats matched controls) | effect -0.0001 [-0.0035, +0.0031] at FP | FP identical | FP identical | confirmatory (legacy): **not met**; S1 to S5 undefined | at INT4 the legacy model is at chance (0.236), matched 0.291, GPTQ 0.319 (FP 0.426); group-removal effect within ±0.5 pp in every arm and state (D-107): **the null is not an artifact of a broken 4-bit model** |
| **Transfer UTILITY** (diagnostic beats strongest baseline by ≥10 %, held-out) | INT4 met, INT6 not met → overall not met | INT4 met, INT6 not met → not met | INT4 met, INT6 not met → not met | pre-set, exploratory | main study: the verdict does not depend on the quantizer (D-105) |
| Transfer UTILITY, **external panel** | met (INT4 19.4 %, INT6 23.4 %) | met (28.7 %, 25.0 %) | INT4 29.6 % met, **INT6 2.6 % [-13.8, 16.8] not met** → not met | same | the legacy and matched "met" at INT6 is lost under GPTQ, where the error is near the floor (D-111) |
| H1 (error above the numerical floor) | supported | supported | supported | pre-set | holds in every arm, state, main and external |
| A3 (context term larger than weights term, group level) | main: mixed (INT4 supported, INT6 inconclusive); external mixed | main INT4 inconclusive; external mixed | main INT4 supported; external inconclusive | pre-set | not stable across arms: do not build on it |
| Study G pre-set H-G1 to H-G3, H-G6 | met | met | met | exploratory | FP-only parts identical across arms (same file) |
| Study G H-G4 (precision, down) | not met | not met | **met** | exploratory | a threshold effect: with two bootstrap errors all arms are met (D-109) |
| Study G H-G5a/b/alpha (repair at INT4) | not met (12 of 27 languages) | not met (10) | not met (4); alpha* above 1 in 21 / 13 / 7 of 27 languages | exploratory | the legacy INT4 "boosting helps" hint is a legacy-rounding effect (D-109) |
| F4 / F5 statements S1, S2, S3 (label predicts function) | S1 met, S2 undefined (2 events), S3 met | S1 met, S2 undefined (4), S3 not run | S1 met, **S2 not met (AUROC 0.49)**, S3 not run | exploratory | the label does not predict function under any quantizer where it can be judged (D-108) |

## 3. Size and uncertainty of the main effects (not only verdict words)

| quantity (median or mean over the stated units; interval as in the source) | legacy RTN | matched RTN | GPTQ | source |
|---|---|---|---|---|
| INT4 survival, mean over 27 languages | 12.7 % | 18.6 % | 30.5 % | D-104 (corrected, see D-104 update of 2026-10-29) |
| INT6 survival | 85.0 % | 89.7 % | 97.5 % | D-104 |
| INT4 loss increase (nats/token) | 0.590 | 0.385 | 0.130 | D-104 |
| Loss task E4, INT4 retention (median language ratio; FP-relative) | 1.54 [1.43, 1.62] | 1.36 [1.33, 1.39] | 1.22 [1.14, 1.24] | D-107 |
| E4 INT6 retention (mean of ratios) | 1.04 | 1.06 | 0.99 (all 'preserved' in 27 languages) | D-107 |
| Group patch recovers (INT4 loss gap) | 2.6 % [2.4, 2.8] | 2.3 % [2.1, 2.4] | 2.0 % [1.6, 2.3] | D-107 |
| Whole layer-12 residual patch recovers | 39.3 % | 35.2 % | 41.4 % | D-107 |
| Belebele INT4 accuracy, no edit (FP 0.426) | 0.236 | 0.291 | 0.318 | D-107 (corrected, see D-107 update of 2026-10-29) |
| Study G INT4 net harm at ×0.088, as a multiple of the FP value | 1.92 | 1.67 | 1.11 | D-109 |
| Transfer main, INT4 error over the numerical floor (median, 2,203 items) | 7.69 | 6.71 | 5.54 | D-105 |
| Transfer main, INT6 error over the floor | 1.88 | 1.75 | 1.52 | D-105 |
| Transfer external, INT4 fingerprint cosine | 0.754 | 0.808 | 0.876 | D-111 |
| F5 INT4 effect similarity W1, median over 1,487 features | 0.839 [0.829, 0.848] | 0.875 [0.869, 0.881] | 0.923 [0.919, 0.928] | D-108 |
| F5 INT4: share of features with W1 at the numerical floor | 0.3 % | 0.7 % | 11.7 % [9.9, 13.6] | D-108 |

All feature-level and per-language intervals are each arm's own, not paired contrasts (the group-level contrasts in D-104, D-105, D-111 are paired over languages and are the only paired ones; the external ones use six languages).

## 4. Do the external results agree with the main ones?

- **Structure and ordering: yes.** In both, the transfer error shrinks legacy > matched > GPTQ at 4 bits; at 6 bits the arms are close; H1 is supported everywhere; the weights term is the largest component of the feature-level error and the interaction the smallest (main D-105; external ordering of C_rel and W_rel in D-111).
- **Verdict words: partly.** Main study: UTILITY not met in all arms. External panel: met for legacy and matched, not met for GPTQ. The GPTQ pattern is the same in both (INT4 met, INT6 not met), so the external "met" of the two RTN arms at INT6 is the odd one out and rests on a 10 % rule with wide intervals (external legacy 23.4 % [12.8, 32.5]; main-study legacy INT6 10.9 % [-1.2, 21.4], D-094). A3 and H3 words move at thresholds.
- **Wikipedia overlap, stated plainly:** the GPTQ calibration corpus is Wikipedia (articles disjoint from the panel by id, same 27-language pool) and the external panel is Wikipedia text in six of them. Under GPTQ the panel is therefore in the calibration domain; the main study's FLORES text is not. The direction and size of GPTQ's advantage on both texts are similar in kind, which is **consistent with** no large domain-match inflation, but the design cannot separate domain match from error compensation, and matched RTN has no calibration text. Do not claim either.

## 5. What the F5 results say about activation survival and effect similarity

F5 (D-053, D-054, D-108; 1,487 stratified features, 27 languages) pairs, per feature and state, the **input-side agreement R1** (token-level correlation of the feature's activation in the quantized and full-precision model) with the **effect similarity W1** (cosine of the effect fingerprints of a single-feature ablation in the two models).

| statement | legacy RTN | matched RTN | GPTQ |
|---|---|---|---|
| Spearman correlation R1 with W1, INT6 (language-cluster bootstrap interval) | 0.533 [0.476, 0.573] | 0.520 [0.465, 0.566] | 0.575 [0.527, 0.615] |
| same, INT4 | 0.413 [0.346, 0.464] | 0.378 [0.319, 0.426] | 0.381 [0.320, 0.431] |
| AUROC of the binary association label (retained vs lost, F4 pairs) for W1 at the floor, INT4 | 0.256 (2 events, undefined) | 0.381 (4 events, undefined) | 0.490 [0.431, 0.560] (79 events) |
| share of features at the numerical floor, INT6 / INT4 | 89.7 % / 0.3 % | 90.6 % / 0.7 % | 92.4 % / 11.7 % |

What these support: (a) the **binary retained/lost label is not a reliable guide** to whether a feature keeps its effect (the pre-set S2 is not met where it can be judged); (b) the **continuous R1 is moderately related to W1**, more at INT6 than at INT4 in every arm; (c) in the earlier F5 controls the change is largely generic (swapping in another feature's direction lowers the similarity almost as much, 0.003 to 0.014; D-063, D-064).
What they do **not** support: that R1 adds anything **beyond** simple baselines (FP effect size, activation scale, firing rate, language) — that has never been tested; any causal reading (both quantities come from the same perturbed model, so a shared cause is the default explanation); any statement about held-out text or a held-out model (R1 is computed on the dev set; the files hold no read-side document IDs; no external F5 exists). At INT6 the outcome is near its ceiling for ~90 % of features, so the relation there rests on small differences above the numerical floor; at INT4 under GPTQ only 11.7 % of features are at the floor.

## 6. Confirmatory versus post hoc

- **Confirmatory (pre-set, legacy RTN only):** Phase 1 T1 to T3 (all null) and the Belebele family (S0 not met, S1 to S5 undefined). These stay exactly as written (`PROJECT_RESULTS.md` section 1). Nothing in the GPTQ extension changes their status; reruns on other quantizers are exploratory (D-106).
- **Pre-set rules but exploratory (examined sentences or families beyond the original):** Study G, transfer (main and external), F4, F5, loss task, rescue.
- **Post hoc:** the recovery analyses R1 to R4 (D-096 to D-099), the Study G extensions, the "matched for activation strength" re-analysis of Phase 1, the extension's three-arm contrasts (each arm's intervals are its own; no multiplicity rule across arms), the F5 relation R1 to W1 discussed above (the numbers existed before this plan; the plan's Stage 3 would be a second, still post hoc, look at the same 1,487 features).
- **Not available:** a held-out second model for F5 (Llama never had F5), Gemma-2-9B (another team; nothing here), any external F5.

## 7. The strongest claim the complete project can support, and the claims it cannot

**Can support (Gemma-2-2B, layer 12, Gemma Scope 16k, weight-only simulated RTN/GPTQ at INT4 to INT8, 27 FLORES languages):**
1. The 64 most language-selective features carry real function for their language's next-word loss (27 of 27 languages, three arms) — a causal effect of a defined edit, exploratory sentences.
2. The apparent selective vulnerability of language features to rounding is explained by activation strength; the pre-set tests give no robust selective penalty, and their verdict words are not stable across quantizers.
3. At 4 bits the effect of the group edit is amplified relative to full precision; the **size** of the amplification and of every INT4 error (survival, loss, transfer error, effect similarity) depends strongly on the quantizer, the **ordering** and the structure of the findings do not; at 6 bits the arms are close.
4. A measured repair is small: the group's full-precision patch closes 2 to 3 % of the INT4 loss gap, the whole layer-12 state about 35 to 41 %; neither number is an additive share or an upper bound.
5. The edit's transfer error is mostly generic (not specific to the feature's own direction; a modest direction-specific part exists on a small subset); a generic sensitivity measure predicts the held-out error at 4 bits in every arm (about 20 to 30 %) and at 6 bits only where enough error is left.
6. Survival-type measures (binary association label, activation correlation) are **not reliable stand-ins** for functional continuity: the label does not predict function; the continuous correlation is moderately related to effect similarity and is untested beyond baselines.

**Cannot support:** that quantization hurts low-resource languages **because** their language features break; that any of this generalizes to other models, layers, SAEs or quantizers (the Llama check was "not consistent", no 9B result exists here); that the INT4 changes are about language rather than generic; that a usable repair exists; anything about deployed integer kernels, activation quantization or the paired-precision audit (never run, Q29); that GPTQ's advantage is due to error compensation alone; that survival predicts function.
A paper-shaped statement of these is a **careful negative-and-limits result with a useful measurement-validity strand**, not a positive mechanism result. This sentence is the review's, not a claim about what any venue would accept.

## 8. Decision gate (plan Stage 1): proceed, narrowed

**Question of the plan:** when a feature's activation pattern changes less after weight quantization, does an intervention on that feature also keep more of its effect?

**Can the saved data answer it? Yes, in a bounded form.** The input inventory ([`f5_input_inventory_2026-10-28.md`](f5_input_inventory_2026-10-28.md)) shows one row per feature per state, aligned across states and arms (1,487 features, 27 languages); R1, support overlap, scale ratio, W1 and the full-precision effect size (dnll) are saved; the full-precision activation scale and firing rate are in the Phase 1 survival files (identical across arms). Not available: margins, read-side document IDs (so no claim of document-level independence), raw fingerprints and sufficient statistics (reproduction starts at the per-feature values), any external F5 (plan Stage 3 step 7 is unavailable and is to be marked so).

**Is there a defensible result to build on? Yes, but modest** (section 7). The F5 relation is a worthwhile bounded test of a measurement, not a rescue of the original hypothesis; a weak or baseline-explained result is an acceptable outcome (plan Stage 3 stop rule).

**Narrowing (to be written into the plan amendments, Stage 2):**
1. Stage 3 on existing Gemma files only, per quantizer and bit width (legacy, matched, GPTQ at INT6 and INT4), unit = feature, hold-out by **language cluster** (27) because contexts and sentences are shared; labelled post hoc; the near-zero-effect rule and the join audit with a reproduction control are fixed in writing **before** any outcome is read (plan update of 2026-10-28, gates 1 to 3).
2. Predictors: continuous R1 (primary); firing overlap and scale ratio (secondary); baselines FP effect size (dnll, effect norm unavailable), FP activation scale and firing rate from Phase 1, number of active tokens. No combined score chosen after seeing results.
3. Stage 4 as written.
4. **Stage 5 (Llama F5) is conditional with a hard time gate:** the Llama group effect at full precision was weak (median +0.048 nats per token, pre-registered P1 not met, D-090), so its F5 effects may be noise-dominated; a smoke test, a noise check of Llama's FP versus INT16 effect fingerprints and a timing test come first; given the 31 October deliverable date the job must be submitted no later than the evening of 29 October to have a result on the 30th; otherwise finish with the Gemma result and say why (plan Stage 5 step 7). The decision to submit is taken after Stage 3 is written, not before.
5. Stage 2 amendments: the two forward-looking plans named in the plan (`2026-10-26-publication-recovery-after-results.md`, `2026-10-26-survival-metric-and-margin-audit-addendum.md`) get dated notes saying: the GPTQ extension is complete (D-101 to D-111) and the all-results assessment is written; the F5 survival-versus-effect test is the one forward action authorized; the precision audit (Q29) remains unstarted and not part of this plan; expired budgets and deadlines are not carried over.

## 9. Limits of this review

Written from committed records, not from new runs; it inherits their limits (one model, layer, SAE; weight-only simulated quantization with floating-point execution; the paired-precision audit never run; the three arms differ in more than one factor; no multiplicity rule across arms; exploratory sentences). The R1-versus-W1 numbers are the committed F5 values, not recomputed here. The statement in section 7 about what a venue would accept is not made.
