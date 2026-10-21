# Functional validation across precision: design (D-039, D-040)

Written 2026-10-21 by Claude after the user's Q16 answer (A/B/C/F). Plan (tasks, tests, gates): [`../plans/2026-10-21-functional-validation.md`](../plans/2026-10-21-functional-validation.md). Live status: [`reports/functional_validation/STATUS.md`](../../../reports/functional_validation/STATUS.md). Selected direction and the user's wording: [`next_steps.md`](../../../reports/association_retention/next_steps.md#selected-functional-study-direction-d-039). Sources: [`reports/research_sources.md`](../../../reports/research_sources.md).

Status: **design, not results.** Every number below that is a setting (sizes, doses, margins) is a proposed default, recorded as D-040 and Q17 in the review file; the user may override. Nothing here was measured on the new tasks yet.

## 1. Questions

The completed D-035 study measured language **association** (which language a feature responds to most). It said nothing about **function**. The functional questions, in the order the evidence has to be built:

| id | question | what it would show | what it would NOT show |
|---|---|---|---|
| F1 | **Necessity in full precision.** Does removing a language's selective feature group (G_l) harm next-token prediction (and task scores) on language l more than removing an equally active, non-selective group of the same size? | that the features carry function beyond being active | exclusive language specificity; semantic meaning; that the SAE direction is the model's own unit (P06) |
| F2 | **Persistence across precision.** Is that effect (F1) kept, attenuated, amplified or reversed in each quantized model (INT8/7/6/5/4), each against its own unintervened baseline? | whether quantization preserves the causal role of language features, not only their activation pattern | a mechanism for any change |
| F3 | **Localisation of quantization damage (rescue).** Of the per-language loss increase that quantization causes, how much is recovered by restoring only G_l's full-precision activations in the quantized model, compared with matched controls, all SAE features, the SAE error term, and the whole layer-12 residual? | where at layer 12 the damage to language l sits (language features, other features, unexplained error) | that the quantized model natively retained the function (a different estimand: F2) |
| F4 | **Association versus function.** Are features whose language association was LOST at precision b (D-035) less functionally important at b than features that RETAINED it, at equal native activity? | whether association retention predicts functional retention (two-sided; a null has a pruning precedent, P31) | causation of the loss; this is exploratory (devtest statuses are conditioned on already examined data) |
| B | **SAE fidelity** (Q15/M2): normalised reconstruction error, active-feature counts and residual norm by language and precision, from the same baseline passes | whether the frozen dictionary still describes the quantized model's layer-12 activations, which bounds how far F2 and F4 can be read as statements about the model rather than about the SAE | whether task-relevant language information changed |
| A | CPU-only analyses of the existing D-035 files (loss against baseline margin, share and scale within languages; floor comparisons; English-excluded views) | describes where loss concentrates | none of it is causal |

Design principle, stated once because it decides the controls: **a zero-ablation changes nothing on text where the feature never fires.** Language-selective features rarely fire on other languages, so "harm on the target language larger than on other languages" is largely implied by selectivity and is not evidence of function. The decisive contrast is therefore **against activity-matched controls on the target language's own text** (F1); the off-target panel is a manipulation check (the edit should do little where the features are silent) and a probe of non-specific damage.

## 2. Fixed setting (inherited, unchanged)

Gemma-2-2B base, bfloat16 on the GPU; Gemma Scope layer-12 residual SAE (canonical 16k, JumpReLU), hook `blocks.12.hook_resid_post`; RTN per-output-channel weight quantization at INT8/7/6/5/4 by the vendored function (restore the original weights before each condition); `mask` regime (one BOS, position 0 never edited and never a target); the frozen D-035 cohort (`reports/association_retention/frozen_cohort_manifest.json`, dev-selected features, 27-language roster of `reports/g0/phase1_languages.json`). INT16/INT12 and INT0 are controls: the numerical floor for F2 (a null result at INT16 must look like INT0) and the restore-only check. bf16 round-off in the forward pass may exceed the weight noise of INT8/INT7: those two precisions are read against the INT16/INT12 floor, never against an ideal zero.

## 3. Data and task coverage (stated, not silently reduced)

**T-LM (all 27 languages): next-token negative log-likelihood** of FLORES-200 **devtest** sentences. The features were selected on **dev**, so devtest is held out *with respect to selection*; devtest was examined in D-035, so every T-LM result is **exploratory** (D-039). Sentence ids are drawn once with a fixed seed from the 1,012 devtest ids and are the same ids in every language (FLORES is parallel, so item i is the same content): 150 pilot ids, 450 evaluation ids, the rest reserved. Score: sum of token NLL over content positions at t >= 1, per sentence (BOS never a target); per-token and per-sentence quantities are both kept.

**T-MC (languages with measured headroom): Belebele** (Bandarkar et al., ACL 2024; 900 four-way multiple-choice reading-comprehension items, parallel across 122 languages, passages from FLORES-200; CC-BY-SA 4.0, revision `7899cdfa4e1e0d733fd77c848e2c273cb1d32be2`, files for the 27 roster languages on Ada under `phase1/data/belebele`). Not yet verified: the exact prompt that a 2B base model can use above chance. The prompt format (letter choice versus cloze likelihood) and the language coverage are **decided on a calibration split in full precision only, before any intervention**, by these rules, which are fixed now:

1. Items are split by the FLORES origin of their passage, established by exact sentence match against the FLORES files (checked on Ada, English: 488 passages, 267 from dev, 220 from devtest, 1 matching neither and excluded and reported): **calibration items** are those whose passage comes from FLORES **dev** (about 495 items per language), **evaluation items** are those from **devtest** (about 405). Full-precision calibration involves no intervention, so using dev-origin text there leaks nothing; the evaluation items are text on which the features were never selected. No passage is on both sides; every language uses the same split (passages are parallel).
2. The format with the higher macro-average full-precision accuracy over the calibration items wins; ties go to the letter format.
3. A language is covered by T-MC if its full-precision calibration accuracy has a 95 % lower bound above chance (0.25) plus 0.05. The uncovered languages are listed with their accuracies in every report; no claim about them from T-MC is made. (Expected: a minority of the 27; the low-resource tier may be mostly uncovered, which is itself a finding about headroom.)
4. Pilot intervention effects on Belebele (before the freeze) use calibration items, which are the dev-origin text the groups were selected on: such effects are inflated by selection and are never used as results, only to check the pipeline.

T-MC is the held-out task evaluation (frozen before scoring, section 7). Score per item: the log-probability of the correct option among the four, from the final-position next-token distribution (letter format) or the length-normalised option likelihood (cloze); accuracy is derived.

**Secondary outputs** from the same passes (no extra cost): (i) *output-language compliance*, the probability mass the model puts on tokens of the target script at each position, defined for the 14 non-Latin-script languages and kept strictly separate from correctness (D-039); (ii) *SAE fidelity* (B): per language and precision, `||x - x_hat||^2 / ||x||^2`, L0, mean `||x||`, on the unintervened pass.

## 4. Feature groups and controls (frozen before any new outcome)

**Source.** Only dev-based quantities from the original `mask/eq` `selectivity.pt`: the 8,086 selective features, their dev peak language, dev mean activation `mu_l(f)` (zeros included) and dev firing rate `r_l(f)`. No devtest outcome and no quantized-model information enters the primary groups.

**Primary group G_l^K.** The K selective features whose dev peak is l, ranked by `mu_l(f)` (dev activation mass on their own language), top K. **K = 64** (every language has at least 73 selective features, Portuguese is the minimum), with K = 16 and the full group (sizes 73 to 1,719) as nested dose/size sensitivities. Group sizes are therefore equal across languages in the primary analysis, which keeps the intervention dose comparable; English (1,719 selective features) is not allowed to dominate pooled numbers.

**Matched controls C_{l,r}^K (r = 1..R).** The only non-selective features of the cohort are the 2,860 supported non-selective features (the whole matched-control pool, `supported & ~selective`; this is small, so draws overlap and cannot give an independent null: stated, and the null is read as a reference set, not a sampling distribution). For each language l and size K, the pool features and the group are cut into cells by (dev firing rate on l, quintile) x (dev mean positive activation on l, tercile); each draw takes, without replacement within a draw, as many pool features per cell as the group has (the existing `matched_sample`), seeded by (language, K, r); cells where the pool is short are recorded and drawn in full. R = 5 for K = 64 and K = 16, R = 3 for the full group. Matching is on the dev statistics for language l **because the amount of activation removed from l's text drives the size of an ablation effect**; the actual removed activation norm is measured for every edit and reported (the matching is checked, not assumed), and the **decoder-row norms** of group and controls are compared and reported (if the decoder rows are not unit-norm, a mismatch is a confound; Makelov et al.'s overshadowing warning, P36).

**Other-language groups.** G_{l'}^K ablated on text l (l' != l) are used only as a descriptive reference (they remove little activation on l by construction; section 1).

**F4 subgroups (exploratory, later).** G_l split into features retained and lost at precision b by the D-035 status (`feature_status.csv`; uses devtest outcomes, hence exploratory), size-matched and matched on **native** activity at b.

## 5. Interventions

All edits happen at the layer-12 residual of the model being studied and **preserve the recipient's SAE reconstruction error**: with `a = SAE.encode(x)` (computed in float32 from the recipient's own residual) and decoder rows `d_f`, the edited residual is `x' = x + sum_{f in G} (a'_f - a_f) d_f` (cast to bf16). Nothing outside G changes; identity (`a' = a`) returns `x` exactly. Applied at all content positions of the prompt; position 0 (BOS) and padding untouched.

| intervention | `a'_f` for f in G | question |
|---|---|---|
| **zero-ablation** (primary) | 0 | necessity (F1, F2) |
| **resample-ablation** | `a_f` of a random position of other-language text of the same model (the whole group vector of one position, so co-activation is kept; pool from FLORES dev, never an evaluation item) | necessity with in-distribution replacement (Heimersheim and Nanda M07 recommend resampling over zeros; zeros are off-distribution) |
| **mean-ablation** (sensitivity) | the dev mean activation of each group feature on **non-target** languages (`mu` of the other 26 languages averaged), a fixed in-distribution value | the Heimersheim and Nanda comparison of zero, mean and resample replacement |
| **scaling** (dose) | `alpha * a_f`, alpha in {0.5, 2} (0 is zero-ablation, 1 is identity) | direction and dose-response, improvement as well as degradation; |
| **sham** | numerically `x' = (x + d) - d` in float32 then bf16, no edit intended | the size of numerical noise in the edit path; must be about 0 |
| **FP-donor patch (rescue, F3)** | `a_f` from the full-precision model at the same tokens and positions (identical inputs, so the alignment is exact), injected into the quantized recipient: `x'_b = x_b + sum_{f in G} (a^FP_f - a^b_f) d_f` | localisation (F3) |

**Difference-in-means baseline (optional, after the pilot).** Directional ablation `x' = x - (x . u) u` of a dev-based language direction `u` (mean residual on language l minus mean on the other languages, normalised; Deng et al.'s directional ablation, P02), same items and doses: it tests whether the SAE group adds anything over a simple direction. Run only if compute allows (it is one extra configuration per language).

Bounds for F3 (same recipient, same tokens): **all features** (`G` = all 16,384: replaces the reconstructed part by the donor's, keeps the recipient's error term), **error term** (replaces the recipient's error by the donor's, keeps the recipient's reconstruction), **full residual** (`x'_b = x^FP`). The decomposition full = reconstruction + error is reported, then language-l's share of the reconstruction part, then the matched-control share.

**Native versus rescue (D-039).** The primary cross-precision comparison (F2) uses only each model's OWN activations (zero/resample/scale are native). FP-donor patching is a separate experiment about where damage sits, not evidence that the quantized model preserves the function.

**Dose bookkeeping.** For every edit record `||Delta x|| / ||x||` (mean over edited positions) and the mean number of group features active. Identical coefficients do not mean identical strength across precisions (feature scales shrink at INT4); a perturbation-norm view (effect per unit `||Delta x||`) is reported next to the raw effect, and is the comparison to quote if the removed norms differ materially between precisions.

## 6. Estimands, uncertainty and decision rules (fixed before evaluation)

For an item i, model precision b and intervention I: **harm** `h_i^b(I) = NLL_i^b(I) - NLL_i^b(none)` in nats per token (T-LM) or `-(logp_correct^b(I) - logp_correct^b(none))` (T-MC); negative harm is an improvement and is reported, never clipped. Each precision is compared with **its own** unintervened baseline on the same items (paired).

- **Functional effect (F1/F2 estimand).** `E_l^b = mean_i h_i^b(G_l^K) - mean_i mean_r h_i^b(C_{l,r}^K)` over the language-l items: harm beyond activity-matched controls. `E_l^b > 0` means the selective group matters more than equally active non-selective features.
- **Retention ratio.** `rho_l^b = E_l^b / E_l^FP`, reported only where the full-precision lower interval bound for `E_l^FP` exceeds 0 (and for the macro-average over covered languages). A precision is **preserved** if the 95 % interval of `rho` lies inside [0.75, 1.33], **attenuated** if it lies below 0.75, **amplified** if above 1.33, otherwise **inconclusive** (non-significance is never read as equivalence, D-039). The band is a proposed default (Q17).
- **Rescue (F3).** `Rec^b_l(P) = (NLL^b(none) - NLL^b(P)) / (NLL^b(none) - NLL^FP(none))` for a patch P, reported only where the denominator exceeds 0.02 nats per token with a lower interval bound above 0 and always next to the raw change; raw nats are the primary quantity.
- **Uncertainty.** Paired **cluster bootstrap** over sentence ids (T-LM: the same id in all 27 languages is one cluster, because translations are not independent, Field and Welsh M04; T-MC: passages are the cluster), 2,000 draws, percentile 95 % intervals, seed fixed; the same resampled clusters are used for every precision and intervention, so differences between precisions are paired. Percentile intervals of ratios and thresholded fractions can be biased (seen in D-035): the observed value is the estimate, the interval is descriptive, and bootstrap means are shown.
- **Confirmatory family (T-MC only, frozen before scoring).** (i) FP: macro-average `E^FP` over covered languages greater than 0; (ii) for each of the five quantized precisions the macro-average `rho` classification above. One-sided/two-sided choices and Holm over the five precision statements are written in the evaluation freeze file; everything else (T-LM, per-language effects, F3, F4, subgroups, doses) is exploratory and labelled.
- **No claims from a single null.** A zero effect of a K = 64 group does not show irrelevance (redundancy, task dependence, prompt distribution: Heimersheim and Nanda M07); the K sensitivity and the resample ablation are reported with it.

## 7. Evaluation protocol and the freeze

1. **Pilot (full precision, small):** sham about 0; edit path parity (cached lower half + edited upper half equals the full forward bit for bit on unedited input); throughput measured; Belebele calibration accuracies; the FP T-LM effect on pilot ids. These results may change *engineering* (batch sizes, which configurations fit the budget) and the Belebele format/coverage by the rules in section 3. They may not change K, alpha, R, the matching cells, the estimand definitions or the decision rules.
2. **Freeze** (a dated file `reports/functional_validation/evaluation_freeze.json` with hashes of the cohort manifest, group lists, control draws, item ids, prompt format, covered languages, doses, margins and the analysis script commit), committed **before** any T-MC evaluation item is scored or any evaluation T-LM intervention is run on a quantized model.
3. **Evaluate** all precisions with the frozen settings. A pre-freeze change is a deviation, recorded with its reason.

## 8. Compute (estimates to be replaced by the pilot's measurement)

One precision = one model state; lower 13 blocks are run once per batch and cached, then each configuration re-runs only blocks 13..25 (TransformerLens `start_at_layer`), the final norm and a chunked unembedding. Estimated token passes per precision for T-LM: baseline plus about 20 configurations per language (zero at K = 16/64/full, resample, two doses, sham, controls on the target text only), about 9 million tokens per precision, about 53 million for six precisions; T-MC about 7 million per precision for about 12 covered languages with R = 3 and 200 evaluation items. At an unmeasured 1.5 to 3 thousand tokens per second on an 11 GB 2080 Ti that is several hours per precision-group; at most two jobs at a time. **Reduction order if the pilot says it does not fit (fixed now): R, then off-target panel, then the T-MC item count, then secondary doses, then INT7; never K, the primary intervention or the FP and INT4/INT6 conditions.**

## 9. Prior art and novelty status

Checked 2026-10-21 ([`prior_art.md`](../../../reports/functional_validation/prior_art.md), register P30 to P38; two central claims re-read by me, the rest agent-reported). **No paper combining language-selective SAE feature groups, a weight-quantization ladder, native and FP-donor interventions and multilingual task scores was found**; search coverage is limited, re-check before submission. What is **not new**: FP-donor patching to localise quantization damage (Chang et al., EMNLP 2025), restoring SAE features to full precision (Sathyanarayanan, arXiv 2609.23125, activation quantisation), language-feature ablation hurting its language in full precision (Deng, Tang, Andrylie), steering directions persisting under quantization (Bhandari and Wade). **What would be new, if the results earn it:** the language-group framing with activity-matched controls across the RTN ladder (F2), per-language localisation (F3), and association-versus-function (F4). **Scoop risk:** Duan (arXiv 2606.03002, the replicated paper, same SAE/layer/model/quantizer, submitted to TMLR) names causal validation as future work. A precedent for a null in F4: Borobia et al. (arXiv 2603.25325) find that geometric feature survival does not predict causal importance under pruning, so F4 is two-sided and a null is an expected, reportable outcome.

## 10. Risks recorded now

- **Headroom floor:** a 2B base model may be near chance on Belebele in many languages (coverage rule, section 3); T-LM covers all 27.
- **Off-manifold edits and drift:** the frozen SAE was trained on the full-precision model; at INT4 its reconstruction error is larger (fidelity B measures it); the error term is preserved, but `a` itself may be a poor description. Interpret INT4 with B beside it.
- **Redundancy:** a missing effect is not proof of absence (section 6).
- **Group definition:** top-K by dev mass favours features that are common; the K sensitivity and the full group are reported.
- **Control pool is small (2,860):** a reference set, not an independent null (section 4).
- **Language length and tokenisation:** NLL is per token; token counts per sentence differ across languages and precisions share the tokens (paired), so within-language comparisons are not affected; cross-language comparisons use per-language effects and ranks, not pooled raw sums.
- **Researcher degrees of freedom:** section 7; every deviation is logged.
