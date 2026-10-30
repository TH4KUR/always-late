> **Current reporting pointer, 30 October 2026:** This is a dated assessment, preserved as written. Its earlier pending-review and publication-priority statements are historical. The 9B original-RTN results have since been reproduced and reviewed (D-127); GPTQ remains pending. Use the [final questions and answers](../FINAL_RESEARCH_QUESTIONS.md), [research paper](../../paper/research-paper/main.tex) and [current status](../CURRENT_STATUS.md) for the completed study.

# Publication reassessment after the main transfer results

> **Dated assessment, before external/recovery results.** For the 27 October evidence and execution state read [current status](../CURRENT_STATUS.md). The original analysis, search record and negative findings below are preserved; their pending-work statements are historical.


Current [novelty reassessment](../novelty_assessment_2026-10-26.md): broad claims have direct precedents; generic BF16/FP32 sensitivity is also known. The planned precision audit must change a substantive feature-audit conclusion to become a contribution; it is not yet a result.

Date: 26 October 2026. Post hoc; written after D-094. This is an assessment, not a new confirmatory result.
Execution priority: ../../docs/superpowers/plans/2026-10-26-publication-recovery-after-results.md
Reproduction: OMP_NUM_THREADS=4 ~/.venvs/anlp-torch/bin/python reports/publication_reassessment_2026-10-26/reassess_existing.py
Machine-readable results: audit.json (input SHA256 included).

## Decision

The present results are not enough to support the original practical diagnostic as the main
A-tier contribution. Preserve them; do not redefine the original failed utility criterion.
The strongest next step is to establish which conclusions remain valid under independent
data and controlled measurement precision. Stop expanding into new layers, tasks or more
gain values until those two issues are resolved.

The revised candidate paper question is:
**Which notions of SAE intervention preservation survive a change in model precision, and
which apparent successes/failures arise from the measurement and intervention definition?**

A defensible answer needs a robust transfer/behavioral boundary. A crossed design, a
Pearson-null table or a significant cosine change alone is not sufficient.

## What survives, what weakens

| Result | Current evidential value | Consequence |
|---|---|---|
| INT4 fixed-edit effects change substantially | Real measurement under the current bf16-body/fp32-readout instrument; about 7.7 INT16 reference floors | Retain as descriptive compression evidence; validate in another arithmetic regime before calling it intrinsic feature damage. |
| INT6 error is about 1.9 reference floors | Smaller signal with limited separation from numerical reference | Do not equate high cosine with equivalence, or a small detectable change with consequential failure. |
| Context exceeds weights for fixed-edit group loss | Endpoint- and intervention-dependent | Compare native edits explicitly; do not claim it explains native group vulnerability. |
| Weights dominate feature-effect vector error | A statement about a different estimand and population | Not a contradiction with group loss, nor by itself a new mechanism. |
| Averaged fingerprints look more stable | Strong descriptive aggregation effect, also present at INT16 | Quantify excess over matched reference and its decision consequence. Averaging suppressing noise is known. |
| Generic-direction diagnostic improves INT4 MAE | Feature-held-out result, with same-context expensive probes and 88% document overlap | It is not yet a cheap, document-general, quantization-specific audit. |
| Pearson survival adds little | Conditional negative result under one predictor family, sampled active contexts and noisy target | Do not conclude no relation or universal metric invalidity. Use an outcome-matched target and stronger calibration-only baselines. |
| Real/control specificity is weak | Original comparison uses different eligibility populations and different FP response scales | Resolve common support before a feature-specific claim. |

INT16 RTN is a small, genuine weight perturbation, NOT a pure measurement-noise oracle.
Failure to beat INT16 predicts lack of specificity; it does not prove that every Q effect is
numerical noise. Conversely, success over INT16 does not prove precision-independent mechanism.

## New descriptive checks of existing data

The script reproduces every original model's held-out MAE before restricting evaluation.
It does not change fits, lambdas or the calibration-selected comparator.

### Document overlap matters

Restrict held-out-feature items to source documents absent from ALL calibration-feature
contexts (including ineligible ones). Only **136 items on 65 documents** remain, versus
1,066 originally eligible held-out items.

| State | Original diagnostic gain | Gain on unseen-document subset |
|---|---:|---:|
| INT16 | 23.4% | 29.4% |
| INT6 | 10.9% | 0.46% |
| INT4 | 21.5% | 6.73% |

These restricted estimates have NO new uncertainty intervals, are post hoc and are based on
a small selected subset. They do not establish that a gain vanished. They do establish that
the headline is sensitive to the evaluation population. Fresh cross-corpus prediction with
locked coefficients is now the decisive test, not another within-corpus feature split.

### The native-amplitude endpoint changes the context interpretation

At INT4, median net harm across 27 languages is:
- fixed FP edit in Q context/Q suffix: .8479 nats/token;
- native Q edit in Q context/Q suffix: .4325;
- fixed FP edit in FP context/Q suffix: .4284.

The median of the PAIRED native-minus-weights-only difference is .01384, with median
absolute difference .05370. These are not differences between medians and no equivalence
is established. The identity native_total = W + (native_QQ - fixed_FP_Q) holds per language.

Conclusion: context dominance for the fixed FP edit cannot be transferred to native edits.
Changes in context and amplitude can offset. Completing a full three-factor causal account
would require missing cells, not another interpretation of the five available cells.

### Current specificity is not a fair same-effect-scale comparison

Requiring BOTH real and swapped directions to satisfy their respective existing FP-effect
eligibility leaves **986 paired items**, versus 2,203 eligible real items. Requiring all four
controls as well leaves only **323**. Even among the 986 pairs, the median real/swapped
FP effect-norm ratio is **2.02**.

Thus shared input perturbation norm does not equalize output-effect scale. Per-kind floors
also create different normalization denominators. Joint eligibility alone does not repair
this comparison. The original specificity table must remain qualified; its small gap cannot
establish equivalence or selective damage.

## The gradient claim is overbroad

The required target was a first-order logit-effect vector. The executed A9 baseline supplies
first-order NLL scalars. These are useful baselines for NLL effects, but a poor result when
predicting vector-error magnitude does not establish that gradient prediction fails at its
own target. The true vector-JVP baseline remains unrun. Evaluate the existing gradient scalars
against signed loss effects using saved records first; do not spend GPU hours to defeat a
baseline on a target it does not estimate.

## Revised claim map

Supported descriptions now: measured INT4 transfer drift; context dependence; distinction
between native/fixed edits; different scalar/vector endpoints; limited conditional Pearson
utility; averaging and precision-reference sensitivity.

Still needed for a strong submission: independent-corpus/frozen-model performance; fair
common-support controls; arithmetic sensitivity of small-effect measurements; an outcome with
behavioral meaning; broader quantizer/model evidence for a broad claim.

Retire as headlines for now: quantization-specific diagnostic; universal metric failure;
selective language-feature destruction; restored language comprehension; “gradients fail”;
“context is the main source of native intervention harm.”

## Literature boundaries and search record

Duan's intervention-validation future work remains the parent question, not a claim that
its numerical protocol is correct. FEGA already studies effect clouds. Chang et al. already
separates noisy activations from restored late weights at a loss-level endpoint.
Vaidyanathan et al., *The Curse of Multiple Mediators: Hidden Interaction Effects in
Activation Patching*, https://arxiv.org/abs/2606.27510v1, primary relevant pages read this
turn: interaction and intervention-distance dependence have direct theoretical/empirical
precedents. Our factorial is not their NIE estimand; interaction existing does not invalidate
ours, but cannot be sold as a new discovery on its own.
The October strength-control and attribution papers remain direct baselines. Margin flips
are already studied by 2608.06564 and 2610.00694.

Three additional web searches combined sparse autoencoders, interventions, bf16/numerical
precision and measurement floors. Returned thesis/project material and tutorials do not
establish a novel precision effect. No identical completed protocol was found in this bounded
check; no broad priority claim is justified. See the existing novelty register for details.

## Current reporting context, 30 October 2026

This dated assessment is retained as a historical record; its old pending-work and publication-priority statements are not the current project status. [Current synthesis](../FINAL_RESEARCH_QUESTIONS.md), [research paper](../../paper/research-paper/main.tex), [current status](../CURRENT_STATUS.md). The completed paper reports a controlled evaluation across three Gemma-2-2B quantizers, task/control checks and scoped scale/architecture extensions. It credits direct precedents and makes no first-ever or acceptance claim. A limited model/layer/SAE scope bounds generalization rather than invalidating controlled effects. Positive associations, supported intervention contrasts and informative nulls all belong in the final account. Original predictions and their outcomes are unchanged.

The 9B original-RTN analysis was reproduced from raw files and reviewed (D-127); GPTQ remains pending, matched RTN is unrun. Belebele's primary interval is in correct-option NLL, not accuracy points. Continuous survival adds about 3–4% predictive improvement beyond baselines, below the 10% convention but not zero. F5 effect changes can coexist with high own-identity retrieval. These later clarifications govern current interpretation where the dated prose differs.
