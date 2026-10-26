# Research strategy: intervention transfer under quantization

> **Update 2026-10-26 (Claude, documentation sweep; Codex's text below is unchanged):** (1) The user replaced Llama-3.2-1B by **Gemma-2-9B with Gemma Scope** (`layer_20/width_16k/canonical`) as the second model (D-089). Read every 'Llama' below as the second model. The 9B runs are done by another team on another cluster (handoff 040, branch `gemma9b-confirmation`), so this study runs its Gemma-2-2B part first and its second-model arm after that branch is merged. The Llama check finished on 2026-10-26 and is reported as exploratory only. (2) Verified and adopted with corrections (D-088): [verification](../../../reports/intervention_transfer/verification_2026-10-26.md) (Chang EMNLP 2025 §6.1-6.2 is the loss-level precedent, so C > W alone is not a contribution; the gradient baseline 2610.15850 is mandatory; Bhandari 2609.06473 added; three citation fixes). The current order of work is the NEXT STEP block of `reports/functional_validation/STATUS.md`.
Date: 26 October 2026. Author: Codex. Execution owner: Claude when the user hands this plan over.
Status: a research design, not authorization to submit jobs from this chat.
Main plan: ../plans/2026-10-26-intervention-transfer-main-and-contingencies.md
Evidence register: ../../../reports/publication_strategy_2026-10-26/future_work_and_novelty.md

## Objective and scope

Build an explanatory paper around **when activation-based feature survival can and cannot
predict whether an intervention still does the same thing after quantization**. Explain the
observed response changes and test an audit that predicts them on new inputs. The working
title is *When Does Feature Survival Predict Intervention Transfer?*

The first new execution is capped at 48 elapsed hours, at most two concurrent Ada jobs,
and a planned ceiling of 32 GPU-hours including conditional follow-up. Queue availability
and throughput are not yet verified. A complete A-tier paper may need later replication
with another quantizer or model scale; the two-day deliverable is evidence for or against
a coherent paper argument, not a guaranteed submission-ready discovery.

Do not mutate frozen Study G settings or retroactively relabel examined data. Preserve the
27-language, 300-sentence, three-draw Study G floor. This new study uses its same population
for group-level explanatory analysis and the established 404-feature F5 control subset
with up to 16 contexts for feature-level diagnosis. Missing or unsupported cells stay
visible. The six-language new-corpus validation panel is a distinct external-validity
sample, not a replacement for the 27-language study.

## Synthesis of what we actually know

| Observation | Consequence for the next explanation | What it does not establish |
|---|---|---|
| Selective-minus-control survival changes sign when magnitude matching is added; the tier slope is near zero after activation-scale adjustment | Measurement scale and comparison design are central. Controls must match perturbation as well as activation statistics. | Resource tier or activation scale causally mediates damage. |
| Activation survival falls strongly at low precision; language association is often lost through a changed peak rather than silence | Correlation, selectivity margin and causal effect are different endpoints. | Lost computation or lost language competence. |
| Group and whole-language activations shrink similarly by ~15% at INT4 | A language-specific attenuation story is incomplete. | A uniform rescaling will restore model behavior. |
| K64 zero-ablation harms own-language loss in Gemma, with a larger effect at INT4 | Changed response sensitivity is plausible even when feature activation shrinks. | Features become more important, or this will replicate in Llama. |
| Language-group FP-donor rescue recovers ~2.6% of the INT4 loss gap; whole-residual rescue ~39.3% | Necessity under deletion differs from responsibility for compression error. Hybrid interventions need interaction-aware interpretation. | The remaining 60.7% is an additive independent downstream share. |
| F5 feature and swapped-direction effects drift similarly; fixed amplitudes change little | Generic response changes deserve direct testing. | Downstream weights cause the drift: background residuals still changed in that control. |
| Fine amplification gives a small INT4 improvement, but inverse shrinkage does not predict optimal gain (rho -0.170; reliability .917) | Simple amplitude compensation is not supported at this resolution. | No possible repair, or the tiny improvement is a useful practical method. |
| State-defined groups change membership but do not give stronger deep-attenuation effects | Chasing a better selective group is not currently supported. | Feature identities have no functional meaning. |
| Belebele group effects are null, including broad group ablation; prompt cross-check favors the original prompt | Own-language next-token loss cannot stand in for comprehension. | No function on any task; a ten-point prompt difference proves task irrelevance. |
| Llama log reports an interim 21/27 positive FP effects and heavy-tailed controls | Architecture/SAE dependence and control quality are real risks. | A completed final replication verdict; the local final result directory is absent. |

Read sources before treating rounded values as paper numbers:
reports/phase1_summary.md; reports/functional_validation/RESULTS_SUMMARY.md;
gain_study/README.md; extension_d/report.md; state_defined/results/compare.csv;
STATUS.md; the dated handoffs. The summary's Llama-access-blocked lines are stale relative
to STATUS. Preserve old text and add dated corrections, never delete the history.

### New descriptive check on existing data

The accompanying reanalyse_existing.py fits each language's Q net-harm curve to k times its
FP curve over the four existing deep-attenuation doses, with no intercept.

| State | Median k, net harm | Median k, control harm | Median relative residual | Across-language rho(net k, control k) |
|---|---:|---:|---:|---:|
| INT6 | 1.030 | 1.042 | .019 | -.304 |
| INT5 | 1.094 | 1.151 | .031 | .043 |
| INT4 | 1.520 | 1.516 | .127 | .017 |

This is a post hoc fit to four points, without propagated sampling uncertainty. It is not a
curvature estimate. Similar INT4 medians suggest a shared population-level increase in
sensitivity; their near-zero languagewise correspondence and occasionally poor fit argue
against declaring a single generic scalar explanation. The 1.52 fitted median is not the
published 2.14 macro mean of per-language zero-ablation retention ratios: different data
summaries/estimands must not be interchanged.

## MAIN: explain native transfer failures with independently controlled components

Future-work anchor F1 explicitly requests causal validation under quantization; F3 requests
downstream validation. The particular decomposition is our proposed response to those requests.

Let h_p be a cached full-sequence residual from prefix state p, F_s the suffix and output
readout in state s, and v_f the fixed FP decoder contribution on the same content positions.
For a matched context x:
`D[p,s,f](x) = center(F_s(h_p - v_f) - F_s(h_p))`.
Use each cell's own unedited baseline. Evaluate p,s in {FP,Q}. Compare native-amplitude
D[Q,Q] separately; keep feature direction, positions and FP amplitude fixed across the core.

- C = D[Q,FP] - D[FP,FP] measures the context contrast at an FP suffix.
- W = D[FP,Q] - D[FP,FP] measures the suffix-weight contrast at FP context.
- I = D[Q,Q] - D[Q,FP] - D[FP,Q] + D[FP,FP] is the interaction.
- D[Q,Q] - D[FP,FP] = C + W + I exactly before aggregation.

The identity is accounting, not a theorem of novelty. Norms/cosines of these vectors are
not additive shares. Hybrid forward passes may be off-distribution; report unedited hybrid
loss, compare both paths of the factorial, and avoid natural-mediation language.

### Scientific hypotheses, deliberately two-sided

H1: native effect error requires more than activation drift or a scalar change in perturbation
size. Test against magnitude, residual norm and generic-direction baselines, not just chance.
H2: independently controlled background and suffix contrasts explain a reproducible portion
of this error, with an interaction that is explicitly measured.
H3: generic sensitivity predicts some transfer error, but feature-/context-specific residual
variation may remain. Either result must be described at its actual scope.
H4: per-context transfer can differ from similarity of averaged fingerprints. Test the same
feature-context pairs rather than comparing two separately selected context clouds.

Mechanistic guide, not a claimed new formula:
for small edits t v, loss change has a first-order gradient term and a second-order
directional-curvature term. Coarse zero-ablation and the large Study G gains do not estimate
a local derivative. If the controlled results motivate a local account, collect signed
finite differences on a prespecified subset at t = +/- .05 and +/- .10 of the same FP edit.
Do not infer Hessian/Fisher behavior from a quadratic interpolation of two positive gains.
Gradient attribution is an existing baseline (2610.15850), not our invention.

### Measurement and inferential commitments

Primary target: continuous per-context transfer error in centered-logit effects, normalized
by the FP effect norm plus a calibration-only numerical-floor term. Also report unnormalized
error, norm changes, cosine and signed loss changes. Report the ratio of error norms to floor;
a small cosine drop or tiny effect should not determine a paper's conclusion by itself.

Eligibility threshold: FP effect norm must exceed five times the calibration sham/rounding
floor; use a floor measured in the same precision, batch regime and readout. Keep and report
counts and absolute outcomes for excluded cases. Freeze the rule before Q test effects are read.
If exact shams are zero, use the repeated-batch/INT16 floor and document it.

Use:
- the real decoder direction, the established matched swapped direction, and norm-matched
  random directions, at the same layer and positions;
- float32 edits/readout, recorded residual execution dtype, BOS excluded, reconstruction error
  retained, split/full equality and same-model hybrid identity gates;
- document/content-cluster uncertainty; reused features and translations require crossed or
  hierarchical resampling, not iid tokens;
- per-language and per-model reporting, with no pooled-bitwidth correlation as the main result;
- a held-out evaluation of generic sensitivity measured using independent directions.
  Fit a simple regularized model on calibration data, compare a nested model adding the
  generic diagnostic to activation correlation, FP magnitude, firing rate and residual norm.
  Include a directional-gradient baseline only if it fits the budget.
- deterministic feature-group holdouts plus document holdouts. Fit all regularization and
  floors on calibration. Do not measure the exact test feature's intervention and present it
  as a cheaper predictor of that same outcome.

Prespecified decision criterion for the diagnostic: at least 10% relative held-out MAE
reduction over the strongest prespecified baseline, with paired cluster interval excluding
zero improvement, separately in both models. This is a project utility criterion, not a
universal publication threshold. Report continuous estimates and uncertainty even when it fails.

A replication need not yield identical C/W/I orderings. A model-by-component interaction can
be interesting if estimated precisely and independently validated; it does not rescue a failed
universal claim. Unknown/high-variance results are inconclusive, not null proof.

### What would earn the paper

Three linked pieces: (a) explain the existing activation/ablation/rescue puzzle, (b) establish
the explanation beyond a single FP/Q comparison and model, (c) show useful prediction of
held-out intervention changes, or a behaviorally meaningful correction grounded in that
explanation. Four factorial cells and a suffix-only counterexample alone are insufficient.
A broad deployment claim additionally needs a validated second quantizer; simulated RTN
and one SAE per model must remain explicit limitations until then.

## Conditional secondary directions: at most ONE within the same time cap

### A — Where does an intervention's effect lose continuity?

**Trigger:** averaged fingerprints obscure sizable per-context errors, or C/W interaction is
large and the final-readout decomposition does not locate its origin.
**Future-work anchor:** F2, FEGA p.22, explicitly proposes successive-layer measurement.
**Question:** does quantization alter an intervention immediately, or amplify a small
difference during later propagation? Is that change shared with matched directions?

Record the paired difference between edited and native residuals after each remaining block,
using hooks on the SAME core forwards. Compare FP/Q effect norms, directions and effective
rank within each layer's own width and normalization convention. Across adjacent layers,
quantify change in within-layer paired transfer error; do not compare unrelated raw
coordinates or interpret CKA as causal proof. Retain final logits and the task endpoint.
Predeclare five checkpoints (SAE site, quarter/half/three-quarter suffix, final pre-readout)
for the primary curve; every-layer traces are exploratory.

No new SAEs, no new model training, no new layer search selected for a large effect.
A candidate result must localize an excess change above matched-direction drift and
predict held-out final failures. “Deeper layers transform representations” is already known.
If all directions behave similarly, report general sensitivity rather than language damage.
Existing single-token layer-depth work and cross-layer decoder-flow work are direct baselines.
Budget: <=4 additional GPU-hours, preferably hooks on existing new runs.

### B — Does transfer depend on the behavior being controlled?

**Trigger:** logit effects transfer reasonably well or are mostly generic, while the
own-language-loss versus Belebele dissociation remains. No redefinition of the Belebele test.
**Future-work anchors:** F1 steering under compression; F3 extrinsic downstream tasks;
F4 naturalistic generation beyond structured prompts.
**Question:** do the same FP-selected language controls retain output-language control under
quantization while leaving semantic comprehension largely unaffected?

Use Deng's known language-gated continuation method as a BASELINE, not a new invention.
Freeze source/target pairs before Q evaluation. Use the already-covered panel
eng_Latn, spa_Latn, rus_Cyrl, hin_Deva, zho_Hans, yor_Latn; Arabic is not in our frozen 27
and must not be silently introduced from the earlier memo. Evaluate 300 fresh prompts per
source language, fixed greedy generation capped at 48 tokens, FP and the same INT6/INT4.
Measure target-language rate AND coherence/content preservation, with a validated language
identifier and a second check on ambiguous outputs. Report failure/missing coverage,
especially Yoruba. Require explicit calibration-only FP headroom; if the task has no
headroom, branch is unavailable, not an excuse to substitute a winning task.
Compare native and fixed-amplitude/direction controls, and preserve the original comprehension null.

Candidate contribution: a replicated task-specific boundary for compression-safe interventions
with an explanation from the main audit. Existing FP steering, translation-feature causality,
and steering transfer in quantized models preempt broad “first” claims. A second quantizer
would strengthen the boundary; absent one, stay narrow. Do not call output-language control
a comprehension improvement.
Budget: <=6 additional GPU-hours. If coherence/language verification cannot be made credible
inside the cap, do not use this branch as a headline.

### Routing and termination

| Main outcome | Action |
|---|---|
| Explanatory components and held-out predictive gain replicate | Finish the main paper argument; spend remaining budget on external corpus/quantizer validation. |
| Per-context instability or strong interaction remains unexplained | Branch A; prioritize localization over a new task. |
| Effects transfer; behavior distinction is the unanswered issue | Branch B, with the fixed task protocol and explicit FP headroom. |
| Generic controls fully explain changes | A possible audit-calibration result, but only if predictive utility and practical consequences are demonstrated; otherwise course-project evidence. |
| No substantial transfer error | Report preservation at the measured precision; no damage paper. Branch B is permitted only for the predeclared task-specific question. |
| Broad intervals, poor controls or pathological hybrids | Repair the instrument within allocated time or conclude inconclusive. No positive/null conclusion. |
| Both branches lack evidence or time | Stop. Preserve all results. Do not open a third direction to search for significance. |

## Approaches rejected

More tiny gain tuning; selection-threshold fishing; a new association metric without a
behavioral target; observer-alignment as proof of recovered computation; a generic pairwise
interaction paper (already done); a norm-matching-only paper (already done); collecting more
bitwidths as the contribution. These may be controls, not the research claim.

## Handoff obligations

Keep this design distinct from executed results. Freeze samples, outcomes and budgets before
reading new outputs. Append all amendments with timestamps and motivations. Check live Ada
jobs before planning occupancy; do not cancel Claude's existing work. Never overwrite frozen
runners or the report in progress. No push, upload, branch merge or paper submission follows
merely from this planning document. The course report and this research path have different
claims; update report prose only after validating actual new evidence.

