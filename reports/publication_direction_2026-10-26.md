> **Current reporting pointer, 30 October 2026:** This is a dated assessment, preserved as written. Its earlier pending-review and publication-priority statements are historical. The 9B original-RTN results have since been reproduced and reviewed (D-127); GPTQ remains pending. Use the [final questions and answers](FINAL_RESEARCH_QUESTIONS.md), [research paper](../paper/research-paper/main.tex) and [current status](CURRENT_STATUS.md) for the completed study.

# Publication direction decision — 26 October 2026

> **Historical assessment / design-stage record.** The [26 October post-D-094 novelty assessment](novelty_assessment_2026-10-26.md) supersedes novelty verdicts and forward priorities below. The broad claims have direct precedents; an exact protocol match was not found, but novelty is not verified. Keep this source/search history; use current results and the recovery plan for decisions.


> **Update 2026-10-26 (Claude, documentation sweep; Codex's text below is unchanged):** (1) The user replaced Llama-3.2-1B by **Gemma-2-9B with Gemma Scope** (`layer_20/width_16k/canonical`) as the second model (D-089). Read every 'Llama' below as the second model. The 9B runs are done by another team on another cluster (handoff 040, branch `gemma9b-confirmation`), so this study runs its Gemma-2-2B part first and its second-model arm after that branch is merged. The Llama check finished on 2026-10-26 and is reported as exploratory only. (2) Verified and adopted with corrections (D-088): [verification](intervention_transfer/verification_2026-10-26.md) (Chang EMNLP 2025 §6.1-6.2 is the loss-level precedent, so C > W alone is not a contribution; the gradient baseline 2610.15850 is mandatory; Bhandari 2609.06473 added; three citation fixes). The current order of work is the NEXT STEP block of `reports/functional_validation/STATUS.md`.

Status: research proposal, not a new experimental finding. No GPU jobs submitted by this chat.
User constraint: build on existing work; at most 48 hours total for the new study.
Recommendation: a factorial audit of SAE intervention transfer under quantization.

## Decision

The best next experiment is to independently vary (1) the activation context at the SAE layer, (2) the downstream model weights, and (3) intervention amplitude. The paper question is: **When an SAE intervention changes after compression, what changed: the measured feature, its context, or the model responding to it?**

Do not pitch another gain sweep, a tiny language-loss improvement, or “geometry does not imply causality.” The latter has direct prior art. A promising A-tier submission would require a replicated explanatory result and a useful audit procedure, not simply another significant comparison. Acceptance cannot be inferred from an experiment plan.

## Why our existing work motivates this

Read: reports/functional_validation/RESULTS_SUMMARY.md, prior_art.md; F5 runner and sensitivity runner; Study G README; handoffs 030 and 032.

- At INT4, the 404-feature comparison gives median functional cosine about .842, versus .828 for swapped directions. Fixed FP amplitudes give about .846 versus .838. This motivates generic response drift; it does not isolate its cause.
- The fixed-amplitude implementation still uses each state's native cached residual. Therefore both background state and downstream weights change. The statement in RESULTS_SUMMARY that this localizes drift to downstream weights is stronger than that experiment establishes.
- Language-group ablation harm increases under quantization, but that can reflect changes in general susceptibility to perturbations. It does not alone establish that language computations became more important.
- The ~.05% loss improvement from fine gain is too small and too narrow to anchor a practical repair paper.
- Feature selection and earlier FLORES evaluation have been repeatedly inspected. Existing data support exploration; a newly frozen analysis does not retroactively create untouched test data.
- Current local STATUS records Llama access granted and consistency jobs launched on 26 October, with an interim FP result positive in 21/27 languages (below its 24/27 criterion). This is newer than the earlier “access blocked” summary. This chat did not verify the live queue or final verdict. Do not assume those jobs will replicate Gemma.

## Primary-source comparison and novelty threats

This is a targeted scoping search, not an exhaustive novelty certificate. Sources read on 26 October 2026.

| Source | What is already covered | What our proposed study would have to add |
|---|---|---|
| Duan, arXiv:2606.03002v2, preprint | Frozen-SAE activation correlation across RTN bitwidths; a measured definition of feature survival. The paper frames transfer reliability as a concern, not a demonstrated causal theorem. | Test whether survival predicts intervention transfer after separating background state and downstream weights. Do not allege that the paper proves causal destruction. |
| Borobia et al., arXiv:2603.25325 and revised author repository | Geometric survival versus local ablation; revised repository includes frozen-dictionary controls, activity-controlled error-preserving ablation, INT8/INT4 pilots and BOS corrections. | A crossed intervention decomposition. “Quantization instead of pruning” and “add causal ablation” are insufficient differences. |
| Chang et al., EMNLP 2025, Why Do Some Inputs Break Low-Bit LLM Quantization? | Residual magnitudes, accumulated errors, late-weight restoration, cross-model module patching and MLP-gate localization. | A feature-intervention transfer audit rather than another module-patching or generic sensitivity result. Include residual magnitude as a baseline explanation. |
| Hoang et al., arXiv:2607.24645, FEGA preprint | Context-dependent clouds of logit effects under feature ablation. | Cross-compression factorial diagnosis. The logit-effect fingerprint itself is not ours. |
| Zhang et al., arXiv:2610.26489, 19 October preprint | Different SAE codes, norm and KL strength controls, fixed groups; output calibration uses the evaluated input and is explicitly diagnostic. | Separate model changes from state changes; if calibrating anything, learn it on disjoint calibration documents and assess on held-out documents. Norm matching is not a novel method. |
| Mei and Ma, arXiv:2610.21240, 15 October preprint | Prompt-induced redistribution across many SAEs; held-out supports; separation of representation change, frozen linear-readout contribution and decision failure. | Nonlinear model interventions under weight compression, with crossed states and weights. “Representation change need not cause failure” is already known. |
| Sathyanarayanan, arXiv:2609.23125, preprint | Residual activation quantization, matched noise, capability probes, oracle feature restoration and rotation. | Weight-compression effects on intervention transfer. Neither oracle restoration nor rotation is new. |

Links:
- https://arxiv.org/abs/2606.03002
- https://github.com/hecboar/sae-pruning-paper
- https://arxiv.org/abs/2603.25325
- https://aclanthology.org/2025.emnlp-main.168/
- https://arxiv.org/abs/2607.24645
- https://arxiv.org/abs/2610.26489
- https://arxiv.org/abs/2610.21240
- https://arxiv.org/abs/2609.23125

The revised Borobia repository explicitly supersedes older values. Criticizing its old BOS protocol as an unaddressed gap would be unfair. Its revised README was read directly, not inferred from a search snippet.

## Primary experiment: crossed states and weights

Write h_p(x) for the residual after the prefix in state p, and F_s for the suffix with weights in state s. For feature f use decoder d_f and a fixed full-precision per-position amplitude a_FP,f(x). Keep exactly the same token positions in all cells.

Define the centered logit intervention effect:
`D[p,s,f](x) = center(F_s(h_p(x) - a_FP,f(x) d_f) - F_s(h_p(x)))`.

For p,s in {FP,Q}, run:
1. FP context / FP suffix: reference.
2. Q context / FP suffix: state-context change with downstream weights fixed.
3. FP context / Q suffix: downstream-weight change with context fixed.
4. Q context / Q suffix: combined change.

Always subtract each cell's OWN unedited output. Otherwise ordinary model-output drift is mixed into the intervention effect.

For vectors or scalar signed loss effects, an exact factorial identity is:
- C = D[Q,FP] - D[FP,FP]
- W = D[FP,Q] - D[FP,FP]
- I = D[Q,Q] - D[Q,FP] - D[FP,Q] + D[FP,FP]
- combined drift = C + W + I.

This algebra is not itself a novel contribution. Its empirical use to explain native intervention-transfer failures is the candidate contribution. Vector magnitudes are not additive; do not report their norms as percentages that sum to 100. Large interactions prevent a clean single-cause story.

Then compare fixed-amplitude D[Q,Q] to native-amplitude D[Q,Q] using a_Q,f. This isolates an amplitude contrast conditional on the Q background and Q suffix. It is not a complete three-factor decomposition; run all eight cells only if a full three-factor claim is needed.

### Measurements and controls

Primary: paired, per-context intervention-vector error relative to FP, normalized by FP effect norm with a noise-derived eligibility rule frozen on calibration data. Also report cosine, effect magnitude, signed next-token loss change, and task-margin change where a real task is available. Do not average vectors over contexts first: cancellation can conceal instability.

- Real feature direction; matched alternative SAE direction; norm-matched random direction. Decoder-direction norms must be matched explicitly.
- Native amplitude, fixed FP amplitude, and a small-effect finite-difference check on a preselected subset. Zeroing is a large perturbation, not automatically a local derivative.
- Sham and numerical floor; split-forward equals full-forward; no edits/statistics at BOS; preserve reconstruction error.
- Patch norms, residual norms, effect norms and unedited hybrid loss. Hybrids are surgical counterfactual models; pathological hybrid behavior limits interpretation.
- Compare against Pearson survival, activation magnitude/firing rate, residual magnitude (Chang baseline), and generic-direction response drift.
- Cluster uncertainty by document and feature where both are reused. Translations of the same source are one content cluster. Never treat tokens as independent samples.
- Report undefined/near-zero cosine cases and excluded features, rather than silently dropping them.

### The useful output beyond a decomposition plot

On calibration documents, measure how the Q suffix changes its response to independent matched/random directions. Test whether this generic sensitivity diagnostic predicts intervention-transfer error on unseen documents/features better than activation correlation and residual magnitude.

Use a simple prespecified calibration model (at most a few coefficients, no feature-specific tuning), leave feature groups out, and evaluate a nested baseline comparison. Computing the exact target feature's Q intervention effect and then predicting that same effect is circular; the generic diagnostic must use independent directions, and total forward-pass cost must be reported.

A practical claim is earned only if the diagnostic predicts a meaningful error or changes an audit decision on held-out data. An exact decomposition that merely restates the observed error is not a predictive method.

## Bounded 48-hour design

Planning assumption: up to two already-authorized RTX 2080 Ti slots, existing Gemma and now-accessible Llama assets, no new model training. Queue availability is unverified. 48 hours is a hard elapsed-time cap, not a guaranteed runtime.

Core matrix:
- Gemma-2-2B and Llama-3.2-1B, existing SAE sites, one SAE each.
- RTN INT6 and INT4; FP and numerical floor.
- Six languages fixed before new outputs: English, Spanish, Russian, Hindi, Arabic, Chinese. This is script coverage, not a claim of representative language-resource sampling.
- Up to 96 feature-context pairs per model (balanced across languages and FP activity strata), eight disjoint evaluation contexts per feature, plus separate calibration contexts. Smaller scope than the existing 404-feature sensitivity run.
- Four crossed cells, actual/matched/random directions; reuse baseline passes and prefix caches. Retain per-context outputs or sufficient statistics, not only global means.
- Secondary K64 group test on 128 passages/language, four crossed cells, true group plus three fixed control draws; only native-vs-fixed amplitude contrasts needed.
- Use fresh documents from a second corpus with dataset/version/document IDs frozen before scoring. Split at document level; deduplicate against previous corpora. Feature IDs can reuse old selection, but do not select evaluation contexts by Q outcomes. If fresh data preparation fails, label results exploratory and weaken the paper claim.
- One second compression method is desirable for a broader paper, but not a hidden requirement squeezed into 48 hours. Only add an already-supported checkpoint/method with correct hook semantics and a validated smoke. Otherwise explicitly limit to RTN.

Historical cost evidence: handoff 032 estimates ~1.5 GPU hours for 404-feature, five-state direction-control run; Study G README says ~3 GPU hours/state for 27 languages and 38 configurations. These are sizing anchors, not a benchmark of the new wrapper.

Schedule:
0–4 h: freeze question, sample IDs, primary outcomes, eligibility and controls; audit existing Llama job occupancy; prepare fresh data.
4–8 h: narrow wrapper around existing lower/cache/upper callables; toy-model factorial identity, sham, hybrid/native equality gates; benchmark 16 feature-context pairs.
8–20 h: core Gemma crossed experiment; compute C/W/I per item; independently run Llama as slots allow.
20–32 h: group-level test and held-out generic-sensitivity diagnostic; finish fixed-amplitude controls.
32–40 h: primary uncertainty, baseline comparisons, model-specific results; second-method smoke only if core already fits.
40–48 h: figures, source/result audit, conclusion and manuscript argument. Hard stop; no exploratory sweep launched in the remaining hours.

GPU target budget: 16–24 GPU hours total, with up to two concurrent jobs. Reserve the rest of elapsed time for implementation, queueing and analysis. At the 8-hour benchmark, estimate total time from measured pair throughput including full-vocabulary readout, then add 50% contingency. If it exceeds budget, reduce feature count by a deterministic prespecified rule or drop optional group expansion; retain both models and all four core cells. Do not pretend extrapolation guarantees completion.

## Go/no-go rules

These are research decision rules to freeze before new results, not universal publication thresholds.
- Instrument gate: identity and sham errors below measured numerical floor; valid hooks in both models. Failure stops inference.
- Mechanism gate: the crossed experiment must distinguish context, weights, or interaction in a reproducible way. A useful target is a component contrast exceeding 3x the calibration numerical floor, with document-cluster interval excluding zero, in both models. Report all components even if none dominates.
- Specificity gate: report the actual-minus-matched/random contrast. If the entire phenomenon is generic, say so; do not call it selective language-feature damage.
- Utility gate for the proposed diagnostic: on held-out data, at least a prespecified 10% relative reduction in prediction MAE over the strongest baseline, with a paired interval excluding no improvement, replicated in the second model. This threshold is a project go/no-go choice, not a meaningful universal standard.
- External-validity gate: at least a fresh corpus and second model. Without a second quantizer, retain a narrow RTN claim.
- Do not declare equivalence from a null test. If intervals are broad, call the experiment underpowered, not evidence of preservation.
- A suffix-only quantization counterexample gives unchanged SAE activations by construction. That alone is a trivial counterexample, not an A-tier paper. The value must come from explaining natural full-model quantization and improving audits.
- If only Gemma supports the mechanism, investigate/report architecture or SAE dependence; do not change the endpoint until Llama appears to agree.
- If there is no robust mechanism or practical audit improvement, this direction has not earned the proposed paper. Preserve it as course-project evidence and stop after the cap.

## Alternatives ranked

1. **Crossed causal-transfer audit (recommended):** maximally reuses F5, addresses an actual identification gap, tests a mechanism and can yield an audit method. Main risks: generic/trivial result, pathological hybrids, failure to generalize.
2. **Observer recalibration:** fit a low-capacity map from Q hidden states to FP states using calibration-only paired data; apply it only before the frozen SAE. Compare identity, diagonal shift/scale, low-rank map, and shuffled-pair controls on held-out documents and causal outcomes. Do not use per-feature affine postprocessing as a supposed Pearson fix: positive affine transforms leave Pearson correlation unchanged. Higher survival alone is fitted measurement agreement, not restored computation. Higher fitting/novelty risk; do not run concurrently with the primary plan.
3. **Intervention-strength audit:** native vs norm/KL-matched language interventions. Cheap and useful as a control, but October's strength-control paper preempts a standalone novelty claim. Incorporate necessary controls into option 1.
Rejected as anchors: further tiny-gain tuning; another bitwidth/language sweep; BOS-only criticism; task shopping after Belebele null; SAE retraining/crosscoders within the time cap.

## Search log and limitations

Sources: alphaXiv discovery and raw PDF queries; web search of arXiv, ACL Anthology, author repositories; existing local source register.
Discovery prompts targeted quantization, SAE survival, multilingual interventions, frozen-SAE calibration/transport, common-input prefix/suffix interventions and function-preserving changes. Two alphaXiv discovery calls used broad coverage; relevant PDFs were then queried directly.
Web queries included:
- sparse autoencoder quantization feature survival alignment calibration
- quantization sparse autoencoder "prefix" "suffix"
- quantization "causal" "feature" "transport"
- quantization SAE "recalibration"
- "crossed" "quantization" "interventions"
- "quantization" "model stitching" interpretability
- "quantization" "Jacobian" "sparse autoencoder"
- "SAE" "quantization" "calibration" feature transfer
- "sparse autoencoder" "intervention" "transportability"

Exact-ID searches covered 2606.03002, 2603.25325 and 2607.24645. Discovery also returned 2609.12591 (cross-layer feature flow; read), 2609.15064 (RL fixed-SAE tracking; not full-text checked), 2610.20680 (training data and reproducibility; not full-text checked), 2607.12166, 2607.20596, 2606.18322, 2609.22782, 2610.20303 and 2609.21450. These broader neighbors remain novelty-check leads, not evidence for unverified assertions. A Zenodo perturbation-ladder preprint was discovered; only its author abstract was inspected, so no detailed criticism is made.

No identical experiment was found in this bounded search. That is not proof of novelty. Papers are mostly preprints; Chang's EMNLP venue was checked against ACL Anthology. Before submission, compare the final actual result against the newest versions, particularly the revised Borobia repository.

## Current reporting context, 30 October 2026

This dated assessment is retained as a historical record; its old pending-work and publication-priority statements are not the current project status. [Current synthesis](FINAL_RESEARCH_QUESTIONS.md), [research paper](../paper/research-paper/main.tex), [current status](CURRENT_STATUS.md). The completed paper reports a controlled evaluation across three Gemma-2-2B quantizers, task/control checks and scoped scale/architecture extensions. It credits direct precedents and makes no first-ever or acceptance claim. A limited model/layer/SAE scope bounds generalization rather than invalidating controlled effects. Positive associations, supported intervention contrasts and informative nulls all belong in the final account. Original predictions and their outcomes are unchanged.

The 9B original-RTN analysis was reproduced from raw files and reviewed (D-127); GPTQ remains pending, matched RTN is unrun. Belebele's primary interval is in correct-option NLL, not accuracy points. Continuous survival adds about 3–4% predictive improvement beyond baselines, below the 10% convention but not zero. F5 effect changes can coexist with high own-identity retrieval. These later clarifications govern current interpretation where the dated prose differs.
