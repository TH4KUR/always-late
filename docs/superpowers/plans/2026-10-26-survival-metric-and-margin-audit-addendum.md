# Addendum: survival-metric validity and decision-margin audit
Date: 26 October 2026. Requested by the user after discussing Duan's future-work statement.
Status: PLAN ONLY; appended to the main intervention-transfer direction. No new job submitted.
Read the current ledger first: reports/intervention_transfer/STATUS.md. Its dated amendments,
including Gemma-2-2B-only scope, supersede older two-model wording.

## Recommendation and budget

Keep the current crossed-intervention study as the MAIN experiment. Add a small metric-validity
and margin audit to its outputs, preferably while the existing forwards already expose logits.
Do not start a competing generic token-flip study or new safety benchmark.

The recorded 48-hour window started 26 October 13:53 IST and ends 28 October 13:53 IST.
The total ceiling remains 32 GPU-hours. This addendum may use at most two EXTRA GPU-hours
and four engineering/analysis hours INSIDE those limits; no reset, extra parallel job or
reduction of the main population floor. If instrumentation threatens completion, execute the
existing matrix first and label this addendum deferred. Existing failed instrument attempts
are engineering evidence, not scientific positive or null results.

## What is established about Duan, and what is not

1. Our audit establishes a substantial BOS protocol sensitivity in the Gemma feature pipeline.
   reports/audit_our_vs_duan.md records matched 256-token chunks with perplexity about
   16.1 with BOS versus 457 without, and different survival behavior. State the exact
   protocol, not that every result of the paper is false.
2. EOS is a separate issue. The official Gemma tokenizer implementation defaults to
   add_bos_token=True and add_eos_token=False. Missing EOS on a continuation prompt is
   not automatically incorrect. Do not bundle BOS and EOS into one unsupported criticism.
3. Different/underdescribed SAEs confound comparison of survival LEVELS across models.
   They do not invalidate a within-model FP/Q comparison that holds its SAE fixed.
   Duan section 3.1 gives SAE identifiers/sites, but we have not in this addendum audited
   both checkpoint configs and encoder nonlinearities. Record that as unresolved rather
   than claiming proven architectural equivalence or incompatibility.
4. The future-work question remains scientifically legitimate regardless of those limits:
   do measured activation changes alter interventions? Our answer must use our validated
   BOS-masked protocol, not inherit the paper's numerical survival rates.

Sources:
- Local audit: ../../../reports/audit_our_vs_duan.md
- https://arxiv.org/html/2606.03002v2 (section 3.1, compression operator, section 7)
- https://raw.githubusercontent.com/huggingface/transformers/v4.51.3/src/transformers/models/gemma/tokenization_gemma.py
  (tokenizer defaults; this is the cited implementation, not a claim about every installed version)

## Translate the pixelated-screen intuition into testable hypotheses

Our vendored target_tl_weight selects transformer attention/MLP weights, not W_E or W_U.
Thus direct collisions of quantized token-embedding vectors are not the mechanism in our
current setup. Rounded internal weights can change the contextual hidden state and hence
token scores even with an unchanged embedding/readout matrix.

Three different events must be separated:
- exact ties at the stored logit dtype;
- close but unequal scores whose ranking changes under a perturbation;
- semantically incorrect predictions, which require ground truth beyond FP agreement.

A deterministic greedy decoder can reproducibly choose a different token without any tie.
Sampling randomness, kernel/batch numerical variation and FP/Q differences are distinct.
Neither small margins nor quantization universally imply lower accuracy; a flip can fix an error.

For FP logits z, let i be the unique winner, m=z_i-max(j!=i)z_j, and
epsilon=max_j |zQ_j-z_j|. If m > 2 epsilon, the winner cannot change.
This is a sufficient bound, not a novel theorem or a necessary condition. A tighter check
uses all competitors' differential changes, not only the FP runner-up.

Pearson survival measures agreement of one feature's activation pattern across tokens.
It is not a token-score ranking metric. If aQ=.1*aFP then r=1 for nonconstant activations,
despite a tenfold scale change. Conversely an unstable feature can have negligible output
effect. Evaluate validity against a specified intervention target; do not pronounce Pearson
“incorrect” without specifying what it was supposed to measure.

## Prior art prevents a generic novelty claim

- Duan explicitly leaves steering/auditing/safety intervention validation open (section 7).
  This is the parent future-work question, not proof of novelty of every subexperiment.
- Wu et al., *Which Decisions Low-Bit Quantization Breaks, and How to Predict Them*,
  https://arxiv.org/abs/2608.06564v5 (preprint, under review, 5 October): already studies
  margin shrinkage/offsets and held-out flip prediction, including practical quantizers.
- Felicio, *How Divergence Becomes Decision Flips in Compressed Language Models*,
  https://arxiv.org/abs/2610.00694v1 (preprint): already relates output divergence and flips
  across compression settings. Its empirical ratios are not universal mathematical identities.
- Gradient intervention attribution (2610.15850), strength controls (2610.26489), and FEGA
  remain required comparisons from the main novelty register.

Proposed narrower question: **Does SAE activation survival predict preservation of the
intervention's effect on a fixed token preference, beyond decision margin, readout precision
and generic response sensitivity?** No identical study was found in this bounded search.
Do not claim that margin-induced flips, affine invariance of Pearson or the sufficient bound
are new. Direct behavioral continuation is the existing Branch B, not another new branch.

## Stage 1: no new model forwards

- [ ] Inventory which per-feature Pearson values, support overlaps and scale summaries exist
  for the 404 features; verify keys and sampling populations. Do not estimate correlation
  from sixteen contexts and call it the same corpus-level survival score.
- [ ] Reuse continuous r as the primary activation predictor; r>.9 is a prespecified secondary
  dichotomy. Show .8/.9/.95 only as a labeled sensitivity, never choose the winning cutoff.
- [ ] Record scale ratio and firing-set overlap as secondary candidates; compute their incremental
  utility, rather than averaging incomparable metrics into a new score without validation.
- [ ] Add hand-computable checks: affine scaling with r=1; a near-margin token flip with unequal
  scores; a large-margin stable prediction; sign-invariant effects with changed decision flips.
  These explain metric limits, not experimental discoveries or a series of model pilots.
- [ ] The main runner stores 5x5 Grams of context-averaged effect vectors. These cannot recover
  token identities, baseline margins, exact ties, full distribution distances, or rowwise flips.
  Do not claim those statistics can be reconstructed from Gram outputs.

## Stage 2: optional streaming sidecar, no change to the frozen primary outputs

Create a separate versioned measurement sidecar and amendment; leave freeze.json, the primary
effect outputs and all jobs already launched unchanged. A completed run can be replayed for
the sidecar only if its measured cost fits the allowance.

Candidate new files:
- scripts/g0/phase1_transfer_margin.py: rowwise summaries from logits, no model loading.
- scripts/g0/tests_torch/test_phase1_transfer_margin.py: the mathematical and key-alignment checks.
- reports/intervention_transfer/margin_audit_freeze.json: exact scope, keys, dtype, endpoints,
  splits, input hashes and calibration rules, written BEFORE new sidecar outcomes.
- reports/intervention_transfer/margin_audit/: generated tables and report.

An optional observer in the NEW transfer runner/core can emit sidecar records before logits
are reduced to an effect vector. Additive API, off by default, no edits to old frozen F5 code.
Verify that enabling it leaves the primary Gram/loss outputs identical on the same inputs.

For each selected content row, retain:
- source document/sentence, feature/direction, cell, row index and state;
- the FP unedited winner and runner-up IDs and their scores; FP top-five IDs;
- whole-vocabulary argmax for each cell before and after the intervention;
- each cell's scores at the FIXED FP candidate IDs, including the true target where available;
- top-two gap, number of tied maxima, correct-token NLL and edit norm;
- separate dtype diagnostics: float32 logits and a CAST-ONLY bf16 view. This cast isolates
  storage rounding, not the whole native bf16 head. A native-head comparison, if recorded,
  must be separately named because multiplication, normalization and soft-cap also differ.

The FP candidate pair must stay fixed for all Q and edited comparisons. Retain global argmax
so a third competitor winning is not silently treated as a top-two event. Full logits can be
discarded after these summaries; never persist vocabulary-wide tensors solely for this audit.
Do not average positions before computing decisions.

## Endpoints and comparison

For fixed FP tokens i,j and each state/cell:
m = z_i-z_j, delta_m = m_edited-m_unedited.
Primary continuous endpoint: |delta_m_Q - delta_m_FP|, with absolute and FP-scale-normalized
versions and undefined near-zero cases explicitly reported.

Distinguish:
1. compression-only argmax change on the same teacher-forced prefix;
2. whether the intervention changes argmax within each model;
3. whether the intervention's SIGNED preference effect changes across models.
Identical delta_m can produce different decisions if baseline margins differ. That is not
by itself evidence of a changed feature effect.

Calibrate a margin-effect noise threshold from the 95th percentile of absolute INT16-minus-FP
delta_m differences on CALIBRATION data. Do not use the main vector-norm floor as a margin
threshold: they have different definitions/scales. Effects within this threshold are unresolved,
not assigned a confident positive/negative sign. Report coverage.

Model comparisons on the same held-out document/feature groups:
A: FP margin, FP effect magnitude, activation scale, residual norm, generic sensitivity,
   mandatory gradient baseline from the main study, and readout-floor diagnostic.
B: A plus continuous Pearson survival.
C: B plus firing support and activation-scale preservation, labeled secondary.
Keep main calibration/held-out identities. Fit preprocessing/regularization only on calibration.
Report within INT6/INT4 separately, cluster by document and reused feature, and retain uncertain
cases. Categorical flip prediction is secondary and gets log loss/Brier score and coverage,
not accuracy alone. Low event counts yield inconclusive classification, not a dropped condition.

Useful conclusion requires a practically meaningful, uncertainty-supported held-out difference.
A project decision target is >=10% relative MAE reduction over A, with paired cluster interval
excluding no improvement. This is not a universal publication threshold. If B adds nothing,
report its interval and power/coverage; don't claim independence from a nonsignificant p-value.
If only C helps, validate it on the fresh corpus before proposing a replacement metric.

## Steering/auditing direction and remaining scope

The current causal-transfer matrix is the first step toward Duan's stated future work.
If a direct behavior result is still needed, activate only the already planned Branch B:
fixed language-continuation steering, with content/coherence checks and FP headroom.
Next-token flips alone are not a safety audit. An actual safety-relevant claim requires
a relevant labeled benchmark and valid controls, not an analogy to unsafe behavior.
Our dequantized RTN experiments are not deployed integer-kernel evidence. Mark that limitation;
a later validated GPTQ/AWQ/kernel arm is needed before a deployment claim.

## Execution decision

1. Finish the main instrument validation and honor its measured runtime/correctness gates.
2. Implement Stage 1 now as analysis/design work when its inputs exist.
3. Instrument Stage 2 only if it fits the stated additional allowance and preserves the main run.
4. If the allowance does not fit, keep the existing main study and record this addendum as
   future work, rather than replacing it with another incomplete pilot.
5. No new safety suite, SAE retraining, architecture sweep or model family inside this addition.

Search record: primary Duan paper re-read for SAE setup/quantization scope; official Gemma
tokenizer source inspected; primary PDFs for 2608.06564v5 and 2610.00694v1 read for
margin/flip novelty threats. Broad searches covered quantization + SAE survival + Pearson +
token-score ties/margins. This is a targeted search, not a guarantee of absence.


## Update 2026-10-28 21:12 IST (Claude; Stage 2 of the survival-and-intervention plan, after D-110 to D-112; ADDED, nothing above is changed)

1. **Window and budget expired.** The 48-hour window ended 28 October 13:53 IST; the 32 GPU-hour ceiling (relaxed by the user in the ledger) and the "at most two EXTRA GPU-hours" for this addendum are **not carried over**. The streaming margin sidecar (this addendum's Stage 2) was never instrumented; it stays **future work**, is not authorized by the survival-and-intervention plan, and the F5 files contain no margins (so every comparison below that needs a margin is unavailable for F5).
2. **Stage 1 (inventory, no new forwards) is partly done, for the F5 sample.** [`f5_input_inventory_2026-10-28.md`](../../../reports/survival_effect_rescue/f5_input_inventory_2026-10-28.md): continuous correlation R1, firing-set overlap, scale ratio, the effect similarity W1 and the full-precision effect size exist per feature and state for 1,487 features and the three arms; full-precision activation scale and firing rate per (language, feature) exist in the Phase 1 survival files (identical across arms). The 404 transfer features were not re-inventoried in this update.
3. **The model-comparison structure and the project decision target of this addendum are carried unchanged into Stage 3 of the new plan** (so that no threshold is chosen after seeing results): A = baselines, B = A plus continuous R1, C = B plus firing support and scale preservation (secondary); R1 is the primary predictor, the 0.9 cutoff is a labeled secondary dichotomy; target: at least 10 % relative reduction of held-out error over A with a paired cluster interval excluding no improvement (a project target, not a universal threshold). **Deviation to be recorded in the Stage 3 freeze:** for F5 the covariates of A that need margins, gradient scalars or the generic sensitivity do not exist, so A is limited to the full-precision effect size (dnll), the full-precision activation scale, the firing rate and the number of active tokens.
4. **Different outcome.** The new plan tests R1 against the effect similarity W1 (cosine of single-feature effect fingerprints), not against the fixed-token margin-preservation endpoint of this addendum; that narrower question stays unanswered here.
5. **Deployment caveat unchanged.** A GPTQ arm now exists (D-101 to D-111), but it is a weight-only dequantized simulation like RTN: no integer-kernel, activation-quantization or safety claim follows.
6. **Prior art.** The new plan's sources (P41, P47, P48; `reports/research_sources.md`, re-read 2026-10-28, D-110) concern the Llama stage and SAE-family differences; they do not change this addendum's prior-art section.
