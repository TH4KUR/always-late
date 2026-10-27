# Publication Recovery After Results — Implementation Plan

Current [novelty reassessment](../../../reports/novelty_assessment_2026-10-26.md): broad claims have direct precedents; generic BF16/FP32 sensitivity is also known. The planned precision audit must change a substantive feature-audit conclusion to become a contribution; it is not yet a result.


> **For agentic workers:** Use superpowers:executing-plans to implement sequentially. This is a POST HOC research revision requested after D-094. Preserve every original freeze, result, failed criterion and amendment.

**Goal:** Turn the transfer results into a defensible statement about precision-stable intervention preservation, with an independently evaluated behavioral endpoint.

**Architecture:** First reuse completed outputs and finish the already-running external validation. Then conduct one paired arithmetic-precision audit only if needed. Reuse the existing margin/behavior sidecar only after the measurement question is resolved. Do not open unrelated experiment families.

**Tech Stack:** Existing transfer runner, saved Gram/loss/amplitude records, Python/NumPy/SciPy/PyTorch, existing Gemma-2-2B/SAE assets and Ada; no new SAE training.

**Spec:** [Result reassessment](../../../reports/publication_reassessment_2026-10-26/README.md), [original design](../specs/2026-10-26-intervention-transfer-research-design.md). This plan supersedes FORWARD-LOOKING priorities in the old plan, not its retrospective protocol or claim verdicts. Current execution scope remains Gemma-2-2B only.

## Decision and expected paper shape

Do not promote the present generic-sensitivity predictor as the central contribution.
Its apparent gain is larger on the rounding reference than on INT4, and shrinks on the small
strictly unseen-document subset. A prediction made using four expensive test-time control
interventions is also not automatically a cheaper audit than measuring the target directly.

Candidate paper: **When Can We Trust Feature-Survival Audits? Precision, Intervention
Definition, and Behavioral Transfer under Quantization.**

This is a working question, not a promised title or accepted contribution. It requires:
1. a reliable distinction between changing the weights and changing the measurement arithmetic;
2. a result that survives truly independent documents/corpus;
3. a consequence for signed loss or a fixed behavior, rather than cosine alone.

The results can support an explanatory evaluation paper if these requirements hold. A practical
repair-method paper is currently unsupported. A broad A-tier claim still needs another
validated quantizer/model; do not disguise one-model RTN evidence as general deployment evidence.

## Global constraints

- Original freezes, D-094 tables and UTILITY=not met remain unchanged.
- Existing external jobs finish; do not cancel, duplicate or change them.
- Old main-plan/spec hashes remain unchanged; all new decisions go into new amendment files.
- The 27-language/300-sentence/three-control main group population is preserved.
- The paired precision follow-up is a distinct instrument-validation sample, not a reduced
  replacement for the main population or new language-generalization evidence.
- The old 32-GPU-hour ceiling was relaxed by the user in the execution ledger; do not reimpose
  it silently. Use measured cost, an explicit <=8 additional GPU-hour planning allowance for
  this revision, and the existing 28 October 13:53 study target. If unavailable/infeasible,
  record the limitation rather than extending by default. The course deadline remains 31 October.
- No target, threshold, model choice or baseline is tuned on external evaluation outcomes.
- All analyses selected after D-094 are labeled post hoc, even if a new file is committed first.
- No unsupported equivalence, safety, mechanistic-mediation, cheap-audit or deployment claim.

## Review focus

- Cross-corpus prediction must reuse actual saved fits, preprocessing and comparator selection.
- FP32-head-only execution is not FP32-suffix execution; assert internal dtypes.
- Pairwise eligible controls still differ in FP effect scale; report common support and matching.
- Scalar NLL gradients must be compared with scalar NLL outcomes, not treated as logit JVPs.
- A third token can win despite unchanged FP top-two ordering; rowwise global argmax remains required.

## Task 0 — Preserve the old experiment and lock evaluation, first 1 hour

**Files:** reports/publication_reassessment_2026-10-26/{README.md,audit.json,locked_flores_models.json,reassess_existing.py}; new reports/intervention_transfer/revision_R1.json at execution.
**Already completed by this planning turn:** source audit, all-original-MAE reproduction, descriptive unseen-document restriction, native/fixed comparison, common-eligibility counts, export of the original FLORES calibration models. No external outcomes used by this script.

- [ ] Independently inspect the saved audit. Original 1,066 eligible held-out items reduce to
  136 on 65 wholly unseen source documents. INT4 diagnostic improvement becomes 6.73%;
  INT6 .46%; INT16 29.4%. These are subset point estimates, not new significance tests.
- [ ] Verify input hashes and model reconstruction from the exported means/scales/coefs/intercepts.
  Before external scoring, ensure ridge_predict on the main rows matches the committed MAEs.
- [ ] Record provenance of the original and new gradient targets and two interventions
  (fixed FP edit versus native Q edit). Preserve the old scalar/vector results separately.
- [ ] Record current queue/run validation state once. Continue report/deck maintenance, but
  no publication language beyond the currently established scope.

## Task 1 — Independent evaluation before any new mechanism experiment, next 2–4 CPU hours

**Create:** scripts/g0/phase1_transfer_recovery.py and tests_torch/test_phase1_transfer_recovery.py.
**Inputs:** locked_flores_models.json; complete validated external chunks; existing per-context tables.
**Output:** reports/intervention_transfer/results/recovery/cross_corpus.csv with explicit target and normalization protocol.

- [ ] Finish validation of all six existing external jobs and perform the unchanged original
  external analysis once. Report it as originally specified, alongside this new analysis.
- [ ] Score EXTERNAL held-out-feature items using locked FLORES fits; no external ridge fit,
  standardization, lambda choice, comparator selection or filtering based on Q errors.
- [ ] PRIMARY frozen-fit target/covariates use the FLORES per-kind floors and the original
  feature transformations, including control-direction sensitivity. This keeps normalization
  units constant across corpora. External-floor-calibrated results are SECONDARY and separately
  named, since changing the denominator changes the prediction target.
- [ ] Test that changing external outcomes never changes coefficients, comparator or eligible set;
  serialize the source-model hash in every output. A missing covariate stays missing with counts.
- [ ] Add the calibration-median null to the comparator set using CALIBRATION fold errors only.
  Do not choose whichever baseline happens to win on external test outcomes. Retain the old
  comparison verbatim as well.
- [ ] Report INT16/INT6/INT4 separately, absolute MAE, relative improvement, coverage, source
  document counts and paired document/feature-cluster intervals. No pooled-state easy predictor.
- [ ] Re-evaluate existing gradient scalars against existing signed NLL changes and shifts,
  per-context and per-state, with document/feature grouped evaluation. Report a first-order
  approximation only at its actual intervention size; large zero-ablation is not locally linear.
  The full logit-vector JVP remains untested and must be named as such.
- [ ] Analyse common real/swapped eligibility with FP effect norm, edit norm and residual norm
  jointly reported. Predefine FP-effect-norm ratio [0.8,1.25] as a diagnostic overlap band;
  count coverage before reporting a contrast. This is post hoc support analysis, not proof of
  equivalence. Do not rescale stored effect vectors to imitate a physically rescaled nonlinear edit.
- [ ] Report group native-versus-fixed contrasts per language: native_total = W + native_remainder.
  Use paired differences and sentence resampling from group chunks, not differences of medians.
  The .4325 native versus .8479 fixed median is not a license to claim amplitude causes half the harm.

**Promotion decision:** a replicated external effect is necessary, not sufficient. Keep the
original 10% utility criterion unchanged. A new quantization-specific predictive claim requires
incremental improvement beyond measurement/reference-sensitivity covariates and an outcome
outside the original overlapping context set. Otherwise this predictor is supporting analysis,
not the abstract's method claim.

## Task 2 — ONE paired precision-control experiment, only after Task 1 identifies the need

**Create:** scripts/g0/phase1_transfer_precision.py, focused tests and precision_audit_freeze.json.
**Do not change:** frozen transfer code, sample or original output files.
**Budget:** <=8 additional GPU-hours inclusive of any later selected follow-up; measure once.

**Question:** which transfer conclusions persist when the numerical implementation of the
suffix changes while cached inputs, stored weight values, edits and readout definition are fixed?

- [ ] Select 128 existing features by stable hash within FP-effect-norm quartiles, four contexts
  per feature by the original frozen FP ranking: 512 pairs. Selection cannot use Q error,
  prediction residual, desired sign or a favorable language. Use real and matched-swapped
  directions, FP/INT16/INT4, same five cells and same input/amplitude tensors.
- [ ] Compare two execution modes: existing bf16 suffix with fp32 readout; genuinely fp32 suffix
  and readout. Promote the SAME stored bf16 weight values to fp32 without requantizing or
  refitting scales. Record hashes of pre-promotion values.
- [ ] Keep cached FP/Q prefix residual values fixed and promote them exactly. This isolates
  SUFFIX arithmetic; it does not validate full-prefix precision or a deployed integer kernel.
- [ ] Assert attention, MLP and normalization internal tensor dtypes at representative hooks;
  model.float() alone is not proof. Preserve attention masks, soft cap, BOS and positions.
  Record any change of attention kernel; if it cannot be held fixed, label arithmetic-plus-kernel
  comparison rather than a dtype-only causal effect.
- [ ] Keep only the suffix/readout resident if necessary; prefix caches already exist. Benchmark
  memory and end-to-end throughput on the fixed first chunk. If full scope exceeds allowance,
  report infeasible; do not repeatedly reduce the sample until a noisy result fits.
- [ ] Add FP-versus-FP repeated-batching controls in BOTH arithmetic modes. Distinguish this
  execution sensitivity from INT16 RTN's genuine small weight perturbation.
- [ ] Primary outputs: paired INT4 transfer norm/cosine, signed NLL change and INT16 reference;
  the magnitude of between-arithmetic change relative to between-weight-state change;
  coverage at both the old fixed floor and a mode-specific calibration floor.
  No blind subtraction of squared noise norms: correlated deterministic rounding does not
  satisfy independent zero-mean noise assumptions.
- [ ] Use exact hand-computable dtype/cell tests and the real-model same-state/sham gates.
  No open-ended sequence of instrument pilots. Freeze any implementation amendment with cause.
- [ ] Report all selected pairs, near-zero effects and mode-specific failures. Use paired
  document/feature intervals. If precision changes conclusions, that is an instrument result;
  if INT4 drift persists, retain it as robust within the tested suffix setup.

This is the only new model experiment prioritized by this revision. It tests a concrete
alternative explanation exposed by the results, rather than adding another bitwidth/layer.

## Task 3 — Behavioral consequence or practical audit, choose at most one next

**Activation condition:** Task 1 external evidence and Task 2 numerical interpretation are usable.
If the existing continuous signed-loss endpoint answers the question robustly, no extra task is required.

A. **Margin sidecar** (existing addendum): reuse the precision-validated runner to measure
fixed-candidate delta-margin transfer and global argmax before/after the edit, with the
original teacher-forced prefixes. Retain the mandatory gradient/margin/FP-effect baselines.
Do not frame generic margin flips as new. A high averaged cosine must be connected to an
actual missed intervention consequence, not merely a second cosine statistic.

B. **Language-continuation branch** (existing Branch B): only if clear behavior beyond the
saved NLL endpoint is needed and calibration FP headroom exists. Keep fixed source/target
pairs, content/coherence checks and the comprehension null. No safety claim from output-language
changes. Do not add translation, QA and refusal suites in search of a positive task.

A practical prediction method additionally needs a fair COST comparison. Four per-feature
test-time control interventions can cost more than measuring the target. Before claiming
efficiency, count suffix calls, prefix caching, wall time and memory at equal reliability.
A global reusable probe scheme may be a later method, but is not implemented or proven here.
If the current audit adds no actionable or economical value, publish it only as a diagnostic result.

## What is deferred

- Layerwise mechanism branch A until an unexplained, precision-stable effect needs localization.
- Full three-factor context/weights/amplitude cube: five cells cannot supply its missing effects.
- More gain-grid tuning, more precision rungs, SAE retraining and new model families.
- New quantizer integration within this revision; first inventory existing compatible assets.
  A broader submission will need a validated second quantizer or narrower claims.
- Any claim that insignificant Pearson increments establish universal metric invalidity.
- Any relaxation of the old success criterion because one precision failed.

## Publication decision after the revision

| Evidence | Honest manuscript direction |
|---|---|
| Precision-stable transfer failures, cross-corpus result, behavioral consequence | Main explanatory evaluation paper; expand external quantizer evidence for a broad venue claim. |
| Apparent failures change substantially with arithmetic | Measurement-validity paper, if the issue affects real audit decisions and the calibration remedy generalizes. |
| Pearson adds no useful information on independently evaluated, reliable outcomes | A scoped negative result supporting the above; not a universal theorem or standalone title by itself. |
| Only mean fingerprints look stable | Report aggregation bias with reference controls and behavioral false reassurance; otherwise supporting descriptive evidence. |
| Predictor works only on shared contexts or rounding reference | Withdraw generalization/quantization-specific-method claim. |
| Effects lack separation from measurement uncertainty | Inconclusive at that precision, not preservation or destruction. |
| No robust external consequence | Strong course report and reproducible negative result; no A-tier claim earned yet. |

The submission standard is evidence, not a planned threshold being crossed. None of these
routes guarantees acceptance. Record failed routes and stop expanding when the identified
question has been answered.

## Deliverables and handoff

- [ ] New recovery report with source hashes and a claim-by-claim disposition.
- [ ] Locked FLORES-to-Wikipedia predictions, with no external-data refit in that row.
- [ ] Precision audit or explicit feasibility limitation.
- [ ] One outcome-matched behavioral conclusion and scope.
- [ ] Updated report/deck with original nulls and deviations intact; official ACL checks after edits.
- [ ] Append-only ledger and dated handoff. No automatic paper submission or external publication.

**Self-review:** original population and freezes preserved; actual new calculations reproducible;
external prediction units specified; gradient target mismatch corrected; all five review hazards
covered; costs counted; negative outcomes do not activate a new unbounded search.

