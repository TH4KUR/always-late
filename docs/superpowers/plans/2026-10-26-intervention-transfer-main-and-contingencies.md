# Intervention Transfer Main Direction and Contingencies Implementation Plan

> **Update 2026-10-26 (Claude, documentation sweep; Codex's text below is unchanged):** (1) The user replaced Llama-3.2-1B by **Gemma-2-9B with Gemma Scope** (`layer_20/width_16k/canonical`) as the second model (D-089). Read every 'Llama' below as the second model. The 9B runs are done by another team on another cluster (handoff 040, branch `gemma9b-confirmation`), so this study runs its Gemma-2-2B part first and its second-model arm after that branch is merged. The Llama check finished on 2026-10-26 and is reported as exploratory only. (2) Verified and adopted with corrections (D-088): [verification](../../../reports/intervention_transfer/verification_2026-10-26.md) (Chang EMNLP 2025 §6.1-6.2 is the loss-level precedent, so C > W alone is not a contribution; the gradient baseline 2610.15850 is mandatory; Bhandari 2609.06473 added; three citation fixes). The current order of work is the NEXT STEP block of `reports/functional_validation/STATUS.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Track the checkboxes below. Claude is the intended execution owner; delegation is optional only with applicable authorization.

**Goal:** Explain native SAE intervention-transfer changes under quantization and evaluate a useful held-out diagnostic, with two evidence-triggered secondary directions.

**Architecture:** Preserve frozen runners. Add a narrow adapter that independently supplies cached prefix residuals, suffix weights and intervention amplitudes, retaining per-context outputs. Use one shared analysis and one outcome-routing report for the main and any activated secondary direction.

**Tech Stack:** Existing Python/PyTorch, TransformerLens, Gemma Scope and Llama SAE loaders, NumPy/SciPy and Slurm on Ada; no SAE training or new inference framework.

**Spec:** [Research design](../specs/2026-10-26-intervention-transfer-research-design.md).

## Global Constraints

- The first new execution is capped at 48 elapsed hours, at most two concurrent Ada jobs, and a planned ceiling of 32 GPU-hours including conditional follow-up.
- Preserve the 27-language, 300-sentence, three-draw Study G floor.
- This new study uses its same population for group-level explanatory analysis and the established 404-feature F5 control subset with up to 16 contexts for feature-level diagnosis.
- The six-language new-corpus validation panel is a distinct external-validity sample, not a replacement for the 27-language study.
- Do not mutate frozen Study G settings or retroactively relabel examined data.
- Missing or unsupported cells stay visible.
- Do not cancel, modify or duplicate Claude's existing Llama jobs.
- At most ONE secondary branch runs inside the same budget.
- No new GPU job, implementation or submission has been performed by this planning chat.

## Review Focus

1. Wrong split point or quantized weight restore corrupts every hybrid; native and same-state hybrid outputs must agree.
2. Float32 residuals, bf16 weights or output soft-capping can create an artificial effect; keep matching precision and sham floors.
3. Averaged vectors, translations and reused features can hide failure or exaggerate sample size; retain per-context records and cluster correctly.
4. Calibration features/documents/directions leaking into held-out evaluation create a circular audit; every split and key is frozen.
5. A pathological hybrid or tiny FP effect produces unstable ratios; report eligibility/coverage and native hybrid losses before interpretation.

## Read first: scope reconciliation

- Read reports/functional_validation/STATUS.md and the newest dated handoff; reconcile stale summary
  lines about Llama licensing with current run logs. There is currently no final local
  reports/functional_validation/llama directory. That absence is not a remote failure verdict.
- Read reports/publication_strategy_2026-10-26/future_work_and_novelty.md. No identical completed
  study was found in the bounded search; no “first ever” claim is authorized.
- The older reports/publication_direction_2026-10-26.md is an initial memo. This plan supersedes
  its smaller feature sample and its Arabic validation-language suggestion. The design preserves
  the established 404-feature comparison and uses Yoruba from the existing panel.
- This is a two-day research decision plan. A publishable broad claim will still need a second
  validated quantizer and possibly a larger-model check; do not hide those requirements.

## Files and ownership

Create only the new study files below. Existing routines remain evidence sources, not edit targets.

| File | Responsibility |
|---|---|
| scripts/g0/phase1_transfer_core.py | Four-cell effects, baseline subtraction and native/fixed-amplitude contrast |
| scripts/g0/phase1_transfer_run.py | Frozen manifest, model adapters, precision state restore, checkpoints, gates and execution metadata |
| scripts/g0/phase1_transfer_analysis.py | Clustered contrasts, eligibility, baseline comparisons, per-context/mean comparison and report |
| scripts/g0/run_transfer.sbatch | One resumable job wrapper, existing node exclusions and resource settings copied from validated runners |
| scripts/g0/tests_torch/test_phase1_transfer_core.py | Exact hand-computable factorial identities and dtype/position invariants |
| scripts/g0/tests_torch/test_phase1_transfer_analysis.py | Key alignment, held-out leakage prevention, clustered sample units and missing data |
| reports/intervention_transfer/freeze.json | All scientific choices, sample/data hashes, endpoints and budgets, frozen before Q outcomes |
| reports/intervention_transfer/STATUS.md | Own run ledger, elapsed/GPU accounting, amendments and active branch |
| reports/intervention_transfer/results/ | Generated tables and figure inputs; run IDs and source hashes required |
| reports/intervention_transfer/research_decision.md | Evidence for each candidate claim, failed hypotheses and next submission requirements |

If Branch A runs, add hooks in the new runner and analysis; do not create a new layerwise SAE pipeline.
If Branch B runs, add phase1_transfer_behavior.py and a corresponding focused test file; keep the
core freeze unchanged and add a branch-specific freeze. Do not scaffold either branch before activation.

### Task 1: Reconcile evidence and freeze an actual study, hours 0–4

**Inputs:** the spec; existing curve tables; validated remote run inventory; current model/SAE manifests.
**Outputs:** freeze.json and STATUS.md with no unresolved scientific configuration.

- [ ] Reproduce the new descriptive calculation:
  OMP_NUM_THREADS=4 ~/.venvs/anlp-torch/bin/python reports/publication_strategy_2026-10-26/reanalyse_existing.py .
  Expected INT4 medians: net scale 1.519910, control scale 1.515682, rho .017094.
  These are descriptive fits, not a discovery or an uncertainty estimate.
- [ ] Read the existing Llama comparison code and current remote metadata read-only. Record the
  completed/pending states and whether the model-specific manifest/loader is ready. Record queue
  occupancy once. Do not make repeated polling a research task.
- [ ] Freeze Gemma and Llama exact model/SAE/hook IDs, the old 404 feature sample hash for Gemma,
  up to 16 existing FP-positive contexts per feature, and the Llama stratified sample rule.
  For Llama use phase1_func_f5_sample.py's same per-language/role cap of three wherever those roles
  can be derived from existing tagging; cap total at 404 by stable SHA256 ordering if larger.
  If association roles are unavailable, stratify by language and FP-activation quartile, label
  the sample difference, and do not run new association tagging solely to match a label.
  A supporting sample cannot silently be presented as the same population.
- [ ] Freeze FP, INT16 floor, INT6 and INT4; fixed/native amplitude; all four FP/Q cells;
  real, matched-swapped and three fixed random direction draws. Random directions are auxiliary
  calibration directions, not extra independent biological/statistical samples.
- [ ] Freeze group-level evaluation: existing 27 languages, same 300 sentence IDs, K64 and three
  matched-control draws. Use paired sentence records, group and each draw separately.
- [ ] Define external validation before new outcomes: six languages eng_Latn, spa_Latn, rus_Cyrl,
  hin_Deva, zho_Hans, yor_Latn, from one versioned multilingual public corpus (prefer a fixed
  Wikimedia snapshot if its configuration/license permits). Resolve actual shard names at execution,
  record dataset revision, document IDs and checksum. Use 100 calibration and 300 evaluation
  passages/language, max 128 tokens, distinct source documents, exclude exact and normalized
  duplicates of FLORES material. If unavailable within four hours, record external validation as
  missing; do not replace it after examining outcomes. Existing-data results remain exploratory.
- [ ] Declare primary continuous endpoint and 10% relative held-out-MAE utility rule from the spec.
  Report all four cells, components and coverage; no post hoc winning-language selection.
- [ ] Freeze sample and budget before reading Q intervention outputs; commit only this study's
  explicit files if repository policy permits commits. Record commit/hash either way.

### Task 2: Implement the independently controlled intervention, hours 4–8

**Files:** phase1_transfer_core.py, phase1_transfer_run.py, core tests.
**Consume:** model-specific lower(tokens), upper_final(residual), logits_fn(hidden), encode(rows);
cache_state and fingerprint_feature semantics from phase1_func_f5_run.py.
**Produce:**
- effect(residual: Tensor, edit: Tensor, suffix: Callable, readout: Callable) -> dict
  with per-context centered effect vector, signed loss effect, FP/edit norms and native loss.
- factorial_effects(caches: dict, suffixes: dict, fp_edit: Tensor) -> dict
  keyed (prefix_state, suffix_state), each subtracting its own baseline.
- decompose(cells: dict) -> dict of C, W, I, total, preserving item keys.

- [ ] Write the small hand-computable tests before implementation. For F_s(h)=w_s h^2,
  h_FP=1, h_Q=2, v=.1, w_FP=1, w_Q=2, effects must be -.19, -.39, -.38, -.78;
  C=-.20, W=-.19, I=-.20, total=-.59. This detects a missing interaction and wrong baseline.
  Also test a linear suffix with zero interaction and unequal native amplitudes.
- [ ] Run the targeted core suite and verify it fails on the missing module.
- [ ] Implement the minimal core. Keep BOS unchanged; preserve the complete recipient residual
  except the decoded edit. Never replace the full residual with its SAE reconstruction.
- [ ] Add assertions for mismatched context keys, feature IDs and token positions, missing cells,
  nonfinite outputs, rounding/sham floors and a same-state hybrid equal to the native forward.
- [ ] Add model adapters. Gemma uses the existing split; Llama resid_pre at block 12 aliases
  block 11 resid_post. phase1_llama_run.py patches the T-LM runner, not automatically F5:
  adapt its loader/namespace deliberately and validate that split independently.
- [ ] Run tests, then ONE <=20-minute correctness/throughput job on frozen sample IDs. Measure
  total context throughput including full-vocabulary readout, memory and cache/weight loads.
  This is an instrument gate, not a scientific pilot used to choose hypotheses.
- [ ] Extrapolate full core + external validation time from this measurement with 50% contingency.
  If it exceeds the cap, drop random-direction expansion and optional Branch B first.
  Preserve all four core cells and the fixed population floor. If it still does not fit,
  record that the proposed complete study is infeasible in 48 hours; do not shrink to a noisy
  subset and imply a full result. No loop of increasingly broad smoke experiments.

### Task 3: Implement the analysis before the full outputs, hours 6–10

**Files:** phase1_transfer_analysis.py and focused analysis tests.
**Consume:** same-key per-context effects and freeze.json.
**Produce:** per_context.csv, component_summary.csv, coverage.csv, predictive_validation.csv,
figure-input tables, claim_status.json.

- [ ] Test exact key joins and coverage denominators; do not inner-join away missing/undefined
  cases. Test that a zero mean effect cloud can contain two large opposing per-context effects.
- [ ] Implement the exact vector decomposition and non-additive norm reporting.
  Compare similarity-of-means with mean paired similarity on the same supported cases.
- [ ] Implement the eligibility floor from calibration only, recording excluded counts and
  absolute errors. Tests must reject a floor recomputed from Q test outcomes.
- [ ] Use a hierarchical bootstrap over source-document/content clusters and reused features
  (2,000 replicates, fixed seed 20261026), preserving paired model conditions.
  For parallel FLORES sources, resample source IDs jointly across languages.
- [ ] Fit nested ridge baselines with calibration-only standardization and regularization:
  A: activation correlation, log FP activation, firing rate, log residual norm;
  B: A plus matched/random-direction sensitivity.
  Select ridge strength from {0.1,1,10} using calibration document folds. Hold feature groups
  out at evaluation; within each fold generic sensitivity must use independent control directions.
  Report within-state/model MAE and paired relative improvement. Do not pool INT4 versus INT6
  to create an easy bitwidth predictor.
- [ ] Run the hand-computable suite and all directly affected tests once. No blanket mutation
  campaign or unrelated refactoring. Scientific correctness and the executed gates are required.

### Task 4: Execute the decisive matrix, hours 8–30

**Files:** run_transfer.sbatch, frozen run manifest and STATUS ledger.

- [ ] Submit resumable model-specific runs from a pinned snapshot. At most two active Ada jobs,
  including earlier Claude jobs. State and cell completion must be atomic and keyed by hash.
- [ ] For each quantized suffix, evaluate both FP and Q cached residuals; for the FP suffix,
  evaluate all required residual states. Reuse unedited baselines across directions.
- [ ] Run the established feature subset and group panel. Retain per-context sufficient
  statistics; do not save only an across-context mean fingerprint. Native-amplitude endpoints
  must be explicit additional records, not substitutions for fixed-amplitude cells.
- [ ] Validate job completion, actual GPU line, CUDA warnings, gates, expected item counts,
  sample hashes and local/remote checksums before analysis. CPU fallback is a failed GPU run.
- [ ] Complete fresh-corpus validation if its freeze exists and the measured budget allows it.
  Never relabel old data as the new corpus.
- [ ] Record all completed, failed and missing cells. Analyze once the prespecified matrix is
  complete; publish no partial “positive” verdict from the first finished state.

### Task 5: Select a branch by evidence, hours 30–40

**Input:** complete main tables, intervals and coverage; spec routing table.
**Output:** one dated routing decision; at most one branch freeze.

- [ ] If the mechanism and held-out utility are established, keep the main direction; use budget
  on external validation. Do not start a backup because a coefficient is nonsignificant.
- [ ] Activate A only for per-context instability/large unexplained interaction: add the five
  predetermined downstream hooks on the same forwards, matched/random controls and held-out
  final-error prediction. Read FEGA's layerwise future-work passage and the current cross-layer
  and depth papers before asserting novelty. <=4 additional GPU-hours.
- [ ] Activate B only for the predeclared behavior distinction: read Deng and the full current
  translation-feature paper 2609.04808; freeze the six-language continuation task before Q
  generation, validate FP headroom on calibration, then run the spec's 300-prompt protocol.
  If language/coherence checking is not credible or <=6 GPU-hours does not fit, mark unavailable.
- [ ] Keep the same primary comparisons regardless of sign. Report every branch trigger and all
  unavailable endpoints. If both models disagree, report the model interaction; do not drop Llama.

### Task 6: Produce a paper decision, hours 40–48

**Files:** research_decision.md, figure sources, results manifest, handoff and append-only progress.

- [ ] Assemble four scientific figures: native activation-vs-effect discrepancy; crossed
  context/weight/interaction contrasts; per-context versus averaged continuity with controls;
  held-out predictive improvement or the one activated branch's behavioral result.
- [ ] For every proposed abstract sentence, list the supporting table, uncertainty, population,
  strongest baseline and limitation. Delete unsupported novelty, causal-mediation or deployment
  language from the proposed claim, while retaining the exploratory trail.
- [ ] Choose one status: supports a coherent paper direction / promising but missing external
  validation / precise negative boundary with practical relevance / inconclusive / stop.
  “Tests green” and “all jobs completed” are execution facts, never publishability criteria.
- [ ] List remaining submission requirements explicitly: second quantizer, larger model if
  necessary, fresh corpus if missing, behavioral significance, and updated novelty review.
  Do not claim these are complete because the 48-hour cap is reached.
- [ ] Append the actual findings to progress_log.md and prior_art.md, add a dated handoff and
  reproducible commands. Preserve the course report and frozen old studies. The human decides
  later whether to submit; no external upload follows from this plan.

## Planning self-review

The main experiment covers all three changing objects: background, suffix and amplitude.
The decomposition includes interactions and the per-cell baseline. Both conditional branches
have verified future-work parents, concrete triggers, direct prior-art threats and budgets.
All five review-focus hazards have explicit checks. Sample counts are sizing assumptions;
their uncertainty must be reported, and they are not guarantees of power. No further research
direction is authorized merely by an unfavorable outcome.

