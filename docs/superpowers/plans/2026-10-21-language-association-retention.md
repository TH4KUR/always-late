# Language Association Retention Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` to implement this plan task by task in the current chat. Checkboxes record future work; this document does not report completed implementation or authorize parallel agents.

**Goal:** Test whether features selected as language-associated before quantization retain that association across the same 27-language panel after quantization.

**Architecture:** Keep the model, frozen SAE, original feature cohort and input positions fixed. Add an optional collector to the existing quantization pass that saves each condition's own language-response profiles and paired sentence summaries. Analyse these sidecar artifacts separately from the historical Pearson-based results.

**Tech stack:** Existing Python/PyTorch/TransformerLens/SAE Lens environment, NumPy/SciPy for CPU analysis, unittest, the existing Ada launcher and artifact tooling. No new model or SAE training.

**Spec:** [Selected measurement design](../../../reports/survival_method_reassessment.md#7-selected-next-study-language-association-retention-d-035). That section defines the scientific quantities, populations, interpretation and uncertainty rules; read both documents before implementation.

**Status, 2026-10-21:** The user selected this question and requested a documented plan. Implementation and new measurements have not started. The numeric settings below are proposed engineering/analysis defaults, not choices independently approved by the user or validated by these future results.

## Global constraints

- `mask` only: one BOS in the input; exclude BOS and padding from feature statistics; do not append EOS.
- Gemma-2-2B; existing layer-12 Gemma Scope canonical 16k SAE; same RTN functions and frozen parameter manifest.
- Study INT8/7/6/5/4, with FP and restore-only INT0 plus INT16/12 numerical controls. Restore the original weights before each condition.
- Freeze the original `mask/eq/selectivity.pt` selective IDs, original language labels and saved frequency-matched controls. Do not retag the cohort, rematch controls or remove failures after inspecting quantized data. Q6 remains separate.
- Equal-token positions are primary; all valid content positions are a sensitivity on the same frozen cohort. Reconstruct the existing seed, split offsets, truncation and budgets from the original inputs/metadata and verify their hashes/counts.
- Existing FLORES dev selects the cohort; existing devtest evaluates pre/post differences. Both splits have already been examined: this is exploratory reassessment, not unseen confirmation.
- Preserve historical code behavior when collection is disabled, all saved numerical results, `third_party/`, and `paper/` prose. Write new outputs to a new run directory.
- Broader Q13-Q15 mechanism/confidence/reconstruction studies, interventions, probes, new SAEs and new corpora are outside this implementation.
- No cron, additional chat, new Duan comparison or retired `none` workload. No automatic commit/push or GPU submission is performed by writing this plan.

## Review focus

1. Quantized arrays must come from quantized activations, not `LangPearson`'s reference-side statistics (Task 2).
2. Inactive, tied, low-support and baseline-unstable features remain visible in fixed-denominator reporting (Tasks 3-4).
3. Every condition sees identical token IDs and selected positions; BOS/padding and batching do not change counts (Tasks 1-2).
4. Resampling respects paired conditions and FLORES translation alignment; token observations are not independent replicates (Task 4).
5. Missing/misordered languages, stale cohort IDs, incomplete condition files and resume duplicates fail validation rather than producing plausible partial results (Tasks 1-2 and 4).

## Task 1: Bounded artifact inventory and frozen input manifest

**Files:** create `scripts/g0/phase1_association_inventory.py`, `scripts/g0/tests/test_phase1_association_inventory.py`, `reports/association_retention/artifact_inventory.md`; later generate the manifest outside Git with the new run.

**Interfaces:** `build_inventory(paths: list[Path]) -> dict` lists availability and provenance without inventing absent statistics. The report records exact local/Ada/cache locations inspected, file/schema hashes, tensor keys/shapes/dtypes and the status of each required quantity: available, derivable, or new collection required. Tensor metadata inspection uses the existing trusted environment; do not scan credentials or print private environment values.

- [ ] Write tests with fixtures for FP-only response statistics, all-language correlations without quantized profiles, complete new profiles and mismatched language order. Require distinct availability verdicts and rejection of inconsistent provenance.
- [ ] Run `python3 -m unittest discover -s scripts/g0/tests -p 'test_phase1_association_inventory.py' -v`; confirm the new tests fail before implementation, then pass after the minimal implementation.
- [ ] Inventory the original local Phase 1 cache, its artifact manifest and the known Ada mask output location. Record uninspected locations explicitly. Do not describe this bounded search as a complete remote-cache search.
- [ ] Freeze model/tokenizer/SAE identifiers and resolved revisions or file hashes, feature IDs, original peak-language codes, language order, corpus hashes, token IDs, sentence IDs, selected-position masks, seed, split offsets, sequence length and budgets. Hash the original selection file and record cohort/control sizes.
- [ ] Decide whether existing artifacts satisfy every data requirement below. If sufficient, skip new GPU collection. If not, document exactly what is missing and estimate the new pass from a smoke measurement before full submission.

**Acceptance:** correlation arrays cannot be mistaken for response profiles; original cohort and input identity are reproducible. The inventory itself does not recompute survival or launch a job.

## Task 2: Optional per-condition response collector

**Files:** create `scripts/g0/phase1_association_collect.py`, `scripts/g0/tests_torch/test_phase1_association_collect.py`; modify `scripts/g0/phase1_run.py` and its existing runner tests only at the optional collection boundary.

**Interfaces:** add `profile_sink=None` to `run_survival(...)`. The sink implements `start_condition(condition: str)`, `update(batch: dict, features: Tensor)` and `finish_condition()`. `features` has shape `[batch, position, feature]`; the caller reshapes encoder output without modifying its values. Condition keys are `FP`, `INT8`, ..., `INT0`. Add CLI options `--association-profiles` and `--association-baseline PATH`; the latter points to the frozen original `mask/eq` directory, and is required when the former is enabled. Collection refuses a `none` regime. Existing runs without the new option retain their old outputs and return schema.

**New sidecar contract:** write `association/manifest.json`, `association/{eq,all}/profiles.pt` and per-condition sentence shards under `association/{eq,all}/sentences/`. `profiles.pt` has schema version, language codes, feature IDs `0..d_sae-1`, and per-condition `n_tokens[L]`, `positive_counts[L,F]`, `activation_sums[L,F]`, `activation_max[L,F]`. Counts are int64; sums/maxima float64 accumulated from the encoder outputs. Derived rates/means are computed from these totals, never copied from reference-side fields. Require all expected conditions and matching masks before marking a run complete.

For the fixed union of selective and saved control IDs, sentence shards retain language code, original aligned sentence ID, selected token count, positive counts and activation sums. Use sparse rows with a shared feature-index array, exact integer counts and float64 sums; unselected/zero rows still retain their sentence ID and token count. Stream shards by language/condition rather than building a `[condition,sentence,16384]` dense object. The manifest records the shard index, shapes, hashes and completion state. Refuse an existing nonempty output unless an explicitly verified resume protocol has been implemented; initial implementation may simply require a fresh directory.

- [ ] Write planted tests where reference and quantized responses differ: Hindi-only becomes Hindi-plus-English; another feature switches language; another becomes inactive. Assert that the saved quantized counts/sums differ from reference arrays and match direct calculations.
- [ ] Test FP/INT0 equality; condition restoration; exact count conservation between sentence shards and aggregates; all-token versus equal-position weighting; positive criterion `z > 0`; unequal sentence lengths; empty selected sentence rows; BOS/padding exclusion; batch-size invariance; missing conditions/languages and duplicate sentence IDs.
- [ ] Run the new test module to establish failure, implement the optional sink/CLI/schema, then run it and the existing `test_phase1_run.py` and `test_phase1_budget.py` modules. Default-path fixtures must remain unchanged.
- [ ] Estimate peak host/GPU memory, on-disk bytes and collection overhead on the smoke; save the extrapolation. Large artifacts stay outside Git and use the established HF artifact store when results are later collected.

**Acceptance:** quantized statistics measure the quantized side; aggregation reproduces saved sentence totals; the default runner remains compatible. Existing `LangPearson` fields are not silently redefined.

## Task 3: Baseline-anchored association metrics

**Files:** create `scripts/g0/phase1_association_metrics.py`, `scripts/g0/tests_torch/test_phase1_association_metrics.py` (uses the existing scientific-Python test environment).

**Interfaces:** `association_metrics(rates: ndarray, means: ndarray, original_peak: ndarray) -> dict[str, ndarray]` consumes finite nonnegative arrays `[language,feature]` and frozen original language indices. Return target rate/mean, strongest off-target response, target-minus-competitor margin, original-language share, global S, target-anchored S and Deng nu. `compare_profiles(fp: ndarray, quant: ndarray) -> dict` returns absolute changes and total-variation distance between normalized language profiles, with explicit undefined masks for all-zero vectors. `association_status(counts: ndarray, n_tokens: ndarray, original_peak: ndarray) -> dict` returns support/activity/peak-set flags and target-specific support.

Use equations and support defaults in the spec. Preserve ties as sets rather than resolving them by language-array order. A numerical equality tolerance of `1e-12` on float64 rates/means is for tie handling only, not a meaningful-change threshold. An all-zero profile has no peak. Invalid shapes, mismatched labels, zero language token totals, negative or nonfinite inputs are errors; zero feature activity is a valid measured outcome.

- [ ] Test identity, uniform positive scaling, off-target broadening with unchanged original-language responses, full language switch, equal-height peaks, inactivity, weak support and a dev-selected peak absent in FP evaluation. Scaling leaves normalized profile distance zero while absolute means change; broadening reduces target share despite unchanged target responses.
- [ ] Run `/home/karma/.venvs/anlp-torch/bin/python -m unittest discover -s scripts/g0/tests_torch -p 'test_phase1_association_metrics.py' -v` before and after implementation.
- [ ] Compare S and nu with the existing `phase1_lib` definitions on shared numeric fixtures; retain their different firing-rate versus mean-activation interpretations.

**Acceptance:** high original-language Pearson correlation is never a prerequisite for inclusion or sufficient evidence of retained specificity. Peak retention and profile preservation are different reported quantities.

## Task 4: Paired uncertainty analysis and report

**Files:** create `scripts/g0/phase1_association_analysis.py`, `scripts/g0/tests_torch/test_phase1_association_analysis.py`; later generate `reports/association_retention/{association_retention.md,feature_status.csv,language_summary.csv,analysis.json}` and a fresh-directory reproduction check.

**Interfaces:** CLI `--run DIR --out DIR --n-boot 1000 --seed 0`; `analyse_association(run_dir: Path, n_boot: int=1000, seed: int=0) -> dict`. Validate the complete manifest, 27-language roster, cohorts and condition/shard integrity before analysis. Recompute profiles and support on each resample; preserve the dev-selected IDs and original language labels.

- [ ] Write tests on paired sentence fixtures: fixed denominator conservation, zero-selective-feature languages yielding NA, asymmetric missing translations rejected, swapped language order rejected, undefined distances not changed to zero, and unchanged result when sparse shards are processed in another order.
- [ ] Implement a paired bootstrap over the 1,012 aligned devtest sentence IDs, drawing IDs with replacement and using the same multiplicities across all languages, FP and every rung. Keep each sentence's original token-selection masks fixed; bootstrap token totals may differ from the original equal budgets. Process resamples in bounded batches and benchmark CPU time/memory before committing to the full workload.
- [ ] Report 95% percentile intervals from 1,000 draws for aggregate fractions and median continuous changes, with seed recorded. Label them pointwise/descriptive, conditional on this cohort, corpus and sampled positions. Do not make simultaneous feature-level significance claims, run a new tier hypothesis family, or interpret an interval crossing zero as equivalence.
- [ ] Emit each original selective feature at each width, including baseline status, quantized status, original/current peak sets, support counts, target share, margin, TV distance, means, nu and the historical correlation when available. Include controls separately; their baseline labels remain their saved peak labels, not a claim of selectivity.
- [ ] Produce 27-language tables grouped by original dev language with full denominators, plus pooled feature-weighted and macro-language summaries (explicitly flag languages with zero cohort members). Show FP evaluation instability, Q inactivity/low support/ties/switches, and the association-retention fraction defined in the spec. All-token sensitivity uses the same cohort. No unit is silently dropped.
- [ ] Run the new analysis tests before/after implementation and regenerate the report twice from saved sidecars into fresh directories; compare canonical JSON/CSV values exactly with the same seed.

**Acceptance:** every selective feature is accounted for and old Pearson results can be compared to the new association outcomes without overwriting them. A retained peak with diminished target share is visible as such.

## Task 5: Smoke, full measurement and compatibility checks

**Files:** modify `scripts/g0/run_phase1.sbatch` only if required to pass through the optional arguments; it already forwards extra runner options. Record exact commands, commit/hash and artifact destinations in the new run manifest and report. Existing deployment/fetch/HF tools are reused; extend their explicit artifact lists only if needed for sidecars.

- [ ] Run both existing stdlib and torch suites using the established environments: `python3 -m unittest discover -s scripts/g0/tests -v` and `/home/karma/.venvs/anlp-torch/bin/python -m unittest discover -s scripts/g0/tests_torch -v`. Resolve environment availability before scheduling work; do not install a replacement stack casually.
- [ ] Smoke on English, Hindi and Khmer, 120 sentences each, FP plus INT6/INT4/INT0. Use a fresh run name and the frozen 27-language cohort IDs; mark the run as a subset so the full-panel analyser refuses to publish a headline fraction from it. Example future launcher: `sbatch scripts/g0/run_phase1.sbatch association-smoke flores mask --languages eng_Latn hin_Deva khm_Khmr --max-sentences 120 --bits 6 4 0 --skip-tagging --association-profiles --association-baseline /home2/karma/duan-verify/out/phase1/p1-mask/eq`.
- [ ] On identical smoke inputs, compare collection-on/off historical outputs; require unchanged outputs. Require exact FP/INT0 positive counts and peak sets; compare floating profiles at `rtol=1e-6, atol=1e-7` and stop to diagnose larger errors. Compare streamed totals with a direct small-batch calculation. These tolerances test numerical implementation, not scientific equivalence.
- [ ] Confirm the original full-run token/corpus/cohort hashes and freeze the analysis configuration before full measurement. Record smoke runtime, sparse density, disk estimate and CPU bootstrap estimate. If artifacts remain missing, run all 27 languages with FP plus `--bits 8 7 6 5 4 16 12 0`, `--skip-tagging` and the two new options in a new output directory. Do not overwrite `p1-mask`.
- [ ] Record job ID, launch command, revision and estimated finish; follow the existing one-check-per-invocation workflow. No polling automation. After completion, verify every expected condition and shard before analysis.
- [ ] Compare the new FP evaluation profiles and legacy correlations against the matching original mask inputs; investigate drift beyond the numerical tolerance rather than mixing incompatible runs. INT16/12 are measured numerical controls, not an assumption that every profile must remain unchanged.
- [ ] After code or result changes, run historical reproduction with `--streams mask/eq mask/all` into a fresh external work directory and the generated summary `--check`. Preserve historical numerical artifacts; the summary's provenance check may read historical outputs without restarting retired analyses. Do not accept a broad default that reruns `none` numerical analyses.

**Acceptance:** complete comparable sidecars exist for all nine conditions; smoke gates pass; changed input provenance and failed controls are diagnosed before claims are made.

## Task 6: Interpretation, documentation and completion

- [ ] Review agreement and disagreements between firing-rate profiles, activation-mean profiles, original-language correlation and controls. A language switch in one measure but not another stays in the report.
- [ ] Answer the selected question at every width and for each original language. State when the FP evaluation fails to reproduce the dev label, when activity/support is inadequate, and when the peak remains while specificity weakens. Absence of a detected change is not proof of unchanged function or meaning.
- [ ] Record whether the fixed SAE permits a clear association conclusion. If evidence suggests instrument mismatch, report the limitation and propose a bounded follow-up; do not automatically launch the wider Q15 or new-SAE work.
- [ ] Update the reassessment, progress log, current plan/progress, artifact inventory and next handoff with actual outcomes and uncertainty. Keep the source register's existing publication/scope limits. Describe this as measurement validation, not a novelty-certified paper contribution.

## Time budget and stop condition

Planning estimate: **1-2 calendar days** for this narrower language-profile study if GPU/cache access is ready and smoke gates pass; allow **2-4 days** for collection, storage or uncertainty-analysis revisions. Rough components: 3-6 h implementation/tests, 1-2 h smoke/review, 4-8 h full collection including I/O contingency, 2-6 h CPU analysis/reporting. Some work overlaps; these are unmeasured estimates. The previous 27-language mask job took 3 h 37 min, but did not save these sentence sidecars. Omitting tagging/logits may save time; sparse serialization and bootstrap may add time. Re-estimate after smoke; queue waits are additional.

Stop when all cohort members are accounted for, the complete ladder and controls have comparable profiles, uncertainty and sensitivity outputs reproduce, compatibility gates pass and the bounded question is answered with its limitations. Do not expand into semantic contrast sets, causal interventions, probes, new corpora or new SAEs to make this study appear complete.
