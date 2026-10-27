# GPTQ versus RTN: matched re-run of the project's quantization experiments (implementation plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (the 40-minute cron firings execute one bounded unit of this plan each; see `reports/gptq_extension/cron_prompt.md`). Steps use checkbox (`- [ ]`) syntax. The ledger is `reports/gptq_extension/STATUS.md`; the plan is never rewritten, only amended by dated "Update" lines.

**Goal:** Re-run the project's quantization experiments with weights chosen by GPTQ (and by a matched RTN comparator) instead of the legacy RTN, changing nothing else, so that "does the old RTN finding depend on the quantizer?" gets a paired answer.

**Architecture:** A new package `scripts/g0/gptq_ext/` builds one immutable dense bf16 weight set per (method, bit width) by sequential calibration on the existing TransformerLens Gemma-2-2B model (pinned upstream GPTQ core, canonical `[out, in]` matrix adapter), and a small weight-state shim replaces the injected RTN callable of the existing entry points. Experiment code, freezes, data, directions, batching and analyses stay as they are; each method writes to its own output directory.

**Tech Stack:** Python, PyTorch 2.14, TransformerLens 3.9.0 (cluster) with a CPU-only local copy for tests, the pinned authors' GPTQ core (IST-DASLab/gptq commit `2d65066e`), Slurm on Ada (RTX 2080 Ti 11 GB, offline compute nodes).

**Spec:** [`reports/quantization_gptq_audit_2026-10-27.md`](../../../reports/quantization_gptq_audit_2026-10-27.md) (the fair-comparison audit; its section numbers are cited below as "audit §n"). User request that starts this work, 2026-10-27 about 17:25 IST: "check the new documentation on using GPTQ instead of RTN. continue the cronjob in similar way as before. run tests again with GPTQ quantization and try to keep other things same as before so that we can draw a fair comparison. cronjob frequency 40 minutes". This request lifts the HOLD of 2026-10-27 09:50 **for this extension only** (D-101).

## Global Constraints

- Unchanged from the old experiments (audit §1): Gemma-2-2B and the exact reference checkpoint, TransformerLens loader and version, Gemma Scope layer-12 canonical 16k SAE, token IDs, the BOS-masked regime (mask only), feature groups, feature/context samples, decoder directions, FP-active edit positions, batching, masks, gradient target, arithmetic precision and readout; both fixed-FP-amplitude and native-amplitude edits.
- Primary GPTQ setting (audit §4): symmetric grid, one scale per output channel (no input grouping), `blocksize=128`, damping 1 % of the mean Hessian diagonal (`percdamp=0.01`), no activation ordering, no MSE clipping search. A tuned variant may be added only by a calibration-only rule fixed in advance; intervention outcomes never select a setting.
- Calibration (audit §6): text independent of FLORES dev/devtest and of the external Wikipedia articles; 8,192 content tokens per language x 27 languages = 221,184; sequences of at most 128 content tokens, each with its own real BOS; padding excluded from the Hessians; one fixed multilingual pool for every configuration; frozen (dataset revision, article ids, token ids, seed, counts) before any artifact is built. A language that cannot reach its budget is reported and settled with the user, never filled with English.
- Calibration is sequential through the whole model (Q/K/V, then O, then input and gate, then MLP output, block after block, inputs recomputed after the quantized earlier stages), on the actual normalised projection inputs.
- One immutable artifact per configuration: prefix and suffix weights of a crossed experiment come from the same artifact; setup never recalibrates.
- Arms: **legacy RTN** (existing results, never rerun), **matched RTN** (same canonical matrices, same grid routine and FP32 quantization arithmetic as GPTQ, no Hessian compensation) and **GPTQ**. The INT16 and INT12 states and the INT0 restore control stay the legacy RTN rounding reference in every arm (audit §7: labelled a small weight perturbation, not a pure noise estimate).
- Never edit `third_party/`, the pristine Duan source, the original freezes or any old result. Add only; each new result has its own directory; no RTN survival file is joined to a GPTQ file by bit width.
- Standing project rules apply (cron prompt, hard rules 3 and 4): progress documents are never shortened; stage explicit paths; secret scan before every commit; HF dataset private; at most two Ada jobs running; deploy only named snapshots while a job runs; read the `GPU 0:` line of every job log.

## Review Focus

- A language of the 27 has too little calibration text (Lao, Igbo, Yoruba, Amharic are the likeliest): expected behaviour is a loud shortfall report and a stop, not silent English filling. Test: `test_pool_reports_shortfall_and_never_fills`.
- A Hessian with dead input columns or a failed Cholesky: expected behaviour is a counted, reported event, never a silent zero or a crash that loses the run. Test: `test_dead_columns_and_damping_failures_are_counted`.
- A node whose CUDA fails runs on the CPU and passes every gate (happened 2026-10-26, job 40402): expected behaviour is refusal. The existing `nvidia-smi` guard of the sbatch files is kept in the new sbatch; Task 8 checks it in the log.
- The same artifact built on two different nodes differs in the last bits: expected behaviour is that the job refuses to start and says which tensor differs (hash recorded in the freeze). Test: `test_artifact_hash_detects_one_flipped_value`; Task 8 compares two nodes.
- Running an entry point without `--weights` must behave exactly as before (default RTN path untouched). Test: `test_default_weights_flag_leaves_the_legacy_quantizer_in_place`.

## Fixed design decisions of this plan (all recorded in D-101; deviations need a new D-entry)

1. **Arms and bits.** Priority bits {4, 6} (strongest distortion and the two states of the transfer study), then {8, 7, 5} for the ladder studies. GPTQ and matched RTN for every study that is run; GPTQ first in the queue, matched RTN next.
2. **Why matched RTN.** The legacy quantizer shares one Q/K/V scale across heads (audit §3) and its grid differs from the authors' (scale 1/7 against 2/15 at four bits). A legacy-RTN versus GPTQ difference therefore mixes algorithm, granularity and grid; only matched RTN versus GPTQ isolates error compensation. Legacy RTN versus matched RTN measures the granularity and grid change.
3. **BOS in the Hessian.** Real BOS stays in the calibration forward passes and its row stays in the Hessian (audit §6 excludes only padding). A BOS-excluded variant is a calibration-only sensitivity check at INT4, run only if the primary arm shows a sink-related anomaly.
4. **Storage.** `/home2` has 2.7 GB free (checked 2026-10-27 17:3x IST) and compute nodes do not mount `/share1`. Artifacts are therefore **rebuilt at the start of every job into node-local memory or `/scratch`**, and the job refuses to continue unless the artifact's sha256 equals the value recorded in the extension freeze after the first build. A compact copy is persisted only if space allows (Q30).
5. **Endpoints, declared now (before any GPTQ number exists).** E1 ordinary degradation: per-language mean next-token NLL minus FP on the frozen FLORES devtest sentences (nats per token). E2 Phase-1 survival: share of active features with token correlation >= 0.9 per language, and the three pre-set tests T1-T3 recomputed by the unchanged analysis. E3 association retention (the D-035 statuses). E4 group-removal effect: effect in the quantized state divided by the FP effect, per language, K64 group, median over languages. E5 rescue: share of the quantization-induced loss that FP-donor patching recovers. E6 transfer: C, W, I (context, weights, interaction) and the pre-set UTILITY rule. E7 Belebele accuracy under group removal (K64 and K = all). Contrasts are always paired by identical keys (language, sentence, feature, context, direction); the comparison files are labelled exploratory; a null stays a null.
6. **Order (audit §11).** Build and gates; then E1 and E2; then E6 (the central fixed/native transfer matrix); then E4, E5, E3, E7 and the F4/F5/Study G panels; then the external panel. The order is fixed here and is not changed by any result.

## File Structure

- `scripts/g0/gptq_ext/__init__.py`, `upstream/` (pinned `gptq.py`, `quant.py`, `LICENSE`, `PROVENANCE.json`): the vendored authors' core, byte-identical, hash-checked.
- `scripts/g0/gptq_ext/canonical.py`: TransformerLens parameter <-> canonical `[out, in]` matrix mapping and the target-name filter.
- `scripts/g0/gptq_ext/grids.py`: the authors' symmetric grid, `rtn_matched`, and the legacy scale for contrast tests.
- `scripts/g0/gptq_ext/calibration.py`: the multilingual calibration pool (selection, exclusions, freeze, batches).
- `scripts/g0/gptq_ext/calibrate.py`: the sequential block-by-block GPTQ driver on a TransformerLens model.
- `scripts/g0/gptq_ext/artifact.py`: artifact hash, in-place apply, untouched-tensor check, manifest.
- `scripts/g0/gptq_ext/states.py`: `add_weights_args`, `install`: the shim for the five entry points.
- `scripts/g0/gptq_ext/compare.py` (later): paired tables arm against arm.
- `scripts/g0/run_gptq_build.sbatch`: build and gate job; existing sbatch files get an optional `WEIGHTS` argument only where needed.
- `reports/gptq_extension/`: `STATUS.md` (ledger), `cron_prompt.md`, `freeze.json` (Task 7), `README.md` and results per stage.
- Tests: `scripts/g0/tests_torch/test_gptq_ext_*.py` (torch; TransformerLens tests need the local CPU environment of Task 2).

## Tasks

### Task 1: vendor the authors' GPTQ core

**Files:** create `scripts/g0/gptq_ext/__init__.py`, `scripts/g0/gptq_ext/upstream/{gptq.py,quant.py,LICENSE,PROVENANCE.json}`; test `scripts/g0/tests_torch/test_gptq_ext_upstream.py`.
**Produces:** `from gptq_ext.upstream import gptq as ug, quant as uq` with `ug.GPTQ`, `uq.Quantizer`, and `gptq_ext.upstream.load()` returning both after verifying the sha256 values of the audit (`gptq.py` bb47468c..., `quant.py` 21152856...).

- [ ] Step 1: write `test_upstream_files_match_pinned_hashes` (fails: files absent) and `test_identity_hessian_gptq_equals_same_grid_nearest_rounding` for bits 4 and 6 in float32 (the audit's CPU check, with the Transformers `Conv1D` import and CUDA synchronise stubbed in the loader only, optimisation arithmetic untouched).
- [ ] Step 2: fetch the two files and the licence from the pinned commit with a one-off script into the new directory (record commit, URL, sha256 in `PROVENANCE.json`); the loader stubs only what the audit stubbed.
- [ ] Step 3: run `~/.venvs/anlp-torch/bin/python -B -m unittest discover -s scripts/g0/tests_torch -p 'test_gptq_ext_upstream.py'` -> PASS; mutation check by flipping one byte of a copy (the hash test must fail).
- [ ] Step 4: commit (`gptq_ext: vendor the pinned GPTQ core`).

### Task 2: CPU test environment with TransformerLens 3.9.0 and the canonical adapter

**Files:** create `scripts/g0/gptq_ext/canonical.py`; test `test_gptq_ext_canonical.py`; environment `~/.venvs/anlp-tl` (outside the repository, same TransformerLens, transformers and torch versions as Ada).
**Produces:** `is_target(name: str) -> bool` (identical to the legacy `target_tl_weight`: substring test on `W_Q, W_K, W_V, W_O, W_in, W_gate, W_out`), `to_linear(name: str, w: Tensor) -> Tensor` (Q/K/V `[h,m,d]` -> `[h*d, m]`; O `[h,d,m]` -> `[m, h*d]`; MLP `[in,out]` -> `[out,in]`; 1-D tensors returned unchanged and flagged), `from_linear(name: str, lin: Tensor, like: Tensor) -> Tensor`, `targets(model) -> list[tuple[str, Parameter]]` in a fixed order (stored `_W_K`, `_W_V` for grouped-query attention, never the expanded views).

- [ ] Step 1: create the venv with the Ada versions (read them from `$BASE/env` on Ada: torch 2.14.1, transformer_lens 3.9.0, transformers 5.19.0); no model weights, no network access to gated files.
- [ ] Step 2: tests: `test_round_trip_is_exact_for_every_target_kind`, `test_target_filter_equals_the_legacy_filter_on_a_tiny_gemma2_model` (random-weight `HookedTransformer` with a Gemma-2 config: 2 layers, 4 heads, 2 key/value heads, d_head 8, d_model 32, d_mlp 64; the set of names must equal the legacy filter's set), `test_gqa_uses_stored_kv_parameters`, `test_canonical_scale_count_is_out_channels` (12 canonical scales against 3 legacy scales for a 4-head example, the audit's number), `test_projection_agrees_with_the_model_forward` (a canonical `Linear` applied to the normalised input equals the TransformerLens projection).
- [ ] Step 3: implement; run; commit.

### Task 3: grids and matched RTN

**Files:** create `scripts/g0/gptq_ext/grids.py`; test `test_gptq_ext_grids.py`.
**Produces:** `authors_grid(w: Tensor, bits: int) -> tuple[Tensor, Tensor]` (scale, zero per output row via the upstream `Quantizer`, `sym=True`, per-channel, no MSE search), `rtn_matched(w: Tensor, bits: int) -> Tensor` (dequantised, FP32 arithmetic, returned in `w.dtype`), `legacy_scale(w: Tensor, bits: int) -> Tensor`.

- [ ] Step 1: tests: `test_four_bit_scale_is_two_fifteenths_for_authors_and_one_seventh_for_legacy` (row spanning [-1, 1]; the audit's numbers), `test_matched_rtn_equals_gptq_with_identity_hessian`, `test_matched_rtn_levels_are_within_the_grid`, `test_matched_rtn_is_deterministic_and_dtype_preserving`.
- [ ] Step 2: implement; run; commit.

### Task 4: the calibration pool

**Files:** create `scripts/g0/gptq_ext/calibration.py`; test `test_gptq_ext_calibration.py`; outputs `reports/gptq_extension/calibration_freeze.json` (ids, counts, hashes; no text) and the text files on Ada.
**Produces:** `select_passages(rows_by_language, budgets, exclude_article_ids, exclude_texts, seed) -> pool` where `pool = {"languages": {code: [{"article_id", "sha256", "n_tokens"}]}, "shortfall": {code: int}}`; `pool_batches(pool, encode, bos_id, pad_id, max_len=128) -> list[dict]` (batches with `tokens`, `valid`, `lang`); `pool_sha256(pool) -> str`.

- [ ] Step 1: tests: `test_pool_reports_shortfall_and_never_fills`, `test_pool_excludes_external_article_ids_and_normalised_flores_duplicates`, `test_budget_is_content_tokens_per_language_and_sequences_have_one_bos`, `test_pool_is_deterministic_for_a_seed`.
- [ ] Step 2: implement, reusing the passage rules of `phase1_transfer_panel.py` (same Wikipedia dump: `wikimedia/wikipedia`, config date 20231101, revision b04c8d1c; same sentence and length rules; the six external languages' calibration and evaluation article ids are excluded). Download on the Ada login node (internet is available there, as for the external panel); the 27-language pool is built there and its freeze committed.
- [ ] Step 3: run the tests; run the build; record shortfalls in the ledger; if any language is short, add a Q-item with the default "report and continue with the shorter budget for that language only" and stop that branch of the plan until the user answers; commit.

### Task 5: the sequential calibrator

**Files:** create `scripts/g0/gptq_ext/calibrate.py`; test `test_gptq_ext_calibrate.py` (tiny TransformerLens Gemma-2 model from Task 2).
**Produces:** `GptqConfig(bits: int, blocksize=128, percdamp=0.01, actorder=False, groupsize=-1, sym=True, mse=False)`; `calibrate(model, batches, cfg, device, method: str, log) -> dict[str, Tensor]` mapping each target parameter name to its dequantised tensor in the parameter's own shape and dtype (`method` is `"gptq"` or `"rtn_matched"`; the latter skips the Hessian and uses `rtn_matched`), plus `report(): dict` (dead columns, damping retries, non-finite counts, peak memory).

- [ ] Step 1: tests: `test_rtn_matched_path_equals_grids_rtn_matched`, `test_gptq_never_increases_layer_output_error_against_rtn_on_calibration_inputs` (the objective GPTQ minimises, per projection, on the tiny model), `test_calibration_is_deterministic`, `test_padding_rows_do_not_enter_the_hessian`, `test_stages_see_the_inputs_of_the_already_quantized_earlier_stages`, `test_non_target_tensors_are_unchanged`, `test_dead_columns_and_damping_failures_are_counted`.
- [ ] Step 2: implement block by block: capture `ln1.hook_normalized` (Q/K/V), `attn.hook_z` flattened (O), `ln2.hook_normalized` (input and gate), `mlp.hook_post` (output); quantize a stage, re-forward the block with the quantized weights for the next stage, then forward the block to produce the next block's inputs; temporary `torch.nn.Linear` proxies around canonical matrices feed the upstream `GPTQ`; offload inactive blocks; FP32 Hessians.
- [ ] Step 3: run; commit.

### Task 6: artifact identity and the weight-state shim

**Files:** create `scripts/g0/gptq_ext/artifact.py`, `scripts/g0/gptq_ext/states.py`; modify the five entry points (`phase1_run.py:312`, `phase1_func_run.py:484,619`, `phase1_func_f5_run.py:386`, `phase1_transfer_run.py:477`, `phase1_transfer_precision_main.py:175`) by three lines each; tests `test_gptq_ext_artifact.py`, `test_gptq_ext_states.py`.
**Produces:** `artifact_sha256(tensors: dict[str, Tensor]) -> str`; `apply(model, tensors) -> None` (copies into the existing parameters, nothing else touched); `add_weights_args(ap)` adding `--weights {rtn,rtn-matched,gptq}` (default `rtn`), `--gptq-freeze PATH`, `--calibration-text DIR`; `install(env, model, args, log) -> dict` returning the weights manifest and replacing `env["quantize_rtn_per_output_channel_tl"]` by a callable with the same signature `(model, bits)`: bits 12 and 16 stay legacy RTN, bits 4 to 8 apply the artifact of the method, other bits raise.

- [ ] Step 1: tests: `test_default_weights_flag_leaves_the_legacy_quantizer_in_place`, `test_artifact_hash_detects_one_flipped_value`, `test_apply_then_restore_returns_exact_original_values`, `test_switching_states_repeatedly_is_exact`, `test_install_refuses_when_the_hash_differs_from_the_freeze`, `test_bits_16_and_12_use_the_legacy_rounding`, `test_resume_refuses_an_output_directory_with_another_weights_manifest`.
- [ ] Step 2: implement; the manifest holds method, config, calibration pool sha256, code sha256s, artifact sha256 per bit width; it is written to `<out>/weights_manifest.json` and copied into `meta`.
- [ ] Step 3: run the new tests and the existing torch suite (`-p 'test_phase1_*.py'`) to show the default path is unchanged; commit.

### Task 7: the extension freeze

**Files:** create `reports/gptq_extension/freeze.json`, `reports/gptq_extension/README.md`.
Written and committed before any real-model calibration: the Global Constraints values, the endpoints E1-E7 and the pairing keys, the arms and bit order, the pool freeze hash, the upstream hashes, the dead-column policy (columns with zero input energy keep their rounded values and are counted), the tolerance table of the gates of Task 8 (FP bridge: identical to the historical FP values; split/full and sham as in the old runs; artifact hash identical on two nodes).

- [ ] Step 1: write the freeze and its test (`test_freeze_names_every_endpoint_and_pins_hashes`); commit.

### Task 8: deploy, instrument job, gates

**Files:** `scripts/g0/run_gptq_build.sbatch`; ledger entries.
- [ ] Step 1: deploy a named snapshot (`deploy_phase1.sh repo_<short commit>`; the deployed paths list gets `scripts/g0/gptq_ext` automatically because `scripts/g0` is shipped; add `reports/gptq_extension/freeze.json calibration_freeze.json` to `PATHS`).
- [ ] Step 2: instrument job A (one GPU, 40 minutes): build the pool batches, calibrate INT4 for both methods on the real model with the real SAE untouched; report time, peak memory, dead columns, damping retries; gates: all targets present, non-targets byte-identical, exact round trips, FP bridge, split/full, zero-amplitude sham in the extension path. Instrument job B on a different node: the same build; the two artifact hashes must be equal. A mismatch is a finding: stop, add a D-entry, decide with the user how to persist artifacts (Q30).
- [ ] Step 3: record the hashes in `freeze.json` (an Update, not an edit of earlier fields); write the handoff.

### Task 9: stage 1, ordinary degradation and survival (E1, E2)

- [ ] Step 1: queue all of stage 1 at once (user rule of 2026-10-25): `phase1_run.py` with `--weights gptq` and `--weights rtn-matched` for the full ladder in the frozen protocol, output directories `out/phase1/gptq-*` and `rtn-matched-*`; E1 from the functional T-LM NLL run of the same states.
- [ ] Step 2: validate each job (gates in the log, complete, hashes equal the freeze); analyse with the unchanged Phase 1 analysis; `compare.py` writes the paired legacy/matched/GPTQ tables; report with limits; handoff.

### Task 10: stage 2, the transfer study (E6)

- [ ] `phase1_transfer_run.py` with `--weights gptq` and `rtn-matched`, states 6 and 4 (INT16 control stays legacy), feature level and group level, FLORES; analysed by the unchanged `phase1_transfer_analysis.py` rules (UTILITY rule as pre-set) plus the paired arm contrasts; D-entry per analysis; external panel in Task 12.

### Task 11: stage 3, functional panels (E3, E4, E5, E7, F4, F5, Study G)

- [ ] Same pattern per panel: queue all, validate, analyse unchanged, paired comparison. E4/E5 (F1-F3 zero-ablation and rescue at INT8-INT4), E3 (association retention collection and analysis), E7 (Belebele, K64 and K = all), then F4, F5 and Study G only with the time left. Each panel states in the ledger which old study it repeats and the file of its old result.

### Task 12: stage 4, the external Wikipedia panel

- [ ] The six-language panel with the calibration pool's exclusion list already in force; same rules.

### Task 13: comparison report and documentation

- [ ] `reports/gptq_extension/RESULTS.md` in plain words (arm against arm, each endpoint, limits, what changed and what did not), a section in `reports/PROJECT_RESULTS.md` with the generated per-language tables extended, index and status pages, a D-entry for every analysis, small result files and logs to the private HF dataset and wandb in the fixed schema (no weights).

### Task 14: close

- [ ] Final handoff; list what was run and what was not (honestly, per stage); the user decides whether the findings enter the course report.

## Self-review

- Spec coverage: audit §1 (Global Constraints, decision 1), §2 (Tasks 2, 6, five entry points), §3 (Tasks 2 and 3), §4 (decisions 1-2, Task 3), §5 (Architecture, Task 5), §6 (Tasks 4 and 5), §7 (Task 8 gates, decision 5), §8 (Task 8; gates 1-7 map to Tasks 1, 2, 3, 5, 6, 8), §9 (endpoints), §11 (order). Not covered on purpose: the packed-kernel deployment validation of audit §5 and a tuned GPTQ configuration (audit §4 optional rows) stay out of scope unless the user asks.
- Open risks recorded in the ledger: disk (Q30), cross-node artifact reproducibility (Task 8), calibration shortfall (Task 4), compute (four days to the 31 October deliverables; stage order protects the central results).

## Update 2026-10-27 17:38 IST (Task 1, after reading the vendored upstream code in full)

Three facts of `gptq.py` at the pinned commit that the plan must respect (they refine, not change, the design):

1. **Dead columns.** `fasterquant` sets `H[dead, dead] = 1` and **zeroes** `W[:, dead]` for every input column whose Hessian diagonal is exactly 0. Task 5 follows upstream (the algorithm is not modified) and counts such columns per matrix; the matched-RTN arm applies the identical zeroing mask so that the two arms differ only by the error compensation (audit section 6). The earlier wording of Task 7 ("keep their rounded values") is superseded by this line.
2. **No damping retry.** `torch.linalg.cholesky` raises if the damped Hessian is not positive definite; upstream has no fallback. Task 5 records the failure (matrix name, damping, diagonal statistics) and aborts that build; a retry with a larger damping is a D-entry and a new protocol line, never a silent default. The test name `test_dead_columns_and_damping_failures_are_counted` of Task 5 therefore covers dead columns (counted) and the failure (reported and raised).
3. **Hessian weighting.** `add_batch` treats the first dimension as the number of samples; a 2-D input counts as one sample, and the running average gives `H = (2 / K) * sum_k X_k^T X_k` over the K calls. The result is a scalar multiple of the plain sum, and the algorithm is invariant to that scalar (damping is relative to the mean diagonal). Task 5 therefore passes the valid (non-padding) rows of each calibration batch as one 2-D tensor per call, in a fixed batch order (the accumulation order is part of the artifact hash).

Task 1 result: `scripts/g0/gptq_ext/upstream/` holds the three files byte for byte (hashes equal the audit's); `upstream.load()` verifies them and extracts `GPTQ`, `Quantizer`, `quantize` by AST; substitutions: dummy `transformers.Conv1D`, no-op `torch.cuda.synchronize()` without CUDA, TF32 switches off. 10 tests pass (`scripts/g0/tests_torch/test_gptq_ext_upstream.py`), 8 of 8 applicable mutants killed.
