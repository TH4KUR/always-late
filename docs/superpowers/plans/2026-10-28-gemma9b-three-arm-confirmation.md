# Gemma-2-9B three-arm confirmation: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the prepared Gemma-2-9B Jarvis run confirmatory for the project as it now stands: three quantizer arms (legacy RTN, matched RTN, GPTQ), tier 1 (Study G) and tier 2 (Phase 1 survival and loss), predictions frozen before any 9B number, everything built and tested up to the point of running.

**Architecture:** Nothing hash-pinned is edited. A new wrapper module (`phase1_gemma9b_ext.py`) puts the adapter's frozen 9B setting around the old entry points; `gptq_ext/run_with_weights.py` swaps the quantizer from outside. `jarvis_run_all.sh` (Shardul0007) is extended in place with arms, stages, gates and a Blackwell-aware env. A new analysis module computes the frozen predictions; a dated amendment freezes their text.

**Tech Stack:** Python (stdlib `unittest` in `scripts/g0/tests`, torch tests in `scripts/g0/tests_torch` run in `~/.venvs/anlp-torch`, TransformerLens tests in `~/.venvs/anlp-tl`), bash, TransformerLens 3.9.0, sae-lens 6.54.5, torch 2.14.1.

**Spec:** `docs/superpowers/specs/2026-10-28-gemma9b-three-arm-confirmation-design.md` (approved by the user 2026-10-28). Worktree `/home/karma/Documents/ANLP/project/gemma9b-v2`, branch `gemma9b-confirmation-v2` (main + both 9B branches merged).

## Global Constraints

- Never edit `third_party/`, `scripts/g0/phase1_run.py`, `phase1_func_run.py`, `phase1_gain_run.py`, `their_code.py`, `phase1_lib.py`, `phase1_gemma9b_run.py`, `run_gemma9b.sbatch`, `tests_torch/test_phase1_gemma9b_run.py`, `reports/g0/phase1_languages.json`, `reports/functional_validation/gemma9b/freeze.json`, `reports/gptq_extension/*` (all hash-pinned or frozen). `scripts/g0/tests_torch/gptq_ext_tiny.py` may only get backward-compatible additions.
- Tests first and watched to fail; mutation check with `scripts/g0/mutcheck.py` on the new analysis and guard modules (as the repository does).
- Arms: `legacy` (`--weights rtn`), `rtn-matched`, `gptq`. Tier 1 states: legacy FP 8 7 6 5 4, new arms 6 4. Tier 2 bits: legacy `8 7 6 5 4 16 12 0`, new arms `7 6 4 16 12 0`. INT16, INT12 and INT0 stay the legacy rounding in every arm.
- `GEMMA9B_N_DEVICES=1`; calibration pool and freeze are the 2B ones (`reports/gptq_extension/calibration_freeze.json`, pool sha256 `543234593913c4434b42bac71de8231618afbd5a035f248c1e13d284f73dd816`).
- Predictions are those of spec section 5 (P1 to P3b unchanged; Q0 to Q3; S1 to S4); `req(x) = ceil(x * n / 27)` is `ga.required(n, x)`; tested languages = group of at least 16 members (`lc.tested_languages`).
- Machine: 1x RTX PRO 6000 Blackwell (cc 12.0), 96 GB VRAM, 160 GB RAM, 250 GB SSD; torch 2.14.1 from the cu130 index when cc >= 12.0 (driver >= 580), cu126 otherwise.
- No token, weight or pool file in git. Commit with explicit paths, trailer `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`; push nothing without the user's word.

## Review Focus

1. Only one GPU and 160 GB RAM: the driver must plan one process per GPU slot and never oversubscribe RAM (`PROCS_PER_GPU=2` only when two artifact-sized jobs fit).
2. A tokenizer that differs from the 2B one (different vocab, special ids or probe encodings) must stop the run before any artifact is built.
3. A GPU with compute capability 12.0 but a driver below 580 (or a missing `nvidia-smi`) must give one clear message, not a failed pip install or a CPU run.
4. A resume after a partial run must skip exactly the finished (stage, arm, state) units, and a GPTQ stage job must refuse to start when `artifact_hashes.json` for 9B has not been recorded by the gate step.
5. A missing or partial arm (a language absent from a state, a state absent from an arm) must make the analysis stop with a named error, never silently drop it; Q1 with a legacy arm that shows no amplification must be reported "not evaluable", not "failed".

## File map

- Create `scripts/g0/gptq_ext/tokenizer_check.py`, `scripts/g0/gptq_ext/ram_guard.py`, `scripts/g0/gptq_ext/record_hashes.py`, `scripts/g0/phase1_gemma9b_ext.py`, `scripts/g0/make_gemma9b_amendment.py`, `scripts/g0/phase1_gemma9b_compare.py`, `scripts/g0/make_jarvis_bundle.sh`.
- Modify `scripts/g0/jarvis_run_all.sh` (extend), `scripts/g0/tests/test_jarvis_run_all.py` (extend), `scripts/g0/tests_torch/gptq_ext_tiny.py` (optional `dims=` parameter), `CLAUDE.md`, `reports/functional_validation/gemma9b/JARVIS_HANDOFF.md`.
- Create tests: `tests/test_gptq_ext_tokenizer_check.py`, `tests/test_gptq_ext_ram_guard.py`, `tests/test_gptq_ext_record_hashes.py`, `tests/test_make_gemma9b_amendment.py` (stdlib); `tests_torch/test_gptq_ext_gemma9b_shapes.py`, `tests_torch/test_phase1_gemma9b_ext.py`, `tests_torch/test_phase1_gemma9b_compare.py`.
- Create docs: `reports/functional_validation/gemma9b/amendment_A1_three_arm.json`, `reports/handoffs/<next number>-...md`, D-entry in `reports/decisions_and_anomalies.md`.

### Task 1: Calibrator and canonical adapter on the 9B layout

**Files:** Modify `tests_torch/gptq_ext_tiny.py`; Create `tests_torch/test_gptq_ext_gemma9b_shapes.py`.

**Interfaces:** Produces `tiny_gemma2(n_layers=2, dtype=..., seed=0, scale=0.3, dims=None)` where `dims` overrides `DIMS` (same keys); default behaviour unchanged. Consumes `canonical.to_linear/from_linear/targets`, `calibrate.calibrate(model, batches, cfg, method, device, log, observer)`.

- [ ] **Step 1: Write the failing tests** (run in `~/.venvs/anlp-tl`; skip when TransformerLens is absent) with the 9B layout `dims = {d_model: 32, d_head: 16, n_heads: 4, n_kv: 2, d_mlp: 64, d_vocab: 100}` (inner attention width 64 != d_model, as in Gemma-2-9B 16x256 vs 3584): `test_canonical_round_trip_for_every_target_when_heads_times_dhead_differs_from_dmodel`; `test_identity_method_reproduces_the_original_activation_at_every_stage` (observer rows equal the hooked activations of the unmodified model); `test_matched_rtn_equals_direct_rounding_of_each_canonical_matrix`; `test_gptq_stage_error_is_not_worse_than_matched_rtn_at_4_bits` (sum over targets of the reported `error` or the output MSE on the calibration rows); `test_model_is_left_exactly_as_found_after_calibrate`.
- [ ] **Step 2:** Run `~/.venvs/anlp-tl/bin/python -B -m unittest tests_torch.test_gptq_ext_gemma9b_shapes -v` from `scripts/g0`; expect failures (`dims` unknown).
- [ ] **Step 3:** Add the optional `dims=None` parameter to `tiny_gemma2` (merge over `DIMS`); fix any real shape assumption the tests expose in `canonical.py` or `calibrate.py` only if a test proves a defect (then also a D-entry line).
- [ ] **Step 4:** Run the new tests and the existing `test_gptq_ext_*` tests; expect all pass.
- [ ] **Step 5:** Commit the three files.

### Task 2: Tokenizer equality check

**Files:** Create `gptq_ext/tokenizer_check.py`, `tests/test_gptq_ext_tokenizer_check.py`.

**Interfaces:** Produces `class TokenizerMismatch(Exception)`; `fingerprint(tok) -> dict` with keys `vocab_size`, `special_ids` (bos, eos, pad, unk), `vocab_sha256` (hash of the sorted `(token, id)` pairs), `probe_ids` (encodings of `PROBES`, a tuple of 8 strings covering Latin, Cyrillic, Devanagari, Thai, Khmer, Han, Ethiopic and a digit string); `check_same(tok_a, tok_b) -> dict` (the two fingerprints and `equal: True`, raises `TokenizerMismatch` naming the first differing field); CLI `python -m gptq_ext.tokenizer_check --a google/gemma-2-2b --b google/gemma-2-9b` loading with `transformers.AutoTokenizer` (offline-capable) and exiting 3 on mismatch.

- [ ] **Step 1: Failing tests** with fake tokenizer objects (`get_vocab`, `encode`, `bos_token_id`...): `test_equal_tokenizers_pass`, `test_a_different_vocab_size_raises_naming_vocab_size`, `test_a_swapped_token_id_changes_the_vocab_hash_and_raises`, `test_a_different_probe_encoding_raises_naming_probe_ids`, `test_different_special_ids_raise`.
- [ ] **Step 2-4:** run (fail), implement, run (pass). **Step 5:** commit.

### Task 3: RAM guard and hash recorder

**Files:** Create `gptq_ext/ram_guard.py`, `gptq_ext/record_hashes.py`, `tests/test_gptq_ext_ram_guard.py`, `tests/test_gptq_ext_record_hashes.py`.

**Interfaces:** `ram_guard.estimate_gb(model_gb: float, target_gb: float, n_artifacts: int, margin: float = 1.1) -> float` = `(model_gb + target_gb + n_artifacts * target_gb) * margin` (FP copy, the calibrator's host copy of the originals, the artifacts; 9B: 18.5 and 17.6); `ram_guard.check(ram_gb: float, **same) -> float` raises `RamShortfall` whose message contains the needed and the available numbers; CLI prints the estimate (`--model-gb 18.5 --target-gb 17.6 --n-artifacts N [--ram-gb X]`), exit 4 on shortfall. `record_hashes.collect(gate_out_dir) -> dict` reads `<dir>/<method>/weights_manifest.json` (or the `weights_manifest_<method>.json` layout of `reports/gptq_extension/gateA`) and returns the same schema as `reports/gptq_extension/artifact_hashes.json` (`artifact_sha256: {method: {bits: sha}}`, `calibration: {freeze_file_sha256, pool_sha256}`); `record_hashes.write(gate_out_dir, out_path)` refuses to overwrite an existing file; CLI `--gate-dir --out`.

- [ ] **Step 1: Failing tests:** `test_estimate_for_one_artifact_is_about_58_gb_with_the_9b_numbers` (between 55 and 60), `test_two_artifacts_need_more_than_one`, `test_check_raises_with_both_numbers_in_the_message`, `test_collect_reads_both_methods_and_the_pool_hash` (fixtures copied from `reports/gptq_extension/gateA`), `test_write_refuses_an_existing_output`, `test_collect_fails_when_two_gate_dirs_disagree` (optional `--gate-dir` repeated; same method and bits with different sha raises).
- [ ] **Step 2-4:** run (fail), implement, run (pass). **Step 5:** commit.

### Task 4: The 9B wrapper entry point

**Files:** Create `phase1_gemma9b_ext.py`, `tests_torch/test_phase1_gemma9b_ext.py`.

**Interfaces:** Consumes `phase1_gemma9b_run.gemma9b_setting(n_devices) -> contextmanager`, `phase1_gemma9b_run.main(argv)`, `phase1_run.main(argv)`, `gptq_ext.gate_run.main(argv)`, `gptq_ext.run_with_weights.main(argv)`. Produces `main(argv=None)`: first token is the task `lm | p1 | gate` (`--n-devices` default `GEMMA9B_N_DEVICES` or 1, as in the adapter); `lm` calls `phase1_gemma9b_run.main(["lm", ...rest])` unchanged; `p1` runs `phase1_run.main(rest)` inside `gemma9b_setting`; `gate` runs `gate_run.main(rest)` inside it. Usable as `run_with_weights.py --entry phase1_gemma9b_ext:main --weights gptq ... -- lm|p1 ...`.

- [ ] **Step 1: Failing tests** (fakes via `unittest.mock.patch`): `test_p1_runs_with_the_9b_constants_and_restores_them` (inside the patched `phase1_run.main` the values of `pr.MODEL_CANDIDATES`, `SAE_RELEASE`, `SAE_ID`, `LAYER` are the 9B ones, afterwards the 2B ones), `test_gate_runs_with_the_9b_constants`, `test_lm_delegates_unchanged_arguments`, `test_unknown_task_is_refused`, `test_through_run_with_weights_the_precisions_are_seen_as_bits_hint` (`run_with_weights.bits_hint(["lm","--precisions","6","4"]) == [4, 6]` and for `p1` `--bits 7 6 4 16 12 0` gives `[0, 4, 6, 7, 12, 16]`), `test_the_frozen_files_are_unchanged` (`make_gemma9b_freeze.verify(freeze path) == []`).
- [ ] **Step 2-4:** run (fail), implement, run (pass; also the existing 18 adapter tests). **Step 5:** commit.

### Task 5: The amendment (frozen predictions)

**Files:** Create `make_gemma9b_amendment.py`, `tests/test_make_gemma9b_amendment.py`, then `reports/functional_validation/gemma9b/amendment_A1_three_arm.json` (generated once, before any 9B number).

**Interfaces:** `extract_text(spec_path) -> str` returns, verbatim, the block of the spec from the line starting `- **Q0` through the end of the paragraph `(The Q0 to Q3 list ... stored in the amendment.)`; `build(spec_path, date, base_commit) -> dict` with keys `kind: "gemma9b_amendment"`, `id: "A1"`, `decision`, `created`, `text`, `text_sha256`, `arms` (the states per tier and arm of the Global Constraints), `code_sha256` (the new modules of this plan), `freeze_sha256` (sha256 of the unchanged `freeze.json`), `rules`; `verify(path) -> list[str]` of mismatching items (text hash, code hashes with LF endings as in `make_gemma9b_freeze.sha`); CLI `--base-commit`, `--out`, `--verify`; refuses to overwrite (a later change is a new dated amendment).

- [ ] **Step 1: Failing tests:** `test_text_is_the_verbatim_spec_block`, `test_hash_changes_when_one_threshold_changes`, `test_refuses_to_overwrite`, `test_verify_lists_a_changed_code_file`, `test_the_text_names_every_prediction` (P-ids Q0 Q1 Q2 Q3 S1 S2 S3 S4 and both verdict words).
- [ ] **Step 2-4:** run, implement, run. The amendment file is generated in Task 10 after the code tasks are final (its `code_sha256` covers them). **Step 5:** commit code and tests.

### Task 6: Analysis, tier 1 (Study G across arms)

**Files:** Create `phase1_gemma9b_compare.py` (part A), `tests_torch/test_phase1_gemma9b_compare.py` (class `TierOne`).

**Interfaces:** Consumes `phase1_llama_compare.evaluate(gains, tested, n_boot, n_boot_languages, seed)`, `tested_languages`, `ga.GainRuns`, `ga.build_table`, `ga.h_g5`, `ga.required`, the synthetic-world helpers of `tests_torch/test_phase1_llama_compare.py` (`world`, `manifest`). Produces `evaluate_arms(arms: dict[str, ga.GainRuns], tested: list[str], n_boot=2000, n_boot_languages=5000, seed=0) -> dict` with keys `arms` (per arm: `lc.evaluate` result; for an arm whose states are FP, INT6, INT4 the `P2a` entry is recomputed on INT6 only and carries `"restricted_to": ["INT6"]`), `Q0`, `Q1`, `Q2`, `Q3`, `verdict_tier1`; `q1(net_by_arm, tested) -> dict` (`evaluable`, `ratios`, `count_lower`, `required`, `met`); `verdict_tier1(q) -> "consistent" | "partly consistent" | "different"`. Raises `ValueError` naming the arm and the language when a tested language or the INT4/FP states are missing.

- [ ] **Step 1: Failing tests** on synthetic worlds (27 languages): `test_world_with_graded_flat_then_rising_legacy_and_flatter_gptq_is_consistent` (factors legacy `{FP:1, INT6:1.0, INT4:1.9}`, matched `{.., INT4:1.7}`, gptq `{.., INT4:1.1}`), `test_q1_fails_when_gptq_is_not_lower_than_legacy_by_0_3`, `test_q1_is_not_evaluable_when_legacy_shows_no_amplification_and_the_verdict_is_partly_consistent`, `test_q2_fails_when_an_arm_leaves_the_band_at_int6`, `test_q0_needs_p1_and_h_g1_at_int4_in_both_new_arms`, `test_q3_reports_h_g5b_and_the_count_of_best_factor_above_one_without_changing_the_verdict`, `test_missing_language_raises_with_arm_and_language_name`, `test_per_language_count_for_q1_is_20_of_27`.
- [ ] **Step 2-4:** run (fail), implement, run (pass). **Step 5:** commit.

### Task 7: Analysis, tier 2 and the command line

**Files:** Modify `phase1_gemma9b_compare.py` (part B, `main`); extend `tests_torch/test_phase1_gemma9b_compare.py` (classes `TierTwo`, `Cli`).

**Interfaces:** Consumes `gptq_ext.compare_stage1.load_arm(dir)` (dict with `survived_pct[lang][bits]`, `nll_delta[lang][bits]`, `languages`, `bits`), `compare_stage1.load_roster(path)` (tier per language). Produces `evaluate_stage1(arms: dict[str, dict], roster: dict) -> dict` with `S1`, `S2`, `S3` (per bit width: arm means, order, counts GPTQ vs legacy; `S3`: high minus low tier gap per arm) and `verdict_tier2(s) -> str`; `reference_2b(repo_root) -> dict` from the committed 2B outputs (`reports/functional_validation/gain_study/results/curves.csv`, `reports/gptq_extension/results/gain/{rtnm,gptq}/curves.csv`, `reports/gptq_extension/results/stage1/`) for the report; `main(argv)` with `--legacy-g DIR... --rtnm-g DIR... --gptq-g DIR... --manifest --ratios-json --legacy-p1 DIR --rtnm-p1 DIR --gptq-p1 DIR --roster --repo-root --out`, writing `analysis.json` and `report.md` (both verdicts, every prediction with rule, 9B value and the 2B reference, the limits sentence of spec section 5, the hardware sentence of spec section 4.3). A tier can be omitted (`--legacy-p1` absent -> tier 2 is "not run", never a verdict).

- [ ] **Step 1: Failing tests:** `test_s1_s2_orders_and_counts_on_planted_survival` (planted `survival.pt` files written with `torch.save` in the layout `load_arm` reads, built like `tests_torch/test_gptq_ext_compare_stage1.py` does), `test_s3_gap_smaller_under_gptq`, `test_verdict_tier2_words`, `test_the_2b_control_gives_the_known_numbers` (the committed 2B outputs through `evaluate_arms` curves and `evaluate_stage1`: legacy/matched/GPTQ median ratio 1.9 / 1.7 / 1.1 within 0.05, INT6 survival 85.0 / 89.7 / 97.5 within 0.1 points), `test_cli_writes_both_files_and_both_verdicts`, `test_cli_with_tier_two_absent_says_not_run`.
- [ ] **Step 2-4:** run (fail), implement, run (pass). **Step 5:** run `scripts/g0/mutcheck.py` on `phase1_gemma9b_compare.py`, `ram_guard.py`, `tokenizer_check.py` and record kills/equivalents; commit.

### Task 8: The driver (`jarvis_run_all.sh`)

**Files:** Modify `scripts/g0/jarvis_run_all.sh`, `scripts/g0/tests/test_jarvis_run_all.py`.

**Interfaces (environment variables, new; the old ones keep their meaning and defaults):** `ARMS="legacy rtn-matched gptq"`, `STAGES="g p1"`, `STATES` (legacy tier 1, default `FP 8 7 6 5 4`), `STATES_NEW="6 4"`, `P1_BITS_LEGACY="8 7 6 5 4 16 12 0"`, `P1_BITS_NEW="7 6 4 16 12 0"`, `PROCS_PER_GPU=1`, `FAKE_CC`, `FAKE_DRIVER` (tests), `POOL_DIR` (default `$BASE/gptq/calibration`). New steps in order: `env` (index chosen from compute capability: `cu130` when >= 12.0 and driver >= 580, `cu126` below 12.0, error otherwise; an explicit `TORCH_INDEX` wins; a tiny bf16 matmul on the GPU proves the build runs on this card), `flores`, `access_and_download`, `checks` (extended: Tasks 1-7 test modules, `make_gemma9b_freeze --verify`, `make_gemma9b_amendment --verify`), `tokenizer` (`tokenizer_check`), `pool` (pool files present and verified against `calibration_freeze.json` through `gptq_ext.calibration.load_pool` / `pool_sha256`), `smoke` (the old smoke), `smoke_p1` (`phase1_gemma9b_ext.py p1 --corpus flores --regime mask --languages eng_Latn hin_Deva khm_Khmr --max-sentences 40 --bits 4`, legacy rounding: proves the tier-2 path before hours are spent), `gate9b` (`phase1_gemma9b_ext.py gate --methods gptq rtn-matched --bits 6 4 --record-weights-hashes`, then `record_hashes` writes `reports/functional_validation/gemma9b/artifact_hashes.json`), `smoke_gptq` (`run_with_weights.py --entry phase1_gemma9b_ext:main --weights gptq --gptq-freeze <9B hashes> -- lm` on the 3 smoke languages, `--precisions 4`: proves the whole GPTQ path, one extra build), `tag` (tagging and manifest, once), then per stage and arm the units. Units: tier 1 legacy `phase1_gemma9b_run.py lm --precisions <S>` per state exactly as before (markers `state_<S>.done`, outputs `g9-lm-<S>`); tier 1 new arms `run_with_weights.py --entry phase1_gemma9b_ext:main --weights <arm> --gptq-freeze <9B hashes> ... -- lm --precisions <S>` per state (markers `unit_g_<arm>_<S>.done`, outputs `g9-lm-<arm>-<S>`); tier 2 `... -- p1 --corpus flores --regime mask --bits <list> --out ... --flores-root ... --commit ...` one process per arm (markers `unit_p1_<arm>.done`, outputs `g9-p1-<arm>`). Order = spec section 9. A unit is validated by `check_lm_log` (tier 1) or the presence of `eq/survival.pt` and `all/survival.pt` plus no CUDA warning (tier 2). Each unit first runs `ram_guard` with its artifact count. Slots = `min(GPUs * PROCS_PER_GPU, RAM / per-unit estimate)`; `DRY_RUN=1` prints every unit with its command and slot plan.

- [ ] **Step 1: Failing tests** (dry run, extend the existing class): `test_dry_run_lists_the_new_steps_in_order` (env, flores, access_and_download, checks, tokenizer, pool, smoke, smoke_p1, gate9b, smoke_gptq, tag, then the units), `test_blackwell_with_a_current_driver_chooses_cu130` (`FAKE_CC=12.0 FAKE_DRIVER=580.65` -> `cu130` in the plan), `test_blackwell_with_an_old_driver_is_refused_with_a_message` (`FAKE_DRIVER=550.0` -> non-zero, stderr names 580), `test_ampere_chooses_cu126` (`FAKE_CC=8.0`), `test_one_gpu_160_gb_gives_one_slot_and_two_with_procs_per_gpu_2_only_when_two_units_fit`, `test_every_arm_and_stage_unit_appears_once_with_the_right_weights_flag` (legacy has no `--weights`, others have), `test_gptq_units_refuse_without_the_recorded_hashes` (a unit command line contains `--gptq-freeze` pointing at the 9B file, and the plan has `gate9b` before every gptq/rtn-matched unit), `test_resume_skips_finished_units` (create marker files in `$BASE/state`, expect `[skip]` lines), `test_old_tests_still_pass` (the five existing tests unchanged).
- [ ] **Step 2: Run** `python3 -B -m unittest tests.test_jarvis_run_all -v` from `scripts/g0`; expect the new tests to fail. **Step 3:** implement the extension keeping the old behaviour when `ARMS=legacy STAGES=g` (the old command line gives the old plan). **Step 4:** run; expect all pass; also `bash -n scripts/g0/jarvis_run_all.sh`. **Step 5:** commit.

### Task 9: The Jarvis bundle

**Files:** Create `scripts/g0/make_jarvis_bundle.sh`.

**Interfaces:** `bash scripts/g0/make_jarvis_bundle.sh <outdir> [--pool-from <dir>|--pool-from-ada]` writes `<outdir>/always-late-gemma9b.tar.gz` (a `git archive` of HEAD under `always-late/` plus a `COMMIT` file, so no push is needed), `<outdir>/gptq_calibration_pool.tar.gz` (the pool token files, from a local directory or, with `--pool-from-ada`, from `/home2/karma/duan-verify/gptq/calibration` over `ssh -o HostKeyAlias=ada-gw1 karma@10.4.24.30` after the user's OK) and `<outdir>/SHA256SUMS`; it refuses a dirty tree. Files go to Jarvis with `scp` and are unpacked by one printed command block (the script prints it).

- [ ] **Step 1:** Write `tests/test_make_jarvis_bundle.py` (temporary git repo in a temp dir: bundle contains COMMIT equal to HEAD, refuses a dirty tree, lists SHA256SUMS). **Step 2-4:** run (fail), implement, run (pass). **Step 5:** commit. (The real pool fetch from Ada happens at hand-over, with the user's OK; if Ada is unreachable the pool can be rebuilt from `gptq_ext/calibration.py` on the login node, command in `reports/gptq_extension/stage_commands.md`.)

### Task 10: Freeze, documents, hand-over

**Files:** Create `reports/functional_validation/gemma9b/amendment_A1_three_arm.json`, `reports/handoffs/<next free number>-2026-10-28-gemma9b-three-arm-ready.md`; Modify `CLAUDE.md`, `reports/functional_validation/gemma9b/JARVIS_HANDOFF.md`, `reports/decisions_and_anomalies.md` (new D-entry, id = next free, in the table's format), `DECISIONS_FOR_REVIEW.md` (one line: the Q-items answered or opened).

- [ ] **Step 1:** Run all stdlib and torch/TL tests of Tasks 1-9 plus the full existing suites (`python3 -B -m unittest discover -s scripts/g0/tests`, `OMP_NUM_THREADS=4 ~/.venvs/anlp-torch/bin/python -B -m unittest discover -s scripts/g0/tests_torch`, TL tests in `~/.venvs/anlp-tl`); expect green (report any pre-existing failure separately).
- [ ] **Step 2:** `python scripts/g0/make_gemma9b_amendment.py --base-commit $(git rev-parse HEAD) --out reports/functional_validation/gemma9b/amendment_A1_three_arm.json`, then `--verify`; commit it alone (the freeze commit; its date is the proof that it precedes every 9B number).
- [ ] **Step 3:** Rewrite `CLAUDE.md` for the new command (one command, arms, stages, hardware facts, what is verified and what is NOT: real 9B on Blackwell, GPTQ build time, cu130 wheel, memory estimate, the smoke and gate steps as the first real execution, drop order, hand-back files).
- [ ] **Step 4:** D-entry and handoff (what was built, tests and mutation numbers, what is unverified, the exact commands to start on Jarvis, the unpack commands of the bundle, the estimate of GPU hours to be replaced by the smoke numbers).
- [ ] **Step 5:** `DRY_RUN=1 FAKE_GPUS=98304 FAKE_RAM_GB=160 FAKE_CC=12.0 FAKE_DRIVER=580.65 bash scripts/g0/jarvis_run_all.sh` prints the full plan; paste its head into the handoff. Commit. Tell the user the branch is ready and what to resume.

## Updates

- **2026-10-28, after the whole-branch review (ledger: `Final:` lines).** Changes to the plan as written: (1) Task 8: the gate step builds every artifact bit width the units use, in gate jobs of at most two bit widths (INT6 and INT4, then INT7 for tier 2) and `record_hashes` takes all gate directories; the smoke tags all 27 languages with 100 sentences and only its `lm` run is restricted to 3 languages; `step()` no longer runs functions on the left of `||` (errexit); the frozen-hash verification (`make_gemma9b_freeze.py --verify`, `make_gemma9b_amendment.py --verify`) runs first on every invocation; `STOP_AFTER=<step>`; `P1_EXTRA`; `nvidia-smi` helpers cannot abort the script. (2) Task 5: the amendment text starts at the line `P1 to P3b (legacy arm) ...` (the definitions), not at `- **Q0`. (3) Task 7: tier 2 must cover the whole roster; the report says "not evaluable" for Q1 with the legacy median and count. (4) Task 10: the amendment is generated AFTER the review, so that its code hashes are final; handoff numbering `9b-NNN`; no D-entry on this branch.
