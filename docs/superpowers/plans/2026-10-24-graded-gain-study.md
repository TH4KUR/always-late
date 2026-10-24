# Study G (graded gain of language-selective features) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **At the start of every run read [`reports/functional_validation/gain_study/README.md`](../../../reports/functional_validation/gain_study/README.md) (the user's request verbatim and the status table) and continue with the first unticked step here.**

**Goal:** Measure, per language and per precision state, how the next-token loss on the language's own text depends on a gain applied to the language's selective SAE features (three ratios down, three up), with matched controls, and judge monotonicity in the ratio, monotonicity across states and consistency across languages.

**Architecture:** The frozen runner (`phase1_func_run.py`, hash-checked) is wrapped by a new module that registers a preset and adds scaled control configurations at run time; results are read by a new analysis module that follows the F1 analysis conventions (paired bootstraps over sentences); the baseline of selectivity per state is a CPU analysis of the D-035 profiles.

**Tech Stack:** Python, torch (Ada env `/home2/karma/duan-verify/env/bin/python`; local `~/.venvs/anlp-torch`), unittest, `scripts/g0/mutcheck.py`, Slurm on Ada (RTX 2080 Ti), matplotlib for figures.

**Spec:** [`../specs/2026-10-24-graded-gain-study-design.md`](../specs/2026-10-24-graded-gain-study-design.md) (hypotheses H-G1 to H-G7 and their decision rules; the ratio rule in section 3).

## Global Constraints

- Mask regime only; no `none`-regime work; no numerical comparison with Duan (D-028).
- `scripts/g0/phase1_func_run.py` and the other frozen files are never edited (`test_phase1_func_freeze_mc` hashes them); new behaviour goes into new modules.
- Groups: frozen `K64` of `reports/functional_validation/groups_manifest.json`; controls: its first three `K64` draws; text: `lm_ids(1012, 150, 300, 0)` (`eval`, 300 sentences) for the main runs, `pilot` (150) for calibration only.
- States: FP, INT8, INT7, INT6, INT5, INT4. At most two Ada jobs at a time; record Slurm ids; never wait or poll; no secrets, tokens, weights or large artifacts in the repo; HF dataset stays private, no HF upload without the user's go-ahead.
- Tests first (watch fail), mutation check each new module, evidence before claims; render and look at every figure; decisions only appended (D-070 onward); documents of the trail are never deleted or shortened.
- Ratios are fixed by the rule of spec section 3 and written to the decision log before any main job is submitted.

## Review Focus

- A group or control index that is out of range or repeated in one configuration: the wrapper must refuse it (test).
- Two configurations with the same name (the frozen validator raises): names carry family, ratio, draw and language (test).
- A ratio that formats ambiguously (`0.35355339` vs `0.354`): names use the exact value through `repr`-stable formatting and the analysis reads ratios from the run's metadata, not from the names (test).
- A state whose job was cut by the time limit: the analysis must refuse a run that lacks a state or a configuration instead of averaging what exists (test).
- The control scaled by α = 1 must equal the unedited loss exactly (identity gate) and the group scaled by 1 likewise (test on the toy engine).

---

### Task 0: Record, design, plan (G0)

**Files:** Create `reports/functional_validation/gain_study/README.md`, `docs/superpowers/specs/2026-10-24-graded-gain-study-design.md`, this plan; modify `cron_prompt.md` (priority 0), `STATUS.md`, `reports/INDEX.md`, memory.

- [x] **Step 1:** Standing record written (README with the request verbatim and the status table; cron priority 0; STATUS `Immediate`; INDEX rows; memory note) - 2026-10-24.
- [x] **Step 2:** Design with pre-registered hypotheses and rules written - 2026-10-24.
- [x] **Step 3:** This plan written.
- [ ] **Step 4:** Append D-070 (the design, its choices and what it does not decide) to `reports/decisions_and_anomalies.md`; handoff 034 started; commit and push.

### Task 1: Baseline of selectivity per state (G1, CPU)

**Files:** Create `scripts/g0/phase1_gain_baseline.py`, `scripts/g0/tests_torch/test_phase1_gain_baseline.py`; outputs `reports/functional_validation/gain_study/baseline/{baseline.csv,baseline.json,report.md}`.

**Interfaces:**
- Consumes: the D-035 `profiles.pt` (keys `languages`, `feature_ids`, `conditions{state: {n_tokens, positive_counts, activation_sums, activation_max}}`; fetched from Ada `/home2/karma/duan-verify/out/phase1/association-full/association/eq/profiles.pt`, sha256 recorded in `artifact_checksums.md`); the frozen groups manifest; the cohort's FP null cutoff and `selective`, `controls` flags (`selectivity.pt` of the cohort, sha256 `2f0422c2…`; copy from Ada if the cutoff is stored there, otherwise recompute from the FP profile and say so).
- Produces: `class_masks(rates, counts, n_tokens, cutoff, min_positives=50, min_panel_rate=1e-5) -> dict[str, bool array [L, F]]` (per language: `own` selective for l at this state, `other` selective for another language, `general` supported and not selective, `unsupported`); `state_row(profile, state_groups, language_index) -> dict` (counts, overlap with `G_l`, mass shares, `group_mean_activation`, `g_ratio` against FP); `baseline_table(profiles, manifest, cutoff) -> list[dict]` with one row per (language, state).

- [x] **Step 1: Write failing tests** on toy profiles (3 languages, 6 features) for: every supported feature falls in exactly one class per language; shares of activation mass sum to 1; `g_ratio` of FP against itself is 1; a never-firing feature is `unsupported`; a feature tied at the peak is not `own` for any language; the toy FP-to-INT shift that halves a group's activations gives `g_ratio` 0.5.
- [x] **Step 2:** Run: `OMP_NUM_THREADS=4 ~/.venvs/anlp-torch/bin/python -B -m unittest scripts.g0.tests_torch.test_phase1_gain_baseline`; expect failure (module missing).
- [x] **Step 3:** Implement the module as specified; mean activation = `activation_sums / n_tokens` (zeros included); rates = `positive_counts / n_tokens`; supported by the Phase 1 rule (`phase1_lib.supported_mask` semantics); `S = max_l r / mean_l r` (`phase1_lib.selectivity_S`).
- [x] **Step 4:** Tests pass; mutation check (`scripts/g0/mutcheck.py`) with at least 15 mutants; every survivor killed by an added test or explained.
- [x] **Step 5:** Fetch `profiles.pt` with one `scp` (sha256 against Ada), run the module, read the report, write the table into the README of the study (counts per language and state; `g_ratio`; mass shares; H-G7 verdict) and the baseline prediction for H-G5 (`1/g_ratio` per language at INT4) **before any GPU result exists**.
- [x] **Step 6:** Commit and push (explicit paths).

### Task 2: Wrapper runner and its job script (G2)

**Files:** Create `scripts/g0/phase1_gain_run.py`, `scripts/g0/tests_torch/test_phase1_gain_run.py`, `scripts/g0/run_gain.sbatch`.

**Interfaces:**
- Consumes: `phase1_func_run.build_configs(man, run_langs, targets, panel, families, off_target_label, ...)`, `PRESETS`, `main(argv)`, `evaluate_precision`; the manifest.
- Produces: `gain_configs(man, run_langs, targets, ratios, label="K64", n_controls=3) -> list[config]` (names `scale{α}_K64@{code}` for the group, `cscale{α}.{r}_K64@{code}` for control draw r; kind `scale`, `kw {"alpha"}`, `langs {index of target}`); `register(ratios, controls=3, zero=True, sham=True)` (adds preset `gain` to the frozen `PRESETS` and replaces `build_configs` in the frozen module by a wrapper that builds the frozen families and appends `gain_configs`, rejecting duplicate names); `main(argv)` parsing `--ratios` (floats), `--n-controls`, then handing the rest to the frozen `main` with `--preset gain`; the run's metadata records ratios.

- [x] **Step 1: Write failing tests** on the toy engine: (a) the preset `gain` appears in `PRESETS` only after `register`, and `phase1_func_run.py` bytes are unchanged; (b) for 2 languages, 3 ratios and 3 controls the config count and names are exactly `1+3+3` per language from the frozen families plus `3 + 9` scaled ones; (c) group `scale{1}` and control `cscale{1}.0` reproduce the unedited NLL exactly through `evaluate_precision`; (d) a control scaled by α changes the NLL only through its own indices (toy model with a known decoder); (e) duplicate or out-of-range ratios (≤ 0, NaN, duplicates) are rejected; (f) a config set never targets the panel (no off-target runs).
- [x] **Step 2:** Run and see them fail.
- [x] **Step 3:** Implement; the wrapper changes behaviour only through the module-level names `PRESETS` and `build_configs` of the frozen module.
- [x] **Step 4:** Tests pass; mutation check; `run_gain.sbatch` copied from `run_functional.sbatch` with the entry point replaced and a header comment (a new file, the old one untouched); a `bash -n` syntax check.
- [x] **Step 5:** Deploy a named snapshot (`ADA_SSH='ssh -o HostKeyAlias=ada-gw1 karma@10.4.24.30' bash scripts/g0/deploy_phase1.sh repo_<sha>`), run a smoke job (FP and INT4, three languages, 40 sentences, calibration grid) and read its log (gates passed, config count, speed).
- [x] **Step 6:** Commit and push.

### Task 3: Ratio calibration on the pilot split (G3)

**Files:** Create `scripts/g0/phase1_gain_ratios.py`, `scripts/g0/tests_torch/test_phase1_gain_ratios.py`; output `reports/functional_validation/gain_study/calibration/{calibration.csv,calibration.md,ratios.json}`.

**Interfaces:**
- Produces: `grid() -> (down list, up list)` (the 8 + 8 values of spec section 3); `select_ratios(net_harm_by_ratio: dict[float, float], z: float) -> dict` implementing the rule of spec section 3 (down ratios by the targets 0.10/0.30/0.60 of z, ties to the next nearest, flat-response fallback 0.25/0.125/0.0625, up ratios the reciprocals) and returning the three down and three up ratios with the table.

- [x] **Step 1: Write failing tests** for the rule: a harness curve whose net harm is exactly `φ z` at three grid values returns them; two targets with the same nearest value return distinct ratios; a flat curve returns the fallback; the up ratios are the reciprocals and lie on the grid; identical results for a permuted input.
- [x] **Step 2-4:** Run to fail, implement, pass; mutation check.
- [~] **Step 5 (FP job 36834 submitted 2026-10-24 07:26; INT4 when a slot is free):** Submit the two calibration jobs on Ada (FP and INT4, pilot split, 150 sentences, ratio grid, two controls, zero-ablation; two slots at most), record Slurm ids in STATUS.
- [ ] **Step 6:** When they finish: validate (gates, completeness, item keys), analyse with the calibration module, **apply the rule exactly as written**, append the chosen ratios, the table and the date to the decision log (D-071), update the README status.
- [ ] **Step 7:** Commit and push.

### Task 4: Main runs (G4)

- [ ] **Step 1:** Submit job `gain-a` (FP, INT8, INT7) and `gain-b` (INT6, INT5, INT4): `--split eval --n-eval 300 --panel --ratios <the six>`, `-t 06:00:00`; record ids; snapshot hash check against the freeze.
- [ ] **Step 2:** On completion (a later firing): sacct, checksums, `meta_lm_eval.json` (complete, nothing skipped for time, gates), config counts, item keys against the frozen eval ids; fetch files one by one; record in `artifact_checksums.md`.
- [ ] **Step 3:** Commit and push.

### Task 5: Analysis (G5)

**Files:** Create `scripts/g0/phase1_gain_analysis.py`, `scripts/g0/tests_torch/test_phase1_gain_analysis.py`; outputs `reports/functional_validation/gain_study/results/{analysis.json,curves.csv,alpha_star.csv,monotonicity.csv,report.md}`.

**Interfaces:**
- Consumes: the run directories of Task 4 (files `func_lm_eval_<state>.pt`, `meta_lm_eval.json`) and the ratios.
- Produces: `load_gain_runs(dirs) -> dict`, `curves(...) -> table of h, h_ctrl, h_net with bootstrap intervals`, `monotone_counts(...)` implementing H-G1 to H-G4, H-G6 literally as written in the spec, `alpha_star(...)`, `baseline_link(...)` for H-G5/H-G7, `report_text(...)` with the verdict line of every hypothesis (met / not met / reversed) and the limits.

- [x] **Step 1: Write failing tests** on planted data: a language whose harm is strictly ordered is counted, a tie is not, a missing state or configuration raises, a planted `α*` is recovered, the bootstrap is seeded and paired (a constant offset in the sentences does not change the interval of a difference), the verdict text names a hypothesis as reversed when the order is reversed.
- [x] **Step 2-4:** Fail, implement, pass; mutation check; reuse the bootstrap helper of `phase1_func_analysis.py` rather than rewriting it.
- [ ] **Step 5:** Run on the real runs, read the report against the pre-registered rules, write the verdicts and every deviation (none are allowed to change a rule) to the README of the study and a D-entry.
- [ ] **Step 6:** Commit and push.

### Task 6: Figures and the written results (G6)

**Files:** Create `scripts/g0/phase1_gain_figures.py` with tests; outputs `reports/functional_validation/gain_study/figures/`; modify the README of the study, `RESULTS_SUMMARY.md` (new claim row), `reproduce_functional.sh` and `REPRODUCE.md`, `artifact_checksums.md`, handoff 034.

- [ ] **Step 1:** Figures of spec section 7 (consult the dataviz skill; tests for data layer and clipping first); render, look in both themes, fix defects.
- [ ] **Step 2:** Write the results section of the README of the study (the one page that answers Q1 to Q6 with the numbers and the limits), the claim map row, the reproduction commands.
- [ ] **Step 3:** Commit and push.

### Task 7: Extensions (only after Task 6; each needs its own appended decision)

- [ ] (a) Size `all` at FP, INT6, INT4; (b) state-defined groups on dev text; (c) Belebele accuracy at the study ratios for the 18 covered languages; (d) second control seed if specificity is marginal.

## Self-review

- Spec coverage: Q1 to Q3 and Q5 by Tasks 4 and 5; Q4 by the scaled controls (Tasks 2, 5); Q6 and H-G7 by Task 1; ratios by Task 3; figures and write-up by Task 6; extensions by Task 7.
- Type consistency: `gain_configs` names (`scale{α}_K64@code`, `cscale{α}.{r}_K64@code`) are what Task 5 parses together with the ratios stored in the run's metadata.
- Proportion: the plan fixes names, signatures, tests and values; bodies are left to the implementer.

## Amendment A1 (2026-10-24 07:12; after Task 1, before any intervention number)

Task 1 is done (29 + 7 tests, 25 mutants all killed; baseline in `reports/functional_validation/gain_study/baseline/`). Its result changes the runs, as written in section 9 of the spec: **two fine up ratios (1.1, 1.2) are added** to the calibration jobs (Task 3: 18 ratios) and to the main jobs (Task 4: 38 configurations per language and state); `gain_configs` takes `ratios` as one list and the analysis (Task 5) separates the six study ratios (three down, three up) from the two fine ones; H-G5 is split into H-G5a and H-G5b (spec section 9); the verdict of H-G7 (not met) is recorded in the README of the study.

## Amendment A2 (2026-10-24 07:26; measured cost, before any intervention number)

The smoke job measured 1,500 to 1,530 token passes per second. The main job has 38 configurations per language (39 passes per token) over 442,613 evaluation tokens: about **3.0 GPU hours per state** (not the 1.5 to 2 estimated in the design): **one job per state** (six jobs, `-t 04:00:00`, two at a time), about 9 hours of wall time. The calibration uses 100 of the 150 pilot sentences (`--max-sentences 100`) and two control draws (59 passes per token, about 1.5 hours per state). Neither choice changes a rule or a hypothesis.

