# Handoff 034: STUDY G (graded gain of language-selective features, the user's major study) is designed, its baseline measured, its code built and tested; the FP calibration job is running (2026-10-24, 08:00 IST)

> The ONLY source of the previous chat. A fresh chat with no memory must be able to continue from this file plus the repository. If this file and the repository disagree, the repository wins: say so in the next handoff. **Start with [`reports/functional_validation/gain_study/README.md`](../functional_validation/gain_study/README.md) (the user's request verbatim and the status table of Study G) and the plan [`docs/superpowers/plans/2026-10-24-graded-gain-study.md`](../../docs/superpowers/plans/2026-10-24-graded-gain-study.md); then `reports/functional_validation/STATUS.md` (Immediate, Log).**

## 0. Read this first

The standing rules are those of [handoff 033](033-2026-10-23-f5-ladder-control-complete.md) section 0 (which carries those of 026 to 032): mask regime only; no `none` work; no comparison with Duan; never edit `paper/` or `third_party/`; no secrets, tokens, weights or large artifacts; explicit `git add` paths only; commit messages end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`; at most two Ada jobs; never wait or poll; HF dataset private, no upload without the user's go-ahead (Q21); the Llama repo is gated (Q22); tests first and watch them fail, mutation check with `scripts/g0/mutcheck.py`; render and look at every figure; decisions only appended; **never delete or shorten progress-tracking documents** (map in `reports/INDEX.md`). **`scripts/g0/phase1_func_run.py` and the other frozen files are never edited** (`test_phase1_func_freeze_mc` hashes them): Study G wraps the runner (`phase1_gain_run.py`).

**The user's instruction of 2026-10-24 (verbatim in the Study G README): do Study G first; record methodology and results very clearly; analyse when the runs are done; keep the documentation current; work on future plans while it runs on Ada; and "maintain a written record that you need to come back to this task at the start of every run".** That record is in the cron prompt (priority 0), STATUS `Immediate`, the README of the study, `reports/INDEX.md` and the memory file `anlp-gain-study.md`.

## 1. Mission in five lines

Per language (27) and per precision (FP, INT8, 7, 6, 5, 4), scale that language's selective SAE features (frozen K64 group, full-precision, dev-selected) down by three ratios and up by three ratios, and measure the next-token loss on that language's own held-out FLORES devtest text; matched controls scaled by the same ratios carry the specificity claim; judge monotonicity in the ratio, across precisions and across languages with rules fixed in advance. The design (hypotheses H-G1 to H-G7, amendment A1) is `docs/superpowers/specs/2026-10-24-graded-gain-study-design.md`; D-070 records the choices. Exploratory (devtest already examined).

## 2. State right now (verified 07:55)

- Git `main` pushed (HEAD after this handoff's commit); tests: torch 1410, stdlib 298, all green; the link checker ends " 0 broken".
- **Ada: `func-mc-pc-256` 36760 (the third positive-control dose, D-069; started 06:35, about 2 h) and `gain-cal-FP` 36834 (the FP ratio calibration, 18 ratios, 100 pilot sentences, 2 control draws; submitted 07:26, about 1.5 h; output `/home2/karma/duan-verify/out/phase1/gain-cal-FP/`) are RUNNING.** The smoke job `gain-smoke` 36827 COMPLETED (mechanics only). Snapshot `repo_7fba511` (frozen-file hashes equal to the freeze).
- Cron entries `955a5cc5`, `e33ac7eb`, `ea207eac` (session-only, expire about 2026-10-29); the cron prompt has Study G as priority 0.
- Study G status (README table): G0 design DONE, G1 baseline DONE, G2 wrapper DONE, G3 calibration RUNNING (FP job; INT4 job to submit), G4 main runs not started, G5 analysis code DONE (tests), G6 figures code DONE (tests), real-data analysis and figures wait for the runs.

## 3. What was done (2026-10-24, this chat)

1. Standing record, design (spec), plan, D-070 (commit `dd784e5`).
2. **G1 baseline** (`phase1_gain_baseline.py`, 36 tests, 25 mutants killed; results in `gain_study/baseline/`): at INT4 the layer-12 SAE activations shrink by about 15 % for every class of feature (group 0.852, whole language 0.843, matched-control pool 0.853 of FP; group over whole-language ratio 1.004), so the shrinkage is generic; H-G7 not met by its rule (mass share does not fall; retained peak falls in 26 of 27 languages but its median wiggles by one member). Amendment A1: two fine up ratios 1.1 and 1.2 are added to the runs; H-G5 split into H-G5a (generic repair) and H-G5b (group-specific).
3. **G2 wrapper** (`phase1_gain_run.py`, 25 tests, 27 mutants killed; `run_gain.sbatch`): registers the preset `gain|<ratios>|c<n>|K64` in the frozen runner's `PRESETS`, wraps its module-level `build_configs` to add scaled matched controls; the preset name carries the exact ratios. Smoke job: gates passed, 17 configurations per language for 3 ratios, about 1,500 token passes per second.
4. **G3 rule module** (`phase1_gain_ratios.py`, 22 tests, 24 mutants killed): grid, the pre-registered rule (down ratios at 0.10, 0.30, 0.60 of the FP zero-ablation net harm, up ratios their reciprocals, flat fallback), the calibration table reader. Measured cost: 3.0 GPU hours per state for the main job (38 configurations per language), 1.5 hours per state for the calibration (plan amendment A2).
5. **G5 analysis module** (`phase1_gain_analysis.py`, 84 tests, 52 mutants killed): loader (refuses incomplete jobs, skipped states, failed gates, preset or ratio mismatches), table with paired bootstrap, rules H-G1 to H-G6 with the counts of the design (24, 20, 14 of 27), alpha*, report, writers. **G6 figures module** (`phase1_gain_figures.py`, 23 tests, 18 mutants killed): language panels, macro curves, heatmaps, repair figure; rendered on synthetic data in both themes and looked at (one overlap found and fixed); real-data renders wait for the runs.

## 4. The next step, exactly

The plan's "Operational commands" appendix has every command. In order:
1. When `gain-cal-FP` 36834 ends: validate it (sacct, `meta_lm_pilot.json` complete, gates, preset equal to the 18 ratios); fetch its two files; **submit `gain-cal-INT4`** (command in the plan; at most two jobs: submit when `func-mc-pc-256` has ended or a slot is free).
2. When 36760 ends: validate it like the other two doses (STATUS Immediate (1), D-069 extension, `reproduce_functional.sh`, checksums).
3. When both calibration jobs are done: apply the rule (`phase1_gain_ratios.py`), append **D-071** (ratios chosen, table, branch) BEFORE submitting the main jobs; then submit the six main jobs (`gain-FP`, `gain-8`, `gain-7`, `gain-6`, `gain-5`, `gain-4`), two at a time.
4. When the main jobs finish: validate, fetch, add checksums, run the analysis and the figures (commands in the plan), look at every figure, read the verdicts against the rules (no rule is changed after the numbers), write the results into the README of the study, `RESULTS_SUMMARY.md`, a D-entry, `reproduce_functional.sh`/`REPRODUCE.md`, handoff 035.
5. While jobs run: future plans (extensions a to d of plan Task 7: size `all`, state-defined groups on dev text, Belebele at the study ratios, second control seed); the literature re-check around 2026-10-30; the user's open questions Q21 to Q26.

## 5. Open questions for the user

Q21 (HF upload), Q22 (Llama licence), Q23, Q24, Q25, Q26 (Qwen as a second model): unanswered, defaults in force (see `DECISIONS_FOR_REVIEW.md`). New for Study G, decided by default and recorded in D-070 (each can be overruled): the performance measure is the next-token loss, the group is the frozen FP-defined K64, three control draws, the ratio rule.

## 6. Environment and pitfalls

- ssh by name fails (WARP DNS): `ssh -o HostKeyAlias=ada-gw1 karma@10.4.24.30`; `scp -o HostKeyAlias=ada-gw1 karma@10.4.24.30:<path> <dest>` one file per call; deploy only named snapshots: `ADA_SSH='ssh -o HostKeyAlias=ada-gw1 karma@10.4.24.30' bash scripts/g0/deploy_phase1.sh repo_<sha>`.
- Python one-liners that import repository modules: use `python -c` with `sys.path.insert(0, 'scripts/g0')`, not `python -I` (isolated mode ignores the working directory). Tests: `OMP_NUM_THREADS=4 ~/.venvs/anlp-torch/bin/python -B -m unittest scripts.g0.tests_torch.<module>`; mutation check: `python3 -B scripts/g0/mutcheck.py FILE MUTANTS.json -- env OMP_NUM_THREADS=4 $HOME/.venvs/anlp-torch/bin/python -B -m unittest ...` (mutant files in the session scratchpad; describe the mutants in the commit).
- The controls draw from the first three or two `K64` draws of the manifest (`n_controls`); the preset name, not the argv, is the record of the ratios.
- Quote bash heredocs; take timestamps from `date`; no foreground `sleep`.

## 7. What I would check if something looks wrong

`git status`, the link checker, `CronList` (three entries), `squeue -u karma`, the gate line and the configuration count in each gain log (`configs` in the per-state summary: 38 x 27 = 1,026 for a main state; the 18-ratio calibration has 59 per language), `sha256sum` of the frozen runner files on the snapshot against `evaluation_freeze_lm.json`.
