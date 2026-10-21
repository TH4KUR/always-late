# Handoff 017: D-035 Tasks 1, 3 and 2 done, smoke queued on Ada (2026-10-21, about 06:40 IST)

> Claude's series. Read this and `reports/association_retention/STATUS.md` (its NEXT STEP block and Log are the live state). If this file and the repository disagree, the repository wins.

## 0. Standing rules (the user's words and decisions)

- Project: ANLP "Does Compression Forget Language?" in `/home/karma/Documents/ANLP/project/always-late`. User instructions of 2026-10-21: verify the plan before implementing; a recurring 20-minute job continues the work; "you are allowed to commit and push codex's work"; "also you can stop trying to communicate with codex now. delete that communication folder ... start implementing the next plan now ... ground with our already done, or new searches for sources and research"; "i will make HF public later, dont worry about it as long as you can access it".
- Done on those: the communication folder is removed (D-037; commit `7a27ae4`; recoverable from `58c3fff`; carried-over points are in the appendix of `reports/association_retention/plan_verification.md`); Codex's work is committed and aligned; the HF dataset stays private until the user publishes it (access verified from Ada, 277 files); method sources M01-M05 for the statistics are in the register.
- Rules unchanged: mask regime only; no `none` work, no numerical comparison with Duan (D-028); never edit `paper/` or `third_party/`; secret scan before each commit; stage explicit paths; commit messages end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`; at most 2 Ada jobs, record the Slurm id and do not wait or poll; do not delete anything on Hugging Face or wandb; do not change the dataset's visibility. Recurring job `5e64064d` (`7,27,47 * * * *`, session-only, expires after 7 days; prompt in `reports/association_retention/cron_prompt.md`) does one step per firing from the status file.

## 1. Mission

D-035: do language-selective SAE features selected before quantization (original `mask/eq` dev cohort: 8,086 selective features, 2,860 controls) keep their language association (full 27-language firing/mean profile, peak language, target share, margin) after RTN INT8/7/6/5/4 on matched devtest inputs, with FP, INT0 and INT16/12 as controls? Plan: `docs/superpowers/plans/2026-10-21-language-association-retention.md`; design: `reports/survival_method_reassessment.md` section 7; verification: `reports/association_retention/plan_verification.md`. A measurement-validation study, not a novelty claim (D-029); prior art found no equivalent (bounded searches).

## 2. State (verified)

- Git: `main` at `e31bc37` (pushed); commits of this stretch: `570d3f4` Task 1, `8536f90` Task 3, `1655450` Task 2, `e31bc37` smoke status. Tree clean after the push.
- Tests: stdlib 180 and torch 342 pass (`python3 -B -m unittest discover -s scripts/g0/tests`; `OMP_NUM_THREADS=4 ~/.venvs/anlp-torch/bin/python -B -m unittest discover -s scripts/g0/tests_torch`). All new code was written test first (RED seen) and mutation-checked with `scripts/g0/mutcheck.py` (inventory 14 of 14, metrics 21 of 21, collector 34 of 34 non-equivalent mutants plus the run hook and CLI; one equivalent mutant survives by design).
- Ada: deployed commit `1655450` (`scripts/g0/deploy_phase1.sh`); jobs `31613` (`assoc-smoke-on`) and `31614` (`assoc-smoke-off`) submitted 06:36 IST, pending (the queue estimate was days; the real start is unknown). `/home2` has 7.9 GB free; `/share1` is NOT mounted on compute nodes.

## 3. What was done

- **Task 1** `scripts/g0/phase1_association_inventory.py` (+19 stdlib, +12 torch tests): schema-only inventory (verdicts available / derivable / incomplete / new_collection_required, provenance checks), `frozen_manifest` (cohort sizes, per-language counts, dev ties 0, FP baseline on devtest: 6,663 of 8,086 established, positives rounded), CLI; outputs `reports/association_retention/artifact_inventory.md` and `frozen_cohort_manifest.json`; local cache files are byte-identical to Ada's.
- **Task 3** `scripts/g0/phase1_association_metrics.py` (27 tests): `association_metrics`, `compare_profiles`, `association_status` (pure NumPy; ties as sets with tolerance 1e-12 for ties only; undefined distances are NaN, never 0).
- **Task 2** `scripts/g0/phase1_association_collect.py` (35 tests), the optional `profile_sink` in `run_survival` (default path unchanged: results identical with and without a sink), CLI options `--association-profiles`, `--association-baseline`, `--association-sentence-streams`, `--association-sums-dtype`. Sidecar: `association/<stream>/profiles.pt`, per-sentence CSR shards, `manifest.json` with hashes and sizes; `validate_run` rejects incomplete, tampered or wrongly rostered runs. The model-loading wiring in `phase1_run.main` is covered only by the real-model smoke.
- Corrections found on the way: `/share1` is not mounted on compute nodes (storage plan changed to `/home2` with the size options); a first FP-baseline count of 31.9 % had an indexing bug (the corrected 82.4 % stands).

## 4. The next step

See STATUS.md NEXT STEP: (1) check `squeue`/`sacct` once per firing; when both smoke jobs finished, verify Gate B (on/off historical outputs identical; `build_inventory` on `association/eq/profiles.pt` says `available`; `validate_run`; FP equals the legacy reference-side counts; INT0 equals FP; peak GPU memory, bytes per condition, nnz per sentence and the extrapolation). (2) On the CPU meanwhile: Task 4 `scripts/g0/phase1_association_analysis.py` (paired bootstrap over aligned sentence ids, sources M01-M05), tests first. Then the full-run decision and Task 6.

## 5. Open questions for the user

Licence of our own statistics in the dataset card (`license: other`); Q6, Q7, Q8, Q11, Q12 in `DECISIONS_FOR_REVIEW.md` remain open (not blocking).

## 6. Environment and pitfalls

Local venv `~/.venvs/anlp-torch`; cached run files `~/.cache/anlp-always-late/phase1_runs/p1-mask/{eq,all}`; Ada `ssh karma@ada-gw1`, project `/home2/karma/duan-verify`, launcher `phase1/repo/scripts/g0/run_phase1.sbatch` (deployed copy), outputs `out/phase1/<name>`; `sbatch --test-only` shows pessimistic start estimates; never print `hf/token`. Scratch scripts (feasibility checks, link checker, mutation JSONs) are in `/tmp/claude-1000/-home-karma-Documents-ANLP-project/239e7a33-140d-4688-8581-25c246ab3668/scratchpad/` and may be lost.

## 7. If something looks wrong

`git log --oneline -6`, `git status --short`, `CronList`, `ssh karma@ada-gw1 squeue -u karma`, the two suites above, `python3 scripts/g0/phase1_summary.py --check`.
