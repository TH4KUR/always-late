# Handoff 024: functional study, pilot results so far, rescue and freeze code done, evaluation jobs next (2026-10-22, 01:40 IST)

> The ONLY source of this chat for a fresh session, together with the repository. If this file and the repository disagree, the repository wins: say so in the next handoff. Start with `reports/functional_validation/STATUS.md` (NEXT STEP block and Log), then `reports/functional_validation/results/pilot/report.md`. Handoff 023 holds the standing rules and the mission; they are unchanged.

## 0. Read this first
- Standing rules and the user's instructions: see handoff 023 section 0 (autonomy for at least 5 to 6 firings of the 40-minute job, plans before code, questions to `DECISIONS_FOR_REVIEW.md`, publishability and methodological correctness first, mask regime only, never edit `paper/` or `third_party/`, at most 2 Ada jobs, secret scan, explicit staging, commit trailer).
- The 40-minute job: two session-only cron entries (see `reports/functional_validation/cron_prompt.md`; they expire about 2026-10-28; recreate if `CronList` shows fewer than two).

## 1. Mission in five lines
D-039/D-040: do the language-selective SAE feature groups carry function beyond activity-matched controls (F1), is it kept in each RTN-quantized model against its own baseline (F2), where does quantization damage sit (FP-donor rescue, F3), does association retention predict function (F4), SAE fidelity (B). Loss task on 27 languages (exploratory), Belebele on languages with headroom (frozen before scoring).

## 2. State right now (verified 2026-10-22 01:40)
- Git `main` at the commit that contains this handoff; tests: stdlib 205, torch 528 (`python3 -B -m unittest discover -s scripts/g0/tests`; `OMP_NUM_THREADS=4 ~/.venvs/anlp-torch/bin/python -B -m unittest discover -s scripts/g0/tests_torch`).
- Ada: pilot jobs 33788 (FP, INT8, INT7, INT6, INT5) and 33789 (INT4, INT16, INT12, INT0) running since 2026-10-21 22:58, limit 4 hours (about 02:58); outputs per precision in `/home2/karma/duan-verify/out/phase1/func-pilot-{a,b}/`; finished: FP, INT8, INT4, INT16. Deployed `phase1/repo` = commit `5a3a733`; snapshot `repo_8d73565` exists. Belebele data and `passage_origin.json` are on Ada under `phase1/data/belebele`.

## 3. What happened since handoff 023
Rescue code (FP-donor patches, donor cache, `--rescue`, recovered-fraction analysis), Belebele calibration module and `--task mc`, per-snapshot deployment with a guard, analysis module (macro retention rule D-041, median ratio, family table), evaluation presets and caps (D-042), freeze helper. Results (pilot, exploratory, `results/pilot/report.md`): full precision: E > 0 in 27 of 27 languages, median 0.27 nats per token for 64 features against 0.02 for matched controls, sham exactly 0, heterogeneous (0.005 to 7.3); INT16 = FP, INT8 = FP within 1 %, INT4 amplified (median ratio 1.43; SAE fidelity worse at INT4). Cost: about 58 minutes per precision for the primary preset on 150 sentences.

## 4. The next step, exactly
See STATUS.md NEXT STEP: read INT7 and later pilot files as they land; commit the loss-task freeze (`phase1_func_freeze.py build`) at the first firing after INT7 is read; deploy a snapshot of the frozen commit and queue the evaluation jobs (command in STATUS); queue INT6/INT5 pilot, Belebele calibration and the rescue smoke job as slots free; then figures (Task 8), the existing-data analyses (A) and documentation consolidation (Task 9).

## 5. Open questions for the user
Q17 (design defaults), Q18 (scoop risk), older Q6, Q7, Q8, Q11, Q12, licence of our statistics, shard backup. None blocks.

## 6. Environment and pitfalls
Deploy only to named snapshots while jobs run (`deploy_phase1.sh repo_<short commit>`; the default `repo` is refused). Time limits cannot be extended; plan jobs at about 58 minutes per precision of the primary preset. Local venv has no matplotlib. TransformerLens 3.9: `model(tokens, stop_at_layer=13)` residual after block 12; `model(resid, start_at_layer=13)` logits. Items of Belebele files are ordered differently per language: align by (link, question_number).

## 7. What I would check if something looks wrong
`git log --oneline -8`, `git status --short`, `CronList`, `ssh karma@ada-gw1 squeue -u karma; sacct -u karma -S 2026-10-21`, both test suites, `python3 scripts/g0/phase1_func_groups.py verify ...` (see handoff 023), and after the freeze `python3 scripts/g0/phase1_func_freeze.py verify --file reports/functional_validation/evaluation_freeze_lm.json`.
