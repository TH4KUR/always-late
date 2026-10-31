# Codebase map

This map describes ownership and entry points. Scientific conclusions belong to
the result files and research status, not to code organization.

## Code families

| Area | Implementation / entry points | Responsibility |
|---|---|---|
| Upstream reference | `third_party/duan2026/`, `scripts/g0/their_code.py`, `verify_vendor.py` | Vendored reference code and verification; preserve pristine files |
| Phase 1 | `phase1_lib.py`, `phase1_data.py`, `phase1_run.py`, `phase1_analysis.py` | Data, measurement and original survival analysis |
| Association | `phase1_association_*`, `phase1_assoc_*` | Language inventory, association retention, margins and functional links |
| Functional interventions | `phase1_func_tasks.py`, `phase1_func_intervene.py`, `phase1_func_groups.py`, `phase1_func_run.py` | Task data, intervention construction, feature groups and evaluation |
| Functional analysis | Other `phase1_func_*analysis.py`, `phase1_func_*figures.py` | Loss/task results, F4/F5, calibration and figures |
| Study G | `phase1_gain_*` | Gain calibration, runs, analysis and extensions B/C/D |
| Exploratory second model | `phase1_llama_*` | Llama SAE loading, run, comparison and robustness |
| Transfer study | `phase1_transfer_freeze.py`, `panel.py`, `core.py`, `external.py`, `run.py`, `analysis.py`, `report.py`, `validate.py` (all with the `phase1_transfer_` prefix) | Frozen samples, model experiments, primary analysis and validation |
| Completed transfer follow-ups | [`scripts/g0/transfer_followup/`](../scripts/g0/transfer_followup/README.md) | Cross-corpus prediction, locked-model recovery, gradient checks, control support, native/fixed contrasts |
| Paired arithmetic audit | `phase1_transfer_precision*.py`, `run_precision.sbatch` | Selection, suffix arithmetic, cross-mode measurements, summaries, checkpoints and execution; separate from completed follow-ups |
| Tracking/artifacts | `g0_wandb.py`, `wandb_upload_v2.py`, `hf_artifacts.py` | Tracking, finished-result uploads and artifact handling; no network action is part of a refactor check |
| Operations | `deploy_phase1.sh`, `fetch_phase1.sh`, `run_*.sbatch`, `reproduce_*.py`, `reproduce_functional.sh` | Deployment snapshots, job entry points and reproduction |
| Maintenance/tests | `check_md_links.py`, `mutcheck.py`, `tests/`, `tests_torch/` | Documentation links, mutation checks and existing unittest suites |
| Course deliverables | `paper/course-report/`, `paper/presentation/`, `scripts/make_final_archive.sh` | ACL report, presentation and final archive |

Unless a full path is shown, code filenames above are relative to `scripts/g0/`.
Some families intentionally retain several files with the same prefix: they are
distinct measurement/analysis stages, not interchangeable implementations.

## Module boundary introduced by the cleanup

```text
legacy phase1_transfer_{cross_corpus,recovery,gradient_check,support,native_fixed}.py
    -> transfer_followup/{cross_corpus,recovery,gradients,support,native_fixed}.py
         -> transfer_followup/tables.py  (CSV and file hashes only)
         -> phase1_transfer_analysis.py (unchanged frozen numerical engine)
```

The legacy entry points remain valid. Tests and new analysis code may import
`transfer_followup` when `scripts/g0` is on `sys.path`, as established by the
existing scripts/tests. This is not a repository-wide packaging/dependency migration.

## Where changes belong

- Numerical changes to a historical study require a new protocol/amendment and
  separately labelled results. A cleanup must not silently change a frozen module.
- New completed follow-up analyses belong beside their related modules; common
  file contracts go in `tables.py`, not statistical decisions.
- Model loading, hooks, dtype promotion and job/checkpoint logic stay in the
  experiment/precision family, outside pure CSV analysis modules.
- Plotting belongs to the corresponding figure modules; report-specific layout
  belongs in `paper/`. Never alter official ACL style files for layout fixes.
- Prefer a new appendix/update or verbatim archive with a pointer to rewriting
  old decision records, handoffs, plans, source registers or logs.

## Verification

From the repository root, using the existing environment:

```bash
python3 -B -m unittest discover -s scripts/g0/tests
OMP_NUM_THREADS=4 ~/.venvs/anlp-torch/bin/python -B -m unittest discover -s scripts/g0/tests_torch -p 'test_phase1_transfer_*.py'
python3 -B scripts/g0/check_md_links.py
```

The focused suite covers the compatibility entry points and shared file contracts.
For this cleanup, also reproduce the five analyses on saved real data and compare
their files byte for byte. The recorded protocol and checks are in the git history
(`docs/maintenance/2026-10-27-cleanup-plan.md`).

Existing paths in old plans, Slurm commands and source hashes refer to the historical
code state. Keep those records; use the new module path for future maintenance.

## Added 2026-10-29: the GPTQ extension and the survival measurement study

| area | files | what |
|---|---|---|
| GPTQ extension (D-101 to D-111) | `scripts/g0/gptq_ext/` (calibration, GPTQ upstream copy, artifact build and hash checks, gates, wrapper `run_with_weights.py`, comparison and validation tools), tests `scripts/g0/tests_torch/test_gptq_ext_*.py`, launcher `scripts/g0/run_gptq.sbatch` | quantize the same model with GPTQ or a matched RTN instead of the legacy RTN and run the old experiments unchanged through the wrapper |
| survival versus effect (D-110 to D-114) | `scripts/g0/survival_effect.py`, `survival_effect_llama.py`, report generators | does the activation correlation predict a feature's effect similarity |
| survival measurement study (D-117 to D-124) | `scripts/g0/survival_lambda.py` (stage 1), `survival_sidecar.py` (streams 21 statistics per condition on the GPU), `survival_validate.py`, `survival_stage3.py` (the frozen analysis); freeze `reports/survival_measurement/analysis_freeze.json` | what the survival correlation measures |

## Final documents

[Course report](../paper/course-report/main.pdf), [complete findings](../reports/COMPLETE_FINDINGS.md). Code organization does not change experiments, frozen rules, numerical outputs or reproduction code.
