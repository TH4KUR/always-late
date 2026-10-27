# Always Late: does compression forget language?

ANLP course project, 2026. Gemma-2-2B, its layer-12 Gemma Scope 16k SAE, 27 FLORES-200 languages, weights rounded to INT8 down to INT4. We test whether language-specific features break first when the model is compressed, and what that does to the language.

**Start with [`reports/PROJECT_RESULTS.md`](reports/PROJECT_RESULTS.md):** all results from the start of the project in plain words, why they are not conclusive, what the null results tell us, the second-model and downstream-task checks, per-language tables and a guide to every data file.

| need | where |
|---|---|
| All results, simple language, per-language tables, data guide | [`reports/PROJECT_RESULTS.md`](reports/PROJECT_RESULTS.md) |
| Current state and what was left open | [`reports/CURRENT_STATUS.md`](reports/CURRENT_STATUS.md) |
| How the project progressed (the record, never shortened) | [decisions](reports/decisions_and_anomalies.md), [open questions](DECISIONS_FOR_REVIEW.md), [progress log](reports/progress_log.md), [handoffs](reports/handoffs/), [plans and designs](docs/superpowers/), [source register](reports/research_sources.md) |
| Map of every document | [`reports/INDEX.md`](reports/INDEX.md) |
| Claim-by-claim table of the functional study, figures | [`RESULTS_SUMMARY.md`](reports/functional_validation/RESULTS_SUMMARY.md), [`FIGURE_INDEX.md`](reports/functional_validation/FIGURE_INDEX.md) |
| Rerun, environment, rules for whoever continues | [`HANDOFF.md`](HANDOFF.md), [`REPRODUCE.md`](reports/functional_validation/REPRODUCE.md), [`docs/CODEBASE.md`](docs/CODEBASE.md) |
| Report, slides, final archive (due 31 October 2026) | [`paper/README.md`](paper/README.md), [`paper/course-report/README.md`](paper/course-report/README.md) |
| Original proposal, course guidelines | [`anlp_revised.txt`](anlp_revised.txt), [`project_guidelines.txt`](project_guidelines.txt) |

Folders: `scripts/g0/` code and tests (completed transfer follow-ups in [`scripts/g0/transfer_followup/`](scripts/g0/transfer_followup/README.md)); `reports/` evidence, freezes, decisions; `docs/` plans and code map; `paper/` report and slides; `third_party/duan2026/` vendored reference code (do not edit); `data/` local data. Large files are outside git: the private [Hugging Face dataset](https://huggingface.co/datasets/karma-skz/always-late-artifacts), the run directories, and wandb (`karma-skz-/anlp-project`, [schema](reports/wandb_schema_v2.md), [artifact inventory](reports/artifacts.md)). No credentials belong in documentation.

The previous README text is kept verbatim in [`reports/archive/doc_cleanup_2026-10-27/`](reports/archive/doc_cleanup_2026-10-27/README.md).
