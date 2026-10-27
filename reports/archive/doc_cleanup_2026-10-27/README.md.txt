# Always Late: SAE feature survival and intervention transfer

ANLP course project, 2026. The main study uses Gemma-2-2B, its layer-12
Gemma Scope 16k SAE, 27 FLORES languages and simulated weight-only RTN.
The external panel uses Wikipedia passages in six languages.

**Start with [current status and next action](reports/CURRENT_STATUS.md).**
That is the one page to update when an experiment, analysis or deliverable changes.
The [research direction](RESEARCH_DIRECTION.md) explains the publication question
and its limits. A completed experiment does not establish a publishable contribution.

## Find the right document

| Need | Entry point |
|---|---|
| Current execution state, results and next action | [Current status](reports/CURRENT_STATUS.md) |
| Take over the work, operational rules and environment | [Handoff/runbook](HANDOFF.md) |
| Code ownership, module structure and tests | [Codebase map](docs/CODEBASE.md) |
| Complete document/history map | [Reports index](reports/INDEX.md) |
| Earlier studies and claim qualifications | [Functional results](reports/functional_validation/RESULTS_SUMMARY.md), [Study G](reports/functional_validation/gain_study/README.md) |
| Main and external transfer results | [Main](reports/intervention_transfer/results/main/README.md), [external](reports/intervention_transfer/results/external/README.md) |
| Locked prediction, gradient, support and native/fixed analyses | [Post hoc recovery results](reports/intervention_transfer/results/recovery/README.md) |
| Novelty search and source history | [26 October assessment](reports/novelty_assessment_2026-10-26.md), [source register](reports/research_sources.md) |
| Decisions and unanswered questions | [Decisions/anomalies](reports/decisions_and_anomalies.md), [review questions](DECISIONS_FOR_REVIEW.md) |
| Report, presentation and submission | [Paper/deliverables](paper/README.md), [course report](paper/course-report/README.md) |
| Reproduction and figure inventory | [Reproduction](reports/functional_validation/REPRODUCE.md), [figures](reports/functional_validation/FIGURE_INDEX.md) |
| Original plans and frozen protocols | [Designs](docs/superpowers/specs/), [plans](docs/superpowers/plans/) |
| Previous page text and experiment history | [Archive](reports/archive/), [progress log](reports/progress_log.md) |

## Repository structure

- `scripts/g0/`: historical experiment/analysis entry points and tests.
  Completed transfer follow-ups live in [their package](scripts/g0/transfer_followup/README.md);
  the old script commands remain valid.
- `reports/`: generated evidence, source registers, freezes, decisions and history.
- `docs/`: code ownership, maintenance records and original research designs/plans.
- `paper/`: ACL course report and viva presentation; the older paper draft is retained.
- `third_party/duan2026/`: vendored reference implementation; preserve pristine code.
- `data/`: local data where available; large artifacts remain outside git.

## Data and tracking

The main dataset is FLORES-200 (dev for feature selection, devtest for evaluation);
Belebele supplies the comprehension task. Model access is gated. Large artifacts are
in the private [Hugging Face dataset](https://huggingface.co/datasets/karma-skz/always-late-artifacts)
and local/cluster run directories. Tracking uses wandb entity `karma-skz-`, project
`anlp-project`; see [artifact inventory](reports/artifacts.md) and
[tracking schema](reports/wandb_schema_v2.md). No credentials belong in documentation.

The [original proposal](anlp_revised.txt) and [course guidelines](project_guidelines.txt)
remain available. Prior README text is [preserved verbatim](reports/archive/modular_cleanup_2026-10-27/README.md).
