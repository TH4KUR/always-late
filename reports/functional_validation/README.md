# Functional validation across precision (D-039, D-040): index

Question: do the SAE features selected as language-selective before quantization carry function for their language (necessity beyond activity-matched controls), is that function kept in each RTN-quantized model (INT8/7/6/5/4), where does quantization damage sit (FP-donor patching), and does association retention (D-035) predict functional retention? Exploratory on already examined FLORES devtest for the loss task; Belebele is the held-out task, frozen before scoring.

| file | what it is |
|---|---|
| [`STATUS.md`](STATUS.md) | live status, NEXT STEP, gates, log (start here) |
| [design](../../docs/superpowers/specs/2026-10-21-functional-validation-design.md) | questions F1 to F4 and B, tasks, groups and controls, interventions, estimands, decision rules, freeze protocol, compute, risks |
| [plan](../../docs/superpowers/plans/2026-10-21-functional-validation.md) | tasks 0 to 9 with tests, mutants, gates and the job queue order |
| [`cron_prompt.md`](cron_prompt.md) | prompt and cron expressions of the 40-minute job |
| [`prior_art.md`](prior_art.md) | dated check of the exact combination (register P30 to P38), scoop risk |
| [`groups_manifest.json`](groups_manifest.json), [`groups_matching.md`](groups_matching.md) | frozen language groups (K16, K64, all) and activity-matched control draws; matching quality per language and size |
| [`evaluation_freeze_lm.json`](evaluation_freeze_lm.json) | **freeze of the loss-task evaluation** (2026-10-22): hashes of the manifests and the evaluation code, the 300 evaluation ids, presets, precision map, analysis constants; verify with `phase1_func_freeze.py verify` (torch venv) |
| [`figures/pilot/`](figures/pilot/) | figures 1, 2, 3, 5, 6, 8 (SVG with live text and PNG, light and dark, byte-reproducible) and three tables (CSV and markdown: retention by precision, per-language effects, configurations and sizes) made by `scripts/g0/phase1_func_figures.py` from the analysis outputs; titles describe, they never state a result; regenerate for the evaluation split when it exists |
| [`results/assoc_rank/`](results/assoc_rank/report.md) | A1 (exploratory, descriptive): D-035 association retention by dev-mass rank of the feature within its language, pooled and as a within-language top-64 contrast; hypothesis written before the numbers; checked against the D-035 headline |
| [`results/eval/`](results/eval/report.md) | **loss-task evaluation** on the 300 frozen devtest sentences (frozen settings; INTERIM while the evaluation jobs run: regenerate with `phase1_func_analysis.py --runs <dir> --split eval --out results/eval`); exploratory by design, the confirmatory family is T-MC |
| [`f4/`](f4/f4_matching.md) | **F4 subgroup manifests** (association versus function at equal native activity): per precision (INT6, INT5, INT4) the lost features of each language matched to retained features of equal native activity, in the schema of `groups_manifest.json`, plus the matching and balance report; the GPU jobs are not run yet (design: plan Task 7c, decision D-046) |
| [`results/pilot/`](results/pilot/report.md) | pilot ladder (150 devtest sentences, exploratory): report, per-language effects, retention, SAE fidelity |

Superseded banners of the front-page documents are kept verbatim in [`reports/archive/superseded_banners_2026-10-22.md`](../archive/superseded_banners_2026-10-22.md); the link checker is `scripts/g0/check_md_links.py`.

Related: [`reports/association_retention/`](../association_retention/README.md) (D-035 study and the selected direction in `next_steps.md`), [`reports/research_sources.md`](../research_sources.md), decisions D-039 to D-046 in [`reports/decisions_and_anomalies.md`](../decisions_and_anomalies.md), review items Q16 to Q19 in [`DECISIONS_FOR_REVIEW.md`](../../DECISIONS_FOR_REVIEW.md).
