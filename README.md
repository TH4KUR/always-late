# Always Late: does compression forget language?

ANLP research project, 2026. We study what weight quantization preserves about multilingual SAE features: their activation patterns, language associations and measured intervention effects. The completed evidence combines 27 FLORES-200 languages, three Gemma-2-2B quantizers (original RTN, matched RTN and GPTQ), a six-language Wikipedia transfer panel, a Gemma-2-9B original-RTN scale extension and a limited Llama-3.2-1B consistency check.

**The main finding:** language-selective groups have a clear causal effect on next-token loss under the specified edits, while activation survival and language labels give only partial information about whether intervention effects are preserved. The size of low-bit changes depends substantially on the quantizer. These are useful, controlled findings within the evaluated systems; they do not establish a complete natural language circuit or universal behavior across models.

**Start with the [final research questions and answers](reports/FINAL_RESEARCH_QUESTIONS.md), the [research paper](paper/research-paper/main.tex), and the [results overview](reports/PROJECT_RESULTS.md).** The final questions organize the completed evidence retrospectively. Original frozen predictions and their pass/fail outcomes are retained.

| need | where |
|---|---|
| Final questions, answers and claim boundaries | [`reports/FINAL_RESEARCH_QUESTIONS.md`](reports/FINAL_RESEARCH_QUESTIONS.md) |
| Research paper and full project account | [`paper/research-paper/main.tex`](paper/research-paper/main.tex), [`reports/FULL_PROJECT_REPORT_DRAFT.md`](reports/FULL_PROJECT_REPORT_DRAFT.md) |
| All results, per-language tables and data guide | [`reports/PROJECT_RESULTS.md`](reports/PROJECT_RESULTS.md) |
| Current evidence and remaining work | [`reports/CURRENT_STATUS.md`](reports/CURRENT_STATUS.md) |
| Quantizer comparison and measurement study | [`reports/gptq_extension/RESULTS.md`](reports/gptq_extension/RESULTS.md), [`reports/survival_measurement/stage3/README.md`](reports/survival_measurement/stage3/README.md) |
| Functional claim map and figures | [`RESULTS_SUMMARY.md`](reports/functional_validation/RESULTS_SUMMARY.md), [`FIGURE_INDEX.md`](reports/functional_validation/FIGURE_INDEX.md) |
| Full documentation map and history | [`reports/INDEX.md`](reports/INDEX.md), [decisions](reports/decisions_and_anomalies.md), [progress log](reports/progress_log.md), [handoffs](reports/handoffs/) |
| Environment, reproduction and code ownership | [`HANDOFF.md`](HANDOFF.md), [`REPRODUCE.md`](reports/functional_validation/REPRODUCE.md), [`docs/CODEBASE.md`](docs/CODEBASE.md) |
| Course report, presentation and submission | [`paper/README.md`](paper/README.md), [`paper/course-report/README.md`](paper/course-report/README.md) |
| Original proposal and course guidelines | [`anlp_revised.txt`](anlp_revised.txt), [`project_guidelines.txt`](project_guidelines.txt) |

**Status, 30 October 2026 (evidence through 29 October):** the present experiments and analyses are the completed body of work. The Gemma-2-9B GPTQ run remains pending and has no reported result. Matched RTN on 9B, arithmetic-precision auditing and broader model/SAE coverage are future work. No recurring jobs should be recreated without the user's request.

Folders: `scripts/g0/` code; `reports/` evidence, freezes and decisions; `docs/` plans and code map; `paper/` research paper, course report and slides; `third_party/duan2026/` vendored reference code; `data/` local data. Large files are outside git: the private [Hugging Face dataset](https://huggingface.co/datasets/karma-skz/always-late-artifacts), run directories and wandb (`karma-skz-/anlp-project`, [schema](reports/wandb_schema_v2.md), [artifact inventory](reports/artifacts.md)). No credentials belong in documentation.

Earlier wording is preserved in [`reports/archive/reporting_revision_2026-10-29/`](reports/archive/reporting_revision_2026-10-29/); earlier cleanup archives remain under `reports/archive/`.
