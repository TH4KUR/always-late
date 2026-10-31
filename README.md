# Always Late: does compression forget language?

ANLP research project, 2026. We study what weight quantization preserves about multilingual SAE features: their activation patterns, language associations and measured intervention effects. The evidence combines 27 FLORES-200 languages, three Gemma-2-2B quantizers (original RTN, matched RTN and GPTQ), a six-language Wikipedia transfer panel, a Gemma-2-9B three-arm confirmation and a limited Llama-3.2-1B consistency check.

**The main finding:** language-selective groups have a clear causal effect on next-token loss under the specified edits, while activation survival and language labels give only partial information about whether intervention effects are preserved. The size of low-bit changes depends substantially on the quantizer.

| need | where |
|---|---|
| Final report, slides, explainer | [`paper/course-report/main.pdf`](paper/course-report/main.pdf), [`paper/slides/main.pdf`](paper/slides/main.pdf), [`paper/project-explainer/main.pdf`](paper/project-explainer/main.pdf) |
| All findings, every study | [`reports/COMPLETE_FINDINGS.md`](reports/COMPLETE_FINDINGS.md) |
| Results in plain words, final research questions | [`reports/PROJECT_RESULTS.md`](reports/PROJECT_RESULTS.md), [`reports/FINAL_RESEARCH_QUESTIONS.md`](reports/FINAL_RESEARCH_QUESTIONS.md) |
| Decision and anomaly log | [`reports/decisions_and_anomalies.md`](reports/decisions_and_anomalies.md) |
| Reproduction and code map | [`REPRODUCE.md`](reports/functional_validation/REPRODUCE.md), [`docs/CODEBASE.md`](docs/CODEBASE.md) |
| Original proposal and course guidelines | [`anlp_revised.txt`](anlp_revised.txt), [`project_guidelines.txt`](project_guidelines.txt) |

Folders: `scripts/g0/` code and tests; `reports/` results per study, with their freezes and amendments; `docs/superpowers/` the frozen designs and plans; `paper/` the final documents; `out/analysis/` the Gemma-2-9B analysis outputs; `third_party/duan2026/` vendored reference code; `data/` local data (not in git).

Large files are outside git: the private [Hugging Face dataset](https://huggingface.co/datasets/karma-skz/always-late-artifacts), run directories and wandb (`karma-skz-/anlp-project`, [schema](reports/wandb_schema_v2.md), [artifact inventory](reports/artifacts.md)). The course submission archive is built with `bash scripts/make_final_archive.sh <output dir>`.
