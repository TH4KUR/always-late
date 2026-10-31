# Always Late: does compression forget language?

ANLP course project (IIIT Hyderabad, Monsoon 2026). We study what weight quantization preserves about multilingual SAE features: their activation patterns, language associations and measured intervention effects. The evidence combines 27 FLORES-200 languages, three Gemma-2-2B quantizers (legacy RTN, matched RTN and GPTQ), a Gemma-2-9B confirmation and a Llama-3.2-1B consistency check.

**The main finding:** language-selective feature groups have a clear causal effect on next-token loss, but activation survival overstates functional damage (13% survival vs 99.7% effect identity at INT4). The features carry almost none of the actual quantization damage (rescue fraction 2.6%).

## Deliverables

| need | where |
|---|---|
| Final report (ACL format, 8 pages) | [`paper/course-report/main.pdf`](paper/course-report/main.pdf) |
| Beamer slides | [`paper/slides/main.pdf`](paper/slides/main.pdf) |
| HTML slides (reveal.js) | [`paper/slides/slides.html`](paper/slides/slides.html), [`paper/slides/slides-html.pdf`](paper/slides/slides-html.pdf) |
| Quick-access copies | [`AlwaysLate-Final-Report.pdf`](AlwaysLate-Final-Report.pdf), [`AlwaysLate-HTML-Slides.pdf`](AlwaysLate-HTML-Slides.pdf) |
| All findings | [`reports/COMPLETE_FINDINGS.md`](reports/COMPLETE_FINDINGS.md) |
| Results summary, research questions | [`reports/PROJECT_RESULTS.md`](reports/PROJECT_RESULTS.md), [`reports/FINAL_RESEARCH_QUESTIONS.md`](reports/FINAL_RESEARCH_QUESTIONS.md) |
| Decision and anomaly log | [`reports/decisions_and_anomalies.md`](reports/decisions_and_anomalies.md) |

## Repository structure

- `scripts/g0/` — experiment code and tests
- `reports/` — results per study, with freezes and amendments
- `paper/` — report, slides (beamer + HTML), and project explainer source
- `out/analysis/` — Gemma-2-9B analysis outputs
- `third_party/duan2026/` — vendored reference code from the reproduced study
- `data/` — local data (not in git)

## External resources

- **GitHub**: [TH4KUR/always-late](https://github.com/TH4KUR/always-late)
- **HF artifacts**: [karma-skz/always-late-artifacts](https://huggingface.co/datasets/karma-skz/always-late-artifacts)
- **wandb**: [karma-skz-/anlp-project](https://wandb.ai/karma-skz-/anlp-project) ([schema](reports/wandb_schema_v2.md), [artifact inventory](reports/artifacts.md))

The course submission archive is built with `bash scripts/make_final_archive.sh <output dir>`.
