# reports/legacy_pipeline: results of the DISCARDED pipeline

These files were produced by our own earlier pipeline (`src/` and the Phase-1 / demo scripts), which was discarded on 2026-10-16 in favour of building on Duan's codebase. The code is recoverable from the git tag `legacy-pipeline-2026-10-16`.

**Status: hypotheses, not findings.** The pipeline's mechanics were cross-checked against Duan's code on WikiText in the BOS regime (`reports/audit_our_vs_duan.md`), but this FLORES-based Phase-1 stage was not re-run under that audit, it was never validated at INT5/INT4, and it carries the design and statistics issues listed in `issues.md` (BOS in every statistic, decode/re-encode token drift, unmatched content, feature-level tests, firing-rate-only controls). They are kept because `paper/main.tex` traces its tables to them and the team needs them to rewrite the paper. Do not cite them as validated results. See the post-mortem entry in `reports/progress_log.md` (2026-10-16, night).

| File | What |
|---|---|
| `selectivity_results*.json`, `selectivity_summary_table.csv` | Phase-1 selectivity extraction (the `.csv` contains hard-coded fallbacks and stale 1.5k numbers, `issues.md` D5) |
| `phase1_survival_results*.json` | survival of selective / matched-universal features (`_27lang.json` is a byte-identical copy of `_27lang_gpu.json`; `phase1_survival_results.json` is the older run) |
| `quantization_survival_demo_result.json` | the 2,048-token WikiText demo (INT7 98.28%, INT6 94.31%) |
| `behavioral_loss_link_results.json`, `language_behavioral_dataset.csv`, `fertility_vs_english_flores_dev.json` | the exploratory behavioural link |
| `w0_sanity_check_result.json`, `multilang_sanity_check_result.json`, `mid_submission_results.json` | early checks and the mid-submission summary |
