# Research manuscript validation, 30 October 2026

The multi-file manuscript builds with the existing local pdfLaTeX/BibTeX toolchain through `bash build.sh`. The compiled PDF is **11 pages**: main content ends on page 8, limitations and references occupy pages 9–10, and the appendix is page 11. No official ACL style or bibliography-style asset was modified; both are byte-identical to the established course-report copies.

Both the pinned official ACL checker (`f626497051df8689e990fac36ba9f42cd630e31c`) and the newest upstream checker report **All Clear!**, with empty findings objects (long-paper mode, default checks). Their console outputs and JSON are retained under `validation/`. No checker flags were disabled. This is a format check, not a publication-acceptance or comprehensive scientific-validation claim.

The final build has no undefined references/citations or overfull-box warnings. All eleven pages were rendered and visually inspected, including the equations, three result tables, two figures, bibliography and appendix. The figures read stored estimates and intervals directly from committed result tables; no inference runs or new statistical analysis were introduced. The accompanying evidence map records estimands, populations, statistical status and distinct bootstrap units.

A fresh evidence review corrected the stricter selectivity threshold, the K64 control-draw count, the positive-control populations and the small positive fresh-sentence Extension C gain. The original 9B prediction-definition problems and all relevant null results remain explicit. The paper reports continuous activation-correlation gains as small positive estimates rather than zero-information findings.

The source was opened in the Codex editor. Its native compiler could not download the Tectonic runtime bundle, so it did not verify compilation. The successful local multi-file build is the basis for the compiled PDF and the formatting-check result. No runtime/plugin installation was needed. The source remains editable.

Primary-source checks for the SAE methodological context are documented in [`../../reports/publication_context_2026-10-30.md`](../../reports/publication_context_2026-10-30.md). The manuscript makes no exhaustive novelty or public-artifact availability claim. Gemma-2-9B GPTQ remains pending and supplies no result.
