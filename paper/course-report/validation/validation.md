# Course-report validation: revision 5, 30 October 2026

The report was rewritten in plain academic language after the user's readability feedback. It explains terms before using them, gives findings before statistical qualifications, and has a Problems Found and Corrected section. That section explicitly describes the missing BOS in Duan's released Gemma evaluation and the correction applied to our later experiments. It also records our retired pipeline, INT16 assumption, hardware rerun and the 9B checklist defects.

`bash build.sh` succeeds with the existing local pdfLaTeX/BibTeX toolchain. The PDF has **10 pages**: the main discussion ends on page 7, limitations also occupy page 7, references are on page 8, and the appendix is pages 9–10. It meets the eight-page main-content and two-page appendix limits. The author order, numerical evidence, bibliography and official ACL assets are preserved.

`bash check.sh` reports **All Clear!** with an empty findings JSON. No checker flags are disabled. The final log has no undefined citations/references or overfull boxes. All fonts are embedded, with no Type 3 fonts. All ten final pages were rendered at 90 dpi and inspected, including the BOS section, equations, quantizer table, references and inventory table. Renders are retained in ignored `rendered/readability/`; hashes and page counts are in `manifest.json`.

This is a writing and presentation revision. No model run or statistical analysis was added. The evidence map retains each finding's source, units, population and uncertainty. The small positive prediction gains, limited repair, comprehension null, Llama difference and pending 9B GPTQ are still explicit. The prior revision-4 manifest is preserved under `previous_revisions`.

The existing multi-file manuscript uses its local build. The native editor runtime was not retried after its earlier download failure. The optional online reference checker was not rerun; its earlier Scholarcy challenge remains an incomplete historical check. Existing verified bibliography records were preserved. The current check makes no new literature or submission claim.
