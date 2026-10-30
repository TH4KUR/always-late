# Course final report: revision 5

Updated 30 October 2026. The report covers the completed study, including the
Gemma-2-2B matched-RTN/GPTQ extension, survival measurement decomposition,
continuous survival-to-effect prediction and reviewed Gemma-2-9B original-RTN
results. The four final questions follow
[`reports/FINAL_RESEARCH_QUESTIONS.md`](../../reports/FINAL_RESEARCH_QUESTIONS.md).
They organize the work retrospectively; frozen hypotheses and original verdicts
remain in their source files. Gemma-2-9B GPTQ contributes no result yet.

The paper reports positive intervention evidence with its measured boundaries:
all-language Gemma next-token effects, quantizer-dependent preservation and
transfer, and modest positive survival-prediction gains. It retains the Belebele
null, limited repair, Llama discrepancy, imperfect matching and documented 9B
criterion errors. Model/dictionary coverage limits generalization; the text does
not use that boundary to dismiss the controlled findings. Shared SAE evaluation
challenges appear with relevant literature. Revision 5 rewrites the text in plain academic language, defines terms before
using them, and adds a clear Problems Found and Corrected section, including the
Duan BOS discovery and our protocol correction. No new experiment or analysis
was introduced.

## Deliverables and page limits

- [`main.tex`](main.tex) is the only active course manuscript.
- [`AlwaysLate-Final-Report.pdf`](AlwaysLate-Final-Report.pdf) is the compiled
  revision-5 report: **10 pages**. Main discussion ends on page 7; the
  separate limitations section is also on page 7. References occupy page 8;
  appendix pages 9–10.
- [`evidence_map.md`](evidence_map.md) maps claims, units, populations, intervals
  and analysis status to completed result files.
- [`validation/validation.md`](validation/validation.md) and
  [`validation/manifest.json`](validation/manifest.json) record the actual build,
  checker result, hashes and limits of validation.
- The aligned presentation is in [`../presentation/`](../presentation/README.md).

The course deadline is 31 October 2026. Its maximum is eight main-content pages,
excluding references and a separate limitations/challenges/future-work section,
plus at most two appendix pages. Revision 5 meets these limits. The course
requires a code archive and presentation in `AlwaysLate-Final.zip`; this directory
provides the report, not evidence that a final ZIP has been submitted. Author
names and order follow the existing roster.

## Build and checks

```bash
bash build.sh
bash check.sh
```

The local build uses installed pdfLaTeX and BibTeX and the existing optional
per-user TeX support cache. `check.sh` runs the pinned official ACL checker with
its default CLI checks and fails if the generated JSON contains findings. The
revision-5 checker prints **All Clear!** and records `{}`. No official style or
checker implementation was modified. No body font, margin or spacing overrides
were introduced.

The official CLI does not enable its optional online reference/name-service
path. The earlier `check_full.py` attempt encountered an external Scholarcy
Cloudflare challenge and remains recorded as incomplete. That optional upload
was not repeated. The local multi-file build is the compilation evidence; a
previous attempt at the editor compiler could not download its runtime bundle.
The preserved official BibTeX records are unchanged; the added GPTQ entry was
checked against its primary arXiv metadata on 30 October 2026. Leask, Makelov and SAEBench field-context references use the official-proceedings-verified entries recorded in `reports/publication_context_2026-10-30.md`.

## Revision record

Revision 1 established the course report on 26 October. Revision 2 added the
Llama and gain-study inventory. Revision 3 emphasized intervention transfer and
added the external panel on 26–27 October. Revision 4 reorganizes the completed
investigation around the four synthesis questions and incorporates the newer
completed studies. Revision 5 simplifies the writing and makes the BOS correction
and project problems explicit. The revision-4 source and PDF remain in Git at
commit `0dc8814`. Detailed older records and exact overwritten source/output
bytes are preserved under
[`reports/archive/reporting_revision_2026-10-29/paper/course-report/`](../../reports/archive/reporting_revision_2026-10-29/paper/course-report/).
The files in `draft/` are superseded working copies, not competing active sources.

## Official assets

`acl.sty`, `acl_natbib.bst`, `official-acl_latex.tex` and
`official-formatting.md` remain unmodified copies from the official
[ACL style repository](https://github.com/acl-org/acl-style-files) at
`d5adc823ff0f80f98c80405ca0ab66c68e684409`. The pinned
[official checker](https://github.com/acl-org/aclpubcheck) revision is
`f626497051df8689e990fac36ba9f42cd630e31c`. A format pass does not certify
scientific novelty or promise course/conference acceptance.
