# Completed-study viva presentation: revision 4

Updated 30 October 2026. `slides.tex` is the active 21-slide 16:9 presentation,
built with plain pdfLaTeX. It preserves the existing visual style and author
roster while aligning the completed results to the four synthesis questions in
[`reports/FINAL_RESEARCH_QUESTIONS.md`](../../reports/FINAL_RESEARCH_QUESTIONS.md).

The deck includes three 2B quantizer recipes, the streamed measurement audit,
positive group effects and graded dependence, reviewed 9B original-RTN results,
the Llama discrepancy, the comprehension null, limited repair, recognizable
individual effects, modest positive R1 prediction and crossed transfer with the
fresh-text panel. It treats model/dictionary coverage as a generalization
boundary and retains substantive control and endpoint qualifications. No
result slide contains a pending placeholder. Gemma-2-9B GPTQ supplies no result.

```bash
bash build.sh final
```

Output: [`AlwaysLate-Final-Presentation.pdf`](AlwaysLate-Final-Presentation.pdf).
The final-mode build refuses any remaining `\pending{...}` result box. Build
records and render inspection are in [`validation.md`](validation.md). Detailed
numerical provenance is in the aligned
[`course-report/evidence_map.md`](../course-report/evidence_map.md), with short
source labels on the slides.

The previous source and README are archived verbatim under
[`reports/archive/reporting_revision_2026-10-29/paper/presentation/`](../../reports/archive/reporting_revision_2026-10-29/paper/presentation/).
The course submission archive can be built separately with
`bash scripts/make_final_archive.sh <outdir> paper/presentation/AlwaysLate-Final-Presentation.pdf`.
This revision does not claim the archive was submitted.
