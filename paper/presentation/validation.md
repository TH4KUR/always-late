# Presentation validation: revision 4

30 October 2026. `bash paper/presentation/build.sh final` succeeds and produces
**21 slides**. The final source contains no pending-result boxes; compilation
has no undefined-reference or overfull-box warning. Every final slide was
rendered and inspected in `build/rendered-rev4/`; a table-flow issue found during
inspection was corrected before the final build. The existing 16:9 style,
21-point body type and author roster are preserved. Tables remain editable
LaTeX objects, with body text below rather than alongside them.

The deck follows the four final retrospective RQs and includes the completed
matched-RTN/GPTQ extension, measurement decomposition, positive group effects,
reviewed 9B original-RTN evidence, Llama discrepancy, comprehension null,
limited repair, recognizable effect identity, modest positive continuous R1
prediction and transfer on both corpora. It distinguishes the small positive
Extension C repair from failed main repair criteria and nonselective positive
controls from other-language selective controls. Gemma-2-9B GPTQ contributes no
result. No model run or new numerical analysis was performed for this revision.

`validation.json` records final source/PDF hashes and build conditions. Detailed
provenance is in `../course-report/evidence_map.md`. Previous source and README
are archived verbatim under
`reports/archive/reporting_revision_2026-10-29/paper/presentation/`.

The deck was inspected as rendered PDF pages, not in PowerPoint. No PowerPoint
compatibility or conference acceptance check is claimed. Submission-archive
creation is separate.
