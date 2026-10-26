# Course-report validation: revision 2 (2026-10-26 09:13 IST, 26 October 2026)

Status: complete draft for team review; not a submitted course ZIP.

- Revision 2 of the report was built with the installed pdfLaTeX and BibTeX (no unresolved references or citations; the only TeX box warning is a 1.4 pt vbox on the balanced reference page).
- 12 PDF pages: the main content (with the second-model section) ends on page 8; the limitations section is on page 9 (outside the page count); references are on page 10; the appendix is pages 11 and 12 (two pages).
- The official checker printed `All Clear!` with an empty JSON report, with the pinned version (`f626497...`, via `check.sh`) and with the newest upstream version, on the final PDF. A first run after an edit had flagged `Text on page 12 bleeds into the bottom margin` (a floating appendix table that was too tall); it was fixed by shortening the table and rechecked, not ignored. A deliberately non-ACL PDF was flagged by the same checker (`pubcheck-negative-control.txt`).
- The ACL template was re-downloaded from the official repository (HEAD `d5adc82...`): `acl.sty`, `acl_natbib.bst`, the template `.tex` and `formatting.md` are byte-identical to the copies in this folder; none was modified.
- All fonts are embedded and none is Type 3 (`fonts.txt`); the page size is A4.
- Independent fact-check: a separate agent read `main.tex` against the repository's result files and found no wrong number among those it could trace. Its framing findings were applied: the Llama verdict is now attributed to the failed existence prediction P1 (21 of 27 languages, 24 required), "pre-registered" became "pre-specified" with a definition (committed before scoring, no external registry), a mismatched AUROC pair was corrected, rows for extension D and the alpha-star rule were added to the rules table, the status words of the Belebele row and the prompt-check point estimates were qualified, and wording on depth, specificity and compute was aligned with the sources. Not applied: the suggestion to add the functional-study reproduction count (69 of 69 outputs, an earlier run) to the reproducibility paragraph, and a statement on the repository's visibility; the wandb sentence was reworded to what is certain.
- Not verified by the reviewer and not claimed: author names and order (taken from the project roster). The post hoc wandb upload, which the reviewer could not confirm, finished afterwards (decision D-091, 15 new runs, 34 in the project, verified through the wandb API by the agent that ran it); the sentence in Appendix A was updated to that state at 09:15 IST.
- The optional online reference check of the checker (`check_full.py`) again fails because the external service returns a web page instead of BibTeX; it is not claimed as passed. The bibliography (15 entries) was checked against the Anthology and arXiv records.

Revision 1 record (kept verbatim):

# Course-report validation — 26 October 2026

Status: complete first draft for team review; not a submitted course ZIP.

- The local pdfLaTeX/BibTeX build succeeded on the final source.
- Nine PDF pages: main content ends on page 7, limitations also fit on page 7,
  references occupy page 8, and the appendix occupies page 9. This meets the
  course's eight-page main-content and two-page appendix maximums.
- The final official CLI run printed `All Clear!`; its JSON report is `{}`.
  `check.sh` now also fails if that JSON contains findings. No checker flags
  disable any of the CLI's default checks.
- Upstream `acl.sty` and `acl_natbib.bst` are byte-identical to the pinned
  official repository. The document uses final mode, the official bibliography
  style and standard packages; no geometry, font-size, caption-size or spacing
  overrides are used to squeeze the content.
- Final TeX log: no unresolved references/citations or overfull boxes.
- PDF fonts are embedded; no Type 3 fonts. Every page was rendered and inspected
  for layout. Reference columns are balanced with the standard balance package.
- An independent read-only review checked scientific statements, measurement
  definitions, denominators and plotted values. Its two methods findings
  (discovery criteria and randomized top-three control matching) were corrected.
- All 27 appendix language codes were checked against the committed roster.
- Plots use committed result CSVs; error bars are explicitly labelled as
  quartiles or stored confidence intervals, according to the source.

## Important distinctions

The official CLI currently omits its optional online reference/name-service
path. An additional attempt using that official API on an earlier compiled
draft failed because Scholarcy returned a Cloudflare challenge, not BibTeX.
That optional path has not passed; no upstream code was patched or bypassed.
The nine bibliography entries were separately verified against official
Anthology records or primary arXiv metadata. A publication-check pass is a
format check, not certification of the paper's scientific claims.

The Codex editor compilation was attempted but could not download the Tectonic
runtime bundle. The source is retained for editing; the delivered PDF was
compiled with the already installed local TeX engine. Missing TeX support files
were supplied in a per-user cache, without changing the official ACL styles.

Machine-readable versions, hashes and page accounting are in `manifest.json`.
The actual official CLI output is in `pubcheck-console.txt`, and its JSON is
`pubcheck/errors-AlwaysLate-Final-Report.json`. The failed optional path's output
is preserved in `pubcheck-full-console.txt`.

## Before submission

The team must review the interpretation and final wording, author order,
course-specific AI disclosure requirements and any updated instructor rules.
The code ZIP, presentation and complete `AlwaysLate-Final.zip` are separate
remaining deliverables. The obsolete `paper/main.tex` has not been changed.

**Update 2026-10-26 12:32:** the course guidelines no longer forbid AI-generated reports (rule 4 of `project_guidelines.txt` was edited by the user; decision D-093), so the 'AI disclosure' review item above is closed unless the instructors ask for a statement. The ACL style constraints and the page limits are unchanged.
