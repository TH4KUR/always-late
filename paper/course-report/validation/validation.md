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
