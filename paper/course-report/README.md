# Course final report (31 October 2026)

This is a fresh draft of the completed study, not a revision of the obsolete
`../main.tex`. The author names follow the existing project roster; the team
must verify authorship/order and final wording before submission.

The user confirmed on 26 October in the current chat that instructors now
permit AI-written drafts. This supersedes the older prohibition recorded in
`../../project_guidelines.txt` for this report's drafting workflow. This is a
course-report draft, not an assertion of conference acceptance or novelty.

## Requirements

- Course deadline: 31 October 2026.
- Main content: no more than eight pages, excluding references.
- A separate limitations/challenges/future-work section immediately before
  references is outside the course page limit; it must not hide project results.
- Up to two appendix pages.
- The final course ZIP must be named `AlwaysLate-Final.zip` and contain the
  report, code archive and a presentation. This directory prepares the report
  only; the complete ZIP and presentation are not delivered here.

## Files and commands

- `main.tex`: editable manuscript, using official ACL final mode.
- `AlwaysLate-Final-Report.pdf`: the built report.
- `references.bib`: six unmodified official Anthology entries and three
  accurately identified preprints with persistent DOI links.
- `make_figures.py`: report plots from committed source CSVs, not model runs.
- `evidence_map.md`: provenance and qualifications for numerical claims.
- `validation/`: actual checker output, PDF metadata and validation summary.

From this directory:

```bash
# Only needed after modifying plots; requires matplotlib.
~/.venvs/anlp-torch/bin/python make_figures.py
bash build.sh
bash check.sh
```

The build uses installed pdfLaTeX and BibTeX. On this host, a small local
TeX-support cache fills missing files in the otherwise installed distribution;
it is outside the repository. A complete TeX Live installation requires no
such cache. The pinned official CLI runs through `uvx` with its default checks
and no disable flags. Inspect its JSON and console output as well as process
status. Upstream's CLI does not currently enable optional reference/name-service
checks. `check_full.py` also requests those through the official API; that
attempt failed because the external Scholarcy service returned a Cloudflare
challenge instead of BibTeX. No checker code was patched or bypassed.
Bibliography records were separately checked against primary sources. The
optional check uploads the PDF to Scholarcy; only rerun it when that service is
available and the authors want this additional check.

The source is opened in the Codex LaTeX editor. Its bundled compiler could not
download its runtime bundle in this session; the local multi-file build is the
PDF and publication-check input. The local build, not the failed editor
compiler, is the basis of any recorded compilation success.

## Official assets

`acl.sty` and `acl_natbib.bst` are copied without modification from
<https://github.com/acl-org/acl-style-files> at commit
`d5adc823ff0f80f98c80405ca0ab66c68e684409`.
The corresponding template and formatting instructions are retained as
`official-acl_latex.tex` and `official-formatting.md` for reference.

The checker is <https://github.com/acl-org/aclpubcheck>, pinned at
`f626497051df8689e990fac36ba9f42cd630e31c`. Upstream styles and checker
implementation are not modified. No margin, caption, line-spacing or base-font
overrides are used to obtain a pass. A checker pass alone does not certify
scientific accuracy, completeness, or course acceptance.

## Team review still required

Check the paper's interpretation and terminology, author order, bibliography,
AI-use disclosure requirements and any newer instructor instructions. Verify
that every claim is retained with its qualification. A separate future
conference submission needs a fresh novelty review and stronger external
validation; the course report does not promise either.
