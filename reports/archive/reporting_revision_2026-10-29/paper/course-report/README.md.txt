# Course final report (31 October 2026)

This is a fresh draft of the completed study, not a revision of the obsolete
`../main.tex`. The author names follow the existing project roster; the team
must verify authorship/order and final wording before submission.

The course guidelines changed on 26 October 2026: rule 4 of
`../../project_guidelines.txt` now reads only 'You are allowed to use AI and
other internet resources for implementation and code', and the sentence that
forbade AI-generated reports was removed (the user's edit; decision D-093).
The report and the paper may therefore be written and updated by the AI
assistant; the earlier note in this file that the permission came from a chat
statement is superseded. This is a course-report draft, not an assertion of
conference acceptance or novelty.

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
- `references.bib`: six unmodified official Anthology entries and nine
  accurately identified preprints with persistent DOI links (checked against
  their arXiv pages).
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

## Revision history

- **Revision 1 (26 October 2026, 03:16):** first complete draft.
- **Revision 2 (26 October 2026, at the user's request):** the text was
  updated to the finished results and made a standalone record: new Section 8
  (second-model check on Llama-3.2-1B, pre-registered verdict "not
  consistent"), a table of the pre-registered rules and outcomes of Study G and
  its extensions and a result inventory in the appendix, software, hardware and
  links, six additional verified background references, and a fixed axis label
  of Figure 2. The ACL style files were re-downloaded from the official
  repository (same upstream commit `d5adc82...`, byte-identical to the copies
  here) and are untouched. The official checker was run again (pinned and
  newest upstream version): `All Clear!` with an empty JSON report; a
  deliberately non-ACL PDF was also run as a negative control and the checker
  flagged it (page size and font), so the check is not vacuous.
- **Revision 3 (26 October 2026, at the user's request, AI-report rule lifted,
  D-093):** the report was rewritten around the intervention-transfer study
  (new Section 7: crossed prefix/suffix decomposition at the group and the
  feature level, averaged fingerprints, prediction of the transfer error,
  claim rules fixed before the outcomes, main matrix analysed once, D-094).
  `main.tex` is the only source (`draft/` holds the working copies of the
  rewrite, see `draft/README.md`). New: `make_transfer_figures.py` (Figures 1
  and 2 from `reports/intervention_transfer/results/main/*.csv`, run with
  `~/.venvs/anlp-torch/bin/python`), the reference Vaidyanathan 2026, an
  appendix table of the pre-specified rules and outcomes with the transfer
  rows. After an independent fact-check against the committed results the
  wording was corrected (no number changed; D-094 Update, commit `542c246`).
  Update 2026-10-27: the six-language Wikipedia panel (external-panel
  paragraph, one abstract sentence, one appendix row, the cross-corpus
  numbers; D-095) was added after its six runs were validated and analysed once
  (`reports/intervention_transfer/results/external/`); `make_transfer_figures.py`
  also renders `figures/transfer-*-ext.*` for that panel. Official checker: `All Clear!`
  (11 pages: the body ends at the top of page 8, then the limitations section,
  references on page 9, two appendix pages 10 and 11). Revision 2 is in the
  committed history.

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

Check the paper's interpretation and terminology, author order, bibliography
and any newer instructor instructions (the guidelines no longer restrict
AI-written reports and name no disclosure statement; add one if the
instructors ask for it). Verify
that every claim is retained with its qualification. A separate future
conference submission needs a fresh novelty review and stronger external
validation; the course report does not promise either.
