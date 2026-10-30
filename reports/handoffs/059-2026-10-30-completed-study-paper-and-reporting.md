# Handoff 059: completed-study paper and reporting revision (30 October 2026)

## User direction

The user asked for all documentation to report the completed work truthfully and proportionately, acknowledge current mechanistic-interpretability limitations, avoid treating finite model/SAE coverage as invalidating controlled results, revise the final research questions and write a publication-quality paper. Gemma-2-9B GPTQ remains the one pending planned extension. This request did not authorize new compute or restarting recurring jobs.

## Completed work

- Four retrospective synthesis questions and quantitative answers: [`../FINAL_RESEARCH_QUESTIONS.md`](../FINAL_RESEARCH_QUESTIONS.md). They organize the completed evidence and preserve the original frozen hypotheses/verdicts.
- New research manuscript: [`../../paper/research-paper/main.tex`](../../paper/research-paper/main.tex), compiled as `Feature-Survival-and-Intervention-Preservation.pdf`; 11 pages, with main content ending on page 8. Its evidence map and primary-source context distinguish actual validity issues from generalization boundaries.
- Revised full evidence narrative, active documentation and result summaries. Positive group effects, recognizable effect identity, R1's modest positive prediction, the tiny fresh-sentence legacy fine-gain result and all task nulls remain visible. Misstated Belebele units, K64 ranking/control counts and positive-control populations were corrected.
- Course report revision 4: 9 pages (body plus limitations 6, references 7, appendix 8–9). Presentation: 21 slides, aligned with the completed evidence.
- Exact pre-edit snapshots: [`../archive/reporting_revision_2026-10-29/REVISION_RECORD.md`](../archive/reporting_revision_2026-10-29/REVISION_RECORD.md). The archive preserves the existing full-report working-copy typo as part of its original bytes; current prose is corrected.

## Verification

- Research paper: local pdfLaTeX/BibTeX build succeeds; pinned and newest official ACL checkers both return **All Clear**, empty findings JSON; no overfull boxes or undefined references/citations. All 11 pages rendered and inspected.
- Course report: final local build and pinned official checker succeed, **All Clear**, empty findings JSON; all final pages inspected. The small standard reference-balancing vertical-box warning is recorded without visible bleed.
- Presentation: final-mode build succeeds, no pending-result boxes, no undefined-reference or overfull-box warnings; all 21 rendered slides inspected.
- `python3 -B scripts/g0/check_md_links.py`: zero broken links at the integrated check. `git diff --check`: clean.
- Original CSV/raw-weight/freeze files have no tracked modifications. Generated result values and frozen rules are not replaced by the final synthesis. No new experiment or statistical analysis was run; figure generation reads stored estimates and intervals.
- The native editor's Tectonic bundle download failed. Local multi-file compilation is the basis for PDF/build claims; source and compiled manuscript were opened in Codex.

## Workspace and next use

The revision is local, on the existing `main` checkout based on `f45cb03`; no commit or push was made. Pre-existing edits were preserved before integration. No cluster queue was inspected live, no job was submitted, and no dataset or wandb upload was performed. No recurring task was created.

The paper can now be reviewed for author-approved wording and venue requirements. The 9B GPTQ implementation remains prepared on its separate branch and supplies no result. Missing matched-RTN 9B, wider dictionary/model sweeps and the precision audit are future extensions, not requirements for acknowledging the completed study's controlled findings. Do not rewrite the original study rules or count the circular/reversed 9B criteria as independent confirmations.
