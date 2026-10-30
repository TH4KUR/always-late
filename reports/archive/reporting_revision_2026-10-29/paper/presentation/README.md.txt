# Viva presentation (course deadline 2026-10-31)

`slides.tex` is a 16:9 deck, one page per slide, built with plain `pdflatex` (no beamer is installed here): `bash build.sh` builds `AlwaysLate-Final-Presentation.pdf` (the PDF and `build/` are not tracked until the final version is added with `git add -f`).

Slides marked `\pending{...}` are result slides whose numbers do not exist yet; `bash build.sh final` refuses to build while any is left. Numbers on the background slides come from `reports/functional_validation/RESULTS_SUMMARY.md`; numbers on result slides only from the committed outputs of `scripts/g0/phase1_transfer_report.py` and `reports/intervention_transfer/STATUS.md`. The author line follows the report's roster and must be verified by the team before submission.

The submission archive is built by `bash scripts/make_final_archive.sh <outdir> paper/presentation/AlwaysLate-Final-Presentation.pdf`.
