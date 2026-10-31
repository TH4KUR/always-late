# Course final report

- [`main.pdf`](main.pdf): the final report (ACL format, 12 pages: 8 main pages, limitations, references, 2 appendix pages).
- [`main.tex`](main.tex), [`references.bib`](references.bib), [`figures/`](figures/): source. The report uses `fig-harm`, `fig-dose`, `fig-dissociation`, `fig-terms` (main text) and `fig-tiers` (appendix), all drawn by `make_figures.py` from committed result files and all for the study's own quantizer (legacy RTN); matched RTN and GPTQ appear in the appendix table. `python make_figures.py --dark` draws the dark slide versions into `../slides/figures/`. The other files in `figures/` are earlier plots kept for reference.
- [`evidence_map.md`](evidence_map.md): maps each claim and number in the report to its result file.

## Build and checks

```bash
bash build.sh        # pdfLaTeX + BibTeX into build/, copies the result to main.pdf
bash check.sh        # official ACL checker (aclpubcheck, pinned); fails on any finding
```

`acl.sty` and `acl_natbib.bst` are the unmodified official ACL style files.
