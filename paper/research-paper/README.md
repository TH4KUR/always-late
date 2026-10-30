# Research manuscript: feature survival and intervention preservation

Readability revision, 30 October 2026 from completed evidence through 29 October. The manuscript presents the study as a controlled empirical contribution to multilingual quantization and mechanistic interpretability. It reports positive intervention effects, modest prediction gains, task nulls and unresolved comparisons together. It explains the measures before using them, puts findings before qualifications, and has a dedicated section on the missing-BOS problem found in the earlier Gemma evaluation. The protocol correction is explicit. SAE features may mix concepts or leave information unexplained; the paired edit results remain reportable.

- [`main.tex`](main.tex): editable manuscript in the unmodified official ACL style; no target venue or acceptance is implied.
- [`Feature-Survival-and-Intervention-Preservation.pdf`](Feature-Survival-and-Intervention-Preservation.pdf): compiled manuscript; current build and page count are recorded in [`validation.md`](validation.md).
- [`evidence_map.md`](evidence_map.md): numerical provenance, uncertainty and evidence status.
- [`../../reports/FINAL_RESEARCH_QUESTIONS.md`](../../reports/FINAL_RESEARCH_QUESTIONS.md): canonical retrospective synthesis questions and answers.
- [`make_figures.py`](make_figures.py): plots stored estimates and existing intervals, with no new analysis.

The bibliography and official style assets are local copies so the manuscript builds as an ordinary multi-file LaTeX project. The course submission has its own constrained report under [`../course-report/`](../course-report/README.md); the old root `paper/main.tex` remains a historical manuscript for the retired pipeline.

Build from this directory with `bash build.sh`. Regenerate figures with the project's matplotlib-enabled Python before building if their source tables change. The evidence map and build record document the actual checks, rather than treating compilation as scientific verification.

Gemma-2-9B GPTQ is pending and contributes no result. The study is otherwise treated as complete. Broader model, dictionary and deployment evaluations are potential extensions, not conditions for acknowledging the controlled findings already obtained. Raw data remain on a private artifact dataset; the paper does not claim unrestricted public availability.
