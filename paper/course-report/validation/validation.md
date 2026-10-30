# Course-report validation: revision 4, 30 October 2026

The completed-study report builds successfully with the existing local
pdfLaTeX/BibTeX `build.sh`. Its final PDF has **9 pages**: main content and the
separate limitations section end on page 6, references are on page 7, and the
appendix is pages 8–9. This satisfies the eight-page main-content and two-page
appendix limits. The four synthesis questions and completed GPTQ, measurement,
continuous prediction and reviewed 9B results appear in the main content.

`bash check.sh` completed on the final PDF with the pinned official ACL checker:
**All Clear!** Its generated JSON is `{}`. No CLI checks were disabled and no
upstream checker or official ACL asset was modified. The style hashes still
match the preserved revision-3 manifest. The official template copies were not
edited.

The final log has no undefined references/citations or overfull horizontal
boxes. Standard reference-column balancing emits a 2.95218 pt vertical-box
warning. The rendered reference page has no visible clipping, and the official
checker records no margin-bleed finding. All fonts are embedded Type 1 fonts,
with no Type 3 fonts. PDF metadata and fonts are retained in `pdfinfo.txt` and
`fonts.txt`. Every final page was rendered and inspected; images and a contact
sheet are retained in `rendered-rev4/`.

The evidence map tracks the distinct endpoints and uncertainty schemes. The
revision reports the positive 3.1–4.4% survival-prediction gains while preserving
the unmet 10% convention. It keeps the fresh-sentence, legacy-only Extension C
benefit alongside the failed main gain criteria. Corrections distinguish the
three frozen K64 evaluation control draws from five stored draws, the stricter
selectivity cutoff from the activation-correlation cutoff, and nonselective
positive controls from other-language selective controls. Belebele loss remains
in nats/item, separate from accuracy. No new inference or statistical analysis
was performed for the manuscript revision.

The optional online reference/name-service checker path was **not rerun**. An
earlier attempt failed when Scholarcy returned a Cloudflare challenge; the
preserved record remains incomplete. The official default CLI pass is the
format-check claim. The prior editor-runtime download failure is historical;
this existing multi-file manuscript uses its working local build. Existing
bibliography records were preserved; the GPTQ entry and the Gao evaluation
metadata were checked on primary arXiv pages on 30 October 2026. The final field-context edit adds Leask, Makelov and SAEBench using the official-proceedings-verified entries recorded in `reports/publication_context_2026-10-30.md`, followed by another successful build and official check. No exhaustive
new literature audit or acceptance claim is implied.

The exact previous manuscript, PDF, README, evidence map, bibliography and
validation records are preserved under
`reports/archive/reporting_revision_2026-10-29/paper/course-report/`. The manifest
retains previous revision records and hashes. The author roster is unchanged.
A submitted course ZIP is not produced or certified by this check.
