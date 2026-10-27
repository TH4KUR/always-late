# Handoff 047: cron jobs stopped, documentation cleanup, single results document (27 October 2026, 14:3x IST)

> A fresh chat with no memory must be able to continue from this file plus the repository. If this file and the repository disagree, the repository wins: say so in the next handoff.

## 0. Read this first (the standing rules)

- **User request of 2026-10-27 (verbatim):** "stop the cron jobs. clean up documentation. reduce it as much as possible. do not tamper with progress docs tho. i want to know exactly how the project progressed. then make a single document with all the results that we got from the start of the projects, their explanations (if found). you say we got null results, but does that also not tell us something? how come we cant decide causality at all. i want all run results so that I can analyse it myself. keep it similar to the other claude chat we had today about the progress. what did the 2nd model check say (if any). what about the downstream task we took? i dont understand why our findings arent conclusive. talk simply. do not use unecessary jargon, keep vocab simple."
- Standing: never remove, shorten or overwrite progress tracking (2026-10-23); HOLD of 2026-10-27 09:50 (no deploy, no Ada jobs, no new experiments until the user says so); mask regime only; HF dataset private; stage explicit paths; times from `date`; never touch `third_party/` or the ACL template files; no secrets in commits.

## 1. Mission in five lines

Compression study of language-specific SAE features (Gemma-2-2B, 27 languages). All science is finished or on hold. This step stopped the cron jobs, shortened the entry pages (old text archived verbatim), and wrote `reports/PROJECT_RESULTS.md`, the one plain-language results document. The only open scientific choice is Q29 (run the built-but-never-run precision audit, or stop). Deliverables are due 31 October.

## 2. State right now

- Cron: **no scheduled jobs** (CronDelete of `955a5cc5`, `e33ac7eb`, `ea207eac`; CronList empty). Ada was not queried or changed. Nothing is queued.
- Tests: `python3 -B -m unittest scripts/g0/tests/test_make_project_results_tables.py` (9 tests, pass; 5 of 5 sensible mutants killed by hand). No other code changed.
- Link check: `python3 -B scripts/g0/check_md_links.py` (stage the new files first; it checks tracked files).

## 3. What this step did

- **`reports/PROJECT_RESULTS.md` (new):** quick answers (why not conclusive; what null results tell us, row by row; why causality is not settled and which part is; second-model check = Llama-3.2-1B "not consistent", Gemma-2-9B with another team and no results here; downstream task = Belebele null with positive control, whole-group and prompt checks); glossary; dated story; study-by-study sections 4.1 to 4.12; what we can and cannot say; data guide (every result path, raw data outside git, HF, wandb); per-language appendix A1 to A10 generated from the committed CSV files by `scripts/g0/make_project_results_tables.py` (rerun: `python3 -B scripts/g0/make_project_results_tables.py --doc reports/PROJECT_RESULTS.md`; test `test_the_committed_document_is_current` fails if the tables drift).
- **Shortened (old text verbatim in `reports/archive/doc_cleanup_2026-10-27/`, sha256 in `manifest.json`):** `README.md` (now leads to PROJECT_RESULTS), `HANDOFF.md` (sections 1 and 2 shortened, 3 to 5 kept with the same numbers because logs cite "HANDOFF section 5"), `RESEARCH_DIRECTION.md` (pointer page), `reports/CURRENT_STATUS.md` (keeps the `## Next action` heading that a protected page links to), `reports/INDEX.md` section 1 (22 rows to 12; a bullet added to section 6).
- **Added only:** an Update line in `reports/functional_validation/cron_prompt.md`; one Log line in `reports/functional_validation/STATUS.md`.
- **Left alone on purpose:** decisions, questions, logs, handoffs, plans and designs, source registers, audits, `reports/archive/`, all result files; the two 26 October publication memos (plans and designs link to them as evidence registers).
- **How much could be reduced:** the live entry pages were already small (about 28 KB). The bulk of the documentation is the progress record (about 1.7 MB), the verbatim archive (about 0.9 MB) and generated result reports (about 0.7 MB), all protected or raw results. So the saving is in what a reader has to open (one results page, one status page), not in bytes.
- No new D-entry: nothing was decided scientifically.

## 4. The next step, exactly

Wait for the user. If they confirm the finalisation plan: (a) the team checks author order and the course AI-use disclosure; (b) build the final archive (`bash scripts/make_final_archive.sh <outdir> paper/presentation/AlwaysLate-Final-Presentation.pdf`); (c) Q29. Do not start jobs or analyses before that.

## 5. Open questions for the user

`DECISIONS_FOR_REVIEW.md`: Q29 (precision audit or stop) and the older open ones listed at its top. Possible follow-up the user may want: publish `reports/PROJECT_RESULTS.md` as a shareable page (not done; asked only for a document).

## 6. Environment and pitfalls

- The results document quotes numbers from committed files; two sources disagree in wording, not in value: the Phase 1 pooled survival (all features pooled: INT6 94.9 %) and the per-language median (INT6 90.4 %, appendix A2). The README of each table says which is which.
- "Average effect 0.704, median language 0.262": heavy tail (Mandarin 7.6, Lao 1.9, Yoruba 1.6). Quote both.
- Explanations in PROJECT_RESULTS are marked tested or guessed; the Belebele "why" is a guess, none of it tested.

## 7. What I would check if something looks wrong

`git status`, the link checker, and `python3 -B -m unittest scripts/g0/tests/test_make_project_results_tables.py`; compare a page against its archived copy in `reports/archive/doc_cleanup_2026-10-27/` (sha256 in the manifest).
