# Handoff 042: wandb Option A done (D-091), course report revision 2 done, cron no longer deals with 9B (D-092) (2026-10-26, 09:16 IST)

> The ONLY source of the previous chat. A fresh chat with no memory must be able to continue from this file plus the repository. If this file and the repository disagree, the repository wins: say so in the next handoff. Start with [`reports/functional_validation/STATUS.md`](../functional_validation/STATUS.md) (NEXT STEP) and the root [`HANDOFF.md`](../../HANDOFF.md). The standing rules, the user's instructions and the environment pitfalls are those of handoffs [035](035-2026-10-25-study-g-main-final-extensions-b-c-running.md), [038](038-2026-10-26-study-g-extension-b-done-promptcheck-running.md), [039](039-2026-10-26-task17-prompt-check-done-all-queues-empty.md) and [041](041-2026-10-26-llama-check-done-not-consistent-intervention-transfer-next.md); two things changed: the course report may be edited (below) and the cron job does not deal with Gemma-2-9B.

## 1. State right now (verified 2026-10-26 09:15 IST)

- **Ada: the queue is EMPTY.** No job is planned until the intervention-transfer study (D-088) reaches its first job.
- Git `main` is pushed up to the commit that carries this handoff. The remote branch `gemma9b-confirmation` was fast-forwarded to `19904ca` (the user's chat request; no force); it does not have the later commits, and the cron job does not touch it (D-092).
- Tests: the Llama modules 98 green (this session); the wandb uploader has 66 tests and 67 mutants (one equivalent survivor); the full stdlib and torch suites were not re-run in this step (last full run 2026-10-25: stdlib 298, torch 1645, plus later modules each green).
- Cron: three session-only entries (`955a5cc5`, `e33ac7eb`, `ea207eac`), expire about 2026-10-29.

## 2. What happened since handoff 041

1. **wandb Option A (user: 'take option A'):** design [`reports/wandb_schema_v2.md`](../wandb_schema_v2.md) written before code; `scripts/g0/wandb_upload_v2.py` (pure payload builder with guards, uploader on an injected module); 15 new runs `v2-*` in `karma-skz-/anlp-project` (34 with the 19 old ones, all finished, verified through the API); run URLs in section 8 of the design. Nothing was deleted on wandb. Five runs hold duplicated rows or figures (two bugs of the first passes, fixed in the code): a clean re-upload is one command if the user asks. Decision D-091.
2. **Course report (`paper/course-report/`), revision 2 (user: 'write the paper according to updated results ... a standalone record', real aclpubcheck, template untouched):** Llama second-model section, rules table and result inventory in the appendix, software and links, six verified background references, a fixed figure label; an independent fact-check by a separate agent was applied (the Llama verdict is attributed to the failed P1; 'pre-specified' instead of 'pre-registered'; a mismatched AUROC pair; extension D rows; status and scope qualifiers). The official checker is `All Clear!` for the final PDF with the pinned (`f626497`) and the newest version; a margin finding of an intermediate state was fixed; a non-ACL PDF is flagged (negative control); the ACL style files are byte-identical to a fresh download of the official repository. 12 pages: body 8, limitations 9, references 10, appendix 11-12. Commits `4357829`, `19904ca` and the one with this handoff. **This lifts 'never edit paper/' for `paper/course-report/` only** (the obsolete `paper/main.tex`, `acl.sty` and `acl_natbib.bst` stay untouched). The user waived the course's no-AI-report rule in chat; the README keeps a note that the team must verify the disclosure requirement.
3. **D-092:** the cron job does not deal with the 9B runs at all; the intervention-transfer study is Gemma-2-2B only.
4. Message for the 9B person (given to the user in chat): pull the branch, same task as handoff 040, add R6 (median of the per-language retention ratios), check the tagging statistics first, order the jobs so P1 comes early, branch rules unchanged.

## 3. The next step, exactly

1. **Intervention-transfer study (D-088), Gemma-2-2B only, Task 1 (freeze)** as in the STATUS NEXT STEP item 1: read the plan and the verification file end to end, write the Task 1 design note in the new ledger `reports/intervention_transfer/STATUS.md`, apply the corrections, tests first; `freeze.json` committed before any quantized intervention output is read.
2. Keep the course report in step with new results (STATUS NEXT STEP item 2: rebuild, both checkers, body within 8 pages and appendix within 2, refresh the manifest). Deadline of the course: 2026-10-31.
3. Standing: literature re-check about 2026-10-30; Q21 to Q28 open with their defaults; recreate the cron entries if they are gone before the work is finished.

## 4. Open questions for the user

Q21, Q23, Q24, Q27, Q28: unanswered, defaults in force. The wandb choice is answered (Option A). The team still has to verify author order and the course's AI-use disclosure for the report; the course ZIP (`AlwaysLate-Final.zip` with the code archive and a presentation) is not prepared here.

## 5. Pitfalls found in this step

- A shell `cd` does not persist between calls; use absolute paths or `cd` in the same call.
- Times in documents must come from `date`: four entries written ahead of the clock were corrected at the end of this step.
- Floating appendix tables in the two-column ACL layout: a `table*` is placed only at the top of a following page; a float page that is too tall makes text bleed into the bottom margin, which the official checker reports (it did once). Rebuild and rerun the checker after every layout-relevant edit; keep the combined appendix table short enough.
