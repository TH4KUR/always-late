# Handoff 039: Task 17 part (b) (Belebele prompt cross-check) is done (D-086): our frozen prompt is not handicapped; every planned job of the project plans is finished and the Ada queue is empty (2026-10-26, 01:30 IST)

> The ONLY source of the previous chat. A fresh chat with no memory must be able to continue from this file plus the repository. If this file and the repository disagree, the repository wins: say so in the next handoff. Start with [`reports/functional_validation/STATUS.md`](../functional_validation/STATUS.md) (NEXT STEP) and [`reports/functional_validation/RESULTS_SUMMARY.md`](../functional_validation/RESULTS_SUMMARY.md) (claims 1 to 16). The standing rules, the user's instructions (Study G, sample-size floor D-075, queue all jobs D-077) and the environment are those of handoffs [035](035-2026-10-25-study-g-main-final-extensions-b-c-running.md) sections 0 and 6 and [038](038-2026-10-26-study-g-extension-b-done-promptcheck-running.md) (pitfalls: quote heredocs, times from `date`/`git log`, read the log head for `CUDA initialization` as well as the `GPU 0:` line, exclude `gnode068` and `gnode069`); nothing changed.

## 1. State right now (verified 2026-10-26 01:30 IST)

- **Ada: the queue is EMPTY.** Every job planned in the project plans has run: the functional study (T-LM, T-MC, F4, rescue, F5, positive controls), Study G (main, calibration, extensions B, C, D) and Task 17 (b). Nothing is queued, nothing is waiting on a decision of the user that blocks a job.
- Git `main`: the commit that carries this handoff is the newest. Tests at the last full run: stdlib 298 OK, torch 1645 OK (2026-10-25 17:22) plus the later modules (each green on its own: extension D figure 18, extension B figure 18, Task 17 wrapper 14, Task 17 analysis 21). Reproduction script: **45 outputs identical, 0 differ** on a fresh run from the input files (the 42 of Study G and the 3 of Task 17 (b)).
- Cron: three session-only entries (`955a5cc5`, `e33ac7eb`, `ea207eac`), expire about 2026-10-29; they fire only while the session is open.

## 2. What this step did

1. `promptcheck-smoke2` 39405 and `promptcheck` 39406 (Task 17 part (b)) COMPLETED on gnode049 (32 min; no CUDA warning; gates passed; complete); sha256 of the four files equal to Ada's; checksums recorded.
2. **Analysis module `phase1_func_promptcheck_analysis.py`**: 21 tests first, 26 mutants killed (two equivalent mutants dropped), committed (`2832860`, 01:15:44) BEFORE it was run on the real data; the rule (margin 0.05, four labels) had been written in the plan before any code.
3. **Result (D-086):** the frozen variant reproduces the stored calibration EXACTLY in 18 of 18 languages; the harness prompt scores lower than ours in all 18 languages: mean difference (harness minus frozen) -0.100 [-0.110, -0.090] = 'ours better'; English 0.516 against 0.596 = -0.080 [-0.132, -0.024] = 'inconclusive' by the rule; 'equivalent' not met (the accuracy level depends on the prompt by about ten points); 'harness prompt better' did not occur, so no Q-item for rerunning the K = all edit with the harness prompt. The Belebele null of the group edits cannot be put down to a weak prompt in this sense; the absolute accuracies are specific to the frozen prompt. RESULTS_SUMMARY claim 16; plan Task 17 closed; outputs `reports/functional_validation/promptcheck/`.

## 3. The next step, exactly

There is no planned job and no pending analysis. In the order of the cron prompt's priorities (d to f) the candidates are, each needing its own written design before any code or job:
1. **Paper-facing material** (priority d): a consolidated results table or figure set for the team's paper from the finished results (the functional-study claims 1 to 16 and Study G claims 12 to 15); the dataviz skill applies; no new numbers.
2. **A question that came out of Study G and is unexplained:** what predicts a language's optimal gain at INT4 (extension D found it reliable, split-half 0.92, but not explained by the group's activation shrinkage): candidate predictors (group size, the group's share of the language's activation, resource level, tokenizer fertility) with a written design first.
3. **Why the harness Belebele prompt scores lower** (a possible small follow-up of D-086): not needed for any claim.
4. Standing: the literature re-check about 2026-10-30 (reports/research_sources.md; search for work after 2026-10-21 on language-selective SAE features under quantization and on feature patching under quantization, Chang et al. EMNLP 2025 and Sathyanarayanan 2609.23125 are the known neighbours); the questions Q21 to Q27 stay open with their defaults (Q27 has an evidence line from D-085); recreate the cron entries if they are gone before the work is finished (they expire about 2026-10-29).
5. If nothing useful can be done, say so in one line and change nothing (cron prompt rule 6): do not invent analyses.

## 4. Open questions for the user

Q21 (HF upload), Q22 (Llama licence), Q23 to Q26, Q27: unanswered, defaults in force (`DECISIONS_FOR_REVIEW.md`).
