# Association-retention study: status file (D-035)

Owner: Claude. Plan: [`docs/superpowers/plans/2026-10-21-language-association-retention.md`](../../docs/superpowers/plans/2026-10-21-language-association-retention.md) (Tasks 1-6 and its "Verification amendments"); design: [`reports/survival_method_reassessment.md` section 7](../survival_method_reassessment.md#7-selected-next-study-language-association-retention-d-035); plan check: [`plan_verification.md`](plan_verification.md); sources: [`reports/research_sources.md`](../research_sources.md).

Updated: 2026-10-21 (after removing the communication folder, D-037).

## State

- Plan verified (D-036): statements about our code and results correct; study not found in prior work (not a novelty claim); register corrected and P20-P29 added.
- Communication with Codex ended on the user's instruction (D-037); the `communication/` folder is removed (recoverable from commit `58c3fff`; its still-useful points are in the appendix of `plan_verification.md`). Shared documents may be edited by Claude as needed; check `git status` before every commit in case Codex edits on her own, and stage explicit paths.
- Hugging Face dataset `karma-skz/always-late-artifacts`: private by the user's decision; the user will make it public later; Claude only needs access (verified from Ada, 277 files, 2026-10-21).
- Implementation: Tasks 1 and 3 complete (see the Log). Ada: `squeue` empty; `/home2` 7.9 GB free; no job queued.

## NEXT STEP (the recurring job does exactly one step per firing, then rewrites this block)

1. Read this file, `git status --short` and `git log --oneline -5`.
2. **Task 2 (plan): the optional per-condition collector.** Source: plan Task 2 and design section 7. Read `scripts/g0/phase1_run.py` (`run_survival`), `scripts/g0/tests_torch/test_phase1_run.py` (fake causal model fixtures) and `phase1_lib.py` first. Tests first in `scripts/g0/tests_torch/test_phase1_association_collect.py`: planted cases where reference and quantized responses differ (Hindi-only becomes Hindi-plus-English; a feature switches language; a feature becomes inactive) and the saved quantized counts and sums match a brute-force calculation and differ from the reference arrays; FP equals INT0; condition restoration; exact count conservation between sentence shards and aggregates; equal-position versus all-token weighting; positive criterion `z > 0`; unequal sentence lengths; empty selected sentence rows; BOS and padding exclusion; batch-size invariance; missing conditions or languages; duplicate sentence ids. Then implement `profile_sink=None` in `run_survival` (default path byte-identical), `scripts/g0/phase1_association_collect.py` (sink, sidecar schema from the inventory module's contract, shards), and the CLI options `--association-profiles` and `--association-baseline`. Estimate bytes per condition and sink memory on the fixture. Ruling already taken: sentence-shard format (dense int16 versus sparse) is decided by measurement in the smoke, so implement an interface that hides it.
3. Then Task 4 (paired bootstrap analysis; sources M01-M05), Task 5 (smoke, gates, GPU), Task 6.

## Gates before any GPU job (all must be recorded here as passed)

- Task 2 tests and the existing `test_phase1_run.py` / `test_phase1_budget.py` pass, collection-off outputs unchanged.
- Measured bytes per condition for sentence shards (smoke extrapolation) times nine conditions times the number of streams is below the free space on Ada (7.9 GB at 2026-10-21) with a stated margin.
- Peak GPU memory with the sink is below the card size (previous run 9.6 of 11 GB).

## Questions for the user (do not block on them)

- Licence of our own statistics in the dataset card (currently `license: other`, none chosen).
- Q6, Q7, Q8, Q11, Q12 remain unanswered in `DECISIONS_FOR_REVIEW.md`; not blocking.

## Log

- 2026-10-21: verification report, HF card commit `43026509` (private; scan clean), Codex's work committed (`e05db89`) and aligned (`58c3fff`).
- 2026-10-21 (later): communication folder removed; carried-over points in the appendix of `plan_verification.md`; D-037.
- 2026-10-21 (Task 1 complete): `scripts/g0/phase1_association_inventory.py` with 19 stdlib tests and 12 torch tests (RED first: no module; then green; 14 of 14 real mutants killed, the inert control survived); full suites green (stdlib 180, torch 280); generated `artifact_inventory.md` and `frozen_cohort_manifest.json` from the original mask/eq files (byte-identical to Ada's). Verdicts: FP profiles, all-language correlations and the cohort are available; the S-vs-nu comparison and the FP baseline are derivable (6,663 of 8,086 established, dev ties 0); quantized profiles, quantized peaks and sentence-level summaries need the Task 2 collector. Ruling: the plan puts the inventory tests in the stdlib directory but loading `.pt` files needs torch, so the pure logic (on file records) is stdlib-tested and the loader/manifest/CLI are torch-tested; cost if wrong: none (same coverage). Ruling: the work continues on `main` (the user's standing practice: commit and push after each green step; no worktree).
- 2026-10-21: method sources M01-M05 (bootstrap resampling of aligned sentences) added to the register.
- 2026-10-21 (Task 3 complete): `scripts/g0/phase1_association_metrics.py` (`association_metrics`, `compare_profiles`, `association_status`; pure NumPy) with 27 tests in `scripts/g0/tests_torch/test_phase1_association_metrics.py`; RED first (no module), green, 21 of 21 mutants killed (one mutant survived at first, `peak_is_original`, and a test was added); suites green (stdlib 180, torch 307). Definitions and tolerances are documented in the module docstring (tie tolerance 1e-12 for ties only; support defaults 50 positives and panel mean rate 1e-5 reuse the tagging defaults). Ruling: positives at a tied peak are the largest count among the tied languages; for unequal token counts the peak is decided on rates and support on counts.
