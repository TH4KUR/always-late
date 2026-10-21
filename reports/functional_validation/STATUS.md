# Functional validation across precision: status file (D-039, D-040)

Owner: Claude. Design: [`docs/superpowers/specs/2026-10-21-functional-validation-design.md`](../../docs/superpowers/specs/2026-10-21-functional-validation-design.md). Plan: [`docs/superpowers/plans/2026-10-21-functional-validation.md`](../../docs/superpowers/plans/2026-10-21-functional-validation.md). Cadence: a 40-minute recurring job (prompt and cron expressions in [`cron_prompt.md`](cron_prompt.md); created 2026-10-21, expires after 7 days). Questions for the user: Q17 (and the older Q6, Q7, Q8, Q11, Q12) in `DECISIONS_FOR_REVIEW.md`; none blocks.

Updated: 2026-10-21 22:50 IST.

## State

- D-035 (association retention) is complete and documented; D-039/D-040 select and design the functional study. No GPU job of this study has been queued; nothing runs on Ada (`squeue -u karma` empty at 22:10).
- Spec and plan written and committed (see Log). Belebele (revision `7899cdfa...`, 27 roster languages) is on Ada under `phase1/data/belebele` with a manifest; a local copy is being fetched into `data/belebele/` (ignored by git).
- A dated prior-art check of the exact combination (cross-precision feature-level causal effects; FP-donor patching to localise quantization damage by language) was started in the background; its result goes to `prior_art.md` and the register before G1.

## NEXT STEP

1. **Task 0 remainder:** when the prior-art report arrives, write `reports/functional_validation/prior_art.md`, add register entries (P30 and later, each with verified venue or "unverified"), amend the design's section 9 and any claim that is not novel. Run the Belebele passage-to-FLORES overlap check on Ada (FLORES is there) and write `belebele_flores_overlap.md`.
2. **Task 1 (task layer)**, then Task 2 (intervention library), Task 3 (groups and controls): each test first, mutation-checked, per the plan. No GPU job before gate G1 (plan, Global constraints).
3. Do not start the pilot (Task 5) before Tasks 0 to 4 are complete.

## Gates

- **G1** (before any GPU job): NOT MET (Tasks 1 to 4 and the prior-art file pending).
- **G2** (before any evaluation run on a quantized model or any T-MC scoring): NOT MET (pilot and freeze file pending).

## Log

- 2026-10-21 22:50 IST: design spec, plan, D-040, Q17, this status file and the cron prompt written; Q16/D-039 documentation (Codex's, uncommitted) committed as `ff1522f` after a link and whitespace check. Facts established while designing: the non-selective supported pool of the frozen cohort has only 2,860 features (equal to the saved controls), group sizes of the 8,086 selective features range from 73 (por_Latn) to 1,719 (eng_Latn); TransformerLens 3.9.0 `forward(..., start_at_layer=...)` exists (edit path); Ada env has matplotlib 3.10.9, local venv has none; `/home2` has 5.6 GB free; Belebele has all 27 roster languages (899 or 900 lines each).
