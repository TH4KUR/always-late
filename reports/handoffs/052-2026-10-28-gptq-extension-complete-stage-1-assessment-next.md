# Handoff 052: GPTQ extension complete (all jobs analysed, HF uploaded); the plan's Stage 1 assessment is next (28 October 2026, 2026-10-28 19:56 IST)

> The ONLY source of the previous chat. A fresh chat with no memory must be able to continue from this file plus the repository. If this file and the repository disagree, the repository wins: say so in the next handoff.

## 0. Read this first (the standing rules)

[Handoff 050](050-2026-10-28-gptq-stages-1-3-and-f5-done-study-g-external-running.md) section 0 and [051](051-2026-10-28-gptq-study-g-done-external-panel-running.md) still apply (the user's request of 2026-10-27 verbatim in 050; main branch only; hash-pinned entry points never edited; progress tracking never removed; times from `date`; no secrets or weights in commits; Q29 not started). **New:** on 2026-10-28 the user wrote in the main chat: "trust what codex added, the plan is fine. you can run a validity check if thats correct or not, and include that in your plans. i also allow you to commit and push codex's work after a validity check" (D-110): Codex's plan `docs/superpowers/plans/2026-10-28-survival-and-intervention-publication-rescue.md` and the USER DIRECTION block of `reports/gptq_extension/cron_prompt.md` are in force, in the order the block gives; the validity gates added to the plan (D-110 and the plan's dated update) are mandatory steps.

## 1. State right now

- **The GPTQ extension's experiments are all validated and analysed (three arms: legacy RTN, matched RTN, GPTQ):** Phase 1 (D-104, D-106), transfer main (D-105), loss task, rescue, Belebele (D-107), F5 (D-108), Study G (D-109), external panel (D-111). Plain-language record: `reports/gptq_extension/RESULTS.md` (sections 4.1 to 4.8); per-language tables A11 to A16 in the generated appendix of `reports/PROJECT_RESULTS.md`. E3 and F4 were not repeated (disk; F4 needs per-arm association data).
- **Raw outputs are on the private Hugging Face dataset** `karma-skz/always-late-artifacts`, `runs/gptq-extension/`: commits `3cce3514c0` (part 1) and `7e4cf26ceb` (part 2), 1,011 files, 820 MB, no weights; root manifest and card untouched (`reports/artifacts.md`).
- **Codex's four files are committed** (`87b779a`), the F5 input inventory of the plan's Stage 0 is written (`reports/survival_effect_rescue/f5_input_inventory_2026-10-28.md`). Ada: nothing queued or running.
- **Cron:** three recurring entries (`bf1ae581`, `ad8593e3`, `8db9afdb`), a firing every 40 minutes, expire about 2026-11-03.

## 2. Headline of the extension (for the assessment)

Better quantizer, same direction, smaller effect: 4-bit survival 12.7 / 18.7 / 30.5 % (legacy / matched / GPTQ), loss increase 0.590 / 0.385 / 0.130; transfer cosine on fresh text at INT4 0.754 / 0.808 / 0.876; Study G INT4 amplification 1.92 / 1.67 / 1.11 times full precision; Phase 1 T1 to T3 verdict flips are fragile and tiny (D-106); verdict words that change in the transfer family sit at thresholds (D-105, D-111: UTILITY met / met / not met on the external panel); F5: the retention label does not predict function (AUROC 0.49 under GPTQ); the continuous R1 versus W1 Spearman is moderate (INT6 0.575, INT4 0.381 under GPTQ). Wording correction carried everywhere: the group patch recovers 2.0 to 2.6 % of the loss gap (a recovery share, not a share of causal responsibility).

## 3. The next step, exactly

Plan Stage 1: write `reports/survival_effect_rescue/all_results_assessment_2026-10-28.md` (see the ledger's NEXT STEP for the required content) and a D-entry (D-112), then the decision gate (stop and say so plainly if the F5 data cannot answer the plan's question or the project has no defensible result to build on), then Stage 2 (dated amendments of the two forward-looking plans), then Stages 3 and 4 on existing Gemma data after the validity gates, then Stage 5 (Llama F5) only if every gate of the plan's update passes. Deadline of the course deliverables: 31 October 2026.

## 4. Open questions for the user

Q32 (wandb for the extension: default build a compact set after Stage 1 if time remains, or skip); Q30 (disk, drop order; E3 and F4 not repeated); Q29 not started. Whether the GPTQ findings enter the course report is the user's call.

## 5. Pitfalls

- Never run `hf_artifacts.py upload` without `--manifest-name` and `--no-card` for a new spec: the default would replace the root manifest and card.
- Study G analysis needs a `meta_lm_eval.json` in every run directory; F5 analysis takes the old `legacy_f5_a` (floor) plus the job directory.
- The group patch numbers (2 to 3 %) are recovery shares; never write "share of the damage".
- Heavy analyses one at a time (about 4 to 5 GB free RAM locally).
