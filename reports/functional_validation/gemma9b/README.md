> **Current reporting pointer, 31 October 2026:** The original-RTN 9B analysis has been reproduced byte for byte from raw per-state files and the review is complete (D-127). The three-arm confirmation (legacy RTN, matched RTN, GPTQ) was completed on a Jarvis Labs RTX PRO 6000 on 2026-10-31 — tier 1 **consistent**, tier 2 **consistent**. Start with [REVIEWED_SUMMARY.md](REVIEWED_SUMMARY.md) for the reviewed legacy-only analysis, and [the three-arm comparison report](../../../out/analysis/g9-compare/report.md) for the full three-arm results.

# Gemma-2-9B confirmation study

Imported 2026-10-29 at the user's request (D-125). The three-arm confirmation was run 2026-10-31 on branch `gemma9b-confirmation` and merged to main.

## Branches and status

- **`gemma9b-confirmation`** was the single maintained 9B branch, now merged to main.
- **Completed:** original-RTN functional validation and Study G (commit `0c3d3c7`), plus three-arm GPTQ/matched-RTN confirmation (commit `49da7e5`).
- All predictions met for both tiers: Q0–Q3 (tier 1) and S1–S4 (tier 2).

## Files and status

- [Pushed verdict](verdict.md): source commit `0c3d3c7`, preserved from consolidation commit `34c6d11`. It reports that all five earlier confirmation predictions passed; independent reproduction is pending.
- [Execution freeze](legacy_rtn_run_freeze_2026-10-29.json): the completed run's original freeze, byte for byte.
- [Three-arm amendment](amendment_A1_three_arm.json): predictions Q0–Q3 and S1–S4.
- [Detailed analysis](results/analysis/report.md), [analysis JSON](results/analysis/analysis.json), [curves](results/analysis/curves.csv), [reliability](results/analysis/reliability.csv), [best gain factors](results/analysis/alpha_star.csv), [run metadata](results/meta_lm_eval.json), [feature groups](results/manifest.json), [tagging metadata](results/eq/meta.json).
- [Import provenance](import_provenance.json): source commit, artifact revision and SHA-256 for every copied file.
- [Review queue](REVIEW_STATUS.md): the existing Claude cron's next task.
- [Artifact hashes](artifact_hashes.json): SHA-256 of GPTQ and matched-RTN weight sets at bits 4, 6, 7.

## Key numbers (three-arm, 2026-10-31)

- **Q1 amplification ratio:** legacy 1.689, matched 1.490, GPTQ 1.165 (GPTQ below legacy in 26/27 languages)
- **S1 INT4 survival:** legacy 22.3%, matched 32.9%, GPTQ 46.6% (GPTQ above legacy in 27/27 languages)
- **S2 INT4 loss:** legacy 0.410, matched 0.287, GPTQ 0.088 nats/token
- **S3 tier gap at INT6:** 9.1 / 6.3 / 1.7 points (legacy / matched / GPTQ)

## Review notes

The Study G analysis was reproduced byte-identically from the raw per-state files (task 2 of the review). Open problems: anomaly A-043 (P3a ordering, P3b relabelling, mean vs median retention). The reviewed statement with intervals is in [`REVIEWED_SUMMARY.md`](REVIEWED_SUMMARY.md) (decision D-127).
