# Handoff 037: STUDY G extension D (optimal gain against the quantization shrinkage) is designed, run and written up (D-083, D-084: not supported); extension B's GPU jobs are still running (2026-10-25, 17:25 IST)

> The ONLY source of the previous chat. A fresh chat with no memory must be able to continue from this file plus the repository. If this file and the repository disagree, the repository wins: say so in the next handoff. **Start with [`reports/functional_validation/gain_study/README.md`](../functional_validation/gain_study/README.md).** The standing rules, the user's instructions (Study G, sample-size floor D-075, queue all jobs D-077) and the environment are those of [handoff 035](035-2026-10-25-study-g-main-final-extensions-b-c-running.md) sections 0 and 6 and [handoff 036](036-2026-10-25-study-g-extension-c-done-extension-b-running.md); nothing in them changed. **Pitfall found this session: do not write a time you did not read from `date` or `git log` (I wrote 17:20 for a design that was committed at 17:12:42 and had to add a time note to D-084).**

## 1. State right now (verified 2026-10-25 17:21 IST)

- Git `main`: this step's commits `bb1b131` (design), `7f3e865` (module) and the commit that carries this handoff. Tests: stdlib 298 OK; torch suite **1645 OK (122 s) at 17:22**; reproduction script with the extension D block: **35 outputs identical, 0 differ** (the 32 of handoff 036 and the 3 of extension D).
- **Ada:** `gain-sd-4` 38816 RUNNING (gnode061, 2 h 42 min of about 3 h; ends about 17:50), `gain-sd-FP` 38817 RUNNING (gnode057, 45 min; ends about 19:40); PENDING behind the scheduler limit: `gain-sd-6` 38818, `gain-sd-FPref-4` 38869, `gain-sd-FPref-6` 38870. Everything is queued; nothing to submit.
- Cron: three session-only entries (`955a5cc5`, `e33ac7eb`, `ea207eac`), expire about 2026-10-29.

## 2. What this step did

1. **Extension D designed first** (spec section 12, D-083, plan D1 to D3; commit `bb1b131` 17:12:42): post hoc, exploratory, CPU only: per language the optimal gain at INT4 on the `gainc-4b` data (grid: the ratio among 1, 1.1, 1.2, 1.4, 1.7 with the lowest group harm; smooth: vertex of a quadratic in ln ratio through the origin), rank correlation with 1 / g_ratio (baseline G1) with a one-sided permutation p (10,000, seed 0); verdict supported (rho >= 0.30, p < 0.05, split-half reliability >= 0.30) / uninformative (reliability < 0.30) / not supported; descriptive: smooth estimator, placebo target 1 / controls_ratio, split-half and the corrected correlation.
2. **`scripts/g0/phase1_gain_extd.py`** with 34 tests written first, 36 mutants killed (commit `7f3e865`).
3. **Result (D-084): R-D1 NOT SUPPORTED.** rho = -0.170, p = 0.80; split-half reliability 0.917 (halves correlate 0.846): the optimum is a reliable property of a language but is not ordered by the group's INT4 activation shrinkage; placebo +0.186; the median optimum (1.157) matches the median of 1 / g (1.174), the order of the languages does not. The compensation story of D-082 is not supported at the language level. Outputs `gain_study/extension_d/` (report.md, analysis.json, alpha.csv).
4. Documents: spec section 12, D-083, D-084, README row and update, `RESULTS_SUMMARY.md` claim 14, plan D1/D2 done and D3 partly, `reproduce_functional.sh` extension D block, STATUS NEXT STEP update and Log.

## 3. The next step, exactly

1. **Extension B jobs** (unchanged from handoff 036 section 4): as each `gain-sd-*` job ends, validate (sacct COMPLETED; `GPU 0:` = RTX 2080 Ti; `split eval sentences per language 300`; gates passed; 999 configurations; `meta_lm_eval.json` complete and nothing skipped; sha256 on Ada equal to the local one after one `scp` per file; ids equal to the freeze; manifest hashes equal to `gain_study/state_defined/meta.json`), fetch into the scratchpad (job directories on Ada: `/home2/karma/duan-verify/out/phase1/<job>/`, logs `/home2/karma/duan-verify/logs/<job>-<id>.out`), checksums into `artifact_checksums.md`. When all five are in: run `phase1_gain_extb.py` and `phase1_gain_extb_figures.py` (commands in handoff 036), look at the figure, **append the extension B verdict as the NEXT FREE D-number (D-085 now)**, README, `RESULTS_SUMMARY.md` claim 15, `reproduce_functional.sh` block, REPRODUCE.md, checksums.
2. **Extension D figure** (plan D3): alpha (grid, with a small deterministic vertical jitter, and smooth) against 1 / g per language with the line alpha = 1 / g, tests first, mutation check, look at the render in both themes; add the svg to the reproduction script.
3. If a further exploratory question about what predicts the per-language optimum is wanted (group size, the group's share of the language's activation, resource level), write its design with rules first (a new spec section and D-entry before any number), as was done for C and D.
4. Standing: literature re-check about 2026-10-30; questions Q21 to Q27 open with their defaults; cron entries expire about 2026-10-29.

## 4. Open questions for the user

Q21 (HF upload), Q22 (Llama licence), Q23 to Q26, Q27: unanswered, defaults in force (`DECISIONS_FOR_REVIEW.md`).
