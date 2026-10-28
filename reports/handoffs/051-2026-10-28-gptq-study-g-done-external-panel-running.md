# Handoff 051: GPTQ extension, Study G analysed (D-109), external panel running (28 October 2026, 2026-10-28 12:33 IST)

> The ONLY source of the previous chat. A fresh chat with no memory must be able to continue from this file plus the repository. If this file and the repository disagree, the repository wins: say so in the next handoff.

## 0. Read this first (the standing rules)

Everything in [handoff 050](050-2026-10-28-gptq-stages-1-3-and-f5-done-study-g-external-running.md) section 0 still applies (user request of 2026-10-27 verbatim there; main branch only; never edit hash-pinned entry points; never remove progress tracking; times from `date`; no secrets or weights in commits; Q29 not started; Codex's and other people's uncommitted files, e.g. `reports/steering_feasibility_review_2026-10-28.md`, are never touched).

## 1. State right now

- **Analysed and pushed:** gates, stage 1, transfer study, E4/E5/E7, F5 (D-108), **Study G (D-109)**: `reports/gptq_extension/results/{stage1,stage2,stage3,f5,gain}/`; plain-language document `reports/gptq_extension/RESULTS.md` (sections 4.1 to 4.7).
- **Study G headline:** INT4 net harm at x0.0884 is 1.92 / 1.67 / 1.11 times the full-precision value (legacy / matched / GPTQ); the 'boosting repairs the 4-bit model' hint (best gain above 1 in 21 of 27 languages) is 7 of 27 under GPTQ (= full precision); INT6 identical; pre-set repair hypotheses not met in all arms; H-G4 flips only on a one-bootstrap-error threshold.
- **Ada:** external group INT4 44209/44210 COMPLETED and NOT yet validated; group INT6 44211/44212 and feature INT4/INT6 44213 to 44216 running (feature jobs about 6.5 h, due about 18:00 to 19:00). Nothing else queued.
- **Cron:** three entries (`bf1ae581`, `ad8593e3`, `8db9afdb`), every 40 minutes, expire about 2026-11-03.

## 2. The next step, exactly

See NEXT STEP in `reports/gptq_extension/STATUS.md`: validate 44209/44210 (`validate_job.py`-style, `--gate-max-abs 3.0`, external core), wait for the other six, then the unchanged `phase1_transfer_report.py` with the external options per arm (old README `reports/intervention_transfer/results/external/`), compare with the committed legacy external results, D-110, RESULTS.md section 4.8, handoff 052.

## 3. Open questions for the user

Q30 (disk, stop point, drop order; E3 and F4 not repeated). Q29 not started. Whether GPTQ findings enter the course report (deadline 31 Oct 2026) is the user's call.

## 4. Pitfalls

- Study G analysis: every run directory needs its own `meta_lm_eval.json`; the FP run is the OLD file (`func_inputs/lg_FP`).
- Caution for the external panel write-up: GPTQ calibration text is Wikipedia and the panel is Wikipedia text in six languages (different articles): an advantage under GPTQ may partly be domain match.
- Run heavy analyses one at a time (about 4 to 5 GB free RAM locally).
