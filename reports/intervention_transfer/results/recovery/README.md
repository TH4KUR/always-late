# Locked FLORES-to-Wikipedia scoring (revision R1, task 1a of the publication recovery plan; POST HOC; written 2026-10-27 02:49 IST)

Protocol fixed and hashed in [`../../revision_R1.json`](../../revision_R1.json) (commit `ed80bca`) BEFORE this run; plan: [`docs/superpowers/plans/2026-10-26-publication-recovery-after-results.md`](../../../../docs/superpowers/plans/2026-10-26-publication-recovery-after-results.md) Task 1 (Codex's plan; the locked models are hers: `reports/publication_reassessment_2026-10-26/locked_flores_models.json`, exported after the main outcomes with no external outcome used). Code: `scripts/g0/phase1_transfer_recovery.py` (26 tests, 26 mutants killed; 328 transfer tests green). Command:

```
python scripts/g0/phase1_transfer_recovery.py --locked reports/publication_reassessment_2026-10-26/locked_flores_models.json --main <transfer_runs/analysis_main> \
  --external <transfer_runs/analysis_external> --committed reports/intervention_transfer/results/main/predictive_validation.csv --out <transfer_runs/analysis_external/recovery>
```

Files: `cross_corpus_locked.csv` (one row per state and protocol), `cross_corpus_locked.json` (all numbers, the hashes of the locked file and of the six per-item inputs), `verification.json`.

## Verification before scoring (all passed, with differences of exactly 0.0)

(V1) a refit on the main calibration items by the frozen protocol equals the exported fit (12 models: strengths equal, mean, scale, coefficients, intercepts equal to the last digit); (V2) the locked models give the committed held-out MAE of every model and state on the main held-out items (12 values); (V3) `error_norm / (fp_norm + FLORES floor of the kind)` and the five-floor eligibility reproduce the normalized error and the eligibility of all 32,320 main items per state. The secondary rows below also reproduce the A11 E3 numbers of [`../external/README.md`](../external/README.md) exactly (INT4 20.63 %, INT6 19.65 %, INT16 26.69 %), an independent confirmation that the exported fits equal the refit used there.

## Result (frozen fits, no refit; external held-out-feature items of the real direction; reduction of the held-out MAE of A+B)

PRIMARY protocol (FLORES floors for the target, the covariates and the eligibility; 1,368 of 6,406 external items eligible, 621 held-out items, 99 features, 459 passages):

| state | held-out MAE: A / A+G / A+B / A+B+G / null | main-calibration CV: A / A+G / null | old comparator (locked baseline) | reduction | new comparator (A, A+G, null by main CV) | reduction | against A (H3 quantity) |
|---|---|---|---|---|---|---|---|
| INT16 (rounding reference) | 0.0213 / 0.0211 / 0.0151 / 0.0154 / 0.0298 | 0.01495 / 0.01484 / 0.02291 | A+G | 28.2 % [21.3, 35.1] | A+G | 28.2 % [21.3, 35.1] | 29.0 % [22.0, 35.6] |
| INT6 | 0.0380 / 0.0380 / 0.0309 / 0.0303 / 0.0367 | 0.02916 / 0.02937 / **0.02820** | A | 18.6 % [4.9, 29.5] | **null** | **15.8 % [-2.7, 28.7]** | 18.6 % [4.9, 29.5] |
| INT4 | 0.1375 / 0.1360 / 0.1088 / 0.1069 / 0.1397 | 0.11137 / 0.11057 / 0.11474 | A+G | 19.9 % [7.8, 31.3] | A+G | 19.9 % [7.8, 31.3] | 20.9 % [8.8, 32.0] |

SECONDARY protocol (the external floors of D-095 for the target and the eligibility; 1,444 eligible items, 659 held-out, 100 features, 483 passages): INT16 26.7 % [19.5, 34.0] (A+G); INT6 19.7 % [7.5, 29.5] against the old comparator A and **17.5 % [1.3, 28.9] against the null**; INT4 20.6 % [9.3, 31.2] (A+G).

## Reading

- The sensitivity to other directions, fitted only on FLORES calibration features, reduces the external held-out error with an interval above zero in every state and under both floors, except one cell: at INT6 with the FLORES floors the null (the calibration median) is a stronger comparator than both activation baselines (main-calibration CV 0.0282 against 0.0292 and 0.0294; on the external items 0.0367 against 0.0380), and the reduction against it is 15.8 % [-2.7, 28.7], an interval that includes zero. Against the old comparator (the locked baseline A) the reduction at INT6 is 18.6 % [4.9, 29.5]. At INT4 the null is not competitive (0.1397 against 0.1360 for A+G) and the reduction is 19.9 % [7.8, 31.3].
- The reduction in the rounding-only INT16 state is as large (28.2 %): the predictor transfers across corpora and languages but is still not shown to be quantization-specific; a quantization-specific claim would need incremental gain over measurement-sensitivity covariates and an outcome outside the original overlapping context set (plan, promotion decision), which this table does not provide.
- The external floors change the numbers little (INT4 20.6 % against 19.9 %): the cross-corpus result does not hinge on the normalization of the target.

## Limits

Post hoc (selected after D-094) and written after the external panel had been analysed under the original rules (D-095) and the A11 E3 row had been read; no choice here depends on those numbers (revision_R1.json, `honesty_note`). Feature-held-out only for the external items (the same feature split as the main study; the external contexts are other documents, which is the point of the test). Six languages and one corpus family; `held-out` items share features with nothing in the calibration half. Pairwise-eligible real items still differ in FP effect scale from their controls (plan: common-support analysis, not done here). The prediction needs four extra control interventions per feature and context, so it is not shown to be a cheaper audit than measuring the target. The original UTILITY verdicts (main: not met; external: met, D-095) are unchanged and are not re-evaluated here.
