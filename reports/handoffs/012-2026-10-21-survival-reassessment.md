# Handoff 012: Q3 reassessment, Q4 widths and communication with Claude (2026-10-21)

## User-approved decisions

- Q1 / D-028: `mask` only, no further `none` work, no further numerical comparison against Duan. Preserve historical evidence and reused-code attribution. BOS is in model input and excluded from feature statistics; appending EOS is not adopted.
- Q2 / D-029: publishability is the main objective. Questions/proposal/direction can change with recorded reasons based on current evidence; check the specific contribution against the closest primary work. Historical OSF status is unverified, not a blocker to justified changes; do not claim retroactive pre-registration.
- Q3 / D-030: reassess survival detection rather than merely selecting between broad/severe damage thresholds. The direction is recorded, but the replacement method and precise experiments remain to be selected.
- Q4 / D-031: include INT5 and INT4 with INT8/INT7/INT6. Existing mask outputs already contain them. No new confirmatory pair or rerun is implied.

## Project state and methodological finding

ANLP project on language-selective feature survival and resource-tier differences under RTN quantization of Gemma-2-2B, using a fixed layer-12 Gemma Scope 16k SAE and 27 FLORES-200 languages. Phase 1 and additional analyses are complete; see `reports/phase1_results.md`, generated `reports/phase1_summary.md`, and cached-output reproduction record `reports/phase1_reproduction.md`. Existing outcomes and their chronology are preserved.

Current tagging occurs once on unquantized dev; `feature_table` in `scripts/g0/phase1_analysis.py` reads each feature's correlation at its original peak language. This measures preservation of an activation pattern for a baseline-selected cohort, not whether language specificity or causal function remains. A feature could preserve Hindi responses while gaining English responses. Do not claim semantic erasure/preservation from r alone.

Important schema finding: `LangPearson.update`/`_finish` in `phase1_lib.py` export firing, maximum and mean from reference `x`/`sx`, not quantized `y`/`sy`. `_pack` therefore returns reference response statistics even in quantized-rung records. `save_outputs` persists these arrays only for FP, plus each rung's correlations/losses. The schema does not export the quantized language-response profiles needed for selectivity reassessment. Check other cached artifacts before concluding a new GPU pass is needed; do not infer missing profiles or sentence-level bootstrap information from aggregate correlations.

## Sources and proposed work

The canonical source/rationale/design record is `reports/survival_method_reassessment.md`. It links official proceedings and published PDFs for Deng et al. (ACL 2025: language features, ablation and steering), Leask et al. (ICLR 2025: non-canonical units), Nelson et al. (ICML 2026: instability/identifiable SAEs), Makelov et al. (ICLR 2025: task-grounded evaluation), and Karvonen et al. (ICML 2025: SAEBench/proxy limitations). None directly establishes the precise quantization-invariance assumption, and accepted status does not establish independent replication. No novelty certification was made.

Candidate sequence: M0 artifact/schema feasibility inventory; M1 full pre/post language-profile comparison; M2 SAE reconstruction/sparsity quality by language/width; M3 language-specific functional interventions; M4 alternative dictionary/probe only if justified. These are proposals, not a request to implement or run them all. First define comparable inputs, discovery/held-out evaluation, cohorts, controls, calibration/uncertainty rules, compute and novelty. Preserve original cohorts rather than filtering to retained selective features. Q15's overlapping quality diagnostic is not automatically approved. Q5/Q9 remain closed. All work stays mask-only and covers the approved width ladder where relevant.

## Communication and ownership

The user authorized communication and requested a dedicated folder. `communication/README.md` defines the shared-file protocol and the dated message carries the full plan. [Claude acknowledged](../../communication/replies/2026-10-21-claude-to-codex.md) Q1-Q4, accepted the protocol and supplied a bounded artifact inventory/schema correction. [Codex replied](../../communication/replies/2026-10-21-codex-to-claude-receipt.md), confirmed the correction in the reassessment record and retained ownership of shared review documents. Claude reports no other files owned or new work started. The folder has no automatic watcher.

Codex edited documentation only: decision review/register, progress log, source/design record, README/PLAN/PROGRESS, result-interpretation notices, routine, communication and handoff. Do not overwrite concurrent changes. Read files before narrow patches and coordinate ownership. No code, saved measurements, vendor files or paper prose were changed. No GPU submission or commit/push was made in this documentation step.

## Next actions

1. Read new shared inbox/replies on resumption. Codex owns the shared review documents; coordinate any transfer before overlapping work.
2. Complete the bounded M0 cached-artifact feasibility inventory; the in-repo code/schema findings above are established, while remote/cache completeness is not.
3. Produce a concrete source-grounded measurement proposal and novelty check before implementing a replacement or measuring new outcomes. Keep candidate findings exploratory until genuinely unseen evaluation is prospectively specified.
4. Continue the user's question review when requested; Q3/Q4 decisions are recorded. Other items, such as Q6/Q13/Q14/Q15, are not silently answered by reassessment.

## Operational notes

Repository `/home/karma/Documents/ANLP/project/always-late`; Claude's active terminal process was observed with working directory `/home/karma/Documents/ANLP/project`. Work in this chat; no cron or second chat for this task. Do not edit `paper/` prose or `third_party/`; the team writes the paper. Never expose stored credentials or commit large artifacts. Runtime for later authorized tasks: Ada `/home2/karma/duan-verify`, environment `env/`; deployment guidance in `reports/scheduled_routine.md` is subordinate to current decisions. Historical `none` tooling and output files remain for provenance, not as instructions to restart retired work.
