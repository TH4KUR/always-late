# Current research direction

Updated 27 October 2026 after the external panel and recovery analyses (D-095–D-100).
Live work priority belongs to [current status](reports/CURRENT_STATUS.md).
Previous versions remain [verbatim in the archive](reports/archive/modular_cleanup_2026-10-27/README.md).

## Candidate contribution

**Which definitions of SAE intervention preservation remain reliable under weight
quantization, independent documents and controlled measurement precision?**

The broad novelty claim remains too strong. Context-dependent effects, intervention
strength controls, interactions, steering versus natural use, and numerical sensitivity
have direct precedents. The [26 October search record](reports/novelty_assessment_2026-10-26.md)
is a bounded literature assessment, not a certificate of priority or acceptance.
Duan's explicit future-work question motivates causal validation under compression;
the exact controlled comparison still needs a substantive answer.

## Evidence now available

- Main and six-language external matrices are complete. Original utility verdicts differ:
  main **not met**, external **met**; neither is redefined by later analyses.
- Locked FLORES models improve Wikipedia INT4 prediction error by 19.9% [7.8, 31.3].
  At INT6, the stronger calibration-median comparator leaves 15.8% [-2.7, 28.7],
  inconclusive. INT16 improves by 28.2%: prediction is not demonstrated to be
  quantization-specific or cheaper than directly measuring the target.
- Scalar gradients work on their signed-NLL target; the old vector-error result
  does not establish gradient failure. A full vector JVP remains untested.
- Similar-FP-effect comparisons show a modest real-versus-swapped direction
  component, with small retained populations and one usable control family.
  This does not establish selective language-feature destruction.
- Native edits cause less harm than fixed-amplitude edits at INT4 in all tested
  languages. The native-versus-weights remainder is small in the median, but
  varies substantially by language; it is not a causal percentage due to amplitude.

Sources and qualifications: [main](reports/intervention_transfer/results/main/README.md),
[external](reports/intervention_transfer/results/external/README.md),
[post hoc recovery](reports/intervention_transfer/results/recovery/README.md).
All recovery analyses were selected after the main outcomes and remain post hoc.

## Decisive remaining question

The paired arithmetic audit asks whether measured transfer drift and the comparison
with controls change when the SAME stored weight values are promoted for FP32 suffix
computation, holding the BF16 prefix cache fixed. The sample and protocol are recorded
in the [audit freeze](reports/intervention_transfer/precision_audit_freeze.json) and
[precision plan](docs/superpowers/plans/2026-10-27-precision-audit-task2-implementation.md).
Implementation progress is not an experimental finding. Generic BF16/FP32 disagreement
alone would not establish novelty. The audit must change a substantive interpretation
or establish a useful, replicated boundary.

The existing layerwise branch, language-control branch and margin sidecar remain
conditional alternatives. They are not automatic parallel launches after an unfavorable
result. Scope remains one model/layer/SAE and RTN; the separate 9B branch is excluded.
Keep the [recovery plan](docs/superpowers/plans/2026-10-26-publication-recovery-after-results.md),
original freezes, negative results and failed criteria. A strong paper is still a
candidate outcome, not an established achievement. Complete the course submission
on the evidence actually obtained.
