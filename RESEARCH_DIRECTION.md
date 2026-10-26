# Main research direction and conditional alternatives

> **Update 2026-10-26 (Claude, documentation sweep; Codex's text below is unchanged):** (1) The user replaced Llama-3.2-1B by **Gemma-2-9B with Gemma Scope** (`layer_20/width_16k/canonical`) as the second model (D-089). Read every 'Llama' below as the second model. The 9B runs are done by another team on another cluster (handoff 040, branch `gemma9b-confirmation`), so this study runs its Gemma-2-2B part first and its second-model arm after that branch is merged. The Llama check finished on 2026-10-26 and is reported as exploratory only. (2) Verified and adopted with corrections (D-088): [verification](reports/intervention_transfer/verification_2026-10-26.md) (Chang EMNLP 2025 §6.1-6.2 is the loss-level precedent, so C > W alone is not a contribution; the gradient baseline 2610.15850 is mandatory; Bhandari 2609.06473 added; three citation fixes). The current order of work is the NEXT STEP block of `reports/functional_validation/STATUS.md`.

**Status, 26 October 2026:** PLANNED. No new model experiment has been run by the planning chat.
This is the execution entry point for Claude after finishing/reconciling existing jobs.

## MAIN: When does feature survival predict intervention transfer?

Independently vary hidden-state context, downstream weights and intervention amplitude.
Explain native transfer failures, then test a diagnostic on held-out documents/features.
The research aim is an explanatory and useful result, not another survival or gain sweep.

- [Scientific design and synthesis](docs/superpowers/specs/2026-10-26-intervention-transfer-research-design.md)
- [Executable plan, tasks and two-day budget](docs/superpowers/plans/2026-10-26-intervention-transfer-main-and-contingencies.md)
- [Verified future-work statements and novelty boundaries](reports/publication_strategy_2026-10-26/future_work_and_novelty.md)
- [New descriptive reanalysis of existing curves](reports/publication_strategy_2026-10-26/existing_curve_patterns.json)
- [Reproduction script](reports/publication_strategy_2026-10-26/reanalyse_existing.py)

## Why this direction

At INT4, activations shrink while language-group removal hurts more; restoration recovers
little of the compression error. Fixed-amplitude controls did not fix the activation
background, so they do not isolate downstream weights. Generic directions drift similarly,
but languagewise amplification is not explained by a shared scalar in the descriptive check.
The task null and failed shrinkage/optimal-gain relationship constrain the mechanism.

Duan explicitly leaves causal validation under quantization open. FEGA explicitly proposes
successive-layer effect measurements. Chang's peer-reviewed EMNLP paper calls for downstream
evaluation. These motivate the work; none certifies that our exact experiment is new or
that it will produce a publishable positive result.

## Secondary directions, activated by evidence

**A: locate where effect continuity breaks along the suffix.**
Use only if per-context instability or interaction remains unexplained. Reuse the main
forwards and collect fixed layer checkpoints. FEGA's explicit future-work proposal is the
parent; our extension is the compression-dependent comparison and predictive consequence.

**B: distinguish output-language control from comprehension under compression.**
Use only if effect transfer is mostly preserved/generic while the task distinction remains.
Run a fixed language-continuation protocol with content/coherence checks, preserving the
Belebele null. Existing FP language steering and translation-feature work are baselines.
This follows the explicit steering/downstream/naturalistic-generation future-work anchors.

At most one branch runs within the same cap. No third experiment is launched just because
a result is unfavorable.

## Execution boundaries

48 elapsed hours, planned ceiling 32 GPU-hours, at most two concurrent Ada jobs including
existing jobs. Benchmark once for correctness/throughput, then execute the frozen matrix.
Preserve the 27-language/300-sentence/three-control-draw Study G floor; use the established
404-feature F5 subset with up to 16 contexts. Six-language fresh-corpus validation is an
additional panel. Do not silently shrink scope to fit or cancel existing Llama jobs.

First read current STATUS and latest handoff: the final Llama result is not present in the
local result tree, and older access-blocked summaries are stale. Freeze samples, model IDs,
endpoints and budgets before reading new Q outputs. Preserve all old reports and history.

## Publication standard

A factorial plot or an obvious suffix-only counterexample is insufficient. The work must
explain the existing puzzle and demonstrate a replicated mechanism, useful held-out audit
prediction, or a behaviorally meaningful boundary. Broad claims require a second quantizer.
The two-day study can earn further paper work; acceptance cannot be promised.

