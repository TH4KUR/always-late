# Completed transfer follow-up analyses

This package owns the completed post hoc analyses. The original study engine,
model runners and precision-audit work remain in their original files because
their paths and hashes are part of experiment provenance.

| Module | Responsibility | Existing command retained |
|---|---|---|
| `cross_corpus.py` | Original A11 cross-corpus diagnostic | `phase1_transfer_cross_corpus.py` |
| `recovery.py` | Locked-model verification, FLORES floors and calibration-only comparator | `phase1_transfer_recovery.py` |
| `gradients.py` | Stored scalar gradients versus signed NLL effects | `phase1_transfer_gradient_check.py` |
| `support.py` | Joint eligibility, output-scale overlap and control contrasts | `phase1_transfer_support.py` |
| `native_fixed.py` | Paired native/fixed group contrasts and sentence bootstrap | `phase1_transfer_native_fixed.py` |
| `tables.py` | Typed per-context CSVs, duplicate checks, ordered CSV output and SHA256 | Shared by the analyses |

Each analysis keeps its own endpoint, eligibility, baseline, statistical protocol,
CLI and JSON schema. Shared I/O does not choose thresholds, filter rows or impute
missing values. CSV columns retain their original first-appearance order; the
native/fixed writer retains its explicit first-row columns.

The five original script files are compatibility entry points. They alias their
implementation module when imported, including private helpers and mutable module
attributes. They call the same `main()` when executed. There are no duplicate
implementations to keep synchronized.

Run existing commands unchanged, for example:

```bash
python scripts/g0/phase1_transfer_support.py --help
```

The package also supports module execution with the established script import root:

```bash
PYTHONPATH=scripts/g0 python -m transfer_followup.support --help
```

`phase1_transfer_analysis.py` remains the frozen numerical engine. This package
depends on it in that direction; frozen modules need no imports from this package.
Existing deployment archives all of `scripts/g0`, including this directory.
Historical source and mutation-test targets in old handoffs refer to their original
commits; future mutation checks should target the implementation modules here.

Tests and preservation proof: [maintenance plan](../../../docs/maintenance/2026-10-27-cleanup-plan.md).
Repository-wide module map: [codebase map](../../../docs/CODEBASE.md).

## Current reporting pointer, 30 October 2026

[Current synthesis](../../../reports/FINAL_RESEARCH_QUESTIONS.md), [research paper](../../../paper/research-paper/main.tex), [current status](../../../reports/CURRENT_STATUS.md). This page retains its technical or dated historical role; the final synthesis uses the completed evidence through 29 October, with 9B GPTQ pending. The reporting revision does not change experiments, frozen rules, numerical outputs or reproduction code.
