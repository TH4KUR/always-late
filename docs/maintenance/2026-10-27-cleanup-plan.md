# Documentation and module cleanup, 27 October 2026

User request: clean documentation and code, make it modular, preserve important files.
This is a behavior-preserving maintenance change, not a new research protocol.

## Design

Move the five completed post hoc transfer analyses into `scripts/g0/transfer_followup/`:
`cross_corpus.py`, `recovery.py`, `gradients.py`, `support.py`, `native_fixed.py`.
A shared `tables.py` owns per-context CSV parsing, SHA256 and ordered CSV output.
Keep all five original script filenames as executable compatibility modules. Importing an
old name resolves to the implementation module so attributes and monkeypatching do not fork.
The package deliberately continues to use the unchanged frozen analysis engine.

Alternatives considered: a repository-wide package migration would break pinned source paths
and ongoing jobs; moving only documentation would leave the flat analysis code unchanged.
The completed follow-up analyses are a bounded first module boundary with real reproduction
checks. Legacy runners and precision-audit code remain outside this refactor.

## Preservation boundary

No datasets, outputs, freezes, protocols, old designs, handoffs or experiment logs are removed.
No runner, precision-audit module, dependency, estimator, seed, output schema or CLI changes.
Existing shell commands, module imports and deployment of the whole `scripts/g0` tree remain valid.
Source organization changes; historical provenance continues to refer to its original commits.
Documentation snapshots preserve full pre-edit bytes with a SHA256 manifest and source paths.
Current execution status lives on one page; other entry points link to it.

## Execution

- [x] Run the existing transfer suite and capture baseline outputs/CLI help before edits.
- [x] Add failing package/import compatibility and table-boundary tests.
- [x] Move implementations; consolidate identical CSV/hash code without altering ordering/errors.
- [x] Run the same suite, legacy CLI checks, and byte-compare all five real-data analyses.
- [x] Add a code map and module README; verify scripts remain included by existing deployment.
- [ ] Integrate only the verified code changes into the active checkout, preserving concurrent work.
- [ ] Archive stale living-doc text verbatim, refresh current pointers/status and log the cleanup.
- [ ] Check links, archive integrity, protected-file hashes and final diff.

## Proof and review focus

Tests must cover duplicate CSV item rejection, blank/typed cells, empty/heterogeneous CSV
column order, the private function exports of old imports, attribute patch propagation, both
CLI entry mechanisms and invocation from another working directory. Reproduction writes only
temporary directories; compare JSON/CSV bytes, not merely headline numbers. The original
frozen analysis engine and all result files must be byte-identical before and after.
