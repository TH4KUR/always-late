# Research scripts

Start with the [codebase map](../../docs/CODEBASE.md) for ownership, dependencies,
test commands and the distinction between frozen experiments and follow-up analyses.

Completed transfer follow-ups now live in
[`transfer_followup/`](transfer_followup/README.md). Their original script names
remain executable compatibility entry points. Other study families retain their
historical paths; do not rename or move files pinned by a freeze merely to make the
directory look uniform.

Tests are in `tests/` (stdlib) and `tests_torch/` (the existing ML environment).
Deployments archive the entire directory, so packages below it must stay included.
No data, model weights, result files or old study entry points were removed by this cleanup.
