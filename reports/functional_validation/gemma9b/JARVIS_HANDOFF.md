# Jarvis Labs hand-over

The instructions for the person (and the Claude) running this job are in the repository root: [`CLAUDE.md`](../../../CLAUDE.md). The three-arm extension of 2026-10-28 (legacy RTN, matched RTN, GPTQ; Study G and Phase 1) is described there; the bundle with the code and the calibration pool is made by `scripts/g0/make_jarvis_bundle.sh`, and the run is one command: `HF_TOKEN=hf_xxx bash scripts/g0/jarvis_run_all.sh` (first `DRY_RUN=1` to read the plan).
