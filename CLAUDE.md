# CLAUDE.md: read this first (Gemma-2-9B confirmation run, prepared for a rented GPU machine)

You are most likely a Claude Code session on a **Jarvis Labs (or similar) GPU instance** started by a friend of the study's author. Your only job is to run the prepared Gemma-2-9B job, keep it healthy, and hand the results back. **Everything else is already built and tested.** Read this file completely, then follow "The one command".

## What this is

The repository is the "Always Late" ANLP project (does compression, i.e. quantization, damage language-specific SAE features?). The main study is on Gemma-2-2B. **This run is the second-model confirmation on Gemma-2-9B** with the same SAE family (Gemma Scope `layer_20/width_16k/canonical`, same relative depth), decision **D-089**; the design is in [`reports/handoffs/040-2026-10-26-gemma9b-confirmation-external-cluster.md`](reports/handoffs/040-2026-10-26-gemma9b-confirmation-external-cluster.md) and [`docs/superpowers/plans/2026-10-26-gemma9b-confirmation.md`](docs/superpowers/plans/2026-10-26-gemma9b-confirmation.md). The experiment: zero-ablate each language's selective SAE feature group and measure the harm to that language's next-token loss, at full precision and at INT8/7/6/5/4 (RTN weight quantization), 27 languages, 300 FLORES-200 sentences, 3 matched-control draws.

Predictions P1 to P3b and the verdict function are **frozen** in [`reports/functional_validation/gemma9b/freeze.json`](reports/functional_validation/gemma9b/freeze.json) (copied verbatim from the Llama plan, sha256 inside). They were written before any 9B number exists. A "not consistent" result is a valid result.

## The one command

Prerequisites (the person who runs this must have them; if one is missing, say so and stop):
- A GPU machine with **at least one NVIDIA GPU of 40 GB or more** (A100 40/80 GB, A6000, L40S, H100), about **80 GB free disk** and **64 GB RAM per concurrent process** (the quantizer keeps a full-precision CPU copy of the model).
- A Hugging Face token (`hf_...`) of an account that has **accepted the Gemma licence** at https://huggingface.co/google/gemma-2-9b . Never print, log, commit or paste the token anywhere.

```bash
cd <this repository> && HF_TOKEN=hf_xxx bash scripts/g0/jarvis_run_all.sh
```

That is all. The script detaches itself (it survives a closed tab or ssh), writes `../run_all.log` (one level above the repo), and is **resumable**: if anything stops, run the same command again and finished steps are skipped. Follow it with `tail -f ../run_all.log`. Do not start a second copy while one is running (`ps aux | grep jarvis_run_all`).

What it does, in order: python venv with the pinned versions (torch 2.14.1 cu126, transformer-lens 3.9.0, transformers 5.19.0, sae-lens 6.54.5) -> FLORES-200 (bundled, hash-checked) -> token check and download of the model and SAE (about 19 GB) -> unit tests and a check that the code is exactly the frozen code -> **smoke test** (3 languages, FP and INT4; if its gates fail the run STOPS) -> full run (tagging and manifest on 27 languages, then the six states FP, INT8, INT7, INT6, INT5, INT4; one process per GPU when there are several GPUs and enough RAM) -> `../gemma9b_results.tar.gz` plus `../artifact_checksums.md`.

Expected cost: the 2B runs took about 3 GPU-hours per state on a 2080 Ti; for 9B on an A100 expect roughly 3 to 6 hours per state, about 20 to 35 GPU-hours in total (not measured; the smoke job prints a first number). With several GPUs the states run in parallel.

## What was verified, and what was NOT

Verified (on a CPU machine, before this hand-over):
- The adapter `scripts/g0/phase1_gemma9b_run.py` (model `google/gemma-2-9b`, SAE `gemma-scope-9b-pt-res-canonical` / `layer_20/width_16k/canonical`, layer 20) with 18 unit tests, the freeze tooling with 6, the driver script with 5 (dry-run plan, GPU/RAM scheduling, refusals), and a stub rehearsal of the whole orchestration (smoke gate pass and fail, parallel states on 2 fake GPUs, resume, packaging).
- The frozen Gemma runners it wraps ran the 2B study; the Llama check used the same wrapper pattern.

**NOT verified: nothing has been run on a real GPU with the real 9B model.** The first real execution is the smoke job. Treat its result as the instrument gate (not a look at results). Likely failure points, in order: (1) the token cannot read the gated model (accept the licence); (2) CUDA/driver mismatch for the cu126 torch wheel (`TORCH_INDEX` can point to another wheel index); (3) out of GPU memory on a 40 GB card (use an 80 GB card, or add `LM_EXTRA="--batch-size 16"`, see below); (4) the frozen builder refusing the `all` group family because the control pool is too small (the driver then builds K16 and K64 only and writes `MANIFEST_FALLBACK.txt`; this is reported, not hidden); (5) the SAE id or release not found in the installed sae-lens registry.

## Rules (they keep the results valid and the merge clean)

- **Never edit** `third_party/`, the frozen runners (`scripts/g0/phase1_run.py`, `phase1_func_run.py`, `phase1_gain_run.py`, `their_code.py`, ...), the files hashed in `freeze.json` (`scripts/g0/phase1_gemma9b_run.py`, `run_gemma9b.sbatch`, `tests_torch/test_phase1_gemma9b_run.py`, `reports/g0/phase1_languages.json`), or the freeze itself. The step "checks" verifies the hashes: if it fails, the code on this machine is not the frozen code; stop and tell the author.
- If something must change to get the run through (for example a smaller batch size on a 40 GB card), it is a **dated amendment**: write what, why and when into `reports/functional_validation/gemma9b/amendments.md` (create it) **before** re-running, and pass the change through `LM_EXTRA` (e.g. `LM_EXTRA="--batch-size 16" HF_TOKEN=... bash scripts/g0/jarvis_run_all.sh`). Never silently change ratios, languages, sentence counts, states, seeds or thresholds. No post hoc language selection.
- Do not skip the smoke test (`SKIP_SMOKE=1` exists for a resume after it already passed, nothing else).
- Do not push to `main`, do not merge, do not change anything on Hugging Face or wandb. No weights or tokens in git. If you commit, commit only new files under `reports/functional_validation/gemma9b/` and say so.
- Report everything honestly: failed or missing states, warnings, the peak-memory numbers the logs print (`peak GPU GB`), wall times. A failed state is reported as failed.

## Monitoring and troubleshooting

- Progress: `tail -f ../run_all.log`; per-job logs in `../logs/` (`g9-smoke-tag.log`, `g9-smoke-lm.log`, `g9-tag.log`, `g9-lm-<state>.log`); step markers in `../state/*.done`; failed states leave `../state/state_<S>.failed`. GPU use: `nvidia-smi`.
- A state is validated only if its log has `gates passed`, has no `CUDA initialization` warning, and the output dir has a `func_lm_eval_*.pt`. Exit code 0 alone is not trusted.
- To re-run one failed state: run the same command again; it retries only unfinished states. To run only some states: `STATES="6 5 4" HF_TOKEN=... bash scripts/g0/jarvis_run_all.sh`.
- Disk full: the model cache is `../hf_cache` (about 19 GB); outputs are in `../out/phase1/`.
- Dry run (prints the plan, runs nothing): `DRY_RUN=1 bash scripts/g0/jarvis_run_all.sh`.

## Hand-back

When it ends (the log says `ALL DONE`, or `PARTIAL` with the failed states), the deliverable is **`../gemma9b_results.tar.gz`** (outputs, logs, checksums, pip freeze, the freeze file) and **`../run_all.log`**. The person running this sends those two files to the study's author, who runs the analysis (comparison with the 2B results, robustness checks R1 to R5, the 2B-vs-9B comparison). That analysis step is **not part of this job** and its scripts (`phase1_llama_compare.py`, `phase1_llama_robust.py`) still need to be generalised to a model-neutral form first (handoff 040, S5).

Summarise for the human: which states finished and validated, the smoke gate result, peak GPU memory, total wall time, any amendment or fallback, and the exact path of the results tarball.
