# CLAUDE.md: read this first (Gemma-2-9B three-arm confirmation, prepared for a rented GPU machine)

You are most likely a Claude Code session on a **Jarvis Labs GPU instance**. Your job is to run the prepared Gemma-2-9B job, keep it healthy, and hand the results back. **Everything else is already built and tested on a CPU machine; nothing has been run on a real GPU with the real 9B model.** Read this file completely, then follow "The one command".

## What this is

The repository is the "Always Late" ANLP project (does compression, i.e. quantization, damage language-specific SAE features?). The main study is on Gemma-2-2B. **This run is the second-model confirmation on Gemma-2-9B** with the same SAE family (Gemma Scope `layer_20/width_16k/canonical`, same relative depth), decision **D-089**, extended on 2026-10-28 so that it confirms the project **as it now stands**, not only the legacy-RTN Study G of 2026-10-26:

- **Three weight-state arms:** `legacy` (the project's original RTN, exactly as frozen), `rtn-matched` (RTN done the standard way), `gptq` (GPTQ, calibrated on Wikipedia text in the 27 languages). Design: [`docs/superpowers/specs/2026-10-28-gemma9b-three-arm-confirmation-design.md`](docs/superpowers/specs/2026-10-28-gemma9b-three-arm-confirmation-design.md); plan: [`docs/superpowers/plans/2026-10-28-gemma9b-three-arm-confirmation.md`](docs/superpowers/plans/2026-10-28-gemma9b-three-arm-confirmation.md); earlier design [`docs/superpowers/plans/2026-10-26-gemma9b-confirmation.md`](docs/superpowers/plans/2026-10-26-gemma9b-confirmation.md).
- **Two stages ("tiers"):** `g` = tier 1, Study G (zero-ablate and scale each language's selective SAE feature group, harm to that language's next-token loss, 27 languages, 300 FLORES-200 sentences, 3 matched-control draws; legacy arm at FP, INT8, INT7, INT6, INT5, INT4, the two new arms at INT6 and INT4); `p1` = tier 2, Phase 1 survival and loss (`phase1_run`, mask regime; legacy bits 8 7 6 5 4 16 12 0, new arms 7 6 4 16 12 0).
- **Predictions are frozen:** P1 to P3b in `reports/functional_validation/gemma9b/freeze.json` (legacy arm, 2026-10-26) and Q0 to Q3 (tier 1) and S1 to S4 (tier 2) in `reports/functional_validation/gemma9b/amendment_A1_three_arm.json` (text, thresholds, verdict functions, code hashes). They were written before any 9B number exists. A "not consistent" or "different" result is a valid result.

## The machine (given 2026-10-28)

1x RTX PRO 6000 (Blackwell, compute capability 12.0), 96 GB VRAM, 28 vCPU, 160 GB RAM, 250 GB SSD, about 210 INR per hour, user `ubuntu`. One GPU: units run one after another (`PROCS_PER_GPU=2` only if the RAM allows two units: a GPTQ unit needs about 59 GB of host RAM, a tier-2 weight-set unit about 98 GB and runs alone). torch 2.14.1 (the pinned version) exists only for CUDA 12.6 and 13.0; Blackwell needs the **13.0 build and an NVIDIA driver of 580 or newer**: the driver script chooses the wheel index from the GPU and refuses with a message when the driver is too old (check `nvidia-smi` first).

## The one command

Files arrive in the home directory as a bundle (made on the author's machine with `scripts/g0/make_jarvis_bundle.sh`: the committed repository, the calibration pool token files, `SHA256SUMS`):

```bash
sha256sum -c SHA256SUMS && tar xzf always-late-gemma9b.tar.gz && tar xzf gptq_calibration_pool.tar.gz
cd always-late && DRY_RUN=1 bash scripts/g0/jarvis_run_all.sh            # prints the plan, runs nothing: read it
HF_TOKEN=hf_xxx bash scripts/g0/jarvis_run_all.sh                        # the real run
```

Prerequisites (if one is missing, say so and stop): an HF token (`hf_...`) of an account that has **accepted the Gemma licences** of `google/gemma-2-9b` AND `google/gemma-2-2b` (the 2B tokenizer is compared with the 9B one); about 80 GB free disk; the files above. Never print, log, commit or paste the token.

The script detaches itself, writes `../run_all.log`, and is **resumable** (run the same command again; finished steps and units are skipped). Follow it with `tail -f ../run_all.log`. Do not start a second copy (`ps aux | grep jarvis_run_all`).

Steps in order: `verify_frozen` (**every** invocation, first: the code must equal the freeze AND the amendment, else the run stops) -> `env` (venv, pinned packages, torch from the right index, a bf16 matmul on the GPU) -> `flores` (hash-checked) -> `access_and_download` (37 GB: the model is eight fp32 shards) -> `checks` (unit tests) -> `tokenizer` (9B equals 2B tokenizer, else STOP) -> `pool` (calibration files equal the frozen pool, else STOP) -> `smoke` (legacy path: tagging on all 27 languages with 100 sentences, `lm` restricted to 3 languages) -> `smoke_p1` (tier-2 path, 3 languages) -> `gate9b` (builds the weight sets of both methods at INT6 and INT4 and, in a second gate job, INT7 (tier 2 needs it), runs the gates, **records their sha256 in `reports/functional_validation/gemma9b/artifact_hashes.json`**; every later unit refuses a different hash) -> `smoke_gptq` (whole GPTQ path on 3 languages) -> `full` (tagging and manifest on 27 languages, then the units in the order tier 1: legacy FP, GPTQ INT4/INT6, matched RTN INT4/INT6, legacy INT4/INT6/INT8/INT7/INT5; tier 2: GPTQ, legacy, matched RTN) -> `../gemma9b_results.tar.gz` plus `../artifact_checksums.md`.

**First invocation: `STOP_AFTER=smoke_gptq HF_TOKEN=... bash scripts/g0/jarvis_run_all.sh`.** It stops cleanly after `smoke_gptq`. Then tell the author the seconds per artifact build (`g9-gate*.log`, `g9-smoke-gptq.log`), the peak GPU memory (`peak GPU GB`) and the host RAM seen (they are unmeasured estimates until then) and **wait for the author's OK** before running the same command without `STOP_AFTER` (about 16 weight-set builds and many hours follow; the machine is billed by the hour).

## What was verified, and what was NOT

Verified on a CPU machine: the adapter (18 tests), the calibrator and the canonical weight adapter on a toy model with the 9B attention layout (heads x d_head differs from d_model), the tokenizer check, the RAM guard, the hash recorder, the wrapper `phase1_gemma9b_ext.py`, the analysis `phase1_gemma9b_compare.py` (with a control: the committed 2B outputs through the same code reproduce the Q1 statistic and the tier-2 survival and loss numbers and satisfy those thresholds; P1, Q0 and Q2 cannot be re-derived on 2B because the raw item files are not in git), the amendment tool, the bundle script, the dry-run plan and a **stub rehearsal of the real orchestration** (units, markers, resume, a failed unit, a failed gate, RAM refusal).

**NOT verified: any real execution.** Never run on the real 9B model or on a Blackwell GPU: the cu130 torch wheel with TransformerLens/sae-lens, the GPTQ build time and memory on 9B, the real memory of the units, `phase1_run` with the 9B setting, the gate on the real model. The first real executions are the smoke steps and the gate: treat them as instrument gates (not a look at results). Likely failure points: (1) a token that cannot read a gated repo; (2) the NVIDIA driver is older than 580 (the script says so); (3) a missing kernel for sm_120 in the torch build (the env step says so); (4) host RAM (the guard refuses with the numbers); (5) the frozen builder refusing the `all` group family (K16/K64 fallback, reported in `MANIFEST_FALLBACK.txt`); (6) the SAE id missing in the installed sae-lens.

## Rules (they keep the results valid)

- **Never edit** `third_party/`, the frozen runners (`scripts/g0/phase1_run.py`, `phase1_func_run.py`, `phase1_gain_run.py`, `phase1_lib.py`, `their_code.py`, ...), the files hashed in `freeze.json` (`phase1_gemma9b_run.py`, `run_gemma9b.sbatch`, its test, `reports/g0/phase1_languages.json`) or in the amendment (`gptq_ext/tokenizer_check.py`, `ram_guard.py`, `record_hashes.py`, `phase1_gemma9b_ext.py`, `phase1_gemma9b_compare.py`), `reports/gptq_extension/*`, the freeze or the amendment. The step `verify_frozen` checks the hashes on every invocation: if it fails the code on this machine is not the frozen code; stop and tell the author.
- If something must change to get the run through (for example a smaller batch size), it is a **dated amendment**: write what, why and when into `reports/functional_validation/gemma9b/amendments.md` (create it) **before** re-running, and pass the change through `LM_EXTRA` (tier-1 runs and the smokes) or `P1_EXTRA` (tier-2 runs); the gate has no such option (a change there is a stop-and-ask). Never silently change ratios, languages, sentence counts, states, bit widths, seeds, thresholds or the calibration pool. No post hoc language selection.
- **Stop and report, do not work around:** a tokenizer mismatch, a calibration pool mismatch, a failed gate, an artifact hash mismatch (`gate9b` or a unit), a compute-capability or driver error. These mean the instrument is not what was tested.
- Do not skip the smoke steps (`SKIP_SMOKE=1` exists for a resume after they passed, nothing else). Do not push to `main`, do not merge, do not change anything on Hugging Face or wandb. No weights, tokens or pool files in git. If you commit, commit only new files under `reports/functional_validation/gemma9b/` and say so.
- Report everything honestly: failed or missing units, warnings, peak memory, wall times. A failed unit is reported as failed.

## Monitoring and troubleshooting

- Progress: `tail -f ../run_all.log`; per-unit logs in `../logs/` (`g9-lm-<S>.log` legacy tier 1, `g9-lm-<arm>-<S>.log` new arms, `g9-p1-<arm>.log` tier 2, `g9-gate.log`, `g9-smoke-*.log`, `g9-tag.log`); step markers `../state/<step>.done`; unit markers `../state/state_<S>.done` (legacy tier 1) and `../state/unit_<stage>_<arm>[_<state>].done`; a failed unit leaves `.failed`. GPU use: `nvidia-smi`.
- A unit is validated only if its log has `gates passed` (tier 1) or its output has `eq/survival.pt` and `all/survival.pt` (tier 2), there is no `CUDA initialization` warning, and (new arms) `weights_manifest.json` exists. Exit code 0 alone is not trusted.
- Subsets and order: `ARMS="legacy gptq"`, `STAGES="g"`, `STATES="FP 6 4"` (always keep FP: every prediction needs the full-precision file), `STATES_NEW="4"`, `P1_BITS_NEW=...` (every artifact bit width must be one the gate recorded: the gate derives its bit widths from `STATES_NEW` and `P1_BITS_NEW`); `ARMS=legacy STAGES=g` runs the same units as the old plan (in the order FP, 4, 6, 8, 7, 5). If time or money runs out the drop order is: legacy INT7/INT5/INT8, then the tier-2 matched-RTN arm; never GPTQ, legacy FP/INT6/INT4, the tagging or the tier-2 legacy and GPTQ arms.
- To re-run a failed unit: run the same command again. Dry run: `DRY_RUN=1 bash scripts/g0/jarvis_run_all.sh`.

## Hand-back and analysis

The deliverable is **`../gemma9b_results.tar.gz`** (outputs, logs, checksums, the frozen files used, the recorded hashes) and **`../run_all.log`**. The author analyses on a CPU machine with the frozen code:

```bash
python scripts/g0/phase1_analysis.py --run <p1 arm dir>/eq --out <dir>      # per arm, the original three tests (S4)
python scripts/g0/phase1_gemma9b_compare.py \
  --legacy-g out/phase1/g9-lm-FP out/phase1/g9-lm-4 out/phase1/g9-lm-6 out/phase1/g9-lm-8 out/phase1/g9-lm-7 out/phase1/g9-lm-5 \
  --rtnm-g out/phase1/g9-lm-rtn-matched-6 out/phase1/g9-lm-rtn-matched-4 --gptq-g out/phase1/g9-lm-gptq-6 out/phase1/g9-lm-gptq-4 \
  --manifest out/phase1/g9-tag/groups_manifest.json --ratios-json reports/functional_validation/gain_study/calibration/ratios.json \
  --legacy-p1 out/phase1/g9-p1-legacy/eq --rtnm-p1 out/phase1/g9-p1-rtn-matched/eq --gptq-p1 out/phase1/g9-p1-gptq/eq --roster reports/g0/phase1_languages.json \
  --analysis-legacy <..>/analysis.json --analysis-rtnm <..>/analysis.json --analysis-gptq <..>/analysis.json --out <results dir>
```

Summarise for the human: which units finished and validated, the smoke and gate results, the recorded artifact hashes' path, peak GPU memory and RAM, seconds per GPTQ build, total wall time, any amendment or fallback, and the exact path of the results tarball.
