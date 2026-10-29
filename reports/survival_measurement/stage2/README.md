# Stage 2 code and its checks (D-119, task 4 of the study ledger)

Written 2026-10-29 by the 40-minute cron session. Design: [`../stage2_design.md`](../stage2_design.md). Ledger: [`../STATUS.md`](../STATUS.md). **No Ada job has been submitted; no number from the model exists for this stage.** Everything here ran on CPU with the toy harness of `tests_torch/test_phase1_run.py`.

## What exists

| file | role |
|---|---|
| `scripts/g0/survival_sidecar.py` | `SurvivalSink` (21 statistics per condition, language and feature; equal-budget stream; FP cache and token-order hash), the gates G1 to G4 and G6 as functions, `run_gates`, `check_committed` (G2(ii) and the committed half of G4 on the fetched sums, run locally), `PreCapture` (the SAE's own `hook_sae_acts_pre`, fallback `x @ W_enc + b_enc`), `save_sums` / `load_sums` / `validate_dir`, `check_thresholds`, `run_job` (everything after the model and the SAE are loaded) and `main` (the entry point). It imports no frozen file and edits none; it calls the unchanged `phase1_run.run_survival(profile_sink=...)`. |
| `scripts/g0/tests_torch/test_survival_sidecar.py` | 47 tests: hand-computed toy values for the statistics, a plain-Python reference on random data, chunking and batching invariance, the sink inside `run_survival` on the toy harness (results identical with and without it), every gate and its failure, the order hash, the files and the validator, the pre-activation capture (hook and fallback, never a stale value), the whole job end to end (`run_job`) including a failing INT0 that ends the job non-zero after writing its files, and the committed comparison. |
| `mutants_survival_sidecar.json`, `mutation_result_2026-10-29.txt` | the mutation battery (59 mutants, `scripts/g0/mutcheck.py`) and its result: 58 killed, 1 survived: `Sab_both` formed over all tokens instead of the co-active ones, an **equivalent mutant** (a times b is 0 wherever either is 0, so the two sums are equal). |
| `scripts/g0/deploy_phase1.sh` | one path added to the deployed set: `reports/phase1/sae_thresholds.csv` (gate G3 compares the loaded SAE's thresholds with it). |

Test totals on 2026-10-29: `scripts/g0/tests` 477 OK; `scripts/g0/tests_torch` 2551 OK (54 skipped, as before).

## Two corrections to the design note found while writing the code (the design stays as written; this is the add-only update)

1. **G2(i) tolerance is 1e-7, not 1e-9.** `phase1_run._pack` stores the correlation as float32 (`r[k].float()`), so the correlation rebuilt in float64 from the sums cannot agree with the returned one to better than the float32 rounding (half a unit in the last place is 6e-8 near 1). The first run of the test showed 3e-8. G2(i) is `max |r_sums - r_run| <= 1e-7`; the sums themselves are float64 and exact in the sense of the design. A wrong formula or a wrong sum moves r by far more than 1e-7 (the mutation battery kills every such mutant).
2. **G2(ii) and the committed half of G4 are evaluated locally**, on the fetched `sums_*.pt`, by `survival_sidecar.check_committed(job_dir, committed_run)`; the committed `survival.pt` files are therefore not copied to Ada (about 16 MB each, `/home2` has 2.3 GB free). The in-job option `--committed-run` still exists and gives the same result. The pre-specified criteria are unchanged (per-language count of active features with r above 0.9 identical; maximum |difference| over active features at most 1e-3; 99.9th percentile over all cells at most 1e-4). A subset (smoke) run never compares with the committed files.

## Planned job commands (NOT run; they need the user's direct confirmation, see the design note section 7)

`REPO` is the deployed snapshot (`bash scripts/g0/deploy_phase1.sh <name>` after the commit). The weights wrapper is the existing `scripts/g0/run_gptq.sbatch` in `wrap` mode (it builds the artifacts of `--bits` before the job starts and checks them against `artifact_hashes.json` with `FROZEN=1`).

```
# smoke (legacy rounding, three languages, 40 sentences each; G1 to G4 and G6 on it; about 10 minutes)
sbatch -t 01:00:00 --exclude=gnode068,gnode069,gnode076 --export=ALL,REPO=$REPO scripts/g0/run_gptq.sbatch sc-smoke wrap --entry survival_sidecar:main --weights rtn -- --out {OUT} --bits 6 0 --flores-root /home2/karma/duan-verify/phase1/data --commit {COMMIT} --languages eng_Latn hin_Deva khm_Khmr --max-sentences 40
# full jobs, queued together after the smoke job and its timing reading pass; at most two running
sbatch -t 05:00:00 ... sc-legacy  wrap --entry survival_sidecar:main --weights rtn         -- --out {OUT} --bits 16 6 4 0 --flores-root ... --commit {COMMIT}
sbatch -t 05:00:00 ... sc-matched wrap --entry survival_sidecar:main --weights rtn-matched -- --out {OUT} --bits 6 4 0     --flores-root ... --commit {COMMIT}
sbatch -t 05:00:00 --export=ALL,REPO=$REPO,FROZEN=1 ... sc-gptq wrap --entry survival_sidecar:main --weights gptq -- --out {OUT} --bits 6 4 0 --flores-root ... --commit {COMMIT}
```

Open point for the review: `main` cannot run locally (no `sae_lens` or `transformer_lens` in this environment, and no model); it copies the setup lines of `phase1_run.main` for the survival pass and is exercised only through `run_job`. The smoke job is its first real test; any failure there is a plain setup error (an attribute name of the installed `sae_lens` 6.54.5, for example `sae.threshold` or `sae.hook_sae_acts_pre`), reported by gate G3 with a clear message before any statistic is trusted.

## Update 2026-10-29 06:05 IST (after the read-only review of commit 8cfdca3 by session project-fe; add-only; the text above stays as written)

Review result: no blocking issue; pre-activation capture, `main` against `phase1_run.main`, the weights path and the two design corrections confirmed. Handled:

1. **FROZEN=1 for BOTH patched full jobs** (`sc-matched` and `sc-gptq`), not only `sc-gptq`: `artifact_hashes.json` holds the matched-RTN hashes too, and without it the matched job's artifacts would not be checked against the recorded ones, which weakens the "same weights as stage 1" claim that G2(ii) relies on. The command block above is superseded on this point: `sc-matched` is submitted with `--export=ALL,REPO=$REPO,FROZEN=1` as well.
2. **A second short smoke on the patched-loader path** (suggestion 1, adopted): after the legacy smoke, `sc-smoke-m` with `--weights rtn-matched --bits 6 0` on the same three languages (about 15 minutes of artifact build plus minutes), so a wiring error of `LazyQuantizer` / prebuild inside the new entry point cannot cost a 5-hour slot. The full jobs are queued after both smoke jobs pass.
3. **The weight method is recorded in `meta.json`** (suggestion 2, adopted): `weights` is read from the quantizer the wrapper installed (`LazyQuantizer.args.weights`; their plain function means `rtn`). `check_committed` reads the method of the committed run from the `weights_manifest.json` next to its `eq/` directory (none = legacy rounding) and refuses a committed run of another method (`weights_match`, part of `passed`); the validator refuses a job whose recorded weights differ from its `--arm`.
4. Found while writing the validator: `validate_dir` crashed with an unpickling error on a corrupt file; it now raises `SidecarError` (a failed validation).

New file `scripts/g0/survival_validate.py` (the local and remote validation of a finished job: sacct COMPLETED 0:0, `GPU 0:` and no CUDA warning in the log head, sha256 of every fetched file against Ada's, the meta gates, the saved sums, the weights manifest against the recorded hashes, and, for a full run, the committed comparison REPORTED but never a failure) with `tests_torch/test_survival_validate.py`. Test counts now: `test_survival_sidecar.py` 49, `test_survival_validate.py` 10. Mutation batteries (`mutants_survival_sidecar.json`, `mutants_survival_validate.json`, result file refreshed): sidecar 63 mutants, 62 killed + the one equivalent survivor; validator 13 mutants, 13 killed.
