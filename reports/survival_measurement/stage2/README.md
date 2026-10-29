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

## Update 2026-10-29 06:10 IST (corrected command lines, after the nit of session project-fe; add-only)

The command block near the top still shows the `sc-matched` line without `FROZEN=1`; do NOT copy that line. The corrected full-job commands (`REPO` = the deployed snapshot; `...` = `--exclude=gnode068,gnode069,gnode076`):

```
sbatch -t 05:00:00 ... --export=ALL,REPO=$REPO         scripts/g0/run_gptq.sbatch sc-legacy  wrap --entry survival_sidecar:main --weights rtn         -- --out {OUT} --bits 16 6 4 0 --flores-root /home2/karma/duan-verify/phase1/data --commit {COMMIT}
sbatch -t 05:00:00 ... --export=ALL,REPO=$REPO,FROZEN=1 scripts/g0/run_gptq.sbatch sc-matched wrap --entry survival_sidecar:main --weights rtn-matched -- --out {OUT} --bits 6 4 0     --flores-root /home2/karma/duan-verify/phase1/data --commit {COMMIT}
sbatch -t 05:00:00 ... --export=ALL,REPO=$REPO,FROZEN=1 scripts/g0/run_gptq.sbatch sc-gptq    wrap --entry survival_sidecar:main --weights gptq        -- --out {OUT} --bits 6 4 0     --flores-root /home2/karma/duan-verify/phase1/data --commit {COMMIT}
```

The second smoke is `sc-smoke-m`: `--export=ALL,REPO=$REPO,FROZEN=1 ... sc-smoke-m wrap --entry survival_sidecar:main --weights rtn-matched -- --out {OUT} --bits 6 0 --flores-root ... --commit {COMMIT} --languages eng_Latn hin_Deva khm_Khmr --max-sentences 40`.

## Update 2026-10-29 07:54 IST (smoke results and a correction of the commands; add-only)

**Smoke job 45775 (legacy weights, bits 6 0, three languages x 40 sentences; snapshot `repo_f91c8b0`, gnode057, RTX 2080 Ti): COMPLETED in 3 min 5 s; every gate passed; `scripts/g0/survival_validate.py --smoke` ok.** Gate values: G1 INT0 equals FP (exact); G2(i) largest |r from the sums - r of `run_survival`| 3.0e-8 (INT6), 0 (INT0); G3 `acts == pre * (pre > theta)` exact (0.0), thresholds equal the committed table to 4.7e-7, theta sha256 `0df56171...` equals the sha256 in the header of `reports/phase1/sae_thresholds.csv`; G4 exact; G6 batch hash `a43c2196...` equal to the one from `make_batches`; pre-activation source `sae.hook_sae_acts_pre`, float32; versions torch 2.14.1+cu126, transformer-lens 3.9.0, sae-lens 6.54.5. 1,206 equal-budget tokens per language. Identity gate of the freeze on these sums: 2,869 cells checked, 0 excluded, largest difference 2.9e-15. **Peak GPU memory 9.38 GB of 11.26 GB** (margin 1.9 GB; if a full job runs out of memory, rerun with `--max-tokens 768`: the batching changes, the results do not beyond float noise, and the expected hash is recomputed from the same arguments). Informational sanity check only (40 sentences against the committed 1,012, so sampling noise is large): on features active in both, the Spearman correlation between the smoke r and the committed legacy INT6 r is 0.78 (eng_Latn), 0.79 (hin_Deva), 0.83 (khm_Khmr); the share of r above 0.9 is 0.954 / 0.965 / 0.907 in the smoke against 0.971 / 0.975 / 0.932 committed; median nu about 0.12 in all three languages; no silent feature. These are smoke numbers, not results.

**Smoke job 45776 (matched RTN, `FROZEN=1`): FAILED after 18 min 8 s, in the weight wrapper's artifact build, before any sidecar work.** The INT6 artifact was built (all 26 blocks, 35 s each) and then refused: `no frozen sha256 for rtn_matched at 6 bits (built e99efbf7...)`. Cause: `artifact_hashes.json` records the INT4 hashes only (the other widths were recorded by the stage jobs in their manifests, D-103), and `FROZEN=1` alone refuses a width without a recorded hash. The stage-1 jobs ran with `FROZEN=1,RECORD=1`. **Correction of the planned commands: the two patched full jobs (`sc-matched`, `sc-gptq`) and the matched smoke take `--export=ALL,REPO=$REPO,FROZEN=1,RECORD=1`** (INT4 enforced, INT6 recorded in the job's `weights_manifest.json`); my earlier statement that INT6 was enforced was wrong. The built hash `e99efbf7c7ba...` equals the matched INT6 hash of the stage-1 job (`transfer_runs/rtnm-p1-mask/weights_manifest.json`), so the weights are the same; the validator now compares every width with the stage-1 manifest of the same arm (`merge_recorded`; the freeze file wins, conflicts are reported). The failure shows that the patched-loader path reaches `prebuild` from the new entry point, but nothing beyond it. Resubmitted as **45796** (`sc-smoke-m2`, `FROZEN=1,RECORD=1`); the failed run's directory `sc-smoke-m` is kept.

**Timing for the full jobs.** The smoke pass took 39 s for 3 conditions (FP, INT6, INT0) x 120 sentences, i.e. at most 0.11 s per sentence and condition (an upper bound: it includes CUDA warm-up). Scaled to 27 languages x 1,012 sentences = 27,324 sentences: at most about 3,000 s per condition, about 4.1 h for the legacy job (FP + 16, 6, 4, 0), 3.3 h of passes for each patched job plus 2 builds (about 15 and 19 min). The design's 5 h limit is too close: **the three full jobs get `-t 07:00:00`** (the stage-1 jobs had 10 h). The true per-condition time is expected to be lower (Phase 1 ran about 1,000 s per condition).

