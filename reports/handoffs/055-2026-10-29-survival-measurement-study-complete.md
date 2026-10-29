# Handoff 055: survival measurement-validity study complete; all compute finished (29 October 2026, 2026-10-29 12:35 IST)

> The ONLY source of the previous chat. A fresh chat with no memory must be able to continue from this file plus the repository. If this file and the repository disagree, the repository wins: say so in the next handoff.

## 0. Read this first

[Handoff 054](054-2026-10-29-wandb-extension-uploaded-all-authorized-work-done.md) section 0 and [053](053-2026-10-28-survival-and-intervention-plan-complete.md) section 0 still apply in full (main branch only; never edit hash-pinned entry points or `third_party/`; never remove progress tracking; times from `date`; no secrets or weights in commits; no 48-hour window ever existed). **New user instructions of 2026-10-29, verbatim:** (D-121) "approved, but report results honestly. the 10% improvement isnt backed by any actual claims so any improvement is reportable and good"; (D-122) "just ensure that all the jobs run eventually. i need to be done with the project (except analysis and reports) by today EOD. after that we will start working on preparing for the viva and for looking at the work done, preparing the final submission paper, and slides accordingly."

## 1. State right now

- **Done:** the D-117 study, all stages: plan (D-117), Stage 1 (D-118), Stage 2 design (D-119), sidecar code and tests, three full Ada jobs 45799 (legacy, 2 h 01), 45836 (matched, 2 h 12), 45837 (GPTQ, 2 h 13) validated (`survival_validate.py`; the committed comparison G2(ii) passed in every arm at float32 resolution; weights hashes equal to stage 1), raw sums on the private HF dataset (commit `7291ebf99b`, 22 files), Stage 3 freeze (D-120), generator reviewed (D-123), the single frozen run, results in plain words (D-124): [`reports/survival_measurement/stage3/README.md`](../survival_measurement/stage3/README.md); `PROJECT_RESULTS.md` section 4.13; `CURRENT_STATUS.md`.
- **Ada:** nothing queued or running; `/home2` 2.2 GB free (nothing deleted). **All project compute is finished (D-122).** **Cron:** three recurring entries (`bf1ae581`, `ad8593e3`, `8db9afdb`) every 40 minutes, expire about 2026-11-03; idle firings only append one ledger line to `reports/gptq_extension/STATUS.md`.
- Tests: `python3 -B -m unittest discover -s scripts/g0/tests` (477) and `OMP_NUM_THREADS=4 ~/.venvs/anlp-torch/bin/python -B -m unittest discover -s scripts/g0/tests_torch -p 'test_*.py'` (about 2,600; 54 skipped as before); the Stage 3 generator alone: `scripts/g0/tests_torch/test_survival_stage3.py` (53), sidecar 49, validator 11. Mutation records in `reports/survival_measurement/stage2/` and `stage3/`.

## 2. What remains: the user's decisions only

Q29 (paired precision audit: built, never deployed, not authorized; flagged to the user in D-122: GPU time after today is excluded unless the user says so); whether any finding enters the course report (deadline 31 October); the user's own next phase: viva preparation, review of the work done, final paper, slides (not started; wait for the user). Q32 stays open (default carried out).

## 3. Pitfalls

- The generated Stage 3 report is `stage3_results.md`; the hand-written `stage3/README.md` follows the D-121 wording (every improvement with its interval; thresholds are conventions). The H4 labels depend on the standardization: state raw slopes and nu medians by tier first.
- scp from Ada is slow (about 110 KB/s per stream): fetch in the background; the HF upload runs on Ada (token stays there; never run `hf_artifacts.py upload` for a new spec without `--manifest-name` and `--no-card`).
- `FROZEN=1` alone refuses a bit width without a recorded hash (`artifact_hashes.json` holds INT4 only): stage jobs use `FROZEN=1,RECORD=1`.
- The mutation tool edits the source in place while it runs: never edit or commit that file during a battery.
