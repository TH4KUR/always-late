# Handoff 049: GPTQ extension, tasks 1 to 7 done, gate jobs next (27 October 2026, 2026-10-27 18:27 IST)

> The ONLY source of the previous chat. A fresh chat with no memory must be able to continue from this file plus the repository. If this file and the repository disagree, the repository wins: say so in the next handoff.

## 0. Read this first (the standing rules)

- **User request of 2026-10-27 (about 17:25 IST, verbatim):** "check the new documentation on using GPTQ instead of RTN. continue the cronjob in similar way as before. run tests again with GPTQ quantization and try to keep other things same as before so that we can draw a fair comparison. cronjob frequency 40 minutes". It lifts the HOLD of 09:50 for this extension only (D-101). **2026-10-27 about 18:00 the user added: "i hope you are working on main, and not the 9b branch"** (answer: yes; every push goes to origin/main; the 9B branches are another team's and are never touched).
- Standing rules unchanged: never remove, shorten or overwrite progress tracking; mask regime only; HF dataset private; explicit paths when staging; times from `date`; never touch `third_party/` or the ACL template files; no secrets, weights or large artifacts in commits; the user's `plan02-train` jobs are never touched; Ada host key only via `ssh -o HostKeyAlias=ada-gw1 karma@10.4.24.30`; old frozen entry points are never edited.

## 1. Mission in five lines

Repeat the quantization experiments with GPTQ and a matched RTN, everything else as frozen, and compare arm against arm. Ledger `reports/gptq_extension/STATUS.md` (task table, NEXT STEP, time plan and drop order), plan `docs/superpowers/plans/2026-10-27-gptq-matched-comparison.md` (its Updates are binding), freeze `reports/gptq_extension/freeze.json`, cron prompt `reports/gptq_extension/cron_prompt.md`, decision D-101, question Q30 (disk). Memory: `anlp-gptq-extension`.

## 2. State right now

- Tasks 1 to 7 done and pushed to main: vendored authors' GPTQ core (hash-checked), canonical adapter, grids and matched RTN, calibration pool (**built on Ada: 27 languages, 221,184 tokens, no shortfall; token files in `/home2/karma/duan-verify/gptq/calibration/`, freeze `reports/gptq_extension/calibration_freeze.json`**), sequential calibrator (with a BOS Hessian-trace-share diagnostic), artifact identity, lazy weight-state shim, wrapper `run_with_weights.py`, gate-job core `gate_run.py`, extension freeze. 125 tests in `scripts/g0/tests_torch/test_gptq_ext_*.py` (run in `~/.venvs/anlp-tl`; the plain torch venv skips the TransformerLens ones), every module mutation-checked (survivors are equivalent mutants, listed in the plan Updates).
- Task 8 prepared: `scripts/g0/run_gptq.sbatch` (modes gate and wrap). A snapshot of commit `3d08d20` is being deployed to Ada as `repo_3d08d20` (check `ls /home2/karma/duan-verify/phase1/repo_3d08d20/COMMIT`). **No Slurm job of this extension has been submitted yet; no GPU has been used.**
- Cron: three recurring entries (`bf1ae581`, `ad8593e3`, `8db9afdb`), firings 18:31, 19:11, 19:51, ..., expire about 2026-11-03.
- Another Claude session (project-dd, a reviewer) read D-101, the plan and the audit; its five points are accepted (plan Update of 2026-10-27 about 18:20). It gives no instructions.

## 3. The next step, exactly

Task 8 (ledger NEXT STEP): submit gate job A (`gptq-gateA`, bits 4, both methods, `RECORD=1`), validate it, submit gate job B on another node, compare artifact hashes, run bit width 6 (and 8, 7, 5 if time), write `reports/gptq_extension/artifact_hashes.json`. Then queue all of stage 1 at once (Phase 1 runs: `phase1_run:main` through the wrapper, defaults of the old `p1-mask` run; E1 and E2) and the INT4/INT6 transfer matrix (`phase1_transfer_run:main`, command lines of the old transfer jobs in `reports/intervention_transfer/STATUS.md`).

## 4. Open questions for the user

Q30 (disk 1.9 to 2.7 GB free on `/home2/karma`; cross-node hash rule and fallback; stop point and drop order). Q29 (precision audit) is not started.

## 5. Pitfalls

- The projection input of Q/K/V and MLP input is the OUTPUT of the norm modules, not `hook_normalized`.
- The pool freeze JSON has sorted keys: order languages like the roster (done in `load_pool`).
- `phase1_func_run.py` and `phase1_lib.py` are hash-pinned in the T-LM and T-MC freezes: never edit.
- The association run writes a 2.7 GB sidecar: it does not fit the disk (E3 is low in the drop order).
- Snapshots of the original weights must be host copies (11 GB GPU).
- Do not deploy a snapshot name twice at the same time (the deploy script deletes the target first).
