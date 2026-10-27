# Handoff 048: GPTQ extension started, plan and ledger written, cron recreated (27 October 2026, 2026-10-27 17:34 IST)

> The ONLY source of the previous chat. A fresh chat with no memory must be able to continue from this file plus the repository. If this file and the repository disagree, the repository wins: say so in the next handoff.

## 0. Read this first (the standing rules)

- **User request of 2026-10-27 (about 17:25 IST, verbatim):** "check the new documentation on using GPTQ instead of RTN. continue the cronjob in similar way as before. run tests again with GPTQ quantization and try to keep other things same as before so that we can draw a fair comparison. cronjob frequency 40 minutes"
- This lifts the HOLD of 2026-10-27 09:50 **for the GPTQ extension only** (D-101). Standing rules unchanged: never remove, shorten or overwrite progress tracking (add only, or move verbatim to `reports/archive/`); mask regime only; HF dataset private; stage explicit paths; times from `date`; never touch `third_party/` or the ACL template files; no secrets, weights or large artifacts in commits; Gemma-2-9B is another team's; the user's `plan02-train` jobs are never touched; Ada host key only via `ssh -o HostKeyAlias=ada-gw1 karma@10.4.24.30`.
- Earlier standing instructions that still hold: 2026-10-26 15:37 the 32 GPU-hour ceiling is soft; 2026-10-25 queue all planned jobs at once; 2026-10-24 sample-size floors (D-075) for the old studies.

## 1. Mission in five lines

Repeat the project's quantization experiments with GPTQ-chosen weights and with a matched RTN comparator, changing nothing else, to learn whether the old RTN findings depend on the quantizer. Documentation: the audit `reports/quantization_gptq_audit_2026-10-27.md` (feasible in design; not verified on Gemma/GPU). Plan: `docs/superpowers/plans/2026-10-27-gptq-matched-comparison.md` (14 tasks, fixed stage order). Ledger and log: `reports/gptq_extension/STATUS.md`. Cron: `reports/gptq_extension/cron_prompt.md`.

## 2. State right now

- Task 0 (governance) is done: D-101, Q30, plan, ledger, cron prompt, this handoff. **No code and no Ada job yet.** Ada was queried once: queue empty.
- Facts found: five entry points call the injected `env["quantize_rtn_per_output_channel_tl"](model, bits=...)` (`phase1_run.py:312`, `phase1_func_run.py:484,619`, `phase1_func_f5_run.py:386`, `phase1_transfer_run.py:477`, `phase1_transfer_precision_main.py:175`), so one shim covers them; local environment has torch (CPU) but no TransformerLens; Ada `/home2/karma` has 2.7 GB free of 50 GB (artifacts of about 4 GB each therefore cannot be stored: rebuilt per job and hash-checked); compute nodes have node-local `/scratch`, no `/share1`; the login node has internet and `datasets`.
- Cron: see the ledger log for the entry ids (CronList is authoritative).

## 3. The next step, exactly

Task 1 of the plan: vendor the authors' pinned GPTQ core into `scripts/g0/gptq_ext/upstream/` (hashes in the audit's `evidence.json`), tests first, mutation check, commit. Then Tasks 2 to 8 in order; no Ada job before Task 8's gates; then queue stage by stage.

## 4. Open questions for the user

Q30 (disk and scope; default carried out) in `DECISIONS_FOR_REVIEW.md`; Q29 (precision audit R5) stays open and is NOT started. Possible questions to come: a calibration shortfall for a low-resource language (Task 4), different artifact hashes on two nodes (Task 8).

## 5. Pitfalls

- The legacy quantizer shares one Q/K/V scale across heads and uses a different grid than the authors' quantizer; compare GPTQ with matched RTN, not only with legacy RTN (audit section 3).
- INT16/INT12/INT0 stay the legacy rounding and restore controls in every arm; never replace them by a low-bit GPTQ state.
- Use absolute rescue and NLL effects before rescue percentages if GPTQ shrinks the original degradation (audit section 7).
- Run model tests only where TransformerLens exists (Task 2's local CPU environment or Ada); no job on Ada before the plan's Task 8 gates.
