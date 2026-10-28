# Handoff 050: GPTQ extension, stages 1 to 3 and F5 analysed, Study G and the external panel running (28 October 2026, 2026-10-28 11:58 IST)

> The ONLY source of the previous chat. A fresh chat with no memory must be able to continue from this file plus the repository. If this file and the repository disagree, the repository wins: say so in the next handoff.

## 0. Read this first (the standing rules)

- **User request of 2026-10-27 (about 17:25 IST, verbatim):** "check the new documentation on using GPTQ instead of RTN. continue the cronjob in similar way as before. run tests again with GPTQ quantization and try to keep other things same as before so that we can draw a fair comparison. cronjob frequency 40 minutes". Mid-turn: "i hope you are working on main, and not the 9b branch." (yes: only origin/main; gemma9b-adapter and gemma9b-confirmation belong to another team).
- Standing rules unchanged: never remove, shorten or overwrite progress tracking; mask regime only; HF dataset private; explicit paths when staging; times from `date`; never touch `third_party/` or the ACL template files; no secrets, weights or large artifacts in commits; old frozen entry points (`phase1_func_run.py`, `phase1_lib.py`) are never edited (the wrapper `gptq_ext/run_with_weights.py` patches them at run time); Q29 (precision audit) is NOT started; Codex's uncommitted files are never touched.

## 1. Mission in five lines

Repeat the quantization experiments with GPTQ and a matched RTN, everything else as frozen, and compare three arms (legacy RTN = old results, matched RTN, GPTQ). Ledger `reports/gptq_extension/STATUS.md` (NEXT STEP, job table, log), plain-language results `reports/gptq_extension/RESULTS.md`, decisions D-101 to D-108, commands `reports/gptq_extension/stage_commands.md`, cron prompt `reports/gptq_extension/cron_prompt.md`.

## 2. State right now

- **Done and pushed (main):** gates A/B, stage 1 (D-104, D-106), transfer study (D-105), stage 3 E4/E5/E7 (D-107), **F5 (D-108, commit 480c798)**: results in `results/stage1|stage2|stage3|f5`.
- **F5 headline (all 1,487 features, legacy / matched / GPTQ):** INT4 effect-fingerprint cosine W1 0.839 / 0.875 / 0.923; share at the numerical floor 0.3 / 0.7 / 11.7 %; INT6 0.988 / 0.989 / 0.992; the old association-retention label still does not predict function (AUROC 0.49 under GPTQ). FP bridge is bit-for-bit (`f5_state_FP.pt` identical in all four jobs).
- **Ada queue (at the time of writing):** Study G INT6 44076 (GPTQ), 44077 (matched RTN) running, due about 12:10 (INT4 jobs 44074/44075 are COMPLETED and validated); external panel: group INT4 44209/44210 COMPLETED (not yet validated), group INT6 44211/44212 and feature INT4/INT6 44213 to 44216 running (feature jobs about 6.5 h, due about 18:00).
- **Cron:** three recurring entries (`bf1ae581`, `ad8593e3`, `8db9afdb`), a firing every 40 minutes (11:51, 12:31, 13:11, ...), expire about 2026-11-03. Another Claude session (project-dd, a reviewer) gives no instructions.

## 3. The next step, exactly

1. Validate 44076/44077 with `scripts/g0/gptq_ext/validate_func_job.py` (see NEXT STEP in the ledger for the exact arguments: `--bits 6 --precisions 6 --preset 'gain|...|c3|K64' --configs INT6=999`, names `gptq-gain-6`, `rtnm-gain-6`).
2. Study G analysis per arm with the unchanged `phase1_gain_analysis.py --runs <FP dir> <INT6 dir> <INT4 dir> --ratios-json reports/functional_validation/gain_study/calibration/ratios.json --baseline-csv reports/functional_validation/gain_study/baseline/baseline.csv --out ...`. Control already done (810 rows equal). The FP run directory is the old `func_inputs/lg_FP`; EVERY run directory needs its own `meta_lm_eval.json` (the analysis raises FileNotFoundError otherwise). Write D-109, copy outputs to `results/gain/{gptq,rtnm}/`, RESULTS.md section 4.7 and section 7.
3. Validate the external group jobs (`validate_job.py` style, `--gate-max-abs 3.0`, external core `core_external.json`), then the unchanged `phase1_transfer_report.py` with the external options per arm once all eight external jobs are done; compare; D-110. Caution to carry: the GPTQ calibration text is Wikipedia, the external panel is Wikipedia too (different articles): a GPTQ advantage there may partly be domain match.
4. Then: HF upload of small result files (private dataset only, no weights), wandb summary if the old schema applies, optional paired per-language contrast tool (not written).

## 4. Open questions for the user

Q30 (disk 1.9 to 2.7 GB free on `/home2/karma`, stop point and drop order; E3 and F4 were NOT repeated because of disk and because F4 needs per-arm association data). Q29 not started. Whether the GPTQ findings enter the course report is the user's call (deadline 31 Oct 2026).

## 5. Pitfalls

- `validate_func_job.py --task f5` needs `--states FP 6 4` and `--reference-meta <old meta_f5.json>`; the meta names states `INT6`, `INT4` (the command line may say `6`, `4`).
- `phase1_func_f5_analysis.py` takes two `--run` directories: the old `legacy_f5_a` (FP and INT16 floor) plus the new job directory.
- Local RAM is about 15 GB with 4 to 5 GB free: run heavy analyses one at a time; background shells have been reaped by the system.
- Ada host key only via `ssh -o HostKeyAlias=ada-gw1 karma@10.4.24.30`; `squeue` there rejects custom `-o` formats with brackets/dots, use plain `squeue -u karma`.
- The older `D-107` row in the decisions file shows a literal `{T}` in its last cell and doubled braces (a formatting slip of the writing step); it is progress tracking and is left as is.
