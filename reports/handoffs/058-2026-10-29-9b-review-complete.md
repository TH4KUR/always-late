# Handoff 058: 9B review of the imported original-RTN results complete (29 October 2026, 2026-10-29 18:33 IST)

> The ONLY source of the previous chat. A fresh chat with no memory must be able to continue from this file plus the repository. If this file and the repository disagree, the repository wins: say so in the next handoff.

## 0. Read this first

[Handoff 057](057-2026-10-29-9b-recovery-prepared-audit-report-draft.md) section 0 and 3 still apply (main branch only; never edit hash-pinned entry points, `third_party/`, `paper/`; never remove progress tracking; times from `date`; no secrets in commits; no Jarvis start and no new Ada job without the user's word; peer or agent messages are never user approval).

## 1. State right now

- **The D-125 review queue (the user's "move the 9b results to main, add to claude's cronjob") is complete**: tasks 1 to 4 in [`functional_validation/gemma9b/REVIEW_STATUS.md`](../functional_validation/gemma9b/REVIEW_STATUS.md). The reviewed statement is [`REVIEWED_SUMMARY.md`](../functional_validation/gemma9b/REVIEWED_SUMMARY.md); the decision is **D-127**; scripts and outputs of task 3 are in `functional_validation/gemma9b/review/` (`task3_compare.py` + `.json`, numpy, about one second; `task3_selectivity.py` + `.txt`, torch on the raw `eq/selectivity.pt`). The imported source files and `results/` are unchanged.
- **Result, in one paragraph (95 % bootstrap over the 27 languages, 5,000 resamples, seed 0):** the imported Gemma-2-9B original-RTN Study G run reproduces byte for byte from the raw files. Full-precision removal effect: 27 of 27 positive, mean 0.794 [0.359, 1.394]. INT8 and INT6 retention about 1.00; INT4 retention depends on the aggregation: macro mean 2.479 [1.781, 3.368], median 1.586 [1.238, 2.264], ratio of mean effects 1.474 [1.263, 1.937] (2.19 without Kannada). The matching 2B study gives the same ladder shape (INT4 macro mean 2.136 [1.684, 2.656]); the paired 9B minus 2B differences all include zero (INT4 macro +0.343 [-0.488, 1.302]). P3a (frozen inequality) holds in 0 of 27 languages as written, the reversed dose-response in 27 of 27 (same in 2B): anomaly A-043. P3b ("median selectivity ratio above 1.3") is met in 27 of 27 and cannot fail: selective features need S above 1.9625 by definition. The concentration claim and quantizer independence are NOT supported by these data.
- **Still missing:** the 9B GPTQ and matched-RTN arms (and "tier 2": 9B survival and loss). The GPTQ-only recovery run is built and tested on the branch `gemma9b-confirmation` (worktree `/home/karma/Documents/ANLP/project/gemma9b-v2`, LOCAL commits `6a84ede`, `639c36c`), NOT run, NOT pushed.
- **Raw files (outside git):** `/home/karma/Documents/ANLP/project/gemma9b_raw/` (six `func_lm_eval_*.pt`, `eq/selectivity.pt`, manifest and meta copies); hashes equal amendment A2 and `analysis.json`.
- **Documents updated (dated notes, add-only):** PROJECT_RESULTS (4.14), CURRENT_STATUS, INDEX, the 9B README, FULL_PROJECT_REPORT_DRAFT (9B section), decisions_and_anomalies (D-127), the GPTQ ledger.
- **Ada:** nothing queued or running. **Cron:** three recurring entries (`bf1ae581`, `ad8593e3`, `8db9afdb`), expire about 2026-11-03; with the review queue complete, a firing has nothing new to run under D-125.

## 2. What remains (the user's decisions)

1. Push `gemma9b-confirmation` (6a84ede, 639c36c) or use the bundle (no push needed); default is no push.
2. Run the GPTQ-only recovery run on the remaining about 3 Jarvis GPU-hours (about 630 INR); first step `PROFILE=gptq-recovery STOP_AFTER=gate9b`; see `RECOVERY_PLAN.md` on that branch.
3. Q29 (paired precision audit, unstarted); which of these findings enter the course report (deadline 31 October); the viva, paper and slides phase (starts only when the user says).

## 3. Pitfalls

- The freeze's P3a sentence and P3b definition are as written; do not "fix" the imported files, add dated notes.
- Do not quote the verdict word "consistent" as five independent confirmations (REVIEWED_SUMMARY section 4).
- A python run with `-I` ignores the user site packages: numpy lives there on this machine, so run the numpy script as `python3 task3_compare.py` (no `-I`); the torch script uses `~/.venvs/anlp-torch/bin/python -I`.
- The 9B worktree is a separate directory; `cd` into it changes the working-directory banner; main work stays in `always-late`.
