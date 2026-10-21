# Handoff 015: D-035 plan verified, recurring job set, HF card updated (2026-10-21, about 06:00 IST)

> The ONLY source of the previous chat from Claude's side (handoffs 010-014 are Codex's). If this file and the repository disagree, the repository wins: say so in the next handoff.

## 0. Read this first (standing rules; user's words where quoted)

- Project: ANLP "Does Compression Forget Language?", repo `/home/karma/Documents/ANLP/project/always-late` (Gemma-2-2B, layer-12 Gemma Scope 16k SAE, RTN quantization, 27 FLORES-200 languages). Submission 2026-10-31.
- Codex (she/her) is a separate AI agent in the same repo. Channel: `communication/` (`messages/`, `replies/`, rules in `communication/README.md`). She owns the review file, register, PLAN/PROGRESS/HANDOFF/README, progress log, reassessment record, research-source register and handoffs 010-014. I never edit them; I send corrections as files in `communication/replies/`. I own `reports/association_retention/`, `scripts/g0/phase1_association_*.py` (to be created), their tests, the optional collector in `scripts/g0/phase1_run.py`, `reports/g0/hf_dataset_card.md`. Stage and commit explicit paths only (never `git add -A`, reset, checkout, stash): she has uncommitted work.
- User's latest instructions (2026-10-21): "ive answered the pressing questions in DECISIONS_FOR_REVIEW.md. lets proceed now, but first look at what codex said, and the other documentation updates as well. i have added some other confounding directions to look at for this project. after that setup a recurring job for every 20 minutes to continue the work. if there is anything you need from me ask, but try to work with the answers i have given right now first"; then: "also first ensure that the recent plan codex has added is factually correct and not already done in any other papers, or overexaggerated before starting implementation and queueing jobs. you can use ada too".
- Earlier rules still in force: mask regime only, no `none` work, no numerical comparison with Duan (D-028); replicate sources as the source did and cite the paper section or code line for every method choice; do not monitor jobs; big artifacts go to the private HF dataset `karma-skz/always-late-artifacts`, wandb fixed schema (project `anlp-project`, entity `karma-skz-`); never put secrets, weights or large files in files, commits or `communication/`; never edit `paper/` or `third_party/`; secret scan before each commit; commit messages end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`. The earlier "no cron, no second chat" rule is superseded by the user's request for the 20-minute job.

## 1. Mission in five lines

Selected study D-035: do the language-selective SAE features selected before quantization (dev split) keep their language association (full 27-language firing/mean profile, peak language, target share, margin) after RTN INT8/7/6/5/4 on matched devtest inputs, with INT0/INT16/INT12 as controls? Plan: `docs/superpowers/plans/2026-10-21-language-association-retention.md` (Tasks 1-6), design: `reports/survival_method_reassessment.md` section 7. It is a measurement-validation step, not a novelty-certified contribution (D-029). Historical Pearson results stay untouched.

## 2. State right now

- Git: HEAD `ac42dcd` (pushed to `origin/main`), previous `df811ba`. The commit holds only my files: `reports/association_retention/{plan_verification.md,STATUS.md}`, `reports/g0/hf_dataset_card.md`, my three files in `communication/replies/`. Codex's 11 modified tracked files and her untracked files are untouched.
- Tests: not run this step (no code changed). Commands: `python3 -B -m unittest discover -s scripts/g0/tests` (161 tests, passed earlier today) and `~/.venvs/anlp-torch/bin/python -B -m unittest discover -s scripts/g0/tests_torch` (268).
- Ada: `squeue` empty; `/home2` 7.9 GB free (85 % used); `out/phase1/p1-mask/eq` holds `meta.json, per_language.csv, selectivity.pt, summary.csv, survival.pt`.
- Hugging Face: the dataset is **private**; card rewritten and uploaded (commit `43026509`); pre-publication scan of the 274 files clean (only `/home2/karma/duan-verify/...` paths). The step making it public was refused by the tool permission layer; it was not retried.
- Recurring job: cron `56dd5dd0`, `7,27,47 * * * *`, session-only, expires after 7 days; each firing does the NEXT STEP in `reports/association_retention/STATUS.md`. Check `CronList` after a restart; if the job is gone, recreate it with `CronCreate` from the prompt copied in `reports/association_retention/cron_prompt.md`.

## 3. What this step did

- Read Codex's Q13-Q15 message, handoffs 013 and 014, the register, the reassessment and the plan.
- Verified the plan: [`reports/association_retention/plan_verification.md`](../association_retention/plan_verification.md). Code and data claims correct (numbers re-read from the committed files); cohort facts (8,086 selective, 2,860 controls, union = 10,946 supported; 6,663 of 8,086 meet the baseline conditions on devtest FP); prior art (no paper found for this study; closest Duan 2606.03002, Gupte 2507.15977, Kurz TACL 2026, Verma ACL 2026, Borobia; UniComp contradicts the tier effect); literature-register corrections for Codex; plan-level technical corrections (subset smoke masks differ, storage, GPU memory, ties, the "no cron" constraint).
- Five subagents (literature audit x3, prior-art search x2) did the source reading; their reports are model output, re-checked by me only where stated in the verification file. Two of my own follow-up fetches were refused by the permission layer and not worked around.
- Sent `communication/replies/2026-10-21-claude-to-codex-q13-q15-ack-and-verification.md`.
- Updated memory files (cron wanted, Codex state).

## 4. The next step, exactly

Task 1 step A of the plan: write `scripts/g0/tests/test_phase1_association_inventory.py` (fixtures: FP-only response statistics, all-language correlations without quantized profiles, complete new profiles, mismatched language order), watch it fail, then implement `scripts/g0/phase1_association_inventory.py`; then generate `reports/association_retention/artifact_inventory.md` and the frozen-input manifest from the saved `mask/eq` files and the Ada listing. Source: plan Task 1 and design section 7. "Done": tests green and mutation-checked, the inventory states available / derivable / new-collection-required for every quantity, hashes of `selectivity.pt` and `survival.pt` recorded, cohort and control sizes recorded, uninspected locations listed. After that: Task 3, Task 2, Task 4; no GPU until the gates in STATUS.md pass.

## 5. Open questions for the user

- Make the HF dataset public (Q10 yes; permission layer refused; card ready; scan clean): needs the user to allow it or run `HfApi().update_repo_settings(repo_id="karma-skz/always-late-artifacts", repo_type="dataset", private=False)` with the token on Ada.
- Licence of our own statistics in the dataset (card says `license: other`; none chosen).
- Q6, Q7, Q8, Q11, Q12 remain unanswered (Codex's review); not blocking.

## 6. Environment and pitfalls

- Local environment: `~/.venvs/anlp-torch`; cached run files `~/.cache/anlp-always-late/phase1_runs/p1-mask/{eq,all}`; byte-identity checks need `OMP_NUM_THREADS=4`. Scratch scripts for this step live in `/tmp/claude-1000/-home-karma-Documents-ANLP-project/239e7a33-140d-4688-8581-25c246ab3668/scratchpad/` (not committed): a first baseline-establishment script had an indexing bug (it indexed features 0..n-1 instead of the selective ids; gave 31.9 %); the corrected figure is 82.4 %.
- Ada: `ssh karma@ada-gw1`, project `/home2/karma/duan-verify` (flat layout: `hf_artifacts.py`, `hf_artifact_spec.json` at top level); token path `hf/token` (never print; `HF_TOKEN_PATH=/home2/karma/duan-verify/hf/token`).
- Smoke `--languages` runs change language indices and the equal-budget minimum: compare by language code, never reuse the full-run masks for a subset.
- The working directory of the shell can drift after `cd`: use absolute paths.

## 7. What I would check if something looks wrong

`git log --oneline -3`, `git status --short` (Codex's files should be the only modified ones plus mine), `CronList`, `ls communication/messages communication/replies`, `ssh karma@ada-gw1 squeue -u karma`, `python3 scripts/g0/phase1_summary.py --check`.
