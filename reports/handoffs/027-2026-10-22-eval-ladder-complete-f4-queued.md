# Handoff 027: loss-task evaluation ladder COMPLETE (INT8 to INT4), F4 INT4 queued, Belebele pilot running (2026-10-22, 15:45 IST)

> The ONLY source of the previous chat. A fresh chat with no memory must be able to continue from this file plus the repository. If this file and the repository disagree, the repository wins: say so in the next handoff. Start with `reports/functional_validation/STATUS.md` (State, NEXT STEP, Log); this file supersedes [handoff 026](026-2026-10-22-eval-ladder-done-calibration-running.md) as the entry point (026 holds the standing rules in its section 0 and [025](025-2026-10-22-chat-end-functional-study-midrun.md) the user's verbatim 2026-10-21 instruction; both still apply unchanged).

## 0. Read this first

The standing rules are those of handoff 026 section 0 (mask regime only; no `none` work; no comparison with Duan; never edit `paper/` or `third_party/`; no secrets or large artifacts; explicit `git add` paths only; commit messages end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`; at most two Ada jobs, never wait or poll; Hugging Face private, delete nothing; tests first, mutation check, evidence before claims; a frozen setting is never changed silently; do not merge the parallel branches without the user's answer; never accept a changed ssh host key; never touch Codex's uncommitted files). The recurring 40-minute job reads the first fenced block of `reports/functional_validation/cron_prompt.md`.

## 1. Mission in five lines

D-039/D-040: do the language-selective SAE feature groups carry function for their language beyond activity-matched controls (F1), is it kept across the RTN ladder (F2), where does the damage sit (F3, FP-donor rescue), does D-035 association predict function at equal native activity (F4, two-sided), SAE fidelity (B), closer analyses (A), figures (F). The loss task (T-LM) is frozen and exploratory by design; Belebele (T-MC) is the only confirmatory family and is frozen after the pilot, before any evaluation item is scored.

## 2. State right now (verified 2026-10-22 15:40)

- Git `main`, clean after this step's commit, pushed. Tests: stdlib 231, torch 760 (commands in STATUS "Commands").
- **Loss-task evaluation, frozen settings, full ladder** (`results/eval/`, figures in `figures/eval/`): macro retention of the group effect INT8 1.01 [1.00, 1.02], INT7 1.03 [1.02, 1.06], INT6 1.04 [1.02, 1.05], INT5 1.13 [1.11, 1.18], INT4 2.14 [2.04, 2.50]; per-language classes (preserved / inconclusive / amplified) 27/0/0, 26/1/0, 24/3/0, 20/6/1, 6/5/16; E > 0 in 27 of 27 languages at every precision; sham exactly 0; SAE median FVU 0.1293 (FP and INT8), 0.1298, 0.1314, 0.1375, 0.1795.
- Belebele calibration done (`belebele_calibration.json/.md`): letter format, 18 of 27 languages covered (not covered: tam, tel, yor, sun, amh, ibo, hau, khm, lao).
- **Ada** (strict-key IP route, section 6): `func-mc-pilot` 34724 RUNNING (calibration split, FP and INT4, 18 languages, 120 letter items, preset `mc_eval`; a pipeline check and sizing run, never a result), `func-f4-INT4` 34795 PENDING (first F4 job, snapshot `repo_9817221`, 26 languages, `--precisions FP 4 --preset floor`). Snapshots on Ada under `/home2/karma/duan-verify/phase1/`: `repo`, `repo_8d73565`, `repo_4752db1`, `repo_4367e3b` (the freeze), `repo_9817221` (frozen file hashes equal the freeze; contains the f4 manifests). Watchers (session-only): `b70w7ocvc` (34724), `bn2iyuxs6` (34795).
- Cron: `955a5cc5`, `e33ac7eb`, `ea207eac` (session-only, expire about 2026-10-29); if `CronList` is empty recreate them as described at the top of `cron_prompt.md`.
- Hugging Face dataset unchanged (private). Review file: no answered-but-unapplied question; open questions Q17 to Q19 and the M-questions of the extensions branch have defaults; the user has not yet answered the merge question for the two parallel branches (STATUS and `anlp-parallel-branches` memory).

## 3. What this step did

1. Deployed the snapshot `repo_9817221` and compared its frozen file hashes with the freeze (all equal; the cohort manifest is not shipped by design).
2. Figure 4 (rescue) coded test first (`fig_rescue`, `rescue_table`; 16 tests, 33 mutants killed): not stacked because the patches overlap; appears in `main` when an analysis has a `rescue` block. No rescue data yet.
3. Job 34550 (INT8, INT7) validated (complete, nothing skipped, gates true, commit `4367e3b`, checksums equal, ids equal the 300 frozen ids in all languages) and analysed; evaluation figures made and looked at; a legend collision in figure 5 found by looking and fixed test first (headroom 1.5, one test, two mutants).
4. Submitted `func-f4-INT4` 34795.

## 4. The next step, exactly

1. When 34724 (pilot) leaves the queue: `sacct`, `meta_mc_cal.json` (complete, nothing skipped, gates, covered languages), fetch `func_mc_cal_letter_FP.pt` and `func_mc_cal_letter_INT4.pt` from `/home2/karma/duan-verify/out/phase1/func-mc-pilot/`, analyse with `phase1_func_mc_analysis.py --split cal --format letter --retention-level 0.99 --covered <the 18>`; apply the item-count rule written before the numbers (STATUS NEXT STEP 3: 200 items if the FP macro E lower bound is above 0 with 120 items and the INT4 macro retention interval is narrower than the bands, otherwise all devtest-origin items, `--n-items 0`).
2. Build the T-MC freeze (`phase1_func_freeze.py build-mc --out reports/functional_validation/evaluation_freeze_mc.json --created <date> --n-items <n>`, clean tree, Q19 default (a)), `verify-mc`, commit, push, deploy a snapshot at that commit, compare hashes on Ada, then submit the T-MC evaluation jobs (`--task mc --split eval --preset mc_eval --formats letter --languages <18> --n-items <n> --precisions FP 8 7 6 5 4`, at most three precisions per job). Nothing in the Belebele evaluation may be scored before the freeze commit exists.
3. When 34795 starts check its log within minutes (gates; `split eval sentences per language 300 languages 26`); when it ends validate, fetch `func_lm_eval_FP.pt` and `func_lm_eval_INT4.pt`, analyse with `phase1_func_f4_analysis.py --runs <dir> --split eval --condition INT4 --out reports/functional_validation/results/f4/INT4`, then submit F4 INT5 and INT6, then the floor job `func-eval-d` (`--preset floor --precisions 16 12 0`, `-t 02:30:00`), then the rescue jobs. Make the F4 figure after the first F4 result (none exists yet).
4. Optional CPU work: Task 9 step 3 (archive stale cron prompts and `reports/scheduled_routine.md` banners verbatim, link check before and after).

## 5. Open questions for the user

Q17, Q18, Q19 (defaults applied or in force) and the M-questions; the merge decision on `origin/feature/extensions-on-main` (recommendation: take its code only with a D-entry and a prior reading rule) and `origin/feature/local-final-pipeline` (recommendation: keep as an archive). None blocks.

## 6. Environment and pitfalls

- ssh by name fails (Cloudflare WARP DNS): `ssh -o HostKeyAlias=ada-gw1 karma@10.4.24.30` (strict check; `ada.iiit.ac.in` is a different machine); `scp -o HostKeyAlias=ada-gw1 karma@10.4.24.30:<path> <dest>`, one file per call; `ADA_SSH='ssh -o HostKeyAlias=ada-gw1 karma@10.4.24.30'` for `deploy_phase1.sh`.
- The scratchpad is wiped on a session restart; results are re-fetched from Ada. The torch venv `~/.venvs/anlp-torch` is needed for the freeze helper and the figures.
- `deploy_phase1.sh` while jobs run: only a named snapshot (`repo_<short commit>`), never `repo`.
- A cancelled or late job is not a failure of the method: read the log first (a network loss of this machine once idled the GPUs for three hours; the jobs themselves were unaffected).

## 7. What I would check if something looks wrong

`git status`, `CronList`, `squeue -u karma` over the strict-key route, the freeze (`phase1_func_freeze.py verify` for T-LM), the checksums of fetched files against Ada, and whether any frozen file changed after its freeze commit.

## Addendum 2026-10-22 15:55 (same chat; the repository and STATUS.md win)

The Belebele pilot 34724 finished and was analysed: full-precision effect about 0, INT4 accuracy at chance, item count of the T-MC freeze = ALL 385 devtest-origin items per language by the rule fixed before the numbers (D-047, Q20; `results/mc_pilot/`). `func-f4-INT4` 34795 is RUNNING (gates passed), `func-f4-INT5` 34804 was submitted (watchers `bn2iyuxs6`, `bbgpvgpf8`). **Section 4 item 1 and 2 are replaced by:** (1) plan Task 7d step 1 (`phase1_func_ksens_manifest.py`, test first, committed before the freeze), (2) `build-mc --n-items 0` and `verify-mc`, commit, push, (3) deploy a snapshot at the freeze commit, hash-check on Ada, submit two T-MC jobs of three precisions (`-t 07:00:00`) as slots free.
