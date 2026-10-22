# Handoff 028: the confirmatory Belebele family analysed (S0 not met, S1 to S5 undefined), F4 complete, floor and rescue jobs running (2026-10-23, 01:20 IST)

> The ONLY source of the previous chat. A fresh chat with no memory must be able to continue from this file plus the repository. If this file and the repository disagree, the repository wins: say so in the next handoff. Start with `reports/functional_validation/STATUS.md` (State, NEXT STEP, Log); this file supersedes [handoff 027](027-2026-10-22-eval-ladder-complete-f4-queued.md) as the entry point ([026](026-2026-10-22-eval-ladder-done-calibration-running.md) holds the standing rules in its section 0 and [025](025-2026-10-22-chat-end-functional-study-midrun.md) the user's verbatim 2026-10-21 instruction; both still apply unchanged).

## 0. Read this first

The standing rules are those of handoff 026 section 0 (mask regime only; no `none` work; no comparison with Duan; never edit `paper/` or `third_party/`; no secrets or large artifacts; explicit `git add` paths only; commit messages end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`; at most two Ada jobs, never wait or poll; Hugging Face private, delete nothing; tests first, mutation check, evidence before claims; a frozen setting is never changed silently; do not merge the parallel branches without the user's answer; never accept a changed ssh host key; never touch Codex's uncommitted files). The recurring 40-minute job reads the first fenced block of `reports/functional_validation/cron_prompt.md`. **Two working rules learned on 2026-10-22:** write documentation edits as a script FILE or with a quoted heredoc (an unquoted heredoc executes backtick text) and read the edited lines back before committing; take every timestamp from `date`, never from a guess.

## 1. Mission in five lines

D-039/D-040: do the language-selective SAE feature groups carry function for their language beyond activity-matched controls (F1), is it kept across the RTN ladder (F2), where does the damage sit (F3, FP-donor rescue), does D-035 association predict function at equal native activity (F4), SAE fidelity (B), closer analyses (A), figures (F). The loss task (T-LM) is frozen and exploratory by design; Belebele (T-MC) is the only confirmatory family and was frozen before any item was scored.

## 2. State right now (verified 2026-10-23 01:16)

- Git `main`, pushed. Tests: stdlib 265, torch 810 (commands in STATUS "Commands").
- **Loss task, held-out, frozen settings (exploratory):** macro retention of the group effect INT8 1.01, INT7 1.03, INT6 1.04, INT5 1.13, INT4 2.14 (`results/eval/`, `figures/eval/`); E > 0 in 27 of 27 languages at every precision.
- **F4 (association versus function at equal native activity; exploratory), INT4, INT5, INT6 all analysed (D-048, D-049, D-051):** no systematic difference; medians +0.0006, +0.0007, +0.0002 nats per token; the macro mean is one language (Lao) and a median and a mean-without-the-dominant-language were added post hoc, labelled, for every condition.
- **CONFIRMATORY BELEBELE FAMILY (D-050, `results/mc_eval/statements.md`): S0 NOT MET (macro effect at full precision -0.0001 nats per item, 95 % interval [-0.0035, +0.0031]); S1 to S5 UNDEFINED (no covered language has a clearly positive full-precision effect).** 18 covered languages, 385 items each, letter format, K64; accuracy 0.426 without edit, 0.426 group zeroed, 0.425 controls; INT4 0.236 (below chance); sham exactly 0. The result was produced by the frozen chain (analysis with `--retention-level 0.99`, statements tool with its provenance check against the freeze) after every file was validated against the freeze (item-key hashes equal).
- **Ada** (strict-key IP route, section 6): `func-eval-d` 35314 (floor INT16, INT12, INT0; started 01:10, expected about 02:50) and `func-rescue-a` 35319 (rescue K64 at INT6, INT5, INT4 with `--preset floor`; started about 01:12, expected about 04:45); both logs checked: gates passed, the rescue job built its donor cache. Watchers (session-only): `bkrugevgd` (35314), `bo5hl7fur` (35319); `bmafsgna6` belonged to a cancelled job. Snapshots under `/home2/karma/duan-verify/phase1/`: `repo_4367e3b` (T-LM freeze), `repo_9817221` (F4), `repo_2b6baf4` (T-MC freeze code, with `ksens/`).
- Cron: `955a5cc5`, `e33ac7eb`, `ea207eac` (session-only, expire about 2026-10-29); recreate as described at the top of `cron_prompt.md` if `CronList` is empty. The queue was empty about 2.5 hours (22:36 to 01:07) because no firing ran in that window.
- Hugging Face dataset unchanged (private). Review file: no answered-but-unapplied question; Q17 to Q20 and the M-questions have defaults; the user has not answered the merge question for the two parallel branches.

## 3. What this chat did since handoff 027

1. F4 INT4 and INT5 analysed (D-048, D-049), figure 9 coded and tested, with a median row and a legend below the axis; F4 INT6 analysed (D-051) and the extreme value labels fixed (they overlapped the gray full-precision interval).
2. Belebele pilot analysed (full-precision effect about 0, INT4 at chance), item count fixed by the pre-written rule (385, D-047), K = all sensitivity manifest (`ksens/`, Q20), **T-MC freeze committed before any evaluation item was scored (15:43 on 2026-10-22)**; both evaluation jobs run and validated; the statements tool (plan Task 6c) and figure 10 written test first before the data existed and rehearsed on synthetic six-precision files (plan Task 6d); the result is D-050.
3. Figures module: dry run on the pilot found and fixed three defects (figure 1 negative ticks and legend, retention figures skipped when fewer than three languages back the macro, with `figures_skipped.txt`).
4. Documentation consolidation Task 9 step 3 (`scheduled_routine.md` banners archived verbatim).
5. Anomaly A-041: a rescue job submitted with `--preset none` cannot be analysed (`analyse` needs `zero_K64` in every run); cancelled after four minutes, resubmitted with `--preset floor`.

## 4. The next step, exactly

STATUS "Immediate" has the commands. In order: (1) when 35314 ends validate it, fetch `func_lm_eval_INT16.pt`, `INT12.pt`, `INT0.pt`, rerun the loss analysis with all nine precisions into `results/eval/` (if the analysis does not tolerate files with only the K64 families, analyse the floor precisions in a separate folder), regenerate `figures/eval`; (2) when 35319 ends validate it and analyse the rescue into `results/rescue/` (FP from the eval job plus the three rescue files in one folder), make figure 4 and `table_rescue`, look at the render; (3) the exploratory K = all Belebele run (Q20 default (a)) in the first free slot from `repo_2b6baf4` with `--manifest .../ksens/ksens_manifest.json --preset mc_eval --precisions FP 6`, every table labelled "K64 stands for the whole group, exploratory"; (4) the paper-facing summary (the team writes the prose; `paper/` is never edited).

## 5. Open questions for the user

Q17 to Q20 (defaults applied or in force; Q20 default (a) is being carried out), the M-questions of the extensions branch, and the merge decision on `origin/feature/extensions-on-main` (recommendation: take its code only with a D-entry and a prior reading rule) and `origin/feature/local-final-pipeline` (recommendation: keep as an archive). None blocks. One point the user may want to decide: how the Belebele null is framed in the paper (D-050 states what it does and does not show).

## 6. Environment and pitfalls

- ssh by name fails (Cloudflare WARP DNS): `ssh -o HostKeyAlias=ada-gw1 karma@10.4.24.30` (strict check; `ada.iiit.ac.in` is a different machine); `scp -o HostKeyAlias=ada-gw1 karma@10.4.24.30:<path> <dest>`, one file per call; `ADA_SSH='ssh -o HostKeyAlias=ada-gw1 karma@10.4.24.30'` for `deploy_phase1.sh`. The scratchpad is wiped on a session restart; results are re-fetched from Ada. The torch venv `~/.venvs/anlp-torch` is needed for the freeze helper, the analyses and the figures.
- `deploy_phase1.sh` while jobs run: only a named snapshot (`repo_<short commit>`), never `repo`.
- Validation of an evaluation job: see the paragraph in STATUS "Immediate" (item-key hashes against the freeze, checksums, metadata, shapes, answers across files).
- A rescue job needs `--preset floor` (or `eval`), never `none` (A-041). The floor job's output files keep the split name `eval` (`func_lm_eval_INT16.pt`).

## 7. What I would check if something looks wrong

`git status`, `CronList`, `squeue -u karma` over the strict-key route, `phase1_func_freeze.py verify` and `verify-mc` (torch venv), the checksums of fetched files against Ada, whether any frozen file changed after its freeze commit, and the timestamps of the log (they must come from `date`).
