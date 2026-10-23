# Handoff 031: the F5 main result is in (through INT8 a feature does the same thing; S1 and S3 met, S2 undefined; the association status carries no functional information); sensitivity and K = all Belebele running (2026-10-23, 07:58 IST)

> The ONLY source of the previous chat. A fresh chat with no memory must be able to continue from this file plus the repository. If this file and the repository disagree, the repository wins: say so in the next handoff. Start with `reports/functional_validation/STATUS.md` (State, NEXT STEP, Log); this file supersedes [handoff 030](030-2026-10-23-f5-smoke-result-full-jobs-queued.md) as the entry point ([026](026-2026-10-22-eval-ladder-done-calibration-running.md) holds the standing rules in its section 0, [025](025-2026-10-22-chat-end-functional-study-midrun.md) the user's verbatim 2026-10-21 instruction, the addendum of [028](028-2026-10-23-tmc-analysed-null-floor-and-rescue-running.md) the user's verbatim 2026-10-23 message; all still apply unchanged).

## 0. Read this first

The standing rules are those of handoffs 026, 028, 029 and 030 section 0 (mask regime only; no `none` work; no comparison with Duan; never edit `paper/` or `third_party/`; no secrets, tokens, weights or large artifacts; explicit `git add` paths only; commit messages end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`; at most two Ada jobs, never wait or poll (background watchers started as the background command itself); Hugging Face private, delete nothing, no upload without the user's go-ahead (Q21); tests first, mutation check, evidence before claims; a frozen setting is never changed silently; do not merge or cherry-pick the parallel branches (D-052); do not work around the gated Llama repository (Q22); never accept a changed ssh host key; never touch Codex's uncommitted files). Working rules: doc edits by script file or quoted heredoc and read back; chain commits with `set -e` AND check the link checker's count before committing (a chain that only prints it lets a broken link through); timestamps from `date`; the shell's working directory can reset, use absolute paths.

## 1. Mission in five lines

D-039/D-040 (functional validation F1 to F4, F3 rescue) is analysed. **D-053, the user's direction of 2026-10-23: establish whether a feature performs the same function before and after quantization, because the survival metric is meaningless otherwise: F5, answered in D-059.** Remaining: the sensitivity comparisons, the direction-swap control (plan Task 15b), the exploratory K = all Belebele run, the paper-facing summary (the team writes the prose), Llama-3.2-1B as a supporting model (D-052, waits for Q22).

## 2. State right now (verified 07:58)

- Git `main` (pushed), working tree clean after the last commit. Tests: stdlib 265, torch 1069.
- **Ada:** `f5-m32` **35487** (F5 sensitivity, M = 32, states FP 16 4, `--per-cell 3`; expected end about 09:00) and `func-mc-ksens` **35505** (exploratory K = all Belebele, Q20 default (a); queued behind it; precisions FP and 6, 18 languages, `-t 05:00:00`, snapshot `repo_2b6baf4`). Every other job of this study is COMPLETED, validated and analysed. Cron entries `955a5cc5`, `e33ac7eb`, `ea207eac` (session-only, expire about 2026-10-29); no watcher is running.

## 3. What this chat did since handoff 030

1. **F3 rescue job analysed (D-058).** The language groups recover 2.6 % of the INT4 loss gap, the whole layer-12 residual 30 to 40 %: 60 to 70 % of the damage arises above layer 12.
2. **F5 full jobs `f5-a` and `f5-b` (stratified subset, 1,487 features) validated and analysed together (D-059; `results/f5/`, `figures/f5/`).** Floor 0.949 (INT16). Median W1: INT8 0.997, INT6 0.988, INT5 0.961, INT4 0.839; share inside the noise 94, 90, 66.5, 0.3 %; each feature retrieved as itself (0.997 to 1.000); different-feature control 0.12 at the 95th percentile. **S1 met (0.902), S3 met (0.824), S2 undefined (2 events at INT4), overall undetermined;** descriptive AUROC of the association status for W1 at INT4 0.506 [0.469, 0.543]. Readings and limits: D-059 (S1 does not discriminate; the activation cutoff misclassifies about 30 % at INT5; the D-035 status carries no functional information; H5b does not hold: function is preserved better than activations).
3. **Sensitivity:** comparison tool `phase1_func_f5_sensitivity.py` (12 tests, 12 mutants); `f5-m8` validated; INT16, M = 8 against M = 16 on 404 features: median W1 0.9962 against 0.9973, Spearman 0.988, 1.5 % classification flips (stable). INT4 and M = 32 comparisons still to do. The plan for the direction-swap control and why the mean ablation is redundant: design section 11.3, plan Task 15.
4. Figures: three defects found on real data and fixed test first. STATUS State block consolidated (old block archived). Handoffs: 030 (with an addendum) and this one.

## 4. The next step, exactly

STATUS "Immediate" has the commands. In order: (1) when `f5-m32` ends: validate, fetch, checksums, run `phase1_func_f5_sensitivity.py` for M = 32 and M = 8 (INT16 and INT4) against the main run (A and B) into `results/f5_sensitivity/`, write the result with its limits into the Log and D-059's evidence; (2) decide and, if justified, build the direction-swap control (tests first on the toy stack; then a job on `--per-cell 3`, states FP, INT16, INT4): the full result has INT4 below the floor for every feature, so what a cosine of 0.84 means is open; (3) when `func-mc-ksens` ends: validate (T-MC chain), analyse (`phase1_func_mc_analysis.py`; every table labelled 'K64 stands for the whole group, exploratory'), checksums; (4) the paper-facing summary of the functional study; (5) Llama-3.2-1B after Q22.

## 5. Open questions for the user

Q17 to Q20 (defaults in force; Q20 (a) is running), Q21 (upload item-level files to the private dataset?), Q22 (Llama licence), Q23 (the rule S1 to S3; amended by D-057; the result suggests the rule's S1 should be tightened in any future version because it is met by the base rate), Q24 (run the whole 4,382-feature sample too? default no). None blocks.

## 6. Environment and pitfalls

- ssh by name fails (WARP DNS): `ssh -o HostKeyAlias=ada-gw1 karma@10.4.24.30`; `scp -o HostKeyAlias=ada-gw1 karma@10.4.24.30:<path> <dest>` one file per call; `ADA_SSH='ssh -o HostKeyAlias=ada-gw1 karma@10.4.24.30' bash scripts/g0/deploy_phase1.sh repo_<short commit>` (named snapshot while jobs run). Torch venv `~/.venvs/anlp-torch`. The scratchpad is wiped on a session restart: re-fetch the F5 folders `f5-a`, `f5-b`, `f5-m8`, `f5-m32` from `/home2/karma/duan-verify/out/phase1/`.
- F5 analysis commands: STATUS "Immediate" and the module docstrings (`phase1_func_f5_analysis.py --run A --run B`, `phase1_func_f5_figures.py`, `phase1_func_f5_sensitivity.py`). The analysis of the real jobs takes seconds.
- The F5 statements are exploratory; never describe them as confirmatory; the verdict is for this model, layer, quantizer and sample.

## 7. What I would check if something looks wrong

`git status`, the link checker (must say 0 broken), `CronList`, `squeue -u karma` over the strict-key route, the sham gate line and logit gates in each F5 log, and that jobs merged in an analysis share their stage A (the analysis refuses otherwise).
