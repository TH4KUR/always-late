# Handoff 032: the F5 controls are complete (the INT4 change of a feature's effect is generic drift) and the K = all Belebele run confirms the null (2026-10-23, 13:20 IST)

> The ONLY source of the previous chat. A fresh chat with no memory must be able to continue from this file plus the repository. If this file and the repository disagree, the repository wins: say so in the next handoff. Start with `reports/functional_validation/STATUS.md` (State, Immediate, Log); this file supersedes [handoff 031](031-2026-10-23-f5-main-result-sensitivity-and-ksens-running.md) as the entry point.

## 0. Read this first

The standing rules are those of handoffs 026, 028, 029, 030 and 031 section 0 (mask regime only; no `none` work; no comparison with Duan; never edit `paper/` or `third_party/`; no secrets, tokens, weights or large artifacts; explicit `git add` paths only; commit messages end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`; at most two Ada jobs, never wait or poll (background watchers started as the background command itself); Hugging Face private, delete nothing, no upload without the user's go-ahead (Q21); tests first, mutation check, evidence before claims; a frozen setting is never changed silently; do not merge or cherry-pick the parallel branches (D-052); do not work around the gated Llama repository (Q22); never accept a changed ssh host key; never touch Codex's uncommitted files). Working rules: doc edits by script file or quoted heredoc and read back; chain commits with `set -e` AND check that the link checker's last line ends " 0 broken" before committing; timestamps from `date`; the shell's working directory can reset, use absolute paths. The user's direction of 2026-10-23 (D-052, D-053) still governs: no merge of `extensions-on-main`, Llama-3.2-1B as a supporting model, and "we should work towards establishing weather feature X still performs the same function pre and post quantizations. that would tell us weather our feature survival rate metric is even sensible or not."

## 1. Mission in five lines

D-039/D-040 (functional validation F1 to F4, F3 rescue) and the confirmatory Belebele family (D-050, null) are analysed. **F5 (D-053): does a feature do the same thing before and after quantization? Answered in D-059 and its controls D-060, D-061, D-063:** through INT8 yes (within weight-rounding noise); at INT6 90 % of features still within noise, INT5 two thirds, INT4 almost none, but each feature stays itself (retrieval 0.997 to 1.000) and the INT4 loss of similarity is generic drift of the quantized downstream map (a swapped-direction control is as dissimilar; taking the activations' own change out changes almost nothing). The activation survival criterion and the association status of the earlier study are not reliable proxies for functional continuity. [`RESULTS_SUMMARY.md`](../functional_validation/RESULTS_SUMMARY.md) is the one-page map for the paper team. Remaining: one more control (Immediate (1) of STATUS), Llama-3.2-1B after the user's Q22, the team's paper prose.

## 2. State right now (verified 13:20)

- Git `main`, pushed after this step's commit; working tree clean apart from files of other people. Tests: stdlib 265, torch 1102 (no code changed in this step). `python3 -B scripts/g0/check_md_links.py` ends " 0 broken".
- **Ada: no job of this study is running** (`squeue -u karma` empty; every job COMPLETED, validated, analysed; checksums in `reports/functional_validation/artifact_checksums.md`, 104 files). Output directories under `/home2/karma/duan-verify/out/phase1/` (`f5-a`, `f5-b`, `f5-m8`, `f5-m32`, `f5-dir`, `f5-fa`, `f5-fa-dir`, `func-mc-ksens`, ...). Snapshots under `/home2/karma/duan-verify/phase1/`: `repo_50d71b6` is the newest F5 code (fixed amplitude and direction control).
- Cron entries `955a5cc5`, `e33ac7eb`, `ea207eac` present at 13:11 (session-only, expire about 2026-10-29).
- Hugging Face dataset private and unchanged; no upload (Q21 open).

## 3. What was done since handoff 031

1. **Sensitivity (D-060):** M = 8, 16, 32 stable (INT4 median W1 0.817, 0.842, 0.859; gap to the floor about 0.10).
2. **Direction-swap control (D-061):** at INT4 the swapped direction has median W1 0.828 against 0.842 for the feature: the loss of similarity is generic.
3. **K = all Belebele (D-062; `results/mc_ksens/`):** zeroing the whole selective group (up to 1,719 features) changes accuracy no more than matched controls (macro effect +0.0007 nats per item [-0.0030, +0.0042]; accuracy 0.4257 / 0.4209 / 0.4226 for no edit / group / controls). The confirmatory null of D-050 is not an artifact of K = 64. Note: the job's argv holds two `--manifest` arguments (the sbatch's, then the `ksens` one); the last wins; recorded in D-062.
4. **Fixed amplitude and its control (D-063; `results/f5_sensitivity/fixed_amplitude/`, `.../fixed_amplitude_direction_swap/`):** with the full-precision activation amplitude in every state the feature's INT4 median W1 is 0.846 (ordinary 0.842), the magnitude ratio 1.115 (ordinary 0.917), and the swapped direction 0.838; so the INT4 change of a feature's effect is not carried by the activations and is shared by other directions. Prediction (b) of design 11.4 was met only negligibly for W1 and not for the magnitude ratio (reported as is).
5. Docs: `RESULTS_SUMMARY.md` updated at each result; STATUS Log, README, progress log, checksums (104 files).

## 4. The next step, exactly

STATUS "Immediate" (1): **the swap control at INT8, INT6 and INT5** on the same 404 features. Source: D-061 and D-063 show the control only at INT16 and INT4; the survival-metric question needs to know whether the shortfall below the numerical floor at INT5 and INT6 (where 66.5 % and 90 % of the features are still within noise) is also generic. Steps: (a) write design section 11.5 with the predictions already put in STATUS (control within 0.01 of the features at each precision; a control more than 0.02 below the features would mean features keep their function better than arbitrary directions); (b) submit `run_f5.sbatch` with `REPO=/home2/karma/duan-verify/phase1/repo_50d71b6` and `--direction-control --per-cell 3 --subset-seed 0 --states FP 16 8 6 5` (state spelling as in the smoke job; `-t 03:00:00`; one job, about 1.5 h); record the Slurm id; (c) validate it exactly like `f5-dir` (sacct, `meta_f5.json` complete with `direction_control` true, sham exact, 404 features, the same sample hash as the main runs, checksums equal to Ada's); (d) compare with `phase1_func_f5_sensitivity.py --main-run <f5-a> --main-run <f5-b> --sens-run <new> --states INT8 INT6 INT5 --figures` (the scratchpad copies `f5a-partial/` and `f5b/` vanish on a restart: re-fetch `f5-a` and `f5-b`); (e) write D-064, Log, `RESULTS_SUMMARY.md`. Done = D-064 committed and pushed.

If that job is judged not worth its hour, the alternatives in priority order: (i) Llama-3.2-1B once the user accepts the licence (Q22; the confirmatory rule is Q25); (ii) a random-direction control (not the matched different feature); (iii) consolidation of documentation (plan Task 9).

## 5. Open questions for the user

Q21 (upload item-level files to the private dataset?), Q22 (accept the Llama licence), Q23 (the rule S1 to S3; amended by D-057; S1 is met by the base rate and should be tightened in any future version), Q24 (run the whole 4,382-feature sample too? default no), Q25 (the confirmatory Llama rule C1 to C5). All have defaults in force; none blocks.

## 6. Environment and pitfalls

- ssh by name fails (WARP DNS): `ssh -o HostKeyAlias=ada-gw1 karma@10.4.24.30`; `scp -o HostKeyAlias=ada-gw1 karma@10.4.24.30:<path> <dest>` one file per call; deploy only named snapshots while jobs run: `ADA_SSH='ssh -o HostKeyAlias=ada-gw1 karma@10.4.24.30' bash scripts/g0/deploy_phase1.sh repo_<short commit>`. Torch venv `~/.venvs/anlp-torch`; Ada job env `/home2/karma/duan-verify/env/bin/python`.
- A fixed-amplitude run and a direction-control run are never merged with an ordinary run (the analysis refuses); compare them only with `phase1_func_f5_sensitivity.py`.
- The F5 statements are exploratory; never describe them as confirmatory; the verdicts are for this model, layer, quantizer and sample.
- The Claude Code session's primary directory can switch between the repository and its parent: use `cd /home/karma/Documents/ANLP/project/always-late` explicitly.

## 7. What I would check if something looks wrong

`git status`, the link checker (last line ends " 0 broken"), `CronList` (three entries), `squeue -u karma` over the strict-key route, the sham gate line and logit gates in each F5 log, the item-key hashes against the freeze for any T-MC file, and that jobs merged in an analysis share their stage A.

## Addendum (2026-10-23 13:52): the next step was started

Design section 11.5 and plan Task 15e (the swap control at INT8, INT6 and INT5, predictions P1 to P3 written first) were committed at 13:51 and the job `f5-dir2` **35885** was submitted at 13:52 (snapshot `repo_50d71b6`, `--states FP 16 8 6 5 --per-cell 3 --subset-seed 0 --direction-control`, `-t 02:30:00`, expected end about 15:25, output `/home2/karma/duan-verify/out/phase1/f5-dir2/`). So section 2's "no job running" is out of date: one job runs, one slot is free. Section 4 steps (a) and (b) are done; (c) to (e) remain, with the exact commands in STATUS "Immediate".
