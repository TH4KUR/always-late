# Handoff 033: the F5 swap control is complete across the whole precision ladder (a feature's effect is preserved exactly as well as any other direction's); the F5 study is finished except the ladder figure and Llama (2026-10-23, 16:00 IST)

> The ONLY source of the previous chat. A fresh chat with no memory must be able to continue from this file plus the repository. If this file and the repository disagree, the repository wins: say so in the next handoff. Start with `reports/functional_validation/STATUS.md` (State, Immediate, Log); this file supersedes [handoff 032](032-2026-10-23-f5-controls-complete-ksens-null.md) as the entry point.

## 0. Read this first

The standing rules are those of [handoff 032](032-2026-10-23-f5-controls-complete-ksens-null.md) section 0 (which carries those of 026 to 031): mask regime only; no `none` work; no comparison with Duan; never edit `paper/` or `third_party/`; no secrets, tokens, weights or large artifacts; explicit `git add` paths only; commit messages end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`; at most two Ada jobs, never wait or poll; Hugging Face private, delete nothing, no upload without the user's go-ahead (Q21); tests first, mutation check, evidence before claims; a frozen setting is never changed silently; do not merge or cherry-pick the parallel branches (D-052); do not work around the gated Llama repository (Q22); never accept a changed ssh host key; never touch Codex's uncommitted files. Working rules: doc edits by script file or quoted heredoc and read back; chain commits with `set -e` AND check that the link checker's last line ends " 0 broken" (the checker only sees git-tracked files: `git add` a new file before checking it); timestamps from `date`; use absolute paths or `cd /home/karma/Documents/ANLP/project/always-late` (the session's primary directory switches between the repository and its parent). The user's direction of 2026-10-23 (D-052, D-053): no merge of `extensions-on-main`; Llama-3.2-1B as a supporting model; "we should work towards establishing weather feature X still performs the same function pre and post quantizations. that would tell us weather our feature survival rate metric is even sensible or not."

## 1. Mission in five lines

D-039/D-040 (functional validation F1 to F4, F3 rescue), the confirmatory Belebele family (D-050, null; K = all confirms it, D-062) and F5 (D-053) are analysed. **F5 answer, now complete (D-059, D-060, D-061, D-063, D-064):** through INT8 a feature's effect on the output is within weight-rounding noise of full precision; at INT6 90 %, INT5 two thirds, INT4 almost none; but **a swapped-direction control (the effect of ablating a different feature's direction at the same sites) loses the same similarity at every precision from INT8 to INT4 (median cosine only 0.003 to 0.014 below the features'), and fixed full-precision amplitudes change almost nothing**, so the loss of similarity is generic drift of the quantized downstream map, not feature-specific; the features stay individually identifiable (retrieval 0.997 to 1.000). For the survival question (the user's): feature-level functional continuity cannot be told apart from the model's response to any direction, so F5 neither confirms nor undermines the activation survival metric at the feature level; the activation criterion misclassifies about 31 % of functionally intact features at INT5 and the association status carries no functional information (D-059). [`RESULTS_SUMMARY.md`](../functional_validation/RESULTS_SUMMARY.md) is the one-page map for the paper team; every number of its section 2 was re-checked against the source files (STATUS Log). Remaining: the ladder figure (plan Task 15f), Llama-3.2-1B after the user's Q22, the team's paper prose.

## 2. State right now (verified 16:00)

- Git `main` (pushed after this step's commit); tests unchanged since handoff 032 (stdlib 265, torch 1102; no code changed in this step). The link checker ends " 0 broken".
- **Ada: no job of this study is running** (`squeue -u karma` empty). All jobs COMPLETED, validated, analysed; checksums of the fetched per-state files in `reports/functional_validation/artifact_checksums.md` (111 files). Newest F5 code snapshot `/home2/karma/duan-verify/phase1/repo_50d71b6`.
- Cron entries `955a5cc5`, `e33ac7eb`, `ea207eac` present at 15:51 (session-only, expire about 2026-10-29).
- The Log of STATUS.md now starts at 2026-10-23; the entries of 2026-10-21 and 22 are in `reports/archive/status_log_2026-10-21_to_22.md`.
- Hugging Face dataset private and unchanged; no upload (Q21 open).

## 3. What was done since handoff 032

1. `f5-fa-dir` validated and compared (D-063): fixed amplitudes raise the feature's INT4 cosine only from 0.842 to 0.846; the control is 0.838.
2. Plan for the swap control at INT8, INT6, INT5 with predictions P1 to P3 written first (design 11.5), job `f5-dir2` 35885 (1 h 27 min), validated, compared (D-064): control 0.0038, 0.0058, 0.0071 below the features' median W1 at INT8, INT6, INT5; all predictions met.
3. `RESULTS_SUMMARY.md` verified number by number (all agree; three wordings tightened).
4. STATUS.md Log halved (plan Task 9).

## 4. The next step, exactly

STATUS "Immediate" (1): **the ladder figure for the F5 control (plan Task 15f).** Median W1 by precision (INT16, INT8, INT6, INT5, INT4) for the features and the swapped directions with the numerical floor as a line, one axis, light and dark, SVG and PNG, a table twin. A function `fig_ladder_control` in `scripts/g0/phase1_func_f5_figures.py` (read the `dataviz` skill first; the figure design is already in the plan item), tests first on planted data (planted generic drift gives equal curves; planted feature-specific drift separates them; undefined states are not drawn), watch them fail, mutation check with `scripts/g0/mutcheck.py`, render and look at the figure (structural tests miss layout), add it to `figures/f5/` and to `RESULTS_SUMMARY.md`. Inputs: the per-state files of the main runs A and B (scratchpad `f5a-partial/`, `f5b/`), of `f5-dir` (INT16, INT4) and `f5-dir2` (INT16, INT8, INT6, INT5); after a restart re-fetch them from `/home2/karma/duan-verify/out/phase1/` (one `scp` per file; checksums are in `artifact_checksums.md`). Done = tests green (torch suite), figure committed and pushed, summary updated.

Afterwards, in priority order: (i) Llama-3.2-1B if the user has accepted the licence (Q22; confirmatory rule Q25); (ii) a dated prior-art check recorded in `reports/research_sources.md` before any Phase 2 work (other models, other quantizers); (iii) further documentation consolidation (plan Task 9: `README.md` index, superseded banners).

## 5. Open questions for the user

Q21 (upload item-level files to the private dataset?), Q22 (accept the Llama licence), Q23 (the rule S1 to S3; S1 is met by the base rate and should be tightened in a future version), Q24 (run the whole 4,382-feature sample too? default no), Q25 (the confirmatory Llama rule C1 to C5). All have defaults in force; none blocks.

## 6. Environment and pitfalls

- ssh by name fails (WARP DNS): `ssh -o HostKeyAlias=ada-gw1 karma@10.4.24.30`; `scp -o HostKeyAlias=ada-gw1 karma@10.4.24.30:<path> <dest>` one file per call; deploy only named snapshots while jobs run: `ADA_SSH='ssh -o HostKeyAlias=ada-gw1 karma@10.4.24.30' bash scripts/g0/deploy_phase1.sh repo_<short commit>`. Torch venv `~/.venvs/anlp-torch`; Ada job env `/home2/karma/duan-verify/env/bin/python`. Job submission form: `sbatch -t HH:MM:SS -J <name> --export=ALL,REPO=/home2/karma/duan-verify/phase1/repo_<sha> scripts/g0/run_f5.sbatch <name> --states FP 16 8 6 5 --per-cell 3 --subset-seed 0 [--direction-control] [--fixed-amplitude]` (state spelling: `16` for INT16).
- A fixed-amplitude run and a direction-control run are never merged with an ordinary run (the analysis refuses); compare them only with `phase1_func_f5_sensitivity.py` (`--main-run` repeatable, `--sens-run`, `--states`, `--figures`, `--figure-title`).
- The F5 statements are exploratory; never describe them as confirmatory; the verdicts are for this model, layer, quantizer and sample.
- Avoid foreground `sleep`; avoid `ugrep` patterns with long `{0,200}` repetitions (complexity limit): use Python `re`.

## 7. What I would check if something looks wrong

`git status`, the link checker (last line ends " 0 broken"), `CronList` (three entries), `squeue -u karma`, the sham gate line and logit gates in each F5 log, the item-key hashes against the freeze for any T-MC file, and that jobs merged in an analysis share their stage A.

## Addendum (16:45): the ladder figure is done

Plan Task 15f is finished: `figures/f5/ladder_control_{light,dark}.{svg,png}` and `ladder_control_table.{csv,md}` (command in STATUS Log, `phase1_func_f5_figures.py ladder ...`); torch suite 1127, stdlib 265. Section 4 above is therefore done; the next step is STATUS "Immediate" (1): a dated prior-art check for Phase 2 (and the choice of a second model if Llama-3.2-1B stays blocked), then Llama after Q22, then consolidation (plan Task 9).

## Addendum 2 (17:13): the Phase 2 prior-art check is done

D-065: no work found that tests the function of SAE features under weight quantization; cite Hoang et al. (arXiv 2607.24645, the feature-effect cloud of logit changes: the F5 fingerprint is not claimed as new) and Cho et al. (2607.20596); Duan's causal step is still unpublished (Q18). New question Q26 (Qwen3-1.7B-Base with Qwen-Scope as a second model; default: keep Llama and wait for Q22). Nothing runs on Ada. The next step is in STATUS "Immediate": no code step waits; the open items are the user's answers (Q21 to Q26); documentation consolidation (plan Task 9) is the default work. For the team: full-text reading of P43 and P41 for the related-work section.

## Addendum 3 (19:20): the association-loss analysis A2 is done

D-066: within language, the feature's own full-precision margin predicts the loss of the D-035 association (AUROC 0.90 / 0.81 / 0.75 at INT6 / INT5 / INT4; scale and share add little; same without English); at INT4 even the best-separated fifth loses 7.3 % against 0.0 % at the rounding floor. New code `scripts/g0/phase1_assoc_margin.py` (29 tests, 26 mutants killed; torch suite 1156, stdlib 265), output `reports/association_retention/results/within-language/`, input checksum in `artifact_checksums.md` (the scratch copy of the file may be re-fetched from Ada `association-full/analysis-eq/feature_status.csv`). Next step in STATUS "Immediate" (1): a figure for A2; no job runs on Ada; the open items are the user's answers (Q21 to Q26).

## Addendum 4 (19:56): the A2 figure is done

`results/within-language/fig_loss_by_margin_{light,dark}.{svg,png}` (module `scripts/g0/phase1_assoc_margin_figures.py`, 18 tests, 20 mutants killed; torch suite 1174, stdlib 265; command in the module docstring). The next step is STATUS "Immediate" (1): A3, a CPU cross-check of function (F5 W1) against the full-precision margin, plan and prediction first. No job runs on Ada; the open items are the user's answers (Q21 to Q26).

## Addendum 5 (20:39): A3 is done

D-067: on the 1,449 F5-tested features that are established D-035 features, the full-precision margin separates the association loss with AUROC 0.86 (INT6) / 0.77 (INT5) but the function event (W1 below the floor) with only 0.58 / 0.53; Spearman(margin, W1) about +0.10. New code `scripts/g0/phase1_assoc_function_link.py` (27 tests, 28 mutants killed; torch suite 1201, stdlib 265), output `reports/association_retention/results/margin-function/`; inputs: the F5 jobs A and B (scratch copies `f5a-partial/`, `f5b/`; Ada `f5-a`, `f5-b`) and the D-035 status file (Ada `association-full/analysis-eq/feature_status.csv`). Next step in STATUS "Immediate" (1): an A3 figure. No job runs on Ada; the open items are the user's answers (Q21 to Q26).

## Addendum 6 (21:16): the A3 figure is done; nothing waits on code

`results/margin-function/fig_margin_auroc_{light,dark}.{svg,png}` (module `scripts/g0/phase1_assoc_function_figures.py`, 19 tests, 20 mutants killed; torch suite 1220, stdlib 265). Every planned analysis and figure of the functional study is done. The next steps in STATUS "Immediate": the user's answers (Q21 to Q26; Q22 and Q26 unlock a second model), documentation consolidation (plan Task 9), a paper-facing digest map (claims, supporting figures, tables, decisions and limits; not prose). No job runs on Ada.

## Addendum 7 (21:51): the claim map

Section 3 of `reports/functional_validation/RESULTS_SUMMARY.md` is now a table of eleven claims with evidence and limits (the 'paper-facing digest' step). Nothing waits on code; remaining defaults: the user's answers (Q21 to Q26), documentation consolidation (plan Task 9: `reports/scheduled_routine.md`), re-running the dated prior-art search in about a week.

## Addendum 8 (22:36): reproducibility check

`scripts/g0/reproduce_functional.sh <inputs root> <scratch dir>` re-runs every analysis of the functional study and compares with the committed outputs (57 of 57 identical on 2026-10-23); layout and commands in `reports/functional_validation/REPRODUCE.md`. The scratch inputs used (symlink layout) are under the session scratchpad `repro_in/` and vanish with it; re-fetch from Ada per the register. Nothing waits on code; defaults for the next firings: the user's answers (Q21 to Q26), documentation consolidation (plan Task 9).

## Addendum 9 (23:12): entry pages consolidated

`PROGRESS.md` (block 'NEXT RUN'S GOAL') and `reports/scheduled_routine.md` (rule 5, steps 1, 4, 6, 'How a step ends') now point at STATUS `Immediate`; the stale texts are in `reports/archive/routine_and_progress_stale_blocks_2026-10-23.md`. Nothing waits on code; the open items are the user's answers (Q21 to Q26).

## Addendum 10 (23:48): documentation clean-up and the standing rule

The user (main chat, 2026-10-23): clean up documentation, remove stale and redundant documents, but **never remove progress tracking** (the trail of all directions explored and considered must survive to the end of the project). Done: `reports/INDEX.md` (map, rule, directions ledger, record of removals); the rule is in `functional_validation/cron_prompt.md` (hard rule 3), `scheduled_routine.md` (rule 7), `HANDOFF.md` and the assistant's memory; 24 generated files removed that were byte-identical to files of `figures/eval/` (last commit holding them: `7147f0e`). Everything else stays.

## Addendum 11 (2026-10-24 00:06): language support, the Belebele reliability gap, plan Task 17

The user asked (main chat) whether Gemma-2-2B supports all 27 languages and whether the Belebele null is reliable / copied from a repository. D-068: the loss-task results hold on the 18 languages with measurable Belebele ability (0.713 [0.686, 0.739], 18 of 18) and the 9 others are positive too; `results/eval_subsets/`, in `reproduce_functional.sh` (69 of 69 identical). The Belebele pipeline is ours (benchmark data, own prompt and scoring); its checks are listed in STATUS; **missing is a positive control**: plan Task 17 (dose-response manifest of language-independent sets of the non-selective pool at 256, 1,024, 2,860 features; predictions P1 to P3 written first; plus a prompt cross-check) is the next step, STATUS `Immediate`.
