# HANDOFF: building on Duan (2026)'s OWN code

> **Latest handoff: [034 -- STUDY G (the user's major study, graded gain of language-selective features) is designed, its baseline measured and its code built; the FP calibration job is running](reports/handoffs/034-2026-10-24-study-g-built-calibration-running.md).** Earlier: [033 -- the F5 swap control is complete across the whole precision ladder (a feature's effect is preserved exactly as well as any other direction's); only the ladder figure and Llama remain](reports/handoffs/033-2026-10-23-f5-ladder-control-complete.md).** Earlier: [032](reports/handoffs/032-2026-10-23-f5-controls-complete-ksens-null.md), [031](reports/handoffs/031-2026-10-23-f5-main-result-sensitivity-and-ksens-running.md), [030](reports/handoffs/030-2026-10-23-f5-smoke-result-full-jobs-queued.md), [029](reports/handoffs/029-2026-10-23-f5-built-smoke-queued.md), [028](reports/handoffs/028-2026-10-23-tmc-analysed-null-floor-and-rescue-running.md), [027](reports/handoffs/027-2026-10-22-eval-ladder-complete-f4-queued.md), [026](reports/handoffs/026-2026-10-22-eval-ladder-done-calibration-running.md), [025](reports/handoffs/025-2026-10-22-chat-end-functional-study-midrun.md) (the work loop, history). The 40-minute cron job is active (entries and recipe: [`reports/functional_validation/cron_prompt.md`](reports/functional_validation/cron_prompt.md); current state: [`reports/functional_validation/STATUS.md`](reports/functional_validation/STATUS.md)). Earlier: [022](reports/handoffs/022-2026-10-21-q16-functional-validation-selected.md) (Q16 selected), 023 and 024 (design, code, pilot).
>
> Older banner paragraphs (the two handoff series of 2026-10-21 and the Codex 014 note) were moved verbatim to [`reports/archive/handoff_md_stale_sections_2026-10-23.md`](reports/archive/handoff_md_stale_sections_2026-10-23.md) on 2026-10-23 (plan Task 9). **The project state is NOT described in the sections below: read the newest handoff and `reports/functional_validation/STATUS.md`.**

> **Documentation map (what is current, what is the progress-tracking trail that is never deleted, what is evidence): [`reports/INDEX.md`](reports/INDEX.md).**
>
> **START HERE (since 2026-10-18): read ONLY the newest file in `reports/handoffs/` first** (highest number; template `reports/handoffs/TEMPLATE.md`; the routine is `reports/scheduled_routine.md`). It is the only record of the previous chat; the sections below are reference material.

Written 2026-10-16, rewritten the same night after the audit and the decision to **discard our previous codebase**. For an agent with no memory of the previous sessions. Read this file fully, then the "POST-MORTEM of our previous codebase" entry at the end of `reports/progress_log.md`, `reports/audit_our_vs_duan.md` (what was found), `third_party/duan2026/UPSTREAM.md` (exactly what was removed and edited) and `reports/duan_codebase_audit.md` (Ada facts, closure protocol). Items marked **UNVERIFIED** were not verified.

## 0. Mission and the rules that matter

**Where we are.** Duan et al.'s own Phase-3 script is vendored and reproduces their published Gemma table on Ada (within 0.15 pp). The audit showed that our earlier pipeline disagreed with theirs because of a **regime difference** (their windows have no BOS token; ours had one), not a bug. Our old pipeline (`src/`, Phase-1 scripts) was **discarded**: recoverable only from the git tag `legacy-pipeline-2026-10-16`. Everything new is built on their code.

**Mission now (2026-10-23).** The Phase 1 and the functional-validation studies are finished and summarised on one page for the team ([`reports/functional_validation/RESULTS_SUMMARY.md`](reports/functional_validation/RESULTS_SUMMARY.md)); the open items are the user's answers (`DECISIONS_FOR_REVIEW.md`, Q21 to Q26) and the team's paper. The original mission of 2026-10-16 (two pending diagnostics, the regime decision) is in the archive file named below.

**The rule.** We run **their code, in their repo layout**. `third_party/duan2026/` is their repository downloaded fresh, pruned to what this study needs, with the working copy of the one script we run differing from their file by a **short explicit whitelist** and nothing else:

| What | Detail |
|---|---|
| Removed (not needed) | Pythia scripts, the feature-class predictor, `src/`, Pythia/pruning-overlap results and figures, env files. Hashes of every removed file stay in `MANIFEST.sha256`. |
| Retained byte-identical (`pristine/`) | `LICENSE`, `README.md`, `scripts/03_gemma_phase3_streaming_sweep.py` (the study), `scripts/06_gemma_sliding_window_ppl_check.py`, and their published Gemma results (`phase3_summary_final.csv`, `gemma_chunked_vs_sliding_ppl_w2048_s512_FULL.csv`, the INT6 per-feature file). |
| Working copy = pristine script + | (tail) delete their cells 22+, which call three functions never defined in the file and run after all results are written; (1) `RUN_PRUNING = False`; (2) `WORK_DEVICE = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE`; (3) `device == "cuda"` -> `str(device).startswith("cuda")` in their dtype test; (4) `load_gemma_tl_model(..., device=WORK_DEVICE)`; (5) `model_work.to(WORK_DEVICE)`. Edits 2-5 are device references forced by 11 GB GPUs. |
| No edit for `display()` | their notebook-only `display(...)` works under `ipython script.py`. |

**Do not edit the working copy by hand, and do not add edits.** The whitelist lives in `scripts/g0/verify_vendor.py` (`EDITS`, `TRUNCATE_AT`). `python3 -m unittest discover -s scripts/g0/tests -v` must print `OK` before any run (54 tests: exact reconstruction of the working copy, a committed mutation suite, report/wandb tooling). Anything beyond the baseline (BOS arms, extra bit-widths, new study code) is a **diagnostic / modified** experiment, must be labelled so and never called a replication. New study code must be a diff against their pristine script (the pattern in `scripts/g0/make_duan_runners_v3.py`), with tests, and must reproduce the baseline when its new switches are off.

**Honest limit (UNVERIFIED).** We never ran their *unmodified* script (needs >= 16 GB; none reachable). That edits 2-5 do not change results is argued from the code and from bit-identical logits across the two cards, not demonstrated on a large GPU.

## 1. Why this exists (short)

The team's own pipeline gave INT7/INT6 survival of 98%/94% where Duan reports 81%/51%. After running their code (reproduces) and a factor-by-factor audit (`reports/audit_our_vs_duan.md`): with one BOS token their own pipeline gives 99.97% / 97.5% (their TransformerLens code and an HF reimplementation agree to 0.1 pp); perplexity is 457 without BOS and 16 with it. Their quantizer's shared q/k/v scales cost ~8 / 3.5 pp but only without BOS; HF's sdpa attention path and a bf16 SAE make no measurable difference. The old code, its flaws and the verdict on whether its direction was worth it are in the post-mortem entry. The paper and README of that period contradicted the team's own data (`issues.md` A1-A6); the paper must be rewritten **by the team** (course policy forbids AI-generated report text; do not write paper prose).

## 2. State of play (moved)

*Moved to [`reports/archive/handoff_md_stale_sections_2026-10-23.md`](reports/archive/handoff_md_stale_sections_2026-10-23.md) on 2026-10-23: the table of the state of 2026-10-16 (review of the old code, baseline, audit, R1 and R5c done, the closure D-NOISE, study design not started).*

## 3. Runbook for the baseline (done once; keep for re-runs)

Commands from the repo root. Ada: `ssh karma@ada-gw1` (key auth works from the user's laptop; `ada.iiit.ac.in` is closed).

1. **Pre-flight (local).** `python3 -m unittest discover -s scripts/g0/tests -v` -> `OK`; `python3 scripts/g0/verify_vendor.py --verify` -> `pristine problems: none | working copy == pristine + whitelist: True`. To re-verify against upstream: download `https://anonymous.4open.science/api/repo/sae-feature-survival-quantization-0141/zip`, extract with a path-safe extractor into an empty directory, run `python3 scripts/g0/verify_vendor.py --build <that dir>`, then `git status third_party/duan2026` must show no change.
2. **Deploy.** `V=third_party/duan2026; scp $V/scripts/03_gemma_phase3_streaming_sweep.py karma@ada-gw1:/home2/karma/duan-verify/vendor/scripts/; scp $V/working.sha256 karma@ada-gw1:/home2/karma/duan-verify/vendor/; scp scripts/g0/run_baseline.sbatch scripts/g0/baseline_report.py scripts/g0/g0_wandb.py karma@ada-gw1:/home2/karma/duan-verify/`. The launcher refuses to run if the deployed script's sha256 does not match `working.sha256`.
3. **Pre-flight (Ada).** `sinfo -N -p u22 -t idle -h -o "%N %f" | grep -c 2080ti` (>= 1), `df -h /home2/karma` (>= 12 GB free), `squeue -u karma` (the user's own `plan02-train` job is not yours: never touch it), outbound network (`curl -sI https://huggingface.co`).
4. **Submit.** `ssh karma@ada-gw1 'cd /home2/karma/duan-verify && sbatch -J base-full run_baseline.sbatch full full'` (~4.1 h compute + 10-20 min model download). No resume.
5. **Watch / collect / judge.** `squeue -u karma`; logs in `/home2/karma/duan-verify/logs/`; results in `/home2/karma/duan-verify/baseline/full/`; judge with `scripts/g0/baseline_report.py` (pre-committed bands; Q1 passes iff INT7 survival within 5 pp of 81.26%, INT6 within 10 pp of 51.30%, active features within 10% of 7,144, perplexity signs hold).

## 4. Environment facts that cost us time (all verified)

- **Cluster.** Slurm. Accounts `nlp` (QOS normal, no per-user cap) and `research` (QOS medium: max 4 GPUs, **40 CPUs**, 125 GB per user, 1 node per job). Only partition `u22` is open. GPUs reachable are 11 GB-class only: GTX 1080 Ti (gnode001-042) and RTX 2080 Ti (gnode044-092, `--constraint=2080ti`). Turing has no native bf16: PyTorch emulates it (correct, ~2x slower).
- **CPU cap trap.** `-c 8` with `--mem=40G` is inflated by Slurm to 14 CPUs (and 18 for the 2-GPU jobs), so several jobs on `research` hit `QOSMaxCpuPerUserLimit`. `-A nlp` starts immediately. Check `scontrol show job <id> | grep NumCPUs`.
- **Filesystems.** `/home2/karma` (NFS, ~19 GB free, **shared with the user's other jobs**): keep the model off it. **`/share1` is NOT mounted on compute nodes** (3,000-file quota). Node-local `/scratch` holds the 10.5 GB gated model per job (deleted at job end). Never write into shared areas you do not own.
- **Software.** `/home2/karma/duan-verify/env`: Python 3.10, torch 2.14.1+cu126, transformer-lens 3.9.0, sae-lens 6.54.5, transformers 5.19.0, wandb 0.30.0, ipython 8.39.0 (freeze in `reports/g0/`). Their `requirements.txt` was unpinned; these versions are part of the result.
- **Hugging Face.** The gated `google/gemma-2-2b` needs a token already stored on Ada (`/home2/karma/duan-verify/hf/token`; jobs use `HF_TOKEN_PATH`). **Never copy or print a token.** **Do not set `HF_HUB_DISABLE_XET=1`** (15x slower, drops connections).
- **wandb.** Key in `~/.netrc` (account `karma-skz`, entity `karma-skz-`), project `anlp-project`. Never ask for or handle a key.
- **Measured costs (TransformerLens, 2 x 2080 Ti).** 40-45 min per condition at 500k tokens, peak GPU 8.2 GiB per card, host RSS 21.8 GB.
- **Fetching results.** rsync filters: **excludes must come before includes** (first match wins); a transient DNS failure on the laptop can leave an empty directory while the loop still echoes "fetched": verify fetches by listing the files.
- **Tests.** Python reuses stale `.pyc` for same-size edits within one second: run mutation checks with `python -B` after clearing `__pycache__`.

## 5. Diagnostics on THEIR code (moved)

*Moved to [`reports/archive/handoff_md_stale_sections_2026-10-23.md`](reports/archive/handoff_md_stale_sections_2026-10-23.md) on 2026-10-23: the tooling paragraph and the table of the jobs R1 (24451) and R5c (24452), both COMPLETED 2026-10-17 and judged; the file locations of their results are in the file map below.*

## 6. Do-not list

- Do not edit `third_party/duan2026/` by hand, add edits, change the baseline constants, or change the comparison bands. Do not describe diagnostics as replication.
- Do not resurrect the legacy pipeline into the working tree. If something from it is needed, check out the tag and port the *idea* as a tested diff on their code.
- **Never remove, shorten or overwrite progress-tracking documents** (decisions, questions, logs, handoffs, plans and designs, source registers, audits, the archive): the user's instruction of 2026-10-23, so that the whole trail of directions explored and considered survives to the end of the project. The map of what is current, what is trail and what is evidence is [`reports/INDEX.md`](reports/INDEX.md); consolidation moves text verbatim into `reports/archive/` with a pointer.
- Do not edit `paper/` prose or write report text.
- Do not touch the user's `plan02-train` jobs, other users' files, `~/.netrc`, `~/.env`, `~/.ssh`, HF tokens. Do not fill `/home2`. No long jobs on the login node.
- Do not commit data, weights, `.pt` files, per-feature CSV dumps or secrets. Commit small CSV/JSON/MD results under `reports/g0/`.

## 7. File map

| Path | What |
|---|---|
| `HANDOFF.md`, `PLAN.md`, `PROGRESS.md`, `reports/progress_log.md` | handoff, plan, status, dated evidence log (ends with the post-mortem of the old codebase) |
| `issues.md` | review of the discarded pipeline and paper |
| `reports/audit_our_vs_duan.md` | why the old pipeline disagreed with theirs: 12 differences, experiments, pre-committed rules, results (sections 6.1-6.4) |
| `reports/duan_codebase_audit.md` | audit of Duan's repo, verified Ada facts, closure protocol (section 7) |
| `reports/artifacts.md`, `scripts/g0/hf_artifacts.py`, `reports/g0/hf_artifact_spec.json`, `hf_dataset_card.md` | where the large artifacts live: private Hugging Face dataset https://huggingface.co/datasets/karma-skz/always-late-artifacts (policy D-012; upload and verify with `hf_artifacts.py`, token only on Ada) |
| `reports/wandb_schema.md`, `scripts/g0/g0_wandb.py`, `wandb_rebuild.py`, `reports/g0/wandb_manifest.json`, `wandb_backup_pre_schema.json` | the fixed wandb schema, its code and tests, what was rebuilt, and the backup of the old runs' metadata |
| `reports/g0/` | `baseline_full/`, `baseline_smoke*/`, `R2_bos_mask/`, `r5/` (+ `r5_report.md`), `r5_smoke/`, benchmark, env freeze, runner diffs |
| `reports/legacy_pipeline/` | Phase-1 results of the discarded pipeline: hypotheses, not findings |
| `third_party/duan2026/` | **their code**: `UPSTREAM.md`, `MANIFEST.sha256`, `pristine/`, `scripts/` (working copy), `patches/`, `working.sha256` |
| `scripts/g0/` | `verify_vendor.py`, `run_baseline.sbatch`, `baseline_report.py`, `make_duan_runners_v3.py`, `run_g0.sbatch`, `unit_test_runner.py`, `g0_wandb.py`, `selftest_wandb.py`, `bench_fit.py`, `r5_compare.py`, `tests/` |

## 8. Baseline result (job 23472, completed 2026-10-16)

**Q1 passed on all four pre-committed bands.** Outputs: `reports/g0/baseline_full/`. wandb run `https://wandb.ai/karma-skz-/anlp-project/runs/86f26788`.

| condition | ppl (ours / published) | survived % (r>0.9) | median r | active features |
|---|---|---|---|---|
| FP16 baseline | 458.4 / 458.5 | 100 / 100 | 1.0000 / 1.0000 | 7,134 / 7,144 |
| RTN INT8 | 469.9 / 470.6 | 99.12 / 99.09 | 0.9707 / 0.9707 | same |
| RTN INT7 | 432.7 / 432.6 | 81.19 / 81.26 | 0.9499 / 0.9499 | same |
| RTN INT6 | 476.8 / 476.8 | 51.15 / 51.30 | 0.9027 / 0.9027 | same |
| RTN INT5 | 539.7 / 539.8 | 27.57 / 27.72 | 0.7775 / 0.7780 | same |
| RTN INT4 | 670.4 / 669.8 | 12.39 / 12.40 | 0.4504 / 0.4512 | same |

Largest survival gap 0.15 pp; per-feature Spearman at INT6 over 7,121 commonly active features 0.99965. Job 23472 on gnode057: 4 h 08 min, exit 0, peak GPU 8,183 / 8,009 MiB. Not demonstrated: that our four device edits equal their unmodified single-device run.

## 9. Open decisions (moved)

*Moved to [`reports/archive/handoff_md_stale_sections_2026-10-23.md`](reports/archive/handoff_md_stale_sections_2026-10-23.md) on 2026-10-23: the four open decisions of 2026-10-16 (the regime, rebuilding the analysis, who writes the text, optional runs); they were answered: Q1 and D-028 (mask only), Q2 and D-029 (publishability), D-035 onward. Open questions now: `DECISIONS_FOR_REVIEW.md`.*

## 10. (COMPLETED 2026-10-17) check R1 and R5c, fetch, judge, finish the audit: both done, closure D-NOISE; kept for reference

*Moved to [`reports/archive/handoff_md_stale_sections_2026-10-23.md`](reports/archive/handoff_md_stale_sections_2026-10-23.md) on 2026-10-23: the six-step checklist for R1 and R5c (completed 2026-10-17, closure D-NOISE).*

## 11. The routine (current; the historical text of 2026-10-18 was moved)

The user's recurring 40-minute job of 2026-10-21 (check Ada jobs, code, analyse, figures, questions, documentation, commit and push; one bounded step per firing) is a session-only cron job: the recipe and the three entry ids are in [`reports/functional_validation/cron_prompt.md`](reports/functional_validation/cron_prompt.md), the live state in [`reports/functional_validation/STATUS.md`](reports/functional_validation/STATUS.md); the step definition and rules of the older routine are in [`reports/scheduled_routine.md`](reports/scheduled_routine.md). After a finished step or job analysis the next numbered handoff is written in `reports/handoffs/` (or an addendum to the newest one) and pushed; open questions live in `DECISIONS_FOR_REVIEW.md`.

*Moved to [`reports/archive/handoff_md_stale_sections_2026-10-23.md`](reports/archive/handoff_md_stale_sections_2026-10-23.md) on 2026-10-23: the 2026-10-18 text ('no scheduled job and no second chat') and the history of the hourly cron jobs.*

## 12. wandb: one fixed schema (since 2026-10-17)

Every run, live or rebuilt, uses the vocabulary in `reports/wandb_schema.md` (15 config keys, 12 per-condition metrics with the condition as data in `bits`, 29 fixed summary keys, 3 fixed table names). The old per-condition-prefixed names (`INT7/median_corr`, `baseline/...`) no longer exist. Rules: never log ad-hoc keys (`python3 scripts/g0/wandb_rebuild.py verify` fails on them); to change the vocabulary change the constants in `g0_wandb.py`, bump `SCHEMA_VERSION`, update the tests and rebuild; run the live check `selftest_wandb.py` (creates and deletes one run) after any logger change. wandb access needs the throwaway venv `/tmp/claude-1000/wbvenv/bin/python` (system python has no wandb; recreate it with `python3 -m venv` and `pip install wandb` if it is gone); never print or handle keys. **Status 2026-10-17: the 13 finished runs are rebuilt under the schema and their 13 old runs deleted (verified first; see `reports/wandb_schema.md`).** The schema-aware `g0_wandb.py`, `run_g0.sbatch`, `run_baseline.sbatch` and `baseline_report.py` **were deployed to Ada on 2026-10-17 after R1 and R5c finished** (sha256 equal to the repo's; the module imports on Ada). R1 and R5c were rebuilt under the schema and their old live runs deleted: all 15 runs in the project conform (`wandb_rebuild.py verify`).

## 13. Phase 1 build on their code (2026-10-17): what exists and how to run it

**Code** (all in `scripts/g0/`, every file has tests; `python3 -B -m unittest discover -s scripts/g0/tests` = stdlib suite, 111 tests; `<torchvenv>/bin/python -B -m unittest discover -s scripts/g0/tests_torch` = torch/numpy/scipy/statsmodels suite, 90+ tests; key behaviours are mutation-checked): `their_code.py` (their functions extracted verbatim by AST from the vendored pristine script, hashes frozen in `reports/g0/their_functions.json`), `phase1_lib.py` (batching with BOS regimes, per-language streaming Pearson in their math, tagging statistics, S(f), support rule, bootstrap stability, shuffled null, Deng nu, matched controls), `phase1_data.py` + `reports/g0/phase1_languages.json` (FLORES loader, 27-language roster), `phase1_run.py` (`main`: one model quantized in place and restored, tagging pass, survival pass, outputs), `phase1_anchor.py` (S5 comparison with pre-declared tolerances), `phase1_analysis.py` (S7: confirmatory T1/T2/T3 + Holm + TOST with stratum-standardised contrasts (D-016), exploratory sections, random-intercept logistic model by quadrature ML, markdown report, command line), `mutcheck.py` (mutation-check harness that always restores the source; never run ad-hoc mutation scripts: A-028), `phase1_wandb_manifest.py`, `fetch_phase1.sh`. Tests leave no temp directories (`/tmp` is a RAM-backed tmpfs of 7.7 GB).

**Run** (needs the Ada environment): `bash scripts/g0/deploy_phase1.sh` (deploys the committed state and records the commit), then `sbatch [-A nlp] run_phase1.sbatch <name> <flores|wikitext> <mask|none> [args]` from `/home2/karma/duan-verify` (examples in the sbatch header). Jobs run **offline from the persistent cache** `/home2/karma/duan-verify/hf_cache` (A-023; refill with `fetch_model.sh` and `prime_hf_cache.py` on the login node). Outputs: `/home2/karma/duan-verify/out/phase1/<name>/` (`summary.csv`, `per_language.csv`, `survival.pt`, `selectivity.pt`, `meta.json`). Analysis of a finished run: `phase1_analysis.run_analysis(run_dir, 'reports/g0/phase1_languages.json', out_dir)` (CPU, minutes).

**Status (2026-10-19): S1-S8 and N1 to N13 done.** Phase 1 (language-selective features; `PLAN.md` S1-S8 and the analyses N1 to N13) has run on top of their code and is **reproduced end to end** from the cached run files by the committed scripts ([`reports/phase1_reproduction.md`](reports/phase1_reproduction.md), D-026). **The state of the evidence is in one place: [`reports/phase1_summary.md`](reports/phase1_summary.md)** (every number generated from the committed result files, a status label per analysis); the full account is `reports/phase1_results.md`. In words:
- the pre-registered selective-versus-control tests are null after correction in the primary regime (BOS masked, equal token budgets);
- a selectivity penalty appears only at strict S thresholds chosen after the fact, and activation magnitude accounts for it;
- the large effect in the data is the resource-tier effect on survival, for selective and universal features alike; it is carried statistically by per-language activation scale (the proposal's "localizes upstream" branch), and character fertility explains only part of it;
- the JumpReLU threshold margin adds nothing beyond raw magnitude; two residuals stay unexplained;
- everything beyond the pre-registered family is additional or exploratory, correlational (27 languages) and labelled so; the BOS-free regime is a sensitivity analysis whose instrument fails its own noise-floor check.
- Open for the team (`DECISIONS_FOR_REVIEW.md`): Q1 (primary regime), Q2 (was the registration frozen), Q13 (magnitude: confounder or mediator), Q14 (which analysis is the confirmatory test of the resource-tier question), Q15 (a GPU diagnostic of why low-resource languages have lower activation scale).
(The step-by-step history is in `reports/progress_log.md`; the newest handoff in `reports/handoffs/` is the only source of the previous chat.)

*Moved to [`reports/archive/handoff_md_stale_sections_2026-10-23.md`](reports/archive/handoff_md_stale_sections_2026-10-23.md) on 2026-10-23: the older status line 'S6 running: Slurm 25745, 25746' (finished long ago; S1 to S8 are done, see the Status paragraph above).*
