# Handoff 030: F5 smoke result in (the feature's effect survives INT4 better than its activations; association status does not predict it), full F5 jobs queued (2026-10-23, 03:42 IST)

> The ONLY source of the previous chat. A fresh chat with no memory must be able to continue from this file plus the repository. If this file and the repository disagree, the repository wins: say so in the next handoff. Start with `reports/functional_validation/STATUS.md` (State, NEXT STEP, Log); this file supersedes [handoff 029](029-2026-10-23-f5-built-smoke-queued.md) as the entry point ([026](026-2026-10-22-eval-ladder-done-calibration-running.md) holds the standing rules in its section 0, [025](025-2026-10-22-chat-end-functional-study-midrun.md) the user's verbatim 2026-10-21 instruction, the addendum of [028](028-2026-10-23-tmc-analysed-null-floor-and-rescue-running.md) the user's verbatim 2026-10-23 message; all still apply unchanged).

## 0. Read this first

The standing rules are those of handoffs 026, 028 and 029 section 0 (mask regime only; no `none` work; no comparison with Duan; never edit `paper/` or `third_party/`; no secrets, tokens, weights or large artifacts; explicit `git add` paths only; commit messages end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`; at most two Ada jobs, record Slurm ids, never wait or poll (background watchers, started as the background command itself); Hugging Face private, delete nothing, no upload without the user's go-ahead (Q21); tests first, mutation check, evidence before claims; a frozen setting is never changed silently; do not merge or cherry-pick the parallel branches (D-052); do not work around the gated Llama repository (Q22); never accept a changed ssh host key; never touch Codex's uncommitted files). Working rules: doc edits by script file or quoted heredoc and read back (a regex with doubled backslashes in a quoted heredoc silently matched nothing once: check the `Updated:` line); chain commits with `set -e`; timestamps from `date`; the shell's working directory can reset, use absolute paths.

## 1. Mission in five lines

D-039/D-040 (functional validation F1 to F4) is analysed (handoff 028). **D-053 (the user's direction of 2026-10-23): F5 establishes whether a feature performs the same function before and after quantization, because the survival metric is meaningless otherwise.** Read side (activations) and write side (the logit effect of switching the single feature off), a numerical floor, a matched negative control and the pre-proposed rule S1 to S3 (Q23, amended by D-057 with minimum counts). Llama-3.2-1B is kept as a supporting model (D-052; waits for Q22).

## 2. State right now (verified 03:42)

- Git `main` (pushed; the last commit of this chat is in `git log`), working tree clean. Tests: stdlib 265, torch 1052.
- **Ada:** `f5-a` **35424** (F5 full job A: states FP, INT16, INT8; stratified subset `--per-cell 12 --subset-seed 0` = 1,487 features; snapshot `repo_530e655`; `-t 04:30:00`; output `/home2/karma/duan-verify/out/phase1/f5-a/`, log `/home2/karma/duan-verify/logs/f5-a-35424.out`) and `func-rescue-a` 35319 (expected about 04:45). **Watcher `bb1ddeg4t` (session-only) submits `f5-b` (states FP, 6, 5, 4; `-t 05:30:00`) when fewer than two jobs are queued;** the by-hand command is in STATUS "Immediate". Other watchers: `bo5hl7fur` (35319).
- Cron entries `955a5cc5`, `e33ac7eb`, `ea207eac` (session-only, expire about 2026-10-29).

## 3. What this chat did since handoff 029

1. **Floor job `func-eval-d` 35314 analysed** (nine precisions in `results/eval/`, figures regenerated): INT0 retention 1.0000, INT16 1.0002, INT12 1.0006. **A-042 / D-055:** INT16 and INT12 are NOT identical to full precision (no-edit per-token NLL differs by 0.0065 nats, INT8 by 0.011), so INT16 is an F5 state and the floor (floor = lower 5th percentile of W1 of FP-rebatched and INT16; adequate only if both medians are at least 0.7; INT8 reported, not in the floor).
2. **F5 smoke job 35399** (150 evenly spaced features, states FP, FP-rebatched, INT16, INT8, INT4; 32 min, exit 0; files checksummed in `artifact_checksums.md`): gates passed, sham exact, 1.5 s per feature and state, peak GPU 8.0 GB. **Exploratory results (`results/f5_smoke/report.md`):** FP-rebatched W1 = 1.000 (batching noise nil); INT16 median W1 0.990, floor 0.898; INT8 0.988; **INT4 0.796, only 3.3 % of the features at or above the floor**; different-feature control median 0.01, p95 0.10; retrieval of identity 1.000 in every state; token correlation at INT4 median 0.666 with only 6.7 % at or above 0.9; so the activation survival criterion understates functional continuity at INT4 on this sample. **D-035 association status did not predict function** (F4 pairs at INT4: median W1 retained 0.728, lost 0.810, descriptive AUROC 0.32 [0.21, 0.44]). All exploratory, small sample.
3. **D-056:** cost-driven stratified subset (`--per-cell 12`, 1,487 features, every role represented), FP-rebatched dropped from the full jobs, two jobs; Q24 asks whether to run the rest. **D-057:** minimum 10 defined features or events for S1 to S3 (otherwise undefined) and a descriptive continuous AUROC S2c.
4. Code: runner `--per-cell`, `DEFAULT_STATES`; analysis `report.md` generator, `--figures`, guard, S2c; figures (a scatter layout defect and an undefined-statement drawing found on the real data); 1052 torch tests, more than 250 mutants killed in total across the F5 modules (the exact per-step counts are in STATUS).

## 4. The next step, exactly

(1) When `f5-a` ends (about 3 h of run time) and `f5-b` ends: validate each (log, `meta_f5.json` complete, sham gate exact, seconds per state), fetch the output folders one file per `scp` (state files about 5 MB each), checksums into `artifact_checksums.md`. (2) Analysis of both jobs together: `~/.venvs/anlp-torch/bin/python scripts/g0/phase1_func_f5_analysis.py --run <A> --run <B> --out reports/functional_validation/results/f5 --figures ../../figures/f5/f5_fig1_w1_distribution_light.png ...` then `scripts/g0/phase1_func_f5_figures.py --run <A> --run <B> --results .../f5_results.json --out reports/functional_validation/figures/f5 --note "exploratory, stratified subset (1,487 features)"`; LOOK at every figure; write the result with its limits (STATUS Log, a D-entry if a decision follows), then a handoff. S1 to S3 are decided by the rule as amended; report whatever the verdict. (3) When 35319 ends: validate, analyse the rescue into `results/rescue/` (FP from `func-eval-a/` plus the three rescue files in one folder), figure 4, checksums. (4) Exploratory K = all Belebele run (Q20 (a)) when a slot is free. (5) Llama-3.2-1B after Q22 (the user must accept the Meta licence for the account of the token).

## 5. Open questions for the user

Q17 to Q20 (defaults in force), Q21 (upload item-level files to the private dataset?), Q22 (Llama licence), Q23 (rule S1 to S3; amended by D-057), **Q24 (run the whole 4,382-feature F5 sample as well as the subset? default: no)**. None blocks.

## 6. Environment and pitfalls

- ssh by name fails (WARP DNS): `ssh -o HostKeyAlias=ada-gw1 karma@10.4.24.30`; `scp -o HostKeyAlias=ada-gw1 karma@10.4.24.30:<path> <dest>` one file per call; `ADA_SSH='ssh -o HostKeyAlias=ada-gw1 karma@10.4.24.30' bash scripts/g0/deploy_phase1.sh repo_<short commit>` (named snapshot only while jobs run). Torch venv `~/.venvs/anlp-torch` for analysis and figures; job environment `/home2/karma/duan-verify/env/bin/python`. The scratchpad is wiped on a session restart (the smoke files are in the repository only as results, the raw files on Ada).
- F5 job script `scripts/g0/run_f5.sbatch <name> [args]`; `meta_f5.json` has per-state seconds, gates, subset; the analysis merges jobs only if they tested the same features in the same contexts.
- The statements are exploratory; the survival metric is "sensible" only if all three are met, for this model, layer, quantizer and sample.

## 7. What I would check if something looks wrong

`git status`, `CronList`, `squeue -u karma` over the strict-key route, whether the watchers are alive (`f5-b` must have been submitted after 35319 ended), the sham gate line and logit gates of each F5 job, the first lines of the log for a GateError, and that the two jobs agree on the number of tested features.

## Addendum 04:18 IST (changes after this handoff was written; the repository and STATUS.md win)

- **The rescue job `func-rescue-a` 35319 COMPLETED and is analysed (D-058):** `results/rescue/`, `figures/rescue/fig4_rescue_*`, `table_rescue.*`, checksums in `artifact_checksums.md`. The language group's full-precision activations recover 2.6 % of the INT4 loss gap (matched controls 1.2 %, group minus controls 1.4 %), nothing at INT5 and INT6; all SAE features 17.4 % at INT4; the whole layer-12 residual 29.7 to 39.3 %, so 60 to 70 % of the damage arises in the quantized blocks above layer 12; partial patches can make the loss worse (components are not independent).
- **F5 job B is `f5-b` 35435** (submitted by the watcher at 04:15; states FP, INT6, INT5, INT4). `f5-a` 35424 is running (started 03:34). Both watchers have finished; no watcher is needed any more (see STATUS "Immediate" for what to do when each job ends). Slots are full until `f5-a` ends (about 06:40); the K = all Belebele run waits for it.
- Scale test of the F5 analysis on a synthetic full-size world: 5.7 s, 312 MB. State files of the F5 jobs are about 5 MB each.

