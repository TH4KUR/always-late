# Handoff 9b-001: Gemma-2-9B three-arm confirmation, built and tested, ready for Jarvis (2026-10-28 IST)

> Own numbering series (`9b-NNN`) on purpose: this branch (`gemma9b-confirmation-v2`) was written beside main, whose cron allocates the numbered handoffs and D-numbers; renumber when merging. If this file and the repository disagree, the repository wins.

## 0. Read this first (the standing rules, in the user's words)
- 2026-10-28: "i at least want to do the runs using GPTQ now, and not RTN." / "keep th4kur's scripts and changes, make updates to that so that the 9b runs are confirmatory for our updates as well and not just the ones we planned at that time. the project has changed since then." / "you can update the implementation part uptil the point of running it on jarvis. we will do that once you are done with this." / "both tier 1 and 2 are important. prepare for both." / "Native, go ahead".
- The 9B branches of Shardul0007 (`gemma9b-adapter`) and th4kur (`gemma9b-confirmation`) were merged into a NEW branch and are untouched. Nothing was pushed.

## 1. Mission in five lines
Make the Jarvis run of Gemma-2-9B confirmatory for the project as it now stands: three weight-state arms (legacy RTN = the frozen run, matched RTN, GPTQ), tier 1 = Study G, tier 2 = Phase 1 survival and loss; predictions frozen before any 9B number (spec section 5, amendment A1); everything built and tested up to the point of running. Machine: 1x RTX PRO 6000 Blackwell, 96 GB VRAM, 160 GB RAM, 250 GB SSD, about 210 INR/h, `ssh -o StrictHostKeyChecking=no ubuntu@151.185.34.68` (paused until the user resumes it).

## 2. State right now
- Worktree `/home/karma/Documents/ANLP/project/gemma9b-v2`, branch `gemma9b-confirmation-v2` (origin/main a205fa3 + both 9B branches merged + this work). Spec `docs/superpowers/specs/2026-10-28-gemma9b-three-arm-confirmation-design.md`, plan `docs/superpowers/plans/2026-10-28-gemma9b-three-arm-confirmation.md`, ledger (git-ignored) `.superpowers/sdd/2026-10-28-gemma9b-three-arm-confirmation/progress.md`.
- Tests (counts are from the final run, see section 3): stdlib `cd scripts/g0 && python3 -B -m unittest discover -s tests`; torch `OMP_NUM_THREADS=4 ~/.venvs/anlp-torch/bin/python -B -m unittest discover -s tests_torch`; TransformerLens `~/.venvs/anlp-tl/bin/python -B -m unittest discover -s tests_torch -p "test_gptq_ext_*.py"`.
- NOT done: any real run; the pool copy from Ada (needs the user's OK: `make_jarvis_bundle.sh --pool-from-ada`, about 1 MB; the pool can also be rebuilt on the Ada login node, `reports/gptq_extension/stage_commands.md`); no push.

## 3. What this step did
- **New code** (`scripts/g0/`): `gptq_ext/tokenizer_check.py` (the 2B pool is reused only if the 9B tokenizer is the same), `gptq_ext/ram_guard.py` (host-RAM estimate: 59.1 GB for a one-weight-set unit, 97.8 GB for tier 2, 117.2 GB for a two-bit-width gate job and 97.8 GB for the INT7 gate job), `gptq_ext/record_hashes.py` (hash file of the 9B artifacts, `--check` for a resume), `phase1_gemma9b_ext.py` (tasks `lm`, `p1`, `gate` under the unchanged adapter setting; through `gptq_ext/run_with_weights.py` they get GPTQ or matched RTN), `phase1_gemma9b_compare.py` (Q0 to Q3, S1 to S4, both verdicts, the 2B control), `make_gemma9b_amendment.py`, `make_jarvis_bundle.sh`; `jarvis_run_all.sh` (Shardul0007's) extended in place; test helper `tests_torch/gptq_ext_tiny.py` got an optional `dims=`.
- **Findings while building:** (a) the calibrator and canonical adapter need no change for the 9B layout (18 new tests, whole GPTQ-extension suite green); (b) the spec's 2B reference values for Q1 were ratios of medians; with the statistic Q1 actually uses (median of per-language ratios, committed `curves.csv`) they are 1.91 / 1.58 / 1.24, GPTQ below legacy in 25 of 27 languages: all thresholds hold on the 2B data (corrected in the spec before freezing); (c) the committed 2B arms through the tier-2 code give survival 85.0 / 89.7 / 97.5 % (INT6) and 12.7 / 18.7 / 30.5 % (INT4), S1, S2, S3 met: the thresholds are consistent with the data they come from; (d) the stub rehearsal of the real driver path found two real defects (slot polling, RAM figure) before they could cost Jarvis hours; (e) torch 2.14.1 has wheels only for CUDA 12.6 and 13.0: a Blackwell card needs the 13.0 build and driver >= 580 (the script chooses and checks).
- **Decisions (to be given D-numbers when merged into main; the amendment says "D-089, amendment A1"):** the three arms and both tiers (user), thresholds Q0 to Q3 and S1 to S4 (spec section 5), Q1 evaluable only if the legacy arm meets P3b as a whole, tier-2 weight-set units run alone, `STATES_NEW=6 4` (the states of the 2B extension), no multi-device GPTQ (the Jarvis card holds the model), no D-entry written on this branch (main's cron allocates them).
- Mutation checks: `phase1_gemma9b_compare.py` 18/19 killed (1 equivalent), `ram_guard.py` 5/5, `tokenizer_check.py` 5/5.

## 4. The next step, exactly
1. The user reviews this branch and says go. 2. (needs the user's OK) fetch the pool: `bash scripts/g0/make_jarvis_bundle.sh <outdir> --pool-from-ada --verify-python ~/.venvs/anlp-torch/bin/python` (refuses an uncommitted tree: commit first). 3. The user resumes the Jarvis machine; `scp` the three files (the script prints the commands); on Jarvis: `sha256sum -c SHA256SUMS`, unpack, `DRY_RUN=1 bash scripts/g0/jarvis_run_all.sh`, then the real run (CLAUDE.md). 4. After `gate9b` and `smoke_gptq` the seconds per build and the peak memory decide whether tier 2 fits; the drop order is in CLAUDE.md. 5. Analysis on a CPU machine with `phase1_gemma9b_compare.py` (command in CLAUDE.md), then a D-entry and the results into the report.

## 5. Open questions for the user
- Q-A: OK to copy the calibration pool from Ada (read-only, about 1 MB)? Default: wait for the answer.
- Q-B: push `gemma9b-confirmation-v2` to the shared remote (the bundle does not need it)? Default: no push.
- Q-C: the amendment A1 text is frozen when its file is committed; the user may still change thresholds before that commit (spec section 5).

## 6. Environment and pitfalls
- Local venvs: `~/.venvs/anlp-torch` (no TransformerLens), `~/.venvs/anlp-tl` (CPU TransformerLens of Ada's version). `pkill -f <script name>` kills the calling shell when the name is in the command line: use `kill <pid>`.
- The 9B freeze pins four files (see CLAUDE.md); the amendment pins five new modules: any edit to them after the amendment commit makes the `checks` step fail by design.
- The calibrator keeps a host copy of the original targets (17.6 GB) and the gate keeps two full-precision copies: the numbers behind the RAM guard.

## 7. What I would check if something looks wrong
`bash -n scripts/g0/jarvis_run_all.sh`; `DRY_RUN=1` plan; `python3 scripts/g0/make_gemma9b_freeze.py --verify` and `make_gemma9b_amendment.py --verify`; `git log --oneline origin/main..HEAD`; the stub rehearsal `python3 -B -m unittest tests.test_jarvis_rehearsal`.
