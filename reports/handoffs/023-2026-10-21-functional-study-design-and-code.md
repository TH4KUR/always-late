# Handoff 023: functional-validation study designed, prior art checked, code Tasks 1 to 3 and the T-LM engine done, smoke job 33771 queued (2026-10-21, 23:40 IST)

> The ONLY source of this chat for a fresh session, together with the repository. If this file and the repository disagree, the repository wins: say so in the next handoff. Start with `reports/functional_validation/STATUS.md` (NEXT STEP block), then the design and plan.

## 0. Read this first (standing rules, the user's words where quoted)

- Project: ANLP "Does Compression Forget Language?" in `/home/karma/Documents/ANLP/project/always-late` (Gemma-2-2B, layer-12 Gemma Scope 16k SAE, RTN weight quantization, 27 FLORES-200 languages). Submission 2026-10-31. Remote `origin` GitHub `TH4KUR/always-late` (private), branch `main`.
- User instruction of this chat (2026-10-21, verbatim parts): Q16 answered A/B/C/F; "work on it, make plans, and setup a cronjob for every 40 minutes to check the ada jobs if anything is queued, add more implementation code, analyse the results and make plots figures and other things for comparison, add questions to the decisions docs, update all documentation, clean up old documentation as needed to remove redundancies. the cronjob can do any of these as per the priority of work available to it. the main thing is to continue writing code and queueing the jobs based on my answers. establish the research questions we are answering and working towards strengthening those, and if done, work towards our original proposal. novelty and publishability are priority, and correctness of methodology is non-negotiable. always make plans first before implementing and ensure that it is sensible. if you have pressing questions for me, add them to the DECISIONS_FOR_REVIEW.md document and tell me when you cant proceed at all without further answers. try to work autonomously till then, at least for the next 5-6 iteration of the scheduled job."
- Rules still in force (earlier chats): mask regime only, no `none` work, no numerical comparison with Duan (D-028); never edit `paper/` or `third_party/`; no secrets/tokens/weights/large artifacts in files or commits (never print the HF token); secret scan before every commit; stage explicit paths (never `git add -A`, reset, checkout, stash); commit messages end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`; at most 2 Ada jobs, record the Slurm id, never wait or poll, never redeploy while a job runs; do not change the Hugging Face dataset's visibility, delete nothing on HF or wandb; tests first (watch them fail), mutation-check with `scripts/g0/mutcheck.py`; important decisions to `DECISIONS_FOR_REVIEW.md` / `reports/decisions_and_anomalies.md`.
- **The 40-minute job runs** as two session-only recurring cron entries (`7,47 0-22/2 * * *` and `27 1-23/2 * * *`, created 2026-10-21, auto-expire after 7 days, i.e. about 2026-10-28; the scheduler adds jitter). Prompt and re-creation instructions: `reports/functional_validation/cron_prompt.md`.

## 1. Mission in five lines

D-039/D-040: do the originally language-associated SAE features carry function for their language beyond activity-matched controls (F1), is that kept in each quantized model against its own baseline (F2), where does quantization damage sit (FP-donor patching, F3), does association retention (D-035) predict functional retention (F4, two-sided), plus SAE fidelity (B), closer analyses of existing files (A) and figures/tables (F). Tasks: next-token NLL on FLORES devtest (27 languages, exploratory) and Belebele multiple choice (languages with measured headroom; calibration = dev-origin passages, evaluation = devtest-origin, frozen before scoring).

## 2. State right now (verified 2026-10-21 23:35)

- Git `main` at `1dbe7a1` (pushed) plus this handoff's documentation commit. Tests: stdlib 205 (`python3 -B -m unittest discover -s scripts/g0/tests`), torch 459 (`OMP_NUM_THREADS=4 ~/.venvs/anlp-torch/bin/python -B -m unittest discover -s scripts/g0/tests_torch`), green.
- Ada: **job 33771 `func-smoke`** (FP and INT6, eng_Latn hin_Deva khm_Khmr, 40 pilot sentences), deployed commit `1dbe7a1`, output `/home2/karma/duan-verify/out/phase1/func-smoke/`, log `/home2/karma/duan-verify/logs/func-smoke-33771.out`. Nothing else queued. `/home2` has about 5.6 GB free.
- Belebele (revision `7899cdfa...`): 27 files on Ada in `phase1/data/belebele` with `MANIFEST.json`; partial local copy in `data/belebele/` (git-ignored; copy still running or finished: check `ls data/belebele | wc -l`).

## 3. What this chat did

Committed Codex's D-039 documentation (`ff1522f`); design, plan, D-040, Q17, status, cron prompt (`c0bafbc`); Task 1 (`db8f04c`), prior-art file and register P30 to P38 plus mean-ablation kind (`0f94dd7`, `eecb653`), Task 3 (`fffd487`), engine and config builder and real-model main (`1dbe7a1` and the commit before it). Decisions taken before any outcome existed: nearest-neighbour matching of controls (cells were too coarse), FLORES-origin split of Belebele, mean ablation added, difference-in-means baseline recorded as optional. Prior art: no paper combines all elements; scoop risk from Duan 2606.03002; F3's method is known (Chang et al., Sathyanarayanan), only the language-group framing is new; F4 has a null precedent (Borobia 2603.25325).

## 4. The next step, exactly

See the NEXT STEP block of `reports/functional_validation/STATUS.md`: check 33771; first look at effects; Task 6 analysis module; T-MC readout and the Belebele calibration job; rescue; freeze; evaluation jobs.

## 5. Open questions for the user

Q17 (confirm design defaults: tasks, group size K=64, decision bands, compute, cadence; not blocking). Earlier open: licence of our own statistics, shard backup, Q6, Q7, Q8, Q11, Q12. **Scoop risk** (Duan's causal follow-up on the same setup) is worth the user's attention for the publication plan.

## 6. Environment and pitfalls

- Local venv `~/.venvs/anlp-torch` (torch, numpy, scipy; no matplotlib: install with pip before figures, or run plotting on Ada where matplotlib 3.10.9 exists). Ada: `ssh karma@ada-gw1`, project `/home2/karma/duan-verify`; deploy with `bash scripts/g0/deploy_phase1.sh` only when `squeue -u karma` is empty; submit with `sbatch -t HH:MM:SS phase1/repo/scripts/g0/run_functional.sbatch <name> <args>`. TransformerLens 3.9.0: `model(tokens, stop_at_layer=13)` returns the residual after block 12; `model(resid, start_at_layer=13)` returns logits.
- The recurring job fires only when the REPL is idle. Do not run `pkill -f` with patterns that occur in an ssh command line.

## 7. What I would check if something looks wrong

`git log --oneline -8`, `git status --short`, `CronList` (two entries with the 40-minute prompt), `ssh karma@ada-gw1 squeue -u karma`, the two test suites, `python3 scripts/g0/phase1_func_groups.py verify` (see STATUS for the command), and that `reports/functional_validation/groups_manifest.json` still equals a fresh rebuild.
