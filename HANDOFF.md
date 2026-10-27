# HANDOFF: how to take over this project

Rewritten 2026-10-26 (documentation cleanup requested by the user). The previous text of this file is kept verbatim in [`reports/archive/root_pages_2026-10-26/HANDOFF.md`](reports/archive/root_pages_2026-10-26/HANDOFF.md), including the baseline runbook and the Phase 1 build notes. Older sections are in [`reports/archive/handoff_md_stale_sections_2026-10-23.md`](reports/archive/handoff_md_stale_sections_2026-10-23.md).

## 1. Start here

1. Read [current status](reports/CURRENT_STATUS.md): current evidence and the next action.
2. Read the newest [numbered handoff](reports/handoffs/) and its latest dated update.
   Handoff 045 covers external results, recovery and precision-audit implementation;
   handoff 046 records the modular cleanup and preservation checks.
3. Read [research direction](RESEARCH_DIRECTION.md) for claim limits and the
   [codebase map](docs/CODEBASE.md) before changing implementation.
4. Apply any unanswered/unapplied user choices from [review questions](DECISIONS_FOR_REVIEW.md).
5. Check the queue once when execution work requires it, using the SSH runbook below.

## 2. The work loop

Work is driven by a 40-minute recurring job. Its full instruction set is the fenced block in [`reports/functional_validation/cron_prompt.md`](reports/functional_validation/cron_prompt.md), which also has the recipe to recreate the entries; they expire after 7 days and die with the Claude session. Without the cron, do the same by hand. Each step:

1. orient (CURRENT_STATUS, newest handoff, git, queue);
2. take the first item of the priority list;
3. tests first, then code;
4. validate each job before analysing it;
5. write the result with its limits;
6. update `reports/CURRENT_STATUS.md` and append to the dated log;
7. commit explicit paths and push;
8. after a finished task, write a handoff (next free number, [`TEMPLATE.md`](reports/handoffs/TEMPLATE.md)).

**Execution priorities:** maintained only in [current status](reports/CURRENT_STATUS.md).
Update that page after a completed step, even when this runbook or an older status
page has uncommitted documentation changes. Keep detailed evidence in dated handoffs,
decisions and the experiment ledger. Do not resume an old task from a historical banner.
Gemma-2-9B remains outside this routine (D-092).
Previous text is [preserved verbatim](reports/archive/modular_cleanup_2026-10-27/README.md).

## 3. Rules that matter (the user's, durable)

- **Never remove, shorten or overwrite progress tracking** (2026-10-23): decisions, questions, logs, handoffs, plans, designs, source registers, audits, the archive. Cleanup means moving text verbatim to `reports/archive/` with a pointer, adding pointers or `Update <date>` lines, or deleting byte-identical generated copies. Map: [`reports/INDEX.md`](reports/INDEX.md).
- **Sample-size floor for Study G and its replications** (2026-10-24, D-075): 27 languages x 6 states x 300 sentences x 3 control draws. Follow-ups only add.
- **Queue every planned Ada job at once** (2026-10-25, D-077) so the GPUs never idle. At most 2 jobs run at once (the scheduler enforces it). Never poll.
- **Publishability and novelty first; methodological correctness is non-negotiable.** Plans before code. Predictions are frozen before the numbers. A deviation is a D-entry with its reason.
- Regime `mask` only (D-028); no numerical comparison with Duan's `none` regime.
- Hugging Face dataset stays private; delete nothing on Hugging Face or wandb.
- Codex (another AI agent) also works in this repo. Never edit, stage or commit her uncommitted files; stage only your own explicit paths.

## 4. Do-not list (verbatim from the previous version of this file)

- Do not edit `third_party/duan2026/` by hand, add edits, change the baseline constants, or change the comparison bands. Do not describe diagnostics as replication.
- Do not resurrect the legacy pipeline into the working tree. If something from it is needed, check out the tag and port the *idea* as a tested diff on their code.
- **Never remove, shorten or overwrite progress-tracking documents** (decisions, questions, logs, handoffs, plans and designs, source registers, audits, the archive): the user's instruction of 2026-10-23, so that the whole trail of directions explored and considered survives to the end of the project; consolidation = move verbatim into `reports/archive/` with a pointer.
- ~~Do not edit `paper/` prose or write report text.~~ **Update 2026-10-26 12:32 (D-093): the course guidelines no longer forbid AI-generated reports (rule 4 of `project_guidelines.txt`, edited by the user), so `paper/` prose and the report may be written and updated. Still untouched: `paper/course-report/acl.sty`, `acl_natbib.bst` and the official template copies (the ACL style), `third_party/`, and the claim discipline (numbers only from committed results with their status and limits; the course page limits; the official checker must stay `All Clear`). Index of the paper files: [`paper/README.md`](paper/README.md).**
- Do not touch the user's `plan02-train` jobs, other users' files, `~/.netrc`, `~/.env`, `~/.ssh`, HF tokens. Do not fill `/home2`. No long jobs on the login node.
- Do not commit data, weights, `.pt` files, per-feature CSV dumps or secrets. Commit small CSV/JSON/MD results under `reports/g0/`.

## 5. Environment facts that cost time (verbatim from the previous version, still valid)

- **Cluster.** Slurm. Accounts `nlp` (QOS normal, no per-user cap) and `research` (QOS medium: max 4 GPUs, **40 CPUs**, 125 GB per user, 1 node per job). Only partition `u22` is open. GPUs reachable are 11 GB-class only: GTX 1080 Ti (gnode001-042) and RTX 2080 Ti (gnode044-092, `--constraint=2080ti`).
- **CPU cap trap.** `-c 8` with `--mem=40G` is inflated by Slurm to 14 CPUs (and 18 for the 2-GPU jobs), so several jobs on `research` hit `QOSMaxCpuPerUserLimit`. `-A nlp` starts immediately. Check `scontrol show job <id> | grep NumCPUs`.
- **Software.** `/home2/karma/duan-verify/env`: Python 3.10, torch 2.14.1+cu126, transformer-lens 3.9.0, sae-lens 6.54.5, transformers 5.19.0, wandb 0.30.0, ipython 8.39.0 (freeze in `reports/g0/`). Their `requirements.txt` was unpinned; these versions are part of the result.
- **Hugging Face.** The gated models need a token already stored on Ada (`/home2/karma/duan-verify/hf/token`; jobs use `HF_TOKEN_PATH`). **Never copy or print a token.** **Do not set `HF_HUB_DISABLE_XET=1`** (15x slower, drops connections).
- **wandb.** Key in `~/.netrc` (account `karma-skz`, entity `karma-skz-`), project `anlp-project`. Never ask for or handle a key.
- **Fetching results.** One `scp` per file, then `sha256sum` against [`artifact_checksums.md`](reports/functional_validation/artifact_checksums.md). With rsync filters, excludes must come before includes.
- **Tests.** `python3 -B -m unittest discover -s scripts/g0/tests` (stdlib) and `OMP_NUM_THREADS=4 ~/.venvs/anlp-torch/bin/python -B -m unittest discover -s scripts/g0/tests_torch` (torch). Python reuses stale `.pyc` for same-size edits within one second: run mutation checks (`scripts/g0/mutcheck.py`) with `python -B`. Deploy to Ada only with `bash scripts/g0/deploy_phase1.sh`, and only when no job of ours is running.
- **Throughput.** About 1.5k token passes per second on a 2080 Ti (bf16 emulated). Study G costs about 3 GPU hours per state.
