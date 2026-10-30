# HANDOFF: how to take over this project

Shortened 2026-10-27 (documentation cleanup requested by the user). The previous text is kept verbatim in [`reports/archive/doc_cleanup_2026-10-27/HANDOFF.md.txt`](reports/archive/doc_cleanup_2026-10-27/HANDOFF.md.txt); older versions: [`reports/archive/root_pages_2026-10-26/HANDOFF.md`](reports/archive/root_pages_2026-10-26/HANDOFF.md), [`reports/archive/handoff_md_stale_sections_2026-10-23.md`](reports/archive/handoff_md_stale_sections_2026-10-23.md).

## 1. Start here

1. Read [`reports/PROJECT_RESULTS.md`](reports/PROJECT_RESULTS.md) (what was found, in plain words) and [current status](reports/CURRENT_STATUS.md) (what is open).
2. Read the newest [numbered handoff](reports/handoffs/) and apply any open choice in [review questions](DECISIONS_FOR_REVIEW.md).
3. Read the [codebase map](docs/CODEBASE.md) before changing code.

## 2. The work loop (stopped)

**No recurring job exists.** The user stopped the 40-minute cron jobs on 2026-10-27 and put the project in finalisation mode (HOLD of 09:50 IST: no deploy, no Ada jobs, no new experiments until the user says so). The old loop and its recipe are history: [`reports/functional_validation/cron_prompt.md`](reports/functional_validation/cron_prompt.md). If work resumes, each step is: orient (current status, newest handoff, git, queue), tests first then code, validate each job before analysing it, write the result with its limits, update current status and the dated log, commit explicit paths and push, write a handoff after a finished task ([`TEMPLATE.md`](reports/handoffs/TEMPLATE.md)). **[Correction 2026-10-29: stale. A recurring 40-minute job has existed since 2026-10-27 17:34 IST (the GPTQ extension, D-101; later carrying the survival measurement study D-117 to D-124 and the 9B review D-125); its instructions are in [`reports/gptq_extension/cron_prompt.md`](reports/gptq_extension/cron_prompt.md). The HOLD of 2026-10-27 09:50 was lifted by the user the same day (D-101) and again on 2026-10-29 (D-121, D-122, D-126).]**

## 3. Rules that matter (the user's, durable)

- **Never remove, shorten or overwrite progress tracking** (2026-10-23): decisions, questions, logs, handoffs, plans, designs, source registers, audits, the archive. Cleanup means moving text verbatim to `reports/archive/` with a pointer, adding pointers or `Update <date>` lines, or deleting byte-identical generated copies. Map: [`reports/INDEX.md`](reports/INDEX.md).
- **Sample-size floor for Study G and its replications** (2026-10-24, D-075): 27 languages x 6 states x 300 sentences x 3 control draws. Follow-ups only add.
- **Queue every planned Ada job at once** (2026-10-25, D-077) so the GPUs never idle; at most 2 run at once. Never poll. The old 32 GPU-hour ceiling is soft (A8).
- Methodological correctness is non-negotiable. Plans before code; predictions frozen before the numbers; a deviation is a D-entry with its reason.
- Regime `mask` only (D-028); no numerical comparison with Duan's `none` regime. Gemma-2-9B is another team's (D-089, D-092). **[Correction 2026-10-29: since D-125 the review and integration of the EXISTING 9B original-RTN results on main is part of this project (`reports/functional_validation/gemma9b/`); the 9B implementation branch is `gemma9b-confirmation` (do not push to it without the user's word); no 9B GPTQ or matched-RTN result exists.]**
- Hugging Face dataset stays private; delete nothing on Hugging Face or wandb.
- Codex (another AI agent) also works in this repo. Never edit, stage or commit her uncommitted files; stage only your own explicit paths.

## 4. Do-not list

- Do not edit `third_party/duan2026/`, the baseline constants or the comparison bands. Do not describe diagnostics as replication. Do not resurrect the legacy pipeline into the working tree (check out the tag and port the idea as a tested diff).
- `paper/` prose may be written (D-093, 2026-10-26) but `paper/course-report/acl.sty`, `acl_natbib.bst` and the official template copies stay untouched; numbers only from committed results with status and limits; the official checker must stay `All Clear`. Index: [`paper/README.md`](paper/README.md).
- Do not touch the user's `plan02-train` jobs, other users' files, `~/.netrc`, `~/.env`, `~/.ssh`, HF tokens. Do not fill `/home2`. No long jobs on the login node. Never accept a changed Ada host key (`ssh -o HostKeyAlias=ada-gw1 karma@10.4.24.30`).
- Do not commit data, weights, `.pt` files, per-feature CSV dumps or secrets. Commit small CSV/JSON/MD results under `reports/g0/`.

## 5. Environment facts that cost time

- **Cluster.** Slurm. Accounts `nlp` (QOS normal, no per-user cap) and `research` (QOS medium: max 4 GPUs, **40 CPUs**, 125 GB per user, 1 node per job). Only partition `u22` is open. GPUs reachable are 11 GB-class only: GTX 1080 Ti (gnode001-042) and RTX 2080 Ti (gnode044-092, `--constraint=2080ti`).
- **CPU cap trap.** `-c 8` with `--mem=40G` is inflated by Slurm to 14 CPUs (and 18 for the 2-GPU jobs), so several jobs on `research` hit `QOSMaxCpuPerUserLimit`. `-A nlp` starts immediately. Check `scontrol show job <id> | grep NumCPUs`.
- **Software.** `/home2/karma/duan-verify/env`: Python 3.10, torch 2.14.1+cu126, transformer-lens 3.9.0, sae-lens 6.54.5, transformers 5.19.0, wandb 0.30.0, ipython 8.39.0 (freeze in `reports/g0/`). Their `requirements.txt` was unpinned; these versions are part of the result.
- **Hugging Face.** The gated models need a token already stored on Ada (`/home2/karma/duan-verify/hf/token`; jobs use `HF_TOKEN_PATH`). **Never copy or print a token.** **Do not set `HF_HUB_DISABLE_XET=1`** (15x slower, drops connections).
- **wandb.** Key in `~/.netrc` (account `karma-skz`, entity `karma-skz-`), project `anlp-project`. Never ask for or handle a key.
- **Fetching results.** One `scp` per file, then `sha256sum` against [`artifact_checksums.md`](reports/functional_validation/artifact_checksums.md). With rsync filters, excludes must come before includes.
- **Tests.** `python3 -B -m unittest discover -s scripts/g0/tests` (stdlib) and `OMP_NUM_THREADS=4 ~/.venvs/anlp-torch/bin/python -B -m unittest discover -s scripts/g0/tests_torch` (torch). Python reuses stale `.pyc` for same-size edits within one second: run mutation checks (`scripts/g0/mutcheck.py`) with `python -B`. Deploy to Ada only with `bash scripts/g0/deploy_phase1.sh`, and only when no job of ours is running. Link check: `python3 -B scripts/g0/check_md_links.py`.
- **Throughput.** About 1.5k token passes per second on a 2080 Ti (bf16 emulated). Study G costs about 3 GPU hours per state.

## Current reporting update, 30 October 2026

The current task finalizes the completed work and its research paper: [final questions and answers](reports/FINAL_RESEARCH_QUESTIONS.md), [paper](paper/research-paper/main.tex), [evidence through 29 October](reports/CURRENT_STATUS.md). No recurring job exists following the user's later instruction to delete all cron jobs; the 27 October and earlier statements above are historical. The Gemma-2-2B three-arm comparison, measurement study, limited Llama check and 9B original-RTN review are complete. Only the 9B GPTQ extension is pending; no result is assumed. Matched RTN on 9B and the precision audit are future work. Do not infer an instruction to run them from old plans.

Report defined causal effects, task differences, positive associations and nulls with their actual endpoints and scope. Limited model/layer/dictionary coverage constrains generalization rather than invalidating controlled estimates. The final paper questions are a retrospective synthesis; original frozen rules and failed criteria remain intact. Belebele's primary effect interval is in nats per item, not accuracy points; F5 effect changes do not imply erased feature identity. Preserve evidence, freezes and history, including the P3a sign error and non-falsifiable P3b on 9B.
