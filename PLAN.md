# Team Plan

## Current plan (updated 2026-10-16): replicate Duan's code as the baseline, then diagnose

Context: our "replication" of Duan (2026) was an independent re-implementation never validated against their code (`issues.md` B1). Their released data show a degraded regime (no BOS, baseline perplexity 458.5). Final submission: **Oct 31, 2026**. **For a new agent: start with `HANDOFF.md`.** Evidence, verified Ada facts and the closure protocol: `reports/duan_codebase_audit.md` (section 4 venue, 6 runs, 7 closure protocol). Their code is at `../duan-sae-feature-survival/` (outside this repo, read-only).

**Principle:** no new claim goes into the paper until the Duan question is closed, and the baseline is a **replication of their code, not a modification of it**.

### Layer 1: BASELINE (their code, vendored; DONE, Q1 passed)
- Code: `third_party/duan2026/` = their repo downloaded fresh, pruned to the Gemma Phase-3 study. The one script we run is their file plus a short whitelist (`UPSTREAM.md`): delete their unrunnable tail cells (22+), their own `RUN_PRUNING` flag off, and four device references so the second bf16 model sits on a second 11 GB card. No generator, no rewrite; `display()` works unedited under IPython.
- Proof: `scripts/g0/tests/test_vendor.py` rebuilds the working copy from their pristine file plus the whitelist and requires byte equality over the whole file (top-level statements included); retained files must match the recorded upstream hashes (upstream content tree hash `a84e43da...adc`); a committed mutation suite (13 mutations) must be rejected. The launcher refuses to run unless the deployed file matches `working.sha256`.
- Launcher: `scripts/g0/run_baseline.sbatch <arm> <full|smoke>`; runs their script under IPython, then compares with their published table (`scripts/g0/baseline_report.py`, pre-committed bands) and uploads to wandb post hoc.
- Configuration: their shipped `3b_full` (500k tokens, INT8/7/6/5/4), pruning off. ~4.1 h compute.
- **Q1 passes** iff INT7 survival is within 5 pp of 81.26%, INT6 within 10 pp of 51.30%, active features within 10% of 7,144, perplexity signs hold (INT7 improves, INT6 worsens). If it fails: do not touch their code; rerun once, record versions, report.

### Layer 2: DIAGNOSTICS (modified code, always labelled as such; only after Layer 1 and the user's OK)
v3 runner (`make_duan_runners_v3.py`: BOS arms, INT0 control, ladder 0/16/12/8/7/6, snapshots, resume, wandb heartbeats), R1-R3 BOS arms, R4 their script 06, R5 our pipeline on their exact tokens, gates and dispositions in audit section 7. These explain *why* the pipelines differ; they are not part of the replication. **Audit in progress (user request 2026-10-16: "audit why our implementation was so off compared to theirs")**: `reports/audit_our_vs_duan.md` lists 12 differences between the pipelines and fixes the reading rules before the results exist; R2 (their code + BOS, job 23643) and R5 (our pipeline on their tokens, one factor flipped per cell; smoke 23648 -> R5a 23649, R5b 23652) are queued/running; HANDOFF section 11 says what to do on the next prompt.

### Venue (verified on Ada 2026-10-15/16)
- **Ada `u22`, 2 x RTX 2080 Ti per job** (`--constraint=2080ti --gres=gpu:2`); `model_ref`+SAE on `cuda:0`, `model_work` on `cuda:1`.
- **No GPU larger than 11 GB is reachable** from our accounts (3090/3080 Ti partitions closed to `nlp`/`research`).
- Gotchas that already cost us jobs: `/share1` is **not mounted on compute nodes** (and has a 3,000-file quota); `/home2` has ~19 GB free and is shared with other jobs (keep the model off it: each job fetches the 10.5 GB model to node-local `/scratch`, ~18 min, deleted afterwards); never set `HF_HUB_DISABLE_XET=1`.
- Not usable: this laptop (6 GB GPU, ~100 KB/s HF downloads), gnode001-042 (GTX 1080 Ti).
- Limits: QOS `medium` caps us at 4 GPUs per user, shared with the semeval job; use `-A nlp` for parallel arms (untested).
- **Logging: wandb** (project `anlp-project`, entity `karma-skz-`; key already in `~/.netrc` on Ada). Baseline results are uploaded after the run from their output files; the v3 runner also logs heartbeats during the run. `G0_WANDB=0` disables.
- **Measured cost (TransformerLens, 2 x 2080 Ti):** peak GPU 8.2 of 11 GiB per card, host RSS 21.8 GB, perplexity pass 3.3 it/s, streaming pass 1.08 it/s: **40 min per condition at 500k tokens**.

### Status
| # | Step | Status |
|---|---|---|
| 0 | Download Duan's repo; audit; verify released tables offline | **done** (their INT6 per-feature data reproduces their headline row exactly) |
| 1 | Probe Ada; build their stack; benchmark; independent re-verification by a second reviewer | **done** |
| 2 | `hf auth login` on Ada; wandb logging (test-first, live-tested) | **done** |
| 3 | v3-runner smoke R0 (job 23355): TL works on both cards, 8.2 GiB peak. Early signal at 20k tokens: INT6 survived 49.8% (published 51.3%), baseline ppl 414 (458) | **done** |
| 4 | Vendor their repo (fresh download, byte-identical twice), prune to the Gemma Phase-3 study, whitelist edits, exact-reconstruction + mutation tests | **done** (43 tests pass); the earlier generator-based "faithful runner" was removed |
| 5 | Smoke of the vendored file on Ada (their `3a_test` mode, job 23463) | **done: exit 0, bit-identical to the two earlier smokes** (max abs diff 0.0 over all summary columns and 3 x 16,384 per-feature correlations); `reports/g0/baseline_smoke_vendored/` |
| 6 | **Baseline: their `3b_full` (pruning off) on Ada**, job 23472, gnode057, 4 h 08 min, exit 0 | **done** |
| 7 | Apply Q1; record in `reports/progress_log.md`; archive `reports/g0/baseline_full/` | **done: Q1 PASSED**: all five rungs within 0.15 pp survival / 0.0008 median r of their published table; per-feature Spearman 0.9996; baseline ppl 458.4 vs 458.5; INT7 perplexity -5.61% vs -5.65% (HANDOFF section 9) |
| 8a | **Audit of the disagreement**: `reports/audit_our_vs_duan.md`; R5 code (`r5_run.py`, `r5_lib.py`, `run_r5.sbatch`, 31 CPU tests + mutation checks), `r5_compare.py` (10 tests) | **done** (found so far, statically: HF `sdpa` skips Gemma-2 attention soft-capping, our `src/` uses sdpa; a TL-RMSNorm claim in `issues.md` was wrong and is corrected) |
| 8b | **R2** (job 23643) and **R5** (smoke 23648, R5a 23649, R5b 23652) on Ada | **running / queued** (2026-10-16); results are read on the next prompt (HANDOFF section 11) |
| 8c | R1/R3 (superseded by the R5 BOS cells), R4 (their script 06), the closure protocol (audit section 7) | optional, after 8b |
| 9 | Re-derive the confirmatory family at discriminating bit-widths (INT6-INT4), language-level permutation test with a script covariate and TOST, content-matched sentence sets, activation-magnitude matching, BOS excluded (`issues.md` section 7) | after 8 |
| 10 | Rewrite abstract/discussion/README results to match `reports/` (`issues.md` A1-A6); remove synthetic figures. **Prose must be written by the team (course policy).** | after 9 |

**Honest limits:** thresholds in the closure protocol are judgement calls; their script has no resume (adding it would modify their loop), so a failed baseline job is rerun whole; we never ran their unmodified script (no >= 16 GB GPU reachable), so equivalence of the four device edits is argued, not demonstrated on a large GPU; Ada's stack (transformers 5.19 / torch 2.14) differs from our paper's (5.17 / 2.13); the protocol is terminal for this question but cannot stop a reviewer asking a different one later.

---

## Historical plan (mid submission, superseded by the section above)

Deadline: **Oct 2, 2026** (3 days out as of this writing). Full guidelines: `project_guidelines.txt`. Full research design: `anlp_revised.txt`.

## What's required (project_guidelines.txt, Section 3)

- A **7-8 page document** (excl. references, +1 optional appendix page).
- Must cover: a refined problem statement, the progress we've made, a concrete timeline for the rest of the project, and a comprehensive literature review.
- Graded on: the write-up + code (this repo) + a viva. Slides optional.
- Repo must be public or grant TA mentors access. Filename on submission: `AlwaysLate-Mid.zip`. **Every team member submits individually.**
- **AI may be used for implementation/code, but not to generate the report text itself** — Claude has been writing the pipeline code and running experiments; the write-up prose needs to be written by the team.

## Scope for this submission (~40-50% of the full project)

Our own proposal's task list (`anlp_revised.txt`, Section 8) already stages the project into a "guaranteed core" (W0-W5) vs. optional extensions (W5+). That's the natural cut for a mid-submission:

**In scope:**
- W0 feasibility spikes + pipeline setup (done, see PROGRESS.md)
- A reduced-scale quantization/survival validation demo (in progress)
- Full literature review, refined problem statement, updated timeline (not started — team work, see below)

**Explicitly out of scope — say so plainly in the write-up, don't imply otherwise:**
- The full 27-language Phase-1 core survival measurement (Section 4.3; scheduled W3-4)
- Llama-3.2-1B cross-model extension (stretch-only per the proposal itself)
- Pruning contrast (magnitude/Wanda) and the Phase-2 causal ablation check

## Task split (so nothing blocks on one person)

Team: Hemang Joshi, Eashaan Thakur, Shardul Joshi, Nikhita Ravi — self-assign against these, none of this is hard-assigned:

1. **Pipeline & experiments.** Code lives in `src/` + `scripts/`. Once the GPU is confirmed working (see PROGRESS.md — currently blocked), rerun `scripts/run_quantization_survival_demo.py` at full scale (`QSD_TARGET_TOKENS=3000 QSD_BITWIDTHS=8,7,6,5,4`), then consider extending sanity-check-style measurement toward more of the 27-language panel if time allows. Real results only — nothing goes in the write-up that isn't in `reports/progress_log.md`.
2. **Literature review.** Expand `anlp_revised.txt` Section 3 into a full review. Already-verified real citations/resources are listed in `reports/progress_log.md` (Gemma Scope, Duan 2026, Borobia et al. 2026 all checked against live sources). Also worth citing as fast-moving related work: arXiv:2608.25941 ("When Pruning Meets Interpretability") and arXiv:2609.04808 ("Recurrence Is Not Enough: Causally Validating Multilingual SAE Translation Features") — neither overlaps our specific compression x language-selectivity question, which is worth stating explicitly as a novelty check.
3. **Problem statement + timeline.** Tighten the Intro/RQ1-3 framing from `anlp_revised.txt`. Write the "concrete timeline for the rest of the project" section by adapting Section 8's W1-W10 schedule to real remaining dates through the Oct 31 Final Submission, keeping the existing guaranteed-core/staged-extension structure (it's a strength, not a gap).
4. **Progress section + viva prep.** Write "progress made" directly from `reports/progress_log.md` — don't round up. Be ready to explain and defend every design decision logged there (why RTN from scratch, why no TransformerLens/sae_lens, why the bf16 mirror, why FLORES via the direct NLLB URL) since the viva will probe this.

## Remaining timeline

- **Sept 29 (today):** GPU driver fix in progress. Once working, rerun the quantization demo at full scale. Lit review / problem statement / timeline drafting can start now, in parallel, no code dependency.
- **Sept 30:** Consolidate whatever real results exist into report sections. Keep `reports/progress_log.md` current.
- **Oct 1:** Full draft assembled, page-limit pass, ACL formatting.
- **Oct 2:** Final proofread, package `AlwaysLate-Mid.zip`, everyone submits individually.

## After this submission (towards Oct 31 Final)

Reuse `anlp_revised.txt` Section 8's W2-W10 schedule, shifted to actual dates once the mid-submission scope is locked in.
