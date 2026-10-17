# Team Plan

## Current plan (updated 2026-10-16, night): build on Duan's code; our old pipeline is discarded

Context: our earlier pipeline (an unverified re-implementation of Duan (2026)) was **discarded** on 2026-10-16 after an audit showed that its disagreement with their numbers is a regime difference (their windows have no BOS token), not a bug on either side; its post-mortem is the last entry of `reports/progress_log.md`, its code lives only at the git tag `legacy-pipeline-2026-10-16`. Their released data show a degraded regime (no BOS, baseline perplexity 458.5). Final submission: **Oct 31, 2026**. **For a new agent: start with `HANDOFF.md`.** Evidence, verified Ada facts and the closure protocol: `reports/duan_codebase_audit.md` (section 4 venue, 6 runs, 7 closure protocol) and `reports/audit_our_vs_duan.md`. Their code is vendored in `third_party/duan2026/`.

**Principle:** no new claim goes into the paper until it is verified against their baseline, and everything we run is **their code** (baseline = replication, never modified) or a **labelled diff on their code** (diagnostics and the future study), with tests.

### Layer 1: BASELINE (their code, vendored; DONE, Q1 passed)
- Code: `third_party/duan2026/` = their repo downloaded fresh, pruned to the Gemma Phase-3 study. The one script we run is their file plus a short whitelist (`UPSTREAM.md`): delete their unrunnable tail cells (22+), their own `RUN_PRUNING` flag off, and four device references so the second bf16 model sits on a second 11 GB card. No generator, no rewrite; `display()` works unedited under IPython.
- Proof: `scripts/g0/tests/test_vendor.py` rebuilds the working copy from their pristine file plus the whitelist and requires byte equality over the whole file (top-level statements included); retained files must match the recorded upstream hashes (upstream content tree hash `a84e43da...adc`); a committed mutation suite (13 mutations) must be rejected. The launcher refuses to run unless the deployed file matches `working.sha256`.
- Launcher: `scripts/g0/run_baseline.sbatch <arm> <full|smoke>`; runs their script under IPython, then compares with their published table (`scripts/g0/baseline_report.py`, pre-committed bands) and uploads to wandb post hoc.
- Configuration: their shipped `3b_full` (500k tokens, INT8/7/6/5/4), pruning off. ~4.1 h compute.
- **Q1 passes** iff INT7 survival is within 5 pp of 81.26%, INT6 within 10 pp of 51.30%, active features within 10% of 7,144, perplexity signs hold (INT7 improves, INT6 worsens). If it fails: do not touch their code; rerun once, record versions, report.

### Layer 2: DIAGNOSTICS (modified code, always labelled as such; only after Layer 1 and the user's OK)
v3 runner (`make_duan_runners_v3.py`: BOS arms, INT0 control, ladder 0/16/12/8/7/6, snapshots, resume, wandb heartbeats), R1-R3 BOS arms, R4 their script 06, R5 our pipeline on their exact tokens, gates and dispositions in audit section 7. These explain *why* the pipelines differ; they are not part of the replication. **Audit in progress (user request 2026-10-16: "audit why our implementation was so off compared to theirs")**: `reports/audit_our_vs_duan.md` lists 12 differences between the pipelines and fixes the reading rules before the results exist; R2 (their code + BOS, job 23643) and R5 (our pipeline on their tokens, one factor flipped per cell; smoke 23648 passed; R5a 23649 and R5b 23660 running) are running; HANDOFF section 11 says what to do on the next prompt.

### Layer 3: the study on top of their code (LIVING plan, grounded in the proposal `anlp_revised.txt`; written 2026-10-17)

Every row names the proposal section, the source whose code/setup we reuse, and the decision or anomaly entry in `reports/decisions_and_anomalies.md` (DEC/ANOM/VER ids below are the table IDs D-, A-, V-). Nothing here is launched without the user's approval; the scheduled hourly check only fetches results, verifies sources and updates this plan.

| Step | Proposal ref | Source / code reused verbatim | Status and next action |
|---|---|---|---|
| **P0-a** G0 (a)-(c): active features, INT7 and INT6 survival | 4.2 | Duan's script 03 (vendored) | **done**: 7,134 active (7,144), INT7 81.19% (81.26), INT6 51.15% (51.30) |
| **P0-b** G0 (d) chunked perplexity signs (INT7 improves, INT6 degrades) | 4.2 | same | **done**: -5.61% / +4.02% |
| **P0-c** G0 (d) sliding-window perplexity signs (W=2048, stride 512; INT6 improves about -3.45%) | 4.2 | Duan's script 06 (does not parse as shipped; diagnostic generator fixes it) | **optional since D-009** (D-005); needs a go-ahead (about 2 h on 2 GPUs) |
| **P0-d** closure: Gate 1 in their regime, Q4 per regime | audit 7 | their TL code via the generator; legacy harness for `O_mask` | **done 2026-10-17**: R1 and R5c finished; Gate 0 pass, Gate 1 `noise_dominated` in their no-BOS regime (by 0.001; passes with BOS masked), Q4 mask passes; **disposition D-NOISE (D-009)**, terminal; R3 not run and no longer needed (audit 6.7) |
| **P0-e** test their claim "even under sliding-window evaluation only 51.3% survive" | 1, 4.2 | their script 03 logic with BOS and sliding windows (diagnostic) | **proposal** (A-007): their sliding rows reuse chunked survival; R2 already shows 97.5% at INT6 in the BOS regime with 256-token windows |
| **P0-f** primary-source reading of Duan (2026) | 1, 4.2 | arXiv:2606.03002v2, 29 pages | **done 2026-10-17** (V-3): never mentions BOS; context-reset explanation contradicted by our data (A-014); q/k/v scale pooling contradicts its Eq. 2 (A-015) |
| **P1-0** decide the regime for Phase 1 | 6 | Gemma Scope paper sections 3.1 and 4 (BOS excluded from training and metrics) | **needs the team (D-001)**: recommended `mask` |
| **P1-1** language roster and tier table (27 languages, 9 per tier, >= 4 scripts per tier) | 5 | Joshi et al. (2020) classes; NLLB data | **V-5 done 2026-10-17**: classes of all 27 read from the authors' file; class 4 straddles High and Mid (A-020), documentation rule proposed as D-011; whether the OSF table was frozen is still open (V-10) |
| **P1-2** data: FLORES-200 dev (tagging) / devtest (survival), equal token budgets | 4.1, 5 | official NLLB archive (FLORES-200, not FLORES+) | **V-6 done 2026-10-17**: 997 dev / 1,012 devtest sentences for every one of the 204 languages, line-aligned, all 27 panel languages present (Mandarin is `zho_Hans`); the equal-budget vs content-matching choice is D-010 (needs the team); note the old pipeline's decode/re-encode token drift (issues D2): feed token ids directly, no decode |
| **P1-3** selectivity: S(f) (pre-registered) with its 3 safeguards, plus Deng's nu as robustness | 4.1 | Deng et al. Eq. 3 and repository `Aatrox103/multilingual-llm-features` | **V-2 done 2026-10-17** (their code: signed difference of mean activations, first token dropped, BOS in the input); still to write: our S(f) plus nu as robustness, tests first (D-003) |
| **P1-4** survival pipeline as a diff on their script 03 (per-language accumulators, BOS switch, FLORES tokens), anchored by reproducing the baseline with the new switches off | 4.3 | their `streaming_feature_metrics`, quantizer and thresholds, verbatim | **TODO (D-004)**; tests first; labelled "diagnostic/modified" |
| **P1-5** controls: firing-rate-quintile-matched universal features (proposal) and activation-magnitude matching (issues C6: Spearman 0.78 vs -0.085 for rate in their data) | 4.3 | proposal; Duan's released per-feature table | **TODO**; document the added control as an amendment |
| **P1-6** statistics: language-level permutation test with script covariate, TOST, mixed-effects logistic regression, Holm over 3 tests | 7 | standard libraries (scipy, statsmodels); no feature-level Fisher as the primary test (issues C4) | **TODO** |
| **P1-7** widths: INT7/INT6 confirmatory plus INT5/INT4 where there is headroom | 4.3, 6 | Duan's protocol | **needs the team (D-006)** |
| **Ext.** Llama-3.2-1B (A), pruning (B), causal ablation (C) | 4.3, 4.4 | seonglae SAE; Borobia et al.; Wanda | cut first if time is short (proposal section 6); not started. V-4 done: Borobia's main instrument (trained TopK SAEs, decoder-cosine matching, FineWeb-Edu) differs from the frozen-SAE Pearson design of our extension, so results are not directly comparable; their Gemma 2 dense perplexity of 410 at 256 tokens carries the same caveat as Duan's (A-021) |

Compute (measured, 2 x RTX 2080 Ti, TransformerLens): about 40-45 min per condition at 500k tokens, so a 27-language panel of about 540k tokens costs about 4.5 h per regime for FP plus INT8..INT4 (the proposal budgeted 30-50 GPU-hours for the core). Realistic calendar to the Oct 31 deadline, subject to the team's decisions: **Oct 17-18** finish P0-d, verification items V-3/V-5/V-6/V-9, team decision D-001/D-006; **Oct 19-22** build and test P1-3/P1-4 (anchor against the baseline); **Oct 23-26** runs; **Oct 27-29** statistics and anomaly review; **Oct 30-31** team write-up and final checks. If P1 does not fit, the audit itself (BOS sensitivity of survival by 18-46 pp at INT7/INT6, two stacks) is the result to write up.

### Phase 1 build sequence (started 2026-10-17; tick a step only when its tests pass and it is pushed)

Design (decisions D-001, D-003, D-004, D-006, D-010, D-011 provisional, `reports/decisions_and_anomalies.md`): one script `scripts/g0/phase1_run.py` on ONE 11 GB GPU. It **reuses their functions extracted verbatim from the vendored pristine script 03** (`load_gemma_tl_model`, `load_sae_compat`, `clone_state_to_cpu`, `quantize_rtn_per_output_channel_tl`, `encode_sae`, thresholds) and adds only: FLORES sentence data with right-padded batches (causal model: padding cannot change earlier positions), BOS regimes `mask` (primary) and `none` (sensitivity), the FP features cached sparsely on CPU so one model is quantized in place and restored, per-language float64 streaming Pearson (their math), a tagging pass on the dev split (S(f) with its safeguards, Deng's nu), per-language loss, wandb under the fixed schema and HF upload at job end. Pure logic lives in `scripts/g0/phase1_lib.py` (CPU tests); the roster/tiers in `reports/g0/phase1_languages.json` (Joshi classes from V-5).

- [x] **S1** (done 2026-10-17, 11 torch tests, 10 mutants caught) `phase1_lib`: batching with BOS regimes and valid masks; per-language streaming Pearson (checked against brute-force numpy); sentence-level tagging statistics
- [x] **S2** (done 2026-10-17, 11 more torch tests, 9 mutants caught) `phase1_lib`: S(f) (max / mean firing-rate ratio), support rule (rate floor, >= 50 positive activations at the peak language), peak language, bootstrap stability (90% of sentences, 100 draws, >= 80 stable), shuffled-label null cutoff (95th percentile), Deng's nu, firing-rate-quintile-matched universal controls, activation-magnitude matching
- [x] **S3** (done 2026-10-17; `phase1_data.py`, `reports/g0/phase1_languages.json`, 5 stdlib tests, 8 mutants caught; anomaly A-022) FLORES loader and roster file (27 languages, tier, Joshi class, script); fake-tokenizer tests; token-count and truncation report
- [x] **S4** (done 2026-10-17; 53 torch tests + 111 stdlib tests, all mutants caught) `phase1_run.py`: their functions extracted by AST from the pristine file (`their_code.py`, frozen sha256 in `reports/g0/their_functions.json`); `run_tagging` / `run_survival` / `selectivity_report` / `save_outputs` / `main` with injectable model, SAE and quantizer, tested on CPU with a fake causal model against brute force (mask excludes position 0, padding excluded, batching-independent, INT0 = FP, rung independent of earlier rungs, restore); `phase1_anchor.py` (anchor comparison, tolerances fixed before the run: ppl 0.5% rel., survival/degraded/damaged 0.5 pp, median/mean r 0.005, active features 10, tokens exact); `run_phase1.sbatch`, `deploy_phase1.sh`; decision D-013. `main()` itself is only exercised on Ada (needs the model).
- [x] **S5** (done 2026-10-18 ~00:05 IST) anchor on Ada **passed, bit-identical**: `--corpus wikitext --regime none --budget 20000 --batch-size 1 --loss tl` (Slurm 25727, commit 235b58c) reproduces the vendored 3a_test smoke exactly: perplexity 414.07274322636493 / 422.6203925446741 / 427.5592814057298, n_active 6702, mean r, median r, survived / degraded / damaged identical in every digit; at batch size 4 (Slurm 25728) only median r differs, in the 15th digit, so batching does not matter. FLORES smoke (Slurm 25711, 3 languages, regime mask): end to end on real tensors; INT0 control = full precision exactly; INT6 survival eng 91.8% / hin 84.7% / khm 63.4% (3 languages x 120 sentences, **not a result**). Findings on the way (A-024, A-025, A-023): their perplexity is quantised by bf16 arithmetic in TransformerLens' loss (the anchor needed their function; production uses exact float64 sums), batches of 2048 tokens run out of memory, the stability requirement must be a fraction of the draws, parallel model downloads stall (persistent offline cache).
- [ ] **S6** full runs on Ada, one GPU per regime (`mask`, `none`): tagging pass (dev, FP) + devtest FP cache + INT8/7/6/5/4 + INT16/12/0 controls; outputs to HF, summaries to git, wandb. **First launch (Slurm 25731 / 25732) cancelled after 25 min: it used all tokens, the proposal pre-registers equal token budgets (A-026, D-015).** Prerequisite now being built test-first: equal-budget position sampling with two streams (`eq` primary, `all` sensitivity) in one pass. Pre-launch checklist (proposal vs runner): equal budgets in tagging AND survival [D-015]; per-language survival at the peak language [D-013]; dev for tagging and devtest for survival [done]; INT7/INT6 confirmatory, INT8/5/4 descriptive, INT16/12/0 noise-floor controls [done]; support floor, 80/100 stability, 95th-percentile shuffled null [done]; firing-rate-quintile-matched controls [done]; activity rule FP firing rate > 0.001 [done].
- [x] **S7** (code done 2026-10-17; 33 tests on synthetic data with planted effects, all mutants caught; decision D-014; **still to run on the real S6 outputs**) `phase1_analysis.py`: selective vs firing-rate-quintile-matched universal survival at INT7 / INT6 (within-stratum permutation, cluster bootstrap), language-level tier test (Freedman-Lane permutation with script-family covariate, populated-language rule, High-minus-Low and Joshi-class sensitivities), TOST, Holm over the 3 confirmatory tests, exploratory INT5 / INT4, random-intercept logistic model (quadrature ML; statsmodels' GLMM was intractable), P1, noise-floor instrument check, markdown report
- [ ] **S8** results into `reports/phase1_results.md` (tables from the analysis script, every anomaly in `decisions_and_anomalies.md`); the team writes the paper text

### Venue (verified on Ada 2026-10-15/16)
- **Ada `u22`, 2 x RTX 2080 Ti per job** (`--constraint=2080ti --gres=gpu:2`); `model_ref`+SAE on `cuda:0`, `model_work` on `cuda:1`.
- **No GPU larger than 11 GB is reachable** from our accounts (3090/3080 Ti partitions closed to `nlp`/`research`).
- Gotchas that already cost us jobs: `/share1` is **not mounted on compute nodes** (and has a 3,000-file quota); `/home2` has ~19 GB free and is shared with other jobs (keep the model off it: each job fetches the 10.5 GB model to node-local `/scratch`, ~18 min, deleted afterwards); never set `HF_HUB_DISABLE_XET=1`.
- Not usable: this laptop (6 GB GPU, ~100 KB/s HF downloads), gnode001-042 (GTX 1080 Ti).
- Limits: QOS `medium` caps us at 4 GPUs per user, shared with the semeval job; `-A nlp` (QOS `normal`, no per-user cap) works for parallel arms (R5b-nlp 23660, 2026-10-16); the 40-CPU cap of `medium` counts the inflated CPU count (`-c 8` + `--mem=40G` became 14).
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
| 8a | **Audit of the disagreement**: `reports/audit_our_vs_duan.md`; R5 tooling (legacy HF harness, now only at the git tag) and `scripts/g0/r5_compare.py` (11 stdlib tests) | **done**: BOS is the dominant factor (INT7 81 -> 99.9%, INT6 51 -> 97%, perplexity 457 -> 16); their q/k/v grouping matters only without BOS; sdpa soft-capping and bf16 SAE do not matter |
| 8b | **R5** (smoke 23648, R5a 23649, R5b 23660) **done and judged** (audit section 6.2: anchor passes; BOS and, without BOS, q/k/v grouping are material; sdpa and bf16 SAE are not); **R2** (job 23643) done: agrees with `T_mask` to 0.1 pp, BOS effect +18.8 pp at INT7 (audit sections 6.3-6.4: closure protocol not yet assignable; what is missing is listed there) |
| 8c | **Discard the old codebase; post-mortem** (`reports/progress_log.md`, last entry); cleanup of the tree; README/HANDOFF rewritten around their code | **done** (2026-10-16 night; tag `legacy-pipeline-2026-10-16`) |
| 8e | **wandb fixed schema**: `reports/wandb_schema.md`; 13 finished runs rebuilt, 13 old runs deleted after verification; R1/R5c to be rebuilt when they finish | **done** (2026-10-17; D-008, A-011) |
| 8d | Completions queued: **R1** (their code, no BOS, ladder INT0/16/12: Gate 1 in their regime), Slurm 24451; **R5c** (`O_mask`, legacy harness copy on Ada: Q4 mask regime), Slurm 24452 | **running** (HANDOFF section 10 says what to do on the next prompt) |
| 9 | **Decide the study's regime (BOS or no BOS)**; then, if the team chooses, rebuild the language-selective analysis as a labelled diff on their script: discriminating widths (INT6-INT4), content-matched sentence sets, language-level statistics, activation-magnitude-matched controls, anchored against the baseline before any new claim | **waits for the team's decision** (`HANDOFF.md` section 9) |
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
