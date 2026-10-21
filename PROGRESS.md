# Progress at a Glance

**Selected next study, 2026-10-21 (D-035):** the user chose the 27-language association-retention check and requested its plan. [Plan and implementation gates](docs/superpowers/plans/2026-10-21-language-association-retention.md); [measurement design](reports/survival_method_reassessment.md#7-selected-next-study-language-association-retention-d-035). Documentation is prepared; no implementation or GPU job has started. Latest handoff: [014](reports/handoffs/014-2026-10-21-language-association-plan.md). This selects bounded M0/M1 work, not all Q13-Q15 follow-ups.

**Q13-Q15 applied, 2026-10-21 (D-032/D-033/D-034):** retain both magnitude-matching comparisons as associations; investigate the narrower magnitude mechanism/selection question without assuming mediation or novelty. The completed Q14 tests distinguish overall tier disparity from an extra selective-feature penalty; preserve both and their chronology. Q15 now investigates confidence, diffuse activity and competing representation/SAE explanations as separate predictions. [All discussed sources, publication status, exact Q14 results and search log](reports/research_sources.md). No specific new GPU job is selected. Latest handoff: [013](reports/handoffs/013-2026-10-21-q13-q15-and-source-register.md).

**Method/width update, 2026-10-21 (Q3/Q4, D-030/D-031):** reassess survival detection; include INT5/INT4 in the full ladder. Existing fixed-SAE correlation measurements do not establish retained language identity or function. [Sources and candidate-check plan](reports/survival_method_reassessment.md). Claude coordination uses [the shared inbox](communication/README.md); [Claude has acknowledged the plan](communication/replies/2026-10-21-claude-to-codex.md).

**Research objective, 2026-10-21 (Q2, D-029):** publishability is the priority. Questions, proposal and direction can change when current results and analysis justify it; before selecting a new direction, check the specific contribution against the closest primary work and document its novelty and evaluation. Keep exploratory/confirmatory distinctions and all result provenance. Historical OSF status is unverified, not a blocker to revisions; no new direction or run is selected by this policy alone.

**Current scope, 2026-10-21 (Q1, D-028):** `mask` is the sole ongoing regime. No further `none` work or numerical comparison with Duan; saved results remain historical evidence. [Audit disposition and result links](reports/audit_our_vs_duan.md#current-disposition----closed-2026-10-21-q1-d-028). Q5 and Q9 are closed without execution; Q15's expanded investigation is mask-only and its exact job remains to be designed. This overrides older status and run proposals below.

Last updated: **2026-10-18 (morning)** (Phase 1 results, below); the rest of this file is from **2026-10-16 (night)**, after the audit of why our old pipeline disagreed with Duan's and the decision to **discard that pipeline and build on Duan's codebase**. **New agent? Start with `HANDOFF.md`.** The post-mortem of the old codebase is the last entry of `reports/progress_log.md`; its code is only at the git tag `legacy-pipeline-2026-10-16`; its Phase-1 results are in `reports/legacy_pipeline/` (hypotheses, not findings). Evidence: `reports/progress_log.md`, `reports/audit_our_vs_duan.md`, `issues.md`, `PLAN.md`.

## NEXT RUN'S GOAL (rewritten at the end of every scheduled invocation; the routine is `reports/scheduled_routine.md`, run in its own chat every 2 hours)

**Updated 2026-10-21 (D-035):** the bounded 27-language association study is now selected and planned. Read [handoff 014](reports/handoffs/014-2026-10-21-language-association-plan.md), the linked design and implementation plan; implementation begins with an artifact inventory and frozen input/cohort manifest. Specific jobs await missing-artifact confirmation and smoke gates. No new measurement has run. Broader Q13-Q15 questions, remaining review items and novelty claims are separate. Q1/Q2 remain in force. The state below is historical where it conflicts with this update.

State at 2026-10-19 ~17:00 IST: Phase 1 (S1-S8) and N1 to N13 are done and **reproduced end to end** (`reports/phase1_reproduction.md`: 89 of 89 committed result files identical when regenerated from the cached run files at 4 BLAS threads). The one-page evidence summary is `reports/phase1_summary.md` (generated; status labels). **No analysis is unblocked: everything left needs the user.** HF verified (274 files; nothing large added), wandb 19 runs conforming, no Ada job running. **Mode (user): everything runs in this one chat, step by step, a handoff after every step (`reports/handoffs/`, newest = only source of the previous chat); no cron, no second chat.** Newest handoff: `reports/handoffs/009-2026-10-19-n13-reproduction-done.md`.

What a next step does, in this order:
1. Read the newest handoff; read `DECISIONS_FOR_REVIEW.md` for answers.
2. **If any question is answered:** apply it as the handoff's section 4 says (D-entry, test-first change, mark `APPLIED`, regenerate `reports/phase1_summary.md`, handoff).
3. **If none is answered: say so and stop.** Do not invent analyses (the CPU analyses available without new data are exhausted and reproduced); low-cost read-only checks (git, Ada `squeue`, HF `verify`, `CronList`) are fine.
4. Rewrite this block; update the progress log; commit and push; write the next handoff.

## Where we really are

- **Phase 1 (language-selective features; `PLAN.md` S1-S8 and the analyses N1 to N13) has run on top of their code and is reproduced end to end from the cached run files by the committed scripts ([`reports/phase1_reproduction.md`](reports/phase1_reproduction.md), D-026). The state of the evidence is in one place: [`reports/phase1_summary.md`](reports/phase1_summary.md) (every number generated from the committed result files, a status label per analysis); the full account is `reports/phase1_results.md`. In words:**
  - the pre-registered selective-versus-control tests are null after correction in the primary regime (BOS masked, equal token budgets);
  - a selectivity penalty appears only at strict S thresholds chosen after the fact, and activation magnitude accounts for it;
  - the large effect in the data is the resource-tier effect on survival, for selective and universal features alike; it is carried statistically by per-language activation scale (the proposal's "localizes upstream" branch), and character fertility explains only part of it;
  - the JumpReLU threshold margin adds nothing beyond raw magnitude; two residuals stay unexplained;
  - everything beyond the pre-registered family is additional or exploratory, correlational (27 languages) and labelled so; the completed BOS-free outputs are historical evidence from a regime that failed its noise-floor check.
  - Q3's reassessment, Q4's inclusion and Q13-Q15's reporting/investigation decisions are recorded; replacement measurements remain to be designed. Other unanswered choices include Q6/Q7/Q8/Q11/Q12; Q10 has a separate earlier answer awaiting its own follow-through. Do not silently resolve these in the measurement review. Historical OSF status remains unverified.

- **The pipeline runs and its core machinery looks correct** (JumpReLU encode, layer-12 hook, RTN quantizer, Pearson/classification, dev/devtest split). The paper tables trace back to the JSON in `reports/`.
- **The headline interpretation is not supported by our own numbers** (see "Corrections" below).
- **The instrument was never validated against Duan (2026).** Gate G0 was not run. Our pipeline is an independent re-implementation, not their code, and it differs from theirs in several ways (BOS token, HF vs TransformerLens, window length, bf16 SAE). Their released data show a degraded regime (no BOS, baseline perplexity 458), so "match their 81% / 51%" is not a valid target either. Decision: **run their vendored code as the baseline** (Layer 1; a short explicit edit whitelist enforced by tests; done, reproduces their table), then run clearly labelled diagnostics (Layer 2) to explain the gap (**in progress**: `reports/audit_our_vs_duan.md`, jobs R2 and R5, HANDOFF section 11; smoke selfcheck results in audit section 6.1). See `PLAN.md`. Ada was probed and a stack built: only 11 GB-class GPUs are reachable, so the baseline uses **2 x RTX 2080 Ti per job with a placement-only edit**.
- **Paper and README are stale** in places (old 1,500-token numbers, a "null → upstream" conclusion, procedures the code does not run). They must not be submitted as they are. Prose must be written by the team (course policy).

## Done (as verified)

- [x] SAE loader for JumpReLU (Gemma Scope), HF residual hooking, RTN quantizer, 27-language data utilities, survival and selectivity functions. All are **our own implementations**.
- [x] W0 precondition check (Kannada vs English) and the 4-language check: both show non-zero selective populations. Coarse tests only, not resource-tier results.
- [x] Quantization demo (WikiText-2 test, **2,048 tokens**, INT8–INT4): survival falls monotonically, 99.03 → 98.28 → 94.31 → 82.37 → 60.40%. **Direction only.** It is not a G0 replication, and its levels are 17 pp (INT7) and 43 pp (INT6) above Duan's.
- [x] Phase-1 selectivity extraction at **20,000 tokens/language**: 2,342 features pass S ≥ 3 (High 839 / Mid 444 / Low 1,059); 11,670 supported. (The earlier 1,500-token run gave 1,996 and is superseded.) S ≥ 3 is a loose criterion, and content coverage differs by language (issues C1–C3).
- [x] Survival (GPU, all selective features, INT8–INT4) on devtest, with firing-rate-quintile-matched universal controls.
- [x] Confirmatory-style tests (Fisher + Holm) and the behavioral loss link. These deviate from the pre-registered plan (issues C4, C5, D6).
- [x] Paper draft exists (`paper/main.tex`). It needs a rewrite of the results and discussion.
- [x] Code review, `issues.md`, and an audit of Duan's codebase (downloaded, hash recorded, released tables verified offline; their INT6 per-feature data reproduces their headline row exactly).
- [x] Ada probed (partitions, GPU types, mounts, quotas, QOS), their stack built, benchmarked, independently re-verified by a second reviewer.
- [x] wandb logging for the G0 runs (`scripts/g0/g0_wandb.py`, project `anlp-project`), built test-first and live-tested against the real service.
- [x] **BASELINE DONE: their own code reproduces their published Gemma Phase-3 table** (job 23472, 4 h 08 min): INT8..INT4 survival within **0.15 pp**, median r within 0.0008, baseline ppl 458.4 vs 458.5, INT7 ppl -5.61% vs -5.65%, per-feature Spearman at INT6 0.9996 over 7,121 features. Q1 passed on all four pre-committed bands. `reports/g0/baseline_full/`; wandb `karma-skz-/anlp-project/runs/86f26788`.
- [x] **Their code, vendored** (`third_party/duan2026/`, fresh download, byte-identical twice; content tree hash `a84e43da...adc`): pruned to the Gemma Phase-3 study; the working copy of the one script we run = their file + 5 whitelisted edits (delete unrunnable tail cells, `RUN_PRUNING=False`, 4 device references). 43 stdlib tests rebuild it from the pristine file plus the whitelist (byte equality over the whole file) and reject 13 committed mutations. `baseline_report.py` compares with their published table using pre-committed bands. The earlier generator-based runner was removed after a self-review found its test compared only 13 function bodies, not top-level statements.
- [x] Earlier smoke with the (now removed) generator-based runner (job 23419) was bit-identical to R0 (max abs difference 0.0 over every summary column and the per-feature correlations of all 16,384 features for 3 conditions). The v3 instrumentation does not change their computation and results are deterministic across nodes. wandb: `karma-skz-/anlp-project/runs/8f6f4d3a`.
- [x] **Closure protocol closed: D-NOISE** (2026-10-17): R1 (their code, no BOS) fails the pre-committed Gate 1 noise-floor check by 0.001 (median r 0.98897 / 0.98895 at INT16 / INT12 against 0.99) while the BOS-masked regime passes; mean L0 is 82.4 with BOS masked (the SAE's nominal 82) and 95.1 without; Q4 passes for the mask regime. Their survival numbers are cited as prior work, ours rest on the section 7.4 checklist (`reports/audit_our_vs_duan.md` 6.7). All Ada jobs of the audit are finished; the schema-aware logger is deployed to Ada.
- [x] **wandb fixed schema** (2026-10-17): one vocabulary for every run (`reports/wandb_schema.md`, `scripts/g0/g0_wandb.py`, `wandb_rebuild.py`); the 13 finished runs rebuilt from local result files and the 13 old runs deleted after verification (summary keys across runs: 158 -> 31); R1 and R5c follow when they finish.
- [x] **Audit of our-vs-their pipeline** (`reports/audit_our_vs_duan.md`): 12 listed differences (BOS, attention soft-capping, SAE dtype, quantizer grouping, corpus/size, population, ...), experiments and pre-committed reading rules; R5 tooling built test-first (`scripts/g0/r5_run.py`, `r5_lib.py`, `run_r5.sbatch`, `r5_compare.py`; 31 torch tests that also pass in the Ada env, 10 stdlib tests, mutation checks run with bytecode disabled). Static findings: HF's default `sdpa` attention path (transformers 5.19) never applies Gemma-2's attention soft-capping (cap 50) and our `src/` pipeline uses it, while TransformerLens and HF `eager` do apply it (smoke selfcheck, 4 windows: sdpa vs eager moves layer-12 resid_post by 1.98% relative L2; its effect on survival is what R5 `T` vs `T_sdpa` measures); an earlier statement that TransformerLens computes RMSNorm in bf16 was wrong (both upcast to fp32) and is corrected in `issues.md` B1. Results pending.
- [x] R0 smoke on Ada (job 23355, v3 runner, 3a_test): TransformerLens runs on both 2080 Ti, peak 8.2 of 11 GiB, 40 min per condition at 500k tokens. At only 20k tokens their regime reproduces: INT6 survived 49.8% (published 51.3%), INT6 median r 0.8994 (0.9027), INT8 median r 0.9713 (0.9707), baseline perplexity 414 (458).

## Corrections to previously recorded claims (superseded)

| Earlier claim | What the data say |
|---|---|
| "Selective features degrade faster than matched universals (INT7/INT6) — directional verification" | 99.87 vs 99.96% (p=0.63) and 97.01 vs 97.35% (p=0.53): at ceiling, indistinguishable. The sign **flips** at INT5 (selective *more* robust, p=4e-4) and INT4 (less, p=3e-4). |
| "RQ2 null *proves* the gap is upstream (tokenization/embeddings)" | A tier gap **exists**: INT6 survival High 99.05 / Mid 99.3 / Low 94.4%; INT5 81.5 / 88.1 / 65.0%; language-level permutation tests give p ≈ 0.0003–0.003. Failing to reject at ceiling is not evidence of absence. |
| "27/27 languages had enough features" | Only 19/27 have ≥ 50 selective features; the pre-registered demotion trigger (< 20 languages) fires. |
| "1,996 features; 96.30% vs 99.15% (n=135/117)" | From the superseded 1,500-token run; not reproducible from the repo. |
| "Quantization sweep reproduced Duan's results" | Direction only. Magnitudes differ massively and the reference pipeline looks degraded (issues B1). |
| "Behavioral link confirms selective damage tracks loss" | Exploratory ρ≈0.49 (n=27), falls to ~0.35 controlling for tier; INT7 ΔL is noise-level. |

## Biggest open problems (details in `issues.md`)

1. **B1**: G0 never run; huge gap to Duan; their pipeline looks degraded. → run their code (PLAN.md).
2. **A1–A2**: paper/README conclusions contradict our data.
3. **C1**: panel is not content-matched (English uses 774 dev sentences, Khmer 127).
4. **C4**: stats are feature-level (pseudo-replication); the pre-registered language-level test was not run.
5. **C6**: control matches firing rate only; Duan's own data show activation magnitude is the main predictor of survival (ρ 0.78 vs −0.085 for rate).
6. **D1**: BOS counted as an ordinary token.
7. **D4–D5**: reproduction commands are wrong; synthetic figures and a stale CSV remain in the repo and wandb.

## Next steps

1. **Baseline (Layer 1): done, Q1 passed** (see above). Do not rerun; do not modify their code.
2. **Audit (2026-10-16): R5 done and judged (see `reports/audit_our_vs_duan.md` section 6.2); R2 (TransformerLens, BOS masked) also done and agrees with `T_mask` to 0.1 pp; the closure protocol's formal disposition is not yet assignable (audit section 6.4 lists the missing runs: Gate 1 in the no-BOS regime, `O_mask`, R3). Nothing is running on Ada. Decisions for the user/team: BOS or no BOS as the study's regime; whether to run the minimal completions.** Afterwards optionally R4 (their script 06) and the closure protocol (`reports/duan_codebase_audit.md` section 7). The protocol is terminal for the Duan question, not against future reviewer questions.
3. Only then re-derive the confirmatory results and rewrite the paper (by the team). Deadline for the final submission: Oct 31.

Gotchas learned on Ada: `/share1` is not mounted on compute nodes and has a 3,000-file quota; `/home2` (~19 GB free) is shared with other jobs, so the 10.5 GB model goes to node-local `/scratch` per job; do **not** disable xet for Hugging Face downloads (15x slower, drops connections). TransformerLens makes a Gemma model ~6.4 GB (un-tied unembedding), so two models fit on two 11 GB cards but not one. Script 06 from Duan's repo does not parse as shipped; the diagnostic generator fixes it.
