# Team Plan — Mid Submission

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
