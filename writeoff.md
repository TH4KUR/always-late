# Handoff brief: ANLP mid-submission, continuing on a new machine

You are a fresh Claude Code session with no memory of prior conversation. This file is a complete handoff from a previous session that did real work on this project. Read this entire file before doing anything. The user has moved to a borrowed machine with an **RTX PRO 6000 96GB VRAM and 250GB disk** — compute/disk are not the constraint they were before.

## 0. DEADLINE IS TODAY, 2026-10-02, 11:59 PM. Move fast, don't over-engineer.

Confirm this with the user if anything seems off, but proceed on this assumption immediately rather than re-asking first.

## 0.5. Major update since this file was first written: a teammate (Shardul) already did most of the guaranteed core

While the previous session was mid-task, it discovered (via `git fetch`) that teammate Shardul Joshi had independently run the **full 27-language Phase-1 pipeline** on IIIT's real Ada HPC cluster and written a **complete 7-8 page LaTeX paper draft**, all already pushed to `main` and merged in. So the big remaining gaps described lower in this file (literature review, timeline, progress write-up, full panel) are **already done** — don't redo them from scratch. What's actually true right now:

- `paper/main.tex` is a full draft (abstract, intro, 5-pillar lit review, methodology, 27-language results, discussion, evaluation-plan matrix, timeline, appendix). Compiles with `pdflatex`+`bibtex` (toolchain confirmed present).
- **Current blocker, in progress when this was last touched: page count is over the limit.** Guidelines cap the body at 7-8 pages (excl. references) + 1 optional appendix page. As compiled: **body = 9 pages, appendix = 2 pages** (should be ≤8 and ≤1 respectively). Two real LaTeX errors (unescaped underscore/backticks in plain text, e.g. `` `data/hf_cache` `` instead of `\texttt{data/hf\_cache}`) were found and fixed. The appendix's big 27-language metadata table was being shrunk (`\scriptsize`, tighter `\tabcolsep`/`\arraystretch`) when interrupted — recompile and check if that alone gets it to 1 page; if not, cut the two short appendix subsections (RTN module counts, null-permutation convergence) down to one sentence each or fold into footnotes.
- Worth a quick, non-blocking sanity flag: the course's explicit policy is *"You are not allowed to use AI to generate reports."* Some of Shardul's commit messages and the fully-formed prose read as possibly AI-generated. Not yours to litigate — just mention it to the user once, don't block on it, they can confirm with Shardul in parallel.
- Repo is still **private** — must be made public or TA mentors granted access before submission.
- No `AlwaysLate-Mid.zip` exists yet — needs to be created (PDF + code) before submission.

## 0.6. What to actually "run" on this new machine (the compute-heavy task)

Given the core pipeline and paper already exist and work, the highest-value use of the new machine's power in the time remaining is **not** redoing everything — it's strengthening the existing, already-written results without changing the paper's structure:

**Primary: re-run the existing Phase-1 pipeline at a much larger token budget for tighter statistics.** Current results used only 1,536 tokens/language (see `reports/selectivity_summary_table.csv` / the appendix metadata table), which is almost certainly why several hypothesis tests came back underpowered (e.g. H2: p=0.7828, n=117-135 per arm). With 96GB VRAM + 250GB disk there's no reason to stay this small. Re-run:
```bash
python scripts/run_phase1_selectivity_pipeline.py --tokens <much larger, e.g. 8000-20000> --output reports/selectivity_results_27lang.json
python scripts/run_phase1_survival_quintile_matching.py --report reports/selectivity_results_27lang.json --bitwidths 7,6 --tokens <same larger value> --output reports/phase1_survival_results_27lang.json
python scripts/statistical_analysis.py --report reports/phase1_survival_results_27lang.json
python scripts/generate_master_results.py
python scripts/generate_mid_submission_figures.py
```
If the numbers change meaningfully, update the corresponding tables/figures in `paper/main.tex` to match — don't let the paper and the data drift apart.

**Secondary, only if time remains after the above + paper is submission-ready:** the Llama-3.2-1B cross-model extension (`src/sae_models.py:TopKSAE`, written but never tested against real `seonglae` weights) — genuinely stretch, don't let it block submission.

**Do not** start the pruning contrast or Phase-2 causal ablation from scratch today — out of scope for tonight regardless of compute available.

## 1. What this project is

Course: Advanced NLP (ANLP), Monsoon 2026, IIIT Hyderabad. Team "AlwaysLate" (4 members: Hemang Joshi, Eashaan Thakur, Shardul Joshi, Nikhita Ravi).

Research question: do **language-specific SAE (sparse autoencoder) features** in multilingual LLMs survive **quantization/pruning** at different rates than universal features, and does survival track a language's **resource level**? Primary model: Gemma-2-2B + the pinned Gemma Scope residual-stream SAE (layer 12, width 16k, "canonical" = average_l0_82). Stretch extension: Llama-3.2-1B.

Full research design: `anlp_revised.txt` (the locked, final, already-submitted proposal — do not change its content, it's a source of truth for methodology, not something to edit).

Official grading requirements: `project_guidelines.txt`. Mid Submission needs a **7-8 page document** (excl. references, +1 optional appendix page) covering: a refined problem statement, progress made, a concrete timeline for the rest of the project, and a comprehensive literature review. Graded on write-up + code (repo, public or TA-mentor accessible) + a viva. Submission filename: `AlwaysLate-Mid.zip`, every team member submits individually.

**Critical, non-negotiable policy from the guidelines: AI may be used for implementation/code, but NOT to generate the report text itself.** Everything in this handoff is about code and experiments. The actual write-up prose must be written by the team — your job is to produce code, run experiments, log real results honestly, and at most draft structural outlines (section headers, bullet content maps), never finished prose for the submitted document.

## 2. Where the code lives

Repo: `https://github.com/TH4KUR/always-late` — **currently PRIVATE**. Before submission this needs to be either made public or have TA mentors granted explicit access (guideline requirement). Clone it on this machine and continue from there; don't start a new repo.

Key files:
- `src/pipeline.py` — model loading (plain HF `transformers`, deliberately **not** TransformerLens) + forward-hook residual-stream capture + SAE encode.
- `src/sae_models.py` — `JumpReLUSAE` (Gemma Scope) and `TopKSAE` (EleutherAI/seonglae, stretch) loaders, implemented from scratch against raw weight files, deliberately **not** depending on `sae_lens`/`transformer_lens` for our own pipeline (see gotchas below for why).
- `src/rtn_quant.py` — custom per-channel round-to-nearest quantization (not bitsandbytes/GPTQ — neither matches the pinned linear-INT protocol).
- `src/survival.py` — firing-rate, Pearson-survival, selectivity-score (`selectivity_score`, Deng et al. 2025-style S(f)) statistics.
- `scripts/run_w0_sanity_check.py`, `scripts/run_multilang_sanity_check.py`, `scripts/run_quantization_survival_demo.py` — the experiments run so far (see results below).
- `reports/progress_log.md` — **the full, honest, timestamped technical log of everything actually run, including bugs found and fixed.** This is the evidentiary backbone for the "progress made" section and for viva defense. Read it in full; don't claim anything in the write-up beyond what's logged here.
- `PLAN.md` / `PROGRESS.md` — team-facing task split and status snapshot (not the report itself).
- `requirements.txt` — exact pinned versions that were verified to work together; reuse rather than re-resolving from scratch.

Also present but **unrelated to this project** — ignore, don't commit, don't touch: `Capture the Offer_ Crustdata @ IIIT Hyderabad.html` and its assets folder (looks like a stray browser save that landed in this directory by accident).

## 3. Real results already secured (do not re-derive, just build on them)

- **W0 sanity check: PASS.** Gemma-2-2B + Gemma Scope layer 12, FLORES-200 Kannada vs. English, 40 sentences each: 1,035 Kannada-selective and 1,918 English-selective features found among 16,384 total.
- **Extended to 4 languages (eng/hin/kan/tel, one per resource tier + English anchor): PASS.** All 6 pairwise comparisons show nonzero selective features in both directions.
- **Reduced-scale quantization-survival demo: DONE, on GPU.** RTN INT8→INT4 on Gemma-2-2B, WikiText-2 (3,000 tokens): survival falls monotonically 99.03% (INT8) → 98.28% → 94.31% → 82.37% → 60.40% (INT4), damage rising 0% → 5.29%. Matches Duan (2026)'s qualitative direction. **Not** the formal replication gate (see below) — smaller sample, not their exact corpus/budget.
- Full numeric results: `reports/w0_sanity_check_result.json`, `reports/multilang_sanity_check_result.json`, `reports/quantization_survival_demo_result.json`.

## 4. In progress when the handoff happened — pick this up first

**Formal Gate G0 replication of Duan (2026)** (`anlp_revised.txt` Section 4.2) was next in priority and had real reconnaissance done, but no run yet:

- Duan's actual anonymous repo is real and reachable: `https://anonymous.4open.science/r/sae-feature-survival-quantization-0141/`. **The main web page 401s (needs JS rendering) but the raw file API works directly**: `https://anonymous.4open.science/api/repo/sae-feature-survival-quantization-0141/files/` (listing) and `.../api/repo/sae-feature-survival-quantization-0141/file/<path>` (raw file content). Use `curl`, not a browser.
- Confirmed via their README: the repo contains auto-exported `.py` versions of their notebooks under `scripts/` (the recommended entry points), e.g. `scripts/03_gemma_phase3_streaming_sweep.py` (the Gemma pipeline, ~36KB, 1259 lines) is what implements their Gate G0 numbers.
- **Confirmed their exact protocol pins** by reading that script: `SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"`, `SAE_ID = "layer_12/width_16k/canonical"`, `FIRING_THRESHOLD = 0.001`, survival/damage thresholds at correlation 0.9/0.5. Also independently verified (via the `google/gemma-scope-2b-pt-res` README) that "canonical" = the SAE with average L0 closest to 100 for that layer/width — for layer_12/width_16k that is **`average_l0_82`**, which is exactly the SAE our own pipeline already uses. This is a real, useful cross-check: our pin was already correct.
- Their script has a built-in reduced mode: `RUN_MODE = "3a_test"` → 20k tokens, INT8+INT6 only. Full mode: `RUN_MODE = "3b_full"` → 500k tokens, INT8/7/6/5/4, plus a pruning comparison.
- Their script uses `transformer_lens.HookedTransformer.from_pretrained_no_processing` + `sae_lens.SAE.from_pretrained` — the exact dependencies our own pipeline deliberately avoids (version churn risk). **That's fine for this one task**: install them temporarily just to run Duan's own script verbatim for the replication gate; don't restructure our own pipeline around them.
- **With 96GB VRAM and 250GB disk, there's no reason to use the reduced "3a_test" mode** — go straight for `RUN_MODE = "3b_full"` (500k tokens, full INT8-INT4 sweep, pruning comparison) for the real, final Gate G0 numbers. Evaluate against the numeric bands in `anlp_revised.txt` Section 4.2 (active-feature count within ±10% of 7,144; INT7 survival 81.26%±5pp; INT6 survival 51.30%±10pp; correct perplexity-direction signs) and log the outcome (pass / fail / descriptive-downgrade per the instrument-validity criterion, Section 6) in `reports/progress_log.md`.

## 5. Priority-ordered next steps (not calendar-bound — do them in order, log each one honestly before moving to the next)

The previous session's explicit, user-confirmed decision was to **stay disciplined at ~40-50% scope** (not push into the full guaranteed core) because compute was the bottleneck. **That constraint is gone now** — re-confirm with the user whether to still cap scope for the mid-submission, or use the new compute to complete substantially more of the guaranteed core (up to and including the full 27-language Phase-1 panel) since it's now cheap. Don't assume; ask once, then proceed.

Recommended order assuming the user says "use the compute, push further":

1. **Finish the Gate G0 replication** (Section 4 above) — highest value, was already in flight, validates the whole project's instrument.
2. **Full 27-language Phase-1 core** (`anlp_revised.txt` Section 4.3): per-language SAE feature survival across selectivity and resource tier, RTN at INT7/INT6 (confirmatory) + INT8/5/4 (descriptive), using the full FLORES-200 devtest panel (27 languages, 9 per tier — see Section 5's language list) with equal token budgets. This needs the real pre-registered selectivity score `S(f)` (`src/survival.py:selectivity_score` already implements the formula) plus the bootstrap-stability check (Section 4.1: peak-language assignment stable under resampling 90% of sentences in ≥80/100 draws) — this requires modifying `encode_texts`/`pipeline.py` to preserve per-sentence boundaries (currently returns one flat concatenated tensor per call), so bootstrap resampling doesn't require re-running the model 100 times.
3. **Llama-3.2-1B cross-model extension** (stretch, Section 4.3): `seonglae/Llama-3.2-1B-sae` via the already-written `TopKSAE` loader in `src/sae_models.py` — written but never tested against real weights. Note it hooks `resid_pre`, not `resid_post`, and isn't in SAELens's official registry (third-party), so label results "replication-with-caveat" per the proposal's own framing.
4. **Pruning contrast** (magnitude/Wanda, matched perplexity, same frozen SAE basis) — extension-phase per the proposal, motivated by Borobia et al. (2026)'s finding that Wanda preserves feature structure best.
5. **Phase-2 causal ablation check** (Section 4.4) — only after 1-4, tests whether geometric survival predicts functional importance post-compression.

**Always running in parallel, no code dependency, team's job not yours:** literature review expansion, refined problem statement, the "concrete timeline for the rest of the project" section, the "progress made" write-up section. Check with the user whether these have been done elsewhere (Docs/Overleaf) — the previous session's last check found zero movement on these in the repo, which is a real risk if the deadline is in fact today or has passed.

## 6. Technical gotchas already paid for — don't re-learn these the hard way

- **Model weights**: use `unsloth/gemma-2-2b` (bf16, ~5.2GB, **not gated**) rather than `google/gemma-2-2b` (fp32, ~10.5GB, gated — needs HF license acceptance). Same underlying checkpoint, just smaller dtype and no gating lag. Keep using this mirror for continuity with existing logged results unless there's a specific reason to switch.
- **FLORES-200**: both HF mirrors (`facebook/flores`, `openlanguagedata/flores_plus`) are gated. Pull directly from the public NLLB release instead: `https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz` (~25.6MB, all 200+ languages, dev+devtest, no gating).
- **JumpReLU "active" definition**: a feature is active iff `pre_acts > threshold` (a learned per-feature threshold), **not** `ReLU(x) > 0`. Our `JumpReLUSAE.encode` in `src/sae_models.py` already implements this correctly — don't reimplement naively if extending.
- **RTN quantization implementation**: use linear per-output-channel round-to-nearest (`src/rtn_quant.py`), not bitsandbytes (NF4 is a non-linear codebook, not INT quant) or GPTQ (different calibration procedure, wrong protocol).
- **Quantize/restore memory**: when staging original weights for restoration after a quantized forward pass, clone them to **CPU**, not GPU (`src/rtn_quant.py`'s `quantized_model` context manager already does this) — cloning on GPU nearly doubles resident weight memory and caused a real OOM on the 8GB card this was developed on. With 96GB VRAM this specific bug won't bite, but the CPU-staging approach is still correct and fine to keep.
- **WikiText-2 loading**: don't use `datasets.load_dataset(...)` for a single split — it materializes train+validation+test all at once on first cache build, which OOM'd a RAM-constrained machine earlier. Not likely to matter with 250GB disk + 96GB VRAM, but if memory ever gets tight, download just the specific parquet shard via `huggingface_hub.hf_hub_download` instead (see `scripts/run_quantization_survival_demo.py` for the pattern).
- **Dependency pins**: `requirements.txt` has exact versions already verified to work together (`transformers==5.17.0`, `torch==2.13.0`, etc.) — reuse them rather than re-resolving, to avoid the version churn that's bitten this ecosystem before (`sae_lens`'s API has changed shape across releases).

## 7. Before actually submitting (whenever that happens)

- Resolve repo visibility: make `https://github.com/TH4KUR/always-late` public, or explicitly grant the TA mentors access.
- Page-limit pass: 7-8 pages excl. references + optional 1-page appendix, ACL style format.
- Package as `AlwaysLate-Mid.zip`; every team member submits individually (course rule).
- Everything claimed in the write-up must trace back to something logged in `reports/progress_log.md` — the team needs to be able to defend every result in the viva without hedging.
