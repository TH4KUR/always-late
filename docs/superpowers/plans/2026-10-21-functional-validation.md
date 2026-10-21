# Functional Validation Across Precision: Implementation Plan

> **For the agent executing this (a 40-minute recurring job, one bounded step per firing, or a chat):** use `superpowers:executing-plans` and `superpowers:test-driven-development`. Checkboxes record work; a ticked box needs the evidence written in [`STATUS.md`](../../../reports/functional_validation/STATUS.md). This plan authorises no parallel agents except one-off literature checks.

**Goal:** answer F1 to F4 and B of the [design](../specs/2026-10-21-functional-validation-design.md): does removing a language's selective SAE feature group harm that language beyond activity-matched controls, is that kept in each quantized model, where does quantization damage sit, and does association retention predict functional retention.

**Architecture:** one new GPU runner that, per precision, runs the 13 lower blocks once per batch (cached) and then re-runs only the upper blocks for each edit configuration, with edits that preserve the SAE error term; item-level results are saved and analysed on the CPU by a separate module (paired cluster bootstrap), then turned into tables and figures. The D-035 collector, `phase1_run.py` and every historical output stay untouched.

**Tech stack:** the existing environment on Ada (torch 2.14, transformer-lens 3.9.0, sae-lens 6.54.5, matplotlib 3.10 on Ada; local venv `~/.venvs/anlp-torch` has torch/numpy/scipy, matplotlib to be installed), unittest, `scripts/g0/mutcheck.py`, the existing sbatch pattern and artifact tooling.

## Global constraints

- Everything in the design's section 2 (mask only; no `none`; no numerical comparison with Duan, D-028; frozen D-035 cohort from dev only for primary groups; never edit `paper/` or `third_party/`).
- Test first and watch it fail; mutation-check the behaviours listed per task with `mutcheck.py`; evidence before claims; secret scan before every commit; stage explicit paths; commit messages end with the attribution line from the session; push after a green step.
- At most **2 Ada jobs** at a time, record the Slurm id, commit, command and expected finish in STATUS.md, check `squeue` once per firing, never wait or poll. Never redeploy to Ada while a job runs (a running job reads the deployed repository). No change to the visibility of the Hugging Face dataset; delete nothing on Hugging Face or wandb. `/home2` has about 5.6 GB free: outputs must stay small (item-level `.pt`/`.json` only; the 2.76 GB association sidecar is not touched).
- Gates: **G1** (before any GPU job): Tasks 0 to 3 done (tests green, mutants killed), the prior-art file written, the spec and plan committed, Q17 in the review file. **G2** (before any evaluation run on a quantized model or any T-MC scoring): pilot results recorded (parity, sham, throughput, Belebele calibration), the freeze file committed (design section 7).
- A step that needs a user decision goes to `DECISIONS_FOR_REVIEW.md` with a default; work continues on the default.

## Task 0: prior art, data feasibility, documents

- [ ] Prior-art result of the background search written to `reports/functional_validation/prior_art.md` and register entries (P30 and later, each with verified venue/status or "unverified"); the spec's novelty paragraph (section 9) updated; any claim the plan makes that turns out not to be novel is reworded.
- [ ] Belebele: files present locally in `data/belebele/` (ignored by git) and on Ada (`phase1/data/belebele/`, `MANIFEST.json` with sha256); the number of items per language and the field names recorded; mapping of each Belebele passage to FLORES dev/devtest by exact sentence match (Ada has FLORES) recorded in `reports/functional_validation/belebele_flores_overlap.md`.
- [ ] D-040 (register), Q17 (review file), handoff 023, README of `reports/functional_validation/`, STATUS.md, `cron_prompt.md`.

## Task 1: task layer `scripts/g0/phase1_func_tasks.py` (+ `tests/test_phase1_func_tasks.py`, stdlib; tokenizer injected)

**Interfaces.** `load_belebele(root, code) -> list[dict]` (validates 900 items, the four options, `correct_answer_num` in 1..4); `passage_split(items, seed, cal_frac=1/3) -> {passage_key: "cal"|"eval"}` (seeded hash of the passage key, same split for every language because passages are parallel; refuses a passage on both sides); `flores_membership(passage, dev_lines, devtest_lines) -> "dev"|"devtest"|"mixed"|"none"` by exact sentence match after whitespace normalisation; `format_prompt(item, fmt)` for `letter` and `cloze`; `encode_mc(tokenize, item, fmt) -> {tokens, cand_ids/option_spans}` (BOS handling as the regime, no special tokens inside); `lm_ids(n_total, n_pilot, n_eval, seed) -> {"pilot": [...], "eval": [...], "reserve": [...]}` (disjoint, seeded, identical for all languages).
**Tests first.** split is deterministic, disjoint, language-independent, fractions right; membership on planted passages (dev only, devtest only, straddling, absent); prompts contain exactly the four options in order and never the answer key; candidate token ids are four distinct ids; `lm_ids` disjoint and sorted; refuses malformed items.
**Mutants.** swapped option order, answer-number off by one, split depends on language, overlap allowed, membership ignoring whitespace normalisation.

## Task 2: intervention library `scripts/g0/phase1_func_intervene.py` (+ `tests_torch/test_phase1_func_intervene.py`)

**Interfaces.** `group_edit(x, a, W_dec, idx, a_new)`: `x' = x + (a_new - a[:, idx]) @ W_dec[idx]` in float32; `ablate`, `scale(alpha)`, `resample(pool, generator)` (whole group vectors of random pool rows), `donor(a_donor)`; `apply_edit(resid [B, L, d], valid [B, L], encode, W_dec, idx, kind, **kw) -> (resid', stats)` that edits valid positions only (never BOS, never padding), returns the input dtype, and `stats` = mean `||Delta x|| / ||x||` over edited positions and mean number of active group features. A toy SAE (random decoder, JumpReLU thresholds) in the tests.
**Tests first.** identity (`a' = a`) returns `x` exactly; ablating all features leaves exactly `x - (x_hat - b_dec)` (the error term preserved); the change is confined to the decoder span of the group; edits at invalid positions are zero; BOS and padding rows bit-identical; `scale(1)` is identity and `scale(0)` equals `ablate`; `resample` draws rows of the pool only and is seeded; `donor` with `a_donor = a` is identity and with the other model's activations moves the residual by exactly `(a_donor - a) @ W_dec[idx]`; sham numerically near zero in bf16; norm statistic correct on a hand case.
**Mutants.** error term dropped (decode-and-replace instead of delta), BOS not excluded, alpha applied to the wrong side, pool index off by one, wrong decoder rows, float32 cast removed, edits applied to padding.

## Task 3: groups and controls `scripts/g0/phase1_func_groups.py` (+ `tests_torch/test_phase1_func_groups.py`; generated `reports/functional_validation/groups_manifest.json`)

**Interfaces.** `rank_groups(sel, K)`: per language the selective features whose dev peak is that language, sorted by dev mean activation on their own language (ties by feature id), top K and full; `build_controls(sel, lang, group, seed, R)`: cells = (dev firing-rate quintile on that language) x (dev mean positive activation tercile), strata computed over group plus pool, draws by the existing `phase1_lib.matched_sample`, shortages recorded; `build_manifest(selectivity_path, Ks=(16, 64, None), Rs=(5, 5, 3), seed)` with sha256 of the source file, language order, ids, shortages; `verify_manifest`.
**Tests first.** nested groups (K=16 inside K=64 inside full); sizes equal K for all languages with at least K members; controls disjoint from every selective feature, drawn from the supported non-selective pool only, matched cell counts equal to the group's (or shortage listed), seeded and independent of other languages; manifest round trip and tamper detection; planted cohort where the answer is known; uses dev-only fields (a test fails if a devtest field is read).
**Mutants.** rank ascending, wrong language's rates in the cells, pool includes selective features, same seed for all draws, cells ignore the magnitude tercile, devtest field used.
**Output check against the real cohort** (local, cached `selectivity.pt`): sizes (minimum 73, Portuguese), shortage report, control-versus-group activity table (matched-mean difference per language) written to the status file.

## Task 4: runner `scripts/g0/phase1_func_run.py` + `run_functional.sbatch` (+ `tests_torch/test_phase1_func_run.py`)

**Interfaces (injected callables as in `run_survival`).** `lower(tokens) -> resid`, `upper(resid_edited) -> final hidden`, `nll_from_hidden(hidden, tokens, valid) -> per-sentence (sum NLL, count)` (chunked unembedding with the final norm and logit soft cap, no whole-vocabulary tensor), `mc_scores(...)`, `encode`, `W_dec`; configuration list built from the groups manifest; `run_precision(bits, tasks, configs)` restores, quantizes, caches `a` for the baseline, runs every configuration, returns item-level arrays keyed by (task, language, config); `run_rescue` (FP pass first, FP features cached sparsely, then the recipient); fidelity quantities (B) from the baseline pass; output `.pt`/`.json` with provenance (commit, versions, hashes, seeds).
**Gates inside the runner (each fails loudly).** (1) cached lower half + upper half equals the full forward bit for bit on unedited input; (2) identity edit equals baseline bit for bit; (3) sham within a stated tolerance; (4) the encoder run on the cached residual reproduces `a` from the full path; (5) the first run records the measured tokens per second and peak memory.
**Tests first** with a toy causal model (cumulative-mean mixing layers, tiny vocabulary) and a toy SAE: gates pass on the toy; an injected bug in each gate path fails the matching gate; item arrays have the right shapes; paired items identical across configurations; the rescue patch with FP = recipient returns the baseline; the all-features and error-term bounds add up to the full-residual patch on the toy; output-language mass computed from a hand vocabulary.
**Mutants.** upper half started at the wrong layer, edit applied before instead of after the hook, BOS edited, rescue donor from the recipient, controls reusing the group ids, soft cap removed, wrong target shift.

## Task 5: pilot on Ada (G1 must hold)

- [ ] Smoke job: full precision plus INT6, three languages (eng_Latn, hin_Deva, khm_Khmr), 40 pilot sentences, K = 64 only, R = 2; verifies the gates on the real model, records tokens per second and peak memory; stop and fix if a gate fails.
- [ ] Belebele calibration job (full precision, both formats, all 27 languages, calibration split): accuracies with intervals; the format and the covered-language list decided by the design's rules; written to `reports/functional_validation/belebele_calibration.md`.
- [ ] T-LM pilot ids, all 27 languages, full precision, all configurations: first look at the full-precision effect (design F1) and at sham; exploratory.
- [ ] Reduced configuration list (compute) decided by the measured throughput using the design's fixed reduction order; recorded.

## Task 6: analysis `scripts/g0/phase1_func_analysis.py` (+ `tests_torch/test_phase1_func_analysis.py`)

Paired cluster bootstrap (clusters = sentence ids across languages, or passages), harm, `E_l^b`, `rho`, classification, rescue fractions, decomposition, F4 subgroup contrasts, per-condition perturbation-norm tables, SAE-fidelity table (B), coverage table; report generator `reports/functional_validation/results/<task>/report.md`, `analysis.json`, CSV tables.
**Tests first** with planted item-level data: a planted effect is recovered, a planted null gives an interval around 0, bootstrap resamples whole clusters (an id appears in all languages or none), the same resamples are used across precisions (a paired difference of identical data is exactly 0 with zero width), ratio classification on planted attenuation/amplification/preservation, rescue fraction on a hand case, negative harm is kept, deterministic under the seed.
**Mutants.** items resampled independently of clusters, different draws per precision, sign flip of harm, ratio inverted, controls averaged over the wrong axis, denominators not fixed.

## Task 7: freeze, evaluation jobs (G2 must hold)

- [ ] `reports/functional_validation/evaluation_freeze.json` committed with the design's section 7 contents before any evaluation run on a quantized model or any T-MC scoring.
- [ ] Evaluation jobs per precision (INT6 and INT4 first, then INT8, INT5, INT7, then the INT16/INT12/INT0 controls and the FP evaluation), at most two at a time; per-job outputs verified (provenance, completeness, a rerun of one job reproduces it byte for byte or the difference is explained, as in D-035).
- [ ] F3 rescue jobs (FP donor plus recipient) for the quantized precisions; F4 subgroup jobs for INT6, INT5, INT4.

## Task 8: figures, tables, closer analysis of the existing results (A, B, F)

- [ ] `scripts/g0/phase1_func_figures.py` (tested on planted data; figures chosen with the `dataviz` skill; colour-blind safe; light and dark; saved as SVG and PNG under `reports/functional_validation/figures/`): (1) full-precision effect per language with its matched-control reference; (2) effect by precision (retention ratio) with the preserved/attenuated/amplified bands; (3) dose-response curves (K, alpha) by precision; (4) rescue decomposition bars (language group, other reconstruction, error term) by precision; (5) association retention (D-035) next to functional retention per precision; (6) SAE fidelity by language and precision; (7) coverage (Belebele headroom by language).
- [ ] Tables for the team's paper (CSV and a rendered markdown): retention by precision; per-language effects; configurations and sizes. Prose stays the team's.
- [ ] A (CPU, tested): within-language models of D-035 loss against baseline margin, share, scale; floor comparison; English-excluded cohort; written to `reports/association_retention/results/within-language/` with limits stated.

## Task 9: documentation consolidation

- [ ] Remove redundancy without losing provenance: one index (`README.md` points to `reports/functional_validation/README.md` and `reports/association_retention/README.md`), superseded banners in `PLAN.md`, `PROGRESS.md`, `HANDOFF.md`, `reports/scheduled_routine.md` shortened to current state with links, old cron prompts marked ended, duplicated status text removed; every removed paragraph exists in git history and, where a document is retired, is moved under `reports/archive/` with a one-line pointer. A link and anchor checker (as used for D-039) must report zero broken local links before and after. No change to `paper/`, `third_party/`, saved numerical results or decision records (decisions are only ever appended).

## Job queue order (what the recurring job follows)

1. Tasks 0 to 3 (no GPU). 2. Task 4 code. 3. Task 5 smoke, then calibration and FP pilot (two jobs). 4. Task 6 on the pilot output (CPU). 5. Freeze. 6. Evaluation jobs (INT6 + INT4, then INT8 + INT5, then INT7 + controls + FP). 7. Rescue and F4 jobs. 8. Task 8 and Task 9 fill any firing in which Ada is busy and no code is pending.
