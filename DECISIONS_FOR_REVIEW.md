# Decisions for your review (the questions I would have asked you)

(`issues.md` already exists and holds the review of the discarded pipeline, so this is a separate file.) Created 2026-10-18.

**How this works.** Every item below is a choice the proposal or the source papers leave open, or one where my default could mislead a result. I have **not** blocked on any of them: each has a *default* that the scheduled routine uses until you answer, chosen as the most literal reading of the source (the rule is "do it as the source did it; anything else is a labelled diagnostic"). To answer, write after `ANSWER:` (a letter, or free text) and save the file. The next routine run reads this file first, applies every answered item that is not yet `APPLIED`, writes a D-entry in `reports/decisions_and_anomalies.md`, changes the code or analysis if needed (test first), and then marks the item `STATUS: APPLIED <date>`. Items are ordered by how much they can change a conclusion.

Current state of the study in one paragraph: Phase 1 ran (2026-10-18). In the primary stream (BOS masked, equal token budgets) none of the three pre-registered confirmatory tests rejects after Holm correction; the BOS-free sensitivity regime shows the opposite sign in an instrument that fails its noise-floor check. Details: `reports/phase1_results.md`.

---

## Q1. Which regime is primary for the multilingual study? (D-001)  [HIGH]
- **Context.** Duan et al.'s windows carry no BOS token (verified in their code and the TransformerLens source, V-8); Gemma Scope excludes BOS, EOS and padding activations from SAE training and metrics (V-1). Our Phase 0 replication is in their regime and reproduces their table. For Phase 1 I ran both: `mask` (BOS prepended, position 0 excluded; the SAE's own convention) and `none` (their convention). With BOS masked the instrument passes the noise-floor gate (median r 0.9957 at INT16/INT12 against 0.99); without BOS on short FLORES sentences it does not (0.982) and the model is nearly broken (perplexity 48,864).
- **Options.** (a) `mask` primary, `none` as sensitivity (current); (b) `none` primary because it is the replication target; (c) report both with equal weight.
- **Default used by the routine:** (a).
- **Caveat that matters for (b):** our `none` run is NOT their regime. Theirs is 256-token windows of continuous text; ours is single sentences of about 26 tokens. See Q5.
- ANSWER:
- STATUS: open

## Q2. Was the OSF registration frozen? (V-10)  [HIGH]
- **Context.** Only you know. If the registration was frozen with specific choices (tier table, thresholds, test definitions), then my provisional choices D-013..D-016 (read r at the peak language, stratum-standardised contrasts, equal budgets by position sampling, populated-language rule, permutation counts) must be reported as deviations or realignments, not as pre-registered analysis.
- **Options.** (a) frozen: send me the registered text/hash and I align or label every deviation; (b) not frozen: my D-entries stand as the analysis plan, fixed before any result was read (D-016, D-017).
- **Default:** treat as (b) in code, label every choice "provisional" in the results (already done).
- ANSWER:
- STATUS: open

## Q3. How is "selective feature damage fraction" defined for the behavioural link?  [HIGH]
- **Context.** `paper/main.tex` (evaluation-plan table) lists a *Behavioral Link*: selective-feature damage fraction against the next-token loss increase ΔL_ℓ per language, with Spearman ρ and multivariable OLS (covariates: overall feature damage, subword token fertility). The proposal text (`anlp_revised.txt`, Phase 2) specifies only a *causal ablation* extension, not this correlational test, and nowhere defines "damage fraction". The proposal's own usage in G0 (b) is "damaged fraction = 1 - survival" (survival at t_s = 0.9); Duan et al. use "damaged" for r < 0.5 (t_d = 0.5). The run outputs already hold per-language loss sums for every width, so ΔL_ℓ can be computed without new GPU time.
- **Options.** (a) damage fraction = 1 - survival at t_s = 0.9 over the selective features peaking in the language (proposal G0 usage), r < 0.5 as sensitivity; (b) r < 0.5 only (their "damaged" class); (c) drop the correlational test and plan only the causal ablation of the proposal.
- **Default:** (a), computed at INT6 and INT7 (the confirmatory widths), INT5 descriptive.
- **Result under the default (a), 2026-10-18** (`reports/phase1_results.md` section 6, D-018, A-032): a positive rank association (rho +0.54 / +0.46 / +0.69 at INT7 / INT6 / INT5) but the OLS cannot separate selective from overall damage (VIF up to 9.5), so it reads as the table's 'partial' branch. Definitions follow Srivastava (2026) for fertility (character fertility, forced by their own whitespace note) and the proposal's G0 usage for damage; if you prefer another definition, answer here and the routine reruns it (the code takes the damage cut-offs as arguments).
- ANSWER:
- STATUS: open (default applied)

## Q4. Are the INT5 / INT4 widths part of the study? (D-006)  [MEDIUM]
- **Context.** The proposal's compression ladder (section 5) contains INT8/7/6/5/4 but pre-registers INT7 and INT6 as confirmatory. With BOS masked INT7 is at the ceiling (99.97% pooled survival), so the confirmatory evidence rests on INT6 alone (A-031); INT5 (59%) and INT4 (14%) are where survival has room. They were run and are reported as exploratory.
- **Options.** (a) keep INT5/INT4 exploratory (current); (b) declare INT6 and INT5 the confirmatory pair in the paper as a documented amendment; (c) drop them from the paper.
- **Default:** (a).
- ANSWER:
- STATUS: open

## Q5. Run the faithful no-BOS regime with their window shape? (diagnostic for A-029 / A-030)  [MEDIUM, needs GPU]
- **Context.** The sign of the selective-versus-control contrast flips between `mask` (-1.4 pp, n.s.) and `none` (+8 to +9 pp, significant). Three untested explanations: (a) the no-BOS instrument is broken on short sentences (perplexity 48,864; noise-floor median r 0.982); (b) selective features differ from controls beyond firing-rate strata; (c) our `none` runs use ~26-token sentences, while Duan's regime is 256-token windows of continuous text (their perplexity 457), so our `none` is not their regime at all.
- **Proposed run (labelled diagnostic, never part of the replication):** concatenate the FLORES sentences of each language into 256-token windows (same equal budget), run both regimes with the unchanged pipeline, ladder INT8/7/6 + INT16/12/0 controls only. About 3 h on one GPU per regime.
- **Options.** (a) run it; (b) do not run it, report the `none` regime as an instrument failure only.
- **Default:** (a) is queued only after you answer; until then the routine does CPU-side work.
- ANSWER:
- STATUS: open

## Q6. Threshold sensitivity of the binary "selective" split (proposal 4.1)  [MEDIUM, needs GPU or a deviation]
- **Context.** 74% of supported features pass the pre-registered rule at 27 languages (A-027; the shuffled-label 95th percentile of S is only 1.83). The proposal asks to report several percentiles. The runs saved S, the support mask and stability counts but **not** the shuffled-null samples, so other percentiles cannot be recomputed from the files.
- **Options.** (a) rerun only the tagging pass (about 1 h on one GPU) with the null samples saved, then recompute at the 90th/95th/99th percentile (same code, nothing new); (b) report fixed S thresholds (2, 3, 5) from the saved S values as a labelled deviation (CPU only, no new run); (c) skip.
- **Default:** (b) now, (a) if you approve.
- ANSWER:
- STATUS: open

## Q7. Confirm or override the other provisional choices  [MEDIUM]
D-003 (S(f) primary, Deng's nu as robustness), D-004 (Phase 1 as a diff on their script 03), D-010/D-015 (all sentences per language; equal budgets by seeded position sampling), D-011 (tier table with Joshi classes; class 4 sits in both High and Mid, A-020), A-022 (the paper says "nine script families", the roster has 14 ISO scripts; I stored a provisional 9-family grouping as the script covariate). Defaults stay unless you change them.
- ANSWER:
- STATUS: open

## Q8. Extensions beyond Phase 1  [LOW now, large]
Proposal extensions (Llama-3.2-1B, pruning, causal ablation) are not started. The causal ablation (proposal 4.4) is the only one the text specifies in detail. Default: not started; after Q3 and Q5 the routine proposes one concrete next run with its source and cost.
- ANSWER:
- STATUS: open

## Q9. Optional replication diagnostics of their paper  [LOW]
Their script 06 (sliding-window perplexity) and arm R3 were never run (D-005, closure D-NOISE is terminal). Default: do not run.
- ANSWER:
- STATUS: open

## Q10. Hugging Face dataset visibility  [LOW]
`karma-skz/always-late-artifacts` is private (274 files, about 400 MB). Make it public when the paper is submitted? Default: stay private.
- ANSWER:
- STATUS: open

---
*Not decisions, reminders:* the prose of `paper/main.tex` is superseded in places (see `issues.md` A1-A6) and must be written by the team (course policy); the routine never edits `paper/`.
