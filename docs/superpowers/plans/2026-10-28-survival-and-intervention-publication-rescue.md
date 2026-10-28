# Plan: test whether feature survival predicts useful behavior

Date: 28 October 2026  
Status: Authorized by the user; start only after the current GPTQ runs are finished and the full results have been reviewed together.  
Scope: Use the current Gemma-2-2B, layer-12, SAE results first. This plan does not promise a publishable result. It is meant to find out whether the evidence supports one.

## Question

When a feature's activation pattern changes less after weight quantization, does an intervention on that feature also keep more of its effect on the model's output?

The project already has feature-activation measures and feature-intervention results for full precision, matched RTN, and GPTQ. The first step is to check whether the saved data can answer this question well enough. Do not assume it can.

This is a test of whether a measurement is useful. It is not, by itself, proof that activation survival causes the model's behavior to survive. Do not turn an association between two scores into a causal claim.

## Why this is the first rescue to test

The current F5 analysis says the old binary “retained/lost” label predicts intervention-effect similarity poorly under GPTQ INT4 (AUROC about 0.49). The continuous activation-survival score has a moderate relationship with effect similarity in the saved data, and that relationship is weaker at INT4 than INT6. Those results make a careful test worthwhile, but do not yet show that survival scores work on new data or add value beyond simple baselines.

The cross-language steering idea is a separate, more expensive pilot with close prior work. Do not add it to this study or use it as a fallback if this analysis is weak. Reconsider it only after the results review, as a separate decision.

## Order of work

### Stage 0 — While current runs are still finishing

- Continue validating and analyzing the GPTQ external-panel runs under their existing frozen instructions. Do not change their settings, rerun legacy results, or start extra model jobs.
- In spare time, do only a data-availability check for F5: locate the saved feature-level files; list their keys, feature IDs, quantizers, bit widths, and sample counts; check whether the same features can be matched across states. Do not calculate new predictor-versus-outcome results yet.
- Record whether the raw files also contain document or example IDs, margins, activation scales, and support-overlap measures. If they do not, mark those comparisons unavailable rather than trying to reconstruct them from aggregates.
- Leave the current plans unchanged until the full run review in Stage 1 is complete.

**Done when:** all currently running GPTQ jobs are validated and the already-planned external analysis is recorded, or a specific failure is documented under the existing GPTQ procedure; the F5 input inventory is ready for review.

### Stage 1 — Review all current results together

After the current GPTQ runs are complete, write one evidence-based review before changing any research plan. Cover the existing full-precision, legacy RTN, matched RTN, and GPTQ results, including the external panel. At minimum, include:

- the original tests and which passed, failed, or remain inconclusive;
- the size and uncertainty of the effects, not just verdict labels;
- whether the external results agree with the main results, with the Wikipedia calibration/evaluation overlap clearly stated;
- what the current F5 results say about activation survival and intervention-effect similarity;
- which evidence is confirmatory under the original rules and which analyses are post hoc;
- the strongest claim the complete project can support, and the claims it cannot support.

Preserve the original results and frozen plans. Add a dated assessment and decision entry; do not revise a past verdict to make the new direction look stronger.

**Decision gate:** if this review shows that the saved F5 data cannot answer the question, or that the project has no defensible result to build on, stop and report that plainly. Do not run an experiment just to keep the project alive.

### Stage 2 — Update forward-looking plans from the review

Only after Stage 1 is written, add dated amendments to the relevant current plans. Check at least:

- `docs/superpowers/plans/2026-10-26-publication-recovery-after-results.md`
- `docs/superpowers/plans/2026-10-26-survival-metric-and-margin-audit-addendum.md`

Keep their history. Amend only future actions, deadlines, and priorities that the completed results actually change. State whether this plan proceeds, is narrowed, or is stopped, and why. Do not silently carry expired budgets or deadlines into new work.

### Stage 3 — Test the survival measure against intervention results

Proceed only if the input audit confirms that feature IDs, states, and outcomes can be matched correctly. Use the existing saved data first; no new model forwards or GPU jobs are part of this stage.

1. Reproduce the existing F5 summary from its saved inputs before adding analyses. Confirm the join counts and that the reported rows refer to the same features in each compared state.
2. Use continuous activation survival as the main predictor. Treat the old binary “retained/lost” label as a separate comparison, since its GPTQ INT4 result was already near chance. If the files support them, examine scale preservation and firing-support overlap as secondary predictors; do not invent a combined score after seeing which one looks best.
3. Use the saved feature-intervention effect-similarity measure as the main outcome. Check its definition, range, noise floor, and baseline effect size first. A feature whose original effect is near zero cannot be judged in the same way as a feature with a clear original effect.
4. Analyze each quantizer and bit width separately before any pooled summary. Compare whether activation survival adds predictive value beyond simple available baselines, such as original effect size and activation scale. Include margin or decision measures only if they were actually saved and are comparable.
5. Test predictive performance on held-out features, keeping all rows for a feature together. Use a feature-level bootstrap or other method that matches the actual sampling units. If document/example IDs are missing, do not claim document-level independence or generalization to new text.
6. Report the relationship, uncertainty, usable sample size, and prediction error. Do not select a survival cutoff, outcome threshold, or model after inspecting the test results. Mark this analysis post hoc, because related F5 results are already known.
7. Use the external-panel results as independent validation only if they contain the same feature-level measure and a genuinely comparable outcome. Group-level scores or different intervention targets are not a substitute.

**Stop rule:** if the feature-level files cannot be joined safely, if held-out evaluation is not possible, or if survival adds no clear predictive value beyond simple baselines, report that result and stop. Do not add a broad sweep, tune thresholds, or switch to steering to rescue a weak result.

### Stage 4 — Record the Gemma result

Create a short methods-and-results report for the Gemma analysis with the data checks, analysis choices, estimates, uncertainty, and limitations. Update the project results and current-status pages only after checking that the report agrees with the raw outputs. Add the next free decision entry and handoff according to the project ledger. Then carry out the final Llama check below if its data, runtime, and validation gates pass.

The strongest acceptable Gemma-only conclusion may be that the current survival score is a limited or unreliable guide to whether feature interventions keep their effects after quantization. A useful negative result is acceptable. A positive result must be described as predictive evidence for this model, layer, SAE, data, and quantization setup; it does not prove a mechanism or establish broad model generality.

### Stage 5 — Final cross-system check on Llama, with analysis

**Why this is credible:** Llama-3.2-1B already has a different outcome from Gemma in the project's earlier group-level check, but it has not had the single-feature F5 survival-versus-effect test. Running that same test on Llama can show whether the Gemma relationship carries over to a second model-plus-SAE setup. It is a useful portability check.

**What it cannot show:** the model and SAE both change, and the tested layers are at different relative depths (Gemma layer 12 of 26; Llama layer 12 of 16). A difference cannot be blamed on model architecture, SAE family, SAE training data, tokenizer, model size, or layer depth separately. Do not compare feature IDs or pool raw feature scores across the two SAEs. The conclusion is about these two model-and-SAE setups only.

1. Before running, freeze a Llama-specific addendum with exact model and SAE revisions, layer/hook, sample and text IDs, quantizer, states, endpoints, gates, and analysis. Use the existing `meta-llama/Llama-3.2-1B` checkpoint and its already-used FineWeb TopK SAE (`blocks.12.hook_resid_pre`, 14,336 features, k=48, seed 42) only if the existing loader/config checks still pass. Do not download a different checkpoint or SAE as part of this step.
2. Reuse the F5 selection method and controls, but select Llama's own features using only its development split. Use the same sample rule (`--per-cell 12 --subset-seed 0`) where the Llama feature counts permit it; report the actual number tested and why any features are missing. Never force matching feature IDs between Llama and Gemma.
3. Use the same per-output-channel RTN method in both systems, comparing FP with INT6 and INT4. Include Llama INT16 as its own numerical-noise reference. This makes the cross-system comparison about the shared RTN method; do not add GPTQ to Llama, since no Llama GPTQ calibration and freeze exist and adding that would broaden the question.
4. Apply the F5 single-feature effect-fingerprint method to Llama's features and paired inputs, with the same sham, logit, and data-completeness gates. Select feature contexts on development data and score effects on the frozen evaluation set. If a truly untouched, comparable document set is already available, use it as a separately labeled external check. Otherwise use the existing held-out FLORES items and clearly label the result post hoc: earlier project studies already used those items.
5. Analyze Llama on its own first: activation-survival correlation (R1), firing overlap and scale as secondary measures, effect-fingerprint similarity (W1), and the relationship between continuous R1 and W1 at INT6 and INT4. Calculate the noise floor from Llama's own FP/INT16 runs. Reuse the same fixed analysis choices and baselines as Stage 3; do not select new cutoffs after seeing Llama outcomes.
6. Compare the Gemma and Llama estimates side by side, each relative to its own noise floor and with uncertainty. Report whether the relationship's direction and predictive value repeat, weaken, reverse, or cannot be resolved. With only two fixed systems, do not make an across-model population claim or treat a non-significant difference as equivalence.
7. Before the full job, run the existing-style smoke test and a small timing/memory test. Prepare enough GPU time and disk space for the full frozen sample, obey the existing two-running-job limit, and stop if the full run cannot fit the available project window or fails a gate. Do not reduce the feature sample after looking at outcomes. If the study cannot pass these checks, report why and finish with the Gemma result.

The literature supports treating this as a system-level check, not a clean architecture comparison: studies report that SAE features can vary even across seeds and that causal effects can differ between SAE families on the same model. These findings are a reason to test with Llama's own SAE and analyze each model separately, not a reason to assume the result will transfer.

**Done when:** the Llama F5 files pass validation; the frozen F5 analysis is reproduced for Llama; the same analysis is reported beside Gemma, including the model/SAE/depth differences and data-reuse limits; and the decision entry and final handoff describe the result without claiming more than this two-system test can show.

Relevant primary research:

- Cho et al. (2026), [Are Single-Token Sparse Autoencoder Features Causally Necessary? Layer-Depth and SAE-Family Effects](https://aclanthology.org/2026.emnlp-main.1794/): reports causal differences between SAE families even on the same base model and layer-depth effects.
- Paulo and Belrose (2025), [Sparse Autoencoders Trained on the Same Data Learn Different Features](https://arxiv.org/abs/2501.16615): reports feature-set differences across SAE training seeds, including TopK SAEs.
- Lan et al. (2024), [Sparse Autoencoders Reveal Universal Feature Spaces Across Large Language Models](https://arxiv.org/abs/2410.06981): shows that cross-model feature comparison requires explicit feature matching; this plan does not assume direct coordinate identity.

## Compute and timing

Use CPU analysis of existing Gemma outputs first. The only new model run authorized here is the bounded Llama F5 check in Stage 5, after the full GPTQ results review and the Gemma analysis. Its sample and analysis must be frozen before outcome runs, and its dry-run must show that it fits the remaining schedule and resources. Do not submit any other GPU jobs under this plan. The current GPTQ runs and the project deadline take priority. Two days can support a careful pilot or analysis, not a guaranteed publication-ready paper.

## Expected files

- This plan.
- One dated overall-results assessment after the current GPTQ panel is complete.
- Additive updates to the two forward-looking plans above, if the review calls for them.
- A focused analysis script and tests only if existing scripts cannot reproduce the required analysis safely.
- A Llama F5 run and cross-system comparison if Stage 5 passes its data, resource, and validation gates.
- A results folder and plain-language report, with no raw weights, secrets, or large model artifacts committed.

## Update 2026-10-28 15:16 IST (Claude, after the user's confirmation and a validity check; D-110)

- **Authorization.** The user wrote in the main chat: "trust what codex added, the plan is fine. you can run a validity check if thats correct or not, and include that in your plans. i also allow you to commit and push codex's work after a validity check." The plan and the order in the cron prompt block are therefore in force (Q31 closed). The order is unchanged: external panel first, then the all-results review (Stage 1), then plan amendments (Stage 2), then Stages 3 and 4, then Stage 5.
- **Validity check done (premises and sources).** The survival-versus-effect relation the plan builds on is real: Spearman correlation between the per-feature input-side agreement R1 and the effect-fingerprint cosine W1 over 1,487 features, INT6 / INT4: legacy 0.533 / 0.413, matched RTN 0.520 / 0.378, GPTQ 0.575 / 0.381 (`reports/gptq_extension/results/f5/*/f5_results.json`, 'validity'); the Llama setting and its earlier 'not consistent' verdict are those of D-087 and D-090; the three cited papers and the steering review's repository numbers were re-checked (D-110 lists the details and the one title change).
- **Validity gates added to this plan (each is a step, not an option).** (1) *Stage 0 / Stage 3 input audit:* before any predictor-versus-outcome number is computed, write down and test the join (feature id, state, language cell, role) with counts; the R1 and W1 columns must be recomputed from the raw `f5_state_*.pt` files and equal the committed tables (the reproduction control of Stage 3 step 1). (2) *Stage 3 analysis code:* any new script is written test first, mutation-checked, and run first on the legacy files with a result that matches the committed F5 tables before it touches the GPTQ or matched files. (3) *Stage 3 step 3:* the near-zero-effect rule (a feature whose full-precision effect is below the numerical floor is reported separately, not pooled) is fixed in writing before the first outcome is read. (4) *Stage 5 gate list, in this order:* the Llama checkpoint and SAE load and reproduce their frozen configuration (hook, k, feature count, d_model); a smoke test passes the F5 gates; a timing and memory test fits the free time and the two-running-job rule; the home disk on Ada has room for the Llama cache and the outputs (about 2.4 GB were free on 2026-10-28; check, do not delete anything); the Llama full-precision effects are not dominated by noise (Llama's group effect at full precision was weak, median +0.048 nats per token, and its pre-registered P1 was not met, D-090: if the per-feature effect-fingerprint cosine between FP and the INT16 repeat is below the noise level for most Llama features, stop and report that). (5) *Reporting:* every number of the Gemma and Llama reports is read from committed files by a generator (as in `make_project_results_tables.py`), and the written conclusions are checked against it by a test.
- **Wording correction that carries into this plan.** The recovery share of the language group (2.0 to 2.6 % of the loss gap) is not 'the share of damage in the group': use 'recovers' wording only (D-110).
- **Stage 0 inventory written (2026-10-28 15:30 IST):** [`reports/survival_effect_rescue/f5_input_inventory_2026-10-28.md`](../../../reports/survival_effect_rescue/f5_input_inventory_2026-10-28.md). Summary: one row per feature per state, aligned across states and arms (1,487 features, 27 languages), R1, support overlap, scale ratio, W1 and original effect size are saved; margins and read-side document IDs are not; the fingerprints and read-side sufficient statistics are not saved (reproduction starts at the per-feature values); full-precision activation scale exists in Phase 1 `survival.pt` (identical across arms); no external F5 equivalent exists (plan Stage 3 step 7 unavailable).
- **Stage 2 done (2026-10-28 21:12 IST):** dated amendments appended to [the publication-recovery plan](2026-10-26-publication-recovery-after-results.md) and [the survival-metric and margin-audit addendum](2026-10-26-survival-metric-and-margin-audit-addendum.md). This plan **proceeds, narrowed** (D-112). Expired budgets and the 28 October 13:53 window are not carried over; the Q29 precision audit and the margin sidecar stay unstarted.

**Correction (2026-10-28 21:30 IST, the user in the main chat: "there was no 48 hour window. what are you saying? continue and complete the work"):** there was **no 48-hour window and no 28 October 13:53 study target set by the user**. Those dates stood in the older plan texts (written by Codex) and I wrongly treated them as real deadlines that had "expired". The statements in the update above that call a window, target or budget "expired" or "not carried over" are therefore wrong in that respect: nothing expired. The GPU-hour figures in these plans (the 32 GPU-hour ceiling, the 8 and 2 GPU-hour allowances) are planning numbers; the user has said the ceiling is soft and that work continues until told to stop. **The only live deadline is the course deliverables on 31 October 2026.** The precision audit (Q29) and the margin sidecar are not started because they are not part of the plan the user authorized (D-110) and Q29 awaits the user's answer, not because a window closed.
